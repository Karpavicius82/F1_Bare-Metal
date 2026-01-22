// engine_prod_v6.cpp
// Gamybinė v6: SoA-prepacked + K=8/16 unrolled AVX2/FMA kernel + gating baked-in + reduce
// + split-plane GA (manager) + double-buffer ring + realūs (stale vs bw) kaštai fitness'e
// + Keff normalizacija + manager core pin + worker pin (neužlipa ant manager core) + sleep_until epoch.
//
// Build (Linux):
//   g++ -O3 -std=c++20 -march=native -mavx2 -mfma -flto -DNDEBUG engine_prod_v6.cpp -pthread -o engine_v6
// Build (Haswell conservative):
//   g++ -O3 -std=c++20 -march=haswell -mavx2 -mfma -flto -DNDEBUG engine_prod_v6.cpp -pthread -o engine_v6
// Build (Win11 MSYS2 ucrt64):
//   g++ -O3 -std=c++20 -march=native -mavx2 -mfma -flto -DNDEBUG engine_prod_v6.cpp -o engine_v6.exe
//
// Run example (Windows/Linux):
//   ./engine_v6 --threads 7 --manager-core 7 --nodes-per-thread 125000000 --ring 16384 --epoch-ms 50 \
//              --refresh-mask 0xFFFF --beta0 0.60 --beta-step 0.04 --speed-bias 0.06 \
//              --stale-penalty 30.0 --bw-penalty 18.0 --target-nps 600000000
//
// Pastaba: jei nori realiai 50ms epoch, manager-core turi būti dedikuotas (ir workeriai neturi jo užimti).

#include <immintrin.h>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <algorithm>

#if defined(_WIN32)
  #define NOMINMAX
  #include <windows.h>
#else
  #include <pthread.h>
  #include <sched.h>
#endif

#if !defined(__AVX2__) || !defined(__FMA__)
#  error "Reikia AVX2+FMA (-mavx2 -mfma) ir tinkamo -march=..."
#endif

// ------------------------- platform -------------------------

static inline void* aligned_malloc(size_t align, size_t size) {
#if defined(_WIN32)
    return _aligned_malloc(size, align);
#else
    void* p = nullptr;
    if (posix_memalign(&p, align, size) != 0) return nullptr;
    return p;
#endif
}
static inline void aligned_free(void* p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    free(p);
#endif
}

static inline void pin_thread_to_core(int core_id, bool hi_prio = false) {
#if defined(_WIN32)
    if (core_id >= 0 && core_id < (int)(8 * sizeof(DWORD_PTR))) {
        DWORD_PTR mask = (DWORD_PTR(1) << core_id);
        (void)SetThreadAffinityMask(GetCurrentThread(), mask);
        if (hi_prio) (void)SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    }
#else
    if (core_id >= 0) {
        cpu_set_t cs;
        CPU_ZERO(&cs);
        CPU_SET(core_id, &cs);
        (void)pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
    }
    (void)hi_prio;
#endif
}

static inline bool is_pow2_u32(uint32_t x) { return x && ((x & (x - 1u)) == 0u); }

// ------------------------- tiny rng -------------------------

static inline uint32_t xorshift32(uint32_t& s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s;
}

static inline float absf(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    u &= 0x7FFFFFFFu;
    std::memcpy(&x, &u, 4);
    return x;
}

static inline float clamp_beta(float b) {
    if (b >  0.999999f) b =  0.999999f;
    if (b < -0.999999f) b = -0.999999f;
    return b;
}

static inline uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static inline float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// ------------------------- config -------------------------

struct EngineConfig {
    int threads = (int)std::thread::hardware_concurrency();
    bool pin_workers = true;

    int manager_core = -1;      // -1 = nepininti; rekomenduojama nustatyti
    bool manager_hi_prio = true;

    uint64_t nodes_per_thread = 100000000ULL;

    uint32_t ring = 16384;      // pow2, per worker
    uint32_t epoch_ms = 50;     // target period
    uint32_t refresh_mask = 0xFFFF; // worker epoch check kas (mask+1) iteracijų

    // GA
    float beta0 = 0.60f;
    float beta_step = 0.04f;
    float speed_bias = 0.06f;     // K=16 bauda (adaptuojama pagal NPS)
    float tail_step = 0.25f;      // K=16 tail kandidatai: 0, step, 2*step

    // Duomenų šviežumas vs bandwidth:
    float stale_penalty = 30.0f;  // bauda už per RETĄ refresh (mažas refresh_frac)
    float bw_penalty = 18.0f;     // bauda už per DAŽNĄ refresh (didelis refresh_frac)

    uint32_t sample = 64;         // GA kokybės sample
    uint32_t perm_cands = 4;      // 1..4 (perm_id 0..3)

    uint64_t target_nps = 600000000ULL; // adaptacijai
    float hard_floor_ratio = 0.85f;     // jei wall_nps < target*ratio -> apribojam kandidatų erdvę
};

// ------------------------- data layout -------------------------

static constexpr int KMAX = 16;

struct alignas(64) NodeSoA16 {
    float t[KMAX];
    float x[KMAX];
};

struct KernelParams {
    __m256 gamma;
    __m256 g_beta;
};

// ------------------------- kernel (unrolled) -------------------------

static inline __m256 rsqrt_nr1(__m256 x) {
    __m256 y = _mm256_rsqrt_ps(x);
    const __m256 half  = _mm256_set1_ps(0.5f);
    const __m256 three = _mm256_set1_ps(3.0f);
    __m256 yy   = _mm256_mul_ps(y, y);
    __m256 term = _mm256_fnmadd_ps(x, yy, three); // 3 - x*y*y
    return _mm256_mul_ps(y, _mm256_mul_ps(half, term));
}

static inline void compute_gamma(float beta_scalar, KernelParams& p) {
    beta_scalar = clamp_beta(beta_scalar);
    __m256 beta  = _mm256_set1_ps(beta_scalar);
    __m256 b2    = _mm256_mul_ps(beta, beta);
    __m256 inv   = _mm256_sub_ps(_mm256_set1_ps(1.0f), b2);
    inv = _mm256_max_ps(inv, _mm256_set1_ps(1.0e-12f));
    p.gamma  = rsqrt_nr1(inv);
    p.g_beta = _mm256_mul_ps(p.gamma, beta);
}

static inline float hsum8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

template<int OFF>
static inline void step16(const NodeSoA16* __restrict n,
                          __m256 gamma, __m256 g_beta,
                          __m256& acc_t, __m256& acc_x) {
    __m256 t = _mm256_load_ps(n->t + OFF);
    __m256 x = _mm256_load_ps(n->x + OFF);
    __m256 tp = _mm256_fnmadd_ps(g_beta, x, _mm256_mul_ps(gamma, t)); // gamma*t - g_beta*x
    __m256 xp = _mm256_fnmadd_ps(g_beta, t, _mm256_mul_ps(gamma, x)); // gamma*x - g_beta*t
    acc_t = _mm256_add_ps(acc_t, tp);
    acc_x = _mm256_add_ps(acc_x, xp);
}

template<int... Offs>
__attribute__((always_inline))
static inline void compute_unrolled(const NodeSoA16* __restrict n,
                                    const KernelParams& p,
                                    float& ot, float& ox,
                                    std::integer_sequence<int, Offs...>) {
    __m256 acc_t = _mm256_setzero_ps();
    __m256 acc_x = _mm256_setzero_ps();
    (step16<Offs>(n, p.gamma, p.g_beta, acc_t, acc_x), ...);
    ot = hsum8(acc_t);
    ox = hsum8(acc_x);
}

using KernelFn = void(*)(const NodeSoA16*, const KernelParams&, float&, float&);

static void kernel_K16(const NodeSoA16* n, const KernelParams& p, float& ot, float& ox) {
    compute_unrolled(n, p, ot, ox, std::integer_sequence<int, 0, 8>{});
}
static void kernel_K8(const NodeSoA16* n, const KernelParams& p, float& ot, float& ox) {
    compute_unrolled(n, p, ot, ox, std::integer_sequence<int, 0>{});
}

// ------------------------- prepack (no LUT) -------------------------

static inline int perm_map(uint32_t perm_id, int k) {
    perm_id &= 3u;
    if (perm_id == 0u) return k;                         // identity
    if (perm_id == 1u) return (k + 8) & 15;              // swap halves
    if (perm_id == 2u) return (k >> 1) + ((k & 1) << 3); // 0,8,1,9,2,10,...
    return 15 - k;                                       // reverse
}

static inline void generate_raw16(float* t_raw, float* x_raw, uint32_t& rng) {
    for (int i = 0; i < KMAX; ++i) {
        uint32_t r = xorshift32(rng);
        float noise = float(int(r & 255u) - 128) * 1e-3f;
        float amp = (i < 8) ? 1.0f : 0.25f;
        t_raw[i] = amp * (10.0f + float(i)*0.05f + noise);
        x_raw[i] = amp * ( 1.0f + float(i)*0.02f);
    }
}

static inline void prepack(NodeSoA16& dst,
                           const float* t_raw, const float* x_raw,
                           int keff, float tail_scale, uint32_t perm_id) {
    float tail = (keff == 16) ? tail_scale : 0.0f; // gating baked-in
    for (int k = 0; k < KMAX; ++k) {
        int src = perm_map(perm_id, k);
        float s = (k < 8) ? 1.0f : tail;
        dst.t[k] = t_raw[src] * s;
        dst.x[k] = x_raw[src] * s;
    }
}

// ------------------------- fitness / GA -------------------------

static inline float fitness_proxy(float ot, float ox) {
    // pigus proxy: "t energija" minus |x| (konkretus modelis vėliau keičiamas)
    return ot - absf(ox);
}

struct Genome {
    float beta = 0.6f;        // GA#1
    int   keff = 8;           // GA#2 (8/16)
    float tail = 0.0f;        // GA#2 (tik kai keff=16)
    uint32_t perm = 0;        // GA#3 (0..3)
    uint32_t feed_shift = 4;  // GA#3 (0=full,2=1/4,4=1/16)
};

static inline KernelFn kernel_of(const Genome& g) {
    return (g.keff == 16) ? kernel_K16 : kernel_K8;
}

static inline void params_of(const Genome& g, KernelParams& p) {
    compute_gamma(g.beta, p);
}

static inline float keff_effective(const Genome& g) {
    if (g.keff == 8) return 8.0f;
    // 8 pilni + 8 su tail
    return 8.0f + 8.0f * g.tail;
}

static inline float refresh_frac(const Genome& g) {
    uint32_t sh = std::min(g.feed_shift, 14u);
    return 1.0f / float(1u << sh);
}

static Genome ga_select_batch(const EngineConfig& cfg,
                              const Genome& cur,
                              const float* t_raw, const float* x_raw,
                              uint32_t scratch_n,
                              uint32_t& rng,
                              double measured_wall_nps)
{
    float adaptive_speed_bias = cfg.speed_bias;
    if (cfg.target_nps) {
        double ratio = measured_wall_nps / double(cfg.target_nps);
        if (ratio < 0.95) adaptive_speed_bias *= 1.25f;
        if (ratio < 0.80) adaptive_speed_bias *= 1.60f;
    }

    // HARD apribojimas kai throughput per mažas:
    bool hard_slow = (cfg.target_nps && measured_wall_nps < double(cfg.target_nps) * cfg.hard_floor_ratio);

    float beta_c[4] = {
        cur.beta,
        cur.beta + cfg.beta_step,
        cur.beta - cfg.beta_step,
        cur.beta + 2.0f*cfg.beta_step
    };
    for (int i = 0; i < 4; ++i) beta_c[i] = clamp_beta(beta_c[i]);

    int keff_c_all[2] = { 8, 16 };
    int keff_c_fast[1] = { 8 };

    float tail_c[3] = { 0.0f, cfg.tail_step, std::min(1.0f, 2.0f*cfg.tail_step) };
    uint32_t permN = std::max(1u, std::min(cfg.perm_cands, 4u));

    // feed_shift kandidatai
    uint32_t feed_shift_c_all[3] = { 0u, 2u, 4u };
    uint32_t feed_shift_c_fast[2] = { 2u, 4u }; // jei labai lėta, nelaikom full refresh

    const int* keff_c = hard_slow ? keff_c_fast : keff_c_all;
    int keff_cnt = hard_slow ? 1 : 2;

    const uint32_t* fs_c = hard_slow ? feed_shift_c_fast : feed_shift_c_all;
    int fs_cnt = hard_slow ? 2 : 3;

    // scratch nodes per candidate
    std::vector<NodeSoA16> scratch_nodes(scratch_n);

    Genome best = cur;
    float best_score = -1.0e30f;

    for (int bi = 0; bi < 4; ++bi) {
        for (int ki = 0; ki < keff_cnt; ++ki) {
            for (int ti = 0; ti < 3; ++ti) {
                for (uint32_t pi = 0; pi < permN; ++pi) {
                    for (int fi = 0; fi < fs_cnt; ++fi) {

                        Genome g = cur;
                        g.beta = beta_c[bi];
                        g.keff = keff_c[ki];
                        g.tail = (g.keff == 16) ? tail_c[ti] : 0.0f;
                        g.perm = pi;
                        g.feed_shift = fs_c[fi];

                        // prepack scratch for THIS candidate
                        for (uint32_t s = 0; s < scratch_n; ++s) {
                            const float* tt = t_raw + s*KMAX;
                            const float* xx = x_raw + s*KMAX;
                            prepack(scratch_nodes[s], tt, xx, g.keff, g.tail, g.perm);
                        }

                        KernelParams p;
                        params_of(g, p);
                        KernelFn fn = kernel_of(g);

                        float q = 0.0f;
                        for (uint32_t it = 0; it < cfg.sample; ++it) {
                            uint32_t idx = xorshift32(rng) % scratch_n;
                            float ot, ox;
                            fn(&scratch_nodes[idx], p, ot, ox);
                            q += fitness_proxy(ot, ox);
                        }

                        // normalizacija, kad K=16 nelaimėtų vien dėl sumos dydžio
                        float qnorm = q / std::max(1.0f, keff_effective(g));

                        // stale vs bw kaštai
                        float rf = refresh_frac(g); // 1.0, 1/4, 1/16
                        float stale_cost = cfg.stale_penalty * (1.0f - rf);
                        float bw_cost    = cfg.bw_penalty    * rf;

                        // K=16 speed bias
                        float sp = (g.keff == 16) ? adaptive_speed_bias : 0.0f;

                        float score = qnorm - stale_cost - bw_cost - sp;

                        if (score > best_score) {
                            best_score = score;
                            best = g;
                        }
                    }
                }
            }
        }
    }

    return best;
}

// ------------------------- shared state -------------------------

struct alignas(64) WorkerShared {
    NodeSoA16* ringA = nullptr;
    NodeSoA16* ringB = nullptr;

    std::atomic<uint64_t> epoch{0};
    std::atomic<NodeSoA16*> ring_ptr{nullptr};

    std::atomic<uint32_t> keff{8};
    std::atomic<uint32_t> perm{0};
    std::atomic<uint32_t> feed_shift{4};

    std::atomic<uint32_t> beta_bits{0};
    std::atomic<uint32_t> tail_bits{0};

    std::atomic<uint64_t> progress{0}; // nodes processed
};

// ------------------------- manager feed -------------------------

static void fill_ring_full(NodeSoA16* ring, uint32_t ring_n, const Genome& g, uint32_t& rng) {
    alignas(32) float t_raw[16];
    alignas(32) float x_raw[16];
    for (uint32_t i = 0; i < ring_n; ++i) {
        generate_raw16(t_raw, x_raw, rng);
        prepack(ring[i], t_raw, x_raw, g.keff, g.tail, g.perm);
    }
}

static void fill_ring_budget(NodeSoA16* ring, uint32_t ring_n, const Genome& g, uint32_t& rng, uint32_t epoch_no) {
    uint32_t shift = std::min(g.feed_shift, 14u);
    uint32_t budget = ring_n >> shift;
    budget = std::max(64u, budget);
    if (budget > ring_n) budget = ring_n;

    uint32_t start = (epoch_no * 9973u) & (ring_n - 1u);
    if (start + budget > ring_n) start = ring_n - budget;

    alignas(32) float t_raw[16];
    alignas(32) float x_raw[16];
    for (uint32_t i = 0; i < budget; ++i) {
        generate_raw16(t_raw, x_raw, rng);
        prepack(ring[start + i], t_raw, x_raw, g.keff, g.tail, g.perm);
    }
}

// ------------------------- worker -------------------------

struct ThreadResult {
    double seconds = 0.0;
    double nps = 0.0;
    float checksum = 0.0f;
};

static int map_worker_core(const EngineConfig& cfg, int tid) {
    // Neužlipam ant manager_core
    int core = tid;
    if (cfg.manager_core >= 0 && core >= cfg.manager_core) core++;
    return core;
}

static ThreadResult worker_main(const EngineConfig& cfg, WorkerShared* sh, int tid)
{
    if (cfg.pin_workers) pin_thread_to_core(map_worker_core(cfg, tid), true);

    while (sh->ring_ptr.load(std::memory_order_acquire) == nullptr) _mm_pause();

    const uint32_t mask = cfg.ring - 1u;

    NodeSoA16* ring = sh->ring_ptr.load(std::memory_order_acquire);
    uint64_t local_epoch = sh->epoch.load(std::memory_order_acquire);

    float beta = u2f(sh->beta_bits.load(std::memory_order_relaxed));
    uint32_t keff = sh->keff.load(std::memory_order_relaxed);

    KernelParams params;
    compute_gamma(beta, params);
    KernelFn fn = (keff == 16) ? kernel_K16 : kernel_K8;

    const uint64_t N = cfg.nodes_per_thread;

    float chk = 0.0f;
    float ot = 0.0f, ox = 0.0f;

    uint64_t last_progress_commit = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (uint64_t i = 0; i < N; ++i) {
        fn(&ring[(uint32_t)i & mask], params, ot, ox);
        chk += (ot + ox) * 1e-6f;

        if (((uint32_t)i & cfg.refresh_mask) == 0u) {
            uint64_t delta = (i - last_progress_commit);
            if (delta) {
                sh->progress.fetch_add(delta, std::memory_order_relaxed);
                last_progress_commit = i;
            }

            uint64_t e = sh->epoch.load(std::memory_order_acquire);
            if (e != local_epoch) {
                local_epoch = e;
                ring = sh->ring_ptr.load(std::memory_order_acquire);

                beta = u2f(sh->beta_bits.load(std::memory_order_relaxed));
                keff = sh->keff.load(std::memory_order_relaxed);
                compute_gamma(beta, params);
                fn = (keff == 16) ? kernel_K16 : kernel_K8;
            }
        }
    }

    if (N > last_progress_commit) {
        sh->progress.fetch_add(N - last_progress_commit, std::memory_order_relaxed);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    ThreadResult r;
    r.seconds = sec;
    r.nps = double(N) / sec;
    r.checksum = chk;
    return r;
}

// ------------------------- manager -------------------------

static void manager_main(const EngineConfig& cfg, std::vector<WorkerShared*>& workers)
{
    if (cfg.manager_core >= 0) pin_thread_to_core(cfg.manager_core, cfg.manager_hi_prio);

    Genome cur;
    cur.beta = cfg.beta0;
    cur.keff = 8;
    cur.tail = 0.0f;
    cur.perm = 0;
    cur.feed_shift = 4;

    // scratch raw (neprepacked): t_raw[scratch_n][16], x_raw[scratch_n][16]
    uint32_t scratch_n = std::max<uint32_t>(64u, cfg.sample);
    std::vector<float> t_raw(scratch_n * KMAX);
    std::vector<float> x_raw(scratch_n * KMAX);

    // initial full fill + publish
    for (size_t wi = 0; wi < workers.size(); ++wi) {
        WorkerShared* sh = workers[wi];
        uint32_t rng = 0xC001D00Du ^ uint32_t(wi * 0x9E3779B9u);

        fill_ring_full(sh->ringA, cfg.ring, cur, rng);
        fill_ring_full(sh->ringB, cfg.ring, cur, rng);

        sh->beta_bits.store(f2u(cur.beta), std::memory_order_relaxed);
        sh->tail_bits.store(f2u(cur.tail), std::memory_order_relaxed);
        sh->keff.store(uint32_t(cur.keff), std::memory_order_relaxed);
        sh->perm.store(cur.perm, std::memory_order_relaxed);
        sh->feed_shift.store(cur.feed_shift, std::memory_order_relaxed);

        sh->ring_ptr.store(sh->ringA, std::memory_order_release);
        sh->epoch.store(1, std::memory_order_release);
    }

    bool useA = true;

    uint64_t last_total_progress = 0;
    auto last_tp = std::chrono::high_resolution_clock::now();

    // schedule
    auto last_epoch_start = std::chrono::high_resolution_clock::now();
    auto next_deadline = std::chrono::high_resolution_clock::now() + std::chrono::milliseconds(cfg.epoch_ms);

    uint32_t epoch_no = 1;

    double nps_smooth = 0.0;

    std::cout << "Epoch | Period | Work | Wall NPS (Smooth) | Beta   | Keff | Perm | Shift | Bdg% | Score\n";
    std::cout << "------+--------+------+-------------------+--------+------+------+-------+------+------\n";

    while (true) {
        std::this_thread::sleep_until(next_deadline);
        auto epoch_start = std::chrono::high_resolution_clock::now();
        
        // Calculate period_ms since last epoch start
        double period_ms = std::chrono::duration<double, std::milli>(epoch_start - last_epoch_start).count();
        last_epoch_start = epoch_start;

        next_deadline += std::chrono::milliseconds(cfg.epoch_ms);

        // measure wall NPS from worker progress
        uint64_t total_progress = 0;
        for (auto* sh : workers) total_progress += sh->progress.load(std::memory_order_relaxed);

        auto now = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(now - last_tp).count();
        uint64_t dnodes = (total_progress - last_total_progress);

        double raw_nps = (dt > 0.0) ? (double(dnodes) / dt) : 0.0;
        if (nps_smooth == 0.0) nps_smooth = raw_nps;
        else nps_smooth = 0.7 * nps_smooth + 0.3 * raw_nps; // EMA smoothing

        last_tp = now;
        last_total_progress = total_progress;

        // generate raw scratch once per epoch (independent of candidate)
        {
            uint32_t rng = 0xA5A5A5A5u ^ uint32_t(total_progress) ^ epoch_no;
            for (uint32_t s = 0; s < scratch_n; ++s) {
                float* tt = t_raw.data() + s*KMAX;
                float* xx = x_raw.data() + s*KMAX;
                generate_raw16(tt, xx, rng);
            }
        }

        // GA selection using smoothed NPS
        {
            uint32_t rng = 0xD15EA5Eu ^ uint32_t(total_progress) ^ (epoch_no * 0x9E3779B9u);
            cur = ga_select_batch(cfg, cur, t_raw.data(), x_raw.data(), scratch_n, rng, nps_smooth);
        }

        // apply to all workers
        uint64_t total_budget_nodes = 0;
        for (size_t wi = 0; wi < workers.size(); ++wi) {
            WorkerShared* sh = workers[wi];

            NodeSoA16* inactive = useA ? sh->ringB : sh->ringA;
            uint32_t e = (uint32_t)sh->epoch.load(std::memory_order_relaxed);

            uint32_t rng = 0xD00DFEEDu ^ uint32_t(wi * 0x3C6EF35Fu) ^ e ^ (epoch_no * 101u);

            // budgeted feed
            uint32_t shift = std::min(cur.feed_shift, 14u);
            uint32_t budget = cfg.ring >> shift;
            budget = std::max(64u, budget);
            if (budget > cfg.ring) budget = cfg.ring;
            total_budget_nodes += budget;

            fill_ring_budget(inactive, cfg.ring, cur, rng, e);

            // publish
            sh->beta_bits.store(f2u(cur.beta), std::memory_order_relaxed);
            sh->tail_bits.store(f2u(cur.tail), std::memory_order_relaxed);
            sh->keff.store(uint32_t(cur.keff), std::memory_order_relaxed);
            sh->perm.store(cur.perm, std::memory_order_relaxed);
            sh->feed_shift.store(cur.feed_shift, std::memory_order_relaxed);

            sh->ring_ptr.store(inactive, std::memory_order_release);
            sh->epoch.fetch_add(1, std::memory_order_release);
        }

        useA = !useA;

        auto epoch_end = std::chrono::high_resolution_clock::now();
        double work_ms = std::chrono::duration<double, std::milli>(epoch_end - epoch_start).count();
        float budget_pct = (float)total_budget_nodes / (float)(workers.size() * cfg.ring) * 100.0f;

        // Log format: Epoch | Period | Work | NPS | Beta | Keff | Perm | Shift | Bdg%
        printf("%5u | %6.1f | %4.1f | %17.0f | %6.4f | %4d | %4d | %5d | %4.0f | --\n",
               epoch_no, period_ms, work_ms, nps_smooth,
               cur.beta, cur.keff, cur.perm, cur.feed_shift, budget_pct);

        epoch_no++;
    }
}

// ------------------------- cli -------------------------

static inline bool arg_eq(const char* a, const char* b) { return std::strcmp(a, b) == 0; }

static void help() {
    std::cout <<
        "Options:\n"
        "  --threads N\n"
        "  --nodes-per-thread N\n"
        "  --ring N                (pow2)\n"
        "  --epoch-ms N\n"
        "  --refresh-mask 0xHEX\n"
        "  --beta0 F\n"
        "  --beta-step F\n"
        "  --speed-bias F\n"
        "  --stale-penalty F\n"
        "  --bw-penalty F\n"
        "  --tail-step F\n"
        "  --perm-cands N          (1..4)\n"
        "  --sample N\n"
        "  --target-nps N\n"
        "  --hard-floor-ratio F\n"
        "  --no-pin\n"
        "  --manager-core N\n";
}

// ------------------------- main -------------------------

int main(int argc, char** argv)
{
    EngineConfig cfg;

    for (int i = 1; i < argc; ++i) {
        if (arg_eq(argv[i], "--help") || arg_eq(argv[i], "-h")) { help(); return 0; }
        else if (arg_eq(argv[i], "--threads") && i+1 < argc) cfg.threads = std::max(1, std::atoi(argv[++i]));
        else if (arg_eq(argv[i], "--nodes-per-thread") && i+1 < argc) cfg.nodes_per_thread = std::strtoull(argv[++i], nullptr, 10);
        else if (arg_eq(argv[i], "--ring") && i+1 < argc) cfg.ring = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        else if (arg_eq(argv[i], "--epoch-ms") && i+1 < argc) cfg.epoch_ms = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        else if (arg_eq(argv[i], "--refresh-mask") && i+1 < argc) cfg.refresh_mask = (uint32_t)std::strtoul(argv[++i], nullptr, 16);
        else if (arg_eq(argv[i], "--beta0") && i+1 < argc) cfg.beta0 = (float)std::atof(argv[++i]);
        else if (arg_eq(argv[i], "--beta-step") && i+1 < argc) cfg.beta_step = (float)std::atof(argv[++i]);
        else if (arg_eq(argv[i], "--speed-bias") && i+1 < argc) cfg.speed_bias = (float)std::atof(argv[++i]);
        else if (arg_eq(argv[i], "--stale-penalty") && i+1 < argc) cfg.stale_penalty = (float)std::atof(argv[++i]);
        else if (arg_eq(argv[i], "--bw-penalty") && i+1 < argc) cfg.bw_penalty = (float)std::atof(argv[++i]);
        else if (arg_eq(argv[i], "--tail-step") && i+1 < argc) cfg.tail_step = (float)std::atof(argv[++i]);
        else if (arg_eq(argv[i], "--perm-cands") && i+1 < argc) cfg.perm_cands = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        else if (arg_eq(argv[i], "--sample") && i+1 < argc) cfg.sample = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        else if (arg_eq(argv[i], "--target-nps") && i+1 < argc) cfg.target_nps = std::strtoull(argv[++i], nullptr, 10);
        else if (arg_eq(argv[i], "--hard-floor-ratio") && i+1 < argc) cfg.hard_floor_ratio = (float)std::atof(argv[++i]);
        else if (arg_eq(argv[i], "--no-pin")) cfg.pin_workers = false;
        else if (arg_eq(argv[i], "--manager-core") && i+1 < argc) cfg.manager_core = std::atoi(argv[++i]);
        else {
            std::cerr << "Unknown arg: " << argv[i] << "\n";
            help();
            return 1;
        }
    }

    if (!is_pow2_u32(cfg.ring)) {
        std::cerr << "Error: --ring must be power-of-two\n";
        return 1;
    }
    cfg.perm_cands = std::max(1u, std::min(cfg.perm_cands, 4u));
    cfg.sample = std::max(8u, cfg.sample);
    cfg.hard_floor_ratio = std::max(0.1f, std::min(0.99f, cfg.hard_floor_ratio));

    const uint64_t total = cfg.nodes_per_thread * (uint64_t)cfg.threads;

    std::cout << "cfg:\n";
    std::cout << "  threads=" << cfg.threads << " manager_core=" << cfg.manager_core << " pin_workers=" << (cfg.pin_workers?1:0) << "\n";
    std::cout << "  nodes_per_thread=" << (unsigned long long)cfg.nodes_per_thread << " total=" << (unsigned long long)total << "\n";
    std::cout << "  ring=" << cfg.ring << " epoch_ms=" << cfg.epoch_ms << " refresh_mask=0x" << std::hex << cfg.refresh_mask << std::dec << "\n";
    std::cout << "  beta0=" << cfg.beta0 << " beta_step=" << cfg.beta_step << " speed_bias=" << cfg.speed_bias << "\n";
    std::cout << "  stale_penalty=" << cfg.stale_penalty << " bw_penalty=" << cfg.bw_penalty << " tail_step=" << cfg.tail_step << "\n";
    std::cout << "  perm_cands=" << cfg.perm_cands << " sample=" << cfg.sample << " target_nps=" << (unsigned long long)cfg.target_nps
              << " hard_floor_ratio=" << cfg.hard_floor_ratio << "\n";
    std::cout << "  sizeof(NodeSoA16)=" << sizeof(NodeSoA16) << " bytes\n";
    std::cout.flush();

    // allocate per-worker rings
    std::vector<WorkerShared> shared((size_t)cfg.threads);
    std::vector<WorkerShared*> ptrs((size_t)cfg.threads);

    for (int i = 0; i < cfg.threads; ++i) {
        shared[(size_t)i].ringA = (NodeSoA16*)aligned_malloc(64, sizeof(NodeSoA16) * (size_t)cfg.ring);
        shared[(size_t)i].ringB = (NodeSoA16*)aligned_malloc(64, sizeof(NodeSoA16) * (size_t)cfg.ring);
        if (!shared[(size_t)i].ringA || !shared[(size_t)i].ringB) {
            std::cerr << "alloc failed\n";
            return 1;
        }
        std::memset(shared[(size_t)i].ringA, 0, sizeof(NodeSoA16) * (size_t)cfg.ring);
        std::memset(shared[(size_t)i].ringB, 0, sizeof(NodeSoA16) * (size_t)cfg.ring);
        ptrs[(size_t)i] = &shared[(size_t)i];
    }

    // manager (runs forever; benchmark mode => detach)
    std::thread mgr([&](){ manager_main(cfg, ptrs); });

    // workers
    std::vector<std::thread> workers;
    std::vector<ThreadResult> res((size_t)cfg.threads);

    auto T0 = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < cfg.threads; ++t) {
        workers.emplace_back([&, t]() {
            res[(size_t)t] = worker_main(cfg, &shared[(size_t)t], t);
        });
    }
    for (auto& th : workers) th.join();

    auto T1 = std::chrono::high_resolution_clock::now();
    double wall = std::chrono::duration<double>(T1 - T0).count();

    mgr.detach();

    double sum_nps = 0.0;
    float checksum = 0.0f;
    for (int t = 0; t < cfg.threads; ++t) {
        sum_nps += res[(size_t)t].nps;
        checksum += res[(size_t)t].checksum;
    }
    double wall_nps = (double)total / wall;

    std::cout << "result:\n";
    std::cout << "  wall_time_s=" << wall << "\n";
    std::cout << "  wall_nodes_per_s=" << wall_nps << "\n";
    std::cout << "  sum_thread_nodes_per_s=" << sum_nps << "\n";
    std::cout << "  checksum=" << checksum << "\n";

    // sample state (thread0)
    float beta0 = u2f(shared[0].beta_bits.load(std::memory_order_relaxed));
    uint32_t keff0 = shared[0].keff.load(std::memory_order_relaxed);
    uint32_t perm0 = shared[0].perm.load(std::memory_order_relaxed);
    uint32_t fs0 = shared[0].feed_shift.load(std::memory_order_relaxed);
    std::cout << "sample_state(thread0): epoch=" << (unsigned long long)shared[0].epoch.load(std::memory_order_relaxed)
              << " beta=" << beta0 << " keff=" << keff0 << " perm=" << perm0 << " feed_shift=" << fs0 << "\n";

    for (int i = 0; i < cfg.threads; ++i) {
        aligned_free(shared[(size_t)i].ringA);
        aligned_free(shared[(size_t)i].ringB);
    }

    return 0;
}
