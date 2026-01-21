// engine_prod_v4.cpp
// Gamybinė versija: SoA-prepacked + fiksuotas K(8/16) + AVX2/FMA branchless unrolled kernel
// + gating (baked į prepack) + reduce + split-plane GA (manager thread) + double-buffer ring.
//
// Esmė:
// - Hot-path (workeriai): tik skaičiuoja (kernel_K8 / kernel_K16), jokių ciklų kernelio viduje.
// - Control/Feed (manager): GA parenka {beta, keff, tail_scale, perm_id, feed_shift}, atnaujina "inactive" ringą,
//   tada atominiu epoch switch perjungia pointerį.
//
// Build (Linux):
//   g++ -O3 -std=c++20 -march=native -mavx2 -mfma -flto -DNDEBUG engine_prod_v4.cpp -lpthread -o engine
// Build (Haswell conservative):
//   g++ -O3 -std=c++20 -march=haswell -mavx2 -mfma -flto -DNDEBUG engine_prod_v4.cpp -lpthread -o engine
// Build (Win11 MSYS2 ucrt64):
//   g++ -O3 -std=c++20 -march=native -mavx2 -mfma -flto -DNDEBUG engine_prod_v4.cpp -o engine.exe
//
// Run example:
//   ./engine --threads 8 --nodes-per-thread 125000000 --ring 16384 --epoch-ms 50 --refresh-mask 0xFFFF \
//            --beta0 0.60 --beta-step 0.04 --speed-bias 0.06 --feed-penalty 30.0
//
// Pastaba: benchmarke "feed" yra sintetinis generatorius. Produkcijoje ring pildymas ateina iš realaus srauto,
// bet architektūra identiška (prepack į inactive ring + epoch swap).

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

// ------------------------- config -------------------------

struct EngineConfig {
    int threads = (int)std::thread::hardware_concurrency();
    bool pin_workers = true;
    int manager_core = -1;          // -1 = nepininti
    bool manager_hi_prio = true;

    uint64_t nodes_per_thread = 100000000ULL;

    uint32_t ring = 16384;          // pow2, per worker
    uint32_t epoch_ms = 50;         // GA+feed period
    uint32_t refresh_mask = 0xFFFF; // worker epoch check kas (mask+1) iteracijų

    // GA: kokybės vs greičio balansavimas
    float beta0 = 0.60f;
    float beta_step = 0.04f;
    float speed_bias = 0.06f;       // bauda K=16 / brangesniems režimams
    float feed_penalty = 30.0f;     // bauda už mažesnį feed (čia modeliuoja "kokybės kritimą", jei per retai atnaujini)

    float tail_step = 0.25f;        // tail_scale kandidatų žingsnis (tik K=16)
    uint32_t sample = 64;           // GA quality sample
    uint64_t target_nps = 600000000ULL; // adaptacijai (nebūtina, bet padeda)

    // Kandidatų aibės (mažos => greita GA)
    uint32_t perm_cands = 4;        // 1..4 (perm_id 0..3)
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

// ------------------------- kernel utils -------------------------

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

// Unrolled step: OFF yra compile-time konstanta.
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

// ------------------------- prepack (be LUT) -------------------------
// perm_id 0..3: identity, swap-halves, interleave, reverse (apskaičiuojama, ne lentelė).
static inline int perm_map(uint32_t perm_id, int k) {
    perm_id &= 3u;
    if (perm_id == 0u) return k;
    if (perm_id == 1u) return (k + 8) & 15;                 // swap halves
    if (perm_id == 2u) return (k >> 1) + ((k & 1) << 3);    // 0,8,1,9,2,10,...
    return 15 - k;                                          // reverse
}

// sintetinis "input" (produkcinėje versijoje čia būtų realūs duomenys)
static inline void generate_raw16(float* t_raw, float* x_raw, uint32_t& rng) {
    for (int i = 0; i < KMAX; ++i) {
        uint32_t r = xorshift32(rng);
        float noise = float(int(r & 255u) - 128) * 1e-3f;
        float amp = (i < 8) ? 1.0f : 0.25f;
        t_raw[i] = amp * (10.0f + float(i)*0.05f + noise);
        x_raw[i] = amp * ( 1.0f + float(i)*0.02f);
    }
}

// gating: baked į SoA (jei K=8 => tail=0; jei K=16 => tail_scale taikomas 8..15)
static inline void prepack(NodeSoA16& dst,
                           const float* t_raw, const float* x_raw,
                           int keff, float tail_scale, uint32_t perm_id) {
    float tail = (keff == 16) ? tail_scale : 0.0f;
    for (int k = 0; k < KMAX; ++k) {
        int src = perm_map(perm_id, k);
        float s = (k < 8) ? 1.0f : tail;
        dst.t[k] = t_raw[src] * s;
        dst.x[k] = x_raw[src] * s;
    }
}

// ------------------------- GA / fitness -------------------------

static inline float fitness_proxy(float ot, float ox) {
    // pigus proxy: "energija" (t) minus |x|
    return ot - absf(ox);
}

struct Genome {
    float beta = 0.6f;       // GA#1
    int   keff = 8;          // GA#2 (8/16)
    float tail = 0.0f;       // GA#2
    uint32_t perm = 0;       // GA#3
    uint32_t feed_shift = 4; // GA#3 (feed budget): budget = ring >> feed_shift (0=full,2=1/4,4=1/16)
};

static inline KernelFn kernel_of(const Genome& g) {
    return (g.keff == 16) ? kernel_K16 : kernel_K8;
}

static inline void params_of(const Genome& g, KernelParams& p) {
    compute_gamma(g.beta, p);
}

static inline float speed_penalty(const EngineConfig& cfg, const Genome& g) {
    // K=16 laikom brangesniu
    float p = (g.keff == 16) ? cfg.speed_bias : 0.0f;
    // didesnis feed (mažesnis shift) => daugiau manager darbo/cache pressure
    // čia BAUDŽIAM "mažesnį feed_shift" (t.y. dažnesnį refill), bet tai per cfg.feed_penalty valdo kokybės-greičio kompromisą
    return p;
}

// GA parenka 1 globalų genome visiems worker'iams (strateginis/batch).
static Genome ga_select_batch(const EngineConfig& cfg,
                              const Genome& cur,
                              const NodeSoA16* scratch, uint32_t scratch_n,
                              uint32_t& rng,
                              double measured_wall_nps)
{
    // adaptuojam "K=16" kainą, jei atsiliekam nuo target
    double ratio = (cfg.target_nps > 0) ? (measured_wall_nps / double(cfg.target_nps)) : 1.0;
    float adaptive_speed_bias = cfg.speed_bias;
    if (ratio < 0.95) adaptive_speed_bias *= 1.25f;
    if (ratio < 0.80) adaptive_speed_bias *= 1.60f;

    float beta_c[4] = {
        cur.beta,
        cur.beta + cfg.beta_step,
        cur.beta - cfg.beta_step,
        cur.beta + 2.0f*cfg.beta_step
    };
    for (int i = 0; i < 4; ++i) beta_c[i] = clamp_beta(beta_c[i]);

    int keff_c[2] = { 8, 16 };
    float tail_c[3] = { 0.0f, cfg.tail_step, std::min(1.0f, 2.0f*cfg.tail_step) };
    uint32_t permN = std::max(1u, std::min(cfg.perm_cands, 4u));
    uint32_t feed_shift_c[3] = { 0u, 2u, 4u }; // full, quarter, 1/16

    Genome best = cur;
    float best_score = -1.0e30f;

    for (int bi = 0; bi < 4; ++bi) {
        for (int ki = 0; ki < 2; ++ki) {
            for (int ti = 0; ti < 3; ++ti) {
                for (uint32_t pi = 0; pi < permN; ++pi) {
                    for (int fi = 0; fi < 3; ++fi) {

                        Genome g = cur;
                        g.beta = beta_c[bi];
                        g.keff = keff_c[ki];
                        g.tail = (g.keff == 16) ? tail_c[ti] : 0.0f;
                        g.perm = pi;
                        g.feed_shift = feed_shift_c[fi];

                        KernelParams p;
                        params_of(g, p);
                        KernelFn fn = kernel_of(g);

                        float q = 0.0f;
                        for (uint32_t s = 0; s < cfg.sample; ++s) {
                            uint32_t idx = xorshift32(rng) % scratch_n;
                            float ot, ox;
                            fn(&scratch[idx], p, ot, ox);
                            q += fitness_proxy(ot, ox);
                        }

                        // Kokybės bauda už retesnį feed (didesnį shift) – minimalus kokybės praradimas už greitį.
                        float feed_quality_pen = cfg.feed_penalty * float(g.feed_shift) * 0.01f;

                        // Greičio bauda: K=16 + (indirektinių efektų) adaptuojama
                        float sp = (g.keff == 16) ? adaptive_speed_bias : 0.0f;

                        float score = q - feed_quality_pen - sp;

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

    // published:
    std::atomic<uint32_t> keff{8};
    std::atomic<uint32_t> perm{0};
    std::atomic<uint32_t> feed_shift{4};
    std::atomic<uint32_t> _pad0{0};

    std::atomic<uint32_t> beta_bits{0};
    std::atomic<uint32_t> tail_bits{0}; // float bits

    std::atomic<uint64_t> progress{0};   // nodes processed (for manager NPS)
};

static inline uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static inline float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// ------------------------- manager feed -------------------------

static void fill_ring_full(NodeSoA16* ring, uint32_t ring_n, const Genome& g, uint32_t& rng) {
    alignas(32) float t_raw[16];
    alignas(32) float x_raw[16];
    for (uint32_t i = 0; i < ring_n; ++i) {
        generate_raw16(t_raw, x_raw, rng);
        prepack(ring[i], t_raw, x_raw, g.keff, g.tail, g.perm);
    }
}

// Feed policy: atnaujina tik dalį ring (budget = ring >> feed_shift), kad sumažintų manager apkrovą/cache thrash.
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

static void build_scratch(std::vector<NodeSoA16>& scratch, const Genome& g, uint32_t& rng) {
    alignas(32) float t_raw[16];
    alignas(32) float x_raw[16];
    for (size_t i = 0; i < scratch.size(); ++i) {
        generate_raw16(t_raw, x_raw, rng);
        prepack(scratch[i], t_raw, x_raw, g.keff, g.tail, g.perm);
    }
}

// ------------------------- worker -------------------------

struct ThreadResult {
    double seconds = 0.0;
    double nps = 0.0;
    float checksum = 0.0f;
};

static ThreadResult worker_main(const EngineConfig& cfg, WorkerShared* sh, int tid)
{
    if (cfg.pin_workers) pin_thread_to_core(tid, true);

    while (sh->ring_ptr.load(std::memory_order_acquire) == nullptr) _mm_pause();

    const uint32_t mask = cfg.ring - 1u;

    NodeSoA16* ring = sh->ring_ptr.load(std::memory_order_acquire);
    uint64_t local_epoch = sh->epoch.load(std::memory_order_acquire);

    float beta = u2f(sh->beta_bits.load(std::memory_order_relaxed));
    float tail = u2f(sh->tail_bits.load(std::memory_order_relaxed));
    uint32_t keff = sh->keff.load(std::memory_order_relaxed);

    (void)tail; // baked into ring, tail čia nereikalingas hot path'e.

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
            // progress commit (retai)
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

    // final progress commit
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

    // init genome
    Genome cur;
    cur.beta = cfg.beta0;
    cur.keff = 8;
    cur.tail = 0.0f;
    cur.perm = 0;
    cur.feed_shift = 4;

    // scratch for GA quality eval
    std::vector<NodeSoA16> scratch(std::max<uint32_t>(cfg.sample, 64u));

    // initial fill
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

    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(cfg.epoch_ms));

        // measure wall NPS from worker progress
        uint64_t total_progress = 0;
        for (auto* sh : workers) total_progress += sh->progress.load(std::memory_order_relaxed);

        auto now = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(now - last_tp).count();
        uint64_t dnodes = (total_progress - last_total_progress);

        double measured_wall_nps = (dt > 0.0) ? (double(dnodes) / dt) : 0.0;
        last_tp = now;
        last_total_progress = total_progress;

        // GA scratch (cheap)
        {
            uint32_t rng = 0xA5A5A5A5u ^ uint32_t(total_progress);
            build_scratch(scratch, cur, rng);
            cur = ga_select_batch(cfg, cur, scratch.data(), (uint32_t)scratch.size(), rng, measured_wall_nps);
        }

        // apply to all workers: feed inactive ring + publish epoch
        for (size_t wi = 0; wi < workers.size(); ++wi) {
            WorkerShared* sh = workers[wi];

            NodeSoA16* inactive = useA ? sh->ringB : sh->ringA;
            uint32_t epoch_no = (uint32_t)sh->epoch.load(std::memory_order_relaxed);

            uint32_t rng = 0xD00DFEEDu ^ uint32_t(wi * 0x3C6EF35Fu) ^ epoch_no;

            // budgeted feed (GA#3)
            fill_ring_budget(inactive, cfg.ring, cur, rng, epoch_no);

            // publish (release by epoch store)
            sh->beta_bits.store(f2u(cur.beta), std::memory_order_relaxed);
            sh->tail_bits.store(f2u(cur.tail), std::memory_order_relaxed);
            sh->keff.store(uint32_t(cur.keff), std::memory_order_relaxed);
            sh->perm.store(cur.perm, std::memory_order_relaxed);
            sh->feed_shift.store(cur.feed_shift, std::memory_order_relaxed);

            sh->ring_ptr.store(inactive, std::memory_order_release);
            sh->epoch.fetch_add(1, std::memory_order_release);
        }

        useA = !useA;
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
        "  --feed-penalty F\n"
        "  --tail-step F\n"
        "  --perm-cands N          (1..4)\n"
        "  --sample N\n"
        "  --target-nps N\n"
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
        else if (arg_eq(argv[i], "--feed-penalty") && i+1 < argc) cfg.feed_penalty = (float)std::atof(argv[++i]);
        else if (arg_eq(argv[i], "--tail-step") && i+1 < argc) cfg.tail_step = (float)std::atof(argv[++i]);
        else if (arg_eq(argv[i], "--perm-cands") && i+1 < argc) cfg.perm_cands = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        else if (arg_eq(argv[i], "--sample") && i+1 < argc) cfg.sample = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        else if (arg_eq(argv[i], "--target-nps") && i+1 < argc) cfg.target_nps = std::strtoull(argv[++i], nullptr, 10);
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

    const uint64_t total = cfg.nodes_per_thread * (uint64_t)cfg.threads;

    std::cout << "cfg:\n";
    std::cout << "  threads=" << cfg.threads << "\n";
    std::cout << "  nodes_per_thread=" << (unsigned long long)cfg.nodes_per_thread << "\n";
    std::cout << "  total=" << (unsigned long long)total << "\n";
    std::cout << "  ring=" << cfg.ring << " epoch_ms=" << cfg.epoch_ms
              << " refresh_mask=0x" << std::hex << cfg.refresh_mask << std::dec << "\n";
    std::cout << "  beta0=" << cfg.beta0 << " beta_step=" << cfg.beta_step
              << " speed_bias=" << cfg.speed_bias << " feed_penalty=" << cfg.feed_penalty
              << " tail_step=" << cfg.tail_step << " perm_cands=" << cfg.perm_cands
              << " sample=" << cfg.sample << " target_nps=" << (unsigned long long)cfg.target_nps << "\n";
    std::cout << "  pin_workers=" << (cfg.pin_workers ? 1 : 0)
              << " manager_core=" << cfg.manager_core << "\n";
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
