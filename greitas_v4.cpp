// engine_v5_linear_prod.cpp
// Gamybinis: "Linear kernel" (no loops in kernel) + split-plane GA + double-buffer ring per worker.
// AVX2+FMA, SoA-prepacked, K=8/16 perjungimas per funkcijos rodyklę (GA), be LUT.
//
// Build (Linux):
//   g++ -O3 -std=c++20 -march=native -mavx2 -mfma -flto -DNDEBUG engine_v5_linear_prod.cpp -lpthread -o engine
// Build (Haswell conservative):
//   g++ -O3 -std=c++20 -march=haswell -mavx2 -mfma -flto -DNDEBUG engine_v5_linear_prod.cpp -lpthread -o engine
// Build (Win11 MSYS2/MinGW ucrt64):
//   g++ -O3 -std=c++20 -march=native -mavx2 -mfma -flto -DNDEBUG engine_v5_linear_prod.cpp -o engine.exe
//
// Run example:
//   ./engine --threads 8 --nodes-per-thread 125000000 --ring 16384 --epoch-ms 50 --beta-step 0.05 --speed-bias 0.03
//
// Pastabos:
// - Kerneliai (K=8/K=16) yra pilnai unrolled: jokių for/while kernelio viduje.
// - Vienintelis "clock loop" yra workerio for ir managerio while (neišvengiama).
// - GA keičia tik: beta, keff, perm_id, tail_scale. Ring perpack daromas managerio plane (ne hot path).
// - Jei manager dalijasi core su workeriais (mažai core) – throughput kris. Geriausia: 1 core manageriui.

#include <immintrin.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>

#if defined(_WIN32)
  #define NOMINMAX
  #include <windows.h>
#else
  #include <pthread.h>
  #include <sched.h>
#endif

#if !defined(__AVX2__) || !defined(__FMA__)
#  error "Reikia AVX2+FMA (-mavx2 -mfma) + tinkamo -march=..."
#endif

// ----------------------------- Platform -----------------------------

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

static inline void pin_thread_to_core(int core_id) {
#if defined(_WIN32)
    DWORD_PTR mask = (core_id >= 0 && core_id < (int)(8 * sizeof(DWORD_PTR))) ? (DWORD_PTR(1) << core_id) : 0;
    if (mask) {
        (void)SetThreadAffinityMask(GetCurrentThread(), mask);
        (void)SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    }
#else
    cpu_set_t cs;
    CPU_ZERO(&cs);
    if (core_id >= 0) CPU_SET(core_id, &cs);
    (void)pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
#endif
}

static inline bool is_pow2_u32(uint32_t x) { return x && ((x & (x - 1u)) == 0u); }

// ----------------------------- RNG -----------------------------

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

// ----------------------------- Config -----------------------------

struct EngineConfig {
    int threads = (int)std::thread::hardware_concurrency(); // worker count
    bool pin = true;
    int manager_core = -1; // -1 => no pin for manager, else pin to specific core

    uint64_t nodes_per_thread = 100000000ULL;

    uint32_t ring = 16384;         // pow2, per worker
    uint32_t epoch_ms = 50;        // GA+swap period
    uint32_t refresh_check_mask = 0xFFFF; // worker checks epoch every (mask+1) iters

    // GA knobs
    float beta0 = 0.6f;
    float beta_step = 0.05f;
    float speed_bias = 0.03f;      // penalize slower modes in GA score
    float degrade_ratio = 0.97f;   // if K=8 drops too much => push K=16 next epoch
    float tail_step = 0.25f;       // tail_scale candidates: 0, step, 2*step (clamped)
    uint32_t perm_cands = 4;       // <=4 (fixed perms in this file)
    uint32_t sample = 64;          // GA eval sample size per worker (kept small)
};

// ----------------------------- Data layout -----------------------------
// SoA-prepacked: t[16], x[16]. gate įkeptas: jei K=8 => indices 8..15 = 0.

static constexpr int KMAX = 16;

struct alignas(64) NodeSoA16 {
    float t[KMAX];
    float x[KMAX];
};

struct KernelParams {
    __m256 gamma;
    __m256 g_beta;
};

// ----------------------------- Unrolled kernels -----------------------------

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
static inline void step16(const NodeSoA16* __restrict n, __m256 gamma, __m256 g_beta,
                          __m256& acc_t, __m256& acc_x) {
    __m256 t = _mm256_load_ps(n->t + OFF);
    __m256 x = _mm256_load_ps(n->x + OFF);
    __m256 tp = _mm256_fnmadd_ps(g_beta, x, _mm256_mul_ps(gamma, t));
    __m256 xp = _mm256_fnmadd_ps(g_beta, t, _mm256_mul_ps(gamma, x));
    acc_t = _mm256_add_ps(acc_t, tp);
    acc_x = _mm256_add_ps(acc_x, xp);
}

template<int... Offs>
__attribute__((always_inline))
static inline void compute_unrolled(const NodeSoA16* __restrict n, const KernelParams& p,
                                    float& ot, float& ox, std::integer_sequence<int, Offs...>) {
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

// ----------------------------- Packing / perms -----------------------------
// 4 fiksuoti perm variantai. Gamyboje čia būtų GA išmoktas top-K mapping.

static constexpr uint8_t PERM4[4][KMAX] = {
    { 0,1,2,3,4,5,6,7, 8,9,10,11,12,13,14,15 },
    { 8,9,10,11,12,13,14,15, 0,1,2,3,4,5,6,7 },
    { 0,8,1,9,2,10,3,11, 4,12,5,13,6,14,7,15 },
    { 15,14,13,12,11,10,9,8, 7,6,5,4,3,2,1,0 }
};

static inline void generate_raw16(float* t_raw, float* x_raw, uint32_t& rng) {
    for (int i = 0; i < KMAX; ++i) {
        uint32_t r = xorshift32(rng);
        float noise = float(int(r & 255u) - 128) * 1e-3f;
        float amp = (i < 8) ? 1.0f : 0.25f; // "important" first half
        t_raw[i] = amp * (10.0f + float(i)*0.05f + noise);
        x_raw[i] = amp * ( 1.0f + float(i)*0.02f);
    }
}

struct Genome {
    float beta = 0.6f;       // GA#1
    int   keff = 8;          // GA#2 (8/16)
    float tail_scale = 0.0f; // GA#2
    uint32_t perm_id = 0;    // GA#3
    KernelFn fn = kernel_K8; // derived
    KernelParams params;     // derived (gamma,g_beta)
};

static inline void derive(Genome& g) {
    if (g.keff == 16) g.fn = kernel_K16; else g.fn = kernel_K8;
    compute_gamma(g.beta, g.params);
}

static inline void prepack(NodeSoA16& dst, const float* t_raw, const float* x_raw,
                           int keff, float tail_scale, uint32_t perm_id) {
    const uint8_t* perm = PERM4[perm_id & 3u];
    float tail = (keff == 16) ? tail_scale : 0.0f;

    for (int k = 0; k < KMAX; ++k) {
        int src = perm[k];
        float s = 1.0f;
        if (k >= 8) s = tail; // 0..1, jei 0 => baked gate
        dst.t[k] = t_raw[src] * s;
        dst.x[k] = x_raw[src] * s;
    }
}

static inline float fitness_proxy(float ot, float ox) {
    return ot - absf(ox);
}

// ----------------------------- Per-worker shared state -----------------------------

struct alignas(64) WorkerShared {
    NodeSoA16* ringA = nullptr;
    NodeSoA16* ringB = nullptr;

    // published state:
    std::atomic<uint64_t> epoch{0};
    std::atomic<NodeSoA16*> ring_ptr{nullptr};
    KernelParams params;
    KernelFn fn = kernel_K8;

    // last GA stats (for reporting)
    std::atomic<float> last_fit{0.0f};
    std::atomic<float> last_beta{0.0f};
    std::atomic<int>   last_keff{8};
    std::atomic<float> last_tail{0.0f};
    std::atomic<uint32_t> last_perm{0};
};

// ----------------------------- GA (manager plane) -----------------------------
// Lightweight 3x GA per worker, evaluated on scratch sample.

static inline float speed_penalty(const EngineConfig& cfg, int keff) {
    return cfg.speed_bias * ((keff == 16) ? 1.0f : 0.0f);
}

static Genome ga_select_for_worker(const EngineConfig& cfg, const Genome& cur,
                                   const NodeSoA16* scratch, uint32_t scratch_n, uint32_t& rng)
{
    Genome best = cur;
    float best_score = -1.0e30f;

    // Candidate sets
    float beta_c[4] = {
        cur.beta,
        cur.beta + cfg.beta_step,
        cur.beta - cfg.beta_step,
        cur.beta + 2.0f*cfg.beta_step
    };
    int keff_c[2] = { 8, 16 };
    float tail_c[3] = { 0.0f, cfg.tail_step, std::min(1.0f, 2.0f*cfg.tail_step) };
    uint32_t permN = cfg.perm_cands ? cfg.perm_cands : 1;
    if (permN > 4) permN = 4;

    // Force clamp
    for (int bi = 0; bi < 4; ++bi) beta_c[bi] = clamp_beta(beta_c[bi]);

    for (int bi = 0; bi < 4; ++bi) {
        for (int ki = 0; ki < 2; ++ki) {
            int keff = keff_c[ki];
            for (int ti = 0; ti < 3; ++ti) {
                float tail = (keff == 16) ? tail_c[ti] : 0.0f;
                for (uint32_t pi = 0; pi < permN; ++pi) {
                    Genome g = cur;
                    g.beta = beta_c[bi];
                    g.keff = keff;
                    g.tail_scale = tail;
                    g.perm_id = pi;
                    derive(g);

                    float acc = 0.0f;
                    // sample a few nodes from scratch (already packed)
                    for (uint32_t s = 0; s < cfg.sample; ++s) {
                        uint32_t idx = xorshift32(rng) % scratch_n;
                        float ot, ox;
                        g.fn(&scratch[idx], g.params, ot, ox);
                        acc += fitness_proxy(ot, ox);
                    }

                    acc -= speed_penalty(cfg, keff);

                    if (acc > best_score) {
                        best_score = acc;
                        best = g;
                    }
                }
            }
        }
    }

    // Degrade guard: jei pereita į K=8 ir krito per daug vs cur score => grąžinti K=16
    // (cur score approx using same best_score baseline čia nenaudojam; pigus guard darysi per worker telemetry)
    return best;
}

// Fill whole ring buffer based on genome (manager plane)
static void fill_ring(NodeSoA16* ring, uint32_t ring_n, const Genome& g, uint32_t& rng) {
    alignas(32) float t_raw[16];
    alignas(32) float x_raw[16];
    for (uint32_t i = 0; i < ring_n; ++i) {
        generate_raw16(t_raw, x_raw, rng);
        prepack(ring[i], t_raw, x_raw, g.keff, g.tail_scale, g.perm_id);
    }
}

// Build scratch sample ring (small) for GA eval (manager plane)
static void build_scratch(std::vector<NodeSoA16>& scratch, const Genome& g, uint32_t& rng) {
    alignas(32) float t_raw[16];
    alignas(32) float x_raw[16];
    for (size_t i = 0; i < scratch.size(); ++i) {
        generate_raw16(t_raw, x_raw, rng);
        prepack(scratch[i], t_raw, x_raw, g.keff, g.tail_scale, g.perm_id);
    }
}

// ----------------------------- Worker -----------------------------

struct ThreadResult {
    double seconds = 0.0;
    double nodes_per_s = 0.0;
    float checksum = 0.0f;
};

static ThreadResult worker_main(const EngineConfig& cfg, WorkerShared* sh, int tid)
{
    if (cfg.pin) pin_thread_to_core(tid);

    const uint32_t mask = cfg.ring - 1u;

    // Wait init
    while (sh->ring_ptr.load(std::memory_order_acquire) == nullptr) {
        _mm_pause();
    }

    NodeSoA16* ring = sh->ring_ptr.load(std::memory_order_acquire);
    KernelParams params = sh->params;
    KernelFn fn = sh->fn;
    uint64_t local_epoch = sh->epoch.load(std::memory_order_acquire);

    const uint64_t N = cfg.nodes_per_thread;

    float chk = 0.0f;
    float ot = 0.0f, ox = 0.0f;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (uint64_t i = 0; i < N; ++i) {
        fn(&ring[(uint32_t)i & mask], params, ot, ox);
        chk += (ot + ox) * 1e-6f;

        // Epoch check retai (control-plane amortizacija)
        if (((uint32_t)i & cfg.refresh_check_mask) == 0u) {
            uint64_t e = sh->epoch.load(std::memory_order_acquire);
            if (e != local_epoch) {
                local_epoch = e;
                ring = sh->ring_ptr.load(std::memory_order_acquire);
                params = sh->params;
                fn = sh->fn;
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    ThreadResult r;
    r.seconds = sec;
    r.nodes_per_s = double(N) / sec;
    r.checksum = chk;
    return r;
}

// ----------------------------- Manager -----------------------------

static void manager_main(const EngineConfig& cfg, std::vector<WorkerShared*>& workers)
{
    if (cfg.manager_core >= 0) pin_thread_to_core(cfg.manager_core);

    // init genomes
    std::vector<Genome> cur(workers.size());
    for (size_t i = 0; i < workers.size(); ++i) {
        cur[i].beta = cfg.beta0;
        cur[i].keff = 8;
        cur[i].tail_scale = 0.0f;
        cur[i].perm_id = 0;
        derive(cur[i]);
    }

    // scratch for GA eval
    std::vector<NodeSoA16> scratch(std::max<uint32_t>(cfg.sample, 64u));

    // Initial fill: ringA active, ringB next
    for (size_t wi = 0; wi < workers.size(); ++wi) {
        WorkerShared* sh = workers[wi];
        uint32_t rng = 0xC001D00Du ^ (uint32_t)(wi * 0x9E3779B9u);

        fill_ring(sh->ringA, cfg.ring, cur[wi], rng);
        fill_ring(sh->ringB, cfg.ring, cur[wi], rng);

        sh->params = cur[wi].params;
        sh->fn = cur[wi].fn;
        sh->ring_ptr.store(sh->ringA, std::memory_order_release);
        sh->epoch.store(1, std::memory_order_release);

        sh->last_beta.store(cur[wi].beta, std::memory_order_relaxed);
        sh->last_keff.store(cur[wi].keff, std::memory_order_relaxed);
        sh->last_tail.store(cur[wi].tail_scale, std::memory_order_relaxed);
        sh->last_perm.store(cur[wi].perm_id, std::memory_order_relaxed);
    }

    bool useA = true;

    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(cfg.epoch_ms));

        // For each worker: GA choose next genome, fill inactive ring, publish
        for (size_t wi = 0; wi < workers.size(); ++wi) {
            WorkerShared* sh = workers[wi];
            uint32_t rng = 0xA5A5A5A5u ^ (uint32_t)(wi * 0x3C6EF35Fu) ^ (uint32_t)sh->epoch.load(std::memory_order_relaxed);

            // Build scratch under current genome (stable distribution reference)
            build_scratch(scratch, cur[wi], rng);

            // GA select
            Genome next = ga_select_for_worker(cfg, cur[wi], scratch.data(), (uint32_t)scratch.size(), rng);

            // Fill inactive ring with "next" (feed-plane, not hot path)
            NodeSoA16* inactive = useA ? sh->ringB : sh->ringA;
            fill_ring(inactive, cfg.ring, next, rng);

            // Publish
            sh->params = next.params;
            sh->fn = next.fn;
            sh->ring_ptr.store(inactive, std::memory_order_release);
            sh->epoch.fetch_add(1, std::memory_order_release);

            // Telemetry
            sh->last_beta.store(next.beta, std::memory_order_relaxed);
            sh->last_keff.store(next.keff, std::memory_order_relaxed);
            sh->last_tail.store(next.tail_scale, std::memory_order_relaxed);
            sh->last_perm.store(next.perm_id, std::memory_order_relaxed);

            cur[wi] = next;
        }

        useA = !useA;
    }
}

// ----------------------------- CLI -----------------------------

static inline bool arg_eq(const char* a, const char* b) { return std::strcmp(a, b) == 0; }

static void help() {
    std::cout <<
        "Options:\n"
        "  --threads N\n"
        "  --nodes-per-thread N\n"
        "  --ring N                 (pow2)\n"
        "  --epoch-ms N\n"
        "  --beta0 F\n"
        "  --beta-step F\n"
        "  --speed-bias F\n"
        "  --tail-step F\n"
        "  --perm-cands N           (<=4)\n"
        "  --sample N\n"
        "  --degrade F\n"
        "  --refresh-check-mask HEX (default 0xFFFF)\n"
        "  --no-pin\n"
        "  --manager-core N\n";
}

int main(int argc, char** argv)
{
    EngineConfig cfg;

    for (int i = 1; i < argc; ++i) {
        if (arg_eq(argv[i], "--help") || arg_eq(argv[i], "-h")) { help(); return 0; }
        else if (arg_eq(argv[i], "--threads") && i+1 < argc) cfg.threads = std::max(1, std::atoi(argv[++i]));
        else if (arg_eq(argv[i], "--nodes-per-thread") && i+1 < argc) cfg.nodes_per_thread = std::strtoull(argv[++i], nullptr, 10);
        else if (arg_eq(argv[i], "--ring") && i+1 < argc) cfg.ring = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        else if (arg_eq(argv[i], "--epoch-ms") && i+1 < argc) cfg.epoch_ms = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        else if (arg_eq(argv[i], "--beta0") && i+1 < argc) cfg.beta0 = (float)std::atof(argv[++i]);
        else if (arg_eq(argv[i], "--beta-step") && i+1 < argc) cfg.beta_step = (float)std::atof(argv[++i]);
        else if (arg_eq(argv[i], "--speed-bias") && i+1 < argc) cfg.speed_bias = (float)std::atof(argv[++i]);
        else if (arg_eq(argv[i], "--tail-step") && i+1 < argc) cfg.tail_step = (float)std::atof(argv[++i]);
        else if (arg_eq(argv[i], "--perm-cands") && i+1 < argc) cfg.perm_cands = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        else if (arg_eq(argv[i], "--sample") && i+1 < argc) cfg.sample = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        else if (arg_eq(argv[i], "--degrade") && i+1 < argc) cfg.degrade_ratio = (float)std::atof(argv[++i]);
        else if (arg_eq(argv[i], "--refresh-check-mask") && i+1 < argc) cfg.refresh_check_mask = (uint32_t)std::strtoul(argv[++i], nullptr, 16);
        else if (arg_eq(argv[i], "--no-pin")) cfg.pin = false;
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
    if (cfg.perm_cands == 0) cfg.perm_cands = 1;
    if (cfg.perm_cands > 4) cfg.perm_cands = 4;
    if (cfg.sample < 8) cfg.sample = 8;

    const uint64_t total = cfg.nodes_per_thread * (uint64_t)cfg.threads;

    std::cout << "cfg:\n";
    std::cout << "  threads=" << cfg.threads << "\n";
    std::cout << "  nodes_per_thread=" << (unsigned long long)cfg.nodes_per_thread << "\n";
    std::cout << "  total=" << (unsigned long long)total << "\n";
    std::cout << "  ring=" << cfg.ring << " epoch_ms=" << cfg.epoch_ms << " sample=" << cfg.sample << "\n";
    std::cout << "  beta0=" << cfg.beta0 << " beta_step=" << cfg.beta_step
              << " speed_bias=" << cfg.speed_bias << " tail_step=" << cfg.tail_step
              << " perm_cands=" << cfg.perm_cands << "\n";
    std::cout << "  refresh_check_mask=0x" << std::hex << cfg.refresh_check_mask << std::dec << "\n";
    std::cout << "  pin=" << (cfg.pin ? 1 : 0) << " manager_core=" << cfg.manager_core << "\n";
    std::cout.flush();

    // Allocate per-worker rings
    std::vector<WorkerShared> shared((size_t)cfg.threads);
    std::vector<WorkerShared*> ptrs((size_t)cfg.threads);

    for (int i = 0; i < cfg.threads; ++i) {
        shared[(size_t)i].ringA = (NodeSoA16*)aligned_malloc(64, sizeof(NodeSoA16) * (size_t)cfg.ring);
        shared[(size_t)i].ringB = (NodeSoA16*)aligned_malloc(64, sizeof(NodeSoA16) * (size_t)cfg.ring);
        if (!shared[(size_t)i].ringA || !shared[(size_t)i].ringB) {
            std::cerr << "alloc failed\n";
            return 1;
        }
        ptrs[(size_t)i] = &shared[(size_t)i];
    }

    // Manager
    std::thread mgr([&]() { manager_main(cfg, ptrs); });

    // Workers
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

    // manager thread not stopped (gamybiškai jis visada on). Teste – atjungiame.
    mgr.detach();

    double sum_nps = 0.0;
    float sum_chk = 0.0f;
    for (int t = 0; t < cfg.threads; ++t) {
        sum_nps += res[(size_t)t].nodes_per_s;
        sum_chk += res[(size_t)t].checksum;
    }
    double wall_nps = (double)total / wall;

    std::cout << "result:\n";
    std::cout << "  wall_time_s=" << wall << "\n";
    std::cout << "  wall_nodes_per_s=" << wall_nps << "\n";
    std::cout << "  sum_thread_nodes_per_s=" << sum_nps << "\n";
    std::cout << "  checksum=" << sum_chk << "\n";

    // sample telemetry from worker0
    std::cout << "sample_state(thread0):\n";
    std::cout << "  epoch=" << shared[0].epoch.load(std::memory_order_relaxed) << "\n";
    std::cout << "  beta=" << shared[0].last_beta.load(std::memory_order_relaxed)
              << " keff=" << shared[0].last_keff.load(std::memory_order_relaxed)
              << " tail=" << shared[0].last_tail.load(std::memory_order_relaxed)
              << " perm=" << shared[0].last_perm.load(std::memory_order_relaxed) << "\n";

    for (int i = 0; i < cfg.threads; ++i) {
        aligned_free(shared[(size_t)i].ringA);
        aligned_free(shared[(size_t)i].ringB);
    }

    return 0;
}
