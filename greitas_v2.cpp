// engine_prod_ga_cache.cpp
// Gamybinė versija (CPU-bound / cache-resident):
// - SoA-prepacked node: t[16], x[16]
// - gate + weights "įkepti" ingest/prepack’e (išjungti kaimynai = 0) => hot-path be gate load/mul
// - Strateginis GA per batch parenka (beta, K_eff ∈ {8,16}) pagal imties fitness
// - AVX2+FMA branchless hot-path (load + FMA + reduce), be DRAM working-set (mažas ring per core)
// - Multi-thread + optional core pinning (Linux/Windows)
// - Runtime CPUID AVX2/FMA guard
//
// Linux build (rekomenduojama):
//   g++ -O3 -std=c++20 -march=native -mavx2 -mfma engine_prod_ga_cache.cpp -lpthread -o engine
//   ./engine --threads 8 --nodes-per-thread 200000000
//
// Haswell target:
//   g++ -O3 -std=c++20 -march=haswell -mavx2 -mfma engine_prod_ga_cache.cpp -lpthread -o engine
//
// Windows (MinGW-w64):
//   g++ -O3 -std=c++20 -march=native -mavx2 -mfma engine_prod_ga_cache.cpp -o engine.exe

#include <immintrin.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

#if defined(_WIN32)
  #define NOMINMAX
  #include <windows.h>
#else
  #include <pthread.h>
  #include <sched.h>
#endif

#if !defined(__AVX2__) || !defined(__FMA__)
  #error "Reikia AVX2+FMA (kompiliuok su -mavx2 -mfma ir tinkamu -march=...)."
#endif

// ----------------------------- Utilities -----------------------------

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

static inline uint32_t xorshift32(uint32_t& s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

static inline float absf(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof(u));
    u &= 0x7FFFFFFFu;
    std::memcpy(&x, &u, sizeof(x));
    return x;
}

static inline float clamp_beta(float b) {
    if (b >  0.999999f) b =  0.999999f;
    if (b < -0.999999f) b = -0.999999f;
    return b;
}

static inline bool is_pow2(uint32_t x) {
    return x && ((x & (x - 1u)) == 0u);
}

// Pin current thread to a core (best-effort)
static inline void pin_thread_to_core(int core_id) {
#if defined(_WIN32)
    DWORD_PTR mask = (core_id >= 0 && core_id < (int)(8 * sizeof(DWORD_PTR))) ? (DWORD_PTR(1) << core_id) : 0;
    if (mask) {
        (void)SetThreadAffinityMask(GetCurrentThread(), mask);
        (void)SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    }
#else
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    if (core_id >= 0) CPU_SET(core_id, &cpuset);
    (void)pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
}

// Runtime AVX2/FMA availability check (x86 only).
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#if defined(_MSC_VER)
  #include <intrin.h>
  static inline void cpuid(int out[4], int leaf, int subleaf) { __cpuidex(out, leaf, subleaf); }
  static inline uint64_t xgetbv0() { return _xgetbv(0); }
#else
  static inline void cpuid(int out[4], int leaf, int subleaf) {
      int a, b, c, d;
      __asm__ __volatile__("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "a"(leaf), "c"(subleaf));
      out[0]=a; out[1]=b; out[2]=c; out[3]=d;
  }
  static inline uint64_t xgetbv0() {
      uint32_t eax, edx;
      __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
      return (uint64_t(edx) << 32) | eax;
  }
#endif

static inline bool cpu_has_avx2_fma() {
    int r[4] = {0,0,0,0};
    cpuid(r, 1, 0);
    const bool osxsave = (r[2] & (1 << 27)) != 0;
    const bool avx     = (r[2] & (1 << 28)) != 0;
    if (!osxsave || !avx) return false;
    const uint64_t xcr0 = xgetbv0();
    if ((xcr0 & 0x6) != 0x6) return false;

    cpuid(r, 7, 0);
    const bool avx2 = (r[1] & (1 << 5)) != 0;

    cpuid(r, 1, 0);
    const bool fma = (r[2] & (1 << 12)) != 0;

    return avx2 && fma;
}
#else
static inline bool cpu_has_avx2_fma() { return true; }
#endif

// ----------------------------- Engine config -----------------------------

struct EngineConfig {
    int threads = (int)std::thread::hardware_concurrency();
    bool pin_threads = true;

    uint64_t nodes_per_thread = 100000000ULL;

    // Cache-resident ring per thread (power-of-two). 256 nodes ~ 32KB (1 core L1 scale for 128B nodes).
    uint32_t ring_nodes = 256;

    // GA (power-of-two)
    uint32_t ga_batch = 4096;
    uint32_t sample = 32;
    uint32_t candidates = 4;
    float beta_step = 0.04f;
    float degrade_ratio = 0.97f;

    bool allow_keff8 = true;
    float initial_beta = 0.6f;
};

struct ThreadResult {
    double seconds = 0.0;
    double nodes_per_sec = 0.0;
    float checksum = 0.0f;
    float last_fit = 0.0f;
    float last_beta = 0.0f;
    int last_keff = 16;
};

// ----------------------------- Data -----------------------------

static constexpr int KMAX = 16;

// Production payload: gate+weights baked in upstream => disabled neighbors are 0.
struct alignas(32) NodeLite16 {
    float t[KMAX];
    float x[KMAX];
};

// Replace this with your real prepack/ingest (NIC/PCIe/graph adjacency):
// Fill t/x with already-gated and already-weighted values.
static inline void init_ring_demo(NodeLite16* ring, uint32_t ring_nodes, uint32_t& rng) {
    for (uint32_t i = 0; i < ring_nodes; ++i) {
        const int deg = 8 + int(xorshift32(rng) & 1u) * 8; // 8 or 16
        for (int k = 0; k < KMAX; ++k) {
            const float on = (k < deg) ? 1.0f : 0.0f;
            const float base = float(i) * 0.01f + float(k) * 0.1f;
            const uint32_t r = xorshift32(rng);
            const float noise = float(int(r & 255u) - 128) * 1e-3f;

            // Example weight profile: first 8 strong, last 8 weaker
            const float w = (k < 8) ? 1.0f : 0.25f;

            ring[i].t[k] = on * w * (10.0f + base + noise);
            ring[i].x[k] = on * w * ( 1.0f + base * 0.05f);
        }
    }
}

// ----------------------------- Math / Kernel -----------------------------

static inline __m256 rsqrt_nr1(__m256 x) {
    __m256 y = _mm256_rsqrt_ps(x);
    const __m256 half  = _mm256_set1_ps(0.5f);
    const __m256 three = _mm256_set1_ps(3.0f);
    __m256 yy   = _mm256_mul_ps(y, y);
    __m256 term = _mm256_fnmadd_ps(x, yy, three);
    return _mm256_mul_ps(y, _mm256_mul_ps(half, term));
}

static inline float hsum8_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

static inline void compute_gamma(float beta_scalar, __m256& gamma, __m256& g_beta) {
    beta_scalar = clamp_beta(beta_scalar);
    __m256 beta = _mm256_set1_ps(beta_scalar);
    __m256 b2   = _mm256_mul_ps(beta, beta);
    __m256 inv  = _mm256_sub_ps(_mm256_set1_ps(1.0f), b2);
    inv = _mm256_max_ps(inv, _mm256_set1_ps(1.0e-12f));
    gamma  = rsqrt_nr1(inv);
    g_beta = _mm256_mul_ps(gamma, beta);
}

static inline void kernel_keff8(const NodeLite16* __restrict n, __m256 gamma, __m256 g_beta,
                                float& out_t, float& out_x) {
    __m256 t = _mm256_load_ps(n->t + 0);
    __m256 x = _mm256_load_ps(n->x + 0);

    __m256 tp = _mm256_fnmadd_ps(g_beta, x, _mm256_mul_ps(gamma, t));
    __m256 xp = _mm256_fnmadd_ps(g_beta, t, _mm256_mul_ps(gamma, x));

    out_t = hsum8_ps(tp);
    out_x = hsum8_ps(xp);
}

static inline void kernel_keff16(const NodeLite16* __restrict n, __m256 gamma, __m256 g_beta,
                                 float& out_t, float& out_x) {
    __m256 t0 = _mm256_load_ps(n->t + 0);
    __m256 x0 = _mm256_load_ps(n->x + 0);
    __m256 t1 = _mm256_load_ps(n->t + 8);
    __m256 x1 = _mm256_load_ps(n->x + 8);

    __m256 tp0 = _mm256_fnmadd_ps(g_beta, x0, _mm256_mul_ps(gamma, t0));
    __m256 xp0 = _mm256_fnmadd_ps(g_beta, t0, _mm256_mul_ps(gamma, x0));
    __m256 tp1 = _mm256_fnmadd_ps(g_beta, x1, _mm256_mul_ps(gamma, t1));
    __m256 xp1 = _mm256_fnmadd_ps(g_beta, t1, _mm256_mul_ps(gamma, x1));

    out_t = hsum8_ps(_mm256_add_ps(tp0, tp1));
    out_x = hsum8_ps(_mm256_add_ps(xp0, xp1));
}

static inline float fitness_proxy(float out_t, float out_x) {
    return out_t - absf(out_x);
}

// ----------------------------- GA (strategic) -----------------------------

struct Pick {
    float beta;
    int keff;
    float score;
};

static inline Pick pick_for_batch(const NodeLite16* ring,
                                  uint32_t ring_mask,
                                  float base_beta,
                                  uint32_t& rng,
                                  const EngineConfig& cfg,
                                  bool allow8) {
    float betas[4] = {
        base_beta,
        base_beta + cfg.beta_step,
        base_beta - cfg.beta_step,
        base_beta + 2.0f * cfg.beta_step
    };

    std::vector<uint32_t> idx(cfg.sample);
    for (uint32_t i = 0; i < cfg.sample; ++i) idx[i] = (xorshift32(rng) & ring_mask);

    Pick best{clamp_beta(betas[0]), 16, -1.0e30f};

    const uint32_t candN = (cfg.candidates <= 4) ? cfg.candidates : 4;
    for (uint32_t bi = 0; bi < candN; ++bi) {
        const float b = clamp_beta(betas[bi]);
        __m256 gamma, g_beta;
        compute_gamma(b, gamma, g_beta);

        for (int mode = 0; mode < 2; ++mode) {
            const int keff = (mode == 0) ? 8 : 16;
            if (keff == 8 && !allow8) continue;

            float s = 0.0f;
            for (uint32_t si = 0; si < cfg.sample; ++si) {
                const NodeLite16* n = &ring[idx[si]];
                float ot, ox;
                if (keff == 8) kernel_keff8(n, gamma, g_beta, ot, ox);
                else           kernel_keff16(n, gamma, g_beta, ot, ox);
                s += fitness_proxy(ot, ox);
            }

            if (s > best.score) best = Pick{b, keff, s};
        }
    }

    return best;
}

// ----------------------------- Worker -----------------------------

static ThreadResult run_worker(const EngineConfig& cfg, int tid, uint64_t nodes_total, uint32_t seed) {
    if (cfg.pin_threads) pin_thread_to_core(tid);

    if (!is_pow2(cfg.ring_nodes) || !is_pow2(cfg.ga_batch)) {
        std::cerr << "Config error: ring_nodes and ga_batch must be power-of-two.\n";
        std::exit(1);
    }

    const uint32_t ring_n = cfg.ring_nodes;
    const uint32_t ring_mask = ring_n - 1u;

    NodeLite16* ring = (NodeLite16*)aligned_malloc(32, sizeof(NodeLite16) * (size_t)ring_n);
    if (!ring) {
        std::cerr << "aligned_malloc failed\n";
        std::exit(1);
    }

    uint32_t rng = seed ^ (uint32_t)(0x9E3779B9u * (uint32_t)(tid + 1));
    init_ring_demo(ring, ring_n, rng);

    float base_beta = cfg.initial_beta;
    float last_fit = -1.0e30f;
    bool force16 = false;

    __m256 gamma = _mm256_set1_ps(1.0f);
    __m256 g_beta = _mm256_setzero_ps();
    compute_gamma(base_beta, gamma, g_beta);

    int current_keff = 16;

    float checksum = 0.0f;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (uint64_t i = 0; i < nodes_total; ++i) {
        if ((i & (cfg.ga_batch - 1u)) == 0u) {
            const bool allow8 = cfg.allow_keff8 && (!force16);
            Pick p = pick_for_batch(ring, ring_mask, base_beta, rng, cfg, allow8);

            if (p.keff == 8 && last_fit > -1.0e29f) {
                force16 = (p.score < last_fit * cfg.degrade_ratio);
            } else {
                force16 = false;
            }

            base_beta = 0.9f * base_beta + 0.1f * p.beta;
            last_fit = p.score;

            compute_gamma(base_beta, gamma, g_beta);

            // politika: 8 greičiui, 16 tik jei priverstinai
            current_keff = force16 ? 16 : 8;
        }

        const NodeLite16* n = &ring[(uint32_t)i & ring_mask];

        float ot, ox;
        if (current_keff == 8) kernel_keff8(n, gamma, g_beta, ot, ox);
        else                   kernel_keff16(n, gamma, g_beta, ot, ox);

        // minimalus side-effect
        checksum += (ot + ox) * 1e-6f;

        // Production ingest integracija:
        // čia (vietoj demo) perrašyk ring[(i+offset)&mask] nauju prepacked node iš savo I/O.
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    aligned_free(ring);

    ThreadResult r;
    r.seconds = sec;
    r.nodes_per_sec = (double)nodes_total / sec;
    r.checksum = checksum;
    r.last_fit = last_fit;
    r.last_beta = base_beta;
    r.last_keff = current_keff;
    return r;
}

// ----------------------------- CLI -----------------------------

static bool arg_eq(const char* a, const char* b) { return std::strcmp(a, b) == 0; }

static void print_help() {
    std::cout <<
        "Usage: engine [options]\n"
        "  --threads N            worker threads\n"
        "  --nodes-per-thread N   iterations per thread\n"
        "  --ring N               ring nodes per thread (pow2)\n"
        "  --ga-batch N           GA period (pow2)\n"
        "  --sample N             GA sample size\n"
        "  --beta-step F          beta mutation step\n"
        "  --degrade F            K=8 quality guard ratio\n"
        "  --no-pin               disable pinning\n"
        "  --beta0 F              initial beta\n";
}

int main(int argc, char** argv) {
    if (!cpu_has_avx2_fma()) {
        std::cerr << "CPU/OS does not support AVX2+FMA (or OSXSAVE not enabled).\n";
        return 1;
    }

    EngineConfig cfg;

    for (int i = 1; i < argc; ++i) {
        if (arg_eq(argv[i], "--help") || arg_eq(argv[i], "-h")) { print_help(); return 0; }
        else if (arg_eq(argv[i], "--threads") && i + 1 < argc) cfg.threads = std::max(1, std::atoi(argv[++i]));
        else if (arg_eq(argv[i], "--nodes-per-thread") && i + 1 < argc) cfg.nodes_per_thread = std::strtoull(argv[++i], nullptr, 10);
        else if (arg_eq(argv[i], "--ring") && i + 1 < argc) cfg.ring_nodes = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        else if (arg_eq(argv[i], "--ga-batch") && i + 1 < argc) cfg.ga_batch = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        else if (arg_eq(argv[i], "--sample") && i + 1 < argc) cfg.sample = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        else if (arg_eq(argv[i], "--beta-step") && i + 1 < argc) cfg.beta_step = (float)std::atof(argv[++i]);
        else if (arg_eq(argv[i], "--degrade") && i + 1 < argc) cfg.degrade_ratio = (float)std::atof(argv[++i]);
        else if (arg_eq(argv[i], "--no-pin")) cfg.pin_threads = false;
        else if (arg_eq(argv[i], "--beta0") && i + 1 < argc) cfg.initial_beta = (float)std::atof(argv[++i]);
        else { std::cerr << "Unknown option: " << argv[i] << "\n"; print_help(); return 1; }
    }

    if (!is_pow2(cfg.ring_nodes) || !is_pow2(cfg.ga_batch)) {
        std::cerr << "Error: --ring and --ga-batch must be power-of-two.\n";
        return 1;
    }

    const uint64_t total_all = cfg.nodes_per_thread * (uint64_t)cfg.threads;
    std::vector<std::thread> threads;
    std::vector<ThreadResult> results((size_t)cfg.threads);

    const uint32_t seed = 0xC001D00Du;

    auto T0 = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < cfg.threads; ++t) {
        threads.emplace_back([&, t]() { results[(size_t)t] = run_worker(cfg, t, cfg.nodes_per_thread, seed); });
    }
    for (auto& th : threads) th.join();

    auto T1 = std::chrono::high_resolution_clock::now();
    const double wall = std::chrono::duration<double>(T1 - T0).count();

    double sum_nps = 0.0;
    float sum_chk = 0.0f;
    for (int t = 0; t < cfg.threads; ++t) { sum_nps += results[(size_t)t].nodes_per_sec; sum_chk += results[(size_t)t].checksum; }

    const double wall_nps = (double)total_all / wall;

    std::cout << "threads=" << cfg.threads
              << " nodes_per_thread=" << cfg.nodes_per_thread
              << " total=" << total_all << "\n";
    std::cout << "wall_time_s=" << wall
              << " wall_nodes_per_s=" << wall_nps
              << " sum_thread_nodes_per_s=" << sum_nps
              << " checksum=" << sum_chk << "\n";

    const ThreadResult& r0 = results[0];
    std::cout << "sample_state: last_beta=" << r0.last_beta
              << " last_fit=" << r0.last_fit
              << " last_keff=" << r0.last_keff << "\n";

    return 0;
}
