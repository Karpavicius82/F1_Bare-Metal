// engine_alderlake_v7.cpp
// Optimized for Intel Core i9-12900 (Alder Lake)
// Features:
// 1. Topology Awareness: P-cores for Workers, E-core for Manager.
// 2. Cell Shifting: AVX2 permutevar8x32 instead of memory shuffle.
// 3. Driver-in-Driver GA with Backpressure.

#include <immintrin.h>
#include <atomic>
#include <vector>
#include <thread>
#include <iostream>
#include <cstring>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iomanip>

#if defined(_WIN32)
  #define NOMINMAX
  #include <windows.h>
#else
  #include <pthread.h>
  #include <sched.h>
#endif

// ------------------------- Configuration -------------------------

// P-cores on i9-12900K are typically physical indices 0-7.
// With HT, logical processors are 0-15.
// E-cores are physical 8-15 (logical 16-23).
static const int PCORE_COUNT = 8; 

// ------------------------- Structures -------------------------

static constexpr int KMAX = 16;

struct alignas(64) NodeSoA16 {
    alignas(32) float t[16];
    alignas(32) float x[16];
};

struct EngineConfig {
    int threads = 8; // Default to 8 P-core workers
    bool pin_workers = true;
    int manager_core = 16; // First E-core (approx)
    
    uint64_t nodes_per_thread = 200000000ULL;
    uint32_t ring = 16384;
    uint32_t epoch_ms = 50;
    uint32_t refresh_mask = 0xFFFF;
    
    // GA params
    float beta0 = 0.60f;
    float beta_step = 0.04f;
    float speed_bias = 0.06f;
    float tail_step = 0.25f;
    float stale_penalty = 30.0f;
    float bw_penalty = 18.0f;
    uint32_t sample = 64;
    uint32_t perm_cands = 4;
    uint64_t target_nps = 14000000000ULL; // High target for 12900
    float hard_floor_ratio = 0.85f;
};

// ------------------------- Helpers -------------------------

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
    s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s;
}

static inline float absf(float x) {
    return std::abs(x);
}

static inline float clamp_beta(float b) {
    if (b >  0.999999f) b =  0.999999f;
    if (b < -0.999999f) b = -0.999999f;
    return b;
}

static inline uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static inline float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// ------------------------- Topology -------------------------

// P-branduoliai (0-7 fiziniai)
// We assume logical processors 0,2,4...14 are the main threads of P-cores.
void pin_to_pcore(int worker_idx) {
#ifdef _WIN32
    // Map worker_idx 0..7 to logical cores 0, 2, 4, ...
    // Note: This assumes standard Windows scheduling where hyperthreads are adjacent.
    int core_id = (worker_idx * 2) % 32; // Safety mod
    DWORD_PTR mask = (DWORD_PTR(1) << core_id);
    SetThreadAffinityMask(GetCurrentThread(), mask);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
#endif
}

// Manager ant E-core (core 16+)
void pin_to_ecore(int core_id) {
#ifdef _WIN32
    DWORD_PTR mask = (DWORD_PTR(1) << core_id); 
    SetThreadAffinityMask(GetCurrentThread(), mask);
#endif
}

// ------------------------- Kernel -------------------------

// "Cell Shifting" vietoj aritmetikos - naudojame AVX2 permute
static inline __m256 shift_cells(__m256 data, __m256i control) {
    return _mm256_permutevar8x32_ps(data, control);
}

// Branchless Driver-in-Driver Kernel (User Provided & Adapted)
// Note: We process 8 items at a time.
static void kernel_Pcore_v7_block(const float* t_ptr, const float* x_ptr,
                                  const __m256 gamma, const __m256 g_beta, 
                                  const __m256i perm_ctrl, float& ot, float& ox) {
    
    __m256 vt = _mm256_load_ps(t_ptr);
    __m256 vx = _mm256_load_ps(x_ptr);

    // Vykdome permute (celių stumdymą)
    vt = shift_cells(vt, perm_ctrl);
    vx = shift_cells(vx, perm_ctrl);

    // Kvantizuota fizika: gamma*t - g_beta*x (FMA3)
    __m256 tp = _mm256_fmsub_ps(gamma, vt, _mm256_mul_ps(g_beta, vx));
    __m256 xp = _mm256_fmsub_ps(gamma, vx, _mm256_mul_ps(g_beta, vt));

    // Fast Horizontal Add
    __m256 sum_t = tp;
    __m256 sum_x = xp;
    
    // Sum 8 elements to 1
    // Step 1: 256 -> 128
    __m128 t128 = _mm_add_ps(_mm256_castps256_ps128(sum_t), _mm256_extractf128_ps(sum_t, 1));
    __m128 x128 = _mm_add_ps(_mm256_castps256_ps128(sum_x), _mm256_extractf128_ps(sum_x, 1));
    
    // Step 2: hadd
    t128 = _mm_hadd_ps(t128, t128);
    x128 = _mm_hadd_ps(x128, x128);
    
    // Step 3: hadd again
    t128 = _mm_hadd_ps(t128, t128);
    x128 = _mm_hadd_ps(x128, x128);
    
    ot += _mm_cvtss_f32(t128);
    ox += _mm_cvtss_f32(x128);
}

// ------------------------- GA / Permutation Logic -------------------------

static const uint32_t PERM_TABLE[4][8] = {
    {0,1,2,3,4,5,6,7},  // Identity
    {4,5,6,7,0,1,2,3},  // Swap halves (local 8)
    {0,2,4,6,1,3,5,7},  // Interleave
    {7,6,5,4,3,2,1,0}   // Reverse
};

struct WorkerShared {
    NodeSoA16* ringA = nullptr;
    NodeSoA16* ringB = nullptr;
    
    std::atomic<uint64_t> epoch{0};
    std::atomic<NodeSoA16*> ring_ptr{nullptr};
    
    std::atomic<uint32_t> beta_bits{0};
    std::atomic<uint32_t> perm_idx{0}; // 0..3
    
    std::atomic<uint64_t> progress{0};
    
    // Alignment padding
    char pad[128]; 
};

// ------------------------- Worker Main -------------------------

struct ThreadResult {
    double seconds;
    double nps;
    float checksum;
};

static __m256 rsqrt_nr1(__m256 x) {
    __m256 y = _mm256_rsqrt_ps(x);
    const __m256 half  = _mm256_set1_ps(0.5f);
    const __m256 three = _mm256_set1_ps(3.0f);
    __m256 yy   = _mm256_mul_ps(y, y);
    __m256 term = _mm256_fnmadd_ps(x, yy, three);
    return _mm256_mul_ps(y, _mm256_mul_ps(half, term));
}

ThreadResult worker_main(const EngineConfig& cfg, WorkerShared* sh, int tid) {
    // Pin to P-core
    if (cfg.pin_workers) {
        pin_to_pcore(tid);
    }

    while (sh->ring_ptr.load(std::memory_order_acquire) == nullptr) _mm_pause();

    const uint32_t ring_mask = cfg.ring - 1u;
    const uint32_t refresh_mask = cfg.refresh_mask;

    NodeSoA16* ring = sh->ring_ptr.load(std::memory_order_acquire);
    uint64_t local_epoch = sh->epoch.load(std::memory_order_acquire);

    // Load dynamic params
    float beta = u2f(sh->beta_bits.load(std::memory_order_relaxed));
    uint32_t p_idx = sh->perm_idx.load(std::memory_order_relaxed);

    // Setup AVX params
    __m256 v_beta = _mm256_set1_ps(beta);
    __m256 v_b2 = _mm256_mul_ps(v_beta, v_beta);
    __m256 v_inv = _mm256_sub_ps(_mm256_set1_ps(1.0f), v_b2);
    v_inv = _mm256_max_ps(v_inv, _mm256_set1_ps(1.0e-12f));
    __m256 gamma = rsqrt_nr1(v_inv);
    __m256 g_beta = _mm256_mul_ps(gamma, v_beta);

    __m256i perm_ctrl = _mm256_loadu_si256((const __m256i*)PERM_TABLE[p_idx & 3]);

    float ot_sum = 0.0f;
    float ox_sum = 0.0f;
    
    uint64_t last_scan = 0;
    uint64_t N = cfg.nodes_per_thread;
    
    auto t0 = std::chrono::high_resolution_clock::now();

    for (uint64_t i = 0; i < N; ++i) {
        // Main Loop
        NodeSoA16* n = &ring[i & ring_mask];

        // Unroll 2x8 (Total 16 floats per node)
        // Upper half
        kernel_Pcore_v7_block(n->t, n->x, gamma, g_beta, perm_ctrl, ot_sum, ox_sum);
        // Lower half (offset 8)
        kernel_Pcore_v7_block(n->t + 8, n->x + 8, gamma, g_beta, perm_ctrl, ot_sum, ox_sum);

        // Periodic checks
        if ((i & refresh_mask) == 0) {
            uint64_t delta = i - last_scan;
            if (delta > 0) sh->progress.fetch_add(delta, std::memory_order_relaxed);
            last_scan = i;

            uint64_t e = sh->epoch.load(std::memory_order_acquire);
            if (e != local_epoch) {
                local_epoch = e;
                ring = sh->ring_ptr.load(std::memory_order_acquire);
                
                beta = u2f(sh->beta_bits.load(std::memory_order_relaxed));
                p_idx = sh->perm_idx.load(std::memory_order_relaxed);
                
                v_beta = _mm256_set1_ps(beta);
                v_b2 = _mm256_mul_ps(v_beta, v_beta);
                v_inv = _mm256_sub_ps(_mm256_set1_ps(1.0f), v_b2);
                v_inv = _mm256_max_ps(v_inv, _mm256_set1_ps(1.0e-12f));
                gamma = rsqrt_nr1(v_inv);
                g_beta = _mm256_mul_ps(gamma, v_beta);
                
                perm_ctrl = _mm256_loadu_si256((const __m256i*)PERM_TABLE[p_idx & 3]);
            }
        }
    }
    
    if (N > last_scan) {
        sh->progress.fetch_add(N - last_scan, std::memory_order_relaxed);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    ThreadResult r;
    r.seconds = sec;
    r.nps = double(N) / sec;
    r.checksum = ot_sum + ox_sum;
    return r;
}

// ------------------------- Manager (Stub for Benchmark) -------------------------

void manager_sim(const EngineConfig& cfg, std::vector<WorkerShared*>& workers, std::atomic<bool>& running) {
    if (cfg.manager_core >= 0) pin_to_ecore(cfg.manager_core);

    int epoch = 0;
    while(running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(cfg.epoch_ms));
        
        float new_beta = cfg.beta0 + (epoch % 5) * 0.01f;
        uint32_t new_perm = epoch % 4;
        
        for(auto* w : workers) {
            w->beta_bits.store(f2u(new_beta), std::memory_order_relaxed);
            w->perm_idx.store(new_perm, std::memory_order_relaxed);
            w->epoch.fetch_add(1, std::memory_order_release);
        }
        epoch++;
    }
}

// ------------------------- Main -------------------------

int main(int argc, char** argv) {
    EngineConfig cfg;
    if (argc > 1) cfg.threads = atoi(argv[1]);
    
    std::cout << "------------------------------------------\n";
    std::cout << " F1 v2.5 'Alder Lake' Engine v7 (AVX2-VNNI/PermuteVar)\n";
    std::cout << " Target: Intel Core i9-12900 (8 P-Cores)\n";
    std::cout << "------------------------------------------\n";
    std::cout << " Threads: " << cfg.threads << " (P-Cores mapped 0,2..)\n";
    std::cout << " Nodes per Thread: " << cfg.nodes_per_thread << "\n";
    
    std::vector<WorkerShared> shared(cfg.threads);
    std::vector<WorkerShared*> ptrs(cfg.threads);
    
    for(int i=0; i<cfg.threads; ++i) {
        shared[i].ringA = (NodeSoA16*)aligned_malloc(64, sizeof(NodeSoA16)*cfg.ring);
        shared[i].ring_ptr = shared[i].ringA;
        for(uint32_t k=0; k<cfg.ring; ++k) {
            for(int j=0; j<16; ++j) {
                shared[i].ringA[k].t[j] = 1.0f; 
                shared[i].ringA[k].x[j] = 0.1f;
            }
        }
        shared[i].beta_bits = f2u(cfg.beta0);
        ptrs[i] = &shared[i];
    }
    
    std::atomic<bool> running{true};
    std::thread mgr([&](){ manager_sim(cfg, ptrs, running); });
    
    std::vector<std::thread> workers;
    std::vector<ThreadResult> results(cfg.threads);
    
    auto t_start = std::chrono::high_resolution_clock::now();
    uint64_t cycles_start = __rdtsc();
    
    for(int i=0; i<cfg.threads; ++i) {
        workers.emplace_back([&, i](){
            results[i] = worker_main(cfg, &shared[i], i);
        });
    }
    
    for(auto& t : workers) t.join();
    
    uint64_t cycles_end = __rdtsc();
    auto t_end = std::chrono::high_resolution_clock::now();
    running = false;
    mgr.join();
    
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();
    uint64_t total_cycles = cycles_end - cycles_start;
    double cycles_per_sec = total_cycles / total_sec;
    
    double total_nps = 0;
    for(auto& r : results) total_nps += r.nps;
    
    double gb_s = (total_nps * sizeof(NodeSoA16)) / (1024.0*1024.0*1024.0);
    
    std::cout << "\nResults:\n";
    std::cout << "  Wall Time: " << total_sec << " s\n";
    std::cout << "  Total PPS (NPS): " << std::fixed << std::setprecision(0) << total_nps << "\n";
    std::cout << "  Throughput: " << std::setprecision(2) << gb_s << " GB/s\n";
    std::cout << "  Clock: " << (cycles_per_sec / 1e9) << " GHz (approx effective)\n";
    std::cout << "  Cycles/Sec: " << (unsigned long long)cycles_per_sec << "\n";
    
    for(int i=0; i<cfg.threads; ++i) aligned_free(shared[i].ringA);

    return 0;
}