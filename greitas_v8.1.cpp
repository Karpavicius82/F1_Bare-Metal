// engine_alderlake_v7.cpp
// Optimized for Intel Core i9-12900 (Alder Lake)
// Features:
// 1. Topology Awareness: P-cores for Workers, E-core for Manager.
// 2. Cell Shifting: AVX2 permutevar8x32 instead of memory shuffle.
// 3. Driver-in-Driver GA with Backpressure.
// 4. Vector Filtering: accumulate tp/xp in __m256 and reduce periodically
// (amortized filter cost).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX
#include <intrin.h> // For __rdtscp
#include <windows.h>

#else
#include <pthread.h>
#include <sched.h>
#include <x86intrin.h> // For __rdtscp
#endif

// ------------------------- Configuration -------------------------

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
  uint64_t target_nps = 14000000000ULL;
  float hard_floor_ratio = 0.85f;
};

// ------------------------- Helpers -------------------------

static inline void *aligned_malloc(size_t align, size_t size) {
#if defined(_WIN32)
  return _aligned_malloc(size, align);
#else
  void *p = nullptr;
  if (posix_memalign(&p, align, size) != 0)
    return nullptr;
  return p;
#endif
}

static inline void aligned_free(void *p) {
#if defined(_WIN32)
  _aligned_free(p);
#else
  free(p);
#endif
}

static inline uint32_t xorshift32(uint32_t &s) {
  s ^= s << 13;
  s ^= s >> 17;
  s ^= s << 5;
  return s;
}

static inline float clamp_beta(float b) {
  if (b > 0.999999f)
    b = 0.999999f;
  if (b < -0.999999f)
    b = -0.999999f;
  return b;
}

static inline uint32_t f2u(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}
static inline float u2f(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// ------------------------- Topology -------------------------

void pin_to_pcore(int worker_idx) {
#ifdef _WIN32
  int core_id = (worker_idx * 2) % 32;
  DWORD_PTR mask = (DWORD_PTR(1) << core_id);
  SetThreadAffinityMask(GetCurrentThread(), mask);
  SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
#endif
}

void pin_to_ecore(int core_id) {
#ifdef _WIN32
  DWORD_PTR mask = (DWORD_PTR(1) << core_id);
  SetThreadAffinityMask(GetCurrentThread(), mask);
#endif
}

// ------------------------- Kernel -------------------------

static inline __m256 shift_cells(__m256 data, __m256i control) {
  return _mm256_permutevar8x32_ps(data, control);
}

// Horizontal sum for __m256 (8 floats -> scalar)
static inline float hsum256_ps(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  return _mm_cvtss_f32(s);
}

// Vector-accumulating kernel: no per-block scalar reduction (filter is
// amortized)
static inline void kernel_Pcore_v7_block_vec(const float *t_ptr,
                                             const float *x_ptr,
                                             const __m256 gamma,
                                             const __m256 g_beta,
                                             const __m256i perm_ctrl,
                                             __m256 &acc_t, __m256 &acc_x) {

  __m256 vt = _mm256_load_ps(t_ptr);
  __m256 vx = _mm256_load_ps(x_ptr);

  vt = shift_cells(vt, perm_ctrl);
  vx = shift_cells(vx, perm_ctrl);

  __m256 tp = _mm256_fmsub_ps(gamma, vt, _mm256_mul_ps(g_beta, vx));
  __m256 xp = _mm256_fmsub_ps(gamma, vx, _mm256_mul_ps(g_beta, vt));

  acc_t = _mm256_add_ps(acc_t, tp);
  acc_x = _mm256_add_ps(acc_x, xp);
}

// ------------------------- GA / Permutation Logic -------------------------

static const uint32_t PERM_TABLE[4][8] = {{0, 1, 2, 3, 4, 5, 6, 7},
                                          {4, 5, 6, 7, 0, 1, 2, 3},
                                          {0, 2, 4, 6, 1, 3, 5, 7},
                                          {7, 6, 5, 4, 3, 2, 1, 0}};

struct WorkerShared {
  NodeSoA16 *ringA = nullptr;
  NodeSoA16 *ringB = nullptr;

  std::atomic<uint64_t> epoch{0};
  std::atomic<NodeSoA16 *> ring_ptr{nullptr};

  std::atomic<uint32_t> beta_bits{0};
  std::atomic<uint32_t> perm_idx{0}; // 0..3

  std::atomic<uint64_t> progress{0};

  char pad[128];
};

// ------------------------- Worker Main -------------------------

struct ThreadResult {
  double seconds;
  double nps;
  float checksum;
  uint64_t cycles;        // NEW
  double cycles_per_node; // NEW
};

static inline uint64_t rdtscp_serialized() {
#if defined(_MSC_VER)
  unsigned int aux = 0;
  return __rdtscp(&aux);
#else
  unsigned int aux = 0;
  return __rdtscp(&aux);
#endif
}

struct StartGate {
  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
};

static __m256 rsqrt_nr1(__m256 x) {
  __m256 y = _mm256_rsqrt_ps(x);
  const __m256 half = _mm256_set1_ps(0.5f);
  const __m256 three = _mm256_set1_ps(3.0f);
  __m256 yy = _mm256_mul_ps(y, y);
  __m256 term = _mm256_fnmadd_ps(x, yy, three);
  return _mm256_mul_ps(y, _mm256_mul_ps(half, term));
}

ThreadResult worker_main(const EngineConfig &cfg, WorkerShared *sh, int tid,
                         StartGate *gate) {
  if (cfg.pin_workers)
    pin_to_pcore(tid);

  while (sh->ring_ptr.load(std::memory_order_acquire) == nullptr)
    _mm_pause();

  // --- start barrier ---
  gate->ready.fetch_add(1, std::memory_order_acq_rel);
  while (!gate->go.load(std::memory_order_acquire))
    _mm_pause();

  // measure inside worker (reliable)
  uint64_t c0 = rdtscp_serialized();
  auto t0 = std::chrono::steady_clock::now();

  const uint32_t ring_mask = cfg.ring - 1u;
  const uint32_t refresh_mask = cfg.refresh_mask;

  NodeSoA16 *ring = sh->ring_ptr.load(std::memory_order_acquire);
  uint64_t local_epoch = sh->epoch.load(std::memory_order_acquire);

  float beta = u2f(sh->beta_bits.load(std::memory_order_relaxed));
  uint32_t p_idx = sh->perm_idx.load(std::memory_order_relaxed);

  __m256 v_beta = _mm256_set1_ps(beta);
  __m256 v_b2 = _mm256_mul_ps(v_beta, v_beta);
  __m256 v_inv = _mm256_sub_ps(_mm256_set1_ps(1.0f), v_b2);
  v_inv = _mm256_max_ps(v_inv, _mm256_set1_ps(1.0e-12f));
  __m256 gamma = rsqrt_nr1(v_inv);
  __m256 g_beta = _mm256_mul_ps(gamma, v_beta);

  __m256i perm_ctrl =
      _mm256_loadu_si256((const __m256i *)PERM_TABLE[p_idx & 3]);

  // Vector accumulators (raw signal kept in SIMD regs)
  __m256 otv = _mm256_setzero_ps();
  __m256 oxv = _mm256_setzero_ps();

  // Scalar "clean" signal (fitness-like aggregate)
  float ot_sum = 0.0f;
  float ox_sum = 0.0f;

  uint64_t last_scan = 0;
  uint64_t N = cfg.nodes_per_thread;

  for (uint64_t i = 0; i < N; ++i) {
    NodeSoA16 *n = &ring[i & ring_mask];

    // Unroll 2x8 (Total 16 floats per node)
    kernel_Pcore_v7_block_vec(n->t, n->x, gamma, g_beta, perm_ctrl, otv, oxv);
    kernel_Pcore_v7_block_vec(n->t + 8, n->x + 8, gamma, g_beta, perm_ctrl, otv,
                              oxv);

    // Periodic checks (amortized filter + GA update)
    if ((i & refresh_mask) == 0) {
      // 1) Progress accounting (unchanged)
      uint64_t delta = i - last_scan;
      if (delta > 0)
        sh->progress.fetch_add(delta, std::memory_order_relaxed);
      last_scan = i;

      // 2) Vector -> scalar reduction (this is the filter; amortized)
      ot_sum += hsum256_ps(otv);
      ox_sum += hsum256_ps(oxv);
      otv = _mm256_setzero_ps();
      oxv = _mm256_setzero_ps();

      // 3) GA/epoch update (unchanged semantics)
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

        perm_ctrl = _mm256_loadu_si256((const __m256i *)PERM_TABLE[p_idx & 3]);
      }
    }
  }

  if (N > last_scan)
    sh->progress.fetch_add(N - last_scan, std::memory_order_relaxed);

  // Final flush of vector accumulators (pay filter cost once)
  ot_sum += hsum256_ps(otv);
  ox_sum += hsum256_ps(oxv);

  auto t1 = std::chrono::steady_clock::now();
  uint64_t c1 = rdtscp_serialized();
  double sec = std::chrono::duration<double>(t1 - t0).count();

  ThreadResult r;
  r.seconds = sec;
  r.nps = double(N) / sec;
  r.checksum = ot_sum + ox_sum;
  r.cycles = (c1 - c0);
  r.cycles_per_node = double(r.cycles) / double(N);
  return r;
}

// ------------------------- Manager (Stub for Benchmark)
// -------------------------

void manager_sim(const EngineConfig &cfg, std::vector<WorkerShared *> &workers,
                 std::atomic<bool> &running) {
  if (cfg.manager_core >= 0)
    pin_to_ecore(cfg.manager_core);

  int epoch = 0;
  while (running) {
    std::this_thread::sleep_for(std::chrono::milliseconds(cfg.epoch_ms));

    float new_beta = cfg.beta0 + (epoch % 5) * 0.01f;
    uint32_t new_perm = epoch % 4;

    for (auto *w : workers) {
      w->beta_bits.store(f2u(new_beta), std::memory_order_relaxed);
      w->perm_idx.store(new_perm, std::memory_order_relaxed);
      w->epoch.fetch_add(1, std::memory_order_release);
    }
    epoch++;
  }
}

// ------------------------- Main -------------------------

int main(int argc, char **argv) {
  EngineConfig cfg;
  if (argc > 1)
    cfg.threads = atoi(argv[1]);
  if (argc > 2)
    cfg.nodes_per_thread =
        (uint64_t)std::strtoull(argv[2], nullptr, 10); // test-friendly

  std::cout << "------------------------------------------\n";
  std::cout << " F1 v2.5 'Alder Lake' Engine v7 (AVX2-VNNI/PermuteVar)\n";
  std::cout << " Target: Intel Core i9-12900 (8 P-Cores)\n";
  std::cout << "------------------------------------------\n";
  std::cout << " Threads: " << cfg.threads << " (P-Cores mapped 0,2..)\n";
  std::cout << " Nodes per Thread: " << cfg.nodes_per_thread << "\n";
  std::cout << " Ring: " << cfg.ring << "  RefreshMask: 0x" << std::hex
            << cfg.refresh_mask << std::dec << "\n";

  std::vector<WorkerShared> shared(cfg.threads);
  std::vector<WorkerShared *> ptrs(cfg.threads);

  for (int i = 0; i < cfg.threads; ++i) {
    shared[i].ringA =
        (NodeSoA16 *)aligned_malloc(64, sizeof(NodeSoA16) * cfg.ring);
    shared[i].ring_ptr = shared[i].ringA;

    std::mt19937 rng(1337 + i);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (uint32_t k = 0; k < cfg.ring; ++k) {
      for (int j = 0; j < 16; ++j) {
        shared[i].ringA[k].t[j] = dist(rng);
        shared[i].ringA[k].x[j] = dist(rng);
      }
    }

    shared[i].beta_bits = f2u(cfg.beta0);
    ptrs[i] = &shared[i];
  }

  std::atomic<bool> running{true};
  std::thread mgr([&]() { manager_sim(cfg, ptrs, running); });

  StartGate gate;
  std::vector<std::thread> workers;
  std::vector<ThreadResult> results(cfg.threads);

  for (int i = 0; i < cfg.threads; ++i) {
    workers.emplace_back(
        [&, i]() { results[i] = worker_main(cfg, &shared[i], i, &gate); });
  }

  // wait until all workers are ready, then start simultaneously
  while (gate.ready.load(std::memory_order_acquire) != cfg.threads)
    _mm_pause();
  auto t_start = std::chrono::steady_clock::now();
  gate.go.store(true, std::memory_order_release);

  for (auto &t : workers)
    t.join();

  auto t_end = std::chrono::steady_clock::now();

  running = false;
  mgr.join();

  double total_sec = std::chrono::duration<double>(t_end - t_start).count();

  double total_nps = 0;
  float total_chk = 0.0f;
  for (auto &r : results) {
    total_nps += r.nps;
    total_chk += r.checksum;
  }

  double gb_s = (total_nps * sizeof(NodeSoA16)) / (1024.0 * 1024.0 * 1024.0);

  std::cout << "\nResults:\n";
  std::cout << "  Wall Time: " << total_sec << " s\n";
  std::cout << "  Total PPS (NPS): " << std::fixed << std::setprecision(0)
            << total_nps << "\n";
  std::cout << "  Throughput(eq): " << std::setprecision(2) << gb_s
            << " GB/s\n";
  std::cout << "  Checksum: " << std::setprecision(3) << total_chk << "\n";

  double min_cpn = 1e100, max_cpn = 0;
  for (auto &r : results) {
    min_cpn = std::min(min_cpn, r.cycles_per_node);
    max_cpn = std::max(max_cpn, r.cycles_per_node);
  }
  std::cout << "  Cycles/Node (min..max): " << min_cpn << " .. " << max_cpn
            << "\n";

  // Sanity check
  uint64_t expected =
      cfg.nodes_per_thread * cfg.threads; // Total expected nodes
  uint64_t got_progress = 0;
  for (int i = 0; i < cfg.threads; ++i)
    got_progress += shared[i].progress.load();

  std::cout << "  Sanity Check (Progress): Expected=" << expected
            << ", Got=" << got_progress << "\n";
  if (got_progress != expected)
    std::cout << "  WARNING: Progress mismatch! Bench invalid.\n";

  for (int i = 0; i < cfg.threads; ++i)
    aligned_free(shared[i].ringA);
  return 0;
}
