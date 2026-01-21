// engine_prod_ga3.cpp
// Gamybinė versija: maksimalus tankumas + maitinimas su 3x GA integracija (be LUT).
//
// Tikslai:
// 1) Hot-path (per node) – AVX2/FMA branchless kernel, be didelio DRAM working-set (cache-resident ring).
// 2) Duomenų tankumas – SoA, gate+weights įkepti prepack'e (išjungti kaimynai = 0), optional FP16 ring (jei įjungsi).
// 3) 3x GA:
//    GA#1 (Compute): parenka beta (Lorentz param) per batch.
//    GA#2 (Density): parenka K_eff ∈ {8,16} + tail_scale (kiek „uodegos“ 8 lane prisideda).
//    GA#3 (Feed): parenka perm_id (kaip perstatyti/įpakuoti kaimynus į top8) + refresh_shift (kaip dažnai maitinti ring).
//
// Pastaba: Hot-path neturi data-dependent branch'ų; GA ir refill yra control-plane (leidžiama).
//
// Build (Linux):
//   g++ -O3 -std=c++20 -march=native -mavx2 -mfma engine_prod_ga3.cpp -lpthread -o engine
//
// Build (Haswell conservative):
//   g++ -O3 -std=c++20 -march=haswell -mavx2 -mfma engine_prod_ga3.cpp -lpthread -o engine
//
// Build (Win11 MSYS2/MinGW ucrt64):
//   g++ -O3 -std=c++20 -march=native -mavx2 -mfma engine_prod_ga3.cpp -o engine.exe
//
// Run:
//   ./engine --threads 8 --nodes-per-thread 100000000 --ring 256 --ga-batch 4096
//
// Optional switches:
//   --no-pin
//   --prefetch 1
//   --speed-bias 0.02
//   --perm-cands 4
//   --refresh-min 4 --refresh-max 10
//   --fp16 0/1  (jei kompiliatorius palaiko F16C; Haswell/Skylake turi)

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
#  error "Reikia AVX2+FMA. Kompiliuok su -mavx2 -mfma ir tinkamu -march=..."
#endif

// ---------------------------- Platform helpers ----------------------------

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

// ---------------------------- RNG / scalar utils ----------------------------

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

// ---------------------------- Engine config ----------------------------

static constexpr int KMAX = 16;

struct EngineConfig {
    int threads = (int)std::thread::hardware_concurrency();
    bool pin_threads = true;

    uint64_t nodes_per_thread = 100000000ULL;

    uint32_t ring_nodes = 256;      // power-of-two; cache-resident per thread
    uint32_t ga_batch   = 4096;     // power-of-two; GA update period

    uint32_t sample = 32;           // GA sample size
    float beta_step = 0.04f;

    // 3x GA knobs
    float tail_scale_step = 0.25f;  // density GA: tail contribution levels: 0, step, 2*step (clamped)
    uint32_t perm_cands = 4;        // number of packing/permutation candidates (<=4 in this file)
    uint32_t refresh_min_shift = 4; // update every 2^shift iters (min)
    uint32_t refresh_max_shift = 10;// update every 2^shift iters (max)

    // Guard: jei 8 lane per daug gadina kokybę, priverstinai grįžti į 16 kitam batch
    float degrade_ratio = 0.97f;

    // Multi-objective: kuo didesnis, tuo labiau GA rinks greitį (mažesnį keff ir retesnį refresh)
    float speed_bias = 0.02f;

    // Prefetch hint (0/1)
    int prefetch = 0;

    // Optional FP16 ring storage (needs F16C at compile+runtime)
    int fp16 = 0;

    float initial_beta = 0.6f;
};

// ---------------------------- Data types (FP32 / FP16 ring) ----------------------------

// Gate+weights baked in: disabled neighbors are 0.0f. Hot-path neturi gate masyvų.

struct alignas(32) NodeFP32 {
    float t[KMAX];
    float x[KMAX];
};

#if defined(__F16C__)
struct alignas(32) NodeFP16 {
    uint16_t t[KMAX];
    uint16_t x[KMAX];
};
#endif

// ---------------------------- Permutation candidates (packing) ----------------------------
// Idėja: perm perstatymas nusprendžia, kurie kaimynai patenka į top8.
// Production: čia statytum top-K reitingavimo/atrankos rezultatą.
// Šiame faile – 4 fiksuoti variantai.

static constexpr uint8_t PERM4[4][KMAX] = {
    // 0: identity
    { 0,1,2,3,4,5,6,7, 8,9,10,11,12,13,14,15 },
    // 1: bring high indices to front (worse if "important" are 0..7)
    { 8,9,10,11,12,13,14,15, 0,1,2,3,4,5,6,7 },
    // 2: interleave
    { 0,8,1,9,2,10,3,11, 4,12,5,13,6,14,7,15 },
    // 3: reverse
    { 15,14,13,12,11,10,9,8, 7,6,5,4,3,2,1,0 }
};

// ---------------------------- Math / kernel helpers ----------------------------

static inline __m256 rsqrt_nr1(__m256 x) {
    __m256 y = _mm256_rsqrt_ps(x);
    const __m256 half  = _mm256_set1_ps(0.5f);
    const __m256 three = _mm256_set1_ps(3.0f);
    __m256 yy   = _mm256_mul_ps(y, y);
    __m256 term = _mm256_fnmadd_ps(x, yy, three); // 3 - x*y*y
    return _mm256_mul_ps(y, _mm256_mul_ps(half, term));
}

// Horizontal sum 8 floats -> scalar (no store)
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

// Kernel: keff=8 (loads 2x256b, FMAs, reduce)
static inline void kernel8_fp32(const NodeFP32* __restrict n, __m256 gamma, __m256 g_beta,
                                float& out_t, float& out_x) {
    __m256 t = _mm256_load_ps(n->t + 0);
    __m256 x = _mm256_load_ps(n->x + 0);
    __m256 tp = _mm256_fnmadd_ps(g_beta, x, _mm256_mul_ps(gamma, t));
    __m256 xp = _mm256_fnmadd_ps(g_beta, t, _mm256_mul_ps(gamma, x));
    out_t = hsum8_ps(tp);
    out_x = hsum8_ps(xp);
}

// Kernel: keff=16
static inline void kernel16_fp32(const NodeFP32* __restrict n, __m256 gamma, __m256 g_beta,
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

#if defined(__F16C__)
// FP16 load: 8 half -> 8 float
static inline __m256 load8_fp16_to_ps(const uint16_t* p) {
    __m128i h = _mm_loadu_si128((const __m128i*)p);
    return _mm256_cvtph_ps(h);
}
static inline void kernel8_fp16(const NodeFP16* __restrict n, __m256 gamma, __m256 g_beta,
                                float& out_t, float& out_x) {
    __m256 t = load8_fp16_to_ps(n->t + 0);
    __m256 x = load8_fp16_to_ps(n->x + 0);
    __m256 tp = _mm256_fnmadd_ps(g_beta, x, _mm256_mul_ps(gamma, t));
    __m256 xp = _mm256_fnmadd_ps(g_beta, t, _mm256_mul_ps(gamma, x));
    out_t = hsum8_ps(tp);
    out_x = hsum8_ps(xp);
}
static inline void kernel16_fp16(const NodeFP16* __restrict n, __m256 gamma, __m256 g_beta,
                                 float& out_t, float& out_x) {
    __m256 t0 = load8_fp16_to_ps(n->t + 0);
    __m256 x0 = load8_fp16_to_ps(n->x + 0);
    __m256 t1 = load8_fp16_to_ps(n->t + 8);
    __m256 x1 = load8_fp16_to_ps(n->x + 8);

    __m256 tp0 = _mm256_fnmadd_ps(g_beta, x0, _mm256_mul_ps(gamma, t0));
    __m256 xp0 = _mm256_fnmadd_ps(g_beta, t0, _mm256_mul_ps(gamma, x0));
    __m256 tp1 = _mm256_fnmadd_ps(g_beta, x1, _mm256_mul_ps(gamma, t1));
    __m256 xp1 = _mm256_fnmadd_ps(g_beta, t1, _mm256_mul_ps(gamma, x1));

    out_t = hsum8_ps(_mm256_add_ps(tp0, tp1));
    out_x = hsum8_ps(_mm256_add_ps(xp0, xp1));
}

// FP32->FP16 store 8 floats
static inline void store8_ps_to_fp16(uint16_t* p, __m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128i hlo = _mm_cvtps_ph(lo, 0);
    __m128i hhi = _mm_cvtps_ph(hi, 0);
    _mm_storeu_si128((__m128i*)p, _mm_unpacklo_epi64(hlo, hhi)); // 16 bytes
}
#endif

static inline float fitness_proxy(float out_t, float out_x) {
    return out_t - absf(out_x);
}

// ---------------------------- Genome (3x GA result) ----------------------------

struct Genome {
    float beta = 0.6f;        // GA#1
    int   keff = 8;           // GA#2 (8 or 16)
    float tail_scale = 0.0f;  // GA#2 (0..1): applied to indices 8..15 before storing
    uint32_t perm_id = 0;     // GA#3 (0..3)
    uint32_t refresh_shift = 6; // GA#3: update every 2^shift iters
};

// ---------------------------- Refill / Prepack (feed-plane) ----------------------------
//
// Gamyboje: čia įdedi tikrą ingest: iš NIC/PCIe arba adjacency -> topK -> SoA pack.
// Šitam faile: generatorius su „importance“: indeksai 0..7 turi didesnę amplitudę.
// Permutation realiai keičia, kas patenka į top8.

static inline void generate_raw16(float* t_raw, float* x_raw, uint32_t& rng) {
    for (int i = 0; i < KMAX; ++i) {
        uint32_t r = xorshift32(rng);
        float noise = float(int(r & 255u) - 128) * 1e-3f;

        // "importance": first half bigger
        float amp = (i < 8) ? 1.0f : 0.25f;

        t_raw[i] = amp * (10.0f + float(i) * 0.05f + noise);
        x_raw[i] = amp * ( 1.0f + float(i) * 0.02f);
    }
}

static inline void prepack_fp32(NodeFP32& dst, const float* t_raw, const float* x_raw, const Genome& g) {
    const uint8_t* perm = PERM4[g.perm_id & 3u];

    // gate baked-in via keff and tail_scale:
    // - if keff=8: indices >=8 => 0
    // - else: indices >=8 scaled by tail_scale
    const float tail = (g.keff == 16) ? g.tail_scale : 0.0f;

    for (int k = 0; k < KMAX; ++k) {
        const int src = perm[k];
        float tk = t_raw[src];
        float xk = x_raw[src];

        float s = 1.0f;
        if (k >= 8) s = tail;        // 0..1
        // if tail==0 => baked gate

        dst.t[k] = tk * s;
        dst.x[k] = xk * s;
    }
}

#if defined(__F16C__)
static inline void prepack_fp16(NodeFP16& dst, const float* t_raw, const float* x_raw, const Genome& g) {
    const uint8_t* perm = PERM4[g.perm_id & 3u];
    const float tail = (g.keff == 16) ? g.tail_scale : 0.0f;

    alignas(32) float ttmp[16];
    alignas(32) float xtmp[16];

    for (int k = 0; k < KMAX; ++k) {
        const int src = perm[k];
        float tk = t_raw[src];
        float xk = x_raw[src];
        float s = 1.0f;
        if (k >= 8) s = tail;
        ttmp[k] = tk * s;
        xtmp[k] = xk * s;
    }

    // store as halves (8+8)
    __m256 t0 = _mm256_load_ps(ttmp + 0);
    __m256 t1 = _mm256_load_ps(ttmp + 8);
    __m256 x0 = _mm256_load_ps(xtmp + 0);
    __m256 x1 = _mm256_load_ps(xtmp + 8);

    // pack: 8 floats -> 8 halves each
    __m128i ht0 = _mm256_cvtps_ph(t0, 0);
    __m128i ht1 = _mm256_cvtps_ph(t1, 0);
    __m128i hx0 = _mm256_cvtps_ph(x0, 0);
    __m128i hx1 = _mm256_cvtps_ph(x1, 0);

    _mm_storeu_si128((__m128i*)(dst.t + 0), ht0);
    _mm_storeu_si128((__m128i*)(dst.t + 8), ht1);
    _mm_storeu_si128((__m128i*)(dst.x + 0), hx0);
    _mm_storeu_si128((__m128i*)(dst.x + 8), hx1);
}
#endif

// Refill one slot according to genome
template <typename NodeT>
static inline void refill_slot(NodeT& slot, const Genome& g, uint32_t& rng) {
    alignas(32) float t_raw[16];
    alignas(32) float x_raw[16];
    generate_raw16(t_raw, x_raw, rng);

    if constexpr (std::is_same<NodeT, NodeFP32>::value) {
        prepack_fp32(slot, t_raw, x_raw, g);
    }
#if defined(__F16C__)
    else if constexpr (std::is_same<NodeT, NodeFP16>::value) {
        prepack_fp16(slot, t_raw, x_raw, g);
    }
#endif
}

// Burst refill (kad genome pasikeitimas greičiau „įsigertų“ į ring)
template <typename NodeT>
static inline void burst_refill(NodeT* ring, uint32_t ring_mask, const Genome& g,
                                uint32_t& rng, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t idx = xorshift32(rng) & ring_mask;
        refill_slot(ring[idx], g, rng);
    }
}

// ---------------------------- 3x GA selection ----------------------------
//
// Multi-objective score:
//   score_adj = fitness - speed_bias*(keff_cost + refresh_cost)
// where:
//   keff_cost = 1 if 16 else 0
//   refresh_cost ~ 1 / 2^shift (dažnesnis mait. = brangiau)

static inline float speed_penalty(const EngineConfig& cfg, int keff, uint32_t refresh_shift) {
    const float keff_cost = (keff == 16) ? 1.0f : 0.0f;
    const float refresh_cost = 1.0f / float(1u << (refresh_shift > 30 ? 30 : refresh_shift));
    return cfg.speed_bias * (keff_cost + refresh_cost);
}

// GA#1: pick beta among 4
template <typename NodeT>
static inline float ga_pick_beta(const EngineConfig& cfg, const NodeT* ring, uint32_t ring_mask,
                                 const Genome& cur, uint32_t& rng)
{
    float cand[4] = { cur.beta,
                      cur.beta + cfg.beta_step,
                      cur.beta - cfg.beta_step,
                      cur.beta + 2.0f * cfg.beta_step };

    float best_beta = clamp_beta(cand[0]);
    float best = -1.0e30f;

    // sample indices from ring
    for (int bi = 0; bi < 4; ++bi) {
        float b = clamp_beta(cand[bi]);
        __m256 gamma, g_beta;
        compute_gamma(b, gamma, g_beta);

        float s = 0.0f;
        for (uint32_t i = 0; i < cfg.sample; ++i) {
            uint32_t idx = xorshift32(rng) & ring_mask;
            float ot, ox;
            if constexpr (std::is_same<NodeT, NodeFP32>::value) {
                if (cur.keff == 8) kernel8_fp32(&ring[idx], gamma, g_beta, ot, ox);
                else               kernel16_fp32(&ring[idx], gamma, g_beta, ot, ox);
            }
#if defined(__F16C__)
            else {
                if (cur.keff == 8) kernel8_fp16(&ring[idx], gamma, g_beta, ot, ox);
                else               kernel16_fp16(&ring[idx], gamma, g_beta, ot, ox);
            }
#endif
            s += fitness_proxy(ot, ox);
        }

        // no speed penalty here (beta selection only), keep it quality-driven
        if (s > best) { best = s; best_beta = b; }
    }
    return best_beta;
}

// GA#2: pick density (keff + tail_scale)
// Candidates: (8,0) and (16, tail_scale in {step, 2*step})
static inline void ga_pick_density(const EngineConfig& cfg, const NodeFP32* ring, uint32_t ring_mask,
                                  Genome& g, uint32_t& rng)
{
    // evaluate using current ring (already packed). Density change affects future refills;
    // still we choose based on current compute + speed pressure.
    float tail1 = cfg.tail_scale_step;
    float tail2 = cfg.tail_scale_step * 2.0f;
    if (tail1 > 1.0f) tail1 = 1.0f;
    if (tail2 > 1.0f) tail2 = 1.0f;

    struct D { int keff; float tail; };
    D cand[3] = { {8, 0.0f}, {16, tail1}, {16, tail2} };

    float best = -1.0e30f;
    int best_keff = g.keff;
    float best_tail = g.tail_scale;

    __m256 gamma, g_beta;
    compute_gamma(g.beta, gamma, g_beta);

    for (int ci = 0; ci < 3; ++ci) {
        int keff = cand[ci].keff;
        float s = 0.0f;
        for (uint32_t i = 0; i < cfg.sample; ++i) {
            uint32_t idx = xorshift32(rng) & ring_mask;
            float ot, ox;
            if (keff == 8) kernel8_fp32(&ring[idx], gamma, g_beta, ot, ox);
            else           kernel16_fp32(&ring[idx], gamma, g_beta, ot, ox);
            s += fitness_proxy(ot, ox);
        }
        s -= cfg.speed_bias * ((keff == 16) ? 1.0f : 0.0f); // density speed pressure

        if (s > best) {
            best = s;
            best_keff = keff;
            best_tail = cand[ci].tail;
        }
    }

    g.keff = best_keff;
    g.tail_scale = best_tail;
}

// GA#3: pick feed policy (perm_id + refresh_shift) using scratch generation + score
template <typename NodeT>
static inline void ga_pick_feed(const EngineConfig& cfg, Genome& g, uint32_t& rng)
{
    // candidates: perm in [0..perm_cands-1], refresh_shift in [min..max] but only 3 around current for cost
    uint32_t permN = cfg.perm_cands;
    if (permN == 0) permN = 1;
    if (permN > 4) permN = 4;

    uint32_t shifts[3];
    uint32_t cur = g.refresh_shift;
    uint32_t mn = cfg.refresh_min_shift;
    uint32_t mx = cfg.refresh_max_shift;
    if (mn > mx) { uint32_t t = mn; mn = mx; mx = t; }
    if (cur < mn) cur = mn;
    if (cur > mx) cur = mx;

    shifts[0] = (cur > mn) ? (cur - 1) : cur;
    shifts[1] = cur;
    shifts[2] = (cur < mx) ? (cur + 1) : cur;

    __m256 gamma, g_beta;
    compute_gamma(g.beta, gamma, g_beta);

    float best = -1.0e30f;
    uint32_t best_perm = g.perm_id;
    uint32_t best_shift = g.refresh_shift;

    // scratch nodes
    std::vector<NodeT> scratch(cfg.sample);

    for (uint32_t p = 0; p < permN; ++p) {
        for (int si = 0; si < 3; ++si) {
            uint32_t sh = shifts[si];

            Genome gg = g;
            gg.perm_id = p;
            gg.refresh_shift = sh;

            // generate scratch according to candidate feed policy
            uint32_t rlocal = rng ^ (p * 0xA5A5A5A5u) ^ (sh * 0x3C6EF35Fu);
            for (uint32_t i = 0; i < cfg.sample; ++i) refill_slot(scratch[i], gg, rlocal);

            // evaluate fitness on scratch
            float s = 0.0f;
            for (uint32_t i = 0; i < cfg.sample; ++i) {
                float ot, ox;
                if constexpr (std::is_same<NodeT, NodeFP32>::value) {
                    if (g.keff == 8) kernel8_fp32(&scratch[i], gamma, g_beta, ot, ox);
                    else             kernel16_fp32(&scratch[i], gamma, g_beta, ot, ox);
                }
#if defined(__F16C__)
                else {
                    if (g.keff == 8) kernel8_fp16(&scratch[i], gamma, g_beta, ot, ox);
                    else             kernel16_fp16(&scratch[i], gamma, g_beta, ot, ox);
                }
#endif
                s += fitness_proxy(ot, ox);
            }

            // speed pressure for refresh + keff
            s -= speed_penalty(cfg, g.keff, sh);

            if (s > best) {
                best = s;
                best_perm = p;
                best_shift = sh;
            }
        }
    }

    g.perm_id = best_perm;
    g.refresh_shift = best_shift;
    rng ^= (best_perm * 0x9E3779B9u) ^ (best_shift * 0x7F4A7C15u);
}

// ---------------------------- Worker ----------------------------

struct ThreadResult {
    double seconds = 0.0;
    double nodes_per_sec = 0.0;
    float checksum = 0.0f;
    float last_fit = 0.0f;
    Genome last_genome{};
};

template <typename NodeT>
static ThreadResult run_worker(const EngineConfig& cfg, int tid, uint32_t seed)
{
    if (cfg.pin_threads) pin_thread_to_core(tid);

    if (!is_pow2_u32(cfg.ring_nodes) || !is_pow2_u32(cfg.ga_batch)) {
        std::cerr << "Config error: --ring and --ga-batch must be power-of-two.\n";
        std::exit(1);
    }

    const uint32_t ring_n = cfg.ring_nodes;
    const uint32_t ring_mask = ring_n - 1u;

    NodeT* ring = (NodeT*)aligned_malloc(32, sizeof(NodeT) * (size_t)ring_n);
    if (!ring) { std::cerr << "aligned_malloc failed\n"; std::exit(1); }

    uint32_t rng = seed ^ (uint32_t)(0x9E3779B9u * (uint32_t)(tid + 1));
    Genome g;
    g.beta = cfg.initial_beta;
    g.keff = 8;
    g.tail_scale = 0.0f;
    g.perm_id = 0;
    g.refresh_shift = cfg.refresh_min_shift;

    // initial fill (full ring)
    for (uint32_t i = 0; i < ring_n; ++i) refill_slot(ring[i], g, rng);

    float last_fit = -1.0e30f;
    bool force16 = false;

    __m256 gamma, g_beta;
    compute_gamma(g.beta, gamma, g_beta);

    float checksum = 0.0f;

    auto t0 = std::chrono::high_resolution_clock::now();

    const uint64_t N = cfg.nodes_per_thread;

    // Burst refill size per GA update (small, cache-friendly)
    const uint32_t burst = (ring_n >= 128) ? (ring_n / 8) : (ring_n / 4);

    for (uint64_t i = 0; i < N; ++i) {
        // Optional prefetch (control-plane)
        if (cfg.prefetch) {
            uint32_t pf = (uint32_t)(i + 32) & ring_mask;
            __builtin_prefetch((const void*)&ring[pf], 0, 3);
        }

        // Feed-plane: refresh one slot periodically (according to genome refresh_shift)
        const uint32_t refresh_mask = (1u << g.refresh_shift) - 1u;
        if ((uint32_t)i & refresh_mask) {
            // no-op (branch predicted taken)
        } else {
            uint32_t idx = (uint32_t)i & ring_mask;
            refill_slot(ring[idx], g, rng);
        }

        // GA update
        if (((uint32_t)i & (cfg.ga_batch - 1u)) == 0u) {
            // GA#1: beta (compute-plane)
            float new_beta = ga_pick_beta(cfg, ring, ring_mask, g, rng);
            g.beta = 0.9f * g.beta + 0.1f * new_beta;

            // refresh gamma cache
            compute_gamma(g.beta, gamma, g_beta);

            // GA#2: density (keff + tail_scale)
            // (density keičia prepack ateityje; hot-path lieka toks pats)
            if constexpr (std::is_same<NodeT, NodeFP32>::value) {
                ga_pick_density(cfg, (const NodeFP32*)ring, ring_mask, g, rng);
            } else {
                // FP16 atveju density parenkam greičio spaudimui: 8 default, 16 tik jei force16.
                // (jei nori – gali analogiškai daryti FP16 ring'e per kernel16_fp16 eval)
                if (force16) { g.keff = 16; g.tail_scale = cfg.tail_scale_step; }
                else         { g.keff = 8;  g.tail_scale = 0.0f; }
            }

            // Situacinis guard: jei per daug krenta (įvertinsim iš GA#3 score)
            // (čia guard bus taikomas po GA#3; bet paruošiam flagą)
            // GA#3: feed (perm + refresh)
            ga_pick_feed<NodeT>(cfg, g, rng);

            // Greitas įsigėrimas: burst refill su nauju genome
            burst_refill(ring, ring_mask, g, rng, burst);

            // Re-evaluate quick fitness on few samples to update force16 guard (cheap)
            float s = 0.0f;
            for (uint32_t k = 0; k < cfg.sample; ++k) {
                uint32_t idx = xorshift32(rng) & ring_mask;
                float ot, ox;
                if constexpr (std::is_same<NodeT, NodeFP32>::value) {
                    if (g.keff == 8) kernel8_fp32(&ring[idx], gamma, g_beta, ot, ox);
                    else             kernel16_fp32(&ring[idx], gamma, g_beta, ot, ox);
                }
#if defined(__F16C__)
                else {
                    if (g.keff == 8) kernel8_fp16(&ring[idx], gamma, g_beta, ot, ox);
                    else             kernel16_fp16(&ring[idx], gamma, g_beta, ot, ox);
                }
#endif
                s += fitness_proxy(ot, ox);
            }

            // Guard: jei K=8 ir nukrito per daug palyginus su praeitu -> priverstinai 16 kitam batch
            if (g.keff == 8 && last_fit > -1.0e29f) {
                force16 = (s < last_fit * cfg.degrade_ratio);
                if (force16) { g.keff = 16; g.tail_scale = cfg.tail_scale_step; }
            } else {
                force16 = false;
            }

            last_fit = s;
        }

        // Compute hot-path
        const uint32_t idx = (uint32_t)i & ring_mask;
        float ot, ox;

        if constexpr (std::is_same<NodeT, NodeFP32>::value) {
            if (g.keff == 8) kernel8_fp32(&ring[idx], gamma, g_beta, ot, ox);
            else             kernel16_fp32(&ring[idx], gamma, g_beta, ot, ox);
        }
#if defined(__F16C__)
        else {
            if (g.keff == 8) kernel8_fp16(&ring[idx], gamma, g_beta, ot, ox);
            else             kernel16_fp16(&ring[idx], gamma, g_beta, ot, ox);
        }
#endif

        // Minimalus side-effect (kad neišmestų)
        checksum += (ot + ox) * 1e-6f;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    aligned_free(ring);

    ThreadResult r;
    r.seconds = sec;
    r.nodes_per_sec = (double)cfg.nodes_per_thread / sec;
    r.checksum = checksum;
    r.last_fit = last_fit;
    r.last_genome = g;
    return r;
}

// ---------------------------- CLI ----------------------------

static inline bool arg_eq(const char* a, const char* b) { return std::strcmp(a, b) == 0; }

static void print_help() {
    std::cout <<
        "Usage: engine [options]\n"
        "  --threads N\n"
        "  --nodes-per-thread N\n"
        "  --ring N                 (pow2, e.g. 256)\n"
        "  --ga-batch N             (pow2, e.g. 4096)\n"
        "  --sample N               (e.g. 32)\n"
        "  --beta-step F\n"
        "  --tail-step F            (e.g. 0.25)\n"
        "  --perm-cands N           (<=4)\n"
        "  --refresh-min S          (shift)\n"
        "  --refresh-max S          (shift)\n"
        "  --degrade F              (e.g. 0.97)\n"
        "  --speed-bias F           (e.g. 0.02)\n"
        "  --beta0 F\n"
        "  --prefetch 0/1\n"
        "  --fp16 0/1               (requires F16C; compile with -march=native or haswell)\n"
        "  --no-pin\n";
}

int main(int argc, char** argv) {
    EngineConfig cfg;

    for (int i = 1; i < argc; ++i) {
        if (arg_eq(argv[i], "--help") || arg_eq(argv[i], "-h")) { print_help(); return 0; }
        else if (arg_eq(argv[i], "--threads") && i + 1 < argc) cfg.threads = std::max(1, std::atoi(argv[++i]));
        else if (arg_eq(argv[i], "--nodes-per-thread") && i + 1 < argc) cfg.nodes_per_thread = std::strtoull(argv[++i], nullptr, 10);
        else if (arg_eq(argv[i], "--ring") && i + 1 < argc) cfg.ring_nodes = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        else if (arg_eq(argv[i], "--ga-batch") && i + 1 < argc) cfg.ga_batch = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        else if (arg_eq(argv[i], "--sample") && i + 1 < argc) cfg.sample = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        else if (arg_eq(argv[i], "--beta-step") && i + 1 < argc) cfg.beta_step = (float)std::atof(argv[++i]);
        else if (arg_eq(argv[i], "--tail-step") && i + 1 < argc) cfg.tail_scale_step = (float)std::atof(argv[++i]);
        else if (arg_eq(argv[i], "--perm-cands") && i + 1 < argc) cfg.perm_cands = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        else if (arg_eq(argv[i], "--refresh-min") && i + 1 < argc) cfg.refresh_min_shift = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        else if (arg_eq(argv[i], "--refresh-max") && i + 1 < argc) cfg.refresh_max_shift = (uint32_t)std::strtoul(argv[++i], nullptr, 10);
        else if (arg_eq(argv[i], "--degrade") && i + 1 < argc) cfg.degrade_ratio = (float)std::atof(argv[++i]);
        else if (arg_eq(argv[i], "--speed-bias") && i + 1 < argc) cfg.speed_bias = (float)std::atof(argv[++i]);
        else if (arg_eq(argv[i], "--beta0") && i + 1 < argc) cfg.initial_beta = (float)std::atof(argv[++i]);
        else if (arg_eq(argv[i], "--prefetch") && i + 1 < argc) cfg.prefetch = std::atoi(argv[++i]) ? 1 : 0;
        else if (arg_eq(argv[i], "--fp16") && i + 1 < argc) cfg.fp16 = std::atoi(argv[++i]) ? 1 : 0;
        else if (arg_eq(argv[i], "--no-pin")) cfg.pin_threads = false;
        else {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            print_help();
            return 1;
        }
    }

    if (!is_pow2_u32(cfg.ring_nodes) || !is_pow2_u32(cfg.ga_batch)) {
        std::cerr << "Error: --ring and --ga-batch must be power-of-two.\n";
        return 1;
    }
    if (cfg.perm_cands > 4) cfg.perm_cands = 4;
    if (cfg.sample == 0) cfg.sample = 1;

    const uint64_t total_all = cfg.nodes_per_thread * (uint64_t)cfg.threads;

    std::cout << "cfg:\n";
    std::cout << "  threads=" << cfg.threads << "\n";
    std::cout << "  nodes_per_thread=" << (unsigned long long)cfg.nodes_per_thread << "\n";
    std::cout << "  total_all=" << (unsigned long long)total_all << "\n";
    std::cout << "  ring=" << cfg.ring_nodes << " ga_batch=" << cfg.ga_batch << " sample=" << cfg.sample << "\n";
    std::cout << "  beta_step=" << cfg.beta_step << " tail_step=" << cfg.tail_scale_step << "\n";
    std::cout << "  perm_cands=" << cfg.perm_cands
              << " refresh_shift=[" << cfg.refresh_min_shift << ".." << cfg.refresh_max_shift << "]\n";
    std::cout << "  degrade_ratio=" << cfg.degrade_ratio << " speed_bias=" << cfg.speed_bias << "\n";
    std::cout << "  prefetch=" << cfg.prefetch << " pin=" << (cfg.pin_threads ? 1 : 0) << "\n";
#if defined(__F16C__)
    std::cout << "  fp16=" << cfg.fp16 << " (compile-time F16C=1)\n";
#else
    std::cout << "  fp16=" << cfg.fp16 << " (compile-time F16C=0 -> forced fp32)\n";
    cfg.fp16 = 0;
#endif
    std::cout.flush();

    std::vector<std::thread> ts;
    std::vector<ThreadResult> res((size_t)cfg.threads);
    const uint32_t seed = 0xC001D00Du;

    auto T0 = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < cfg.threads; ++t) {
        ts.emplace_back([&, t]() {
#if defined(__F16C__)
            if (cfg.fp16) res[(size_t)t] = run_worker<NodeFP16>(cfg, t, seed);
            else          res[(size_t)t] = run_worker<NodeFP32>(cfg, t, seed);
#else
            res[(size_t)t] = run_worker<NodeFP32>(cfg, t, seed);
#endif
        });
    }
    for (auto& th : ts) th.join();

    auto T1 = std::chrono::high_resolution_clock::now();
    const double wall = std::chrono::duration<double>(T1 - T0).count();

    double sum_nps = 0.0;
    float sum_chk = 0.0f;
    for (int t = 0; t < cfg.threads; ++t) {
        sum_nps += res[(size_t)t].nodes_per_sec;
        sum_chk += res[(size_t)t].checksum;
    }
    const double wall_nps = (double)total_all / wall;

    std::cout << "result:\n";
    std::cout << "  wall_time_s=" << wall << "\n";
    std::cout << "  wall_nodes_per_s=" << wall_nps << "\n";
    std::cout << "  sum_thread_nodes_per_s=" << sum_nps << "\n";
    std::cout << "  checksum=" << sum_chk << "\n";

    // report one thread state
    const ThreadResult& r0 = res[0];
    std::cout << "sample_state(thread0):\n";
    std::cout << "  last_fit=" << r0.last_fit << "\n";
    std::cout << "  beta=" << r0.last_genome.beta
              << " keff=" << r0.last_genome.keff
              << " tail=" << r0.last_genome.tail_scale
              << " perm=" << r0.last_genome.perm_id
              << " refresh_shift=" << r0.last_genome.refresh_shift << "\n";

    return 0;
}
