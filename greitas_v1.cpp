// engine_soa_ga_batch.cpp
// SoA-prepacked + AVX2/FMA branchless kernel + gating(mul) + reduce
// + Strategic GA per batch: parenka (beta, K_eff ∈ {8,16}) pagal imties fitness.
//
// Target: Haswell (AVX2+FMA). Vienas failas, C++20.
//
// Kompiliavimas (Debian):
// g++ -O3 -std=c++20 -march=haswell -mavx2 -mfma engine_soa_ga_batch.cpp -o engine && ./engine

#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include <chrono>

#if !defined(__AVX2__) || !defined(__FMA__)
#  error "Reikia AVX2+FMA. Kompiliuok su -march=haswell arba -mavx2 -mfma"
#endif

// -------------------- Konfigūracija --------------------
static constexpr int KMAX   = 16;     // fiksuotas maksimalus K (8 arba 16 naudojamas realiai)
static constexpr int BATCH  = 1024;   // GA vienetas
static constexpr int SAMPLE = 32;     // kiek mazgų per batch tikrinam kandidatams
static constexpr int CAND_BETA = 4;   // beta kandidatų skaičius
static constexpr float BETA_STEP = 0.04f;
static constexpr float DEGRADE_RATIO = 0.97f; // jei K=8 per blogai -> kitam batch draudžiam
static constexpr int FITNESS_STRIDE = 16;     // per kiek mazgų skaičiuoti batch fitness (taupo laiką)

// -------------------- Pagalbiniai --------------------
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

static inline __m256 rsqrt_nr1(__m256 x) {
    __m256 y = _mm256_rsqrt_ps(x);
    const __m256 half  = _mm256_set1_ps(0.5f);
    const __m256 three = _mm256_set1_ps(3.0f);
    __m256 yy   = _mm256_mul_ps(y, y);
    __m256 term = _mm256_fnmadd_ps(x, yy, three);   // 3 - x*y*y
    return _mm256_mul_ps(y, _mm256_mul_ps(half, term));
}

static inline float hsum8(__m256 v) {
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, v);
    float s = 0.0f;
    for (int i = 0; i < 8; ++i) s += tmp[i];
    return s;
}

// -------------------- Duomenų formatas: SoA prepacked --------------------
struct alignas(32) NodeSoA16 {
    // t,x,w,gate yra 16 float ir dėl offset'ų (0,64,128,192) išlieka 32B aligned.
    float t[KMAX];
    float x[KMAX];
    float w[KMAX];
    float gate[KMAX]; // 0.0f arba 1.0f
};

struct Candidate {
    float beta;
    int keff; // 8 arba 16
};

// -------------------- Gamma skaičiavimas (1x per batch/kandidatui) --------------------
__attribute__((noinline, target("avx2,fma")))
static inline void compute_gamma(float beta_scalar, __m256& gamma, __m256& g_beta) {
    beta_scalar = clamp_beta(beta_scalar);
    __m256 beta = _mm256_set1_ps(beta_scalar);

    __m256 b2 = _mm256_mul_ps(beta, beta);
    __m256 inv_g2 = _mm256_sub_ps(_mm256_set1_ps(1.0f), b2);
    inv_g2 = _mm256_max_ps(inv_g2, _mm256_set1_ps(1.0e-12f));

    gamma  = rsqrt_nr1(inv_g2);
    g_beta = _mm256_mul_ps(gamma, beta);
}

// -------------------- Kerneliai (branchless data path) --------------------
__attribute__((noinline, target("avx2,fma")))
static inline void kernel_keff8(const NodeSoA16* __restrict n,
                                __m256 gamma, __m256 g_beta,
                                float& out_t, float& out_x)
{
    __m256 t = _mm256_load_ps(n->t + 0);
    __m256 x = _mm256_load_ps(n->x + 0);
    __m256 w = _mm256_load_ps(n->w + 0);
    __m256 g = _mm256_load_ps(n->gate + 0);

    // t' = gamma*t - g_beta*x ; x' = gamma*x - g_beta*t
    __m256 tp = _mm256_fnmadd_ps(g_beta, x, _mm256_mul_ps(gamma, t));
    __m256 xp = _mm256_fnmadd_ps(g_beta, t, _mm256_mul_ps(gamma, x));

    tp = _mm256_mul_ps(tp, g);
    xp = _mm256_mul_ps(xp, g);

    __m256 acc_t = _mm256_mul_ps(tp, w);
    __m256 acc_x = _mm256_mul_ps(xp, w);

    out_t = hsum8(acc_t);
    out_x = hsum8(acc_x);
}

__attribute__((noinline, target("avx2,fma")))
static inline void kernel_keff16(const NodeSoA16* __restrict n,
                                 __m256 gamma, __m256 g_beta,
                                 float& out_t, float& out_x)
{
    __m256 acc_t = _mm256_setzero_ps();
    __m256 acc_x = _mm256_setzero_ps();

    // block 0
    {
        __m256 t = _mm256_load_ps(n->t + 0);
        __m256 x = _mm256_load_ps(n->x + 0);
        __m256 w = _mm256_load_ps(n->w + 0);
        __m256 g = _mm256_load_ps(n->gate + 0);

        __m256 tp = _mm256_fnmadd_ps(g_beta, x, _mm256_mul_ps(gamma, t));
        __m256 xp = _mm256_fnmadd_ps(g_beta, t, _mm256_mul_ps(gamma, x));

        tp = _mm256_mul_ps(tp, g);
        xp = _mm256_mul_ps(xp, g);

        acc_t = _mm256_fmadd_ps(tp, w, acc_t);
        acc_x = _mm256_fmadd_ps(xp, w, acc_x);
    }
    // block 1
    {
        __m256 t = _mm256_load_ps(n->t + 8);
        __m256 x = _mm256_load_ps(n->x + 8);
        __m256 w = _mm256_load_ps(n->w + 8);
        __m256 g = _mm256_load_ps(n->gate + 8);

        __m256 tp = _mm256_fnmadd_ps(g_beta, x, _mm256_mul_ps(gamma, t));
        __m256 xp = _mm256_fnmadd_ps(g_beta, t, _mm256_mul_ps(gamma, x));

        tp = _mm256_mul_ps(tp, g);
        xp = _mm256_mul_ps(xp, g);

        acc_t = _mm256_fmadd_ps(tp, w, acc_t);
        acc_x = _mm256_fmadd_ps(xp, w, acc_x);
    }

    out_t = hsum8(acc_t);
    out_x = hsum8(acc_x);
}

// Fitness proxy (pigi): out_t - |out_x|
static inline float fitness_proxy(float out_t, float out_x) {
    return out_t - absf(out_x);
}

// -------------------- Strategic GA per batch --------------------
static Candidate pick_candidate_for_batch(const NodeSoA16* nodes,
                                         int count,
                                         float base_beta,
                                         uint32_t& rng,
                                         bool allow_keff8)
{
    float betas[CAND_BETA];
    betas[0] = base_beta;
    betas[1] = base_beta + BETA_STEP;
    betas[2] = base_beta - BETA_STEP;
    betas[3] = base_beta + 2.0f * BETA_STEP;

    // sample indeksai
    int sample_idx[SAMPLE];
    for (int i = 0; i < SAMPLE; ++i) {
        uint32_t r = xorshift32(rng);
        sample_idx[i] = (int)(r % (uint32_t)count);
    }

    Candidate best{clamp_beta(betas[0]), 16};
    float best_score = -1.0e30f;

    for (int bi = 0; bi < CAND_BETA; ++bi) {
        float b = clamp_beta(betas[bi]);
        __m256 gamma, g_beta;
        compute_gamma(b, gamma, g_beta);

        for (int mode = 0; mode < 2; ++mode) {
            int keff = (mode == 0) ? 8 : 16;
            if (keff == 8 && !allow_keff8) continue;

            float score = 0.0f;
            for (int si = 0; si < SAMPLE; ++si) {
                const NodeSoA16* n = &nodes[sample_idx[si]];
                float ot, ox;
                if (keff == 8) kernel_keff8(n, gamma, g_beta, ot, ox);
                else           kernel_keff16(n, gamma, g_beta, ot, ox);
                score += fitness_proxy(ot, ox);
            }

            if (score > best_score) {
                best_score = score;
                best = Candidate{b, keff};
            }
        }
    }

    return best;
}

// -------------------- Prepack demo (čia tik generatorius) --------------------
// Realioj sistemoje čia dėtum adjacency -> topK -> SoA pack.
static void build_prepacked_nodes(std::vector<NodeSoA16>& out_nodes, int N, uint32_t& rng)
{
    out_nodes.resize((size_t)N);

    for (int i = 0; i < N; ++i) {
        NodeSoA16& n = out_nodes[(size_t)i];

        // deg: 8 arba 16 (kad K_eff=8 turėtų prasmę)
        int deg = 8 + (int)(xorshift32(rng) & 1u) * 8;

        for (int k = 0; k < KMAX; ++k) {
            float base = (float)(i & 1023) * 0.01f + (float)k * 0.1f;
            uint32_t r = xorshift32(rng);

            // t,x (demo)
            n.t[k] = 10.0f + base + (float)((int)(r & 255) - 128) * 1e-3f;
            n.x[k] =  1.0f + base * 0.05f;

            // svoriai: pirma 8 laikom "sunkesnius" (kad 8 būtų artimas 16)
            n.w[k] = (k < 8) ? 1.0f : 0.25f;

            // gating (jūsų pataisyta eilutė)
            n.gate[k] = (k < deg) ? 1.0f : 0.0f;
        }
    }
}

// -------------------- Pipeline demo --------------------
int main()
{
    const int N = 1 << 18; // 262144 mazgai (demo)
    uint32_t rng = 0xC001D00Du;

    std::vector<NodeSoA16> nodes;
    build_prepacked_nodes(nodes, N, rng);

    std::vector<float> out_t((size_t)N);
    std::vector<float> out_x((size_t)N);

    float base_beta = 0.6f;
    float last_batch_fit = -1.0e30f;
    bool force_keff16_next = false;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int base = 0; base < N; base += BATCH) {
        int cnt = (base + BATCH <= N) ? BATCH : (N - base);
        const NodeSoA16* batch_nodes = &nodes[(size_t)base];

        // situacinis: jei praeitą kartą K=8 "per blogai", draudžiam K=8 šitam batch
        bool allow_keff8 = !force_keff16_next;

        Candidate c = pick_candidate_for_batch(batch_nodes, cnt, base_beta, rng, allow_keff8);

        // 1x per batch
        __m256 gamma, g_beta;
        compute_gamma(c.beta, gamma, g_beta);

        // kernel pass
        float fit_acc = 0.0f;
        int fit_cnt = 0;

        for (int i = 0; i < cnt; ++i) {
            float ot, ox;
            if (c.keff == 8) kernel_keff8(&batch_nodes[i], gamma, g_beta, ot, ox);
            else             kernel_keff16(&batch_nodes[i], gamma, g_beta, ot, ox);

            out_t[(size_t)base + (size_t)i] = ot;
            out_x[(size_t)base + (size_t)i] = ox;

            // pigi batch kokybės metrika (ne kiekvienam mazgui)
            if ((i % FITNESS_STRIDE) == 0) {
                fit_acc += fitness_proxy(ot, ox);
                ++fit_cnt;
            }
        }

        float batch_fit = (fit_cnt > 0) ? (fit_acc / (float)fit_cnt) : last_batch_fit;

        // jei pasirinkta K=8 ir smuko per daug -> kitam batch priverstinai K=16
        if (c.keff == 8 && last_batch_fit > -1.0e29f) {
            force_keff16_next = (batch_fit < last_batch_fit * DEGRADE_RATIO);
        } else {
            force_keff16_next = false;
        }

        // lėtas bazinės beta adaptavimas
        base_beta = 0.9f * base_beta + 0.1f * c.beta;
        last_batch_fit = batch_fit;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    double nodes_per_sec = (double)N / sec;

    std::cout << "N=" << N << " time=" << sec << " s, nodes/s=" << nodes_per_sec << "\n";
    std::cout << "out_t[0]=" << out_t[0] << " out_x[0]=" << out_x[0] << "\n";
    std::cout << "final base_beta=" << base_beta
              << " last_fit=" << last_batch_fit
              << " force_keff16_next=" << (force_keff16_next ? 1 : 0) << "\n";

    return 0;
}
