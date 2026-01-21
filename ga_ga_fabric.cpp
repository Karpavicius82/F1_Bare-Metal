// ga_ga_fabric.cpp
// Multi-core GA↔GA fabric: N GA threads, each pair has a dedicated SPSC channel (i -> j).
// - Fast path: SPSC commit (payload first, header last with release; consumer acquire).
// - Doorbell per receiver: pending_mask[j] bit i means "channel i->j likely has data".
// - Uses busy-spin (_mm_pause) for max throughput; optional Windows pinning.
// - Protocol in 64-bit header: len(32) | mode(8) | type(8) | src(8) | dst(8).
// - [[likely]] / [[unlikely]] hints.
//
// Build (MSYS2/MinGW UCRT64):
//   g++ -std=c++20 -O3 -march=native -flto -DNDEBUG -pthread ga_ga_fabric.cpp -o ga_fabric.exe
//
// Run examples:
//   ./ga_fabric.exe                 (auto N = min(hw,24), no pin)
//   ./ga_fabric.exe --ga 24         (force 24 GA threads)
//   ./ga_fabric.exe --pin           (pin threads to CPU0..CPU(N-1))
//   ./ga_fabric.exe --msgs 5000000  (initial messages)
//
// Notes:
// - RTX Ti is irrelevant; this is CPU memory + atomics.
// - If you want peak Mops on hybrid i9: try --pin and test different CPU sets (P-cores vs E-cores).

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <thread>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cassert>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
  #include <immintrin.h>
  #define GA_HAS_X86_PAUSE 1
#else
  #define GA_HAS_X86_PAUSE 0
#endif

#if defined(_WIN32)
  #define NOMINMAX
  #include <windows.h>
#endif

// ---------------- Tunables ----------------
static constexpr std::size_t RING_SIZE   = 64;     // slots per channel
static constexpr std::size_t PAYLOAD_MAX = 256;    // keep small for high Mops; increase if needed

// Busy-spin tuning:
static constexpr int SPIN_PAUSE_ITERS = 400;
static constexpr int SPIN_YIELD_ITERS = 40;

// Max supported GA threads for doorbell bitmask in this implementation:
static constexpr int MAX_GA = 64;

// ---------------- Protocol ----------------
enum class MsgType : std::uint8_t {
    Data    = 0,
    Control = 1,
    Ack     = 2,
};

enum class Mode : std::uint8_t {
    Raw        = 0,
    Compressed = 1,
};

static inline std::uint64_t pack_header(std::uint32_t len,
                                        std::uint8_t mode,
                                        std::uint8_t type,
                                        std::uint8_t src,
                                        std::uint8_t dst)
{
    // len(32) | mode(8) | type(8) | src(8) | dst(8)
    std::uint64_t v = 0;
    v |= (std::uint64_t)len;
    v |= (std::uint64_t)mode << 32;
    v |= (std::uint64_t)type << 40;
    v |= (std::uint64_t)src  << 48;
    v |= (std::uint64_t)dst  << 56;
    return v;
}

struct HeaderFields {
    std::uint32_t len = 0;
    std::uint8_t  mode = 0;
    std::uint8_t  type = 0;
    std::uint8_t  src = 0;
    std::uint8_t  dst = 0;
};

static inline HeaderFields unpack_header(std::uint64_t raw) {
    HeaderFields h;
    h.len  = (std::uint32_t)(raw & 0xFFFFFFFFull);
    h.mode = (std::uint8_t)((raw >> 32) & 0xFFu);
    h.type = (std::uint8_t)((raw >> 40) & 0xFFu);
    h.src  = (std::uint8_t)((raw >> 48) & 0xFFu);
    h.dst  = (std::uint8_t)((raw >> 56) & 0xFFu);
    return h;
}

// ---------------- Backoff ----------------
struct Backoff {
    int pause_iters = 0;
    int yield_iters = 0;

    void reset() { pause_iters = 0; yield_iters = 0; }

    void step() {
#if GA_HAS_X86_PAUSE
        if (pause_iters < SPIN_PAUSE_ITERS) {
            ++pause_iters;
            _mm_pause();
            return;
        }
#endif
        if (yield_iters < SPIN_YIELD_ITERS) {
            ++yield_iters;
            std::this_thread::yield();
            return;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
};

static inline int ctz64(std::uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(x);
#else
    // Fallback (should not be used with MSYS2 g++)
    int n = 0;
    while ((x & 1ull) == 0ull) { x >>= 1; ++n; }
    return n;
#endif
}

// ---------------- SPSC Channel ----------------
struct alignas(64) Slot {
    std::atomic<std::uint64_t> header{0}; // len==0 => empty
    alignas(8) std::uint8_t payload[PAYLOAD_MAX];
};

struct SPSCChannel {
    Slot slots[RING_SIZE];
    // head used only by producer, tail only by consumer
    std::size_t head = 0;
    std::size_t tail = 0;

    bool try_send(const std::uint8_t* data,
                  std::uint32_t len,
                  std::uint8_t mode,
                  std::uint8_t type,
                  std::uint8_t src,
                  std::uint8_t dst)
    {
        if (!data || len == 0 || len > PAYLOAD_MAX) return false;
        Slot& s = slots[head];

        // Check empty
        const std::uint64_t cur = s.header.load(std::memory_order_relaxed);
        if (cur != 0) [[unlikely]] return false;

        // payload first
        std::memcpy(s.payload, data, len);

        // commit header last
        s.header.store(pack_header(len, mode, type, src, dst), std::memory_order_release);

        head = (head + 1) % RING_SIZE;
        return true;
    }

    // Non-blocking recv. Returns true if got a packet.
    bool try_recv(HeaderFields& h, const std::uint8_t*& payload_ptr) {
        Slot& s = slots[tail];
        const std::uint64_t raw = s.header.load(std::memory_order_acquire);
        if (raw == 0) [[likely]] return false;

        h = unpack_header(raw);
        payload_ptr = s.payload;
        return true;
    }

    // Must be called after processing the packet.
    void pop_commit() {
        Slot& s = slots[tail];
        s.header.store(0, std::memory_order_release);
        tail = (tail + 1) % RING_SIZE;
    }

    bool peek_has_data() const {
        const Slot& s = slots[tail];
        return s.header.load(std::memory_order_relaxed) != 0;
    }
};

// ---------------- GA Fabric ----------------
struct GAFabric {
    int n = 0;

    // channels[src*n + dst] is SPSC src->dst
    std::vector<SPSCChannel> channels;

    // pending_mask[dst] bit src indicates "src->dst may have data"
    std::vector<std::atomic<std::uint64_t>> pending_mask;

    explicit GAFabric(int n_)
        : n(n_), channels((std::size_t)n_ * (std::size_t)n_), pending_mask((std::size_t)n_)
    {
        for (int i = 0; i < n; ++i) pending_mask[(std::size_t)i].store(0, std::memory_order_relaxed);
    }

    inline SPSCChannel& ch(int src, int dst) {
        return channels[(std::size_t)src * (std::size_t)n + (std::size_t)dst];
    }

    bool send(int src, int dst, MsgType type, Mode mode, const std::uint8_t* data, std::uint32_t len) {
        if (src < 0 || src >= n || dst < 0 || dst >= n) return false;
        auto& c = ch(src, dst);
        if (!c.try_send(data, len, (std::uint8_t)mode, (std::uint8_t)type, (std::uint8_t)src, (std::uint8_t)dst)) {
            return false;
        }
        // Doorbell: mark pending for receiver
        const std::uint64_t bit = 1ull << (unsigned)src;
        pending_mask[(std::size_t)dst].fetch_or(bit, std::memory_order_release);
        return true;
    }

    // Receiver drains one sender channel indicated by bit (src->dst).
    // Returns number drained from that channel (0 if empty despite bit).
    std::uint32_t drain_one(int dst, int src, std::uint64_t bit,
                            std::atomic<std::uint64_t>& global_processed,
                            std::atomic<bool>& run_flag)
    {
        auto& c = ch(src, dst);
        std::uint32_t drained = 0;

        for (;;) {
            HeaderFields h;
            const std::uint8_t* p = nullptr;
            if (!c.try_recv(h, p)) break;

            // Basic validation
            if (h.dst != (std::uint8_t)dst || h.src != (std::uint8_t)src) [[unlikely]] {
                // Still consume to avoid deadlock
                c.pop_commit();
                continue;
            }

            // Dispatch (99% Data/Raw can be hinted if that is your real profile)
            switch ((MsgType)h.type) {
                case MsgType::Data: [[likely]] {
                    if ((Mode)h.mode == Mode::Raw) [[likely]] {
                        // Simulate processing: touch first byte
                        (void)p[0];
                    } else [[unlikely]] {
                        // Compressed path stub
                        (void)p[0];
                    }
                    break;
                }
                case MsgType::Ack: [[unlikely]] {
                    break;
                }
                case MsgType::Control: [[unlikely]] {
                    // Control payload[0] = code. 0 => stop
                    if (h.len >= 1 && p[0] == 0) {
                        run_flag.store(false, std::memory_order_relaxed);
                    }
                    break;
                }
            }

            c.pop_commit();
            ++drained;
            global_processed.fetch_add(1, std::memory_order_relaxed);

            if (!run_flag.load(std::memory_order_relaxed)) break;
        }

        // If channel empty, clear bit; if not empty, keep it set.
        if (!c.peek_has_data()) {
            pending_mask[(std::size_t)dst].fetch_and(~bit, std::memory_order_release);
            // Race fix: if producer added after we checked, re-set.
            if (c.peek_has_data()) {
                pending_mask[(std::size_t)dst].fetch_or(bit, std::memory_order_release);
            }
        }
        return drained;
    }
};

// ---------------- Optional Windows pinning ----------------
#if defined(_WIN32)
static void pin_thread_to_cpu(int cpu_index) {
    // cpu_index 0..63 (no CPU groups handling here)
    const DWORD_PTR mask = (DWORD_PTR)1ull << (unsigned)cpu_index;
    SetThreadAffinityMask(GetCurrentThread(), mask);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
}
static void set_process_priority() {
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
}
#endif

// ---------------- GA Worker ----------------
struct GAWorkerCfg {
    int id = 0;
    bool pin = false;
};

static void ga_worker_loop(GAFabric& f,
                           GAWorkerCfg cfg,
                           std::atomic<std::uint64_t>& processed,
                           std::atomic<bool>& run_flag)
{
#if defined(_WIN32)
    if (cfg.pin) pin_thread_to_cpu(cfg.id);
#endif

    Backoff backoff;

    while (run_flag.load(std::memory_order_relaxed)) {
        std::uint64_t mask = f.pending_mask[(std::size_t)cfg.id].load(std::memory_order_acquire);
        if (mask == 0) [[likely]] {
            backoff.step();
            continue;
        }
        backoff.reset();

        // Service one sender at a time (lowest bit). This keeps it simple and fast.
        const int src = ctz64(mask);
        const std::uint64_t bit = 1ull << (unsigned)src;

        (void)f.drain_one(cfg.id, src, bit, processed, run_flag);
    }

    // Drain leftovers quickly (optional)
    for (;;) {
        std::uint64_t mask = f.pending_mask[(std::size_t)cfg.id].load(std::memory_order_acquire);
        if (mask == 0) break;
        const int src = ctz64(mask);
        const std::uint64_t bit = 1ull << (unsigned)src;
        (void)f.drain_one(cfg.id, src, bit, processed, run_flag);
    }
}

// ---------------- CLI ----------------
static int arg_int(int argc, char** argv, const char* key, int defv) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], key) == 0) return std::atoi(argv[i + 1]);
    }
    return defv;
}
static bool arg_flag(int argc, char** argv, const char* key) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], key) == 0) return true;
    }
    return false;
}

int main(int argc, char** argv) {
#if defined(_WIN32)
    set_process_priority();
#endif

    int hw = (int)std::thread::hardware_concurrency();
    if (hw <= 0) hw = 8;

    int ga_n = arg_int(argc, argv, "--ga", (hw > 24 ? 24 : hw));
    if (ga_n < 2) ga_n = 2;
    if (ga_n > MAX_GA) ga_n = MAX_GA;

    const bool pin = arg_flag(argc, argv, "--pin");
    const int msgs = arg_int(argc, argv, "--msgs", 5'000'000);

    std::printf("GA threads: %d  pin=%s  msgs=%d  ring=%zu payload_max=%zu\n",
                ga_n, pin ? "true" : "false", msgs, RING_SIZE, PAYLOAD_MAX);

    GAFabric f(ga_n);

    std::atomic<std::uint64_t> processed{0};
    std::atomic<bool> run_flag{true};

    // Start GA workers
    std::vector<std::thread> threads;
    threads.reserve((std::size_t)ga_n);
    for (int i = 0; i < ga_n; ++i) {
        GAWorkerCfg cfg;
        cfg.id = i;
        cfg.pin = pin;
        threads.emplace_back([&f, cfg, &processed, &run_flag] {
            ga_worker_loop(f, cfg, processed, run_flag);
        });
    }

    // Producer: distribute initial messages round-robin from GA0 -> all others (including itself optional).
    // This is "GA integration entry point": external producer can map to any src id you want.
    std::uint8_t payload[PAYLOAD_MAX];
    payload[0] = 0xAB; // dummy

    auto t0 = std::chrono::steady_clock::now();

    std::uint64_t sent = 0;
    for (int i = 0; i < msgs; ++i) {
        const int dst = (i % ga_n);
        const int src = 0;

        // Keep payload small for speed test
        const std::uint32_t len = 32;
        std::memset(payload, 0, len);
        payload[0] = (std::uint8_t)(i & 0xFF);

        while (!f.send(src, dst, MsgType::Data, Mode::Raw, payload, len)) {
#if GA_HAS_X86_PAUSE
            _mm_pause();
#else
            std::this_thread::yield();
#endif
        }
        ++sent;
    }

    // Wait until processed reaches sent (each message consumed exactly once).
    while (processed.load(std::memory_order_relaxed) < sent) {
        std::this_thread::yield();
    }

    // Stop all GA threads: send Control(stop) to each destination from src=0.
    std::uint8_t stop_payload[1] = {0};
    for (int dst = 0; dst < ga_n; ++dst) {
        while (!f.send(0, dst, MsgType::Control, Mode::Raw, stop_payload, 1)) {
#if GA_HAS_X86_PAUSE
            _mm_pause();
#else
            std::this_thread::yield();
#endif
        }
    }

    run_flag.store(false, std::memory_order_relaxed);

    for (auto& th : threads) th.join();

    auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();

    const std::uint64_t proc = processed.load(std::memory_order_relaxed);

    std::printf("sent=%llu processed=%llu time=%.3f s\n",
                (unsigned long long)sent,
                (unsigned long long)proc,
                sec);

    const double mops = (double)proc / sec / 1e6;
    const double mbps = ((double)proc * 32.0) / sec / (1024.0 * 1024.0);
    std::printf("Mops/s=%.2f  MB/s(32B payload)=%.2f\n", mops, mbps);

    return 0;
}
