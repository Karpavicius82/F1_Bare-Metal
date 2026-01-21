// antigravity_win.cpp
// "Antigravity" Edition for Windows 11
// Features:
// 1. Thread Affinity (Pinning) - Prirakina gijas prie branduolių.
// 2. Busy Wait (_mm_pause) - Jokio sleep/yield.
// 3. No CRC - Grynas transporto greitis.
// 4. Safe Shutdown - Baigia darbą automatiškai.

#include <atomic>
#include <cstdint>
#include <cstring>
#include <thread>
#include <chrono>
#include <cstdio>
#include <cassert>
#include <vector>

// Windows API gijų valdymui
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <immintrin.h> // _mm_pause()

static constexpr std::size_t RING_SIZE    = 256;   // Didesnis buferis geriau sugeria Windows jitter'į
static constexpr std::size_t PAYLOAD_MAX  = 32;    // Mažas paketas = didelis PPS testas

enum class Mode : std::uint8_t {
    Raw        = 0,
    Stop       = 2,
};

// -------- Helper: Thread Pinning --------
void pin_thread_to_core(int core_id) {
    HANDLE thread = GetCurrentThread();
    DWORD_PTR mask = (1ULL << core_id);
    SetThreadAffinityMask(thread, mask);
    // Pakeliam prioritetą, kad Windows netrukdytų
    SetThreadPriority(thread, THREAD_PRIORITY_TIME_CRITICAL);
}

// -------- Header packing/unpacking --------
struct HeaderFields {
    std::uint32_t length = 0;
    std::uint8_t  mode   = 0;
    std::uint8_t  gen_id = 0;
    std::uint16_t checksum = 0;
};

static inline std::uint64_t pack_header(const HeaderFields& h) {
    std::uint64_t v = 0;
    v |= (std::uint64_t)h.length;
    v |= (std::uint64_t)h.mode      << 32;
    v |= (std::uint64_t)h.gen_id    << 40;
    v |= (std::uint64_t)h.checksum << 48;
    return v;
}

static inline HeaderFields unpack_header(std::uint64_t raw) {
    HeaderFields h;
    h.length   = (std::uint32_t)(raw & 0xFFFFFFFFull);
    h.mode     = (std::uint8_t)((raw >> 32) & 0xFFu);
    h.gen_id   = (std::uint8_t)((raw >> 40) & 0xFFu);
    h.checksum = (std::uint16_t)((raw >> 48) & 0xFFFFu);
    return h;
}

// -------- Platform wait/wake --------
// Windows "Busy Wait" - sukame ciklą su lengva pauze CPU vamzdynui
static inline void platform_wait() {
    _mm_pause(); 
}

static inline void platform_wake() {
    // No-op for busy wait loop
}

// -------- Data structures --------
struct alignas(64) DataPacket {
    alignas(8) std::atomic<std::uint64_t> header_raw{0};
    alignas(8) std::uint8_t payload[PAYLOAD_MAX]{};
};

struct alignas(64) RingBuffer {
    DataPacket data[RING_SIZE]{};
    alignas(64) std::size_t head = 0;
    alignas(64) std::size_t tail = 0;
};

// -------- Dummy processing hooks --------
// Kad kompiliatorius neišmestų kodo
static void apdoroti_raw(volatile std::uint8_t* p, std::uint32_t len) {
    // Tik paliečiam atmintį, jokios matematikos
    (void)p[0]; 
}

// -------- Producer (Maitintojas) --------
static bool maitintojas(RingBuffer& rb,
                        const std::uint8_t* data,
                        std::uint32_t len,
                        Mode mode,
                        std::uint8_t gen_id)
{
    DataPacket& pkt = rb.data[rb.head];

    const std::uint64_t cur = pkt.header_raw.load(std::memory_order_acquire);
    if (cur != 0) {
        return false; // slot busy
    }

    // 1) Write PAYLOAD first
    std::memcpy(pkt.payload, data, len);

    // 2) Commit HEADER last (release)
    HeaderFields h;
    h.length   = len;
    h.mode     = (std::uint8_t)mode;
    h.gen_id   = gen_id;
    h.checksum = 0; // IŠJUNGTA: CRC nedarome, testuojam transportą

    pkt.header_raw.store(pack_header(h), std::memory_order_release);

    rb.head = (rb.head + 1) % RING_SIZE;
    return true;
}

// -------- Consumer (Girnos worker) --------
static std::uint64_t girnos_worker(RingBuffer& rb) {
    // PRIRAKINAM PRIE BRANDUOLIO #2 (kad nesipeštų su Producer)
    pin_thread_to_core(2);
    
    std::uint64_t processed = 0;

    for (;;) {
        DataPacket& pkt = rb.data[rb.tail];

        const std::uint64_t raw = pkt.header_raw.load(std::memory_order_acquire);
        HeaderFields h = unpack_header(raw);

        if (h.length == 0) {
            platform_wait();
            continue;
        }

        if (h.mode == (std::uint8_t)Mode::Stop) {
            pkt.header_raw.store(0, std::memory_order_release);
            rb.tail = (rb.tail + 1) % RING_SIZE;
            return processed;
        }

        // Apdorojam
        apdoroti_raw(pkt.payload, h.length);

        // Release slot
        pkt.header_raw.store(0, std::memory_order_release);

        rb.tail = (rb.tail + 1) % RING_SIZE;
        ++processed;
    }
}

// -------- Main Test --------
int main() {
    // PRIRAKINAM MAIN THREAD (PRODUCER) PRIE BRANDUOLIO #1
    pin_thread_to_core(1);

    RingBuffer* rb = new RingBuffer(); // Heap allocation for safety

    // Didiname paketų kiekį iki 50 milijonų, kad testas truktų ilgiau
    static constexpr std::uint32_t N = 50000000; 

    printf("--- ANTIGRAVITY WINDOWS TEST ---\n");
    printf("Core Affinity: Producer=1, Worker=2\n");
    printf("Payload: %zu bytes\n", PAYLOAD_MAX);
    printf("CRC: OFF (Pure Transport Speed)\n");
    printf("Scheduling: TIME_CRITICAL (Busy Wait)\n");
    printf("Simulating %u packets...\n", N);

    std::atomic<std::uint64_t> consumed{0};
    
    auto start_time = std::chrono::high_resolution_clock::now();

    std::thread worker([&]{
        const std::uint64_t c = girnos_worker(*rb);
        consumed.store(c, std::memory_order_relaxed);
    });

    // Producer Loop
    std::uint8_t buf[PAYLOAD_MAX];
    std::memset(buf, 0xAA, PAYLOAD_MAX); // Užpildom vieną kartą

    for (std::uint32_t i = 0; i < N; ++i) {
        // Spin-wait producer side
        while (!maitintojas(*rb, buf, PAYLOAD_MAX, Mode::Raw, 1)) {
            platform_wait();
        }
    }

    // Stop signal
    while (!maitintojas(*rb, buf, 1, Mode::Stop, 1)) {
        platform_wait();
    }

    worker.join();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;

    const std::uint64_t c = consumed.load(std::memory_order_relaxed);
    double seconds = diff.count();
    double pps = (double)N / seconds;

    printf("\n--- RESULTS ---\n");
    printf("Processed: %llu packets\n", (unsigned long long)c);
    printf("Time: %.4f seconds\n", seconds);
    printf("Throughput: %.2f MILLION packets/sec\n", pps / 1e6);
    
    double mb_s = (pps * PAYLOAD_MAX) / (1024 * 1024);
    printf("Data Rate: %.2f MB/s\n", mb_s);

    delete rb;
    return 0;
}