// spsc_ring.cpp
// Single-producer / single-consumer ring buffer with "payload first, header
// last" commit. Fixes UB from casting non-atomic memory to atomic, adds bounds
// checks, validates mode, and provides portable wait/wake with optional ARM
// WFE/SEV. Adds C++20 [[likely]] / [[unlikely]] to guide branch prediction for
// mode dispatch.
//
// Build (Linux/Debian):
//   g++ -std=c++20 -O2 -pthread spsc_ring.cpp -o spsc_ring
//
// Build (MinGW/Windows):
//   g++ -std=c++20 -O2 -pthread spsc_ring.cpp -o spsc_ring.exe
//
// Run:
//   ./spsc_ring

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>


static constexpr std::size_t RING_SIZE = 64;     // Must be > 1
static constexpr std::size_t PAYLOAD_MAX = 1024; // Max bytes per packet

// Optional: define to force ARM WFE/SEV when compiling for ARM.
// #define USE_ARM_SEV_WFE 1

enum class Mode : std::uint8_t {
  Raw = 0,
  Compressed = 1,
  Stop = 2, // control packet to stop worker in test
};

// -------- CRC16 (CCITT-FALSE) --------
static std::uint16_t crc16_ccitt_false(const std::uint8_t *data,
                                       std::size_t len) {
  std::uint16_t crc = 0xFFFF;
  for (std::size_t i = 0; i < len; ++i) {
    crc ^= (std::uint16_t)data[i] << 8;
    for (int b = 0; b < 8; ++b) {
      crc = (crc & 0x8000) ? (std::uint16_t)((crc << 1) ^ 0x1021)
                           : (std::uint16_t)(crc << 1);
    }
  }
  return crc;
}

// -------- Header packing/unpacking --------
// Layout (64-bit):
//   bits  0..31  length (uint32_t)
//   bits 32..39  mode   (uint8_t)
//   bits 40..47  gen_id (uint8_t)
//   bits 48..63  checksum (uint16_t)
// length==0 means "empty slot" sentinel.
struct HeaderFields {
  std::uint32_t length = 0;
  std::uint8_t mode = 0;
  std::uint8_t gen_id = 0;
  std::uint16_t checksum = 0;
};

static inline std::uint64_t pack_header(const HeaderFields &h) {
  std::uint64_t v = 0;
  v |= (std::uint64_t)h.length;
  v |= (std::uint64_t)h.mode << 32;
  v |= (std::uint64_t)h.gen_id << 40;
  v |= (std::uint64_t)h.checksum << 48;
  return v;
}

static inline HeaderFields unpack_header(std::uint64_t raw) {
  HeaderFields h;
  h.length = (std::uint32_t)(raw & 0xFFFFFFFFull);
  h.mode = (std::uint8_t)((raw >> 32) & 0xFFu);
  h.gen_id = (std::uint8_t)((raw >> 40) & 0xFFu);
  h.checksum = (std::uint16_t)((raw >> 48) & 0xFFFFu);
  return h;
}

// -------- Platform wait/wake (optional low-power on ARM) --------
static inline void platform_wait() {
#if (defined(USE_ARM_SEV_WFE) && USE_ARM_SEV_WFE) &&                           \
    (defined(__aarch64__) || defined(__arm__))
  __asm__ volatile("wfe" ::: "memory");
#else
  std::this_thread::yield();
#endif
}

static inline void platform_wake() {
#if (defined(USE_ARM_SEV_WFE) && USE_ARM_SEV_WFE) &&                           \
    (defined(__aarch64__) || defined(__arm__))
  __asm__ volatile("sev" ::: "memory");
#else
  // no-op
#endif
}

// -------- Data structures --------
struct alignas(64) DataPacket {
  alignas(8) std::atomic<std::uint64_t> header_raw{0}; // MUST be atomic
  alignas(8) std::uint8_t payload[PAYLOAD_MAX]{};
};

struct alignas(64) RingBuffer {
  DataPacket data[RING_SIZE]{};
  // head is producer-only, tail is consumer-only (SPSC). No atomics needed.
  std::size_t head = 0;
  std::size_t tail = 0;
};

// -------- Example processing hooks --------
static void apdoroti_raw(const std::uint8_t *p, std::uint32_t len) {
  (void)p;
  (void)len;
  // Real code: process raw packet bytes.
}

static void apdoroti_unzip(const std::uint8_t *p, std::uint32_t len) {
  (void)p;
  (void)len;
  // Real code: decompress + process.
}

// -------- Producer (Maitintojas) --------
// Returns true if pushed, false if slot busy or invalid input.
static bool maitintojas(RingBuffer &rb, const std::uint8_t *data,
                        std::uint32_t len, Mode mode, std::uint8_t gen_id) {
  if (!data)
    return false;

  // length==0 reserved as "empty slot"
  if (len == 0 || len > PAYLOAD_MAX)
    return false;

  const auto m = (std::uint8_t)mode;
  if (m > (std::uint8_t)Mode::Stop)
    return false;

  DataPacket &pkt = rb.data[rb.head];

  // ACQUIRE here prevents reordering around the "slot free" observation if code
  // evolves.
  const std::uint64_t cur = pkt.header_raw.load(std::memory_order_acquire);
  if (cur != 0) {
    return false; // slot busy
  }

  // 1) Write PAYLOAD first
  std::memcpy(pkt.payload, data, len);

  // 2) Commit HEADER last (release)
  HeaderFields h;
  h.length = len;
  h.mode = m;
  h.gen_id = gen_id;
  h.checksum = crc16_ccitt_false(data, len);

  pkt.header_raw.store(pack_header(h), std::memory_order_release);

  rb.head = (rb.head + 1) % RING_SIZE;

  platform_wake();
  return true;
}

// -------- Consumer (Girnos worker) --------
// Returns number of data packets processed (Stop packet not counted).
static std::uint64_t girnos_worker(RingBuffer &rb) {
  std::uint64_t processed = 0;

  for (;;) {
    DataPacket &pkt = rb.data[rb.tail];

    // 1) ACQUIRE: observe header commit, then safely read payload
    // written-before release store.
    const std::uint64_t raw = pkt.header_raw.load(std::memory_order_acquire);
    HeaderFields h = unpack_header(raw);

    if (h.length == 0) {
      platform_wait();
      continue;
    }

    // Validation to avoid UB and OOB even under corruption/bugs.
    if (h.length > PAYLOAD_MAX) {
      pkt.header_raw.store(0, std::memory_order_release);
      rb.tail = (rb.tail + 1) % RING_SIZE;
      continue;
    }

    if (h.mode > (std::uint8_t)Mode::Stop) {
      pkt.header_raw.store(0, std::memory_order_release);
      rb.tail = (rb.tail + 1) % RING_SIZE;
      continue;
    }

    // Optional integrity check
    const std::uint16_t got = crc16_ccitt_false(pkt.payload, h.length);
    if (got != h.checksum) {
      pkt.header_raw.store(0, std::memory_order_release);
      rb.tail = (rb.tail + 1) % RING_SIZE;
      continue;
    }

    // Dispatch with branch prediction hints.
    // Assumption: ~99% packets are Mode::Raw.
    switch ((Mode)h.mode) {
    case Mode::Raw:
      [[likely]] {
        apdoroti_raw(pkt.payload, h.length);
        break;
      }
    case Mode::Compressed:
      [[unlikely]] {
        apdoroti_unzip(pkt.payload, h.length);
        break;
      }
    case Mode::Stop:
      [[unlikely]] {
        // 2) RELEASE: clear slot before exit (producer may reuse).
        pkt.header_raw.store(0, std::memory_order_release);
        rb.tail = (rb.tail + 1) % RING_SIZE;
        return processed;
      }
    }

    // 2) RELEASE: clear header to signal producer "slot free".
    pkt.header_raw.store(0, std::memory_order_release);

    rb.tail = (rb.tail + 1) % RING_SIZE;
    ++processed;
  }
}

// -------- Short test --------
int main() {
  RingBuffer rb;

  static constexpr std::uint32_t N = 10000000; // 10 Million

  std::atomic<std::uint64_t> consumed{0};
  std::atomic<bool> worker_done{false};

  auto start_time = std::chrono::high_resolution_clock::now();

  std::thread worker([&] {
    const std::uint64_t c = girnos_worker(rb);
    consumed.store(c, std::memory_order_relaxed);
    worker_done.store(true, std::memory_order_relaxed);
  });

  // Producer: send patterned packets
  std::uint8_t buf[PAYLOAD_MAX];
  std::uint64_t produced = 0;

  for (std::uint32_t i = 0; i < N; ++i) {
    const std::uint32_t len = 32; // fixed for test
    std::memset(buf, 0, len);
    // encode sequence in first 4 bytes
    buf[0] = (std::uint8_t)(i & 0xFF);
    buf[1] = (std::uint8_t)((i >> 8) & 0xFF);
    buf[2] = (std::uint8_t)((i >> 16) & 0xFF);
    buf[3] = (std::uint8_t)((i >> 24) & 0xFF);

    while (!maitintojas(rb, buf, len, Mode::Raw, /*gen_id*/ 1)) {
      std::this_thread::yield();
    }
    ++produced;
  }

  // Send stop packet
  std::uint8_t stopbyte = 0xAA;
  while (!maitintojas(rb, &stopbyte, 1, Mode::Stop, 1)) {
    std::this_thread::yield();
  }

  worker.join();

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end_time - start_time;

  const std::uint64_t c = consumed.load(std::memory_order_relaxed);
  double seconds = diff.count();
  double packets_per_sec = (double)N / seconds;

  std::printf("produced=%llu consumed=%llu\n", (unsigned long long)produced,
              (unsigned long long)c);
  std::printf("Time: %.4f seconds\n", seconds);
  std::printf("Throughput: %.2f million packets/sec\n", packets_per_sec / 1e6);

  assert(c == produced);
  return 0;
}
