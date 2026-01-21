#include <Arduino.h>
#include <atomic>


// SKIRTA Arduino® UNO R4 WiFi


// --- KONFIGŪRACIJA (Arduino Diet) ---
// Mažiname dydžius, kad tilptų į 32KB RAM
static constexpr size_t RING_SIZE = 8;        // 8 slotai
static constexpr size_t PAYLOAD_MAX = 128;    // 128 baitai per paketą

// --- 32-BIT HEADERIS (Cortex-M4 Optimizacija) ---
// Bit packing į 32 bitus:
// [0..15] Length (16 bit)
// [16..23] Mode (8 bit)
// [24..31] GenID/Checksum (8 bit)
struct Header32 {
  uint16_t length;
  uint8_t  mode;
  uint8_t  check;
};

union HeaderUnion {
  uint32_t raw;
  Header32 fields;
};

// --- DATA STRUCTURES ---
// Arduino neturi L1/L2/L3 cache problemų kaip Pi 5, 
// todėl 'alignas(64)' yra perteklinis ir švaisto RAM.
// Naudojame alignas(4) dėl 32-bit CPU.
struct DataPacket {
  std::atomic<uint32_t> header_raw{0}; // 32-bit atomic (Native for Cortex-M4)
  uint8_t payload[PAYLOAD_MAX];
};

struct RingBuffer {
  DataPacket data[RING_SIZE];
  volatile size_t head = 0; // Producer index
  volatile size_t tail = 0; // Consumer index
};

RingBuffer rb;
volatile unsigned long processed_count = 0;

// --- MAITINTOJAS (PRODUCER) - Veikia "Pertraukimo" režimu ---
// Arduino pasaulyje tai būtų: Serial.onReceive(), TimerInterrupt, arba WiFi callback.
bool maitintojas_isr(const uint8_t* data, uint16_t len, uint8_t mode) {
  DataPacket& pkt = rb.data[rb.head];

  // 1. ACQUIRE (Check if slot is free)
  uint32_t cur = pkt.header_raw.load(std::memory_order_acquire);
  if (cur != 0) return false; // Buffer full!

  // 2. Write Payload
  memcpy(pkt.payload, data, len);

  // 3. Pack Header
  HeaderUnion h;
  h.fields.length = len;
  h.fields.mode = mode;
  h.fields.check = 0xAA; // Paprasta checksum

  // 4. RELEASE (Commit)
  pkt.header_raw.store(h.raw, std::memory_order_release);

  // Advance Head
  rb.head = (rb.head + 1) % RING_SIZE;
  return true;
}

// --- GIRNOS (WORKER) - Veikia Main Loop ---
void girnos_loop() {
  DataPacket& pkt = rb.data[rb.tail];

  // 1. ACQUIRE (Check for new data)
  uint32_t raw = pkt.header_raw.load(std::memory_order_acquire);
  
  if (raw == 0) {
    return; // Empty
  }

  HeaderUnion h;
  h.raw = raw;

  // --- APDOROJIMAS ---
  // Čia tavo logika. Dabar tik spausdinam debug (lėta, bet testui tinka)
  // Realybėje čia darytum bit-banging, motor control ir pan.
  
  // Imituojam darbą
  processed_count++;

  // 2. RELEASE (Free slot)
  pkt.header_raw.store(0, std::memory_order_release);
  
  rb.tail = (rb.tail + 1) % RING_SIZE;
}

// --- ARDUINO SETUP ---
void setup() {
  Serial.begin(115200);
  while(!Serial);
  
  Serial.println("--- Arduino UNO R4 SPSC Ring Buffer ---");
  Serial.println("Architecture: 32-bit Cortex-M4");
  Serial.print("Payload size: "); Serial.println(PAYLOAD_MAX);
}

// --- ARDUINO LOOP ---
void loop() {
  // 1. Sukame Girnas (kiek įmanoma greičiau)
  girnos_loop();

  // 2. Imituojam Maitintoją (Kas 1ms)
  // Realybėje tai vyktų FspTimer pertraukime (Interrupt)
  static unsigned long last_feed = 0;
  if (millis() - last_feed > 1) {
    last_feed = millis();
    
    uint8_t dummy_data[32];
    memset(dummy_data, 0x55, 32);
    
    // Bandome įdėti paketą
    if (!maitintojas_isr(dummy_data, 32, 1)) {
      // Jei buferis pilnas - LED mirktelėjimas ar pan.
    }
  }

  // Statistika kas sekundę
  static unsigned long last_stat = 0;
  if (millis() - last_stat > 1000) {
    last_stat = millis();
    Serial.print("PPS (Packets Per Sec): ");
    Serial.println(processed_count);
    processed_count = 0;
  }
}