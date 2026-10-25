// AES-128 encrypt-only, FIPS-197. See include/opencrab2/lowlevel/aes128.hpp.
#include "opencrab2/lowlevel/aes128.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>

// The hardware path: x86-64 with a GNU-style compiler (target attribute, <cpuid.h>) or MSVC
// (intrinsics always available, __cpuid from <intrin.h>). OPENCRAB2_AES_NO_HARDWARE removes it
// at compile time.
#if !defined(OPENCRAB2_AES_NO_HARDWARE) && (defined(__x86_64__) || defined(_M_X64))
#if defined(__GNUC__) || defined(__clang__)
#define OPENCRAB2_AES_HARDWARE 1
#define OPENCRAB2_AES_TARGET __attribute__((target("aes,sse2")))
#include <cpuid.h>
#include <emmintrin.h>
#include <wmmintrin.h>
#elif defined(_MSC_VER)
#define OPENCRAB2_AES_HARDWARE 1
#define OPENCRAB2_AES_TARGET
#include <intrin.h>
#include <emmintrin.h>
#include <wmmintrin.h>
#endif
#endif

namespace opencrab2 {
namespace lowlevel {

namespace {

const uint8_t kSbox[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16};

const uint8_t kRcon[10] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36};

inline uint8_t xtime(uint8_t x) {
    return static_cast<uint8_t>((x << 1) ^ ((x & 0x80u) ? 0x1bu : 0u));
}

inline uint32_t rotr8(uint32_t x) { return (x >> 8) | (x << 24); }
inline uint32_t rotr16(uint32_t x) { return (x >> 16) | (x << 16); }
inline uint32_t rotr24(uint32_t x) { return (x >> 24) | (x << 8); }

// Te0[x] packs the MixColumns contribution of S-box output a = S[x] sitting in row 0:
// (2a, a, a, 3a) from the most to the least significant byte. Rows 1..3 are byte rotations.
struct Tables {
    uint32_t te0[256];
    uint32_t te1[256];
    uint32_t te2[256];
    uint32_t te3[256];
    Tables() {
        for (int i = 0; i < 256; ++i) {
            const uint8_t a = kSbox[i];
            const uint8_t a2 = xtime(a);
            const uint8_t a3 = static_cast<uint8_t>(a2 ^ a);
            const uint32_t w = (static_cast<uint32_t>(a2) << 24) | (static_cast<uint32_t>(a) << 16) |
                               (static_cast<uint32_t>(a) << 8) | static_cast<uint32_t>(a3);
            te0[i] = w;
            te1[i] = rotr8(w);
            te2[i] = rotr16(w);
            te3[i] = rotr24(w);
        }
    }
};

const Tables& tables() {
    static const Tables t;
    return t;
}

inline uint32_t load_be32(const uint8_t* p) {
    return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}

inline void store_be32(uint8_t* p, uint32_t v) {
    p[0] = static_cast<uint8_t>(v >> 24);
    p[1] = static_cast<uint8_t>(v >> 16);
    p[2] = static_cast<uint8_t>(v >> 8);
    p[3] = static_cast<uint8_t>(v);
}

inline uint32_t bswap32(uint32_t v) {
    return (v >> 24) | ((v >> 8) & 0x0000ff00u) | ((v << 8) & 0x00ff0000u) | (v << 24);
}

inline uint32_t sub_word(uint32_t w) {
    return (static_cast<uint32_t>(kSbox[(w >> 24) & 0xff]) << 24) |
           (static_cast<uint32_t>(kSbox[(w >> 16) & 0xff]) << 16) |
           (static_cast<uint32_t>(kSbox[(w >> 8) & 0xff]) << 8) |
           static_cast<uint32_t>(kSbox[w & 0xff]);
}

// ---------------------------------------------------------------------------------------------
// run-time selection
// ---------------------------------------------------------------------------------------------
#if defined(OPENCRAB2_AES_HARDWARE)
bool cpu_has_aes() {
#if defined(_MSC_VER) && !defined(__clang__)
    int regs[4] = {0, 0, 0, 0};
    __cpuid(regs, 0);
    if (regs[0] < 1) return false;
    __cpuid(regs, 1);
    return ((static_cast<unsigned>(regs[2]) >> 25) & 1u) != 0 && ((static_cast<unsigned>(regs[3]) >> 26) & 1u) != 0;
#else
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return false;
    return ((ecx >> 25) & 1u) != 0 && ((edx >> 26) & 1u) != 0;   // AES and SSE2
#endif
}

// Fixed for the life of the process: CPUID, and the OPENCRAB2_AES=portable override.
struct Selection {
    bool available;
    bool env_portable;
    Selection() : available(cpu_has_aes()), env_portable(false) {
        const char* e = std::getenv("OPENCRAB2_AES");
        env_portable = (e != nullptr && std::strcmp(e, "portable") == 0);
    }
};

const Selection& selection() {
    static const Selection s;
    return s;
}
#endif

std::atomic<bool> g_force_portable{false};

inline bool use_hardware() {
#if defined(OPENCRAB2_AES_HARDWARE)
    const Selection& s = selection();
    return s.available && !s.env_portable && !g_force_portable.load(std::memory_order_relaxed);
#else
    return false;
#endif
}

#if defined(OPENCRAB2_AES_HARDWARE)
// Shape of the two kernels below. Eight block states are named locals and stay in registers
// through the rounds (the AESENC unit is pipelined: latency 3..4 cycles, throughput 1 or 2 per
// cycle, so eight independent blocks keep it busy). Two things cost a factor of two to three
// each while this was written, and the code is shaped to avoid them:
//   * states kept in an array filled by a variable-count loop were spilled to the stack inside
//     the rounds;
//   * round keys read through the caller's pointer inside the loop may alias the stores to out,
//     so they were reloaded every round, and a key load that waits on an unrelated store delays
//     all eight blocks. The kernels copy the 11 round keys into a local array first.

// Nine middle rounds and the last round over eight states.
#define OPENCRAB2_AES_ROUNDS8(kp, b0, b1, b2, b3, b4, b5, b6, b7)        \
    for (int r = 1; r < 10; ++r) {                                        \
        const __m128i kr = _mm_loadu_si128((kp) + r);                     \
        b0 = _mm_aesenc_si128(b0, kr);                                    \
        b1 = _mm_aesenc_si128(b1, kr);                                    \
        b2 = _mm_aesenc_si128(b2, kr);                                    \
        b3 = _mm_aesenc_si128(b3, kr);                                    \
        b4 = _mm_aesenc_si128(b4, kr);                                    \
        b5 = _mm_aesenc_si128(b5, kr);                                    \
        b6 = _mm_aesenc_si128(b6, kr);                                    \
        b7 = _mm_aesenc_si128(b7, kr);                                    \
    }                                                                     \
    {                                                                     \
        const __m128i kl = _mm_loadu_si128((kp) + 10);                    \
        b0 = _mm_aesenclast_si128(b0, kl);                                \
        b1 = _mm_aesenclast_si128(b1, kl);                                \
        b2 = _mm_aesenclast_si128(b2, kl);                                \
        b3 = _mm_aesenclast_si128(b3, kl);                                \
        b4 = _mm_aesenclast_si128(b4, kl);                                \
        b5 = _mm_aesenclast_si128(b5, kl);                                \
        b6 = _mm_aesenclast_si128(b6, kl);                                \
        b7 = _mm_aesenclast_si128(b7, kl);                                \
    }

OPENCRAB2_AES_TARGET inline __m128i ni_one(const __m128i* kp, __m128i b) {
    for (int r = 1; r < 10; ++r) b = _mm_aesenc_si128(b, _mm_loadu_si128(kp + r));
    return _mm_aesenclast_si128(b, _mm_loadu_si128(kp + 10));
}

// AES-128 ECB over n blocks with the 11 expanded round keys rk (176 bytes): eight blocks in
// flight, then single blocks for the tail. in == out is allowed: every block of a group is
// loaded before the first result of that group is stored.
OPENCRAB2_AES_TARGET void ni_encrypt(const uint8_t* rk, const uint8_t* in, uint8_t* out, std::size_t n) {
    // A private copy of the round keys: loads from it cannot alias the stores to out, so the
    // compiler keeps them out of the loop (a key load that waits on an unrelated store delays
    // all eight blocks of a round).
    __m128i kp[11];
    for (int r = 0; r < 11; ++r) kp[r] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rk) + r);
    const __m128i k0 = kp[0];
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m128i* p = reinterpret_cast<const __m128i*>(in + 16 * i);
        __m128i b0 = _mm_xor_si128(_mm_loadu_si128(p + 0), k0);
        __m128i b1 = _mm_xor_si128(_mm_loadu_si128(p + 1), k0);
        __m128i b2 = _mm_xor_si128(_mm_loadu_si128(p + 2), k0);
        __m128i b3 = _mm_xor_si128(_mm_loadu_si128(p + 3), k0);
        __m128i b4 = _mm_xor_si128(_mm_loadu_si128(p + 4), k0);
        __m128i b5 = _mm_xor_si128(_mm_loadu_si128(p + 5), k0);
        __m128i b6 = _mm_xor_si128(_mm_loadu_si128(p + 6), k0);
        __m128i b7 = _mm_xor_si128(_mm_loadu_si128(p + 7), k0);
        OPENCRAB2_AES_ROUNDS8(kp, b0, b1, b2, b3, b4, b5, b6, b7)
        __m128i* q = reinterpret_cast<__m128i*>(out + 16 * i);
        _mm_storeu_si128(q + 0, b0);
        _mm_storeu_si128(q + 1, b1);
        _mm_storeu_si128(q + 2, b2);
        _mm_storeu_si128(q + 3, b3);
        _mm_storeu_si128(q + 4, b4);
        _mm_storeu_si128(q + 5, b5);
        _mm_storeu_si128(q + 6, b6);
        _mm_storeu_si128(q + 7, b7);
    }
    for (; i < n; ++i) {
        const __m128i b = _mm_xor_si128(_mm_loadu_si128(reinterpret_cast<const __m128i*>(in + 16 * i)), k0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 16 * i), ni_one(kp, b));
    }
}

// out block (k * repeat + j) = E(b_k xor (0, hx[k * repeat + j])) xor b_k, eight blocks in flight.
// The operands are built from 64-bit loads (_mm_set_epi64x): the callers write seeds and tweak
// words as 64-bit stores, and a 128-bit load on top of two fresh 64-bit stores would miss
// store forwarding on every block. The base block of an output block is parked in its output
// slot during the rounds (it is needed again only for the final xor), which keeps it out of the
// register budget.
OPENCRAB2_AES_TARGET void ni_mmo(const uint8_t* rk, const uint64_t* base, std::size_t n_base, std::size_t repeat,
                                 const uint64_t* hx, uint64_t* out) {
    // A private copy of the round keys: loads from it cannot alias the stores to out, so the
    // compiler keeps them out of the loop (a key load that waits on an unrelated store delays
    // all eight blocks of a round).
    __m128i kp[11];
    for (int r = 0; r < 11; ++r) kp[r] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rk) + r);
    const __m128i k0 = kp[0];
    const std::size_t n = n_base * repeat;
    std::size_t kb = 0;   // base block of the next output block
    std::size_t j = 0;    // its index within the base block
    std::size_t i = 0;
#define OPENCRAB2_MMO_LOAD(u, bu)                                                                                    \
    __m128i bu;                                                                                                      \
    {                                                                                                                \
        const __m128i sd = _mm_set_epi64x(static_cast<long long>(base[2 * kb + 1]), static_cast<long long>(base[2 * kb])); \
        _mm_storeu_si128(q + (u), sd);                                                                               \
        bu = _mm_xor_si128(_mm_xor_si128(sd, _mm_set_epi64x(static_cast<long long>(hx[i + (u)]), 0)), k0);          \
        if (++j == repeat) {                                                                                         \
            j = 0;                                                                                                   \
            ++kb;                                                                                                    \
        }                                                                                                            \
    }
    for (; i + 8 <= n; i += 8) {
        __m128i* q = reinterpret_cast<__m128i*>(out + 2 * i);
        OPENCRAB2_MMO_LOAD(0, b0)
        OPENCRAB2_MMO_LOAD(1, b1)
        OPENCRAB2_MMO_LOAD(2, b2)
        OPENCRAB2_MMO_LOAD(3, b3)
        OPENCRAB2_MMO_LOAD(4, b4)
        OPENCRAB2_MMO_LOAD(5, b5)
        OPENCRAB2_MMO_LOAD(6, b6)
        OPENCRAB2_MMO_LOAD(7, b7)
        OPENCRAB2_AES_ROUNDS8(kp, b0, b1, b2, b3, b4, b5, b6, b7)
        _mm_storeu_si128(q + 0, _mm_xor_si128(b0, _mm_loadu_si128(q + 0)));
        _mm_storeu_si128(q + 1, _mm_xor_si128(b1, _mm_loadu_si128(q + 1)));
        _mm_storeu_si128(q + 2, _mm_xor_si128(b2, _mm_loadu_si128(q + 2)));
        _mm_storeu_si128(q + 3, _mm_xor_si128(b3, _mm_loadu_si128(q + 3)));
        _mm_storeu_si128(q + 4, _mm_xor_si128(b4, _mm_loadu_si128(q + 4)));
        _mm_storeu_si128(q + 5, _mm_xor_si128(b5, _mm_loadu_si128(q + 5)));
        _mm_storeu_si128(q + 6, _mm_xor_si128(b6, _mm_loadu_si128(q + 6)));
        _mm_storeu_si128(q + 7, _mm_xor_si128(b7, _mm_loadu_si128(q + 7)));
    }
    for (; i < n; ++i) {
        __m128i* q = reinterpret_cast<__m128i*>(out + 2 * i);
        OPENCRAB2_MMO_LOAD(0, b)
        _mm_storeu_si128(q, _mm_xor_si128(ni_one(kp, b), _mm_loadu_si128(q)));
    }
#undef OPENCRAB2_MMO_LOAD
}
#undef OPENCRAB2_AES_ROUNDS8
#endif

}  // namespace

bool Aes128::hardware_available() {
#if defined(OPENCRAB2_AES_HARDWARE)
    return selection().available;
#else
    return false;
#endif
}

bool Aes128::hardware_enabled() { return use_hardware(); }

void Aes128::force_portable(bool on) { g_force_portable.store(on, std::memory_order_relaxed); }

const char* Aes128::implementation() { return use_hardware() ? "aes-ni" : "portable"; }

void Aes128::set_key(const uint8_t key[16]) {
    for (int i = 0; i < 4; ++i) rk_[i] = load_be32(key + 4 * i);
    for (int i = 4; i < 44; ++i) {
        uint32_t temp = rk_[i - 1];
        if ((i & 3) == 0) {
            const uint32_t rot = (temp << 8) | (temp >> 24);
            temp = sub_word(rot) ^ (static_cast<uint32_t>(kRcon[i / 4 - 1]) << 24);
        }
        rk_[i] = rk_[i - 4] ^ temp;
    }
    // The hardware path wants the round keys as bytes: word i is bytes 4i..4i+3, big-endian.
    for (int i = 0; i < 44; ++i) store_be32(rkb_ + 4 * i, rk_[i]);
}

void Aes128::encrypt_state(uint32_t s[4]) const {
    const Tables& T = tables();
    uint32_t s0 = s[0] ^ rk_[0];
    uint32_t s1 = s[1] ^ rk_[1];
    uint32_t s2 = s[2] ^ rk_[2];
    uint32_t s3 = s[3] ^ rk_[3];
    for (int r = 1; r < 10; ++r) {
        const uint32_t* k = rk_ + 4 * r;
        const uint32_t t0 = T.te0[s0 >> 24] ^ T.te1[(s1 >> 16) & 0xff] ^ T.te2[(s2 >> 8) & 0xff] ^
                            T.te3[s3 & 0xff] ^ k[0];
        const uint32_t t1 = T.te0[s1 >> 24] ^ T.te1[(s2 >> 16) & 0xff] ^ T.te2[(s3 >> 8) & 0xff] ^
                            T.te3[s0 & 0xff] ^ k[1];
        const uint32_t t2 = T.te0[s2 >> 24] ^ T.te1[(s3 >> 16) & 0xff] ^ T.te2[(s0 >> 8) & 0xff] ^
                            T.te3[s1 & 0xff] ^ k[2];
        const uint32_t t3 = T.te0[s3 >> 24] ^ T.te1[(s0 >> 16) & 0xff] ^ T.te2[(s1 >> 8) & 0xff] ^
                            T.te3[s2 & 0xff] ^ k[3];
        s0 = t0;
        s1 = t1;
        s2 = t2;
        s3 = t3;
    }
    const uint32_t* k = rk_ + 40;
    s[0] = ((static_cast<uint32_t>(kSbox[s0 >> 24]) << 24) | (static_cast<uint32_t>(kSbox[(s1 >> 16) & 0xff]) << 16) |
            (static_cast<uint32_t>(kSbox[(s2 >> 8) & 0xff]) << 8) | static_cast<uint32_t>(kSbox[s3 & 0xff])) ^ k[0];
    s[1] = ((static_cast<uint32_t>(kSbox[s1 >> 24]) << 24) | (static_cast<uint32_t>(kSbox[(s2 >> 16) & 0xff]) << 16) |
            (static_cast<uint32_t>(kSbox[(s3 >> 8) & 0xff]) << 8) | static_cast<uint32_t>(kSbox[s0 & 0xff])) ^ k[1];
    s[2] = ((static_cast<uint32_t>(kSbox[s2 >> 24]) << 24) | (static_cast<uint32_t>(kSbox[(s3 >> 16) & 0xff]) << 16) |
            (static_cast<uint32_t>(kSbox[(s0 >> 8) & 0xff]) << 8) | static_cast<uint32_t>(kSbox[s1 & 0xff])) ^ k[2];
    s[3] = ((static_cast<uint32_t>(kSbox[s3 >> 24]) << 24) | (static_cast<uint32_t>(kSbox[(s0 >> 16) & 0xff]) << 16) |
            (static_cast<uint32_t>(kSbox[(s1 >> 8) & 0xff]) << 8) | static_cast<uint32_t>(kSbox[s2 & 0xff])) ^ k[3];
}

// ---------------------------------------------------------------------------------------------
// portable path
// ---------------------------------------------------------------------------------------------
void Aes128::encrypt_blocks_portable(const uint8_t* in, uint8_t* out, std::size_t n_blocks) const {
    for (std::size_t i = 0; i < n_blocks; ++i) {
        const uint8_t* p = in + 16 * i;
        uint32_t s[4] = {load_be32(p), load_be32(p + 4), load_be32(p + 8), load_be32(p + 12)};
        encrypt_state(s);
        uint8_t* q = out + 16 * i;
        store_be32(q, s[0]);
        store_be32(q + 4, s[1]);
        store_be32(q + 8, s[2]);
        store_be32(q + 12, s[3]);
    }
}

void Aes128::encrypt_words_batch_portable(const uint64_t* in, uint64_t* out, std::size_t n_blocks) const {
    for (std::size_t i = 0; i < n_blocks; ++i) {
        // Byte j of the block is (lo >> 8j) for j < 8 and (hi >> 8(j-8)) for j >= 8. The state
        // words are big-endian column loads, hence the byte swaps.
        const uint64_t lo = in[2 * i];
        const uint64_t hi = in[2 * i + 1];
        uint32_t s[4] = {bswap32(static_cast<uint32_t>(lo)), bswap32(static_cast<uint32_t>(lo >> 32)),
                         bswap32(static_cast<uint32_t>(hi)), bswap32(static_cast<uint32_t>(hi >> 32))};
        encrypt_state(s);
        out[2 * i] = static_cast<uint64_t>(bswap32(s[0])) | (static_cast<uint64_t>(bswap32(s[1])) << 32);
        out[2 * i + 1] = static_cast<uint64_t>(bswap32(s[2])) | (static_cast<uint64_t>(bswap32(s[3])) << 32);
    }
}

void Aes128::mmo_words_batch_portable(const uint64_t* base, std::size_t n_base, std::size_t repeat, const uint64_t* hi_xor,
                                      uint64_t* out) const {
    for (std::size_t k = 0; k < n_base; ++k) {
        const uint64_t lo = base[2 * k];
        const uint64_t hi = base[2 * k + 1];
        const uint32_t lo0 = bswap32(static_cast<uint32_t>(lo));
        const uint32_t lo1 = bswap32(static_cast<uint32_t>(lo >> 32));
        for (std::size_t j = 0; j < repeat; ++j) {
            const std::size_t i = k * repeat + j;
            const uint64_t xhi = hi ^ hi_xor[i];
            uint32_t s[4] = {lo0, lo1, bswap32(static_cast<uint32_t>(xhi)), bswap32(static_cast<uint32_t>(xhi >> 32))};
            encrypt_state(s);
            out[2 * i] = (static_cast<uint64_t>(bswap32(s[0])) | (static_cast<uint64_t>(bswap32(s[1])) << 32)) ^ lo;
            out[2 * i + 1] = (static_cast<uint64_t>(bswap32(s[2])) | (static_cast<uint64_t>(bswap32(s[3])) << 32)) ^ hi;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// hardware path (x86-64 is little-endian: a (lo, hi) word pair in memory IS the 16 block bytes)
// ---------------------------------------------------------------------------------------------
bool Aes128::encrypt_blocks_hardware(const uint8_t* in, uint8_t* out, std::size_t n_blocks) const {
#if defined(OPENCRAB2_AES_HARDWARE)
    if (!selection().available) return false;
    ni_encrypt(rkb_, in, out, n_blocks);
    return true;
#else
    (void)in;
    (void)out;
    (void)n_blocks;
    return false;
#endif
}

bool Aes128::encrypt_words_batch_hardware(const uint64_t* in, uint64_t* out, std::size_t n_blocks) const {
#if defined(OPENCRAB2_AES_HARDWARE)
    if (!selection().available) return false;
    ni_encrypt(rkb_, reinterpret_cast<const uint8_t*>(in), reinterpret_cast<uint8_t*>(out), n_blocks);
    return true;
#else
    (void)in;
    (void)out;
    (void)n_blocks;
    return false;
#endif
}

bool Aes128::mmo_words_batch_hardware(const uint64_t* base, std::size_t n_base, std::size_t repeat, const uint64_t* hi_xor,
                                      uint64_t* out) const {
#if defined(OPENCRAB2_AES_HARDWARE)
    if (!selection().available) return false;
    ni_mmo(rkb_, base, n_base, repeat, hi_xor, out);
    return true;
#else
    (void)base;
    (void)n_base;
    (void)repeat;
    (void)hi_xor;
    (void)out;
    return false;
#endif
}

// ---------------------------------------------------------------------------------------------
// selected path
// ---------------------------------------------------------------------------------------------
void Aes128::encrypt_blocks(const uint8_t* in, uint8_t* out, std::size_t n_blocks) const {
#if defined(OPENCRAB2_AES_HARDWARE)
    if (use_hardware()) {
        ni_encrypt(rkb_, in, out, n_blocks);
        return;
    }
#endif
    encrypt_blocks_portable(in, out, n_blocks);
}

void Aes128::encrypt_words_batch(const uint64_t* in, uint64_t* out, std::size_t n_blocks) const {
#if defined(OPENCRAB2_AES_HARDWARE)
    if (use_hardware()) {
        ni_encrypt(rkb_, reinterpret_cast<const uint8_t*>(in), reinterpret_cast<uint8_t*>(out), n_blocks);
        return;
    }
#endif
    encrypt_words_batch_portable(in, out, n_blocks);
}

void Aes128::mmo_words_batch(const uint64_t* base, std::size_t n_base, std::size_t repeat, const uint64_t* hi_xor,
                             uint64_t* out) const {
#if defined(OPENCRAB2_AES_HARDWARE)
    if (use_hardware()) {
        ni_mmo(rkb_, base, n_base, repeat, hi_xor, out);
        return;
    }
#endif
    mmo_words_batch_portable(base, n_base, repeat, hi_xor, out);
}

void Aes128::encrypt_block(const uint8_t in[16], uint8_t out[16]) const { encrypt_blocks(in, out, 1); }

void Aes128::encrypt_words(uint64_t lo, uint64_t hi, uint64_t& out_lo, uint64_t& out_hi) const {
    uint64_t w[2] = {lo, hi};
    encrypt_words_batch(w, w, 1);
    out_lo = w[0];
    out_hi = w[1];
}

}  // namespace lowlevel
}  // namespace opencrab2
