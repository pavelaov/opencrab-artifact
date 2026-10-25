// Fixed-key Matyas-Meyer-Oseas PRG for the GGM tree of the DCF keys.
//
//     B_j(s) = AES_K(s xor c_j) xor s,   K = 2b7e151628aed2a6abf7158809cf4f3c,
//     c_j = the 16-byte big-endian integer j.
//
// Base layout (tweaks 0..3). The 64-byte expansion out = concat_{j=0..3} B_j(s) of a 16-byte seed
// is split as
//
//     sL = out[0:16], sR = out[16:32], vL = out[32:40] (LE u64), vR = out[40:48] (LE u64),
//     tL = out[48] & 1, tR = out[49] & 1.
//
// Component-specific Convert (the comparison-key repair of paper, Section 5.3 and Appendix D,
// Disclosure; binding in both languages). Every payload component c of a DCF key reads its own
// pseudo-random words:
//
//     seeds             sL = B_0(s), sR = B_1(s)
//     control bits      tL = B_3(s)[0] & 1, tR = B_3(s)[1] & 1
//     Z_{2^64}, c = 0   vL_0 = LE64(B_2(s)[0:8]),     vR_0 = LE64(B_2(s)[8:16])      (the base layout)
//     Z_{2^64}, c >= 1  vL_c = LE64(B_{3+c}(s)[0:8]), vR_c = LE64(B_{3+c}(s)[8:16])  (tweaks 4, 5, ...)
//     Z_t, any c        vL_c = LE128(B_{4+2c}(s)) mod t, vR_c = LE128(B_{5+2c}(s)) mod t
//                       (block 2 is unused in Z_t mode)
//
// The reuse of ONE value word for every payload component (the pre-repair rule of the superseded
// engine) let a party read alpha and beta_1 - beta_0 from its own key; see the
// KEY_COMPONENT_INDEPENDENCE checks in tests/test_lowlevel.cpp. A 64-bit word reduced mod t has
// statistical distance t / 2^64 from uniform; the 128-bit block reduced mod t has at most
// t / 2^128 (below 2^-66 for every admitted t < 2^62).
//
// Blocks are handled as two little-endian uint64 words (lo = bytes 0..7, hi = bytes 8..15), which
// is the view numpy takes of an (M,2) uint64 array; the tweak c_j therefore only touches the hi
// word (byte 15 carries j & 0xff, byte 14 the next byte, and so on). tR is taken from byte 49,
// as in the Python implementation; the byte index is the compile-time constant
// OPENCRAB2_PRG_TR_BYTE (default 49).
//
// Security holds in the ideal-cipher model with a public fixed key; this PRG must only be used
// inside the GGM tree, never as a general CSPRNG (see csprng.hpp for that).
//
// Tested (tests/test_lowlevel.cpp): expand(seed 00..0f) against a direct AES computation, the
// byte layout, mmo on tweaks 0..9 and on multi-byte tweaks against a byte-level computation, the
// 128-bit reduction against an independent shift-and-reduce routine, and
// tests/vectors/cpp_prg_vectors.json is emitted for the Python cross-check.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "opencrab2/lowlevel/aes128.hpp"

#ifndef OPENCRAB2_PRG_TR_BYTE
#define OPENCRAB2_PRG_TR_BYTE 49
#endif

namespace opencrab2 {
namespace lowlevel {

constexpr int PRG_TL_BYTE = 48;
constexpr int PRG_TR_BYTE = OPENCRAB2_PRG_TR_BYTE;
static_assert(PRG_TR_BYTE >= 48 && PRG_TR_BYTE < 64, "tR must come from the fourth AES block");

extern const uint8_t PRG_FIXED_KEY[16];

// A 16-byte block as two little-endian uint64 words.
struct Block128 {
    uint64_t lo = 0;
    uint64_t hi = 0;
};

inline bool operator==(const Block128& a, const Block128& b) { return a.lo == b.lo && a.hi == b.hi; }
inline bool operator!=(const Block128& a, const Block128& b) { return !(a == b); }
inline Block128 operator^(const Block128& a, const Block128& b) { return Block128{a.lo ^ b.lo, a.hi ^ b.hi}; }

Block128 block_from_bytes(const uint8_t b[16]);
void block_to_bytes(const Block128& blk, uint8_t out[16]);

// The fixed-key cipher AES_K (constructed once).
const Aes128& prg_cipher();

// One MMO block B_j(s) = AES_K(s xor c_j) xor s; c_j is the 16-byte big-endian integer j (any
// 32-bit j; the DCF uses j <= 5 + 2 * 254).
Block128 mmo(const Block128& s, uint32_t tweak);

// The tweak c_j as it meets a block held as two little-endian words: c_j is zero in bytes 0..11
// and keeps the 32-bit j big-endian in bytes 12..15 (byte 15 = j & 0xff), so it only touches the
// hi word; byte 15 is bits 56..63 of that word, byte 14 bits 48..55, and so on (the byte-swapped
// j in the top half). For j < 256 this is j << 56. s xor c_j == Block128{s.lo, s.hi ^ prg_tweak_word(j)}.
constexpr uint64_t prg_tweak_word(uint32_t j) {
    return ((static_cast<uint64_t>(j) & 0xffu) << 56) | (((static_cast<uint64_t>(j) >> 8) & 0xffu) << 48) |
           (((static_cast<uint64_t>(j) >> 16) & 0xffu) << 40) | (((static_cast<uint64_t>(j) >> 24) & 0xffu) << 32);
}

// n MMO blocks in one cipher call: out[i] = mmo(seeds[i], tweaks[i]). The blocks are independent,
// so the hardware AES path keeps eight of them in flight. out must not overlap seeds.
void mmo_batch(const Block128* seeds, const uint32_t* tweaks, Block128* out, std::size_t n);

// per_seed blocks of each of n_seeds seeds in one cipher call:
//     out[k * per_seed + j] = B_j'(seeds[k]),  tweak_words[k * per_seed + j] = prg_tweak_word(j').
// This is what the DCF loops use: all blocks of one tree level of a group of keys per call, with
// the tweak words laid out once. out must not overlap seeds or tweak_words.
void mmo_batch_repeat(const Block128* seeds, std::size_t n_seeds, std::size_t per_seed, const uint64_t* tweak_words,
                      Block128* out);

// Tweak indices of the component-specific Convert.
constexpr uint32_t PRG_TWEAK_SEED_L = 0;
constexpr uint32_t PRG_TWEAK_SEED_R = 1;
constexpr uint32_t PRG_TWEAK_VALUE0 = 2;    // Z_{2^64} component 0 (vL = lo word, vR = hi word)
constexpr uint32_t PRG_TWEAK_CONTROL = 3;
// Z_{2^64} component c: block 2 for c = 0, block 3 + c for c >= 1 (vL = lo word, vR = hi word).
constexpr uint32_t prg_ring_value_tweak(uint32_t c) { return c == 0 ? PRG_TWEAK_VALUE0 : 3u + c; }
// Z_t component c, side 0 = L, 1 = R: block 4 + 2c + side, reduced mod t as a 128-bit integer.
constexpr uint32_t prg_zt_value_tweak(uint32_t c, uint32_t side) { return 4u + 2u * c + (side & 1u); }

struct Expansion {
    Block128 sL;
    Block128 sR;
    uint64_t vL = 0;   // Z_{2^64} component 0 only; other components and Z_t use the Convert below
    uint64_t vR = 0;
    uint64_t tL = 0;
    uint64_t tR = 0;
};

// Control bits of the fourth block (tweak 3): tL from byte 48, tR from byte PRG_TR_BYTE.
inline uint64_t prg_tl_bit(const Block128& block3) { return block3.lo & 1u; }
inline uint64_t prg_tr_bit(const Block128& block3) {
    const int local = PRG_TR_BYTE - 48;
    const uint64_t word = (local >> 3) ? block3.hi : block3.lo;
    return (word >> ((local & 7) * 8)) & 1u;
}

// The base layout (tweaks 0..3).
Expansion expand(const Block128& seed);
Expansion expand(const uint8_t seed[16]);
void expand_batch(const std::vector<Block128>& seeds, std::vector<Expansion>& out);

// The raw 64-byte layout above (cross-language test vectors).
void expand_bytes(const uint8_t seed[16], uint8_t out[64]);
// One raw MMO block B_j(seed) as 16 bytes (cross-language vectors for the tweaks above 3).
void mmo_bytes(const uint8_t seed[16], uint32_t tweak, uint8_t out[16]);

// ---------------------------------------------------------------------------------------------
// Component-specific Convert (one set of pseudo-random words per payload component)
// ---------------------------------------------------------------------------------------------
// LE128(block) mod t for 1 <= t < 2^63 (the DCF admits odd t < 2^62). On x86-64 one 64-bit
// remainder and one DIV with a 128-bit dividend; elsewhere unsigned __int128.
uint64_t block_mod(const Block128& b, uint64_t t);
// The same value by a shift-and-reduce loop without 128-bit arithmetic (test cross-check, and the
// fallback on compilers that lack __int128).
uint64_t block_mod_portable(const Block128& b, uint64_t t);

// Convert words of one seed for payload component c.
struct ValueWords {
    uint64_t vL = 0;
    uint64_t vR = 0;
};
ValueWords value_words_ring(const Block128& s, uint32_t c);              // one AES block
ValueWords value_words_zt(const Block128& s, uint32_t c, uint64_t t);    // two AES blocks
// Only the word of one side (0 = L, 1 = R): what evaluation needs, one AES block either way.
uint64_t value_word_ring(const Block128& s, uint32_t c, uint32_t side);
uint64_t value_word_zt(const Block128& s, uint32_t c, uint32_t side, uint64_t t);

}  // namespace lowlevel
}  // namespace opencrab2
