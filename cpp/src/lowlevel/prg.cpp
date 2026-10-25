// Fixed-key MMO PRG. See include/opencrab2/lowlevel/prg.hpp.
#include "opencrab2/lowlevel/prg.hpp"

namespace opencrab2 {
namespace lowlevel {

const uint8_t PRG_FIXED_KEY[16] = {0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
                                   0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};

Block128 block_from_bytes(const uint8_t b[16]) {
    Block128 out;
    for (int i = 7; i >= 0; --i) out.lo = (out.lo << 8) | b[i];
    for (int i = 15; i >= 8; --i) out.hi = (out.hi << 8) | b[i];
    return out;
}

void block_to_bytes(const Block128& blk, uint8_t out[16]) {
    for (int i = 0; i < 8; ++i) out[i] = static_cast<uint8_t>(blk.lo >> (8 * i));
    for (int i = 0; i < 8; ++i) out[8 + i] = static_cast<uint8_t>(blk.hi >> (8 * i));
}

const Aes128& prg_cipher() {
    static const Aes128 cipher(PRG_FIXED_KEY);
    return cipher;
}

static_assert(sizeof(Block128) == 16, "Block128 is two uint64 words without padding");

Block128 mmo(const Block128& s, uint32_t tweak) {
    const uint64_t tw = prg_tweak_word(tweak);
    Block128 y;
    prg_cipher().mmo_words_batch(&s.lo, 1, 1, &tw, &y.lo);
    return y;
}

void mmo_batch(const Block128* seeds, const uint32_t* tweaks, Block128* out, std::size_t n) {
    // tweak words on the stack, 64 blocks per cipher call
    const std::size_t kChunk = 64;
    uint64_t tw[kChunk];
    for (std::size_t i0 = 0; i0 < n; i0 += kChunk) {
        const std::size_t w = (n - i0 < kChunk) ? (n - i0) : kChunk;
        for (std::size_t i = 0; i < w; ++i) tw[i] = prg_tweak_word(tweaks[i0 + i]);
        // A Block128 array is a (lo, hi) uint64 array: the word interface of the cipher.
        prg_cipher().mmo_words_batch(&seeds[i0].lo, w, 1, tw, &out[i0].lo);
    }
}

void mmo_batch_repeat(const Block128* seeds, std::size_t n_seeds, std::size_t per_seed, const uint64_t* tweak_words,
                      Block128* out) {
    if (n_seeds == 0 || per_seed == 0) return;
    prg_cipher().mmo_words_batch(&seeds[0].lo, n_seeds, per_seed, tweak_words, &out[0].lo);
}

Expansion expand(const Block128& seed) {
    static const uint64_t kTweaks[4] = {prg_tweak_word(PRG_TWEAK_SEED_L), prg_tweak_word(PRG_TWEAK_SEED_R),
                                        prg_tweak_word(PRG_TWEAK_VALUE0), prg_tweak_word(PRG_TWEAK_CONTROL)};
    Block128 b[4];
    mmo_batch_repeat(&seed, 1, 4, kTweaks, b);
    Expansion e;
    e.sL = b[0];
    e.sR = b[1];
    e.vL = b[2].lo;
    e.vR = b[2].hi;
    e.tL = prg_tl_bit(b[3]);
    e.tR = prg_tr_bit(b[3]);
    return e;
}

Expansion expand(const uint8_t seed[16]) { return expand(block_from_bytes(seed)); }

void expand_batch(const std::vector<Block128>& seeds, std::vector<Expansion>& out) {
    out.resize(seeds.size());
    // 16 seeds (64 blocks) per cipher call, on the stack
    const std::size_t kChunk = 16;
    Block128 b[4 * kChunk];
    uint64_t tw[4 * kChunk];
    for (std::size_t j = 0; j < 4 * kChunk; ++j) tw[j] = prg_tweak_word(static_cast<uint32_t>(j & 3u));
    for (std::size_t i0 = 0; i0 < seeds.size(); i0 += kChunk) {
        const std::size_t w = (seeds.size() - i0 < kChunk) ? (seeds.size() - i0) : kChunk;
        mmo_batch_repeat(&seeds[i0], w, 4, tw, b);
        for (std::size_t k = 0; k < w; ++k) {
            Expansion& e = out[i0 + k];
            e.sL = b[4 * k + 0];
            e.sR = b[4 * k + 1];
            e.vL = b[4 * k + 2].lo;
            e.vR = b[4 * k + 2].hi;
            e.tL = prg_tl_bit(b[4 * k + 3]);
            e.tR = prg_tr_bit(b[4 * k + 3]);
        }
    }
}

void expand_bytes(const uint8_t seed[16], uint8_t out[64]) {
    static const uint64_t kTweaks[4] = {prg_tweak_word(0), prg_tweak_word(1), prg_tweak_word(2), prg_tweak_word(3)};
    const Block128 s = block_from_bytes(seed);
    Block128 b[4];
    mmo_batch_repeat(&s, 1, 4, kTweaks, b);
    for (uint32_t j = 0; j < 4; ++j) block_to_bytes(b[j], out + 16 * j);
}

void mmo_bytes(const uint8_t seed[16], uint32_t tweak, uint8_t out[16]) {
    block_to_bytes(mmo(block_from_bytes(seed), tweak), out);
}

// ---------------------------------------------------------------------------------------------
// Component-specific Convert (one set of pseudo-random words per payload component)
// ---------------------------------------------------------------------------------------------
uint64_t block_mod_portable(const Block128& b, uint64_t t) {
    // Horner over the 128 bits, most significant first. r < t < 2^63, so 2r + 1 < 2^64: no wrap.
    uint64_t r = 0;
    for (int i = 127; i >= 0; --i) {
        const uint64_t bit = (i >= 64) ? ((b.hi >> (i - 64)) & 1u) : ((b.lo >> i) & 1u);
        r = (r << 1) | bit;
        r -= t & (uint64_t(0) - static_cast<uint64_t>(r >= t));
    }
    return r;
}

uint64_t block_mod(const Block128& b, uint64_t t) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    // (hi * 2^64 + lo) mod t = ((hi mod t) * 2^64 + lo) mod t. With hi mod t < t the quotient of
    // the second division fits 64 bits, so one DIV instruction (128-bit dividend in RDX:RAX)
    // does it; the generic 128-bit remainder goes through a library call instead. Same value.
    uint64_t rem = b.hi % t;
    uint64_t quo = b.lo;
    __asm__("divq %[d]" : "+a"(quo), "+d"(rem) : [d] "r"(t) : "cc");
    (void)quo;
    return rem;
#elif defined(__SIZEOF_INT128__)
    const unsigned __int128 v = (static_cast<unsigned __int128>(b.hi) << 64) | b.lo;
    return static_cast<uint64_t>(v % t);
#else
    return block_mod_portable(b, t);
#endif
}

ValueWords value_words_ring(const Block128& s, uint32_t c) {
    const Block128 b = mmo(s, prg_ring_value_tweak(c));
    return ValueWords{b.lo, b.hi};
}

ValueWords value_words_zt(const Block128& s, uint32_t c, uint64_t t) {
    const uint64_t tw[2] = {prg_tweak_word(prg_zt_value_tweak(c, 0)), prg_tweak_word(prg_zt_value_tweak(c, 1))};
    Block128 b[2];
    mmo_batch_repeat(&s, 1, 2, tw, b);
    ValueWords w;
    w.vL = block_mod(b[0], t);
    w.vR = block_mod(b[1], t);
    return w;
}

uint64_t value_word_ring(const Block128& s, uint32_t c, uint32_t side) {
    const Block128 b = mmo(s, prg_ring_value_tweak(c));
    const uint64_t m = uint64_t(0) - static_cast<uint64_t>(side & 1u);   // all ones for the R word
    return b.lo ^ ((b.lo ^ b.hi) & m);
}

uint64_t value_word_zt(const Block128& s, uint32_t c, uint32_t side, uint64_t t) {
    return block_mod(mmo(s, prg_zt_value_tweak(c, side)), t);
}

}  // namespace lowlevel
}  // namespace opencrab2
