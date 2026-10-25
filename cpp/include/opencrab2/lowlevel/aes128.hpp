// AES-128 block encryption, FIPS-197, encrypt-only.
//
// Used by the fixed-key Matyas-Meyer-Oseas PRG of the GGM tree. Only the forward cipher is
// needed and implemented. Two implementations compute the same function:
//
//   portable   the classic 32-bit T-table formulation (four 1 KiB tables derived from the S-box
//              at first use). It has no data-dependent branches, but like every table-based AES
//              it is not cache-timing hardened. This is acceptable here because the key is public
//              (fixed) and the inputs are public masked values or dealer seeds that never cross a
//              trust boundary through this cipher.
//   hardware   x86-64 AES-NI (<wmmintrin.h>), compiled only into the functions that carry the
//              "aes" target attribute, so the translation unit needs no -maes flag and the rest
//              of the library still runs on a CPU without the extension. Eight blocks are kept
//              in flight per round, which is where the batch interfaces below gain their speed.
//
// Selection happens at run time: the hardware path is used when the build has it, CPUID leaf 1
// reports AES (ECX bit 25) and nobody forced the portable path. The portable path is forced by
// the environment variable OPENCRAB2_AES=portable (read once, at the first use) or by
// Aes128::force_portable(true); both exist so that the tests can run every vector on both paths.
// The two paths are bit-identical by construction and by test.
//
// Tested (tests/test_lowlevel.cpp): FIPS-197 Appendix C.1 vector
// (key 000102..0f, pt 00112233..ff -> ct 69c4e0d86a7b0430d8cdb78070b4c55a) and the
// Appendix B worked example (key 2b7e1516..., the PRG fixed key), on both paths and through every
// interface; random batches of every length 0..40 hardware against portable; whole DCF key
// generations and evaluations with the portable path forced against the default path.
#pragma once

#include <cstddef>
#include <cstdint>

namespace opencrab2 {
namespace lowlevel {

class Aes128 {
public:
    static constexpr std::size_t block_bytes = 16;
    static constexpr std::size_t key_bytes = 16;

    Aes128() = default;
    explicit Aes128(const uint8_t key[16]) { set_key(key); }

    void set_key(const uint8_t key[16]);

    // One block, byte interface.
    void encrypt_block(const uint8_t in[16], uint8_t out[16]) const;

    // One block, little-endian word interface: lo holds bytes 0..7, hi holds bytes 8..15,
    // each read as a little-endian uint64 (the layout numpy uses for an (M,2) uint64 view).
    void encrypt_words(uint64_t lo, uint64_t hi, uint64_t& out_lo, uint64_t& out_hi) const;

    // n_blocks independent blocks (ECB), byte interface: in and out hold 16 * n_blocks bytes.
    // in == out is allowed; a partial overlap is not.
    void encrypt_blocks(const uint8_t* in, uint8_t* out, std::size_t n_blocks) const;

    // n_blocks independent blocks, word interface: block i is (in[2i], in[2i+1]) = (lo, hi) as in
    // encrypt_words. in == out is allowed; a partial overlap is not.
    void encrypt_words_batch(const uint64_t* in, uint64_t* out, std::size_t n_blocks) const;

    // Fixed-key hashing in one pass (the Matyas-Meyer-Oseas shape of lowlevel/prg.hpp): `repeat`
    // blocks from each of n_base base blocks b_k = (base[2k], base[2k+1]),
    //     out block (k * repeat + j) = E(b_k xor (0, hi_xor[k * repeat + j])) xor b_k,
    // where (0, h) is the block whose lo word is 0 and whose hi word is h. The hardware path keeps
    // the whole computation in registers (no intermediate buffer, eight blocks in flight). out
    // must not overlap base or hi_xor.
    void mmo_words_batch(const uint64_t* base, std::size_t n_base, std::size_t repeat, const uint64_t* hi_xor,
                         uint64_t* out) const;

    // The same batch functions on one named path, whatever the run-time selection says
    // (differential tests). The hardware variants return false and write nothing when the build
    // or the CPU has no AES-NI.
    void encrypt_blocks_portable(const uint8_t* in, uint8_t* out, std::size_t n_blocks) const;
    void encrypt_words_batch_portable(const uint64_t* in, uint64_t* out, std::size_t n_blocks) const;
    bool encrypt_blocks_hardware(const uint8_t* in, uint8_t* out, std::size_t n_blocks) const;
    bool encrypt_words_batch_hardware(const uint64_t* in, uint64_t* out, std::size_t n_blocks) const;
    void mmo_words_batch_portable(const uint64_t* base, std::size_t n_base, std::size_t repeat, const uint64_t* hi_xor,
                                  uint64_t* out) const;
    bool mmo_words_batch_hardware(const uint64_t* base, std::size_t n_base, std::size_t repeat, const uint64_t* hi_xor,
                                  uint64_t* out) const;

    // True when this build carries the AES-NI path and the CPU reports the extension.
    static bool hardware_available();
    // True when the hardware path is the one in use (available and not forced off).
    static bool hardware_enabled();
    // Process-wide switch: true routes every call through the portable path, false returns to the
    // run-time selection (which still honours OPENCRAB2_AES=portable). Safe to call at any time
    // from any thread; both paths compute the same function, so a concurrent caller only changes
    // speed.
    static void force_portable(bool on);
    // "aes-ni" or "portable": the path in use right now (evidence records and test logs).
    static const char* implementation();

private:
    void encrypt_state(uint32_t s[4]) const;
    uint32_t rk_[44] = {};            // round keys as big-endian column words (portable path)
    alignas(16) uint8_t rkb_[176] = {};   // the same round keys as 11 * 16 bytes (hardware path)
};

}  // namespace lowlevel
}  // namespace opencrab2
