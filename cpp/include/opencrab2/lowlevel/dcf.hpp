// Batched GGM Distributed Comparison Function (paper, Section 2.2, DCF comparison).
//
// GGM key batches on the fixed-key MMO PRG (prg.hpp) with a payload group option, the same key
// layout as the Python implementation (lowlevel/dcf.py). For M independent keys this module realizes
//
//     f^{<}_{alpha[m], beta[m]}(x) = beta[m]  if  x < alpha[m]  else  0
//
// over an n_bits-bit input domain (1 <= n_bits <= 64), with the payload group
//
//   * modulus == 0 : Z_{2^64}^P, P payload words, uint64 wrap;
//   * modulus == t : Z_t^P (t odd, 3 <= t < 2^62). Payload words are reduced mod t, PRG value words
//                    are 128-bit blocks reduced mod t, every correction word is computed mod t,
//                    outputs are below t and party 1 negates mod t.
//
// Construction: Boyle, Chandran, Gilboa, Gupta, Ishai, Kumar and Rathee (Eurocrypt 2021, Fig. 1)
// with the product-group Convert: one
// GGM level per input bit with a seed correction word, P value correction words and two
// control-bit corrections, plus a final value correction word.
//
// Convert rule (the comparison-key repair: paper, Section 5.3 and Appendix D, Disclosure; see
// prg.hpp). Every payload
// component p reads its OWN pseudo-random words from a seed s, with B_j(s) = AES_K(s xor c_j) xor s:
//     Z_{2^64}: p = 0 uses block 2 (vL = LE64 bytes 0..7, vR = LE64 bytes 8..15), p >= 1 uses block
//               3 + p (tweaks 4, 5, ...). P = 1 keys are bit-identical to those of the pre-repair rule.
//     Z_t:      vL_p = LE128(B_{4+2p}(s)) mod t, vR_p = LE128(B_{5+2p}(s)) mod t; block 2 is unused.
//     leaf:     the final correction word of component p uses vL_p of the leaf seed.
// The pre-repair rule (of the superseded engine) reused one value word for all components.
// The public correction words then satisfied cw_values[i][1] - cw_values[i][0] =
// +-(a_i - a_{i-1}) (beta_1 - beta_0), which gave a party every bit of alpha and beta_1 - beta_0
// from its own key (for DReLU material: the mask r, hence the comparison input). The regression
// tests KEY_COMPONENT_INDEPENDENCE in tests/test_lowlevel.cpp reproduce the attack and pin its
// failure on the repaired keys.
//
// Evaluation expands only the branch selected by the public input bit: per key and level one seed
// block, one control block and one value block per payload component (three AES blocks at P = 1
// in either group, one more per additional component). Control-bit dependent corrections are
// applied with masks, never with branches, and message sizes depend only on (M, n_bits, P).
// Keys are evaluated 16 at a time, level by level, so that all the AES blocks of one level of the
// group go to the cipher in one call (lowlevel/aes128.hpp keeps eight blocks in flight on the
// hardware path); generation batches the blocks of one level of one key the same way. The outputs
// are bit-identical to a key-by-key, block-by-block walk (tests/test_lowlevel.cpp, "aes paths").
//
// In-memory layout (identical to the numpy arrays of the Python implementation, word-major per level):
//     seed      2*M      seed[2m] = lo word (bytes 0..7), seed[2m+1] = hi word (bytes 8..15)
//     cw_seeds  n*2*M    [(i*2 + w)*M + m]
//     cw_values n*P*M    [(i*P + p)*M + m]
//     cw_tL     n*M      [i*M + m]          cw_tR  n*M
//     cw_final  P*M      [p*M + m]
//
// Wire layout (binding, to_bytes default; the bytes of the Python lowlevel/dcf.py serialization):
//     u8 version = 2, u8 party, u8 n_bits, u8 P, u64 LE M, u64 LE modulus (0 = none),
//     then seed[M][2], cw_seeds[n][2][M], cw_values[n][P][M], cw_tL[n][M], cw_tR[n][M],
//     cw_final[P][M], all u64 LE.
// from_bytes also accepts the "OC2DCF01" legacy layout, which the Python reader takes (32-byte header,
// control bits packed one byte per level and key), and to_bytes(WireLayout::PythonV2) writes it.
//
// Key wire size per element (the quantity the cost model charges): 16 + n_bits*(16 + 8P + 1) + 8P.
//
// Tested (tests/test_lowlevel.cpp): exhaustive correctness for n_bits <= 8 in both payload
// groups and P in {1, 2, 3}, random correctness at 32 and 64 bits, serialization round trips in
// both layouts, slicing and concatenation, the byte formula, the four KEY_COMPONENT_INDEPENDENCE
// checks, and a pinned P = 1 vector of the pre-repair rule. Design limits: a Z_t value word
// is a 128-bit PRG block reduced mod t, statistical distance at most t / 2^128 from uniform per
// word (below 2^-66 for every admitted t < 2^62, 2^-92 for t = 68720050177; the session budget
// charges n_word * t / 2^128); PRG security is in the ideal-cipher model; keys are one-time
// objects enforced by the ledger, not here. What is NOT proven here: the tests show the specific
// correction-word relation is gone and that components use disjoint PRG blocks; pseudo-randomness
// of the key rests on the cited construction, not on these tests.
#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "opencrab2/expected.hpp"
#include "opencrab2/lowlevel/csprng.hpp"
#include "opencrab2/lowlevel/prg.hpp"

namespace opencrab2 {
namespace lowlevel {
namespace dcf {

using u64 = uint64_t;

constexpr u64 MAX_MODULUS = u64(1) << 62;
constexpr uint8_t WIRE_VERSION = 2;
constexpr std::size_t WIRE_HEADER_BYTES = 20;         // binding layout
constexpr std::size_t PY_WIRE_HEADER_BYTES = 32;      // "OC2DCF01" legacy layout (the Python reader's)

enum class WireLayout { Binding, PythonV2 };

Status check_n_bits(int n_bits);
Status check_modulus(u64 modulus);   // 0 (no modulus) is accepted
std::size_t key_bytes_per_element(int n_bits, int payloads);

// ---------------------------------------------------------------------------------------------
// Little-endian byte helpers shared by every material serializer of this layer
// ---------------------------------------------------------------------------------------------
namespace wire {
inline void put_u8(std::vector<uint8_t>& b, uint8_t v) { b.push_back(v); }
inline void put_u16le(std::vector<uint8_t>& b, uint16_t v) {
    b.push_back(static_cast<uint8_t>(v));
    b.push_back(static_cast<uint8_t>(v >> 8));
}
inline void put_u32le(std::vector<uint8_t>& b, uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back(static_cast<uint8_t>(v >> (8 * i)));
}
inline void put_u64le(std::vector<uint8_t>& b, u64 v) {
    for (int i = 0; i < 8; ++i) b.push_back(static_cast<uint8_t>(v >> (8 * i)));
}
inline void put_u64le_array(std::vector<uint8_t>& b, const std::vector<u64>& a) {
    for (u64 v : a) put_u64le(b, v);
}
inline uint16_t get_u16le(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }
inline uint32_t get_u32le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) | (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}
inline u64 get_u64le(const uint8_t* p) {
    u64 v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}
inline void get_u64le_array(const uint8_t* p, std::size_t count, std::vector<u64>& out) {
    out.resize(count);
    for (std::size_t i = 0; i < count; ++i) out[i] = get_u64le(p + 8 * i);
}
}  // namespace wire

// ---------------------------------------------------------------------------------------------
// Key batch
// ---------------------------------------------------------------------------------------------
struct DcfKeyBatch {
    std::vector<u64> seed;
    std::vector<u64> cw_seeds;
    std::vector<u64> cw_values;
    std::vector<u64> cw_tL;
    std::vector<u64> cw_tR;
    std::vector<u64> cw_final;
    uint8_t n_bits = 0;
    uint8_t party = 0;
    uint8_t P = 0;
    u64 modulus = 0;   // 0 = Z_{2^64}

    std::size_t M() const { return seed.size() / 2; }
    Block128 seed_block(std::size_t m) const { return Block128{seed[2 * m], seed[2 * m + 1]}; }

    // Wire size charged by the cost model: M * key_bytes_per_element(n_bits, P).
    std::size_t nbytes() const;
    // Exact size of to_bytes(layout).
    std::size_t serialized_bytes(WireLayout layout = WireLayout::Binding) const;

    Status validate() const;
    Expected<DcfKeyBatch> slice(std::size_t lo, std::size_t hi) const;
    std::vector<uint8_t> to_bytes(WireLayout layout = WireLayout::Binding) const;
    static Expected<DcfKeyBatch> from_bytes(const uint8_t* data, std::size_t len);
    static Expected<DcfKeyBatch> from_bytes(const std::vector<uint8_t>& b) { return from_bytes(b.data(), b.size()); }
};

// Dealer: M key pairs for f^{<}_{alpha[m], beta[m]}. alpha has M entries (reduced mod 2^n_bits);
// beta has M*P entries row-major (beta[m*P + p]), reduced mod `modulus` when one is given.
Expected<std::pair<DcfKeyBatch, DcfKeyBatch>> gen_dcf_batch(const std::vector<u64>& alpha, const std::vector<u64>& beta,
                                                            int n_bits, Rng& rng, u64 modulus = 0);

// Party: evaluate the M keys on M public inputs; returns M*P output shares row-major (out[m*P + p]).
Expected<std::vector<u64>> eval_dcf_batch(const DcfKeyBatch& key, const std::vector<u64>& x);

Expected<DcfKeyBatch> concat_keys(const DcfKeyBatch& a, const DcfKeyBatch& b);

// Plaintext reconstruction of two output share vectors (test helper).
Expected<std::vector<u64>> reconstruct_dcf(const std::vector<u64>& out0, const std::vector<u64>& out1, u64 modulus);

}  // namespace dcf
}  // namespace lowlevel
}  // namespace opencrab2
