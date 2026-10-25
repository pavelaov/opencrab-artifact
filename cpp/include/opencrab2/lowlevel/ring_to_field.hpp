// A64 -> Zt resharing in one round (RingToField, paper, Section 5.2).
//
// Inputs: shares (a0, a1) of u over q = 2^k (k = 64 natively, k <= 64 for the exhaustive
// small-ring tests); odd prime t < 2^62. Output: fresh shares (z0, z1) over Z_t of
// z = x_signed mod t where x_signed = u - 1{u >= q/2} q is the representative in [-q/2, q/2).
//
// Dealer per element: uniform r in Z_q; Z_t shares of (r mod t); DCF key A: alpha = r, n = k,
// payload group Z_t, beta = 1; DCF key C: alpha = (r + 2^(k-1)) mod q, n = k, payload Z_t,
// beta = 1; Z_t shares of the constant c = 1{r > (r + 2^(k-1)) mod q} (the 1{p > q'} term of the
// DReLU identity).
//
// Online:
//     1. open y = u + [r] mod q                                    (one round)
//     2. [w]_t  = DCF_A(y)                      (= 1{y < r}, the wrap of y - r)
//     3. [nn]_t = DCF_C(y) - DCF_A(y) + [c]_t   (= 1{u is non-negative as signed}, DReLU identity with n = k)
//        [b]_t  = 1 - [nn]_t                    (MSB of u; P0 adds the 1)
//     4. [u mod t]_t = (y mod t) 1{party 0} - [r mod t]_t + (q mod t) [w]_t
//     5. [z]_t = [u mod t]_t - (q mod t) [b]_t                 (mod t)
//
// Nothing but y is communicated; w and b stay shared. The counterexample (q = 256, t = 17,
// a0 = 255, a1 = 2) gives z = 1 here and 2 for the naive per-share reduction.
//
// Serialization (header "OC2R2F01"; the Python lowlevel/ring_to_field.py writes the same bytes):
//     magic[8] = "OC2R2F01", u8 k, u8 party, u16 LE reserved = 0, u32 LE flags = 0, u64 LE M,
//     u64 LE t, u64 LE len_key_A, u64 LE len_key_C,
//     r_share[M], r_mod_t_share[M], c_share[M] (u64 LE), key_A bytes, key_C bytes (DCF wire layout).
//
// The two DCF keys carry Z_t payloads, so their value words follow the 128-bit Convert rule
// (prg.hpp): statistical distance at most t / 2^128 per word, which is what the session budget
// charges for this gate (paper, Section 6).
//
// Tested (tests/test_lowlevel.cpp): exhaustive small rings (k <= 8, t in {3,5,7,11,13,17}, all
// masks, all u, ALL share splits: brute force for k <= 5; for k in {6,7,8} the gate output is
// checked for every (u, r) and every split (a0, a1) is checked to open the same y, which decides
// every split because r2f_eval depends on the material and y only), the counterexamples above,
// random 64-bit with t = 68720050177, serialization round trip, byte formula (3272 bytes per
// element). Design limit: primality of t is the admission profile's contract and is not tested
// here.
#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "opencrab2/expected.hpp"
#include "opencrab2/lowlevel/csprng.hpp"
#include "opencrab2/lowlevel/dcf.hpp"

namespace opencrab2 {
namespace lowlevel {
namespace r2f {

using u64 = uint64_t;

constexpr std::size_t WIRE_HEADER_BYTES = 48;
constexpr u64 ADMITTED_T = 68720050177ULL;   // the plaintext modulus of the admitted BFV A2H route

Status check_r2f_params(u64 t, int k = 64);

// (a * b) mod t for a, b < t < 2^62.
u64 mulmod(u64 a, u64 b, u64 t);
// 2^k mod t for k in [1, 64].
u64 pow2_mod(int k, u64 t);

struct R2FMaterial {
    std::vector<u64> r_share;         // share of r over Z_q
    std::vector<u64> r_mod_t_share;   // Z_t share of r mod t
    dcf::DcfKeyBatch key_A;           // alpha = r, n = k, Z_t payload 1
    dcf::DcfKeyBatch key_C;           // alpha = (r + 2^(k-1)) mod q, n = k, Z_t payload 1
    std::vector<u64> c_share;         // Z_t share of 1{r > (r + 2^(k-1)) mod q}
    u64 t = 0;
    uint8_t party = 0;

    std::size_t M() const { return r_share.size(); }
    int k() const { return key_A.n_bits; }
    std::size_t nbytes() const;
    Status validate() const;
    Expected<R2FMaterial> slice(std::size_t lo, std::size_t hi) const;
    std::vector<uint8_t> to_bytes(dcf::WireLayout key_layout = dcf::WireLayout::Binding) const;
    static Expected<R2FMaterial> from_bytes(const uint8_t* data, std::size_t len);
    static Expected<R2FMaterial> from_bytes(const std::vector<uint8_t>& b) { return from_bytes(b.data(), b.size()); }
};

// Native (k = 64) material per element: 3 shares + 2 keys(64, 1) = 24 + 2 * 1624 = 3272 (independent of t).
std::size_t r2f_bytes_per_element(u64 t);

Expected<std::pair<R2FMaterial, R2FMaterial>> r2f_material_from_mask(const std::vector<u64>& r, u64 t, Rng& rng, int k = 64);
Expected<std::pair<R2FMaterial, R2FMaterial>> r2f_material_k(std::size_t M, u64 t, Rng& rng, int k);
Expected<std::pair<R2FMaterial, R2FMaterial>> r2f_material(std::size_t M, u64 t, Rng& rng);

// This party's message for the single open: [u] + [r] mod 2^k.
Expected<std::vector<u64>> r2f_open_share(const std::vector<u64>& u_share, const R2FMaterial& mat, int party);
// Steps 2 to 5 on the public y: this party's Z_t share of x_signed mod t.
Expected<std::vector<u64>> r2f_eval(const R2FMaterial& mat, const std::vector<u64>& y, int party);

// Plaintext reference: (u mod 2^k interpreted in [-2^(k-1), 2^(k-1))) mod t.
u64 signed_mod(u64 u, u64 t, int k = 64);
std::vector<u64> signed_mod(const std::vector<u64>& u, u64 t, int k = 64);

Expected<R2FMaterial> concat_r2f(const R2FMaterial& a, const R2FMaterial& b);

}  // namespace r2f
}  // namespace lowlevel
}  // namespace opencrab2
