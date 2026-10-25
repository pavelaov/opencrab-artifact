// One-round DReLU / ReLU from two DCF keys (paper, Section 5.3, one-round DReLU).
//
// With a public masked value y = x + r (mod 2^64), an n-bit comparison ring, p = r mod 2^n and
// q = (r + 2^(n-1)) mod 2^n, for |x| < 2^(n-1):
//
//     1{x >= 0} = DCF_q(y_n) - DCF_p(y_n) + 1{p > q},      y_n = y mod 2^n.
//
// 1{x >= 0} = 1{(y_n - p) mod 2^n < 2^(n-1)} = 1{y_n in the cyclic interval [p, q)}, and the
// indicator of a cyclic interval is 1{y < q} - 1{y < p} + 1{p > q}. With payload (1, r) the same
// two evaluations give shares of r * 1{x >= 0}, hence ReLU(x) = y*[b] - [r b] in one round. The
// dealer-known constants 1{p > q} and 1{p > q} r are shipped as additive shares (revealing
// 1{p > q} would leak one bit of r).
//
// The comparison-key finding (paper, Section 5.3 and Appendix D, Disclosure). The payload (1, r)
// is a two-component payload whose second component is the mask itself. With one PRG value word
// shared by both components (the pre-repair rule of the superseded engine) a party read r, hence
// x = y - r, from the public correction words of its own key. The DCF now derives the words of
// every component from its own PRG blocks (dcf.hpp); tests/test_lowlevel.cpp
// (KEY_COMPONENT_INDEPENDENCE) reproduces the attack
// on a replica of the old key generation (2000 of 2000 masks) and pins 0 of 2000 on this material.
// Material generated before the repair must not be used. Design limit: the wire format did not
// change (no version bump was agreed across the two languages), so this code cannot recognize
// such material; it parses and then evaluates to wrong shares. Dealer bundles are bound to a
// setup id, so stock from an earlier setup is rejected at ingest, not here.
//
// Material per element: two DCF keys (n bits, P in {1, 2}) plus three ring shares.
// Serialization (same bytes as the Python implementation, header "OC2DRL01"):
//     magic[8] = "OC2DRL01", u8 n_bits, u8 party, u16 LE with_product, u32 LE flags = 0,
//     u64 LE M, u64 LE len_key_p, u64 LE len_key_q,
//     r_share[M], c_b[M], c_rb[M] (u64 LE), key_p bytes, key_q bytes (DCF wire layout).
//
// Tested (tests/test_lowlevel.cpp): random x at n in {16, 32, 64} with and without the product
// payload, sign and ReLU reconstruction, serialization round trip, byte formula. Design limit:
// out-of-domain inputs (|x| >= 2^(n-1)) are not detected; the numeric contract bounds x.
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
namespace drelu {

using u64 = uint64_t;

constexpr std::size_t WIRE_HEADER_BYTES = 40;

struct DreluMaterial {
    std::vector<u64> r_share;    // share of the 64-bit mask r
    dcf::DcfKeyBatch key_p;      // alpha = r mod 2^n, payload (1[, r])
    dcf::DcfKeyBatch key_q;      // alpha = (r + 2^(n-1)) mod 2^n, payload (1[, r])
    std::vector<u64> c_b;        // share of 1{p > q}
    std::vector<u64> c_rb;       // share of 1{p > q} r (zeros without product payload)
    uint8_t n_bits = 0;
    uint8_t party = 0;

    std::size_t M() const { return r_share.size(); }
    bool with_product() const { return key_p.P >= 2; }
    std::size_t nbytes() const;
    Status validate() const;
    Expected<DreluMaterial> slice(std::size_t lo, std::size_t hi) const;
    std::vector<uint8_t> to_bytes(dcf::WireLayout key_layout = dcf::WireLayout::Binding) const;
    static Expected<DreluMaterial> from_bytes(const uint8_t* data, std::size_t len);
    static Expected<DreluMaterial> from_bytes(const std::vector<uint8_t>& b) { return from_bytes(b.data(), b.size()); }
};

struct DreluOutput {
    std::vector<u64> b;    // shares of 1{x >= 0}
    std::vector<u64> rb;   // shares of r 1{x >= 0}; empty without product payload
};

// Two DCF keys (P = 2 with product, else 1) plus three ring shares.
std::size_t drelu_bytes_per_element(int n_bits, bool with_product = true);

// Dealer: material for the given 64-bit masks r (tests pass chosen masks). n_bits in [2, 64].
Expected<std::pair<DreluMaterial, DreluMaterial>> drelu_material_from_mask(const std::vector<u64>& r, int n_bits, Rng& rng,
                                                                           bool with_product = true);
// Dealer: material for M gates with uniform masks.
Expected<std::pair<DreluMaterial, DreluMaterial>> drelu_material(std::size_t M, int n_bits, Rng& rng,
                                                                 bool with_product = true);

// This party's share of r, to be added to its share of x before the single open.
inline const std::vector<u64>& drelu_mask_share(const DreluMaterial& mat) { return mat.r_share; }

// From the public y = x + r return shares of (1{x >= 0}, r 1{x >= 0}).
Expected<DreluOutput> drelu_eval(const DreluMaterial& mat, const std::vector<u64>& y);

// ReLU(x) share = y*[b] - [r b] for the public y = x + r (mod 2^64).
Expected<std::vector<u64>> relu_from_drelu(const std::vector<u64>& y, const std::vector<u64>& b_share,
                                           const std::vector<u64>& rb_share);

Expected<DreluMaterial> concat_drelu(const DreluMaterial& a, const DreluMaterial& b);

}  // namespace drelu
}  // namespace lowlevel
}  // namespace opencrab2
