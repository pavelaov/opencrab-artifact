// Faithful truncation gate with the signed offset (TruncFaithful, paper, Section 5.1).
//
// Parameters: ring q = 2^k (k = 64 natively, k <= 64 for the exhaustive small-ring tests),
// logical width L, shift f with 1 <= f <= L-1 <= k-1, input x in [-2^(L-1), 2^(L-1)) (signed),
// a = 2^(L-1).
//
// Dealer per element: uniform r in Z_q; shares [r]; shares [floor(r / 2^f)]; DCF key A (wrap):
// alpha = r, n_bits = k, payload 1; DCF key B (borrow): alpha = r mod 2^f, n_bits = f, payload 1.
//
// Online (one round):
//     1. P0 adds the public offset: x' = x + a (P0 only). Both: open y = x' + [r] mod q.
//     2. Locally: [w] = DCF_A(y) (= 1{y < r}), [b] = DCF_B(y mod 2^f) (= 1{y mod 2^f < r mod 2^f}).
//     3. [z] = floor(y / 2^f) 1{party 0} - [floor(r / 2^f)] - [b] + 2^(k-f) [w] - 2^(L-1-f) 1{party 0}  (mod q)
//     4. [z] are shares of floor(x / 2^f) with signed semantics.
//
// Correctness: with z0 = x + a in [0, 2^L), y = z0 + r - w q and the low-f-bit borrow b,
// floor(z0 / 2^f) = floor(y / 2^f) - floor(r / 2^f) - b + w 2^(k-f); a is divisible by 2^f so
// subtracting a / 2^f = 2^(L-1-f) gives the signed floor. A formula without the last term is
// wrong: the regression (k=8, L=6, f=3, x=-13, r=37) returns -2 here and 2 without it.
// Privacy: the only message is y = x' + r with uniform r, independent of x; w
// and b stay shared. Keys are one-time objects (ledger enforced).
//
// trunc_local (SecureML local probabilistic truncation) is kept for contract-bounded products:
// for |x| < 2^l_x it is off by at most one unit except with probability about 2^(l_x + 1 - 64),
// when it is wrong by a full 2^(64-f) wrap (design limit; the registry row trunc_local of the
// Python implementation states the resulting per-session rate).
//
// Serialization (same bytes as the Python implementation, header "OC2TRN01"):
//     magic[8] = "OC2TRN01", u8 f, u8 L, u8 k, u8 party, u32 LE flags = 0, u64 LE M,
//     u64 LE len_wrap, u64 LE len_borrow, r_share[M], r_high_share[M] (u64 LE),
//     key_wrap bytes, key_borrow bytes (DCF wire layout).
//
// Tested (tests/test_lowlevel.cpp): the regression value, exhaustive agreement of the gate (real
// DCF keys, real shares) with floor(x / 2^f) and with reference_trunc_small for every
// (k <= 10, L, f, x, r) (165 configurations, about 22 million gate evaluations, run on a thread
// pool), 10^6 random 64-bit elements across magnitudes 2^20 .. 2^39, boundary values, sign
// changes, serialization round trip, byte formula (2064 bytes at f = 16), trunc_local on bounded
// inputs.
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
namespace trunc {

using u64 = uint64_t;
using i64 = int64_t;

constexpr std::size_t WIRE_HEADER_BYTES = 40;

// Validate 1 <= f <= L-1 <= k-1 with k in [2, 64].
Status check_trunc_params(int f, int L, int k = 64);

struct TruncMaterial {
    std::vector<u64> r_share;        // share of the mask r over Z_q
    std::vector<u64> r_high_share;   // share of floor(r / 2^f) over Z_q
    dcf::DcfKeyBatch key_wrap;       // alpha = r, n_bits = k, payload 1
    dcf::DcfKeyBatch key_borrow;     // alpha = r mod 2^f, n_bits = f, payload 1
    uint8_t f = 0;
    uint8_t L = 0;
    uint8_t party = 0;

    std::size_t M() const { return r_share.size(); }
    int k() const { return key_wrap.n_bits; }   // ring width, carried by the wrap key's domain
    std::size_t nbytes() const;
    Status validate() const;
    Expected<TruncMaterial> slice(std::size_t lo, std::size_t hi) const;
    std::vector<uint8_t> to_bytes(dcf::WireLayout key_layout = dcf::WireLayout::Binding) const;
    static Expected<TruncMaterial> from_bytes(const uint8_t* data, std::size_t len);
    static Expected<TruncMaterial> from_bytes(const std::vector<uint8_t>& b) { return from_bytes(b.data(), b.size()); }
};

// Native (k = 64) material per element: key(64,1) + key(f,1) + 16 (2064 bytes at f = 16).
std::size_t trunc_bytes_per_element(int f);

// Dealer: material for the given masks r over Z_{2^k} (tests pass chosen masks).
Expected<std::pair<TruncMaterial, TruncMaterial>> trunc_material_from_mask(const std::vector<u64>& r, int f, int L, Rng& rng,
                                                                           int k = 64);
// Dealer: M elements over Z_{2^k} with uniform masks.
Expected<std::pair<TruncMaterial, TruncMaterial>> trunc_material_k(std::size_t M, int f, int L, Rng& rng, int k);
// Dealer: M native (k = 64) elements.
Expected<std::pair<TruncMaterial, TruncMaterial>> trunc_material(std::size_t M, int f, int L, Rng& rng);

// Step 1: P0 adds the public offset a = 2^(L-1); P1 returns its share unchanged (mod 2^k).
Expected<std::vector<u64>> trunc_offset(const std::vector<u64>& x_share, int party, int L, int k = 64);
// This party's message for the single open: x' + [r] mod 2^k (x' includes P0's offset).
Expected<std::vector<u64>> trunc_open_share(const std::vector<u64>& x_share, const TruncMaterial& mat, int party);
// Steps 2 and 3 on the public y: this party's share of floor(x / 2^f) (mod 2^k), including the
// -2^(L-1-f) term on P0.
Expected<std::vector<u64>> trunc_faithful_eval(const TruncMaterial& mat, const std::vector<u64>& y, int party);

// SecureML local probabilistic truncation by f bits (k = 64 only, zero rounds).
Expected<u64> trunc_local(u64 share, int f, int party);
Expected<std::vector<u64>> trunc_local(const std::vector<u64>& share, int f, int party);

// Pure-integer evaluation of the gate formula (the DCF outputs replaced by the comparisons they
// encode); returns the signed result in [-2^(k-1), 2^(k-1)). Independent oracle for the tests.
Expected<i64> reference_trunc_small(int k, int L, int f, i64 x, u64 r);
// The incorrect formula without the -2^(L-1-f) term, kept for the regression test.
Expected<i64> reference_trunc_small_no_offset_term(int k, int L, int f, i64 x, u64 r);

Expected<TruncMaterial> concat_trunc(const TruncMaterial& a, const TruncMaterial& b);

}  // namespace trunc
}  // namespace lowlevel
}  // namespace opencrab2
