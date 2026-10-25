// Z_{2^64} ring arithmetic and fixed-point encoding (paper, Section 2.1), header-only.
//
// All ring elements are uint64_t interpreted as two's complement integers modulo 2^64. Unsigned
// arithmetic wraps by definition in C++, so add/sub/mul never invoke undefined behaviour.
// Generic small rings Z_{2^k} (k <= 64) are supported by reduce_k / to_signed_k so that the
// faithful truncation and ring-to-field gates run one code path for the exhaustive small-k tests
// and for the native k = 64 case. encode() refuses values outside the logical range with
// CONTRACT_VIOLATION; there is no silent saturation anywhere.
//
// Tested (tests/test_lowlevel.cpp): wrap-around, signed views, msb, arithmetic shift against a
// floor-division reference, encode/decode round trips and the saturation refusal.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "opencrab2/expected.hpp"

namespace opencrab2 {
namespace lowlevel {
namespace ring {

using u64 = uint64_t;
using i64 = int64_t;

constexpr u64 MASK64 = 0xFFFFFFFFFFFFFFFFULL;
constexpr u64 SIGN_BIT = 0x8000000000000000ULL;

inline u64 add(u64 a, u64 b) { return a + b; }
inline u64 sub(u64 a, u64 b) { return a - b; }
inline u64 mul(u64 a, u64 b) { return a * b; }
inline u64 neg(u64 a) { return u64(0) - a; }

// Two's complement view of v as a signed 64-bit integer (no implementation-defined casts).
inline i64 to_signed(u64 v) {
    if (v & SIGN_BIT) return -static_cast<i64>(~v) - 1;
    return static_cast<i64>(v);
}

// Modular conversion of a signed value to its residue (well defined for every int64).
inline u64 from_signed(i64 v) { return static_cast<u64>(v); }

inline u64 msb(u64 v) { return v >> 63; }

// floor(signed(v) / 2^bits) as a ring element; total: bits <= 0 returns v, bits >= 64 returns the
// sign fill. Plaintext helper (the SecureML local truncation of trunc.hpp is built on it).
inline u64 arith_shift_right(u64 v, int bits) {
    if (bits <= 0) return v;
    if (bits >= 64) return (v & SIGN_BIT) ? MASK64 : 0;
    if (v & SIGN_BIT) return ~((~v) >> bits);
    return v >> bits;
}

// ---------------------------------------------------------------------------------------------
// Generic Z_{2^k}, k in [1, 64]
// ---------------------------------------------------------------------------------------------
inline Status check_k(int k) {
    if (k < 1 || k > 64) return fail_status("RING_BITS: k must be in [1, 64]");
    return ok_status();
}

inline u64 mask_k(int k) { return k >= 64 ? MASK64 : ((u64(1) << k) - 1); }

inline u64 reduce_k(u64 v, int k) { return k >= 64 ? v : (v & mask_k(k)); }

// Representative of v mod 2^k in [-2^(k-1), 2^(k-1)).
inline i64 to_signed_k(u64 v, int k) {
    if (k >= 64) return to_signed(v);
    const u64 r = reduce_k(v, k);
    const u64 half = u64(1) << (k - 1);
    if (r >= half) return static_cast<i64>(r) - static_cast<i64>(u64(1) << k);
    return static_cast<i64>(r);
}

inline u64 from_signed_k(i64 v, int k) { return reduce_k(from_signed(v), k); }

// ---------------------------------------------------------------------------------------------
// Fixed point (contract fields f = frac_bits, l = logical_bits; the full NumericContract lives in
// the highlevel layer and passes these two numbers down)
// ---------------------------------------------------------------------------------------------
struct FixedPoint {
    int frac_bits = 16;
    int logical_bits = 40;
};

inline Status check_fixed_point(int f, int l) {
    if (f < 0 || f > 63) return fail_status("CONTRACT_PARAMS: frac_bits must be in [0, 63]");
    if (l < 1 || l > 64) return fail_status("CONTRACT_PARAMS: logical_bits must be in [1, 64]");
    return ok_status();
}

// rint(x * 2^f) mod 2^64; CONTRACT_VIOLATION when the value is not finite or |rint(x*2^f)| >= 2^(l-1).
inline Expected<u64> encode(double x, int f, int l) {
    using R = Expected<u64>;
    const Status st = check_fixed_point(f, l);
    if (!st.ok) return R::failure(st.code);
    if (!std::isfinite(x)) return R::failure("CONTRACT_VIOLATION: encode received a non-finite value");
    const double scaled = std::nearbyint(x * std::ldexp(1.0, f));   // round half to even, like numpy rint
    const double bound = std::ldexp(1.0, l - 1);
    if (std::fabs(scaled) >= bound) return R::failure("CONTRACT_VIOLATION: |rint(x*2^f)| >= 2^(l-1)");
    return R::success(from_signed(static_cast<i64>(scaled)));
}

inline Expected<u64> encode(double x, const FixedPoint& c) { return encode(x, c.frac_bits, c.logical_bits); }

inline double decode(u64 v, int f) { return static_cast<double>(to_signed(v)) * std::ldexp(1.0, -f); }

inline double decode(u64 v, const FixedPoint& c) { return decode(v, c.frac_bits); }

inline Expected<std::vector<u64>> encode(const std::vector<double>& xs, int f, int l) {
    using R = Expected<std::vector<u64>>;
    std::vector<u64> out(xs.size());
    for (std::size_t i = 0; i < xs.size(); ++i) {
        const Expected<u64> e = encode(xs[i], f, l);
        if (!e.ok) return R::failure(e.code);
        out[i] = e.value;
    }
    return R::success(std::move(out));
}

inline std::vector<double> decode(const std::vector<u64>& vs, int f) {
    std::vector<double> out(vs.size());
    for (std::size_t i = 0; i < vs.size(); ++i) out[i] = decode(vs[i], f);
    return out;
}

// ---------------------------------------------------------------------------------------------
// Elementwise vector helpers (sizes must agree; mismatch fails closed)
// ---------------------------------------------------------------------------------------------
namespace detail {
template <class Op>
Expected<std::vector<u64>> zip(const std::vector<u64>& a, const std::vector<u64>& b, Op op) {
    using R = Expected<std::vector<u64>>;
    if (a.size() != b.size()) return R::failure("RING_SHAPE: operand sizes differ");
    std::vector<u64> out(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) out[i] = op(a[i], b[i]);
    return R::success(std::move(out));
}
}  // namespace detail

inline Expected<std::vector<u64>> add(const std::vector<u64>& a, const std::vector<u64>& b) {
    return detail::zip(a, b, [](u64 x, u64 y) { return x + y; });
}
inline Expected<std::vector<u64>> sub(const std::vector<u64>& a, const std::vector<u64>& b) {
    return detail::zip(a, b, [](u64 x, u64 y) { return x - y; });
}
inline Expected<std::vector<u64>> mul(const std::vector<u64>& a, const std::vector<u64>& b) {
    return detail::zip(a, b, [](u64 x, u64 y) { return x * y; });
}
inline std::vector<u64> neg(const std::vector<u64>& a) {
    std::vector<u64> out(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) out[i] = neg(a[i]);
    return out;
}
inline std::vector<u64> add_scalar(const std::vector<u64>& a, u64 c) {
    std::vector<u64> out(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) out[i] = a[i] + c;
    return out;
}
inline std::vector<u64> reduce_k(const std::vector<u64>& a, int k) {
    std::vector<u64> out(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) out[i] = reduce_k(a[i], k);
    return out;
}

}  // namespace ring
}  // namespace lowlevel
}  // namespace opencrab2
