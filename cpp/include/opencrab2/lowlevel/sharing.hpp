// Additive and XOR secret sharing (paper, Section 2.2), header-only.
//
// share / reconstruct over Z_{2^64}; share_k / reconstruct_k over Z_{2^k} (small-ring tests);
// share_mod / reconstruct_mod over Z_t for an odd modulus 3 <= t < 2^62 (values held as uint64
// below t); share_bits / reconstruct_bits are XOR shares of bits held as uint8. Randomness comes
// from the caller's Rng (csprng.hpp); the dealer seeds it from the OS.
//
// Tested (tests/test_lowlevel.cpp): reconstruction identities for all four variants and the
// range refusal of share_mod.
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "opencrab2/expected.hpp"
#include "opencrab2/lowlevel/csprng.hpp"
#include "opencrab2/lowlevel/ring.hpp"

namespace opencrab2 {
namespace lowlevel {
namespace sharing {

using ring::u64;

constexpr u64 MAX_MODULUS = u64(1) << 62;

inline std::pair<u64, u64> share(u64 x, Rng& rng) {
    const u64 s0 = rng.next_u64();
    return {s0, x - s0};
}

inline u64 reconstruct(u64 s0, u64 s1) { return s0 + s1; }

inline std::pair<std::vector<u64>, std::vector<u64>> share(const std::vector<u64>& x, Rng& rng) {
    std::vector<u64> s0(x.size()), s1(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        s0[i] = rng.next_u64();
        s1[i] = x[i] - s0[i];
    }
    return {std::move(s0), std::move(s1)};
}

inline Expected<std::vector<u64>> reconstruct(const std::vector<u64>& s0, const std::vector<u64>& s1) {
    return ring::add(s0, s1);
}

// Z_{2^k}: both shares held below 2^k.
inline std::pair<u64, u64> share_k(u64 x, int k, Rng& rng) {
    const u64 xr = ring::reduce_k(x, k);
    const u64 s0 = ring::reduce_k(rng.next_u64(), k);
    return {s0, ring::reduce_k(xr - s0, k)};
}

inline u64 reconstruct_k(u64 s0, u64 s1, int k) { return ring::reduce_k(s0 + s1, k); }

inline std::pair<std::vector<u64>, std::vector<u64>> share_k(const std::vector<u64>& x, int k, Rng& rng) {
    std::vector<u64> s0(x.size()), s1(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const std::pair<u64, u64> p = share_k(x[i], k, rng);
        s0[i] = p.first;
        s1[i] = p.second;
    }
    return {std::move(s0), std::move(s1)};
}

// Deterministic Miller-Rabin for every n < 2^64: the twelve prime bases 2..37 leave no composite
// below 3.3 * 10^24 undetected. Used by the Z_t payload group of the DCF (DCF_MODULUS) and by
// ring_to_field (R2F_MODULUS): both protocols admit odd PRIMES only, and the
// Python side refuses composites, so 9, 561 or 2^62 - 1 must not be accepted here either.
inline u64 mulmod_u64(u64 a, u64 b, u64 n) {
#if defined(__SIZEOF_INT128__)
    return static_cast<u64>((static_cast<unsigned __int128>(a) * b) % n);
#else
    u64 r = 0;
    a %= n;
    while (b) {
        if (b & 1u) r = (r >= n - a) ? r - (n - a) : r + a;
        a = (a >= n - a) ? a - (n - a) : a + a;
        b >>= 1;
    }
    return r;
#endif
}

inline bool is_prime_u64(u64 n) {
    static const u64 bases[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    if (n < 2) return false;
    for (u64 p : bases) {
        if (n % p == 0) return n == p;
    }
    u64 d = n - 1;
    int s = 0;
    while ((d & 1u) == 0) {
        d >>= 1;
        ++s;
    }
    for (u64 a : bases) {
        u64 x = 1, base = a % n, e = d;
        while (e) {
            if (e & 1u) x = mulmod_u64(x, base, n);
            base = mulmod_u64(base, base, n);
            e >>= 1;
        }
        if (x == 1 || x == n - 1) continue;
        bool composite = true;
        for (int r = 1; r < s; ++r) {
            x = mulmod_u64(x, x, n);
            if (x == n - 1) {
                composite = false;
                break;
            }
        }
        if (composite) return false;
    }
    return true;
}

// Odd modulus with 3 <= t < 2^62 for plain additive sharing over Z_t (any odd t is a valid group;
// the Python sharing.check_modulus has the same rule). The protocols that are restricted to odd
// primes check primality themselves: dcf::check_modulus and check_r2f_params.
inline Status check_modulus(u64 t) {
    if (t < 3 || t >= MAX_MODULUS || (t & 1u) == 0) return fail_status("SHARING_MODULUS: t must be odd with 3 <= t < 2^62");
    return ok_status();
}

inline Expected<std::pair<u64, u64>> share_mod(u64 x, u64 t, Rng& rng) {
    using R = Expected<std::pair<u64, u64>>;
    const Status st = check_modulus(t);
    if (!st.ok) return R::failure(st.code);
    if (x >= t) return R::failure("SHARING_RANGE: share_mod input must be below t");
    const u64 s0 = rng.below(t);
    const u64 s1 = (x + t - s0) % t;   // x + t < 2^63: no wrap
    return R::success(std::make_pair(s0, s1));
}

inline Expected<std::pair<std::vector<u64>, std::vector<u64>>> share_mod(const std::vector<u64>& x, u64 t, Rng& rng) {
    using R = Expected<std::pair<std::vector<u64>, std::vector<u64>>>;
    const Status st = check_modulus(t);
    if (!st.ok) return R::failure(st.code);
    std::vector<u64> s0(x.size()), s1(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (x[i] >= t) return R::failure("SHARING_RANGE: share_mod input must be below t");
        s0[i] = rng.below(t);
        s1[i] = (x[i] + t - s0[i]) % t;
    }
    return R::success(std::make_pair(std::move(s0), std::move(s1)));
}

inline u64 reconstruct_mod(u64 s0, u64 s1, u64 t) { return ((s0 % t) + (s1 % t)) % t; }

inline Expected<std::vector<u64>> reconstruct_mod(const std::vector<u64>& s0, const std::vector<u64>& s1, u64 t) {
    using R = Expected<std::vector<u64>>;
    const Status st = check_modulus(t);
    if (!st.ok) return R::failure(st.code);
    if (s0.size() != s1.size()) return R::failure("SHARING_SHAPE: share vectors differ in size");
    std::vector<u64> out(s0.size());
    for (std::size_t i = 0; i < s0.size(); ++i) out[i] = reconstruct_mod(s0[i], s1[i], t);
    return R::success(std::move(out));
}

inline std::pair<std::vector<uint8_t>, std::vector<uint8_t>> share_bits(const std::vector<uint8_t>& b, Rng& rng) {
    std::vector<uint8_t> s0(b.size()), s1(b.size());
    for (std::size_t i = 0; i < b.size(); ++i) {
        s0[i] = static_cast<uint8_t>(rng.next_u64() & 1u);
        s1[i] = static_cast<uint8_t>((b[i] & 1u) ^ s0[i]);
    }
    return {std::move(s0), std::move(s1)};
}

inline Expected<std::vector<uint8_t>> reconstruct_bits(const std::vector<uint8_t>& s0, const std::vector<uint8_t>& s1) {
    using R = Expected<std::vector<uint8_t>>;
    if (s0.size() != s1.size()) return R::failure("SHARING_SHAPE: share vectors differ in size");
    std::vector<uint8_t> out(s0.size());
    for (std::size_t i = 0; i < s0.size(); ++i) out[i] = static_cast<uint8_t>((s0[i] ^ s1[i]) & 1u);
    return R::success(std::move(out));
}

}  // namespace sharing
}  // namespace lowlevel
}  // namespace opencrab2
