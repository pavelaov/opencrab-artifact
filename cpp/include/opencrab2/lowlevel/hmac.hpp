// HMAC (RFC 2104) over the hash classes of this directory, header-only.
//
// hmac_sha256 authenticates dealer bundles (paper, Appendix B, DealerDelivery) and telemetry
// frames (paper, Section 4.2). These are integrity tools for a semi-honest deployment, not
// security proofs.
// hmac_sha1 exists for completeness and tests only. constant_time_equal compares tags without
// data-dependent early exit.
//
// Tested (tests/test_lowlevel.cpp): RFC 4231 cases 1, 2, 3, 4 (HMAC-SHA-256) and RFC 2202
// cases 1, 2, 3 (HMAC-SHA-1).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "opencrab2/lowlevel/sha1.hpp"
#include "opencrab2/lowlevel/sha256.hpp"

namespace opencrab2 {
namespace lowlevel {

template <class Hash>
std::array<uint8_t, Hash::digest_size> hmac(const uint8_t* key, std::size_t key_len,
                                            const uint8_t* msg, std::size_t msg_len) {
    constexpr std::size_t B = Hash::block_size;
    uint8_t k[B];
    for (std::size_t i = 0; i < B; ++i) k[i] = 0;
    if (key_len > B) {
        const auto d = Hash::digest(key, key_len);
        for (std::size_t i = 0; i < d.size(); ++i) k[i] = d[i];
    } else {
        for (std::size_t i = 0; i < key_len; ++i) k[i] = key[i];
    }
    uint8_t ipad[B];
    uint8_t opad[B];
    for (std::size_t i = 0; i < B; ++i) {
        ipad[i] = static_cast<uint8_t>(k[i] ^ 0x36);
        opad[i] = static_cast<uint8_t>(k[i] ^ 0x5c);
    }
    Hash inner;
    inner.update(ipad, B);
    inner.update(msg, msg_len);
    const auto inner_digest = inner.finish();
    Hash outer;
    outer.update(opad, B);
    outer.update(inner_digest.data(), inner_digest.size());
    return outer.finish();
}

inline std::array<uint8_t, 32> hmac_sha256(const uint8_t* key, std::size_t key_len, const uint8_t* msg,
                                           std::size_t msg_len) {
    return hmac<Sha256>(key, key_len, msg, msg_len);
}

inline std::array<uint8_t, 32> hmac_sha256(const std::vector<uint8_t>& key, const std::string& msg) {
    return hmac<Sha256>(key.data(), key.size(), reinterpret_cast<const uint8_t*>(msg.data()), msg.size());
}

inline std::array<uint8_t, 20> hmac_sha1(const uint8_t* key, std::size_t key_len, const uint8_t* msg,
                                         std::size_t msg_len) {
    return hmac<Sha1>(key, key_len, msg, msg_len);
}

// Constant-time comparison of two byte strings of equal length n.
inline bool constant_time_equal(const uint8_t* a, const uint8_t* b, std::size_t n) {
    unsigned diff = 0;
    for (std::size_t i = 0; i < n; ++i) diff |= static_cast<unsigned>(a[i] ^ b[i]);
    return diff == 0;
}

template <std::size_t N>
bool constant_time_equal(const std::array<uint8_t, N>& a, const std::array<uint8_t, N>& b) {
    return constant_time_equal(a.data(), b.data(), N);
}

}  // namespace lowlevel
}  // namespace opencrab2
