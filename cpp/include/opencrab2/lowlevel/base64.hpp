// Textual encodings, header-only: base64 (RFC 4648 Sec. 4, with padding) and hex.
//
// base64 is needed by the WebSocket handshake (RFC 6455) and by JSON-carried binary fields of
// dealer bundles; hex is used for digests, ids and test vectors. Decoding fails closed on any
// character outside the alphabet, on bad padding and on a length that is not a multiple of 4.
//
// Tested (tests/test_lowlevel.cpp): the RFC 4648 Sec. 10 vectors ("", "f", "fo", "foo", "foob",
// "fooba", "foobar"), round trips on random data, and rejection of malformed input.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "opencrab2/expected.hpp"

namespace opencrab2 {
namespace lowlevel {

inline std::string base64_encode(const uint8_t* data, std::size_t len) {
    static const char* const tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    std::size_t i = 0;
    while (i + 3 <= len) {
        const uint32_t v = (static_cast<uint32_t>(data[i]) << 16) | (static_cast<uint32_t>(data[i + 1]) << 8) |
                           static_cast<uint32_t>(data[i + 2]);
        out.push_back(tbl[(v >> 18) & 63]);
        out.push_back(tbl[(v >> 12) & 63]);
        out.push_back(tbl[(v >> 6) & 63]);
        out.push_back(tbl[v & 63]);
        i += 3;
    }
    if (len - i == 1) {
        const uint32_t v = static_cast<uint32_t>(data[i]) << 16;
        out.push_back(tbl[(v >> 18) & 63]);
        out.push_back(tbl[(v >> 12) & 63]);
        out.push_back('=');
        out.push_back('=');
    } else if (len - i == 2) {
        const uint32_t v = (static_cast<uint32_t>(data[i]) << 16) | (static_cast<uint32_t>(data[i + 1]) << 8);
        out.push_back(tbl[(v >> 18) & 63]);
        out.push_back(tbl[(v >> 12) & 63]);
        out.push_back(tbl[(v >> 6) & 63]);
        out.push_back('=');
    }
    return out;
}

inline std::string base64_encode(const std::vector<uint8_t>& data) {
    return base64_encode(data.data(), data.size());
}

inline std::string base64_encode(const std::string& s) {
    return base64_encode(reinterpret_cast<const uint8_t*>(s.data()), s.size());
}

inline int base64_value(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

// Strict decoder: length must be a multiple of 4, padding only at the end ("=" or "==").
inline Expected<std::vector<uint8_t>> base64_decode(const std::string& s) {
    using R = Expected<std::vector<uint8_t>>;
    if (s.size() % 4 != 0) return R::failure("BASE64_LENGTH");
    std::vector<uint8_t> out;
    out.reserve(s.size() / 4 * 3);
    for (std::size_t i = 0; i < s.size(); i += 4) {
        const bool last = (i + 4 == s.size());
        int pad = 0;
        if (last) {
            if (s[i + 3] == '=') pad = 1;
            if (s[i + 2] == '=') {
                if (s[i + 3] != '=') return R::failure("BASE64_PADDING");
                pad = 2;
            }
        }
        int v[4];
        for (int j = 0; j < 4; ++j) {
            if (j >= 4 - pad) {
                v[j] = 0;
                continue;
            }
            v[j] = base64_value(s[i + j]);
            if (v[j] < 0) return R::failure("BASE64_ALPHABET");
        }
        const uint32_t w = (static_cast<uint32_t>(v[0]) << 18) | (static_cast<uint32_t>(v[1]) << 12) |
                           (static_cast<uint32_t>(v[2]) << 6) | static_cast<uint32_t>(v[3]);
        out.push_back(static_cast<uint8_t>(w >> 16));
        if (pad < 2) out.push_back(static_cast<uint8_t>(w >> 8));
        if (pad < 1) out.push_back(static_cast<uint8_t>(w));
    }
    return R::success(std::move(out));
}

inline std::string to_hex(const uint8_t* data, std::size_t len) {
    static const char* const digits = "0123456789abcdef";
    std::string out;
    out.reserve(len * 2);
    for (std::size_t i = 0; i < len; ++i) {
        out.push_back(digits[data[i] >> 4]);
        out.push_back(digits[data[i] & 15]);
    }
    return out;
}

template <std::size_t N>
std::string to_hex(const std::array<uint8_t, N>& a) {
    return to_hex(a.data(), N);
}

inline std::string to_hex(const std::vector<uint8_t>& v) { return to_hex(v.data(), v.size()); }

inline int hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

inline Expected<std::vector<uint8_t>> from_hex(const std::string& s) {
    using R = Expected<std::vector<uint8_t>>;
    if (s.size() % 2 != 0) return R::failure("HEX_LENGTH");
    std::vector<uint8_t> out(s.size() / 2);
    for (std::size_t i = 0; i < out.size(); ++i) {
        const int hi = hex_value(s[2 * i]);
        const int lo = hex_value(s[2 * i + 1]);
        if (hi < 0 || lo < 0) return R::failure("HEX_ALPHABET");
        out[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return R::success(std::move(out));
}

}  // namespace lowlevel
}  // namespace opencrab2
