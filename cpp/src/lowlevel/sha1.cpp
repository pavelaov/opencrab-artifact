// SHA-1, FIPS 180-4. See include/opencrab2/lowlevel/sha1.hpp.
#include "opencrab2/lowlevel/sha1.hpp"

#include <cstring>

namespace opencrab2 {
namespace lowlevel {

namespace {

inline uint32_t rotl(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }
inline uint32_t load_be32(const uint8_t* p) {
    return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}
inline void store_be32(uint8_t* p, uint32_t v) {
    p[0] = static_cast<uint8_t>(v >> 24);
    p[1] = static_cast<uint8_t>(v >> 16);
    p[2] = static_cast<uint8_t>(v >> 8);
    p[3] = static_cast<uint8_t>(v);
}

}  // namespace

Sha1::Sha1() { reset(); }

void Sha1::reset() {
    h_[0] = 0x67452301;
    h_[1] = 0xEFCDAB89;
    h_[2] = 0x98BADCFE;
    h_[3] = 0x10325476;
    h_[4] = 0xC3D2E1F0;
    buf_len_ = 0;
    total_len_ = 0;
}

void Sha1::compress(const uint8_t block[64]) {
    uint32_t w[80];
    for (int i = 0; i < 16; ++i) w[i] = load_be32(block + 4 * i);
    for (int i = 16; i < 80; ++i) w[i] = rotl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4];
    for (int i = 0; i < 80; ++i) {
        uint32_t f, k;
        if (i < 20) {
            f = (b & c) | (~b & d);
            k = 0x5A827999;
        } else if (i < 40) {
            f = b ^ c ^ d;
            k = 0x6ED9EBA1;
        } else if (i < 60) {
            f = (b & c) | (b & d) | (c & d);
            k = 0x8F1BBCDC;
        } else {
            f = b ^ c ^ d;
            k = 0xCA62C1D6;
        }
        const uint32_t temp = rotl(a, 5) + f + e + k + w[i];
        e = d;
        d = c;
        c = rotl(b, 30);
        b = a;
        a = temp;
    }
    h_[0] += a;
    h_[1] += b;
    h_[2] += c;
    h_[3] += d;
    h_[4] += e;
}

void Sha1::update(const uint8_t* data, std::size_t len) {
    total_len_ += len;
    if (buf_len_ > 0) {
        const std::size_t take = (64 - buf_len_ < len) ? (64 - buf_len_) : len;
        std::memcpy(buf_ + buf_len_, data, take);
        buf_len_ += take;
        data += take;
        len -= take;
        if (buf_len_ == 64) {
            compress(buf_);
            buf_len_ = 0;
        }
    }
    while (len >= 64) {
        compress(data);
        data += 64;
        len -= 64;
    }
    if (len > 0) {
        std::memcpy(buf_, data, len);
        buf_len_ = len;
    }
}

Sha1::Digest Sha1::finish() {
    const uint64_t bit_len = total_len_ * 8u;
    uint8_t pad = 0x80;
    update(&pad, 1);
    const uint8_t zero = 0;
    while (buf_len_ != 56) update(&zero, 1);
    uint8_t len_be[8];
    for (int i = 0; i < 8; ++i) len_be[i] = static_cast<uint8_t>(bit_len >> (56 - 8 * i));
    update(len_be, 8);
    Digest out{};
    for (int i = 0; i < 5; ++i) store_be32(out.data() + 4 * i, h_[i]);
    return out;
}

Sha1::Digest Sha1::digest(const uint8_t* data, std::size_t len) {
    Sha1 h;
    h.update(data, len);
    return h.finish();
}

}  // namespace lowlevel
}  // namespace opencrab2
