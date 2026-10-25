// SHA-1 (FIPS 180-4, RFC 3174), streaming and one-shot.
//
// SHA-1 is NOT used for any security purpose in OpenCrab v2. It exists only because the
// WebSocket opening handshake (RFC 6455 Sec. 4.2.2) requires
// base64(SHA1(Sec-WebSocket-Key || "258EAFA5-E914-47DA-95CA-C5AB0DC85B11")).
// Tested (tests/test_lowlevel.cpp) against the FIPS 180-4 vectors: "", "abc", the 56-byte
// message, one million 'a'.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace opencrab2 {
namespace lowlevel {

class Sha1 {
public:
    static constexpr std::size_t digest_size = 20;
    static constexpr std::size_t block_size = 64;
    using Digest = std::array<uint8_t, digest_size>;

    Sha1();
    void reset();
    void update(const uint8_t* data, std::size_t len);
    void update(const std::string& s) { update(reinterpret_cast<const uint8_t*>(s.data()), s.size()); }
    Digest finish();

    static Digest digest(const uint8_t* data, std::size_t len);
    static Digest digest(const std::string& s) {
        return digest(reinterpret_cast<const uint8_t*>(s.data()), s.size());
    }

private:
    void compress(const uint8_t block[64]);
    uint32_t h_[5];
    uint8_t buf_[64];
    std::size_t buf_len_;
    uint64_t total_len_;
};

}  // namespace lowlevel
}  // namespace opencrab2
