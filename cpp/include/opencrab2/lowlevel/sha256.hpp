// SHA-256 (FIPS 180-4), streaming and one-shot (bundle tags, telemetry tags, the contract hash).
//
// Tested (tests/test_lowlevel.cpp) against the FIPS 180-4 / RFC 6234 vectors: "", "abc",
// the 56-byte two-block message and one million 'a'.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace opencrab2 {
namespace lowlevel {

class Sha256 {
public:
    static constexpr std::size_t digest_size = 32;
    static constexpr std::size_t block_size = 64;
    using Digest = std::array<uint8_t, digest_size>;

    Sha256();
    void reset();
    void update(const uint8_t* data, std::size_t len);
    void update(const std::string& s) { update(reinterpret_cast<const uint8_t*>(s.data()), s.size()); }
    Digest finish();   // the object must be reset() before reuse

    static Digest digest(const uint8_t* data, std::size_t len);
    static Digest digest(const std::string& s) {
        return digest(reinterpret_cast<const uint8_t*>(s.data()), s.size());
    }

private:
    void compress(const uint8_t block[64]);
    uint32_t h_[8];
    uint8_t buf_[64];
    std::size_t buf_len_;
    uint64_t total_len_;
};

}  // namespace lowlevel
}  // namespace opencrab2
