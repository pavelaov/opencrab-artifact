// Randomness, header-only: the operating system CSPRNG and the ChaCha20 stream it keys (paper,
// Appendix D, Disclosure).
//
// os_random_bytes: the operating system CSPRNG. Windows: BCryptGenRandom with
// BCRYPT_USE_SYSTEM_PREFERRED_RNG, resolved at run time from bcrypt.dll so no extra link flag is
// needed. Elsewhere: /dev/urandom. Fails closed with CSPRNG_UNAVAILABLE / CSPRNG_FAILED.
//
// Rng: a ChaCha20 (RFC 7539) keystream generator. Production code seeds it from the OS
// (Rng::from_os / from_os_or_abort); tests seed it deterministically (Rng::from_seed or an
// explicit 32-byte key) so that every fixture is reproducible. The 32-bit block counter of RFC
// 7539 is extended by carrying into the first nonce word after 2^32 blocks (256 GiB), a design
// limit that only matters for streams longer than any bundle this code produces.
//
// Tested (tests/test_lowlevel.cpp): the RFC 7539 Sec. 2.3.2 block function vector, determinism of
// seeded streams, that the OS source returns and is not constant, uniformity of below().
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "opencrab2/expected.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace opencrab2 {
namespace lowlevel {

inline Status os_random_bytes(uint8_t* out, std::size_t n) {
#ifdef _WIN32
    typedef LONG(WINAPI * BCryptGenRandomFn)(void*, unsigned char*, ULONG, ULONG);
    static const BCryptGenRandomFn fn = []() -> BCryptGenRandomFn {
        HMODULE h = LoadLibraryW(L"bcrypt.dll");
        if (!h) return nullptr;
        return reinterpret_cast<BCryptGenRandomFn>(reinterpret_cast<void*>(GetProcAddress(h, "BCryptGenRandom")));
    }();
    if (!fn) return fail_status("CSPRNG_UNAVAILABLE");
    const ULONG kUseSystemPreferredRng = 0x00000002;
    while (n > 0) {
        const ULONG chunk = n > 0x7fffffffu ? 0x7fffffffu : static_cast<ULONG>(n);
        const LONG st = fn(nullptr, out, chunk, kUseSystemPreferredRng);
        if (st != 0) return fail_status("CSPRNG_FAILED");
        out += chunk;
        n -= chunk;
    }
    return ok_status();
#else
    const int fd = ::open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    if (fd < 0) return fail_status("CSPRNG_UNAVAILABLE");
    while (n > 0) {
        const ssize_t got = ::read(fd, out, n);
        if (got < 0 && errno == EINTR) continue;   // interrupted by a signal handler: nothing was read
        if (got <= 0) {
            ::close(fd);
            return fail_status("CSPRNG_FAILED");
        }
        out += static_cast<std::size_t>(got);
        n -= static_cast<std::size_t>(got);
    }
    ::close(fd);
    return ok_status();
#endif
}

inline Status os_random_bytes(std::vector<uint8_t>& out) { return os_random_bytes(out.data(), out.size()); }

// ---------------------------------------------------------------------------------------------
// ChaCha20 block function (RFC 7539 Sec. 2.3)
// ---------------------------------------------------------------------------------------------
namespace detail {
inline uint32_t rotl32(uint32_t v, int c) { return (v << c) | (v >> (32 - c)); }
inline uint32_t load_le32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) | (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}
inline void store_le32(uint8_t* p, uint32_t v) {
    p[0] = static_cast<uint8_t>(v);
    p[1] = static_cast<uint8_t>(v >> 8);
    p[2] = static_cast<uint8_t>(v >> 16);
    p[3] = static_cast<uint8_t>(v >> 24);
}
inline void quarter_round(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
    a += b; d ^= a; d = rotl32(d, 16);
    c += d; b ^= c; b = rotl32(b, 12);
    a += b; d ^= a; d = rotl32(d, 8);
    c += d; b ^= c; b = rotl32(b, 7);
}
}  // namespace detail

inline void chacha20_block(const uint8_t key[32], uint32_t counter, const uint8_t nonce[12], uint8_t out[64]) {
    uint32_t st[16];
    st[0] = 0x61707865u;
    st[1] = 0x3320646eu;
    st[2] = 0x79622d32u;
    st[3] = 0x6b206574u;
    for (int i = 0; i < 8; ++i) st[4 + i] = detail::load_le32(key + 4 * i);
    st[12] = counter;
    for (int i = 0; i < 3; ++i) st[13 + i] = detail::load_le32(nonce + 4 * i);
    uint32_t x[16];
    std::memcpy(x, st, sizeof(x));
    for (int i = 0; i < 10; ++i) {
        detail::quarter_round(x[0], x[4], x[8], x[12]);
        detail::quarter_round(x[1], x[5], x[9], x[13]);
        detail::quarter_round(x[2], x[6], x[10], x[14]);
        detail::quarter_round(x[3], x[7], x[11], x[15]);
        detail::quarter_round(x[0], x[5], x[10], x[15]);
        detail::quarter_round(x[1], x[6], x[11], x[12]);
        detail::quarter_round(x[2], x[7], x[8], x[13]);
        detail::quarter_round(x[3], x[4], x[9], x[14]);
    }
    for (int i = 0; i < 16; ++i) detail::store_le32(out + 4 * i, x[i] + st[i]);
}

// ---------------------------------------------------------------------------------------------
// Rng
// ---------------------------------------------------------------------------------------------
class Rng {
public:
    using Key = std::array<uint8_t, 32>;
    using Nonce = std::array<uint8_t, 12>;

    explicit Rng(const Key& key, const Nonce& nonce = Nonce{}) : key_(key), nonce_(nonce) {}

    // Deterministic generator for tests: key = little-endian seed in bytes 0..7, zero elsewhere.
    static Rng from_seed(uint64_t seed) {
        Key k{};
        for (int i = 0; i < 8; ++i) k[i] = static_cast<uint8_t>(seed >> (8 * i));
        return Rng(k);
    }

    // Reseeds `out` from the OS CSPRNG (key and nonce). On failure `out` is left unchanged.
    static Status from_os(Rng& out) {
        uint8_t raw[44];
        const Status st = os_random_bytes(raw, sizeof(raw));
        if (!st.ok) return st;
        Key k;
        Nonce n;
        std::memcpy(k.data(), raw, 32);
        std::memcpy(n.data(), raw + 32, 12);
        out = Rng(k, n);
        out.os_seeded_ = true;
        return ok_status();
    }

    // True only for a generator keyed by from_os (or from_os_or_abort): nobody outside this process
    // can replay it. A generator built from a seed or an explicit key reports false, and material
    // drawn from it is never security evidence (highlevel::Dealer labels its bundles insecure).
    bool os_seeded() const { return os_seeded_; }

    // Production convenience: an OS-seeded generator, or a loud abort (fail closed, never a weak default).
    static Rng from_os_or_abort() {
        Rng r = from_seed(0);
        const Status st = from_os(r);
        if (!st.ok) {
            std::fprintf(stderr, "%s: operating system CSPRNG not available\n", st.code.c_str());
            std::abort();
        }
        return r;
    }

    uint64_t next_u64() {
        uint8_t b[8];
        fill(b, 8);
        uint64_t v = 0;
        for (int i = 7; i >= 0; --i) v = (v << 8) | b[i];
        return v;
    }

    uint32_t next_u32() { return static_cast<uint32_t>(next_u64()); }

    void fill(uint8_t* out, std::size_t n) {
        while (n > 0) {
            if (pos_ == 64) refill();
            const std::size_t take = (64 - pos_ < n) ? (64 - pos_) : n;
            std::memcpy(out, buf_ + pos_, take);
            pos_ += take;
            out += take;
            n -= take;
        }
    }

    void fill(std::vector<uint8_t>& out) { fill(out.data(), out.size()); }

    // Uniform in [0, bound) by rejection sampling; bound == 0 returns 0.
    uint64_t below(uint64_t bound) {
        if (bound <= 1) return 0;
        if ((bound & (bound - 1)) == 0) return next_u64() & (bound - 1);
        const uint64_t zone = UINT64_MAX - (UINT64_MAX % bound);   // largest multiple of bound <= UINT64_MAX
        for (;;) {
            const uint64_t x = next_u64();
            if (x < zone) return x % bound;
        }
    }

private:
    void refill() {
        chacha20_block(key_.data(), counter_, nonce_.data(), buf_);
        if (++counter_ == 0) {
            // carry into the first nonce word (design limit: deviates from RFC 7539 after 2^32 blocks)
            const uint32_t n0 = detail::load_le32(nonce_.data()) + 1u;
            detail::store_le32(nonce_.data(), n0);
        }
        pos_ = 0;
    }

    Key key_{};
    Nonce nonce_{};
    uint32_t counter_ = 0;
    uint8_t buf_[64] = {};
    std::size_t pos_ = 64;
    bool os_seeded_ = false;
};

}  // namespace lowlevel
}  // namespace opencrab2
