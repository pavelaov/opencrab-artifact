// OpenCrab v2 C++ lowlevel test suite.
//
// Prints "PASS <n> checks" and exits 0, or prints the first failing check and exits 1.
// With a directory argument it also writes the cross-language vectors cpp_prg_vectors.json,
// cpp_dcf_vectors.json and cpp_material_vectors.json into that directory (LF line ends on every
// host); python/tests/test_cross_language.py checks them with the Python package.
//
// Build: python cpp/build.py test_lowlevel, or the CMake target test_lowlevel.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "opencrab2/expected.hpp"
#include "opencrab2/lowlevel/aes128.hpp"
#include "opencrab2/lowlevel/base64.hpp"
#include "opencrab2/lowlevel/csprng.hpp"
#include "opencrab2/lowlevel/dcf.hpp"
#include "opencrab2/lowlevel/drelu.hpp"
#include "opencrab2/lowlevel/hmac.hpp"
#include "opencrab2/lowlevel/prg.hpp"
#include "opencrab2/lowlevel/ring.hpp"
#include "opencrab2/lowlevel/ring_to_field.hpp"
#include "opencrab2/lowlevel/sha1.hpp"
#include "opencrab2/lowlevel/sha256.hpp"
#include "opencrab2/lowlevel/sharing.hpp"
#include "opencrab2/lowlevel/trunc.hpp"

namespace {

using namespace opencrab2;
using namespace opencrab2::lowlevel;
using u64 = uint64_t;
using i64 = int64_t;

long g_checks = 0;

[[noreturn]] void fail_at(const char* expr, const char* file, int line, const std::string& detail) {
    std::printf("FAIL %s at %s:%d%s%s\n", expr, file, line, detail.empty() ? "" : ": ", detail.c_str());
    std::fflush(stdout);
    std::exit(1);
}

#define CHECK(cond)                                                   \
    do {                                                              \
        if (!(cond)) fail_at(#cond, __FILE__, __LINE__, "");          \
        ++g_checks;                                                   \
    } while (0)

#define CHECK_EQ_U64(a, b)                                                                                   \
    do {                                                                                                     \
        const u64 _a = static_cast<u64>(a);                                                                  \
        const u64 _b = static_cast<u64>(b);                                                                  \
        if (_a != _b) fail_at(#a " == " #b, __FILE__, __LINE__, "got " + std::to_string(_a) + " want " + std::to_string(_b)); \
        ++g_checks;                                                                                          \
    } while (0)

#define CHECK_EQ_I64(a, b)                                                                                   \
    do {                                                                                                     \
        const i64 _a = static_cast<i64>(a);                                                                  \
        const i64 _b = static_cast<i64>(b);                                                                  \
        if (_a != _b) fail_at(#a " == " #b, __FILE__, __LINE__, "got " + std::to_string(_a) + " want " + std::to_string(_b)); \
        ++g_checks;                                                                                          \
    } while (0)

#define CHECK_OK(expr)                                                        \
    do {                                                                      \
        const auto& _e = (expr);                                              \
        if (!_e.ok) fail_at(#expr, __FILE__, __LINE__, _e.code);              \
        ++g_checks;                                                           \
    } while (0)

#define CHECK_FAILS(expr, prefix)                                                                       \
    do {                                                                                                \
        const auto& _e = (expr);                                                                        \
        if (_e.ok) fail_at(#expr " should fail", __FILE__, __LINE__, "");                               \
        if (_e.code.compare(0, std::strlen(prefix), prefix) != 0)                                       \
            fail_at(#expr " code", __FILE__, __LINE__, "got " + _e.code + " want prefix " + prefix);    \
        ++g_checks;                                                                                     \
    } while (0)

struct Timer {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit Timer(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~Timer() {
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "[test_lowlevel] %-28s %7.2f s  (%ld checks so far)\n", name, s, g_checks);
    }
};

// Worker-side result of one parallel job. Workers never touch g_checks and never call fail_at:
// they count locally and report the first failure as text; run_parallel folds the results on the
// calling thread.
struct JobResult {
    long checks = 0;
    std::string failure;
};

unsigned test_threads() {
    unsigned n = std::thread::hardware_concurrency();
    if (n == 0) n = 2;
    if (n > 12) n = 12;
    if (const char* e = std::getenv("OPENCRAB2_TEST_THREADS")) {
        const int v = std::atoi(e);
        if (v >= 1 && v <= 64) n = static_cast<unsigned>(v);
    }
    return n;
}

// Runs fn(job, result) for job in [0, n_jobs) on a small thread pool (dynamic scheduling).
template <class F>
void run_parallel(std::size_t n_jobs, F fn, const char* what) {
    std::vector<JobResult> results(n_jobs);
    std::atomic<std::size_t> next{0};
    const unsigned T = static_cast<unsigned>(std::min<std::size_t>(test_threads(), n_jobs ? n_jobs : 1));
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < T; ++t)
        pool.emplace_back([&]() {
            for (;;) {
                const std::size_t j = next.fetch_add(1);
                if (j >= n_jobs) break;
                fn(j, results[j]);
            }
        });
    for (std::thread& th : pool) th.join();
    for (const JobResult& r : results) {
        if (!r.failure.empty()) fail_at(what, __FILE__, __LINE__, r.failure);
        g_checks += r.checks;
    }
}

std::vector<uint8_t> hexv(const std::string& s) {
    const Expected<std::vector<uint8_t>> r = from_hex(s);
    if (!r.ok) fail_at("from_hex", __FILE__, __LINE__, s);
    return r.value;
}

std::vector<uint8_t> bytes_of(const std::string& s) { return std::vector<uint8_t>(s.begin(), s.end()); }

std::string dec(u64 v) { return std::to_string(v); }

i64 floor_div_pow2(i64 x, int f) { return ring::to_signed(ring::arith_shift_right(ring::from_signed(x), f)); }

// ---------------------------------------------------------------------------------------------
// AES, SHA, HMAC, base64, CSPRNG
// ---------------------------------------------------------------------------------------------
void test_aes() {
    Timer t("aes128");
    {
        std::vector<uint8_t> key = hexv("000102030405060708090a0b0c0d0e0f");
        std::vector<uint8_t> pt = hexv("00112233445566778899aabbccddeeff");
        uint8_t ct[16];
        Aes128 aes(key.data());
        aes.encrypt_block(pt.data(), ct);
        CHECK(to_hex(ct, 16) == "69c4e0d86a7b0430d8cdb78070b4c55a");   // FIPS-197 C.1
        u64 lo, hi;
        const Block128 b = block_from_bytes(pt.data());
        aes.encrypt_words(b.lo, b.hi, lo, hi);
        uint8_t ct2[16];
        block_to_bytes(Block128{lo, hi}, ct2);
        CHECK(std::memcmp(ct, ct2, 16) == 0);
    }
    {
        std::vector<uint8_t> key = hexv("2b7e151628aed2a6abf7158809cf4f3c");   // FIPS-197 Appendix B (the PRG key)
        std::vector<uint8_t> pt = hexv("3243f6a8885a308d313198a2e0370734");
        uint8_t ct[16];
        Aes128 aes(key.data());
        aes.encrypt_block(pt.data(), ct);
        CHECK(to_hex(ct, 16) == "3925841d02dc09fbdc118597196a0b32");
    }
}

void test_sha() {
    Timer t("sha256 / sha1");
    const std::string m56 = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    CHECK(to_hex(Sha256::digest(std::string(""))) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(to_hex(Sha256::digest(std::string("abc"))) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK(to_hex(Sha256::digest(m56)) == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    CHECK(to_hex(Sha1::digest(std::string(""))) == "da39a3ee5e6b4b0d3255bfef95601890afd80709");
    CHECK(to_hex(Sha1::digest(std::string("abc"))) == "a9993e364706816aba3e25717850c26c9cd0d89d");
    CHECK(to_hex(Sha1::digest(m56)) == "84983e441c3bd26ebaae4aa1f95129e5e54670f1");
    // one million 'a', streamed in irregular chunks
    {
        Sha256 h256;
        Sha1 h1;
        std::vector<uint8_t> chunk(997, 'a');
        std::size_t remaining = 1000000;
        while (remaining > 0) {
            const std::size_t n = remaining < chunk.size() ? remaining : chunk.size();
            h256.update(chunk.data(), n);
            h1.update(chunk.data(), n);
            remaining -= n;
        }
        CHECK(to_hex(h256.finish()) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
        CHECK(to_hex(h1.finish()) == "34aa973cd4c4daa4f61eeb2bdbad27316534016f");
    }
    // streaming equals one-shot on random data with every split point
    {
        Rng rng = Rng::from_seed(1);
        std::vector<uint8_t> data(300);
        rng.fill(data);
        const auto ref = Sha256::digest(data.data(), data.size());
        for (std::size_t split = 0; split <= data.size(); split += 37) {
            Sha256 h;
            h.update(data.data(), split);
            h.update(data.data() + split, data.size() - split);
            CHECK(h.finish() == ref);
        }
    }
}

void test_hmac() {
    Timer t("hmac");
    auto h256 = [](const std::vector<uint8_t>& k, const std::vector<uint8_t>& m) {
        return to_hex(hmac_sha256(k.data(), k.size(), m.data(), m.size()));
    };
    auto h1 = [](const std::vector<uint8_t>& k, const std::vector<uint8_t>& m) {
        return to_hex(hmac_sha1(k.data(), k.size(), m.data(), m.size()));
    };
    // RFC 4231
    CHECK(h256(std::vector<uint8_t>(20, 0x0b), bytes_of("Hi There")) == "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7");
    CHECK(h256(bytes_of("Jefe"), bytes_of("what do ya want for nothing?")) == "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
    CHECK(h256(std::vector<uint8_t>(20, 0xaa), std::vector<uint8_t>(50, 0xdd)) == "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe");
    {
        std::vector<uint8_t> k(25);
        for (int i = 0; i < 25; ++i) k[i] = static_cast<uint8_t>(i + 1);
        CHECK(h256(k, std::vector<uint8_t>(50, 0xcd)) == "82558a389a443c0ea4cc819899f2083a85f0faa3e578f8077a2e3ff46729665b");
    }
    CHECK(h256(std::vector<uint8_t>(131, 0xaa), bytes_of("Test Using Larger Than Block-Size Key - Hash Key First")) ==
          "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54");
    CHECK(h256(std::vector<uint8_t>(131, 0xaa),
               bytes_of("This is a test using a larger than block-size key and a larger than block-size data. The key needs to be "
                        "hashed before being used by the HMAC algorithm.")) ==
          "9b09ffa71b942fcb27635fbcd5b0e944bfdc63644f0713938a7f51535c3a35e2");
    // RFC 2202
    CHECK(h1(std::vector<uint8_t>(20, 0x0b), bytes_of("Hi There")) == "b617318655057264e28bc0b6fb378c8ef146be00");
    CHECK(h1(bytes_of("Jefe"), bytes_of("what do ya want for nothing?")) == "effcdf6ae5eb2fa2d27416d5f184df9c259a7c79");
    CHECK(h1(std::vector<uint8_t>(20, 0xaa), std::vector<uint8_t>(50, 0xdd)) == "125d7342b9ac11cd91a39af48aa17b4f63f175d3");
    // constant-time compare
    {
        std::array<uint8_t, 32> a{}, b{};
        CHECK(constant_time_equal(a, b));
        b[31] = 1;
        CHECK(!constant_time_equal(a, b));
    }
}

void test_base64_hex() {
    Timer t("base64 / hex");
    const char* plain[] = {"", "f", "fo", "foo", "foob", "fooba", "foobar"};
    const char* enc[] = {"", "Zg==", "Zm8=", "Zm9v", "Zm9vYg==", "Zm9vYmE=", "Zm9vYmFy"};
    for (int i = 0; i < 7; ++i) {
        CHECK(base64_encode(std::string(plain[i])) == enc[i]);
        const Expected<std::vector<uint8_t>> d = base64_decode(enc[i]);
        CHECK_OK(d);
        CHECK(std::string(d.value.begin(), d.value.end()) == plain[i]);
    }
    CHECK_FAILS(base64_decode("Zg="), "BASE64_LENGTH");
    CHECK_FAILS(base64_decode("Zg=a"), "BASE64_PADDING");
    CHECK_FAILS(base64_decode("Z*=="), "BASE64_ALPHABET");
    // RFC 6455 Sec. 1.3 WebSocket accept key
    {
        const std::string s = "dGhlIHNhbXBsZSBub25jZQ==258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
        const auto d = Sha1::digest(s);
        CHECK(base64_encode(d.data(), d.size()) == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");
    }
    Rng rng = Rng::from_seed(2);
    for (int n = 0; n < 64; ++n) {
        std::vector<uint8_t> data(static_cast<std::size_t>(n));
        rng.fill(data);
        const Expected<std::vector<uint8_t>> back = base64_decode(base64_encode(data));
        CHECK(back.ok && back.value == data);
        const Expected<std::vector<uint8_t>> hb = from_hex(to_hex(data));
        CHECK(hb.ok && hb.value == data);
    }
    CHECK_FAILS(from_hex("abc"), "HEX_LENGTH");
    CHECK_FAILS(from_hex("zz"), "HEX_ALPHABET");
}

void test_csprng() {
    Timer t("csprng / chacha20");
    {
        uint8_t key[32];
        for (int i = 0; i < 32; ++i) key[i] = static_cast<uint8_t>(i);
        const std::vector<uint8_t> nonce = hexv("000000090000004a00000000");
        uint8_t out[64];
        chacha20_block(key, 1, nonce.data(), out);   // RFC 7539 Sec. 2.3.2
        CHECK(to_hex(out, 64) ==
              "10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4ed2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e");
    }
    {
        Rng a = Rng::from_seed(123), b = Rng::from_seed(123), c = Rng::from_seed(124);
        bool differs = false;
        for (int i = 0; i < 1000; ++i) {
            const u64 va = a.next_u64();
            CHECK_EQ_U64(va, b.next_u64());
            if (va != c.next_u64()) differs = true;
        }
        CHECK(differs);
        // fill and next_u64 read the same stream (little-endian words)
        Rng d = Rng::from_seed(5), e = Rng::from_seed(5);
        std::vector<uint8_t> buf(200);
        d.fill(buf);
        for (int i = 0; i < 25; ++i) {
            u64 v = 0;
            for (int j = 7; j >= 0; --j) v = (v << 8) | buf[8 * i + j];
            CHECK_EQ_U64(v, e.next_u64());
        }
        CHECK_EQ_U64(d.below(1), 0);
        for (int i = 0; i < 10000; ++i) {
            CHECK(d.below(1000) < 1000);
            CHECK(d.below(u64(1) << 20) < (u64(1) << 20));
        }
        // the rejection zone cannot break uniformity on a tiny bound: every residue appears
        int seen[7] = {0, 0, 0, 0, 0, 0, 0};
        for (int i = 0; i < 7000; ++i) ++seen[d.below(7)];
        for (int i = 0; i < 7; ++i) CHECK(seen[i] > 700 && seen[i] < 1300);
    }
    {
        uint8_t a[32], b[32];
        CHECK_OK(os_random_bytes(a, 32));
        CHECK_OK(os_random_bytes(b, 32));
        CHECK(std::memcmp(a, b, 32) != 0);
        Rng r = Rng::from_seed(0);
        CHECK_OK(Rng::from_os(r));
        Rng z = Rng::from_seed(0);
        CHECK(r.next_u64() != z.next_u64());
    }
}

// ---------------------------------------------------------------------------------------------
// ring, prg, sharing
// ---------------------------------------------------------------------------------------------
void test_ring() {
    Timer t("ring");
    CHECK_EQ_U64(ring::add(ring::MASK64, 1), 0);
    CHECK_EQ_U64(ring::sub(0, 1), ring::MASK64);
    CHECK_EQ_U64(ring::mul(u64(1) << 63, 2), 0);
    CHECK_EQ_U64(ring::neg(1), ring::MASK64);
    CHECK_EQ_I64(ring::to_signed(ring::MASK64), -1);
    CHECK_EQ_I64(ring::to_signed(u64(1) << 63), INT64_MIN);
    CHECK_EQ_I64(ring::to_signed(5), 5);
    CHECK_EQ_U64(ring::from_signed(-1), ring::MASK64);
    CHECK_EQ_U64(ring::msb(u64(1) << 63), 1);
    CHECK_EQ_U64(ring::msb(123), 0);
    // arithmetic shift against floor division
    Rng rng = Rng::from_seed(3);
    for (int i = 0; i < 20000; ++i) {
        const u64 v = rng.next_u64();
        const int f = static_cast<int>(rng.below(63));   // 0..62
        const i64 s = ring::to_signed(v);
        const i64 d = static_cast<i64>(u64(1) << f);
        i64 q = s / d;
        if ((s % d) != 0 && s < 0) q -= 1;
        CHECK_EQ_I64(ring::to_signed(ring::arith_shift_right(v, f)), q);
    }
    CHECK_EQ_I64(ring::to_signed(ring::arith_shift_right(ring::from_signed(-1), 63)), -1);
    CHECK_EQ_I64(ring::to_signed(ring::arith_shift_right(12345, 63)), 0);
    CHECK_EQ_U64(ring::arith_shift_right(ring::from_signed(-7), 64), ring::MASK64);
    // fixed point
    {
        Expected<u64> e = ring::encode(1.5, 16, 40);
        CHECK_OK(e);
        CHECK_EQ_U64(e.value, 98304);
        CHECK(ring::decode(e.value, 16) == 1.5);
        e = ring::encode(-1.5, 16, 40);
        CHECK_OK(e);
        CHECK_EQ_U64(e.value, ring::from_signed(-98304));
        CHECK(ring::decode(e.value, 16) == -1.5);
        CHECK_FAILS(ring::encode(8388608.0, 16, 40), "CONTRACT_VIOLATION");           // 2^23 * 2^16 = 2^39
        CHECK_OK(ring::encode(8388608.0 - 1.0 / 65536.0, 16, 40));
        CHECK_FAILS(ring::encode(-8388608.0 - 1.0 / 65536.0, 16, 40), "CONTRACT_VIOLATION");
        CHECK_FAILS(ring::encode(-8388608.0, 16, 40), "CONTRACT_VIOLATION");          // |v| = 2^(l-1) is refused on both sides
        CHECK_OK(ring::encode(-8388608.0 + 1.0 / 65536.0, 16, 40));
    }
    CHECK_FAILS(ring::encode(std::nan(""), 16, 40), "CONTRACT_VIOLATION");
    CHECK_FAILS(ring::encode(1.0 / 0.0, 16, 40), "CONTRACT_VIOLATION");
    CHECK_FAILS(ring::encode(1.0, 64, 40), "CONTRACT_PARAMS");
    CHECK_EQ_U64(ring::encode(0.5 / 65536.0, 16, 40).value, 0);   // rint half to even
    CHECK_EQ_U64(ring::encode(1.5 / 65536.0, 16, 40).value, 2);
    {
        const std::vector<double> xs = {0.0, 1.0, -2.25, 1000.5};
        const Expected<std::vector<u64>> v = ring::encode(xs, 16, 40);
        CHECK_OK(v);
        const std::vector<double> back = ring::decode(v.value, 16);
        for (std::size_t i = 0; i < xs.size(); ++i) CHECK(back[i] == xs[i]);
    }
    // small rings
    CHECK_EQ_I64(ring::to_signed_k(255, 8), -1);
    CHECK_EQ_I64(ring::to_signed_k(127, 8), 127);
    CHECK_EQ_I64(ring::to_signed_k(128, 8), -128);
    CHECK_EQ_U64(ring::from_signed_k(-1, 8), 255);
    CHECK_EQ_U64(ring::reduce_k(0x1234, 8), 0x34);
    CHECK_EQ_U64(ring::reduce_k(ring::MASK64, 64), ring::MASK64);
    CHECK_FAILS(ring::check_k(0), "RING_BITS");
    CHECK_FAILS(ring::check_k(65), "RING_BITS");
    {
        const std::vector<u64> a = {1, 2}, b = {3, 4, 5};
        CHECK_FAILS(ring::add(a, b), "RING_SHAPE");
    }
}

void test_prg() {
    Timer t("prg");
    uint8_t seed[16];
    for (int i = 0; i < 16; ++i) seed[i] = static_cast<uint8_t>(i);
    uint8_t out[64];
    expand_bytes(seed, out);
    // identical in Python (opencrab2.lowlevel.prg.expand_bytes) and in a direct cryptography AES-ECB computation
    CHECK(to_hex(out, 64) ==
          "50ff65cf9d6834b1d2003de297a2e26f922a73060b96decbfe08492b03d6868e5e1b4a1ba11e6a59b159221f73334c70b8b54cb235923debc21bd439a15689f8");
    const Expansion e = expand(seed);
    CHECK(e.sL == block_from_bytes(out));
    CHECK(e.sR == block_from_bytes(out + 16));
    CHECK_EQ_U64(e.vL, dcf::wire::get_u64le(out + 32));
    CHECK_EQ_U64(e.vR, dcf::wire::get_u64le(out + 40));
    CHECK_EQ_U64(e.tL, out[48] & 1u);
    CHECK_EQ_U64(e.tR, out[PRG_TR_BYTE] & 1u);
    CHECK(PRG_TR_BYTE == 49);
    // mmo(s, j) == AES_K(s xor c_j) xor s, computed independently on bytes
    Rng rng = Rng::from_seed(11);
    for (int it = 0; it < 200; ++it) {
        uint8_t s[16];
        rng.fill(s, 16);
        uint8_t exp64[64];
        expand_bytes(s, exp64);
        for (unsigned j = 0; j < 4; ++j) {
            uint8_t x[16], y[16];
            for (int i = 0; i < 16; ++i) x[i] = s[i];
            x[15] = static_cast<uint8_t>(x[15] ^ j);   // c_j is the big-endian integer j: byte 15
            prg_cipher().encrypt_block(x, y);
            for (int i = 0; i < 16; ++i) y[i] = static_cast<uint8_t>(y[i] ^ s[i]);
            CHECK(std::memcmp(y, exp64 + 16 * j, 16) == 0);
            uint8_t m[16];
            block_to_bytes(mmo(block_from_bytes(s), j), m);
            CHECK(std::memcmp(m, y, 16) == 0);
        }
        const Expansion ex = expand(s);
        CHECK(ex.sL == block_from_bytes(exp64) && ex.sR == block_from_bytes(exp64 + 16));
        CHECK_EQ_U64(ex.vL, dcf::wire::get_u64le(exp64 + 32));
        CHECK_EQ_U64(ex.vR, dcf::wire::get_u64le(exp64 + 40));
        CHECK_EQ_U64(ex.tL, exp64[48] & 1u);
        CHECK_EQ_U64(ex.tR, exp64[49] & 1u);
        uint8_t rt[16];
        block_to_bytes(block_from_bytes(s), rt);
        CHECK(std::memcmp(rt, s, 16) == 0);
    }
    std::vector<Block128> seeds = {Block128{1, 2}, Block128{3, 4}};
    std::vector<Expansion> outs;
    expand_batch(seeds, outs);
    CHECK(outs.size() == 2 && outs[1].sL == expand(seeds[1]).sL);

    // --- component-specific Convert (prg.hpp) -----------------------------------------------------
    // tweak indices of the rule
    CHECK(prg_ring_value_tweak(0) == 2 && prg_ring_value_tweak(1) == 4 && prg_ring_value_tweak(2) == 5 && prg_ring_value_tweak(254) == 257);
    CHECK(prg_zt_value_tweak(0, 0) == 4 && prg_zt_value_tweak(0, 1) == 5 && prg_zt_value_tweak(1, 0) == 6 && prg_zt_value_tweak(2, 1) == 9);
    CHECK(prg_zt_value_tweak(254, 1) == 513);
    // mmo on the tweaks above 3 and on multi-byte tweaks: c_j is the 16-byte big-endian integer j
    for (int it = 0; it < 100; ++it) {
        uint8_t s[16];
        rng.fill(s, 16);
        const Block128 sb = block_from_bytes(s);
        for (uint32_t j : {4u, 5u, 6u, 7u, 8u, 9u, 255u, 256u, 257u, 513u, 0x01020304u}) {
            uint8_t x[16], y[16], m[16], mb[16];
            for (int i = 0; i < 16; ++i) x[i] = s[i];
            x[15] = static_cast<uint8_t>(x[15] ^ (j & 0xffu));
            x[14] = static_cast<uint8_t>(x[14] ^ ((j >> 8) & 0xffu));
            x[13] = static_cast<uint8_t>(x[13] ^ ((j >> 16) & 0xffu));
            x[12] = static_cast<uint8_t>(x[12] ^ ((j >> 24) & 0xffu));
            prg_cipher().encrypt_block(x, y);
            for (int i = 0; i < 16; ++i) y[i] = static_cast<uint8_t>(y[i] ^ s[i]);
            block_to_bytes(mmo(sb, j), m);
            CHECK(std::memcmp(m, y, 16) == 0);
            mmo_bytes(s, j, mb);
            CHECK(std::memcmp(mb, y, 16) == 0);
        }
        // Z_{2^64} words: component 0 from block 2 (the base layout), component c >= 1 from block 3 + c
        const Expansion ex = expand(s);
        CHECK_EQ_U64(value_words_ring(sb, 0).vL, ex.vL);
        CHECK_EQ_U64(value_words_ring(sb, 0).vR, ex.vR);
        for (uint32_t c = 1; c <= 3; ++c) {
            uint8_t blk[16];
            mmo_bytes(s, 3 + c, blk);
            const ValueWords w = value_words_ring(sb, c);
            CHECK_EQ_U64(w.vL, dcf::wire::get_u64le(blk));
            CHECK_EQ_U64(w.vR, dcf::wire::get_u64le(blk + 8));
            CHECK_EQ_U64(value_word_ring(sb, c, 0), w.vL);
            CHECK_EQ_U64(value_word_ring(sb, c, 1), w.vR);
            CHECK(w.vL != ex.vL && w.vR != ex.vR);   // a component never reuses component 0's words
        }
        // Z_t words: LE128 of blocks 4 + 2c and 5 + 2c reduced mod t (128 bits)
        for (u64 tt : {u64(3), u64(17), u64(1000003), u64(68720050177ULL), (u64(1) << 62) - 57}) {
            for (uint32_t c = 0; c <= 2; ++c) {
                const ValueWords w = value_words_zt(sb, c, tt);
                for (uint32_t side = 0; side < 2; ++side) {
                    uint8_t blk[16];
                    mmo_bytes(s, 4 + 2 * c + side, blk);
                    // independent reduction: Horner over the 16 bytes, most significant first
                    u64 r = 0;
                    for (int i = 15; i >= 0; --i) {
                        for (int bit = 7; bit >= 0; --bit) {
                            r = (r << 1) | ((blk[i] >> bit) & 1u);
                            if (r >= tt) r -= tt;
                        }
                    }
                    const u64 got = side ? w.vR : w.vL;
                    CHECK_EQ_U64(got, r);
                    CHECK(got < tt);
                    CHECK_EQ_U64(value_word_zt(sb, c, side, tt), r);
                    CHECK_EQ_U64(block_mod_portable(block_from_bytes(blk), tt), r);
                    CHECK_EQ_U64(block_mod(block_from_bytes(blk), tt), r);
                }
            }
        }
    }
    // the 128-bit reduction really uses the high word: 2^64 mod t differs from 0 for odd t
    CHECK_EQ_U64(block_mod(Block128{0, 1}, 17), 1);                       // 2^64 = 1 mod 17
    CHECK_EQ_U64(block_mod(Block128{5, 3}, 17), (3 * 1 + 5) % 17);
    CHECK_EQ_U64(block_mod(Block128{ring::MASK64, ring::MASK64}, 3), 0);   // 2^128 - 1 = 0 mod 3
    CHECK_EQ_U64(block_mod(Block128{0, 1}, 68720050177ULL), block_mod_portable(Block128{0, 1}, 68720050177ULL));
#if defined(__SIZEOF_INT128__)
    CHECK_EQ_U64(block_mod(Block128{0, 1}, 68720050177ULL), static_cast<u64>((static_cast<unsigned __int128>(1) << 64) % 68720050177ULL));
#endif
}

void test_sharing() {
    Timer t("sharing");
    Rng rng = Rng::from_seed(4);
    for (int i = 0; i < 1000; ++i) {
        const u64 x = rng.next_u64();
        const std::pair<u64, u64> s = sharing::share(x, rng);
        CHECK_EQ_U64(sharing::reconstruct(s.first, s.second), x);
        const std::pair<u64, u64> sk = sharing::share_k(x, 8, rng);
        CHECK(sk.first < 256 && sk.second < 256);
        CHECK_EQ_U64(sharing::reconstruct_k(sk.first, sk.second, 8), x & 0xff);
        for (u64 tt : {u64(17), u64(68720050177ULL)}) {
            const u64 v = rng.below(tt);
            const Expected<std::pair<u64, u64>> sm = sharing::share_mod(v, tt, rng);
            CHECK_OK(sm);
            CHECK(sm.value.first < tt && sm.value.second < tt);
            CHECK_EQ_U64(sharing::reconstruct_mod(sm.value.first, sm.value.second, tt), v);
        }
    }
    CHECK_FAILS(sharing::share_mod(17, 17, rng), "SHARING_RANGE");
    CHECK_FAILS(sharing::share_mod(1, 16, rng), "SHARING_MODULUS");
    CHECK_FAILS(sharing::share_mod(1, u64(1) << 62, rng), "SHARING_MODULUS");
    {
        std::vector<u64> xs(100);
        for (u64& v : xs) v = rng.next_u64();
        const auto p = sharing::share(xs, rng);
        const Expected<std::vector<u64>> r = sharing::reconstruct(p.first, p.second);
        CHECK(r.ok && r.value == xs);
        std::vector<u64> xm(100);
        for (u64& v : xm) v = rng.below(17);
        const Expected<std::pair<std::vector<u64>, std::vector<u64>>> pm = sharing::share_mod(xm, 17, rng);
        CHECK_OK(pm);
        const Expected<std::vector<u64>> rm = sharing::reconstruct_mod(pm.value.first, pm.value.second, 17);
        CHECK(rm.ok && rm.value == xm);
        std::vector<uint8_t> bits(100);
        for (uint8_t& b : bits) b = static_cast<uint8_t>(rng.next_u64() & 1u);
        const auto pb = sharing::share_bits(bits, rng);
        const Expected<std::vector<uint8_t>> rb = sharing::reconstruct_bits(pb.first, pb.second);
        CHECK(rb.ok && rb.value == bits);
    }
}

// ---------------------------------------------------------------------------------------------
// DCF
// ---------------------------------------------------------------------------------------------
bool same_key(const dcf::DcfKeyBatch& a, const dcf::DcfKeyBatch& b) {
    return a.n_bits == b.n_bits && a.party == b.party && a.P == b.P && a.modulus == b.modulus && a.seed == b.seed &&
           a.cw_seeds == b.cw_seeds && a.cw_values == b.cw_values && a.cw_tL == b.cw_tL && a.cw_tR == b.cw_tR &&
           a.cw_final == b.cw_final;
}

u64 rand_payload(u64 mod, Rng& rng) { return mod ? rng.below(mod) : rng.next_u64(); }

void dcf_exhaustive_group(u64 mod, std::size_t P, int n_max, Rng& rng) {
    for (int n = 1; n <= n_max; ++n) {
        const std::size_t M = std::size_t(1) << n;
        std::vector<u64> alpha(M), beta(M * P);
        for (std::size_t m = 0; m < M; ++m) {
            alpha[m] = m;
            for (std::size_t p = 0; p < P; ++p) beta[m * P + p] = rand_payload(mod, rng);
        }
        const Expected<std::pair<dcf::DcfKeyBatch, dcf::DcfKeyBatch>> kp = dcf::gen_dcf_batch(alpha, beta, n, rng, mod);
        CHECK_OK(kp);
        const dcf::DcfKeyBatch& k0 = kp.value.first;
        const dcf::DcfKeyBatch& k1 = kp.value.second;
        CHECK_EQ_U64(k0.nbytes(), M * dcf::key_bytes_per_element(n, static_cast<int>(P)));
        for (u64 xv = 0; xv < M; ++xv) {
            const std::vector<u64> x(M, xv);
            const Expected<std::vector<u64>> o0 = dcf::eval_dcf_batch(k0, x);
            const Expected<std::vector<u64>> o1 = dcf::eval_dcf_batch(k1, x);
            CHECK(o0.ok && o1.ok);
            const Expected<std::vector<u64>> rec = dcf::reconstruct_dcf(o0.value, o1.value, mod);
            CHECK_OK(rec);
            for (std::size_t m = 0; m < M; ++m)
                for (std::size_t p = 0; p < P; ++p) {
                    const u64 want = (xv < alpha[m]) ? beta[m * P + p] : 0;
                    CHECK_EQ_U64(rec.value[m * P + p], want);
                    if (mod) CHECK(o0.value[m * P + p] < mod && o1.value[m * P + p] < mod);
                }
        }
        // serialization, both layouts
        for (dcf::WireLayout layout : {dcf::WireLayout::Binding, dcf::WireLayout::PythonV2}) {
            const std::vector<uint8_t> b = k1.to_bytes(layout);
            CHECK_EQ_U64(b.size(), k1.serialized_bytes(layout));
            if (layout == dcf::WireLayout::PythonV2) CHECK_EQ_U64(b.size(), dcf::PY_WIRE_HEADER_BYTES + k1.nbytes());
            const Expected<dcf::DcfKeyBatch> back = dcf::DcfKeyBatch::from_bytes(b);
            CHECK_OK(back);
            CHECK(same_key(back.value, k1));
            std::vector<uint8_t> bad = b;
            bad.pop_back();
            CHECK_FAILS(dcf::DcfKeyBatch::from_bytes(bad), "DCF_WIRE");
            bad = b;
            bad.push_back(0);
            CHECK_FAILS(dcf::DcfKeyBatch::from_bytes(bad), "DCF_WIRE");
        }
        {
            std::vector<uint8_t> bad = k0.to_bytes();
            bad[0] = 3;
            CHECK_FAILS(dcf::DcfKeyBatch::from_bytes(bad), "DCF_WIRE");
            bad = k0.to_bytes(dcf::WireLayout::PythonV2);
            bad[28] = 1;   // flags
            CHECK_FAILS(dcf::DcfKeyBatch::from_bytes(bad), "DCF_WIRE");
        }
        // slice and concat
        if (M >= 2) {
            const std::size_t lo = M / 2;
            const Expected<dcf::DcfKeyBatch> s1 = k0.slice(0, lo);
            const Expected<dcf::DcfKeyBatch> s2 = k0.slice(lo, M);
            CHECK(s1.ok && s2.ok);
            const Expected<dcf::DcfKeyBatch> c = dcf::concat_keys(s1.value, s2.value);
            CHECK_OK(c);
            CHECK(same_key(c.value, k0));
            const std::vector<u64> xs(lo, 1);
            const Expected<std::vector<u64>> os = dcf::eval_dcf_batch(s1.value, xs);
            const Expected<std::vector<u64>> of = dcf::eval_dcf_batch(k0, std::vector<u64>(M, 1));
            CHECK(os.ok && of.ok);
            for (std::size_t i = 0; i < lo * P; ++i) CHECK_EQ_U64(os.value[i], of.value[i]);
            CHECK_FAILS(k0.slice(lo, M + 1), "DCF_SLICE");
        }
    }
}

void test_dcf_exhaustive() {
    Timer t("dcf exhaustive n<=8");
    Rng rng = Rng::from_seed(20);
    // KEY_COMPONENT_INDEPENDENCE, functional correctness for
    // P = 1, 2, 3 in both payload groups with the component-specific Convert words.
    dcf_exhaustive_group(0, 1, 8, rng);                 // Z_{2^64}, one payload word (bit-identical to the pre-repair rule)
    dcf_exhaustive_group(0, 2, 8, rng);                 // Z_{2^64}, two payload words (the DReLU / ReLU shape)
    dcf_exhaustive_group(0, 3, 8, rng);                 // Z_{2^64}, three payload words
    dcf_exhaustive_group(17, 1, 8, rng);                // Z_17
    dcf_exhaustive_group(17, 2, 8, rng);
    dcf_exhaustive_group(17, 3, 8, rng);
    dcf_exhaustive_group(1000003, 2, 6, rng);           // Z_t with a 20-bit prime, two payload words
    dcf_exhaustive_group(68720050177ULL, 1, 8, rng);    // the admitted t
    dcf_exhaustive_group(68720050177ULL, 3, 6, rng);
    dcf_exhaustive_group((u64(1) << 62) - 57, 2, 5, rng);   // the largest admitted width (odd t < 2^62)
    // parameter refusals
    const std::vector<u64> a = {1}, b = {1};
    CHECK_FAILS(dcf::gen_dcf_batch(a, b, 0, rng, 0), "DCF_N_BITS");
    CHECK_FAILS(dcf::gen_dcf_batch(a, b, 65, rng, 0), "DCF_N_BITS");
    CHECK_FAILS(dcf::gen_dcf_batch(a, b, 8, rng, 16), "DCF_MODULUS");
    CHECK_FAILS(dcf::gen_dcf_batch(a, b, 8, rng, u64(1) << 62), "DCF_MODULUS");
    // Odd PRIME only. Composite odd moduli are refused, as in Python (Miller-Rabin, 12 bases).
    for (u64 composite : {u64(9), u64(15), u64(561), u64(206160150531ULL), (u64(1) << 62) - 1, u64(4611686018427387845ULL)}) {
        CHECK(!sharing::is_prime_u64(composite));
        CHECK_FAILS(dcf::check_modulus(composite), "DCF_MODULUS");
        CHECK_FAILS(dcf::gen_dcf_batch(a, b, 8, rng, composite), "DCF_MODULUS");
    }
    for (u64 prime : {u64(3), u64(5), u64(17), u64(68720050177ULL), (u64(1) << 61) - 1, (u64(1) << 62) - 57}) {
        CHECK(sharing::is_prime_u64(prime));
        CHECK_OK(dcf::check_modulus(prime));
    }
    CHECK_OK(dcf::check_modulus(0));   // no modulus: payload group Z_{2^64}
    {
        // Miller-Rabin against trial division below 20000, and on strong pseudoprimes / Carmichael numbers
        for (u64 n = 0; n < 20000; ++n) {
            bool prime = n >= 2;
            for (u64 d = 2; d * d <= n && prime; ++d) prime = (n % d) != 0;
            CHECK(sharing::is_prime_u64(n) == prime);
        }
        for (u64 n : {u64(2047), u64(3215031751ULL), u64(3825123056546413051ULL), u64(341550071728321ULL), u64(1373653), u64(25326001),
                      u64(9007199254740993ULL)})
            CHECK(!sharing::is_prime_u64(n));
        CHECK(sharing::is_prime_u64(18446744073709551557ULL));   // the largest prime below 2^64
        CHECK(!sharing::is_prime_u64(18446744073709551615ULL));
    }
    CHECK_FAILS(dcf::gen_dcf_batch(std::vector<u64>{1, 2}, std::vector<u64>{1, 2, 3}, 8, rng, 0), "DCF_SHAPE");
    CHECK_OK(dcf::gen_dcf_batch(a, std::vector<u64>{1, 2, 3}, 8, rng, 0));   // M = 1, P = 3 is legitimate
    CHECK_FAILS(dcf::gen_dcf_batch(std::vector<u64>{}, std::vector<u64>{}, 8, rng, 0), "DCF_SHAPE");
    const Expected<std::pair<dcf::DcfKeyBatch, dcf::DcfKeyBatch>> kp = dcf::gen_dcf_batch(a, b, 8, rng, 0);
    CHECK_OK(kp);
    CHECK_FAILS(dcf::eval_dcf_batch(kp.value.first, std::vector<u64>{1, 2}), "DCF_SHAPE");
    const Expected<std::pair<dcf::DcfKeyBatch, dcf::DcfKeyBatch>> kq = dcf::gen_dcf_batch(a, b, 8, rng, 17);
    CHECK_FAILS(dcf::concat_keys(kp.value.first, kq.value.first), "DCF_CONCAT");
}

void dcf_random_group(int n, u64 mod, std::size_t P, std::size_t M, Rng& rng) {
    const u64 dmask = n >= 64 ? ring::MASK64 : ((u64(1) << n) - 1);
    std::vector<u64> alpha(M), beta(M * P), x(M);
    for (std::size_t m = 0; m < M; ++m) {
        alpha[m] = rng.next_u64() & dmask;
        for (std::size_t p = 0; p < P; ++p) beta[m * P + p] = rand_payload(mod, rng);
        switch (m % 6) {
            case 0: x[m] = rng.next_u64() & dmask; break;
            case 1: x[m] = alpha[m]; break;
            case 2: x[m] = (alpha[m] - 1) & dmask; break;
            case 3: x[m] = (alpha[m] + 1) & dmask; break;
            case 4: x[m] = 0; break;
            default: x[m] = dmask; break;
        }
    }
    const Expected<std::pair<dcf::DcfKeyBatch, dcf::DcfKeyBatch>> kp = dcf::gen_dcf_batch(alpha, beta, n, rng, mod);
    CHECK_OK(kp);
    const Expected<std::vector<u64>> o0 = dcf::eval_dcf_batch(kp.value.first, x);
    const Expected<std::vector<u64>> o1 = dcf::eval_dcf_batch(kp.value.second, x);
    CHECK(o0.ok && o1.ok);
    const Expected<std::vector<u64>> rec = dcf::reconstruct_dcf(o0.value, o1.value, mod);
    CHECK_OK(rec);
    for (std::size_t m = 0; m < M; ++m)
        for (std::size_t p = 0; p < P; ++p) CHECK_EQ_U64(rec.value[m * P + p], (x[m] < alpha[m]) ? beta[m * P + p] : 0);
    // out-of-domain inputs are reduced mod 2^n (n < 64): the same answer as the reduced input
    if (n < 64) {
        std::vector<u64> xh(M);
        for (std::size_t m = 0; m < M; ++m) xh[m] = x[m] | (u64(1) << n);
        const Expected<std::vector<u64>> oh = dcf::eval_dcf_batch(kp.value.first, xh);
        CHECK(oh.ok && oh.value == o0.value);
    }
}

void test_dcf_random() {
    Timer t("dcf random 32/64");
    Rng rng = Rng::from_seed(21);
    // P = 1, 2, 3 at 32 and 64 bits in both payload groups (KEY_COMPONENT_INDEPENDENCE check 3)
    dcf_random_group(32, 0, 1, 4096, rng);
    dcf_random_group(32, 0, 2, 4096, rng);
    dcf_random_group(32, 0, 3, 2048, rng);
    dcf_random_group(32, 68720050177ULL, 1, 4096, rng);
    dcf_random_group(32, 68720050177ULL, 2, 2048, rng);
    dcf_random_group(32, 68720050177ULL, 3, 2048, rng);
    dcf_random_group(64, 0, 1, 4096, rng);
    dcf_random_group(64, 0, 2, 2048, rng);
    dcf_random_group(64, 0, 3, 2048, rng);
    dcf_random_group(64, 68720050177ULL, 1, 4096, rng);
    dcf_random_group(64, 68720050177ULL, 2, 2048, rng);
    dcf_random_group(64, 17, 2, 2048, rng);
    dcf_random_group(64, 17, 3, 2048, rng);
    dcf_random_group(64, (u64(1) << 62) - 57, 3, 1024, rng);
}

// ---------------------------------------------------------------------------------------------
// KEY_COMPONENT_INDEPENDENCE (the comparison-key finding: paper, Section 5.3 and Appendix D, Disclosure)
// ---------------------------------------------------------------------------------------------
// The pre-repair rule (of the superseded engine) derived ONE pseudo-random value word per
// level and reused it for every payload component. This is a test-only replica of that key
// generation for one key (Z_{2^64}); it returns the public value correction words cw[i*P + p].
// It exists only as the positive control of the attack: a regression test that reports zero
// recoveries must also show that the same attack code recovers everything on vulnerable keys.
std::vector<u64> legacy_shared_word_cw_values(u64 alpha, const std::vector<u64>& beta, int n_bits, Rng& rng) {
    const std::size_t n = static_cast<std::size_t>(n_bits), P = beta.size();
    Block128 s[2] = {Block128{rng.next_u64(), rng.next_u64()}, Block128{rng.next_u64(), rng.next_u64()}};
    u64 t[2] = {0, 1};
    std::vector<u64> v_alpha(P, 0), cw(n * P);
    for (std::size_t i = 0; i < n; ++i) {
        const u64 a_i = (alpha >> (n - 1 - i)) & 1u;
        const bool keep_L = (a_i == 0);
        const Expansion E0 = expand(s[0]);
        const Expansion E1 = expand(s[1]);
        const Block128 s_cw = (keep_L ? E0.sR : E0.sL) ^ (keep_L ? E1.sR : E1.sL);
        const u64 v_lose0 = keep_L ? E0.vR : E0.vL, v_lose1 = keep_L ? E1.vR : E1.vL;   // one word for every component
        const u64 v_keep0 = keep_L ? E0.vL : E0.vR, v_keep1 = keep_L ? E1.vL : E1.vR;
        const u64 sign1 = t[1];
        for (std::size_t p = 0; p < P; ++p) {
            u64 vc = (v_lose1 - v_lose0) - v_alpha[p] + (keep_L ? 0 : beta[p]);
            if (sign1) vc = u64(0) - vc;
            cw[i * P + p] = vc;
            v_alpha[p] = v_alpha[p] - v_keep1 + v_keep0 + (sign1 ? u64(0) - vc : vc);
        }
        const u64 t_cw_L = E0.tL ^ E1.tL ^ a_i ^ 1u, t_cw_R = E0.tR ^ E1.tR ^ a_i;
        const u64 t_cw_keep = keep_L ? t_cw_L : t_cw_R;
        const Expansion* E[2] = {&E0, &E1};
        for (int b = 0; b < 2; ++b) {
            Block128 s_keep = keep_L ? E[b]->sL : E[b]->sR;
            u64 t_keep = keep_L ? E[b]->tL : E[b]->tR;
            if (t[b]) {
                s_keep = s_keep ^ s_cw;
                t_keep ^= t_cw_keep;
            }
            s[b] = s_keep;
            t[b] = t_keep;
        }
    }
    return cw;
}

struct KciCounts {
    std::size_t blind_recovered = 0;       // check 1: r in {1 - d, 1 + d}, d = first nonzero difference
    std::size_t level_equals_delta = 0;    // check 2: some level has cw[i][1] - cw[i][0] = +-(beta_1 - beta_0)
    std::size_t alpha_bits_read = 0;       // every bit of alpha read from |d_i| pattern (legacy keys only)
};

// The blind attack on the PUBLIC part of one P = 2 key (payload (1, r), group Z_{2^64} or Z_t): a
// party looks only at its own correction words. cw(i, p) returns cw_values[i][p] of the key.
template <class CwAt>
void kci_attack_one(CwAt cw, std::size_t n, u64 alpha, u64 beta0, u64 beta1, u64 modulus, KciCounts& counts) {
    auto sub = [modulus](u64 a, u64 b) { return modulus ? (a + modulus - b) % modulus : a - b; };
    auto add = [modulus](u64 a, u64 b) { return modulus ? (a + b) % modulus : a + b; };
    const u64 delta = sub(beta1, beta0);
    const u64 neg_delta = sub(0, delta);
    u64 d_first = 0;
    bool hit = false;
    // With a shared word, d_i = +-(a_i - a_{i-1}) delta: nonzero exactly where the bit of alpha flips.
    u64 prev_bit = 0, alpha_guess = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const u64 d = sub(cw(i, 1), cw(i, 0));
        if (d != 0 && d_first == 0) d_first = d;
        if (d == delta || d == neg_delta) hit = true;
        const u64 bit = prev_bit ^ (d != 0 ? 1u : 0u);
        alpha_guess = (alpha_guess << 1) | bit;
        prev_bit = bit;
    }
    if (d_first != 0 && (beta1 == add(beta0, d_first) || beta1 == sub(beta0, d_first))) ++counts.blind_recovered;
    if (hit) ++counts.level_equals_delta;
    if (alpha_guess == alpha) ++counts.alpha_bits_read;
}

void test_key_component_independence() {
    Timer t("KEY_COMPONENT_INDEPENDENCE");
    const std::size_t T = 2000;
    Rng rng = Rng::from_seed(0x4b4349ULL);

    // Positive control: on keys of the pre-repair rule the attack reads the DReLU mask r
    // (beta = (1, r), alpha = r mod 2^n) and every bit of alpha in 2000 of 2000 trials.
    for (int n : {32, 64}) {
        KciCounts legacy;
        for (std::size_t trial = 0; trial < T; ++trial) {
            const u64 r = rng.next_u64();
            const u64 alpha = ring::reduce_k(r, n);
            const std::vector<u64> cw = legacy_shared_word_cw_values(alpha, std::vector<u64>{1, r}, n, rng);
            kci_attack_one([&](std::size_t i, std::size_t p) { return cw[i * 2 + p]; }, static_cast<std::size_t>(n), alpha, 1, r, 0, legacy);
        }
        std::fprintf(stderr, "[KEY_COMPONENT_INDEPENDENCE] legacy shared-word keys n=%d: blind mask recovery %zu/%zu, level = +-delta %zu/%zu, alpha read %zu/%zu\n",
                     n, legacy.blind_recovered, T, legacy.level_equals_delta, T, legacy.alpha_bits_read, T);
        CHECK_EQ_U64(legacy.blind_recovered, T);
        CHECK_EQ_U64(legacy.level_equals_delta, T);
        CHECK_EQ_U64(legacy.alpha_bits_read, T);
    }

    // Checks 1 and 2 on the shipped DReLU material (P = 2), both keys of both parties: 0 of 2000.
    for (int n : {32, 64}) {
        const auto mats = drelu::drelu_material(T, n, rng, true);
        CHECK_OK(mats);
        const u64 half = u64(1) << (n - 1);
        for (int party = 0; party < 2; ++party) {
            const drelu::DreluMaterial& mat = party == 0 ? mats.value.first : mats.value.second;
            for (int which = 0; which < 2; ++which) {
                const dcf::DcfKeyBatch& key = which == 0 ? mat.key_p : mat.key_q;
                CHECK(key.P == 2 && key.modulus == 0 && key.M() == T);
                KciCounts c;
                for (std::size_t m = 0; m < T; ++m) {
                    const u64 r = mats.value.first.r_share[m] + mats.value.second.r_share[m];
                    const u64 alpha = ring::reduce_k(which == 0 ? r : r + half, n);
                    kci_attack_one([&](std::size_t i, std::size_t p) { return key.cw_values[(i * 2 + p) * T + m]; },
                                   static_cast<std::size_t>(n), alpha, 1, r, 0, c);
                }
                std::fprintf(stderr, "[KEY_COMPONENT_INDEPENDENCE] repaired DReLU keys n=%d party=%d key_%c: blind mask recovery %zu/%zu, level = +-delta %zu/%zu\n",
                             n, party, which == 0 ? 'p' : 'q', c.blind_recovered, T, c.level_equals_delta, T);
                CHECK_EQ_U64(c.blind_recovered, 0);       // check 1
                CHECK_EQ_U64(c.level_equals_delta, 0);    // check 2
                CHECK_EQ_U64(c.alpha_bits_read, 0);
            }
        }
    }

    // The same two checks in the Z_t payload group (P = 2, the admitted t), where S2 also applies.
    {
        const u64 tt = 68720050177ULL;
        std::vector<u64> alpha(T), beta(2 * T);
        for (std::size_t m = 0; m < T; ++m) {
            alpha[m] = rng.next_u64();
            beta[2 * m] = 1;
            beta[2 * m + 1] = 2 + rng.below(tt - 2);   // beta_1 - beta_0 is nonzero mod t
        }
        const auto kp = dcf::gen_dcf_batch(alpha, beta, 64, rng, tt);
        CHECK_OK(kp);
        KciCounts c;
        for (std::size_t m = 0; m < T; ++m)
            kci_attack_one([&](std::size_t i, std::size_t p) { return kp.value.first.cw_values[(i * 2 + p) * T + m]; }, 64, alpha[m], 1,
                           beta[2 * m + 1], tt, c);
        std::fprintf(stderr, "[KEY_COMPONENT_INDEPENDENCE] repaired Z_t keys (t=68720050177, P=2): blind recovery %zu/%zu, level = +-delta %zu/%zu\n",
                     c.blind_recovered, T, c.level_equals_delta, T);
        CHECK_EQ_U64(c.blind_recovered, 0);
        CHECK_EQ_U64(c.level_equals_delta, 0);
        CHECK_EQ_U64(c.alpha_bits_read, 0);
    }

    // Structural form of the same property: with EQUAL payload components a shared word makes the
    // components' correction words identical at every level; component-specific words never do.
    for (u64 mod : {u64(0), u64(68720050177ULL)}) {
        const std::size_t M = 64, P = 3;
        std::vector<u64> alpha(M), beta(M * P);
        for (std::size_t m = 0; m < M; ++m) {
            alpha[m] = rng.next_u64();
            const u64 b = rand_payload(mod, rng);
            for (std::size_t p = 0; p < P; ++p) beta[m * P + p] = b;
        }
        const auto kp = dcf::gen_dcf_batch(alpha, beta, 64, rng, mod);
        CHECK_OK(kp);
        std::size_t equal_levels = 0, equal_final = 0;
        for (std::size_t m = 0; m < M; ++m) {
            for (std::size_t i = 0; i < 64; ++i)
                for (std::size_t p = 1; p < P; ++p)
                    if (kp.value.first.cw_values[(i * P + p) * M + m] == kp.value.first.cw_values[(i * P) * M + m]) ++equal_levels;
            for (std::size_t p = 1; p < P; ++p)
                if (kp.value.first.cw_final[p * M + m] == kp.value.first.cw_final[m]) ++equal_final;
        }
        CHECK_EQ_U64(equal_levels, 0);
        CHECK_EQ_U64(equal_final, 0);
        // and the keys still evaluate to the same payload in every component
        std::vector<u64> x(M);
        for (std::size_t m = 0; m < M; ++m) x[m] = (m & 1) ? alpha[m] - 1 : alpha[m];
        const auto o0 = dcf::eval_dcf_batch(kp.value.first, x);
        const auto o1 = dcf::eval_dcf_batch(kp.value.second, x);
        CHECK(o0.ok && o1.ok);
        const auto rec = dcf::reconstruct_dcf(o0.value, o1.value, mod);
        CHECK_OK(rec);
        for (std::size_t m = 0; m < M; ++m)
            for (std::size_t p = 0; p < P; ++p) CHECK_EQ_U64(rec.value[m * P + p], (x[m] < alpha[m]) ? beta[m * P + p] : 0);
    }

    // Check 4: P = 1 keys and outputs over Z_{2^64} are bit-identical to those of the pre-repair rule.
    // The digests and shares below were produced by the pre-repair build on 2026-10-10 (same seeded
    // ChaCha20 stream); the Convert rule must never change them.
    {
        Rng pr = Rng::from_seed(0x50494e31ULL);
        const std::vector<u64> alpha = {77, 200}, beta = {1, 0x0123456789abcdefULL}, x = {76, 200};
        const auto kp = dcf::gen_dcf_batch(alpha, beta, 8, pr, 0);
        CHECK_OK(kp);
        const std::vector<uint8_t> b0 = kp.value.first.to_bytes(), b1 = kp.value.second.to_bytes();
        CHECK_EQ_U64(b0.size(), 708);
        CHECK(to_hex(Sha256::digest(b0.data(), b0.size())) == "6d74808c5c8137f02223584d7c45a979546d07f9853df714a47801e78a04c986");
        CHECK(to_hex(Sha256::digest(b1.data(), b1.size())) == "5c0eff321f09f5e026a1e75f06b5115dacee70c228d9fb5a7980b9d5d4e9902f");
        CHECK(to_hex(b0.data() + 20, 16) == "946f71bb67e4a9e689613eb547a477f7");     // P0 seed of key 0
        CHECK_EQ_U64(kp.value.first.cw_final[0], 0xdd7b12e426bd1d98ULL);
        CHECK_EQ_U64(kp.value.first.cw_final[1], 0x3dff40118625982fULL);
        const auto o0 = dcf::eval_dcf_batch(kp.value.first, x);
        const auto o1 = dcf::eval_dcf_batch(kp.value.second, x);
        CHECK(o0.ok && o1.ok);
        CHECK_EQ_U64(o0.value[0], 3849884810853995144ULL);
        CHECK_EQ_U64(o0.value[1], 14521926565186295988ULL);
        CHECK_EQ_U64(o1.value[0], 14596859262855556473ULL);
        CHECK_EQ_U64(o1.value[1], 3924817508523255628ULL);
        CHECK_EQ_U64(o0.value[0] + o1.value[0], 1);   // 76 < 77
        CHECK_EQ_U64(o0.value[1] + o1.value[1], 0);   // 200 < 200 is false
        std::vector<u64> a2(16), b2(16), x2(16);
        for (int i = 0; i < 16; ++i) {
            a2[i] = pr.next_u64();
            b2[i] = pr.next_u64();
            x2[i] = (i & 1) ? a2[i] - 1 : pr.next_u64();
        }
        const auto k2 = dcf::gen_dcf_batch(a2, b2, 64, pr, 0);
        CHECK_OK(k2);
        const auto p0 = dcf::eval_dcf_batch(k2.value.first, x2);
        const auto p1 = dcf::eval_dcf_batch(k2.value.second, x2);
        CHECK(p0.ok && p1.ok);
        std::vector<uint8_t> ob;
        for (u64 v : p0.value) dcf::wire::put_u64le(ob, v);
        for (u64 v : p1.value) dcf::wire::put_u64le(ob, v);
        const std::vector<uint8_t> c0 = k2.value.first.to_bytes(), c1 = k2.value.second.to_bytes();
        CHECK(to_hex(Sha256::digest(c0.data(), c0.size())) == "8773feb79977f97267b9623f1c3fd9755fe7e79aef253532c1068b1f69c962e5");
        CHECK(to_hex(Sha256::digest(c1.data(), c1.size())) == "3dd4e3611523c5dbfa5d3dd00fd016fdbc50079f5bf6ac52df80e5780e1f6596");
        CHECK(to_hex(Sha256::digest(ob.data(), ob.size())) == "0be592c0266e0cb23a7f953bb8f241a02fe66b78e6375deb1aba99ff649002c0");
    }

    // Key sizes are unchanged by the repair.
    CHECK_EQ_U64(dcf::key_bytes_per_element(32, 2), 1088);
    CHECK_EQ_U64(dcf::key_bytes_per_element(64, 1), 1624);
    CHECK_EQ_U64(drelu::drelu_bytes_per_element(32, true), 2200);
}

// ---------------------------------------------------------------------------------------------
// DReLU
// ---------------------------------------------------------------------------------------------
bool same_drelu(const drelu::DreluMaterial& a, const drelu::DreluMaterial& b) {
    return a.n_bits == b.n_bits && a.party == b.party && a.r_share == b.r_share && a.c_b == b.c_b && a.c_rb == b.c_rb &&
           same_key(a.key_p, b.key_p) && same_key(a.key_q, b.key_q);
}

void drelu_check_batch(const drelu::DreluMaterial& m0, const drelu::DreluMaterial& m1, const std::vector<i64>& x, Rng& rng) {
    const std::size_t M = x.size();
    std::vector<u64> y(M), r(M);
    std::vector<u64> xu(M);
    for (std::size_t m = 0; m < M; ++m) xu[m] = ring::from_signed(x[m]);
    const auto xs = sharing::share(xu, rng);
    for (std::size_t m = 0; m < M; ++m) {
        r[m] = m0.r_share[m] + m1.r_share[m];
        y[m] = (xs.first[m] + drelu::drelu_mask_share(m0)[m]) + (xs.second[m] + drelu::drelu_mask_share(m1)[m]);
    }
    const Expected<drelu::DreluOutput> o0 = drelu::drelu_eval(m0, y);
    const Expected<drelu::DreluOutput> o1 = drelu::drelu_eval(m1, y);
    CHECK(o0.ok && o1.ok);
    const bool wp = m0.with_product();
    CHECK((o0.value.rb.size() == M) == wp);
    for (std::size_t m = 0; m < M; ++m) {
        const u64 b = o0.value.b[m] + o1.value.b[m];
        CHECK_EQ_U64(b, x[m] >= 0 ? 1 : 0);
        if (wp) {
            CHECK_EQ_U64(o0.value.rb[m] + o1.value.rb[m], r[m] * b);
        }
    }
    if (wp) {
        const Expected<std::vector<u64>> z0 = drelu::relu_from_drelu(y, o0.value.b, o0.value.rb);
        const Expected<std::vector<u64>> z1 = drelu::relu_from_drelu(y, o1.value.b, o1.value.rb);
        CHECK(z0.ok && z1.ok);
        for (std::size_t m = 0; m < M; ++m) CHECK_EQ_I64(ring::to_signed(z0.value[m] + z1.value[m]), x[m] >= 0 ? x[m] : 0);
    }
}

void test_drelu() {
    Timer t("drelu");
    Rng rng = Rng::from_seed(30);
    // exhaustive over (x, r mod 2^8) at n = 8
    for (bool wp : {true, false}) {
        std::vector<u64> r(256);
        for (std::size_t m = 0; m < 256; ++m) r[m] = (rng.next_u64() & ~u64(0xff)) | m;
        const auto mats = drelu::drelu_material_from_mask(r, 8, rng, wp);
        CHECK_OK(mats);
        for (i64 xv = -128; xv < 128; ++xv) drelu_check_batch(mats.value.first, mats.value.second, std::vector<i64>(256, xv), rng);
    }
    // random at n in {16, 32, 64}
    for (int n : {16, 32, 64}) {
        for (bool wp : {true, false}) {
            const std::size_t M = 4096;
            const auto mats = drelu::drelu_material(M, n, rng, wp);
            CHECK_OK(mats);
            std::vector<i64> x(M);
            for (std::size_t m = 0; m < M; ++m) {
                if (n == 64) {
                    x[m] = ring::to_signed(rng.next_u64());
                } else {
                    const u64 v = rng.below(u64(1) << n);
                    x[m] = static_cast<i64>(v) - static_cast<i64>(u64(1) << (n - 1));
                }
            }
            x[0] = 0;
            x[1] = -1;
            x[2] = 1;
            x[3] = (n == 64) ? INT64_MAX : static_cast<i64>((u64(1) << (n - 1)) - 1);
            x[4] = (n == 64) ? INT64_MIN : -static_cast<i64>(u64(1) << (n - 1));
            drelu_check_batch(mats.value.first, mats.value.second, x, rng);
            CHECK_EQ_U64(mats.value.first.nbytes(), M * drelu::drelu_bytes_per_element(n, wp));
            for (dcf::WireLayout layout : {dcf::WireLayout::Binding, dcf::WireLayout::PythonV2}) {
                const std::vector<uint8_t> b = mats.value.first.to_bytes(layout);
                const Expected<drelu::DreluMaterial> back = drelu::DreluMaterial::from_bytes(b);
                CHECK_OK(back);
                CHECK(same_drelu(back.value, mats.value.first));
                std::vector<uint8_t> bad = b;
                bad.pop_back();
                CHECK_FAILS(drelu::DreluMaterial::from_bytes(bad), "D");   // DRELU_WIRE or DCF_WIRE
            }
            const Expected<drelu::DreluMaterial> s1 = mats.value.second.slice(0, 100);
            const Expected<drelu::DreluMaterial> s2 = mats.value.second.slice(100, M);
            CHECK(s1.ok && s2.ok);
            const Expected<drelu::DreluMaterial> c = drelu::concat_drelu(s1.value, s2.value);
            CHECK_OK(c);
            CHECK(same_drelu(c.value, mats.value.second));
        }
    }
    CHECK_EQ_U64(drelu::drelu_bytes_per_element(32, true), 2200);
    CHECK_EQ_U64(drelu::drelu_bytes_per_element(32, false), 2 * 824 + 24);   // key(32,1) = 16 + 32*25 + 8 = 824
    CHECK_FAILS(drelu::drelu_material(4, 1, rng, true), "DRELU_N_BITS");
    CHECK_FAILS(drelu::drelu_material(4, 65, rng, true), "DRELU_N_BITS");
}

// ---------------------------------------------------------------------------------------------
// Faithful truncation
// ---------------------------------------------------------------------------------------------
bool same_trunc(const trunc::TruncMaterial& a, const trunc::TruncMaterial& b) {
    return a.f == b.f && a.L == b.L && a.party == b.party && a.r_share == b.r_share && a.r_high_share == b.r_high_share &&
           same_key(a.key_wrap, b.key_wrap) && same_key(a.key_borrow, b.key_borrow);
}

// Runs the gate on M elements with the given x values and checks floor(x / 2^f). Thread-safe:
// counts into `res` and reports the first failure there (no global state).
bool trunc_gate_run(const trunc::TruncMaterial& m0, const trunc::TruncMaterial& m1, const std::vector<i64>& x, Rng& rng,
                    bool compare_reference, JobResult& res) {
    const std::size_t M = x.size();
    const int k = m0.k();
    const int f = m0.f;
    const int L = m0.L;
    const std::string where = "k=" + std::to_string(k) + " L=" + std::to_string(L) + " f=" + std::to_string(f);
    std::vector<u64> x0(M), x1(M);
    for (std::size_t m = 0; m < M; ++m) {
        x0[m] = ring::reduce_k(rng.next_u64(), k);
        x1[m] = ring::reduce_k(ring::from_signed(x[m]) - x0[m], k);
    }
    const Expected<std::vector<u64>> msg0 = trunc::trunc_open_share(x0, m0, 0);
    const Expected<std::vector<u64>> msg1 = trunc::trunc_open_share(x1, m1, 1);
    if (!msg0.ok || !msg1.ok) {
        res.failure = "trunc_open_share failed (" + where + "): " + msg0.code + msg1.code;
        return false;
    }
    ++res.checks;
    std::vector<u64> y(M);
    for (std::size_t m = 0; m < M; ++m) y[m] = ring::reduce_k(msg0.value[m] + msg1.value[m], k);
    const Expected<std::vector<u64>> z0 = trunc::trunc_faithful_eval(m0, y, 0);
    const Expected<std::vector<u64>> z1 = trunc::trunc_faithful_eval(m1, y, 1);
    if (!z0.ok || !z1.ok) {
        res.failure = "trunc_faithful_eval failed (" + where + "): " + z0.code + z1.code;
        return false;
    }
    ++res.checks;
    for (std::size_t m = 0; m < M; ++m) {
        const i64 z = ring::to_signed_k(z0.value[m] + z1.value[m], k);
        const i64 want = floor_div_pow2(x[m], f);
        if (z != want) {
            res.failure = "trunc gate " + where + " x=" + std::to_string(x[m]) + " got " + std::to_string(z) + " want " + std::to_string(want);
            return false;
        }
        ++res.checks;
        if (compare_reference) {
            const u64 r = ring::reduce_k(m0.r_share[m] + m1.r_share[m], k);
            const Expected<i64> ref = trunc::reference_trunc_small(k, L, f, x[m], r);
            if (!ref.ok || ref.value != want) {
                res.failure = "reference_trunc_small " + where + " x=" + std::to_string(x[m]) + " r=" + std::to_string(r) + " " + ref.code;
                return false;
            }
            res.checks += 2;
        }
    }
    return true;
}

void trunc_gate_check(const trunc::TruncMaterial& m0, const trunc::TruncMaterial& m1, const std::vector<i64>& x, Rng& rng,
                      bool compare_reference) {
    JobResult res;
    if (!trunc_gate_run(m0, m1, x, rng, compare_reference, res)) fail_at("trunc gate", __FILE__, __LINE__, res.failure);
    g_checks += res.checks;
}

void test_trunc() {
    Timer t("trunc");
    Rng rng = Rng::from_seed(40);
    // the signed-offset regression of trunc.hpp: (k=8, L=6, f=3, x=-13, r=37)
    CHECK_EQ_I64(trunc::reference_trunc_small(8, 6, 3, -13, 37).value, -2);
    CHECK_EQ_I64(trunc::reference_trunc_small_no_offset_term(8, 6, 3, -13, 37).value, 2);
    {
        const auto mats = trunc::trunc_material_from_mask(std::vector<u64>{37}, 3, 6, rng, 8);
        CHECK_OK(mats);
        trunc_gate_check(mats.value.first, mats.value.second, std::vector<i64>{-13}, rng, true);
    }
    CHECK_FAILS(trunc::check_trunc_params(0, 6, 8), "TRUNC_PARAMS");
    CHECK_FAILS(trunc::check_trunc_params(6, 6, 8), "TRUNC_PARAMS");
    CHECK_FAILS(trunc::check_trunc_params(3, 9, 8), "TRUNC_PARAMS");
    CHECK_FAILS(trunc::reference_trunc_small(8, 6, 3, 32, 0), "TRUNC_RANGE");
    CHECK_FAILS(trunc::trunc_material(4, 16, 65, rng), "TRUNC_PARAMS");
    CHECK_FAILS(trunc::trunc_material(4, 0, 40, rng), "TRUNC_PARAMS");
    CHECK_EQ_U64(trunc::trunc_bytes_per_element(16), 2064);
}

// Exhaustive small ring of the faithful truncation gate: every (k <= 10, L, f, x, r) through the real
// gate (real DCF keys, random share split of x) against floor(x / 2^f) and the integer reference.
// One job per (k, L, f); jobs run on a thread pool because k = 10 alone is 16.8 million gate
// evaluations. Every job owns a seeded generator, so the run is deterministic for any thread count.
void test_trunc_exhaustive() {
    Timer t("trunc exhaustive k<=10");
    struct Cfg {
        int k, L, f;
    };
    std::vector<Cfg> cfgs;
    for (int k = 10; k >= 2; --k)            // largest jobs first
        for (int L = k; L >= 2; --L)
            for (int f = 1; f <= L - 1; ++f) cfgs.push_back(Cfg{k, L, f});
    run_parallel(
        cfgs.size(),
        [&](std::size_t j, JobResult& res) {
            const Cfg c = cfgs[j];
            Rng rng = Rng::from_seed(0x7472756e63ULL + 1000003ULL * static_cast<u64>(c.k) + 1009ULL * static_cast<u64>(c.L) + static_cast<u64>(c.f));
            std::vector<u64> r(std::size_t(1) << c.k);
            for (std::size_t m = 0; m < r.size(); ++m) r[m] = m;   // every mask of the ring
            const auto mats = trunc::trunc_material_from_mask(r, c.f, c.L, rng, c.k);
            if (!mats.ok || mats.value.first.k() != c.k) {
                res.failure = "trunc_material_from_mask k=" + std::to_string(c.k) + ": " + mats.code;
                return;
            }
            res.checks += 2;
            const i64 a = static_cast<i64>(u64(1) << (c.L - 1));
            for (i64 xv = -a; xv < a; ++xv)   // every admitted input
                if (!trunc_gate_run(mats.value.first, mats.value.second, std::vector<i64>(r.size(), xv), rng, true, res)) return;
        },
        "trunc exhaustive");
    CHECK(cfgs.size() == 165);   // sum over k = 2..10 of k (k - 1) / 2 configurations
}

void test_trunc_random() {
    Timer t("trunc random 64-bit");
    Rng rng = Rng::from_seed(42);
    // 10^6 random 64-bit elements across magnitudes 2^20 .. 2^(L-1), in
    // 100 chunks of 10^4 on the thread pool (one chunk of material is about 65 MB in memory).
    {
        const std::size_t chunks = 100, per_chunk = 10000;
        const int L = 40, f = 16;
        run_parallel(
            chunks,
            [&](std::size_t j, JobResult& res) {
                Rng crng = Rng::from_seed(0x72616e64ULL + 7919ULL * static_cast<u64>(j));
                const auto mats = trunc::trunc_material(per_chunk, f, L, crng);
                if (!mats.ok || mats.value.first.nbytes() != per_chunk * 2064) {
                    res.failure = "trunc_material chunk " + std::to_string(j) + ": " + mats.code;
                    return;
                }
                res.checks += 2;
                std::vector<i64> x(per_chunk);
                for (std::size_t m = 0; m < per_chunk; ++m) {
                    const int bits = 20 + static_cast<int>(m % 20);   // magnitudes 2^20 .. 2^39
                    const u64 mag = crng.below(u64(1) << bits);
                    x[m] = (crng.next_u64() & 1u) ? -static_cast<i64>(mag) : static_cast<i64>(mag);
                }
                const i64 top = static_cast<i64>((u64(1) << (L - 1)) - 1);
                x[0] = top;                                   // boundary values +-(2^(L-1) - 1) and sign changes
                x[1] = -top - 1;
                x[2] = 0;
                x[3] = -1;
                x[4] = 1;
                x[5] = -top;
                x[6] = static_cast<i64>(u64(1) << f) - 1;
                x[7] = -static_cast<i64>(u64(1) << f);
                trunc_gate_run(mats.value.first, mats.value.second, x, crng, true, res);
            },
            "trunc random 10^6");
    }
    {
        const std::size_t M = 20000;
        const int L = 40, f = 16;
        const auto mats = trunc::trunc_material(M, f, L, rng);
        CHECK_OK(mats);
        CHECK_EQ_U64(mats.value.first.nbytes(), M * 2064);
        std::vector<i64> x(M);
        for (std::size_t m = 0; m < M; ++m) {
            const int bits = 20 + static_cast<int>(m % 20);   // magnitudes 2^20 .. 2^39
            const u64 mag = rng.below(u64(1) << bits);
            x[m] = (rng.next_u64() & 1u) ? -static_cast<i64>(mag) : static_cast<i64>(mag);
        }
        trunc_gate_check(mats.value.first, mats.value.second, x, rng, true);
        for (dcf::WireLayout layout : {dcf::WireLayout::Binding, dcf::WireLayout::PythonV2}) {
            const Expected<trunc::TruncMaterial> s = mats.value.second.slice(0, 64);
            CHECK_OK(s);
            const std::vector<uint8_t> b = s.value.to_bytes(layout);
            const Expected<trunc::TruncMaterial> back = trunc::TruncMaterial::from_bytes(b);
            CHECK_OK(back);
            CHECK(same_trunc(back.value, s.value));
            std::vector<uint8_t> bad = b;
            bad[8] = 0;   // f = 0
            CHECK_FAILS(trunc::TruncMaterial::from_bytes(bad), "TRUNC_");
        }
        const Expected<trunc::TruncMaterial> s1 = mats.value.first.slice(0, 10);
        const Expected<trunc::TruncMaterial> s2 = mats.value.first.slice(10, 30);
        const Expected<trunc::TruncMaterial> c = trunc::concat_trunc(s1.value, s2.value);
        CHECK_OK(c);
        CHECK(same_trunc(c.value, mats.value.first.slice(0, 30).value));
    }
    // full-width logical range L = 64 and the SmolLM contract L = 46
    for (int L : {64, 46}) {
        const std::size_t M = 10000;
        const auto mats = trunc::trunc_material(M, 16, L, rng);
        CHECK_OK(mats);
        std::vector<i64> x(M);
        for (std::size_t m = 0; m < M; ++m) {
            if (L == 64) {
                x[m] = ring::to_signed(rng.next_u64());
            } else {
                const u64 mag = rng.below(u64(1) << (L - 1));
                x[m] = (rng.next_u64() & 1u) ? -static_cast<i64>(mag) : static_cast<i64>(mag);
            }
        }
        x[0] = (L == 64) ? INT64_MAX : static_cast<i64>((u64(1) << (L - 1)) - 1);
        x[1] = (L == 64) ? INT64_MIN : -static_cast<i64>(u64(1) << (L - 1));
        trunc_gate_check(mats.value.first, mats.value.second, x, rng, true);
    }
    // SecureML local truncation on contract-bounded inputs: within one unit
    {
        int off_by_one = 0;
        for (int i = 0; i < 10000; ++i) {
            const u64 mag = rng.below(u64(1) << 39);
            const i64 x = (rng.next_u64() & 1u) ? -static_cast<i64>(mag) : static_cast<i64>(mag);
            const std::pair<u64, u64> s = sharing::share(ring::from_signed(x), rng);
            const Expected<u64> t0 = trunc::trunc_local(s.first, 16, 0);
            const Expected<u64> t1 = trunc::trunc_local(s.second, 16, 1);
            CHECK(t0.ok && t1.ok);
            const i64 z = ring::to_signed(t0.value + t1.value);
            const i64 want = floor_div_pow2(x, 16);
            CHECK(z == want || z == want + 1 || z == want - 1);
            if (z != want) ++off_by_one;
        }
        CHECK(off_by_one > 0);   // the probabilistic gate does round differently sometimes
        CHECK_FAILS(trunc::trunc_local(u64(5), 64, 0), "TRUNC_PARAMS");
        CHECK_EQ_U64(trunc::trunc_local(u64(5), 0, 1).value, 5);
    }
}

// ---------------------------------------------------------------------------------------------
// Ring to field
// ---------------------------------------------------------------------------------------------
bool same_r2f(const r2f::R2FMaterial& a, const r2f::R2FMaterial& b) {
    return a.t == b.t && a.party == b.party && a.r_share == b.r_share && a.r_mod_t_share == b.r_mod_t_share &&
           a.c_share == b.c_share && same_key(a.key_A, b.key_A) && same_key(a.key_C, b.key_C);
}

// Runs the conversion on M elements given u values and P0's shares a0; checks z = x_signed mod t.
// Thread-safe: counts into `res` and reports the first failure there.
bool r2f_gate_run(const r2f::R2FMaterial& m0, const r2f::R2FMaterial& m1, const std::vector<u64>& u, const std::vector<u64>& a0,
                  JobResult& res) {
    const std::size_t M = u.size();
    const int k = m0.k();
    const u64 t = m0.t;
    const std::string where = "k=" + std::to_string(k) + " t=" + std::to_string(t);
    std::vector<u64> a1(M);
    for (std::size_t m = 0; m < M; ++m) a1[m] = ring::reduce_k(u[m] - a0[m], k);
    const Expected<std::vector<u64>> msg0 = r2f::r2f_open_share(a0, m0, 0);
    const Expected<std::vector<u64>> msg1 = r2f::r2f_open_share(a1, m1, 1);
    if (!msg0.ok || !msg1.ok) {
        res.failure = "r2f_open_share failed (" + where + "): " + msg0.code + msg1.code;
        return false;
    }
    ++res.checks;
    std::vector<u64> y(M);
    for (std::size_t m = 0; m < M; ++m) y[m] = ring::reduce_k(msg0.value[m] + msg1.value[m], k);
    const Expected<std::vector<u64>> z0 = r2f::r2f_eval(m0, y, 0);
    const Expected<std::vector<u64>> z1 = r2f::r2f_eval(m1, y, 1);
    if (!z0.ok || !z1.ok) {
        res.failure = "r2f_eval failed (" + where + "): " + z0.code + z1.code;
        return false;
    }
    ++res.checks;
    for (std::size_t m = 0; m < M; ++m) {
        const u64 z = (z0.value[m] + z1.value[m]) % t;
        const u64 want = r2f::signed_mod(u[m], t, k);
        if (!(z0.value[m] < t && z1.value[m] < t) || z != want) {
            res.failure = "r2f gate " + where + " u=" + std::to_string(u[m]) + " a0=" + std::to_string(a0[m]) + " got " + std::to_string(z) +
                          " want " + std::to_string(want);
            return false;
        }
        res.checks += 2;
    }
    return true;
}

void r2f_gate_check(const r2f::R2FMaterial& m0, const r2f::R2FMaterial& m1, const std::vector<u64>& u, const std::vector<u64>& a0) {
    JobResult res;
    if (!r2f_gate_run(m0, m1, u, a0, res)) fail_at("r2f gate", __FILE__, __LINE__, res.failure);
    g_checks += res.checks;
}

void test_r2f() {
    Timer t("ring_to_field");
    Rng rng = Rng::from_seed(50);
    CHECK_EQ_U64(r2f::pow2_mod(8, 17), 1);
    CHECK_EQ_U64(r2f::pow2_mod(64, 3), 1);
    CHECK_EQ_U64(r2f::pow2_mod(64, 5), 1);
    CHECK_EQ_U64(r2f::pow2_mod(64, 7), 2);
    CHECK_EQ_U64(r2f::pow2_mod(64, 17), 1);
    CHECK_EQ_U64(r2f::mulmod(3, 5, 7), 1);
    CHECK_EQ_U64(r2f::mulmod((u64(1) << 61) + 3, (u64(1) << 61) + 5, (u64(1) << 61) + 1), 8);   // (a+2)(a+4) mod (a) with a = 2^61+1 -> 2*4 = 8
#if defined(__SIZEOF_INT128__)
    CHECK_EQ_U64(r2f::pow2_mod(64, 68720050177ULL),
                 static_cast<u64>((static_cast<unsigned __int128>(1) << 64) % 68720050177ULL));
#endif
    CHECK_EQ_U64(r2f::signed_mod(255, 17, 8), 16);   // -1 mod 17
    CHECK_EQ_U64(r2f::signed_mod(1, 17, 8), 1);
    CHECK_EQ_U64(r2f::signed_mod(128, 17, 8), (17 - 128 % 17) % 17);
    CHECK_EQ_U64(r2f::r2f_bytes_per_element(17), 3272);
    CHECK_FAILS(r2f::check_r2f_params(16, 64), "R2F_MODULUS");
    // composite odd moduli: refused by the parameter check and by the dealer
    for (u64 composite : {u64(9), u64(561), u64(206160150531ULL), (u64(1) << 62) - 1, u64(4611686018427387845ULL)}) {
        CHECK_FAILS(r2f::check_r2f_params(composite, 64), "R2F_MODULUS");
        Rng crng = Rng::from_seed(5);
        CHECK_FAILS(r2f::r2f_material(2, composite, crng), "R2F_MODULUS");
    }
    for (u64 prime : {u64(3), u64(68720050177ULL), (u64(1) << 61) - 1, (u64(1) << 62) - 57}) CHECK_OK(r2f::check_r2f_params(prime, 64));
    CHECK_FAILS(r2f::check_r2f_params(17, 0), "RING_BITS");
    // the counterexamples of ring_to_field.hpp (q = 256, t = 17)
    {
        const auto mats = r2f::r2f_material_k(8, 17, rng, 8);
        CHECK_OK(mats);
        CHECK(((255 % 17) + (2 % 17)) % 17 == 2);   // the naive per-share reduction is wrong
        r2f_gate_check(mats.value.first, mats.value.second, std::vector<u64>(8, (255 + 2) % 256), std::vector<u64>(8, 255));
        CHECK_EQ_U64(r2f::signed_mod((255 + 2) % 256, 17, 8), 1);
        r2f_gate_check(mats.value.first, mats.value.second, std::vector<u64>(8, (128 + 129) % 256), std::vector<u64>(8, 128));
    }
}

// Exhaustive small ring of the ring-to-field gate: q = 2^k for k <= 8, every t in
// {3, 5, 7, 11, 13, 17}, every mask r, every value u and EVERY share split (a0, a1).
//   k <= 5: the whole gate runs for every (u, a0, r) (brute force, 2^(3k) evaluations per t).
//   k in {6, 7, 8}: the same set is covered in two exact steps. (a) For every (u, r) the gate runs
//       on the opened y = u + r and its output is checked. (b) For every (u, a0, r) the two open
//       messages are formed with r2f_open_share and their sum is checked to equal that same y.
//       r2f_eval(mat, y, party) is a deterministic function of the material and y alone (its
//       signature takes no share), so (a) and (b) together decide every split without paying 2^24
//       DCF evaluations per t. One random split per u additionally runs through the full gate.
void test_r2f_exhaustive() {
    Timer t("r2f exhaustive k<=8");
    struct Cfg {
        int k;
        u64 t;
    };
    std::vector<Cfg> cfgs;
    for (int k = 8; k >= 1; --k)
        for (u64 tt : {u64(3), u64(5), u64(7), u64(11), u64(13), u64(17)}) cfgs.push_back(Cfg{k, tt});
    run_parallel(
        cfgs.size(),
        [&](std::size_t j, JobResult& res) {
            const Cfg c = cfgs[j];
            const int k = c.k;
            Rng rng = Rng::from_seed(0x723266ULL + 104729ULL * static_cast<u64>(k) + c.t);
            const std::size_t q = std::size_t(1) << k;
            std::vector<u64> r(q);
            for (std::size_t m = 0; m < q; ++m) r[m] = m;   // every mask of the ring
            const auto mats = r2f::r2f_material_from_mask(r, c.t, rng, k);
            if (!mats.ok) {
                res.failure = "r2f_material_from_mask k=" + std::to_string(k) + ": " + mats.code;
                return;
            }
            ++res.checks;
            const r2f::R2FMaterial& m0 = mats.value.first;
            const r2f::R2FMaterial& m1 = mats.value.second;
            for (u64 u = 0; u < q; ++u) {
                const std::vector<u64> uv(q, u);
                if (k <= 5) {
                    for (u64 a0 = 0; a0 < q; ++a0)
                        if (!r2f_gate_run(m0, m1, uv, std::vector<u64>(q, a0), res)) return;
                    continue;
                }
                if (!r2f_gate_run(m0, m1, uv, uv, res)) return;   // (a): split a0 = u, a1 = 0
                for (u64 a0 = 0; a0 < q; ++a0) {                    // (b): every split opens the same y
                    const Expected<std::vector<u64>> msg0 = r2f::r2f_open_share(std::vector<u64>(q, a0), m0, 0);
                    const Expected<std::vector<u64>> msg1 = r2f::r2f_open_share(std::vector<u64>(q, ring::reduce_k(u - a0, k)), m1, 1);
                    if (!msg0.ok || !msg1.ok) {
                        res.failure = "r2f_open_share failed: " + msg0.code + msg1.code;
                        return;
                    }
                    for (std::size_t m = 0; m < q; ++m) {
                        if (ring::reduce_k(msg0.value[m] + msg1.value[m], k) != ring::reduce_k(u + r[m], k)) {
                            res.failure = "r2f open k=" + std::to_string(k) + " u=" + std::to_string(u) + " a0=" + std::to_string(a0) +
                                          " r=" + std::to_string(r[m]) + ": opened y differs from u + r";
                            return;
                        }
                    }
                    res.checks += static_cast<long>(q);
                }
                std::vector<u64> a0(q);
                for (u64& v : a0) v = rng.below(q);
                if (!r2f_gate_run(m0, m1, uv, a0, res)) return;
            }
        },
        "r2f exhaustive");
    CHECK(cfgs.size() == 48);
}

void test_r2f_random() {
    Timer t("r2f random 64-bit");
    Rng rng = Rng::from_seed(52);
    const u64 tt = 68720050177ULL;
    const std::size_t M = 20000;
    const auto mats = r2f::r2f_material(M, tt, rng);
    CHECK_OK(mats);
    CHECK_EQ_U64(mats.value.first.nbytes(), M * 3272);
    std::vector<u64> u(M), a0(M);
    for (std::size_t m = 0; m < M; ++m) {
        u[m] = rng.next_u64();
        a0[m] = rng.next_u64();
    }
    u[0] = 0;
    u[1] = 1;
    u[2] = ring::MASK64;
    u[3] = u64(1) << 63;
    u[4] = (u64(1) << 63) - 1;
    u[5] = tt;
    u[6] = ring::MASK64 - tt + 1;
    r2f_gate_check(mats.value.first, mats.value.second, u, a0);
    for (dcf::WireLayout layout : {dcf::WireLayout::Binding, dcf::WireLayout::PythonV2}) {
        const Expected<r2f::R2FMaterial> s = mats.value.second.slice(0, 32);
        CHECK_OK(s);
        const std::vector<uint8_t> b = s.value.to_bytes(layout);
        const Expected<r2f::R2FMaterial> back = r2f::R2FMaterial::from_bytes(b);
        CHECK_OK(back);
        CHECK(same_r2f(back.value, s.value));
        std::vector<uint8_t> bad = b;
        bad[24] = 16;   // even t
        CHECK_FAILS(r2f::R2FMaterial::from_bytes(bad), "R2F_");
    }
    const Expected<r2f::R2FMaterial> s1 = mats.value.first.slice(0, 7);
    const Expected<r2f::R2FMaterial> s2 = mats.value.first.slice(7, 20);
    const Expected<r2f::R2FMaterial> c = r2f::concat_r2f(s1.value, s2.value);
    CHECK_OK(c);
    CHECK(same_r2f(c.value, mats.value.first.slice(0, 20).value));
}

// ---------------------------------------------------------------------------------------------
// Cross-language vectors
// ---------------------------------------------------------------------------------------------
std::string json_str_list(const std::vector<u64>& v) {
    std::string s = "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) s += ", ";
        s += "\"" + dec(v[i]) + "\"";
    }
    return s + "]";
}

std::string json_i64_list(const std::vector<i64>& v) {
    std::string s = "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(v[i]);
    }
    return s + "]";
}

void emit_prg_vectors(const std::string& path) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);   // LF line ends on every host
    f << "{\n  \"description\": \"MMO PRG blocks B_j(s) = AES_K(s xor c_j) xor s; base layout j = 0..3 and the "
         "component-specific Convert words (one set of words per payload component)\",\n";
    f << "  \"fixed_key_hex\": \"" << to_hex(PRG_FIXED_KEY, 16) << "\",\n";
    f << "  \"layout\": {\"sL\": \"out[0:16]\", \"sR\": \"out[16:32]\", \"vL\": \"LE u64 of out[32:40]\", \"vR\": \"LE u64 of out[40:48]\", "
         "\"tL\": \"out[48] & 1\", \"tR\": \"out[49] & 1\"},\n";
    f << "  \"convert\": {\"blocks_hex\": \"B_0 .. B_9, 16 bytes each\", \"ring_words\": \"component c >= 1: [vL, vR] = LE u64 halves of B_{3+c}\", "
         "\"zt_words\": \"per modulus t, component c: [vL, vR] = [LE128(B_{4+2c}) mod t, LE128(B_{5+2c}) mod t]\"},\n";
    f << "  \"vectors\": [\n";
    std::vector<std::array<uint8_t, 16>> seeds;
    std::array<uint8_t, 16> s0{};
    for (int i = 0; i < 16; ++i) s0[i] = static_cast<uint8_t>(i);
    seeds.push_back(s0);
    Rng rng = Rng::from_seed(0x5052475632ULL);   // fixed: the file must be reproducible
    for (int i = 0; i < 16; ++i) {
        std::array<uint8_t, 16> s{};
        rng.fill(s.data(), 16);
        seeds.push_back(s);
    }
    for (std::size_t i = 0; i < seeds.size(); ++i) {
        uint8_t out[64];
        expand_bytes(seeds[i].data(), out);
        const Expansion e = expand(seeds[i].data());
        uint8_t sl[16], sr[16];
        block_to_bytes(e.sL, sl);
        block_to_bytes(e.sR, sr);
        f << "    {\"seed_hex\": \"" << to_hex(seeds[i]) << "\", \"out_hex\": \"" << to_hex(out, 64) << "\", \"sL_hex\": \"" << to_hex(sl, 16)
          << "\", \"sR_hex\": \"" << to_hex(sr, 16) << "\", \"vL\": \"" << dec(e.vL) << "\", \"vR\": \"" << dec(e.vR) << "\", \"tL\": " << e.tL
          << ", \"tR\": " << e.tR << ",\n";
        // component-specific Convert: raw blocks 0..9, ring words of components 1 and 2,
        // Z_t words of components 0..2 for t = 17 and the admitted t
        const Block128 sb = block_from_bytes(seeds[i].data());
        f << "     \"blocks_hex\": [";
        for (uint32_t j = 0; j < 10; ++j) {
            uint8_t blk[16];
            mmo_bytes(seeds[i].data(), j, blk);
            f << (j ? ", " : "") << "\"" << to_hex(blk, 16) << "\"";
        }
        f << "],\n     \"ring_words\": {";
        for (uint32_t c = 1; c <= 2; ++c) {
            const ValueWords w = value_words_ring(sb, c);
            f << (c > 1 ? ", " : "") << "\"" << c << "\": [\"" << dec(w.vL) << "\", \"" << dec(w.vR) << "\"]";
        }
        f << "},\n     \"zt_words\": {";
        bool first_t = true;
        for (u64 tt : {u64(17), u64(68720050177ULL)}) {
            f << (first_t ? "" : ", ") << "\"" << dec(tt) << "\": [";
            first_t = false;
            for (uint32_t c = 0; c <= 2; ++c) {
                const ValueWords w = value_words_zt(sb, c, tt);
                f << (c ? ", " : "") << "[\"" << dec(w.vL) << "\", \"" << dec(w.vR) << "\"]";
            }
            f << "]";
        }
        f << "}}" << (i + 1 < seeds.size() ? "," : "") << "\n";
    }
    f << "  ]\n}\n";
}

void emit_dcf_vectors(const std::string& path) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);   // LF line ends on every host
    f << "{\n  \"description\": \"DCF key batches in both wire layouts with alpha, beta (row-major M*P), evaluation points x and both parties' output shares (row-major M*P)\",\n";
    f << "  \"cases\": [\n";
    struct Case {
        const char* name;
        int n;
        std::size_t P;
        u64 mod;
    };
    // The first four cases keep their names and seeds; the P = 3 cases exercise the tweaks above 5
    // of the component-specific Convert. ring_n32_p1 is unchanged by the repair.
    const Case cases[] = {{"ring_n8_p2", 8, 2, 0},        {"ring_n32_p1", 32, 1, 0},
                          {"zt17_n16_p2", 16, 2, 17},      {"zt_admitted_n64_p1", 64, 1, 68720050177ULL},
                          {"ring_n16_p3", 16, 3, 0},       {"zt_admitted_n32_p3", 32, 3, 68720050177ULL}};
    const std::size_t n_cases = sizeof(cases) / sizeof(cases[0]);
    Rng rng = Rng::from_seed(0x4443465632ULL);
    for (std::size_t ci = 0; ci < n_cases; ++ci) {
        const Case& c = cases[ci];
        const std::size_t M = 4;
        const u64 dmask = c.n >= 64 ? ring::MASK64 : ((u64(1) << c.n) - 1);
        std::vector<u64> alpha(M), beta(M * c.P), x(M);
        for (std::size_t m = 0; m < M; ++m) {
            alpha[m] = rng.next_u64() & dmask;
            for (std::size_t p = 0; p < c.P; ++p) beta[m * c.P + p] = rand_payload(c.mod, rng);
            x[m] = (m == 0) ? alpha[m] : (m == 1) ? ((alpha[m] - 1) & dmask) : (rng.next_u64() & dmask);
        }
        const auto kp = dcf::gen_dcf_batch(alpha, beta, c.n, rng, c.mod);
        CHECK_OK(kp);
        const auto o0 = dcf::eval_dcf_batch(kp.value.first, x);
        const auto o1 = dcf::eval_dcf_batch(kp.value.second, x);
        CHECK(o0.ok && o1.ok);
        f << "    {\"name\": \"" << c.name << "\", \"n_bits\": " << c.n << ", \"P\": " << c.P << ", \"M\": " << M << ", \"modulus\": \"" << dec(c.mod)
          << "\",\n";
        f << "     \"alpha\": " << json_str_list(alpha) << ",\n     \"beta\": " << json_str_list(beta) << ",\n";
        f << "     \"x\": " << json_str_list(x) << ",\n";
        f << "     \"out0\": " << json_str_list(o0.value) << ",\n     \"out1\": " << json_str_list(o1.value) << ",\n";
        f << "     \"key0_binding_hex\": \"" << to_hex(kp.value.first.to_bytes()) << "\",\n";
        f << "     \"key1_binding_hex\": \"" << to_hex(kp.value.second.to_bytes()) << "\",\n";
        f << "     \"key0_pythonv2_hex\": \"" << to_hex(kp.value.first.to_bytes(dcf::WireLayout::PythonV2)) << "\",\n";
        f << "     \"key1_pythonv2_hex\": \"" << to_hex(kp.value.second.to_bytes(dcf::WireLayout::PythonV2)) << "\"}" << (ci + 1 < n_cases ? "," : "") << "\n";
    }
    f << "  ]\n}\n";
}

void emit_material_vectors(const std::string& path) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);   // LF line ends on every host
    Rng rng = Rng::from_seed(0x4d41545632ULL);
    f << "{\n  \"description\": \"Composite material pairs (both key layouts), the opened value y and both parties' outputs\",\n";
    // trunc: k = 8, L = 6, f = 3, masks {37, 200}, x = {-13, 5}
    {
        const auto mats = trunc::trunc_material_from_mask(std::vector<u64>{37, 200}, 3, 6, rng, 8);
        CHECK_OK(mats);
        const std::vector<i64> x = {-13, 5};
        const std::vector<u64> x0 = {77, 3};
        std::vector<u64> x1(2), y(2);
        for (std::size_t m = 0; m < 2; ++m) x1[m] = ring::reduce_k(ring::from_signed(x[m]) - x0[m], 8);
        const auto msg0 = trunc::trunc_open_share(x0, mats.value.first, 0);
        const auto msg1 = trunc::trunc_open_share(x1, mats.value.second, 1);
        for (std::size_t m = 0; m < 2; ++m) y[m] = ring::reduce_k(msg0.value[m] + msg1.value[m], 8);
        const auto z0 = trunc::trunc_faithful_eval(mats.value.first, y, 0);
        const auto z1 = trunc::trunc_faithful_eval(mats.value.second, y, 1);
        CHECK(z0.ok && z1.ok);
        f << "  \"trunc\": {\"k\": 8, \"L\": 6, \"f\": 3, \"r\": [\"37\", \"200\"], \"x\": " << json_i64_list(x) << ", \"x_share0\": " << json_str_list(x0)
          << ", \"x_share1\": " << json_str_list(x1) << ", \"y\": " << json_str_list(y) << ",\n";
        f << "    \"z0\": " << json_str_list(z0.value) << ", \"z1\": " << json_str_list(z1.value) << ", \"floor_x_div_2f\": [-2, 0],\n";
        f << "    \"material0_binding_hex\": \"" << to_hex(mats.value.first.to_bytes()) << "\",\n";
        f << "    \"material1_binding_hex\": \"" << to_hex(mats.value.second.to_bytes()) << "\",\n";
        f << "    \"material0_pythonv2_hex\": \"" << to_hex(mats.value.first.to_bytes(dcf::WireLayout::PythonV2)) << "\",\n";
        f << "    \"material1_pythonv2_hex\": \"" << to_hex(mats.value.second.to_bytes(dcf::WireLayout::PythonV2)) << "\"},\n";
    }
    // drelu: n = 8 with product, masks {0x1122334455667705, 0x0f0e0d0c0b0a0980}, x = {-3, 100}
    {
        const std::vector<u64> r = {0x1122334455667705ULL, 0x0f0e0d0c0b0a0980ULL};
        const auto mats = drelu::drelu_material_from_mask(r, 8, rng, true);
        CHECK_OK(mats);
        const std::vector<i64> x = {-3, 100};
        std::vector<u64> y(2);
        for (std::size_t m = 0; m < 2; ++m) y[m] = ring::from_signed(x[m]) + r[m];
        const auto o0 = drelu::drelu_eval(mats.value.first, y);
        const auto o1 = drelu::drelu_eval(mats.value.second, y);
        CHECK(o0.ok && o1.ok);
        f << "  \"drelu\": {\"n_bits\": 8, \"with_product\": true, \"r\": " << json_str_list(r) << ", \"x\": " << json_i64_list(x) << ", \"y\": "
          << json_str_list(y) << ",\n";
        f << "    \"b0\": " << json_str_list(o0.value.b) << ", \"b1\": " << json_str_list(o1.value.b) << ", \"rb0\": " << json_str_list(o0.value.rb)
          << ", \"rb1\": " << json_str_list(o1.value.rb) << ", \"drelu_x\": [0, 1],\n";
        f << "    \"material0_binding_hex\": \"" << to_hex(mats.value.first.to_bytes()) << "\",\n";
        f << "    \"material1_binding_hex\": \"" << to_hex(mats.value.second.to_bytes()) << "\",\n";
        f << "    \"material0_pythonv2_hex\": \"" << to_hex(mats.value.first.to_bytes(dcf::WireLayout::PythonV2)) << "\",\n";
        f << "    \"material1_pythonv2_hex\": \"" << to_hex(mats.value.second.to_bytes(dcf::WireLayout::PythonV2)) << "\"},\n";
    }
    // r2f: k = 8, t = 17, masks {5, 250}, the counterexample (a0 = 255, a1 = 2) and (a0 = 128, a1 = 129)
    {
        const std::vector<u64> r = {5, 250};
        const auto mats = r2f::r2f_material_from_mask(r, 17, rng, 8);
        CHECK_OK(mats);
        const std::vector<u64> a0 = {255, 128}, a1 = {2, 129};
        std::vector<u64> u(2), y(2);
        for (std::size_t m = 0; m < 2; ++m) u[m] = ring::reduce_k(a0[m] + a1[m], 8);
        const auto msg0 = r2f::r2f_open_share(a0, mats.value.first, 0);
        const auto msg1 = r2f::r2f_open_share(a1, mats.value.second, 1);
        for (std::size_t m = 0; m < 2; ++m) y[m] = ring::reduce_k(msg0.value[m] + msg1.value[m], 8);
        const auto z0 = r2f::r2f_eval(mats.value.first, y, 0);
        const auto z1 = r2f::r2f_eval(mats.value.second, y, 1);
        CHECK(z0.ok && z1.ok);
        f << "  \"ring_to_field\": {\"k\": 8, \"t\": \"17\", \"r\": " << json_str_list(r) << ", \"a0\": " << json_str_list(a0) << ", \"a1\": " << json_str_list(a1)
          << ", \"u\": " << json_str_list(u) << ", \"y\": " << json_str_list(y) << ",\n";
        f << "    \"z0\": " << json_str_list(z0.value) << ", \"z1\": " << json_str_list(z1.value) << ", \"z\": " << json_str_list(r2f::signed_mod(u, 17, 8))
          << ",\n";
        f << "    \"material0_binding_hex\": \"" << to_hex(mats.value.first.to_bytes()) << "\",\n";
        f << "    \"material1_binding_hex\": \"" << to_hex(mats.value.second.to_bytes()) << "\",\n";
        f << "    \"material0_pythonv2_hex\": \"" << to_hex(mats.value.first.to_bytes(dcf::WireLayout::PythonV2)) << "\",\n";
        f << "    \"material1_pythonv2_hex\": \"" << to_hex(mats.value.second.to_bytes(dcf::WireLayout::PythonV2)) << "\"}\n";
    }
    f << "}\n";
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// AES paths: hardware (AES-NI) against portable, and the batched DCF loops against a scalar walk
// ---------------------------------------------------------------------------------------------
// Scalar reference evaluation of one DCF key batch: one key at a time, one mmo() call per block,
// plain % arithmetic. This is the loop the batched eval_dcf_batch replaced; the two must agree on
// every word, including on keys whose correction words are NOT reduced mod t (a key read from the
// wire may carry any 64-bit word).
std::vector<u64> dcf_eval_reference(const dcf::DcfKeyBatch& key, const std::vector<u64>& x) {
    const std::size_t M = key.M();
    const std::size_t n = key.n_bits;
    const std::size_t P = key.P;
    const u64 t = key.modulus;
    const u64 dmask = n >= 64 ? ~u64(0) : ((u64(1) << n) - 1);
    auto add = [t](u64 a, u64 b) { return t ? ((a + b) % t) : (a + b); };
    auto word = [t](const Block128& s, std::size_t p, u64 side) {
        const uint32_t c = static_cast<uint32_t>(p);
        if (t) return block_mod_portable(mmo(s, prg_zt_value_tweak(c, static_cast<uint32_t>(side))), t);
        const Block128 b = mmo(s, prg_ring_value_tweak(c));
        return side ? b.hi : b.lo;
    };
    std::vector<u64> out(M * P, 0);
    for (std::size_t m = 0; m < M; ++m) {
        Block128 s = key.seed_block(m);
        u64 tb = key.party;
        std::vector<u64> acc(P, 0);
        const u64 xv = x[m] & dmask;
        for (std::size_t i = 0; i < n; ++i) {
            const u64 x_i = (xv >> (n - 1 - i)) & 1u;
            const u64 mt = u64(0) - tb;
            const Block128 y0 = mmo(s, static_cast<uint32_t>(x_i));
            const Block128 y3 = mmo(s, PRG_TWEAK_CONTROL);
            Block128 s_next = y0;
            s_next.lo ^= key.cw_seeds[(i * 2 + 0) * M + m] & mt;
            s_next.hi ^= key.cw_seeds[(i * 2 + 1) * M + m] & mt;
            const u64 t_next = (x_i ? prg_tr_bit(y3) : prg_tl_bit(y3)) ^ ((x_i ? key.cw_tR[i * M + m] : key.cw_tL[i * M + m]) & tb);
            for (std::size_t p = 0; p < P; ++p)
                acc[p] = add(acc[p], add(word(s, p, x_i), key.cw_values[(i * P + p) * M + m] & mt));
            s = s_next;
            tb = t_next;
        }
        const u64 mt = u64(0) - tb;
        for (std::size_t p = 0; p < P; ++p) {
            u64 o = add(acc[p], add(word(s, p, 0), key.cw_final[p * M + m] & mt));
            if (key.party == 1) o = t ? ((t - o) % t) : (u64(0) - o);
            out[m * P + p] = o;
        }
    }
    return out;
}

// Restores the run-time selection when a test section that forced the portable path ends.
struct PortableScope {
    explicit PortableScope(bool on) { Aes128::force_portable(on); }
    ~PortableScope() { Aes128::force_portable(false); }
};

void test_aes_paths() {
    Timer t("aes paths (hw vs portable)");
    const bool hw = Aes128::hardware_available();
    std::fprintf(stderr, "[test_lowlevel] AES: hardware path %s, in use: %s\n", hw ? "available" : "not available",
                 Aes128::implementation());
    CHECK(Aes128::hardware_enabled() == (std::string(Aes128::implementation()) == "aes-ni"));
    CHECK(!Aes128::hardware_enabled() || hw);
    {
        PortableScope forced(true);
        CHECK(!Aes128::hardware_enabled());
        CHECK(std::string(Aes128::implementation()) == "portable");
    }
    // --- FIPS-197 vectors on each named path, through the byte and the word interface -------------
    struct Kat {
        const char* key;
        const char* pt;
        const char* ct;
    };
    const Kat kats[] = {
        {"000102030405060708090a0b0c0d0e0f", "00112233445566778899aabbccddeeff", "69c4e0d86a7b0430d8cdb78070b4c55a"},   // C.1
        {"2b7e151628aed2a6abf7158809cf4f3c", "3243f6a8885a308d313198a2e0370734", "3925841d02dc09fbdc118597196a0b32"},   // B
        {"2b7e151628aed2a6abf7158809cf4f3c", "6bc1bee22e409f96e93d7e117393172a", "3ad77bb40d7a3660a89ecaf32466ef97"},   // SP 800-38A F.1.1
    };
    for (const Kat& kat : kats) {
        const std::vector<uint8_t> key = hexv(kat.key);
        const std::vector<uint8_t> pt = hexv(kat.pt);
        const Aes128 aes(key.data());
        // 11 copies: one full group of eight and a tail of three
        std::vector<uint8_t> in(16 * 11), out(16 * 11, 0);
        for (int i = 0; i < 11; ++i) std::memcpy(&in[16 * i], pt.data(), 16);
        aes.encrypt_blocks_portable(in.data(), out.data(), 11);
        for (int i = 0; i < 11; ++i) CHECK(to_hex(&out[16 * i], 16) == kat.ct);
        std::vector<uint8_t> out_hw(16 * 11, 0);
        CHECK(aes.encrypt_blocks_hardware(in.data(), out_hw.data(), 11) == hw);
        if (hw)
            for (int i = 0; i < 11; ++i) CHECK(to_hex(&out_hw[16 * i], 16) == kat.ct);
        const Block128 b = block_from_bytes(pt.data());
        std::vector<u64> win(22), wout(22, 0);
        for (int i = 0; i < 11; ++i) {
            win[2 * i] = b.lo;
            win[2 * i + 1] = b.hi;
        }
        aes.encrypt_words_batch_portable(win.data(), wout.data(), 11);
        for (int i = 0; i < 11; ++i) {
            uint8_t ct[16];
            block_to_bytes(Block128{wout[2 * i], wout[2 * i + 1]}, ct);
            CHECK(to_hex(ct, 16) == kat.ct);
        }
        if (hw) {
            std::vector<u64> wout_hw(22, 0);
            CHECK(aes.encrypt_words_batch_hardware(win.data(), wout_hw.data(), 11));
            CHECK(wout_hw == wout);
        }
        // the selected path, whichever it is, and the forced portable path
        for (int forced = 0; forced < 2; ++forced) {
            PortableScope scope(forced == 1);
            uint8_t ct[16];
            aes.encrypt_block(pt.data(), ct);
            CHECK(to_hex(ct, 16) == kat.ct);
            u64 lo = 0, hi = 0;
            aes.encrypt_words(b.lo, b.hi, lo, hi);
            block_to_bytes(Block128{lo, hi}, ct);
            CHECK(to_hex(ct, 16) == kat.ct);
        }
    }
    // --- random keys, every batch length 0..40: hardware == portable == block by block -----------
    Rng rng = Rng::from_seed(0xae5);
    for (int it = 0; it < 20; ++it) {
        uint8_t key[16];
        rng.fill(key, 16);
        const Aes128 aes(key);
        for (std::size_t n = 0; n <= 40; ++n) {
            std::vector<uint8_t> in(16 * n + 1), ref(16 * n + 1, 0xa5), got(16 * n + 1, 0xa5);
            rng.fill(in.data(), in.size());
            aes.encrypt_blocks_portable(in.data(), ref.data(), n);
            CHECK(ref[16 * n] == 0xa5);   // nothing written past the last block
            for (std::size_t i = 0; i < n; ++i) {
                PortableScope scope(true);
                uint8_t one[16];
                aes.encrypt_block(&in[16 * i], one);
                CHECK(std::memcmp(one, &ref[16 * i], 16) == 0);
            }
            aes.encrypt_blocks(in.data(), got.data(), n);   // the selected path
            CHECK(got == ref);
            if (hw) {
                std::vector<uint8_t> h(16 * n + 1, 0xa5);
                CHECK(aes.encrypt_blocks_hardware(in.data(), h.data(), n));
                CHECK(h == ref);
                // in place
                std::vector<uint8_t> inplace = in;
                CHECK(aes.encrypt_blocks_hardware(inplace.data(), inplace.data(), n));
                CHECK(std::memcmp(inplace.data(), ref.data(), 16 * n) == 0 && inplace[16 * n] == in[16 * n]);
            }
            std::vector<uint8_t> inplace = in;
            aes.encrypt_blocks_portable(inplace.data(), inplace.data(), n);
            CHECK(std::memcmp(inplace.data(), ref.data(), 16 * n) == 0);
            // word interface on the same blocks
            std::vector<u64> win(2 * n + 1), wref(2 * n + 1, 7), wgot(2 * n + 1, 7);
            for (std::size_t i = 0; i < n; ++i) {
                const Block128 b = block_from_bytes(&in[16 * i]);
                win[2 * i] = b.lo;
                win[2 * i + 1] = b.hi;
            }
            aes.encrypt_words_batch_portable(win.data(), wref.data(), n);
            for (std::size_t i = 0; i < n; ++i) CHECK(Block128({wref[2 * i], wref[2 * i + 1]}) == block_from_bytes(&ref[16 * i]));
            aes.encrypt_words_batch(win.data(), wgot.data(), n);
            CHECK(wgot == wref);
            if (hw) {
                std::vector<u64> wh(2 * n + 1, 7);
                CHECK(aes.encrypt_words_batch_hardware(win.data(), wh.data(), n));
                CHECK(wh == wref);
            }
        }
        // the one-pass MMO form: out = E(b xor (0, h)) xor b, every (n_base, repeat) shape up to 5 x 9
        for (std::size_t n_base = 0; n_base <= 5; ++n_base)
            for (std::size_t repeat = 0; repeat <= 9; ++repeat) {
                const std::size_t n = n_base * repeat;
                std::vector<u64> base(2 * n_base + 1), hx(n + 1), ref(2 * n + 1, 9), got(2 * n + 1, 9);
                for (u64& v : base) v = rng.next_u64();
                for (u64& v : hx) v = rng.next_u64();
                for (std::size_t k = 0; k < n_base; ++k)
                    for (std::size_t j = 0; j < repeat; ++j) {
                        const std::size_t i = k * repeat + j;
                        uint8_t x[16], y[16];
                        block_to_bytes(Block128{base[2 * k], base[2 * k + 1] ^ hx[i]}, x);
                        {
                            PortableScope scope(true);
                            aes.encrypt_block(x, y);
                        }
                        const Block128 e = block_from_bytes(y);
                        ref[2 * i] = e.lo ^ base[2 * k];
                        ref[2 * i + 1] = e.hi ^ base[2 * k + 1];
                    }
                aes.mmo_words_batch_portable(base.data(), n_base, repeat, hx.data(), got.data());
                CHECK(got == ref);
                std::fill(got.begin(), got.end(), 9);
                aes.mmo_words_batch(base.data(), n_base, repeat, hx.data(), got.data());
                CHECK(got == ref);
                if (hw) {
                    std::fill(got.begin(), got.end(), 9);
                    CHECK(aes.mmo_words_batch_hardware(base.data(), n_base, repeat, hx.data(), got.data()));
                    CHECK(got == ref);
                }
            }
    }
    // --- PRG: the batch forms against mmo(), on the selected and on the forced portable path ------
    CHECK(prg_tweak_word(0) == 0 && prg_tweak_word(1) == (u64(1) << 56) && prg_tweak_word(3) == (u64(3) << 56));
    CHECK(prg_tweak_word(0x01020304u) == ((u64(0x04) << 56) | (u64(0x03) << 48) | (u64(0x02) << 40) | (u64(0x01) << 32)));
    for (std::size_t n : {std::size_t(0), std::size_t(1), std::size_t(7), std::size_t(8), std::size_t(9), std::size_t(64), std::size_t(65),
                          std::size_t(200)}) {
        std::vector<Block128> seeds(n), a(n), b(n), c(n);
        std::vector<uint32_t> tw(n);
        std::vector<u64> tww(n);
        for (std::size_t i = 0; i < n; ++i) {
            seeds[i] = Block128{rng.next_u64(), rng.next_u64()};
            tw[i] = (i % 5 == 4) ? rng.next_u32() : static_cast<uint32_t>(rng.below(514));
            tww[i] = prg_tweak_word(tw[i]);
        }
        mmo_batch(seeds.data(), tw.data(), a.data(), n);
        mmo_batch_repeat(seeds.data(), n, 1, tww.data(), c.data());
        {
            PortableScope scope(true);
            mmo_batch(seeds.data(), tw.data(), b.data(), n);
            for (std::size_t i = 0; i < n; ++i) CHECK(mmo(seeds[i], tw[i]) == b[i]);
        }
        for (std::size_t i = 0; i < n; ++i) CHECK(a[i] == b[i] && c[i] == b[i] && mmo(seeds[i], tw[i]) == b[i]);
        // per_seed = 3: the evaluation shape
        if (n % 3 == 0 && n > 0) {
            std::vector<Block128> d(n);
            mmo_batch_repeat(seeds.data(), n / 3, 3, tww.data(), d.data());
            for (std::size_t i = 0; i < n; ++i) CHECK(d[i] == mmo(seeds[i / 3], tw[i]));
        }
    }
    {
        std::vector<Block128> seeds(37);
        for (Block128& sd : seeds) sd = Block128{rng.next_u64(), rng.next_u64()};
        std::vector<Expansion> e_sel, e_port;
        expand_batch(seeds, e_sel);
        {
            PortableScope scope(true);
            expand_batch(seeds, e_port);
        }
        CHECK(e_sel.size() == 37 && e_port.size() == 37);
        for (std::size_t i = 0; i < seeds.size(); ++i) {
            const Expansion one = expand(seeds[i]);
            for (const Expansion* e : {&e_sel[i], &e_port[i]})
                CHECK(e->sL == one.sL && e->sR == one.sR && e->vL == one.vL && e->vR == one.vR && e->tL == one.tL && e->tR == one.tR);
            CHECK(one.sL == mmo(seeds[i], 0) && one.sR == mmo(seeds[i], 1) && one.vL == mmo(seeds[i], 2).lo &&
                  one.vR == mmo(seeds[i], 2).hi && one.tL == prg_tl_bit(mmo(seeds[i], 3)) && one.tR == prg_tr_bit(mmo(seeds[i], 3)));
        }
    }
    // --- DCF: the same generation and evaluation on both paths, and against the scalar walk -------
    struct Shape {
        int n;
        u64 mod;
        std::size_t P;
        std::size_t M;
    };
    const Shape shapes[] = {
        {1, 0, 1, 1},   {7, 0, 1, 37},  {32, 0, 1, 37}, {32, 0, 2, 37}, {64, 0, 3, 50},  {64, 0, 1, 16},  {12, 0, 255, 3},
        {1, 17, 1, 5},  {9, 17, 3, 33}, {40, 68720050177ULL, 1, 37},    {40, 68720050177ULL, 2, 37},      {64, (u64(1) << 62) - 57, 2, 19},
        {10, 68720050177ULL, 255, 2},
    };
    for (const Shape& sh : shapes) {
        std::vector<u64> alpha(sh.M), beta(sh.M * sh.P), x(sh.M);
        Rng in_rng = Rng::from_seed(0xdcf0 + static_cast<u64>(sh.n) * 131 + sh.P);
        for (u64& v : alpha) v = in_rng.next_u64();
        for (u64& v : beta) v = in_rng.next_u64();   // not reduced: generation reduces the payload itself
        for (std::size_t m = 0; m < sh.M; ++m) x[m] = (m % 3 == 0) ? alpha[m] : in_rng.next_u64();
        Rng r_sel = Rng::from_seed(0x5eed);
        Rng r_port = Rng::from_seed(0x5eed);
        const Expected<std::pair<dcf::DcfKeyBatch, dcf::DcfKeyBatch>> k_sel = dcf::gen_dcf_batch(alpha, beta, sh.n, r_sel, sh.mod);
        CHECK_OK(k_sel);
        Expected<std::pair<dcf::DcfKeyBatch, dcf::DcfKeyBatch>> k_port = Expected<std::pair<dcf::DcfKeyBatch, dcf::DcfKeyBatch>>::failure("unset");
        {
            PortableScope scope(true);
            k_port = dcf::gen_dcf_batch(alpha, beta, sh.n, r_port, sh.mod);
        }
        CHECK_OK(k_port);
        CHECK(k_sel.value.first.to_bytes() == k_port.value.first.to_bytes());
        CHECK(k_sel.value.second.to_bytes() == k_port.value.second.to_bytes());
        CHECK_EQ_U64(r_sel.next_u64(), r_port.next_u64());   // both consumed the same randomness
        for (int party = 0; party < 2; ++party) {
            dcf::DcfKeyBatch key = party ? k_sel.value.second : k_sel.value.first;
            for (int unreduced = 0; unreduced < 2; ++unreduced) {
                if (unreduced) {
                    // a key as a malformed wire message could carry it: correction words above t
                    if (!sh.mod) break;
                    for (std::size_t i = 0; i < key.cw_values.size(); i += 3) key.cw_values[i] = in_rng.next_u64() >> (i % 2);
                    for (std::size_t i = 0; i < key.cw_final.size(); i += 2) key.cw_final[i] = in_rng.next_u64() >> 1;
                }
                const Expected<std::vector<u64>> o_sel = dcf::eval_dcf_batch(key, x);
                CHECK_OK(o_sel);
                const std::vector<u64> ref = dcf_eval_reference(key, x);
                CHECK(o_sel.value == ref);
                PortableScope scope(true);
                const Expected<std::vector<u64>> o_port = dcf::eval_dcf_batch(key, x);
                CHECK_OK(o_port);
                CHECK(o_port.value == ref);
                CHECK(dcf_eval_reference(key, x) == ref);
            }
        }
        // and the pair still shares the comparison
        const Expected<std::vector<u64>> o0 = dcf::eval_dcf_batch(k_sel.value.first, x);
        const Expected<std::vector<u64>> o1 = dcf::eval_dcf_batch(k_sel.value.second, x);
        CHECK(o0.ok && o1.ok);
        const Expected<std::vector<u64>> rec = dcf::reconstruct_dcf(o0.value, o1.value, sh.mod);
        CHECK_OK(rec);
        const u64 dmask = sh.n >= 64 ? ~u64(0) : ((u64(1) << sh.n) - 1);
        for (std::size_t m = 0; m < sh.M; ++m)
            for (std::size_t p = 0; p < sh.P; ++p) {
                const u64 b = sh.mod ? (beta[m * sh.P + p] % sh.mod) : beta[m * sh.P + p];
                CHECK_EQ_U64(rec.value[m * sh.P + p], ((x[m] & dmask) < (alpha[m] & dmask)) ? b : 0);
            }
    }
    // --- the gates, same seed, both paths: identical material bytes and identical output shares ---
    {
        const std::size_t M = 41;
        std::vector<u64> r(M), y(M);
        for (u64& v : r) v = rng.next_u64() & 0xffffffffu;
        for (u64& v : y) v = rng.next_u64() & 0xffffffffu;
        std::vector<std::vector<uint8_t>> mats[2];
        std::vector<std::vector<u64>> outs[2];
        for (int forced = 0; forced < 2; ++forced) {
            PortableScope scope(forced == 1);
            Rng gr = Rng::from_seed(0x6a7e);
            const auto dm = drelu::drelu_material_from_mask(r, 32, gr, true);
            CHECK_OK(dm);
            mats[forced].push_back(dm.value.first.to_bytes());
            mats[forced].push_back(dm.value.second.to_bytes());
            for (const drelu::DreluMaterial* mat : {&dm.value.first, &dm.value.second}) {
                const auto o = drelu::drelu_eval(*mat, y);
                CHECK_OK(o);
                outs[forced].push_back(o.value.b);
                outs[forced].push_back(o.value.rb);
            }
            const auto tm = trunc::trunc_material_from_mask(r, 16, 40, gr, 64);
            CHECK_OK(tm);
            mats[forced].push_back(tm.value.first.to_bytes());
            mats[forced].push_back(tm.value.second.to_bytes());
            for (int party = 0; party < 2; ++party) {
                const auto o = trunc::trunc_faithful_eval(party ? tm.value.second : tm.value.first, y, party);
                CHECK_OK(o);
                outs[forced].push_back(o.value);
            }
            const auto fm = r2f::r2f_material_from_mask(r, 68720050177ULL, gr);
            CHECK_OK(fm);
            mats[forced].push_back(fm.value.first.to_bytes());
            mats[forced].push_back(fm.value.second.to_bytes());
            for (int party = 0; party < 2; ++party) {
                const auto o = r2f::r2f_eval(party ? fm.value.second : fm.value.first, y, party);
                CHECK_OK(o);
                outs[forced].push_back(o.value);
            }
        }
        CHECK(mats[0].size() == 6 && mats[0] == mats[1]);
        CHECK(outs[0].size() == 8 && outs[0] == outs[1]);
    }
}

int main(int argc, char** argv) {
    const std::string vectors_dir = argc > 1 ? argv[1] : "";
    test_aes();
    test_sha();
    test_hmac();
    test_base64_hex();
    test_csprng();
    test_ring();
    test_prg();
    test_sharing();
    test_dcf_exhaustive();
    test_dcf_random();
    test_key_component_independence();
    test_aes_paths();
    if (Aes128::hardware_available()) {
        // The published vectors and the KEY_COMPONENT_INDEPENDENCE checks once more with the
        // portable cipher forced (the run above used the hardware path). OPENCRAB2_AES=portable
        // does the same for a whole executable, test_vectors included.
        PortableScope forced(true);
        Timer t("portable path rerun");
        test_aes();
        test_prg();
        test_key_component_independence();
    }
    test_drelu();
    test_trunc();
    test_trunc_exhaustive();
    test_trunc_random();
    test_r2f();
    test_r2f_exhaustive();
    test_r2f_random();
    if (!vectors_dir.empty()) {
        Timer t("emit vectors");
        emit_prg_vectors(vectors_dir + "/cpp_prg_vectors.json");
        emit_dcf_vectors(vectors_dir + "/cpp_dcf_vectors.json");
        emit_material_vectors(vectors_dir + "/cpp_material_vectors.json");
    }
    std::printf("PASS %ld checks\n", g_checks);
    return 0;
}
