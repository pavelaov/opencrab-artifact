// OpenCrab v2: toy power-of-two BFV reference scheme for the secure bridge (paper, Section 5.2).
//
// Status: TEST_ONLY reference, mirror of python/opencrab2/lowlevel/toy_bfv.py (same names, same
// arithmetic, same noise bound). Ring R_q = Z_q[X]/(X^N + 1) with q = 2^64 (native uint64 wrap),
// plaintext modulus t = 2^log_t (log_t <= 32), N in {64, 128, 256, 512}, Delta = 2^(64 - log_t),
// ternary secret s, errors uniform in [-B_e, B_e] (B_e = 8), public key pk = (-(a s + e), a).
//
//     Enc(m; u, e0, e1) = (pk0 u + e0 + Delta m,  pk1 u + e1)
//     Dec(c0, c1)       = floor((c0 + c1 s + Delta/2) / Delta) mod t
//
// Because t divides q, Delta t = 0 mod q: plaintext arithmetic is exactly arithmetic mod t and
// decryption is a faithful truncation by (64 - log_t) bits of (c0 + c1 s + Delta/2). That is the
// property H2A_ref relies on: it reproduces this decryption on secret shares with one Beaver
// convolution and one faithful truncation gate; decrypt_arith_steps exposes the intermediate
// quantities so that a test can check the two computations agree.
//
// Coefficient-packed matvec (Cheetah/Iron trick): for x packed as sum_i x_i X^i and the weight
// block p = sum_k sum_j W[k, d_in-1-j] X^(j + k d_in), coefficient (k+1) d_in - 1 of x p equals
// <W[k,:], x>; no cross term and no negacyclic wrap reaches those positions when k d_in + d_in
// <= N, so one ciphertext carries N / d_in output rows.
//
// Noise: ||e0 - e u + e1 s||_inf <= B_e (2N + 1) for a fresh ciphertext; after one packed matvec
// the worst case is fresh * nnz(p) * ||W||_inf with nnz(p) <= (N / d_in) d_in. matvec_pt refuses
// (TOY_BFV_NOISE_BUDGET) whenever that bound reaches Delta / 2; a raw mul_pt only checks shapes.
//
// What is tested (tests/test_toy_bfv.cpp): 1000 encrypt/decrypt round trips per admitted N,
// ct+ct, ct+pt, ct*pt against an integer reference, matvec equals W x mod t at (N=128, d_in=8,
// d_out=8) and (N=512, d_in=64, d_out=8), the noise bound refuses a violating parameter set, the
// decrypt_arith_steps decomposition, and the A2H_toy local-reduction pattern.
//
// Design limits (not security claims): q = 2^64 with N <= 512 offers no computational security
// and the scheme is never used on real data; the bound is a worst-case infinity-norm bound, so a
// refused parameter set may decrypt correctly most of the time, it is refused anyway. Real
// parameter BFV (SEAL, N = 8192) is not emulated here. Codes: TOY_BFV_PARAM, TOY_BFV_SHAPE,
// TOY_BFV_PLAINTEXT_RANGE, TOY_BFV_NOISE_BUDGET.
#pragma once

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "opencrab2/expected.hpp"

namespace opencrab2 {
namespace lowlevel {

using Poly = std::vector<std::uint64_t>;                 // length N, coefficients in Z_{2^64}
using Matrix = std::vector<std::vector<std::uint64_t>>;  // row-major, entries are ring values

struct Ciphertext {
    Poly c0;
    Poly c1;
};

// Randomness source: returns uniform 64-bit words. Tests use deterministic_source(seed); the
// dealer/bridge code uses os_random_source() (BCryptGenRandom / /dev/urandom).
using RandomSource = std::function<std::uint64_t()>;
RandomSource deterministic_source(std::uint64_t seed);   // SplitMix64, tests only
RandomSource os_random_source();                          // OS CSPRNG, buffered

constexpr int kRingBits = 64;
constexpr int kMaxLogT = 32;

struct ToyParams {
    int N = 128;
    int log_t = 16;
    int B_e = 8;

    std::uint64_t t() const { return std::uint64_t(1) << log_t; }
    std::uint64_t delta() const { return std::uint64_t(1) << (kRingBits - log_t); }
    int shift() const { return kRingBits - log_t; }      // low bits removed by the decryption rounding

    static Expected<ToyParams> make(int N, int log_t, int B_e = 8);   // TOY_BFV_PARAM on a bad set
};

// Product of two length-N polynomials modulo (X^N + 1) and 2^64, schoolbook O(N^2).
Poly negacyclic_mul(const Poly& a, const Poly& b);

struct DecryptSteps {
    Poly c1_s;                          // c1 * s
    Poly phase;                         // c0 + c1 s
    Poly pre_round;                     // phase + Delta/2 (mod 2^64)
    Poly plaintext;                     // pre_round >> shift, values in [0, t)
    int shift = 0;
    std::uint64_t half_delta = 0;
    std::vector<std::int64_t> noise_signed;   // phase - Delta * plaintext, centred (test diagnostic only)
};

class ToyBFV {
public:
    explicit ToyBFV(ToyParams params);
    const ToyParams& params() const { return params_; }
    std::int64_t ct_pt_products() const { return ct_pt_products_; }

    struct KeyPair {
        Poly sk;          // ternary in {-1, 0, 1} as two's complement uint64
        Ciphertext pk;    // (pk0, pk1) = (-(a s + e), a)
    };
    KeyPair keygen(RandomSource& rng) const;
    Expected<Ciphertext> encrypt(const Ciphertext& pk, const Poly& m, RandomSource& rng) const;
    Expected<Poly> decrypt(const Poly& sk, const Ciphertext& ct) const;
    Expected<DecryptSteps> decrypt_arith_steps(const Poly& sk, const Ciphertext& ct) const;

    Expected<Ciphertext> add_ct(const Ciphertext& a, const Ciphertext& b) const;
    Expected<Ciphertext> add_pt(const Ciphertext& ct, const Poly& m) const;      // adds Delta m into c0
    Expected<Ciphertext> mul_pt(const Ciphertext& ct, const Poly& p);            // counts one ct-pt product

    // Encoding helpers.
    Expected<Poly> encode_vector(const std::vector<std::uint64_t>& x) const;      // x[j] mod t at coefficient j
    std::uint64_t signed_mod_t(std::uint64_t v) const;         // representative in [-t/2, t/2) as two's complement
    std::uint64_t w_inf(const Matrix& W) const;                // infinity norm of the signed representatives
    Expected<int> rows_per_ct(int d_in) const;                 // N / d_in
    Expected<std::pair<Poly, std::vector<std::int64_t>>> encode_weight_block(const Matrix& W_block) const;

    // Noise bound.
    std::uint64_t fresh_noise_bound() const;                   // B_e (2N + 1)
    Expected<std::uint64_t> noise_bound(int d_in, std::uint64_t w_inf) const;   // saturating at UINT64_MAX
    Expected<bool> noise_budget_ok(int d_in, std::uint64_t w_inf) const;

    // Matvec.
    struct MatvecResult {
        Ciphertext ct_y;
        std::vector<std::int64_t> positions;   // positions[k] holds (W x)[k] mod t after decryption
    };
    Expected<MatvecResult> matvec_pt(const Ciphertext& ct_x, const Matrix& W);   // d_out <= N / d_in
    struct MatvecBlock {
        Ciphertext ct_y;
        std::vector<std::int64_t> positions;
        int row_lo = 0;
        int row_hi = 0;
    };
    Expected<std::vector<MatvecBlock>> matvec_pt_blocks(const Ciphertext& ct_x, const Matrix& W);   // any d_out
    Expected<std::vector<std::uint64_t>> matvec_reference(const Matrix& W, const std::vector<std::uint64_t>& x) const;

private:
    Status check_pt(const Poly& m) const;
    Status check_ct(const Ciphertext& ct) const;
    Poly ternary(RandomSource& rng) const;
    Poly error(RandomSource& rng) const;
    Poly uniform(RandomSource& rng) const;

    ToyParams params_;
    std::int64_t ct_pt_products_ = 0;
};

// Local arithmetic of A2H_toy without the channel: P1 reduces its A64 share mod t,
// encrypts under its own key; P0 adds Delta (x0 mod t) into c0. The result encrypts x mod t
// because Delta t = 0 mod 2^64. The two-party version with one message lives in the bridge.
Expected<Ciphertext> a2h_toy_local(const ToyBFV& bfv, const Ciphertext& pk, const std::vector<std::uint64_t>& x1_share,
                                   const std::vector<std::uint64_t>& x0_share, RandomSource& rng);

}  // namespace lowlevel
}  // namespace opencrab2
