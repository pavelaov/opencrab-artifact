// OpenCrab v2 C++17 mirror: secure toy-BFV bridge A2H_toy, Eval_toy, H2A_ref
// (paper, Section 5.2 and Appendix B, H2A-ref; mirror of python/opencrab2/highlevel/bridge.py).
//
// Status: reference design over the power-of-two toy scheme of lowlevel/toy_bfv.hpp (TEST_ONLY).
// Roles: P0 (model owner) holds the evaluated ciphertext and the plaintext weights W; P1 (client)
// holds the toy secret key s. Nothing but the protocol messages below crosses the channel.
//
//   a2h_toy(party, [x]_A64, bfv, pk, rng)
//       Both parties reduce their A64 share modulo t = 2^log_t locally (t divides 2^64, so the
//       reduction commutes with the sharing). P1 encrypts its reduced share under its own key
//       and sends (c0, c1) to P0: ONE message P1 -> P0 of 2 * 8 * N data bytes. P0 adds
//       Delta * (x0 mod t) into c0. Result: an encryption of x mod t held by P0 only; P1 gets an
//       empty ciphertext object back.
//   eval_toy(bfv, ct, W)
//       P0's plaintext-weight coefficient-packed matvec (ToyBFV::matvec_pt). No message. The
//       toy module refuses parameter sets whose worst-case noise reaches Delta / 2.
//   h2a_ref(party, bfv, ct | null, sk | null, positions)
//       Secure internal decryption. c1 * s is a negacyclic product of P0's private polynomial
//       with P1's private polynomial; [c1] = (c1, 0) and [s] = (0, s) are valid additive
//       sharings, so the product is one Beaver round through Party:
//         KARATSUBA_LEAVES (default, what the Python module does): Party::beaver_mul on the
//             3^log2(N) Karatsuba leaf products (2187 at N = 128, 19683 at N = 512), opening
//             2 * 8 * 3^log2(N) data bytes per party;
//         OUTER_TRIPLE (the schoolbook N^2 products): Party::matmul_shared on
//             the (N x 1) by (1 x N) outer product with one matrix triple, opening 2 * 8 * N
//             data bytes per party at the price of 8 * (2N + N^2) bytes of material per party.
//       P0 adds c0 and Delta / 2 to its share. Party::trunc_faithful with f = 64 - log_t and
//       L = 64 then gives shares of floor((c0 + c1 s + Delta/2) / 2^f) in the signed semantics,
//       which equals the plaintext coefficient modulo t. The coefficients at `positions` are
//       the A64 shares of y. Rounds: 1 (Beaver open) + 1 (truncation open) = 2. P1 never
//       receives ct, c1 s or the noise; P0 never receives s: each party's two incoming frames
//       are the peer's shares of values masked by dealer randomness.
//
// Message sizes depend only on (N, convolution mode); no branch on a secret changes a frame.
//
// Material. One h2a_ref call consumes 3^log2(N) Beaver triples (or one (N,1,N) matrix triple)
// and N faithful-truncation elements with (f = 64 - log_t, L = 64). DealerClient::take_trunc
// pools by f only and the refill link asks the dealer for the contract's logical width, so
// the L = 64 truncation material must be provisioned before the call (h2a_trunc_bundles
// below); with any other material Party::trunc_faithful fails closed with MATERIAL_PARAMS.
//
// What is tested (tests/test_bridge.cpp): the shared Karatsuba product and the outer-triple
// product against lowlevel::negacyclic_mul; end to end share -> a2h_toy -> eval_toy(W) ->
// h2a_ref -> reconstruct == W x mod t at (N = 128, d_in = d_out = 8) and (N = 512, d_in = 64,
// d_out = 8), 200 trials each; frame counts and byte counts by direction, constant across
// trials; P1 receives no frame of ciphertext shape and no frame that contains a ciphertext
// component; role violations fail closed before any message.
//
// Design limits (not security claims). The gate with L = 64 reads the pre-rounding value as a
// signed 64-bit integer, so the reconstructed output equals the plaintext modulo t (signed
// representative), not necessarily the value in [0, t): consumers reduce modulo t (t divides
// 2^64, so reducing each share locally is a valid Z_t sharing). q = 2^64 with N <= 512 offers no
// computational security; the scheme never touches real data. Real-parameter (SEAL, N = 8192)
// H2A_ref is NOT executed anywhere; any cost figure for it is an estimate derived from these
// toy measurements. Semi-honest parties only (profile SH2P-TrustedOffline-v1).
//
// Codes: BRIDGE_PARTY (role violation), BRIDGE_SHAPE, BRIDGE_PARAMS; codes of the toy scheme,
// of Party and of the dealer client pass through unchanged.
#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <opencrab2/expected.hpp>
#include <opencrab2/highlevel/dealer.hpp>
#include <opencrab2/highlevel/party.hpp>
#include <opencrab2/lowlevel/toy_bfv.hpp>

namespace opencrab2 {
namespace highlevel {
namespace bridge {

// Logical width handed to the faithful gate: the whole ring (see the header).
constexpr int BRIDGE_L = 64;

enum class Convolution {
    KARATSUBA_LEAVES,   // elementwise Beaver triples on the Karatsuba leaves (Python parity)
    OUTER_TRIPLE        // one (N, 1, N) matrix triple, schoolbook N^2 products
};

// 3^log2(N) for a power of two N (number of Karatsuba leaf products); 0 when N is not one.
std::size_t karatsuba_leaf_count(std::size_t N);

// Split two shared length-N polynomials into the 3^log2(N) Karatsuba leaf factors. Level by
// level every polynomial of length n becomes three of length n/2: [lo, hi, lo + hi], rows
// ordered (all lo, all hi, all mid) so that karatsuba_recombine undoes the split. Linear in
// the inputs: each party applies it to its own shares.
Expected<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>> karatsuba_leaf_operands(const std::vector<uint64_t>& sa,
                                                                                          const std::vector<uint64_t>& sb);
// Recombine the leaf products into the product modulo (X^N + 1) and 2^64 (linear as well).
Expected<std::vector<uint64_t>> karatsuba_recombine(const std::vector<uint64_t>& leaf_products, std::size_t N);

// Shares of a * b mod (X^N + 1) from shares of a and b: one Beaver round.
Expected<std::vector<uint64_t>> negacyclic_mul_shared(Party& party, const std::vector<uint64_t>& sa, const std::vector<uint64_t>& sb,
                                                      Convolution conv = Convolution::KARATSUBA_LEAVES);

// What one h2a_ref call consumes (public: a function of N, log_t and the mode only).
struct H2aMaterial {
    std::size_t beaver_triples = 0;       // KARATSUBA_LEAVES
    std::size_t matrix_triples = 0;       // OUTER_TRIPLE: one triple of shape (N, 1, N)
    std::size_t trunc_elements = 0;       // N
    int trunc_f = 0;                      // 64 - log_t
    int trunc_L = BRIDGE_L;
    std::size_t open_bytes_beaver = 0;    // data bytes each party sends in the Beaver round
    std::size_t open_bytes_trunc = 0;     // data bytes each party sends in the truncation round
    std::size_t rounds = 2;
};
H2aMaterial h2a_material(const lowlevel::ToyParams& params, Convolution conv = Convolution::KARATSUBA_LEAVES);

// Dealer side: the truncation bundles (f = 64 - log_t, L = 64) for `calls` h2a_ref calls. Both
// halves must be ingested (same order on both parties) before the calls.
Expected<BundlePair> h2a_trunc_bundles(Dealer& dealer, const lowlevel::ToyParams& params, std::size_t calls = 1);

// A64 share -> toy ciphertext of x mod t held by P0. One message P1 -> P0.
//   P1: pk and rng are required (its own public key, its encryption randomness); the returned
//       ciphertext is empty.
//   P0: pk and rng are ignored; returns the ciphertext with Delta * (x0 mod t) added into c0.
Expected<lowlevel::Ciphertext> a2h_toy(Party& party, const std::vector<uint64_t>& share, const lowlevel::ToyBFV& bfv,
                                       const lowlevel::Ciphertext* pk_or_null, lowlevel::RandomSource* rng_or_null);

// P0: plaintext-weight matvec on the toy ciphertext; returns (ct_y, positions). No message.
Expected<lowlevel::ToyBFV::MatvecResult> eval_toy(lowlevel::ToyBFV& bfv, const lowlevel::Ciphertext& ct, const lowlevel::Matrix& W);

// Secure internal decryption; returns this party's A64 shares of y at `positions` (see the header).
//   P0: ct required, sk must be null.   P1: sk required, ct must be null.
Expected<std::vector<uint64_t>> h2a_ref(Party& party, const lowlevel::ToyBFV& bfv, const lowlevel::Ciphertext* ct_or_null,
                                        const lowlevel::Poly* sk_or_null, const std::vector<int64_t>& positions,
                                        Convolution conv = Convolution::KARATSUBA_LEAVES);

}  // namespace bridge
}  // namespace highlevel
}  // namespace opencrab2
