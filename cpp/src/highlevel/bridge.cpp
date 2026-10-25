// Implementation of bridge.hpp: secure toy-BFV bridge (A2H_toy, Eval_toy, H2A_ref).
#include <opencrab2/highlevel/bridge.hpp>

#include <string>

namespace opencrab2 {
namespace highlevel {
namespace bridge {

namespace {

using u64 = uint64_t;
using Rows = std::vector<std::vector<u64>>;

bool is_pow2(std::size_t n) { return n != 0 && (n & (n - 1)) == 0; }

// One Karatsuba split of every row: n -> three rows of n/2, ordered (all lo, all hi, all mid).
Rows split_level(const Rows& in) {
    const std::size_t rows = in.size();
    const std::size_t h = in[0].size() / 2;
    Rows out(3 * rows, std::vector<u64>(h));
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t i = 0; i < h; ++i) {
            const u64 lo = in[r][i], hi = in[r][h + i];
            out[r][i] = lo;
            out[rows + r][i] = hi;
            out[2 * rows + r][i] = lo + hi;
        }
    }
    return out;
}

std::vector<u64> leaves_of(const std::vector<u64>& poly) {
    Rows cur(1, poly);
    while (cur[0].size() > 1) cur = split_level(cur);
    std::vector<u64> out(cur.size());
    for (std::size_t r = 0; r < cur.size(); ++r) out[r] = cur[r][0];
    return out;
}

}  // namespace

std::size_t karatsuba_leaf_count(std::size_t N) {
    if (!is_pow2(N)) return 0;
    std::size_t leaves = 1;
    for (std::size_t n = N; n > 1; n /= 2) leaves *= 3;
    return leaves;
}

Expected<std::pair<std::vector<u64>, std::vector<u64>>> karatsuba_leaf_operands(const std::vector<u64>& sa, const std::vector<u64>& sb) {
    if (!is_pow2(sa.size())) return fail("BRIDGE_SHAPE: polynomial length must be a power of two, got " + std::to_string(sa.size()));
    if (sb.size() != sa.size())
        return fail("BRIDGE_SHAPE: operand lengths differ (" + std::to_string(sa.size()) + " vs " + std::to_string(sb.size()) + ")");
    return ok(std::make_pair(leaves_of(sa), leaves_of(sb)));
}

Expected<std::vector<u64>> karatsuba_recombine(const std::vector<u64>& leaf_products, std::size_t N) {
    if (!is_pow2(N)) return fail("BRIDGE_SHAPE: polynomial length must be a power of two, got " + std::to_string(N));
    if (leaf_products.size() != karatsuba_leaf_count(N))
        return fail("BRIDGE_SHAPE: expected " + std::to_string(karatsuba_leaf_count(N)) + " leaf products, got " +
                    std::to_string(leaf_products.size()));
    // a product of two length-1 polynomials is stored as a length-2 row (coefficient, 0)
    Rows P(leaf_products.size(), std::vector<u64>(2, 0));
    for (std::size_t r = 0; r < P.size(); ++r) P[r][0] = leaf_products[r];
    while (P.size() > 1) {
        const std::size_t Bn = P.size() / 3;
        const std::size_t Lc = P[0].size();   // product length of the current level
        const std::size_t h = Lc / 2;         // operand length of the current level
        Rows res(Bn, std::vector<u64>(2 * Lc, 0));
        for (std::size_t r = 0; r < Bn; ++r) {
            const std::vector<u64>& lo = P[r];
            const std::vector<u64>& hi = P[Bn + r];
            const std::vector<u64>& mid = P[2 * Bn + r];
            for (std::size_t i = 0; i < Lc; ++i) {
                res[r][i] += lo[i];
                res[r][Lc + i] += hi[i];
                res[r][h + i] += mid[i] - lo[i] - hi[i];
            }
        }
        P = std::move(res);
    }
    const std::vector<u64>& full = P[0];
    if (full.size() != 2 * N) return fail("BRIDGE_SHAPE: recombined length differs from 2N");
    std::vector<u64> out(N);
    for (std::size_t i = 0; i < N; ++i) out[i] = full[i] - full[N + i];   // X^N = -1
    return ok(std::move(out));
}

Expected<std::vector<u64>> negacyclic_mul_shared(Party& party, const std::vector<u64>& sa, const std::vector<u64>& sb, Convolution conv) {
    const std::size_t N = sa.size();
    if (!is_pow2(N)) return fail("BRIDGE_SHAPE: polynomial length must be a power of two, got " + std::to_string(N));
    if (sb.size() != N) return fail("BRIDGE_SHAPE: operand lengths differ (" + std::to_string(N) + " vs " + std::to_string(sb.size()) + ")");
    Party::Track tr(party, "negacyclic_mul_shared", {N});
    if (conv == Convolution::KARATSUBA_LEAVES) {
        OC2_TRY(ops, karatsuba_leaf_operands(sa, sb));
        OC2_TRY(prod, party.beaver_mul(Mat::row_vector(std::move(ops.first)), Mat::row_vector(std::move(ops.second))));   // one round
        return karatsuba_recombine(prod.a, N);
    }
    // Schoolbook: shares of the outer product Z[i][j] = a_i b_j from one (N, 1, N) matrix triple,
    // then the negacyclic fold (linear, local).
    OC2_TRY(Z, party.matmul_shared(Mat::from_vector(sa, N, 1), Mat::from_vector(sb, 1, N), false));   // one round
    if (Z.rows != N || Z.cols != N) return fail("BRIDGE_SHAPE: outer product has an unexpected shape");
    std::vector<u64> out(N, 0);
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            const std::size_t k = i + j;
            if (k < N) out[k] += Z.a[i * N + j];
            else out[k - N] -= Z.a[i * N + j];
        }
    }
    return ok(std::move(out));
}

H2aMaterial h2a_material(const lowlevel::ToyParams& params, Convolution conv) {
    H2aMaterial m;
    const std::size_t N = static_cast<std::size_t>(params.N);
    if (conv == Convolution::KARATSUBA_LEAVES) {
        m.beaver_triples = karatsuba_leaf_count(N);
        m.open_bytes_beaver = 2 * 8 * m.beaver_triples;
    } else {
        m.matrix_triples = 1;
        m.open_bytes_beaver = 2 * 8 * N;
    }
    m.trunc_elements = N;
    m.trunc_f = params.shift();
    m.trunc_L = BRIDGE_L;
    m.open_bytes_trunc = 8 * N;
    m.rounds = 2;
    return m;
}

Expected<BundlePair> h2a_trunc_bundles(Dealer& dealer, const lowlevel::ToyParams& params, std::size_t calls) {
    if (calls == 0) return fail("BRIDGE_PARAMS: calls must be positive");
    if (params.N < 1 || params.shift() < 1 || params.shift() > BRIDGE_L - 1) return fail("BRIDGE_PARAMS: toy parameters out of range");
    return dealer.make_trunc(static_cast<std::size_t>(params.N) * calls, params.shift(), BRIDGE_L, "h2a_ref");
}

Expected<lowlevel::Ciphertext> a2h_toy(Party& party, const std::vector<u64>& share, const lowlevel::ToyBFV& bfv,
                                       const lowlevel::Ciphertext* pk_or_null, lowlevel::RandomSource* rng_or_null) {
    const std::size_t N = static_cast<std::size_t>(bfv.params().N);
    if (share.size() > N) return fail("BRIDGE_SHAPE: share length " + std::to_string(share.size()) + " exceeds N = " + std::to_string(N));
    // every role check happens before the first message
    if (party.party() == 1) {
        if (!pk_or_null) return fail("BRIDGE_PARTY: P1 needs its public key to encrypt");
        if (!rng_or_null || !*rng_or_null) return fail("BRIDGE_PARTY: P1 needs a randomness source to encrypt");
    } else if (party.party() != 0) {
        return fail("BRIDGE_PARTY: party must be 0 or 1");
    }
    // local reduction mod t (t divides 2^64, so it commutes with the additive sharing)
    Expected<lowlevel::Poly> x_t = bfv.encode_vector(share);
    if (!x_t.ok) return fail(x_t.code);
    Party::Track tr(party, "a2h_toy", {share.size(), N});
    Channel& ch = party.channel();
    if (party.party() == 1) {
        Expected<lowlevel::Ciphertext> ct = bfv.encrypt(*pk_or_null, x_t.value, *rng_or_null);
        if (!ct.ok) return fail(ct.code);
        std::vector<u64> flat;
        flat.reserve(2 * N);
        flat.insert(flat.end(), ct.value.c0.begin(), ct.value.c0.end());
        flat.insert(flat.end(), ct.value.c1.begin(), ct.value.c1.end());
        OC2_CHECK(ch.send(u64_array(flat, {2, N})));   // the one message of A2H_toy: P1 -> P0
        ch.clear_pending();                            // the round is counted explicitly on both sides
        party.count_round();
        return ok(lowlevel::Ciphertext{});             // the keyholder keeps no ciphertext
    }
    OC2_TRY(msg, ch.recv());
    party.count_round();
    OC2_TRY(arr, msg.as_array());
    if (arr.dtype != DType::u64 || arr.dims.size() != 2 || arr.dims[0] != 2 || arr.dims[1] != N || arr.data.size() != 16 * N)
        return fail("BRIDGE_SHAPE: expected a (2, " + std::to_string(N) + ") uint64 ciphertext frame");
    const std::vector<u64> flat = arr.as_u64();
    lowlevel::Ciphertext ct;
    ct.c0.assign(flat.begin(), flat.begin() + static_cast<std::ptrdiff_t>(N));
    ct.c1.assign(flat.begin() + static_cast<std::ptrdiff_t>(N), flat.end());
    Expected<lowlevel::Ciphertext> sum = bfv.add_pt(ct, x_t.value);   // adds Delta * (x0 mod t) into c0
    if (!sum.ok) return fail(sum.code);
    return ok(std::move(sum.value));
}

Expected<lowlevel::ToyBFV::MatvecResult> eval_toy(lowlevel::ToyBFV& bfv, const lowlevel::Ciphertext& ct, const lowlevel::Matrix& W) {
    Expected<lowlevel::ToyBFV::MatvecResult> r = bfv.matvec_pt(ct, W);
    if (!r.ok) return fail(r.code);
    return ok(std::move(r.value));
}

Expected<std::vector<u64>> h2a_ref(Party& party, const lowlevel::ToyBFV& bfv, const lowlevel::Ciphertext* ct_or_null,
                                   const lowlevel::Poly* sk_or_null, const std::vector<int64_t>& positions, Convolution conv) {
    const lowlevel::ToyParams& params = bfv.params();
    const std::size_t N = static_cast<std::size_t>(params.N);
    const int shift = params.shift();
    if (!is_pow2(N)) return fail("BRIDGE_PARAMS: N must be a power of two");
    if (shift < 1 || shift > BRIDGE_L - 1) return fail("BRIDGE_PARAMS: need 1 <= 64 - log_t <= 63");
    for (int64_t p : positions)
        if (p < 0 || static_cast<std::size_t>(p) >= N) return fail("BRIDGE_SHAPE: positions must lie in [0, " + std::to_string(N) + ")");
    // Role checks before any message. [c1] = (c1, 0) and [s] = (0, s) are additive sharings.
    std::vector<u64> sa, sb;
    if (party.party() == 0) {
        if (!ct_or_null) return fail("BRIDGE_PARTY: P0 must supply the evaluated ciphertext");
        if (sk_or_null) return fail("BRIDGE_PARTY: P0 must not hold the secret key");
        if (ct_or_null->c0.size() != N || ct_or_null->c1.size() != N)
            return fail("BRIDGE_SHAPE: ciphertext components must have length N = " + std::to_string(N));
        sa = ct_or_null->c1;
        sb.assign(N, 0);
    } else if (party.party() == 1) {
        if (ct_or_null) return fail("BRIDGE_PARTY: the keyholder P1 must never receive the evaluated ciphertext");
        if (!sk_or_null) return fail("BRIDGE_PARTY: P1 must supply its secret key");
        if (sk_or_null->size() != N) return fail("BRIDGE_SHAPE: secret key must have length N = " + std::to_string(N));
        sa.assign(N, 0);
        sb = *sk_or_null;
    } else {
        return fail("BRIDGE_PARTY: party must be 0 or 1");
    }
    Party::Track tr(party, "h2a_ref", {N, positions.size()});
    OC2_TRY(prod, negacyclic_mul_shared(party, sa, sb, conv));   // round 1: shares of c1 * s
    if (party.party() == 0) {
        const u64 half_delta = params.delta() / 2;
        for (std::size_t i = 0; i < N; ++i) prod[i] += ct_or_null->c0[i] + half_delta;   // local: c0 + Delta/2
    }
    // round 2: faithful truncation by 64 - log_t bits over the whole ring (L = 64)
    OC2_TRY(z, party.trunc_faithful(Mat::row_vector(std::move(prod)), shift, BRIDGE_L));
    if (z.size() != N) return fail("BRIDGE_SHAPE: truncation returned an unexpected shape");
    std::vector<u64> out(positions.size());
    for (std::size_t k = 0; k < positions.size(); ++k) out[k] = z.a[static_cast<std::size_t>(positions[k])];
    return ok(std::move(out));
}

}  // namespace bridge
}  // namespace highlevel
}  // namespace opencrab2
