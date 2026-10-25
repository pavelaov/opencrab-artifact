// OpenCrab v2: tests of the toy BFV reference scheme (paper, Section 5.2).
//
// Checks: parameter refusals, negacyclic product identities, 1000 encrypt/decrypt round trips per
// admitted N (with homomorphic ct+ct, ct+pt, ct*pt spot checks against an integer reference),
// round trips at log_t = 1 and log_t = 32, the decrypt_arith_steps decomposition, coefficient
// packed matvec equals W x mod t at (N=128, d_in=8, d_out=8) and (N=512, d_in=64, d_out=8) for
// log_t = 8 and 16, 1000 random matvec trials at each of the two admitted configurations (log_t = 16)
// with the noise bound and the signed-floor identity that H2A_ref relies on, blocked matvec for
// d_out > N / d_in, the noise bound refusing N=512, log_t=32 with 2^31 weights, and the A2H_toy
// local reduction pattern. The two-party secure bridge (a2h_toy, eval_toy, h2a_ref over Party)
// is tested in tests/test_bridge.cpp.
// Build: python -m ziglang c++ -std=c++17 -O2 -I include src/lowlevel/toy_bfv.cpp tests/test_toy_bfv.cpp -o test_toy_bfv
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "opencrab2/lowlevel/toy_bfv.hpp"

using namespace opencrab2;
using namespace opencrab2::lowlevel;

static int g_checks = 0;

static void check(bool cond, const char* what) {
    ++g_checks;
    if (!cond) {
        std::printf("FAIL check %d: %s\n", g_checks, what);
        std::exit(1);
    }
}

static bool starts_with(const std::string& s, const char* prefix) { return s.rfind(prefix, 0) == 0; }

// Integer reference of a negacyclic product modulo 2^64 using the same wrap arithmetic (independent loop order).
static Poly reference_negacyclic(const Poly& a, const Poly& b) {
    const std::size_t N = a.size();
    Poly out(N, 0);
    for (std::size_t k = 0; k < N; ++k) {
        std::uint64_t acc = 0;
        for (std::size_t i = 0; i < N; ++i) {
            const std::size_t j = (k + N - i) % N;   // i + j = k or k + N
            const std::uint64_t term = a[i] * b[j];
            if (i + j == k) acc += term;
            else acc -= term;
        }
        out[k] = acc;
    }
    return out;
}

static std::vector<std::uint64_t> random_vector(RandomSource& rng, std::size_t len, std::uint64_t t) {
    std::vector<std::uint64_t> v(len);
    for (auto& x : v) x = rng() % t;
    return v;
}

static void test_params_and_poly() {
    check(!ToyParams::make(100, 16).ok, "N = 100 refused");
    check(starts_with(ToyParams::make(100, 16).code, "TOY_BFV_PARAM"), "N refusal code");
    check(!ToyParams::make(128, 33).ok, "log_t = 33 refused");
    check(!ToyParams::make(128, 0).ok, "log_t = 0 refused");
    check(!ToyParams::make(128, 8, 0).ok, "B_e = 0 refused");
    Expected<ToyParams> p = ToyParams::make(256, 12);
    check(p.ok, "N = 256, log_t = 12 accepted");
    check(p.value.t() == (1ULL << 12), "t = 2^log_t");
    check(p.value.delta() == (1ULL << 52), "Delta = 2^(64 - log_t)");
    check(p.value.shift() == 52, "shift = 64 - log_t");

    // X^(N-1) * X = X^N = -1 mod (X^N + 1)
    const std::size_t N = 64;
    Poly a(N, 0), b(N, 0);
    a[N - 1] = 1;
    b[1] = 1;
    Poly c = negacyclic_mul(a, b);
    check(c[0] == ~0ULL, "X^(N-1) * X = -1");
    for (std::size_t i = 1; i < N; ++i) check(c[i] == 0, "X^(N-1) * X other coefficients zero");

    RandomSource rng = deterministic_source(7);
    for (int trial = 0; trial < 5; ++trial) {
        Poly x(N), y(N), z(N);
        for (std::size_t i = 0; i < N; ++i) { x[i] = rng(); y[i] = rng(); z[i] = rng(); }
        Poly xy = negacyclic_mul(x, y);
        check(xy == reference_negacyclic(x, y), "negacyclic product matches reference");
        check(xy == negacyclic_mul(y, x), "negacyclic product commutes");
        Poly yz(N);
        for (std::size_t i = 0; i < N; ++i) yz[i] = y[i] + z[i];
        Poly lhs = negacyclic_mul(x, yz);
        Poly xz = negacyclic_mul(x, z);
        bool dist = true;
        for (std::size_t i = 0; i < N; ++i) dist = dist && (lhs[i] == xy[i] + xz[i]);
        check(dist, "negacyclic product distributes");
    }
}

static void test_round_trips(int N) {
    Expected<ToyParams> p = ToyParams::make(N, 16);
    check(p.ok, "params");
    ToyBFV bfv(p.value);
    RandomSource rng = deterministic_source(1000 + static_cast<std::uint64_t>(N));
    ToyBFV::KeyPair kp = bfv.keygen(rng);
    const std::uint64_t t = p.value.t();
    int ok_round = 0;
    for (int trial = 0; trial < 1000; ++trial) {
        std::vector<std::uint64_t> m = random_vector(rng, static_cast<std::size_t>(N), t);
        Expected<Ciphertext> ct = bfv.encrypt(kp.pk, m, rng);
        if (!ct.ok) break;
        Expected<Poly> dec = bfv.decrypt(kp.sk, ct.value);
        if (!dec.ok || dec.value != m) break;
        ++ok_round;
        if (trial % 10 == 0) {
            // ct + ct
            std::vector<std::uint64_t> m2 = random_vector(rng, static_cast<std::size_t>(N), t);
            Expected<Ciphertext> ct2 = bfv.encrypt(kp.pk, m2, rng);
            check(ct2.ok, "second encryption");
            Expected<Ciphertext> sum = bfv.add_ct(ct.value, ct2.value);
            check(sum.ok, "add_ct");
            Expected<Poly> dsum = bfv.decrypt(kp.sk, sum.value);
            check(dsum.ok, "decrypt sum");
            bool okb = true;
            for (int i = 0; i < N; ++i) okb = okb && (dsum.value[static_cast<std::size_t>(i)] == ((m[static_cast<std::size_t>(i)] + m2[static_cast<std::size_t>(i)]) & (t - 1)));
            check(okb, "ct + ct decrypts to m1 + m2 mod t");
            // ct + pt
            Expected<Ciphertext> apt = bfv.add_pt(ct.value, m2);
            check(apt.ok, "add_pt");
            Expected<Poly> dapt = bfv.decrypt(kp.sk, apt.value);
            check(dapt.ok && dapt.value == dsum.value, "ct + pt decrypts to m1 + m2 mod t");
            // ct * pt with a sparse small plaintext polynomial (noise stays far below Delta/2)
            Poly pp(static_cast<std::size_t>(N), 0);
            pp[0] = 3;
            pp[static_cast<std::size_t>(N / 2)] = 5;
            const std::int64_t before = bfv.ct_pt_products();
            Expected<Ciphertext> prod = bfv.mul_pt(ct.value, pp);
            check(prod.ok, "mul_pt");
            check(bfv.ct_pt_products() == before + 1, "ct-pt product counted");
            Expected<Poly> dprod = bfv.decrypt(kp.sk, prod.value);
            check(dprod.ok, "decrypt product");
            Poly ref = reference_negacyclic(m, pp);
            bool okp = true;
            for (int i = 0; i < N; ++i) okp = okp && (dprod.value[static_cast<std::size_t>(i)] == (ref[static_cast<std::size_t>(i)] & (t - 1)));
            check(okp, "ct * pt decrypts to m * p mod t");
        }
    }
    check(ok_round == 1000, "1000 encrypt/decrypt round trips");
    std::printf("  N=%d: %d round trips ok\n", N, ok_round);

    // extreme plaintext moduli
    for (int log_t : {1, 32}) {
        Expected<ToyParams> pe = ToyParams::make(N, log_t);
        check(pe.ok, "extreme log_t params");
        ToyBFV bfv2(pe.value);
        ToyBFV::KeyPair kp2 = bfv2.keygen(rng);
        for (int trial = 0; trial < 20; ++trial) {
            std::vector<std::uint64_t> m = random_vector(rng, static_cast<std::size_t>(N), pe.value.t());
            Expected<Ciphertext> ct = bfv2.encrypt(kp2.pk, m, rng);
            check(ct.ok, "encrypt at extreme log_t");
            Expected<Poly> dec = bfv2.decrypt(kp2.sk, ct.value);
            check(dec.ok && dec.value == m, "round trip at extreme log_t");
        }
    }

    // refusals
    std::vector<std::uint64_t> bad(static_cast<std::size_t>(N), 0);
    bad[0] = t;
    Expected<Ciphertext> e1 = bfv.encrypt(kp.pk, bad, rng);
    check(!e1.ok && starts_with(e1.code, "TOY_BFV_PLAINTEXT_RANGE"), "plaintext >= t refused");
    Ciphertext short_ct;
    short_ct.c0.assign(static_cast<std::size_t>(N - 1), 0);
    short_ct.c1.assign(static_cast<std::size_t>(N), 0);
    Expected<Ciphertext> e2 = bfv.add_ct(short_ct, short_ct);
    check(!e2.ok && starts_with(e2.code, "TOY_BFV_SHAPE"), "short ciphertext refused");
}

static void test_arith_steps() {
    Expected<ToyParams> p = ToyParams::make(128, 12);
    ToyBFV bfv(p.value);
    RandomSource rng = deterministic_source(99);
    ToyBFV::KeyPair kp = bfv.keygen(rng);
    std::vector<std::uint64_t> m = random_vector(rng, 128, p.value.t());
    Expected<Ciphertext> ct = bfv.encrypt(kp.pk, m, rng);
    Expected<DecryptSteps> st = bfv.decrypt_arith_steps(kp.sk, ct.value);
    check(st.ok, "decrypt_arith_steps");
    check(st.value.shift == 52 && st.value.half_delta == (1ULL << 51), "shift and half delta");
    check(st.value.plaintext == m, "steps plaintext equals m");
    const std::uint64_t bound = bfv.fresh_noise_bound();
    bool ok = true;
    for (std::size_t i = 0; i < 128; ++i) {
        ok = ok && (st.value.phase[i] == ct.value.c0[i] + st.value.c1_s[i]);
        ok = ok && (st.value.pre_round[i] == st.value.phase[i] + st.value.half_delta);
        ok = ok && ((st.value.pre_round[i] >> 52) == st.value.plaintext[i]);
        const std::int64_t nz = st.value.noise_signed[i];
        ok = ok && (nz <= static_cast<std::int64_t>(bound)) && (nz >= -static_cast<std::int64_t>(bound));
    }
    check(ok, "phase, pre_round, rounding and centred noise consistent");
    check(st.value.c1_s == negacyclic_mul(ct.value.c1, kp.sk), "c1_s is the negacyclic product");
}

static void test_matvec(int N, int d_in, int d_out, int log_t, std::uint64_t seed) {
    Expected<ToyParams> p = ToyParams::make(N, log_t);
    check(p.ok, "matvec params");
    ToyBFV bfv(p.value);
    RandomSource rng = deterministic_source(seed);
    ToyBFV::KeyPair kp = bfv.keygen(rng);
    const std::uint64_t t = p.value.t();
    Matrix W(static_cast<std::size_t>(d_out));
    for (auto& row : W) row = random_vector(rng, static_cast<std::size_t>(d_in), t);
    std::vector<std::uint64_t> x = random_vector(rng, static_cast<std::size_t>(d_in), t);
    Expected<bool> nb = bfv.noise_budget_ok(d_in, bfv.w_inf(W));
    check(nb.ok && nb.value, "noise budget admits the parameter set");
    Expected<Poly> px = bfv.encode_vector(x);
    check(px.ok, "encode_vector");
    Expected<Ciphertext> ct_x = bfv.encrypt(kp.pk, px.value, rng);
    check(ct_x.ok, "encrypt packed x");
    const std::int64_t before = bfv.ct_pt_products();
    Expected<ToyBFV::MatvecResult> mv = bfv.matvec_pt(ct_x.value, W);
    check(mv.ok, "matvec_pt");
    check(bfv.ct_pt_products() == before + 1, "one ct-pt product per block");
    Expected<Poly> dec = bfv.decrypt(kp.sk, mv.value.ct_y);
    check(dec.ok, "decrypt matvec result");
    Expected<std::vector<std::uint64_t>> ref = bfv.matvec_reference(W, x);
    check(ref.ok, "matvec reference");
    bool ok = static_cast<int>(mv.value.positions.size()) == d_out;
    for (int k = 0; ok && k < d_out; ++k) {
        const std::int64_t pos = mv.value.positions[static_cast<std::size_t>(k)];
        ok = ok && (pos == static_cast<std::int64_t>(k + 1) * d_in - 1);
        ok = ok && (dec.value[static_cast<std::size_t>(pos)] == ref.value[static_cast<std::size_t>(k)]);
    }
    check(ok, "matvec equals W x mod t at the packed positions");
    std::printf("  matvec N=%d d_in=%d d_out=%d log_t=%d ok\n", N, d_in, d_out, log_t);
}

// 1000 random trials per configuration (fresh W, x and encryption
// randomness under one key). Besides W x mod t this checks, on every coefficient of every trial,
// the identity the secure bridge relies on (highlevel/bridge.hpp, H2A_ref): the faithful
// truncation gate with L = 64 returns the SIGNED floor of the pre-rounding value by 64 - log_t
// bits, and that value is congruent modulo t to the plaintext coefficient. The noise bound is
// checked per trial and the centred noise of the evaluated ciphertext is compared with it.
static void test_matvec_trials(int N, int d_in, int d_out, int log_t, int trials, std::uint64_t seed) {
    Expected<ToyParams> p = ToyParams::make(N, log_t);
    check(p.ok, "trial params");
    ToyBFV bfv(p.value);
    RandomSource rng = deterministic_source(seed);
    ToyBFV::KeyPair kp = bfv.keygen(rng);
    const std::uint64_t t = p.value.t();
    const int shift = p.value.shift();
    long mismatches = 0, identity_failures = 0, noise_failures = 0;
    std::uint64_t worst_noise = 0;
    std::uint64_t worst_bound = 0;
    for (int trial = 0; trial < trials; ++trial) {
        Matrix W(static_cast<std::size_t>(d_out));
        for (auto& row : W) row = random_vector(rng, static_cast<std::size_t>(d_in), t);
        std::vector<std::uint64_t> x(static_cast<std::size_t>(d_in));
        for (auto& v : x) v = rng();   // full-ring A64 values; encode_vector reduces mod t
        Expected<bool> nb = bfv.noise_budget_ok(d_in, bfv.w_inf(W));
        Expected<std::uint64_t> bound = bfv.noise_bound(d_in, bfv.w_inf(W));
        if (!(nb.ok && nb.value && bound.ok)) {
            ++noise_failures;
            continue;
        }
        if (bound.value > worst_bound) worst_bound = bound.value;
        Expected<Poly> px = bfv.encode_vector(x);
        Expected<Ciphertext> ct_x = bfv.encrypt(kp.pk, px.value, rng);
        Expected<ToyBFV::MatvecResult> mv = bfv.matvec_pt(ct_x.value, W);
        Expected<DecryptSteps> st = bfv.decrypt_arith_steps(kp.sk, mv.value.ct_y);
        Expected<std::vector<std::uint64_t>> ref = bfv.matvec_reference(W, x);
        if (!(px.ok && ct_x.ok && mv.ok && st.ok && ref.ok)) {
            ++mismatches;
            continue;
        }
        for (int k = 0; k < d_out; ++k) {
            const std::size_t pos = static_cast<std::size_t>(mv.value.positions[static_cast<std::size_t>(k)]);
            if (st.value.plaintext[pos] != ref.value[static_cast<std::size_t>(k)]) ++mismatches;
        }
        for (std::size_t i = 0; i < static_cast<std::size_t>(N); ++i) {
            // signed floor by `shift` bits without relying on a signed right shift
            const std::uint64_t pre = st.value.pre_round[i];
            const std::uint64_t unsigned_floor = pre >> shift;
            const std::uint64_t signed_floor = (pre >> 63) ? unsigned_floor - (std::uint64_t(1) << (64 - shift)) : unsigned_floor;
            if ((signed_floor & (t - 1)) != st.value.plaintext[i]) ++identity_failures;
            const std::int64_t nz = st.value.noise_signed[i];
            const std::uint64_t mag = nz < 0 ? static_cast<std::uint64_t>(-(nz + 1)) + 1 : static_cast<std::uint64_t>(nz);
            if (mag > worst_noise) worst_noise = mag;
            if (mag > bound.value) ++noise_failures;
        }
    }
    check(mismatches == 0, "matvec equals W x mod t in every trial");
    check(identity_failures == 0, "signed floor by 64 - log_t bits is the plaintext modulo t on every coefficient");
    check(noise_failures == 0, "noise bound admitted and respected in every trial");
    check(worst_noise < p.value.delta() / 2, "observed noise below Delta / 2");
    std::printf("  matvec trials N=%d d_in=%d d_out=%d log_t=%d: %d trials, 0 mismatches, worst |noise| = %llu <= bound %llu < Delta/2 = %llu\n", N, d_in,
                d_out, log_t, trials, static_cast<unsigned long long>(worst_noise), static_cast<unsigned long long>(worst_bound),
                static_cast<unsigned long long>(p.value.delta() / 2));
}

static void test_matvec_blocks_and_refusal() {
    // blocked: N = 128, d_in = 8 -> 16 rows per ciphertext; d_out = 40 -> 3 blocks
    Expected<ToyParams> p = ToyParams::make(128, 8);
    ToyBFV bfv(p.value);
    RandomSource rng = deterministic_source(5);
    ToyBFV::KeyPair kp = bfv.keygen(rng);
    const std::uint64_t t = p.value.t();
    Matrix W(40);
    for (auto& row : W) row = random_vector(rng, 8, t);
    std::vector<std::uint64_t> x = random_vector(rng, 8, t);
    Expected<Poly> px = bfv.encode_vector(x);
    Expected<Ciphertext> ct_x = bfv.encrypt(kp.pk, px.value, rng);
    Expected<ToyBFV::MatvecResult> too_big = bfv.matvec_pt(ct_x.value, W);
    check(!too_big.ok && starts_with(too_big.code, "TOY_BFV_SHAPE"), "d_out > N / d_in refused by matvec_pt");
    Expected<std::vector<ToyBFV::MatvecBlock>> blocks = bfv.matvec_pt_blocks(ct_x.value, W);
    check(blocks.ok && blocks.value.size() == 3, "three blocks for 40 rows of width 8 at N = 128");
    Expected<std::vector<std::uint64_t>> ref = bfv.matvec_reference(W, x);
    bool ok = true;
    for (const auto& b : blocks.value) {
        Expected<Poly> dec = bfv.decrypt(kp.sk, b.ct_y);
        ok = ok && dec.ok;
        for (int k = b.row_lo; ok && k < b.row_hi; ++k) {
            const std::int64_t pos = b.positions[static_cast<std::size_t>(k - b.row_lo)];
            ok = ok && (dec.value[static_cast<std::size_t>(pos)] == ref.value[static_cast<std::size_t>(k)]);
        }
    }
    check(ok, "blocked matvec equals W x mod t");

    // refusal: N = 512, log_t = 32, weights of magnitude 2^31
    Expected<ToyParams> pr = ToyParams::make(512, 32);
    ToyBFV big(pr.value);
    Expected<bool> nb = big.noise_budget_ok(64, 1ULL << 31);
    check(nb.ok && !nb.value, "noise bound refuses N=512, log_t=32, w_inf=2^31");
    Expected<std::uint64_t> bound = big.noise_bound(64, 1ULL << 31);
    check(bound.ok && bound.value >= pr.value.delta() / 2, "bound at least Delta/2");
    ToyBFV::KeyPair kb = big.keygen(rng);
    Matrix Wb(8, std::vector<std::uint64_t>(64, 1ULL << 31));
    std::vector<std::uint64_t> xb(64, 1);
    Expected<Poly> pxb = big.encode_vector(xb);
    Expected<Ciphertext> ctb = big.encrypt(kb.pk, pxb.value, rng);
    Expected<ToyBFV::MatvecResult> mvb = big.matvec_pt(ctb.value, Wb);
    check(!mvb.ok && starts_with(mvb.code, "TOY_BFV_NOISE_BUDGET"), "matvec_pt fails closed on the noise budget");
    // the same shape with small weights is admitted
    Expected<bool> nb2 = big.noise_budget_ok(64, 255);
    check(nb2.ok && nb2.value, "small weights admitted at log_t = 32");
}

static void test_a2h_local() {
    Expected<ToyParams> p = ToyParams::make(256, 16);
    ToyBFV bfv(p.value);
    RandomSource rng = deterministic_source(31);
    ToyBFV::KeyPair kp = bfv.keygen(rng);
    const std::size_t d = 100;
    std::vector<std::uint64_t> x0(d), x1(d);
    for (std::size_t i = 0; i < d; ++i) { x0[i] = rng(); x1[i] = rng(); }   // full 64-bit A64 shares
    Expected<Ciphertext> ct = a2h_toy_local(bfv, kp.pk, x1, x0, rng);
    check(ct.ok, "a2h_toy_local");
    Expected<Poly> dec = bfv.decrypt(kp.sk, ct.value);
    check(dec.ok, "decrypt a2h result");
    bool ok = true;
    for (std::size_t i = 0; i < d; ++i) ok = ok && (dec.value[i] == ((x0[i] + x1[i]) & (p.value.t() - 1)));
    for (std::size_t i = d; i < 256; ++i) ok = ok && (dec.value[i] == 0);
    check(ok, "A2H_toy encrypts x mod t with zero padding");
}

static void test_os_random() {
    RandomSource os = os_random_source();
    const std::uint64_t a = os(), b = os();
    check(a != b, "OS randomness yields distinct words");
    std::uint64_t acc = 0;
    for (int i = 0; i < 1024; ++i) acc |= os();
    check(acc != 0, "OS randomness is not all zero");
}

int main() {
    test_params_and_poly();
    test_os_random();
    for (int N : {64, 128, 256, 512}) test_round_trips(N);
    test_arith_steps();
    test_matvec(128, 8, 8, 8, 11);
    test_matvec(128, 8, 8, 16, 12);
    test_matvec(512, 64, 8, 8, 13);
    test_matvec(512, 64, 8, 16, 14);
    test_matvec_trials(128, 8, 8, 16, 1000, 21);
    test_matvec_trials(512, 64, 8, 16, 1000, 22);
    test_matvec_blocks_and_refusal();
    test_a2h_local();
    std::printf("PASS %d checks\n", g_checks);
    return 0;
}
