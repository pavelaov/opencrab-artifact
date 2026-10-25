// OpenCrab v2: toy BFV implementation (see include/opencrab2/lowlevel/toy_bfv.hpp).
#include "opencrab2/lowlevel/toy_bfv.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>

#ifdef _WIN32
#include <windows.h>
// BCryptGenRandom lives in bcrypt.dll. It is resolved at run time through kernel32 (always linked)
// so that no build needs an extra -lbcrypt flag; an unavailable generator fails closed.
#else
#include <fstream>
#endif

namespace opencrab2 {
namespace lowlevel {

namespace {

struct SplitMix {
    std::uint64_t s;
    std::uint64_t next() {
        std::uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
};

// Buffered OS randomness.
struct OsBuffer {
    std::uint8_t buf[4096];
    std::size_t pos = sizeof(buf);
    bool refill() {
#ifdef _WIN32
        typedef LONG(WINAPI * GenRandomFn)(PVOID, PUCHAR, ULONG, ULONG);
        static GenRandomFn fn = nullptr;
        if (!fn) {
            HMODULE h = LoadLibraryA("bcrypt.dll");
            if (!h) return false;
            fn = reinterpret_cast<GenRandomFn>(reinterpret_cast<void*>(GetProcAddress(h, "BCryptGenRandom")));
            if (!fn) return false;
        }
        const ULONG use_system_preferred_rng = 0x00000002;   // BCRYPT_USE_SYSTEM_PREFERRED_RNG
        const LONG st = fn(nullptr, buf, static_cast<ULONG>(sizeof(buf)), use_system_preferred_rng);
        if (st != 0) return false;
#else
        std::ifstream in("/dev/urandom", std::ios::binary);
        if (!in.read(reinterpret_cast<char*>(buf), sizeof(buf))) return false;
#endif
        pos = 0;
        return true;
    }
    std::uint64_t next() {
        if (pos + 8 > sizeof(buf)) {
            if (!refill()) {
                // Fail closed: an unreadable OS generator must not silently degrade to a weak
                // source. Abort the process; the caller cannot continue securely.
                std::fprintf(stderr, "OS_RANDOM_UNAVAILABLE: BCryptGenRandom or /dev/urandom failed\n");
                std::abort();
            }
        }
        std::uint64_t v = 0;
        for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(buf[pos + static_cast<std::size_t>(i)]) << (8 * i);
        pos += 8;
        return v;
    }
};

// Uniform in [lo, hi] (inclusive) for small signed ranges, via rejection on 64-bit words.
std::int64_t uniform_signed(RandomSource& rng, std::int64_t lo, std::int64_t hi) {
    const std::uint64_t span = static_cast<std::uint64_t>(hi - lo + 1);
    // Unbiased rejection sampling; when span divides 2^64 every word is accepted (no wrapped limit).
    const std::uint64_t rem = (std::uint64_t(0) - span) % span;   // 2^64 mod span
    std::uint64_t v = rng();
    if (rem != 0) {
        const std::uint64_t limit = std::uint64_t(0) - rem;
        while (v >= limit) v = rng();
    }
    return lo + static_cast<std::int64_t>(v % span);
}

inline std::uint64_t to_u64(std::int64_t v) { return static_cast<std::uint64_t>(v); }

}  // namespace

RandomSource deterministic_source(std::uint64_t seed) {
    auto st = std::make_shared<SplitMix>();
    st->s = seed;
    return [st]() { return st->next(); };
}

RandomSource os_random_source() {
    auto st = std::make_shared<OsBuffer>();
    return [st]() { return st->next(); };
}

Expected<ToyParams> ToyParams::make(int N, int log_t, int B_e) {
    if (!(N == 64 || N == 128 || N == 256 || N == 512))
        return Expected<ToyParams>::failure("TOY_BFV_PARAM: N must be one of 64, 128, 256, 512");
    if (log_t < 1 || log_t > kMaxLogT) return Expected<ToyParams>::failure("TOY_BFV_PARAM: log_t must satisfy 1 <= log_t <= 32");
    if (B_e < 1) return Expected<ToyParams>::failure("TOY_BFV_PARAM: B_e must be >= 1");
    ToyParams p;
    p.N = N;
    p.log_t = log_t;
    p.B_e = B_e;
    return Expected<ToyParams>::success(p);
}

Poly negacyclic_mul(const Poly& a, const Poly& b) {
    const std::size_t N = a.size();
    Poly out(N, 0);
    if (b.size() != N) return out;   // callers validate shapes; keep the function total
    for (std::size_t i = 0; i < N; ++i) {
        const std::uint64_t ai = a[i];
        if (ai == 0) continue;
        // k = i + j < N: add; k >= N: subtract at k - N
        const std::size_t split = N - i;
        for (std::size_t j = 0; j < split; ++j) out[i + j] += ai * b[j];
        for (std::size_t j = split; j < N; ++j) out[i + j - N] -= ai * b[j];
    }
    return out;
}

ToyBFV::ToyBFV(ToyParams params) : params_(params) {}

Poly ToyBFV::ternary(RandomSource& rng) const {
    Poly v(static_cast<std::size_t>(params_.N));
    for (auto& c : v) c = to_u64(uniform_signed(rng, -1, 1));
    return v;
}

Poly ToyBFV::error(RandomSource& rng) const {
    Poly v(static_cast<std::size_t>(params_.N));
    for (auto& c : v) c = to_u64(uniform_signed(rng, -params_.B_e, params_.B_e));
    return v;
}

Poly ToyBFV::uniform(RandomSource& rng) const {
    Poly v(static_cast<std::size_t>(params_.N));
    for (auto& c : v) c = rng();
    return v;
}

Status ToyBFV::check_pt(const Poly& m) const {
    if (m.size() != static_cast<std::size_t>(params_.N)) return fail_status("TOY_BFV_SHAPE: plaintext must have length N");
    const std::uint64_t t = params_.t();
    for (std::uint64_t c : m)
        if (c >= t) return fail_status("TOY_BFV_PLAINTEXT_RANGE: coefficient >= t");
    return ok_status();
}

Status ToyBFV::check_ct(const Ciphertext& ct) const {
    if (ct.c0.size() != static_cast<std::size_t>(params_.N) || ct.c1.size() != static_cast<std::size_t>(params_.N))
        return fail_status("TOY_BFV_SHAPE: ciphertext components must have length N");
    return ok_status();
}

ToyBFV::KeyPair ToyBFV::keygen(RandomSource& rng) const {
    KeyPair kp;
    kp.sk = ternary(rng);
    const Poly a = uniform(rng);
    const Poly e = error(rng);
    Poly as = negacyclic_mul(a, kp.sk);
    Poly pk0(as.size());
    for (std::size_t i = 0; i < as.size(); ++i) pk0[i] = std::uint64_t(0) - (as[i] + e[i]);
    kp.pk.c0 = std::move(pk0);
    kp.pk.c1 = a;
    return kp;
}

Expected<Ciphertext> ToyBFV::encrypt(const Ciphertext& pk, const Poly& m, RandomSource& rng) const {
    Status st = check_ct(pk);
    if (!st.ok) return Expected<Ciphertext>::failure(st.code);
    st = check_pt(m);
    if (!st.ok) return Expected<Ciphertext>::failure(st.code);
    const Poly u = ternary(rng);
    const Poly e0 = error(rng);
    const Poly e1 = error(rng);
    Ciphertext ct;
    ct.c0 = negacyclic_mul(pk.c0, u);
    ct.c1 = negacyclic_mul(pk.c1, u);
    const std::uint64_t delta = params_.delta();
    for (std::size_t i = 0; i < ct.c0.size(); ++i) {
        ct.c0[i] += e0[i] + m[i] * delta;
        ct.c1[i] += e1[i];
    }
    return Expected<Ciphertext>::success(std::move(ct));
}

Expected<DecryptSteps> ToyBFV::decrypt_arith_steps(const Poly& sk, const Ciphertext& ct) const {
    Status st = check_ct(ct);
    if (!st.ok) return Expected<DecryptSteps>::failure(st.code);
    if (sk.size() != static_cast<std::size_t>(params_.N)) return Expected<DecryptSteps>::failure("TOY_BFV_SHAPE: sk must have length N");
    DecryptSteps d;
    d.shift = params_.shift();
    d.half_delta = params_.delta() / 2;
    d.c1_s = negacyclic_mul(ct.c1, sk);
    const std::size_t N = d.c1_s.size();
    d.phase.resize(N);
    d.pre_round.resize(N);
    d.plaintext.resize(N);
    d.noise_signed.resize(N);
    const std::uint64_t delta = params_.delta();
    for (std::size_t i = 0; i < N; ++i) {
        d.phase[i] = ct.c0[i] + d.c1_s[i];
        d.pre_round[i] = d.phase[i] + d.half_delta;
        d.plaintext[i] = d.pre_round[i] >> d.shift;
        d.noise_signed[i] = static_cast<std::int64_t>(d.phase[i] - d.plaintext[i] * delta);
    }
    return Expected<DecryptSteps>::success(std::move(d));
}

Expected<Poly> ToyBFV::decrypt(const Poly& sk, const Ciphertext& ct) const {
    Expected<DecryptSteps> d = decrypt_arith_steps(sk, ct);
    if (!d.ok) return Expected<Poly>::failure(d.code);
    return Expected<Poly>::success(std::move(d.value.plaintext));
}

Expected<Ciphertext> ToyBFV::add_ct(const Ciphertext& a, const Ciphertext& b) const {
    Status st = check_ct(a);
    if (!st.ok) return Expected<Ciphertext>::failure(st.code);
    st = check_ct(b);
    if (!st.ok) return Expected<Ciphertext>::failure(st.code);
    Ciphertext out;
    out.c0.resize(a.c0.size());
    out.c1.resize(a.c1.size());
    for (std::size_t i = 0; i < a.c0.size(); ++i) {
        out.c0[i] = a.c0[i] + b.c0[i];
        out.c1[i] = a.c1[i] + b.c1[i];
    }
    return Expected<Ciphertext>::success(std::move(out));
}

Expected<Ciphertext> ToyBFV::add_pt(const Ciphertext& ct, const Poly& m) const {
    Status st = check_ct(ct);
    if (!st.ok) return Expected<Ciphertext>::failure(st.code);
    st = check_pt(m);
    if (!st.ok) return Expected<Ciphertext>::failure(st.code);
    Ciphertext out = ct;
    const std::uint64_t delta = params_.delta();
    for (std::size_t i = 0; i < out.c0.size(); ++i) out.c0[i] += m[i] * delta;
    return Expected<Ciphertext>::success(std::move(out));
}

Expected<Ciphertext> ToyBFV::mul_pt(const Ciphertext& ct, const Poly& p) {
    Status st = check_ct(ct);
    if (!st.ok) return Expected<Ciphertext>::failure(st.code);
    if (p.size() != static_cast<std::size_t>(params_.N)) return Expected<Ciphertext>::failure("TOY_BFV_SHAPE: p must have length N");
    ++ct_pt_products_;
    Ciphertext out;
    out.c0 = negacyclic_mul(ct.c0, p);
    out.c1 = negacyclic_mul(ct.c1, p);
    return Expected<Ciphertext>::success(std::move(out));
}

Expected<Poly> ToyBFV::encode_vector(const std::vector<std::uint64_t>& x) const {
    if (x.size() > static_cast<std::size_t>(params_.N)) return Expected<Poly>::failure("TOY_BFV_SHAPE: vector length exceeds N");
    const std::uint64_t mask = params_.t() - 1;   // t is a power of two
    Poly p(static_cast<std::size_t>(params_.N), 0);
    for (std::size_t i = 0; i < x.size(); ++i) p[i] = x[i] & mask;
    return Expected<Poly>::success(std::move(p));
}

std::uint64_t ToyBFV::signed_mod_t(std::uint64_t v) const {
    const std::uint64_t t = params_.t();
    const std::uint64_t r = v & (t - 1);
    if (r >= t / 2 && t > 1) return r - t;   // wraps to the two's complement of a negative value
    return r;
}

std::uint64_t ToyBFV::w_inf(const Matrix& W) const {
    std::uint64_t best = 0;
    for (const auto& row : W) {
        for (std::uint64_t v : row) {
            const std::int64_t s = static_cast<std::int64_t>(signed_mod_t(v));
            const std::uint64_t a = s < 0 ? static_cast<std::uint64_t>(-(s + 1)) + 1 : static_cast<std::uint64_t>(s);
            if (a > best) best = a;
        }
    }
    return best;
}

Expected<int> ToyBFV::rows_per_ct(int d_in) const {
    if (d_in < 1 || d_in > params_.N) return Expected<int>::failure("TOY_BFV_SHAPE: d_in must satisfy 1 <= d_in <= N");
    return Expected<int>::success(params_.N / d_in);
}

Expected<std::pair<Poly, std::vector<std::int64_t>>> ToyBFV::encode_weight_block(const Matrix& W_block) const {
    using R = Expected<std::pair<Poly, std::vector<std::int64_t>>>;
    if (W_block.empty()) return R::failure("TOY_BFV_SHAPE: weight block has no rows");
    const int d_in = static_cast<int>(W_block[0].size());
    for (const auto& row : W_block)
        if (static_cast<int>(row.size()) != d_in) return R::failure("TOY_BFV_SHAPE: ragged weight block");
    Expected<int> rows = rows_per_ct(d_in);
    if (!rows.ok) return R::failure(rows.code);
    const int r = static_cast<int>(W_block.size());
    if (r > rows.value) return R::failure("TOY_BFV_SHAPE: rows exceed N / d_in rows per ciphertext");
    Poly p(static_cast<std::size_t>(params_.N), 0);
    std::vector<std::int64_t> positions(static_cast<std::size_t>(r));
    for (int k = 0; k < r; ++k) {
        for (int j = 0; j < d_in; ++j) p[static_cast<std::size_t>(k * d_in + j)] = signed_mod_t(W_block[static_cast<std::size_t>(k)][static_cast<std::size_t>(d_in - 1 - j)]);
        positions[static_cast<std::size_t>(k)] = static_cast<std::int64_t>(k + 1) * d_in - 1;
    }
    return R::success(std::make_pair(std::move(p), std::move(positions)));
}

std::uint64_t ToyBFV::fresh_noise_bound() const {
    return static_cast<std::uint64_t>(params_.B_e) * (2ULL * static_cast<std::uint64_t>(params_.N) + 1ULL);
}

Expected<std::uint64_t> ToyBFV::noise_bound(int d_in, std::uint64_t w_inf) const {
    Expected<int> rows = rows_per_ct(d_in);
    if (!rows.ok) return Expected<std::uint64_t>::failure(rows.code);
    const std::uint64_t nnz = static_cast<std::uint64_t>(rows.value) * static_cast<std::uint64_t>(d_in);
    const std::uint64_t fresh = fresh_noise_bound();
    const std::uint64_t maxv = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t acc = fresh;
    if (nnz != 0 && acc > maxv / nnz) return Expected<std::uint64_t>::success(maxv);
    acc *= nnz;
    if (w_inf != 0 && acc > maxv / w_inf) return Expected<std::uint64_t>::success(maxv);
    acc *= w_inf;
    return Expected<std::uint64_t>::success(acc);
}

Expected<bool> ToyBFV::noise_budget_ok(int d_in, std::uint64_t w_inf) const {
    Expected<std::uint64_t> b = noise_bound(d_in, w_inf);
    if (!b.ok) return Expected<bool>::failure(b.code);
    return Expected<bool>::success(b.value < params_.delta() / 2);
}

Expected<ToyBFV::MatvecResult> ToyBFV::matvec_pt(const Ciphertext& ct_x, const Matrix& W) {
    using R = Expected<MatvecResult>;
    Status st = check_ct(ct_x);
    if (!st.ok) return R::failure(st.code);
    if (W.empty() || W[0].empty()) return R::failure("TOY_BFV_SHAPE: W must be a non-empty matrix");
    const int d_in = static_cast<int>(W[0].size());
    Expected<int> rows = rows_per_ct(d_in);
    if (!rows.ok) return R::failure(rows.code);
    if (static_cast<int>(W.size()) > rows.value) return R::failure("TOY_BFV_SHAPE: d_out exceeds N / d_in; use matvec_pt_blocks");
    const std::uint64_t winf = w_inf(W);
    Expected<bool> okb = noise_budget_ok(d_in, winf);
    if (!okb.ok) return R::failure(okb.code);
    if (!okb.value) return R::failure("TOY_BFV_NOISE_BUDGET: worst-case noise reaches Delta/2 for this N, log_t, d_in, w_inf");
    Expected<std::pair<Poly, std::vector<std::int64_t>>> enc = encode_weight_block(W);
    if (!enc.ok) return R::failure(enc.code);
    Expected<Ciphertext> ct_y = mul_pt(ct_x, enc.value.first);
    if (!ct_y.ok) return R::failure(ct_y.code);
    MatvecResult res;
    res.ct_y = std::move(ct_y.value);
    res.positions = std::move(enc.value.second);
    return R::success(std::move(res));
}

Expected<std::vector<ToyBFV::MatvecBlock>> ToyBFV::matvec_pt_blocks(const Ciphertext& ct_x, const Matrix& W) {
    using R = Expected<std::vector<MatvecBlock>>;
    if (W.empty() || W[0].empty()) return R::failure("TOY_BFV_SHAPE: W must be a non-empty matrix");
    const int d_in = static_cast<int>(W[0].size());
    Expected<int> rows = rows_per_ct(d_in);
    if (!rows.ok) return R::failure(rows.code);
    std::vector<MatvecBlock> out;
    const int d_out = static_cast<int>(W.size());
    for (int lo = 0; lo < d_out; lo += rows.value) {
        const int hi = std::min(d_out, lo + rows.value);
        Matrix block(W.begin() + lo, W.begin() + hi);
        Expected<MatvecResult> r = matvec_pt(ct_x, block);
        if (!r.ok) return R::failure(r.code);
        MatvecBlock b;
        b.ct_y = std::move(r.value.ct_y);
        b.positions = std::move(r.value.positions);
        b.row_lo = lo;
        b.row_hi = hi;
        out.push_back(std::move(b));
    }
    return R::success(std::move(out));
}

Expected<std::vector<std::uint64_t>> ToyBFV::matvec_reference(const Matrix& W, const std::vector<std::uint64_t>& x) const {
    using R = Expected<std::vector<std::uint64_t>>;
    const std::uint64_t mask = params_.t() - 1;
    std::vector<std::uint64_t> y;
    y.reserve(W.size());
    for (const auto& row : W) {
        if (row.size() != x.size()) return R::failure("TOY_BFV_SHAPE: W columns must match x length");
        std::uint64_t acc = 0;
        for (std::size_t j = 0; j < row.size(); ++j) acc += row[j] * x[j];   // wraps mod 2^64, reduced mod t below
        y.push_back(acc & mask);
    }
    return R::success(std::move(y));
}

Expected<Ciphertext> a2h_toy_local(const ToyBFV& bfv, const Ciphertext& pk, const std::vector<std::uint64_t>& x1_share,
                                   const std::vector<std::uint64_t>& x0_share, RandomSource& rng) {
    Expected<Poly> m1 = bfv.encode_vector(x1_share);
    if (!m1.ok) return Expected<Ciphertext>::failure(m1.code);
    Expected<Ciphertext> ct1 = bfv.encrypt(pk, m1.value, rng);
    if (!ct1.ok) return ct1;
    Expected<Poly> m0 = bfv.encode_vector(x0_share);
    if (!m0.ok) return Expected<Ciphertext>::failure(m0.code);
    return bfv.add_pt(ct1.value, m0.value);
}

}  // namespace lowlevel
}  // namespace opencrab2
