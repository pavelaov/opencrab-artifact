// Implementation of gadgets.hpp (polynomial contracts, references) and of the
// nonlinear Party methods (paper, Section 5.4): clipped polynomial activations,
// exp_neg, recip, rsqrt, softmax_rows, layernorm_rows.
#include <opencrab2/highlevel/gadgets.hpp>
#include <opencrab2/highlevel/party.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>

namespace opencrab2 {
namespace highlevel {

// ---------------------------------------------------------------------------
// least squares
// ---------------------------------------------------------------------------
Expected<std::vector<double>> lstsq(const std::vector<std::vector<double>>& A, const std::vector<double>& y) {
    size_t n = A.size();
    if (n == 0) return fail("LSTSQ_SHAPE: empty system");
    size_t m = A[0].size();
    if (m == 0 || n < m || y.size() != n) return fail("LSTSQ_SHAPE: need n >= m > 0");
    // Householder QR on a copy (column-major work array).
    std::vector<double> R(n * m);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < m; ++j) R[j * n + i] = A[i][j];
    std::vector<double> b = y;
    for (size_t j = 0; j < m; ++j) {
        double norm = 0.0;
        for (size_t i = j; i < n; ++i) norm += R[j * n + i] * R[j * n + i];
        norm = std::sqrt(norm);
        if (norm == 0.0) return fail("LSTSQ_RANK: rank deficient");
        double alpha = R[j * n + j] > 0 ? -norm : norm;
        std::vector<double> v(n - j);
        for (size_t i = j; i < n; ++i) v[i - j] = R[j * n + i];
        v[0] -= alpha;
        double vnorm = 0.0;
        for (double x : v) vnorm += x * x;
        if (vnorm == 0.0) continue;
        // apply H = I - 2 v v^T / (v^T v) to the remaining columns and to b
        for (size_t c = j; c < m; ++c) {
            double dot = 0.0;
            for (size_t i = j; i < n; ++i) dot += v[i - j] * R[c * n + i];
            double s = 2.0 * dot / vnorm;
            for (size_t i = j; i < n; ++i) R[c * n + i] -= s * v[i - j];
        }
        double dotb = 0.0;
        for (size_t i = j; i < n; ++i) dotb += v[i - j] * b[i];
        double sb = 2.0 * dotb / vnorm;
        for (size_t i = j; i < n; ++i) b[i] -= sb * v[i - j];
    }
    // back substitution R c = b (upper m x m)
    std::vector<double> c(m, 0.0);
    for (size_t jj = m; jj-- > 0;) {
        double s = b[jj];
        for (size_t k = jj + 1; k < m; ++k) s -= R[k * n + jj] * c[k];
        double d = R[jj * n + jj];
        if (std::fabs(d) < 1e-300) return fail("LSTSQ_RANK: zero pivot");
        c[jj] = s / d;
    }
    return ok(std::move(c));
}

// ---------------------------------------------------------------------------
// polynomial contracts
// ---------------------------------------------------------------------------
double PolyContract::evaluate(double x) const {
    double u = affine ? (2.0 * (x - lo) / (hi - lo) - 1.0) : x / scale;
    double v = even_in_u ? u * u : u;
    double acc = 0.0;
    for (size_t i = coeffs.size(); i-- > 0;) acc = acc * v + coeffs[i];
    return acc;
}

Value PolyContract::to_value() const {
    List cs;
    for (double c : coeffs) cs.push_back(Value(c));
    return dict({{"name", name},
                 {"coeffs", Value(cs)},
                 {"lo", lo},
                 {"hi", hi},
                 {"scale", scale},
                 {"affine", affine},
                 {"even_in_u", even_in_u},
                 {"max_abs_err", max_abs_err},
                 {"degree", static_cast<int64_t>(degree)}});
}

Expected<PolyContract> fit_poly(const std::string& name, const std::function<double(double)>& f, double lo, double hi,
                                int degree, bool even, bool affine, int nodes) {
    if (!(hi > lo) || degree < 1 || nodes <= degree) return fail("APPROX_PARAM: bad fit parameters for " + name);
    std::vector<double> u(static_cast<size_t>(nodes)), x(static_cast<size_t>(nodes));
    double scale = 1.0;
    for (int k = 0; k < nodes; ++k) u[static_cast<size_t>(k)] = std::cos((2.0 * k + 1.0) * 3.141592653589793 / (2.0 * nodes));
    if (affine) {
        for (int k = 0; k < nodes; ++k) x[static_cast<size_t>(k)] = lo + (u[static_cast<size_t>(k)] + 1.0) * 0.5 * (hi - lo);
    } else {
        scale = std::max(std::fabs(lo), std::fabs(hi));
        if (lo >= 0) {
            for (int k = 0; k < nodes; ++k) u[static_cast<size_t>(k)] = (u[static_cast<size_t>(k)] + 1.0) * 0.5 * (hi / scale);
        }
        for (int k = 0; k < nodes; ++k) x[static_cast<size_t>(k)] = u[static_cast<size_t>(k)] * scale;
    }
    size_t ncoef = even ? static_cast<size_t>(degree / 2 + 1) : static_cast<size_t>(degree + 1);
    std::vector<std::vector<double>> V(static_cast<size_t>(nodes), std::vector<double>(ncoef));
    std::vector<double> y(static_cast<size_t>(nodes));
    for (int k = 0; k < nodes; ++k) {
        double var = even ? u[static_cast<size_t>(k)] * u[static_cast<size_t>(k)] : u[static_cast<size_t>(k)];
        double p = 1.0;
        for (size_t j = 0; j < ncoef; ++j) {
            V[static_cast<size_t>(k)][j] = p;
            p *= var;
        }
        y[static_cast<size_t>(k)] = f(x[static_cast<size_t>(k)]);
    }
    OC2_TRY(c, lstsq(V, y));
    PolyContract pc;
    pc.name = name;
    pc.coeffs = c;
    pc.lo = lo;
    pc.hi = hi;
    pc.scale = scale;
    pc.affine = affine;
    pc.even_in_u = even;
    pc.degree = degree;
    double err = 0.0;
    const int G = 20001;
    for (int g = 0; g < G; ++g) {
        double xg = lo + (hi - lo) * static_cast<double>(g) / (G - 1);
        err = std::max(err, std::fabs(pc.evaluate(xg) - f(xg)));
    }
    pc.max_abs_err = err;
    return ok(std::move(pc));
}

double gelu_ref(double x) { return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0))); }
double silu_ref(double x) { return x / (1.0 + std::exp(-x)); }

namespace {
struct ContractTable {
    std::map<std::string, PolyContract> table;
    std::string error;
};
const ContractTable& contract_table() {
    static ContractTable tbl = []() {
        ContractTable t;
        struct Spec {
            const char* name;
            std::function<double(double)> f;
            double lo, hi;
            int degree;
            bool even, affine;
        };
        std::vector<Spec> specs = {
            {"gelu_clip12", gelu_ref, -5.0, 5.0, 12, false, false},
            {"silu_clip12", silu_ref, -8.0, 8.0, 12, false, false},
            {"tanh_clip", [](double x) { return std::tanh(x); }, -4.0, 4.0, 9, false, false},
            {"recip_norm", [](double v) { return 1.0 / v; }, 0.5, 1.0, 5, false, true},
            {"rsqrt_norm", [](double v) { return 1.0 / std::sqrt(v); }, 0.25, 1.0, 5, false, true},
        };
        for (const auto& s : specs) {
            auto r = fit_poly(s.name, s.f, s.lo, s.hi, s.degree, s.even, s.affine);
            if (!r.ok) {
                t.error = r.code;
                break;
            }
            t.table[s.name] = r.value;
        }
        return t;
    }();
    return tbl;
}
}  // namespace

Expected<const PolyContract*> poly_contract_by_name(const std::string& name) {
    const ContractTable& t = contract_table();
    if (!t.error.empty()) return fail(t.error);
    auto it = t.table.find(name);
    if (it == t.table.end()) return fail("APPROX_UNKNOWN_CONTRACT: " + name);
    return ok(&it->second);
}

Value poly_contract_table() {
    Dict d;
    for (const auto& kv : contract_table().table) d.push_back(DictEntry{kv.first, kv.second.to_value()});
    return Value(d);
}

std::vector<double> softmax_ref(const std::vector<double>& row, const std::vector<uint8_t>* mask) {
    size_t n = row.size();
    std::vector<double> out(n, 0.0);
    double m = -1e300;
    for (size_t j = 0; j < n; ++j)
        if (!mask || (*mask)[j]) m = std::max(m, row[j]);
    double s = 0.0;
    for (size_t j = 0; j < n; ++j) {
        if (mask && !(*mask)[j]) continue;
        out[j] = std::exp(row[j] - m);
        s += out[j];
    }
    for (size_t j = 0; j < n; ++j) out[j] /= s;
    return out;
}

std::vector<double> layernorm_ref(const std::vector<double>& row, const std::vector<double>* gamma,
                                  const std::vector<double>* beta, double eps, bool rms) {
    size_t d = row.size();
    double mu = 0.0;
    if (!rms) {
        for (double v : row) mu += v;
        mu /= static_cast<double>(d);
    }
    double var = 0.0;
    for (double v : row) var += (v - mu) * (v - mu);
    var /= static_cast<double>(d);
    double rs = 1.0 / std::sqrt(var + eps);
    std::vector<double> out(d);
    for (size_t j = 0; j < d; ++j) {
        double y = (row[j] - mu) * rs;
        if (gamma) y *= (*gamma)[j];
        if (beta) y += (*beta)[j];
        out[j] = y;
    }
    return out;
}

size_t argmax_ref(const std::vector<double>& row) {
    size_t best = 0;
    for (size_t j = 1; j < row.size(); ++j)
        if (row[j] > row[best]) best = j;
    return best;
}

// ---------------------------------------------------------------------------
// Party: polynomials
// ---------------------------------------------------------------------------
Expected<Mat> Party::poly_horner(const Mat& u, const std::vector<double>& coeffs) {
    if (coeffs.empty()) return fail("APPROX_PARAM: empty coefficient list");
    Mat acc = const_mat(coeffs.back(), u.rows, u.cols);
    for (size_t i = coeffs.size() - 1; i-- > 0;) {
        OC2_TRY(prod, mul_fxp(acc, u));
        acc = add_public_real(prod, coeffs[i]);
    }
    return ok(std::move(acc));
}

Expected<Mat> Party::poly_contract(const Mat& x, const std::string& name) {
    OC2_TRY(c, poly_contract_by_name(name));
    Mat u;
    if (c->affine) {
        double kk = 2.0 / (c->hi - c->lo);
        u = mul_public_real(x, kk);
        u = add_public_real(u, -(2.0 * c->lo / (c->hi - c->lo) + 1.0));
    } else {
        u = c->scale != 1.0 ? mul_public_real(x, 1.0 / c->scale) : x;
    }
    if (c->even_in_u) {
        OC2_TRY(sq, square_fxp(u));
        u = std::move(sq);
    }
    return poly_horner(u, c->coeffs);
}

// 4.1 clipped polynomial activation: b_lo = 1{x >= -R}, b_hi = 1{x >= R},
// act = b_hi x + (b_lo - b_hi) p(x / R). Rounds 1 + degree + 1.
Expected<Mat> Party::clip_poly_act(const Mat& x, const std::string& name) {
    OC2_TRY(c, poly_contract_by_name(name));
    double R = c->hi;
    Track tr(*this, name, {x.rows, x.cols});
    Mat both = Mat::vstack(add_public_real(x, R), add_public_real(x, -R));
    OC2_TRY(bits, drelu(both));
    Mat b_lo = bits.rows_range(0, x.rows), b_hi = bits.rows_range(x.rows, 2 * x.rows);
    Mat u = mul_public_real(x, 1.0 / R);
    OC2_TRY(p, poly_horner(u, c->coeffs));
    Mat w = ring_sub(b_lo, b_hi);
    if (name.rfind("tanh", 0) == 0) {
        OC2_TRY(wp, beaver_mul(w, p));
        Mat act = ring_add(wp, ring_mul_const(ring_add(b_hi, b_lo), enc(1.0)));
        return ok(add_public_real(act, -1.0));
    }
    OC2_TRY(prods, beaver_mul_batch({b_hi, w}, {x, p}));
    return ok(ring_add(prods[0], prods[1]));
}

Expected<Mat> Party::gelu(const Mat& x) { return clip_poly_act(x, contract_.gelu); }
Expected<Mat> Party::silu(const Mat& x) { return clip_poly_act(x, contract_.silu); }
Expected<Mat> Party::tanh(const Mat& x) { return clip_poly_act(x, "tanh_clip"); }

// 4.2 exp on (-inf, 0]: clamp at -Z, base 1 + w + w^2/2, r squarings at scale 2^(f+e).
Expected<Mat> Party::exp_neg(const Mat& z) {
    double Z = static_cast<double>(contract_.exp_Z);
    int r = contract_.exp_r, e = contract_.exp_extra_bits, f = contract_.frac_bits;
    int order = contract_.exp_order;
    if (r < e) return fail("EXP_PARAMS: need r >= extra bits");
    Track tr(*this, "exp_neg", {z.rows, z.cols});
    Mat zZ = add_public_real(z, Z);
    OC2_TRY(b, drelu(zZ));
    OC2_TRY(bz, beaver_mul(b, zZ));
    Mat zc = add_public_real(bz, -Z);                 // max(z, -Z)
    Mat w = trunc_local(zc, r - e);                   // zc 2^-r at scale 2^(f+e)
    Mat t = add_public(w, Mat(z.rows, z.cols, static_cast<uint64_t>(1) << (f + e)));
    if (order >= 2) {
        OC2_TRY(ww, beaver_mul(w, w));
        t = ring_add(t, trunc_local(ww, f + e + 1));  // + w^2 / 2
    }
    for (int i = 0; i < r; ++i) {
        OC2_TRY(tt, beaver_mul(t, t));
        t = trunc_local(tt, f + e);
    }
    return ok(trunc_local(t, e));
}

// Batched DReLU(S - T_j); thresholds beyond the DCF bound are answered 0 (contract).
Expected<std::vector<Mat>> Party::threshold_bits(const Mat& S, const std::vector<double>& thresholds, std::vector<bool>& live) {
    double bound = std::ldexp(1.0, contract_.dcf_domain_bits - 1 - contract_.frac_bits);
    live.assign(thresholds.size(), false);
    std::vector<size_t> idx;
    for (size_t j = 0; j < thresholds.size(); ++j)
        if (thresholds[j] < bound) {
            live[j] = true;
            idx.push_back(j);
        }
    std::vector<Mat> out(thresholds.size());
    if (idx.empty()) return ok(std::move(out));
    Mat stack(idx.size() * S.rows, S.cols);
    for (size_t q = 0; q < idx.size(); ++q) {
        Mat s = add_public_real(S, -thresholds[idx[q]]);
        std::copy(s.a.begin(), s.a.end(), stack.a.begin() + static_cast<std::ptrdiff_t>(q * S.size()));
    }
    OC2_TRY(bits, drelu(stack));
    for (size_t q = 0; q < idx.size(); ++q) out[idx[q]] = bits.rows_range(q * S.rows, (q + 1) * S.rows);
    return ok(std::move(out));
}

// 4.3 normalised reciprocal for S in [1, 2^(K+1)); output at scale 2^(f + extra_bits).
Expected<Mat> Party::recip(const Mat& S, int extra_bits) {
    int K = contract_.recip_K;
    int newton = contract_.recip_newton;
    Track tr(*this, "recip", {S.rows, S.cols});
    std::vector<double> th;
    for (int j = 1; j <= K; ++j) th.push_back(std::ldexp(1.0, j));
    std::vector<bool> live;
    OC2_TRY(bits, threshold_bits(S, th, live));
    Mat s = const_mat(0.5, S.rows, S.cols);
    for (int j = 1; j <= K; ++j) {
        if (!live[static_cast<size_t>(j - 1)]) continue;
        s = ring_sub(s, ring_mul_const(bits[static_cast<size_t>(j - 1)], enc(std::ldexp(1.0, -(j + 1)))));
    }
    OC2_TRY(v, mul_fxp(S, s));  // v in [0.5, 1)
    OC2_TRY(q0, poly_contract(v, "recip_norm"));
    Mat q = std::move(q0);
    for (int i = 0; i < newton; ++i) {
        OC2_TRY(vq, mul_fxp(v, q));
        OC2_TRY(qn, mul_fxp(q, add_public_real(ring_neg(vq), 2.0)));
        q = std::move(qn);
    }
    return mul_fxp(q, s, contract_.frac_bits - extra_bits);
}

// 4.4 normalised inverse square root for V in [4^j0, 4^(j0+m+1)); s2 held at scale 2^(f+e2),
// e2 = max(0, 2 (j0+1+m) - f) (rsqrt_scale_bits).
int rsqrt_scale_bits(const NumericContract& c) {
    int e2 = 2 * (c.rsqrt_j0 + 1 + c.rsqrt_m) - c.frac_bits;
    return e2 > 0 ? e2 : 0;
}
int rsqrt_product_bits(const NumericContract& c) { return 2 * c.frac_bits + rsqrt_scale_bits(c); }

Expected<Mat> Party::rsqrt(const Mat& V, int extra_bits) {
    int j0 = contract_.rsqrt_j0, m = contract_.rsqrt_m, newton = contract_.rsqrt_newton, f = contract_.frac_bits;
    Track tr(*this, "rsqrt", {V.rows, V.cols});
    std::vector<double> th;
    for (int i = 1; i <= m; ++i) th.push_back(std::pow(4.0, j0 + i));
    std::vector<bool> live;
    OC2_TRY(bits, threshold_bits(V, th, live));
    // s2 = 4^-(j0+1+k) is an exact integer at scale 2^(f+e2) as soon as f + e2 >= 2 (j0+1+m). The
    // smallest such e2 keeps the Beaver product V * s2 = v * 2^(2f+e2) (v < 1) inside the declared
    // bound of the local truncation: 2^34 ring units under NC-v2, failure <= 2^-29 per element.
    // (e2 = f made it a 2^48 product: a wrong row with probability 2^-16.6; paper, Appendix D,
    // Disclosure.)
    int e2 = rsqrt_scale_bits(contract_);
    double sc2 = std::ldexp(1.0, f + e2);
    Mat s2(V.rows, V.cols, party_ == 0 ? static_cast<uint64_t>(std::llround(std::pow(4.0, -(j0 + 1)) * sc2)) : 0);
    Mat s1 = const_mat(std::ldexp(1.0, -(j0 + 1)), V.rows, V.cols);
    for (int i = 1; i <= m; ++i) {
        if (!live[static_cast<size_t>(i - 1)]) continue;
        uint64_t k2 = static_cast<uint64_t>(std::llround(std::pow(4.0, -(j0 + 1)) * 3.0 * std::pow(4.0, -i) * sc2));
        uint64_t k1 = static_cast<uint64_t>(std::llround(std::ldexp(1.0, -(j0 + 1)) * std::ldexp(1.0, -i) * std::ldexp(1.0, f)));
        s2 = ring_sub(s2, ring_mul_const(bits[static_cast<size_t>(i - 1)], k2));
        s1 = ring_sub(s1, ring_mul_const(bits[static_cast<size_t>(i - 1)], k1));
    }
    OC2_TRY(v, mul_fxp(V, s2, f + e2));  // v in [0.25, 1)
    OC2_TRY(q0, poly_contract(v, "rsqrt_norm"));
    Mat q = std::move(q0);
    for (int i = 0; i < newton; ++i) {
        OC2_TRY(vq, mul_fxp(v, q));
        OC2_TRY(vqq, mul_fxp(vq, q));
        Mat h = add_public_real(ring_neg(trunc_local(vqq, 1)), 1.5);
        OC2_TRY(qn, mul_fxp(q, h));
        q = std::move(qn);
    }
    return mul_fxp(q, s1, f - extra_bits);
}

// 4.5 row softmax with a public mask (1 = attend); masked positions get exactly 0.
// The row maximum is the fused k-ary tournament of party.cpp (2 ceil(log_k n) rounds). Precondition
// (the magnitude condition of paper, Section 5.3; not checked at run time because a check on secret values would
// leak): the scores of a row, masked positions included, differ by less than 2^(n-1) ring units
// for k_max = 2 and by less than 2^(n-2-ceil(log2 g)) inside a fused node of g >= 3 members
// (2^12 real units for k_max = 4 at n = 32, f = 16). Admission enforces it from the declared
// score bound (candidates.hpp, code KARY_MAGNITUDE).
Expected<Mat> Party::softmax_rows(const Mat& x, const std::vector<uint8_t>* mask, int k_max) {
    size_t n = x.cols, R = x.rows;
    Track tr(*this, "softmax_rows_k" + std::to_string(k_max), {R, n});
    if (mask && mask->size() != x.size()) return fail("SOFTMAX_MASK: mask shape differs from the input");
    OC2_TRY(mx, max_rows(x, mask, k_max));
    Mat z(R, n);
    uint64_t negZ = party_ == 0 ? enc(-static_cast<double>(contract_.exp_Z)) : 0;
    for (size_t r = 0; r < R; ++r)
        for (size_t j = 0; j < n; ++j) {
            bool valid = !mask || (*mask)[r * n + j];
            z.a[r * n + j] = valid ? x.a[r * n + j] - mx.a[r] : negZ;
        }
    OC2_TRY(t, exp_neg(z));
    if (mask)
        for (size_t i = 0; i < t.size(); ++i)
            if (!(*mask)[i]) t.a[i] = 0;
    Mat S(R, 1);
    for (size_t r = 0; r < R; ++r) {
        uint64_t s = 0;
        for (size_t j = 0; j < n; ++j) s += t.a[r * n + j];
        S.a[r] = s;
    }
    int e = std::min(static_cast<int>(std::ceil(std::log2(static_cast<double>(std::max<size_t>(n, 2))))), 10);
    OC2_TRY(inv, recip(S, e));
    Mat invb(R, n);
    for (size_t r = 0; r < R; ++r)
        for (size_t j = 0; j < n; ++j) invb.a[r * n + j] = inv.a[r];
    return mul_fxp(t, invb, contract_.frac_bits + e);
}

// 4.6 LayerNorm / RMSNorm over the last axis; gamma / beta are shares (1 x d) or null.
Expected<Mat> Party::layernorm_rows(const Mat& x, const Mat* gamma, const Mat* beta, double eps, bool rms) {
    size_t d = x.cols, R = x.rows;
    int extra = contract_.layernorm_extra_bits;
    Track tr(*this, rms ? "rmsnorm_rows" : "layernorm_rows", {R, d});
    Mat dx = x;
    if (!rms) {
        Mat sum(R, 1);
        for (size_t r = 0; r < R; ++r) {
            uint64_t s = 0;
            for (size_t j = 0; j < d; ++j) s += x.a[r * d + j];
            sum.a[r] = s;
        }
        Mat mu = mul_public_real(sum, 1.0 / static_cast<double>(d));
        for (size_t r = 0; r < R; ++r)
            for (size_t j = 0; j < d; ++j) dx.a[r * d + j] = x.a[r * d + j] - mu.a[r];
    }
    bool faithful = contract_.trunc_square_ln == "faithful";
    OC2_TRY(sq, square_fxp(dx, faithful));
    Mat ssum(R, 1);
    for (size_t r = 0; r < R; ++r) {
        uint64_t s = 0;
        for (size_t j = 0; j < d; ++j) s += sq.a[r * d + j];
        ssum.a[r] = s;
    }
    Mat var = add_public_real(mul_public_real(ssum, 1.0 / static_cast<double>(d)), eps);
    OC2_TRY(rs, rsqrt(var, extra));  // scale 2^(f + extra)
    Mat rsb(R, d);
    for (size_t r = 0; r < R; ++r)
        for (size_t j = 0; j < d; ++j) rsb.a[r * d + j] = rs.a[r];
    OC2_TRY(y, mul_fxp(dx, rsb, contract_.frac_bits + extra));
    if (gamma) {
        if (gamma->size() != d) return fail("LAYERNORM_SHAPE: gamma size");
        Mat gb(R, d);
        for (size_t r = 0; r < R; ++r)
            for (size_t j = 0; j < d; ++j) gb.a[r * d + j] = gamma->a[j];
        OC2_TRY(yg, mul_fxp(y, gb));
        y = std::move(yg);
    }
    if (beta) {
        if (beta->size() != d) return fail("LAYERNORM_SHAPE: beta size");
        for (size_t r = 0; r < R; ++r)
            for (size_t j = 0; j < d; ++j) y.a[r * d + j] += beta->a[j];
    }
    return ok(std::move(y));
}

}  // namespace highlevel
}  // namespace opencrab2
