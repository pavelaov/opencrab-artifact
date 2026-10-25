// Implementation of party.hpp: Mat helpers, ReleasePolicy, the Party core
// (sharing, opening, public ops, truncation, Beaver products, matrix products,
// linear layers, install_wmask, comparison layer, k-ary tournaments,
// ring_to_field) and the per-gadget counters. Nonlinear gadgets: gadgets.cpp.
#include <opencrab2/highlevel/party.hpp>

#include <opencrab2/lowlevel/ring.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace opencrab2 {
namespace highlevel {

// ---------------------------------------------------------------------------
// Mat
// ---------------------------------------------------------------------------
Mat Mat::from_vector(std::vector<uint64_t> v, size_t rows, size_t cols) {
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.a = std::move(v);
    m.a.resize(rows * cols);
    return m;
}

Mat Mat::row(size_t i) const { return rows_range(i, i + 1); }

Mat Mat::rows_range(size_t lo, size_t hi) const {
    Mat m(hi - lo, cols);
    std::copy(a.begin() + static_cast<std::ptrdiff_t>(lo * cols), a.begin() + static_cast<std::ptrdiff_t>(hi * cols), m.a.begin());
    return m;
}

Mat Mat::cols_range(size_t lo, size_t hi) const {
    Mat m(rows, hi - lo);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = lo; j < hi; ++j) m.a[i * (hi - lo) + (j - lo)] = a[i * cols + j];
    return m;
}

Mat Mat::transpose() const {
    Mat m(cols, rows);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j) m.a[j * rows + i] = a[i * cols + j];
    return m;
}

Mat Mat::reshape(size_t r, size_t c) const {
    Mat m = *this;
    m.rows = r;
    m.cols = c;
    m.a.resize(r * c);
    return m;
}

Mat Mat::vstack(const Mat& x, const Mat& y) {
    Mat m(x.rows + y.rows, x.cols);
    std::copy(x.a.begin(), x.a.end(), m.a.begin());
    std::copy(y.a.begin(), y.a.end(), m.a.begin() + static_cast<std::ptrdiff_t>(x.a.size()));
    return m;
}

Mat Mat::hstack(const Mat& x, const Mat& y) {
    Mat m(x.rows, x.cols + y.cols);
    for (size_t i = 0; i < x.rows; ++i) {
        std::copy(x.a.begin() + static_cast<std::ptrdiff_t>(i * x.cols), x.a.begin() + static_cast<std::ptrdiff_t>((i + 1) * x.cols),
                  m.a.begin() + static_cast<std::ptrdiff_t>(i * m.cols));
        std::copy(y.a.begin() + static_cast<std::ptrdiff_t>(i * y.cols), y.a.begin() + static_cast<std::ptrdiff_t>((i + 1) * y.cols),
                  m.a.begin() + static_cast<std::ptrdiff_t>(i * m.cols + x.cols));
    }
    return m;
}

Mat Mat::concat_flat(const std::vector<const Mat*>& ms) {
    size_t total = 0;
    for (const Mat* m : ms) total += m->size();
    Mat out(1, total);
    size_t at = 0;
    for (const Mat* m : ms) {
        std::copy(m->a.begin(), m->a.end(), out.a.begin() + static_cast<std::ptrdiff_t>(at));
        at += m->size();
    }
    return out;
}

Mat ring_add(const Mat& a, const Mat& b) {
    Mat m(a.rows, a.cols);
    for (size_t i = 0; i < a.size(); ++i) m.a[i] = a.a[i] + b.a[i];
    return m;
}

Mat ring_sub(const Mat& a, const Mat& b) {
    Mat m(a.rows, a.cols);
    for (size_t i = 0; i < a.size(); ++i) m.a[i] = a.a[i] - b.a[i];
    return m;
}

Mat ring_neg(const Mat& a) {
    Mat m(a.rows, a.cols);
    for (size_t i = 0; i < a.size(); ++i) m.a[i] = 0 - a.a[i];
    return m;
}

Mat ring_mul(const Mat& a, const Mat& b) {
    Mat m(a.rows, a.cols);
    for (size_t i = 0; i < a.size(); ++i) m.a[i] = a.a[i] * b.a[i];
    return m;
}

Mat ring_mul_const(const Mat& a, uint64_t c) {
    Mat m(a.rows, a.cols);
    for (size_t i = 0; i < a.size(); ++i) m.a[i] = a.a[i] * c;
    return m;
}

Mat ring_matmul(const Mat& A, const Mat& B) {
    Mat C(A.rows, B.cols);
    ring_matmul(A.a, A.rows, A.cols, B.a, B.cols, C.a);
    return C;
}

// ---------------------------------------------------------------------------
// ReleasePolicy
// ---------------------------------------------------------------------------
ReleasePolicy::ReleasePolicy(int recipient, std::vector<std::string> kinds, int max_count)
    : declared_(true), recipient_(recipient), kinds_(std::move(kinds)), max_count_(max_count) {}

Expected<Unit> ReleasePolicy::authorize(const std::string& kind) {
    if (!declared_) return fail("UNAUTHORIZED_RELEASE: no ReleasePolicy declared");
    if (recipient_ != 1) return fail("UNAUTHORIZED_RELEASE: only P1 may receive model-derived values");
    if (std::find(kinds_.begin(), kinds_.end(), kind) == kinds_.end())
        return fail("UNAUTHORIZED_RELEASE: kind " + kind + " not declared");
    if (count_ >= max_count_) return fail("UNAUTHORIZED_RELEASE: release count exhausted");
    count_ += 1;
    return ok_unit();
}

Value ReleasePolicy::to_value() const {
    List k;
    for (const auto& s : kinds_) k.push_back(Value(s));
    return dict({{"recipient", static_cast<int64_t>(recipient_)},
                 {"kinds", Value(k)},
                 {"max_count", static_cast<int64_t>(max_count_)},
                 {"count", static_cast<int64_t>(count_)}});
}

Value GadgetStat::to_value() const {
    List sh;
    for (size_t s : shape) sh.push_back(Value(static_cast<int64_t>(s)));
    return dict({{"kind", kind},
                 {"seconds", seconds},
                 {"bytes", static_cast<int64_t>(bytes)},
                 {"rounds", static_cast<int64_t>(rounds)},
                 {"material_bytes", static_cast<int64_t>(material_bytes)},
                 {"shape", Value(sh)}});
}

// ---------------------------------------------------------------------------
// Track
// ---------------------------------------------------------------------------
Party::Track::Track(Party& party, std::string k, std::vector<size_t> sh)
    : p(party), kind(std::move(k)), shape(std::move(sh)) {
    t0 = now_s();
    b0 = p.ch_.stats.bytes_sent + p.ch_.stats.bytes_recv;
    r0 = p.ch_.stats.rounds;
    m0 = p.dealer_.bytes_consumed();
    p.depth_ += 1;
}

Party::Track::~Track() {
    p.depth_ -= 1;
    GadgetStat s;
    s.kind = kind;
    s.shape = shape;
    s.seconds = now_s() - t0;
    s.bytes = p.ch_.stats.bytes_sent + p.ch_.stats.bytes_recv - b0;
    s.rounds = p.ch_.stats.rounds - r0;
    s.material_bytes = p.dealer_.bytes_consumed() - m0;
    if (p.depth_ == 0) p.stats.push_back(s);
    p.all_stats.push_back(s);
}

// ---------------------------------------------------------------------------
// Party core
// ---------------------------------------------------------------------------
Party::Party(int party, Channel& channel, DealerClient& dealer, const NumericContract& contract, ReleasePolicy release,
             ll::Rng rng, std::string model_version, std::string client_key_domain)
    : party_(party), ch_(channel), dealer_(dealer), contract_(contract), release_(std::move(release)), rng_(rng),
      model_version_(std::move(model_version)), client_key_domain_(std::move(client_key_domain)) {}

void Party::set_gate(const std::string& gate_id, int64_t epoch) { dealer_.set_gate(gate_id, epoch); }

uint64_t Party::enc(double x) const {
    double v = std::nearbyint(x * std::ldexp(1.0, contract_.frac_bits));
    return static_cast<uint64_t>(static_cast<int64_t>(v));
}

double Party::dec(uint64_t v) const { return static_cast<double>(static_cast<int64_t>(v)) / std::ldexp(1.0, contract_.frac_bits); }

Expected<Mat> Party::encode(const std::vector<double>& x, size_t rows, size_t cols) const {
    if (x.size() != rows * cols) return fail("SHARE_SHAPE: encode size mismatch");
    Mat m(rows, cols);
    double bound = std::ldexp(1.0, contract_.logical_bits - 1);
    double sc = std::ldexp(1.0, contract_.frac_bits);
    for (size_t i = 0; i < x.size(); ++i) {
        if (!std::isfinite(x[i])) return fail("CONTRACT_VIOLATION: non-finite value");
        double v = std::nearbyint(x[i] * sc);
        if (std::fabs(v) >= bound) return fail("CONTRACT_VIOLATION: |v| >= 2^(l-1) (no silent clamp)");
        m.a[i] = static_cast<uint64_t>(static_cast<int64_t>(v));
    }
    return ok(std::move(m));
}

std::vector<double> Party::decode(const Mat& m) const {
    std::vector<double> out(m.size());
    for (size_t i = 0; i < m.size(); ++i) out[i] = dec(m.a[i]);
    return out;
}

Mat Party::const_mat(double value, size_t rows, size_t cols) const {
    return Mat(rows, cols, party_ == 0 ? enc(value) : 0);
}

Mat Party::const_ring(uint64_t value, size_t rows, size_t cols) const {
    return Mat(rows, cols, party_ == 0 ? value : 0);
}

Expected<Mat> Party::share_input(const Mat* value_enc, int owner, size_t rows, size_t cols) {
    if (party_ == owner) {
        if (!value_enc) return fail("SHARE_OWNER: the owner must pass its value");
        if (value_enc->rows != rows || value_enc->cols != cols) return fail("SHARE_SHAPE: value shape differs from declared");
        Mat mask(rows, cols);
        for (auto& v : mask.a) v = rng_.next_u64();
        Mat peer = ring_sub(*value_enc, mask);
        OC2_CHECK(ch_.send(u64_array(peer.a, {rows, cols})));
        ch_.clear_pending();  // the round is counted explicitly on both sides
        count_round();
        return ok(std::move(mask));
    }
    OC2_TRY(msg, ch_.recv());
    count_round();
    OC2_TRY(arr, msg.as_array());
    if (arr.dtype != DType::u64 || arr.numel() != rows * cols) return fail("SHARE_SHAPE: received shape differs from declared");
    return ok(Mat::from_vector(arr.as_u64(), rows, cols));
}

Expected<Mat> Party::open(const Mat& share) {
    OC2_TRY(other, ch_.exchange(u64_array(share.a, {share.rows, share.cols})));
    OC2_TRY(arr, other.as_array());
    if (arr.dtype != DType::u64 || arr.numel() != share.size()) return fail("OPEN_SHAPE: peer sent a different shape");
    Mat o = Mat::from_vector(arr.as_u64(), share.rows, share.cols);
    return ok(ring_add(share, o));
}

Expected<std::vector<uint64_t>> Party::open_flat(const std::vector<uint64_t>& share) {
    OC2_TRY(other, ch_.exchange(u64_array(share, {share.size()})));
    OC2_TRY(arr, other.as_array());
    if (arr.dtype != DType::u64 || arr.dims.size() != 1 || arr.numel() != share.size())
        return fail("OPEN_SHAPE: peer sent a different shape");
    std::vector<uint64_t> o = arr.as_u64();
    for (size_t i = 0; i < o.size(); ++i) o[i] += share[i];
    return ok(std::move(o));
}

Expected<Mat> Party::reveal(const Mat& share) { return open(share); }

Expected<Mat> Party::reveal_final(const Mat& share, const std::string& kind, bool flat) {
    OC2_CHECK(release_.authorize(kind));
    release_log.push_back(dict({{"kind", kind},
                                {"rows", static_cast<int64_t>(share.rows)},
                                {"cols", static_cast<int64_t>(share.cols)},
                                {"count", static_cast<int64_t>(release_.count())}}));
    if (party_ == release_.recipient()) {
        OC2_TRY(msg, ch_.recv());
        count_round();
        OC2_TRY(arr, msg.as_array());
        if (arr.dtype != DType::u64 || arr.numel() != share.size()) return fail("OPEN_SHAPE: peer sent a different shape");
        return ok(ring_add(share, Mat::from_vector(arr.as_u64(), share.rows, share.cols)));
    }
    OC2_CHECK(ch_.send(flat ? u64_array(share.a, {share.size()}) : u64_array(share.a, {share.rows, share.cols})));
    ch_.clear_pending();
    count_round();
    return ok(Mat());
}

Mat Party::add_public(const Mat& share, const Mat& pub) const { return party_ == 0 ? ring_add(share, pub) : share; }

Mat Party::add_public_real(const Mat& share, double value) const {
    if (party_ != 0) return share;
    Mat m = share;
    uint64_t c = enc(value);
    for (auto& v : m.a) v += c;
    return m;
}

Mat Party::mul_public_real(const Mat& share, double value) const {
    if (value == std::nearbyint(value)) return ring_mul_const(share, static_cast<uint64_t>(static_cast<int64_t>(value)));
    int e = 0;
    double m = std::frexp(value, &e);  // value = m 2^e, 0.5 <= |m| < 1
    int pre = std::max(0, -e - 4);
    int shift = contract_.frac_bits - e - pre;
    if (shift < 0) return ring_mul_const(share, static_cast<uint64_t>(static_cast<int64_t>(std::nearbyint(value))));
    Mat s = pre ? trunc_local(share, pre) : share;
    return trunc_local(ring_mul_const(s, enc(m)), shift);
}

Mat Party::matmul_public(const Mat& share, const Mat& pub, bool pub_on_left) const {
    return pub_on_left ? ring_matmul(pub, share) : ring_matmul(share, pub);
}

Mat Party::trunc_local(const Mat& share, int f) const {
    if (f < 0) f = contract_.frac_bits;
    if (f == 0) return share;
    // P0: floor(s0 / 2^f); P1: -floor(-s1 / 2^f). The floor of the signed value is taken with the
    // explicit two's complement shift of the lowlevel ring (total for every f, no signed >>).
    Mat m(share.rows, share.cols);
    for (size_t i = 0; i < share.size(); ++i) {
        if (party_ == 0) {
            m.a[i] = lowlevel::ring::arith_shift_right(share.a[i], f);
        } else {
            m.a[i] = 0 - lowlevel::ring::arith_shift_right(0 - share.a[i], f);
        }
    }
    return m;
}

Expected<Mat> Party::trunc_faithful(const Mat& share, int f, int L) {
    if (f < 0) f = contract_.frac_bits;
    if (L < 0) L = contract_.logical_bits;
    if (!(1 <= f && f <= L - 1 && L - 1 <= 63)) return fail("TRUNC_PARAMS: need 1 <= f <= L-1 <= 63");
    Track tr(*this, "trunc_faithful", {share.rows, share.cols});
    if (share.size() == 0) return ok(share);
    OC2_TRY(mat, dealer_.take_trunc(share.size(), f, L));
    if (mat.f != f || mat.L != L) return fail("MATERIAL_PARAMS: trunc material (f, L) differs from the request");
    OC2_TRY(xo, lowlevel::trunc::trunc_offset(share.a, party_, L));
    Mat masked = Mat::from_vector(xo, share.rows, share.cols);
    for (size_t i = 0; i < masked.size(); ++i) masked.a[i] += mat.r_share[i];
    OC2_TRY(y, open(masked));
    OC2_TRY(out, lowlevel::trunc::trunc_faithful_eval(mat, y.a, party_));
    return ok(Mat::from_vector(std::move(out), share.rows, share.cols));
}

// --- Beaver -------------------------------------------------------------------------
Expected<std::vector<Mat>> Party::beaver_mul_batch(const std::vector<Mat>& xs, const std::vector<Mat>& ys) {
    if (xs.size() != ys.size()) return fail("BEAVER_SHAPE: batch size mismatch");
    size_t total = 0;
    for (size_t i = 0; i < xs.size(); ++i) {
        if (!xs[i].same_shape(ys[i])) return fail("BEAVER_SHAPE: operand shapes differ");
        total += xs[i].size();
    }
    std::vector<Mat> out(xs.size());
    if (total == 0) {
        for (size_t i = 0; i < xs.size(); ++i) out[i] = Mat(xs[i].rows, xs[i].cols);
        return ok(std::move(out));
    }
    OC2_TRY(triple, dealer_.take_beaver(total));
    // masked = [x - a | y - b] for every operand, one opening
    Mat masked(1, 2 * total);
    size_t at = 0;
    for (size_t i = 0; i < xs.size(); ++i) {
        for (size_t j = 0; j < xs[i].size(); ++j) {
            masked.a[at + j] = xs[i].a[j] - triple.a[at + j];
            masked.a[total + at + j] = ys[i].a[j] - triple.b[at + j];
        }
        at += xs[i].size();
    }
    OC2_TRY(de, open(masked));
    at = 0;
    for (size_t i = 0; i < xs.size(); ++i) {
        Mat z(xs[i].rows, xs[i].cols);
        for (size_t j = 0; j < z.size(); ++j) {
            uint64_t d = de.a[at + j], e = de.a[total + at + j];
            uint64_t v = triple.c[at + j] + d * triple.b[at + j] + e * triple.a[at + j];
            if (party_ == 0) v += d * e;
            z.a[j] = v;
        }
        out[i] = std::move(z);
        at += xs[i].size();
    }
    return ok(std::move(out));
}

Expected<Mat> Party::beaver_mul(const Mat& sx, const Mat& sy) {
    Mat x = sx, y = sy;
    // broadcasting of a 1 x 1 or a per-row (rows x 1) operand
    if (!x.same_shape(y)) {
        if (y.rows == x.rows && y.cols == 1) {
            Mat yb(x.rows, x.cols);
            for (size_t i = 0; i < x.rows; ++i)
                for (size_t j = 0; j < x.cols; ++j) yb.a[i * x.cols + j] = y.a[i];
            y = yb;
        } else if (y.size() == 1) {
            y = Mat(x.rows, x.cols, y.a[0]);
        } else if (x.size() == 1) {
            x = Mat(y.rows, y.cols, x.a[0]);
        } else {
            return fail("BEAVER_SHAPE: operand shapes differ");
        }
    }
    OC2_TRY(r, beaver_mul_batch({x}, {y}));
    return ok(std::move(r[0]));
}

Expected<Mat> Party::mul_fxp(const Mat& sx, const Mat& sy, int trunc_bits) {
    OC2_TRY(z, beaver_mul(sx, sy));
    return ok(trunc_local(z, trunc_bits));
}

Expected<Mat> Party::square_fxp(const Mat& sx, bool exact) {
    if (exact) {
        OC2_TRY(z, beaver_mul(sx, sx));
        return trunc_faithful(z);
    }
    return mul_fxp(sx, sx);
}

// --- matrix products ------------------------------------------------------------------
Expected<std::vector<Mat>> Party::matmul_shared_batch(const std::vector<Mat>& Xs, const std::vector<Mat>& Ys, bool trunc) {
    if (Xs.size() != Ys.size()) return fail("MATMUL_SHAPE: batch size mismatch");
    std::vector<size_t> shape;
    if (!Xs.empty()) shape = {Xs.size(), Xs[0].rows, Xs[0].cols, Ys[0].cols};
    Track tr(*this, "matmul_shared", shape);
    std::vector<MatrixTripleShare> triples;
    std::vector<const Mat*> parts;
    std::vector<Mat> masked;
    for (size_t i = 0; i < Xs.size(); ++i) {
        if (Xs[i].cols != Ys[i].rows) return fail("MATMUL_SHAPE: inner dimensions differ");
        OC2_TRY(t, dealer_.take_matrix_triple(Xs[i].rows, Xs[i].cols, Ys[i].cols));
        triples.push_back(std::move(t));
    }
    for (size_t i = 0; i < Xs.size(); ++i) {
        Mat dx = ring_sub(Xs[i], Mat::from_vector(triples[i].A, Xs[i].rows, Xs[i].cols));
        Mat ey = ring_sub(Ys[i], Mat::from_vector(triples[i].B, Ys[i].rows, Ys[i].cols));
        masked.push_back(std::move(dx));
        masked.push_back(std::move(ey));
    }
    for (const Mat& m : masked) parts.push_back(&m);
    Mat flat = Mat::concat_flat(parts);
    OC2_TRY(opened, open(flat));
    std::vector<Mat> out;
    size_t at = 0;
    for (size_t i = 0; i < Xs.size(); ++i) {
        size_t n = Xs[i].rows, k = Xs[i].cols, m = Ys[i].cols;
        Mat D = Mat::from_vector(std::vector<uint64_t>(opened.a.begin() + static_cast<std::ptrdiff_t>(at),
                                                       opened.a.begin() + static_cast<std::ptrdiff_t>(at + n * k)), n, k);
        at += n * k;
        Mat E = Mat::from_vector(std::vector<uint64_t>(opened.a.begin() + static_cast<std::ptrdiff_t>(at),
                                                       opened.a.begin() + static_cast<std::ptrdiff_t>(at + k * m)), k, m);
        at += k * m;
        Mat A = Mat::from_vector(triples[i].A, n, k);
        Mat B = Mat::from_vector(triples[i].B, k, m);
        Mat Z = Mat::from_vector(triples[i].C, n, m);
        Z = ring_add(Z, ring_add(ring_matmul(D, B), ring_matmul(A, E)));
        if (party_ == 0) Z = ring_add(Z, ring_matmul(D, E));
        out.push_back(trunc ? trunc_local(Z) : Z);
    }
    return ok(std::move(out));
}

Expected<Mat> Party::matmul_shared(const Mat& sX, const Mat& sY, bool trunc) {
    OC2_TRY(r, matmul_shared_batch({sX}, {sY}, trunc));
    return ok(std::move(r[0]));
}

// --- linear layers ------------------------------------------------------------------------
void Party::set_vector_share(const std::string& name, Mat share) { vectors_[name] = std::move(share); }

bool Party::has_vector_share(const std::string& name) const {
    return vectors_.count(name) > 0 || dealer_.has_vector_share(name);
}

Expected<Mat> Party::vector_share(const std::string& name) {
    auto it = vectors_.find(name);
    if (it != vectors_.end()) return ok(it->second);
    OC2_TRY(v, dealer_.take_vector_share(name));
    Mat m = Mat::row_vector(std::move(v));
    vectors_[name] = m;
    return ok(std::move(m));
}

Expected<Mat> Party::linear_generic(const Mat& sX, const std::string& mat_id, bool trunc) {
    OC2_TRY(W, vector_share(mat_id));
    if (W.rows != sX.cols) return fail("LINEAR_SHAPE: weight rows differ from input cols for " + mat_id);
    Track tr(*this, "linear_generic", {sX.rows, W.rows, W.cols});
    OC2_TRY(y, matmul_shared(sX, W, trunc));
    if (has_vector_share(mat_id + ".b")) {
        OC2_TRY(b, vector_share(mat_id + ".b"));
        if (b.size() != y.cols) return fail("LINEAR_SHAPE: bias size for " + mat_id);
        for (size_t i = 0; i < y.rows; ++i)
            for (size_t j = 0; j < y.cols; ++j) y.a[i * y.cols + j] += b.a[j];
    }
    return ok(std::move(y));
}

Value Party::version() const {
    return dict({{"model_version", model_version_}, {"client_key_domain", client_key_domain_}});
}

bool Party::installed(const std::string& mat_id) const {
    auto it = installed_.find(mat_id);
    if (it == installed_.end()) return false;
    if (!version_equal(it->second.version, version())) return false;
    return dealer_.ledger().installed(install_object_id(mat_id), version());
}

Value Party::installed_objects() const {
    Dict d;
    for (const auto& kv : installed_) d.push_back(DictEntry{kv.first, kv.second.version});
    return Value(d);
}

Expected<Mat> Party::wmask_mask(const std::string& mat_id, size_t d, size_t m) {
    auto it = wmask_b_.find(mat_id);
    if (it != wmask_b_.end()) return ok(it->second);
    OC2_TRY(wm, dealer_.wmask_mask_share(mat_id, d, m));
    Mat B = Mat::from_vector(wm.b_share, wm.d, wm.m);
    wmask_b_[mat_id] = B;
    return ok(std::move(B));
}

Expected<Value> Party::install_wmask(const std::string& mat_id, const Mat* W_enc_or_null) {
    Value ver = version();
    if (installed(mat_id)) {
        return ok(dict({{"mat_id", mat_id}, {"bytes", static_cast<int64_t>(0)}, {"seconds", 0.0},
                        {"rounds", static_cast<int64_t>(0)}, {"version", ver}, {"skipped", true}}));
    }
    // public mask shape: W on P0, the registered weight share otherwise
    size_t d = 0, m = 0;
    if (W_enc_or_null) {
        d = W_enc_or_null->rows;
        m = W_enc_or_null->cols;
    } else {
        auto vs = vectors_.find(mat_id);
        if (vs == vectors_.end()) return fail("INSTALL_SHAPE: unknown mask shape for " + mat_id + " (no weight share registered)");
        d = vs->second.rows;
        m = vs->second.cols;
    }
    uint64_t b0 = ch_.stats.bytes_sent + ch_.stats.bytes_recv;
    double t0 = now_s();
    Mat E;
    Mat B;
    {
        // The Track is created BEFORE the mask bundle is taken: the 8 d m bytes of the mask are
        // the material of this gadget (material_formula and the Python party charge them here).
        Track tr(*this, "install_wmask", {d, m});
        OC2_TRY(Bm, wmask_mask(mat_id, d, m));
        B = std::move(Bm);
        if (party_ == 1) {
            if (W_enc_or_null) return fail("INSTALL_OWNER: the client holds no plaintext weights");
            OC2_CHECK(ch_.send(u64_array(B.a, {B.rows, B.cols})));
            OC2_TRY(msg, ch_.recv());
            OC2_TRY(arr, msg.as_array());
            if (arr.numel() != B.size()) return fail("INSTALL_SHAPE: E differs from the mask shape");
            E = Mat::from_vector(arr.as_u64(), B.rows, B.cols);
        } else {
            if (!W_enc_or_null) return fail("INSTALL_OWNER: P0 must supply the encoded weight matrix");
            if (!W_enc_or_null->same_shape(B)) return fail("INSTALL_SHAPE: W differs from the mask shape");
            OC2_TRY(msg, ch_.recv());
            OC2_TRY(arr, msg.as_array());
            if (arr.numel() != B.size()) return fail("INSTALL_SHAPE: peer mask share differs from the mask shape");
            Mat B1 = Mat::from_vector(arr.as_u64(), B.rows, B.cols);
            E = ring_sub(ring_sub(*W_enc_or_null, B), B1);
            OC2_CHECK(ch_.send(u64_array(E.a, {E.rows, E.cols})));
            ch_.clear_pending();
            count_round();
        }
    }
    installed_[mat_id] = Installed{E, ver};
    OC2_CHECK(dealer_.ledger().register_installed(install_object_id(mat_id), ver));
    Value rec = dict({{"mat_id", mat_id},
                      {"bytes", static_cast<int64_t>(ch_.stats.bytes_sent + ch_.stats.bytes_recv - b0)},
                      {"seconds", now_s() - t0},
                      {"rounds", static_cast<int64_t>(1)},
                      {"version", ver},
                      {"skipped", false},
                      {"shape", Value(List{Value(static_cast<int64_t>(B.rows)), Value(static_cast<int64_t>(B.cols))})}});
    install_log.push_back(rec);
    return ok(std::move(rec));
}

Expected<Mat> Party::linear_wmask(const Mat& sX, const std::string& mat_id, bool trunc) {
    if (!installed(mat_id)) return fail("OBJECT_NOT_INSTALLED: wmask linear on " + mat_id + " before its install");
    const Mat& E = installed_[mat_id].E;
    OC2_TRY(B, wmask_mask(mat_id, E.rows, E.cols));
    if (sX.cols != B.rows) return fail("LINEAR_SHAPE: input cols differ from the mask rows for " + mat_id);
    size_t n = sX.rows;
    Track tr(*this, "linear_wmask", {n, B.rows, B.cols});
    OC2_TRY(call, dealer_.take_wmask_call(mat_id, n));
    if (call.d != B.rows || call.m != B.cols) return fail("MATERIAL_PARAMS: wmask call shape");
    Mat A = Mat::from_vector(call.a, n, B.rows);
    Mat C = Mat::from_vector(call.c, n, B.cols);
    OC2_TRY(D, open(ring_sub(sX, A)));
    Mat z = ring_add(C, ring_add(ring_matmul(D, B), ring_matmul(A, E)));
    if (party_ == 0) z = ring_add(z, ring_matmul(D, E));
    if (trunc) z = trunc_local(z);
    if (has_vector_share(mat_id + ".b")) {
        OC2_TRY(b, vector_share(mat_id + ".b"));
        if (b.size() != z.cols) return fail("LINEAR_SHAPE: bias size for " + mat_id);
        for (size_t i = 0; i < z.rows; ++i)
            for (size_t j = 0; j < z.cols; ++j) z.a[i * z.cols + j] += b.a[j];
    }
    return ok(std::move(z));
}

// --- comparison layer ----------------------------------------------------------------------
Expected<std::pair<Mat, Mat>> Party::drelu_raw(const Mat& x, bool with_product, Mat* rb_out) {
    size_t M = x.size();
    OC2_TRY(mat, dealer_.take_drelu(M, contract_.dcf_domain_bits, with_product));
    Mat masked = x;
    for (size_t i = 0; i < M; ++i) masked.a[i] += mat.r_share[i];
    OC2_TRY(y, open(masked));
    OC2_TRY(res, lowlevel::drelu::drelu_eval(mat, y.a));
    Mat b = Mat::from_vector(std::move(res.b), x.rows, x.cols);
    if (rb_out && with_product) *rb_out = Mat::from_vector(std::move(res.rb), x.rows, x.cols);
    return ok(std::make_pair(std::move(y), std::move(b)));
}

Expected<Mat> Party::drelu(const Mat& x) {
    Track tr(*this, "drelu", {x.rows, x.cols});
    if (x.size() == 0) return ok(x);
    OC2_TRY(yb, drelu_raw(x, false, nullptr));
    return ok(std::move(yb.second));
}

Expected<Mat> Party::relu(const Mat& x) {
    Track tr(*this, "relu", {x.rows, x.cols});
    if (x.size() == 0) return ok(x);
    Mat rb;
    OC2_TRY(yb, drelu_raw(x, true, &rb));
    OC2_TRY(out, lowlevel::drelu::relu_from_drelu(yb.first.a, yb.second.a, rb.a));
    return ok(Mat::from_vector(std::move(out), x.rows, x.cols));
}

Expected<Mat> Party::compare_public(const Mat& x, double c, const std::string& op) {
    Track tr(*this, "compare_public", {x.rows, x.cols});
    OC2_TRY(b, drelu(add_public_real(x, -c)));
    if (op == "ge") return ok(std::move(b));
    if (op == "lt") return ok(add_public(ring_neg(b), Mat(x.rows, x.cols, 1)));
    return fail("COMPARE_OP: op must be ge or lt");
}

Expected<Mat> Party::select(const Mat& bit, const Mat& a, const Mat& b) {
    Track tr(*this, "select", {a.rows, a.cols});
    OC2_TRY(prod, beaver_mul(bit, ring_sub(a, b)));
    return ok(ring_add(b, prod));
}

// --- k-ary tournament -------------------------------------------------------------------------
// Fused two-round node, a line-by-line port of python/opencrab2/highlevel/party.py
// (Party._tournament); see party.hpp for the protocol and the magnitude bound. The order of
// material takes and of the opened elements is the Python order: per level, the round-1 DReLU
// batch (P = 1, live pairs in row-major order over (row, pair)), then, group by group, the
// Beaver select of a binary node (value, then index when the indices are secret) or the payload
// DReLU batch of a fused node (value, then index), each in row-major order over (row, member).
int Party::kary_ceil_log2(size_t g) {
    int c = 0;
    while (c < 63 && (static_cast<size_t>(1) << c) < g) ++c;
    return c;
}

uint64_t Party::kary_node_modulus(int n_bits, size_t g) {
    int e = n_bits - 1 - kary_ceil_log2(g);
    if (e < 0 || e > 63) return 0;
    return static_cast<uint64_t>(1) << e;
}

uint64_t Party::kary_difference_bound(int n_bits, size_t g) {
    if (g <= 2) return (n_bits >= 1 && n_bits <= 64) ? static_cast<uint64_t>(1) << (n_bits - 1) : 0;
    return kary_node_modulus(n_bits, g) / 2;
}

Expected<std::pair<Mat, Mat>> Party::tournament(const Mat& x, const std::vector<uint8_t>* mask, int k, bool with_index) {
    const size_t R = x.rows, n = x.cols;
    if (n == 0) return fail("MAX_ROWS_EMPTY: cannot take the maximum of an empty row");
    std::vector<uint8_t> valid(R * n, 1);
    if (mask) {
        if (mask->size() != R * n) return fail("MAX_ROWS_MASK: mask shape differs from the input");
        valid = *mask;
        for (size_t r = 0; r < R; ++r) {
            bool any = false;
            for (size_t j = 0; j < n; ++j) any = any || valid[r * n + j];
            if (!any) return fail("MAX_ROWS_EMPTY: every row needs at least one valid entry");
        }
    }
    const int nb = contract_.dcf_domain_bits;
    const size_t K = static_cast<size_t>(k);
    // Public admission of the node geometry (fail closed before any material is taken): the node
    // modulus must leave room for the half offset, and the index differences of an argmax (at most
    // n - 1, a public quantity) must respect the fused bound of the largest node.
    {
        const size_t g_max = std::min(K, n);
        if (g_max >= 3) {
            const uint64_t M = kary_node_modulus(nb, g_max);
            if (M < 2) return fail("KARY_MAGNITUDE: dcf_domain_bits too small for a fused node of " + std::to_string(g_max) + " members");
            if (with_index && static_cast<uint64_t>(n - 1) >= M / 2)
                return fail("KARY_MAGNITUDE: row length exceeds the index bound of the fused node");
        }
    }
    Mat vals = x;
    Mat idx;
    bool idx_public = true;
    if (with_index) {
        idx = Mat(R, n);
        if (party_ == 0)
            for (size_t r = 0; r < R; ++r)
                for (size_t j = 0; j < n; ++j) idx.a[r * n + j] = static_cast<uint64_t>(j);
    }
    const uint64_t one = party_ == 0 ? 1 : 0;
    size_t w = n;
    while (w > 1) {
        struct Group { size_t s, g; };
        std::vector<Group> groups;
        for (size_t s = 0; s < w; s += K) groups.push_back({s, std::min<size_t>(K, w - s)});
        const size_t ng = groups.size();
        // ---- round 1: all pairs inside every node (DReLU without payload) ----------------------
        std::vector<size_t> gi, gj;           // pair -> (column i, column j), node order
        std::vector<size_t> pair_pos(ng, 0);  // first pair of each node
        for (size_t q = 0; q < ng; ++q) {
            pair_pos[q] = gi.size();
            for (size_t i = 0; i < groups[q].g; ++i)
                for (size_t j = i + 1; j < groups[q].g; ++j) {
                    gi.push_back(groups[q].s + i);
                    gj.push_back(groups[q].s + j);
                }
        }
        const size_t P = gi.size();
        std::vector<uint64_t> c(R * P, 0);  // shares of 1{x_i >= x_j}
        {
            std::vector<uint64_t> flat;
            std::vector<size_t> where;
            for (size_t r = 0; r < R; ++r)
                for (size_t p = 0; p < P; ++p) {
                    const bool vi = valid[r * w + gi[p]] != 0, vj = valid[r * w + gj[p]] != 0;
                    if (vi && vj) {
                        flat.push_back(vals.a[r * w + gi[p]] - vals.a[r * w + gj[p]]);
                        where.push_back(r * P + p);
                    } else if (vi || !vj) {
                        c[r * P + p] = one;  // decided publicly: i wins (P0 holds the 1)
                    }
                    // else: i invalid, j valid: c = 0 on both parties
                }
            if (!flat.empty()) {
                Track tr1(*this, "drelu", {flat.size()});
                OC2_TRY(mat, dealer_.take_drelu(flat.size(), nb, false));
                std::vector<uint64_t> masked(flat.size());
                for (size_t q = 0; q < flat.size(); ++q) masked[q] = flat[q] + mat.r_share[q];
                OC2_TRY(y, open_flat(masked));
                OC2_TRY(res, lowlevel::drelu::drelu_eval(mat, y));
                for (size_t q = 0; q < flat.size(); ++q) c[where[q]] = res.b[q];
            }
        }
        // rank_i = sum_{j > i} c_ij + sum_{j < i} (1 - c_ji)
        std::vector<uint64_t> rank(R * w, 0);
        for (size_t r = 0; r < R; ++r)
            for (size_t p = 0; p < P; ++p) {
                const uint64_t cv = c[r * P + p];
                rank[r * w + gi[p]] += cv;
                rank[r * w + gj[p]] += one - cv;
            }
        // ---- round 2: fused selection, one opening for the whole level -----------------------------
        struct Request {
            int kind = 0;  // 0 pass, 1 pair, 2 node
            BeaverShare val, idx;
            ll::DReLUMaterial mat, mat_idx;
            uint64_t M = 0;
        };
        std::vector<Request> req(ng);
        std::vector<uint64_t> opens;
        const bool secret_idx = with_index && !idx_public;
        for (size_t q = 0; q < ng; ++q) {
            const size_t s = groups[q].s, g = groups[q].g;
            if (g == 1) continue;
            if (g == 2) {
                req[q].kind = 1;
                // masked = [bit - a | dv - b] with bit = c(s, s+1), dv = x_s - x_{s+1}
                OC2_TRY(tv, dealer_.take_kary_select(R));
                const size_t at = opens.size();
                opens.resize(at + 2 * R);
                for (size_t r = 0; r < R; ++r) {
                    opens[at + r] = c[r * P + pair_pos[q]] - tv.a[r];
                    opens[at + R + r] = (vals.a[r * w + s] - vals.a[r * w + s + 1]) - tv.b[r];
                }
                req[q].val = std::move(tv);
                if (secret_idx) {
                    OC2_TRY(ti, dealer_.take_kary_select(R));
                    const size_t ai = opens.size();
                    opens.resize(ai + 2 * R);
                    for (size_t r = 0; r < R; ++r) {
                        opens[ai + r] = c[r * P + pair_pos[q]] - ti.a[r];
                        opens[ai + R + r] = (idx.a[r * w + s] - idx.a[r * w + s + 1]) - ti.b[r];
                    }
                    req[q].idx = std::move(ti);
                }
                continue;
            }
            req[q].kind = 2;
            const size_t m = g - 1;
            const uint64_t M = kary_node_modulus(nb, g);
            const uint64_t offset = (0 - static_cast<uint64_t>(g - 1) * M) + M / 2;  // -(g-1) M + M/2 mod 2^64
            req[q].M = M;
            OC2_TRY(mu, dealer_.take_drelu(R * m, nb, true));
            {
                const size_t at = opens.size();
                opens.resize(at + R * m);
                for (size_t r = 0; r < R; ++r)
                    for (size_t i = 1; i < g; ++i) {
                        const size_t col = s + i;
                        const uint64_t d = valid[r * w + col] ? vals.a[r * w + col] - vals.a[r * w + s] : 0;
                        uint64_t u = rank[r * w + col] * M + d;
                        if (party_ == 0) u += offset;
                        const size_t e = r * m + (i - 1);
                        opens[at + e] = u + mu.r_share[e];
                    }
            }
            req[q].mat = std::move(mu);
            if (secret_idx) {
                OC2_TRY(mv, dealer_.take_drelu(R * m, nb, true));
                const size_t at = opens.size();
                opens.resize(at + R * m);
                for (size_t r = 0; r < R; ++r)
                    for (size_t i = 1; i < g; ++i) {
                        const size_t col = s + i;
                        uint64_t v = rank[r * w + col] * M + (idx.a[r * w + col] - idx.a[r * w + s]);
                        if (party_ == 0) v += offset;
                        const size_t e = r * m + (i - 1);
                        opens[at + e] = v + mv.r_share[e];
                    }
                req[q].mat_idx = std::move(mv);
            }
        }
        std::vector<uint64_t> opened;
        if (!opens.empty()) {
            OC2_TRY(o, open_flat(opens));
            opened = std::move(o);
        }
        size_t cursor = 0;
        Mat new_vals(R, ng), new_idx;
        std::vector<uint8_t> new_valid(R * ng, 0);
        if (with_index) new_idx = Mat(R, ng);
        auto beaver_end = [&](const BeaverShare& t, size_t at, size_t r) {
            const uint64_t d = opened[at + r], e = opened[at + R + r];
            uint64_t z = t.c[r] + d * t.b[r] + e * t.a[r];
            if (party_ == 0) z += d * e;
            return z;
        };
        for (size_t q = 0; q < ng; ++q) {
            const size_t s = groups[q].s, g = groups[q].g;
            for (size_t r = 0; r < R; ++r) {
                bool any = false;
                for (size_t i = 0; i < g; ++i) any = any || valid[r * w + s + i];
                new_valid[r * ng + q] = any ? 1 : 0;
            }
            if (req[q].kind == 0) {
                for (size_t r = 0; r < R; ++r) {
                    new_vals.a[r * ng + q] = vals.a[r * w + s];
                    if (with_index) new_idx.a[r * ng + q] = idx.a[r * w + s];
                }
                continue;
            }
            if (req[q].kind == 1) {
                const size_t at = cursor;
                cursor += 2 * R;
                for (size_t r = 0; r < R; ++r) new_vals.a[r * ng + q] = vals.a[r * w + s + 1] + beaver_end(req[q].val, at, r);
                if (with_index) {
                    if (idx_public) {
                        // idx_0 - idx_1 = -1 publicly: idx = (s + 1) - bit
                        for (size_t r = 0; r < R; ++r)
                            new_idx.a[r * ng + q] = one * static_cast<uint64_t>(s + 1) - c[r * P + pair_pos[q]];
                    } else {
                        const size_t ai = cursor;
                        cursor += 2 * R;
                        for (size_t r = 0; r < R; ++r)
                            new_idx.a[r * ng + q] = idx.a[r * w + s + 1] + beaver_end(req[q].idx, ai, r);
                    }
                }
                continue;
            }
            const size_t m = g - 1;
            const uint64_t half = req[q].M / 2;
            std::vector<uint64_t> y(opened.begin() + static_cast<std::ptrdiff_t>(cursor),
                                    opened.begin() + static_cast<std::ptrdiff_t>(cursor + R * m));
            cursor += R * m;
            OC2_TRY(res, lowlevel::drelu::drelu_eval(req[q].mat, y));
            OC2_TRY(relu, lowlevel::drelu::relu_from_drelu(y, res.b, res.rb));
            for (size_t r = 0; r < R; ++r) {
                uint64_t acc = vals.a[r * w + s];
                for (size_t e = r * m; e < (r + 1) * m; ++e) acc += relu[e] - res.b[e] * half;
                new_vals.a[r * ng + q] = acc;
            }
            if (with_index) {
                if (idx_public) {
                    // idx_i - idx_0 = i publicly: idx = s + sum_i w_i i
                    for (size_t r = 0; r < R; ++r) {
                        uint64_t acc = one * static_cast<uint64_t>(s);
                        for (size_t i = 1; i < g; ++i) acc += res.b[r * m + (i - 1)] * static_cast<uint64_t>(i);
                        new_idx.a[r * ng + q] = acc;
                    }
                } else {
                    std::vector<uint64_t> yv(opened.begin() + static_cast<std::ptrdiff_t>(cursor),
                                             opened.begin() + static_cast<std::ptrdiff_t>(cursor + R * m));
                    cursor += R * m;
                    OC2_TRY(rv, lowlevel::drelu::drelu_eval(req[q].mat_idx, yv));
                    OC2_TRY(relu_v, lowlevel::drelu::relu_from_drelu(yv, rv.b, rv.rb));
                    for (size_t r = 0; r < R; ++r) {
                        uint64_t acc = idx.a[r * w + s];
                        for (size_t e = r * m; e < (r + 1) * m; ++e) acc += relu_v[e] - rv.b[e] * half;
                        new_idx.a[r * ng + q] = acc;
                    }
                }
            }
        }
        if (cursor != opened.size()) return fail("TOURNAMENT_INTERNAL: opening cursor mismatch");
        vals = std::move(new_vals);
        valid = std::move(new_valid);
        if (with_index) {
            idx = std::move(new_idx);
            idx_public = false;
        }
        w = ng;
    }
    Mat out_val(R, 1), out_idx;
    for (size_t r = 0; r < R; ++r) out_val.a[r] = vals.a[r];
    if (with_index) {
        out_idx = Mat(R, 1);
        for (size_t r = 0; r < R; ++r) out_idx.a[r] = idx.a[r];
    }
    return ok(std::make_pair(std::move(out_val), std::move(out_idx)));
}

Expected<Mat> Party::max_rows(const Mat& x, const std::vector<uint8_t>* mask, int k) {
    if (k != 2 && k != 4 && k != 8) return fail("MAX_ROWS_K: k must be one of 2, 4, 8");
    Track tr(*this, "max_rows_k" + std::to_string(k), {x.rows, x.cols});
    OC2_TRY(res, tournament(x, mask, k, false));
    return ok(std::move(res.first));
}

Expected<Mat> Party::argmax_row(const Mat& x, int k, bool release, const std::string& kind, const std::vector<uint8_t>* mask) {
    if (k != 2 && k != 4 && k != 8) return fail("ARGMAX_K: k must be one of 2, 4, 8");
    Track tr(*this, "argmax_k" + std::to_string(k), {x.rows, x.cols});
    OC2_TRY(res, tournament(x, mask, k, true));
    if (!release) return ok(std::move(res.second));
    // RELEASE_FINAL of an argmax: one message of 8 bytes per row in a 1-D frame, as the Python
    // party sends it (formulas.json "conventions"); the recipient accepts any shape of that size.
    return reveal_final(res.second, kind, true);
}

// --- ring to field -------------------------------------------------------------------------------
Expected<Mat> Party::ring_to_field(const Mat& share, uint64_t t) {
    Track tr(*this, "ring_to_field", {share.rows, share.cols});
    if (share.size() == 0) return ok(share);
    OC2_TRY(mat, dealer_.take_r2f(share.size(), t));
    if (mat.t != t) return fail("MATERIAL_PARAMS: r2f material modulus differs");
    Mat masked = share;
    for (size_t i = 0; i < masked.size(); ++i) masked.a[i] += mat.r_share[i];
    OC2_TRY(y, open(masked));
    OC2_TRY(out, lowlevel::r2f::r2f_eval(mat, y.a, party_));
    return ok(Mat::from_vector(std::move(out), share.rows, share.cols));
}

Value Party::stats_value() const {
    List l;
    for (const auto& s : stats) l.push_back(s.to_value());
    return Value(l);
}

}  // namespace highlevel
}  // namespace opencrab2
