// Implementation of tiny_decoder.hpp.
#include <opencrab2/highlevel/tiny_decoder.hpp>
#include <opencrab2/highlevel/gadgets.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>

namespace opencrab2 {
namespace highlevel {

// ---------------------------------------------------------------------------
// config / weights
// ---------------------------------------------------------------------------
Value TinyConfig::to_value() const {
    return dict({{"n_layers", static_cast<int64_t>(n_layers)},
                 {"d_model", static_cast<int64_t>(d_model)},
                 {"n_head", static_cast<int64_t>(n_head)},
                 {"vocab", static_cast<int64_t>(vocab)},
                 {"max_pos", static_cast<int64_t>(max_pos)},
                 {"d_ff", static_cast<int64_t>(d_ff)}});
}

namespace {
void fill_normal(std::vector<float>& v, size_t n, ll::Rng& rng, float std_dev, float mean = 0.0f) {
    v.resize(n);
    for (auto& x : v) x = mean + std_dev * static_cast<float>(ll::rng_normal(rng));
}
void put_f32(Bytes& out, const std::vector<float>& v) {
    size_t at = out.size();
    out.resize(at + 4 * v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        uint32_t bits;
        std::memcpy(&bits, &v[i], 4);
        for (int k = 0; k < 4; ++k) out[at + 4 * i + static_cast<size_t>(k)] = static_cast<uint8_t>(bits >> (8 * k));
    }
}
void put_u32(Bytes& out, uint32_t v) {
    for (int k = 0; k < 4; ++k) out.push_back(static_cast<uint8_t>(v >> (8 * k)));
}
struct Reader {
    const Bytes& b;
    size_t at = 0;
    bool fail_ = false;
    uint32_t u32() {
        if (at + 4 > b.size()) { fail_ = true; return 0; }
        uint32_t v = 0;
        for (int k = 3; k >= 0; --k) v = (v << 8) | b[at + static_cast<size_t>(k)];
        at += 4;
        return v;
    }
    std::vector<float> f32(size_t n) {
        std::vector<float> v(n);
        if (at + 4 * n > b.size()) { fail_ = true; return v; }
        for (size_t i = 0; i < n; ++i) {
            uint32_t bits = 0;
            for (int k = 3; k >= 0; --k) bits = (bits << 8) | b[at + 4 * i + static_cast<size_t>(k)];
            std::memcpy(&v[i], &bits, 4);
        }
        at += 4 * n;
        return v;
    }
};
}  // namespace

TinyWeights TinyWeights::random(const TinyConfig& cfg, uint64_t seed, float std_dev, float emb_std) {
    ll::Rng rng = ll::Rng::from_seed(seed);
    TinyWeights w;
    w.cfg = cfg;
    size_t d = cfg.d_model, V = cfg.vocab, F = cfg.d_ff;
    fill_normal(w.wte, V * d, rng, emb_std);
    fill_normal(w.wpe, static_cast<size_t>(cfg.max_pos) * d, rng, 0.1f);
    for (uint32_t l = 0; l < cfg.n_layers; ++l) {
        TinyLayer L;
        fill_normal(L.ln1_g, d, rng, 0.1f, 1.0f);
        fill_normal(L.ln1_b, d, rng, 0.1f);
        fill_normal(L.w_qkv, d * 3 * d, rng, std_dev);
        fill_normal(L.b_qkv, 3 * d, rng, std_dev);
        fill_normal(L.w_out, d * d, rng, std_dev);
        fill_normal(L.b_out, d, rng, std_dev);
        fill_normal(L.ln2_g, d, rng, 0.1f, 1.0f);
        fill_normal(L.ln2_b, d, rng, 0.1f);
        fill_normal(L.w_fc, d * F, rng, std_dev);
        fill_normal(L.b_fc, F, rng, std_dev);
        fill_normal(L.w_proj, F * d, rng, std_dev);
        fill_normal(L.b_proj, d, rng, std_dev);
        w.layers.push_back(std::move(L));
    }
    fill_normal(w.lnf_g, d, rng, 0.1f, 1.0f);
    fill_normal(w.lnf_b, d, rng, 0.1f);
    fill_normal(w.w_lm, d * V, rng, emb_std);
    return w;
}

Bytes TinyWeights::serialize() const {
    Bytes out;
    const char* magic = "OC2TDW01";
    out.insert(out.end(), magic, magic + 8);
    put_u32(out, cfg.n_layers);
    put_u32(out, cfg.d_model);
    put_u32(out, cfg.n_head);
    put_u32(out, cfg.vocab);
    put_u32(out, cfg.max_pos);
    put_u32(out, cfg.d_ff);
    put_f32(out, wte);
    put_f32(out, wpe);
    for (const auto& L : layers) {
        put_f32(out, L.ln1_g); put_f32(out, L.ln1_b); put_f32(out, L.w_qkv); put_f32(out, L.b_qkv);
        put_f32(out, L.w_out); put_f32(out, L.b_out); put_f32(out, L.ln2_g); put_f32(out, L.ln2_b);
        put_f32(out, L.w_fc); put_f32(out, L.b_fc); put_f32(out, L.w_proj); put_f32(out, L.b_proj);
    }
    put_f32(out, lnf_g);
    put_f32(out, lnf_b);
    put_f32(out, w_lm);
    return out;
}

Expected<TinyWeights> TinyWeights::deserialize(const Bytes& b) {
    if (b.size() < 32 || std::memcmp(b.data(), "OC2TDW01", 8) != 0) return fail("WEIGHTS_MAGIC: not an OC2TDW01 file");
    Reader r{b, 8, false};
    TinyWeights w;
    w.cfg.n_layers = r.u32();
    w.cfg.d_model = r.u32();
    w.cfg.n_head = r.u32();
    w.cfg.vocab = r.u32();
    w.cfg.max_pos = r.u32();
    w.cfg.d_ff = r.u32();
    if (r.fail_ || w.cfg.d_model == 0 || w.cfg.n_head == 0 || w.cfg.d_model % w.cfg.n_head != 0 || w.cfg.vocab == 0 ||
        w.cfg.n_layers > 256)
        return fail("WEIGHTS_HEADER: inconsistent header");
    size_t d = w.cfg.d_model, V = w.cfg.vocab, F = w.cfg.d_ff;
    w.wte = r.f32(V * d);
    w.wpe = r.f32(static_cast<size_t>(w.cfg.max_pos) * d);
    for (uint32_t l = 0; l < w.cfg.n_layers; ++l) {
        TinyLayer L;
        L.ln1_g = r.f32(d); L.ln1_b = r.f32(d); L.w_qkv = r.f32(d * 3 * d); L.b_qkv = r.f32(3 * d);
        L.w_out = r.f32(d * d); L.b_out = r.f32(d); L.ln2_g = r.f32(d); L.ln2_b = r.f32(d);
        L.w_fc = r.f32(d * F); L.b_fc = r.f32(F); L.w_proj = r.f32(F * d); L.b_proj = r.f32(d);
        w.layers.push_back(std::move(L));
    }
    w.lnf_g = r.f32(d);
    w.lnf_b = r.f32(d);
    w.w_lm = r.f32(d * V);
    if (r.fail_) return fail("WEIGHTS_TRUNCATED");
    if (r.at != b.size()) return fail("WEIGHTS_TRAILING");
    return ok(std::move(w));
}

Expected<Unit> TinyWeights::save(const std::string& path) const {
    Bytes b = serialize();
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return fail("WEIGHTS_IO: cannot open " + path);
    size_t n = std::fwrite(b.data(), 1, b.size(), f);
    std::fclose(f);
    if (n != b.size()) return fail("WEIGHTS_IO: short write");
    return ok_unit();
}

Expected<TinyWeights> TinyWeights::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return fail("WEIGHTS_IO: cannot open " + path);
    Bytes b((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return deserialize(b);
}

std::string TinyWeights::digest() const { return sha256_hex(serialize()); }

// ---------------------------------------------------------------------------
// plan
// ---------------------------------------------------------------------------
Json TinyPlan::to_json() const {
    Json j = Json::object();
    j.set("embedding", embedding);
    j.set("linear", linear);
    j.set("argmax_k", argmax_k);
    j.set("max_k", max_k);
    return j;
}

Expected<TinyPlan> TinyPlan::from_json(const Json& j) {
    TinyPlan p;
    OC2_TRY(e, j.get_string("embedding"));
    OC2_TRY(l, j.get_string("linear"));
    OC2_TRY(ak, j.get_int("argmax_k"));
    OC2_TRY(mk, j.get_int("max_k"));
    if ((e != "wmask" && e != "generic") || (l != "wmask" && l != "generic")) return fail("PLAN_INVALID: realization");
    if (ak != 2 && ak != 4 && ak != 8) return fail("PLAN_INVALID: argmax_k");
    if (mk != 2 && mk != 4) return fail("PLAN_INVALID: max_k");
    p.embedding = e;
    p.linear = l;
    p.argmax_k = static_cast<int>(ak);
    p.max_k = static_cast<int>(mk);
    return ok(p);
}

std::string TinyPlan::key() const { return to_json().dump(); }

// ---------------------------------------------------------------------------
// plaintext reference
// ---------------------------------------------------------------------------
namespace {
std::vector<double> layernorm_d(const std::vector<double>& x, const std::vector<float>& g, const std::vector<float>& b,
                                double eps) {
    std::vector<double> gd(g.begin(), g.end()), bd(b.begin(), b.end());
    return layernorm_ref(x, &gd, &bd, eps, false);
}
// y = x W + b, x (n x din), W (din x dout) float
std::vector<double> linear_d(const std::vector<double>& x, size_t n, size_t din, const std::vector<float>& W, size_t dout,
                             const std::vector<float>* b) {
    std::vector<double> y(n * dout, 0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t k = 0; k < din; ++k) {
            double xv = x[i * din + k];
            for (size_t j = 0; j < dout; ++j) y[i * dout + j] += xv * W[k * dout + j];
        }
    if (b)
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < dout; ++j) y[i * dout + j] += (*b)[j];
    return y;
}
}  // namespace

std::vector<double> TinyReference::last_logits(const std::vector<int>& tokens) const {
    size_t n = tokens.size(), d = w_.cfg.d_model, H = w_.cfg.n_head, dh = d / H, F = w_.cfg.d_ff, V = w_.cfg.vocab;
    std::vector<double> x(n * d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j)
            x[i * d + j] = static_cast<double>(w_.wte[static_cast<size_t>(tokens[i]) * d + j]) +
                           static_cast<double>(w_.wpe[i * d + j]);
    for (const auto& L : w_.layers) {
        std::vector<double> h(n * d);
        for (size_t i = 0; i < n; ++i) {
            std::vector<double> row(x.begin() + static_cast<std::ptrdiff_t>(i * d), x.begin() + static_cast<std::ptrdiff_t>((i + 1) * d));
            auto ln = layernorm_d(row, L.ln1_g, L.ln1_b, 1e-5);
            std::copy(ln.begin(), ln.end(), h.begin() + static_cast<std::ptrdiff_t>(i * d));
        }
        auto qkv = linear_d(h, n, d, L.w_qkv, 3 * d, &L.b_qkv);
        std::vector<double> att(n * d, 0.0);
        for (size_t hh = 0; hh < H; ++hh) {
            for (size_t i = 0; i < n; ++i) {
                std::vector<double> sc(i + 1);
                for (size_t j = 0; j <= i; ++j) {
                    double s = 0.0;
                    for (size_t k = 0; k < dh; ++k)
                        s += qkv[i * 3 * d + hh * dh + k] * qkv[j * 3 * d + d + hh * dh + k];
                    sc[j] = s / std::sqrt(static_cast<double>(dh));
                }
                auto pr = softmax_ref(sc);
                for (size_t j = 0; j <= i; ++j)
                    for (size_t k = 0; k < dh; ++k) att[i * d + hh * dh + k] += pr[j] * qkv[j * 3 * d + 2 * d + hh * dh + k];
            }
        }
        auto o = linear_d(att, n, d, L.w_out, d, &L.b_out);
        for (size_t i = 0; i < n * d; ++i) x[i] += o[i];
        std::vector<double> h2(n * d);
        for (size_t i = 0; i < n; ++i) {
            std::vector<double> row(x.begin() + static_cast<std::ptrdiff_t>(i * d), x.begin() + static_cast<std::ptrdiff_t>((i + 1) * d));
            auto ln = layernorm_d(row, L.ln2_g, L.ln2_b, 1e-5);
            std::copy(ln.begin(), ln.end(), h2.begin() + static_cast<std::ptrdiff_t>(i * d));
        }
        auto f = linear_d(h2, n, d, L.w_fc, F, &L.b_fc);
        for (auto& v : f) v = gelu_ref(v);
        auto pj = linear_d(f, n, F, L.w_proj, d, &L.b_proj);
        for (size_t i = 0; i < n * d; ++i) x[i] += pj[i];
    }
    std::vector<double> last(x.begin() + static_cast<std::ptrdiff_t>((n - 1) * d), x.end());
    auto hf = layernorm_d(last, w_.lnf_g, w_.lnf_b, 1e-5);
    return linear_d(hf, 1, d, w_.w_lm, V, nullptr);
}

std::vector<int> TinyReference::generate(const std::vector<int>& prompt, int n_generate) const {
    std::vector<int> toks = prompt, out;
    for (int g = 0; g < n_generate; ++g) {
        auto lg = last_logits(toks);
        int t = static_cast<int>(argmax_ref(lg));
        out.push_back(t);
        toks.push_back(t);
    }
    return out;
}

// ---------------------------------------------------------------------------
// TinyDecoder
// ---------------------------------------------------------------------------
TinyDecoder::TinyDecoder(Party& party, const TinyWeights* weights_or_null, const TinyConfig& cfg, TinyPlan plan)
    : p_(party), W_(weights_or_null), cfg_(cfg), plan_(std::move(plan)) {
    K_.assign(cfg.n_layers, Mat(0, cfg.d_model));
    V_.assign(cfg.n_layers, Mat(0, cfg.d_model));
}

void TinyDecoder::reset_cache() {
    K_.assign(cfg_.n_layers, Mat(0, cfg_.d_model));
    V_.assign(cfg_.n_layers, Mat(0, cfg_.d_model));
    T_ = 0;
}

Expected<Unit> TinyDecoder::share_vector(const std::string& name, const std::vector<float>* v, size_t rows, size_t cols) {
    if (p_.party() == 0) {
        if (!v || v->size() != rows * cols) return fail("WEIGHTS_SHAPE: " + name);
        std::vector<double> d(v->begin(), v->end());
        OC2_TRY(enc, p_.encode(d, rows, cols));
        if (name == "wte" || name == "lm_head" || name.find(".qkv") != std::string::npos || name.find(".out") != std::string::npos ||
            name.find(".fc") != std::string::npos || name.find(".proj") != std::string::npos)
            enc_mats_[name] = enc;
        OC2_TRY(sh, p_.share_input(&enc, 0, rows, cols));
        p_.set_vector_share(name, sh);
    } else {
        OC2_TRY(sh, p_.share_input(nullptr, 0, rows, cols));
        p_.set_vector_share(name, sh);
    }
    return ok_unit();
}

Expected<Value> TinyDecoder::setup() {
    uint64_t b0 = p_.channel().stats.bytes_sent + p_.channel().stats.bytes_recv;
    uint64_t r0 = p_.channel().stats.rounds;
    double t0 = now_s();
    size_t d = cfg_.d_model, V = cfg_.vocab, F = cfg_.d_ff;
    const TinyWeights* W = W_;
    if (p_.party() == 0 && !W) return fail("WEIGHTS_MISSING: P0 needs the model weights");
    if (p_.party() == 0 && (W->cfg.d_model != d || W->cfg.vocab != V || W->cfg.n_layers != cfg_.n_layers))
        return fail("WEIGHTS_SHAPE: weights differ from the public config");
    OC2_CHECK(share_vector("wte", W ? &W->wte : nullptr, V, d));
    for (uint32_t l = 0; l < cfg_.n_layers; ++l) {
        std::string h = "h" + std::to_string(l) + ".";
        const TinyLayer* L = W ? &W->layers[l] : nullptr;
        OC2_CHECK(share_vector(h + "ln_1.g", L ? &L->ln1_g : nullptr, 1, d));
        OC2_CHECK(share_vector(h + "ln_1.b", L ? &L->ln1_b : nullptr, 1, d));
        OC2_CHECK(share_vector(h + "qkv", L ? &L->w_qkv : nullptr, d, 3 * d));
        OC2_CHECK(share_vector(h + "qkv.b", L ? &L->b_qkv : nullptr, 1, 3 * d));
        OC2_CHECK(share_vector(h + "out", L ? &L->w_out : nullptr, d, d));
        OC2_CHECK(share_vector(h + "out.b", L ? &L->b_out : nullptr, 1, d));
        OC2_CHECK(share_vector(h + "ln_2.g", L ? &L->ln2_g : nullptr, 1, d));
        OC2_CHECK(share_vector(h + "ln_2.b", L ? &L->ln2_b : nullptr, 1, d));
        OC2_CHECK(share_vector(h + "fc", L ? &L->w_fc : nullptr, d, F));
        OC2_CHECK(share_vector(h + "fc.b", L ? &L->b_fc : nullptr, 1, F));
        OC2_CHECK(share_vector(h + "proj", L ? &L->w_proj : nullptr, F, d));
        OC2_CHECK(share_vector(h + "proj.b", L ? &L->b_proj : nullptr, 1, d));
    }
    OC2_CHECK(share_vector("ln_f.g", W ? &W->lnf_g : nullptr, 1, d));
    OC2_CHECK(share_vector("ln_f.b", W ? &W->lnf_b : nullptr, 1, d));
    OC2_CHECK(share_vector("lm_head", W ? &W->w_lm : nullptr, d, V));
    // The LM head is wmask under every plan: installed here, charged to T_cold.
    const Mat* wlm = p_.party() == 0 ? &enc_mats_["lm_head"] : nullptr;
    OC2_TRY(inst, p_.install_wmask("lm_head", wlm));
    (void)inst;
    setup_done_ = true;
    return ok(dict({{"bytes", static_cast<int64_t>(p_.channel().stats.bytes_sent + p_.channel().stats.bytes_recv - b0)},
                    {"seconds", now_s() - t0},
                    {"rounds", static_cast<int64_t>(p_.channel().stats.rounds - r0)}}));
}

std::vector<std::string> TinyDecoder::required_objects(const TinyPlan& plan) const {
    std::vector<std::string> out;
    if (plan.embedding == "wmask") out.push_back("wte");
    if (plan.linear == "wmask") {
        for (uint32_t l = 0; l < cfg_.n_layers; ++l) {
            std::string h = "h" + std::to_string(l) + ".";
            out.push_back(h + "qkv");
            out.push_back(h + "out");
            out.push_back(h + "fc");
            out.push_back(h + "proj");
        }
    }
    out.push_back("lm_head");
    return out;
}

std::vector<std::string> TinyDecoder::missing_objects(const TinyPlan& plan) const {
    std::vector<std::string> out;
    for (const auto& o : required_objects(plan))
        if (!p_.installed(o)) out.push_back(o);
    return out;
}

Expected<Value> TinyDecoder::install_for_plan(const TinyPlan& plan) {
    uint64_t b0 = p_.channel().stats.bytes_sent + p_.channel().stats.bytes_recv;
    uint64_t r0 = p_.channel().stats.rounds;
    double t0 = now_s();
    List objs;
    for (const auto& o : missing_objects(plan)) {
        const Mat* W = nullptr;
        if (p_.party() == 0) {
            auto it = enc_mats_.find(o);
            if (it == enc_mats_.end()) return fail("WEIGHTS_MISSING: no encoded matrix for " + o);
            W = &it->second;
        }
        OC2_TRY(rec, p_.install_wmask(o, W));
        (void)rec;
        objs.push_back(Value(o));
    }
    plan_ = plan;
    return ok(dict({{"objects", Value(objs)},
                    {"count", static_cast<int64_t>(objs.size())},
                    {"bytes", static_cast<int64_t>(p_.channel().stats.bytes_sent + p_.channel().stats.bytes_recv - b0)},
                    {"seconds", now_s() - t0},
                    {"rounds", static_cast<int64_t>(p_.channel().stats.rounds - r0)}}));
}

Expected<Mat> TinyDecoder::linear(const Mat& x, const std::string& mat_id) {
    if (plan_.linear == "wmask") return p_.linear_wmask(x, mat_id, true);
    return p_.linear_generic(x, mat_id, true);
}

Expected<Mat> TinyDecoder::embed(const std::vector<int>* ids, size_t n, size_t pos0) {
    size_t d = cfg_.d_model, V = cfg_.vocab;
    // The client's one-hot rows are its own share (server share = 0): no traffic.
    Mat e(n, V);
    if (p_.party() == 1) {
        if (!ids || ids->size() != n) return fail("TOKENS_SHAPE: P1 must pass n token ids");
        for (size_t i = 0; i < n; ++i) {
            int t = (*ids)[i];
            if (t < 0 || static_cast<size_t>(t) >= V) return fail("TOKEN_RANGE: token id outside the vocabulary");
            e.a[i * V + static_cast<size_t>(t)] = 1;
        }
    }
    Mat x;
    if (plan_.embedding == "wmask") {
        OC2_TRY(xw, p_.linear_wmask(e, "wte", false));
        x = std::move(xw);
    } else {
        OC2_TRY(xg, p_.linear_generic(e, "wte", false));
        x = std::move(xg);
    }
    if (pos0 + n > cfg_.max_pos) return fail("POSITION_RANGE: position beyond max_pos");
    if (p_.party() == 0) {
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d; ++j) x.a[i * d + j] += p_.enc(static_cast<double>(W_->wpe[(pos0 + i) * d + j]));
    }
    return ok(std::move(x));
}

Expected<Mat> TinyDecoder::attention(size_t li, const Mat& h) {
    size_t n = h.rows, d = cfg_.d_model, H = cfg_.n_head, dh = d / H;
    std::string hid = "h" + std::to_string(li) + ".";
    OC2_TRY(qkv, linear(h, hid + "qkv"));  // n x 3d
    Mat q = qkv.cols_range(0, d), k = qkv.cols_range(d, 2 * d), v = qkv.cols_range(2 * d, 3 * d);
    K_[li] = K_[li].rows == 0 ? k : Mat::vstack(K_[li], k);
    V_[li] = V_[li].rows == 0 ? v : Mat::vstack(V_[li], v);
    size_t T = K_[li].rows;
    std::vector<Mat> Qs, KTs;
    for (size_t hh = 0; hh < H; ++hh) {
        Qs.push_back(q.cols_range(hh * dh, (hh + 1) * dh));
        KTs.push_back(K_[li].cols_range(hh * dh, (hh + 1) * dh).transpose());  // dh x T
    }
    OC2_TRY(scores, p_.matmul_shared_batch(Qs, KTs, true));  // H x (n x T)
    Mat S(H * n, T);
    std::vector<uint8_t> mask(H * n * T, 0);
    for (size_t hh = 0; hh < H; ++hh) {
        Mat sc = p_.mul_public_real(scores[hh], 1.0 / std::sqrt(static_cast<double>(dh)));
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < T; ++j) {
                S.a[(hh * n + i) * T + j] = sc.a[i * T + j];
                mask[(hh * n + i) * T + j] = (j <= T - n + i) ? 1 : 0;
            }
    }
    OC2_TRY(P, p_.softmax_rows(S, &mask, plan_.max_k));
    std::vector<Mat> Ps, Vs;
    for (size_t hh = 0; hh < H; ++hh) {
        Ps.push_back(P.rows_range(hh * n, (hh + 1) * n));
        Vs.push_back(V_[li].cols_range(hh * dh, (hh + 1) * dh));
    }
    OC2_TRY(os, p_.matmul_shared_batch(Ps, Vs, true));  // H x (n x dh)
    Mat o(n, d);
    for (size_t hh = 0; hh < H; ++hh)
        for (size_t i = 0; i < n; ++i)
            for (size_t kk = 0; kk < dh; ++kk) o.a[i * d + hh * dh + kk] = os[hh].a[i * dh + kk];
    return linear(o, hid + "out");
}

Expected<Mat> TinyDecoder::layer(size_t li, const Mat& x) {
    std::string hid = "h" + std::to_string(li) + ".";
    OC2_TRY(g1, p_.vector_share(hid + "ln_1.g"));
    OC2_TRY(b1, p_.vector_share(hid + "ln_1.b"));
    OC2_TRY(h, p_.layernorm_rows(x, &g1, &b1, 1e-5, false));
    OC2_TRY(a, attention(li, h));
    Mat x1 = ring_add(x, a);
    OC2_TRY(g2, p_.vector_share(hid + "ln_2.g"));
    OC2_TRY(b2, p_.vector_share(hid + "ln_2.b"));
    OC2_TRY(h2, p_.layernorm_rows(x1, &g2, &b2, 1e-5, false));
    OC2_TRY(f, linear(h2, hid + "fc"));
    OC2_TRY(g, p_.gelu(f));
    OC2_TRY(y, linear(g, hid + "proj"));
    return ok(ring_add(x1, y));
}

Expected<int> TinyDecoder::head(const Mat& x_last) {
    OC2_TRY(gf, p_.vector_share("ln_f.g"));
    OC2_TRY(bf, p_.vector_share("ln_f.b"));
    OC2_TRY(h, p_.layernorm_rows(x_last, &gf, &bf, 1e-5, false));
    OC2_TRY(logits, p_.linear_wmask(h, "lm_head", true));  // 1 x V
    OC2_TRY(idx, p_.argmax_row(logits, plan_.argmax_k, true, "token"));
    if (p_.party() == 1) {
        if (idx.size() != 1) return fail("RELEASE_SHAPE: expected one index");
        return ok(static_cast<int>(static_cast<int64_t>(idx.a[0])));
    }
    return ok(-1);
}

Expected<int> TinyDecoder::prefill(const std::vector<int>* token_ids, size_t n) {
    if (!setup_done_) return fail("DECODER_SETUP: call setup() first");
    if (n == 0) return fail("TOKENS_SHAPE: empty prompt");
    reset_cache();
    OC2_TRY(x0, embed(token_ids, n, 0));
    Mat x = std::move(x0);
    for (uint32_t l = 0; l < cfg_.n_layers; ++l) {
        OC2_TRY(xl, layer(l, x));
        x = std::move(xl);
    }
    T_ = n;
    return head(x.row(n - 1));
}

Expected<int> TinyDecoder::decode_step(int token_id, size_t position) {
    if (!setup_done_) return fail("DECODER_SETUP: call setup() first");
    if (position != T_) return fail("POSITION_RANGE: decode position must equal the KV length");
    std::vector<int> ids{token_id};
    OC2_TRY(x0, embed(p_.party() == 1 ? &ids : nullptr, 1, position));
    Mat x = std::move(x0);
    for (uint32_t l = 0; l < cfg_.n_layers; ++l) {
        OC2_TRY(xl, layer(l, x));
        x = std::move(xl);
    }
    T_ = position + 1;
    return head(x);
}

}  // namespace highlevel
}  // namespace opencrab2
