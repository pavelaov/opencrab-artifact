// Implementation of dealer.hpp: Dealer (recipient-bound bundles), material
// estimate, DealerClient (verification, pools, ledger accounting, burn on
// cancellation) and the in-process dealer link used by the tests and the
// single-process session.
#include <opencrab2/highlevel/dealer.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>

namespace opencrab2 {
namespace highlevel {

// Cross-language dealer bundles: DCF-based payloads carry their keys in the
// "OC2DCF01" legacy layout. The C++ readers
// (DcfKeyBatch::from_bytes and the composite materials) auto-detect both
// layouts; the Python reader accepts only this one, so every producer writes it.
static constexpr lowlevel::dcf::WireLayout kBundleKeyLayout = lowlevel::dcf::WireLayout::PythonV2;

namespace ll {

double rng_uniform(Rng& rng) { return static_cast<double>(rng.next_u64() >> 11) * (1.0 / 9007199254740992.0); }

double rng_normal(Rng& rng) {
    double u1 = rng_uniform(rng), u2 = rng_uniform(rng);
    if (u1 < 1e-300) u1 = 1e-300;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}

void put_u64_vec(Bytes& out, const std::vector<uint64_t>& v) {
    size_t at = out.size();
    out.resize(at + v.size() * 8);
    for (size_t i = 0; i < v.size(); ++i)
        for (int k = 0; k < 8; ++k) out[at + 8 * i + static_cast<size_t>(k)] = static_cast<uint8_t>(v[i] >> (8 * k));
}

Expected<std::vector<uint64_t>> get_u64_vec(const uint8_t* p, size_t len, size_t count, size_t* consumed) {
    if (count > (len - std::min(len, *consumed)) / 8) return fail("MATERIAL_WIRE: truncated u64 vector");
    std::vector<uint64_t> v(count);
    for (size_t i = 0; i < count; ++i) v[i] = get_u64le(p + *consumed + 8 * i);
    *consumed += count * 8;
    return ok(std::move(v));
}

}  // namespace ll

// ---------------------------------------------------------------------------
// ring matmul
// ---------------------------------------------------------------------------
void ring_matmul(const std::vector<uint64_t>& A, size_t n, size_t k, const std::vector<uint64_t>& B, size_t m,
                 std::vector<uint64_t>& C) {
    C.assign(n * m, 0);
    for (size_t i = 0; i < n; ++i) {
        uint64_t* crow = &C[i * m];
        for (size_t j = 0; j < k; ++j) {
            uint64_t a = A[i * k + j];
            if (a == 0) continue;
            const uint64_t* brow = &B[j * m];
            for (size_t l = 0; l < m; ++l) crow[l] += a * brow[l];
        }
    }
}

// ---------------------------------------------------------------------------
// header params (Python spellings; the earlier C++ spellings are still read)
// ---------------------------------------------------------------------------
namespace {

// A list of positive ints under `key` (shape: [d, m]); false when absent or malformed.
bool read_shape(const Value& params, const char* key, std::vector<int64_t>& out) {
    out.clear();
    const Value* v = params.get(key);
    if (!v || !v->is_list()) return false;
    for (const Value& e : *v->as_list().value) {
        if (!e.is_int()) return false;
        int64_t d = e.as_int().value;
        if (d <= 0) return false;
        out.push_back(d);
    }
    return !out.empty();
}

Value shape_value(std::initializer_list<size_t> dims) {
    List l;
    for (size_t d : dims) l.push_back(Value(static_cast<int64_t>(d)));
    return Value(std::move(l));
}

// A positive int under `key`; -1 when absent or malformed.
int64_t pos_int(const Value& params, const char* key) {
    const Value* v = params.get(key);
    if (!v || !v->is_int()) return -1;
    int64_t x = v->as_int().value;
    return x > 0 ? x : -1;
}

// matrix_triple: {shape_a: [n, k], shape_b: [k, m]} (binding) or {n, k, m}.
Expected<std::array<size_t, 3>> matrix_triple_shape(const Value& params) {
    std::vector<int64_t> sa, sb;
    if (read_shape(params, "shape_a", sa) || read_shape(params, "shape_b", sb)) {
        if (!read_shape(params, "shape_a", sa) || !read_shape(params, "shape_b", sb) || sa.size() != 2 || sb.size() != 2 ||
            sa[1] != sb[0])
            return fail("MATERIAL_PARAMS: matrix_triple needs shape_a [n, k] and shape_b [k, m]");
        return ok(std::array<size_t, 3>{static_cast<size_t>(sa[0]), static_cast<size_t>(sa[1]), static_cast<size_t>(sb[1])});
    }
    int64_t n = pos_int(params, "n"), k = pos_int(params, "k"), m = pos_int(params, "m");
    if (n < 0 || k < 0 || m < 0) return fail("MATERIAL_PARAMS: matrix_triple needs shape_a and shape_b");
    return ok(std::array<size_t, 3>{static_cast<size_t>(n), static_cast<size_t>(k), static_cast<size_t>(m)});
}

// wmask_mask: {mat_id, shape: [d, m]} (binding) or {mat_id, d, m}.
Expected<std::array<size_t, 2>> wmask_mask_shape(const Value& params) {
    std::vector<int64_t> sh;
    if (params.has("shape")) {
        if (!read_shape(params, "shape", sh) || sh.size() != 2) return fail("MATERIAL_PARAMS: wmask_mask shape must be [d, m]");
        return ok(std::array<size_t, 2>{static_cast<size_t>(sh[0]), static_cast<size_t>(sh[1])});
    }
    int64_t d = pos_int(params, "d"), m = pos_int(params, "m");
    if (d < 0 || m < 0) return fail("MATERIAL_PARAMS: wmask_mask needs shape [d, m]");
    return ok(std::array<size_t, 2>{static_cast<size_t>(d), static_cast<size_t>(m)});
}

// Element count of a shape list when present (product of the dims); `fallback` otherwise.
Expected<size_t> numel_or(const Value& params, const char* key, size_t fallback) {
    if (!params.has(key)) return ok(fallback);
    std::vector<int64_t> sh;
    if (!read_shape(params, key, sh)) return fail(std::string("MATERIAL_PARAMS: malformed ") + key);
    uint64_t n = 1;
    for (int64_t d : sh) {
        if (static_cast<uint64_t>(d) > (static_cast<uint64_t>(1) << 40) / n) return fail("MATERIAL_PARAMS: shape too large");
        n *= static_cast<uint64_t>(d);
    }
    return ok(static_cast<size_t>(n));
}

}  // namespace

// ---------------------------------------------------------------------------
// Dealer
// ---------------------------------------------------------------------------
Dealer::Dealer(DealerConfig cfg, ll::Rng rng, Bytes hmac_key0, Bytes hmac_key1)
    : cfg_(std::move(cfg)), rng_(rng), os_seeded_(rng.os_seeded()), key0_(std::move(hmac_key0)), key1_(std::move(hmac_key1)) {}

std::string Dealer::next_material_id(const std::string& type) {
    uint8_t rnd[8];
    rng_.fill(rnd, 8);
    std::ostringstream os;
    os << cfg_.setup_id << "-" << type << "-" << (counter_++) << "-" << to_hex(rnd, 8);
    return os.str();
}

Value Dealer::base_header(int recipient, const std::string& type, const std::string& material_id, size_t count,
                          const std::string& gate_id, Value params, List parts) {
    return dict({{"profile_id", cfg_.profile_id},
                 {"setup_id", cfg_.setup_id},
                 {"model_version", cfg_.model_version},
                 {"numeric_contract_hash", cfg_.numeric_contract_hash},
                 {"recipient", static_cast<int64_t>(recipient)},
                 {"gate_id", gate_id},
                 {"material_type", type},
                 {"material_id", material_id},
                 {"count", static_cast<int64_t>(count)},
                 {"bytes", static_cast<int64_t>(0)},
                 {"params", std::move(params)},
                 {"parts", Value(std::move(parts))},
                 {"dealer_mode", cfg_.mode},
                 {"insecure", insecure()}});
}

Expected<Bundle> Dealer::seal(int recipient, Value header, Bytes payload) {
    if (recipient != 0 && recipient != 1) return fail("DEALER_RECIPIENT");
    header.set("recipient", static_cast<int64_t>(recipient));
    header.set("bytes", static_cast<int64_t>(payload.size()));
    const Bytes& key = recipient == 0 ? key0_ : key1_;
    OC2_TRY(tag, hmac_tag_with_payload(key, header, payload));
    Bundle b;
    b.header = std::move(header);
    b.payload = std::move(payload);
    b.tag = std::move(tag);
    return ok(std::move(b));
}

Expected<BundlePair> Dealer::seal_pair(const std::string& type, size_t count, const std::string& gate_id, Value params,
                                       List parts, Bytes payload0, Bytes payload1) {
    std::string mid = next_material_id(type);
    OC2_TRY(b0, seal(0, base_header(0, type, mid, count, gate_id, params, parts), std::move(payload0)));
    OC2_TRY(b1, seal(1, base_header(1, type, mid, count, gate_id, params, parts), std::move(payload1)));
    bytes_generated_ += b0.payload.size() + b1.payload.size();
    bundles_generated_ += 2;
    BundlePair bp;
    bp.for0 = std::move(b0);
    bp.for1 = std::move(b1);
    return ok(std::move(bp));
}

Expected<BundlePair> Dealer::make_beaver_like(const std::string& type, size_t count, const std::string& gate_id) {
    if (count == 0) return fail("MATERIAL_PARAMS: count = 0");
    std::vector<uint64_t> a(count), b(count), c(count);
    for (size_t i = 0; i < count; ++i) {
        a[i] = rng_.next_u64();
        b[i] = rng_.next_u64();
        c[i] = a[i] * b[i];
    }
    auto as = ll::share_vec(a, rng_), bs = ll::share_vec(b, rng_), cs = ll::share_vec(c, rng_);
    Bytes p0, p1;
    ll::put_u64_vec(p0, as.first); ll::put_u64_vec(p0, bs.first); ll::put_u64_vec(p0, cs.first);
    ll::put_u64_vec(p1, as.second); ll::put_u64_vec(p1, bs.second); ll::put_u64_vec(p1, cs.second);
    Value params = dict({{"shape", shape_value({count})}, {"elements", static_cast<int64_t>(count)}});
    return seal_pair(type, count, gate_id, params, List{Value("a"), Value("b"), Value("c")}, std::move(p0), std::move(p1));
}

Expected<BundlePair> Dealer::make_beaver(size_t count, const std::string& gate_id) {
    return make_beaver_like("beaver", count, gate_id);
}

Expected<BundlePair> Dealer::make_kary_select(size_t count, const std::string& gate_id) {
    return make_beaver_like("kary_select", count, gate_id);
}

Expected<BundlePair> Dealer::make_matrix_triples(size_t n, size_t k, size_t m, size_t count, const std::string& gate_id) {
    if (count == 0 || n == 0 || k == 0 || m == 0) return fail("MATERIAL_PARAMS: matrix triple shape");
    Bytes p0, p1;
    std::vector<uint64_t> A(n * k), B(k * m), C;
    for (size_t t = 0; t < count; ++t) {
        for (auto& v : A) v = rng_.next_u64();
        for (auto& v : B) v = rng_.next_u64();
        ring_matmul(A, n, k, B, m, C);
        auto as = ll::share_vec(A, rng_), bs = ll::share_vec(B, rng_), cs = ll::share_vec(C, rng_);
        ll::put_u64_vec(p0, as.first); ll::put_u64_vec(p0, bs.first); ll::put_u64_vec(p0, cs.first);
        ll::put_u64_vec(p1, as.second); ll::put_u64_vec(p1, bs.second); ll::put_u64_vec(p1, cs.second);
    }
    Value params = dict({{"shape_a", shape_value({n, k})}, {"shape_b", shape_value({k, m})}});
    return seal_pair("matrix_triple", count, gate_id, params, List{Value("A"), Value("B"), Value("C")}, std::move(p0),
                     std::move(p1));
}

Expected<BundlePair> Dealer::make_drelu(size_t M, int n_bits, bool with_product, const std::string& gate_id) {
    if (M == 0) return fail("MATERIAL_PARAMS: M = 0");
    OC2_TRY(mm, lowlevel::drelu::drelu_material(M, n_bits, rng_, with_product));
    Value params = dict({{"n_bits", static_cast<int64_t>(n_bits)}, {"with_product", with_product},
                         {"P", static_cast<int64_t>(with_product ? 2 : 1)}});
    return seal_pair("drelu", M, gate_id, params, List{Value("drelu_material")}, mm.first.to_bytes(kBundleKeyLayout),
                     mm.second.to_bytes(kBundleKeyLayout));
}

Expected<BundlePair> Dealer::make_trunc(size_t M, int f, int L, const std::string& gate_id) {
    if (M == 0) return fail("MATERIAL_PARAMS: M = 0");
    OC2_TRY(mm, lowlevel::trunc::trunc_material(M, f, L, rng_));
    Value params = dict({{"f", static_cast<int64_t>(f)}, {"L", static_cast<int64_t>(L)}});
    return seal_pair("trunc_faithful", M, gate_id, params, List{Value("trunc_material")}, mm.first.to_bytes(kBundleKeyLayout),
                     mm.second.to_bytes(kBundleKeyLayout));
}

Expected<BundlePair> Dealer::make_r2f(size_t M, uint64_t t, const std::string& gate_id) {
    if (M == 0) return fail("MATERIAL_PARAMS: M = 0");
    OC2_TRY(mm, lowlevel::r2f::r2f_material(M, t, rng_));
    Value params = dict({{"t", static_cast<int64_t>(t)}});
    return seal_pair("ring_to_field", M, gate_id, params, List{Value("r2f_material")}, mm.first.to_bytes(kBundleKeyLayout),
                     mm.second.to_bytes(kBundleKeyLayout));
}

Expected<BundlePair> Dealer::make_wmask_mask(const std::string& mat_id, size_t d, size_t m, const std::string& gate_id,
                                             int64_t generation) {
    if (d == 0 || m == 0) return fail("MATERIAL_PARAMS: wmask shape");
    if (generation < 0) return fail("MATERIAL_PARAMS: negative generation");
    auto known = mask_generation_.find(mat_id);
    if (known != mask_generation_.end() && generation <= known->second)
        return fail("DEALER_MASK_EXISTS: a mask for " + mat_id + " was already issued for this generation");
    // A fresh uniform B on every issue: a new generation (or a new model version) never reuses B
    // (E1 - E0 = W1 - W0 would leak the weight difference otherwise).
    std::vector<uint64_t> B(d * m);
    for (auto& v : B) v = rng_.next_u64();
    auto bs = ll::share_vec(B, rng_);
    masks_[mat_id] = std::move(B);
    mask_shapes_[mat_id] = {d, m};
    mask_generation_[mat_id] = generation;
    Bytes p0, p1;
    ll::put_u64_vec(p0, bs.first);
    ll::put_u64_vec(p1, bs.second);
    Value params = dict({{"mat_id", mat_id}, {"shape", shape_value({d, m})}, {"model_version", cfg_.model_version}});
    if (generation > 0) params.set("generation", generation);
    return seal_pair("wmask_mask", 1, gate_id, params, List{Value("B_share")}, std::move(p0), std::move(p1));
}

Expected<BundlePair> Dealer::make_wmask_calls(const std::string& mat_id, size_t n_rows, size_t count,
                                              const std::string& gate_id) {
    auto it = masks_.find(mat_id);
    if (it == masks_.end()) return fail("DEALER_UNKNOWN_MASK: " + mat_id);
    if (count == 0 || n_rows == 0) return fail("MATERIAL_PARAMS: wmask call count");
    size_t d = mask_shapes_[mat_id].first, m = mask_shapes_[mat_id].second;
    Bytes p0, p1;
    std::vector<uint64_t> A(n_rows * d), C;
    for (size_t t = 0; t < count; ++t) {
        for (auto& v : A) v = rng_.next_u64();
        ring_matmul(A, n_rows, d, it->second, m, C);
        auto as = ll::share_vec(A, rng_), cs = ll::share_vec(C, rng_);
        ll::put_u64_vec(p0, as.first); ll::put_u64_vec(p0, cs.first);
        ll::put_u64_vec(p1, as.second); ll::put_u64_vec(p1, cs.second);
    }
    Value params = dict({{"mat_id", mat_id}, {"n_rows", static_cast<int64_t>(n_rows)}, {"d", static_cast<int64_t>(d)},
                         {"m", static_cast<int64_t>(m)}});
    return seal_pair("wmask_call", count, gate_id, params, List{Value("A"), Value("C")}, std::move(p0), std::move(p1));
}

Expected<BundlePair> Dealer::make_vector_share(const std::string& name, const std::vector<uint64_t>& v_enc,
                                               const std::string& gate_id) {
    if (v_enc.empty()) return fail("MATERIAL_PARAMS: empty vector");
    if (!insecure())
        return fail("DEALER_PLAINTEXT_REFUSED: vector_share hands a plaintext vector to the dealer; an eligible dealer "
                    "refuses it (share server-private vectors with Party::share_input)");
    auto vs = ll::share_vec(v_enc, rng_);
    Bytes p0, p1;
    ll::put_u64_vec(p0, vs.first);
    ll::put_u64_vec(p1, vs.second);
    Value params = dict({{"name", name}, {"shape", shape_value({v_enc.size()})}});
    return seal_pair("vector_share", 1, gate_id, params, List{Value("share")}, std::move(p0), std::move(p1));
}

// Request dispatch. Request params follow the Python spellings: Beaver-like material
// {shape: [...]} with count bundles of that shape, DCF-based material {M, ...} with count batches
// of M gates; the earlier C++ form (no shape / no M, count = elements) is still accepted. One
// request yields ONE bundle pair holding all requested elements.
Expected<BundlePair> Dealer::make(const std::string& type, const Value& params, size_t count, const std::string& gate_id) {
    auto geti = [&](const char* k) -> Expected<int64_t> { return params.get_int(k); };
    if (count == 0) return fail("MATERIAL_PARAMS: count = 0");
    // elements = M * count when params.M is given, count otherwise
    auto gates = [&]() -> Expected<size_t> {
        if (!params.has("M")) return ok(count);
        int64_t M = pos_int(params, "M");
        if (M < 0) return fail("MATERIAL_PARAMS: M must be a positive int");
        if (static_cast<uint64_t>(M) > (static_cast<uint64_t>(1) << 40) / count) return fail("MATERIAL_PARAMS: M too large");
        return ok(static_cast<size_t>(M) * count);
    };
    if (type == "beaver" || type == "kary_select") {
        OC2_TRY(per, numel_or(params, "shape", 1));
        return make_beaver_like(type, per * count, gate_id);
    }
    if (type == "matrix_triple") {
        OC2_TRY(sh, matrix_triple_shape(params));
        return make_matrix_triples(sh[0], sh[1], sh[2], count, gate_id);
    }
    if (type == "drelu") {
        OC2_TRY(nb, geti("n_bits"));
        bool wp = true;
        if (const Value* w = params.get("with_product")) {
            auto r = w->as_bool();
            if (r.ok) wp = r.value;
        }
        OC2_TRY(M, gates());
        return make_drelu(M, static_cast<int>(nb), wp, gate_id);
    }
    if (type == "trunc_faithful") {
        OC2_TRY(f, geti("f"));
        int64_t L = cfg_.logical_bits;
        if (params.has("L")) { OC2_TRY(Lv, geti("L")); L = Lv; }
        OC2_TRY(M, gates());
        return make_trunc(M, static_cast<int>(f), static_cast<int>(L), gate_id);
    }
    if (type == "ring_to_field") {
        OC2_TRY(t, geti("t"));
        OC2_TRY(M, gates());
        return make_r2f(M, static_cast<uint64_t>(t), gate_id);
    }
    if (type == "wmask_mask") {
        OC2_TRY(mid, params.get_string("mat_id"));
        OC2_TRY(sh, wmask_mask_shape(params));
        int64_t generation = 0;
        if (params.has("generation")) { OC2_TRY(g, geti("generation")); generation = g; }
        return make_wmask_mask(mid, sh[0], sh[1], gate_id, generation);
    }
    if (type == "wmask_call") {
        OC2_TRY(mid, params.get_string("mat_id")); OC2_TRY(n, geti("n_rows"));
        if (n <= 0) return fail("MATERIAL_PARAMS: n_rows must be positive");
        return make_wmask_calls(mid, static_cast<size_t>(n), count, gate_id);
    }
    if (type == "vector_share") {
        OC2_TRY(name, params.get_string("name"));
        OC2_TRY(arr, params.get_array("values"));
        return make_vector_share(name, arr.as_u64(), gate_id);
    }
    return fail("DEALER_UNKNOWN_TYPE: " + type);
}

// ---------------------------------------------------------------------------
// material estimate
// ---------------------------------------------------------------------------
Expected<TournamentMaterial> tournament_material(size_t n, int k, bool with_index) {
    if (n < 1) return fail("TOURNAMENT_PARAMS: n must be >= 1");
    if (k != 2 && k != 4 && k != 8) return fail("TOURNAMENT_PARAMS: k must be in {2, 4, 8}");
    TournamentMaterial t;
    size_t w = n;
    bool idx_public = true;
    const size_t K = static_cast<size_t>(k);
    while (w > 1) {
        const size_t extra = (with_index && !idx_public) ? 1 : 0;
        size_t groups = 0;
        for (size_t s = 0; s < w; s += K) {
            const size_t g = std::min(K, w - s);
            ++groups;
            t.drelu_p1 += g * (g - 1) / 2;
            if (g == 2) t.selects += 1 + extra;
            else if (g >= 3) t.drelu_p2 += (g - 1) * (1 + extra);
        }
        w = groups;
        idx_public = false;
        t.levels += 1;
    }
    t.rounds = 2 * t.levels;
    return ok(t);
}

namespace {
size_t numel(const std::vector<size_t>& s) {
    size_t n = 1;
    for (size_t v : s) n *= v;
    return s.empty() ? 1 : n;
}
struct Est {
    size_t beaver = 0, kary = 0, drelu_p1 = 0, drelu_p2 = 0, matrix = 0, wmask_call = 0, wmask_mask = 0, trunc = 0, r2f = 0;
    void add(const Est& o) {
        beaver += o.beaver; kary += o.kary; drelu_p1 += o.drelu_p1; drelu_p2 += o.drelu_p2; matrix += o.matrix;
        wmask_call += o.wmask_call; wmask_mask += o.wmask_mask; trunc += o.trunc; r2f += o.r2f;
    }
};
Expected<Est> estimate(const std::string& op, const std::vector<size_t>& shape, int k_ary) {
    Est e;
    if (op == "beaver_mul" || op == "mul" || op == "mul_fxp" || op == "square" || op == "select") {
        e.beaver = numel(shape);
    } else if (op == "trunc_faithful") {
        e.trunc = numel(shape);
    } else if (op == "drelu" || op == "compare_public") {
        e.drelu_p1 = numel(shape);
    } else if (op == "relu") {
        e.drelu_p2 = numel(shape);
    } else if (op == "ring_to_field" || op == "a2h_exact") {
        e.r2f = numel(shape);
    } else if (op == "matmul_shared") {
        if (shape.size() != 3) return fail("ESTIMATE_SHAPE: matmul needs (n, k, m)");
        e.matrix = shape[0] * shape[1] + shape[1] * shape[2] + shape[0] * shape[2];
    } else if (op == "linear_generic" || op == "embedding_generic") {
        if (shape.size() != 3) return fail("ESTIMATE_SHAPE: linear needs (n, d, m)");
        e.matrix = shape[0] * shape[1] + shape[1] * shape[2] + shape[0] * shape[2];
    } else if (op == "linear_wmask" || op == "embedding_wmask") {
        if (shape.size() != 3) return fail("ESTIMATE_SHAPE: linear needs (n, d, m)");
        e.wmask_call = shape[0] * shape[1] + shape[0] * shape[2];
    } else if (op == "install_wmask") {
        if (shape.size() != 2) return fail("ESTIMATE_SHAPE: install needs (d, m)");
        e.wmask_mask = shape[0] * shape[1];
    } else if (op == "gelu_clip12" || op == "silu_clip12" || op == "gelu" || op == "silu") {
        size_t N = numel(shape);
        e.drelu_p1 = 2 * N;
        e.beaver = (12 + 2) * N;
    } else if (op == "exp_neg") {
        size_t N = numel(shape);
        e.drelu_p1 = N;
        e.beaver = (2 + 7) * N;
    } else if (op == "recip") {
        size_t N = numel(shape);
        e.drelu_p1 = 11 * N;
        e.beaver = (1 + 5 + 4 + 1) * N;
    } else if (op == "rsqrt") {
        size_t N = numel(shape);
        e.drelu_p1 = 15 * N;
        e.beaver = (1 + 5 + 6 + 1) * N;
    } else if (op == "max_rows" || op == "argmax_row" || op == "argmax_k") {
        if (shape.empty()) return fail("ESTIMATE_SHAPE");
        size_t n = shape.back();
        size_t R = numel(std::vector<size_t>(shape.begin(), shape.end() - 1));
        OC2_TRY(t, tournament_material(n, k_ary, op != "max_rows"));
        e.drelu_p1 = R * t.drelu_p1;   // pairwise comparisons (P = 1)
        e.drelu_p2 = R * t.drelu_p2;   // fused node selects (payload)
        e.kary = R * t.selects;        // Beaver selects of binary nodes
    } else if (op == "softmax_rows") {
        if (shape.empty()) return fail("ESTIMATE_SHAPE");
        size_t n = shape.back();
        std::vector<size_t> rows(shape.begin(), shape.end() - 1);
        size_t R = numel(rows);
        OC2_TRY(mx, estimate("max_rows", shape, k_ary));
        OC2_TRY(ex, estimate("exp_neg", shape, k_ary));
        OC2_TRY(rc, estimate("recip", rows, k_ary));
        e.add(mx); e.add(ex); e.add(rc);
        e.beaver += R * n;
    } else if (op == "layernorm_rows" || op == "rmsnorm_rows") {
        if (shape.empty()) return fail("ESTIMATE_SHAPE");
        size_t d = shape.back();
        std::vector<size_t> rows(shape.begin(), shape.end() - 1);
        size_t R = numel(rows);
        OC2_TRY(rs, estimate("rsqrt", rows, k_ary));
        e.add(rs);
        e.beaver += 3 * R * d;
        e.trunc += R * d;
    } else {
        return fail("ESTIMATE_UNKNOWN_OP: " + op);
    }
    return ok(e);
}
}  // namespace

Expected<Value> material_estimate(const std::string& op_kind, const std::vector<size_t>& shape, int n_bits, int frac_bits,
                                  int k_ary) {
    OC2_TRY(e, estimate(op_kind, shape, k_ary));
    uint64_t bb = 24 * e.beaver;
    uint64_t kb = 24 * e.kary;
    uint64_t db = ll::drelu_bytes_per_element(n_bits, false) * e.drelu_p1 + ll::drelu_bytes_per_element(n_bits, true) * e.drelu_p2;
    uint64_t mb = 8 * e.matrix;
    uint64_t wcb = 8 * e.wmask_call;
    uint64_t wmb = 8 * e.wmask_mask;
    uint64_t tb = ll::trunc_bytes_per_element(frac_bits) * e.trunc;
    uint64_t rb = ll::r2f_bytes_per_element(0) * e.r2f;
    auto pair = [](size_t c, uint64_t b) { return Value(List{Value(static_cast<int64_t>(c)), Value(static_cast<int64_t>(b))}); };
    return ok(dict({{"beaver", pair(e.beaver, bb)},
                    {"kary_select", pair(e.kary, kb)},
                    {"drelu", pair(e.drelu_p1 + e.drelu_p2, db)},
                    {"drelu_p1", pair(e.drelu_p1, ll::drelu_bytes_per_element(n_bits, false) * e.drelu_p1)},
                    {"drelu_p2", pair(e.drelu_p2, ll::drelu_bytes_per_element(n_bits, true) * e.drelu_p2)},
                    {"matrix_triple", pair(e.matrix, mb)},
                    {"wmask_call", pair(e.wmask_call, wcb)},
                    {"wmask_mask", pair(e.wmask_mask, wmb)},
                    {"trunc_faithful", pair(e.trunc, tb)},
                    {"ring_to_field", pair(e.r2f, rb)},
                    {"bytes", static_cast<int64_t>(bb + kb + db + mb + wcb + wmb + tb + rb)}}));
}

// ---------------------------------------------------------------------------
// DealerClient
// ---------------------------------------------------------------------------
DealerClient::DealerClient(int party, MaterialLedger& ledger, Bytes hmac_key, std::string contract_hash)
    : party_(party), ledger_(ledger), key_(std::move(hmac_key)), contract_hash_(std::move(contract_hash)) {}

std::string DealerClient::type_key_drelu(int n_bits, bool with_product) const {
    return std::to_string(n_bits) + "|" + (with_product ? "2" : "1");
}

Expected<Unit> DealerClient::ingest(const Value& header, const Bytes& payload, const std::string& tag) {
    OC2_TRY(valid, hmac_verify_with_payload(key_, header, payload, tag));
    if (!valid) return fail("BUNDLE_TAG: HMAC verification failed");
    OC2_TRY(recipient, header.get_int("recipient"));
    if (recipient != party_) return fail("BUNDLE_RECIPIENT: bundle is not addressed to this party");
    OC2_TRY(ch, header.get_string("numeric_contract_hash"));
    if (ch != contract_hash_) return fail("CONTRACT_MISMATCH: bundle contract hash differs");
    OC2_TRY(nbytes, header.get_int("bytes"));
    if (nbytes < 0 || static_cast<size_t>(nbytes) != payload.size()) return fail("BUNDLE_BYTES: header bytes != payload size");
    OC2_CHECK(ledger_.check_binding(header));
    OC2_TRY(mid, header.get_string("material_id"));
    if (ledger_.seen(mid)) return fail("MATERIAL_REUSE: material_id seen before: " + mid);
    OC2_TRY(type, header.get_string("material_type"));
    OC2_TRY(count, header.get_int("count"));
    if (count < 0) return fail("MATERIAL_PARAMS: negative count");
    // Fail closed: a bundle counts as eligible only when its header DECLARES isolated
    // generation, i.e. insecure == false and dealer_mode == ISOLATED_RECIPIENT_BUNDLES are both
    // present. A missing or malformed declaration is treated as insecure.
    bool insecure = true;
    {
        bool declared_secure = false, isolated = false;
        if (const Value* ins = header.get("insecure")) {
            auto b = ins->as_bool();
            if (b.ok && !b.value) declared_secure = true;
        }
        if (const Value* dm = header.get("dealer_mode")) {
            auto s = dm->as_string();
            if (s.ok && s.value == "ISOLATED_RECIPIENT_BUNDLES") isolated = true;
        }
        insecure = !(declared_secure && isolated);
    }
    const Value* params = header.get("params");
    Value empty = dict({});
    if (!params || !params->is_dict()) params = &empty;
    Value version = dict({{"setup_id", header.get("setup_id") ? *header.get("setup_id") : Value("")},
                          {"model_version", header.get("model_version") ? *header.get("model_version") : Value("")},
                          {"numeric_contract_hash", ch},
                          {"profile_id", header.get("profile_id") ? *header.get("profile_id") : Value("")}});
    size_t cnt = static_cast<size_t>(count);
    const uint8_t* p = payload.data();
    size_t len = payload.size(), at = 0;
    if (type == "beaver" || type == "kary_select") {
        // params.shape (Python spelling) must describe exactly count elements when present
        OC2_TRY(declared, numel_or(*params, "shape", cnt));
        if (declared != cnt) return fail("MATERIAL_PARAMS: Beaver shape does not hold count elements");
        OC2_TRY(a, ll::get_u64_vec(p, len, cnt, &at));
        OC2_TRY(b, ll::get_u64_vec(p, len, cnt, &at));
        OC2_TRY(c, ll::get_u64_vec(p, len, cnt, &at));
        if (at != len) return fail("MATERIAL_WIRE: beaver trailing bytes");
        PoolItem<BeaverShare> it;
        it.material_id = mid;
        it.mat.count = cnt;
        it.mat.a = std::move(a); it.mat.b = std::move(b); it.mat.c = std::move(c);
        OC2_CHECK(ledger_.add_available(mid, type, payload.size(), version));
        (type == "beaver" ? beaver_ : kary_).push_back(std::move(it));
    } else if (type == "matrix_triple") {
        OC2_TRY(sh, matrix_triple_shape(*params));
        const size_t n = sh[0], k = sh[1], m = sh[2];
        std::string key = std::to_string(n) + "," + std::to_string(k) + "," + std::to_string(m);
        std::vector<PoolItem<MatrixTripleShare>> items;
        for (size_t i = 0; i < cnt; ++i) {
            PoolItem<MatrixTripleShare> it;
            it.material_id = mid;
            it.mat.n = n; it.mat.k = k; it.mat.m = m;
            OC2_TRY(A, ll::get_u64_vec(p, len, n * k, &at));
            OC2_TRY(B, ll::get_u64_vec(p, len, k * m, &at));
            OC2_TRY(C, ll::get_u64_vec(p, len, n * m, &at));
            it.mat.A = std::move(A); it.mat.B = std::move(B); it.mat.C = std::move(C);
            items.push_back(std::move(it));
        }
        if (at != len) return fail("MATERIAL_WIRE: matrix triple trailing bytes");
        OC2_CHECK(ledger_.add_available(mid, type, payload.size(), version));
        for (auto& it : items) matrix_[key].push_back(std::move(it));
    } else if (type == "wmask_mask") {
        OC2_TRY(mat_id, params->get_string("mat_id"));
        OC2_TRY(sh, wmask_mask_shape(*params));
        OC2_TRY(bsh, ll::get_u64_vec(p, len, sh[0] * sh[1], &at));
        if (at != len) return fail("MATERIAL_WIRE: wmask mask trailing bytes");
        if (wmask_masks_.count(mat_id)) return fail("MATERIAL_REUSE: wmask mask already present for " + mat_id);
        PoolItem<WMaskMask> it;
        it.material_id = mid;
        it.mat.mat_id = mat_id; it.mat.d = sh[0]; it.mat.m = sh[1];
        it.mat.b_share = std::move(bsh);
        OC2_CHECK(ledger_.add_available(mid, type, payload.size(), version));
        wmask_masks_[mat_id] = std::move(it);
    } else if (type == "wmask_call") {
        OC2_TRY(mat_id, params->get_string("mat_id")); OC2_TRY(n, params->get_int("n_rows"));
        // The isolated dealer service echoes the request params {mat_id, n_rows}; the shape
        // then comes from the installed mask of the same matrix (fail closed when absent).
        int64_t d = 0, m = 0;
        if (params->has("d") && params->has("m")) {
            OC2_TRY(dd, params->get_int("d")); OC2_TRY(mm, params->get_int("m"));
            d = dd; m = mm;
        } else {
            auto mk = wmask_masks_.find(mat_id);
            if (mk == wmask_masks_.end()) return fail("MATERIAL_PARAMS: wmask_call for " + mat_id + " without d, m or an installed mask");
            d = static_cast<int64_t>(mk->second.mat.d); m = static_cast<int64_t>(mk->second.mat.m);
        }
        if (d <= 0 || m <= 0 || n <= 0) return fail("MATERIAL_PARAMS: wmask_call shape");
        std::string key = mat_id + "|" + std::to_string(n);
        std::vector<PoolItem<WMaskCall>> items;
        for (size_t i = 0; i < cnt; ++i) {
            PoolItem<WMaskCall> it;
            it.material_id = mid;
            it.mat.mat_id = mat_id; it.mat.n = static_cast<size_t>(n); it.mat.d = static_cast<size_t>(d);
            it.mat.m = static_cast<size_t>(m);
            OC2_TRY(a, ll::get_u64_vec(p, len, it.mat.n * it.mat.d, &at));
            OC2_TRY(c, ll::get_u64_vec(p, len, it.mat.n * it.mat.m, &at));
            it.mat.a = std::move(a); it.mat.c = std::move(c);
            items.push_back(std::move(it));
        }
        if (at != len) return fail("MATERIAL_WIRE: wmask call trailing bytes");
        OC2_CHECK(ledger_.add_available(mid, type, payload.size(), version));
        for (auto& it : items) wmask_calls_[key].push_back(std::move(it));
    } else if (type == "drelu") {
        OC2_TRY(mat, ll::DReLUMaterial::from_bytes(payload));
        if (mat.party != party_) return fail("BUNDLE_RECIPIENT: drelu material party mismatch");
        if (mat.M() != cnt) return fail("MATERIAL_PARAMS: drelu material disagrees with its header count");
        PoolItem<ll::DReLUMaterial> it;
        it.material_id = mid;
        std::string key = type_key_drelu(mat.n_bits, mat.with_product());
        it.mat = std::move(mat);
        OC2_CHECK(ledger_.add_available(mid, type, payload.size(), version));
        drelu_[key].push_back(std::move(it));
    } else if (type == "trunc_faithful") {
        OC2_TRY(mat, ll::TruncMaterial::from_bytes(payload));
        if (mat.party != party_) return fail("BUNDLE_RECIPIENT: trunc material party mismatch");
        if (mat.M() != cnt) return fail("MATERIAL_PARAMS: trunc material disagrees with its header count");
        PoolItem<ll::TruncMaterial> it;
        it.material_id = mid;
        const std::pair<int, int> fl(mat.f, mat.L);
        it.mat = std::move(mat);
        OC2_CHECK(ledger_.add_available(mid, type, payload.size(), version));
        trunc_[fl].push_back(std::move(it));
    } else if (type == "ring_to_field") {
        OC2_TRY(mat, ll::R2FMaterial::from_bytes(payload));
        if (mat.party != party_) return fail("BUNDLE_RECIPIENT: r2f material party mismatch");
        if (mat.M() != cnt) return fail("MATERIAL_PARAMS: r2f material disagrees with its header count");
        PoolItem<ll::R2FMaterial> it;
        it.material_id = mid;
        uint64_t t = mat.t;
        it.mat = std::move(mat);
        OC2_CHECK(ledger_.add_available(mid, type, payload.size(), version));
        r2f_[t].push_back(std::move(it));
    } else if (type == "vector_share") {
        OC2_TRY(name, params->get_string("name"));
        size_t size = 0;
        if (params->has("shape")) {
            OC2_TRY(ne, numel_or(*params, "shape", 0));
            size = ne;
        } else {
            OC2_TRY(sz, params->get_int("size"));
            if (sz <= 0) return fail("MATERIAL_PARAMS: vector share size");
            size = static_cast<size_t>(sz);
        }
        OC2_TRY(v, ll::get_u64_vec(p, len, size, &at));
        if (at != len) return fail("MATERIAL_WIRE: vector share trailing bytes");
        PoolItem<std::vector<uint64_t>> it;
        it.material_id = mid;
        it.mat = std::move(v);
        OC2_CHECK(ledger_.add_available(mid, type, payload.size(), version));
        vectors_[name] = std::move(it);
    } else {
        return fail("BUNDLE_TYPE: unknown material type " + type);
    }
    if (insecure) insecure_seen_ = true;
    else eligible_seen_ = true;
    bytes_ingested_ += payload.size();
    bundles_ingested_ += 1;
    return ok_unit();
}

// Ledger transition of the bundle a slice comes from. The bundle is journaled RESERVED then
// SPENT when first touched (durable before the material is returned); later slices of the same
// bundle (a multi-object bundle or a partially consumed batch) find it SPENT already. A BURNED
// bundle is never served.
Expected<Unit> DealerClient::ensure_spent(const std::string& material_id, bool& spent_flag) {
    if (spent_flag) return ok_unit();
    OC2_TRY(st, ledger_.state(material_id));
    if (st == "BURNED") return fail("MATERIAL_REUSE: " + material_id + " is BURNED");
    if (st == "AVAILABLE") OC2_CHECK(ledger_.reserve({material_id}, gate_id_, epoch_));
    if (st != "SPENT") OC2_CHECK(ledger_.spend({material_id}));
    spent_flag = true;
    return ok_unit();
}

Expected<Unit> DealerClient::pre_lookup() {
    if (drain_) return drain_();
    return ok_unit();
}

Expected<Unit> DealerClient::refill_or_stall(const std::string& type, Value params, size_t count) {
    if (!refill_) return fail("MATERIAL_STALL: no " + type + " material and no refill link");
    refills_ += 1;
    if (generation_ > 0) params.set("generation", generation_);
    OC2_CHECK(refill_(type, params, count));
    return pre_lookup();
}

Expected<BeaverShare> DealerClient::take_beaver_like(std::deque<PoolItem<BeaverShare>>& pool, const std::string& type,
                                                     size_t count) {
    if (count == 0) return ok(BeaverShare{});
    OC2_CHECK(pre_lookup());
    for (int attempt = 0; attempt < 2; ++attempt) {
        size_t avail = 0;
        for (const auto& it : pool) avail += it.mat.count - it.offset;
        if (avail < count) {
            if (attempt == 1) return fail("MATERIAL_STALL: " + type);
            // Python request spelling: one bundle of shape [need]
            OC2_CHECK(refill_or_stall(type, dict({{"shape", shape_value({count - avail})}}), 1));
            continue;
        }
        BeaverShare out;
        out.count = count;
        out.a.reserve(count); out.b.reserve(count); out.c.reserve(count);
        size_t need = count;
        while (need > 0) {
            auto& it = pool.front();
            OC2_CHECK(ensure_spent(it.material_id, it.spent));
            size_t take = std::min(need, it.mat.count - it.offset);
            auto lo = static_cast<std::ptrdiff_t>(it.offset), hi = static_cast<std::ptrdiff_t>(it.offset + take);
            out.a.insert(out.a.end(), it.mat.a.begin() + lo, it.mat.a.begin() + hi);
            out.b.insert(out.b.end(), it.mat.b.begin() + lo, it.mat.b.begin() + hi);
            out.c.insert(out.c.end(), it.mat.c.begin() + lo, it.mat.c.begin() + hi);
            it.offset += take;
            need -= take;
            bytes_consumed_ += 24 * take;
            if (it.offset == it.mat.count) pool.pop_front();
        }
        return ok(std::move(out));
    }
    return fail("MATERIAL_STALL: " + type);
}

Expected<BeaverShare> DealerClient::take_beaver(size_t count) { return take_beaver_like(beaver_, "beaver", count); }

Expected<BeaverShare> DealerClient::take_kary_select(size_t count) { return take_beaver_like(kary_, "kary_select", count); }

Expected<MatrixTripleShare> DealerClient::take_matrix_triple(size_t n, size_t k, size_t m) {
    OC2_CHECK(pre_lookup());
    std::string key = std::to_string(n) + "," + std::to_string(k) + "," + std::to_string(m);
    for (int attempt = 0; attempt < 2; ++attempt) {
        auto it = matrix_.find(key);
        if (it == matrix_.end() || it->second.empty()) {
            if (attempt == 1) return fail("MATERIAL_STALL: matrix_triple " + key);
            OC2_CHECK(refill_or_stall("matrix_triple", dict({{"shape_a", shape_value({n, k})}, {"shape_b", shape_value({k, m})}}), 1));
            continue;
        }
        auto& item = it->second.front();
        OC2_CHECK(ensure_spent(item.material_id, item.spent));
        MatrixTripleShare out = std::move(item.mat);
        bytes_consumed_ += 8 * (out.A.size() + out.B.size() + out.C.size());
        it->second.pop_front();
        return ok(std::move(out));
    }
    return fail("MATERIAL_STALL: matrix_triple");
}

Expected<WMaskCall> DealerClient::take_wmask_call(const std::string& mat_id, size_t n_rows) {
    OC2_CHECK(pre_lookup());
    std::string key = mat_id + "|" + std::to_string(n_rows);
    for (int attempt = 0; attempt < 2; ++attempt) {
        auto it = wmask_calls_.find(key);
        if (it == wmask_calls_.end() || it->second.empty()) {
            if (attempt == 1) return fail("MATERIAL_STALL: wmask_call " + key);
            OC2_CHECK(refill_or_stall("wmask_call", dict({{"mat_id", mat_id}, {"n_rows", static_cast<int64_t>(n_rows)}}), 1));
            continue;
        }
        auto& item = it->second.front();
        OC2_CHECK(ensure_spent(item.material_id, item.spent));
        WMaskCall out = std::move(item.mat);
        bytes_consumed_ += 8 * (out.a.size() + out.c.size());
        it->second.pop_front();
        return ok(std::move(out));
    }
    return fail("MATERIAL_STALL: wmask_call");
}

// Generic sliced take over a deque of partially consumed batches. A refill asks for one batch
// of M = shortfall gates (Python request spelling: params.M with count = 1).
#define OC2_TAKE_SLICED(POOL, COUNT, TYPE, PARAMS, CONCAT, STALL)                                   \
    OC2_CHECK(pre_lookup());                                                                         \
    for (int attempt = 0; attempt < 2; ++attempt) {                                                  \
        auto& dq = (POOL);                                                                           \
        size_t avail = 0;                                                                            \
        for (const auto& it : dq) avail += it.mat.M() - it.offset;                                  \
        if (avail < (COUNT)) {                                                                       \
            if (attempt == 1) return fail(STALL);                                                    \
            Value req_params = (PARAMS);                                                             \
            req_params.set("M", static_cast<int64_t>((COUNT) - avail));                              \
            OC2_CHECK(refill_or_stall(TYPE, req_params, 1));                                         \
            continue;                                                                                \
        }                                                                                            \
        bool first = true;                                                                           \
        size_t need = (COUNT);                                                                       \
        decltype(dq.front().mat) out;                                                                \
        while (need > 0) {                                                                           \
            auto& it = dq.front();                                                                   \
            OC2_CHECK(ensure_spent(it.material_id, it.spent));                                       \
            size_t take = std::min(need, it.mat.M() - it.offset);                                    \
            OC2_TRY(part, it.mat.slice(it.offset, it.offset + take));                                \
            bytes_consumed_ += part.nbytes();                                                        \
            if (first) {                                                                             \
                out = std::move(part);                                                               \
                first = false;                                                                       \
            } else {                                                                                 \
                OC2_TRY(cat, CONCAT(out, part));                                                     \
                out = std::move(cat);                                                                \
            }                                                                                        \
            it.offset += take;                                                                       \
            need -= take;                                                                            \
            if (it.offset == it.mat.M()) dq.pop_front();                                             \
        }                                                                                            \
        return ok(std::move(out));                                                                   \
    }                                                                                                \
    return fail(STALL);

Expected<ll::DReLUMaterial> DealerClient::take_drelu(size_t count, int n_bits, bool with_product) {
    if (count == 0) return fail("MATERIAL_PARAMS: take_drelu count = 0");
    std::string key = type_key_drelu(n_bits, with_product);
    Value params = dict({{"n_bits", static_cast<int64_t>(n_bits)}, {"with_product", with_product}});
    OC2_TAKE_SLICED(drelu_[key], count, "drelu", params, lowlevel::drelu::concat_drelu, "MATERIAL_STALL: drelu " + key)
}

Expected<ll::TruncMaterial> DealerClient::take_trunc(size_t count, int f, int L) {
    if (count == 0) return fail("MATERIAL_PARAMS: take_trunc count = 0");
    if (!(1 <= f && f <= L - 1 && L - 1 <= 63)) return fail("MATERIAL_PARAMS: take_trunc needs 1 <= f <= L-1 <= 63");
    // the pool key and the refill request both carry (f, L): material of another width is never
    // handed out, so nothing is spent before the (f, L) of the gate is known to match
    Value params = dict({{"f", static_cast<int64_t>(f)}, {"L", static_cast<int64_t>(L)}});
    const std::pair<int, int> fl(f, L);
    OC2_TAKE_SLICED(trunc_[fl], count, "trunc_faithful", params, lowlevel::trunc::concat_trunc,
                    "MATERIAL_STALL: trunc_faithful f=" + std::to_string(f) + " L=" + std::to_string(L))
}

Expected<ll::R2FMaterial> DealerClient::take_r2f(size_t count, uint64_t t) {
    if (count == 0) return fail("MATERIAL_PARAMS: take_r2f count = 0");
    Value params = dict({{"t", static_cast<int64_t>(t)}});
    OC2_TAKE_SLICED(r2f_[t], count, "ring_to_field", params, lowlevel::r2f::concat_r2f, "MATERIAL_STALL: ring_to_field")
}

#undef OC2_TAKE_SLICED

Expected<std::vector<uint64_t>> DealerClient::take_vector_share(const std::string& name) {
    OC2_CHECK(pre_lookup());
    auto it = vectors_.find(name);
    if (it == vectors_.end()) return fail("MATERIAL_STALL: vector_share " + name);
    const bool first_use = !it->second.spent;
    OC2_CHECK(ensure_spent(it->second.material_id, it->second.spent));
    if (first_use) bytes_consumed_ += 8 * it->second.mat.size();
    return ok(it->second.mat);
}

Expected<WMaskMask> DealerClient::wmask_mask_share(const std::string& mat_id, size_t d, size_t m) {
    OC2_CHECK(pre_lookup());
    for (int attempt = 0; attempt < 2; ++attempt) {
        auto it = wmask_masks_.find(mat_id);
        if (it == wmask_masks_.end()) {
            if (attempt == 1 || d == 0 || m == 0) return fail("MATERIAL_STALL: wmask_mask " + mat_id);
            OC2_CHECK(refill_or_stall("wmask_mask", dict({{"mat_id", mat_id}, {"shape", shape_value({d, m})}}), 1));
            continue;
        }
        if ((d && it->second.mat.d != d) || (m && it->second.mat.m != m)) return fail("MATERIAL_PARAMS: wmask mask shape for " + mat_id);
        const bool first_use = !it->second.spent;
        OC2_CHECK(ensure_spent(it->second.material_id, it->second.spent));
        // the 8 d m bytes of the mask are consumed by the first use (the install), as in the
        // Python client and in material_formula("install_wmask")
        if (first_use) bytes_consumed_ += 8 * it->second.mat.b_share.size();
        return ok(it->second.mat);
    }
    return fail("MATERIAL_STALL: wmask_mask " + mat_id);
}

size_t DealerClient::pooled_count(const std::string& type_key) const {
    size_t n = 0;
    if (type_key == "beaver") {
        for (const auto& it : beaver_) n += it.mat.count - it.offset;
    } else if (type_key == "kary_select") {
        for (const auto& it : kary_) n += it.mat.count - it.offset;
    } else if (type_key.rfind("drelu:", 0) == 0) {
        auto it = drelu_.find(type_key.substr(6));
        if (it != drelu_.end())
            for (const auto& x : it->second) n += x.mat.M() - x.offset;
    } else if (type_key.rfind("trunc:", 0) == 0) {
        // "trunc:<f>" counts every width, "trunc:<f>|<L>" one pool
        const std::string spec = type_key.substr(6);
        const size_t bar = spec.find('|');
        const int f = std::atoi(spec.c_str());
        const int L = bar == std::string::npos ? -1 : std::atoi(spec.c_str() + bar + 1);
        for (const auto& kv : trunc_) {
            if (kv.first.first != f || (L >= 0 && kv.first.second != L)) continue;
            for (const auto& x : kv.second) n += x.mat.M() - x.offset;
        }
    } else if (type_key.rfind("r2f:", 0) == 0) {
        auto it = r2f_.find(std::strtoull(type_key.c_str() + 4, nullptr, 10));
        if (it != r2f_.end())
            for (const auto& x : it->second) n += x.mat.M() - x.offset;
    }
    return n;
}

// ---------------------------------------------------------------------------
// cancellation / resumption
// ---------------------------------------------------------------------------
Value DealerClient::BurnReport::to_value() const {
    return dict({{"reserved_ids", static_cast<int64_t>(reserved_ids)},
                 {"available_ids", static_cast<int64_t>(available_ids)},
                 {"bytes_burned", static_cast<int64_t>(bytes_burned)},
                 {"bytes_dropped", static_cast<int64_t>(bytes_dropped)}});
}

std::vector<std::string> DealerClient::reserved_ids() const {
    std::set<std::string> ids;
    auto note = [&](const std::string& id) {
        auto st = ledger_.state(id);
        if (st.ok && st.value == "RESERVED") ids.insert(id);
    };
    for (const auto& it : beaver_) note(it.material_id);
    for (const auto& it : kary_) note(it.material_id);
    for (const auto& kv : matrix_)
        for (const auto& it : kv.second) note(it.material_id);
    for (const auto& kv : wmask_calls_)
        for (const auto& it : kv.second) note(it.material_id);
    for (const auto& kv : drelu_)
        for (const auto& it : kv.second) note(it.material_id);
    for (const auto& kv : trunc_)
        for (const auto& it : kv.second) note(it.material_id);
    for (const auto& kv : r2f_)
        for (const auto& it : kv.second) note(it.material_id);
    for (const auto& kv : wmask_masks_) note(kv.second.material_id);
    for (const auto& kv : vectors_) note(kv.second.material_id);
    return std::vector<std::string>(ids.begin(), ids.end());
}

Expected<DealerClient::BurnReport> DealerClient::burn_outstanding(const std::string& reason, bool drop_masks) {
    OC2_CHECK(pre_lookup());  // bundles in flight are accounted (and burned) too
    BurnReport rep;
    std::vector<std::string> burn_ids;
    std::set<std::string> seen;
    const uint64_t burned_before = ledger_.bytes_burned();
    // one pooled item: its bundle is burned unless it is SPENT (then only the in-memory rest is dropped)
    auto visit = [&](const std::string& id, uint64_t remainder_bytes) -> Expected<Unit> {
        OC2_TRY(st, ledger_.state(id));
        if (st == "SPENT") {
            rep.bytes_dropped += remainder_bytes;
            return ok_unit();
        }
        if (!seen.insert(id).second) return ok_unit();
        if (st == "RESERVED") rep.reserved_ids += 1;
        else if (st == "AVAILABLE") rep.available_ids += 1;
        else return ok_unit();  // BURNED already
        burn_ids.push_back(id);
        return ok_unit();
    };
    for (const auto& it : beaver_) OC2_CHECK(visit(it.material_id, 24 * (it.mat.count - it.offset)));
    for (const auto& it : kary_) OC2_CHECK(visit(it.material_id, 24 * (it.mat.count - it.offset)));
    for (const auto& kv : matrix_)
        for (const auto& it : kv.second) OC2_CHECK(visit(it.material_id, 8 * (it.mat.A.size() + it.mat.B.size() + it.mat.C.size())));
    for (const auto& kv : wmask_calls_)
        for (const auto& it : kv.second) OC2_CHECK(visit(it.material_id, 8 * (it.mat.a.size() + it.mat.c.size())));
    for (const auto& kv : drelu_)
        for (const auto& it : kv.second)
            OC2_CHECK(visit(it.material_id, (it.mat.M() - it.offset) * ll::drelu_bytes_per_element(it.mat.n_bits, it.mat.with_product())));
    for (const auto& kv : trunc_)
        for (const auto& it : kv.second) OC2_CHECK(visit(it.material_id, (it.mat.M() - it.offset) * ll::trunc_bytes_per_element(it.mat.f)));
    for (const auto& kv : r2f_)
        for (const auto& it : kv.second) OC2_CHECK(visit(it.material_id, (it.mat.M() - it.offset) * ll::r2f_bytes_per_element(it.mat.t)));
    if (drop_masks)
        for (const auto& kv : wmask_masks_) OC2_CHECK(visit(kv.second.material_id, 0));
    // persistent objects that stay: a RESERVED entry must not survive either
    if (!drop_masks)
        for (const auto& kv : wmask_masks_) {
            auto st = ledger_.state(kv.second.material_id);
            if (st.ok && st.value == "RESERVED") OC2_CHECK(visit(kv.second.material_id, 0));
        }
    for (const auto& kv : vectors_) {
        auto st = ledger_.state(kv.second.material_id);
        if (st.ok && st.value == "RESERVED") OC2_CHECK(visit(kv.second.material_id, 0));
    }
    if (!burn_ids.empty()) OC2_CHECK(ledger_.burn(burn_ids, reason));
    rep.bytes_burned = ledger_.bytes_burned() - burned_before;
    // the pools are emptied only after the journal accepted the burn
    beaver_.clear();
    kary_.clear();
    matrix_.clear();
    wmask_calls_.clear();
    drelu_.clear();
    trunc_.clear();
    r2f_.clear();
    if (drop_masks) {
        wmask_masks_.clear();
    } else {
        for (auto it = wmask_masks_.begin(); it != wmask_masks_.end();) {
            auto st = ledger_.state(it->second.material_id);
            if (st.ok && st.value == "BURNED") it = wmask_masks_.erase(it);
            else ++it;
        }
    }
    for (auto it = vectors_.begin(); it != vectors_.end();) {
        auto st = ledger_.state(it->second.material_id);
        if (st.ok && st.value == "BURNED") it = vectors_.erase(it);
        else ++it;
    }
    return ok(rep);
}

// ---------------------------------------------------------------------------
// InProcessDealerLink
// ---------------------------------------------------------------------------
InProcessDealerLink::InProcessDealerLink(std::shared_ptr<Dealer> dealer) : dealer_(std::move(dealer)) {}

void InProcessDealerLink::attach(DealerClient& client) {
    int party = client.party();
    DealerClient* cp = &client;
    client.set_refill([this, party](const std::string& type, const Value& params, size_t count) {
        return this->request(party, type, params, count);
    });
    client.set_drain([this, party, cp]() { return this->drain(party, *cp); });
}

Expected<Unit> InProcessDealerLink::request(int party, const std::string& type, const Value& params, size_t count) {
    std::lock_guard<std::mutex> g(mu_);
    // A request arrives only after the requesting party drained its inbox and
    // still found its pool short, so the material is genuinely missing on both
    // sides (both consume identical sequences). Every pair is appended to both
    // inboxes in one global order; a concurrent request by the peer at most
    // generates one extra pair, which both parties pool identically.
    requests_[party] += 1;
    if (type == "wmask_mask" || type == "vector_share") {
        // persistent objects exist once per id (and generation): the peer's request already covers this one
        OC2_TRY(cj, canonical_json(params));
        std::string key = type + "|" + cj;
        if (persistent_.count(key)) return ok_unit();
        persistent_.insert(key);
    }
    double t0 = now_s();
    OC2_TRY(bp, dealer_->make(type, params, count));
    gen_seconds_ += now_s() - t0;
    generations_ += 1;
    transferred_[0] += bp.for0.payload.size();
    transferred_[1] += bp.for1.payload.size();
    inbox_[0].push_back(std::move(bp.for0));
    inbox_[1].push_back(std::move(bp.for1));
    return ok_unit();
}

Expected<Unit> InProcessDealerLink::drain(int party, DealerClient& client) {
    std::deque<Bundle> mine;
    {
        std::lock_guard<std::mutex> g(mu_);
        mine.swap(inbox_[party]);
    }
    for (auto& b : mine) OC2_CHECK(client.ingest(b));
    return ok_unit();
}

uint64_t InProcessDealerLink::bytes_generated() const { return dealer_->bytes_generated(); }

}  // namespace highlevel
}  // namespace opencrab2
