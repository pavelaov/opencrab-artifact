// OpenCrab v2: cost model implementation (see include/opencrab2/highlevel/costmodel.hpp).
#include "opencrab2/highlevel/costmodel.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <mutex>
#include <numeric>

namespace opencrab2 {
namespace highlevel {

namespace {

const std::vector<std::string> kComponents = {
    "linear_wmask", "linear_generic", "embedding_wmask", "embedding_generic", "install_wmask", "gelu_clip12",
    "softmax_rows_k2", "softmax_rows_k4", "layernorm_rows", "argmax_k2", "argmax_k4", "argmax_k8",
    "trunc_faithful", "ring_to_field", "a2h_exact", "matmul_shared"};

std::int64_t ceil_div(std::int64_t a, std::int64_t b) { return (a + b - 1) / b; }

int argmax_k(const std::string& c) {
    if (c == "argmax_k2") return 2;
    if (c == "argmax_k4") return 4;
    if (c == "argmax_k8") return 8;
    return 0;
}

int softmax_k(const std::string& c) {
    if (c == "softmax_rows_k2") return 2;
    if (c == "softmax_rows_k4") return 4;
    return 0;
}

bool is_linear(const std::string& c) {
    return c == "linear_wmask" || c == "linear_generic" || c == "embedding_wmask" || c == "embedding_generic" || c == "matmul_shared";
}

// Derived gadget counts of the contract (the twin of python costmodel.ProtocolParams). A threshold
// comparison is live when the threshold lies below the DCF bound 2^(n-1-f) real units
// (Party::threshold_bits answers the others with a public 0 and issues no DReLU for them).
int compare_bound_log2(const NumericContract& c) { return c.dcf_domain_bits - 1 - c.frac_bits; }
std::int64_t recip_live(const NumericContract& c) {
    std::int64_t live = 0;
    for (int j = 1; j <= c.recip_K; ++j)
        if (j < compare_bound_log2(c)) ++live;                    // thresholds 2^j
    return live;
}
std::int64_t rsqrt_live(const NumericContract& c) {
    std::int64_t live = 0;
    for (int i = 1; i <= c.rsqrt_m; ++i)
        if (2 * (c.rsqrt_j0 + i) < compare_bound_log2(c)) ++live;   // thresholds 4^(j0+i)
    return live;
}
std::int64_t exp_beaver(const NumericContract& c) { return 1 + (c.exp_order >= 2 ? 1 : 0) + c.exp_r; }   // clamp + base square + r squarings
std::int64_t exp_rounds(const NumericContract& c) { return 1 + exp_beaver(c); }
std::int64_t recip_beaver(const NumericContract& c) { return 1 + c.recip_degree + 2 * c.recip_newton + 1; }
std::int64_t recip_rounds(const NumericContract& c) { return (recip_live(c) ? 1 : 0) + recip_beaver(c); }
std::int64_t rsqrt_beaver(const NumericContract& c) { return 1 + c.rsqrt_degree + 3 * c.rsqrt_newton + 1; }
std::int64_t rsqrt_rounds(const NumericContract& c) { return (rsqrt_live(c) ? 1 : 0) + rsqrt_beaver(c); }
bool faithful_square(const NumericContract& c) { return c.trunc_square_ln == "faithful"; }

// Process-wide shared constants: the three integers are atomics (read on every formula call),
// the two labels sit behind a mutex.
std::atomic<std::int64_t> g_frame_bytes{kFrameBytes};
std::atomic<std::int64_t> g_bfv_slots{kBfvSlots};
std::atomic<std::int64_t> g_bfv_ct_bytes{kBfvCiphertextBytes};
std::mutex& labels_mutex() {
    static std::mutex m;
    return m;
}
struct Labels {
    std::string source = "builtin";
    std::string a2h_status = "MODELED";
};
Labels& labels() {
    static Labels l;
    return l;
}

}  // namespace

Json Env::to_json() const {
    Json j = Json::object();
    j.set("bandwidth_bps", bandwidth_bps);
    j.set("one_way_latency_s", one_way_latency_s);
    j.set("cpu_share", cpu_share);
    return j;
}

Expected<Env> Env::from_json(const Json& j) {
    if (!j.is_object()) return Expected<Env>::failure("ENV_JSON: expected an object");
    Env e;
    Expected<double> bw = j.get_double("bandwidth_bps");
    if (!bw.ok) return Expected<Env>::failure("ENV_JSON: " + bw.code);
    Expected<double> lat = j.get_double("one_way_latency_s");
    if (!lat.ok) return Expected<Env>::failure("ENV_JSON: " + lat.code);
    e.bandwidth_bps = bw.value;
    e.one_way_latency_s = lat.value;
    if (j.has("cpu_share")) {
        Expected<double> cs = j.get_double("cpu_share");
        if (!cs.ok) return Expected<Env>::failure("ENV_JSON: " + cs.code);
        e.cpu_share = cs.value;
    }
    if (e.cpu_share <= 0.0 || e.bandwidth_bps < 0.0 || e.one_way_latency_s < 0.0)
        return Expected<Env>::failure("ENV_PARAMS: cpu_share must be > 0 and bandwidth, latency >= 0");
    return Expected<Env>::success(e);
}

// ---------------------------------------------------------------------------
// shared constants
// ---------------------------------------------------------------------------
Json FormulaConstants::to_json() const {
    Json j = Json::object();
    j.set("frame_bytes", static_cast<long long>(frame_bytes));
    j.set("open_bytes", static_cast<long long>(kOpenBytes));
    j.set("beaver_open_bytes", static_cast<long long>(kBeaverOpenBytes));
    j.set("oneway_bytes", static_cast<long long>(kOnewayBytes));
    j.set("bfv_slots", static_cast<long long>(bfv_slots));
    j.set("bfv_ciphertext_bytes", static_cast<long long>(bfv_ciphertext_bytes));
    j.set("source", source);
    j.set("a2h_status", a2h_status);
    return j;
}

FormulaConstants formula_constants() {
    FormulaConstants c;
    c.frame_bytes = g_frame_bytes.load();
    c.bfv_slots = g_bfv_slots.load();
    c.bfv_ciphertext_bytes = g_bfv_ct_bytes.load();
    std::lock_guard<std::mutex> g(labels_mutex());
    c.source = labels().source;
    c.a2h_status = labels().a2h_status;
    return c;
}

Status set_formula_constants(const FormulaConstants& c) {
    if (c.frame_bytes < 0 || c.bfv_slots < 1 || c.bfv_ciphertext_bytes < 1)
        return fail_status("FORMULAS_PARAMS: frame_bytes >= 0, bfv_slots >= 1 and bfv_ciphertext_bytes >= 1 required");
    g_frame_bytes.store(c.frame_bytes);
    g_bfv_slots.store(c.bfv_slots);
    g_bfv_ct_bytes.store(c.bfv_ciphertext_bytes);
    std::lock_guard<std::mutex> g(labels_mutex());
    labels().source = c.source;
    labels().a2h_status = c.a2h_status;
    return ok_status();
}

void reset_formula_constants() { (void)set_formula_constants(FormulaConstants()); }

// ---------------------------------------------------------------------------
// formulas
// ---------------------------------------------------------------------------
const std::vector<std::string>& components() { return kComponents; }

bool is_component(const std::string& name) {
    return std::find(kComponents.begin(), kComponents.end(), name) != kComponents.end();
}

TournamentCounts tournament_counts(std::int64_t L, int k, bool with_index) {
    TournamentCounts t;
    if (L <= 1 || (k != 2 && k != 4 && k != 8)) return t;
    std::int64_t w = L;
    bool idx_public = true;
    while (w > 1) {
        const std::int64_t extra = (with_index && !idx_public) ? 1 : 0;
        const std::int64_t full = w / k;
        const std::int64_t rem = w % k;
        auto node = [&](std::int64_t g, std::int64_t count) {
            if (g < 2 || count < 1) return;
            t.comparisons += count * (g * (g - 1) / 2);
            if (g == 2) t.selects += count * (1 + extra);
            else t.payload_drelu += count * (g - 1) * (1 + extra);
        };
        node(k, full);
        node(rem, rem ? 1 : 0);
        w = ceil_div(w, k);
        idx_public = false;
        ++t.levels;
    }
    t.rounds = 2 * t.levels;
    return t;
}

int kary_ceil_log2(std::int64_t g) {
    int c = 0;
    while (c < 62 && (static_cast<std::int64_t>(1) << c) < g) ++c;
    return c;
}

std::int64_t kary_difference_bound_ring(int dcf_domain_bits, std::int64_t g) {
    const int e = g <= 2 ? dcf_domain_bits - 1 : dcf_domain_bits - 2 - kary_ceil_log2(g);
    if (e < 0) return 0;
    if (e >= 63) return std::numeric_limits<std::int64_t>::max();
    return static_cast<std::int64_t>(1) << e;
}

double kary_operand_bound_real(const NumericContract& contract, int k, std::int64_t L) {
    const std::int64_t g_max = std::min<std::int64_t>(k, std::max<std::int64_t>(L, 1));
    const std::int64_t bound = kary_difference_bound_ring(contract.dcf_domain_bits, g_max);
    return std::ldexp(static_cast<double>(bound), -(contract.frac_bits + 1));
}

Status kary_magnitude_check(const NumericContract& contract, int k, std::int64_t L, double declared_bound_real, bool with_index) {
    if (k != 2 && k != 4 && k != 8) return fail_status("KARY_PARAMS: k must be 2, 4 or 8");
    if (L < 1) return fail_status("KARY_PARAMS: row length must be >= 1");
    if (L == 1) return ok_status();   // nothing is compared
    if (!std::isfinite(declared_bound_real) || declared_bound_real <= 0.0)
        return fail_status("KARY_MAGNITUDE: no declared public operand bound for a k-ary site (k = " + std::to_string(k) + ")");
    const std::int64_t g_max = std::min<std::int64_t>(k, L);
    const std::int64_t bound = kary_difference_bound_ring(contract.dcf_domain_bits, g_max);
    const double admitted = kary_operand_bound_real(contract, k, L);
    // |x| <= B for every operand gives |x_i - x_j| <= 2 B; the node needs the difference strictly
    // below its bound, so B must stay strictly below bound / 2 (in real units: admitted).
    if (bound <= 0 || !(declared_bound_real < admitted))
        return fail_status("KARY_MAGNITUDE: declared operand bound " + format_double_repr(declared_bound_real) +
                           " is not below the admitted " + format_double_repr(admitted) + " real units for k = " + std::to_string(k) +
                           " (nodes of " + std::to_string(g_max) + ", n = " + std::to_string(contract.dcf_domain_bits) +
                           ", f = " + std::to_string(contract.frac_bits) + ")");
    if (with_index && g_max >= 3 && L - 1 >= bound)
        return fail_status("KARY_MAGNITUDE: row length " + std::to_string(L) + " exceeds the index bound of a node of " +
                           std::to_string(g_max));
    return ok_status();
}

std::int64_t dcf_key_bytes(int n_bits, int payloads) {
    return 16 + static_cast<std::int64_t>(n_bits) * (16 + 8 * payloads + 1) + 8 * payloads;
}

std::int64_t drelu_material_bytes(const NumericContract& c, int payloads) { return 2 * dcf_key_bytes(c.dcf_domain_bits, payloads) + 24; }
std::int64_t trunc_material_bytes(const NumericContract& c) { return 16 + dcf_key_bytes(64, 1) + dcf_key_bytes(c.frac_bits, 1); }
std::int64_t r2f_material_bytes() { return 24 + 2 * dcf_key_bytes(64, 1); }

Expected<GadgetCounts> gadget_counts(const std::string& comp, std::int64_t n, std::int64_t d_in, std::int64_t d_out,
                                     const NumericContract& c) {
    using R = Expected<GadgetCounts>;
    if (!is_component(comp)) return R::failure("UNKNOWN_COMPONENT: " + comp);
    if (n < 1 || d_in < 1) return R::failure("COSTMODEL_SHAPE: n and d_in must be >= 1");
    const bool uses_d_out = is_linear(comp) || comp == "install_wmask";
    if (uses_d_out && d_out < 1) return R::failure("COSTMODEL_SHAPE: d_out must be >= 1 for " + comp);
    GadgetCounts g;
    const std::int64_t N = n * d_in;
    if (comp == "linear_wmask" || comp == "embedding_wmask") {
        g.wmask_call = n * d_in + n * d_out;
        g.open_elems = n * d_in;                                     // open X - A
        g.rounds = 1;
        g.messages = 2;
    } else if (comp == "linear_generic" || comp == "embedding_generic" || comp == "matmul_shared") {
        g.matrix_triple = n * d_in + d_in * d_out + n * d_out;
        g.open_elems = n * d_in + d_in * d_out;                      // open X - A and W - B, both directions
        g.rounds = 1;
        g.messages = 2;
    } else if (comp == "install_wmask") {
        g.wmask_mask = d_in * d_out;
        g.oneway_elems = 2 * d_in * d_out;                           // P1 sends B1 up, P0 returns E down
        g.rounds = 1;
        g.messages = 2;
    } else if (comp == "gelu_clip12") {
        const std::int64_t deg = c.gelu_degree();
        g.drelu = 2 * N;                                             // clip at -R and +R
        g.beaver = (deg + 2) * N;                                    // Horner + two blend products
        g.rounds = 1 + deg + 1;
        g.messages = 2 * g.rounds;
    } else if (comp == "layernorm_rows") {
        const bool faithful = faithful_square(c);
        g.beaver = 3 * N + rsqrt_beaver(c) * n;                      // square, dx * rs, gamma; rsqrt per row
        g.drelu = rsqrt_live(c) * n;
        g.trunc_faithful = faithful ? N : 0;
        g.rounds = 1 + (faithful ? 1 : 0) + rsqrt_rounds(c) + 1 + 1;
        g.messages = 2 * g.rounds;
    } else if (softmax_k(comp)) {
        const TournamentCounts t = tournament_counts(d_in, softmax_k(comp), false);   // n rows of d_in keys
        g.drelu = n * t.comparisons + N + recip_live(c) * n;
        g.drelu_p2 = n * t.payload_drelu;
        g.kary_select = n * t.selects;
        g.beaver = exp_beaver(c) * N + recip_beaver(c) * n + N;
        g.rounds = t.rounds + exp_rounds(c) + recip_rounds(c) + 1;
        g.messages = 2 * g.rounds;
    } else if (argmax_k(comp)) {
        const TournamentCounts t = tournament_counts(d_in, argmax_k(comp), true);     // n rows of d_in logits
        g.drelu = n * t.comparisons;
        g.drelu_p2 = n * t.payload_drelu;
        g.kary_select = n * t.selects;
        g.rounds = t.rounds;
        g.messages = 2 * g.rounds;
    } else if (comp == "trunc_faithful") {
        g.trunc_faithful = N;
        g.rounds = 1;
        g.messages = 2;
    } else if (comp == "ring_to_field") {
        g.ring_to_field = N;
        g.rounds = 1;
        g.messages = 2;
    } else if (comp == "a2h_exact") {
        g.ring_to_field = N;
        g.bfv_ct = ceil_div(N, g_bfv_slots.load());
        g.rounds = 2;                        // ring_to_field opening, then ONE message P1 -> P0 with every ciphertext
        g.messages = 3;
    } else {
        return R::failure("UNKNOWN_COMPONENT: " + comp);
    }
    return R::success(g);
}

Expected<std::int64_t> macs(const std::string& comp, std::int64_t n, std::int64_t d_in, std::int64_t d_out) {
    using R = Expected<std::int64_t>;
    if (!is_component(comp)) return R::failure("UNKNOWN_COMPONENT: " + comp);
    if (comp == "install_wmask") {
        if (d_in < 1 || d_out < 1) return R::failure("COSTMODEL_SHAPE: d_in and d_out must be >= 1");
        return R::success(d_in * d_out);
    }
    if (n < 1 || d_in < 1) return R::failure("COSTMODEL_SHAPE: n and d_in must be >= 1");
    if (is_linear(comp)) {
        if (d_out < 1) return R::failure("COSTMODEL_SHAPE: d_out must be >= 1 for " + comp);
        return R::success(n * d_in * d_out);
    }
    return R::success(n * d_in);   // elementwise and row components: element count
}

Expected<std::int64_t> payload_bytes(const std::string& comp, std::int64_t n, std::int64_t d_in, std::int64_t d_out) {
    Expected<GadgetCounts> g = gadget_counts(comp, n, d_in, d_out, default_contract());
    if (!g.ok) return Expected<std::int64_t>::failure(g.code);
    const GadgetCounts& c = g.value;
    return Expected<std::int64_t>::success(kOpenBytes * (c.drelu + c.drelu_p2 + c.trunc_faithful + c.ring_to_field + c.open_elems) +
                                           kBeaverOpenBytes * (c.beaver + c.kary_select) + kOnewayBytes * c.oneway_elems +
                                           (c.bfv_ct ? g_bfv_ct_bytes.load() * c.bfv_ct : 0));
}

Expected<std::int64_t> messages_formula(const std::string& comp, std::int64_t n, std::int64_t d_in, std::int64_t d_out) {
    Expected<GadgetCounts> g = gadget_counts(comp, n, d_in, d_out, default_contract());
    if (!g.ok) return Expected<std::int64_t>::failure(g.code);
    return Expected<std::int64_t>::success(g.value.messages);
}

Expected<std::int64_t> rounds_formula(const std::string& comp, std::int64_t n, std::int64_t d_in, std::int64_t d_out) {
    Expected<GadgetCounts> g = gadget_counts(comp, n, d_in, d_out, default_contract());
    if (!g.ok) return Expected<std::int64_t>::failure(g.code);
    return Expected<std::int64_t>::success(g.value.rounds);
}

Expected<std::int64_t> bytes_formula(const std::string& comp, std::int64_t n, std::int64_t d_in, std::int64_t d_out) {
    Expected<std::int64_t> p = payload_bytes(comp, n, d_in, d_out);
    if (!p.ok) return p;
    Expected<std::int64_t> m = messages_formula(comp, n, d_in, d_out);
    if (!m.ok) return m;
    return Expected<std::int64_t>::success(p.value + g_frame_bytes.load() * m.value);
}

Expected<std::map<std::string, std::int64_t>> material_by_type(const std::string& comp, std::int64_t n, std::int64_t d_in,
                                                               std::int64_t d_out, const NumericContract& contract) {
    using R = Expected<std::map<std::string, std::int64_t>>;
    Expected<GadgetCounts> g = gadget_counts(comp, n, d_in, d_out, contract);
    if (!g.ok) return R::failure(g.code);
    const GadgetCounts& c = g.value;
    std::map<std::string, std::int64_t> m;
    if (c.beaver) m["beaver"] = 24 * c.beaver;
    if (c.kary_select) m["kary_select"] = 24 * c.kary_select;
    if (c.drelu || c.drelu_p2) m["drelu"] = drelu_material_bytes(contract, 1) * c.drelu + drelu_material_bytes(contract, 2) * c.drelu_p2;
    if (c.trunc_faithful) m["trunc_faithful"] = trunc_material_bytes(contract) * c.trunc_faithful;
    if (c.ring_to_field) m["ring_to_field"] = r2f_material_bytes() * c.ring_to_field;
    if (c.matrix_triple) m["matrix_triple"] = 8 * c.matrix_triple;
    // cost-model vocabulary (shared with Python): wmask_mask covers the per-call (A, C = A B) shares
    // of the weight-masked linear and the reusable mask B consumed by an install
    if (c.wmask_call || c.wmask_mask) m["wmask_mask"] = 8 * (c.wmask_call + c.wmask_mask);
    return R::success(std::move(m));
}

Expected<std::int64_t> material_formula(const std::string& comp, std::int64_t n, std::int64_t d_in, std::int64_t d_out,
                                        const NumericContract& contract) {
    Expected<std::map<std::string, std::int64_t>> m = material_by_type(comp, n, d_in, d_out, contract);
    if (!m.ok) return Expected<std::int64_t>::failure(m.code);
    std::int64_t total = 0;
    for (const auto& kv : m.value) total += kv.second;
    return Expected<std::int64_t>::success(total);
}

// ---------------------------------------------------------------------------
// formulas.json (shared constants and cross-language vectors)
// ---------------------------------------------------------------------------
namespace {

bool file_exists(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return static_cast<bool>(in);
}

std::string norm_key(const std::string& s) {
    std::string o;
    for (char ch : s)
        if (std::isalnum(static_cast<unsigned char>(ch))) o.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    return o;
}

bool key_in(const std::string& normalized, std::initializer_list<const char*> names) {
    for (const char* n : names)
        if (normalized == n) return true;
    return false;
}

// An integer constant: a JSON int, an integral double, or an object carrying one under
// value / bytes / measured_bytes / measured. Returns false when no integer can be read.
bool int_constant(const Json& v, std::int64_t& out) {
    if (v.is_int()) {
        out = v.as_int();
        return true;
    }
    if (v.is_double()) {
        const double d = v.as_double();
        if (std::isfinite(d) && d == std::floor(d) && std::fabs(d) < 9e15) {
            out = static_cast<std::int64_t>(d);
            return true;
        }
        return false;
    }
    if (v.is_object()) {
        for (const char* k : {"value", "bytes", "measured_bytes", "measured"}) {
            const Json* x = v.find(k);
            if (x && (x->is_int() || x->is_double()) && int_constant(*x, out)) return true;
        }
    }
    return false;
}

struct ConstantScan {
    bool have_frame = false, have_ct = false, have_slots = false;
    std::int64_t frame = 0, ct = 0, slots = 0;
    std::string a2h_status;
    std::vector<std::string> problems;   // FORMULAS_PARAMS / FORMULAS_MISMATCH details
};

void scan_constants(const Json& j, int depth, ConstantScan& s) {
    if (!j.is_object() || depth > 3) return;
    for (const auto& kv : j.as_object()) {
        const std::string k = norm_key(kv.first);
        std::int64_t v = 0;
        const bool is_frame = key_in(k, {"framebytes", "framingbytes", "framebytespermessage"});
        const bool is_ct = key_in(k, {"ctbytes", "bfvctbytes", "bfvciphertextbytes", "a2hexactciphertextbytes", "a2hciphertextbytes",
                                      "ciphertextbytes", "a2hctbytes", "a2hexactctbytes"});
        const bool is_slots = key_in(k, {"bfvn", "bfvslots", "slotsperciphertext"});
        const bool is_open = key_in(k, {"openbytes"});
        const bool is_beaver = key_in(k, {"beaveropenbytes", "selectbytes"});
        const bool is_oneway = key_in(k, {"onewaybytes"});
        if (is_oneway) {
            if (int_constant(kv.second, v) && v != kOnewayBytes)
                s.problems.push_back("FORMULAS_MISMATCH: " + kv.first + " = " + std::to_string(v) + ", this implementation sends " +
                                     std::to_string(kOnewayBytes) + " bytes per one-way ring element");
            continue;
        }
        if (is_frame || is_ct || is_slots || is_open || is_beaver) {
            if (!int_constant(kv.second, v)) {
                s.problems.push_back("FORMULAS_PARAMS: " + kv.first + " is not an integer");
                continue;
            }
            if (is_frame && !s.have_frame) {
                if (v < 0) s.problems.push_back("FORMULAS_PARAMS: " + kv.first + " must be >= 0");
                else { s.have_frame = true; s.frame = v; }
            } else if (is_ct && !s.have_ct) {
                if (v < 1) s.problems.push_back("FORMULAS_PARAMS: " + kv.first + " must be >= 1");
                else {
                    s.have_ct = true;
                    s.ct = v;
                    // the evidence label of the measurement, when the file states one next to the value
                    for (const Json* holder : {&kv.second, &j}) {
                        if (!holder->is_object()) continue;
                        for (const char* sk : {"status", "evidence_status", "a2h_status"}) {
                            const Json* st = holder->find(sk);
                            if (st && st->is_string() && s.a2h_status.empty()) s.a2h_status = st->as_string();
                        }
                    }
                }
            } else if (is_slots) {
                if (v < 1) s.problems.push_back("FORMULAS_PARAMS: " + kv.first + " must be >= 1");
                else if (!s.have_slots) { s.have_slots = true; s.slots = v; }
                else if (v != s.slots) s.problems.push_back("FORMULAS_PARAMS: the file states two different BFV slot counts");
            } else if (is_open && v != kOpenBytes) {
                s.problems.push_back("FORMULAS_MISMATCH: " + kv.first + " = " + std::to_string(v) + ", this implementation opens " +
                                     std::to_string(kOpenBytes) + " bytes per ring element");
            } else if (is_beaver && v != kBeaverOpenBytes) {
                s.problems.push_back("FORMULAS_MISMATCH: " + kv.first + " = " + std::to_string(v) + ", this implementation opens " +
                                     std::to_string(kBeaverOpenBytes) + " bytes per Beaver product");
            }
            continue;
        }
        if (kv.second.is_object()) scan_constants(kv.second, depth + 1, s);
    }
}

const Json* find_any(const Json& obj, std::initializer_list<const char*> names) {
    for (const char* n : names) {
        const Json* v = obj.find(n);
        if (v) return v;
    }
    return nullptr;
}

void note_mismatch(FormulasReport& rep, const std::string& what, const std::string& field, std::int64_t file_v, std::int64_t cpp_v) {
    rep.fields_compared += 1;
    if (file_v == cpp_v) return;
    if (rep.mismatches.size() < 200)
        rep.mismatches.push_back(what + " " + field + ": file " + std::to_string(file_v) + ", cpp " + std::to_string(cpp_v));
    else if (rep.mismatches.size() == 200)
        rep.mismatches.push_back("... further mismatches omitted");
}

void compare_component_vector(const Json& v, const NumericContract& contract, FormulasReport& rep) {
    const std::string comp = v.get("component").as_string();
    std::int64_t n = 0, d_in = 0, d_out = 1;
    if (!int_constant(v.get("n"), n) || !int_constant(v.get("d_in"), d_in)) {
        rep.notes.push_back("component vector without integer n / d_in skipped: " + comp);
        return;
    }
    if (v.has("d_out") && !v.get("d_out").is_null()) (void)int_constant(v.get("d_out"), d_out);
    const std::string what = comp + " (" + std::to_string(n) + ", " + std::to_string(d_in) + ", " + std::to_string(d_out) + ")";
    if (!is_component(comp)) {
        rep.fields_compared += 1;
        rep.mismatches.push_back(what + ": component unknown to the C++ model");
        return;
    }
    std::int64_t fv = 0;
    bool any = false;
    auto field = [&](std::initializer_list<const char*> names, const char* label, const Expected<std::int64_t>& mine) {
        const Json* f = find_any(v, names);
        if (!f || !int_constant(*f, fv)) return;
        any = true;
        if (!mine.ok) {
            rep.fields_compared += 1;
            rep.mismatches.push_back(what + " " + label + ": file " + std::to_string(fv) + ", cpp fails with " + mine.code);
            return;
        }
        note_mismatch(rep, what, label, fv, mine.value);
    };
    field({"macs"}, "macs", macs(comp, n, d_in, d_out));
    field({"bytes", "bytes_formula", "total_bytes"}, "bytes", bytes_formula(comp, n, d_in, d_out));
    field({"payload_bytes", "payload"}, "payload_bytes", payload_bytes(comp, n, d_in, d_out));
    field({"rounds", "rounds_formula"}, "rounds", rounds_formula(comp, n, d_in, d_out));
    field({"messages", "messages_formula"}, "messages", messages_formula(comp, n, d_in, d_out));
    field({"material_bytes", "material_formula"}, "material_bytes", material_formula(comp, n, d_in, d_out, contract));
    // gate counts of the python document (GateCounts.to_dict)
    const Json* gates = v.find("gates");
    if (gates && gates->is_object()) {
        Expected<GadgetCounts> mine = gadget_counts(comp, n, d_in, d_out, contract);
        if (mine.ok) {
            const GadgetCounts& gc = mine.value;
            const std::pair<const char*, std::int64_t> known[] = {
                {"drelu_p1", gc.drelu},           {"drelu_p2", gc.drelu_p2},       {"beaver", gc.beaver},
                {"kary_select", gc.kary_select},  {"trunc", gc.trunc_faithful},    {"r2f", gc.ring_to_field},
                {"matrix_triple", gc.matrix_triple}, {"wmask_call", gc.wmask_call}, {"wmask_mask", gc.wmask_mask},
                {"open_elems", gc.open_elems},    {"oneway_elems", gc.oneway_elems}, {"bfv_ct", gc.bfv_ct},
                {"rounds", gc.rounds},            {"messages", gc.messages}};
            for (const auto& kv : gates->as_object()) {
                if (!int_constant(kv.second, fv)) continue;
                bool matched = false;
                for (const auto& kn : known) {
                    if (kv.first != kn.first) continue;
                    matched = true;
                    any = true;
                    note_mismatch(rep, what, "gates[" + kv.first + "]", fv, kn.second);
                }
                if (!matched && fv != 0) {
                    // a gate kind this implementation does not know and the file counts as non-zero
                    rep.fields_compared += 1;
                    rep.mismatches.push_back(what + " gates[" + kv.first + "]: file " + std::to_string(fv) + ", unknown to the C++ model");
                }
            }
        }
    }
    const Json* mat = v.find("material");
    if (!mat) mat = v.find("material_by_type");
    if (mat && mat->is_object()) {
        Expected<std::map<std::string, std::int64_t>> mine = material_by_type(comp, n, d_in, d_out, contract);
        if (mine.ok) {
            any = true;
            for (const auto& kv : mat->as_object()) {
                if (!int_constant(kv.second, fv)) continue;
                auto it = mine.value.find(kv.first);
                note_mismatch(rep, what, "material[" + kv.first + "]", fv, it == mine.value.end() ? 0 : it->second);
            }
        }
    } else if (mat && int_constant(*mat, fv)) {
        any = true;
        Expected<std::int64_t> mine = material_formula(comp, n, d_in, d_out, contract);
        if (mine.ok) note_mismatch(rep, what, "material", fv, mine.value);
    }
    if (any) rep.vectors_compared += 1;
}

void compare_tournament_vector(const Json& v, FormulasReport& rep) {
    std::int64_t k = 0, L = 0;
    if (!int_constant(v.get("k"), k)) return;
    const Json* len = find_any(v, {"n", "L", "V", "length"});
    if (!len || !int_constant(*len, L)) return;
    bool with_index = false;
    if (const Json* wi = find_any(v, {"with_index", "index"})) with_index = wi->is_bool() ? wi->as_bool() : wi->as_int() != 0;
    const std::string what = "tournament (L = " + std::to_string(L) + ", k = " + std::to_string(k) + (with_index ? ", index)" : ")");
    if (k != 2 && k != 4 && k != 8) {
        rep.notes.push_back(what + ": arity outside {2, 4, 8} skipped");
        return;
    }
    const TournamentCounts t = tournament_counts(L, static_cast<int>(k), with_index);
    std::int64_t fv = 0;
    bool any = false;
    auto field = [&](std::initializer_list<const char*> names, const char* label, std::int64_t mine) {
        const Json* f = find_any(v, names);
        if (!f || !int_constant(*f, fv)) return;
        any = true;
        note_mismatch(rep, what, label, fv, mine);
    };
    field({"drelu_p1", "p1", "comparisons", "pairwise"}, "drelu_p1", t.comparisons);
    field({"drelu_p2", "p2", "payload_drelu"}, "drelu_p2", t.payload_drelu);
    field({"beaver_select", "beaver", "bv", "selects", "kary_select", "beaver_selects"}, "beaver_selects", t.selects);
    field({"levels"}, "levels", t.levels);
    field({"rounds"}, "rounds", t.rounds);
    field({"payload_bytes"}, "payload_bytes", kOpenBytes * (t.comparisons + t.payload_drelu) + kBeaverOpenBytes * t.selects);
    if (any) rep.tournament_compared += 1;
}

void scan_vectors(const Json& j, int depth, const NumericContract& contract, FormulasReport& rep) {
    if (depth > 4) return;
    if (j.is_array()) {
        for (const Json& e : j.as_array()) {
            if (e.is_object()) {
                if (e.get("component").is_string() && e.has("n") && e.has("d_in")) {
                    compare_component_vector(e, contract, rep);
                    continue;
                }
                if (e.has("k") && !e.has("component") && find_any(e, {"n", "L", "V", "length"})) {
                    compare_tournament_vector(e, rep);
                    continue;
                }
            }
            scan_vectors(e, depth + 1, contract, rep);
        }
    } else if (j.is_object()) {
        for (const auto& kv : j.as_object()) scan_vectors(kv.second, depth + 1, contract, rep);
    }
}

Expected<Json> read_formulas_file(const std::string& resolved) {
    Expected<Json> j = Json::load_file(resolved);
    if (!j.ok) {
        if (j.code.rfind("JSON_IO", 0) == 0) return Expected<Json>::failure("FORMULAS_IO: cannot read " + resolved);
        return Expected<Json>::failure("FORMULAS_JSON: " + j.code);
    }
    if (!j.value.is_object()) return Expected<Json>::failure("FORMULAS_JSON: top level must be an object");
    // a document that names another schema version is not interpreted at all
    const Json* schema = j.value.find("schema");
    if (schema && (!schema->is_string() || schema->as_string() != "opencrab2-formulas-v1"))
        return Expected<Json>::failure("FORMULAS_JSON: unsupported schema (expected opencrab2-formulas-v1)");
    return j;
}

}  // namespace

Json FormulasReport::to_json() const {
    Json j = Json::object();
    j.set("path", path);
    j.set("found", found);
    Json a = Json::array();
    for (const auto& s : adopted) a.push_back(s);
    j.set("adopted", a);
    j.set("vectors_compared", static_cast<long long>(vectors_compared));
    j.set("tournament_compared", static_cast<long long>(tournament_compared));
    j.set("fields_compared", static_cast<long long>(fields_compared));
    Json m = Json::array();
    for (const auto& s : mismatches) m.push_back(s);
    j.set("mismatches", m);
    Json nn = Json::array();
    for (const auto& s : notes) nn.push_back(s);
    j.set("notes", nn);
    return j;
}

std::string find_calibration_file(const std::string& name, const std::string& dir_hint) {
    std::vector<std::string> candidates;
    if (!dir_hint.empty()) candidates.push_back(dir_hint + "/" + name);
    if (const char* env = std::getenv("OPENCRAB2_CALIBRATION_DIR")) {
        if (*env) candidates.push_back(std::string(env) + "/" + name);
    }
    for (const char* prefix : {"", "../", "../../"}) candidates.push_back(std::string(prefix) + "results/calibration/" + name);
    for (const auto& c : candidates)
        if (file_exists(c)) return c;
    return "";
}

namespace {
// Reads the file and adopts its constants; hands the parsed document back for the vector pass.
Expected<FormulasReport> load_formulas_impl(const std::string& path, Json* doc_out) {
    using R = Expected<FormulasReport>;
    FormulasReport rep;
    std::string resolved = path;
    if (resolved.empty()) resolved = find_calibration_file("formulas.json");
    else if (!file_exists(resolved)) resolved.clear();
    if (resolved.empty()) {
        rep.path = path;
        rep.found = false;
        return R::success(rep);
    }
    rep.path = resolved;
    rep.found = true;
    Expected<Json> doc = read_formulas_file(resolved);
    if (!doc.ok) return R::failure(doc.code);
    ConstantScan scan;
    scan_constants(doc.value, 0, scan);
    if (!scan.problems.empty()) return R::failure(scan.problems.front());
    FormulaConstants c = formula_constants();
    if (scan.have_frame) {
        c.frame_bytes = scan.frame;
        rep.adopted.push_back("frame_bytes=" + std::to_string(scan.frame));
    }
    if (scan.have_slots) {
        c.bfv_slots = scan.slots;
        rep.adopted.push_back("bfv_slots=" + std::to_string(scan.slots));
    }
    if (scan.have_ct) {
        c.bfv_ciphertext_bytes = scan.ct;
        c.a2h_status = scan.a2h_status.empty() ? "FILE" : scan.a2h_status;
        rep.adopted.push_back("bfv_ciphertext_bytes=" + std::to_string(scan.ct));
    }
    c.source = resolved;
    Status st = set_formula_constants(c);
    if (!st.ok) return R::failure(st.code);
    if (doc_out) *doc_out = std::move(doc.value);
    return R::success(rep);
}
}  // namespace

Expected<FormulasReport> load_formulas(const std::string& path) { return load_formulas_impl(path, nullptr); }

Expected<FormulasReport> check_formulas(const std::string& path, const NumericContract& contract) {
    Json doc;
    Expected<FormulasReport> rep = load_formulas_impl(path, &doc);
    if (!rep.ok || !rep.value.found) return rep;
    // material sizes and contract-derived counts the file states (python formulas_document)
    auto constant = [&](const Json* holder, const char* key, std::int64_t mine) {
        if (!holder || !holder->is_object()) return;
        const Json* v = holder->find(key);
        std::int64_t fv = 0;
        if (v && int_constant(*v, fv)) note_mismatch(rep.value, "constant", key, fv, mine);
    };
    const Json* consts = doc.find("constants");
    constant(consts, "BEAVER_BYTES", 24);
    constant(consts, "beaver_material_bytes", 24);
    constant(consts, "drelu_p1_bytes", drelu_material_bytes(contract, 1));
    constant(consts, "drelu_p2_bytes", drelu_material_bytes(contract, 2));
    constant(consts, "drelu_p1_material_bytes", drelu_material_bytes(contract, 1));
    constant(consts, "drelu_p2_material_bytes", drelu_material_bytes(contract, 2));
    constant(consts, "trunc_faithful_bytes", trunc_material_bytes(contract));
    constant(consts, "trunc_material_bytes", trunc_material_bytes(contract));
    constant(consts, "ring_to_field_bytes", r2f_material_bytes());
    constant(consts, "r2f_material_bytes", r2f_material_bytes());
    constant(consts, "matrix_triple_elem_bytes", 8);
    constant(consts, "wmask_elem_bytes", 8);
    const Json* derived = doc.get("contract").find("derived");
    constant(derived, "exp_products", exp_beaver(contract));
    constant(derived, "exp_rounds", exp_rounds(contract));
    constant(derived, "gelu_products", contract.gelu_degree() + 2);
    constant(derived, "gelu_rounds", 1 + contract.gelu_degree() + 1);
    constant(derived, "recip_live", recip_live(contract));
    constant(derived, "recip_products", recip_beaver(contract));
    constant(derived, "recip_rounds", recip_rounds(contract));
    constant(derived, "rsqrt_live", rsqrt_live(contract));
    constant(derived, "rsqrt_products", rsqrt_beaver(contract));
    constant(derived, "rsqrt_rounds", rsqrt_rounds(contract));
    // the file is written for one numeric contract: its hash must be the one being compared
    const Json* hash = doc.get("contract").find("hash");
    if (hash && hash->is_string()) {
        rep.value.fields_compared += 1;
        if (hash->as_string() != contract.contract_hash())
            rep.value.mismatches.push_back("contract hash: file " + hash->as_string() + ", cpp " + contract.contract_hash());
    }
    scan_vectors(doc, 0, contract, rep.value);
    return rep;
}

Json formulas_document(const NumericContract& contract) {
    Json root = Json::object();
    root.set("schema", "opencrab2-formulas-v1");
    root.set("generator", "cpp/src/highlevel/costmodel.cpp formulas_document()");
    const FormulaConstants fc = formula_constants();
    Json framing = Json::object();
    framing.set("frame_bytes", static_cast<long long>(fc.frame_bytes));
    framing.set("dim_bytes", 8);
    framing.set("rule", "bytes = payload_bytes + frame_bytes * messages; frame_bytes is the framing of a frame carrying one 1-D uint64 "
                        "array under the wire codec of API.md (8 prefix + tag + dtype + ndim + one u64 dimension)");
    root.set("framing", framing);
    Json bfv = Json::object();
    bfv.set("N", static_cast<long long>(fc.bfv_slots));
    bfv.set("slots_per_ciphertext", static_cast<long long>(fc.bfv_slots));
    bfv.set("ct_bytes", static_cast<long long>(fc.bfv_ciphertext_bytes));
    bfv.set("status", fc.a2h_status);
    bfv.set("constants_source", fc.source);
    root.set("bfv", bfv);
    Json consts = Json::object();
    consts.set("OPEN_BYTES", static_cast<long long>(kOpenBytes));
    consts.set("BEAVER_OPEN_BYTES", static_cast<long long>(kBeaverOpenBytes));
    consts.set("SELECT_BYTES", static_cast<long long>(kBeaverOpenBytes));
    consts.set("ONEWAY_BYTES", static_cast<long long>(kOnewayBytes));
    consts.set("BFV_N", static_cast<long long>(fc.bfv_slots));
    consts.set("BEAVER_BYTES", 24);
    consts.set("drelu_p1_bytes", static_cast<long long>(drelu_material_bytes(contract, 1)));
    consts.set("drelu_p2_bytes", static_cast<long long>(drelu_material_bytes(contract, 2)));
    consts.set("trunc_faithful_bytes", static_cast<long long>(trunc_material_bytes(contract)));
    consts.set("ring_to_field_bytes", static_cast<long long>(r2f_material_bytes()));
    consts.set("matrix_triple_elem_bytes", 8);
    consts.set("wmask_elem_bytes", 8);
    root.set("constants", consts);
    Json cj = Json::object();
    cj.set("name", contract.name);
    cj.set("hash", contract.contract_hash());
    Json derived = Json::object();
    derived.set("exp_products", static_cast<long long>(exp_beaver(contract)));
    derived.set("exp_rounds", static_cast<long long>(exp_rounds(contract)));
    derived.set("gelu_products", contract.gelu_degree() + 2);
    derived.set("gelu_rounds", 1 + contract.gelu_degree() + 1);
    derived.set("recip_live", static_cast<long long>(recip_live(contract)));
    derived.set("recip_products", static_cast<long long>(recip_beaver(contract)));
    derived.set("recip_rounds", static_cast<long long>(recip_rounds(contract)));
    derived.set("rsqrt_live", static_cast<long long>(rsqrt_live(contract)));
    derived.set("rsqrt_products", static_cast<long long>(rsqrt_beaver(contract)));
    derived.set("rsqrt_rounds", static_cast<long long>(rsqrt_rounds(contract)));
    cj.set("derived", derived);
    root.set("contract", cj);
    Json tv = Json::array();
    for (std::int64_t L : {1, 2, 3, 5, 8, 17, 32, 64, 1000, 50257})
        for (int k : {2, 4, 8})
            for (bool wi : {false, true}) {
                const TournamentCounts t = tournament_counts(L, k, wi);
                Json e = Json::object();
                e.set("n", static_cast<long long>(L));
                e.set("k", k);
                e.set("with_index", wi);
                e.set("drelu_p1", static_cast<long long>(t.comparisons));
                e.set("drelu_p2", static_cast<long long>(t.payload_drelu));
                e.set("beaver_select", static_cast<long long>(t.selects));
                e.set("levels", static_cast<long long>(t.levels));
                e.set("rounds", static_cast<long long>(t.rounds));
                e.set("payload_bytes", static_cast<long long>(kOpenBytes * (t.comparisons + t.payload_drelu) + kBeaverOpenBytes * t.selects));
                tv.push_back(e);
            }
    root.set("tournament", tv);
    struct Shape {
        const char* comp;
        std::int64_t n, d_in, d_out;
    };
    const Shape grid[] = {
        {"linear_wmask", 1, 768, 2304},     {"linear_wmask", 8, 256, 256},     {"linear_generic", 1, 768, 2304},
        {"linear_generic", 8, 256, 256},    {"embedding_wmask", 1, 50257, 768}, {"embedding_wmask", 4, 1000, 64},
        {"embedding_generic", 1, 50257, 768}, {"embedding_generic", 4, 1000, 64}, {"matmul_shared", 12, 64, 32},
        {"install_wmask", 1, 768, 3072},    {"install_wmask", 1, 50257, 768},  {"gelu_clip12", 4, 3072, 3072},
        {"gelu_clip12", 1, 256, 256},       {"layernorm_rows", 4, 768, 768},   {"layernorm_rows", 1, 64, 64},
        {"softmax_rows_k2", 12, 32, 32},    {"softmax_rows_k2", 48, 17, 17},   {"softmax_rows_k2", 2, 5, 5},
        {"softmax_rows_k4", 12, 32, 32},    {"softmax_rows_k4", 48, 17, 17},   {"softmax_rows_k4", 2, 5, 5},
        {"argmax_k2", 1, 50257, 1},         {"argmax_k2", 3, 17, 1},           {"argmax_k2", 1, 1000, 1},
        {"argmax_k4", 1, 50257, 1},         {"argmax_k4", 3, 17, 1},           {"argmax_k4", 1, 1000, 1},
        {"argmax_k8", 1, 50257, 1},         {"argmax_k8", 3, 17, 1},           {"argmax_k8", 1, 1000, 1},
        {"trunc_faithful", 10, 10, 1},      {"ring_to_field", 10, 10, 1},      {"a2h_exact", 1, 100, 1},
        {"a2h_exact", 1, 8193, 1}};
    Json vs = Json::array();
    for (const Shape& s : grid) {
        Json e = Json::object();
        e.set("component", s.comp);
        e.set("n", static_cast<long long>(s.n));
        e.set("d_in", static_cast<long long>(s.d_in));
        e.set("d_out", static_cast<long long>(s.d_out));
        e.set("macs", static_cast<long long>(macs(s.comp, s.n, s.d_in, s.d_out).value));
        e.set("payload_bytes", static_cast<long long>(payload_bytes(s.comp, s.n, s.d_in, s.d_out).value));
        e.set("bytes", static_cast<long long>(bytes_formula(s.comp, s.n, s.d_in, s.d_out).value));
        e.set("rounds", static_cast<long long>(rounds_formula(s.comp, s.n, s.d_in, s.d_out).value));
        e.set("messages", static_cast<long long>(messages_formula(s.comp, s.n, s.d_in, s.d_out).value));
        e.set("material_bytes", static_cast<long long>(material_formula(s.comp, s.n, s.d_in, s.d_out, contract).value));
        Json mat = Json::object();
        for (const auto& kv : material_by_type(s.comp, s.n, s.d_in, s.d_out, contract).value) mat.set(kv.first, static_cast<long long>(kv.second));
        e.set("material", mat);
        const GadgetCounts gc = gadget_counts(s.comp, s.n, s.d_in, s.d_out, contract).value;
        Json gates = Json::object();
        gates.set("drelu_p1", static_cast<long long>(gc.drelu));
        gates.set("drelu_p2", static_cast<long long>(gc.drelu_p2));
        gates.set("beaver", static_cast<long long>(gc.beaver));
        gates.set("kary_select", static_cast<long long>(gc.kary_select));
        gates.set("trunc", static_cast<long long>(gc.trunc_faithful));
        gates.set("r2f", static_cast<long long>(gc.ring_to_field));
        gates.set("matrix_triple", static_cast<long long>(gc.matrix_triple));
        gates.set("wmask_call", static_cast<long long>(gc.wmask_call));
        gates.set("wmask_mask", static_cast<long long>(gc.wmask_mask));
        gates.set("open_elems", static_cast<long long>(gc.open_elems));
        gates.set("oneway_elems", static_cast<long long>(gc.oneway_elems));
        gates.set("bfv_ct", static_cast<long long>(gc.bfv_ct));
        gates.set("rounds", static_cast<long long>(gc.rounds));
        gates.set("messages", static_cast<long long>(gc.messages));
        e.set("gates", gates);
        vs.push_back(e);
    }
    root.set("shapes", vs);
    return root;
}

// ---------------------------------------------------------------------------
// fits
// ---------------------------------------------------------------------------
Json ComponentFit::to_json() const {
    Json j = Json::object();
    j.set("compute_per_mac_s", compute_per_mac_s);
    j.set("compute_per_byte_s", compute_per_byte_s);
    j.set("compute_fixed_s", compute_fixed_s);
    j.set("sigma_compute_s", sigma_compute_s);
    j.set("n_train", static_cast<long long>(n_train));
    j.set("status", status);
    j.set("source", source);
    return j;
}

Expected<ComponentFit> ComponentFit::from_json(const Json& j) {
    using R = Expected<ComponentFit>;
    if (!j.is_object()) return R::failure("COSTMODEL_JSON: component fit must be an object");
    ComponentFit f;
    Expected<double> a = j.get_double("compute_per_mac_s");
    if (!a.ok) return R::failure("COSTMODEL_JSON: " + a.code);
    Expected<double> c = j.get_double("compute_per_byte_s");
    if (!c.ok) return R::failure("COSTMODEL_JSON: " + c.code);
    Expected<double> b = j.get_double("compute_fixed_s");
    if (!b.ok) return R::failure("COSTMODEL_JSON: " + b.code);
    Expected<double> s = j.get_double("sigma_compute_s");
    if (!s.ok) return R::failure("COSTMODEL_JSON: " + s.code);
    f.compute_per_mac_s = a.value;
    f.compute_per_byte_s = c.value;
    f.compute_fixed_s = b.value;
    f.sigma_compute_s = s.value;
    if (j.has("n_train")) {
        Expected<std::int64_t> k = j.get_int("n_train");
        if (!k.ok) return R::failure("COSTMODEL_JSON: " + k.code);
        f.n_train = k.value;
    }
    if (j.has("status")) {
        if (!j.get("status").is_string()) return R::failure("COSTMODEL_JSON: status must be a string");
        f.status = j.get("status").as_string();
    }
    if (j.has("source")) {
        if (!j.get("source").is_string()) return R::failure("COSTMODEL_JSON: source must be a string");
        f.source = j.get("source").as_string();
    }
    if (!std::isfinite(f.compute_per_mac_s) || !std::isfinite(f.compute_per_byte_s) || !std::isfinite(f.compute_fixed_s) ||
        !std::isfinite(f.sigma_compute_s))
        return R::failure("COSTMODEL_JSON: coefficients must be finite");
    if (f.compute_per_mac_s < 0 || f.compute_per_byte_s < 0 || f.compute_fixed_s < 0 || f.sigma_compute_s < 0)
        return R::failure("COSTMODEL_JSON: coefficients must be non-negative");
    return R::success(f);
}

// ---------------------------------------------------------------------------
// NNLS (Lawson-Hanson active set) for small dense problems
// ---------------------------------------------------------------------------
namespace {

// Solve least squares X[:, cols] w = y by normal equations with Gaussian elimination (small p).
bool solve_ls_subset(const std::vector<std::vector<double>>& X, const std::vector<double>& y, const std::vector<int>& cols,
                     std::vector<double>& w) {
    const std::size_t p = cols.size();
    const std::size_t m = X.size();
    std::vector<std::vector<double>> A(p, std::vector<double>(p + 1, 0.0));
    for (std::size_t i = 0; i < p; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            double s = 0.0;
            for (std::size_t r = 0; r < m; ++r) s += X[r][static_cast<std::size_t>(cols[i])] * X[r][static_cast<std::size_t>(cols[j])];
            A[i][j] = s;
        }
        double s = 0.0;
        for (std::size_t r = 0; r < m; ++r) s += X[r][static_cast<std::size_t>(cols[i])] * y[r];
        A[i][p] = s;
    }
    // Gaussian elimination with partial pivoting and a tiny ridge for singular subsets.
    for (std::size_t i = 0; i < p; ++i) A[i][i] += 1e-18;
    for (std::size_t col = 0; col < p; ++col) {
        std::size_t piv = col;
        for (std::size_t r = col + 1; r < p; ++r)
            if (std::fabs(A[r][col]) > std::fabs(A[piv][col])) piv = r;
        if (std::fabs(A[piv][col]) < 1e-300) return false;
        std::swap(A[piv], A[col]);
        for (std::size_t r = 0; r < p; ++r) {
            if (r == col) continue;
            const double f = A[r][col] / A[col][col];
            for (std::size_t k = col; k <= p; ++k) A[r][k] -= f * A[col][k];
        }
    }
    w.assign(p, 0.0);
    for (std::size_t i = 0; i < p; ++i) w[i] = A[i][p] / A[i][i];
    return true;
}

}  // namespace

Expected<std::vector<double>> nnls(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
    using R = Expected<std::vector<double>>;
    if (X.empty() || X.size() != y.size()) return R::failure("NNLS_SHAPE: rows of X must match y");
    const std::size_t p = X[0].size();
    for (const auto& row : X)
        if (row.size() != p) return R::failure("NNLS_SHAPE: ragged X");
    std::vector<double> x(p, 0.0);
    std::vector<bool> active(p, true);   // in the zero set
    const std::size_t m = X.size();
    auto gradient = [&](std::vector<double>& w) {
        std::vector<double> r(m);
        for (std::size_t i = 0; i < m; ++i) {
            double s = 0.0;
            for (std::size_t j = 0; j < p; ++j) s += X[i][j] * x[j];
            r[i] = y[i] - s;
        }
        w.assign(p, 0.0);
        for (std::size_t j = 0; j < p; ++j)
            for (std::size_t i = 0; i < m; ++i) w[j] += X[i][j] * r[i];
    };
    std::vector<double> w;
    for (int outer = 0; outer < 3 * static_cast<int>(p) + 10; ++outer) {
        gradient(w);
        int best = -1;
        double bestw = 1e-12;
        for (std::size_t j = 0; j < p; ++j)
            if (active[j] && w[j] > bestw) { bestw = w[j]; best = static_cast<int>(j); }
        if (best < 0) break;
        active[static_cast<std::size_t>(best)] = false;
        for (int inner = 0; inner < 3 * static_cast<int>(p) + 10; ++inner) {
            std::vector<int> cols;
            for (std::size_t j = 0; j < p; ++j)
                if (!active[j]) cols.push_back(static_cast<int>(j));
            std::vector<double> z;
            if (!solve_ls_subset(X, y, cols, z)) return R::failure("NNLS_SINGULAR: cannot solve the passive subset");
            bool all_pos = true;
            for (double v : z)
                if (v <= 0.0) all_pos = false;
            if (all_pos) {
                for (std::size_t k = 0; k < cols.size(); ++k) x[static_cast<std::size_t>(cols[k])] = z[k];
                break;
            }
            // step toward z until a coordinate hits zero
            double alpha = 1.0;
            for (std::size_t k = 0; k < cols.size(); ++k) {
                if (z[k] <= 0.0) {
                    const double xj = x[static_cast<std::size_t>(cols[k])];
                    const double a = xj / (xj - z[k]);
                    if (a < alpha) alpha = a;
                }
            }
            for (std::size_t k = 0; k < cols.size(); ++k) {
                std::size_t j = static_cast<std::size_t>(cols[k]);
                x[j] += alpha * (z[k] - x[j]);
                if (x[j] <= 1e-15) { x[j] = 0.0; active[j] = true; }
            }
        }
    }
    for (double& v : x)
        if (v < 0.0) v = 0.0;
    return R::success(x);
}

Expected<FittedCostModel> FittedCostModel::fit(const std::vector<CalibrationRow>& rows, double rel_bw, double rel_rtt) {
    using R = Expected<FittedCostModel>;
    if (rows.empty()) return R::failure("COSTMODEL_FIT: no calibration rows");
    FittedCostModel model;
    model.rel_bw = rel_bw;
    model.rel_rtt = rel_rtt;
    std::map<std::string, std::vector<const CalibrationRow*>> by_comp;
    for (const CalibrationRow& r : rows) {
        if (!is_component(r.component)) return R::failure("UNKNOWN_COMPONENT: " + r.component);
        by_comp[r.component].push_back(&r);
    }
    for (const auto& kv : by_comp) {
        const std::string& comp = kv.first;
        const auto& rs = kv.second;
        std::vector<std::vector<double>> X;
        std::vector<double> y;
        std::vector<double> scale(3, 1e-300);
        for (const CalibrationRow* r : rs) {
            Expected<std::int64_t> mc = macs(comp, r->n, r->d_in, r->d_out);
            if (!mc.ok) return R::failure(mc.code);
            Env env;
            env.bandwidth_bps = r->bandwidth_bps;
            env.one_way_latency_s = r->one_way_latency_s;
            env.cpu_share = r->cpu_share > 0 ? r->cpu_share : 1.0;
            double transport = static_cast<double>(r->rounds) * env.rtt_s();
            if (env.bandwidth_bps > 0) transport += static_cast<double>(r->bytes) * 8.0 / env.bandwidth_bps;
            // compute term is scaled by cpu_share: (a*macs + c*bytes + b) / cpu_share = measured - transport
            const double target = (r->measured_s - transport) * env.cpu_share;
            X.push_back({static_cast<double>(mc.value), static_cast<double>(r->bytes), 1.0});
            y.push_back(target);
            for (int j = 0; j < 3; ++j) scale[static_cast<std::size_t>(j)] = std::max(scale[static_cast<std::size_t>(j)], std::fabs(X.back()[static_cast<std::size_t>(j)]));
        }
        std::vector<std::vector<double>> Xs = X;
        for (auto& row : Xs)
            for (int j = 0; j < 3; ++j) row[static_cast<std::size_t>(j)] /= scale[static_cast<std::size_t>(j)];
        Expected<std::vector<double>> coef = nnls(Xs, y);
        if (!coef.ok) return R::failure(coef.code);
        ComponentFit f;
        f.compute_per_mac_s = coef.value[0] / scale[0];
        f.compute_per_byte_s = coef.value[1] / scale[1];
        f.compute_fixed_s = coef.value[2] / scale[2];
        f.n_train = static_cast<std::int64_t>(rs.size());
        // repeat spread per configuration
        std::map<std::string, std::vector<double>> groups;
        for (std::size_t i = 0; i < rs.size(); ++i) {
            const CalibrationRow* r = rs[i];
            const std::string key = std::to_string(r->n) + "," + std::to_string(r->d_in) + "," + std::to_string(r->d_out) + "," +
                                    format_double_repr(r->bandwidth_bps) + "," + format_double_repr(r->one_way_latency_s);
            groups[key].push_back(y[i]);
        }
        double ss = 0.0;
        int cnt = 0;
        for (const auto& g : groups) {
            if (g.second.size() < 2) continue;
            double mean = 0.0;
            for (double v : g.second) mean += v;
            mean /= static_cast<double>(g.second.size());
            double var = 0.0;
            for (double v : g.second) var += (v - mean) * (v - mean);
            var /= static_cast<double>(g.second.size() - 1);
            ss += var;
            ++cnt;
        }
        f.sigma_compute_s = cnt ? std::sqrt(ss / cnt) : 0.0;
        model.fits[comp] = f;
    }
    return R::success(model);
}

Expected<Prediction> FittedCostModel::predict(const std::string& comp, std::int64_t n, std::int64_t d_in, std::int64_t d_out,
                                              const Env& env) const {
    using R = Expected<Prediction>;
    if (!is_component(comp)) return R::failure("UNKNOWN_COMPONENT: " + comp);
    auto it = fits.find(comp);
    if (it == fits.end()) return R::failure("COSTMODEL_NOT_FITTED: " + comp);
    if (!(env.cpu_share > 0.0) || !std::isfinite(env.cpu_share)) return R::failure("ENV_PARAMS: cpu_share must be finite and > 0");
    if (!std::isfinite(env.bandwidth_bps) || env.bandwidth_bps < 0.0)
        return R::failure("ENV_PARAMS: bandwidth_bps must be finite and >= 0 (0 = unlimited)");
    if (!std::isfinite(env.one_way_latency_s) || env.one_way_latency_s < 0.0)
        return R::failure("ENV_PARAMS: one_way_latency_s must be finite and >= 0");
    Expected<std::int64_t> mc = macs(comp, n, d_in, d_out);
    if (!mc.ok) return R::failure(mc.code);
    Expected<std::int64_t> by = bytes_formula(comp, n, d_in, d_out);
    if (!by.ok) return R::failure(by.code);
    Expected<std::int64_t> rd = rounds_formula(comp, n, d_in, d_out);
    if (!rd.ok) return R::failure(rd.code);
    const ComponentFit& f = it->second;
    Prediction p;
    p.bytes = by.value;
    p.rounds = rd.value;
    p.compute_s = (f.compute_per_mac_s * static_cast<double>(mc.value) + f.compute_per_byte_s * static_cast<double>(by.value) +
                   f.compute_fixed_s) / env.cpu_share;
    p.comm_s = env.bandwidth_bps > 0.0 ? static_cast<double>(by.value) * 8.0 / env.bandwidth_bps : 0.0;
    p.round_s = static_cast<double>(rd.value) * env.rtt_s();
    p.mean_s = p.compute_s + p.comm_s + p.round_s;
    const double sc = f.sigma_compute_s / env.cpu_share;
    p.sigma_s = std::sqrt(sc * sc + (p.comm_s * rel_bw) * (p.comm_s * rel_bw) + (p.round_s * rel_rtt) * (p.round_s * rel_rtt));
    return R::success(p);
}

std::string FittedCostModel::status_of(const std::string& component) const {
    auto it = fits.find(component);
    return it == fits.end() ? std::string("MISSING") : it->second.status;
}

Json FittedCostModel::to_json() const {
    Json j = Json::object();
    j.set("rel_bw", rel_bw);
    j.set("rel_rtt", rel_rtt);
    j.set("label", label);
    Json comps = Json::object();
    for (const auto& kv : fits) comps.set(kv.first, kv.second.to_json());
    j.set("components", comps);
    return j;
}

Expected<FittedCostModel> FittedCostModel::from_json(const Json& j) {
    using R = Expected<FittedCostModel>;
    if (!j.is_object()) return R::failure("COSTMODEL_JSON: expected an object");
    FittedCostModel m;
    if (j.has("rel_bw")) {
        Expected<double> v = j.get_double("rel_bw");
        if (!v.ok) return R::failure("COSTMODEL_JSON: " + v.code);
        m.rel_bw = v.value;
    }
    if (j.has("rel_rtt")) {
        Expected<double> v = j.get_double("rel_rtt");
        if (!v.ok) return R::failure("COSTMODEL_JSON: " + v.code);
        m.rel_rtt = v.value;
    }
    if (!(m.rel_bw >= 0.0) || !(m.rel_rtt >= 0.0)) return R::failure("COSTMODEL_JSON: rel_bw and rel_rtt must be non-negative");
    if (j.has("label")) {
        if (!j.get("label").is_string()) return R::failure("COSTMODEL_JSON: label must be a string");
        m.label = j.get("label").as_string();
    }
    const Json* comps = j.find("components");
    if (!comps || !comps->is_object()) return R::failure("COSTMODEL_JSON: components object required");
    for (const auto& kv : comps->as_object()) {
        if (!is_component(kv.first)) return R::failure("UNKNOWN_COMPONENT: " + kv.first);
        Expected<ComponentFit> f = ComponentFit::from_json(kv.second);
        if (!f.ok) return R::failure(f.code + " (" + kv.first + ")");
        m.fits[kv.first] = f.value;
    }
    return R::success(m);
}

Expected<FittedCostModel> FittedCostModel::load(const std::string& path) {
    Expected<Json> j = Json::load_file(path);
    if (!j.ok) return Expected<FittedCostModel>::failure(j.code);
    return from_json(j.value);
}

Status FittedCostModel::save(const std::string& path) const { return Json::save_file(path, to_json(), true); }

FittedCostModel historical_prior_model() {
    // Retained Cycle-3 fit (results/native/native_calibration_metrics.json), status HISTORICAL.
    // mpc_wmask -> linear/embedding wmask, mpc_generic -> generic lanes and matmul_shared,
    // gelu / layernorm / softmax / argmax carried over; new components get the nearest analogue.
    FittedCostModel m;
    m.label = "HISTORICAL";
    auto put = [&](const char* name, double a, double c, double b, double s) {
        ComponentFit f;
        f.compute_per_mac_s = a;
        f.compute_per_byte_s = c;
        f.compute_fixed_s = b;
        f.sigma_compute_s = s;
        f.n_train = 0;
        f.status = "HISTORICAL";
        f.source = "historical_prior_model";
        m.fits[name] = f;
    };
    put("linear_wmask", 8.386853427599183e-09, 0.0, 0.0, 0.011876389285424472);
    put("embedding_wmask", 8.386853427599183e-09, 0.0, 0.0, 0.011876389285424472);
    put("linear_generic", 7.246562834215189e-09, 3.781701344572254e-09, 0.0, 0.03237468890266239);
    put("embedding_generic", 7.246562834215189e-09, 3.781701344572254e-09, 0.0, 0.03237468890266239);
    put("matmul_shared", 7.246562834215189e-09, 3.781701344572254e-09, 0.0, 0.03237468890266239);
    put("install_wmask", 0.0, 3.781701344572254e-09, 0.002, 0.005);
    put("gelu_clip12", 5.9701287822591305e-06, 0.0, 0.029399970603295914, 0.023656370579697408);
    put("softmax_rows_k2", 2.7322804816411642e-05, 0.0, 0.030354139232777085, 0.004737581603556359);
    put("softmax_rows_k4", 3.0e-05, 0.0, 0.030354139232777085, 0.004737581603556359);
    put("layernorm_rows", 2.235558365892894e-06, 0.0, 0.026014880064250623, 0.0024341225314535825);
    put("argmax_k2", 3.072624346399207e-06, 0.0, 0.0, 0.015933937797522467);
    put("argmax_k4", 3.6e-06, 0.0, 0.0, 0.015933937797522467);
    put("argmax_k8", 4.5e-06, 0.0, 0.0, 0.015933937797522467);
    put("trunc_faithful", 2.4e-06, 0.0, 0.002, 0.002);
    put("ring_to_field", 2.6e-06, 0.0, 0.002, 0.002);
    put("a2h_exact", 2.6e-06, 4.478109494765893e-07, 0.04109047626504255, 0.05);
    return m;
}

Expected<DefaultModel> default_model(const std::string& calibration_dir) {
    using R = Expected<DefaultModel>;
    DefaultModel d;
    const std::string path = find_calibration_file("fitted_model.json", calibration_dir);
    if (path.empty()) {
        d.model = historical_prior_model();
        d.status = "HISTORICAL";
        d.source = "historical_prior_model";
        return R::success(std::move(d));
    }
    // A calibration file that exists but is broken is an error, never a silent fallback.
    Expected<FittedCostModel> m = FittedCostModel::load(path);
    if (!m.ok) return R::failure(m.code + " (" + path + ")");
    d.model = std::move(m.value);
    d.status = d.model.label;
    d.source = path;
    return R::success(std::move(d));
}

namespace {
std::vector<double> rankdata(const std::vector<double>& v) {
    std::vector<std::size_t> order(v.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return v[a] < v[b]; });
    std::vector<double> ranks(v.size(), 0.0);
    std::size_t i = 0;
    while (i < order.size()) {
        std::size_t j = i;
        while (j + 1 < order.size() && v[order[j + 1]] == v[order[i]]) ++j;
        const double r = (static_cast<double>(i) + static_cast<double>(j)) / 2.0 + 1.0;
        for (std::size_t k = i; k <= j; ++k) ranks[order[k]] = r;
        i = j + 1;
    }
    return ranks;
}
}  // namespace

Expected<Json> error_metrics(const std::vector<double>& measured, const std::vector<double>& predicted) {
    using R = Expected<Json>;
    if (measured.empty() || measured.size() != predicted.size()) return R::failure("METRICS_SHAPE: vectors must be non-empty and equal length");
    std::vector<double> rel(measured.size());
    for (std::size_t i = 0; i < measured.size(); ++i) rel[i] = std::fabs(predicted[i] - measured[i]) / std::max(measured[i], 1e-12);
    std::vector<double> sorted = rel;
    std::sort(sorted.begin(), sorted.end());
    auto pct = [&](double q) {
        const double pos = q / 100.0 * static_cast<double>(sorted.size() - 1);
        const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
        const std::size_t hi = std::min(lo + 1, sorted.size() - 1);
        return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - static_cast<double>(lo));
    };
    double mape = 0.0;
    for (double v : rel) mape += v;
    mape /= static_cast<double>(rel.size());
    double spearman = std::nan("");
    if (measured.size() >= 2) {
        const std::vector<double> ra = rankdata(measured), rb = rankdata(predicted);
        double ma = 0, mb = 0;
        for (std::size_t i = 0; i < ra.size(); ++i) { ma += ra[i]; mb += rb[i]; }
        ma /= static_cast<double>(ra.size());
        mb /= static_cast<double>(rb.size());
        double num = 0, da = 0, db = 0;
        for (std::size_t i = 0; i < ra.size(); ++i) {
            num += (ra[i] - ma) * (rb[i] - mb);
            da += (ra[i] - ma) * (ra[i] - ma);
            db += (rb[i] - mb) * (rb[i] - mb);
        }
        if (da > 0 && db > 0) spearman = num / std::sqrt(da * db);
    }
    Json j = Json::object();
    j.set("mape", mape);
    j.set("median_rel_err", pct(50.0));
    j.set("p95_rel_err", pct(95.0));
    j.set("spearman", spearman);
    j.set("n", static_cast<long long>(measured.size()));
    return R::success(j);
}

}  // namespace highlevel
}  // namespace opencrab2
