// OpenCrab v2: candidate enumeration implementation (see include/opencrab2/highlevel/candidates.hpp).
#include "opencrab2/highlevel/candidates.hpp"

#include <algorithm>
#include <cmath>

namespace opencrab2 {
namespace highlevel {

// ---------------------------------------------------------------------------
// SessionState
// ---------------------------------------------------------------------------
bool SessionState::object_installed(const std::string& object_id) const {
    auto it = installed.find(object_id);
    if (it == installed.end()) return false;
    const Json& v = it->second;
    if (!v.is_object()) return false;
    const Json* mv = v.find("model_version");
    if (!mv || !mv->is_string() || mv->as_string() != model_version) return false;
    const Json* ch = v.find("contract_hash");
    if (ch && (!ch->is_string() || ch->as_string() != contract_hash)) return false;
    return true;
}

Json SessionState::to_json() const {
    Json j = Json::object();
    j.set("incumbent", incumbent);
    Json inst = Json::object();
    for (const auto& kv : installed) inst.set(kv.first, kv.second);
    j.set("installed", inst);
    Json st = Json::object();
    for (const auto& kv : stock) {
        Json pair = Json::array();
        pair.push_back(static_cast<long long>(kv.second.first));
        pair.push_back(static_cast<long long>(kv.second.second));
        st.set(kv.first, pair);
    }
    j.set("stock", st);
    j.set("kv_bucket", kv_bucket);
    j.set("remaining_epochs", remaining_epochs);
    j.set("session_id", session_id);
    j.set("model_version", model_version);
    j.set("contract_hash", contract_hash);
    return j;
}

Expected<SessionState> SessionState::from_json(const Json& j) {
    using R = Expected<SessionState>;
    if (!j.is_object()) return R::failure("STATE_JSON: expected an object");
    SessionState s;
    const Json* inc = j.find("incumbent");
    if (!inc || !inc->is_object()) return R::failure("STATE_JSON: incumbent plan required");
    s.incumbent = *inc;
    const Json* inst = j.find("installed");
    if (inst) {
        if (!inst->is_object()) return R::failure("STATE_JSON: installed must be an object");
        for (const auto& kv : inst->as_object()) s.installed[kv.first] = kv.second;
    }
    const Json* st = j.find("stock");
    if (st) {
        if (!st->is_object()) return R::failure("STATE_JSON: stock must be an object");
        for (const auto& kv : st->as_object()) {
            if (!kv.second.is_array() || kv.second.size() != 2) return R::failure("STATE_JSON: stock entries are [count, bytes]");
            s.stock[kv.first] = std::make_pair(kv.second[0].as_int(), kv.second[1].as_int());
        }
    }
    Expected<std::int64_t> kv = j.get_int("kv_bucket");
    if (!kv.ok) return R::failure("STATE_JSON: " + kv.code);
    Expected<std::int64_t> re = j.get_int("remaining_epochs");
    if (!re.ok) return R::failure("STATE_JSON: " + re.code);
    s.kv_bucket = static_cast<int>(kv.value);
    s.remaining_epochs = static_cast<int>(re.value);
    s.session_id = j.get("session_id").as_string();
    s.model_version = j.get("model_version").as_string();
    s.contract_hash = j.get("contract_hash").as_string();
    return R::success(s);
}

// ---------------------------------------------------------------------------
// Candidate
// ---------------------------------------------------------------------------
Json Candidate::to_json() const {
    Json j = Json::object();
    j.set("plan", plan);
    Json objs = Json::array();
    for (const auto& o : install_objects) objs.push_back(o);
    j.set("install_objects", objs);
    Json costs = Json::object();
    for (const auto& kv : install_cost_by_object) costs.set(kv.first, kv.second);
    j.set("install_cost_by_object", costs);
    j.set("c_install_s", c_install_s);
    j.set("c_install_sigma", c_install_sigma);
    j.set("exec_s_per_epoch", exec_s_per_epoch);
    j.set("exec_sigma", exec_sigma);
    j.set("material_bytes_per_epoch", static_cast<long long>(material_bytes_per_epoch));
    j.set("mem_peak_bytes", static_cast<long long>(mem_peak_bytes));
    j.set("feasible", feasible);
    j.set("reason", reason);
    if (!detail.empty()) j.set("detail", detail);
    j.set("c_wait_s", c_wait_s);
    Json mat = Json::object();
    for (const auto& kv : material_by_type) mat.set(kv.first, static_cast<long long>(kv.second));
    j.set("material_by_type", mat);
    j.set("online_bytes_per_epoch", static_cast<long long>(online_bytes_per_epoch));
    j.set("rounds_per_epoch", static_cast<long long>(rounds_per_epoch));
    Json by_s = Json::array();
    for (double v : exec_s_by_epoch) by_s.push_back(v);
    j.set("exec_s_by_epoch", by_s);
    Json by_g = Json::array();
    for (double v : exec_sigma_by_epoch) by_g.push_back(v);
    j.set("exec_sigma_by_epoch", by_g);
    return j;
}

Status validate_schedule(const Candidate& c) {
    if (!c.exec_s_by_epoch.empty() && c.exec_sigma_by_epoch.size() != c.exec_s_by_epoch.size())
        return fail_status("CANDIDATE_SCHEDULE: " + std::to_string(c.exec_s_by_epoch.size()) + " step costs but " +
                           std::to_string(c.exec_sigma_by_epoch.size()) + " step sigmas");
    return ok_status();
}

double exec_at(const Candidate& c, std::size_t j) {
    if (c.exec_s_by_epoch.empty()) return c.exec_s_per_epoch;
    return c.exec_s_by_epoch[std::min(j, c.exec_s_by_epoch.size() - 1)];
}

double exec_sigma_at(const Candidate& c, std::size_t j) {
    if (c.exec_sigma_by_epoch.empty()) return c.exec_sigma;
    return c.exec_sigma_by_epoch[std::min(j, c.exec_sigma_by_epoch.size() - 1)];
}

Expected<Candidate> Candidate::from_json(const Json& j) {
    using R = Expected<Candidate>;
    if (!j.is_object()) return R::failure("CANDIDATE_JSON: expected an object");
    Candidate c;
    const Json* plan = j.find("plan");
    if (!plan || !plan->is_object()) return R::failure("CANDIDATE_JSON: plan required");
    c.plan = *plan;
    for (const Json& o : j.get("install_objects").as_array()) c.install_objects.push_back(o.as_string());
    for (const auto& kv : j.get("install_cost_by_object").as_object()) c.install_cost_by_object[kv.first] = kv.second.as_double();
    Expected<double> v = j.get_double("c_install_s");
    if (!v.ok) return R::failure("CANDIDATE_JSON: " + v.code);
    c.c_install_s = v.value;
    c.c_install_sigma = j.get("c_install_sigma").as_double();
    v = j.get_double("exec_s_per_epoch");
    if (!v.ok) return R::failure("CANDIDATE_JSON: " + v.code);
    c.exec_s_per_epoch = v.value;
    c.exec_sigma = j.get("exec_sigma").as_double();
    c.material_bytes_per_epoch = j.get("material_bytes_per_epoch").as_int();
    c.mem_peak_bytes = j.get("mem_peak_bytes").as_int();
    c.feasible = j.has("feasible") ? j.get("feasible").as_bool() : true;
    c.reason = j.get("reason").as_string();
    c.detail = j.get("detail").as_string();
    c.c_wait_s = j.get("c_wait_s").as_double();
    for (const auto& kv : j.get("material_by_type").as_object()) c.material_by_type[kv.first] = kv.second.as_int();
    c.online_bytes_per_epoch = j.get("online_bytes_per_epoch").as_int();
    c.rounds_per_epoch = j.get("rounds_per_epoch").as_int();
    // per-step horizon lists: absent or empty = one shape
    auto read_list = [&](const char* key, std::vector<double>& dst) -> std::string {
        if (!j.has(key)) return "";
        const Json& arr = j.get(key);
        if (!arr.is_array()) return std::string("CANDIDATE_JSON: ") + key + " must be an array";
        for (const Json& x : arr.as_array()) {
            if (!x.is_number()) return std::string("CANDIDATE_JSON: ") + key + " entries must be numbers";
            dst.push_back(x.as_double());
        }
        return "";
    };
    std::string list_err = read_list("exec_s_by_epoch", c.exec_s_by_epoch);
    if (list_err.empty()) list_err = read_list("exec_sigma_by_epoch", c.exec_sigma_by_epoch);
    if (!list_err.empty()) return R::failure(list_err);
    Status sched = validate_schedule(c);
    if (!sched.ok) return R::failure(sched.code);
    return R::success(c);
}

// ---------------------------------------------------------------------------
// Catalog / Shapes
// ---------------------------------------------------------------------------
Catalog Catalog::full() { return Catalog(); }

Json Catalog::to_json() const {
    Json j = Json::object();
    Json e = Json::array();
    for (const auto& s : embedding) e.push_back(s);
    Json l = Json::array();
    for (const auto& s : linear) l.push_back(s);
    Json a = Json::array();
    for (int k : argmax_k) a.push_back(k);
    Json m = Json::array();
    for (int k : max_k) m.push_back(k);
    j.set("embedding", e);
    j.set("linear", l);
    j.set("argmax_k", a);
    j.set("max_k", m);
    return j;
}

Expected<Catalog> Catalog::from_json(const Json& j) {
    using R = Expected<Catalog>;
    if (!j.is_object()) return R::failure("CATALOG_JSON: expected an object");
    Catalog c;
    c.embedding.clear();
    c.linear.clear();
    c.argmax_k.clear();
    c.max_k.clear();
    for (const Json& v : j.get("embedding").as_array()) c.embedding.push_back(v.as_string());
    for (const Json& v : j.get("linear").as_array()) c.linear.push_back(v.as_string());
    for (const Json& v : j.get("argmax_k").as_array()) c.argmax_k.push_back(static_cast<int>(v.as_int()));
    for (const Json& v : j.get("max_k").as_array()) c.max_k.push_back(static_cast<int>(v.as_int()));
    if (c.embedding.empty() || c.linear.empty() || c.argmax_k.empty() || c.max_k.empty())
        return R::failure("CATALOG_JSON: every site needs at least one option");
    return R::success(c);
}

Json Shapes::to_json() const {
    Json j = Json::object();
    j.set("n_layers", n_layers);
    j.set("d_model", d_model);
    j.set("d_ff", d_ff);
    j.set("n_head", n_head);
    j.set("vocab", vocab);
    j.set("n_rows", n_rows);
    j.set("kv_len", kv_len);
    j.set("mem_limit_bytes", static_cast<long long>(mem_limit_bytes));
    j.set("refill_bytes_per_s", refill_bytes_per_s);
    j.set("logits_bound", logits_bound);
    j.set("score_bound", score_bound);
    return j;
}

Expected<Shapes> Shapes::from_json(const Json& j) {
    using R = Expected<Shapes>;
    if (!j.is_object()) return R::failure("SHAPES_JSON: expected an object");
    Shapes s;
    auto geti = [&](const char* k, int& dst) -> std::string {
        if (!j.has(k)) return "";
        Expected<std::int64_t> v = j.get_int(k);
        if (!v.ok) return v.code;
        dst = static_cast<int>(v.value);
        return "";
    };
    std::string err;
    if (!(err = geti("n_layers", s.n_layers)).empty()) return R::failure("SHAPES_JSON: " + err);
    if (!(err = geti("d_model", s.d_model)).empty()) return R::failure("SHAPES_JSON: " + err);
    if (!(err = geti("d_ff", s.d_ff)).empty()) return R::failure("SHAPES_JSON: " + err);
    if (!(err = geti("n_head", s.n_head)).empty()) return R::failure("SHAPES_JSON: " + err);
    if (!(err = geti("vocab", s.vocab)).empty()) return R::failure("SHAPES_JSON: " + err);
    if (!(err = geti("n_rows", s.n_rows)).empty()) return R::failure("SHAPES_JSON: " + err);
    if (!(err = geti("kv_len", s.kv_len)).empty()) return R::failure("SHAPES_JSON: " + err);
    if (j.has("mem_limit_bytes")) s.mem_limit_bytes = j.get("mem_limit_bytes").as_int();
    if (j.has("refill_bytes_per_s")) s.refill_bytes_per_s = j.get("refill_bytes_per_s").as_double();
    if (j.has("logits_bound")) {
        if (!j.get("logits_bound").is_number()) return R::failure("SHAPES_JSON: logits_bound must be a number");
        s.logits_bound = j.get("logits_bound").as_double();
    }
    if (j.has("score_bound")) {
        if (!j.get("score_bound").is_number()) return R::failure("SHAPES_JSON: score_bound must be a number");
        s.score_bound = j.get("score_bound").as_double();
    }
    if (s.n_layers < 1 || s.d_model < 1 || s.d_ff < 1 || s.n_head < 1 || s.vocab < 1 || s.n_rows < 1 || s.kv_len < 1 ||
        s.d_model % s.n_head != 0)
        return R::failure("SHAPES_PARAMS: positive dimensions with n_head dividing d_model required");
    return R::success(s);
}

// ---------------------------------------------------------------------------
// plans
// ---------------------------------------------------------------------------
Json make_plan(const std::string& embedding, const std::string& linear, int argmax_k, int max_k) {
    Json p = Json::object();
    p.set("embedding", embedding);
    p.set("linear", linear);
    p.set("argmax_k", argmax_k);
    p.set("max_k", max_k);
    return p;
}

Status validate_plan(const Json& plan) {
    if (!plan.is_object()) return fail_status("PLAN_INVALID: plan must be an object");
    const std::string emb = plan.get("embedding").as_string();
    const std::string lin = plan.get("linear").as_string();
    if (emb != "wmask" && emb != "generic") return fail_status("PLAN_INVALID: embedding must be wmask or generic");
    if (lin != "wmask" && lin != "generic") return fail_status("PLAN_INVALID: linear must be wmask or generic");
    const Json* ak = plan.find("argmax_k");
    const Json* mk = plan.find("max_k");
    if (!ak || !ak->is_int() || (ak->as_int() != 2 && ak->as_int() != 4 && ak->as_int() != 8))
        return fail_status("PLAN_INVALID: argmax_k must be 2, 4 or 8");
    if (!mk || !mk->is_int() || (mk->as_int() != 2 && mk->as_int() != 4)) return fail_status("PLAN_INVALID: max_k must be 2 or 4");
    return ok_status();
}

std::string plan_key(const Json& plan) { return plan.dump(); }

bool same_plan(const Json& a, const Json& b) { return plan_key(a) == plan_key(b); }

std::vector<std::pair<std::string, std::pair<std::int64_t, std::int64_t>>> wmask_objects_for_plan(const Json& plan, const Shapes& sh) {
    std::vector<std::pair<std::string, std::pair<std::int64_t, std::int64_t>>> out;
    const std::int64_t d = sh.d_model, f = sh.d_ff, V = sh.vocab;
    if (plan.get("embedding").as_string() == "wmask") out.push_back({"wte", {V, d}});
    if (plan.get("linear").as_string() == "wmask") {
        for (int li = 0; li < sh.n_layers; ++li) {
            const std::string h = "h" + std::to_string(li) + ".";
            out.push_back({h + "qkv", {d, 3 * d}});
            out.push_back({h + "out", {d, d}});
            out.push_back({h + "fc", {d, f}});
            out.push_back({h + "proj", {f, d}});
        }
    }
    out.push_back({"lm_head", {d, V}});
    return out;
}

// ---------------------------------------------------------------------------
// evaluation
// ---------------------------------------------------------------------------
std::vector<PlanCall> plan_calls(const Json& plan, const Shapes& sh) {
    const std::string emb = plan.get("embedding").as_string();
    const std::string lin = plan.get("linear").as_string();
    const int argmax_k = static_cast<int>(plan.get("argmax_k").as_int());
    const int max_k = static_cast<int>(plan.get("max_k").as_int());
    const std::int64_t n = sh.n_rows, d = sh.d_model, f = sh.d_ff, V = sh.vocab, H = sh.n_head, T = sh.kv_len;
    const std::int64_t dh = d / H;
    const std::string lin_comp = "linear_" + lin;
    const std::string softmax_comp = "softmax_rows_k" + std::to_string(max_k);
    std::vector<PlanCall> calls;
    calls.push_back({"embedding_" + emb, n, V, d, 1});
    for (int li = 0; li < sh.n_layers; ++li) {
        calls.push_back({"layernorm_rows", n, d, d, 1});
        calls.push_back({lin_comp, n, d, 3 * d, 1});
        calls.push_back({"matmul_shared", n, dh, T, H});          // Q K^T per head
        calls.push_back({softmax_comp, H * n, T, T, 1});          // H n rows of T keys
        calls.push_back({"matmul_shared", n, T, dh, H});          // P V per head
        calls.push_back({lin_comp, n, d, d, 1});
        calls.push_back({"layernorm_rows", n, d, d, 1});
        calls.push_back({lin_comp, n, d, f, 1});
        calls.push_back({"gelu_clip12", n, f, f, 1});
        calls.push_back({lin_comp, n, f, d, 1});
    }
    calls.push_back({"layernorm_rows", 1, d, d, 1});              // final LayerNorm on the last row
    calls.push_back({"linear_wmask", 1, d, V, 1});                // LM head on the last row
    calls.push_back({"argmax_k" + std::to_string(argmax_k), 1, V, 1, 1});
    return calls;
}

std::string shape_key(const Shapes& sh) {
    return std::to_string(sh.n_layers) + "|" + std::to_string(sh.n_rows) + "|" + std::to_string(sh.kv_len) + "|" + std::to_string(sh.d_model) +
           "|" + std::to_string(sh.d_ff) + "|" + std::to_string(sh.n_head) + "|" + std::to_string(sh.vocab);
}

std::vector<Shapes> horizon_schedule(const Shapes& current, int t, int n_epochs, int prompt_len, int length_bucket) {
    std::vector<Shapes> out;
    out.push_back(current);
    const int b = std::max(1, length_bucket);
    for (int u = t + 1; u < n_epochs; ++u) {
        Shapes step = current;
        step.n_rows = 1;
        const int keys = prompt_len + u;
        step.kv_len = keys > 0 ? ((keys + b - 1) / b) * b : 0;   // ceil(keys / b) * b
        out.push_back(step);
    }
    return out;
}

Expected<std::pair<double, double>> exec_cost(const Json& plan, const Shapes& sh, const FittedCostModel& cm, const Env& env) {
    using R = Expected<std::pair<double, double>>;
    double mean = 0.0, var = 0.0;
    for (const PlanCall& call : plan_calls(plan, sh)) {
        Expected<Prediction> p = cm.predict(call.component, call.n, call.d_in, call.d_out, env);
        if (!p.ok) return R::failure(p.code);
        const double k = static_cast<double>(call.count);
        mean += k * p.value.mean_s;
        var += k * p.value.sigma_s * p.value.sigma_s;
    }
    return R::success(std::make_pair(mean, std::sqrt(var)));
}

Status kary_feasibility(const Json& plan, const Shapes& sh, const NumericContract& contract) {
    Status st = validate_plan(plan);
    if (!st.ok) return st;
    const int argmax_k = static_cast<int>(plan.get("argmax_k").as_int());
    const int max_k = static_cast<int>(plan.get("max_k").as_int());
    // softmax max over kv_len scores (no index), argmax over vocab logits (index tracked)
    Status sm = kary_magnitude_check(contract, max_k, sh.kv_len, sh.score_bound, false);
    if (!sm.ok) return fail_status(sm.code + " [site softmax max_k]");
    Status am = kary_magnitude_check(contract, argmax_k, sh.vocab, sh.logits_bound, true);
    if (!am.ok) return fail_status(am.code + " [site argmax_k]");
    return ok_status();
}

Expected<Candidate> evaluate_plan(const Json& plan, const SessionState& state, const FittedCostModel& cm, const Env& env,
                                  const Shapes& sh, const NumericContract& contract, const std::vector<Shapes>& schedule) {
    using R = Expected<Candidate>;
    Status st = validate_plan(plan);
    if (!st.ok) return R::failure(st.code);
    if (sh.n_layers < 1 || sh.d_model < 1 || sh.d_ff < 1 || sh.n_head < 1 || sh.vocab < 1 || sh.n_rows < 1 || sh.kv_len < 1 ||
        sh.d_model % sh.n_head != 0)
        return R::failure("SHAPES_PARAMS: positive dimensions with n_head dividing d_model required");
    const std::string key0 = shape_key(sh);
    if (!schedule.empty() && shape_key(schedule[0]) != key0)
        return R::failure("SCHEDULE_FIRST_STEP: the schedule starts with shape " + shape_key(schedule[0]) + ", the current epoch has " + key0);
    Candidate c;
    c.plan = plan;
    const std::int64_t n = sh.n_rows, d = sh.d_model, f = sh.d_ff, T = sh.kv_len;

    double mean = 0.0, var = 0.0;
    for (const PlanCall& call : plan_calls(plan, sh)) {
        Expected<Prediction> p = cm.predict(call.component, call.n, call.d_in, call.d_out, env);
        if (!p.ok) return R::failure(p.code);
        const double k = static_cast<double>(call.count);
        mean += k * p.value.mean_s;
        var += k * p.value.sigma_s * p.value.sigma_s;
        c.online_bytes_per_epoch += call.count * p.value.bytes;
        c.rounds_per_epoch += call.count * p.value.rounds;
        Expected<std::map<std::string, std::int64_t>> mat = material_by_type(call.component, call.n, call.d_in, call.d_out, contract);
        if (!mat.ok) return R::failure(mat.code);
        for (const auto& kv : mat.value) c.material_by_type[kv.first] += call.count * kv.second;
    }
    c.exec_s_per_epoch = mean;
    c.exec_sigma = std::sqrt(var);
    for (const auto& kv : c.material_by_type) c.material_bytes_per_epoch += kv.second;

    // per-step horizon: every scheduled step priced under the same environment, each distinct shape once
    if (!schedule.empty()) {
        std::map<std::string, std::pair<double, double>> priced;
        priced[key0] = std::make_pair(c.exec_s_per_epoch, c.exec_sigma);
        for (const Shapes& step : schedule) {
            const std::string key = shape_key(step);
            auto it = priced.find(key);
            if (it == priced.end()) {
                if (step.n_layers < 1 || step.d_model < 1 || step.d_ff < 1 || step.n_head < 1 || step.vocab < 1 || step.n_rows < 1 ||
                    step.kv_len < 1 || step.d_model % step.n_head != 0)
                    return R::failure("SHAPES_PARAMS: positive dimensions with n_head dividing d_model required (scheduled step)");
                Expected<std::pair<double, double>> ec = exec_cost(plan, step, cm, env);
                if (!ec.ok) return R::failure(ec.code);
                it = priced.emplace(key, ec.value).first;
            }
            c.exec_s_by_epoch.push_back(it->second.first);
            c.exec_sigma_by_epoch.push_back(it->second.second);
        }
    }

    // candidate-specific install: objects not installed under the current version
    double imean = 0.0, ivar = 0.0;
    std::int64_t object_bytes = 0;
    for (const auto& obj : wmask_objects_for_plan(plan, sh)) {
        object_bytes += 16 * obj.second.first * obj.second.second;   // mask share per party + public E
        if (state.object_installed(obj.first)) continue;
        Expected<Prediction> p = cm.predict("install_wmask", 1, obj.second.first, obj.second.second, env);
        if (!p.ok) return R::failure(p.code);
        c.install_objects.push_back(obj.first);
        c.install_cost_by_object[obj.first] = p.value.mean_s;
        imean += p.value.mean_s;
        ivar += p.value.sigma_s * p.value.sigma_s;
    }
    c.c_install_s = imean;
    c.c_install_sigma = std::sqrt(ivar);

    // hard filter 1 (numerics, the magnitude condition): the declared operand bounds must keep every k-ary site
    // inside the fused node bound of the contract
    Status kary = kary_feasibility(plan, sh, contract);
    if (!kary.ok) {
        c.feasible = false;
        c.reason = "KARY_MAGNITUDE";
        c.detail = kary.code;
    }

    // hard filter 2: memory peak estimate (per party): KV shares, mask shares of wmask objects, material, activations
    const std::int64_t kv_bytes = 2 * static_cast<std::int64_t>(sh.n_layers) * T * d * 8;
    const std::int64_t act_bytes = 4 * n * f * 8;
    c.mem_peak_bytes = kv_bytes + object_bytes + c.material_bytes_per_epoch + act_bytes;
    if (c.feasible && sh.mem_limit_bytes > 0 && c.mem_peak_bytes > sh.mem_limit_bytes) {
        c.feasible = false;
        c.reason = "MEMORY_INFEASIBLE";
        c.detail = "peak " + std::to_string(c.mem_peak_bytes) + " B > limit " + std::to_string(sh.mem_limit_bytes) + " B";
    }

    // hard filter 3: material feasibility against the declared stock (types absent from the stock are not checked)
    if (!state.stock.empty()) {
        for (const auto& kv : c.material_by_type) {
            auto it = state.stock.find(kv.first);
            if (it == state.stock.end()) continue;
            const std::int64_t have = it->second.second;
            if (have >= kv.second) continue;
            const std::int64_t shortfall = kv.second - have;
            if (sh.refill_bytes_per_s > 0.0) {
                c.c_wait_s += static_cast<double>(shortfall) / sh.refill_bytes_per_s;
            } else if (c.feasible) {
                c.feasible = false;
                c.reason = "INSUFFICIENT_MATERIAL";
                c.detail = kv.first + " needs " + std::to_string(kv.second) + " B, stock " + std::to_string(have) + " B";
            }
        }
    }
    return R::success(c);
}

Expected<std::vector<Candidate>> enumerate_candidates(const SessionState& state, const Catalog& catalog, const FittedCostModel& cm,
                                                      const Env& env, const Shapes& sh, const NumericContract& contract,
                                                      const std::vector<Shapes>& schedule) {
    using R = Expected<std::vector<Candidate>>;
    Status st = validate_plan(state.incumbent);
    if (!st.ok) return R::failure("STATE_INCUMBENT: " + st.code);
    if (!schedule.empty() && shape_key(schedule[0]) != shape_key(sh))
        return R::failure("SCHEDULE_FIRST_STEP: the schedule starts with shape " + shape_key(schedule[0]) + ", the current epoch has " + shape_key(sh));
    std::vector<Candidate> out;
    Expected<Candidate> keep = evaluate_plan(state.incumbent, state, cm, env, sh, contract, schedule);
    if (!keep.ok) return R::failure(keep.code);
    out.push_back(keep.value);
    for (const auto& e : catalog.embedding)
        for (const auto& l : catalog.linear)
            for (int ak : catalog.argmax_k)
                for (int mk : catalog.max_k) {
                    Json plan = make_plan(e, l, ak, mk);
                    if (same_plan(plan, state.incumbent)) continue;
                    Expected<Candidate> c = evaluate_plan(plan, state, cm, env, sh, contract, schedule);
                    if (!c.ok) return R::failure(c.code);
                    out.push_back(c.value);
                }
    return R::success(std::move(out));
}

}  // namespace highlevel
}  // namespace opencrab2
