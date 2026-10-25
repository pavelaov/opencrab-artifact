// OpenCrab v2: controller and policies implementation (see include/opencrab2/highlevel/controller.hpp).
#include "opencrab2/highlevel/controller.hpp"

#include "opencrab2/highlevel/planner.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <set>

namespace opencrab2 {
namespace highlevel {

// ---------------------------------------------------------------------------
// TelemetrySample / ControllerConfig / Decision
// ---------------------------------------------------------------------------
Json TelemetrySample::to_json() const {
    Json j = Json::object();
    j.set("seq", seq);
    j.set("t", t);
    j.set("bandwidth_bps", bandwidth_bps);
    j.set("rtt_s", rtt_s);
    j.set("epoch", epoch);
    j.set("cpu_load", cpu_load);
    return j;
}

Expected<TelemetrySample> TelemetrySample::from_json(const Json& j) {
    using R = Expected<TelemetrySample>;
    if (!j.is_object()) return R::failure("TELEMETRY_JSON: expected an object");
    TelemetrySample s;
    Expected<double> bw = j.get_double("bandwidth_bps");
    if (!bw.ok) return R::failure("TELEMETRY_JSON: " + bw.code);
    Expected<double> rtt = j.get_double("rtt_s");
    if (!rtt.ok) return R::failure("TELEMETRY_JSON: " + rtt.code);
    s.bandwidth_bps = bw.value;
    s.rtt_s = rtt.value;
    s.seq = static_cast<int>(j.get("seq").as_int());
    s.t = j.get("t").as_double();
    s.epoch = static_cast<int>(j.get("epoch").as_int());
    s.cpu_load = j.get("cpu_load").as_double();
    return R::success(s);
}

Json ControllerConfig::to_json() const {
    Json j = Json::object();
    j.set("hysteresis_s", hysteresis_s);
    j.set("z", z);
    j.set("min_residency", min_residency);
    j.set("dwell_prior", dwell_prior);
    j.set("env_change_rel", env_change_rel);
    j.set("ignore_install", ignore_install);
    j.set("use_uncertainty", use_uncertainty);
    j.set("epoch_corr", epoch_corr);
    j.set("stationary_survival", stationary_survival);
    return j;
}

Expected<ControllerConfig> ControllerConfig::from_json(const Json& j) {
    using R = Expected<ControllerConfig>;
    if (!j.is_object()) return R::failure("CONTROLLER_JSON: expected an object");
    ControllerConfig c;
    if (j.has("hysteresis_s")) c.hysteresis_s = j.get("hysteresis_s").as_double();
    if (j.has("z")) c.z = j.get("z").as_double();
    if (j.has("min_residency")) c.min_residency = static_cast<int>(j.get("min_residency").as_int());
    if (j.has("dwell_prior")) c.dwell_prior = j.get("dwell_prior").as_double();
    if (j.has("env_change_rel")) c.env_change_rel = j.get("env_change_rel").as_double();
    if (j.has("ignore_install")) c.ignore_install = j.get("ignore_install").as_bool();
    if (j.has("use_uncertainty")) c.use_uncertainty = j.get("use_uncertainty").as_bool();
    if (j.has("epoch_corr")) c.epoch_corr = j.get("epoch_corr").as_double();
    if (j.has("stationary_survival")) c.stationary_survival = j.get("stationary_survival").as_bool();
    if (c.hysteresis_s < 0 || c.z < 0 || c.min_residency < 0 || c.env_change_rel < 0 || c.epoch_corr < 0 || c.epoch_corr > 1)
        return R::failure("CONTROLLER_PARAMS: non-negative parameters required, epoch_corr in [0, 1]");
    return R::success(c);
}

Json Decision::to_json() const {
    Json j = Json::object();
    j.set("chosen", chosen);
    j.set("switched", switched);
    j.set("q_keep", q_keep);
    j.set("q_best", q_best);
    j.set("lower_bound_gain", lower_bound_gain);
    j.set("u_diff", u_diff);
    j.set("reason", reason);
    Json c = Json::array();
    for (const Json& x : candidates) c.push_back(x);
    j.set("candidates", c);
    Json a = Json::array();
    for (const std::string& s : assumptions) a.push_back(s);
    j.set("assumptions", a);
    j.set("policy", policy);
    return j;
}

// ---------------------------------------------------------------------------
// Qhat
// ---------------------------------------------------------------------------
QHat q_hat(const Candidate& c, const SessionState& /*state*/, const std::vector<double>& survival, bool include_install,
           double epoch_corr) {
    QHat q;
    double var_exec = 0.0;
    if (!c.exec_s_by_epoch.empty()) {
        // per-step horizon: step j costs exec_at(c, j) with standard deviation exec_sigma_at(c, j)
        double ex = 0.0, a1 = 0.0, a2 = 0.0;
        for (std::size_t j = 0; j < survival.size(); ++j) {
            const double s = survival[j];
            ex += s * exec_at(c, j);
            const double a = s * exec_sigma_at(c, j);
            a1 += a;
            a2 += a * a;
        }
        q.exec_sum_s = ex;
        var_exec = a2 + epoch_corr * (a1 * a1 - a2);
    } else {
        // one shape: every step costs exec_s_per_epoch
        double s1 = 0.0, s2 = 0.0;
        for (double s : survival) {
            s1 += s;
            s2 += s * s;
        }
        q.exec_sum_s = s1 * c.exec_s_per_epoch;
        const double sig2 = c.exec_sigma * c.exec_sigma;
        var_exec = sig2 * (s2 + epoch_corr * (s1 * s1 - s2));
    }
    q.install_s = include_install ? c.c_install_s : 0.0;
    q.wait_s = c.c_wait_s;
    q.mean = q.install_s + q.exec_sum_s + q.wait_s;
    q.variance = var_exec + (include_install ? c.c_install_sigma * c.c_install_sigma : 0.0);
    return q;
}

double horizon_exec(const Candidate& c, int H) {
    if (c.exec_s_by_epoch.empty()) return static_cast<double>(H) * c.exec_s_per_epoch;
    double total = 0.0;
    for (int j = 0; j < H; ++j) total += exec_at(c, static_cast<std::size_t>(j));
    return total;
}

// ---------------------------------------------------------------------------
// DwellModel
// ---------------------------------------------------------------------------
DwellModel::DwellModel(double dwell_prior, double env_change_rel) : prior_(dwell_prior), change_rel_(env_change_rel) {}

void DwellModel::observe(const TelemetrySample& s) {
    if (!have_last_) {
        have_last_ = true;
        last_bw_ = s.bandwidth_bps;
        last_rtt_ = s.rtt_s;
        age_ = 1;
        return;
    }
    const double rel_bw = std::fabs(s.bandwidth_bps - last_bw_) / std::max(std::fabs(last_bw_), 1e-9);
    const double rel_rtt = std::fabs(s.rtt_s - last_rtt_) / std::max(std::fabs(last_rtt_), 1e-12);
    ++transitions_;
    if (std::max(rel_bw, rel_rtt) > change_rel_) {
        dwells_.push_back(age_);
        age_ = 1;
    } else {
        ++age_;
    }
    last_bw_ = s.bandwidth_bps;
    last_rtt_ = s.rtt_s;
}

double DwellModel::hazard() const {
    if (stationary()) return 0.0;
    const double h = (static_cast<double>(dwells_.size()) + 1.0) / (static_cast<double>(transitions_) + prior_);
    return std::min(1.0, std::max(0.0, h));
}

double DwellModel::mean_dwell() const {
    if (stationary()) return std::numeric_limits<double>::infinity();
    return 1.0 / hazard();
}

std::vector<double> DwellModel::survival(int horizon) const {
    std::vector<double> s(static_cast<std::size_t>(std::max(horizon, 0)), 1.0);
    if (stationary() || s.empty()) return s;
    // (1 - h)^j through pow, as the Python survival computes it
    const double keep = 1.0 - hazard();
    for (std::size_t j = 0; j < s.size(); ++j) s[j] = std::pow(keep, static_cast<double>(j));
    return s;
}

Json DwellModel::to_json() const {
    Json j = Json::object();
    j.set("prior", prior_);
    j.set("env_change_rel", change_rel_);
    j.set("completed_dwells", static_cast<int>(dwells_.size()));
    j.set("transitions", transitions_);
    j.set("current_age", age_);
    j.set("hazard", hazard());
    j.set("mean_dwell", stationary() ? Json("stationary") : Json(mean_dwell()));
    return j;
}

// ---------------------------------------------------------------------------
// StateAwareController
// ---------------------------------------------------------------------------
StateAwareController::StateAwareController(ControllerConfig cfg)
    : cfg_(cfg), dwell_(cfg.stationary_survival ? 0.0 : cfg.dwell_prior, cfg.env_change_rel) {}

void StateAwareController::observe(const TelemetrySample& s) { dwell_.observe(s); }

std::vector<double> StateAwareController::survival(int horizon) const { return dwell_.survival(horizon); }

Expected<Decision> StateAwareController::decide(const SessionState& state, const std::vector<Candidate>& candidates, int epoch) {
    const int H = std::max(state.remaining_epochs, 1);
    return decide_with_survival(state, candidates, epoch, survival(H));
}

Expected<Decision> StateAwareController::decide_with_survival(const SessionState& state, const std::vector<Candidate>& candidates,
                                                              int epoch, const std::vector<double>& survival) {
    using R = Expected<Decision>;
    (void)epoch;
    if (candidates.empty()) return R::failure("NO_CANDIDATES");
    if (survival.empty()) return R::failure("CONTROLLER_PARAMS: empty survival vector");
    int keep_idx = -1;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        Status sched = validate_schedule(candidates[i]);
        if (!sched.ok) return R::failure(sched.code);
        if (keep_idx < 0 && same_plan(candidates[i].plan, state.incumbent)) keep_idx = static_cast<int>(i);
    }
    if (keep_idx < 0) return R::failure("NO_KEEP_CANDIDATE");
    const bool include_install = !cfg_.ignore_install;
    const Candidate& keep = candidates[static_cast<std::size_t>(keep_idx)];
    const QHat qk = q_hat(keep, state, survival, include_install, cfg_.epoch_corr);

    Decision d;
    d.q_keep = qk.mean;
    int best = -1;
    double best_lb = -std::numeric_limits<double>::infinity();
    double best_gain = 0.0, best_u = 0.0, best_q = 0.0;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const Candidate& c = candidates[i];
        const QHat q = q_hat(c, state, survival, include_install, cfg_.epoch_corr);
        const bool is_keep = static_cast<int>(i) == keep_idx;
        const double gain = qk.mean - q.mean;
        // U_diff(p) is the standard deviation of the cumulative difference Qhat(keep) - Qhat(p); the
        // confidence multiplier z enters only the lower bound (same convention as the Python controller,
        // so the controller_vectors.json fields compare directly).
        const double u = (is_keep || !cfg_.use_uncertainty) ? 0.0 : std::sqrt(std::max(qk.variance + q.variance, 0.0));
        const double lb = gain - cfg_.z * u;
        Json rec = Json::object();
        rec.set("plan", c.plan);
        rec.set("is_keep", is_keep);
        rec.set("feasible", c.feasible);
        rec.set("reason", c.reason);
        rec.set("q_mean", q.mean);
        rec.set("q_variance", q.variance);
        rec.set("q_install_s", q.install_s);
        rec.set("q_exec_sum_s", q.exec_sum_s);
        rec.set("q_wait_s", q.wait_s);
        rec.set("gain", gain);
        rec.set("u_diff", u);
        rec.set("lower_bound_gain", lb);
        Json objs = Json::array();
        for (const auto& o : c.install_objects) objs.push_back(o);
        rec.set("install_objects", objs);
        d.candidates.push_back(rec);
        if (is_keep || !c.feasible) continue;
        if (lb > best_lb) {
            best_lb = lb;
            best = static_cast<int>(i);
            best_gain = gain;
            best_u = u;
            best_q = q.mean;
        }
    }

    d.chosen = keep.plan;
    d.q_best = qk.mean;
    d.switched = false;
    if (best >= 0) {
        d.lower_bound_gain = best_lb;
        d.u_diff = best_u;
        if (best_lb > cfg_.hysteresis_s && best_lb > 0.0) {
            if (epochs_since_switch_ < cfg_.min_residency) {
                d.reason = "MIN_RESIDENCY_HOLD";
            } else {
                d.switched = true;
                d.chosen = candidates[static_cast<std::size_t>(best)].plan;
                d.q_best = best_q;
                d.reason = "LOWER_BOUND_GAIN_EXCEEDS_HYSTERESIS";
            }
        } else if (best_gain > 0.0) {
            d.reason = "GAIN_BELOW_UNCERTAINTY_OR_HYSTERESIS";
        } else {
            d.reason = "INCUMBENT_OPTIMAL";
        }
    } else {
        d.reason = "INCUMBENT_OPTIMAL";
    }

    d.assumptions.push_back("PUBLIC_TELEMETRY_ONLY: costs predicted from public shapes, telemetry and declared state");
    d.assumptions.push_back("HORIZON_DECLARED_BUDGET: H = " + std::to_string(survival.size()) + " remaining public epochs");
    if (dwell_.stationary() || cfg_.stationary_survival)
        d.assumptions.push_back("SURVIVAL_STATIONARY: explicit fixed-capacity assumption s_j = 1");
    else
        d.assumptions.push_back("SURVIVAL_GEOMETRIC_DWELL: mean dwell " + format_double_repr(dwell_.mean_dwell()) + " epochs from " +
                                std::to_string(dwell_.completed_dwells()) + " completed dwells and a prior of " +
                                format_double_repr(cfg_.dwell_prior));
    d.assumptions.push_back(include_install ? "INSTALL_COST_INCLUDED: candidate-specific c_install(p | S_t)"
                                            : "INSTALL_COST_IGNORED: ablation, the session still pays the real install");
    d.assumptions.push_back(cfg_.use_uncertainty
                                ? "UNCERTAINTY_CUMULATIVE: U_diff = sqrt(var(keep) + var(p)) with per-epoch variances summed over the horizon, lower bound = gain - z U_diff, epoch_corr = " +
                                      format_double_repr(cfg_.epoch_corr)
                                : "UNCERTAINTY_IGNORED: ablation, U_diff = 0");
    d.assumptions.push_back("COST_INTERVALS_SIMULTANEOUS: the conditional result holds only if the true costs lie in the declared intervals");
    d.assumptions.push_back("NO_UNCONDITIONAL_NO_HARM: arbitrary future changes can defeat the horizon prediction");
    d.assumptions.push_back("SENSING_PLANNING_SUNK: already paid sensing and planning are excluded from Qhat and charged in session totals");

    // The counter is the number of epochs the incumbent will have executed when the next decision is
    // taken: the switching epoch counts as one, every kept epoch adds one. A plan chosen at epoch t0 can
    // be replaced at the first epoch t >= t0 + min_residency.
    if (d.switched) epochs_since_switch_ = 1;
    else epochs_since_switch_ = std::min(epochs_since_switch_ + 1, 1 << 20);
    return R::success(std::move(d));
}

// ---------------------------------------------------------------------------
// OracleTrace
// ---------------------------------------------------------------------------
int OracleTrace::index_of(const std::string& key) const {
    for (std::size_t i = 0; i < plan_keys.size(); ++i)
        if (plan_keys[i] == key) return static_cast<int>(i);
    return -1;
}

Json OracleTrace::to_json() const {
    Json j = Json::object();
    Json keys = Json::array();
    for (const auto& k : plan_keys) keys.push_back(k);
    j.set("plan_keys", keys);
    Json rows = Json::array();
    for (const auto& r : exec_s) {
        Json row = Json::array();
        for (double v : r) row.push_back(v);
        rows.push_back(row);
    }
    j.set("exec_s", rows);
    return j;
}

Expected<OracleTrace> OracleTrace::from_json(const Json& j) {
    using R = Expected<OracleTrace>;
    if (!j.is_object()) return R::failure("TRACE_JSON: expected an object");
    OracleTrace t;
    for (const Json& k : j.get("plan_keys").as_array()) t.plan_keys.push_back(k.as_string());
    for (const Json& row : j.get("exec_s").as_array()) {
        std::vector<double> r;
        for (const Json& v : row.as_array()) r.push_back(v.as_double());
        if (r.size() != t.plan_keys.size()) return R::failure("TRACE_JSON: row length must match plan_keys");
        t.exec_s.push_back(r);
    }
    return R::success(t);
}

// ---------------------------------------------------------------------------
// brute force over switching sequences
// ---------------------------------------------------------------------------
Expected<SequenceOptimum> brute_force_sequence(const std::vector<std::vector<double>>& exec, const std::vector<double>& install,
                                               int incumbent) {
    using R = Expected<SequenceOptimum>;
    const int H = static_cast<int>(exec.size());
    const int C = static_cast<int>(install.size());
    if (H < 1 || C < 1) return R::failure("BRUTE_FORCE_SHAPE: need at least one epoch and one plan");
    if (incumbent < 0 || incumbent >= C) return R::failure("BRUTE_FORCE_SHAPE: incumbent out of range");
    for (const auto& row : exec)
        if (static_cast<int>(row.size()) != C) return R::failure("BRUTE_FORCE_SHAPE: ragged exec table");
    double total_seq = 1.0;
    for (int t = 0; t < H; ++t) total_seq *= C;
    if (total_seq > 2e7) return R::failure("BRUTE_FORCE_TOO_LARGE");
    std::vector<int> seq(static_cast<std::size_t>(H), 0);
    SequenceOptimum best;
    best.total = std::numeric_limits<double>::infinity();
    while (true) {
        double cost = 0.0;
        std::vector<bool> used(static_cast<std::size_t>(C), false);
        used[static_cast<std::size_t>(incumbent)] = true;
        for (int t = 0; t < H; ++t) {
            const int p = seq[static_cast<std::size_t>(t)];
            if (!used[static_cast<std::size_t>(p)]) {
                cost += install[static_cast<std::size_t>(p)];
                used[static_cast<std::size_t>(p)] = true;
            }
            cost += exec[static_cast<std::size_t>(t)][static_cast<std::size_t>(p)];
        }
        if (cost < best.total) {
            best.total = cost;
            best.sequence = seq;
        }
        int pos = H - 1;
        while (pos >= 0) {
            if (++seq[static_cast<std::size_t>(pos)] < C) break;
            seq[static_cast<std::size_t>(pos)] = 0;
            --pos;
        }
        if (pos < 0) break;
    }
    return R::success(best);
}

// ---------------------------------------------------------------------------
// Policies
// ---------------------------------------------------------------------------
namespace {

// The controller behind the Policy interface, with the optional planner of the planner x
// controller factorial: the planner assigns the plan of the epoch, the controller rule (B1, B2 or
// the full rule, depending on the configuration) decides between keep and that single challenger.
// The planner ranks the candidates at the rule's horizon cost (Qhat without the install term over
// the controller's survival vector, plus the install when the rule counts it); static_assignment
// (B3) ranks them by the current epoch's execution cost instead.
// per_epoch (B1): the rule and every reported figure use the current epoch's cost alone, so the
// per-step lists of the candidates are dropped before the controller prices them.
class ControllerPolicy : public Policy {
public:
    ControllerPolicy(std::string name, ControllerConfig cfg, PlannerHook planner = PlannerHook(), std::string planner_name = "",
                     bool per_epoch = false, bool static_assignment = false)
        : name_(std::move(name)), ctl_(cfg), planner_(std::move(planner)), planner_name_(std::move(planner_name)), per_epoch_(per_epoch),
          static_assignment_(static_assignment) {}
    std::string name() const override { return name_; }
    void observe(const TelemetrySample& s) override { ctl_.observe(s); }
    Expected<Decision> decide(const SessionState& state, const std::vector<Candidate>& all, const Env& env, int epoch) override {
        std::vector<Candidate> flat;
        if (per_epoch_) {
            flat = all;
            for (Candidate& c : flat) {
                c.exec_s_by_epoch.clear();
                c.exec_sigma_by_epoch.clear();
            }
        }
        const std::vector<Candidate>& candidates = per_epoch_ ? flat : all;
        if (!planner_) {
            Expected<Decision> d = ctl_.decide(state, candidates, epoch);
            if (d.ok) d.value.policy = name_;
            return d;
        }
        using R = Expected<Decision>;
        if (candidates.empty()) return R::failure("NO_CANDIDATES");
        int keep = -1;
        for (std::size_t i = 0; i < candidates.size(); ++i)
            if (same_plan(candidates[i].plan, state.incumbent)) { keep = static_cast<int>(i); break; }
        if (keep < 0) return R::failure("NO_KEEP_CANDIDATE");
        std::vector<double> cost(candidates.size(), 0.0);
        if (static_assignment_) {
            for (std::size_t i = 0; i < candidates.size(); ++i) cost[i] = candidates[i].exec_s_per_epoch;
        } else {
            const std::vector<double> survival = ctl_.survival(std::max(state.remaining_epochs, 1));
            const bool include_install = !ctl_.config().ignore_install;
            for (std::size_t i = 0; i < candidates.size(); ++i) {
                Status sched = validate_schedule(candidates[i]);
                if (!sched.ok) return R::failure(sched.code);
                cost[i] = q_hat(candidates[i], state, survival, false, ctl_.config().epoch_corr).mean +
                          (include_install ? candidates[i].c_install_s : 0.0);
            }
        }
        Expected<int> assigned = planner_(state, candidates, env, cost);
        if (!assigned.ok) return R::failure("PLANNER_FAILED: " + assigned.code);
        if (assigned.value < 0 || assigned.value >= static_cast<int>(candidates.size()))
            return R::failure("PLANNER_FAILED: assignment index out of range");
        const int target = assigned.value;
        // An infeasible assignment is never executed: the epoch then keeps the incumbent.
        std::vector<Candidate> restricted;
        restricted.push_back(candidates[static_cast<std::size_t>(keep)]);
        if (target != keep && candidates[static_cast<std::size_t>(target)].feasible)
            restricted.push_back(candidates[static_cast<std::size_t>(target)]);
        Expected<Decision> d = ctl_.decide(state, restricted, epoch);
        if (!d.ok) return d;
        d.value.policy = name_;
        for (Json& rec : d.value.candidates) rec.set("planner_assigned", !rec.get("is_keep").as_bool() || target == keep);
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            if (static_cast<int>(i) == keep || (static_cast<int>(i) == target && candidates[i].feasible)) continue;
            Json rec = Json::object();
            rec.set("plan", candidates[i].plan);
            rec.set("is_keep", false);
            rec.set("feasible", candidates[i].feasible);
            rec.set("reason", candidates[i].reason);
            rec.set("exec_s_per_epoch", candidates[i].exec_s_per_epoch);
            rec.set("c_install_s", candidates[i].c_install_s);
            rec.set("planner_cost", cost[i]);
            rec.set("planner_assigned", false);
            rec.set("excluded", "NOT_PLANNER_ASSIGNMENT");
            d.value.candidates.push_back(rec);
        }
        d.value.assumptions.push_back("PLANNER_ASSIGNMENT: planner " + (planner_name_.empty() ? std::string("custom") : planner_name_) +
                                      (static_assignment_ ? " assigns the per-epoch execution optimum over the feasible candidates"
                                                          : " assigns the feasible candidate with the least horizon cost (Qhat without "
                                                            "the install term, plus the install when the rule counts it)") +
                                      "; the control rule decides between keep and that single challenger");
        return d;
    }

private:
    std::string name_;
    StateAwareController ctl_;
    PlannerHook planner_;
    std::string planner_name_;
    bool per_epoch_;
    bool static_assignment_;
};

Decision base_record(const SessionState& state, const std::vector<Candidate>& candidates, const std::string& policy);

// delayed:<k> (paper, Section 7.1: B1 with its replanning delayed): the per-epoch cheapest feasible
// plan, install ignored, as B1, but a newly cheapest plan is adopted only once it has been the
// cheapest k + 1 epochs in a row (the streak counts the epochs with the same cheapest plan, the
// incumbent included). The rule of the Python NaivePolicy with a delay.
class DelayedPolicy : public Policy {
public:
    explicit DelayedPolicy(int k) : k_(k) {}
    std::string name() const override { return "delayed:" + std::to_string(k_); }
    Expected<Decision> decide(const SessionState& state, const std::vector<Candidate>& candidates, const Env&, int) override {
        int keep = -1, best = -1;
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            if (keep < 0 && same_plan(candidates[i].plan, state.incumbent)) keep = static_cast<int>(i);
            if (!candidates[i].feasible) continue;
            if (best < 0 || candidates[i].exec_s_per_epoch < candidates[static_cast<std::size_t>(best)].exec_s_per_epoch) best = static_cast<int>(i);
        }
        if (keep < 0) return Expected<Decision>::failure("NO_KEEP_CANDIDATE");
        if (best < 0) best = keep;
        const Candidate& kc = candidates[static_cast<std::size_t>(keep)];
        const Candidate& bc = candidates[static_cast<std::size_t>(best)];
        const std::string key = plan_key(bc.plan);
        streak_ = (key == streak_key_) ? streak_ + 1 : 1;
        streak_key_ = key;
        const double H = static_cast<double>(std::max(state.remaining_epochs, 1));
        Decision d = base_record(state, candidates, name());
        d.q_keep = H * kc.exec_s_per_epoch;
        d.q_best = H * bc.exec_s_per_epoch;
        d.assumptions.push_back("INSTALL_COST_IGNORED: per-epoch cheapest execution, the session still pays the real install");
        d.assumptions.push_back("DELAYED_REPLAN: a newly cheapest plan is adopted after " + std::to_string(k_ + 1) + " epochs in a row");
        if (best == keep || same_plan(bc.plan, kc.plan)) {
            d.q_best = d.q_keep;
            d.reason = "ALREADY_CHEAPEST";
            return Expected<Decision>::success(d);
        }
        d.lower_bound_gain = d.q_keep - d.q_best;
        if (streak_ <= k_) {
            d.reason = "DELAYED_HOLD: " + std::to_string(streak_) + "/" + std::to_string(k_);
            return Expected<Decision>::success(d);
        }
        d.switched = true;
        d.chosen = bc.plan;
        d.reason = "DELAYED_REPLAN: k=" + std::to_string(k_);
        return Expected<Decision>::success(d);
    }

private:
    int k_;
    std::string streak_key_;
    int streak_ = 0;
};

Decision base_record(const SessionState& state, const std::vector<Candidate>& candidates, const std::string& policy) {
    Decision d;
    d.policy = policy;
    d.chosen = state.incumbent;
    const int H = std::max(state.remaining_epochs, 1);
    for (const Candidate& c : candidates) {
        Json rec = Json::object();
        rec.set("plan", c.plan);
        rec.set("is_keep", same_plan(c.plan, state.incumbent));
        rec.set("feasible", c.feasible);
        rec.set("reason", c.reason);
        rec.set("c_install_s", c.c_install_s);
        rec.set("exec_s_per_epoch", c.exec_s_per_epoch);
        rec.set("exec_horizon_s", horizon_exec(c, H));   // the per-step sum over the remaining H epochs
        d.candidates.push_back(rec);
    }
    return d;
}

class FixedPolicy : public Policy {
public:
    explicit FixedPolicy(Json plan) : plan_(std::move(plan)) {}
    std::string name() const override { return "B0_fixed:" + plan_key(plan_); }
    Expected<Decision> decide(const SessionState& state, const std::vector<Candidate>& candidates, const Env&, int) override {
        Decision d = base_record(state, candidates, name());
        d.assumptions.push_back("FIXED_PLAN: no adaptation; the fixed plan is installed at the first epoch when needed");
        const Candidate* keep = nullptr;
        const Candidate* target = nullptr;
        for (const Candidate& c : candidates) {
            if (same_plan(c.plan, state.incumbent)) keep = &c;
            if (same_plan(c.plan, plan_)) target = &c;
        }
        if (!keep) return Expected<Decision>::failure("NO_KEEP_CANDIDATE");
        const int H = std::max(state.remaining_epochs, 1);
        d.q_keep = keep->c_install_s + horizon_exec(*keep, H);
        if (!target || !target->feasible) {
            d.reason = "FIXED_PLAN_NOT_AVAILABLE";
            d.q_best = d.q_keep;
            return Expected<Decision>::success(d);
        }
        d.q_best = target->c_install_s + horizon_exec(*target, H);
        d.chosen = plan_;
        d.switched = !same_plan(plan_, state.incumbent);
        d.reason = d.switched ? "FIXED_PLAN_INSTALL" : "FIXED_PLAN_KEEP";
        return Expected<Decision>::success(d);
    }

private:
    Json plan_;
};

// Oracle helpers: true exec cost of candidate c at epoch e, which is step j of the remaining horizon;
// an epoch or plan the trace does not cover falls back to the candidate's per-step prediction.
double true_exec(const OracleTrace& tr, const Candidate& c, int epoch, int step) {
    const int idx = tr.index_of(plan_key(c.plan));
    if (idx < 0 || epoch < 0 || epoch >= static_cast<int>(tr.exec_s.size())) return exec_at(c, static_cast<std::size_t>(std::max(step, 0)));
    return tr.exec_s[static_cast<std::size_t>(epoch)][static_cast<std::size_t>(idx)];
}

class HindsightFixedPolicy : public Policy {
public:
    std::string name() const override { return "B0_fixed_best_hindsight"; }
    void set_trace(const OracleTrace& t) override { trace_ = t; have_trace_ = true; }
    Expected<Decision> decide(const SessionState& state, const std::vector<Candidate>& candidates, const Env&, int epoch) override {
        if (!have_trace_) return Expected<Decision>::failure("ORACLE_NEEDS_TRACE");
        if (!chosen_once_) {
            const int H = std::max(state.remaining_epochs, 1);
            double best = std::numeric_limits<double>::infinity();
            for (const Candidate& c : candidates) {
                if (!c.feasible) continue;
                double total = c.c_install_s;
                for (int j = 0; j < H; ++j) total += true_exec(trace_, c, epoch + j, j);
                if (total < best) { best = total; fixed_ = c.plan; }
            }
            if (!std::isfinite(best)) return Expected<Decision>::failure("NO_FEASIBLE_CANDIDATE");
            chosen_once_ = true;
        }
        FixedPolicy fp(fixed_);
        Expected<Decision> d = fp.decide(state, candidates, Env(), epoch);
        if (d.ok) {
            d.value.policy = name();
            d.value.assumptions.push_back("HINDSIGHT: the fixed plan minimizes the true total over the trace (reads the future)");
        }
        return d;
    }

private:
    OracleTrace trace_;
    bool have_trace_ = false;
    bool chosen_once_ = false;
    Json fixed_;
};

class OraclePolicy : public Policy {
public:
    std::string name() const override { return "B5_oracle"; }
    void set_trace(const OracleTrace& t) override { trace_ = t; have_trace_ = true; }
    Expected<Decision> decide(const SessionState& state, const std::vector<Candidate>& candidates, const Env&, int epoch) override {
        if (!have_trace_) return Expected<Decision>::failure("ORACLE_NEEDS_TRACE");
        Decision d = base_record(state, candidates, name());
        d.assumptions.push_back("ORACLE_READS_TRACE: true future costs are read from the trace; real install is paid");
        const int H = std::max(state.remaining_epochs, 1);
        std::vector<int> feas;
        int keep = -1;
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            if (same_plan(candidates[i].plan, state.incumbent)) keep = static_cast<int>(i);
            if (candidates[i].feasible || same_plan(candidates[i].plan, state.incumbent)) feas.push_back(static_cast<int>(i));
        }
        if (keep < 0) return Expected<Decision>::failure("NO_KEEP_CANDIDATE");
        int first = keep;
        double best_total = std::numeric_limits<double>::infinity();
        if (feas.size() <= 10) {
            // exact DP over (epoch, used set, current plan); install objects are unioned across used plans
            const int C = static_cast<int>(feas.size());
            int keep_pos = 0;
            for (int i = 0; i < C; ++i)
                if (feas[static_cast<std::size_t>(i)] == keep) keep_pos = i;
            auto install_cost = [&](int mask, int p) {
                const Candidate& c = candidates[static_cast<std::size_t>(feas[static_cast<std::size_t>(p)])];
                double cost = 0.0;
                for (const auto& kv : c.install_cost_by_object) {
                    bool covered = false;
                    for (int q = 0; q < C && !covered; ++q) {
                        if (!(mask & (1 << q))) continue;
                        const Candidate& cq = candidates[static_cast<std::size_t>(feas[static_cast<std::size_t>(q)])];
                        for (const auto& o : cq.install_objects)
                            if (o == kv.first) { covered = true; break; }
                    }
                    if (!covered) cost += kv.second;
                }
                return cost;
            };
            const int M = 1 << C;
            // V[t][mask][cur]
            std::vector<std::vector<std::vector<double>>> V(static_cast<std::size_t>(H + 1),
                std::vector<std::vector<double>>(static_cast<std::size_t>(M), std::vector<double>(static_cast<std::size_t>(C), 0.0)));
            std::vector<std::vector<std::vector<int>>> A(static_cast<std::size_t>(H),
                std::vector<std::vector<int>>(static_cast<std::size_t>(M), std::vector<int>(static_cast<std::size_t>(C), 0)));
            for (int t = H - 1; t >= 0; --t) {
                for (int mask = 0; mask < M; ++mask) {
                    for (int cur = 0; cur < C; ++cur) {
                        double bestv = std::numeric_limits<double>::infinity();
                        int besta = cur;
                        for (int p = 0; p < C; ++p) {
                            const int nmask = mask | (1 << p);
                            const double inst = (mask & (1 << p)) ? 0.0 : install_cost(mask, p);
                            const double v = inst + true_exec(trace_, candidates[static_cast<std::size_t>(feas[static_cast<std::size_t>(p)])], epoch + t, t) +
                                             V[static_cast<std::size_t>(t + 1)][static_cast<std::size_t>(nmask)][static_cast<std::size_t>(p)];
                            if (v < bestv - 1e-12 || (std::fabs(v - bestv) <= 1e-12 && p == cur)) { bestv = v; besta = p; }
                        }
                        V[static_cast<std::size_t>(t)][static_cast<std::size_t>(mask)][static_cast<std::size_t>(cur)] = bestv;
                        A[static_cast<std::size_t>(t)][static_cast<std::size_t>(mask)][static_cast<std::size_t>(cur)] = besta;
                    }
                }
            }
            const int mask0 = 1 << keep_pos;
            best_total = V[0][static_cast<std::size_t>(mask0)][static_cast<std::size_t>(keep_pos)];
            first = feas[static_cast<std::size_t>(A[0][static_cast<std::size_t>(mask0)][static_cast<std::size_t>(keep_pos)])];
            d.assumptions.push_back("ORACLE_EXACT_DP: exact over switching sequences (candidates <= 10)");
        } else {
            for (int i : feas) {
                const Candidate& c = candidates[static_cast<std::size_t>(i)];
                double total = c.c_install_s;
                for (int j = 0; j < H; ++j) total += true_exec(trace_, c, epoch + j, j);
                if (total < best_total) { best_total = total; first = i; }
            }
            d.assumptions.push_back("ORACLE_GREEDY_HINDSIGHT: one switch at a time (candidates > 10)");
        }
        double keep_total = candidates[static_cast<std::size_t>(keep)].c_install_s;
        for (int j = 0; j < H; ++j) keep_total += true_exec(trace_, candidates[static_cast<std::size_t>(keep)], epoch + j, j);
        d.q_keep = keep_total;
        d.q_best = best_total;
        d.lower_bound_gain = keep_total - best_total;
        d.chosen = candidates[static_cast<std::size_t>(first)].plan;
        d.switched = first != keep;
        d.reason = d.switched ? "ORACLE_SWITCH" : "ORACLE_KEEP";
        return Expected<Decision>::success(d);
    }

private:
    OracleTrace trace_;
    bool have_trace_ = false;
};

class ZeroOverheadPolicy : public Policy {
public:
    std::string name() const override { return "B6_zero_overhead"; }
    void set_trace(const OracleTrace& t) override { trace_ = t; have_trace_ = true; }
    Expected<Decision> decide(const SessionState& state, const std::vector<Candidate>& candidates, const Env&, int epoch) override {
        if (!have_trace_) return Expected<Decision>::failure("ORACLE_NEEDS_TRACE");
        Decision d = base_record(state, candidates, name());
        d.assumptions.push_back("ZERO_OVERHEAD_LOWER_BOUND: per-epoch true cheapest plan, install not charged (not a policy)");
        int keep = -1, best = -1;
        double bestv = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            if (same_plan(candidates[i].plan, state.incumbent)) keep = static_cast<int>(i);
            if (!candidates[i].feasible) continue;
            const double v = true_exec(trace_, candidates[i], epoch, 0);
            if (v < bestv) { bestv = v; best = static_cast<int>(i); }
        }
        if (keep < 0) return Expected<Decision>::failure("NO_KEEP_CANDIDATE");
        if (best < 0) best = keep;
        d.q_keep = true_exec(trace_, candidates[static_cast<std::size_t>(keep)], epoch, 0);
        d.q_best = bestv;
        d.lower_bound_gain = d.q_keep - d.q_best;
        d.chosen = candidates[static_cast<std::size_t>(best)].plan;
        d.switched = best != keep;
        d.reason = d.switched ? "ZERO_OVERHEAD_SWITCH" : "ZERO_OVERHEAD_KEEP";
        return Expected<Decision>::success(d);
    }

private:
    OracleTrace trace_;
    bool have_trace_ = false;
};

}  // namespace

// ---------------------------------------------------------------------------
// planner hooks (planner x controller factorial)
// ---------------------------------------------------------------------------
namespace {

int keep_index(const SessionState& state, const std::vector<Candidate>& candidates) {
    for (std::size_t i = 0; i < candidates.size(); ++i)
        if (same_plan(candidates[i].plan, state.incumbent)) return static_cast<int>(i);
    return -1;
}

// One-operator chain: the realizations are the feasible candidates priced by cost[i] (the
// construction of the single-operator chain of the Python planner hook). The realization name is
// the candidate index, so equal plans can never be confused.
template <class Planner>
Expected<int> assign_with(const Planner& planner, const SessionState& state, const std::vector<Candidate>& candidates, const Env& env,
                          const std::vector<double>& cost) {
    using R = Expected<int>;
    const int keep = keep_index(state, candidates);
    if (keep < 0) return R::failure("NO_KEEP_CANDIDATE");
    if (cost.size() != candidates.size()) return R::failure("PLANNER_PARAMS: one cost per candidate required");
    PlannerCatalog cat;
    std::vector<Realization> rs;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (!candidates[i].feasible) continue;
        Realization r;
        r.name = std::to_string(i);
        r.fixed_cost_s = cost[i];
        r.in_domain = "A64";
        r.out_domain = "A64";
        rs.push_back(r);
    }
    if (rs.empty()) return R::success(keep);   // nothing feasible: the assignment is the incumbent
    cat.realizations["plan"] = rs;
    std::vector<ChainOp> chain;
    ChainOp op;
    op.op_id = "plan";
    op.kind = "plan";
    chain.push_back(op);
    PlannerState st;
    Expected<PlanResult> res = planner.plan(chain, cat, env, nullptr, st);
    if (!res.ok) return R::failure(res.code);
    if (res.value.assignment.size() != 1) return R::failure("PLANNER_ASSIGNMENT_SHAPE");
    const std::string& name = res.value.assignment[0];
    char* end = nullptr;
    const long idx = std::strtol(name.c_str(), &end, 10);
    if (name.empty() || *end != '\0' || idx < 0 || idx >= static_cast<long>(candidates.size())) return R::failure("PLANNER_ASSIGNMENT_NAME");
    // tie rule shared by every hook: the first feasible candidate with the minimal cost
    const double best = cost[static_cast<std::size_t>(idx)];
    for (std::size_t i = 0; i < candidates.size(); ++i)
        if (candidates[i].feasible && cost[i] == best) return R::success(static_cast<int>(i));
    return R::success(static_cast<int>(idx));
}

Expected<int> assign_argmin(const SessionState& state, const std::vector<Candidate>& candidates, const Env&, const std::vector<double>& cost) {
    using R = Expected<int>;
    const int keep = keep_index(state, candidates);
    if (keep < 0) return R::failure("NO_KEEP_CANDIDATE");
    if (cost.size() != candidates.size()) return R::failure("PLANNER_PARAMS: one cost per candidate required");
    int best = -1;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (!candidates[i].feasible) continue;
        if (best < 0 || cost[i] < cost[static_cast<std::size_t>(best)]) best = static_cast<int>(i);
    }
    return R::success(best < 0 ? keep : best);
}

// "<policy>+planner:<name>" -> (policy, planner name); planner name empty when there is no suffix.
void split_planner_suffix(const std::string& name, std::string& policy, std::string& planner) {
    const std::string marker = "+planner:";
    const std::string::size_type at = name.rfind(marker);
    // a B0_fixed:<plan-json> name is never split (its JSON cannot contain the marker, but be explicit)
    if (at == std::string::npos || name.rfind("B0_fixed:", 0) == 0) {
        policy = name;
        planner.clear();
        return;
    }
    policy = name.substr(0, at);
    planner = name.substr(at + marker.size());
}

}  // namespace

Expected<PlannerHook> planner_hook(const std::string& planner_name) {
    using R = Expected<PlannerHook>;
    using Costs = std::vector<double>;
    if (planner_name == "exact")
        return R::success([](const SessionState& s, const std::vector<Candidate>& c, const Env& e, const Costs& k) { return assign_with(ExactAssign(true), s, c, e, k); });
    if (planner_name == "ilp")
        return R::success([](const SessionState& s, const std::vector<Candidate>& c, const Env& e, const Costs& k) { return assign_with(ILPAdapter(), s, c, e, k); });
    if (planner_name == "greedy")
        return R::success([](const SessionState& s, const std::vector<Candidate>& c, const Env& e, const Costs& k) { return assign_with(GreedyPerOperator(), s, c, e, k); });
    if (planner_name == "argmin") return R::success(PlannerHook(assign_argmin));
    return R::failure("UNKNOWN_PLANNER: " + planner_name);
}

std::vector<std::string> policy_names() {
    return {"B0_fixed:<plan-json>", "B0_fixed_best_hindsight", "B1_naive", "B2_remaining_plus_install", "B3_planner_plus_B2",
            "B4_opencrab", "B4_no_install", "B4_no_hyst", "B4_no_unc", "delayed:<k>", "B5_oracle", "B6_zero_overhead"};
}

namespace {

Expected<std::unique_ptr<Policy>> make_policy_impl(const std::string& full_name, const std::string& name, const ControllerConfig& base,
                                                   PlannerHook planner, const std::string& planner_name) {
    using R = Expected<std::unique_ptr<Policy>>;
    const bool has_planner = static_cast<bool>(planner);
    auto controller = [&](const ControllerConfig& c, bool per_epoch = false) {
        return R::success(std::unique_ptr<Policy>(new ControllerPolicy(full_name, c, planner, planner_name, per_epoch)));
    };
    if (name == "B1_naive") {
        // per-epoch cheapest execution, install ignored: the per-step lists are not read
        ControllerConfig c = base;
        c.ignore_install = true;
        c.use_uncertainty = false;
        c.hysteresis_s = 0.0;
        c.min_residency = 0;
        c.stationary_survival = true;
        return controller(c, true);
    }
    if (name == "B2_remaining_plus_install" || name == "B3_planner_plus_B2") {
        ControllerConfig c = base;
        c.use_uncertainty = false;
        c.hysteresis_s = 0.0;
        c.min_residency = 0;
        c.stationary_survival = true;
        if (name == "B3_planner_plus_B2") {
            // B3 is B2 control on the static per-epoch assignment: ExactAssign unless another planner is given
            PlannerHook hook = planner;
            std::string hook_name = planner_name;
            if (!has_planner) {
                Expected<PlannerHook> h = planner_hook("exact");
                if (!h.ok) return R::failure(h.code);
                hook = h.value;
                hook_name = "exact";
            }
            return R::success(std::unique_ptr<Policy>(new ControllerPolicy(full_name, c, hook, hook_name, false, true)));
        }
        return controller(c);
    }
    if (name == "B4_opencrab") return controller(base);
    if (name == "B4_no_install") {
        ControllerConfig c = base;
        c.ignore_install = true;
        return controller(c);
    }
    if (name == "B4_no_hyst") {
        ControllerConfig c = base;
        c.hysteresis_s = 0.0;
        c.min_residency = 0;
        return controller(c);
    }
    if (name == "B4_no_unc") {
        ControllerConfig c = base;
        c.use_uncertainty = false;
        return controller(c);
    }
    // policies below are not backed by the controller: a planner has no meaning for them
    if (has_planner) return R::failure("POLICY_PARAMS: policy " + name + " takes no planner");
    if (name.rfind("B0_fixed:", 0) == 0) {
        Expected<Json> plan = Json::parse(name.substr(9));
        if (!plan.ok) return R::failure("POLICY_PLAN_JSON: " + plan.code);
        Status st = validate_plan(plan.value);
        if (!st.ok) return R::failure(st.code);
        return R::success(std::unique_ptr<Policy>(new FixedPolicy(plan.value)));
    }
    if (name == "B0_fixed_best_hindsight") return R::success(std::unique_ptr<Policy>(new HindsightFixedPolicy()));
    if (name.rfind("delayed:", 0) == 0) {
        const int k = std::atoi(name.c_str() + 8);
        if (k < 1) return R::failure("POLICY_PARAMS: delayed:<k> needs k >= 1");
        return R::success(std::unique_ptr<Policy>(new DelayedPolicy(k)));
    }
    if (name == "B5_oracle") return R::success(std::unique_ptr<Policy>(new OraclePolicy()));
    if (name == "B6_zero_overhead") return R::success(std::unique_ptr<Policy>(new ZeroOverheadPolicy()));
    return R::failure("UNKNOWN_POLICY: " + name);
}

}  // namespace

Expected<std::unique_ptr<Policy>> make_policy(const std::string& name, const ControllerConfig& base) {
    using R = Expected<std::unique_ptr<Policy>>;
    std::string policy, planner;
    split_planner_suffix(name, policy, planner);
    if (planner.empty()) {
        if (policy != name) return R::failure("UNKNOWN_PLANNER: empty planner name");
        return make_policy_impl(name, policy, base, PlannerHook(), "");
    }
    Expected<PlannerHook> h = planner_hook(planner);
    if (!h.ok) return R::failure(h.code);
    return make_policy_impl(name, policy, base, h.value, planner);
}

Expected<std::unique_ptr<Policy>> make_policy(const std::string& name, const ControllerConfig& base, PlannerHook planner) {
    using R = Expected<std::unique_ptr<Policy>>;
    std::string policy, suffix;
    split_planner_suffix(name, policy, suffix);
    if (!suffix.empty() || policy != name) return R::failure("POLICY_PARAMS: give the planner either as a hook or as a name suffix, not both");
    return make_policy_impl(name, policy, base, std::move(planner), "custom");
}

}  // namespace highlevel
}  // namespace opencrab2
