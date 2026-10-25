// OpenCrab v2: state-aware runtime controller and baseline policies (paper, Section 4.2, objective
// and decision; python/opencrab2/highlevel/controller.py).
//
// Objective (keep included as a candidate):
//     Qhat_t(p | S_t) = c_incremental_switch(p | S_t)
//                     + sum_{j=0}^{H-1} s_j * E[T_exec(p, t+j) | public history]
//                     + c_wait_or_material(p | S_t)
// H is the remaining declared budget (public), s_j the regime survival estimate from a dwell
// model fitted on the public telemetry seen so far (no future access; s_0 = 1).
//
// Step by step: E[T_exec(p, t+j)] is entry j of the candidate's exec_s_by_epoch (the remaining
// epochs have different public shapes: the prefill processes the prompt, a decode step one row),
// a step beyond the list takes its last entry, and the sum is never H times the current epoch. A
// candidate without the lists (no schedule was given) is priced on one shape, exec_s_per_epoch for
// every step. U_diff(p) is the uncertainty of the cumulative difference Qhat(keep) - Qhat(p): with
// a_j = s_j sigma_j (sigma_j entry j of exec_sigma_by_epoch) the exec variance over the horizon is
// sum_j a_j^2 + epoch_corr ((sum_j a_j)^2 - sum_j a_j^2), the install variance is added, and keep
// and p are treated as independent (conservative). The baselines that count the remaining horizon
// without survival weights (B0, B2, B3, the oracle fallbacks) use the same per-step sum
// (horizon_exec); B1 is per epoch by definition and never reads the lists.
// Decision: p* = argmax_p (Qhat(keep) - Qhat(p) - z U_diff(p)) with the confidence multiplier z; switch iff p* != keep and
// lower_bound_gain(p*) > hysteresis_s, subject to min_residency epochs since the last switch.
// Sensing/planning already paid is sunk for the decision (not in Qhat) but charged in session totals.
//
// Dwell model (paper, Section 4.2, and the Python DwellModel): a regime change is declared when
// bandwidth or RTT moves more than env_change_rel relative to the previous sample. With n_c changes
// over n_s transitions (one per frame after the first) the hazard is h = (n_c + 1) / (n_s +
// dwell_prior), clamped to [0, 1], and the survival s_j = (1 - h)^j. A stationary observed link
// therefore lengthens the effective horizon as its age grows; stationary_survival = true (or
// dwell_prior <= 0) declares the explicit fixed-capacity assumption s_j = 1.
//
// Every Decision carries the assumption set under which the lower-bound-gain argument holds
// (a conditional result; no unconditional no-harm claim is made).
//
// Policies (same engine, catalog and calibration): B0_fixed:<plan-json>, B0_fixed_best_hindsight,
// B1_naive (per-epoch cheapest, install ignored) and delayed:<k> (the B1 rule with its replanning
// delayed by k epochs: a newly cheapest plan is adopted once it has been the cheapest k + 1 epochs
// in a row), B2_remaining_plus_install (remaining execution + candidate-specific install, no
// uncertainty, no hysteresis), B3_planner_plus_B2 (the ExactAssign planner assigns the per-epoch
// plan, the control rule is B2), B4_opencrab (the full rule) with ablations B4_no_install,
// B4_no_hyst, B4_no_unc, B5_oracle (reads the trace of true per-epoch costs, pays the real install; exact DP over
// switching sequences for <= 10 candidates, greedy hindsight beyond; an epoch the trace does not
// cover is priced on the candidate's per-step prediction), B6_zero_overhead (per-epoch true
// cheapest, install not charged: a lower bound, not a policy).
//
// Planner x controller factorial: every controller-backed policy
// (B1_naive, B2_remaining_plus_install, B3_planner_plus_B2, B4_opencrab and its ablations) accepts
// an OPTIONAL planner for the per-epoch assignment. With a planner the epoch is decided in two
// steps: the planner assigns one plan, the feasible candidate with the least cost on a
// one-operator chain, then the policy's own control rule decides between keep and that single
// challenger. The chain prices every candidate at the policy's horizon cost, as the Python
// planner chain does: Qhat without the install term over the controller's survival vector, plus
// c_install_s when the rule counts the install (for B2 the per-step sum horizon_exec plus the
// install; for B1 the current epoch's cost times H). B3_planner_plus_B2 is the static planner of
// the paper: it assigns the per-epoch execution optimum (exec_s_per_epoch), which knows neither
// the install state nor the horizon, and B2 control decides. Without a planner the controller
// ranks every candidate itself (the default for B1, B2 and B4). The planner is given either as a
// PlannerHook to make_policy or by the name suffix "+planner:<exact | ilp | greedy | argmin>"
// (ExactAssign, ILPAdapter, GreedyPerOperator over the chain, or the plain minimum);
// B3_planner_plus_B2 takes ExactAssign unless another planner is named.
//
// Tests (tests/test_decision.cpp): the install example (keep 50 / A 105 / B 21 -> B), the
// stationary cold start (installs, then no further switch), the assumption set, exhaustive
// agreement of the no-uncertainty no-hysteresis stationary rule with the brute-force minimizer
// over 3 candidates x 5 epochs x 50 random tables, min_residency, delayed:2, oracle sanity, the
// planner chain at the horizon cost, the per-step objective
// (mean, variance, the one-shape fallback, horizon_exec, B0/B1/B2/B3 and the oracle on scheduled
// candidates, CANDIDATE_SCHEDULE), and the cross-language vector file
// tests/vectors/controller_vectors.json.
#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "opencrab2/expected.hpp"
#include "opencrab2/highlevel/candidates.hpp"
#include "opencrab2/highlevel/costmodel.hpp"
#include "opencrab2/highlevel/json.hpp"

namespace opencrab2 {
namespace highlevel {

// Public telemetry observation (mirror of the Python TelemetryFrame fields the controller reads).
struct TelemetrySample {
    int seq = 0;
    double t = 0.0;
    double bandwidth_bps = 0.0;
    double rtt_s = 0.0;
    int epoch = 0;
    double cpu_load = 0.0;
    Json to_json() const;
    static Expected<TelemetrySample> from_json(const Json& j);
};

struct ControllerConfig {
    double hysteresis_s = 0.5;
    double z = 1.0;                       // uncertainty multiplier on the difference sigma
    int min_residency = 1;                // epochs a plan executes (switching epoch included) before another switch
    double dwell_prior = 5.0;             // prior mean dwell (epochs); <= 0 means stationary assumption
    double env_change_rel = 0.25;         // relative change that declares a new regime
    bool ignore_install = false;          // ablation: c_install = 0 in the decision (session still pays)
    bool use_uncertainty = true;          // ablation: U_diff = 0
    double epoch_corr = 0.0;              // share of fully correlated exec residual across epochs (0 = independent)
    bool stationary_survival = false;     // explicit fixed-capacity assumption s_j = 1
    Json to_json() const;
    static Expected<ControllerConfig> from_json(const Json& j);
};

struct Decision {
    Json chosen;                          // plan to execute next
    bool switched = false;
    double q_keep = 0.0;
    double q_best = 0.0;                  // Qhat of the chosen plan
    double lower_bound_gain = 0.0;        // of the best challenger (0 when none)
    double u_diff = 0.0;                  // of the best challenger
    std::string reason;
    std::vector<Json> candidates;         // per-candidate record (plan, q, gain, u_diff, lower bound, feasibility)
    std::vector<std::string> assumptions;
    std::string policy;                   // filled by Policy wrappers
    Json to_json() const;
};

struct QHat {
    double mean = 0.0;
    double variance = 0.0;
    double install_s = 0.0;
    double exec_sum_s = 0.0;
    double wait_s = 0.0;
};

// Qhat of one candidate over the survival-weighted horizon (survival.size() = H, s_0 = 1). With
// per-step lists on the candidate: mean = c_install (if included) + sum_j s_j exec_at(c, j) +
// c_wait_s and, with a_j = s_j exec_sigma_at(c, j), variance = sum a_j^2 + epoch_corr ((sum a_j)^2
// - sum a_j^2) + c_install_sigma^2 (if included). Without them: mean = c_install + S1 exec +
// c_wait_s and variance = exec_sigma^2 (S2 + epoch_corr (S1^2 - S2)) + c_install_sigma^2 with
// S1 = sum s_j, S2 = sum s_j^2. Plain left-to-right sums, the same figures as the Python q_hat.
// The caller checks validate_schedule(c) first (decide_with_survival does).
QHat q_hat(const Candidate& c, const SessionState& state, const std::vector<double>& survival, bool include_install,
           double epoch_corr = 0.0);

// Execution time of the next H steps without survival weights (the baselines' horizon term): the
// per-step sum of exec_s_by_epoch (a step beyond the list takes its last entry), or H * exec_s_per_epoch.
double horizon_exec(const Candidate& c, int H);

class DwellModel {
public:
    DwellModel(double dwell_prior, double env_change_rel);
    void observe(const TelemetrySample& s);
    double hazard() const;                     // (changes + 1) / (transitions + prior), in [0, 1]; 0 when stationary
    double mean_dwell() const;                 // 1 / hazard, epochs
    std::vector<double> survival(int horizon) const;   // s_j = (1 - hazard)^j
    int completed_dwells() const { return static_cast<int>(dwells_.size()); }
    int transitions() const { return transitions_; }
    int current_age() const { return age_; }
    bool stationary() const { return prior_ <= 0.0; }
    Json to_json() const;

private:
    double prior_;
    double change_rel_;
    bool have_last_ = false;
    double last_bw_ = 0.0, last_rtt_ = 0.0;
    int age_ = 0;
    int transitions_ = 0;
    std::vector<int> dwells_;
};

class StateAwareController {
public:
    explicit StateAwareController(ControllerConfig cfg = ControllerConfig());
    void observe(const TelemetrySample& s);
    std::vector<double> survival(int horizon) const;
    // One decision per epoch; updates the residency counter according to the outcome.
    Expected<Decision> decide(const SessionState& state, const std::vector<Candidate>& candidates, int epoch);
    // Same rule with an explicit survival vector (cross-language vectors, tests).
    Expected<Decision> decide_with_survival(const SessionState& state, const std::vector<Candidate>& candidates, int epoch,
                                            const std::vector<double>& survival);
    void reset_residency() { epochs_since_switch_ = 1 << 20; }
    int epochs_since_switch() const { return epochs_since_switch_; }
    const ControllerConfig& config() const { return cfg_; }
    const DwellModel& dwell() const { return dwell_; }

private:
    ControllerConfig cfg_;
    DwellModel dwell_;
    int epochs_since_switch_ = 1 << 20;
};

// True per-epoch execution costs for the oracles (exec_s[epoch][plan index]).
struct OracleTrace {
    std::vector<std::string> plan_keys;
    std::vector<std::vector<double>> exec_s;
    int index_of(const std::string& key) const;   // -1 when absent
    Json to_json() const;
    static Expected<OracleTrace> from_json(const Json& j);
};

class Policy {
public:
    virtual ~Policy() = default;
    virtual std::string name() const = 0;
    virtual void observe(const TelemetrySample&) {}
    virtual void set_trace(const OracleTrace&) {}
    virtual Expected<Decision> decide(const SessionState& state, const std::vector<Candidate>& candidates, const Env& env,
                                      int epoch) = 0;
};

// Per-epoch assignment hook of the planner x controller factorial: returns the index (into
// candidates) of the plan the planner assigns for this epoch. cost[i] is the price of candidate i
// the policy ranks by (see the header comment: the horizon cost, or exec_s_per_epoch for the
// static planner of B3). It sees public quantities only.
using PlannerHook = std::function<Expected<int>(const SessionState& state, const std::vector<Candidate>& candidates, const Env& env,
                                                const std::vector<double>& cost)>;

// Built-in hooks by name: "exact" (ExactAssign), "ilp" (ILPAdapter), "greedy" (GreedyPerOperator),
// "argmin" (no planner object). Each assigns the feasible candidate with the least cost (the
// first one in candidate order on ties) and assigns keep when no candidate is feasible. Codes:
// UNKNOWN_PLANNER; a hook fails with NO_KEEP_CANDIDATE when keep is absent and with
// PLANNER_PARAMS when cost and candidates differ in length.
Expected<PlannerHook> planner_hook(const std::string& planner_name);

std::vector<std::string> policy_names();
// name may carry the suffix "+planner:<exact|ilp|greedy|argmin>" (see the header comment).
Expected<std::unique_ptr<Policy>> make_policy(const std::string& name, const ControllerConfig& base = ControllerConfig());
// Same with an explicit planner hook (may be empty: no planner). Policies that are not backed by
// the controller (B0, B5, B6, delayed) refuse a planner with POLICY_PARAMS.
Expected<std::unique_ptr<Policy>> make_policy(const std::string& name, const ControllerConfig& base, PlannerHook planner);

// Exact minimum over switching sequences of a stationary or epoch-indexed cost table: install[p]
// is paid the first time plan p is used (unless p is the incumbent: install_incumbent), exec[t][p]
// per epoch. Returns the optimal total and the plan sequence. Used by the tests and the oracle.
struct SequenceOptimum {
    double total = 0.0;
    std::vector<int> sequence;
};
Expected<SequenceOptimum> brute_force_sequence(const std::vector<std::vector<double>>& exec, const std::vector<double>& install,
                                               int incumbent);

}  // namespace highlevel
}  // namespace opencrab2
