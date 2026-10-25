// OpenCrab v2: session state and candidate enumeration (paper, Section 4.1, session state and
// candidates; python/opencrab2/highlevel/candidates.py).
//
// State S_t = {incumbent plan, installed objects with versions, ledger stock per material type, KV
// length bucket, remaining public token budget, session ids}. enumerate_candidates(S_t, catalog)
// always includes keep (first). Each candidate carries the realization per site (the plan), its
// candidate-specific install cost c_install(p | S_t) (only objects not yet installed under the
// current version), the predicted execution cost per epoch with its sigma, the material need per
// epoch (by dealer material type), a memory peak estimate and feasibility flags.
//
// Horizon schedule. The remaining epochs do not share one public shape: the prefill processes the
// whole prompt, every later decode step one row with a longer KV bucket. Given the schedule of
// public shapes of the remaining epochs (entry j for epoch t + j; horizon_schedule builds it the
// way the Python session does), every candidate also carries exec_s_by_epoch and
// exec_sigma_by_epoch, the predicted time and standard deviation of each remaining step, which
// the controller sums step by step. exec_s_per_epoch and exec_sigma stay the values of the
// current epoch (entry 0); the feasibility checks and the per-epoch baseline use them. A shape
// that occurs in several steps is priced once per plan. Without a schedule both lists stay empty
// and the controller prices every step on the current shape.
//
// Plan (same keys as the Python dict): {"embedding": "wmask"|"generic", "linear": "wmask"|"generic",
// "argmax_k": 2|4|8, "max_k": 2|4}. Installed objects of the wmask lane: "wte" (vocab x d_model),
// "h<l>.qkv", "h<l>.out", "h<l>.fc", "h<l>.proj" per layer, and "lm_head" (always wmask, charged
// in the cold-start total). An installed object counts only when its version matches the state's
// (model_version, contract_hash); a mismatch is treated as not installed (never reused).
//
// The per-epoch execution cost of a plan sums the cost-model predictions of every component of a
// GPT-2 style block over n_rows query rows attending kv_len keys (plan_calls: the same call list
// as python candidates.plan_calls, each call with its repetition count; component sigmas combine
// as independent). Hard feasibility is separated from the objective; the filters run in the
// admission order (numerics before resources) and the first failing one names the reason:
//   KARY_MAGNITUDE         a k-ary site whose declared public operand bound (softmax
//                          scores for max_k, logits for argmax_k; Shapes::score_bound and
//                          Shapes::logits_bound, real units) does not keep the compared differences
//                          inside the fused node bound 2^(n - 2 - ceil(log2 g)) of the contract.
//                          Decided from the contract and the declared bounds only, never from
//                          secret values (costmodel.hpp kary_magnitude_check).
//   MEMORY_INFEASIBLE      the memory peak estimate exceeds a declared limit.
//   INSUFFICIENT_MATERIAL  a material shortfall against the declared stock without a refill rate;
//                          with a refill rate the shortfall becomes a wait c_wait_s = shortfall /
//                          refill_bytes_per_s instead.
// An infeasible candidate stays in the list (keep included) so that the decision record can show
// what was rejected and why; the controller never selects it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "opencrab2/expected.hpp"
#include "opencrab2/highlevel/contract.hpp"
#include "opencrab2/highlevel/costmodel.hpp"
#include "opencrab2/highlevel/json.hpp"

namespace opencrab2 {
namespace highlevel {

struct SessionState {
    Json incumbent;                                                   // plan object
    std::map<std::string, Json> installed;                            // object_id -> version {"model_version", "contract_hash"}
    std::map<std::string, std::pair<std::int64_t, std::int64_t>> stock;   // material_type -> (count, bytes) AVAILABLE
    int kv_bucket = 0;
    int remaining_epochs = 0;
    std::string session_id;
    std::string model_version;
    std::string contract_hash;

    bool object_installed(const std::string& object_id) const;       // version must match the state
    Json to_json() const;
    static Expected<SessionState> from_json(const Json& j);
};

struct Candidate {
    Json plan;
    std::vector<std::string> install_objects;                        // objects to install for this plan (not yet installed)
    std::map<std::string, double> install_cost_by_object;            // predicted seconds per object
    double c_install_s = 0.0;
    double c_install_sigma = 0.0;
    double exec_s_per_epoch = 0.0;
    double exec_sigma = 0.0;
    std::int64_t material_bytes_per_epoch = 0;
    std::int64_t mem_peak_bytes = 0;
    bool feasible = true;
    std::string reason;                                              // empty when feasible, else a stable code
    std::string detail;                                              // human-readable detail of the reason (may be empty)
    double c_wait_s = 0.0;                                           // material refill wait for the first epoch
    std::map<std::string, std::int64_t> material_by_type;            // per-epoch bytes by dealer material type
    std::int64_t online_bytes_per_epoch = 0;
    std::int64_t rounds_per_epoch = 0;
    // Per-step horizon: entry j is the predicted seconds / standard deviation of remaining step j
    // (entry 0 equals exec_s_per_epoch / exec_sigma). Both empty = one shape for every step. A
    // non-empty exec_s_by_epoch needs an exec_sigma_by_epoch of the same length (CANDIDATE_SCHEDULE).
    std::vector<double> exec_s_by_epoch;
    std::vector<double> exec_sigma_by_epoch;

    Json to_json() const;
    static Expected<Candidate> from_json(const Json& j);
};

// The per-step lists of a candidate are consistent (see Candidate): ok, or CANDIDATE_SCHEDULE.
Status validate_schedule(const Candidate& c);
// Predicted execution time and standard deviation of remaining step j: entry min(j, n - 1) of the
// per-step lists, or exec_s_per_epoch / exec_sigma when the candidate carries no schedule.
double exec_at(const Candidate& c, std::size_t j);
double exec_sigma_at(const Candidate& c, std::size_t j);

struct Catalog {
    std::vector<std::string> embedding = {"wmask", "generic"};
    std::vector<std::string> linear = {"wmask", "generic"};
    std::vector<int> argmax_k = {2, 4, 8};
    std::vector<int> max_k = {2, 4};
    static Catalog full();
    Json to_json() const;
    static Expected<Catalog> from_json(const Json& j);
};

struct Shapes {
    int n_layers = 4;
    int d_model = 768;
    int d_ff = 3072;
    int n_head = 12;
    int vocab = 50257;
    int n_rows = 1;                 // query rows of this epoch (1 for a decode step, prompt length for prefill)
    int kv_len = 32;                // keys attended (KV length bucket)
    std::int64_t mem_limit_bytes = 0;      // 0 = no declared limit
    double refill_bytes_per_s = 0.0;       // dealer refill rate (0 = no refill: a shortfall is infeasible)
    // Declared public operand bounds of the k-ary sites, in real units (|x| <= bound for every
    // compared value; a property of the model and the contract, not of a particular input).
    double logits_bound = 512.0;           // LM head logits entering argmax_k
    double score_bound = 512.0;            // attention scores entering the softmax max (max_k)
    Json to_json() const;
    static Expected<Shapes> from_json(const Json& j);
};

Json make_plan(const std::string& embedding, const std::string& linear, int argmax_k, int max_k);
Status validate_plan(const Json& plan);                               // PLAN_INVALID
std::string plan_key(const Json& plan);                               // canonical serialization
bool same_plan(const Json& a, const Json& b);

// One gadget call of an epoch: component, shape and repetition count (python candidates.plan_calls).
struct PlanCall {
    std::string component;
    std::int64_t n = 1, d_in = 1, d_out = 1;
    std::int64_t count = 1;
};
// Every gadget call of one epoch under a validated plan.
std::vector<PlanCall> plan_calls(const Json& plan, const Shapes& shapes);

// The public shape of one epoch as a key: everything plan_calls reads (n_layers, n_rows, kv_len,
// d_model, d_ff, n_head, vocab); the limits and bounds are not part of it.
std::string shape_key(const Shapes& shapes);

// Public shapes of the remaining epochs t .. n_epochs - 1 (paper, Section 4.2, objective): entry 0
// is `current` (the prefill, a recovery prefill or one decode step, as the caller built it), every
// later epoch u is one decode step over prompt_len + u keys rounded up to the length bucket (a
// bucket of 1 keeps the true key count, as the C++ session prices its current epoch). The same
// construction as the Python session's horizon_schedule.
std::vector<Shapes> horizon_schedule(const Shapes& current, int t, int n_epochs, int prompt_len, int length_bucket);

// (mean_s, sigma_s) of one epoch under a validated plan: the cost-model predictions of plan_calls
// summed, the component sigmas combined as independent (the same figures evaluate_plan stores).
Expected<std::pair<double, double>> exec_cost(const Json& plan, const Shapes& shapes, const FittedCostModel& costmodel, const Env& env);

// The magnitude condition of the fused node (paper, Section 5.3) as a feasibility filter: ok, or
// KARY_MAGNITUDE with the failing site in the detail.
Status kary_feasibility(const Json& plan, const Shapes& shapes, const NumericContract& contract = default_contract());

// wmask objects a plan needs: (object_id, (d_in, d_out)), lm_head included for every plan.
std::vector<std::pair<std::string, std::pair<std::int64_t, std::int64_t>>> wmask_objects_for_plan(const Json& plan, const Shapes& shapes);

// Per-plan evaluation (exec cost, install cost given the state, material, memory, feasibility).
// `schedule` lists the public shapes of the remaining epochs, entry j for epoch t + j; its entry 0
// must have the shape key of `shapes` (SCHEDULE_FIRST_STEP otherwise). With a schedule the
// candidate carries exec_s_by_epoch and exec_sigma_by_epoch, one entry per scheduled step priced
// under `env`; a shape that occurs in several steps is priced once per plan. Material, memory and
// the k-ary filter are checked for the current epoch only. Without a schedule both lists stay empty.
Expected<Candidate> evaluate_plan(const Json& plan, const SessionState& state, const FittedCostModel& costmodel, const Env& env,
                                  const Shapes& shapes, const NumericContract& contract = default_contract(),
                                  const std::vector<Shapes>& schedule = std::vector<Shapes>());

// Keep first, then every other plan of the catalog (schedule as for evaluate_plan).
Expected<std::vector<Candidate>> enumerate_candidates(const SessionState& state, const Catalog& catalog,
                                                      const FittedCostModel& costmodel, const Env& env, const Shapes& shapes,
                                                      const NumericContract& contract = default_contract(),
                                                      const std::vector<Shapes>& schedule = std::vector<Shapes>());

}  // namespace highlevel
}  // namespace opencrab2
