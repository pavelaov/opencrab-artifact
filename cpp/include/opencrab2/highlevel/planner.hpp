// OpenCrab v2: assignment planners over a serialized operator chain (paper, Section 4.4).
//
// A chain is a list of operators; the catalog lists, per operator kind, the admitted realizations
// with their input/output domain, cost (a cost-model component priced on the operator shape, or a
// fixed cost), the installed objects they require, and the HE level they consume or reset to.
// Conversions between domains carry their own cost. Every planner prices a transition with the
// same evaluate_step rule (the shared admission gate in miniature: domain match or conversion,
// level feasibility, candidate-specific install).
//
//   ExactAssign   Pareto/Markov dynamic program whose state after operator i is the typed triple
//                 (ending domain, installed object set, level tag). Exact for a serialized chain, a
//                 fixed finite catalog and no global resource constraint. Complexity
//                 O(n * |S| * P * (1 + C)) with |S| <= |D| * 2^|O| * (L_max + 1) states, P
//                 realizations per operator and C conversions. With typed_state = false the state
//                 collapses to the ending domain only, the O(n P^2) rule, which discards the only
//                 level-feasible prefix of the typed-state test.
//   ILPAdapter    branch and bound over the same catalog (depth-first over operators, bound =
//                 partial cost + sum of the cheapest realization of every remaining operator, valid
//                 because conversions and installs are non-negative). A faithful re-implementation
//                 of the assignment formulation of Silph/NetAdapt style planners on this catalog,
//                 not a rerun of those systems. Exact, exponential worst case.
//   GreedyPerOperator  cheapest feasible next step; not exact.
//   brute_force   enumeration of every assignment (tests only; refuses more than 10^7).
//
// Tests (tests/test_decision.cpp): ExactAssign == ILPAdapter == brute_force on random chains with
// n <= 6, the typed state (finds the plan the domain-only state misses), greedy feasibility.
#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "opencrab2/expected.hpp"
#include "opencrab2/highlevel/costmodel.hpp"
#include "opencrab2/highlevel/json.hpp"

namespace opencrab2 {
namespace highlevel {

struct Realization {
    std::string name;                        // e.g. "linear.wmask"
    std::string component;                   // cost-model component (empty: use fixed_cost_s)
    double fixed_cost_s = 0.0;
    std::string in_domain;                   // required input domain
    std::string out_domain;                  // produced domain
    std::vector<std::string> install_objects;
    int level_consumed = 0;                  // HE levels consumed (0 for MPC/FSS lanes)
    int level_reset_to = -1;                 // >= 0: output level is reset to this value (fresh encryption)
    std::int64_t material_bytes = 0;
    Json to_json() const;
    static Expected<Realization> from_json(const Json& j);
};

struct ChainOp {
    std::string op_id;
    std::string kind;                        // catalog key
    std::int64_t n = 1, d_in = 1, d_out = 1; // shape used to price component-based realizations
    Json to_json() const;
    static Expected<ChainOp> from_json(const Json& j);
};

struct Conversion {
    std::string from_domain;
    std::string to_domain;
    std::string component;                   // optional cost-model component
    double fixed_cost_s = 0.0;
    int level_reset_to = -1;                 // >= 0: the conversion produces a fresh level
    Json to_json() const;
    static Expected<Conversion> from_json(const Json& j);
};

struct PlannerCatalog {
    std::map<std::string, std::vector<Realization>> realizations;   // by operator kind
    std::vector<Conversion> conversions;
    std::map<std::string, double> install_cost_s;                   // per object
    Json to_json() const;
    static Expected<PlannerCatalog> from_json(const Json& j);
};

struct PlannerState {
    std::string start_domain = "A64";
    int start_level = 0;
    std::set<std::string> installed;
    std::string required_end_domain;         // empty: any
};

struct PlanResult {
    std::vector<std::string> assignment;     // realization name per operator
    double cost_s = 0.0;
    std::string end_domain;
    int end_level = 0;
    std::set<std::string> installed_after;
    std::int64_t states_expanded = 0;        // work counter (DP states or search nodes)
    Json to_json() const;
};

// Price one realization on one operator (component via the cost model, or the fixed cost).
Expected<double> realization_cost(const Realization& r, const ChainOp& op, const Env& env, const FittedCostModel* costmodel);
Expected<double> conversion_cost(const Conversion& c, const ChainOp& op, const Env& env, const FittedCostModel* costmodel);

// Shared transition rule: returns the incremental cost and the new typed state, or a failure code
// (PLAN_DOMAIN_MISMATCH, PLAN_LEVEL_INFEASIBLE, UNKNOWN_INSTALL_OBJECT).
struct StepResult {
    double cost_s = 0.0;
    std::string domain;
    int level = 0;
    std::set<std::string> installed;
};
Expected<StepResult> evaluate_step(const Realization& r, const ChainOp& op, const std::string& domain, int level,
                                   const std::set<std::string>& installed, const PlannerCatalog& catalog, const Env& env,
                                   const FittedCostModel* costmodel);

// Price a full assignment (names per operator) with the shared rule.
Expected<PlanResult> evaluate_assignment(const std::vector<ChainOp>& chain, const PlannerCatalog& catalog, const Env& env,
                                         const FittedCostModel* costmodel, const PlannerState& state,
                                         const std::vector<std::string>& assignment);

class ExactAssign {
public:
    explicit ExactAssign(bool typed_state = true) : typed_(typed_state) {}
    Expected<PlanResult> plan(const std::vector<ChainOp>& chain, const PlannerCatalog& catalog, const Env& env,
                              const FittedCostModel* costmodel, const PlannerState& state) const;

private:
    bool typed_;
};

class ILPAdapter {
public:
    Expected<PlanResult> plan(const std::vector<ChainOp>& chain, const PlannerCatalog& catalog, const Env& env,
                              const FittedCostModel* costmodel, const PlannerState& state) const;
};

class GreedyPerOperator {
public:
    Expected<PlanResult> plan(const std::vector<ChainOp>& chain, const PlannerCatalog& catalog, const Env& env,
                              const FittedCostModel* costmodel, const PlannerState& state) const;
};

Expected<PlanResult> brute_force(const std::vector<ChainOp>& chain, const PlannerCatalog& catalog, const Env& env,
                                 const FittedCostModel* costmodel, const PlannerState& state);

}  // namespace highlevel
}  // namespace opencrab2
