// OpenCrab v2: planner implementation (see include/opencrab2/highlevel/planner.hpp).
#include "opencrab2/highlevel/planner.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace opencrab2 {
namespace highlevel {

// ---------------------------------------------------------------------------
// JSON (de)serialization
// ---------------------------------------------------------------------------
Json Realization::to_json() const {
    Json j = Json::object();
    j.set("name", name);
    j.set("component", component);
    j.set("fixed_cost_s", fixed_cost_s);
    j.set("in_domain", in_domain);
    j.set("out_domain", out_domain);
    Json objs = Json::array();
    for (const auto& o : install_objects) objs.push_back(o);
    j.set("install_objects", objs);
    j.set("level_consumed", level_consumed);
    j.set("level_reset_to", level_reset_to);
    j.set("material_bytes", static_cast<long long>(material_bytes));
    return j;
}

Expected<Realization> Realization::from_json(const Json& j) {
    using R = Expected<Realization>;
    if (!j.is_object()) return R::failure("PLANNER_JSON: realization must be an object");
    Realization r;
    r.name = j.get("name").as_string();
    if (r.name.empty()) return R::failure("PLANNER_JSON: realization name required");
    r.component = j.get("component").as_string();
    r.fixed_cost_s = j.get("fixed_cost_s").as_double();
    r.in_domain = j.get("in_domain").as_string();
    r.out_domain = j.get("out_domain").as_string();
    if (r.in_domain.empty() || r.out_domain.empty()) return R::failure("PLANNER_JSON: realization domains required");
    for (const Json& o : j.get("install_objects").as_array()) r.install_objects.push_back(o.as_string());
    r.level_consumed = static_cast<int>(j.get("level_consumed").as_int());
    r.level_reset_to = j.has("level_reset_to") ? static_cast<int>(j.get("level_reset_to").as_int()) : -1;
    r.material_bytes = j.get("material_bytes").as_int();
    if (r.level_consumed < 0 || r.fixed_cost_s < 0) return R::failure("PLANNER_JSON: negative level or cost");
    return R::success(r);
}

Json ChainOp::to_json() const {
    Json j = Json::object();
    j.set("op_id", op_id);
    j.set("kind", kind);
    j.set("n", static_cast<long long>(n));
    j.set("d_in", static_cast<long long>(d_in));
    j.set("d_out", static_cast<long long>(d_out));
    return j;
}

Expected<ChainOp> ChainOp::from_json(const Json& j) {
    using R = Expected<ChainOp>;
    if (!j.is_object()) return R::failure("PLANNER_JSON: chain op must be an object");
    ChainOp op;
    op.op_id = j.get("op_id").as_string();
    op.kind = j.get("kind").as_string();
    if (op.kind.empty()) return R::failure("PLANNER_JSON: op kind required");
    op.n = j.has("n") ? j.get("n").as_int() : 1;
    op.d_in = j.has("d_in") ? j.get("d_in").as_int() : 1;
    op.d_out = j.has("d_out") ? j.get("d_out").as_int() : 1;
    return R::success(op);
}

Json Conversion::to_json() const {
    Json j = Json::object();
    j.set("from_domain", from_domain);
    j.set("to_domain", to_domain);
    j.set("component", component);
    j.set("fixed_cost_s", fixed_cost_s);
    j.set("level_reset_to", level_reset_to);
    return j;
}

Expected<Conversion> Conversion::from_json(const Json& j) {
    using R = Expected<Conversion>;
    if (!j.is_object()) return R::failure("PLANNER_JSON: conversion must be an object");
    Conversion c;
    c.from_domain = j.get("from_domain").as_string();
    c.to_domain = j.get("to_domain").as_string();
    if (c.from_domain.empty() || c.to_domain.empty()) return R::failure("PLANNER_JSON: conversion domains required");
    c.component = j.get("component").as_string();
    c.fixed_cost_s = j.get("fixed_cost_s").as_double();
    c.level_reset_to = j.has("level_reset_to") ? static_cast<int>(j.get("level_reset_to").as_int()) : -1;
    if (c.fixed_cost_s < 0) return R::failure("PLANNER_JSON: negative conversion cost");
    return R::success(c);
}

Json PlannerCatalog::to_json() const {
    Json j = Json::object();
    Json re = Json::object();
    for (const auto& kv : realizations) {
        Json arr = Json::array();
        for (const auto& r : kv.second) arr.push_back(r.to_json());
        re.set(kv.first, arr);
    }
    j.set("realizations", re);
    Json conv = Json::array();
    for (const auto& c : conversions) conv.push_back(c.to_json());
    j.set("conversions", conv);
    Json inst = Json::object();
    for (const auto& kv : install_cost_s) inst.set(kv.first, kv.second);
    j.set("install_cost_s", inst);
    return j;
}

Expected<PlannerCatalog> PlannerCatalog::from_json(const Json& j) {
    using R = Expected<PlannerCatalog>;
    if (!j.is_object()) return R::failure("PLANNER_JSON: catalog must be an object");
    PlannerCatalog c;
    for (const auto& kv : j.get("realizations").as_object()) {
        std::vector<Realization> rs;
        for (const Json& rj : kv.second.as_array()) {
            Expected<Realization> r = Realization::from_json(rj);
            if (!r.ok) return R::failure(r.code);
            rs.push_back(r.value);
        }
        c.realizations[kv.first] = rs;
    }
    for (const Json& cj : j.get("conversions").as_array()) {
        Expected<Conversion> cv = Conversion::from_json(cj);
        if (!cv.ok) return R::failure(cv.code);
        c.conversions.push_back(cv.value);
    }
    for (const auto& kv : j.get("install_cost_s").as_object()) c.install_cost_s[kv.first] = kv.second.as_double();
    return R::success(c);
}

Json PlanResult::to_json() const {
    Json j = Json::object();
    Json a = Json::array();
    for (const auto& s : assignment) a.push_back(s);
    j.set("assignment", a);
    j.set("cost_s", cost_s);
    j.set("end_domain", end_domain);
    j.set("end_level", end_level);
    Json inst = Json::array();
    for (const auto& s : installed_after) inst.push_back(s);
    j.set("installed_after", inst);
    j.set("states_expanded", static_cast<long long>(states_expanded));
    return j;
}

// ---------------------------------------------------------------------------
// shared pricing
// ---------------------------------------------------------------------------
Expected<double> realization_cost(const Realization& r, const ChainOp& op, const Env& env, const FittedCostModel* cm) {
    if (r.component.empty()) return Expected<double>::success(r.fixed_cost_s);
    if (!cm) return Expected<double>::failure("PLANNER_NO_COSTMODEL: component-priced realization without a cost model");
    Expected<Prediction> p = cm->predict(r.component, op.n, op.d_in, op.d_out, env);
    if (!p.ok) return Expected<double>::failure(p.code);
    return Expected<double>::success(p.value.mean_s + r.fixed_cost_s);
}

Expected<double> conversion_cost(const Conversion& c, const ChainOp& op, const Env& env, const FittedCostModel* cm) {
    if (c.component.empty()) return Expected<double>::success(c.fixed_cost_s);
    if (!cm) return Expected<double>::failure("PLANNER_NO_COSTMODEL: component-priced conversion without a cost model");
    Expected<Prediction> p = cm->predict(c.component, op.n, op.d_in, op.d_out, env);
    if (!p.ok) return Expected<double>::failure(p.code);
    return Expected<double>::success(p.value.mean_s + c.fixed_cost_s);
}

Expected<StepResult> evaluate_step(const Realization& r, const ChainOp& op, const std::string& domain, int level,
                                   const std::set<std::string>& installed, const PlannerCatalog& catalog, const Env& env,
                                   const FittedCostModel* cm) {
    using R = Expected<StepResult>;
    StepResult s;
    s.cost_s = 0.0;
    s.level = level;
    s.installed = installed;
    if (r.in_domain != domain) {
        const Conversion* found = nullptr;
        for (const Conversion& c : catalog.conversions)
            if (c.from_domain == domain && c.to_domain == r.in_domain) { found = &c; break; }
        if (!found) return R::failure("PLAN_DOMAIN_MISMATCH: no conversion " + domain + " -> " + r.in_domain);
        Expected<double> cc = conversion_cost(*found, op, env, cm);
        if (!cc.ok) return R::failure(cc.code);
        s.cost_s += cc.value;
        if (found->level_reset_to >= 0) s.level = found->level_reset_to;
    }
    if (r.level_consumed > s.level) return R::failure("PLAN_LEVEL_INFEASIBLE: " + r.name + " needs " + std::to_string(r.level_consumed) + " levels");
    s.level -= r.level_consumed;
    if (r.level_reset_to >= 0) s.level = r.level_reset_to;
    for (const std::string& obj : r.install_objects) {
        if (s.installed.count(obj)) continue;
        auto it = catalog.install_cost_s.find(obj);
        if (it == catalog.install_cost_s.end()) return R::failure("UNKNOWN_INSTALL_OBJECT: " + obj);
        s.cost_s += it->second;
        s.installed.insert(obj);
    }
    Expected<double> rc = realization_cost(r, op, env, cm);
    if (!rc.ok) return R::failure(rc.code);
    s.cost_s += rc.value;
    s.domain = r.out_domain;
    return R::success(std::move(s));
}

namespace {

const Realization* find_realization(const PlannerCatalog& catalog, const std::string& kind, const std::string& name) {
    auto it = catalog.realizations.find(kind);
    if (it == catalog.realizations.end()) return nullptr;
    for (const Realization& r : it->second)
        if (r.name == name) return &r;
    return nullptr;
}

Status check_chain(const std::vector<ChainOp>& chain, const PlannerCatalog& catalog) {
    if (chain.empty()) return fail_status("PLANNER_EMPTY_CHAIN");
    for (const ChainOp& op : chain) {
        auto it = catalog.realizations.find(op.kind);
        if (it == catalog.realizations.end() || it->second.empty()) return fail_status("PLANNER_NO_REALIZATION: " + op.kind);
    }
    return ok_status();
}

}  // namespace

Expected<PlanResult> evaluate_assignment(const std::vector<ChainOp>& chain, const PlannerCatalog& catalog, const Env& env,
                                         const FittedCostModel* cm, const PlannerState& state,
                                         const std::vector<std::string>& assignment) {
    using R = Expected<PlanResult>;
    Status st = check_chain(chain, catalog);
    if (!st.ok) return R::failure(st.code);
    if (assignment.size() != chain.size()) return R::failure("PLANNER_ASSIGNMENT_LENGTH");
    PlanResult res;
    std::string domain = state.start_domain;
    int level = state.start_level;
    std::set<std::string> installed = state.installed;
    for (std::size_t i = 0; i < chain.size(); ++i) {
        const Realization* r = find_realization(catalog, chain[i].kind, assignment[i]);
        if (!r) return R::failure("PLANNER_UNKNOWN_REALIZATION: " + assignment[i]);
        Expected<StepResult> step = evaluate_step(*r, chain[i], domain, level, installed, catalog, env, cm);
        if (!step.ok) return R::failure(step.code);
        res.cost_s += step.value.cost_s;
        domain = step.value.domain;
        level = step.value.level;
        installed = step.value.installed;
        res.assignment.push_back(assignment[i]);
    }
    if (!state.required_end_domain.empty() && domain != state.required_end_domain) return R::failure("PLAN_END_DOMAIN_MISMATCH");
    res.end_domain = domain;
    res.end_level = level;
    res.installed_after = installed;
    return R::success(res);
}

// ---------------------------------------------------------------------------
// ExactAssign
// ---------------------------------------------------------------------------
namespace {

struct DpState {
    std::string domain;
    int level;
    std::set<std::string> installed;
    double cost;
    std::vector<std::string> assignment;
};

std::string state_key(const DpState& s, bool typed) {
    if (!typed) return s.domain;
    std::string k = s.domain + "|" + std::to_string(s.level) + "|";
    for (const auto& o : s.installed) k += o + ",";
    return k;
}

}  // namespace

Expected<PlanResult> ExactAssign::plan(const std::vector<ChainOp>& chain, const PlannerCatalog& catalog, const Env& env,
                                       const FittedCostModel* cm, const PlannerState& state) const {
    using R = Expected<PlanResult>;
    Status st = check_chain(chain, catalog);
    if (!st.ok) return R::failure(st.code);
    std::vector<DpState> frontier;
    frontier.push_back({state.start_domain, state.start_level, state.installed, 0.0, {}});
    std::int64_t expanded = 0;
    std::string last_error = "NO_FEASIBLE_PLAN";
    for (const ChainOp& op : chain) {
        std::map<std::string, DpState> next;
        for (const DpState& s : frontier) {
            for (const Realization& r : catalog.realizations.at(op.kind)) {
                ++expanded;
                Expected<StepResult> step = evaluate_step(r, op, s.domain, s.level, s.installed, catalog, env, cm);
                if (!step.ok) {
                    if (step.code_prefix() == "UNKNOWN_INSTALL_OBJECT" || step.code_prefix() == "PLANNER_NO_COSTMODEL" ||
                        step.code_prefix() == "UNKNOWN_COMPONENT" || step.code_prefix() == "COSTMODEL_NOT_FITTED")
                        return R::failure(step.code);
                    last_error = step.code;
                    continue;
                }
                DpState ns{step.value.domain, step.value.level, step.value.installed, s.cost + step.value.cost_s, s.assignment};
                ns.assignment.push_back(r.name);
                const std::string key = state_key(ns, typed_);
                auto it = next.find(key);
                if (it == next.end() || ns.cost < it->second.cost) next[key] = std::move(ns);
            }
        }
        if (next.empty()) return R::failure("NO_FEASIBLE_PLAN: " + last_error);
        frontier.clear();
        for (auto& kv : next) frontier.push_back(std::move(kv.second));
    }
    const DpState* best = nullptr;
    for (const DpState& s : frontier) {
        if (!state.required_end_domain.empty() && s.domain != state.required_end_domain) continue;
        if (!best || s.cost < best->cost) best = &s;
    }
    if (!best) return R::failure("NO_FEASIBLE_PLAN: PLAN_END_DOMAIN_MISMATCH");
    PlanResult res;
    res.assignment = best->assignment;
    res.cost_s = best->cost;
    res.end_domain = best->domain;
    res.end_level = best->level;
    res.installed_after = best->installed;
    res.states_expanded = expanded;
    return R::success(res);
}

// ---------------------------------------------------------------------------
// ILPAdapter (branch and bound)
// ---------------------------------------------------------------------------
namespace {

struct BnB {
    const std::vector<ChainOp>& chain;
    const PlannerCatalog& catalog;
    const Env& env;
    const FittedCostModel* cm;
    const PlannerState& state;
    std::vector<double> suffix_min;             // lower bound of the remaining operators
    std::vector<std::vector<std::pair<double, const Realization*>>> ordered;   // realizations sorted by base cost
    double best_cost = std::numeric_limits<double>::infinity();
    std::vector<std::string> best_assignment;
    std::string end_domain;
    int end_level = 0;
    std::set<std::string> end_installed;
    std::int64_t nodes = 0;
    std::string fatal;

    void dfs(std::size_t i, const std::string& domain, int level, const std::set<std::string>& installed, double cost,
             std::vector<std::string>& assignment) {
        if (!fatal.empty()) return;
        if (i == chain.size()) {
            if (!state.required_end_domain.empty() && domain != state.required_end_domain) return;
            if (cost < best_cost) {
                best_cost = cost;
                best_assignment = assignment;
                end_domain = domain;
                end_level = level;
                end_installed = installed;
            }
            return;
        }
        if (cost + suffix_min[i] >= best_cost) return;
        for (const auto& pr : ordered[i]) {
            ++nodes;
            Expected<StepResult> step = evaluate_step(*pr.second, chain[i], domain, level, installed, catalog, env, cm);
            if (!step.ok) {
                if (step.code_prefix() == "UNKNOWN_INSTALL_OBJECT") fatal = step.code;
                continue;
            }
            const double nc = cost + step.value.cost_s;
            if (nc + suffix_min[i + 1] >= best_cost) continue;
            assignment.push_back(pr.second->name);
            dfs(i + 1, step.value.domain, step.value.level, step.value.installed, nc, assignment);
            assignment.pop_back();
        }
    }
};

}  // namespace

Expected<PlanResult> ILPAdapter::plan(const std::vector<ChainOp>& chain, const PlannerCatalog& catalog, const Env& env,
                                      const FittedCostModel* cm, const PlannerState& state) const {
    using R = Expected<PlanResult>;
    Status st = check_chain(chain, catalog);
    if (!st.ok) return R::failure(st.code);
    BnB b{chain, catalog, env, cm, state, {}, {}, std::numeric_limits<double>::infinity(), {}, "", 0, {}, 0, ""};
    b.ordered.resize(chain.size());
    std::vector<double> per_op_min(chain.size(), 0.0);
    for (std::size_t i = 0; i < chain.size(); ++i) {
        double mn = std::numeric_limits<double>::infinity();
        for (const Realization& r : catalog.realizations.at(chain[i].kind)) {
            Expected<double> c = realization_cost(r, chain[i], env, cm);
            if (!c.ok) return R::failure(c.code);
            b.ordered[i].push_back({c.value, &r});
            mn = std::min(mn, c.value);
        }
        std::stable_sort(b.ordered[i].begin(), b.ordered[i].end(),
                         [](const std::pair<double, const Realization*>& x, const std::pair<double, const Realization*>& y) { return x.first < y.first; });
        per_op_min[i] = mn;
    }
    b.suffix_min.assign(chain.size() + 1, 0.0);
    for (std::size_t i = chain.size(); i-- > 0;) b.suffix_min[i] = b.suffix_min[i + 1] + per_op_min[i];
    std::vector<std::string> assignment;
    b.dfs(0, state.start_domain, state.start_level, state.installed, 0.0, assignment);
    if (!b.fatal.empty()) return R::failure(b.fatal);
    if (!std::isfinite(b.best_cost)) return R::failure("NO_FEASIBLE_PLAN");
    PlanResult res;
    res.assignment = b.best_assignment;
    res.cost_s = b.best_cost;
    res.end_domain = b.end_domain;
    res.end_level = b.end_level;
    res.installed_after = b.end_installed;
    res.states_expanded = b.nodes;
    return R::success(res);
}

// ---------------------------------------------------------------------------
// Greedy
// ---------------------------------------------------------------------------
Expected<PlanResult> GreedyPerOperator::plan(const std::vector<ChainOp>& chain, const PlannerCatalog& catalog, const Env& env,
                                             const FittedCostModel* cm, const PlannerState& state) const {
    using R = Expected<PlanResult>;
    Status st = check_chain(chain, catalog);
    if (!st.ok) return R::failure(st.code);
    PlanResult res;
    std::string domain = state.start_domain;
    int level = state.start_level;
    std::set<std::string> installed = state.installed;
    for (const ChainOp& op : chain) {
        const Realization* best = nullptr;
        StepResult best_step;
        for (const Realization& r : catalog.realizations.at(op.kind)) {
            ++res.states_expanded;
            Expected<StepResult> step = evaluate_step(r, op, domain, level, installed, catalog, env, cm);
            if (!step.ok) continue;
            if (!best || step.value.cost_s < best_step.cost_s) {
                best = &r;
                best_step = step.value;
            }
        }
        if (!best) return R::failure("NO_FEASIBLE_PLAN: greedy dead end at " + op.op_id);
        res.assignment.push_back(best->name);
        res.cost_s += best_step.cost_s;
        domain = best_step.domain;
        level = best_step.level;
        installed = best_step.installed;
    }
    if (!state.required_end_domain.empty() && domain != state.required_end_domain) return R::failure("NO_FEASIBLE_PLAN: PLAN_END_DOMAIN_MISMATCH");
    res.end_domain = domain;
    res.end_level = level;
    res.installed_after = installed;
    return R::success(res);
}

// ---------------------------------------------------------------------------
// brute force
// ---------------------------------------------------------------------------
Expected<PlanResult> brute_force(const std::vector<ChainOp>& chain, const PlannerCatalog& catalog, const Env& env,
                                 const FittedCostModel* cm, const PlannerState& state) {
    using R = Expected<PlanResult>;
    Status st = check_chain(chain, catalog);
    if (!st.ok) return R::failure(st.code);
    double total = 1.0;
    std::vector<std::size_t> counts;
    for (const ChainOp& op : chain) {
        counts.push_back(catalog.realizations.at(op.kind).size());
        total *= static_cast<double>(counts.back());
    }
    if (total > 1e7) return R::failure("BRUTE_FORCE_TOO_LARGE");
    std::vector<std::size_t> idx(chain.size(), 0);
    PlanResult best;
    best.cost_s = std::numeric_limits<double>::infinity();
    std::int64_t evaluated = 0;
    while (true) {
        std::vector<std::string> assignment;
        for (std::size_t i = 0; i < chain.size(); ++i) assignment.push_back(catalog.realizations.at(chain[i].kind)[idx[i]].name);
        ++evaluated;
        Expected<PlanResult> r = evaluate_assignment(chain, catalog, env, cm, state, assignment);
        if (r.ok && r.value.cost_s < best.cost_s) best = r.value;
        std::size_t pos = chain.size();
        while (pos-- > 0) {
            if (++idx[pos] < counts[pos]) break;
            idx[pos] = 0;
            if (pos == 0) { pos = static_cast<std::size_t>(-1); break; }
        }
        if (pos == static_cast<std::size_t>(-1)) break;
    }
    if (!std::isfinite(best.cost_s)) return R::failure("NO_FEASIBLE_PLAN");
    best.states_expanded = evaluated;
    return R::success(best);
}

}  // namespace highlevel
}  // namespace opencrab2
