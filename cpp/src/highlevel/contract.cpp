// OpenCrab v2: numeric contract implementation (see include/opencrab2/highlevel/contract.hpp).
#include "opencrab2/highlevel/contract.hpp"

#include <cmath>
#include <cstdlib>
#include <set>

namespace opencrab2 {
namespace highlevel {

namespace {

const char* const kFields[] = {
    "ring_bits", "frac_bits", "logical_bits", "dcf_domain_bits", "name", "gelu", "silu",
    "exp_order", "exp_r", "exp_Z", "exp_extra_bits", "recip_K", "recip_degree", "recip_newton",
    "rsqrt_j0", "rsqrt_m", "rsqrt_degree", "rsqrt_newton", "layernorm_extra_bits", "argmax_tie",
    "trunc_products", "trunc_square_ln", "activation_semantics"};

}  // namespace

double NumericContract::compare_bound_real() const {
    return std::ldexp(1.0, dcf_domain_bits - 1 - frac_bits);
}

Expected<int> NumericContract::softmax_extra_bits(int n) const {
    if (n < 1) return Expected<int>::failure("CONTRACT_PARAMS: softmax row length must be >= 1");
    int bits = 0;
    unsigned long long v = static_cast<unsigned long long>(n - 1);
    while (v) { ++bits; v >>= 1; }
    return Expected<int>::success(bits < 10 ? bits : 10);
}

int NumericContract::gelu_degree() const {
    // "gelu_clip12" -> 12; unknown names fall back to 12 (the only admitted contract).
    std::string digits;
    for (char c : gelu) if (c >= '0' && c <= '9') digits.push_back(c);
    if (digits.empty()) return 12;
    return std::atoi(digits.c_str());
}

Json NumericContract::to_json() const {
    Json j = Json::object();
    j.set("ring_bits", ring_bits);
    j.set("frac_bits", frac_bits);
    j.set("logical_bits", logical_bits);
    j.set("dcf_domain_bits", dcf_domain_bits);
    j.set("name", name);
    j.set("gelu", gelu);
    j.set("silu", silu);
    j.set("exp_order", exp_order);
    j.set("exp_r", exp_r);
    j.set("exp_Z", exp_Z);
    j.set("exp_extra_bits", exp_extra_bits);
    j.set("recip_K", recip_K);
    j.set("recip_degree", recip_degree);
    j.set("recip_newton", recip_newton);
    j.set("rsqrt_j0", rsqrt_j0);
    j.set("rsqrt_m", rsqrt_m);
    j.set("rsqrt_degree", rsqrt_degree);
    j.set("rsqrt_newton", rsqrt_newton);
    j.set("layernorm_extra_bits", layernorm_extra_bits);
    j.set("argmax_tie", argmax_tie);
    j.set("trunc_products", trunc_products);
    j.set("trunc_square_ln", trunc_square_ln);
    j.set("activation_semantics", activation_semantics);
    return j;
}

std::string NumericContract::contract_hash() const { return json_sha256_hex(to_json()); }

Status NumericContract::validate() const {
    if (ring_bits != 64) return fail_status("CONTRACT_PARAMS: ring_bits must be 64");
    if (frac_bits < 1 || frac_bits > 62) return fail_status("CONTRACT_PARAMS: frac_bits must be in [1, 62]");
    if (logical_bits < frac_bits + 1 || logical_bits > 63)
        return fail_status("CONTRACT_PARAMS: logical_bits must satisfy frac_bits+1 <= l <= 63");
    if (dcf_domain_bits < 2 || dcf_domain_bits > 64)
        return fail_status("CONTRACT_PARAMS: dcf_domain_bits must be in [2, 64]");
    if (exp_order != 1 && exp_order != 2) return fail_status("CONTRACT_PARAMS: exp_order must be 1 or 2");
    if (exp_r < 1 || exp_r > frac_bits) return fail_status("CONTRACT_PARAMS: exp_r must satisfy 1 <= r <= frac_bits");
    if (exp_Z < 0 || exp_extra_bits < 0 || recip_K < 0 || recip_degree < 0 || recip_newton < 0 || rsqrt_m < 0 ||
        rsqrt_degree < 0 || rsqrt_newton < 0 || layernorm_extra_bits < 0)
        return fail_status("CONTRACT_PARAMS: gadget parameters must be non-negative");
    if (argmax_tie != "lowest_index") return fail_status("CONTRACT_PARAMS: argmax_tie must be lowest_index");
    if (trunc_products != "local_prob" && trunc_products != "faithful")
        return fail_status("CONTRACT_PARAMS: trunc_products unknown");
    if (trunc_square_ln != "local_prob" && trunc_square_ln != "faithful")
        return fail_status("CONTRACT_PARAMS: trunc_square_ln unknown");
    if (activation_semantics != "erf_gelu") return fail_status("CONTRACT_PARAMS: activation_semantics must be erf_gelu");
    if (name.empty()) return fail_status("CONTRACT_PARAMS: name must be a non-empty string");
    return ok_status();
}

Expected<NumericContract> NumericContract::from_json(const Json& j) {
    using R = Expected<NumericContract>;
    if (!j.is_object()) return R::failure("CONTRACT_JSON: expected an object");
    std::set<std::string> expected(std::begin(kFields), std::end(kFields));
    std::set<std::string> present;
    for (const auto& kv : j.as_object()) present.insert(kv.first);
    if (present != expected) return R::failure("CONTRACT_JSON: field set mismatch");
    NumericContract c;
    auto geti = [&](const char* k, int& dst) -> std::string {
        Expected<std::int64_t> v = j.get_int(k);
        if (!v.ok) return v.code;
        dst = static_cast<int>(v.value);
        return "";
    };
    auto gets = [&](const char* k, std::string& dst) -> std::string {
        Expected<std::string> v = j.get_string(k);
        if (!v.ok) return v.code;
        dst = v.value;
        return "";
    };
    std::string err;
    if (!(err = geti("ring_bits", c.ring_bits)).empty()) return R::failure("CONTRACT_JSON: " + err);
    if (!(err = geti("frac_bits", c.frac_bits)).empty()) return R::failure("CONTRACT_JSON: " + err);
    if (!(err = geti("logical_bits", c.logical_bits)).empty()) return R::failure("CONTRACT_JSON: " + err);
    if (!(err = geti("dcf_domain_bits", c.dcf_domain_bits)).empty()) return R::failure("CONTRACT_JSON: " + err);
    if (!(err = gets("name", c.name)).empty()) return R::failure("CONTRACT_JSON: " + err);
    if (!(err = gets("gelu", c.gelu)).empty()) return R::failure("CONTRACT_JSON: " + err);
    if (!(err = gets("silu", c.silu)).empty()) return R::failure("CONTRACT_JSON: " + err);
    if (!(err = geti("exp_order", c.exp_order)).empty()) return R::failure("CONTRACT_JSON: " + err);
    if (!(err = geti("exp_r", c.exp_r)).empty()) return R::failure("CONTRACT_JSON: " + err);
    if (!(err = geti("exp_Z", c.exp_Z)).empty()) return R::failure("CONTRACT_JSON: " + err);
    if (!(err = geti("exp_extra_bits", c.exp_extra_bits)).empty()) return R::failure("CONTRACT_JSON: " + err);
    if (!(err = geti("recip_K", c.recip_K)).empty()) return R::failure("CONTRACT_JSON: " + err);
    if (!(err = geti("recip_degree", c.recip_degree)).empty()) return R::failure("CONTRACT_JSON: " + err);
    if (!(err = geti("recip_newton", c.recip_newton)).empty()) return R::failure("CONTRACT_JSON: " + err);
    if (!(err = geti("rsqrt_j0", c.rsqrt_j0)).empty()) return R::failure("CONTRACT_JSON: " + err);
    if (!(err = geti("rsqrt_m", c.rsqrt_m)).empty()) return R::failure("CONTRACT_JSON: " + err);
    if (!(err = geti("rsqrt_degree", c.rsqrt_degree)).empty()) return R::failure("CONTRACT_JSON: " + err);
    if (!(err = geti("rsqrt_newton", c.rsqrt_newton)).empty()) return R::failure("CONTRACT_JSON: " + err);
    if (!(err = geti("layernorm_extra_bits", c.layernorm_extra_bits)).empty()) return R::failure("CONTRACT_JSON: " + err);
    if (!(err = gets("argmax_tie", c.argmax_tie)).empty()) return R::failure("CONTRACT_JSON: " + err);
    if (!(err = gets("trunc_products", c.trunc_products)).empty()) return R::failure("CONTRACT_JSON: " + err);
    if (!(err = gets("trunc_square_ln", c.trunc_square_ln)).empty()) return R::failure("CONTRACT_JSON: " + err);
    if (!(err = gets("activation_semantics", c.activation_semantics)).empty()) return R::failure("CONTRACT_JSON: " + err);
    Status st = c.validate();
    if (!st.ok) return R::failure(st.code);
    return R::success(c);
}

NumericContract default_contract() { return NumericContract(); }

NumericContract smollm_contract() {
    NumericContract c;
    c.logical_bits = 46;
    c.rsqrt_m = 16;
    c.name = "NC-v2-f16-l46-smollm";
    return c;
}

Expected<NumericContract> contract_by_name(const std::string& name) {
    if (name == "NC-v2-f16-l40") return Expected<NumericContract>::success(default_contract());
    if (name == "NC-v2-f16-l46-smollm") return Expected<NumericContract>::success(smollm_contract());
    return Expected<NumericContract>::failure("CONTRACT_UNKNOWN: " + name);
}

}  // namespace highlevel
}  // namespace opencrab2
