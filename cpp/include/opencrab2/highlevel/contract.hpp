// OpenCrab v2: numeric contract (mirror of python/opencrab2/highlevel/contract.py).
//
// NumericContract is the versioned description of the discrete fixed-point functionality every
// realization implements. Its identity is contract_hash(): the SHA-256 hex of the canonical JSON
// (sorted keys, no spaces) of all fields, with integers serialized as integers and strings as
// strings. The field list is exactly the Python one; the C++ and Python hashes agree
// (tests/test_decision.cpp checks the default hash against the Python hashlib value).
//
// Validation mirrors the Python __post_init__ checks and fails closed with CONTRACT_PARAMS.
#pragma once

#include <cstdint>
#include <string>

#include "opencrab2/expected.hpp"
#include "opencrab2/highlevel/json.hpp"

namespace opencrab2 {
namespace highlevel {

struct NumericContract {
    int ring_bits = 64;
    int frac_bits = 16;
    int logical_bits = 40;
    int dcf_domain_bits = 32;
    std::string name = "NC-v2-f16-l40";
    std::string gelu = "gelu_clip12";
    std::string silu = "silu_clip12";
    int exp_order = 2;
    int exp_r = 7;
    int exp_Z = 16;
    int exp_extra_bits = 4;
    int recip_K = 11;
    int recip_degree = 5;
    int recip_newton = 2;
    int rsqrt_j0 = -7;
    int rsqrt_m = 15;
    int rsqrt_degree = 5;
    int rsqrt_newton = 2;
    int layernorm_extra_bits = 4;
    std::string argmax_tie = "lowest_index";
    std::string trunc_products = "local_prob";
    std::string trunc_square_ln = "faithful";
    std::string activation_semantics = "erf_gelu";

    // Derived quantities.
    std::uint64_t scale() const { return std::uint64_t(1) << frac_bits; }
    std::uint64_t logical_bound() const { return std::uint64_t(1) << (logical_bits - 1); }
    double compare_bound_real() const;                 // 2^(n-1-f)
    Expected<int> softmax_extra_bits(int n) const;     // min(ceil(log2 n), 10)
    int gelu_degree() const;                           // parsed from the gelu contract name (12 for gelu_clip12)

    // Identity.
    Json to_json() const;
    std::string contract_hash() const;
    Status validate() const;                           // CONTRACT_PARAMS on an unimplementable parameter set
    static Expected<NumericContract> from_json(const Json& j);   // exact field set required
};

NumericContract default_contract();                    // NC-v2-f16-l40
NumericContract smollm_contract();                     // NC-v2-f16-l46-smollm (l = 46, rsqrt_m = 16)
Expected<NumericContract> contract_by_name(const std::string& name);   // CONTRACT_UNKNOWN otherwise

}  // namespace highlevel
}  // namespace opencrab2
