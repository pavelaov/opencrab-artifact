// OpenCrab v2 C++17 mirror: polynomial approximation contracts and gadget
// reference formulas (mirror of python/opencrab2/highlevel/approx.py; paper,
// Section 5.4). The nonlinear Party methods themselves live in src/highlevel/gadgets.cpp.
//
// Every clipped activation and every normalised reciprocal evaluates a
// polynomial on a normalised argument u inside a declared range (the
// approximation contract). Coefficients are computed at startup by least
// squares on 4000 Chebyshev nodes (Householder QR), exactly like approx.py:
//   gelu_clip12  erf-GELU on [-5, 5], u = x / 5, degree 12 (plain fit, not even)
//   silu_clip12  SiLU on [-8, 8], u = x / 8, degree 12
//   tanh_clip    tanh on [-4, 4], u = x / 4, degree 9
//   recip_norm   1/v on [0.5, 1), affine u = 2 (v - lo) / (hi - lo) - 1, degree 5
//   rsqrt_norm   1/sqrt(v) on [0.25, 1), affine, degree 5
// The recorded max_abs_err is measured on a 20001-point grid (the Python module
// reports 4.84e-3 for gelu_clip12 and 1.28e-2 for silu_clip12; the C++ fit
// reproduces these to the printed digits, tests/test_engine.cpp checks it).
//
// Plaintext references (double) of the fixed-point gadgets are provided for
// the tests: gelu_ref, silu_ref, softmax_ref, layernorm_ref.
#pragma once

#include <opencrab2/highlevel/codec.hpp>

#include <functional>
#include <string>
#include <vector>

namespace opencrab2 {
namespace highlevel {

struct PolyContract {
    std::string name;
    std::vector<double> coeffs;  // ascending powers of the fitted variable
    double lo = 0.0;
    double hi = 0.0;
    double scale = 1.0;          // u = x / scale (symmetric range)
    bool affine = false;         // u = 2 (x - lo) / (hi - lo) - 1
    bool even_in_u = false;      // variable is u^2
    double max_abs_err = 0.0;
    int degree = 0;
    double evaluate(double x) const;  // plaintext evaluation of the contract
    Value to_value() const;
};

// Least-squares fit on Chebyshev nodes (fail-closed on degenerate parameters).
Expected<PolyContract> fit_poly(const std::string& name, const std::function<double(double)>& f, double lo, double hi,
                                int degree, bool even = false, bool affine = false, int nodes = 4000);
// Named contracts (built once, deterministic). Code APPROX_UNKNOWN_CONTRACT otherwise.
Expected<const PolyContract*> poly_contract_by_name(const std::string& name);
Value poly_contract_table();

// Dense least squares min ||A c - y|| by Householder QR (A is n x m, n >= m).
Expected<std::vector<double>> lstsq(const std::vector<std::vector<double>>& A, const std::vector<double>& y);

// Plaintext references.
double gelu_ref(double x);   // erf-GELU
double silu_ref(double x);
std::vector<double> softmax_ref(const std::vector<double>& row, const std::vector<uint8_t>* mask = nullptr);
std::vector<double> layernorm_ref(const std::vector<double>& row, const std::vector<double>* gamma,
                                  const std::vector<double>* beta, double eps, bool rms);
// Lowest-index argmax of a double row (reference for the k-ary tournaments).
size_t argmax_ref(const std::vector<double>& row);

}  // namespace highlevel
}  // namespace opencrab2
