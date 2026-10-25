// OpenCrab v2: evaluation statistics (paper, Section 7.1; mirror of python/opencrab2/highlevel/statistics.py).
//
// Paired percentile bootstrap for mean differences and ratios of means, the one-sided upper
// bound on a failure probability after n failure-free trials (1 - alpha^(1/n)), and a
// descriptive summary (n, mean, median, p95, min, max, std).
//
// Determinism: every bootstrap draws its resampling indices from a SplitMix64 stream seeded by
// the caller, so a report regenerated from the same raw records is byte-identical. The stream
// differs from numpy's PCG64, so the C++ and Python intervals agree in distribution, not bit
// for bit; the percentile function (numpy "linear" interpolation) is identical on both sides.
//
// Design limits: the percentile bootstrap assumes independent resampled units (resample
// sessions or traces, never correlated tokens); the zero-failure bound is a bound on a
// Bernoulli parameter, not a claim about correlated decisions. Codes: STATISTICS_EMPTY,
// STATISTICS_NONFINITE, STATISTICS_PARAM, STATISTICS_RATIO_DENOMINATOR.
#pragma once

#include <cstdint>
#include <vector>

#include "opencrab2/expected.hpp"
#include "opencrab2/highlevel/json.hpp"

namespace opencrab2 {
namespace highlevel {

// Small deterministic generator (SplitMix64) for tests and bootstraps.
class SplitMix64 {
public:
    explicit SplitMix64(std::uint64_t seed) : state_(seed) {}
    std::uint64_t next();
    double uniform01();                                  // [0, 1)
    std::uint64_t below(std::uint64_t n);                // uniform in [0, n), n > 0 (rejection sampling)
private:
    std::uint64_t state_;
};

struct BootstrapCI {
    double point = 0.0;   // sample statistic (mean of diffs, or ratio of means)
    double lo = 0.0;
    double hi = 0.0;
    Json to_json() const;
};

struct Summary {
    std::int64_t n = 0;
    double mean = 0.0, median = 0.0, p95 = 0.0, min = 0.0, max = 0.0, std = 0.0;   // std: sample std, 0 for n = 1
    Json to_json() const;
};

Expected<BootstrapCI> paired_bootstrap_ci(const std::vector<double>& diffs, int n_boot = 10000,
                                          double alpha = 0.05, std::uint64_t seed = 0);
Expected<BootstrapCI> ratio_ci(const std::vector<double>& a, const std::vector<double>& b, int n_boot = 10000,
                               double alpha = 0.05, std::uint64_t seed = 0);
Expected<double> zero_failure_upper_bound(std::int64_t n, double alpha = 0.05);
Expected<Summary> summarize(const std::vector<double>& values);

// numpy.percentile(values, q) with the default linear interpolation; values need not be sorted.
Expected<double> percentile_linear(std::vector<double> values, double q);

}  // namespace highlevel
}  // namespace opencrab2
