// OpenCrab v2: statistics implementation (see include/opencrab2/highlevel/statistics.hpp).
#include "opencrab2/highlevel/statistics.hpp"

#include <algorithm>
#include <cmath>

namespace opencrab2 {
namespace highlevel {

std::uint64_t SplitMix64::next() {
    std::uint64_t z = (state_ += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

double SplitMix64::uniform01() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }

std::uint64_t SplitMix64::below(std::uint64_t n) {
    if (n <= 1) return 0;
    // Unbiased rejection sampling: accept v < 2^64 - (2^64 mod n). When n divides 2^64 (powers of
    // two) every word is accepted; the limit must not be formed by wrapping arithmetic in that case.
    const std::uint64_t rem = (std::uint64_t(0) - n) % n;   // 2^64 mod n
    std::uint64_t v = next();
    if (rem != 0) {
        const std::uint64_t limit = std::uint64_t(0) - rem;
        while (v >= limit) v = next();
    }
    return v % n;
}

Json BootstrapCI::to_json() const {
    Json j = Json::object();
    j.set("point", point);
    j.set("lo", lo);
    j.set("hi", hi);
    return j;
}

Json Summary::to_json() const {
    Json j = Json::object();
    j.set("n", static_cast<long long>(n));
    j.set("mean", mean);
    j.set("median", median);
    j.set("p95", p95);
    j.set("min", min);
    j.set("max", max);
    j.set("std", std);
    return j;
}

namespace {

Status check_vector(const std::vector<double>& v, const char* name) {
    if (v.empty()) return fail_status(std::string("STATISTICS_EMPTY: ") + name + " has no values");
    for (double x : v)
        if (!std::isfinite(x)) return fail_status(std::string("STATISTICS_NONFINITE: ") + name + " contains NaN or infinity");
    return ok_status();
}

Status check_boot(int n_boot, double alpha) {
    if (n_boot < 1) return fail_status("STATISTICS_PARAM: n_boot must be a positive int");
    if (!(alpha > 0.0 && alpha < 1.0)) return fail_status("STATISTICS_PARAM: alpha must lie in (0, 1)");
    return ok_status();
}

double mean_of(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}

double percentile_sorted(const std::vector<double>& sorted, double q) {
    const std::size_t n = sorted.size();
    if (n == 1) return sorted[0];
    const double pos = (q / 100.0) * static_cast<double>(n - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = lo + 1 < n ? lo + 1 : lo;
    const double frac = pos - static_cast<double>(lo);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * frac;
}

}  // namespace

Expected<double> percentile_linear(std::vector<double> values, double q) {
    Status st = check_vector(values, "values");
    if (!st.ok) return Expected<double>::failure(st.code);
    if (!(q >= 0.0 && q <= 100.0)) return Expected<double>::failure("STATISTICS_PARAM: q must lie in [0, 100]");
    std::sort(values.begin(), values.end());
    return Expected<double>::success(percentile_sorted(values, q));
}

Expected<BootstrapCI> paired_bootstrap_ci(const std::vector<double>& diffs, int n_boot, double alpha, std::uint64_t seed) {
    Status st = check_boot(n_boot, alpha);
    if (!st.ok) return Expected<BootstrapCI>::failure(st.code);
    st = check_vector(diffs, "diffs");
    if (!st.ok) return Expected<BootstrapCI>::failure(st.code);
    SplitMix64 rng(seed);
    const std::size_t n = diffs.size();
    std::vector<double> samples(static_cast<std::size_t>(n_boot));
    for (int b = 0; b < n_boot; ++b) {
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i) s += diffs[static_cast<std::size_t>(rng.below(n))];
        samples[static_cast<std::size_t>(b)] = s / static_cast<double>(n);
    }
    std::sort(samples.begin(), samples.end());
    BootstrapCI ci;
    ci.point = mean_of(diffs);
    ci.lo = percentile_sorted(samples, 100.0 * alpha / 2.0);
    ci.hi = percentile_sorted(samples, 100.0 * (1.0 - alpha / 2.0));
    return Expected<BootstrapCI>::success(ci);
}

Expected<BootstrapCI> ratio_ci(const std::vector<double>& a, const std::vector<double>& b, int n_boot, double alpha,
                               std::uint64_t seed) {
    Status st = check_boot(n_boot, alpha);
    if (!st.ok) return Expected<BootstrapCI>::failure(st.code);
    st = check_vector(a, "a");
    if (!st.ok) return Expected<BootstrapCI>::failure(st.code);
    st = check_vector(b, "b");
    if (!st.ok) return Expected<BootstrapCI>::failure(st.code);
    if (mean_of(b) == 0.0) return Expected<BootstrapCI>::failure("STATISTICS_RATIO_DENOMINATOR: mean(b) is zero");
    const bool paired = a.size() == b.size();
    SplitMix64 rng(seed);
    std::vector<double> samples(static_cast<std::size_t>(n_boot));
    for (int k = 0; k < n_boot; ++k) {
        double num = 0.0, den = 0.0;
        if (paired) {
            for (std::size_t i = 0; i < a.size(); ++i) {
                const std::size_t idx = static_cast<std::size_t>(rng.below(a.size()));
                num += a[idx];
                den += b[idx];
            }
            num /= static_cast<double>(a.size());
            den /= static_cast<double>(b.size());
        } else {
            for (std::size_t i = 0; i < a.size(); ++i) num += a[static_cast<std::size_t>(rng.below(a.size()))];
            for (std::size_t i = 0; i < b.size(); ++i) den += b[static_cast<std::size_t>(rng.below(b.size()))];
            num /= static_cast<double>(a.size());
            den /= static_cast<double>(b.size());
        }
        if (den == 0.0) return Expected<BootstrapCI>::failure("STATISTICS_RATIO_DENOMINATOR: a resampled mean of b is zero");
        samples[static_cast<std::size_t>(k)] = num / den;
    }
    std::sort(samples.begin(), samples.end());
    BootstrapCI ci;
    ci.point = mean_of(a) / mean_of(b);
    ci.lo = percentile_sorted(samples, 100.0 * alpha / 2.0);
    ci.hi = percentile_sorted(samples, 100.0 * (1.0 - alpha / 2.0));
    return Expected<BootstrapCI>::success(ci);
}

Expected<double> zero_failure_upper_bound(std::int64_t n, double alpha) {
    if (n < 1) return Expected<double>::failure("STATISTICS_PARAM: n must be a positive int");
    if (!(alpha > 0.0 && alpha < 1.0)) return Expected<double>::failure("STATISTICS_PARAM: alpha must lie in (0, 1)");
    return Expected<double>::success(1.0 - std::pow(alpha, 1.0 / static_cast<double>(n)));
}

Expected<Summary> summarize(const std::vector<double>& values) {
    Status st = check_vector(values, "values");
    if (!st.ok) return Expected<Summary>::failure(st.code);
    std::vector<double> sorted(values);
    std::sort(sorted.begin(), sorted.end());
    Summary s;
    s.n = static_cast<std::int64_t>(values.size());
    s.mean = mean_of(values);
    s.median = percentile_sorted(sorted, 50.0);
    s.p95 = percentile_sorted(sorted, 95.0);
    s.min = sorted.front();
    s.max = sorted.back();
    if (values.size() > 1) {
        double ss = 0.0;
        for (double x : values) ss += (x - s.mean) * (x - s.mean);
        s.std = std::sqrt(ss / static_cast<double>(values.size() - 1));
    }
    return Expected<Summary>::success(s);
}

}  // namespace highlevel
}  // namespace opencrab2
