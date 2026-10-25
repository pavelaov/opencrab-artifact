// OpenCrab v2 C++17 mirror: public telemetry frames and the fixed-size channel
// probe (paper, Section 4.2, objective and decision; python/opencrab2/highlevel/telemetry.py).
//
// A TelemetryFrame is one sensing record of the PUBLIC environment (bandwidth,
// RTT, CPU load) at a fixed epoch. Frames are HMAC-SHA256 signed over their
// canonical JSON (sorted keys, no spaces, Python float repr: the same bytes as
// the Python module) and carry a sequence number. Both parties feed identical
// verified frames to identical deterministic controllers; a frame that is
// replayed, out of order, non-monotone in time, non-finite, out of range or
// wrongly tagged is rejected with a stable TELEMETRY_* code.
//
// The probe is fixed-size by construction: `samples` ping/pong pairs of
// 16-byte payloads followed by one `burst`-byte transfer answered by a 16-byte
// acknowledgement. Its message sizes depend on nothing secret.
//
// Estimator (probe_estimate, the same figures as the Python module): the
// round-trip time is the median of the ping times; the bandwidth is one pooled
// figure per probe, burst * 8 / (median(burst times) - min(ping times)) with
// the difference floored at PROBE_MIN_NET_S. The smallest ping is the least
// delayed observation of the round trip, so one late ping (a scheduler or
// shaper tick) cannot cancel the burst time of its own sample, which a median
// of per-sample ratios lets happen when the burst time is close to the timer
// resolution. This narrows the upper tail of the estimate; it is not a bound.
// sense_epoch then caps the bandwidth at max_bandwidth_bps and floors the RTT
// at 1e-6 s before the frame is signed.
//
// Integrity note (not a security proof): the tag binds a frame to the shared
// telemetry key; it protects the two honest parties against a corrupted
// transport, not against a corrupted party that holds the key.
#pragma once

#include <opencrab2/highlevel/channel.hpp>
#include <opencrab2/highlevel/codec.hpp>
#include <opencrab2/highlevel/controller.hpp>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace opencrab2 {
namespace highlevel {

constexpr size_t PROBE_PING_BYTES = 16;
constexpr size_t PROBE_ACK_BYTES = 16;
constexpr int PROBE_MAX_SAMPLES = 1024;
constexpr size_t PROBE_MAX_BURST_BYTES = static_cast<size_t>(1) << 28;
constexpr double PROBE_MIN_NET_S = 1e-7;   // floor of the burst time net of the round trip (seconds)

struct TelemetryFrame {
    int64_t seq = 0;
    double t = 0.0;
    double bandwidth_bps = 0.0;
    double rtt_s = 0.0;
    int64_t epoch = 0;
    double cpu_load = 0.0;
    std::string source = "probe";

    Value to_value() const;
    static Expected<TelemetryFrame> from_value(const Value& v);  // strictly typed, exact key set
    Expected<std::string> canonical() const;
    Expected<std::string> digest() const;  // sha256 hex of the canonical frame
    TelemetrySample to_sample() const;    // the controller's view
    Env to_env() const;                   // bandwidth, one-way latency = rtt / 2
};

// HMAC-SHA256 hex tag over the canonical JSON; the key must have >= 16 bytes.
Expected<std::string> sign_frame(const TelemetryFrame& frame, const Bytes& key);

class TelemetryVerifier {
public:
    explicit TelemetryVerifier(Bytes key, int64_t first_seq = 0, double max_bandwidth_bps = 1e12, double max_rtt_s = 10.0);
    // Codes: TELEMETRY_BAD_TAG, TELEMETRY_BAD_SEQ, TELEMETRY_REPLAY, TELEMETRY_OUT_OF_ORDER,
    // TELEMETRY_NON_MONOTONE_TIME, TELEMETRY_NON_FINITE, TELEMETRY_OUT_OF_RANGE, TELEMETRY_BAD_EPOCH.
    Expected<Unit> accept(const TelemetryFrame& frame, const std::string& tag);
    size_t accepted() const { return accepted_.size(); }
    size_t rejected() const { return rejected_; }
    const std::vector<TelemetryFrame>& frames() const { return accepted_; }

private:
    Bytes key_;
    int64_t next_seq_;
    double max_bw_;
    double max_rtt_;
    bool have_last_ = false;
    double last_t_ = 0.0;
    int64_t last_epoch_ = -1;
    std::vector<TelemetryFrame> accepted_;
    size_t rejected_ = 0;
};

// (bandwidth_bps, rtt_s) of one probe from its samples: rtts[i] is the ping/pong
// time and elapsed[i] the burst/ack time of sample i. rtt_s = median(rtts),
// bandwidth_bps = burst * 8 / max(median(elapsed) - min(rtts), PROBE_MIN_NET_S);
// the median of an even number of samples is the mean of the two middle ones.
// TELEMETRY_PROBE_PARAMS: no sample, unequal lengths or burst < 1.
Expected<std::pair<double, double>> probe_estimate(const std::vector<double>& rtts, const std::vector<double>& elapsed, size_t burst);

// P1 drives the probe and returns probe_estimate over the samples; P0 answers and
// returns (nan, nan). Rounds are accounted explicitly: 2 per sample on both sides.
// Fails closed with TELEMETRY_PARAMS before any message when samples < 1,
// samples > PROBE_MAX_SAMPLES, or burst outside [1, PROBE_MAX_BURST_BYTES]
// (the parameters are public, so both parties refuse the same configuration).
Expected<std::pair<double, double>> probe_channel(Channel& ch, int party, int samples = 3, size_t burst = 65536);

// One sensing step: probe, P1 signs and sends the frame (one round), both
// verify the same frame and return it.
Expected<TelemetryFrame> sense_epoch(Channel& ch, int party, int64_t seq, int64_t epoch, double t_session,
                                     TelemetryVerifier& verifier, const Bytes& key, double max_bandwidth_bps = 1e12,
                                     int samples = 3, size_t burst = 65536);

}  // namespace highlevel
}  // namespace opencrab2
