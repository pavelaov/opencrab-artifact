// Implementation of telemetry.hpp.
#include <opencrab2/highlevel/telemetry.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace opencrab2 {
namespace highlevel {

Value TelemetryFrame::to_value() const {
    return dict({{"seq", seq},
                 {"t", t},
                 {"bandwidth_bps", bandwidth_bps},
                 {"rtt_s", rtt_s},
                 {"epoch", epoch},
                 {"cpu_load", cpu_load},
                 {"source", source}});
}

Expected<TelemetryFrame> TelemetryFrame::from_value(const Value& v) {
    OC2_TRY(d, v.as_dict());
    static const char* names[] = {"seq", "t", "bandwidth_bps", "rtt_s", "epoch", "cpu_load", "source"};
    if (d->size() != 7) return fail("TELEMETRY_MALFORMED: frame must have exactly 7 keys");
    for (const char* n : names)
        if (!v.has(n)) return fail(std::string("TELEMETRY_MALFORMED: missing ") + n);
    TelemetryFrame f;
    const Value* seqv = v.get("seq");
    const Value* epv = v.get("epoch");
    if (!seqv->is_int() || !epv->is_int()) return fail("TELEMETRY_MALFORMED: seq and epoch must be ints");
    f.seq = std::get<int64_t>(seqv->v);
    f.epoch = std::get<int64_t>(epv->v);
    for (const char* n : {"t", "bandwidth_bps", "rtt_s", "cpu_load"}) {
        const Value* x = v.get(n);
        if (x->is_bool() || !(x->is_int() || x->is_double())) return fail(std::string("TELEMETRY_MALFORMED: ") + n + " must be a number");
    }
    f.t = v.get_number("t").value;
    f.bandwidth_bps = v.get_number("bandwidth_bps").value;
    f.rtt_s = v.get_number("rtt_s").value;
    f.cpu_load = v.get_number("cpu_load").value;
    OC2_TRY(src, v.get_string("source"));
    f.source = src;
    return ok(std::move(f));
}

Expected<std::string> TelemetryFrame::canonical() const { return canonical_json(to_value()); }

Expected<std::string> TelemetryFrame::digest() const {
    OC2_TRY(c, canonical());
    return ok(sha256_hex(c));
}

TelemetrySample TelemetryFrame::to_sample() const {
    TelemetrySample s;
    s.seq = static_cast<int>(seq);
    s.t = t;
    s.bandwidth_bps = bandwidth_bps;
    s.rtt_s = rtt_s;
    s.epoch = static_cast<int>(epoch);
    s.cpu_load = cpu_load;
    return s;
}

Env TelemetryFrame::to_env() const {
    Env e;
    e.bandwidth_bps = bandwidth_bps;
    e.one_way_latency_s = rtt_s / 2.0;
    e.cpu_share = std::max(0.05, 1.0 - cpu_load);
    return e;
}

Expected<std::string> sign_frame(const TelemetryFrame& frame, const Bytes& key) {
    if (key.size() < 16) return fail("TELEMETRY_KEY: key must be at least 16 bytes");
    return hmac_tag(key, frame.to_value());
}

TelemetryVerifier::TelemetryVerifier(Bytes key, int64_t first_seq, double max_bandwidth_bps, double max_rtt_s)
    : key_(std::move(key)), next_seq_(first_seq), max_bw_(max_bandwidth_bps), max_rtt_(max_rtt_s) {}

Expected<Unit> TelemetryVerifier::accept(const TelemetryFrame& frame, const std::string& tag) {
    auto reject = [this](const std::string& code) -> Expected<Unit> {
        rejected_ += 1;
        return fail(code);
    };
    if (frame.seq < 0) return reject("TELEMETRY_BAD_SEQ: negative sequence number");
    if (frame.seq < next_seq_) return reject("TELEMETRY_REPLAY: sequence number already consumed");
    if (frame.seq > next_seq_) return reject("TELEMETRY_OUT_OF_ORDER: gap in sequence numbers");
    for (double x : {frame.t, frame.bandwidth_bps, frame.rtt_s, frame.cpu_load})
        if (!std::isfinite(x)) return reject("TELEMETRY_NON_FINITE");
    if (frame.bandwidth_bps < 0 || frame.bandwidth_bps > max_bw_ || frame.rtt_s < 0 || frame.rtt_s > max_rtt_ ||
        frame.cpu_load < 0 || frame.cpu_load > 1.0)
        return reject("TELEMETRY_OUT_OF_RANGE");
    if (have_last_ && frame.t < last_t_) return reject("TELEMETRY_NON_MONOTONE_TIME");
    if (frame.epoch < last_epoch_) return reject("TELEMETRY_BAD_EPOCH: epoch decreased");
    if (key_.size() < 16) return reject("TELEMETRY_KEY: key must be at least 16 bytes");
    auto okv = hmac_verify(key_, frame.to_value(), tag);
    if (!okv.ok) return reject(okv.code);
    if (!okv.value) return reject("TELEMETRY_BAD_TAG");
    next_seq_ = frame.seq + 1;
    have_last_ = true;
    last_t_ = frame.t;
    last_epoch_ = frame.epoch;
    accepted_.push_back(frame);
    return ok_unit();
}

namespace {

// The median of a sample: the middle value, or the mean of the two middle values of an even count
// (the rule of Python statistics.median, so both languages compute the same frame).
double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    if (n % 2 == 1) return v[n / 2];
    return (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

}  // namespace

Expected<std::pair<double, double>> probe_estimate(const std::vector<double>& rtts, const std::vector<double>& elapsed, size_t burst) {
    if (rtts.empty() || rtts.size() != elapsed.size() || burst < 1)
        return fail("TELEMETRY_PROBE_PARAMS: one ping time per burst time, at least one sample, burst >= 1");
    const double net = std::max(median_of(elapsed) - *std::min_element(rtts.begin(), rtts.end()), PROBE_MIN_NET_S);
    return ok(std::make_pair(static_cast<double>(burst) * 8.0 / net, median_of(rtts)));
}

Expected<std::pair<double, double>> probe_channel(Channel& ch, int party, int samples, size_t burst) {
    // Public parameters are validated before the first message: an empty sample set has no median
    // (the unchecked code indexed an empty vector), and the probe stays a fixed, bounded size.
    if (party != 0 && party != 1) return fail("TELEMETRY_PARAMS: party must be 0 or 1");
    if (samples < 1) return fail("TELEMETRY_PARAMS: probe_samples must be >= 1");
    if (samples > PROBE_MAX_SAMPLES) return fail("TELEMETRY_PARAMS: probe_samples above the declared maximum");
    if (burst < 1 || burst > PROBE_MAX_BURST_BYTES) return fail("TELEMETRY_PARAMS: probe_burst must be in [1, 2^28] bytes");
    std::vector<double> rtts, els;
    Bytes ping(PROBE_PING_BYTES, 0), payload(burst, 0), ack(PROBE_ACK_BYTES, 0);
    for (int s = 0; s < samples; ++s) {
        // Rounds are accounted explicitly (2 per sample on both sides): every
        // bare send clears the channel's pending flag so that no automatic
        // send-then-recv round is added on top.
        if (party == 1) {
            double t0 = now_s();
            OC2_CHECK(ch.send(Value(ping)));
            ch.clear_pending();
            OC2_TRY(pong, ch.recv());
            if (!pong.is_bytes()) return fail("TELEMETRY_PROBE: unexpected pong");
            double rtt = now_s() - t0;
            t0 = now_s();
            OC2_CHECK(ch.send(Value(payload)));
            ch.clear_pending();
            OC2_TRY(a, ch.recv());
            if (!a.is_bytes()) return fail("TELEMETRY_PROBE: unexpected ack");
            double el = now_s() - t0;
            rtts.push_back(rtt);
            els.push_back(el);
        } else {
            OC2_TRY(p, ch.recv());
            if (!p.is_bytes()) return fail("TELEMETRY_PROBE: unexpected ping");
            OC2_CHECK(ch.send(Value(ping)));
            ch.clear_pending();
            OC2_TRY(b, ch.recv());
            if (!b.is_bytes()) return fail("TELEMETRY_PROBE: unexpected burst");
            OC2_CHECK(ch.send(Value(ack)));
            ch.clear_pending();
        }
        ch.stats.rounds += 2;
    }
    if (party == 1) return probe_estimate(rtts, els, burst);
    double nan = std::numeric_limits<double>::quiet_NaN();
    return ok(std::make_pair(nan, nan));
}

Expected<TelemetryFrame> sense_epoch(Channel& ch, int party, int64_t seq, int64_t epoch, double t_session,
                                     TelemetryVerifier& verifier, const Bytes& key, double max_bandwidth_bps, int samples,
                                     size_t burst) {
    OC2_TRY(probe, probe_channel(ch, party, samples, burst));
    TelemetryFrame frame;
    std::string tag;
    if (party == 1) {
        frame.seq = seq;
        frame.t = t_session;
        frame.bandwidth_bps = std::min(probe.first, max_bandwidth_bps);
        frame.rtt_s = std::max(probe.second, 1e-6);
        frame.epoch = epoch;
        frame.cpu_load = 0.0;
        frame.source = "probe";
        OC2_TRY(tg, sign_frame(frame, key));
        tag = tg;
        OC2_CHECK(ch.send(dict({{"frame", frame.to_value()}, {"tag", tag}})));
        ch.clear_pending();
    } else {
        OC2_TRY(msg, ch.recv());
        const Value* fv = msg.get("frame");
        if (!fv) return fail("TELEMETRY_MALFORMED: missing frame");
        OC2_TRY(f, TelemetryFrame::from_value(*fv));
        OC2_TRY(tg, msg.get_string("tag"));
        frame = f;
        tag = tg;
    }
    ch.clear_pending();
    ch.stats.rounds += 1;
    OC2_CHECK(verifier.accept(frame, tag));
    return ok(std::move(frame));
}

}  // namespace highlevel
}  // namespace opencrab2
