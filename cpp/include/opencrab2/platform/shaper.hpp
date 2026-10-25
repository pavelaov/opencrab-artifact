// opencrab2/platform/shaper.hpp
//
// Exogenous trace files and wall-clock link shaping (paper, Section 7.1).
//
// Trace JSON (python/opencrab2/experiments/traces.py, make_trace):
//   {trace_id, trace_hash, family, variant, split, seed, duration_s,
//    points: [[elapsed_s, bandwidth_bps, one_way_delay_s, cpu_share, stock_refill_bps], ...],
//    events: [[elapsed_s, kind], ...]            (S7: cancel | resume | disconnect)
//    initial_installed: [object id, ...]         (S8)
//    length_bucket, mem_limit_bytes              (S6)
//    initial_stock_bytes                         (S5)}
// A point holds from its elapsed_s until the next point; the last point holds
// forever. bandwidth_bps <= 0 means unlimited. Rows may also be objects with
// the five column names (a C++ convenience, never produced by Python).
//
// trace_hash is the Python points_hash: sha256 of the compact JSON of the
// points rows as floats. A file that declares a trace_hash is verified against
// its points and refused with TRACE_HASH_MISMATCH otherwise (as traces.py
// load_trace does); a file without one gets the computed hash. The JSON text
// is produced by the one JSON implementation of the tree (highlevel::Json).
//
// Shaping has ONE token-bucket implementation, highlevel::WallClockShaper
// (channel.hpp). This header adds what a platform process needs around it:
//   * Trace: the parsed file with the public side columns (cpu_share, stock
//     refill) and the S5 to S8 metadata;
//   * WallClockShaper: the trace-aware adapter used by the WebSocket channel.
//     It indexes the trace by elapsed wall time since a shared origin (Unix
//     time), so two processes started with the same origin see the same link
//     state at the same instant and a change applies in the middle of a
//     transfer; apply() queues back-to-back messages on one FIFO link and
//     sleeps for the delay;
//   * make_channel_shaper: the highlevel shaper for a highlevel::Channel, with
//     the shared Unix origin converted to this process's monotonic clock;
//   * ContentionLoad: cpu_share < 1 as a public busy-loop thread with duty
//     cycle 1 - cpu_share (public load, not a secret);
//   * TraceDriver: trace + shaper + contention for one process.
//
// Codes: TRACE_MALFORMED, TRACE_EMPTY, TRACE_UNSORTED, TRACE_IO,
// TRACE_HASH_MISMATCH.
//
// Tested (tests/test_platform.cpp): parse of a valid trace, the Python hash
// of a make_trace file, rejection of malformed traces and of a tampered
// point, state_at before/after boundaries, delay_for across a bandwidth
// change (closed form), unlimited link gives delay == one_way_delay, FIFO
// queueing, agreement of the adapter with the highlevel shaper.
#pragma once

#include <opencrab2/platform/support.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace opencrab2 {
namespace platform {

struct TracePoint {
  double elapsed_s = 0.0;
  double bandwidth_bps = 0.0;  // <= 0: unlimited
  double one_way_delay_s = 0.0;
  double cpu_share = 1.0;
  double stock_refill_bps = 0.0;  // bytes per second of dealer delivery (S5); 0 = not declared
};

struct TraceEvent {
  double elapsed_s = 0.0;
  std::string kind;  // cancel | resume | disconnect
};

struct Trace {
  std::string trace_id;
  std::string trace_hash;
  std::string family;
  std::string variant;
  std::string split;
  int64_t seed = 0;
  double declared_duration_s = 0.0;           // duration_s of the file, 0 when absent
  std::vector<TracePoint> points;
  std::vector<TraceEvent> events;             // S7
  std::vector<std::string> initial_installed; // S8
  int64_t length_bucket = 0;                  // S6, 0 when absent
  int64_t mem_limit_bytes = 0;                // S6, 0 when absent
  int64_t initial_stock_bytes = 0;            // S5, 0 when absent
  bool hash_verified = false;                 // the file declared a hash and it matched

  const TracePoint& at(double elapsed_s) const;  // piecewise constant lookup
  double duration_s() const { return points.empty() ? 0.0 : points.back().elapsed_s; }
  // Index of the first point with elapsed_s > t, or points.size().
  size_t next_index_after(double elapsed_s) const;
  // True when some point declares cpu_share < 1 (S4) or a stock refill (S5).
  bool has_contention() const;
  double max_stock_refill_bps() const;
  json::Value to_json() const;
  // Python points_hash of the five-column rows (sha256 hex).
  std::string compute_hash() const;
  // (elapsed, bandwidth, one-way delay) rows for the highlevel shaper.
  std::vector<highlevel::TracePoint> link_points() const;
};

Expected<Trace> parse_trace_json(const std::string& text);
Expected<Trace> load_trace_file(const std::string& path);
Trace make_constant_trace(double bandwidth_bps, double one_way_delay_s, double duration_s, double cpu_share = 1.0,
                          const std::string& trace_id = "const");

struct LinkState {
  double bandwidth_bps = 0.0;
  double one_way_delay_s = 0.0;
  bool unlimited() const { return bandwidth_bps <= 0.0; }
};

// The highlevel shaper of `trace` for a highlevel::Channel. origin_unix_s is
// the shared origin in Unix seconds (exchanged between the parties); the
// returned shaper is indexed by this process's monotonic clock
// (highlevel::now_s), which is what highlevel::Channel passes to it.
std::shared_ptr<highlevel::WallClockShaper> make_channel_shaper(const Trace& trace, double origin_unix_s);

class WallClockShaper {
 public:
  WallClockShaper(Trace trace, double origin_wall_s);

  LinkState state_at(double wall_s) const;
  // Delay observed by the receiver for a message of nbytes that starts
  // leaving at wall_s on an idle link: serialization across segments plus
  // the one-way delay at the start instant.
  double delay_for(size_t nbytes, double wall_s) const;
  // Serialization time only (no propagation delay), starting at wall_s.
  double serialization_s(size_t nbytes, double wall_s) const;
  // Queue the message behind earlier ones (FIFO link), sleep for the delay,
  // return it. Thread safe.
  double apply(size_t nbytes);
  double origin_wall_s() const { return origin_; }
  const Trace& trace() const { return trace_; }
  double total_delay_s() const;
  uint64_t messages() const;

 private:
  Trace trace_;
  double origin_;
  std::vector<highlevel::TracePoint> points_;
  std::unique_ptr<highlevel::WallClockShaper> fifo_;  // the one token bucket (link-busy state)
  mutable std::mutex mu_;
  double total_delay_ = 0.0;
  uint64_t messages_ = 0;
};

class ContentionLoad {
 public:
  ContentionLoad() = default;
  ~ContentionLoad() { stop(); }
  ContentionLoad(const ContentionLoad&) = delete;
  ContentionLoad& operator=(const ContentionLoad&) = delete;

  // Spawns `threads` busy-loop threads whose duty cycle follows
  // 1 - cpu_share(elapsed) with a 20 ms period.
  void start(const Trace& trace, double origin_wall_s, int threads = 1);
  void stop();
  bool running() const { return running_.load(); }
  double current_busy_fraction() const { return busy_fraction_.load(); }

 private:
  void run();
  Trace trace_;
  double origin_ = 0.0;
  std::atomic<bool> running_{false};
  std::atomic<double> busy_fraction_{0.0};
  std::vector<std::thread> threads_;
};

// Bundles a trace, its shaper and the contention load for one process.
class TraceDriver {
 public:
  TraceDriver(Trace trace, double origin_wall_s);
  WallClockShaper& shaper() { return shaper_; }
  const Trace& trace() const { return trace_; }
  double origin_wall_s() const { return origin_; }
  double elapsed_s() const;
  void start_contention(int threads = 1);
  void stop();
  bool contention_running() const { return load_.running(); }
  double current_busy_fraction() const { return load_.current_busy_fraction(); }

 private:
  Trace trace_;
  double origin_;
  WallClockShaper shaper_;
  ContentionLoad load_;
};

}  // namespace platform
}  // namespace opencrab2
