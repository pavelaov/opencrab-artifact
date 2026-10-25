// opencrab2/platform/shaper.cpp (see shaper.hpp)
#include <opencrab2/platform/shaper.hpp>
#include <opencrab2/platform/tcp.hpp>

#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>

namespace opencrab2 {
namespace platform {

namespace hl = opencrab2::highlevel;

// ---------------------------------------------------------------------------
// Trace
// ---------------------------------------------------------------------------
const TracePoint& Trace::at(double elapsed_s) const {
  static const TracePoint kDefault;
  if (points.empty()) return kDefault;
  if (elapsed_s <= points.front().elapsed_s) return points.front();
  // binary search for the last point with elapsed_s <= t
  size_t lo = 0, hi = points.size();
  while (hi - lo > 1) {
    size_t mid = (lo + hi) / 2;
    if (points[mid].elapsed_s <= elapsed_s) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return points[lo];
}

size_t Trace::next_index_after(double elapsed_s) const {
  size_t lo = 0, hi = points.size();
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    if (points[mid].elapsed_s <= elapsed_s) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

bool Trace::has_contention() const {
  for (const auto& p : points) {
    if (p.cpu_share < 1.0) return true;
  }
  return false;
}

double Trace::max_stock_refill_bps() const {
  double m = 0.0;
  for (const auto& p : points) {
    if (p.stock_refill_bps > m) m = p.stock_refill_bps;
  }
  return m;
}

namespace {

json::Value point_rows(const std::vector<TracePoint>& points) {
  json::Value pts = json::Value::array();
  for (const auto& p : points) {
    json::Value row = json::Value::array();
    row.push(json::Value::number(p.elapsed_s));
    row.push(json::Value::number(p.bandwidth_bps));
    row.push(json::Value::number(p.one_way_delay_s));
    row.push(json::Value::number(p.cpu_share));
    row.push(json::Value::number(p.stock_refill_bps));
    pts.push(std::move(row));
  }
  return pts;
}

// Python traces.points_hash: sha256 of json.dumps(rows of floats, separators=(",", ":")).
std::string rows_hash(const json::Value& rows) {
  auto h = detail::sha256(rows.dump(true));
  return detail::hex_encode(h.data(), h.size());
}

}  // namespace

json::Value Trace::to_json() const {
  json::Value o = json::Value::object();
  o.set("trace_id", json::Value::string(trace_id));
  o.set("trace_hash", json::Value::string(trace_hash));
  o.set("family", json::Value::string(family));
  o.set("variant", json::Value::string(variant));
  o.set("split", json::Value::string(split));
  o.set("seed", json::Value::integer(seed));
  if (declared_duration_s > 0) o.set("duration_s", json::Value::number(declared_duration_s));
  o.set("points", point_rows(points));
  json::Value ev = json::Value::array();
  for (const auto& e : events) {
    json::Value row = json::Value::array();
    row.push(json::Value::number(e.elapsed_s));
    row.push(json::Value::string(e.kind));
    ev.push(std::move(row));
  }
  o.set("events", std::move(ev));
  json::Value inst = json::Value::array();
  for (const auto& s : initial_installed) inst.push(json::Value::string(s));
  o.set("initial_installed", std::move(inst));
  o.set("length_bucket", json::Value::integer(length_bucket));
  if (mem_limit_bytes > 0) o.set("mem_limit_bytes", json::Value::integer(mem_limit_bytes));
  if (initial_stock_bytes > 0) o.set("initial_stock_bytes", json::Value::integer(initial_stock_bytes));
  return o;
}

std::string Trace::compute_hash() const { return rows_hash(point_rows(points)); }

std::vector<hl::TracePoint> Trace::link_points() const {
  std::vector<hl::TracePoint> out;
  out.reserve(points.size());
  for (const auto& p : points) {
    hl::TracePoint q;
    q.elapsed_s = p.elapsed_s;
    q.state.bandwidth_bps = p.bandwidth_bps;
    q.state.one_way_delay_s = p.one_way_delay_s;
    out.push_back(q);
  }
  return out;
}

Expected<Trace> parse_trace_json(const std::string& text) {
  auto j = json::parse(text);
  if (!j.ok) return make_fail<Trace>("TRACE_MALFORMED");
  const json::Value& root = j.value;
  if (!root.is_object()) return make_fail<Trace>("TRACE_MALFORMED");
  Trace t;
  t.trace_id = root.str_or("trace_id", "");
  t.trace_hash = root.str_or("trace_hash", "");
  t.family = root.str_or("family", "");
  t.variant = root.str_or("variant", "");
  t.split = root.str_or("split", "");
  t.seed = root.int_or("seed", 0);
  t.declared_duration_s = root.double_or("duration_s", 0.0);
  t.length_bucket = root.int_or("length_bucket", 0);
  t.mem_limit_bytes = root.int_or("mem_limit_bytes", 0);
  t.initial_stock_bytes = root.int_or("initial_stock_bytes", 0);
  const json::Value* pts = root.find("points");
  if (!pts || !pts->is_array()) return make_fail<Trace>("TRACE_MALFORMED");
  if (pts->arr.empty()) return make_fail<Trace>("TRACE_EMPTY");
  double last = -std::numeric_limits<double>::infinity();
  json::Value hashed_rows = json::Value::array();  // the rows as the Python hash sees them
  for (const auto& row : pts->arr) {
    TracePoint p;
    json::Value hrow = json::Value::array();
    if (row.is_array()) {
      if (row.arr.size() < 3) return make_fail<Trace>("TRACE_MALFORMED");
      for (const auto& c : row.arr) {
        if (!c.is_number() && !c.is_null()) return make_fail<Trace>("TRACE_MALFORMED");
        hrow.push(json::Value::number(c.is_null() ? 0.0 : c.as_double()));
      }
      p.elapsed_s = row.arr[0].as_double();
      p.bandwidth_bps = row.arr[1].is_null() ? 0.0 : row.arr[1].as_double();
      p.one_way_delay_s = row.arr[2].as_double();
      p.cpu_share = row.arr.size() > 3 && !row.arr[3].is_null() ? row.arr[3].as_double() : 1.0;
      p.stock_refill_bps = row.arr.size() > 4 && !row.arr[4].is_null() ? row.arr[4].as_double() : 0.0;
    } else if (row.is_object()) {
      p.elapsed_s = row.double_or("elapsed_s", 0.0);
      const json::Value* bw = row.find("bandwidth_bps");
      p.bandwidth_bps = (bw && bw->is_number()) ? bw->as_double() : 0.0;
      p.one_way_delay_s = row.double_or("one_way_delay_s", 0.0);
      p.cpu_share = row.double_or("cpu_share", 1.0);
      p.stock_refill_bps = row.double_or("stock_refill_bps", 0.0);
      hrow.push(json::Value::number(p.elapsed_s));
      hrow.push(json::Value::number(p.bandwidth_bps));
      hrow.push(json::Value::number(p.one_way_delay_s));
      hrow.push(json::Value::number(p.cpu_share));
      hrow.push(json::Value::number(p.stock_refill_bps));
    } else {
      return make_fail<Trace>("TRACE_MALFORMED");
    }
    if (!std::isfinite(p.elapsed_s) || !std::isfinite(p.bandwidth_bps) || !std::isfinite(p.one_way_delay_s) ||
        !std::isfinite(p.cpu_share) || !std::isfinite(p.stock_refill_bps))
      return make_fail<Trace>("TRACE_MALFORMED");
    if (!(p.elapsed_s >= 0.0) || !(p.one_way_delay_s >= 0.0) || p.cpu_share < 0.0 || p.cpu_share > 1.0 ||
        p.stock_refill_bps < 0.0)
      return make_fail<Trace>("TRACE_MALFORMED");
    if (p.elapsed_s < last) return make_fail<Trace>("TRACE_UNSORTED");
    last = p.elapsed_s;
    t.points.push_back(p);
    hashed_rows.push(std::move(hrow));
  }
  // S7 events and S8 installed objects (optional, public metadata)
  if (const json::Value* ev = root.find("events")) {
    if (!ev->is_array()) return make_fail<Trace>("TRACE_MALFORMED");
    for (const auto& e : ev->arr) {
      if (!e.is_array() || e.arr.size() != 2 || !e.arr[0].is_number() || !e.arr[1].is_string())
        return make_fail<Trace>("TRACE_MALFORMED");
      TraceEvent te;
      te.elapsed_s = e.arr[0].as_double();
      te.kind = e.arr[1].str;
      if (!(te.elapsed_s >= 0.0) || (te.kind != "cancel" && te.kind != "resume" && te.kind != "disconnect"))
        return make_fail<Trace>("TRACE_MALFORMED");
      t.events.push_back(te);
    }
  }
  if (const json::Value* inst = root.find("initial_installed")) {
    if (!inst->is_array()) return make_fail<Trace>("TRACE_MALFORMED");
    for (const auto& s : inst->arr) {
      if (!s.is_string()) return make_fail<Trace>("TRACE_MALFORMED");
      t.initial_installed.push_back(s.str);
    }
  }
  // The declared hash binds the points (traces.py load_trace does the same check).
  const std::string computed = rows_hash(hashed_rows);
  if (t.trace_hash.empty()) {
    t.trace_hash = computed;
  } else {
    if (!detail::constant_time_equal(t.trace_hash, computed)) return make_fail<Trace>("TRACE_HASH_MISMATCH");
    t.hash_verified = true;
  }
  return make_ok(std::move(t));
}

Expected<Trace> load_trace_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return make_fail<Trace>("TRACE_IO");
  std::stringstream ss;
  ss << in.rdbuf();
  return parse_trace_json(ss.str());
}

Trace make_constant_trace(double bandwidth_bps, double one_way_delay_s, double duration_s, double cpu_share,
                          const std::string& trace_id) {
  Trace t;
  t.trace_id = trace_id;
  t.family = "S0";
  t.split = "test";
  TracePoint p;
  p.elapsed_s = 0.0;
  p.bandwidth_bps = bandwidth_bps;
  p.one_way_delay_s = one_way_delay_s;
  p.cpu_share = cpu_share;
  t.points.push_back(p);
  if (duration_s > 0) {
    TracePoint q = p;
    q.elapsed_s = duration_s;
    t.points.push_back(q);
  }
  t.declared_duration_s = duration_s > 0 ? duration_s : 0.0;
  t.trace_hash = t.compute_hash();
  return t;
}

std::shared_ptr<hl::WallClockShaper> make_channel_shaper(const Trace& trace, double origin_unix_s) {
  // Same instant on two clocks: Unix time for the shared origin, the monotonic
  // clock for the shaper that highlevel::Channel drives.
  const double origin_monotonic = hl::now_s() - (wall_s() - origin_unix_s);
  return std::make_shared<hl::WallClockShaper>(trace.link_points(), origin_monotonic);
}

// ---------------------------------------------------------------------------
// WallClockShaper: trace-aware adapter over highlevel::WallClockShaper
// ---------------------------------------------------------------------------
WallClockShaper::WallClockShaper(Trace trace, double origin_wall_s)
    : trace_(std::move(trace)), origin_(origin_wall_s), points_(trace_.link_points()),
      fifo_(new hl::WallClockShaper(points_, origin_wall_s)) {}

LinkState WallClockShaper::state_at(double wall_s) const {
  const TracePoint& p = trace_.at(wall_s - origin_);
  LinkState s;
  s.bandwidth_bps = p.bandwidth_bps;
  s.one_way_delay_s = p.one_way_delay_s;
  return s;
}

double WallClockShaper::delay_for(size_t nbytes, double wall_s) const {
  // An idle link is a fresh token bucket: the highlevel shaper integrates the
  // serialization across the trace segments and adds the one-way delay.
  hl::WallClockShaper idle(points_, origin_);
  return idle.delay_for(nbytes, wall_s);
}

double WallClockShaper::serialization_s(size_t nbytes, double wall_s) const {
  // the one-way delay belongs to the state at the END of the serialization, so the serialization
  // time is asked for directly instead of subtracting a delay from delay_for
  hl::WallClockShaper idle(points_, origin_);
  return idle.serialization_time(nbytes, wall_s);
}

double WallClockShaper::apply(size_t nbytes) {
  // FIFO link: the shared highlevel shaper keeps the link-busy time, so a
  // message queues behind the previous ones.
  double delay = fifo_->delay_for(nbytes, wall_s());
  if (delay < 0) delay = 0;
  {
    std::lock_guard<std::mutex> lk(mu_);
    total_delay_ += delay;
    messages_ += 1;
  }
  hl::sleep_s(delay);
  return delay;
}

double WallClockShaper::total_delay_s() const {
  std::lock_guard<std::mutex> lk(mu_);
  return total_delay_;
}

uint64_t WallClockShaper::messages() const {
  std::lock_guard<std::mutex> lk(mu_);
  return messages_;
}

// ---------------------------------------------------------------------------
// ContentionLoad
// ---------------------------------------------------------------------------
void ContentionLoad::start(const Trace& trace, double origin_wall_s, int threads) {
  if (running_.exchange(true)) return;
  trace_ = trace;
  origin_ = origin_wall_s;
  if (threads < 1) threads = 1;
  for (int i = 0; i < threads; ++i) threads_.emplace_back([this]() { run(); });
}

void ContentionLoad::stop() {
  if (!running_.exchange(false)) return;
  for (auto& t : threads_) {
    if (t.joinable()) t.join();
  }
  threads_.clear();
}

void ContentionLoad::run() {
  const double period = 0.020;
  volatile uint64_t sink = 0;
  while (running_) {
    double elapsed = wall_s() - origin_;
    double share = trace_.at(elapsed).cpu_share;
    if (share < 0) share = 0;
    if (share > 1) share = 1;
    double busy = (1.0 - share) * period;
    busy_fraction_.store(1.0 - share);
    if (busy <= 0) {
      sleep_s(period);
      continue;
    }
    double t0 = monotonic_s();
    while (monotonic_s() - t0 < busy && running_) {
      for (int i = 0; i < 1000; ++i) sink = sink * 6364136223846793005ULL + 1442695040888963407ULL;
    }
    double rest = period - busy;
    if (rest > 0) sleep_s(rest);
  }
}

// ---------------------------------------------------------------------------
// TraceDriver
// ---------------------------------------------------------------------------
TraceDriver::TraceDriver(Trace trace, double origin_wall_s)
    : trace_(trace), origin_(origin_wall_s), shaper_(trace, origin_wall_s) {}

double TraceDriver::elapsed_s() const { return wall_s() - origin_; }

void TraceDriver::start_contention(int threads) { load_.start(trace_, origin_, threads); }

void TraceDriver::stop() { load_.stop(); }

}  // namespace platform
}  // namespace opencrab2
