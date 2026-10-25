// opencrab2/platform/session_bridge.cpp (see session_bridge.hpp)
#include <opencrab2/platform/session_bridge.hpp>

#include <opencrab2/highlevel/contract.hpp>
#include <opencrab2/highlevel/costmodel.hpp>
#include <opencrab2/highlevel/json.hpp>

#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <thread>

namespace opencrab2 {
namespace platform {

namespace hl = opencrab2::highlevel;

const char* const kTinySessionGate = "tiny-session";

namespace {

constexpr double kP0RecvTimeoutS = 600.0;    // P0 waits this long for the next P1 request
constexpr double kFinishTimeoutS = 600.0;    // "finish" waits this long for the P0 thread
constexpr double kPollS = 0.05;

std::string temp_dir() {
  const char* base = std::getenv("TEMP");
  if (!base) base = std::getenv("TMPDIR");
  if (!base) base = "/tmp";
  return base;
}

std::string journal_path(const std::string& dir, const std::string& session_id, int party) {
  std::string safe;
  for (char c : session_id) safe.push_back((std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_') ? c : '_');
  return (dir.empty() ? temp_dir() : dir) + "/oc2_tiny_" + safe + "_p" + std::to_string(party) + ".jsonl";
}

hl::Value ledger_binding(const hl::SessionConfig& cfg) {
  return hl::dict({{"session_id", cfg.session_id},
                   {"setup_id", cfg.setup_id},
                   {"model_version", cfg.model_version},
                   {"numeric_contract_hash", cfg.contract.contract_hash()},
                   {"profile_id", cfg.profile_id}});
}

// The raw record of one run: the highlevel loop appends one JSON line to this
// private file (also when it fails), which is read back, brought into the
// evidence schema and only then appended to the shared evidence file.
std::string raw_record_path(const std::string& dir, const std::string& session_id, int party) {
  std::string safe;
  for (char c : session_id) safe.push_back((std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_') ? c : '_');
  return (dir.empty() ? temp_dir() : dir) + "/oc2_tiny_" + safe + "_p" + std::to_string(party) + ".record.jsonl";
}

// Last complete JSON line of a file as a highlevel Value (failure when absent).
Expected<hl::Value> read_last_record(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return Expected<hl::Value>::failure("RECORD_IO: no raw record");
  std::string line, last;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (!line.empty()) last = line;
  }
  if (last.empty()) return Expected<hl::Value>::failure("RECORD_IO: empty raw record");
  return hl::parse_json(last);
}

// The record of a run: the returned one, else the line the loop wrote before
// failing, else a minimal aborted record that states the failure code.
hl::Value record_of_run(const Expected<hl::Value>& run, const std::string& raw_path, const hl::SessionConfig& cfg,
                        const std::string& code_commit) {
  if (run.ok) return run.value;
  auto last = read_last_record(raw_path);
  if (last.ok && last.value.is_dict()) return last.value;
  hl::Value r = evidence::new_record(code_commit);
  r.set("status", "DIAGNOSTIC");
  r.set("profile_id", cfg.profile_id);
  r.set("numeric_contract_hash", cfg.contract.contract_hash());
  r.set("dealer_mode", cfg.dealer_mode);
  r.set("workload_scope", cfg.workload_scope);
  r.set("model_revision", cfg.model_version);
  r.set("clock_basis", "wall-clock-exogenous");
  r.set("policy_id", cfg.policy);
  r.set("run_id", cfg.run_id);
  r.set("repeat", static_cast<int64_t>(cfg.repeat));
  r.set("aborts", static_cast<int64_t>(1));
  r.set("failure", run.code);
  r.set("completed", false);
  r.set("session_id", cfg.session_id);
  return r;
}

hl::Value erase_key(const hl::Value& v, const std::string& key) {
  const hl::Dict* d = std::get_if<hl::Dict>(&v.v);
  if (!d) return v;
  hl::Dict out;
  for (const auto& e : *d) {
    if (e.key != key) out.push_back(e);
  }
  return hl::Value(std::move(out));
}

SessionRecordContext record_context(const TinySessionOptions& opts, const Trace& trace, const std::string& dealer_mode,
                                    bool os_rng, bool weights_disclosed) {
  SessionRecordContext rc;
  rc.record_kind = "web";
  rc.code_commit = opts.code_commit;
  rc.implementation = opts.implementation;
  rc.trace_id = trace.trace_id;
  rc.trace_hash = trace.trace_hash;
  rc.trace_family = trace.family;
  rc.trace_variant = trace.variant;
  rc.trace_split = trace.split;
  rc.raw_measurement_path = opts.out_path;
  rc.dealer_mode = dealer_mode;
  rc.party_rng_os_seeded = os_rng;
  rc.weights_disclosed_to_p1 = weights_disclosed;
  rc.model_from_public_seed = opts.model_from_public_seed;
  return rc;
}

}  // namespace

// ---------------------------------------------------------------------------
// conversions
// ---------------------------------------------------------------------------
Expected<hl::Value> to_highlevel(const wire::Value& v) { return Expected<hl::Value>::success(v.to_highlevel()); }

Expected<wire::Value> to_wire(const hl::Value& v) { return make_ok(wire::Value::from_highlevel(v)); }

// ---------------------------------------------------------------------------
// session helpers shared by the apps
// ---------------------------------------------------------------------------
int private_session_seed() {
  std::vector<uint8_t> b = detail::os_random_bytes(4);
  uint32_t v = static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8) | (static_cast<uint32_t>(b[2]) << 16) |
               (static_cast<uint32_t>(b[3]) << 24);
  return static_cast<int>(v & 0x7fffffffu);
}

hl::SessionTrace to_session_trace(const Trace& trace) {
  hl::SessionTrace t;
  t.trace_id = trace.trace_id.empty() ? std::string("trace") : trace.trace_id;
  t.family = trace.family.empty() ? std::string("S0") : trace.family;
  t.split = trace.split.empty() ? std::string("test") : trace.split;
  t.points = trace.link_points();
  return t;
}

hl::Value finalize_session_record(const hl::Value& session_record, const SessionRecordContext& ctx) {
  hl::Value out = evidence::conform(session_record, ctx.record_kind, ctx.code_commit);
  if (!ctx.code_commit.empty()) out.set("code_commit", ctx.code_commit);
  if (!ctx.trace_id.empty()) out.set("trace_id", ctx.trace_id);
  if (!ctx.trace_hash.empty()) {
    // the hash of the trace file (Python points_hash); the session's own digest stays in extra
    const hl::Value* own = out.get("trace_hash");
    if (own && own->is_string()) evidence::set_extra(out, "session_trace_digest", *own);
    out.set("trace_hash", ctx.trace_hash);
  }
  if (!ctx.trace_family.empty()) out.set("family", ctx.trace_family);
  if (!ctx.trace_variant.empty()) out.set("variant", ctx.trace_variant);
  if (!ctx.trace_split.empty()) out.set("split", ctx.trace_split);
  if (!ctx.raw_measurement_path.empty()) out.set("raw_measurement_path", evidence::record_path(ctx.raw_measurement_path));
  if (!ctx.dealer_mode.empty()) out.set("dealer_mode", ctx.dealer_mode);  // what the dealer announced
  // The session seed is private party randomness: it never appears in a record.
  out = erase_key(out, "seed");
  if (const hl::Value* e = out.get("extra")) out.set("extra", erase_key(*e, "seed"));
  if (!ctx.implementation.empty()) evidence::set_extra(out, "implementation", hl::Value(ctx.implementation));
  evidence::set_extra(out, "party_rng", hl::Value(ctx.party_rng_os_seeded ? "os_csprng" : "private_seed_31bit"));
  evidence::set_extra(out, "public_cpu_contention", hl::Value(ctx.contention));
  if (!ctx.party_rng_os_seeded)
    evidence::withdraw_eligibility(out,
                                   "PARTY_RNG_SEEDED: the highlevel session derives the party share randomness from a 31-bit "
                                   "session seed (private to each party here), below the security parameter",
                                   true);
  if (ctx.weights_disclosed_to_p1)
    evidence::withdraw_eligibility(out, "VERIFY_WEIGHTS_DISCLOSED: the model weights were given to P1 for the output check", true);
  if (ctx.model_from_public_seed)
    evidence::withdraw_eligibility(out, "MODEL_FROM_PUBLIC_SEED: the model weights derive from a public seed", true);
  evidence::enforce(out);
  return out;
}

// ---------------------------------------------------------------------------
// dealer refill through the isolated dealer
// ---------------------------------------------------------------------------
void attach_dealer_connection(hl::DealerClient& client, std::shared_ptr<DealerConnection> conn,
                              std::shared_ptr<std::mutex> conn_mu, int logical_bits) {
  hl::DealerClient* cp = &client;
  client.set_refill([cp, conn, conn_mu, logical_bits](const std::string& type, const hl::Value& params,
                                                     size_t count) -> Expected<Unit> {
    hl::Value p = params;
    if (type == "trunc_faithful" && !p.has("L")) p.set("L", static_cast<int64_t>(logical_bits));
    auto wp = to_wire(p);
    if (!wp.ok) return Expected<Unit>::failure(wp.code);
    Expected<Bundle> b;
    {
      std::lock_guard<std::mutex> lk(*conn_mu);
      b = conn->request(type, wp.value, static_cast<int64_t>(count), "*");
    }
    if (!b.ok) return Expected<Unit>::failure(b.code);
    auto header = to_highlevel(b.value.header);
    if (!header.ok) return Expected<Unit>::failure(header.code);
    return cp->ingest(header.value, b.value.payload, b.value.tag);
  });
}

// ---------------------------------------------------------------------------
// ServerSessionChannel (P0)
// ---------------------------------------------------------------------------
ServerSessionChannel::ServerSessionChannel(std::shared_ptr<hl::WallClockShaper> shaper, std::atomic<bool>* cancelled)
    : hl::Channel(0), cancelled_(cancelled) {
  set_shaper(std::move(shaper));
}

ServerSessionChannel::~ServerSessionChannel() { abort(); }

void ServerSessionChannel::push_inbound(std::vector<uint8_t> bytes) {
  std::lock_guard<std::mutex> lk(mu_);
  inbound_.push_back(std::move(bytes));
  cv_.notify_all();
}

Expected<std::vector<uint8_t>> ServerSessionChannel::pop_outbound(double timeout_s) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(static_cast<int64_t>(timeout_s * 1000.0));
  std::unique_lock<std::mutex> lk(mu_);
  while (outbound_.empty()) {
    if (aborted_) return make_fail<std::vector<uint8_t>>("CHANNEL_ABORTED");
    if (cancelled_ && cancelled_->load()) return make_fail<std::vector<uint8_t>>("CANCELLED");
    if (cv_.wait_until(lk, deadline) == std::cv_status::timeout && outbound_.empty())
      return make_fail<std::vector<uint8_t>>("CHANNEL_TIMEOUT: P0 produced no message");
  }
  std::vector<uint8_t> out = std::move(outbound_.front());
  outbound_.pop_front();
  return make_ok(std::move(out));
}

void ServerSessionChannel::abort() {
  std::lock_guard<std::mutex> lk(mu_);
  aborted_ = true;
  cv_.notify_all();
}

Expected<std::vector<uint8_t>> ServerSessionChannel::wait_inbound(double timeout_s) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(static_cast<int64_t>(timeout_s * 1000.0));
  std::unique_lock<std::mutex> lk(mu_);
  while (inbound_.empty()) {
    if (aborted_) return make_fail<std::vector<uint8_t>>("CHANNEL_ABORTED");
    if (cancelled_ && cancelled_->load()) return make_fail<std::vector<uint8_t>>("CANCELLED");
    if (std::chrono::steady_clock::now() >= deadline) return make_fail<std::vector<uint8_t>>("CHANNEL_TIMEOUT: no message from P1");
    cv_.wait_for(lk, std::chrono::milliseconds(static_cast<int64_t>(kPollS * 1000.0)));
  }
  std::vector<uint8_t> in = std::move(inbound_.front());
  inbound_.pop_front();
  return make_ok(std::move(in));
}

void ServerSessionChannel::apply_shaping(size_t nbytes) {
  auto sh = shaper();
  if (!sh) return;
  double d = sh->delay_for(nbytes, hl::now_s());
  if (!(d >= 0.0)) return;   // an invalid shaper (NaN): send() refuses the frame, nothing to sleep for
  if (d > 0) hl::sleep_s(d);
  stats.shaping_delay_s += d;
}

Expected<Unit> ServerSessionChannel::send(const hl::Value& obj) {
  hl::Bytes enc = hl::encode_value(obj);
  apply_shaping(enc.size() + 8);
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (aborted_) return Expected<Unit>::failure("CHANNEL_ABORTED");
    outbound_.push_back(std::vector<uint8_t>(enc.begin(), enc.end()));
    cv_.notify_all();
  }
  stats.bytes_sent += enc.size() + 8;
  stats.messages_sent += 1;
  pending_send_ = true;
  return ok_status();
}

Expected<hl::Value> ServerSessionChannel::recv() {
  auto in = wait_inbound(kP0RecvTimeoutS);
  if (!in.ok) return Expected<hl::Value>::failure(in.code);
  stats.bytes_recv += in.value.size() + 8;
  stats.messages_recv += 1;
  auto v = hl::decode_value(in.value);
  if (!v.ok) return v;
  if (pending_send_) {
    stats.rounds += 1;
    pending_send_ = false;
  }
  return v;
}

Expected<hl::Value> ServerSessionChannel::exchange(const hl::Value& obj) {
  stats.rounds += 1;
  pending_send_ = false;
  auto s = send(obj);
  pending_send_ = false;
  if (!s.ok) return Expected<hl::Value>::failure(s.code);
  auto in = wait_inbound(kP0RecvTimeoutS);
  if (!in.ok) return Expected<hl::Value>::failure(in.code);
  stats.bytes_recv += in.value.size() + 8;
  stats.messages_recv += 1;
  return hl::decode_value(in.value);
}

void ServerSessionChannel::close() { abort(); }

// ---------------------------------------------------------------------------
// WsSessionChannel (P1)
// ---------------------------------------------------------------------------
WsSessionChannel::WsSessionChannel(WebSocketChannel& ws, std::shared_ptr<hl::WallClockShaper> shaper,
                                   std::function<bool()> cancelled)
    : hl::Channel(1), ws_(ws), cancelled_(std::move(cancelled)) {
  set_shaper(std::move(shaper));
}

Expected<wire::Value> WsSessionChannel::request(const std::string& op, const std::vector<uint8_t>* v, wire::Value extra) {
  wire::Value payload = extra.is_dict() ? extra : wire::Value::make_dict();
  payload.set("op", wire::Value::from_str(op));
  if (v) payload.set("v", wire::Value::from_bytes(*v));
  wire::Value m = wire::Value::make_dict();
  m.set("type", wire::Value::from_str("MSG"));
  m.set("gate_id", wire::Value::from_str(kTinySessionGate));
  m.set("seq", wire::Value::from_int(seq_++));
  m.set("payload", std::move(payload));
  auto s = ws_.send(m);
  if (!s.ok) return forward_fail<wire::Value>(s);
  while (true) {
    auto r = ws_.recv();
    if (!r.ok) {
      if (r.code == "TCP_TIMEOUT") {
        if (cancelled_ && cancelled_()) return make_fail<wire::Value>("CANCELLED");
        continue;
      }
      return r;
    }
    const std::string type = r.value.str_or("type", "");
    if (type == "FLOW" || type == "PONG") continue;
    if (type == "REJECT") return make_fail<wire::Value>(r.value.str_or("code", "REJECTED"));
    if (type != "MSG") return make_fail<wire::Value>("WS_UNEXPECTED_FRAME: " + type);
    const wire::Value* p = r.value.get("payload");
    if (!p || !p->is_dict()) return make_fail<wire::Value>("SESSION_BAD_REPLY");
    return make_ok(*p);
  }
}

void WsSessionChannel::apply_shaping(size_t nbytes) {
  auto sh = shaper();
  if (!sh) return;
  double d = sh->delay_for(nbytes, hl::now_s());
  if (!(d >= 0.0)) return;   // an invalid shaper (NaN): send() refuses the frame, nothing to sleep for
  if (d > 0) hl::sleep_s(d);
  stats.shaping_delay_s += d;
}

Expected<Unit> WsSessionChannel::send(const hl::Value& obj) {
  hl::Bytes enc = hl::encode_value(obj);
  apply_shaping(enc.size() + 8);
  std::vector<uint8_t> bytes(enc.begin(), enc.end());
  auto r = request("send", &bytes);
  if (!r.ok) return Expected<Unit>::failure(r.code);
  stats.bytes_sent += enc.size() + 8;
  stats.messages_sent += 1;
  pending_send_ = true;
  return ok_status();
}

Expected<hl::Value> WsSessionChannel::recv() {
  auto r = request("recv", nullptr);
  if (!r.ok) return Expected<hl::Value>::failure(r.code);
  const wire::Value* v = r.value.get("v");
  if (!v || !v->is_bytes()) return Expected<hl::Value>::failure("SESSION_BAD_REPLY: recv without v");
  stats.bytes_recv += v->bytes.size() + 8;
  stats.messages_recv += 1;
  auto d = hl::decode_value(v->bytes);
  if (!d.ok) return d;
  if (pending_send_) {
    stats.rounds += 1;
    pending_send_ = false;
  }
  return d;
}

Expected<hl::Value> WsSessionChannel::exchange(const hl::Value& obj) {
  stats.rounds += 1;
  pending_send_ = false;
  hl::Bytes enc = hl::encode_value(obj);
  apply_shaping(enc.size() + 8);
  std::vector<uint8_t> bytes(enc.begin(), enc.end());
  auto r = request("xchg", &bytes);
  if (!r.ok) return Expected<hl::Value>::failure(r.code);
  stats.bytes_sent += enc.size() + 8;
  stats.messages_sent += 1;
  const wire::Value* v = r.value.get("v");
  if (!v || !v->is_bytes()) return Expected<hl::Value>::failure("SESSION_BAD_REPLY: xchg without v");
  stats.bytes_recv += v->bytes.size() + 8;
  stats.messages_recv += 1;
  return hl::decode_value(v->bytes);
}

void WsSessionChannel::close() {}

// ---------------------------------------------------------------------------
// P0 handler
// ---------------------------------------------------------------------------
namespace {

struct TinyP0State {
  std::shared_ptr<ServerSessionChannel> ch;
  std::unique_ptr<hl::MaterialLedger> ledger;
  std::unique_ptr<hl::DealerClient> client;
  hl::SessionConfig cfg;
  hl::SessionTrace trace;
  std::thread th;
  std::mutex mu;
  std::condition_variable cv;
  bool done = false;
  Expected<hl::Value> result;
  ~TinyP0State() {
    if (ch) ch->abort();
    if (th.joinable()) th.join();
  }
};

wire::Value tiny_config_value(const hl::TinyConfig& c) {
  wire::Value t = wire::Value::make_dict();
  t.set("n_layers", wire::Value::from_int(c.n_layers));
  t.set("d_model", wire::Value::from_int(c.d_model));
  t.set("n_head", wire::Value::from_int(c.n_head));
  t.set("vocab", wire::Value::from_int(c.vocab));
  t.set("max_pos", wire::Value::from_int(c.max_pos));
  t.set("d_ff", wire::Value::from_int(c.d_ff));
  return t;
}

wire::Value plan_value(const hl::TinyPlan& p) {
  wire::Value t = wire::Value::make_dict();
  t.set("embedding", wire::Value::from_str(p.embedding));
  t.set("linear", wire::Value::from_str(p.linear));
  t.set("argmax_k", wire::Value::from_int(p.argmax_k));
  t.set("max_k", wire::Value::from_int(p.max_k));
  return t;
}

}  // namespace

ProtocolHandler make_tiny_session_handler(std::shared_ptr<hl::TinyWeights> weights, std::shared_ptr<DealerConnection> dealer,
                                          std::vector<uint8_t> hmac_key, ModelServerConfig server_cfg, TinySessionOptions opts) {
  auto conn_mu = std::make_shared<std::mutex>();
  auto costmodel = std::make_shared<hl::FittedCostModel>(hl::historical_prior_model());
  std::string costmodel_error;
  if (!opts.costmodel_path.empty()) {
    auto cm = hl::FittedCostModel::load(opts.costmodel_path);
    if (cm.ok) {
      *costmodel = std::move(cm.value);
    } else {
      costmodel_error = cm.code;  // fail closed at setup: no silent fallback to the prior
    }
  }
  const std::string costmodel_source = opts.costmodel_path.empty() ? "HISTORICAL_PRIOR" : "FITTED_FILE";
  const hl::NumericContract contract = hl::default_contract();
  const hl::Bytes key(hmac_key.begin(), hmac_key.end());
  // The public trace of every session of this server: a file of experiments/traces.py
  // or the stable link of the options. A broken file fails every setup (fail closed).
  auto ptrace = std::make_shared<Trace>(
      make_constant_trace(opts.trace_bandwidth_bps, opts.trace_one_way_delay_s, 0.0, 1.0, opts.trace_id));
  std::string trace_error;
  if (!opts.trace_file.empty()) {
    auto t = load_trace_file(opts.trace_file);
    if (t.ok) {
      *ptrace = std::move(t.value);
      if (ptrace->trace_id.empty()) ptrace->trace_id = opts.trace_id;
    } else {
      trace_error = t.code;
    }
  }
  return [weights, dealer, conn_mu, costmodel, costmodel_error, costmodel_source, server_cfg, opts, contract, key, ptrace,
          trace_error](Session& session, const std::string& gate_id, int64_t, const wire::Value& payload) -> Expected<wire::Value> {
    if (gate_id != kTinySessionGate) return make_fail<wire::Value>("SESSION_BAD_REQUEST: gate " + gate_id);
    if (!payload.is_dict()) return make_fail<wire::Value>("SESSION_BAD_REQUEST: payload");
    const std::string op = payload.str_or("op", "");
    auto* st = static_cast<TinyP0State*>(session.handler_state.get());

    if (op == "setup") {
      if (st) return make_fail<wire::Value>("SESSION_ALREADY_SETUP");
      if (!trace_error.empty()) return make_fail<wire::Value>(trace_error + ": server trace file");
      if (!costmodel_error.empty()) return make_fail<wire::Value>(costmodel_error + ": server cost model file");
      if (server_cfg.numeric_contract_hash != contract.contract_hash())
        return make_fail<wire::Value>("CONTRACT_UNSUPPORTED: the tiny decoder runs under the default contract only");
      if (!dealer->isolated() && !opts.allow_insecure_dealer)
        return make_fail<wire::Value>("DEALER_NOT_ISOLATED: the dealer announced " + dealer->dealer_mode());
      int64_t n_generate = payload.int_or("n_generate", 0);
      std::string policy = payload.str_or("policy", opts.policy);
      double origin_unix = payload.float_or("origin_unix_s", wall_s());
      if (n_generate < 1 || n_generate > 4096 || policy.empty()) return make_fail<wire::Value>("SESSION_BAD_REQUEST: setup");
      if (!(origin_unix > 0.0) || origin_unix > wall_s() + 3600.0 || origin_unix < wall_s() - 3600.0)
        return make_fail<wire::Value>("SESSION_BAD_REQUEST: origin_unix_s out of range");
      auto state = std::make_shared<TinyP0State>();
      hl::SessionConfig& cfg = state->cfg;
      cfg.model = "tiny-decoder@" + weights->digest().substr(0, 16);
      cfg.n_layers = static_cast<int>(weights->cfg.n_layers);
      cfg.n_generate = static_cast<int>(n_generate);
      cfg.policy = policy;
      cfg.trace_id = ptrace->trace_id;
      cfg.seed = private_session_seed();  // private to P0, never sent
      cfg.contract = contract;
      cfg.profile_id = server_cfg.profile_id;
      cfg.length_bucket = static_cast<int>(session.binding.length_bucket);
      cfg.tiny = weights->cfg;
      cfg.initial_plan = hl::TinyPlan{};
      cfg.session_id = session.binding.session_id;
      cfg.setup_id = server_cfg.setup_id;
      cfg.model_version = server_cfg.model_version;
      cfg.dealer_mode = dealer->dealer_mode().empty() ? "INSECURE_TEST_ONLY_SHARED_SEED" : dealer->dealer_mode();
      std::vector<uint8_t> tk = detail::os_random_bytes(32);
      cfg.telemetry_key = hl::Bytes(tk.begin(), tk.end());
      cfg.run_id = "web-" + session.binding.session_id;
      cfg.workload_scope = "full-model";
      cfg.code_commit = opts.code_commit;
      cfg.controller = opts.controller;
      if (ptrace->max_stock_refill_bps() > 0) cfg.refill_bytes_per_s = ptrace->max_stock_refill_bps();
      const bool os_rng = session_caps::request_os_party_rng(cfg);
      session_caps::set_cancel_flag(cfg, static_cast<const std::atomic<bool>*>(&session.cancelled), 0);
      state->trace = to_session_trace(*ptrace);
      std::shared_ptr<hl::WallClockShaper> shaper;
      if (opts.shape) shaper = make_channel_shaper(*ptrace, origin_unix);
      state->ch = std::make_shared<ServerSessionChannel>(shaper, &session.cancelled);
      state->ledger =
          std::make_unique<hl::MaterialLedger>(journal_path(opts.journal_dir, cfg.session_id, 0), ledger_binding(cfg));
      auto lo = state->ledger->open();
      if (!lo.ok) return make_fail<wire::Value>(lo.code);
      if (state->ledger->needs_recovery()) {
        // an existing journal with RESERVED entries: the restart rule burns them (ledger.hpp)
        auto rec = state->ledger->recover();
        if (!rec.ok) return make_fail<wire::Value>(rec.code);
      }
      state->client = std::make_unique<hl::DealerClient>(0, *state->ledger, key, contract.contract_hash());
      attach_dealer_connection(*state->client, dealer, conn_mu, contract.logical_bits);
      TinyP0State* raw = state.get();
      std::shared_ptr<hl::TinyWeights> w = weights;
      std::shared_ptr<hl::FittedCostModel> cm = costmodel;
      const std::string raw_path = raw_record_path(opts.journal_dir, cfg.session_id, 0);
      std::remove(raw_path.c_str());
      const SessionRecordContext rc = record_context(opts, *ptrace, cfg.dealer_mode, os_rng, false);
      const std::string out_path = opts.out_path;
      auto on_record = opts.on_record;
      state->th = std::thread([raw, w, cm, raw_path, out_path, rc, on_record]() {
        Expected<hl::Value> r = hl::run_session_party(0, raw->cfg, *raw->ch, *raw->client, *raw->ledger, *cm, raw->trace,
                                                      w.get(), nullptr, raw_path, nullptr);
        hl::Value rec = finalize_session_record(record_of_run(r, raw_path, raw->cfg, rc.code_commit), rc);
        std::remove(raw_path.c_str());
        if (!out_path.empty()) {
          auto ap = evidence::append(out_path, rec);
          if (!ap.ok && r.ok) r = Expected<hl::Value>::failure(ap.code);
        }
        if (on_record) on_record(rec);
        if (r.ok) r.value = rec;
        std::lock_guard<std::mutex> lk(raw->mu);
        raw->result = std::move(r);
        raw->done = true;
        raw->cv.notify_all();
      });
      session.handler_state = std::static_pointer_cast<void>(state);

      wire::Value reply = wire::Value::make_dict();
      reply.set("ok", wire::Value::from_bool(true));
      reply.set("tiny", tiny_config_value(cfg.tiny));
      reply.set("model", wire::Value::from_str(cfg.model));
      reply.set("model_version", wire::Value::from_str(cfg.model_version));
      reply.set("telemetry_key", wire::Value::from_bytes(tk));
      reply.set("controller", wire::Value::from_str(cfg.controller.to_json().dump()));
      reply.set("costmodel", wire::Value::from_str(costmodel->to_json().dump()));
      reply.set("costmodel_source", wire::Value::from_str(costmodel_source));
      // the public trace: rows in five columns, the file's hash next to it
      Trace sent = *ptrace;
      sent.trace_hash.clear();
      reply.set("trace", wire::Value::from_str(sent.to_json().dump()));
      reply.set("trace_hash", wire::Value::from_str(ptrace->trace_hash));
      reply.set("shape", wire::Value::from_bool(opts.shape));
      reply.set("plan", plan_value(cfg.initial_plan));
      reply.set("policy", wire::Value::from_str(policy));
      reply.set("code_commit", wire::Value::from_str(opts.code_commit));
      reply.set("dealer_mode", wire::Value::from_str(cfg.dealer_mode));
      return make_ok(std::move(reply));
    }

    if (!st) return make_fail<wire::Value>("SESSION_NOT_SETUP");
    if (op == "send" || op == "xchg") {
      const wire::Value* v = payload.get("v");
      if (!v || !v->is_bytes()) return make_fail<wire::Value>("SESSION_BAD_REQUEST: " + op + " without v");
      st->ch->push_inbound(v->bytes);
      wire::Value reply = wire::Value::make_dict();
      if (op == "send") {
        reply.set("ok", wire::Value::from_bool(true));
        return make_ok(std::move(reply));
      }
      auto out = st->ch->pop_outbound(kP0RecvTimeoutS);
      if (!out.ok) return forward_fail<wire::Value>(out);
      reply.set("v", wire::Value::from_bytes(std::move(out.value)));
      return make_ok(std::move(reply));
    }
    if (op == "recv") {
      auto out = st->ch->pop_outbound(kP0RecvTimeoutS);
      if (!out.ok) return forward_fail<wire::Value>(out);
      wire::Value reply = wire::Value::make_dict();
      reply.set("v", wire::Value::from_bytes(std::move(out.value)));
      return make_ok(std::move(reply));
    }
    if (op == "finish") {
      {
        std::unique_lock<std::mutex> lk(st->mu);
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(static_cast<int64_t>(kFinishTimeoutS * 1000.0));
        while (!st->done) {
          if (st->cv.wait_until(lk, deadline) == std::cv_status::timeout && !st->done) {
            st->ch->abort();
            return make_fail<wire::Value>("CHANNEL_TIMEOUT: P0 session did not finish");
          }
        }
      }
      if (st->th.joinable()) st->th.join();
      wire::Value reply = wire::Value::make_dict();
      reply.set("ok", wire::Value::from_bool(st->result.ok));
      reply.set("code", wire::Value::from_str(st->result.code));
      if (st->result.ok) {
        auto b = st->result.value.get_int("bytes");
        auto r = st->result.value.get_int("rounds");
        auto t = st->result.value.get_number("T_online_s");
        reply.set("bytes", wire::Value::from_int(b.ok ? b.value : -1));
        reply.set("rounds", wire::Value::from_int(r.ok ? r.value : -1));
        reply.set("T_online_s", wire::Value::from_float(t.ok ? t.value : -1.0));
        reply.set("material_burned", wire::Value::from_int(static_cast<int64_t>(st->ledger->bytes_burned())));
      }
      return make_ok(std::move(reply));
    }
    return make_fail<wire::Value>("SESSION_BAD_REQUEST: op " + op);
  };
}

// ---------------------------------------------------------------------------
// P1 engine
// ---------------------------------------------------------------------------
ClientEngine make_tiny_session_engine(TinySessionOptions opts) {
  return [opts](JobContext& ctx) -> Expected<bool> {
    ctx.set_engine_name("tiny_decoder");
    const SidecarConfig& scfg = ctx.config();
    const hl::NumericContract contract = hl::default_contract();
    if (scfg.numeric_contract_hash != contract.contract_hash())
      return make_fail<bool>("CONTRACT_UNSUPPORTED: the tiny decoder runs under the default contract only");
    // 1. dealer connection as recipient 1
    auto dconn = DealerConnection::connect(scfg.dealer_host, scfg.dealer_port, 1, scfg.setup_id, scfg.numeric_contract_hash,
                                           scfg.dealer_key, 30.0, 20);
    if (!dconn.ok) return forward_fail<bool>(dconn);
    auto dealer = std::make_shared<DealerConnection>(std::move(dconn.value));
    if (!dealer->isolated() && !opts.allow_insecure_dealer) {
      dealer->close();
      return make_fail<bool>("DEALER_NOT_ISOLATED: the dealer announced " + dealer->dealer_mode());
    }
    auto conn_mu = std::make_shared<std::mutex>();
    // 2. WebSocket session with the model server (HELLO binding)
    auto ws = websocket_connect_url(scfg.server_ws_url, "", 10.0);
    if (!ws.ok) return forward_fail<bool>(ws);
    WebSocketChannel wsch(std::move(ws.value), 1);
    std::vector<uint8_t> nonce = detail::os_random_bytes(16);
    wire::Value hello = wire::Value::make_dict();
    hello.set("type", wire::Value::from_str("HELLO"));
    hello.set("session_id", wire::Value::from_str(ctx.session_id()));
    hello.set("model_version", wire::Value::from_str(scfg.model_version));
    hello.set("numeric_contract_hash", wire::Value::from_str(scfg.numeric_contract_hash));
    hello.set("profile_id", wire::Value::from_str(scfg.profile_id));
    hello.set("nonce", wire::Value::from_bytes(nonce));
    hello.set("length_bucket", wire::Value::from_int(scfg.length_bucket));
    auto hs = wsch.send(hello);
    if (!hs.ok) return forward_fail<bool>(hs);
    auto hok = wsch.recv();
    if (!hok.ok) return forward_fail<bool>(hok);
    if (hok.value.str_or("type", "") != "HELLO_OK") return make_fail<bool>(hok.value.str_or("code", "HELLO_REJECTED"));
    json::Value binding = json::Value::object();
    binding.set("session_id", json::Value::string(ctx.session_id()));
    binding.set("model_version", json::Value::string(scfg.model_version));
    binding.set("numeric_contract_hash", json::Value::string(scfg.numeric_contract_hash));
    binding.set("profile_id", json::Value::string(scfg.profile_id));
    binding.set("nonce", json::Value::string(detail::hex_encode(nonce)));
    binding.set("length_bucket", json::Value::integer(scfg.length_bucket));
    binding.set("ttl_s", json::Value::number(hok.value.float_or("ttl_s", 0)));
    binding.set("engine", json::Value::string("tiny_decoder"));
    ctx.set_binding(binding);
    wsch.ws().set_timeout(1.0);
    // 3. setup (public session parameters); the prompt never leaves this process
    const double origin_unix = wall_s();
    WsSessionChannel ch(wsch, nullptr, [&ctx]() { return ctx.cancelled(); });
    wire::Value setup = wire::Value::make_dict();
    setup.set("n_generate", wire::Value::from_int(ctx.n_tokens()));
    setup.set("policy", wire::Value::from_str(opts.policy));
    setup.set("origin_unix_s", wire::Value::from_float(origin_unix));
    auto sr = ch.request("setup", nullptr, setup);
    if (!sr.ok) return forward_fail<bool>(sr);
    const wire::Value& rep = sr.value;
    const wire::Value* tiny = rep.get("tiny");
    const wire::Value* tkey = rep.get("telemetry_key");
    if (!rep.bool_or("ok", false) || !tiny || !tiny->is_dict() || !tkey || !tkey->is_bytes())
      return make_fail<bool>("SESSION_BAD_REPLY: setup");
    hl::SessionConfig cfg;
    cfg.tiny.n_layers = static_cast<uint32_t>(tiny->int_or("n_layers", 0));
    cfg.tiny.d_model = static_cast<uint32_t>(tiny->int_or("d_model", 0));
    cfg.tiny.n_head = static_cast<uint32_t>(tiny->int_or("n_head", 0));
    cfg.tiny.vocab = static_cast<uint32_t>(tiny->int_or("vocab", 0));
    cfg.tiny.max_pos = static_cast<uint32_t>(tiny->int_or("max_pos", 0));
    cfg.tiny.d_ff = static_cast<uint32_t>(tiny->int_or("d_ff", 0));
    if (cfg.tiny.n_layers == 0 || cfg.tiny.d_model == 0 || cfg.tiny.n_head == 0 || cfg.tiny.vocab == 0 || cfg.tiny.max_pos == 0 ||
        cfg.tiny.d_ff == 0)
      return make_fail<bool>("SESSION_BAD_REPLY: tiny config");
    // the public trace announced by P0 (verified shape, P0's hash adopted for the record)
    auto tr = parse_trace_json(rep.str_or("trace", ""));
    if (!tr.ok) return make_fail<bool>("SESSION_BAD_REPLY: trace (" + tr.code + ")");
    Trace ptrace = std::move(tr.value);
    if (!rep.str_or("trace_hash", "").empty()) ptrace.trace_hash = rep.str_or("trace_hash", "");
    cfg.model = rep.str_or("model", "tiny-decoder");
    cfg.n_layers = static_cast<int>(cfg.tiny.n_layers);
    cfg.n_generate = static_cast<int>(ctx.n_tokens());
    cfg.policy = rep.str_or("policy", opts.policy);
    cfg.trace_id = ptrace.trace_id;
    cfg.seed = private_session_seed();  // private to P1, never sent
    cfg.contract = contract;
    cfg.profile_id = scfg.profile_id;
    cfg.length_bucket = static_cast<int>(scfg.length_bucket);
    const wire::Value* plan = rep.get("plan");
    if (plan && plan->is_dict()) {
      cfg.initial_plan.embedding = plan->str_or("embedding", "generic");
      cfg.initial_plan.linear = plan->str_or("linear", "generic");
      cfg.initial_plan.argmax_k = static_cast<int>(plan->int_or("argmax_k", 2));
      cfg.initial_plan.max_k = static_cast<int>(plan->int_or("max_k", 2));
    }
    cfg.session_id = ctx.session_id();
    cfg.setup_id = scfg.setup_id;
    cfg.model_version = scfg.model_version;
    cfg.dealer_mode = dealer->dealer_mode().empty() ? "INSECURE_TEST_ONLY_SHARED_SEED" : dealer->dealer_mode();
    cfg.telemetry_key = hl::Bytes(tkey->bytes.begin(), tkey->bytes.end());
    cfg.run_id = "web-" + ctx.session_id();
    cfg.workload_scope = "full-model";
    cfg.code_commit = opts.code_commit;
    if (ptrace.max_stock_refill_bps() > 0) cfg.refill_bytes_per_s = ptrace.max_stock_refill_bps();
    const bool os_rng = session_caps::request_os_party_rng(cfg);
    session_caps::set_cancel_flag(cfg, ctx.cancel_flag(), 0);
    {
      auto cj = hl::Json::parse(rep.str_or("controller", "{}"));
      if (!cj.ok) return make_fail<bool>("SESSION_BAD_REPLY: controller json");
      auto cc = hl::ControllerConfig::from_json(cj.value);
      if (!cc.ok) return forward_fail<bool>(cc);
      cfg.controller = cc.value;
    }
    // prompt bytes -> token ids modulo the public vocabulary (local only)
    const std::string& prompt = ctx.prompt();
    for (char c : prompt) cfg.prompt_ids.push_back(static_cast<int>(static_cast<uint8_t>(c) % cfg.tiny.vocab));
    if (cfg.prompt_ids.empty()) cfg.prompt_ids.push_back(0);
    if (cfg.prompt_ids.size() + static_cast<size_t>(cfg.n_generate) > cfg.tiny.max_pos)
      return make_fail<bool>("SESSION_PROMPT_TOO_LONG: prompt + n_tokens exceeds max_pos");
    cfg.on_token = [&ctx](int, int tok) { ctx.emit_token(tok, "<tiny:" + std::to_string(tok) + ">"); };
    // trace and shaping (adopted from P0; the origin is this party's announcement)
    hl::SessionTrace trace = to_session_trace(ptrace);
    if (rep.bool_or("shape", opts.shape)) ch.set_shaper(make_channel_shaper(ptrace, origin_unix));
    // ledger and dealer client
    hl::MaterialLedger ledger(journal_path(opts.journal_dir, cfg.session_id, 1), ledger_binding(cfg));
    auto lo = ledger.open();
    if (!lo.ok) return forward_fail<bool>(lo);
    if (ledger.needs_recovery()) {
      // an existing journal with RESERVED entries: the restart rule burns them (ledger.hpp)
      auto rec = ledger.recover();
      if (!rec.ok) return forward_fail<bool>(rec);
    }
    hl::DealerClient client(1, ledger, hl::Bytes(scfg.dealer_key.begin(), scfg.dealer_key.end()), contract.contract_hash());
    attach_dealer_connection(client, dealer, conn_mu, contract.logical_bits);
    // test mode: plaintext reference from the disclosed weights
    std::vector<int> reference;
    const std::vector<int>* ref_ptr = nullptr;
    const bool weights_disclosed = !opts.verify_weights_path.empty();
    if (weights_disclosed) {
      auto W = hl::TinyWeights::load(opts.verify_weights_path);
      if (!W.ok) return forward_fail<bool>(W);
      if ("tiny-decoder@" + W.value.digest().substr(0, 16) != cfg.model)
        return make_fail<bool>("VERIFY_WEIGHTS_MISMATCH: the verification weights differ from the server's model");
      hl::TinyReference ref(W.value);
      reference = ref.generate(cfg.prompt_ids, cfg.n_generate);
      ref_ptr = &reference;
    }
    // the public cost model announced by P0 (identical decisions need identical models)
    hl::FittedCostModel cm = hl::historical_prior_model();
    if (rep.has("costmodel")) {
      auto cmj = hl::Json::parse(rep.str_or("costmodel", ""));
      auto cmr = cmj.ok ? hl::FittedCostModel::from_json(cmj.value) : Expected<hl::FittedCostModel>::failure(cmj.code);
      if (!cmr.ok) return make_fail<bool>("SESSION_BAD_REPLY: costmodel (" + cmr.code + ")");
      cm = std::move(cmr.value);
    }
    // The highlevel loop writes its line to a private file; the conformed record is
    // appended to the shared evidence file after "finish" returns (P0 has appended its
    // own line by then; the lock file of evidence::append covers the remaining race).
    const std::string raw_path = raw_record_path(opts.journal_dir, cfg.session_id, 1);
    std::remove(raw_path.c_str());
    Expected<hl::Value> run = hl::run_session_party(1, cfg, ch, client, ledger, cm, trace, nullptr, ref_ptr, raw_path, nullptr);
    auto fin = ch.request("finish", nullptr);
    SessionRecordContext rc = record_context(opts, ptrace, cfg.dealer_mode, os_rng, weights_disclosed);
    hl::Value rec = finalize_session_record(record_of_run(run, raw_path, cfg, rc.code_commit), rc);
    std::remove(raw_path.c_str());
    // public counters of this job for GET /v1/status (sizes and seconds only)
    ctx.report_metric("material_transferred_bytes", static_cast<double>(client.bytes_ingested()));
    ctx.report_metric("material_consumed_bytes", static_cast<double>(client.bytes_consumed()));
    ctx.report_metric("material_burned_bytes", static_cast<double>(ledger.bytes_burned()));
    for (const char* k : {"T_cold_s", "T_warm_s", "T_online_s"}) {
      auto v = rec.get_number(k);
      if (v.ok) ctx.report_metric(k, v.value);
    }
    {
      auto b = rec.get_int("bytes");
      auto r = rec.get_int("rounds");
      if (b.ok) ctx.report_metric("protocol_bytes", static_cast<double>(b.value));
      if (r.ok) ctx.report_metric("protocol_rounds", static_cast<double>(r.value));
    }
    Expected<bool> result = make_ok(true);
    if (!opts.out_path.empty()) {
      auto ap = evidence::append(opts.out_path, rec);
      if (!ap.ok) result = make_fail<bool>(ap.code);
    }
    if (opts.on_record) opts.on_record(rec);
    if (!run.ok) result = make_fail<bool>(run.code);
    if (result.ok && !fin.ok) result = make_fail<bool>("P0_FINISH: " + fin.code);
    if (result.ok && !fin.value.bool_or("ok", false)) result = make_fail<bool>("P0_FAILED: " + fin.value.str_or("code", ""));
    wsch.close(1000);
    dealer->close();
    return result;
  };
}

}  // namespace platform
}  // namespace opencrab2
