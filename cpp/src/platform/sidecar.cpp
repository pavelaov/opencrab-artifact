// opencrab2/platform/sidecar.cpp (see sidecar.hpp)
#include <opencrab2/platform/sidecar.hpp>

#include <opencrab2/platform/dealer_service.hpp>
#include <opencrab2/platform/model_server.hpp>
#include <opencrab2/platform/websocket.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace opencrab2 {
namespace platform {

// ---------------------------------------------------------------------------
// JobStatus / JobContext
// ---------------------------------------------------------------------------
json::Value JobStatus::to_json() const {
  json::Value o = json::Value::object();
  o.set("job_id", json::Value::string(job_id));
  o.set("session_id", json::Value::string(session_id));
  o.set("state", json::Value::string(state));
  o.set("error_code", json::Value::string(error_code));
  o.set("engine", json::Value::string(engine));
  o.set("n_tokens", json::Value::integer(n_tokens));
  o.set("tokens_emitted", json::Value::integer(tokens_emitted));
  o.set("created_wall_s", json::Value::number(created_wall_s));
  o.set("started_wall_s", json::Value::number(started_wall_s));
  o.set("finished_wall_s", json::Value::number(finished_wall_s));
  o.set("ttft_s", json::Value::number(ttft_s));
  o.set("cold", json::Value::boolean(cold));
  json::Value it = json::Value::object();
  it.set("n", json::Value::integer(tokens_emitted > 1 ? tokens_emitted - 1 : 0));
  it.set("mean", json::Value::number(inter_token_mean_s()));
  it.set("p50", json::Value::number(inter_token_p50_s));
  it.set("max", json::Value::number(tokens_emitted > 1 ? inter_token_max_s : -1.0));
  o.set("inter_token_s", it);
  o.set("completed", json::Value::boolean(completed()));
  o.set("completion_s", json::Value::number(completion_s()));
  json::Value c = json::Value::object();
  c.set("requested", json::Value::boolean(cancel_requested_wall_s > 0));
  c.set("latency_s", json::Value::number(cancel_latency_s()));
  c.set("tokens_after_cancel", json::Value::integer(tokens_after_cancel));
  o.set("cancel", c);
  o.set("metrics", metrics.is_object() ? metrics : json::Value::object());
  o.set("binding", binding);
  return o;
}

namespace {

// Median of a sample (-1 when empty); the sample is consumed.
double median_of(std::vector<double> v) {
  if (v.empty()) return -1.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

json::Value summary_of(const std::vector<double>& v) {
  json::Value o = json::Value::object();
  double sum = 0.0, mx = -1.0;
  for (double x : v) {
    sum += x;
    if (x > mx) mx = x;
  }
  o.set("n", json::Value::integer(static_cast<int64_t>(v.size())));
  o.set("mean", json::Value::number(v.empty() ? -1.0 : sum / static_cast<double>(v.size())));
  o.set("p50", json::Value::number(median_of(v)));
  o.set("max", json::Value::number(mx));
  return o;
}

// Gaps between consecutive token emission times.
std::vector<double> token_gaps(const std::vector<TokenEvent>& tokens) {
  std::vector<double> gaps;
  for (size_t i = 1; i < tokens.size(); ++i) gaps.push_back(tokens[i].t_wall_s - tokens[i - 1].t_wall_s);
  return gaps;
}

bool code_is(const std::string& code, const char* prefix) { return code.compare(0, std::strlen(prefix), prefix) == 0; }

}  // namespace

void JobContext::emit_token(int64_t token_id, const std::string& text) {
  if (emit_) emit_(token_id, text);
}
void JobContext::set_binding(const json::Value& b) {
  if (bind_) bind_(b);
}
void JobContext::set_engine_name(const std::string& name) {
  if (name_) name_(name);
}
void JobContext::report_metric(const std::string& name, double value) {
  if (metric_) metric_(name, value);
}

std::string generate_bearer_token() { return detail::hex_encode(detail::os_random_bytes(32)); }

// ---------------------------------------------------------------------------
// Sidecar
// ---------------------------------------------------------------------------
Sidecar::Sidecar(SidecarConfig cfg, ClientEngine engine) : cfg_(std::move(cfg)), engine_(std::move(engine)) {
  if (cfg_.token.empty()) cfg_.token = generate_bearer_token();
  if (!cfg_.log) {
    cfg_.log = [](const std::string& line) { std::fprintf(stderr, "%s\n", line.c_str()); };
  }
}

Sidecar::~Sidecar() { stop(); }

uint16_t Sidecar::port() const { return http_ ? http_->port() : cfg_.port; }

std::string Sidecar::launch_url() const {
  return "http://" + cfg_.host + ":" + std::to_string(port()) + "/#token=" + cfg_.token;
}

void Sidecar::log(const std::string& event, const std::map<std::string, json::Value>& fields) {
  json::Value o = json::Value::object();
  o.set("ts", json::Value::number(wall_s()));
  o.set("component", json::Value::string("sidecar"));
  o.set("event", json::Value::string(event));
  for (const auto& kv : fields) o.set(kv.first, kv.second);
  cfg_.log(o.dump(false));
}

Expected<bool> Sidecar::start() {
  if (cfg_.host != "127.0.0.1" && cfg_.host != "localhost" && cfg_.host != "::1") {
    return make_fail<bool>("SIDECAR_MUST_BIND_LOOPBACK");
  }
  auto l = TcpListener::bind(cfg_.host, cfg_.port, 32);
  if (!l.ok) return forward_fail<bool>(l);
  if (cfg_.origin_allowlist.empty()) {
    cfg_.origin_allowlist.push_back("http://" + cfg_.host + ":" + std::to_string(l.value.port()));
  }
  http_.reset(new HttpServer(std::move(l.value), [this](HttpServer::Exchange& ex) { handle(ex); }));
  auto s = http_->start();
  if (!s.ok) return s;
  started_wall_s_ = wall_s();
  log("start", {{"host", json::Value::string(cfg_.host)}, {"port", json::Value::integer(http_->port())},
                {"origins", json::Value::integer(static_cast<int64_t>(cfg_.origin_allowlist.size()))}});
  return make_ok(true);
}

void Sidecar::stop() {
  if (!http_) return;
  std::vector<std::shared_ptr<Job>> jobs;
  {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& kv : jobs_) {
      kv.second->cancel = true;
      jobs.push_back(kv.second);
    }
  }
  for (auto& j : jobs) {
    if (j->worker.joinable()) j->worker.join();
  }
  http_->stop();
  http_.reset();
}

Sidecar::Stats Sidecar::stats() const {
  std::lock_guard<std::mutex> lk(mu_);
  return stats_;
}

std::shared_ptr<Sidecar::Job> Sidecar::find_job(const std::string& id) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = jobs_.find(id);
  return it == jobs_.end() ? nullptr : it->second;
}

Expected<JobStatus> Sidecar::job_status(const std::string& job_id) const {
  auto j = find_job(job_id);
  if (!j) return make_fail<JobStatus>("JOB_UNKNOWN");
  std::lock_guard<std::mutex> lk(j->mu);
  return make_ok(j->status);
}

Expected<std::vector<TokenEvent>> Sidecar::job_tokens(const std::string& job_id) const {
  auto j = find_job(job_id);
  if (!j) return make_fail<std::vector<TokenEvent>>("JOB_UNKNOWN");
  std::lock_guard<std::mutex> lk(j->mu);
  return make_ok(j->tokens);
}

Expected<bool> Sidecar::cancel_job(const std::string& job_id) {
  auto j = find_job(job_id);
  if (!j) return make_fail<bool>("JOB_UNKNOWN");
  j->cancel = true;
  {
    std::lock_guard<std::mutex> lk(j->mu);
    if (j->status.cancel_requested_wall_s < 0 && (j->status.state == "queued" || j->status.state == "running"))
      j->status.cancel_requested_wall_s = wall_s();
    if (j->status.state == "queued") {
      j->status.state = "cancelled";
      j->status.finished_wall_s = wall_s();
    }
    j->cv.notify_all();
  }
  std::lock_guard<std::mutex> lk(mu_);
  stats_.jobs_cancelled++;
  return make_ok(true);
}

Expected<JobStatus> Sidecar::wait_job(const std::string& job_id, double timeout_s) const {
  auto j = find_job(job_id);
  if (!j) return make_fail<JobStatus>("JOB_UNKNOWN");
  std::unique_lock<std::mutex> lk(j->mu);
  auto deadline = std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                                          std::chrono::duration<double>(timeout_s));
  while (j->status.state == "queued" || j->status.state == "running") {
    if (j->cv.wait_until(lk, deadline) == std::cv_status::timeout) break;
  }
  return make_ok(j->status);
}

HttpResponse Sidecar::with_cors(HttpResponse r, const HttpRequest& req) const {
  const std::string* origin = req.header("origin");
  if (origin && origin_allowed(req, cfg_.origin_allowlist, false)) {
    r.set_header("Access-Control-Allow-Origin", *origin);
    r.set_header("Vary", "Origin");
  }
  r.set_header("Cache-Control", "no-store");
  r.set_header("X-Content-Type-Options", "nosniff");
  return r;
}

void Sidecar::handle(HttpServer::Exchange& ex) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stats_.requests++;
  }
  const HttpRequest& req = ex.request;
  bool api = req.path.compare(0, 4, "/v1/") == 0;
  if (!api) {
    handle_static(ex);
    return;
  }
  // 1. Origin (403), 2. token (401)
  if (!origin_allowed(req, cfg_.origin_allowlist, true)) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stats_.rejected_origin++;
    }
    log("rejected_origin", {{"path", json::Value::string(req.path)}});
    ex.respond(HttpResponse::error(403, "ORIGIN_REJECTED"));
    return;
  }
  if (req.method == "OPTIONS") {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stats_.preflights++;
    }
    HttpResponse r;
    r.status = 204;
    r.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    r.set_header("Access-Control-Allow-Headers", "Authorization, Content-Type");
    r.set_header("Access-Control-Max-Age", "600");
    ex.respond(with_cors(r, req));
    return;
  }
  if (!bearer_token_ok(req, cfg_.token)) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stats_.rejected_auth++;
    }
    log("rejected_auth", {{"path", json::Value::string(req.path)}});
    HttpResponse r = HttpResponse::error(401, "TOKEN_REQUIRED");
    r.set_header("WWW-Authenticate", "Bearer realm=\"opencrab2-sidecar\"");
    ex.respond(with_cors(r, req));
    return;
  }
  if (req.path == "/v1/generate" && req.method == "POST") return handle_generate(ex);
  if (req.path == "/v1/cancel" && req.method == "POST") return handle_cancel(ex);
  if (req.path == "/v1/stream" && req.method == "GET") return handle_stream(ex);
  if (req.path == "/v1/status" && req.method == "GET") return handle_status(ex);
  ex.respond(with_cors(HttpResponse::error(404, "NOT_FOUND"), req));
}

void Sidecar::handle_generate(HttpServer::Exchange& ex) {
  const HttpRequest& req = ex.request;
  auto j = json::parse(req.body);
  if (!j.ok || !j.value.is_object()) {
    ex.respond(with_cors(HttpResponse::error(400, "SIDECAR_BAD_JSON"), req));
    return;
  }
  std::string prompt = j.value.str_or("prompt", "");
  int64_t n_tokens = j.value.int_or("n_tokens", 16);
  if (n_tokens <= 0 || n_tokens > 4096) {
    ex.respond(with_cors(HttpResponse::error(400, "SIDECAR_BAD_REQUEST"), req));
    return;
  }
  std::string session_id = j.value.str_or("session_id", "");
  if (session_id.empty()) session_id = "s-" + detail::hex_encode(detail::os_random_bytes(8));
  auto job = std::make_shared<Job>();
  std::string job_id;
  {
    std::lock_guard<std::mutex> lk(mu_);
    size_t active = 0;
    for (const auto& kv : jobs_) {
      std::lock_guard<std::mutex> jl(kv.second->mu);
      if (kv.second->status.state == "queued" || kv.second->status.state == "running") ++active;
    }
    if (active >= cfg_.max_jobs) {
      ex.respond(with_cors(HttpResponse::error(429, "JOB_LIMIT"), req));
      return;
    }
    char buf[32];
    std::snprintf(buf, sizeof buf, "job-%06llu", static_cast<unsigned long long>(++job_counter_));
    job_id = buf;
    job->status.job_id = job_id;
    job->status.session_id = session_id;
    job->status.state = "queued";
    job->status.n_tokens = n_tokens;
    job->status.created_wall_s = wall_s();
    job->status.cold = (stats_.jobs_started == 0);
    job->status.binding = json::Value::object();
    job->status.metrics = json::Value::object();
    job->prompt = prompt;  // stays in this process
    jobs_[job_id] = job;
    stats_.jobs_started++;
  }
  job->worker = std::thread([this, job]() { run_job(job); });
  json::Value o = json::Value::object();
  o.set("job_id", json::Value::string(job_id));
  o.set("session_id", json::Value::string(session_id));
  o.set("n_tokens", json::Value::integer(n_tokens));
  log("generate", {{"job_id", json::Value::string(job_id)}, {"session_id", json::Value::string(session_id)},
                   {"prompt_bytes", json::Value::integer(static_cast<int64_t>(prompt.size()))},
                   {"n_tokens", json::Value::integer(n_tokens)}});
  ex.respond(with_cors(HttpResponse::json(202, o), req));
}

void Sidecar::run_job(std::shared_ptr<Job> job) {
  {
    std::lock_guard<std::mutex> lk(job->mu);
    if (job->status.state != "queued") return;
    job->status.state = "running";
    job->status.started_wall_s = wall_s();
    job->cv.notify_all();
  }
  JobContext ctx;
  ctx.prompt_ = job->prompt;
  ctx.n_tokens_ = job->status.n_tokens;
  ctx.session_id_ = job->status.session_id;
  ctx.job_id_ = job->status.job_id;
  ctx.cfg_ = &cfg_;
  ctx.cancel_ = &job->cancel;
  ctx.emit_ = [job](int64_t id, const std::string& text) {
    std::lock_guard<std::mutex> lk(job->mu);
    TokenEvent ev;
    ev.index = static_cast<int64_t>(job->tokens.size());
    ev.token_id = id;
    ev.text = text;
    ev.t_wall_s = wall_s();
    if (job->status.ttft_s < 0) job->status.ttft_s = ev.t_wall_s - job->status.started_wall_s;
    if (job->status.last_token_wall_s >= 0) {
      double gap = ev.t_wall_s - job->status.last_token_wall_s;
      job->status.inter_token_sum_s += gap;
      if (gap > job->status.inter_token_max_s) job->status.inter_token_max_s = gap;
    }
    job->status.last_token_wall_s = ev.t_wall_s;
    if (job->cancel) job->status.tokens_after_cancel += 1;
    job->tokens.push_back(ev);
    job->status.tokens_emitted = static_cast<int64_t>(job->tokens.size());
    job->cv.notify_all();
  };
  ctx.bind_ = [job](const json::Value& b) {
    std::lock_guard<std::mutex> lk(job->mu);
    job->status.binding = b;
  };
  ctx.name_ = [job](const std::string& n) {
    std::lock_guard<std::mutex> lk(job->mu);
    job->status.engine = n;
  };
  ctx.metric_ = [job](const std::string& name, double value) {
    std::lock_guard<std::mutex> lk(job->mu);
    if (!job->status.metrics.is_object()) job->status.metrics = json::Value::object();
    if (value == value && value - value == 0.0) job->status.metrics.set(name, json::Value::number(value));  // finite only
  };
  Expected<bool> r = engine_ ? engine_(ctx) : make_fail<bool>("NO_ENGINE");
  // Outcome first, then the counters, then the published state. The two locks are never
  // nested here: the request handlers take mu_ and then a job's mu (status, metrics), so
  // taking mu_ while holding job->mu would invert that order. The counters are updated
  // before the state becomes visible, so a waiter that sees a final state sees them too.
  const bool cancelled = job->cancel.load();
  const std::string final_state = cancelled ? "cancelled" : (r.ok ? "done" : "error");
  {
    std::lock_guard<std::mutex> gl(mu_);
    if (final_state == "error") {
      stats_.jobs_error++;
      // REJECT codes of the model server (HELLO / RESUME binding checks)
      if (code_is(r.code, "VERSION_MISMATCH")) stats_.rejected_version++;
      if (code_is(r.code, "CONTRACT_MISMATCH")) stats_.rejected_contract++;
      if (code_is(r.code, "PROFILE_MISMATCH")) stats_.rejected_profile++;
      if (code_is(r.code, "SESSION_EXPIRED") || code_is(r.code, "SESSION_UNKNOWN") || code_is(r.code, "NONCE_MISMATCH"))
        stats_.rejected_stale++;
      if (code_is(r.code, "DUPLICATE_SESSION")) stats_.rejected_duplicate_session++;
    } else if (final_state == "done") {
      stats_.jobs_done++;
    }
  }
  std::string job_id;
  int64_t tokens = 0;
  double seconds = 0.0;
  {
    std::lock_guard<std::mutex> lk(job->mu);
    job->status.finished_wall_s = wall_s();
    job->status.state = final_state;
    if (final_state == "error") job->status.error_code = r.code;
    job_id = job->status.job_id;
    tokens = job->status.tokens_emitted;
    seconds = job->status.finished_wall_s - job->status.started_wall_s;
    job->cv.notify_all();
  }
  log("job_finished", {{"job_id", json::Value::string(job_id)},
                       {"state", json::Value::string(final_state)},
                       {"tokens", json::Value::integer(tokens)},
                       {"seconds", json::Value::number(seconds)}});
}

void Sidecar::handle_cancel(HttpServer::Exchange& ex) {
  const HttpRequest& req = ex.request;
  auto j = json::parse(req.body);
  if (!j.ok || !j.value.is_object()) {
    ex.respond(with_cors(HttpResponse::error(400, "SIDECAR_BAD_JSON"), req));
    return;
  }
  std::string job_id = j.value.str_or("job_id", "");
  auto c = cancel_job(job_id);
  if (!c.ok) {
    ex.respond(with_cors(HttpResponse::error(404, c.code), req));
    return;
  }
  auto st = job_status(job_id);
  json::Value o = json::Value::object();
  o.set("job_id", json::Value::string(job_id));
  o.set("state", json::Value::string(st.ok ? st.value.state : "unknown"));
  log("cancel", {{"job_id", json::Value::string(job_id)}});
  ex.respond(with_cors(HttpResponse::json(200, o), req));
}

void Sidecar::handle_stream(HttpServer::Exchange& ex) {
  const HttpRequest& req = ex.request;
  std::string job_id = req.query_param("job_id", "");
  auto job = find_job(job_id);
  if (!job) {
    ex.respond(with_cors(HttpResponse::error(404, "JOB_UNKNOWN"), req));
    return;
  }
  ex.hijacked = true;
  ex.responded = true;
  {
    std::lock_guard<std::mutex> lk(mu_);
    stats_.sse_clients++;
  }
  SseStream sse(ex.socket);
  std::vector<std::pair<std::string, std::string>> extra;
  const std::string* origin = req.header("origin");
  if (origin && origin_allowed(req, cfg_.origin_allowlist, false)) extra.emplace_back("Access-Control-Allow-Origin", *origin);
  if (!sse.begin(extra).ok) return;
  size_t sent = 0;
  while (!ex.stopping) {
    std::vector<TokenEvent> batch;
    std::string final_state;
    JobStatus snapshot;
    {
      std::unique_lock<std::mutex> lk(job->mu);
      if (job->tokens.size() == sent && (job->status.state == "queued" || job->status.state == "running")) {
        job->cv.wait_for(lk, std::chrono::duration<double>(cfg_.sse_heartbeat_s));
      }
      for (size_t i = sent; i < job->tokens.size(); ++i) batch.push_back(job->tokens[i]);
      sent = job->tokens.size();
      if (job->status.state != "queued" && job->status.state != "running") final_state = job->status.state;
      snapshot = job->status;
    }
    for (const auto& t : batch) {
      json::Value o = json::Value::object();
      o.set("index", json::Value::integer(t.index));
      o.set("token_id", json::Value::integer(t.token_id));
      o.set("text", json::Value::string(t.text));
      o.set("t_wall_s", json::Value::number(t.t_wall_s));
      o.set("engine", json::Value::string(snapshot.engine));
      o.set("job_id", json::Value::string(job_id));
      if (!sse.event("token", o.dump(false), std::to_string(t.index)).ok) return;
    }
    if (!final_state.empty()) {
      json::Value o = snapshot.to_json();
      sse.event(final_state, o.dump(false), "");
      return;
    }
    if (batch.empty()) {
      if (!sse.comment("keepalive").ok) return;
    }
  }
}

void Sidecar::handle_status(HttpServer::Exchange& ex) {
  const HttpRequest& req = ex.request;
  json::Value o = json::Value::object();
  json::Value jobs = json::Value::array();
  size_t active = 0;
  {
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto& kv : jobs_) {
      std::lock_guard<std::mutex> jl(kv.second->mu);
      JobStatus st = kv.second->status;
      st.inter_token_p50_s = median_of(token_gaps(kv.second->tokens));
      jobs.push(st.to_json());
      if (kv.second->status.state == "queued" || kv.second->status.state == "running") ++active;
    }
  }
  Stats s = stats();
  json::Value st = json::Value::object();
  st.set("requests", json::Value::integer(static_cast<int64_t>(s.requests)));
  st.set("rejected_origin", json::Value::integer(static_cast<int64_t>(s.rejected_origin)));
  st.set("rejected_auth", json::Value::integer(static_cast<int64_t>(s.rejected_auth)));
  st.set("jobs_started", json::Value::integer(static_cast<int64_t>(s.jobs_started)));
  st.set("jobs_done", json::Value::integer(static_cast<int64_t>(s.jobs_done)));
  st.set("jobs_cancelled", json::Value::integer(static_cast<int64_t>(s.jobs_cancelled)));
  st.set("jobs_error", json::Value::integer(static_cast<int64_t>(s.jobs_error)));
  st.set("sse_clients", json::Value::integer(static_cast<int64_t>(s.sse_clients)));
  st.set("static_served", json::Value::integer(static_cast<int64_t>(s.static_served)));
  st.set("rejected_version", json::Value::integer(static_cast<int64_t>(s.rejected_version)));
  st.set("rejected_contract", json::Value::integer(static_cast<int64_t>(s.rejected_contract)));
  st.set("rejected_profile", json::Value::integer(static_cast<int64_t>(s.rejected_profile)));
  st.set("rejected_stale", json::Value::integer(static_cast<int64_t>(s.rejected_stale)));
  st.set("rejected_duplicate_session", json::Value::integer(static_cast<int64_t>(s.rejected_duplicate_session)));
  o.set("metrics", metrics_json());
  o.set("jobs", jobs);
  o.set("active_jobs", json::Value::integer(static_cast<int64_t>(active)));
  o.set("stats", st);
  o.set("server", json::Value::string(cfg_.server_ws_url));
  o.set("dealer", json::Value::string(cfg_.dealer_host + ":" + std::to_string(cfg_.dealer_port)));
  json::Value origins = json::Value::array();
  for (const auto& a : cfg_.origin_allowlist) origins.push(json::Value::string(a));
  o.set("origins", origins);
  o.set("model_version", json::Value::string(cfg_.model_version));
  o.set("numeric_contract_hash", json::Value::string(cfg_.numeric_contract_hash));
  o.set("profile_id", json::Value::string(cfg_.profile_id));
  o.set("uptime_s", json::Value::number(wall_s() - started_wall_s_));
  o.set("token_required", json::Value::boolean(true));
  ex.respond(with_cors(HttpResponse::json(200, o), req));
}

json::Value Sidecar::metrics_json() const {
  std::vector<double> ttft, ttft_warm, gaps, completion, cancel_latency;
  double ttft_cold = -1.0;
  int64_t finished = 0, completed = 0, cancelled = 0, errors = 0, tokens_after_cancel = 0;
  double burned = 0.0, unused = 0.0, cancelled_wall = 0.0;
  {
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto& kv : jobs_) {
      std::lock_guard<std::mutex> jl(kv.second->mu);
      const JobStatus& st = kv.second->status;
      if (st.ttft_s >= 0) {
        ttft.push_back(st.ttft_s);
        if (st.cold) {
          ttft_cold = st.ttft_s;
        } else {
          ttft_warm.push_back(st.ttft_s);
        }
      }
      for (double g : token_gaps(kv.second->tokens)) gaps.push_back(g);
      if (st.state == "queued" || st.state == "running") continue;
      ++finished;
      if (st.completed()) {
        ++completed;
        if (st.completion_s() >= 0) completion.push_back(st.completion_s());
      }
      if (st.state == "error") ++errors;
      if (st.state == "cancelled") {
        ++cancelled;
        if (st.cancel_latency_s() >= 0) cancel_latency.push_back(st.cancel_latency_s());
        tokens_after_cancel += st.tokens_after_cancel;
        if (st.completion_s() >= 0) cancelled_wall += st.completion_s();
        double b = st.metrics.double_or("material_burned_bytes", 0.0);
        double tr = st.metrics.double_or("material_transferred_bytes", 0.0);
        double co = st.metrics.double_or("material_consumed_bytes", 0.0);
        burned += b;
        if (tr > co) unused += tr - co;
      }
    }
  }
  Stats s = stats();
  json::Value m = json::Value::object();
  json::Value t = summary_of(ttft);
  t.set("cold_s", json::Value::number(ttft_cold));
  json::Value warm = summary_of(ttft_warm);
  t.set("warm_mean_s", json::Value::number(warm.double_or("mean", -1.0)));
  t.set("warm_n", json::Value::integer(static_cast<int64_t>(ttft_warm.size())));
  m.set("ttft_s", t);
  m.set("inter_token_s", summary_of(gaps));
  json::Value c = json::Value::object();
  c.set("finished", json::Value::integer(finished));
  c.set("completed", json::Value::integer(completed));
  c.set("cancelled", json::Value::integer(cancelled));
  c.set("error", json::Value::integer(errors));
  c.set("rate", json::Value::number(finished > 0 ? static_cast<double>(completed) / static_cast<double>(finished) : -1.0));
  m.set("completion", c);
  m.set("completion_s", summary_of(completion));
  json::Value w = json::Value::object();
  json::Value lat = summary_of(cancel_latency);
  w.set("jobs", json::Value::integer(cancelled));
  w.set("latency_s_mean", json::Value::number(lat.double_or("mean", -1.0)));
  w.set("latency_s_max", json::Value::number(lat.double_or("max", -1.0)));
  w.set("tokens_after_cancel", json::Value::integer(tokens_after_cancel));
  w.set("material_burned_bytes", json::Value::integer(static_cast<int64_t>(burned)));
  w.set("material_unused_bytes", json::Value::integer(static_cast<int64_t>(unused)));
  w.set("wall_s", json::Value::number(cancelled_wall));
  m.set("cancel_waste", w);
  json::Value r = json::Value::object();
  r.set("origin", json::Value::integer(static_cast<int64_t>(s.rejected_origin)));
  r.set("auth", json::Value::integer(static_cast<int64_t>(s.rejected_auth)));
  r.set("version_mismatch", json::Value::integer(static_cast<int64_t>(s.rejected_version)));
  r.set("contract_mismatch", json::Value::integer(static_cast<int64_t>(s.rejected_contract)));
  r.set("profile_mismatch", json::Value::integer(static_cast<int64_t>(s.rejected_profile)));
  r.set("stale_session", json::Value::integer(static_cast<int64_t>(s.rejected_stale)));
  r.set("duplicate_session", json::Value::integer(static_cast<int64_t>(s.rejected_duplicate_session)));
  m.set("rejections", r);
  return m;
}

namespace {
std::string content_type_for(const std::string& name) {
  auto ends = [&](const char* suf) {
    size_t n = std::strlen(suf);
    return name.size() >= n && name.compare(name.size() - n, n, suf) == 0;
  };
  if (ends(".html")) return "text/html; charset=utf-8";
  if (ends(".js")) return "application/javascript; charset=utf-8";
  if (ends(".css")) return "text/css; charset=utf-8";
  if (ends(".json")) return "application/json";
  if (ends(".svg")) return "image/svg+xml";
  if (ends(".png")) return "image/png";
  if (ends(".ico")) return "image/x-icon";
  return "application/octet-stream";
}

bool safe_static_name(const std::string& name) {
  if (name.empty() || name.size() > 64) return false;
  if (name[0] == '.') return false;
  for (char c : name) {
    bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
    if (!ok) return false;
  }
  if (name.find("..") != std::string::npos) return false;
  return true;
}
}  // namespace

void Sidecar::handle_static(HttpServer::Exchange& ex) {
  const HttpRequest& req = ex.request;
  if (req.method != "GET" && req.method != "HEAD") {
    ex.respond(HttpResponse::error(405, "METHOD_NOT_ALLOWED"));
    return;
  }
  std::string name = req.path == "/" ? "index.html" : req.path.substr(1);
  if (!safe_static_name(name)) {
    ex.respond(HttpResponse::error(403, "SIDECAR_STATIC_FORBIDDEN"));
    return;
  }
  HttpResponse r;
  if (cfg_.static_dir.empty()) {
    if (name != "index.html") {
      ex.respond(HttpResponse::error(404, "NOT_FOUND"));
      return;
    }
    r = HttpResponse::text(200, builtin_browser_page(), "text/html; charset=utf-8");
  } else {
    std::string path = cfg_.static_dir + "/" + name;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      ex.respond(HttpResponse::error(404, "NOT_FOUND"));
      return;
    }
    std::stringstream ss;
    ss << in.rdbuf();
    r = HttpResponse::text(200, ss.str(), content_type_for(name));
  }
  r.set_header("Content-Security-Policy", "default-src 'self'; connect-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'");
  r.set_header("Referrer-Policy", "no-referrer");
  r.set_header("X-Content-Type-Options", "nosniff");
  {
    std::lock_guard<std::mutex> lk(mu_);
    stats_.static_served++;
  }
  if (req.method == "HEAD") r.body.clear();
  ex.respond(r);
}

// ---------------------------------------------------------------------------
// Engines
// ---------------------------------------------------------------------------
ClientEngine make_stub_engine(double delay_per_token_s) {
  return [delay_per_token_s](JobContext& ctx) -> Expected<bool> {
    ctx.set_engine_name("stub");
    json::Value b = json::Value::object();
    b.set("session_id", json::Value::string(ctx.session_id()));
    ctx.set_binding(b);
    for (int64_t i = 0; i < ctx.n_tokens(); ++i) {
      if (ctx.cancelled()) return make_ok(true);
      sleep_s(delay_per_token_s);
      ctx.emit_token(1000 + i, "tok" + std::to_string(i));
    }
    return make_ok(true);
  };
}

ClientEngine make_demo_beaver_engine() {
  return [](JobContext& ctx) -> Expected<bool> {
    ctx.set_engine_name("demo_beaver");
    const SidecarConfig& cfg = ctx.config();
    size_t n = cfg.demo_vector_len;
    // 1. dealer connection as recipient 1
    auto dealer = DealerConnection::connect(cfg.dealer_host, cfg.dealer_port, 1, cfg.setup_id, cfg.numeric_contract_hash,
                                            cfg.dealer_key, 30.0, 20);
    if (!dealer.ok) return forward_fail<bool>(dealer);
    // 2. WebSocket session with the model server
    auto ws = websocket_connect_url(cfg.server_ws_url, "", 10.0);
    if (!ws.ok) return forward_fail<bool>(ws);
    WebSocketChannel ch(std::move(ws.value), 1);
    std::vector<uint8_t> nonce = detail::os_random_bytes(16);
    wire::Value hello = wire::Value::make_dict();
    hello.set("type", wire::Value::from_str("HELLO"));
    hello.set("session_id", wire::Value::from_str(ctx.session_id()));
    hello.set("model_version", wire::Value::from_str(cfg.model_version));
    hello.set("numeric_contract_hash", wire::Value::from_str(cfg.numeric_contract_hash));
    hello.set("profile_id", wire::Value::from_str(cfg.profile_id));
    hello.set("nonce", wire::Value::from_bytes(nonce));
    hello.set("length_bucket", wire::Value::from_int(cfg.length_bucket));
    auto s = ch.send(hello);
    if (!s.ok) return forward_fail<bool>(s);
    ch.count_round();
    auto ok = ch.recv();
    if (!ok.ok) return forward_fail<bool>(ok);
    if (ok.value.str_or("type", "") != "HELLO_OK") return make_fail<bool>(ok.value.str_or("code", "HELLO_REJECTED"));
    json::Value binding = json::Value::object();
    binding.set("session_id", json::Value::string(ctx.session_id()));
    binding.set("model_version", json::Value::string(cfg.model_version));
    binding.set("numeric_contract_hash", json::Value::string(cfg.numeric_contract_hash));
    binding.set("profile_id", json::Value::string(cfg.profile_id));
    binding.set("nonce", json::Value::string(detail::hex_encode(nonce)));
    binding.set("length_bucket", json::Value::integer(cfg.length_bucket));
    binding.set("ttl_s", json::Value::number(ok.value.float_or("ttl_s", 0)));
    ctx.set_binding(binding);
    // 3. the client input: prompt bytes encoded into an n-vector (stays local;
    //    only uniformly random shares leave the process)
    std::vector<uint64_t> x(n, 0);
    const std::string& prompt = ctx.prompt();
    for (size_t i = 0; i < n; ++i) x[i] = prompt.empty() ? 0 : static_cast<uint8_t>(prompt[i % prompt.size()]);
    detail::ChaCha20 rng = detail::ChaCha20::from_os();
    // Request / response against the server: send, then receive with a short
    // socket timeout so that a cancel is noticed while waiting.
    ch.ws().set_timeout(1.0);
    auto recv_reply = [&]() -> Expected<wire::Value> {
      while (true) {
        auto r = ch.recv();
        if (r.ok) {
          if (r.value.str_or("type", "") == "FLOW") continue;  // backpressure frames are advisory here
          return r;
        }
        if (r.code != "TCP_TIMEOUT") return r;
        if (ctx.cancelled()) return make_fail<wire::Value>("CANCELLED");
      }
    };
    auto send_msg = [&](const std::string& gate, int64_t seq, wire::Value payload) -> Expected<wire::Value> {
      wire::Value m = wire::Value::make_dict();
      m.set("type", wire::Value::from_str("MSG"));
      m.set("gate_id", wire::Value::from_str(gate));
      m.set("seq", wire::Value::from_int(seq));
      m.set("payload", std::move(payload));
      auto s = ch.send(m);
      if (!s.ok) return forward_fail<wire::Value>(s);
      ch.count_round();
      auto r = recv_reply();
      if (!r.ok) return r;
      if (r.value.str_or("type", "") == "REJECT") return make_fail<wire::Value>(r.value.str_or("code", "REJECTED"));
      const wire::Value* p = r.value.get("payload");
      if (!p) return make_fail<wire::Value>("DEMO_BAD_PAYLOAD");
      return make_ok(*p);
    };
    for (int64_t step = 0; step < ctx.n_tokens(); ++step) {
      if (ctx.cancelled()) {
        wire::Value c = wire::Value::make_dict();
        c.set("type", wire::Value::from_str("CANCEL"));
        ch.send(c);
        ch.close(1000);
        dealer.value.close();
        return make_ok(true);
      }
      // material for this step (same gate_id on both sides)
      wire::Value params = wire::Value::make_dict();
      params.set("shape", wire::Value::from_list({wire::Value::from_int(static_cast<int64_t>(n))}));
      auto bundle = dealer.value.request("beaver", params, 1, ctx.session_id() + ":" + std::to_string(step));
      if (!bundle.ok) return forward_fail<bool>(bundle);
      auto half = demo::parse_beaver_half(bundle.value, n);
      if (!half.ok) return forward_fail<bool>(half);
      // round 1: share x (send x0 to P0), receive w1
      std::vector<uint64_t> x0(n);
      rng.fill_u64(x0.data(), n);
      std::vector<uint64_t> x1 = demo::ring_sub(x, x0);
      wire::Value p1 = wire::Value::make_dict();
      p1.set("x0", wire::Value::from_u64_vector(x0));
      auto r1 = send_msg("share_input", step, std::move(p1));
      if (!r1.ok) {
        if (r1.code == "CANCELLED") return make_ok(true);
        return forward_fail<bool>(r1);
      }
      auto w1 = demo::array_field(r1.value, "w1", n);
      if (!w1.ok) return forward_fail<bool>(w1);
      // round 2: Beaver opening
      std::vector<uint64_t> d1 = demo::ring_sub(x1, half.value.a);
      std::vector<uint64_t> e1 = demo::ring_sub(w1.value, half.value.b);
      wire::Value p2 = wire::Value::make_dict();
      p2.set("d1", wire::Value::from_u64_vector(d1));
      p2.set("e1", wire::Value::from_u64_vector(e1));
      auto r2 = send_msg("beaver_open", step, std::move(p2));
      if (!r2.ok) {
        if (r2.code == "CANCELLED") return make_ok(true);
        return forward_fail<bool>(r2);
      }
      auto d0 = demo::array_field(r2.value, "d0", n);
      auto e0 = demo::array_field(r2.value, "e0", n);
      auto z0 = demo::array_field(r2.value, "z0", n);
      if (!d0.ok || !e0.ok || !z0.ok) return make_fail<bool>("DEMO_BAD_PAYLOAD");
      std::vector<uint64_t> d = demo::ring_add(d0.value, d1);
      std::vector<uint64_t> e = demo::ring_add(e0.value, e1);
      std::vector<uint64_t> z1 = demo::beaver_combine(half.value, d, e, 1);
      std::vector<uint64_t> z = demo::ring_add(z0.value, z1);  // x * w, released to P1 only
      uint64_t tok = z[static_cast<size_t>(step) % n] % 50257u;
      ctx.emit_token(static_cast<int64_t>(tok), "<demo:" + std::to_string(tok) + ">");
    }
    ch.close(1000);
    dealer.value.close();
    return make_ok(true);
  };
}

std::string builtin_browser_page() {
  return "<!doctype html>\n<html><head><meta charset=\"utf-8\"><title>OpenCrab v2 sidecar</title>\n"
         "<style>body{font-family:system-ui,sans-serif;margin:2rem;max-width:48rem}pre{background:#f4f4f4;padding:.5rem;"
         "white-space:pre-wrap}button{margin-right:.5rem}</style></head><body>\n"
         "<h1>OpenCrab v2 sidecar (built-in page)</h1>\n"
         "<p>This page is served by the local sidecar on 127.0.0.1. The prompt below never leaves this machine: the "
         "sidecar shares it into the two-party protocol and only shares leave the process.</p>\n"
         "<p>Session binding: <code id=\"binding\">(none)</code></p>\n"
         "<textarea id=\"prompt\" rows=\"3\" cols=\"60\">hello opencrab</textarea><br>\n"
         "<label>tokens <input id=\"n\" type=\"number\" value=\"8\" min=\"1\" max=\"256\"></label>\n"
         "<button id=\"go\">generate</button><button id=\"cancel\" disabled>cancel</button>\n"
         "<pre id=\"out\"></pre>\n"
         "<script>\n"
         "const token=(location.hash.match(/token=([0-9a-f]+)/)||[])[1]||'';\n"
         "const hdr={'Authorization':'Bearer '+token,'Content-Type':'application/json'};\n"
         "let job=null;const out=document.getElementById('out');\n"
         "function log(s){out.textContent+=s+'\\n';}\n"
         "function onEvent(name,data){if(name==='token'){const t=JSON.parse(data);log('token '+t.index+': '+t.text+' ['+t.engine+']');}\n"
         " else if(name==='done'||name==='cancelled'||name==='error'){const s=JSON.parse(data);document.getElementById('binding').textContent=JSON.stringify(s.binding);log(name+' '+(s.error_code||''));document.getElementById('cancel').disabled=true;}}\n"
         "async function stream(id){const r=await fetch('/v1/stream?job_id='+encodeURIComponent(id),{headers:{'Authorization':'Bearer '+token}});\n"
         " if(!r.ok){log('stream failed: '+r.status);return;}\n"
         " const rd=r.body.getReader();const dec=new TextDecoder();let buf='';\n"
         " for(;;){const {value,done}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});\n"
         "  let i;while((i=buf.indexOf('\\n\\n'))>=0){const block=buf.slice(0,i);buf=buf.slice(i+2);let ev='message',data=[];\n"
         "   for(const line of block.split('\\n')){if(line.startsWith('event: '))ev=line.slice(7);else if(line.startsWith('data: '))data.push(line.slice(6));}\n"
         "   if(data.length)onEvent(ev,data.join('\\n'));}}}\n"
         "document.getElementById('go').onclick=async()=>{out.textContent='';\n"
         " const r=await fetch('/v1/generate',{method:'POST',headers:hdr,body:JSON.stringify({prompt:document.getElementById('prompt').value,n_tokens:+document.getElementById('n').value})});\n"
         " if(!r.ok){log('generate failed: '+r.status);return;}\n"
         " const j=await r.json();job=j.job_id;document.getElementById('cancel').disabled=false;log('job '+job+' session '+j.session_id);\n"
         " stream(job);\n"
         "};\n"
         "document.getElementById('cancel').onclick=async()=>{if(!job)return;await fetch('/v1/cancel',{method:'POST',headers:hdr,body:JSON.stringify({job_id:job})});};\n"
         "</script>\n"
         "<p>The token comes from the URL fragment printed at launch and is sent only in the Authorization header "
         "(fetch streaming, not EventSource, so no token ever appears in a URL).</p>\n"
         "</body></html>\n";
}

}  // namespace platform
}  // namespace opencrab2
