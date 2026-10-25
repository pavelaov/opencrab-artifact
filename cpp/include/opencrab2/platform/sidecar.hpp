// opencrab2/platform/sidecar.hpp
//
// P1 sidecar (paper, Section 3, design): a local HTTP server bound to 127.0.0.1 with
// an Origin allowlist and a per-launch bearer token. The browser page talks
// only to the sidecar; the prompt never leaves this process. Endpoints:
//
//   POST /v1/generate   {prompt, n_tokens, session_id?}  -> 202 {job_id, session_id}
//   POST /v1/cancel     {job_id}                         -> 200 {job_id, state}
//   GET  /v1/stream?job_id=...  Server-Sent Events: token / done / cancelled / error
//   GET  /v1/status     -> 200 {jobs, active_jobs, stats, metrics, server, origins, uptime_s}
//   GET  /              static browser page (browser_client dir, read only) or a
//                       built-in minimal page; GET /<file> for app.js, style.css
//   OPTIONS /v1/*       CORS preflight for allowed origins only
//
// Checks on every /v1 request, in this order: Origin present and not in the
// allowlist -> 403 ORIGIN_REJECTED (a missing Origin is allowed: non-browser
// clients are protected by the token); Authorization: Bearer <token>
// mismatch or missing -> 401 TOKEN_REQUIRED. The static page needs neither;
// it receives the token from the URL fragment printed at launch
// (http://127.0.0.1:<port>/#token=...), which browsers never send to servers.
//
// The generation engine is injected (ClientEngine). The sidecar app runs
// make_tiny_session_engine (session_bridge.hpp: the highlevel session loop
// with the tiny decoder) by default; make_demo_beaver_engine runs the
// two-round Beaver product protocol of model_server.hpp for every token step
// and labels its tokens engine="demo_beaver" (not language model output);
// make_stub_engine needs no network.
//
// Web metrics (paper, Section 7.4, Web service), all in GET /v1/status and all derived from
// public timestamps and counters, never from a prompt or a token value:
//   per job    ttft_s (first token minus job start), cold (first job of this
//              launch), inter_token_s {n, mean, p50, max}, completed (state
//              done with every requested token emitted), completion_s (job
//              start to finish), cancel {requested, latency_s (cancel request
//              to job stop), tokens_after_cancel}, engine metrics reported by
//              the engine through JobContext::report_metric (the tiny engine
//              reports material_transferred_bytes, material_consumed_bytes,
//              material_burned_bytes, T_online_s, T_cold_s)
//   metrics    ttft_s {n, mean, p50, max, cold_s, warm_mean_s},
//              inter_token_s {n, mean, p50, max}, completion {finished,
//              completed, cancelled, error, rate}, completion_s {n, mean, p50,
//              max}, cancel_waste {jobs, latency_s_mean, latency_s_max,
//              tokens_after_cancel, material_burned_bytes,
//              material_unused_bytes (transferred minus consumed of cancelled
//              jobs), wall_s (job time spent on cancelled jobs)}, rejections
//              {origin, auth, version_mismatch, contract_mismatch,
//              profile_mismatch, stale_session, duplicate_session}
// The first two rejection counters are counted on the HTTP surface of the
// sidecar; the others are the REJECT codes the model server answered to this
// sidecar's HELLO or RESUME (the job then ends in state error with that code).
// Recovery cost after a disconnect is reported once the engine resumes a
// session (cancel_waste and the rejections are what a launch without resume
// can measure).
//
// Codes: SIDECAR_BAD_JSON, SIDECAR_BAD_REQUEST, JOB_UNKNOWN, JOB_LIMIT,
// SIDECAR_STATIC_FORBIDDEN, plus engine codes (DEALER_*, WS_*, TCP_*).
//
// Tested: 403 on wrong Origin, 401 on missing token, 200 status with token,
// generate + stream + cancel round trip with an in-process stub engine,
// static path traversal rejected, the metrics block after a completed, a
// cancelled and a rejected job.
#pragma once

#include <opencrab2/platform/http.hpp>
#include <opencrab2/platform/support.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace opencrab2 {
namespace platform {

struct SidecarConfig {
  std::string host = "127.0.0.1";
  uint16_t port = 8787;
  std::vector<std::string> origin_allowlist;  // e.g. http://127.0.0.1:8787
  std::string token;                          // per-launch bearer; generated when empty
  std::string server_ws_url = "ws://127.0.0.1:8765/session";
  std::string dealer_host = "127.0.0.1";
  uint16_t dealer_port = 0;
  std::vector<uint8_t> dealer_key;  // HMAC key of recipient 1
  std::string setup_id = "setup-0";
  std::string model_version = "gpt2-4L";
  std::string numeric_contract_hash;
  std::string profile_id = "SH2P-TrustedOffline-v1";
  int64_t length_bucket = 32;
  std::string static_dir;  // browser_client directory; empty = built-in page
  size_t max_jobs = 16;
  double sse_heartbeat_s = 5.0;
  size_t demo_vector_len = 64;
  std::function<void(const std::string&)> log;
};

struct TokenEvent {
  int64_t index = 0;
  int64_t token_id = 0;
  std::string text;
  double t_wall_s = 0.0;
};

struct JobStatus {
  std::string job_id;
  std::string session_id;
  std::string state;  // queued | running | done | cancelled | error
  std::string error_code;
  std::string engine;
  int64_t n_tokens = 0;
  int64_t tokens_emitted = 0;
  double created_wall_s = 0.0;
  double started_wall_s = 0.0;
  double finished_wall_s = 0.0;
  double ttft_s = -1.0;  // time to first token
  bool cold = false;     // first job of this sidecar launch (cold TTFT)
  // inter-token latency over the emitted tokens (n = tokens_emitted - 1 gaps)
  double last_token_wall_s = -1.0;
  double inter_token_sum_s = 0.0;
  double inter_token_max_s = 0.0;
  double inter_token_p50_s = -1.0;  // filled when the status is read
  // cancellation
  double cancel_requested_wall_s = -1.0;
  int64_t tokens_after_cancel = 0;
  json::Value binding;   // session binding echoed from the server (ids only)
  json::Value metrics;   // engine-reported public counters (name -> number)
  bool completed() const { return state == "done" && tokens_emitted == n_tokens; }
  double completion_s() const { return finished_wall_s > 0 && started_wall_s > 0 ? finished_wall_s - started_wall_s : -1.0; }
  double inter_token_mean_s() const { return tokens_emitted > 1 ? inter_token_sum_s / static_cast<double>(tokens_emitted - 1) : -1.0; }
  double cancel_latency_s() const {
    return cancel_requested_wall_s > 0 && finished_wall_s >= cancel_requested_wall_s ? finished_wall_s - cancel_requested_wall_s : -1.0;
  }
  json::Value to_json() const;
};

class Sidecar;

// Handed to the engine. The prompt is readable only here, inside the process.
class JobContext {
 public:
  bool cancelled() const { return cancel_->load(); }
  // The job's cancel request as a flag an engine can hand to a session loop.
  const std::atomic<bool>* cancel_flag() const { return cancel_; }
  const std::string& prompt() const { return prompt_; }
  int64_t n_tokens() const { return n_tokens_; }
  const std::string& session_id() const { return session_id_; }
  const std::string& job_id() const { return job_id_; }
  const SidecarConfig& config() const { return *cfg_; }
  void emit_token(int64_t token_id, const std::string& text);
  void set_binding(const json::Value& b);
  void set_engine_name(const std::string& name);
  // A public counter or timing of this job for /v1/status (sizes, seconds,
  // counts; never a secret-dependent value).
  void report_metric(const std::string& name, double value);

 private:
  friend class Sidecar;
  JobContext() = default;
  std::string prompt_;
  int64_t n_tokens_ = 0;
  std::string session_id_;
  std::string job_id_;
  const SidecarConfig* cfg_ = nullptr;
  std::atomic<bool>* cancel_ = nullptr;
  std::function<void(int64_t, const std::string&)> emit_;
  std::function<void(const json::Value&)> bind_;
  std::function<void(const std::string&)> name_;
  std::function<void(const std::string&, double)> metric_;
};

// Runs one generation job to completion; a failure code marks the job error.
using ClientEngine = std::function<Expected<bool>(JobContext&)>;

class Sidecar {
 public:
  Sidecar(SidecarConfig cfg, ClientEngine engine);
  ~Sidecar();
  Sidecar(const Sidecar&) = delete;
  Sidecar& operator=(const Sidecar&) = delete;

  Expected<bool> start();
  void stop();
  uint16_t port() const;
  const std::string& token() const { return cfg_.token; }
  const SidecarConfig& config() const { return cfg_; }
  std::string launch_url() const;  // http://host:port/#token=...

  struct Stats {
    uint64_t requests = 0, rejected_origin = 0, rejected_auth = 0, jobs_started = 0, jobs_done = 0;
    uint64_t jobs_cancelled = 0, jobs_error = 0, sse_clients = 0, static_served = 0, preflights = 0;
    // REJECT codes the model server answered to this sidecar (job error codes)
    uint64_t rejected_version = 0, rejected_contract = 0, rejected_profile = 0, rejected_stale = 0;
    uint64_t rejected_duplicate_session = 0;
  };
  Stats stats() const;
  // The "metrics" block of GET /v1/status (see the header comment).
  json::Value metrics_json() const;
  Expected<JobStatus> job_status(const std::string& job_id) const;
  Expected<std::vector<TokenEvent>> job_tokens(const std::string& job_id) const;
  Expected<bool> cancel_job(const std::string& job_id);
  // Blocks until the job leaves queued/running or the timeout passes.
  Expected<JobStatus> wait_job(const std::string& job_id, double timeout_s) const;

 private:
  struct Job {
    JobStatus status;
    std::string prompt;
    std::vector<TokenEvent> tokens;
    std::atomic<bool> cancel{false};
    mutable std::mutex mu;
    mutable std::condition_variable cv;
    std::thread worker;
  };
  void handle(HttpServer::Exchange& ex);
  void handle_generate(HttpServer::Exchange& ex);
  void handle_cancel(HttpServer::Exchange& ex);
  void handle_stream(HttpServer::Exchange& ex);
  void handle_status(HttpServer::Exchange& ex);
  void handle_static(HttpServer::Exchange& ex);
  void run_job(std::shared_ptr<Job> job);
  void log(const std::string& event, const std::map<std::string, json::Value>& fields);
  HttpResponse with_cors(HttpResponse r, const HttpRequest& req) const;
  std::shared_ptr<Job> find_job(const std::string& id) const;

  SidecarConfig cfg_;
  ClientEngine engine_;
  std::unique_ptr<HttpServer> http_;
  mutable std::mutex mu_;
  std::map<std::string, std::shared_ptr<Job>> jobs_;
  Stats stats_;
  double started_wall_s_ = 0.0;
  uint64_t job_counter_ = 0;
};

// Demo engine: dealer + model server Beaver product per token step (see
// model_server.hpp). Emits token ids z[k mod n] mod 50257 with text
// "<demo:ID>"; labelled engine="demo_beaver".
ClientEngine make_demo_beaver_engine();
// Local stub for tests: emits n_tokens tokens "tok<i>" with a small delay,
// no network.
ClientEngine make_stub_engine(double delay_per_token_s = 0.01);

// Random 32-byte hex token.
std::string generate_bearer_token();
// Built-in minimal page (used when no browser_client directory is configured).
std::string builtin_browser_page();

}  // namespace platform
}  // namespace opencrab2
