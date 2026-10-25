// opencrab2/platform/model_server.hpp
//
// P0 model server (paper, Section 3, design): a WebSocket endpoint /session over
// the HTTP server of http.hpp. Frames are binary WebSocket messages carrying
// one wire-codec dict each (codec.hpp, the bytes of the Python channel):
//
//   client HELLO  {type:"HELLO", session_id, model_version, numeric_contract_hash,
//                  profile_id, nonce(bytes), length_bucket(int)}
//   server        {type:"HELLO_OK", session_id, ttl_s, max_queue_frames, length_bucket}
//              or {type:"REJECT", code} followed by close(1008)
//   client RESUME {type:"RESUME", session_id, nonce}
//   server        {type:"RESUME_OK", session_id, last_seq{gate_id -> seq}} or REJECT
//   client MSG    {type:"MSG", gate_id, seq, payload}
//   server MSG    {type:"MSG", gate_id, seq, payload}     (byte-identical on replay)
//   client CANCEL {type:"CANCEL"}            server {type:"CANCEL_OK"}
//   client PING   {type:"PING"}              server {type:"PONG"}
//   server FLOW   {type:"FLOW", pause(bool), queue_depth}   (backpressure)
//
// Rules enforced here (tested): version, contract and profile binding at
// HELLO (REJECT VERSION_MISMATCH / CONTRACT_MISMATCH / PROFILE_MISMATCH);
// the public length bucket must be an integer in [1, 2^20] (REJECT
// BAD_LENGTH_BUCKET otherwise, also for a missing or non-integer value) and
// is echoed in HELLO_OK and bound to the session;
// session TTL since last activity (REJECT SESSION_EXPIRED on RESUME,
// close(1008) on idle); duplicate session_id while attached (REJECT
// DUPLICATE_SESSION); duplicate (gate_id, seq) answered from the
// retransmission log without re-executing the handler (identical bytes); the
// same (gate_id, seq) with a different request digest is a protocol
// violation (REJECT SEQ_REUSED_DIFFERENT_INPUT, close): a new input never
// reuses a sequence number, which is the server-side half of ledger
// rule that an id is spent at most once (ledger.hpp); bounded inbound queue with FLOW pause
// and resume frames, and close(1008, FLOW_VIOLATION) when the client ignores
// them; frames larger than max_frame_bytes close the connection. The server
// has no code path that accepts a plaintext prompt: MSG payloads that are
// strings or dicts with keys prompt/text/tokens are rejected
// (PLAINTEXT_NOT_ACCEPTED) as defence in depth. Log lines carry ids, sizes
// and timings only, never payload bytes.
//
// The protocol handler (what P0 computes on each MSG) is injected. The
// session bridge (session_bridge.hpp) installs the highlevel session loop;
// without it the server app uses DemoBeaverP0 below, a real two-round
// semi-honest elementwise product with dealer Beaver triples, so the full
// dealer + server + sidecar path runs natively.
#pragma once

#include <opencrab2/platform/dealer_service.hpp>
#include <opencrab2/platform/http.hpp>
#include <opencrab2/platform/support.hpp>
#include <opencrab2/platform/websocket.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace opencrab2 {
namespace platform {

struct ModelServerConfig {
  std::string host = "127.0.0.1";
  uint16_t port = 8765;
  std::string model_version = "gpt2-4L";
  std::string numeric_contract_hash;
  std::string profile_id = "SH2P-TrustedOffline-v1";
  std::string setup_id = "setup-0";
  double session_ttl_s = 300.0;
  size_t max_queue_frames = 64;
  size_t max_frame_bytes = static_cast<size_t>(64) << 20;
  size_t retransmit_log_entries = 1024;
  std::vector<std::string> origin_allowlist;  // empty: Origin not checked (peer is the sidecar process)
  std::function<void(const std::string&)> log;  // JSON line sink; default stderr
};

struct SessionBinding {
  std::string session_id;
  std::string model_version;
  std::string numeric_contract_hash;
  std::string profile_id;
  std::vector<uint8_t> nonce;
  int64_t length_bucket = 0;
  json::Value to_json() const;  // nonce as hex
};

// Server side retransmission log: (gate_id, seq) -> request digest + reply bytes.
class RetransmitLog {
 public:
  explicit RetransmitLog(size_t max_entries = 1024) : max_entries_(max_entries) {}
  // nullptr when unseen; fails SEQ_REUSED_DIFFERENT_INPUT when seen with another digest.
  Expected<const std::vector<uint8_t>*> lookup(const std::string& gate_id, int64_t seq,
                                                const std::string& request_digest) const;
  void record(const std::string& gate_id, int64_t seq, const std::string& request_digest, std::vector<uint8_t> reply);
  size_t size() const { return order_.size(); }
  uint64_t bytes() const { return bytes_; }
  std::map<std::string, int64_t> last_seq() const;

 private:
  struct Entry {
    std::string digest;
    std::vector<uint8_t> reply;
  };
  size_t max_entries_;
  std::map<std::string, Entry> entries_;  // key = gate_id + "#" + seq
  std::deque<std::string> order_;
  std::map<std::string, int64_t> last_seq_;
  uint64_t bytes_ = 0;
};

struct Session {
  SessionBinding binding;
  double created_wall_s = 0.0;
  double last_active_wall_s = 0.0;
  bool attached = false;
  std::atomic<bool> cancelled{false};
  uint64_t frames_in = 0, frames_out = 0, bytes_in = 0, bytes_out = 0, replays = 0;
  RetransmitLog log;
  std::shared_ptr<void> handler_state;  // owned by the protocol handler
  std::mutex mu;
  explicit Session(size_t log_entries) : log(log_entries) {}
};

// Computes the reply payload for one MSG. Failure codes become REJECT frames.
using ProtocolHandler = std::function<Expected<wire::Value>(Session& session, const std::string& gate_id, int64_t seq,
                                                            const wire::Value& payload)>;

class ModelServer {
 public:
  ModelServer(ModelServerConfig cfg, ProtocolHandler handler);
  ~ModelServer();
  ModelServer(const ModelServer&) = delete;
  ModelServer& operator=(const ModelServer&) = delete;

  Expected<bool> start();
  void stop();
  uint16_t port() const;
  const ModelServerConfig& config() const { return cfg_; }

  struct Stats {
    uint64_t connections = 0, sessions_opened = 0, sessions_resumed = 0, sessions_expired = 0;
    uint64_t rejected_version = 0, rejected_contract = 0, rejected_profile = 0, rejected_expired = 0;
    uint64_t rejected_duplicate_session = 0, rejected_origin = 0, rejected_unknown_session = 0, rejected_nonce = 0;
    uint64_t rejected_length_bucket = 0;
    uint64_t duplicate_seq_retransmits = 0, seq_violations = 0, plaintext_rejections = 0, handler_rejections = 0;
    uint64_t flow_pause_frames = 0, flow_resume_frames = 0, flow_violations = 0;
    uint64_t frames_in = 0, frames_out = 0, bytes_in = 0, bytes_out = 0, cancels = 0, bad_frames = 0;
  };
  Stats stats() const;
  json::Value stats_json() const;
  size_t live_sessions() const;
  // Removes idle detached sessions older than the TTL; returns how many.
  size_t expire_sessions(double now_wall_s);

 private:
  struct Connection;
  void on_http(HttpServer::Exchange& ex);
  void run_connection(std::shared_ptr<Connection> conn);
  void reader_loop(std::shared_ptr<Connection> conn);
  void housekeeping_loop();
  Expected<bool> send_frame(Connection& conn, const wire::Value& v);
  void reject(Connection& conn, const std::string& code, bool close);
  void log(const std::string& event, const std::map<std::string, json::Value>& fields);
  Expected<std::shared_ptr<Session>> handle_hello(Connection& conn, const wire::Value& v);
  Expected<std::shared_ptr<Session>> handle_resume(Connection& conn, const wire::Value& v);
  void handle_msg(Connection& conn, Session& s, const wire::Value& v);

  ModelServerConfig cfg_;
  ProtocolHandler handler_;
  std::unique_ptr<HttpServer> http_;
  std::thread housekeeping_;
  std::atomic<bool> stopping_{false};
  mutable std::mutex mu_;
  std::map<std::string, std::shared_ptr<Session>> sessions_;
  Stats stats_;
};

// ---------------------------------------------------------------------------
// Demo protocol (used by the server app and the sidecar demo engine until the
// highlevel Party lands). Public n-element vectors over Z_2^64.
//   gate "share_input" seq k: P1 -> {x0}         P0 -> {w1}
//   gate "beaver_open" seq k: P1 -> {d1, e1}     P0 -> {d0, e0, z0}
// with Beaver triple (a, b, c = a*b) from the dealer under
// material beaver(shape=[n]), gate_id = session_id + ":" + k, so both
// parties receive the halves of the same tuple. z = x * w elementwise is
// released to P1 only. Two rounds, 5 n-vectors on the wire per step.
// ---------------------------------------------------------------------------
namespace demo {
struct BeaverHalf {
  std::vector<uint64_t> a, b, c;
};
Expected<BeaverHalf> parse_beaver_half(const Bundle& bundle, size_t n);
std::vector<uint64_t> ring_sub(const std::vector<uint64_t>& x, const std::vector<uint64_t>& y);
std::vector<uint64_t> ring_add(const std::vector<uint64_t>& x, const std::vector<uint64_t>& y);
// z_share = c + d*b + e*a (+ d*e for party 0)
std::vector<uint64_t> beaver_combine(const BeaverHalf& h, const std::vector<uint64_t>& d, const std::vector<uint64_t>& e,
                                     int party);
Expected<std::vector<uint64_t>> array_field(const wire::Value& dict, const std::string& key, size_t n);
}  // namespace demo

class DemoBeaverP0 {
 public:
  DemoBeaverP0(std::shared_ptr<DealerConnection> dealer, std::vector<uint64_t> w);
  Expected<wire::Value> handle(Session& session, const std::string& gate_id, int64_t seq, const wire::Value& payload);
  ProtocolHandler as_handler();
  size_t n() const { return w_.size(); }

 private:
  struct StepState {
    std::vector<uint64_t> x0, w0;
    demo::BeaverHalf triple;
    bool have_input = false;
  };
  struct SessionState {
    std::map<int64_t, StepState> steps;
  };
  std::shared_ptr<DealerConnection> dealer_;
  std::vector<uint64_t> w_;
  std::mutex mu_;
  detail::ChaCha20 rng_;
};

// Accepts every MSG and answers {ack:true, bytes:n}; no computation. Useful
// for transport tests and as the fallback when no dealer is reachable.
ProtocolHandler make_ack_handler();

}  // namespace platform
}  // namespace opencrab2
