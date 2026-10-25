// opencrab2/platform/session_bridge.hpp
//
// Runs the highlevel session loop (opencrab2::highlevel::run_session_party with
// the tiny decoder) across the three
// platform processes: the isolated dealer (dealer_service.hpp), the P0 model
// server (model_server.hpp) and the P1 sidecar (sidecar.hpp).
//
// Transport. The model server answers one MSG with one reply, so the P1 side
// drives every channel operation as a request under gate_id "tiny-session":
//
//   payload {op:"setup", n_generate, policy, origin_unix_s}
//          -> {ok, tiny{...}, model_version, telemetry_key, controller (json),
//              trace (the public trace as JSON text), costmodel (json), shape, plan{...}}
// The cost model is public and must be the same on both parties (identical
// decisions): P0 loads it (TinySessionOptions::costmodel_path, a fitted model
// such as results/calibration/fitted_model.json, or the HISTORICAL prior)
// and sends it; P1 adopts it. A file that does not load fails every setup.
//   payload {op:"send", v: bytes}   P1 bare send      -> {ok}
//   payload {op:"recv"}             P1 bare recv      -> {v: bytes}
//   payload {op:"xchg", v: bytes}   P1 exchange       -> {v: bytes}  (P0 sends first)
//   payload {op:"finish"}           -> {ok, code, bytes, rounds} of the P0 record
//
// `v` is the wire-codec encoding (codec.hpp, byte-identical to the Python
// Channel) of the highlevel Value that the session sends; the two Channel
// adapters below count exactly those bytes plus the 8-byte frame prefix, so
// the evidence records carry the protocol bytes of the session loop
// (WebSocket framing is reported separately by WebSocketChannel::stats).
// Round accounting mirrors highlevel::Channel: one round per exchange, one per
// send/recv pair.
//
// P0 runs run_session_party in one thread per WebSocket session behind a
// queue-backed Channel (ServerSessionChannel); the handler thread moves
// bytes between the queues and the frames. Both parties draw material from
// the isolated dealer through DealerConnection (attach_dealer_connection):
// each party requests on its own pool miss, with identical parameters in the
// same order, and the dealer service pairs the two REQs (the second recipient
// pops the half its peer's identical request left behind).
//
// Shaping: when the server options ask for it, both parties sleep for the
// highlevel WallClockShaper delay of the server's trace before every send
// (same model as highlevel::Channel). The trace is public: P0 sends it to P1
// at setup (a stable link by default, any experiments/traces.py file with
// TinySessionOptions::trace_file), and P1 announces the shared origin as Unix
// time; each party converts it to its own monotonic clock.
//
// Party randomness. Every app asks the session for share randomness seeded
// from the OS CSPRNG (SessionConfig::party_rng_from_os, requested through
// session_caps::request_os_party_rng). A session seed shared by the two
// parties would let each one recompute the other's input masks, so no seed is
// ever sent: the seed field is filled with a private 31-bit value from the OS
// (never sent, never written to a record) and matters only against a session
// build without OS seeding. In that case the share randomness has 31 bits of
// entropy, far below the security parameter, and finalize_session_record
// marks the record DIAGNOSTIC and not eligible with the note PARTY_RNG_SEEDED.
//
// Evidence. Records leave this layer in the schema of
// python/opencrab2/experiments/schema.py (evidence.hpp): schema fields flat,
// tokens as a count, everything else under extra (language "cpp"), appended
// under the cross-process lock file.
//
// Test mode: TinySessionOptions::verify_weights_path lets the sidecar load the
// same random weights the server uses and compare the released tokens with the
// plaintext greedy reference (output_match in the P1 record). It discloses the
// weights to P1, so such a record is DIAGNOSTIC (note VERIFY_WEIGHTS_DISCLOSED).
//
// Codes: SESSION_ALREADY_SETUP, SESSION_NOT_SETUP, SESSION_BAD_REQUEST,
// CONTRACT_UNSUPPORTED, CHANNEL_TIMEOUT, CHANNEL_ABORTED, CANCELLED,
// VERIFY_WEIGHTS_MISMATCH, plus the codes of the layers below.
#pragma once

#include <opencrab2/highlevel/session.hpp>
#include <opencrab2/platform/dealer_service.hpp>
#include <opencrab2/platform/model_server.hpp>
#include <opencrab2/platform/sidecar.hpp>
#include <opencrab2/platform/support.hpp>
#include <opencrab2/platform/websocket.hpp>

#include <opencrab2/platform/evidence.hpp>
#include <opencrab2/platform/shaper.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace opencrab2 {
namespace platform {

extern const char* const kTinySessionGate;  // "tiny-session"

// Conversions between the platform wire::Value and the highlevel Value
// (structural; both are views of the one wire codec).
Expected<highlevel::Value> to_highlevel(const wire::Value& v);
Expected<wire::Value> to_wire(const highlevel::Value& v);

// ---------------------------------------------------------------------------
// Capabilities of the highlevel session interface, detected at compile time.
// The session loop is owned by the highlevel layer; cancellation, resumption
// and OS-seeded party randomness are optional members of
// highlevel::SessionConfig as far as this layer is concerned. Each setter
// assigns the member when SessionConfig has one of that name that accepts the
// value and returns false otherwise, so a caller can fail closed
// (SESSION_CANCEL_UNSUPPORTED, SESSION_RESUME_UNSUPPORTED) instead of
// silently ignoring an option.
// ---------------------------------------------------------------------------
namespace session_caps {

#define OPENCRAB2_SESSION_CAP(name)                                                              \
  template <class C, class V>                                                                    \
  auto set_##name(C& c, V&& v, int)->decltype(c.name = std::forward<V>(v), bool()) {             \
    c.name = std::forward<V>(v);                                                                 \
    return true;                                                                                 \
  }                                                                                              \
  template <class C, class V>                                                                    \
  bool set_##name(C&, V&&, long) {                                                               \
    return false;                                                                                \
  }

OPENCRAB2_SESSION_CAP(cancel_after_epoch)  // int: declared cancellation after that epoch
OPENCRAB2_SESSION_CAP(resume)              // bool: resume the session bound to session_id
OPENCRAB2_SESSION_CAP(state_path)          // std::string: public session state for a cold resume
OPENCRAB2_SESSION_CAP(cancel_flag)         // const std::atomic<bool>*: asynchronous cancel request
OPENCRAB2_SESSION_CAP(party_rng_from_os)   // bool: true  = party randomness from the OS CSPRNG (session.hpp)
OPENCRAB2_SESSION_CAP(deterministic_rng)   // alternative spellings of the same switch, probed so that a
OPENCRAB2_SESSION_CAP(os_seeded_rng)       // renamed field degrades to PARTY_RNG_SEEDED, never to a false claim
#undef OPENCRAB2_SESSION_CAP

// Asks the session for OS-seeded party randomness under any of the spellings
// above; true when one of them exists (the record may then be eligible).
template <class C>
bool request_os_party_rng(C& cfg) {
  bool a = set_deterministic_rng(cfg, false, 0);
  bool b = set_party_rng_from_os(cfg, true, 0);
  bool c = set_os_seeded_rng(cfg, true, 0);
  return a || b || c;
}

}  // namespace session_caps

// A private 31-bit session seed from the OS CSPRNG (never sent, never logged).
int private_session_seed();

// The highlevel view of a public trace (identity and the link columns).
highlevel::SessionTrace to_session_trace(const Trace& trace);

// What the caller knows about a finished session run and the record must say.
struct SessionRecordContext {
  std::string record_kind = "session";      // schema record_kind
  std::string code_commit;                  // OPENCRAB2_CODE_COMMIT_LITERAL of the app
  std::string implementation;               // app name, goes to extra.implementation
  std::string trace_id;                     // of the trace file (overrides the record when not empty)
  std::string trace_hash;                   // Python points_hash of the trace file (same rule)
  std::string trace_family;
  std::string trace_variant;
  std::string trace_split;
  std::string raw_measurement_path;         // the evidence file the record is appended to (stored as evidence::record_path)
  std::string dealer_mode;                  // announced by the dealer at CHALLENGE
  bool party_rng_os_seeded = false;         // session_caps::request_os_party_rng result
  bool weights_disclosed_to_p1 = false;     // verification mode
  bool model_from_public_seed = false;      // the weights derive from a public seed (not private to P0)
  bool contention = false;                  // public CPU load of the trace was running
};

// conform (evidence.hpp) + the context above + evidence::enforce. The result
// validates against the schema; security_eligible is true only when the
// highlevel session claimed it AND the dealer is isolated AND the party
// randomness is OS seeded AND the weights are private to P0 (not drawn from a
// public seed, not disclosed to P1 for verification).
highlevel::Value finalize_session_record(const highlevel::Value& session_record, const SessionRecordContext& ctx);

struct TinySessionOptions {
  std::string out_path;                       // evidence JSONL (one line per party per session)
  std::string journal_dir;                    // ledger journals; empty = TEMP
  std::string policy = "B4_opencrab";         // P1 proposes, P0 adopts
  std::string trace_id = "S0-stable-200M-0.5ms";
  double trace_bandwidth_bps = 200e6;
  double trace_one_way_delay_s = 0.0005;
  std::string trace_file;                     // P0: experiments/traces.py file; replaces the stable trace above
  std::string costmodel_path;                 // P0: fitted cost model JSON; empty = the HISTORICAL prior
  bool shape = true;                          // P0 decides, P1 adopts
  bool allow_insecure_dealer = false;         // refuse a dealer that is not ISOLATED_RECIPIENT_BUNDLES unless set
  bool model_from_public_seed = false;        // P0: the weights were drawn from a public seed (verification run)
  std::string code_commit = "cpp-tree-unknown";
  std::string implementation = "session_bridge";
  highlevel::ControllerConfig controller;     // P0 decides, P1 adopts
  std::string verify_weights_path;            // P1 test mode only
  std::function<void(const highlevel::Value&)> on_record;  // called with the finished record
};

// Installs a refill hook on the client that requests from the isolated dealer
// and ingests the verified bundle. trunc_faithful requests carry L =
// logical_bits (the dealer service defaults to 40 otherwise).
void attach_dealer_connection(highlevel::DealerClient& client, std::shared_ptr<DealerConnection> conn,
                              std::shared_ptr<std::mutex> conn_mu, int logical_bits);

// P0 side: a queue-backed highlevel::Channel driven by the handler thread.
class ServerSessionChannel : public highlevel::Channel {
 public:
  ServerSessionChannel(std::shared_ptr<highlevel::WallClockShaper> shaper, std::atomic<bool>* cancelled);
  ~ServerSessionChannel() override;
  // handler thread
  void push_inbound(std::vector<uint8_t> bytes);
  Expected<std::vector<uint8_t>> pop_outbound(double timeout_s);
  void abort();
  // session thread (highlevel::Channel interface)
  Expected<Unit> send(const highlevel::Value& obj) override;
  Expected<highlevel::Value> recv() override;
  Expected<highlevel::Value> exchange(const highlevel::Value& obj) override;
  void close() override;

 private:
  Expected<std::vector<uint8_t>> wait_inbound(double timeout_s);
  void apply_shaping(size_t nbytes);
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::vector<uint8_t>> inbound_, outbound_;
  bool aborted_ = false;
  std::atomic<bool>* cancelled_;
};

// P1 side: every channel operation is one MSG request against the model server.
class WsSessionChannel : public highlevel::Channel {
 public:
  WsSessionChannel(WebSocketChannel& ws, std::shared_ptr<highlevel::WallClockShaper> shaper,
                   std::function<bool()> cancelled);
  Expected<wire::Value> request(const std::string& op, const std::vector<uint8_t>* v, wire::Value extra = wire::Value::none());
  Expected<Unit> send(const highlevel::Value& obj) override;
  Expected<highlevel::Value> recv() override;
  Expected<highlevel::Value> exchange(const highlevel::Value& obj) override;
  void close() override;
  int64_t requests() const { return seq_; }

 private:
  void apply_shaping(size_t nbytes);
  WebSocketChannel& ws_;
  std::function<bool()> cancelled_;
  int64_t seq_ = 0;
};

// P0: protocol handler that runs the tiny decoder session for every WebSocket
// session. weights: P0's model (random or exported); dealer: connection as
// recipient 0. The server's model_version, setup_id, profile and contract hash
// are the ledger binding; the contract must be the default contract.
ProtocolHandler make_tiny_session_handler(std::shared_ptr<highlevel::TinyWeights> weights,
                                          std::shared_ptr<DealerConnection> dealer, std::vector<uint8_t> hmac_key,
                                          ModelServerConfig server_cfg, TinySessionOptions opts);

// P1: sidecar engine. The prompt bytes become token ids modulo the public
// vocabulary inside the sidecar process; n_tokens is the number of epochs.
ClientEngine make_tiny_session_engine(TinySessionOptions opts);

}  // namespace platform
}  // namespace opencrab2
