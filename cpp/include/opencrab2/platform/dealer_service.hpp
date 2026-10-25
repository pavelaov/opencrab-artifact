// opencrab2/platform/dealer_service.hpp
//
// The isolated dealer process (paper, Appendix B, DealerDelivery, and Appendix D, Dealer
// Isolation and Journal) and the dealer bundle wire format of the Python
// opencrab2.platform.dealer_service:
//
//   client:  HELLO{recipient, setup_id, contract_hash}
//   server:  CHALLENGE{nonce}
//   client:  AUTH{tag = hex(HMAC-SHA256(key_recipient, nonce))}
//   server:  AUTH_OK{}
//   client:  REQ{material_type, params(dict), count, gate_id?, request_id?}
//   server:  BUNDLE{header(dict), tag(str), remaining(int)} then one bytes frame (payload)
//
// Two request forms (both are served on the same connection):
//   * named (REQ carries request_id; the form the Python dealer client
//     uses): the request stands for `count` bundles, each holding ONE item of the
//     declared shape (one tuple of `shape`, one matrix triple, one wmask call, one
//     batch of M gates). One REQ returns one bundle; the client repeats the same
//     REQ while `remaining` > 0. Both recipients name the same request_id and get
//     the two halves of the same pairs in the same order. A request_id reused with
//     other parameters is refused (DEALER_REQUEST_CONFLICT); a recipient that asks
//     again after it drained the request is refused (DEALER_ALREADY_DELIVERED)
//     while its peer still has halves pending. A request that both recipients
//     drained is forgotten.
//   * unnamed (no request_id; the C++ highlevel client): ONE bundle holds all
//     `count` items (remaining = 0); identical REQ frames of the two recipients
//     pair first come, first served.
//   server:  ERROR{code} then close, on any failure (fail closed)
//
// Every message is a wire-codec dict with a "type" key, framed with the
// 8-byte big-endian length prefix over TCP. The bundle header is the dict
// {profile_id, setup_id, model_version, numeric_contract_hash, recipient,
// gate_id, material_type, material_id, count, bytes, params, parts,
// dealer_mode, insecure}, exactly the field set that the highlevel dealers of
// both languages seal (opencrab2/highlevel/dealer.hpp, python dealer.py);
// material_type is the plain base name. params is the request dict of the
// recipient (payload-like values such as a u64 array are replaced by their
// size, a header never carries material) completed with the binding
// spellings of the Python dealer next to the keys the C++ highlevel client reads:
// shape and elements (beaver), shape_a / shape_b and n, k, m (matrix_triple),
// shape and d, m (wmask_mask), d, m (wmask_call), shape and size
// (vector_share), M with n_bits, with_product, P (drelu), f, L
// (trunc_faithful), t (ring_to_field). The tag is the ONE bundle tag of the
// tree, highlevel::hmac_tag_with_payload: hex(HMAC-SHA256(key_recipient,
// len(canonical header) as 8-byte BE || canonical header || payload)), so
// opencrab2::highlevel::DealerClient::ingest verifies these bundles
// unchanged.
//
// Randomness (the same rule as the Python dealer): in ISOLATED_RECIPIENT_BUNDLES mode
// the master stream is the lowlevel CSPRNG, opencrab2::lowlevel::Rng seeded
// from the operating system (BCryptGenRandom on Windows, /dev/urandom
// elsewhere; the process aborts when the OS source is unavailable). The DCF
// based generators draw from that same stream. A seed exists only in
// INSECURE_TEST_ONLY_SHARED_SEED mode, whose bundles carry insecure = true.
// The CHALLENGE nonce always comes from the OS.
//
// Isolation: the dealer owns the master randomness. Both halves of a
// correlated tuple are produced in one generation; the requesting recipient
// receives its half immediately and the other half waits (bounded) for the
// peer's identical REQ. material_id is identical on both halves so the
// parties' ledgers can bind them. A recipient only ever sees bundles whose
// header names it; the service never sends a bundle to a connection whose
// authenticated recipient differs from header.recipient.
//
// Material generators and payload layouts (little-endian u64 unless stated):
//   beaver, kary_select  params {shape}             parts a, b, c (n = prod(shape) * count each)
//   matrix_triple        {n, k, m} or {shape_a, shape_b}   parts A, B, C, count repetitions
//   vector_share         {name, size}                      parts share (zero sharing, weight-blind);
//                        {name, values} hands a plaintext vector to the dealer: served by the
//                        insecure test mode only, DEALER_PLAINTEXT_REFUSED in the isolated mode
//   wmask_mask           {mat_id, d, m} or {mat_id, shape} parts B_share (B kept dealer-private)
//   wmask_call           {mat_id, n_rows}            parts A, C with C = A B, count repetitions
//   drelu                {M?, n_bits, with_product}  parts drelu_material (DreluMaterial::to_bytes)
//   trunc_faithful       {M?, f, L?}                 parts trunc_material (TruncMaterial::to_bytes)
//   ring_to_field        {M?, t?}                    parts r2f_material (R2FMaterial::to_bytes)
// The three DCF-based types are generated by the lowlevel headers
// when they are present at compile time (M defaults to count); without them
// a REQ for those types is answered ERROR{MATERIAL_TYPE_UNSUPPORTED} and
// lowlevel_material_available() returns false. register_generator adds or
// replaces a type.
//
// Dialogue: client-first HELLO as written above, the binding dialogue of the
// Python dealer service and client (paper, Appendix B, DealerDelivery).
//
// Codes: DEALER_BAD_MESSAGE, DEALER_UNKNOWN_RECIPIENT, SETUP_MISMATCH,
// CONTRACT_MISMATCH, AUTH_FAILED, DEALER_NOT_AUTHENTICATED,
// MATERIAL_TYPE_UNSUPPORTED, MATERIAL_BAD_PARAMS, MATERIAL_TOO_LARGE,
// MATERIAL_STALL, DEALER_PENDING_OVERFLOW, WMASK_NOT_INSTALLED,
// BUNDLE_BAD_TAG, BUNDLE_WRONG_RECIPIENT, BUNDLE_CONTRACT_MISMATCH,
// BUNDLE_SETUP_MISMATCH, MATERIAL_REUSE, BUNDLE_SIZE_MISMATCH, plus TCP_*.
//
// Tested: loopback dealer with two recipients receiving matching beaver
// halves (c = a*b after reconstruction, test-only cross check), tag
// verification through the highlevel verifier, wrong key rejected, wrong
// recipient rejected, material_id replay rejected, unsupported type rejected,
// header params in both spellings, no array content in a header, two
// isolated-mode dealers produce different material while two dealers with
// the same test seed produce identical material.
// Design limit: these are integrity tools, not a security proof; the
// per-recipient HMAC keys are distributed by the launcher out of band.
#pragma once

#include <opencrab2/platform/support.hpp>
#include <opencrab2/platform/tcp.hpp>

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace opencrab2 {
namespace platform {

struct DealerConfig {
  std::string profile_id = "SH2P-TrustedOffline-v1";
  std::string setup_id = "setup-0";
  std::string model_version = "gpt2-4L";
  std::string numeric_contract_hash;  // hex of NumericContract.contract_hash()
  std::string mode = "ISOLATED_RECIPIENT_BUNDLES";  // or INSECURE_TEST_ONLY_SHARED_SEED
  uint64_t test_seed = 0;                           // used only in the insecure test mode
  std::map<int, std::vector<uint8_t>> recipient_keys;  // recipient -> HMAC key
  double io_timeout_s = 30.0;
  uint64_t refill_bytes_per_s = 0;  // 0 = unlimited provisioning rate
  double max_refill_wait_s = 30.0;  // beyond this the REQ fails MATERIAL_STALL
  size_t max_payload_bytes = static_cast<size_t>(1) << 30;
  size_t max_pending_halves = 4096;
};

struct MaterialRequest {
  std::string material_type;  // base name: beaver, matrix_triple, ...
  wire::Value params;         // dict
  int64_t count = 1;
  std::string gate_id;
  int recipient = -1;
};

struct MaterialPair {
  std::vector<std::string> parts;
  std::vector<uint8_t> payload0;
  std::vector<uint8_t> payload1;
  int64_t count = 0;
  wire::Value header_params = wire::Value::make_dict();  // public shape keys added to the header params
};

// Persistent dealer state shared by generators (e.g. the wmask B matrix per
// mat_id). Accessed under the service mutex.
struct DealerState {
  std::map<std::string, std::vector<uint64_t>> matrices;
  std::map<std::string, std::pair<uint64_t, uint64_t>> matrix_shapes;  // mat_id -> (rows, cols)
};

using MaterialGenerator =
    std::function<Expected<MaterialPair>(const MaterialRequest& req, detail::ChaCha20& rng, DealerState& state)>;

struct Bundle {
  wire::Value header;  // dict
  std::string tag;     // hex HMAC
  std::vector<uint8_t> payload;
  std::string material_id() const { return header.str_or("material_id", ""); }
  std::string material_type() const { return header.str_or("material_type", ""); }
  int64_t count() const { return header.int_or("count", 0); }
  std::vector<std::string> parts() const;
};

// base + canonical JSON of params; used only as a pairing key.
std::string material_type_string(const std::string& base, const wire::Value& params);
// len(canonical header) 8-byte BE || canonical header || payload (empty when
// the header is not canonicalisable, i.e. carries bytes or an array).
std::vector<uint8_t> mac_input(const wire::Value& header, const std::vector<uint8_t>& payload);
// highlevel::hmac_tag_with_payload over the header and the payload; an empty
// string when the header is not canonicalisable or the key is empty.
std::string header_tag(const std::vector<uint8_t>& key, const wire::Value& header, const std::vector<uint8_t>& payload);
// A params dict without payload-like values: bytes and arrays are replaced by
// {"__bytes__": n} / {"__ndarray__": n} (sizes only).
wire::Value public_params(const wire::Value& params);
// hex(HMAC-SHA256(key, nonce bytes)), the AUTH tag of the dialogue above
std::string auth_tag(const std::vector<uint8_t>& key, const std::vector<uint8_t>& nonce);

// Ring helpers (mod 2^64, little-endian serialisation) shared by generators.
std::vector<uint8_t> u64_to_le(const std::vector<uint64_t>& v);
std::vector<uint64_t> le_to_u64(const uint8_t* p, size_t nbytes);
void ring_matmul(const std::vector<uint64_t>& A, size_t m, size_t k, const std::vector<uint64_t>& B, size_t n,
                 std::vector<uint64_t>& C);

class DealerService {
 public:
  DealerService(DealerConfig cfg, TcpListener&& listener);
  ~DealerService();
  DealerService(const DealerService&) = delete;
  DealerService& operator=(const DealerService&) = delete;

  void register_generator(const std::string& material_type, MaterialGenerator gen);
  void register_builtin_generators();
  bool supports(const std::string& material_type) const;
  std::vector<std::string> supported_types() const;
  static bool lowlevel_material_available();  // compiled with the lowlevel DCF generators

  Expected<bool> start();
  void stop();
  uint16_t port() const { return listener_.port(); }
  const DealerConfig& config() const { return cfg_; }

  struct Stats {
    uint64_t connections = 0;
    uint64_t auth_failures = 0;
    uint64_t bundles_sent = 0;
    uint64_t bytes_sent = 0;
    uint64_t requests_rejected = 0;
    uint64_t pending_halves = 0;
    uint64_t pairs_generated = 0;
  };
  Stats stats() const;

  // Generates (or pops the pending half of) the requested material and seals
  // it for `recipient`. Thread safe. Exposed for tests and for in-process use.
  Expected<Bundle> produce(const MaterialRequest& req);
  // Named request (see the file header): the next undelivered bundle of
  // `request_id` for req.recipient; `remaining` receives the number of bundles
  // this recipient still has to fetch. Thread safe.
  Expected<Bundle> produce_named(const MaterialRequest& req, const std::string& request_id, int64_t& remaining);

 private:
  struct PendingHalf {
    std::string material_id;
    std::vector<std::string> parts;
    std::vector<uint8_t> payload;
    int64_t count = 0;
    wire::Value header_params = wire::Value::make_dict();
  };
  struct NamedRequest {
    std::string fingerprint;                               // pending_key of the first REQ
    std::vector<std::pair<PendingHalf, PendingHalf>> pairs;  // (half of recipient 0, half of recipient 1)
    size_t next[2] = {0, 0};
  };
  // Runs the generator once and returns the two halves (first: recipient 0). Lock held by the caller.
  Expected<std::pair<PendingHalf, PendingHalf>> generate_halves(const MaterialRequest& req);
  // Seals one half for req.recipient (refill pacing included). Lock held by the caller.
  Expected<Bundle> seal_half(const MaterialRequest& req, PendingHalf& half);
  size_t pending_total() const;
  void accept_loop();
  void serve(std::shared_ptr<TcpSocket> sock);
  Expected<bool> send_msg(TcpSocket& sock, const wire::Value& v);
  Expected<wire::Value> recv_msg(TcpSocket& sock);
  void send_error(TcpSocket& sock, const std::string& code);
  std::string pending_key(const MaterialRequest& req) const;
  Expected<bool> wait_refill(size_t nbytes);
  wire::Value make_header(int recipient, const MaterialRequest& req, const std::string& material_id, int64_t count,
                          size_t nbytes, const std::vector<std::string>& parts, const wire::Value& header_params) const;

  DealerConfig cfg_;
  TcpListener listener_;
  std::map<std::string, MaterialGenerator> generators_;
  std::thread accept_thread_;
  std::atomic<bool> stopping_{false};
  std::atomic<bool> started_{false};
  mutable std::mutex mu_;
  std::vector<std::thread> workers_;
  std::vector<std::weak_ptr<TcpSocket>> live_;
  std::map<std::string, std::deque<PendingHalf>> pending_;
  std::map<std::string, NamedRequest> named_;
  DealerState state_;
  detail::ChaCha20 rng_;  // the lowlevel CSPRNG stream (OS seeded unless the insecure test mode is selected)
  uint64_t material_counter_ = 0;
  Stats stats_;
  double refill_tokens_ = 0.0;
  double refill_last_ = 0.0;
};

// Party-side connection to the dealer service (network client). Verifies
// every bundle: tag, recipient, setup, contract, material_id uniqueness,
// declared bytes == payload size.
class DealerConnection {
 public:
  DealerConnection() = default;
  static Expected<DealerConnection> connect(const std::string& host, uint16_t port, int recipient,
                                           const std::string& setup_id, const std::string& contract_hash,
                                           const std::vector<uint8_t>& hmac_key, double timeout_s = 30.0,
                                           int retries = 50);
  Expected<Bundle> request(const std::string& material_type, const wire::Value& params, int64_t count,
                           const std::string& gate_id = "");
  // Verification only (also used by tests on bundles produced in process).
  static Expected<bool> verify_bundle(const Bundle& b, int recipient, const std::string& setup_id,
                                      const std::string& contract_hash, const std::vector<uint8_t>& key,
                                      std::set<std::string>& seen_ids);
  uint64_t bytes_received() const { return bytes_received_; }
  uint64_t bundles_received() const { return bundles_received_; }
  const std::string& dealer_mode() const { return dealer_mode_; }  // from CHALLENGE
  // True when the dealer announced ISOLATED_RECIPIENT_BUNDLES. Callers that
  // produce evidence refuse any other dealer unless a test flag allows it.
  bool isolated() const { return dealer_mode_ == "ISOLATED_RECIPIENT_BUNDLES"; }
  // The binding the dealer stamps on every bundle header (announced at CHALLENGE).
  const std::string& model_version() const { return model_version_; }
  const std::string& profile_id() const { return profile_id_; }
  void close();
  bool valid() const { return sock_.valid(); }

 private:
  TcpSocket sock_;
  std::string dealer_mode_;
  std::string model_version_;
  std::string profile_id_;
  int recipient_ = -1;
  std::string setup_id_;
  std::string contract_hash_;
  std::vector<uint8_t> key_;
  std::set<std::string> seen_ids_;
  uint64_t bytes_received_ = 0;
  uint64_t bundles_received_ = 0;
};

// Reads a keys file: {"0": "<hex key>", "1": "<hex key>"} or
// {"recipients": {"0": "...", "1": "..."}}. Generates one with
// write_random_keys_file when missing (launcher convenience).
Expected<std::map<int, std::vector<uint8_t>>> load_keys_file(const std::string& path);
Expected<bool> write_random_keys_file(const std::string& path);

}  // namespace platform
}  // namespace opencrab2
