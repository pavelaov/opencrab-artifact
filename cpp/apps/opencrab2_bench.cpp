// opencrab2_bench: native timings of the gates and of the platform primitives
// as evidence records (schema of python/opencrab2/experiments/schema.py,
// validated here before every write).
//
//   opencrab2_bench [--out FILE] [--only LIST] [--batches 64,256,1024,4096] [--repeats 7]
//                   [--dealer host:port --keys keys.json [--setup-id ID]]
//                   [--argmax-n 256,4096,50257] [--argmax-k 2,4,8] [--argmax-repeats R]
//                   [--argmax-on-demand] [--bridge-n 128] [--seed N] [--help]
//   opencrab2_bench --print-schema     the evidence vocabulary of this build as one JSON line
//                                      (compared with schema.py by tests/smoke_three_process.py)
//
// Dealer mode and evidence status (one mode per run, stated in every record):
//   with --dealer and --keys   the material comes from the isolated dealer process
//       (opencrab2_dealer, OS seeded, recipient-bound bundles over two authenticated
//       connections, one per recipient). dealer_mode is what that dealer announces;
//       when it is ISOLATED_RECIPIENT_BUNDLES the records are MEASURED_NATIVE.
//   without --dealer           the material is generated in this process from the
//       public --seed (shared seed). dealer_mode is INSECURE_TEST_ONLY_SHARED_SEED
//       and every record of the run is DIAGNOSTIC (paper, Section 3, evidence classes).
// security_eligible is false in every record of this app: both parties' halves
// are evaluated inside one process, so a bench line is a component timing, not
// a session under the profile (extra.security_eligible_reason says so).
//
// Benches (--only names, comma separated):
//   dcf         eval_dcf_batch of one party's key, n_bits 32 (the p key of the P = 1
//               DReLU material) and 64 (the wrap key of the truncation material)
//   drelu       one-round DReLU, n_bits 32, P = 1 (bit only) and P = 2 (bit and r b):
//               material request / generation and the online evaluation of both parties
//   trunc       faithful truncation f = 16, L = 40: material and online (offset, open, eval)
//   r2f         ring_to_field at t = 68720050177: material and online
//   argmax      fused k-ary argmax of the highlevel Party (Party::argmax_row) over one row of
//               n shared logits for k in --argmax-k and n in --argmax-n: two party threads over
//               a loopback highlevel::Channel, index released to P1 only; checks the released
//               index against the plaintext lowest-index argmax and records rounds, protocol
//               bytes and material bytes. The material of one tournament (tournament_counts of
//               the cost model: pairwise DReLUs, payload DReLUs, selects) is provisioned from
//               the run's dealer BEFORE the timed call, as a dealer that runs ahead of the online
//               phase does: argmax_k<K>_material is that offline step, argmax_k<K> the online
//               gate. --argmax-on-demand skips the provisioning (every pool miss inside the gate
//               then costs a dealer request and journal writes; diagnostic of that path).
//               The logits are drawn in [-8, 8), inside the declared magnitude bound of the
//               fused nodes (paper, Section 5.3); the bound for (k, n) is recorded and checked.
//   toy_bridge  the local reference path (a2h_toy_local + matvec_pt + decrypt) and, when
//               highlevel/bridge.hpp is present, the two-party secure bridge a2h_toy ->
//               eval_toy -> h2a_ref (N = --bridge-n, d_in = d_out = 8, t = 2^16) with the
//               L = 64 truncation material provisioned before the call
//   platform    wire codec round trip, HMAC-SHA256, WebSocket loopback echo, Beaver pair
//               delivery and combine
//
// Record shape: the schema fields flat (workload_scope primitive, record_kind primitive,
// clock_basis process-perf-counter, T_online_s = median seconds of the timed call, bytes
// and rounds of the online phase, material_* = bytes of dealer material per run,
// output_match = the correctness check where one exists) and under extra: language "cpp",
// implementation, layer, component, params, batch, repeats, seconds (every repeat),
// median_s, min_s, mean_s, p95_s, per_element_s, ns_per_element, material_wait_s,
// uses_dealer_material, host. Timing uses the steady clock around the timed call only;
// for gates that can be re-evaluated on the same material the first repeat warms caches
// and is excluded. Nothing here is modelled or estimated.
//
// Without --out the records are printed to stdout only. Exit status 0 when at
// least one record was produced and every record validated; 1 otherwise.
#include <opencrab2/highlevel/party.hpp>
#include <opencrab2/highlevel/session.hpp>
#include <opencrab2/lowlevel/csprng.hpp>
#include <opencrab2/lowlevel/dcf.hpp>
#include <opencrab2/lowlevel/drelu.hpp>
#include <opencrab2/lowlevel/ring_to_field.hpp>
#include <opencrab2/lowlevel/toy_bfv.hpp>
#include <opencrab2/lowlevel/trunc.hpp>
#include <opencrab2/platform/dealer_service.hpp>
#include <opencrab2/platform/evidence.hpp>
#include <opencrab2/platform/http.hpp>
#include <opencrab2/platform/model_server.hpp>
#include <opencrab2/platform/session_bridge.hpp>
#include <opencrab2/platform/support.hpp>
#include <opencrab2/platform/tcp.hpp>
#include <opencrab2/platform/websocket.hpp>

#if __has_include(<opencrab2/highlevel/bridge.hpp>)
#include <opencrab2/highlevel/bridge.hpp>
#define BENCH_HAVE_SECURE_BRIDGE 1
#endif

#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace opencrab2::platform;
using opencrab2::Expected;
using opencrab2::Unit;
namespace hl = opencrab2::highlevel;
namespace ll = opencrab2::lowlevel;

namespace {

const char* const kCodeCommit = OPENCRAB2_CODE_COMMIT_LITERAL;
const char* const kIsolated = "ISOLATED_RECIPIENT_BUNDLES";
const char* const kSharedSeed = "INSECURE_TEST_ONLY_SHARED_SEED";

int usage(const char* prog, int code) {
  std::fprintf(code == 0 ? stdout : stderr,
               "usage: %s [--out FILE] [--only dcf,drelu,trunc,r2f,argmax,toy_bridge,platform]\n"
               "          [--batches 64,256,1024,4096] [--repeats 7] [--seed N]\n"
               "          [--dealer host:port --keys keys.json [--setup-id ID]]\n"
               "          [--argmax-n 256,4096,50257] [--argmax-k 2,4,8] [--argmax-repeats R] [--argmax-on-demand]\n"
               "          [--bridge-n 128]\n"
               "  with --dealer: material from the isolated dealer process, records MEASURED_NATIVE;\n"
               "  without: shared-seed material generated in process, every record DIAGNOSTIC.\n",
               prog);
  return code;
}

std::string temp_dir() {
  const char* base = std::getenv("TEMP");
  if (!base) base = std::getenv("TMPDIR");
  if (!base) base = "/tmp";
  return base;
}

std::vector<double> time_runs(const std::function<void()>& fn, int repeats, bool drop_first) {
  std::vector<double> t;
  for (int i = 0; i < repeats + (drop_first ? 1 : 0); ++i) {
    double a = monotonic_s();
    fn();
    double b = monotonic_s();
    if (!drop_first || i > 0) t.push_back(b - a);
  }
  return t;
}

// ---------------------------------------------------------------------------
// Material source: the isolated dealer process or the shared-seed dealer of
// this process. One per run; it decides dealer_mode and the record status.
// ---------------------------------------------------------------------------
class MaterialSource {
 public:
  virtual ~MaterialSource() = default;
  virtual std::string dealer_mode() const = 0;
  virtual Expected<std::pair<ll::drelu::DreluMaterial, ll::drelu::DreluMaterial>> drelu(size_t M, int n_bits, bool with_product) = 0;
  virtual Expected<std::pair<ll::trunc::TruncMaterial, ll::trunc::TruncMaterial>> trunc(size_t M, int f, int L) = 0;
  virtual Expected<std::pair<ll::r2f::R2FMaterial, ll::r2f::R2FMaterial>> r2f(size_t M, uint64_t t) = 0;
  // Beaver halves (a, b, c) of n elements for both recipients.
  virtual Expected<std::pair<demo::BeaverHalf, demo::BeaverHalf>> beaver(size_t n) = 0;
  // Wires the DealerClient of `party` to this source (on-demand refill).
  virtual void attach(hl::DealerClient& client, int party) = 0;
  // Faithful-truncation material with an explicit logical width, ingested into both clients.
  virtual Expected<bool> provision_trunc(hl::DealerClient& c0, hl::DealerClient& c1, size_t M, int f, int L) = 0;
  // One bundle pair of `type` (params and count in the spelling of the DealerClient refill
  // requests) ingested into both clients ahead of use; returns the bytes of one half.
  virtual Expected<int64_t> provision(hl::DealerClient& c0, hl::DealerClient& c1, const std::string& type,
                                      const hl::Value& params, size_t count) = 0;
  virtual hl::Value binding() const = 0;  // ledger binding of the clients
  virtual hl::Bytes key(int party) const = 0;
  // Cumulative seconds spent producing / delivering material for attached clients.
  virtual double wait_seconds() const = 0;
};

class SeededSource : public MaterialSource {
 public:
  SeededSource(uint64_t seed, const hl::NumericContract& contract) : rng_(ll::Rng::from_seed(seed)) {
    dc_.setup_id = "bench-setup";
    dc_.model_version = "bench";
    dc_.numeric_contract_hash = contract.contract_hash();
    dc_.mode = kSharedSeed;
    dc_.dcf_domain_bits = contract.dcf_domain_bits;
    dc_.frac_bits = contract.frac_bits;
    dc_.logical_bits = contract.logical_bits;
    key_[0] = hl::Bytes(32, 0x51);
    key_[1] = hl::Bytes(32, 0x52);
    dealer_ = std::make_shared<hl::Dealer>(dc_, ll::Rng::from_seed(seed + 1), key_[0], key_[1]);
    link_ = std::make_shared<hl::InProcessDealerLink>(dealer_);
  }
  std::string dealer_mode() const override { return kSharedSeed; }
  Expected<std::pair<ll::drelu::DreluMaterial, ll::drelu::DreluMaterial>> drelu(size_t M, int n_bits, bool with_product) override {
    return ll::drelu::drelu_material(M, n_bits, rng_, with_product);
  }
  Expected<std::pair<ll::trunc::TruncMaterial, ll::trunc::TruncMaterial>> trunc(size_t M, int f, int L) override {
    return ll::trunc::trunc_material(M, f, L, rng_);
  }
  Expected<std::pair<ll::r2f::R2FMaterial, ll::r2f::R2FMaterial>> r2f(size_t M, uint64_t t) override {
    return ll::r2f::r2f_material(M, t, rng_);
  }
  Expected<std::pair<demo::BeaverHalf, demo::BeaverHalf>> beaver(size_t n) override {
    std::pair<demo::BeaverHalf, demo::BeaverHalf> out;
    std::vector<uint64_t> a(n), b(n), c(n);
    for (size_t i = 0; i < n; ++i) {
      a[i] = rng_.next_u64();
      b[i] = rng_.next_u64();
      c[i] = a[i] * b[i];
    }
    auto as = ll::sharing::share(a, rng_), bs = ll::sharing::share(b, rng_), cs = ll::sharing::share(c, rng_);
    out.first.a = as.first;
    out.first.b = bs.first;
    out.first.c = cs.first;
    out.second.a = as.second;
    out.second.b = bs.second;
    out.second.c = cs.second;
    return make_ok(std::move(out));
  }
  void attach(hl::DealerClient& client, int) override { link_->attach(client); }
  Expected<bool> provision_trunc(hl::DealerClient& c0, hl::DealerClient& c1, size_t M, int f, int L) override {
    auto bp = dealer_->make_trunc(M, f, L);
    if (!bp.ok) return make_fail<bool>(bp.code);
    auto i0 = c0.ingest(bp.value.for0);
    if (!i0.ok) return make_fail<bool>(i0.code);
    auto i1 = c1.ingest(bp.value.for1);
    if (!i1.ok) return make_fail<bool>(i1.code);
    return make_ok(true);
  }
  Expected<int64_t> provision(hl::DealerClient& c0, hl::DealerClient& c1, const std::string& type, const hl::Value& params,
                              size_t count) override {
    auto bp = dealer_->make(type, params, count);
    if (!bp.ok) return make_fail<int64_t>(bp.code);
    auto i0 = c0.ingest(bp.value.for0);
    if (!i0.ok) return make_fail<int64_t>(i0.code);
    auto i1 = c1.ingest(bp.value.for1);
    if (!i1.ok) return make_fail<int64_t>(i1.code);
    return make_ok(static_cast<int64_t>(bp.value.for0.payload.size()));
  }
  hl::Value binding() const override {
    return hl::dict({{"setup_id", dc_.setup_id}, {"model_version", dc_.model_version},
                     {"numeric_contract_hash", dc_.numeric_contract_hash}, {"profile_id", dc_.profile_id}});
  }
  hl::Bytes key(int party) const override { return key_[party]; }
  double wait_seconds() const override { return link_->generation_seconds(); }

 private:
  ll::Rng rng_;
  hl::DealerConfig dc_;
  hl::Bytes key_[2];
  std::shared_ptr<hl::Dealer> dealer_;
  std::shared_ptr<hl::InProcessDealerLink> link_;
};

class IsolatedSource : public MaterialSource {
 public:
  static Expected<std::shared_ptr<IsolatedSource>> connect(const std::string& host, uint16_t port, const std::string& keys_path,
                                                           const std::string& setup_id, const hl::NumericContract& contract) {
    using R = std::shared_ptr<IsolatedSource>;
    auto keys = load_keys_file(keys_path);
    if (!keys.ok) return make_fail<R>(keys.code);
    auto src = std::shared_ptr<IsolatedSource>(new IsolatedSource());
    src->setup_id_ = setup_id;
    src->contract_hash_ = contract.contract_hash();
    src->logical_bits_ = contract.logical_bits;
    for (int p = 0; p < 2; ++p) {
      auto c = DealerConnection::connect(host, port, p, setup_id, src->contract_hash_, keys.value[p], 60.0, 100);
      if (!c.ok) return make_fail<R>(c.code);
      src->key_[p] = hl::Bytes(keys.value[p].begin(), keys.value[p].end());
      src->conn_[p] = std::make_shared<DealerConnection>(std::move(c.value));
    }
    if (src->conn_[0]->dealer_mode() != src->conn_[1]->dealer_mode()) return make_fail<R>("DEALER_BAD_MESSAGE");
    return make_ok(std::move(src));
  }
  ~IsolatedSource() override {
    for (auto& c : conn_) {
      if (c) c->close();
    }
  }
  std::string dealer_mode() const override { return conn_[0]->isolated() ? kIsolated : kSharedSeed; }

  Expected<std::pair<ll::drelu::DreluMaterial, ll::drelu::DreluMaterial>> drelu(size_t M, int n_bits, bool with_product) override {
    using R = std::pair<ll::drelu::DreluMaterial, ll::drelu::DreluMaterial>;
    wire::Value p = wire::Value::make_dict();
    p.set("M", wire::Value::from_int(static_cast<int64_t>(M)));
    p.set("n_bits", wire::Value::from_int(n_bits));
    p.set("with_product", wire::Value::from_bool(with_product));
    auto b = request_pair("drelu", p);
    if (!b.ok) return make_fail<R>(b.code);
    auto m0 = ll::drelu::DreluMaterial::from_bytes(b.value.first.payload);
    auto m1 = ll::drelu::DreluMaterial::from_bytes(b.value.second.payload);
    if (!m0.ok || !m1.ok) return make_fail<R>(m0.ok ? m1.code : m0.code);
    return make_ok(R(std::move(m0.value), std::move(m1.value)));
  }
  Expected<std::pair<ll::trunc::TruncMaterial, ll::trunc::TruncMaterial>> trunc(size_t M, int f, int L) override {
    using R = std::pair<ll::trunc::TruncMaterial, ll::trunc::TruncMaterial>;
    auto b = request_pair("trunc_faithful", trunc_params(M, f, L));
    if (!b.ok) return make_fail<R>(b.code);
    auto m0 = ll::trunc::TruncMaterial::from_bytes(b.value.first.payload);
    auto m1 = ll::trunc::TruncMaterial::from_bytes(b.value.second.payload);
    if (!m0.ok || !m1.ok) return make_fail<R>(m0.ok ? m1.code : m0.code);
    return make_ok(R(std::move(m0.value), std::move(m1.value)));
  }
  Expected<std::pair<ll::r2f::R2FMaterial, ll::r2f::R2FMaterial>> r2f(size_t M, uint64_t t) override {
    using R = std::pair<ll::r2f::R2FMaterial, ll::r2f::R2FMaterial>;
    wire::Value p = wire::Value::make_dict();
    p.set("M", wire::Value::from_int(static_cast<int64_t>(M)));
    p.set("t", wire::Value::from_int(static_cast<int64_t>(t)));
    auto b = request_pair("ring_to_field", p);
    if (!b.ok) return make_fail<R>(b.code);
    auto m0 = ll::r2f::R2FMaterial::from_bytes(b.value.first.payload);
    auto m1 = ll::r2f::R2FMaterial::from_bytes(b.value.second.payload);
    if (!m0.ok || !m1.ok) return make_fail<R>(m0.ok ? m1.code : m0.code);
    return make_ok(R(std::move(m0.value), std::move(m1.value)));
  }
  Expected<std::pair<demo::BeaverHalf, demo::BeaverHalf>> beaver(size_t n) override {
    using R = std::pair<demo::BeaverHalf, demo::BeaverHalf>;
    wire::Value p = wire::Value::make_dict();
    p.set("shape", wire::Value::from_list({wire::Value::from_int(static_cast<int64_t>(n))}));
    auto b = request_pair("beaver", p);
    if (!b.ok) return make_fail<R>(b.code);
    auto h0 = demo::parse_beaver_half(b.value.first, n);
    auto h1 = demo::parse_beaver_half(b.value.second, n);
    if (!h0.ok || !h1.ok) return make_fail<R>(h0.ok ? h1.code : h0.code);
    return make_ok(R(std::move(h0.value), std::move(h1.value)));
  }
  void attach(hl::DealerClient& client, int party) override {
    hl::DealerClient* cp = &client;
    client.set_refill([this, cp, party](const std::string& type, const hl::Value& params, size_t count) -> Expected<Unit> {
      double t0 = monotonic_s();
      hl::Value p = params;
      if (type == "trunc_faithful" && !p.has("L")) p.set("L", static_cast<int64_t>(logical_bits_));
      Expected<Bundle> b;
      {
        std::lock_guard<std::mutex> lk(mu_[party]);
        b = conn_[party]->request(type, wire::Value::from_highlevel(p), static_cast<int64_t>(count), "*");
      }
      {
        std::lock_guard<std::mutex> lk(wait_mu_);
        wait_[party] += monotonic_s() - t0;
      }
      if (!b.ok) return Expected<Unit>::failure(b.code);
      return cp->ingest(b.value.header.to_highlevel(), b.value.payload, b.value.tag);
    });
  }
  Expected<bool> provision_trunc(hl::DealerClient& c0, hl::DealerClient& c1, size_t M, int f, int L) override {
    auto b = request_pair("trunc_faithful", trunc_params(M, f, L));
    if (!b.ok) return make_fail<bool>(b.code);
    auto i0 = c0.ingest(b.value.first.header.to_highlevel(), b.value.first.payload, b.value.first.tag);
    if (!i0.ok) return make_fail<bool>(i0.code);
    auto i1 = c1.ingest(b.value.second.header.to_highlevel(), b.value.second.payload, b.value.second.tag);
    if (!i1.ok) return make_fail<bool>(i1.code);
    return make_ok(true);
  }
  Expected<int64_t> provision(hl::DealerClient& c0, hl::DealerClient& c1, const std::string& type, const hl::Value& params,
                              size_t count) override {
    const std::string gate = "bench-prov:" + std::to_string(counter_++);
    Expected<Bundle> b0, b1;
    {
      std::lock_guard<std::mutex> lk(mu_[0]);
      b0 = conn_[0]->request(type, wire::Value::from_highlevel(params), static_cast<int64_t>(count), gate);
    }
    if (!b0.ok) return make_fail<int64_t>(b0.code);
    {
      std::lock_guard<std::mutex> lk(mu_[1]);
      b1 = conn_[1]->request(type, wire::Value::from_highlevel(params), static_cast<int64_t>(count), gate);
    }
    if (!b1.ok) return make_fail<int64_t>(b1.code);
    auto i0 = c0.ingest(b0.value.header.to_highlevel(), b0.value.payload, b0.value.tag);
    if (!i0.ok) return make_fail<int64_t>(i0.code);
    auto i1 = c1.ingest(b1.value.header.to_highlevel(), b1.value.payload, b1.value.tag);
    if (!i1.ok) return make_fail<int64_t>(i1.code);
    return make_ok(static_cast<int64_t>(b0.value.payload.size()));
  }
  hl::Value binding() const override {
    // setup and contract are this run's choice; model_version and profile are the ones the dealer announced
    hl::Value b = hl::dict({{"setup_id", setup_id_}, {"numeric_contract_hash", contract_hash_}});
    if (!conn_[0]->model_version().empty()) b.set("model_version", conn_[0]->model_version());
    if (!conn_[0]->profile_id().empty()) b.set("profile_id", conn_[0]->profile_id());
    return b;
  }
  hl::Bytes key(int party) const override { return key_[party]; }
  double wait_seconds() const override {
    std::lock_guard<std::mutex> lk(wait_mu_);
    return std::max(wait_[0], wait_[1]);  // both parties wait for the same generation at the same protocol point
  }

 private:
  IsolatedSource() = default;
  static wire::Value trunc_params(size_t M, int f, int L) {
    wire::Value p = wire::Value::make_dict();
    p.set("M", wire::Value::from_int(static_cast<int64_t>(M)));
    p.set("f", wire::Value::from_int(f));
    p.set("L", wire::Value::from_int(L));
    return p;
  }
  // The same REQ on both recipient connections; the dealer pairs the halves by (type, params, count, gate).
  Expected<std::pair<Bundle, Bundle>> request_pair(const std::string& type, const wire::Value& params) {
    using R = std::pair<Bundle, Bundle>;
    const std::string gate = "bench:" + std::to_string(counter_++);
    Expected<Bundle> b0, b1;
    {
      std::lock_guard<std::mutex> lk(mu_[0]);
      b0 = conn_[0]->request(type, params, 1, gate);
    }
    if (!b0.ok) return make_fail<R>(b0.code);
    {
      std::lock_guard<std::mutex> lk(mu_[1]);
      b1 = conn_[1]->request(type, params, 1, gate);
    }
    if (!b1.ok) return make_fail<R>(b1.code);
    if (b0.value.material_id() != b1.value.material_id()) return make_fail<R>("DEALER_PAIRING_MISMATCH");
    return make_ok(R(std::move(b0.value), std::move(b1.value)));
  }
  std::shared_ptr<DealerConnection> conn_[2];
  std::mutex mu_[2];
  mutable std::mutex wait_mu_;
  double wait_[2] = {0.0, 0.0};
  hl::Bytes key_[2];
  std::string setup_id_;
  std::string contract_hash_;
  int logical_bits_ = 40;
  uint64_t counter_ = 0;
};

// ---------------------------------------------------------------------------
// Record sink: builds, validates and writes one evidence record per timing.
// ---------------------------------------------------------------------------
struct Measurement {
  std::string component;                 // gate name
  std::string layer;                     // lowlevel | highlevel | platform
  hl::Value params = hl::dict({});
  size_t batch = 0;
  std::vector<double> seconds;           // one per repeat (already without the warm-up run)
  int64_t online_bytes = 0;              // protocol bytes of the timed call (0 for local work)
  int64_t rounds = 0;
  int64_t material_bytes = 0;            // dealer material consumed by one run (one party)
  double material_wait_s = 0.0;          // dealer generation / delivery inside the timed call (median run)
  bool uses_dealer_material = false;
  int correct = -1;                      // -1 no check, 0 failed, 1 passed
  std::string note;
  hl::Value more = hl::dict({});         // additional extra fields
};

struct Sink {
  std::string out_path;
  std::string run_tag;
  std::string dealer_mode;
  std::string contract_hash;
  hl::Value host;
  int written = 0;
  int invalid = 0;
  int incorrect = 0;

  void record(const Measurement& m) {
    std::vector<double> t = m.seconds;
    if (t.empty()) return;
    std::sort(t.begin(), t.end());
    const double median = t[t.size() / 2];
    double sum = 0.0;
    for (double v : t) sum += v;
    const double mean = sum / static_cast<double>(t.size());
    size_t p95i = static_cast<size_t>(std::ceil(0.95 * static_cast<double>(t.size())));
    if (p95i < 1) p95i = 1;
    const double p95 = t[std::min(t.size() - 1, p95i - 1)];
    const double online_s = std::max(0.0, median - m.material_wait_s);

    hl::Value r = evidence::new_record(kCodeCommit);
    // Evidence class: a shared seed makes the record DIAGNOSTIC; MEASURED_NATIVE only with the isolated dealer.
    r.set("status", dealer_mode == kIsolated ? "MEASURED_NATIVE" : "DIAGNOSTIC");
    r.set("security_eligible", false);
    r.set("numeric_contract_hash", contract_hash);
    r.set("dealer_mode", dealer_mode);
    r.set("workload_scope", "primitive");
    r.set("record_kind", "primitive");
    r.set("clock_basis", "process-perf-counter");
    r.set("run_id", "bench:" + m.component + ":" + std::to_string(m.batch) + ":" + run_tag + ":" + std::to_string(written));
    r.set("T_online_s", online_s);
    r.set("bytes", m.online_bytes);
    r.set("rounds", m.rounds);
    r.set("material_generated", m.material_bytes * (m.uses_dealer_material ? 2 : 0));  // both recipients' halves
    r.set("material_transferred", m.uses_dealer_material ? m.material_bytes : static_cast<int64_t>(0));
    r.set("material_consumed", m.uses_dealer_material ? m.material_bytes : static_cast<int64_t>(0));
    r.set("peak_rss_client", static_cast<int64_t>(hl::peak_rss_bytes()));
    r.set("output_match", m.correct < 0 ? hl::Value(hl::None{}) : hl::Value(m.correct == 1));
    r.set("raw_measurement_path", evidence::record_path(out_path));
    r.set("label", m.component);
    hl::List secs;
    for (double v : m.seconds) secs.push_back(hl::Value(v));
    hl::Value extra = hl::dict({{"language", "cpp"},
                                {"implementation", "opencrab2_bench"},
                                {"layer", m.layer},
                                {"component", m.component},
                                {"params", m.params},
                                {"batch", static_cast<int64_t>(m.batch)},
                                {"repeats", static_cast<int64_t>(m.seconds.size())},
                                {"seconds", hl::Value(std::move(secs))},
                                {"median_s", median},
                                {"min_s", t.front()},
                                {"mean_s", mean},
                                {"p95_s", p95},
                                {"per_element_s", m.batch ? online_s / static_cast<double>(m.batch) : 0.0},
                                {"ns_per_element", m.batch ? online_s * 1e9 / static_cast<double>(m.batch) : 0.0},
                                {"material_wait_s", m.material_wait_s},
                                {"material_bytes", m.material_bytes},
                                {"uses_dealer_material", m.uses_dealer_material},
                                {"note", m.note},
                                {"security_eligible_reason",
                                 "primitive timing: both recipients' halves are evaluated in one process; not a session under the profile"},
                                {"host", host}});
    if (const hl::Dict* d = std::get_if<hl::Dict>(&m.more.v)) {
      for (const auto& e : *d) extra.set(e.key, e.val);
    }
    r.set("extra", extra);
    evidence::enforce(r);
    auto v = evidence::validate(r);
    if (!v.ok) {
      ++invalid;
      std::fprintf(stderr, "%s: record of %s rejected by the evidence schema\n", v.code.c_str(), m.component.c_str());
      return;
    }
    if (m.correct == 0) {
      ++incorrect;
      std::fprintf(stderr, "BENCH_INCORRECT: %s batch %zu produced a wrong result\n", m.component.c_str(), m.batch);
    }
    auto text = hl::canonical_json(r);
    if (!text.ok) {
      ++invalid;
      return;
    }
    if (!out_path.empty()) {
      auto ap = evidence::append(out_path, r);
      if (!ap.ok) {
        ++invalid;
        std::fprintf(stderr, "%s: cannot append to the evidence file\n", ap.code.c_str());
        return;
      }
    }
    std::printf("%s\n", text.value.c_str());
    std::fflush(stdout);
    ++written;
  }
};

bool wants(const std::vector<std::string>& only, const std::string& name) {
  if (only.empty()) return true;
  for (const auto& o : only) {
    if (o == name) return true;
  }
  return false;
}

uint64_t low_mask(int bits) { return bits >= 64 ? ~uint64_t(0) : ((uint64_t(1) << bits) - 1); }

// ---------------------------------------------------------------------------
// lowlevel gates
// ---------------------------------------------------------------------------
void bench_drelu_and_dcf32(Sink& sink, MaterialSource& src, const std::vector<size_t>& batches, int repeats, bool do_drelu,
                           bool do_dcf, ll::Rng& rng) {
  const int n_bits = 32;
  for (bool with_product : {false, true}) {
    for (size_t M : batches) {
      Expected<std::pair<ll::drelu::DreluMaterial, ll::drelu::DreluMaterial>> mat;
      auto tg = time_runs([&]() { mat = src.drelu(M, n_bits, with_product); }, 1, false);
      if (!mat.ok) {
        std::fprintf(stderr, "%s: drelu material\n", mat.code.c_str());
        return;
      }
      const ll::drelu::DreluMaterial& m0 = mat.value.first;
      const ll::drelu::DreluMaterial& m1 = mat.value.second;
      hl::Value params = hl::dict({{"n_bits", static_cast<int64_t>(n_bits)}, {"P", static_cast<int64_t>(with_product ? 2 : 1)}});
      // x in the admitted signed range of the comparison domain, y = x + r public
      std::vector<uint64_t> x(M), y(M), r(M);
      for (size_t i = 0; i < M; ++i) {
        int64_t xs = static_cast<int64_t>(rng.next_u64() & low_mask(n_bits - 1)) - (int64_t(1) << (n_bits - 2));
        x[i] = static_cast<uint64_t>(xs);
        r[i] = m0.r_share[i] + m1.r_share[i];
        y[i] = x[i] + r[i];
      }
      if (do_drelu) {
        Measurement off;
        off.component = with_product ? "drelu_p2_material" : "drelu_p1_material";
        off.layer = "lowlevel";
        off.params = params;
        off.params.set("phase", "offline");
        off.batch = M;
        off.seconds = tg;
        off.material_bytes = static_cast<int64_t>(m0.nbytes());
        off.uses_dealer_material = true;
        off.note = "dealer generation (shared-seed run) or request and delivery of both halves (isolated dealer)";
        sink.record(off);

        ll::drelu::DreluOutput o0, o1;
        auto to = time_runs(
            [&]() {
              auto a = ll::drelu::drelu_eval(m0, y);
              auto b = ll::drelu::drelu_eval(m1, y);
              if (a.ok) o0 = std::move(a.value);
              if (b.ok) o1 = std::move(b.value);
            },
            repeats, true);
        bool good = o0.b.size() == M && o1.b.size() == M;
        for (size_t i = 0; i < M && good; ++i) {
          const uint64_t bit = static_cast<int64_t>(x[i]) >= 0 ? 1 : 0;
          if (o0.b[i] + o1.b[i] != bit) good = false;
          if (with_product && (o0.rb.size() != M || o0.rb[i] + o1.rb[i] != r[i] * bit)) good = false;
        }
        Measurement on;
        on.component = with_product ? "drelu_p2" : "drelu_p1";
        on.layer = "lowlevel";
        on.params = params;
        on.params.set("phase", "online_both_parties");
        on.batch = M;
        on.seconds = to;
        on.online_bytes = static_cast<int64_t>(16 * M);  // the single open of y: 8 bytes per element and direction
        on.rounds = 1;
        on.material_bytes = static_cast<int64_t>(m0.nbytes());
        on.uses_dealer_material = true;
        on.correct = good ? 1 : 0;
        on.note = "drelu_eval of both parties on the opened y (local work of the one-round gate)";
        sink.record(on);
      }
      if (do_dcf && !with_product) {
        // one party's evaluation of the p key: shares of 1{y_n < r mod 2^n}
        std::vector<uint64_t> yn(M);
        for (size_t i = 0; i < M; ++i) yn[i] = y[i] & low_mask(n_bits);
        std::vector<uint64_t> out0, out1;
        auto td = time_runs(
            [&]() {
              auto e = ll::dcf::eval_dcf_batch(m0.key_p, yn);
              if (e.ok) out0 = std::move(e.value);
            },
            repeats, true);
        auto e1 = ll::dcf::eval_dcf_batch(m1.key_p, yn);
        bool good = e1.ok && out0.size() == M;
        if (good) out1 = std::move(e1.value);
        for (size_t i = 0; i < M && good; ++i) {
          const uint64_t want = yn[i] < (r[i] & low_mask(n_bits)) ? 1 : 0;
          if (out0[i] + out1[i] != want) good = false;
        }
        Measurement d;
        d.component = "dcf_eval";
        d.layer = "lowlevel";
        d.params = hl::dict({{"n_bits", static_cast<int64_t>(n_bits)}, {"P", static_cast<int64_t>(1)}, {"modulus", static_cast<int64_t>(0)},
                             {"key", "drelu p key"}});
        d.batch = M;
        d.seconds = td;
        d.material_bytes = static_cast<int64_t>(m0.key_p.nbytes());
        d.uses_dealer_material = true;
        d.correct = good ? 1 : 0;
        d.note = "eval_dcf_batch of one party";
        sink.record(d);
      }
    }
  }
}

void bench_trunc_and_dcf64(Sink& sink, MaterialSource& src, const std::vector<size_t>& batches, int repeats, bool do_trunc,
                           bool do_dcf, ll::Rng& rng, const hl::NumericContract& contract) {
  const int f = contract.frac_bits, L = contract.logical_bits;
  for (size_t M : batches) {
    Expected<std::pair<ll::trunc::TruncMaterial, ll::trunc::TruncMaterial>> mat;
    auto tg = time_runs([&]() { mat = src.trunc(M, f, L); }, 1, false);
    if (!mat.ok) {
      std::fprintf(stderr, "%s: trunc material\n", mat.code.c_str());
      return;
    }
    const ll::trunc::TruncMaterial& m0 = mat.value.first;
    const ll::trunc::TruncMaterial& m1 = mat.value.second;
    hl::Value params = hl::dict({{"f", static_cast<int64_t>(f)}, {"L", static_cast<int64_t>(L)}});
    std::vector<uint64_t> x(M), x0(M), x1(M), y(M);
    for (size_t i = 0; i < M; ++i) {
      int64_t xs = static_cast<int64_t>(rng.next_u64() & low_mask(L - 1)) - (int64_t(1) << (L - 2));  // |x| < 2^(L-2)
      x[i] = static_cast<uint64_t>(xs);
      x0[i] = rng.next_u64();
      x1[i] = x[i] - x0[i];
    }
    if (do_trunc) {
      Measurement off;
      off.component = "trunc_faithful_material";
      off.layer = "lowlevel";
      off.params = params;
      off.params.set("phase", "offline");
      off.batch = M;
      off.seconds = tg;
      off.material_bytes = static_cast<int64_t>(m0.nbytes());
      off.uses_dealer_material = true;
      off.note = "dealer generation (shared-seed run) or request and delivery of both halves (isolated dealer)";
      sink.record(off);

      std::vector<uint64_t> z0, z1;
      auto to = time_runs(
          [&]() {
            // trunc_open_share applies P0's public offset itself (step 1 of the gate)
            auto y0 = ll::trunc::trunc_open_share(x0, m0, 0);
            auto y1 = ll::trunc::trunc_open_share(x1, m1, 1);
            if (!y0.ok || !y1.ok) return;
            for (size_t i = 0; i < M; ++i) y[i] = y0.value[i] + y1.value[i];
            auto a = ll::trunc::trunc_faithful_eval(m0, y, 0);
            auto b = ll::trunc::trunc_faithful_eval(m1, y, 1);
            if (a.ok) z0 = std::move(a.value);
            if (b.ok) z1 = std::move(b.value);
          },
          repeats, true);
      bool good = z0.size() == M && z1.size() == M;
      for (size_t i = 0; i < M && good; ++i) {
        const int64_t xs = static_cast<int64_t>(x[i]);
        const int64_t want = xs >= 0 ? (xs >> f) : -(((-xs) + (int64_t(1) << f) - 1) >> f);  // floor(x / 2^f)
        if (static_cast<int64_t>(z0[i] + z1[i]) != want) good = false;
      }
      Measurement on;
      on.component = "trunc_faithful";
      on.layer = "lowlevel";
      on.params = params;
      on.params.set("phase", "online_both_parties");
      on.batch = M;
      on.seconds = to;
      on.online_bytes = static_cast<int64_t>(16 * M);
      on.rounds = 1;
      on.material_bytes = static_cast<int64_t>(m0.nbytes());
      on.uses_dealer_material = true;
      on.correct = good ? 1 : 0;
      on.note = "offset, open share and faithful evaluation of both parties (local work of the one-round gate)";
      sink.record(on);
    }
    if (do_dcf) {
      // one party's evaluation of the 64-bit wrap key: shares of 1{y < r}
      std::vector<uint64_t> yy(M), r(M), out0, out1;
      for (size_t i = 0; i < M; ++i) {
        r[i] = m0.r_share[i] + m1.r_share[i];
        yy[i] = rng.next_u64();
      }
      auto td = time_runs(
          [&]() {
            auto e = ll::dcf::eval_dcf_batch(m0.key_wrap, yy);
            if (e.ok) out0 = std::move(e.value);
          },
          repeats, true);
      auto e1 = ll::dcf::eval_dcf_batch(m1.key_wrap, yy);
      bool good = e1.ok && out0.size() == M;
      if (good) out1 = std::move(e1.value);
      for (size_t i = 0; i < M && good; ++i) {
        if (out0[i] + out1[i] != (yy[i] < r[i] ? 1u : 0u)) good = false;
      }
      Measurement d;
      d.component = "dcf_eval";
      d.layer = "lowlevel";
      d.params = hl::dict({{"n_bits", static_cast<int64_t>(64)}, {"P", static_cast<int64_t>(1)}, {"modulus", static_cast<int64_t>(0)},
                           {"key", "truncation wrap key"}});
      d.batch = M;
      d.seconds = td;
      d.material_bytes = static_cast<int64_t>(m0.key_wrap.nbytes());
      d.uses_dealer_material = true;
      d.correct = good ? 1 : 0;
      d.note = "eval_dcf_batch of one party";
      sink.record(d);
    }
  }
}

void bench_r2f(Sink& sink, MaterialSource& src, const std::vector<size_t>& batches, int repeats, ll::Rng& rng) {
  const uint64_t t = ll::r2f::ADMITTED_T;
  for (size_t M : batches) {
    Expected<std::pair<ll::r2f::R2FMaterial, ll::r2f::R2FMaterial>> mat;
    auto tg = time_runs([&]() { mat = src.r2f(M, t); }, 1, false);
    if (!mat.ok) {
      std::fprintf(stderr, "%s: r2f material\n", mat.code.c_str());
      return;
    }
    const ll::r2f::R2FMaterial& m0 = mat.value.first;
    const ll::r2f::R2FMaterial& m1 = mat.value.second;
    hl::Value params = hl::dict({{"t", static_cast<int64_t>(t)}});
    Measurement off;
    off.component = "ring_to_field_material";
    off.layer = "lowlevel";
    off.params = params;
    off.params.set("phase", "offline");
    off.batch = M;
    off.seconds = tg;
    off.material_bytes = static_cast<int64_t>(m0.nbytes());
    off.uses_dealer_material = true;
    off.note = "dealer generation (shared-seed run) or request and delivery of both halves (isolated dealer)";
    sink.record(off);

    std::vector<uint64_t> u0(M), u1(M), y(M), z0, z1;
    for (size_t i = 0; i < M; ++i) {
      u0[i] = rng.next_u64();
      u1[i] = rng.next_u64();
    }
    auto to = time_runs(
        [&]() {
          auto y0 = ll::r2f::r2f_open_share(u0, m0, 0);
          auto y1 = ll::r2f::r2f_open_share(u1, m1, 1);
          if (!y0.ok || !y1.ok) return;
          for (size_t i = 0; i < M; ++i) y[i] = y0.value[i] + y1.value[i];
          auto a = ll::r2f::r2f_eval(m0, y, 0);
          auto b = ll::r2f::r2f_eval(m1, y, 1);
          if (a.ok) z0 = std::move(a.value);
          if (b.ok) z1 = std::move(b.value);
        },
        repeats, true);
    bool good = z0.size() == M && z1.size() == M;
    for (size_t i = 0; i < M && good; ++i) {
      const uint64_t want = ll::r2f::signed_mod(u0[i] + u1[i], t);
      uint64_t got = z0[i] + z1[i];
      if (got >= t) got -= t;
      if (got != want) good = false;
    }
    Measurement on;
    on.component = "ring_to_field";
    on.layer = "lowlevel";
    on.params = params;
    on.params.set("phase", "online_both_parties");
    on.batch = M;
    on.seconds = to;
    on.online_bytes = static_cast<int64_t>(16 * M);
    on.rounds = 1;
    on.material_bytes = static_cast<int64_t>(m0.nbytes());
    on.uses_dealer_material = true;
    on.correct = good ? 1 : 0;
    on.note = "open share and evaluation of both parties (local work of the one-round gate)";
    sink.record(on);
  }
}

// ---------------------------------------------------------------------------
// highlevel two-party rig: loopback channel, ledgers, dealer clients
// ---------------------------------------------------------------------------
struct Rig {
  std::unique_ptr<hl::Channel> ch[2];
  std::unique_ptr<hl::MaterialLedger> ledger[2];
  std::unique_ptr<hl::DealerClient> client[2];
  std::string journal[2];
  hl::NumericContract contract;
  ~Rig() {
    for (int p = 0; p < 2; ++p) {
      client[p].reset();
      ledger[p].reset();
      if (!journal[p].empty()) std::remove(journal[p].c_str());
    }
  }
};

Expected<std::unique_ptr<Rig>> make_rig(MaterialSource& src, const hl::NumericContract& contract, const std::string& tag) {
  using R = std::unique_ptr<Rig>;
  auto rig = std::make_unique<Rig>();
  rig->contract = contract;
  auto pair = hl::loopback_pair();
  if (!pair.ok) return make_fail<R>(pair.code);
  rig->ch[0] = std::move(pair.value.first);
  rig->ch[1] = std::move(pair.value.second);
  for (int p = 0; p < 2; ++p) {
    rig->journal[p] = temp_dir() + "/oc2_bench_" + tag + "_p" + std::to_string(p) + ".jsonl";
    std::remove(rig->journal[p].c_str());
    rig->ledger[p] = std::make_unique<hl::MaterialLedger>(rig->journal[p], src.binding());
    auto lo = rig->ledger[p]->open();
    if (!lo.ok) return make_fail<R>(lo.code);
    rig->client[p] = std::make_unique<hl::DealerClient>(p, *rig->ledger[p], src.key(p), contract.contract_hash());
    src.attach(*rig->client[p], p);
  }
  return make_ok(std::move(rig));
}

void bench_argmax(Sink& sink, MaterialSource& src, const hl::NumericContract& contract, const std::vector<size_t>& ns,
                  const std::vector<int>& ks, int repeats_arg, ll::Rng& rng, const std::string& tag, bool on_demand) {
  const double kLogitBound = 8.0;
  for (size_t n : ns) {
    for (int k : ks) {
      // the magnitude bound of the fused node from the cost model layer (public: contract, k and n only)
      const double bound = hl::kary_operand_bound_real(contract, k, static_cast<int64_t>(n));
      if (!(kLogitBound < bound)) {
        std::fprintf(stderr, "KARY_MAGNITUDE: logits bound %.1f is not below the fused bound %.3f for k=%d n=%zu; skipped\n",
                     kLogitBound, bound, k, n);
        continue;
      }
      const int repeats = repeats_arg > 0 ? repeats_arg : (n > 10000 ? 1 : 3);
      auto rig_r = make_rig(src, contract, tag + "_am" + std::to_string(n) + "_" + std::to_string(k));
      if (!rig_r.ok) {
        std::fprintf(stderr, "%s: argmax rig\n", rig_r.code.c_str());
        return;
      }
      Rig& rig = *rig_r.value;
      std::vector<double> secs, waits, prov_secs;
      int64_t bytes = 0, rounds = 0, material = 0, prov_bytes = 0;
      bool all_good = true;
      std::string fail_code;
      // What one tournament over a row of n consumes (public: n, k and the index path only).
      const hl::TournamentCounts tc = hl::tournament_counts(static_cast<int64_t>(n), k, true);
      for (int rep = 0; rep < repeats && fail_code.empty(); ++rep) {
        if (!on_demand) {
          // offline: the dealer runs ahead and both pools hold the material of this tournament
          const double p0 = monotonic_s();
          int64_t pb = 0;
          auto prov = [&](const std::string& type, const hl::Value& params, int64_t count) {
            if (count <= 0 || !fail_code.empty()) return;
            auto r = src.provision(*rig.client[0], *rig.client[1], type, params, static_cast<size_t>(count));
            if (!r.ok) {
              fail_code = r.code;
            } else {
              pb += r.value;
            }
          };
          prov("drelu", hl::dict({{"n_bits", static_cast<int64_t>(contract.dcf_domain_bits)}, {"with_product", false}}), tc.comparisons);
          prov("drelu", hl::dict({{"n_bits", static_cast<int64_t>(contract.dcf_domain_bits)}, {"with_product", true}}), tc.payload_drelu);
          if (tc.selects > 0)
            prov("kary_select", hl::dict({{"shape", hl::Value(hl::List{hl::Value(static_cast<int64_t>(tc.selects))})}}), 1);
          if (!fail_code.empty()) break;
          prov_secs.push_back(monotonic_s() - p0);
          prov_bytes = pb;
        }
        // public-shape row of n logits in [-8, 8), shared additively
        std::vector<uint64_t> x(n), x0(n), x1(n);
        int64_t best = 0;
        for (size_t i = 0; i < n; ++i) {
          const double v = (static_cast<double>(rng.next_u64() >> 11) / 9007199254740992.0) * 2.0 * kLogitBound - kLogitBound;
          x[i] = static_cast<uint64_t>(static_cast<int64_t>(std::llrint(v * std::ldexp(1.0, contract.frac_bits))));
          x0[i] = rng.next_u64();
          x1[i] = x[i] - x0[i];
          if (static_cast<int64_t>(x[i]) > static_cast<int64_t>(x[static_cast<size_t>(best)])) best = static_cast<int64_t>(i);
        }
        Expected<hl::Mat> res[2];
        double wall[2] = {0.0, 0.0};
        const double wait_before = src.wait_seconds();
        const uint64_t b0 = rig.ch[1]->stats.bytes_sent + rig.ch[1]->stats.bytes_recv;
        const uint64_t r0 = rig.ch[1]->stats.rounds;
        const uint64_t m0 = rig.client[1]->bytes_consumed();
        auto run_party = [&](int p) {
          hl::ReleasePolicy rel(1, {"token"}, 1);
          hl::Party party(p, *rig.ch[p], *rig.client[p], contract, rel, ll::Rng::from_os_or_abort(), "bench");
          hl::Mat share = hl::Mat::from_vector(p == 0 ? x0 : x1, 1, n);
          const double t0 = monotonic_s();
          res[p] = party.argmax_row(share, k, true, "token");
          wall[p] = monotonic_s() - t0;
        };
        std::thread th0([&]() { run_party(0); });
        run_party(1);
        th0.join();
        if (!res[0].ok || !res[1].ok) {
          fail_code = res[0].ok ? res[1].code : res[0].code;
          break;
        }
        const bool good = res[1].value.a.size() == 1 && static_cast<int64_t>(res[1].value.a[0]) == best && res[0].value.a.empty();
        all_good = all_good && good;
        secs.push_back(wall[1]);
        waits.push_back(src.wait_seconds() - wait_before);
        bytes = static_cast<int64_t>(rig.ch[1]->stats.bytes_sent + rig.ch[1]->stats.bytes_recv - b0);
        rounds = static_cast<int64_t>(rig.ch[1]->stats.rounds - r0);
        material = static_cast<int64_t>(rig.client[1]->bytes_consumed() - m0);
      }
      if (!fail_code.empty()) {
        std::fprintf(stderr, "%s: argmax k=%d n=%zu\n", fail_code.c_str(), k, n);
        continue;
      }
      std::vector<double> ws = waits;
      std::sort(ws.begin(), ws.end());
      if (!on_demand && !prov_secs.empty()) {
        Measurement off;
        off.component = "argmax_k" + std::to_string(k) + "_material";
        off.layer = "highlevel";
        off.params = hl::dict({{"k", static_cast<int64_t>(k)}, {"n", static_cast<int64_t>(n)}, {"phase", "offline"},
                               {"drelu_p1", tc.comparisons}, {"drelu_p2", tc.payload_drelu}, {"kary_select", tc.selects}});
        off.batch = n;
        off.seconds = prov_secs;
        off.material_bytes = prov_bytes;
        off.uses_dealer_material = true;
        off.note = "material of one tournament (tournament_counts): generation and delivery of both halves, ingestion by both "
                   "dealer clients";
        sink.record(off);
      }
      Measurement m;
      m.component = "argmax_k" + std::to_string(k);
      m.layer = "highlevel";
      m.params = hl::dict({{"k", static_cast<int64_t>(k)}, {"n", static_cast<int64_t>(n)}, {"rows", static_cast<int64_t>(1)},
                           {"realization", "fused k-ary tournament (Party::argmax_row)"}});
      m.batch = n;
      m.seconds = secs;
      m.online_bytes = bytes;
      m.rounds = rounds;
      m.material_bytes = material;
      m.material_wait_s = ws.empty() ? 0.0 : ws[ws.size() / 2];
      m.uses_dealer_material = true;
      m.correct = all_good ? 1 : 0;
      m.note = on_demand
                   ? "two party threads over a loopback highlevel::Channel; material on demand: every pool miss inside the gate "
                     "costs a dealer request and journal writes (reserve and spend are fsynced before first use); "
                     "T_online_s subtracts the dealer wait only"
                   : "two party threads over a loopback highlevel::Channel; the material was provisioned before the timed call; "
                     "seconds is P1's wall time of the gate, including the journal writes of the bundles it touches (reserve "
                     "and spend are fsynced before first use); index released to P1 only";
      const int levels = [&]() {
        int l = 0;
        size_t w = n;
        while (w > 1) {
          w = (w + static_cast<size_t>(k) - 1) / static_cast<size_t>(k);
          ++l;
        }
        return l;
      }();
      m.more = hl::dict({{"levels", static_cast<int64_t>(levels)},
                         {"declared_operand_bound_real", kLogitBound},
                         {"kary_operand_bound_real", bound},
                         {"tie_rule", "lowest_index"},
                         {"material_mode", on_demand ? "on_demand" : "provisioned_ahead"},
                         {"expected_drelu_p1", tc.comparisons},
                         {"expected_drelu_p2", tc.payload_drelu},
                         {"expected_selects", tc.selects},
                         {"expected_rounds", tc.rounds},
                         {"ledger_journal", "durable (fsync per state change), included in seconds"},
                         {"transport", "loopback tcp, unshaped"}});
      sink.record(m);
    }
  }
}

// ---------------------------------------------------------------------------
// toy bridge: local reference path and the secure two-party bridge
// ---------------------------------------------------------------------------
void bench_toy_local(Sink& sink, const std::vector<size_t>& batches, int repeats, uint64_t seed, int N) {
  auto params = ll::ToyParams::make(N, 16, 8);
  if (!params.ok) return;
  ll::ToyBFV bfv(params.value);
  ll::RandomSource rs = ll::deterministic_source(seed + 19);
  auto kp = bfv.keygen(rs);
  const int d_in = 8, d_out = 8;
  ll::Matrix W(d_out, std::vector<uint64_t>(d_in));
  for (auto& row : W) {
    for (auto& v : row) v = rs() % 64;
  }
  for (size_t batch : batches) {
    size_t vectors = batch / 64 ? batch / 64 : 1;  // one vector per 64 "elements" of batch
    std::vector<std::vector<uint64_t>> x1(vectors, std::vector<uint64_t>(d_in)), x0(vectors, std::vector<uint64_t>(d_in));
    for (size_t v = 0; v < vectors; ++v) {
      for (int i = 0; i < d_in; ++i) {
        x0[v][static_cast<size_t>(i)] = rs() % 64;
        x1[v][static_cast<size_t>(i)] = rs() % 64;
      }
    }
    int64_t ct_bytes = 0;
    bool good = true;
    auto t = time_runs(
        [&]() {
          for (size_t v = 0; v < vectors; ++v) {
            auto ct = ll::a2h_toy_local(bfv, kp.pk, x1[v], x0[v], rs);
            if (!ct.ok) {
              good = false;
              return;
            }
            auto mv = bfv.matvec_pt(ct.value, W);
            if (!mv.ok) {
              good = false;
              return;
            }
            auto pt = bfv.decrypt(kp.sk, mv.value.ct_y);
            std::vector<uint64_t> x(static_cast<size_t>(d_in));
            for (int i = 0; i < d_in; ++i) x[static_cast<size_t>(i)] = x0[v][static_cast<size_t>(i)] + x1[v][static_cast<size_t>(i)];
            auto want = bfv.matvec_reference(W, x);
            if (!pt.ok || !want.ok) {
              good = false;
              return;
            }
            for (int kk = 0; kk < d_out; ++kk) {
              if (pt.value[static_cast<size_t>(mv.value.positions[static_cast<size_t>(kk)])] != want.value[static_cast<size_t>(kk)]) good = false;
            }
            ct_bytes = static_cast<int64_t>((ct.value.c0.size() + ct.value.c1.size()) * 8);
          }
        },
        repeats, true);
    Measurement m;
    m.component = "toy_bridge_local";
    m.layer = "lowlevel";
    m.params = hl::dict({{"N", static_cast<int64_t>(N)}, {"log_t", static_cast<int64_t>(16)}, {"d_in", static_cast<int64_t>(d_in)},
                         {"d_out", static_cast<int64_t>(d_out)}});
    m.batch = vectors;
    m.seconds = t;
    m.online_bytes = ct_bytes;
    m.rounds = 1;
    m.correct = good ? 1 : 0;
    m.note = "a2h_toy_local + matvec_pt + decrypt: the local reference path of the toy BFV bridge (TEST_ONLY scheme, "
             "public test key, no dealer material)";
    sink.record(m);
  }
}

#ifdef BENCH_HAVE_SECURE_BRIDGE
void bench_secure_bridge(Sink& sink, MaterialSource& src, const hl::NumericContract& contract, int repeats_arg, uint64_t seed,
                         int N, const std::string& tag) {
  namespace br = hl::bridge;
  const int log_t = 16, d_in = 8, d_out = 8;
  auto params = ll::ToyParams::make(N, log_t, 8);
  if (!params.ok) {
    std::fprintf(stderr, "%s: toy parameters\n", params.code.c_str());
    return;
  }
  const int repeats = repeats_arg > 0 ? repeats_arg : 3;
  auto rig_r = make_rig(src, contract, tag + "_br" + std::to_string(N));
  if (!rig_r.ok) {
    std::fprintf(stderr, "%s: bridge rig\n", rig_r.code.c_str());
    return;
  }
  Rig& rig = *rig_r.value;
  const br::H2aMaterial need = br::h2a_material(params.value, br::Convolution::KARATSUBA_LEAVES);
  // The L = 64 truncation material must be in both pools before the calls (bridge.hpp).
  auto prov = src.provision_trunc(*rig.client[0], *rig.client[1], need.trunc_elements * static_cast<size_t>(repeats), need.trunc_f,
                                  need.trunc_L);
  if (!prov.ok) {
    std::fprintf(stderr, "%s: bridge truncation material\n", prov.code.c_str());
    return;
  }
  ll::ToyBFV bfv_keys(params.value);
  ll::RandomSource krs = src.dealer_mode() == kIsolated ? ll::os_random_source() : ll::deterministic_source(seed + 23);
  auto kp = bfv_keys.keygen(krs);  // P1's key pair
  ll::RandomSource wrs = ll::deterministic_source(seed + 29);
  ll::Matrix W(d_out, std::vector<uint64_t>(d_in));
  for (auto& row : W) {
    for (auto& v : row) v = wrs() % 64;
  }
  std::vector<int64_t> positions(static_cast<size_t>(d_out));
  for (int kk = 0; kk < d_out; ++kk) positions[static_cast<size_t>(kk)] = static_cast<int64_t>(kk + 1) * d_in - 1;  // public
  const uint64_t t_mask = (uint64_t(1) << log_t) - 1;
  std::vector<double> total_s, a2h_s, eval_s, h2a_s, waits;
  int64_t bytes = 0, rounds = 0, material = 0;
  bool all_good = true;
  std::string fail_code;
  for (int rep = 0; rep < repeats && fail_code.empty(); ++rep) {
    std::vector<uint64_t> x(static_cast<size_t>(d_in)), sx0(static_cast<size_t>(d_in)), sx1(static_cast<size_t>(d_in));
    for (int i = 0; i < d_in; ++i) {
      x[static_cast<size_t>(i)] = wrs() % 64;
      sx0[static_cast<size_t>(i)] = wrs();
      sx1[static_cast<size_t>(i)] = x[static_cast<size_t>(i)] - sx0[static_cast<size_t>(i)];
    }
    Expected<std::vector<uint64_t>> ys[2];
    double t_a2h = 0.0, t_eval = 0.0, t_h2a = 0.0, t_total = 0.0;
    const double wait_before = src.wait_seconds();
    const uint64_t b0 = rig.ch[1]->stats.bytes_sent + rig.ch[1]->stats.bytes_recv;
    const uint64_t r0 = rig.ch[1]->stats.rounds;
    const uint64_t m0 = rig.client[1]->bytes_consumed();
    auto run_party = [&](int p) {
      hl::ReleasePolicy rel(1, {"token"}, 1);
      hl::Party party(p, *rig.ch[p], *rig.client[p], contract, rel, ll::Rng::from_os_or_abort(), "bench");
      ll::ToyBFV bfv(params.value);
      ll::RandomSource ers = src.dealer_mode() == kIsolated ? ll::os_random_source() : ll::deterministic_source(seed + 31 + static_cast<uint64_t>(p));
      const double t0 = monotonic_s();
      auto ct = br::a2h_toy(party, p == 0 ? sx0 : sx1, bfv, p == 1 ? &kp.pk : nullptr, p == 1 ? &ers : nullptr);
      const double t1 = monotonic_s();
      if (!ct.ok) {
        ys[p] = Expected<std::vector<uint64_t>>::failure(ct.code);
        return;
      }
      ll::Ciphertext ct_y;
      if (p == 0) {
        auto mv = br::eval_toy(bfv, ct.value, W);
        if (!mv.ok) {
          ys[p] = Expected<std::vector<uint64_t>>::failure(mv.code);
          return;
        }
        ct_y = std::move(mv.value.ct_y);
      }
      const double t2 = monotonic_s();
      ys[p] = br::h2a_ref(party, bfv, p == 0 ? &ct_y : nullptr, p == 1 ? &kp.sk : nullptr, positions, br::Convolution::KARATSUBA_LEAVES);
      const double t3 = monotonic_s();
      if (p == 1) {
        t_a2h = t1 - t0;
        t_h2a = t3 - t2;
        t_total = t3 - t0;
      } else {
        t_eval = t2 - t1;
      }
    };
    std::thread th0([&]() { run_party(0); });
    run_party(1);
    th0.join();
    if (!ys[0].ok || !ys[1].ok) {
      fail_code = ys[0].ok ? ys[1].code : ys[0].code;
      break;
    }
    auto want = bfv_keys.matvec_reference(W, x);
    bool good = want.ok && ys[0].value.size() == static_cast<size_t>(d_out) && ys[1].value.size() == static_cast<size_t>(d_out);
    for (int kk = 0; kk < d_out && good; ++kk) {
      const uint64_t got = (ys[0].value[static_cast<size_t>(kk)] + ys[1].value[static_cast<size_t>(kk)]) & t_mask;  // y mod t
      if (got != (want.value[static_cast<size_t>(kk)] & t_mask)) good = false;
    }
    all_good = all_good && good;
    total_s.push_back(t_total);
    a2h_s.push_back(t_a2h);
    eval_s.push_back(t_eval);
    h2a_s.push_back(t_h2a);
    waits.push_back(src.wait_seconds() - wait_before);
    bytes = static_cast<int64_t>(rig.ch[1]->stats.bytes_sent + rig.ch[1]->stats.bytes_recv - b0);
    rounds = static_cast<int64_t>(rig.ch[1]->stats.rounds - r0);
    material = static_cast<int64_t>(rig.client[1]->bytes_consumed() - m0);
  }
  if (!fail_code.empty()) {
    std::fprintf(stderr, "%s: secure toy bridge N=%d\n", fail_code.c_str(), N);
    return;
  }
  auto list_of = [](const std::vector<double>& v) {
    hl::List l;
    for (double d : v) l.push_back(hl::Value(d));
    return hl::Value(std::move(l));
  };
  std::vector<double> ws = waits;
  std::sort(ws.begin(), ws.end());
  Measurement m;
  m.component = "toy_bridge_h2a_ref";
  m.layer = "highlevel";
  m.params = hl::dict({{"N", static_cast<int64_t>(N)}, {"log_t", static_cast<int64_t>(log_t)}, {"d_in", static_cast<int64_t>(d_in)},
                       {"d_out", static_cast<int64_t>(d_out)}, {"convolution", "KARATSUBA_LEAVES"}});
  m.batch = static_cast<size_t>(N);
  m.seconds = total_s;
  m.online_bytes = bytes;
  m.rounds = rounds;
  m.material_bytes = material;
  m.material_wait_s = ws.empty() ? 0.0 : ws[ws.size() / 2];
  m.uses_dealer_material = true;
  m.correct = all_good ? 1 : 0;
  m.note = "a2h_toy -> eval_toy -> h2a_ref with two party threads over a loopback highlevel::Channel; TEST_ONLY toy "
           "scheme (q = 2^64, N <= 512: no computational security); real-parameter H2A_ref is not executed anywhere";
  m.more = hl::dict({{"a2h_toy_s", list_of(a2h_s)},
                     {"eval_toy_s_p0", list_of(eval_s)},
                     {"h2a_ref_s", list_of(h2a_s)},
                     {"beaver_triples", static_cast<int64_t>(need.beaver_triples)},
                     {"trunc_elements", static_cast<int64_t>(need.trunc_elements)},
                     {"trunc_f", static_cast<int64_t>(need.trunc_f)},
                     {"trunc_L", static_cast<int64_t>(need.trunc_L)},
                     {"declared_rounds_h2a", static_cast<int64_t>(need.rounds)},
                     {"declared_open_bytes_beaver", static_cast<int64_t>(need.open_bytes_beaver)},
                     {"declared_open_bytes_trunc", static_cast<int64_t>(need.open_bytes_trunc)},
                     {"component_status", "TEST_ONLY"}});
  sink.record(m);
}
#endif

// ---------------------------------------------------------------------------
// platform primitives
// ---------------------------------------------------------------------------
void bench_platform(Sink& sink, MaterialSource& src, const std::vector<size_t>& batches, int repeats, ll::Rng& rng) {
  for (size_t n : batches) {
    std::vector<uint64_t> v(n);
    for (auto& x : v) x = rng.next_u64();
    wire::Value d = wire::Value::make_dict();
    d.set("x", wire::Value::from_u64_vector(v));
    std::vector<uint8_t> f;
    bool good = true;
    auto tc = time_runs(
        [&]() {
          f = wire::frame(d);
          auto back = wire::decode_frame(f);
          good = good && back.ok && back.value.get("x") && back.value.get("x")->array.to_u64() == v;
        },
        repeats, true);
    Measurement c;
    c.component = "wire_codec_roundtrip";
    c.layer = "platform";
    c.batch = n;
    c.seconds = tc;
    c.online_bytes = static_cast<int64_t>(f.size());
    c.correct = good ? 1 : 0;
    c.note = "frame + decode_frame of one u64 array (the highlevel codec through the platform value type)";
    sink.record(c);

    std::vector<uint8_t> key(32, 0x42);
    auto th = time_runs(
        [&]() {
          auto t = detail::hmac_sha256(key, f);
          (void)t;
        },
        repeats, true);
    Measurement h;
    h.component = "hmac_sha256";
    h.layer = "platform";
    h.batch = n;
    h.seconds = th;
    h.online_bytes = static_cast<int64_t>(f.size());
    h.note = "HMAC-SHA256 over the frame bytes";
    sink.record(h);
  }
  // WebSocket loopback echo: server thread echoes binary messages
  auto l = TcpListener::bind("127.0.0.1", 0, 4);
  if (!l.ok) return;
  uint16_t port = l.value.port();
  std::thread server([&l]() {
    auto c = l.value.accept(10.0);
    if (!c.ok) return;
    auto req = read_http_request(c.value);
    if (!req.ok) return;
    auto ws = websocket_accept(std::move(c.value), req.value);
    if (!ws.ok) return;
    while (true) {
      auto m = ws.value.recv_message();
      if (!m.ok || m.value.is_close()) break;
      ws.value.send_binary(m.value.payload);
    }
  });
  auto ws = websocket_connect("127.0.0.1", port, "/session", "", {}, 10.0);
  if (ws.ok) {
    for (size_t n : batches) {
      std::vector<uint8_t> msg(n * 8, 0x5a);
      bool good = true;
      auto t = time_runs(
          [&]() {
            ws.value.send_binary(msg);
            auto r = ws.value.recv_message();
            good = good && r.ok && r.value.payload == msg;
          },
          repeats, true);
      Measurement m;
      m.component = "ws_loopback_roundtrip";
      m.layer = "platform";
      m.params = hl::dict({{"masked_client_frames", true}});
      m.batch = n;
      m.seconds = t;
      m.online_bytes = static_cast<int64_t>(2 * msg.size());
      m.rounds = 1;
      m.correct = good ? 1 : 0;
      m.note = "one binary message echoed over a loopback WebSocket";
      sink.record(m);
    }
    ws.value.send_close(1000, "");
    ws.value.close();
  }
  server.join();

  // Beaver halves for both recipients and one combine step
  for (size_t n : batches) {
    int64_t bytes = 0;
    bool good = true;
    auto t = time_runs(
        [&]() {
          auto b = src.beaver(n);
          if (!b.ok) {
            good = false;
            return;
          }
          bytes = static_cast<int64_t>(3 * 8 * n);
          std::vector<uint64_t> dd(n, 3), ee(n, 5);
          auto z0 = demo::beaver_combine(b.value.first, dd, ee, 0);
          auto z1 = demo::beaver_combine(b.value.second, dd, ee, 1);
          for (size_t i = 0; i < n && good; ++i) {
            // (a + 3)(b + 5) with d = 3, e = 5: z = c + d b + e a + d e
            const uint64_t a = b.value.first.a[i] + b.value.second.a[i], bb = b.value.first.b[i] + b.value.second.b[i];
            if (z0[i] + z1[i] != (a + 3) * (bb + 5)) good = false;
          }
        },
        repeats, false);
    Measurement m;
    m.component = "dealer_beaver_pair_plus_combine";
    m.layer = "platform";
    m.params = hl::dict({{"material", "beaver"}});
    m.batch = n;
    m.seconds = t;
    m.material_bytes = bytes;
    m.uses_dealer_material = true;
    m.correct = good ? 1 : 0;
    m.note = "Beaver halves for both recipients (generation or isolated delivery) and one combine step";
    sink.record(m);
  }
}

std::vector<size_t> parse_sizes(const std::string& s, const std::vector<size_t>& def) {
  std::vector<size_t> out;
  for (const auto& part : detail::split(s, ',')) {
    std::string p = detail::trim(part);
    if (p.empty()) continue;
    long long v = std::strtoll(p.c_str(), nullptr, 10);
    if (v > 0) out.push_back(static_cast<size_t>(v));
  }
  if (out.empty()) out = def;
  return out;
}

}  // namespace

int main(int argc, char** argv) {
#ifndef _WIN32
  std::signal(SIGPIPE, SIG_IGN);  // a peer that went away is an error return of the write, not the end of the process
#endif
  cli::Args args(argc, argv);
  if (args.flag("help") || args.flag("h")) return usage(argv[0], 0);
  if (args.flag("print-schema")) {
    auto list_of = [](const std::vector<std::string>& v) {
      hl::List l;
      for (const auto& s : v) l.push_back(hl::Value(s));
      return hl::Value(std::move(l));
    };
    hl::Value o = hl::dict({{"schema", evidence::kSchemaId},
                            {"profile_id", evidence::kProfileId},
                            {"record_fields", list_of(evidence::record_fields())},
                            {"optional_fields", list_of(evidence::optional_fields())},
                            {"statuses", list_of(evidence::statuses())},
                            {"workload_scopes", list_of(evidence::workload_scopes())},
                            {"dealer_modes", list_of(evidence::dealer_modes())},
                            {"clock_bases", list_of(evidence::clock_bases())},
                            {"record_kinds", list_of(evidence::record_kinds())},
                            {"code_commit", kCodeCommit},
                            {"default_contract_hash", hl::default_contract().contract_hash()}});
    auto text = hl::canonical_json(o);
    std::printf("%s\n", text.ok ? text.value.c_str() : "{}");
    return text.ok ? 0 : 1;
  }
  {
    auto unknown = args.unknown_keys({"out", "only", "batches", "repeats", "seed", "dealer", "keys", "setup-id", "argmax-n",
                                      "argmax-k", "argmax-repeats", "argmax-on-demand", "bridge-n", "print-schema", "help", "h"});
    if (!unknown.empty()) {
      std::fprintf(stderr, "BENCH_ARGS: unknown option --%s\n", unknown[0].c_str());
      return usage(argv[0], 2);
    }
  }
  auto init = tcp_global_init();
  if (!init.ok) {
    std::fprintf(stderr, "%s\n", init.code.c_str());
    return 1;
  }
  const hl::NumericContract contract = hl::default_contract();
  const std::string out_path = args.get("out", "");
  const std::vector<size_t> batches = parse_sizes(args.get("batches", ""), {64, 256, 1024, 4096});
  int repeats = static_cast<int>(args.get_int("repeats", 7));
  if (repeats < 1) repeats = 1;
  const uint64_t seed = static_cast<uint64_t>(args.get_int("seed", 20261010));
  const std::vector<size_t> argmax_n = parse_sizes(args.get("argmax-n", ""), {256, 4096, 50257});
  std::vector<int> argmax_k;
  for (size_t k : parse_sizes(args.get("argmax-k", ""), {2, 4, 8})) {
    if (k != 2 && k != 4 && k != 8) {
      std::fprintf(stderr, "BENCH_ARGS: --argmax-k admits 2, 4 and 8\n");
      return 2;
    }
    argmax_k.push_back(static_cast<int>(k));
  }
  const int argmax_repeats = static_cast<int>(args.get_int("argmax-repeats", 0));
  const int bridge_n = static_cast<int>(args.get_int("bridge-n", 128));
  std::vector<std::string> only;
  for (const auto& o : detail::split(args.get("only", ""), ',')) {
    if (!detail::trim(o).empty()) only.push_back(detail::trim(o));
  }
  for (const auto& o : only) {
    if (o != "dcf" && o != "drelu" && o != "trunc" && o != "r2f" && o != "argmax" && o != "toy_bridge" && o != "platform") {
      std::fprintf(stderr, "BENCH_ARGS: unknown bench '%s'\n", o.c_str());
      return usage(argv[0], 2);
    }
  }

  // The material source of the run decides dealer_mode and the status of every record.
  std::shared_ptr<MaterialSource> src;
  if (args.has("dealer")) {
    auto dhp = parse_host_port(args.get("dealer"));
    if (!dhp.ok || args.get("keys").empty()) {
      std::fprintf(stderr, "BENCH_ARGS: --dealer host:port needs --keys keys.json\n");
      return 2;
    }
    auto iso = IsolatedSource::connect(dhp.value.first, dhp.value.second, args.get("keys"), args.get("setup-id", "setup-0"), contract);
    if (!iso.ok) {
      std::fprintf(stderr, "%s: cannot use the dealer at %s\n", iso.code.c_str(), args.get("dealer").c_str());
      return 1;
    }
    src = iso.value;
  } else {
    src = std::make_shared<SeededSource>(seed, contract);
    std::fprintf(stderr, "[opencrab2-bench] no --dealer: shared-seed material, every record is DIAGNOSTIC\n");
  }

  Sink sink;
  sink.out_path = out_path;
  sink.run_tag = detail::hex_encode(detail::os_random_bytes(6));
  sink.dealer_mode = src->dealer_mode();
  sink.contract_hash = contract.contract_hash();
  sink.host = evidence::host_info();
  // Test inputs of the timed calls: OS randomness with the isolated dealer, the public seed otherwise.
  ll::Rng input_rng = sink.dealer_mode == kIsolated ? ll::Rng::from_os_or_abort() : ll::Rng::from_seed(seed ^ 0x62656e6368ULL);

  const bool do_dcf = wants(only, "dcf"), do_drelu = wants(only, "drelu"), do_trunc = wants(only, "trunc");
  if (do_dcf || do_drelu) bench_drelu_and_dcf32(sink, *src, batches, repeats, do_drelu, do_dcf, input_rng);
  if (do_dcf || do_trunc) bench_trunc_and_dcf64(sink, *src, batches, repeats, do_trunc, do_dcf, input_rng, contract);
  if (wants(only, "r2f")) bench_r2f(sink, *src, batches, repeats, input_rng);
  if (wants(only, "argmax"))
    bench_argmax(sink, *src, contract, argmax_n, argmax_k, argmax_repeats, input_rng, sink.run_tag, args.flag("argmax-on-demand"));
  if (wants(only, "toy_bridge")) {
    bench_toy_local(sink, batches, repeats, seed, bridge_n);
#ifdef BENCH_HAVE_SECURE_BRIDGE
    bench_secure_bridge(sink, *src, contract, argmax_repeats, seed, bridge_n, sink.run_tag);
#else
    std::fprintf(stderr, "toy_bridge: highlevel/bridge.hpp absent at compile time, h2a_ref not timed\n");
#endif
  }
  if (wants(only, "platform")) bench_platform(sink, *src, batches, repeats, input_rng);
  std::fprintf(stderr, "[opencrab2-bench] %d records (%s, %s)%s%s, %d rejected by the schema, %d incorrect\n", sink.written,
               sink.dealer_mode.c_str(), sink.dealer_mode == kIsolated ? "MEASURED_NATIVE" : "DIAGNOSTIC",
               out_path.empty() ? "" : " appended to ", out_path.c_str(), sink.invalid, sink.incorrect);
  return (sink.written > 0 && sink.invalid == 0 && sink.incorrect == 0) ? 0 : 1;
}
