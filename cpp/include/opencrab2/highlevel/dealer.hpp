// OpenCrab v2 C++17 mirror: dealer (material generation as recipient-bound
// bundles, paper, Appendix B, DealerDelivery) and DealerClient (party-side verification
// and pooling). Mirrors python/opencrab2/highlevel/dealer.py.
//
// Lowlevel dependencies: opencrab2::lowlevel::dcf::DcfKeyBatch,
// drelu::DreluMaterial, trunc::TruncMaterial, r2f::R2FMaterial, Rng (ChaCha20
// CSPRNG seeded from the OS or from a fixed test seed), sharing::share.
// The aliases in namespace `ll` below keep the Python names (DCFKeyBatch,
// DReLUMaterial, ...) available in this layer.
//
// Bundle header (canonical JSON): {profile_id, setup_id,
// model_version, numeric_contract_hash, recipient, gate_id, material_type,
// material_id, count, bytes, params, parts, dealer_mode, insecure}. The tag is
// HMAC-SHA256 under the recipient's key over
//     len(header) 8-byte BE || canonical header || payload
// (the hmac_auth.tag_with_payload framing of the Python module). A party
// rejects a bundle whose tag fails, whose recipient != itself, whose contract
// hash differs, whose binding tuple differs from the ledger binding, or whose
// material_id was seen before (MATERIAL_REUSE).
//
// Material types, header params and payload layouts (integers little-endian
// u64). The params use the PYTHON spellings of python/opencrab2/highlevel/
// dealer.py (shape_a / shape_b / shape):
//   beaver         {shape: [count], elements: count}        a[count] b[count] c[count]
//   kary_select    {shape: [count], elements: count}        same layout as beaver; Beaver triples
//                                                           reserved for the selects of the k-ary
//                                                           tournaments (binary nodes)
//   matrix_triple  {shape_a: [n, k], shape_b: [k, m]}       count x (A[n*k] B[k*m] C[n*m])
//   wmask_mask     {mat_id, shape: [d, m], model_version}   B_share[d*m]               (persistent)
//   wmask_call     {mat_id, n_rows, d, m}                   count x (A[n*d] C[n*m]), C = A B
//   drelu          {n_bits, with_product, P}                DreluMaterial::to_bytes
//   trunc_faithful {f, L}                                   TruncMaterial::to_bytes
//   ring_to_field  {t}                                      R2FMaterial::to_bytes
//   vector_share   {name, shape: [size]}                    share[size]                (persistent)
// The reader (DealerClient::ingest, Dealer::make) also accepts the earlier C++
// spellings {n, k, m}, {mat_id, d, m} and {name, size}, so that bundles of a
// dealer service that still echoes them keep verifying; every header this file
// PRODUCES and every refill request it SENDS uses the Python spellings.
//
// Both parties consume pools in the same order because every bundle pair is
// delivered to both inboxes in one global order (InProcessDealerLink); the
// isolated dealer process of the platform layer streams bundles the same way.
// Ledger accounting is at bundle granularity: a bundle is RESERVED and SPENT
// (journaled, fsynced) when first touched, before its material is returned.
#pragma once

#include <opencrab2/highlevel/codec.hpp>
#include <opencrab2/highlevel/ledger.hpp>
#include <opencrab2/lowlevel/csprng.hpp>
#include <opencrab2/lowlevel/dcf.hpp>
#include <opencrab2/lowlevel/drelu.hpp>
#include <opencrab2/lowlevel/ring_to_field.hpp>
#include <opencrab2/lowlevel/sharing.hpp>
#include <opencrab2/lowlevel/trunc.hpp>

#include <array>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace opencrab2 {
namespace highlevel {
namespace ll {

using Rng = lowlevel::Rng;
using DCFKeyBatch = lowlevel::dcf::DcfKeyBatch;
using DReLUMaterial = lowlevel::drelu::DreluMaterial;
using TruncMaterial = lowlevel::trunc::TruncMaterial;
using R2FMaterial = lowlevel::r2f::R2FMaterial;

constexpr uint64_t RING_MASK = ~static_cast<uint64_t>(0);

inline size_t drelu_bytes_per_element(int n_bits, bool with_product = true) {
    return lowlevel::drelu::drelu_bytes_per_element(n_bits, with_product);
}
inline size_t trunc_bytes_per_element(int f) { return lowlevel::trunc::trunc_bytes_per_element(f); }
inline size_t r2f_bytes_per_element(uint64_t t) { return lowlevel::r2f::r2f_bytes_per_element(t); }

// Additive sharing helpers over Z_{2^64}.
inline std::pair<std::vector<uint64_t>, std::vector<uint64_t>> share_vec(const std::vector<uint64_t>& x, Rng& rng) {
    return lowlevel::sharing::share(x, rng);
}

// Uniform double in [0, 1) and a standard normal (tests and random weights only).
double rng_uniform(Rng& rng);
double rng_normal(Rng& rng);

// Little-endian payload helpers.
void put_u64_vec(Bytes& out, const std::vector<uint64_t>& v);
Expected<std::vector<uint64_t>> get_u64_vec(const uint8_t* p, size_t len, size_t count, size_t* consumed);

}  // namespace ll

// Exact ring matrix product (row-major), wraps mod 2^64.
void ring_matmul(const std::vector<uint64_t>& A, size_t n, size_t k, const std::vector<uint64_t>& B, size_t m,
                 std::vector<uint64_t>& C);

// ---------------------------------------------------------------------------
// Dealer side
// ---------------------------------------------------------------------------
struct Bundle {
    Value header;
    Bytes payload;
    std::string tag;
};

struct BundlePair {
    Bundle for0;
    Bundle for1;
};

struct BeaverShare {
    size_t count = 0;
    std::vector<uint64_t> a, b, c;
};

struct MatrixTripleShare {
    size_t n = 0, k = 0, m = 0;
    std::vector<uint64_t> A;  // n x k
    std::vector<uint64_t> B;  // k x m
    std::vector<uint64_t> C;  // n x m
};

struct WMaskCall {
    std::string mat_id;
    size_t n = 0, d = 0, m = 0;
    std::vector<uint64_t> a;  // n x d
    std::vector<uint64_t> c;  // n x m  (= A B)
};

struct WMaskMask {
    std::string mat_id;
    size_t d = 0, m = 0;
    std::vector<uint64_t> b_share;  // d x m
};

struct DealerConfig {
    std::string profile_id = "SH2P-TrustedOffline-v1";
    std::string setup_id = "setup-0";
    std::string model_version = "model-0";
    std::string numeric_contract_hash;
    std::string mode = "ISOLATED_RECIPIENT_BUNDLES";  // or INSECURE_TEST_ONLY_SHARED_SEED
    int dcf_domain_bits = 32;
    int frac_bits = 16;
    int logical_bits = 40;
};

class Dealer {
public:
    // The dealer remembers whether `rng` was keyed from the OS (ll::Rng::os_seeded()). A dealer
    // built on Rng::from_seed or an explicit key is replayable by whoever chose the seed: every
    // bundle it seals carries insecure = true whatever the mode (the rule of the Python dealer),
    // and a DealerClient that ingests such a bundle is never security eligible.
    Dealer(DealerConfig cfg, ll::Rng rng, Bytes hmac_key0, Bytes hmac_key1);

    Expected<BundlePair> make_beaver(size_t count, const std::string& gate_id = "*");
    // Beaver triples reserved for the selects of the k-ary tournaments (material type kary_select).
    Expected<BundlePair> make_kary_select(size_t count, const std::string& gate_id = "*");
    Expected<BundlePair> make_matrix_triples(size_t n, size_t k, size_t m, size_t count, const std::string& gate_id = "*");
    Expected<BundlePair> make_drelu(size_t M, int n_bits, bool with_product = true, const std::string& gate_id = "*");
    Expected<BundlePair> make_trunc(size_t M, int f, int L, const std::string& gate_id = "*");
    Expected<BundlePair> make_r2f(size_t M, uint64_t t, const std::string& gate_id = "*");
    // Per-matrix reusable mask B (d x m) shared to both parties. The dealer does
    // not receive W: E = W - B is produced by the online install protocol.
    // A second mask for the same mat_id is refused (DEALER_MASK_EXISTS) unless
    // it belongs to a newer session generation (resumption, paper, Section 4.3):
    // then a FRESH uniform B replaces the old one and later
    // wmask_call bundles are bound to the new mask.
    Expected<BundlePair> make_wmask_mask(const std::string& mat_id, size_t d, size_t m, const std::string& gate_id = "*",
                                         int64_t generation = 0);
    Expected<BundlePair> make_wmask_calls(const std::string& mat_id, size_t n_rows, size_t count,
                                          const std::string& gate_id = "*");
    // Shares of a vector given to the dealer IN THE CLEAR. Insecure unit-test harness only: the
    // dealer of profile SH2P-TrustedOffline-v1 never receives a model value (the tiny decoder
    // shares its vectors with Party::share_input). An eligible dealer (isolated mode, OS keyed)
    // refuses the request with DEALER_PLAINTEXT_REFUSED.
    Expected<BundlePair> make_vector_share(const std::string& name, const std::vector<uint64_t>& v_enc,
                                           const std::string& gate_id = "*");
    // Dispatch by material_type and params dict (used by the dealer link / service).
    Expected<BundlePair> make(const std::string& material_type, const Value& params, size_t count,
                              const std::string& gate_id = "*");

    Expected<Bundle> seal(int recipient, Value header, Bytes payload);

    uint64_t bytes_generated() const { return bytes_generated_; }
    uint64_t bundles_generated() const { return bundles_generated_; }
    const DealerConfig& config() const { return cfg_; }
    bool eligible_mode() const { return cfg_.mode == "ISOLATED_RECIPIENT_BUNDLES"; }
    bool os_seeded() const { return os_seeded_; }
    // The label of every header: true unless the mode is isolated AND the randomness is OS keyed.
    bool insecure() const { return !eligible_mode() || !os_seeded_; }

private:
    std::string next_material_id(const std::string& type);
    Value base_header(int recipient, const std::string& type, const std::string& material_id, size_t count,
                      const std::string& gate_id, Value params, List parts);
    Expected<BundlePair> seal_pair(const std::string& type, size_t count, const std::string& gate_id, Value params,
                                   List parts, Bytes payload0, Bytes payload1);
    Expected<BundlePair> make_beaver_like(const std::string& type, size_t count, const std::string& gate_id);

    DealerConfig cfg_;
    ll::Rng rng_;
    bool os_seeded_ = false;
    Bytes key0_, key1_;
    uint64_t counter_ = 0;
    uint64_t bytes_generated_ = 0;
    uint64_t bundles_generated_ = 0;
    std::map<std::string, std::vector<uint64_t>> masks_;  // mat_id -> full B (dealer-private)
    std::map<std::string, std::pair<size_t, size_t>> mask_shapes_;
    std::map<std::string, int64_t> mask_generation_;
};

// Gates of the fused k-ary tournament that Party::tournament issues for ONE row
// of n candidates (the C++ twin of python dealer.tournament_material and
// session.tournament_counts): per node of g members, g (g - 1) / 2 pairwise
// DReLUs without payload in round 1; in round 2 a binary node spends one Beaver
// select (kary_select) for the value plus one for the index once indices are
// secret (every level after the first), a node of g >= 3 members spends g - 1
// payload DReLUs (P = 2) for the value and the same again for a secret index.
// rounds = 2 * levels. Codes: TOURNAMENT_PARAMS.
struct TournamentMaterial {
    size_t drelu_p1 = 0;   // pairwise comparisons, DReLU without payload
    size_t drelu_p2 = 0;   // fused node selects, DReLU with product payload
    size_t selects = 0;    // Beaver selects of binary nodes (material type kary_select)
    size_t levels = 0;     // ceil(log_k n)
    size_t rounds = 0;     // 2 * levels
};
Expected<TournamentMaterial> tournament_material(size_t n, int k, bool with_index);

// Material volume per operator as a dict {material_type: [count, bytes], ...,
// "bytes": total} with the dealer vocabulary (beaver, kary_select, drelu,
// matrix_triple, wmask_call, wmask_mask, trunc_faithful, ring_to_field).
// op kinds: beaver_mul, mul, mul_fxp, square, select, drelu, relu,
// compare_public, matmul_shared (n, k, m), linear_generic / linear_wmask /
// embedding_* (n, d, m), install_wmask (d, m), gelu_clip12, silu_clip12, gelu,
// silu, exp_neg, recip, rsqrt (element counts), max_rows, argmax_row, argmax_k,
// softmax_rows, layernorm_rows, rmsnorm_rows (rows, n), trunc_faithful,
// ring_to_field, a2h_exact (element counts). k_ary selects the tournament
// arity. Counts are elements for Beaver-like and matrix material and gates for
// DReLU / truncation / ring-to-field material. DReLU bytes distinguish the
// payload width: comparisons (drelu, compare_public, thresholds, tournament
// round 1) use P = 1 keys, relu and the fused tournament nodes P = 2 keys. The
// tournaments are priced exactly as Party::tournament issues them
// (tournament_material above).
Expected<Value> material_estimate(const std::string& op_kind, const std::vector<size_t>& shape, int n_bits, int frac_bits,
                                  int k_ary = 2);

// ---------------------------------------------------------------------------
// Party side
// ---------------------------------------------------------------------------
class DealerClient {
public:
    using RefillFn = std::function<Expected<Unit>(const std::string& material_type, const Value& params, size_t count)>;

    DealerClient(int party, MaterialLedger& ledger, Bytes hmac_key, std::string contract_hash);

    // Verifies tag, recipient, contract hash, binding and freshness; parses the
    // payload into the pool and records AVAILABLE in the ledger.
    Expected<Unit> ingest(const Value& header, const Bytes& payload, const std::string& tag);
    Expected<Unit> ingest(const Bundle& b) { return ingest(b.header, b.payload, b.tag); }

    // Consumers (ledger: RESERVED then SPENT before the material is returned).
    Expected<BeaverShare> take_beaver(size_t count);
    // Beaver triples of the k-ary tournament selects (own pool, material type kary_select).
    Expected<BeaverShare> take_kary_select(size_t count);
    Expected<MatrixTripleShare> take_matrix_triple(size_t n, size_t k, size_t m);
    Expected<WMaskCall> take_wmask_call(const std::string& mat_id, size_t n_rows);
    Expected<ll::DReLUMaterial> take_drelu(size_t count, int n_bits, bool with_product = true);
    // Faithful truncation material for the shift f and the logical width L. The pool is keyed by
    // (f, L) and a refill request carries both (params {f, L}, the Python spelling), so a gate
    // never receives, and never spends, material of another width.
    Expected<ll::TruncMaterial> take_trunc(size_t count, int f, int L);
    Expected<ll::R2FMaterial> take_r2f(size_t count, uint64_t t);
    // Persistent objects (spent in the ledger on first use, then reusable).
    Expected<std::vector<uint64_t>> take_vector_share(const std::string& name);
    // With a public shape (d, m) a missing mask is requested through the refill link.
    Expected<WMaskMask> wmask_mask_share(const std::string& mat_id, size_t d = 0, size_t m = 0);
    bool has_wmask_mask(const std::string& mat_id) const { return wmask_masks_.count(mat_id) > 0; }
    bool has_vector_share(const std::string& name) const { return vectors_.count(name) > 0; }

    void set_gate(const std::string& gate_id, int64_t epoch) {
        gate_id_ = gate_id;
        epoch_ = epoch;
    }
    void set_refill(RefillFn fn) { refill_ = std::move(fn); }
    // Hook run before every pool lookup (drains an inbox in the in-process link).
    void set_drain(std::function<Expected<Unit>()> fn) { drain_ = std::move(fn); }

    // --- cancellation and resumption (paper, Section 4.3) --------------------------------
    // Burns, THROUGH THE LEDGER, every pooled bundle that is RESERVED (reserved
    // for a gate that will not run) and every untouched AVAILABLE one-time
    // bundle (safe waste), and drops the in-memory remainder of partially
    // consumed bundles (already SPENT in the ledger). One-time pools: beaver,
    // kary_select, matrix_triple, wmask_call, drelu, trunc_faithful,
    // ring_to_field. With drop_masks = true the per-matrix masks are dropped as
    // well (an untouched mask bundle is burned), so that a resumed generation
    // installs under fresh masks. Vector shares are kept. The inbox is drained
    // first, so bundles in flight are accounted too. After the call no id known
    // to this client is RESERVED.
    struct BurnReport {
        uint64_t reserved_ids = 0;     // RESERVED -> BURNED
        uint64_t available_ids = 0;    // AVAILABLE -> BURNED
        uint64_t bytes_burned = 0;     // ledger bytes of the burned bundles
        uint64_t bytes_dropped = 0;    // unconsumed remainder of SPENT bundles (in-memory only)
        Value to_value() const;
    };
    Expected<BurnReport> burn_outstanding(const std::string& reason, bool drop_masks = false);
    // Ids pooled by this client that are RESERVED in the ledger (normally empty:
    // a take reserves and spends in one step, durable before the material is returned).
    std::vector<std::string> reserved_ids() const;
    // Session generation (0 for a fresh session, +1 per resumption). A positive
    // generation is added to every refill request as params.generation, which
    // gives the resumed session its own request namespace at the dealer (no
    // stale half of an aborted generation can pair with a new request) and
    // makes the dealer issue fresh per-matrix masks.
    void set_generation(int64_t generation) { generation_ = generation; }
    int64_t generation() const { return generation_; }

    // Counters
    uint64_t bytes_ingested() const { return bytes_ingested_; }
    uint64_t bytes_consumed() const { return bytes_consumed_; }
    uint64_t bundles_ingested() const { return bundles_ingested_; }
    uint64_t refills() const { return refills_; }
    // Eligibility (paper, Section 3, evidence classes), default false: true only after at
    // least one bundle of a dealer that declares
    // isolated generation (dealer_mode ISOLATED_RECIPIENT_BUNDLES and insecure = false, i.e.
    // OS keyed), and never again once any bundle without that declaration was ingested.
    bool security_eligible() const { return eligible_seen_ && !insecure_seen_; }
    // True once a bundle without the isolated-generation declaration was ingested.
    bool insecure_material_seen() const { return insecure_seen_; }
    int party() const { return party_; }
    MaterialLedger& ledger() { return ledger_; }
    // Pooled element count of a type key: "beaver", "kary_select", "drelu:<n_bits>|<P>",
    // "trunc:<f>" (every L) or "trunc:<f>|<L>", "r2f:<t>".
    size_t pooled_count(const std::string& type_key) const;
    std::string type_key_drelu(int n_bits, bool with_product) const;

private:
    template <class Mat>
    struct PoolItem {
        std::string material_id;
        Mat mat;
        size_t offset = 0;
        bool spent = false;
    };
    Expected<Unit> ensure_spent(const std::string& material_id, bool& spent_flag);
    Expected<Unit> refill_or_stall(const std::string& type, Value params, size_t count);
    Expected<Unit> pre_lookup();
    Expected<BeaverShare> take_beaver_like(std::deque<PoolItem<BeaverShare>>& pool, const std::string& type, size_t count);

    int party_;
    MaterialLedger& ledger_;
    Bytes key_;
    std::string contract_hash_;
    std::string gate_id_ = "";
    int64_t epoch_ = -1;
    RefillFn refill_;
    std::function<Expected<Unit>()> drain_;
    bool eligible_seen_ = false;
    bool insecure_seen_ = false;
    int64_t generation_ = 0;

    std::deque<PoolItem<BeaverShare>> beaver_;
    std::deque<PoolItem<BeaverShare>> kary_;                                       // kary_select
    std::map<std::string, std::deque<PoolItem<MatrixTripleShare>>> matrix_;      // "n,k,m"
    std::map<std::string, std::deque<PoolItem<WMaskCall>>> wmask_calls_;         // "mat_id|n"
    std::map<std::string, std::deque<PoolItem<ll::DReLUMaterial>>> drelu_;       // "n_bits|P"
    std::map<std::pair<int, int>, std::deque<PoolItem<ll::TruncMaterial>>> trunc_;  // (f, L)
    std::map<uint64_t, std::deque<PoolItem<ll::R2FMaterial>>> r2f_;              // t
    std::map<std::string, PoolItem<std::vector<uint64_t>>> vectors_;             // name
    std::map<std::string, PoolItem<WMaskMask>> wmask_masks_;                      // mat_id
    uint64_t bytes_ingested_ = 0;
    uint64_t bytes_consumed_ = 0;
    uint64_t bundles_ingested_ = 0;
    uint64_t refills_ = 0;
};

// In-process stand-in for the isolated dealer service: both party threads share
// one Dealer; every generated pair is appended to BOTH inboxes in one global
// order so that both pools consume matching halves in the same sequence. A
// party requests only after draining its inbox, so the peer's earlier request
// for the same material already covers it (one generation per miss in the
// common case; a concurrent miss on both sides generates one extra pair).
class InProcessDealerLink {
public:
    explicit InProcessDealerLink(std::shared_ptr<Dealer> dealer);
    // Attach a client: installs its refill and drain hooks.
    void attach(DealerClient& client);
    Expected<Unit> request(int party, const std::string& type, const Value& params, size_t count);
    Expected<Unit> drain(int party, DealerClient& client);
    uint64_t bytes_transferred(int party) const { return transferred_[party]; }
    uint64_t bytes_generated() const;
    double generation_seconds() const { return gen_seconds_; }
    uint64_t generations() const { return generations_; }
    uint64_t requests(int party) const { return requests_[party]; }

private:
    std::shared_ptr<Dealer> dealer_;
    std::mutex mu_;
    std::deque<Bundle> inbox_[2];
    uint64_t requests_[2] = {0, 0};
    std::set<std::string> persistent_;  // generated persistent objects (type|params)
    uint64_t transferred_[2] = {0, 0};
    double gen_seconds_ = 0.0;
    uint64_t generations_ = 0;
};

}  // namespace highlevel
}  // namespace opencrab2
