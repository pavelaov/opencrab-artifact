// OpenCrab v2 C++17 mirror: the session epoch loop (paper, Section 4.2, SessionControl)
// and its evidence record (mirror of python/opencrab2/highlevel/session.py).
//
// Session start, both parties in lock step:
//   hello:     exchange of the public session binding (session_id, setup_id,
//              model_version, numeric_contract_hash, profile_id, n_generate,
//              policy, model shape digest, incumbent plan, resume flag and
//              epoch, generation).
//              A model_version difference is refused with OBJECT_VERSION_MISMATCH,
//              a contract difference with CONTRACT_MISMATCH, a profile difference
//              with PROFILE_MISMATCH, anything else with SESSION_BINDING_MISMATCH.
//   ready:     each party checks its OWN durable state against its declared
//              binding (ledger binding, and on a resume the public state file)
//              and both exchange a readiness flag with the stable code; a party
//              that is not ready returns its code, its peer SESSION_PEER_REFUSED.
//   setup:     server-private vectors are shared, the LM head is installed.
// Per epoch:
//   0. cancel: both parties exchange their declared cancel flags (public). If
//              either is raised the epoch is not started: RESERVED material and
//              every untouched one-time bundle are burned THROUGH THE LEDGER
//              (DealerClient::burn_outstanding), the public state is saved with
//              cancelled = true and the session stops. It may be resumed later
//              under the same session_id.
//   1. sense:     fixed-size probe (P1 measures), signed TelemetryFrame, both verify
//   2. candidates: SessionState (incumbent plan, installed objects with versions,
//                 ledger stock, KV bucket, remaining epochs) -> enumerate_candidates
//                 (infeasible plans, e.g. KARY_MAGNITUDE, stay listed and are never chosen)
//   3. decide:    the configured policy (B0_fixed:<plan>, B1_naive, B2_..., B4_opencrab,
//                 optionally with "+planner:<name>") fed with identical frames on both parties
//   4. commit:    two-phase commit of the decision digest. The digest is the SHA-256
//                 of the canonical JSON of {epoch, generation, plan, installed objects
//                 WITH their versions, target objects of the plan, the object version
//                 of this party, frame digest, policy, session binding (session_id,
//                 setup_id, model_version, numeric_contract_hash, profile_id)}; a
//                 mismatch in either phase aborts the session (fail closed)
//   5. install:   transactional install of the missing wmask objects (real protocol)
//   6. execute:   epoch 0 = prefill of the private prompt, later epochs = one decode
//                 step on the A64-shared KV cache; the token is released to P1 only
//                 through the ReleasePolicy
//   7. state:     the public state (epochs done, plan, installed objects, counts;
//                 on P1 also its own released tokens) is written atomically
//
// Cancellation (paper, Section 4.3). Declared sources: SessionConfig::
// cancel_after_epoch (raise the flag at the boundary after that epoch) and
// SessionConfig::cancel_flag (an asynchronous request, e.g. the sidecar's
// POST /v1/cancel; read at every boundary). A cancelled run returns its record
// with cancelled = true, completed = false, security_eligible = false, and
// leaves no RESERVED id in the journal. Any failure of the loop also burns the
// outstanding material (uncertainty: the two pools may no longer be aligned).
//
// Resumption (SessionConfig::resume): the same session_id continues from the
// public state. The binding is checked first (a model_version that differs from
// the state or from the ledger binding is refused with OBJECT_VERSION_MISMATCH,
// a missing state with SESSION_STATE_MISSING, a completed session with
// SESSION_COMPLETED). The two parties continue at the SMALLER of their recorded
// epochs (a party that already holds a later token re-derives it; the outputs
// are deterministic, so nothing new is released, and P1 compares the re-derived
// token with the recorded one: RESUME_OUTPUT_MISMATCH), with the incumbent plan
// recorded by the party that is behind, under a new generation (the larger of
// the two proposals, each party proposing its recorded one + 1):
// every pooled bundle of the previous generation is burned or dropped, the
// per-matrix masks included, and new bundles are requested with
// params.generation (fresh masks, a fresh request namespace at the dealer).
// In-memory protocol state does not survive this function, so the first
// resumed epoch is the recovery gate of the Python runtime: it re-runs the
// prefill over the prompt plus the tokens released so far, which rebuilds the
// shared KV cache and releases the next token; later epochs decode normally.
// Design limit: the public state file is local and does not resist a rolled
// back host (same limit as the ledger journal).
//
// Timing boundaries (the cold, warm and online times of paper, Section 7.4):
// T_cold = everything from session start (hello,
// setup, vector sharing, LM head install, sensing, planning, installs, execution,
// release); T_warm = T_cold minus the declared reusable setup; T_online = the
// sum of the protocol execution intervals minus in-process dealer generation
// time (the isolated dealer runs ahead of the online phase in a deployment).
// One evidence JSON line per party is appended to out_path.
#pragma once

#include <opencrab2/highlevel/candidates.hpp>
#include <opencrab2/highlevel/channel.hpp>
#include <opencrab2/highlevel/controller.hpp>
#include <opencrab2/highlevel/costmodel.hpp>
#include <opencrab2/highlevel/dealer.hpp>
#include <opencrab2/highlevel/ledger.hpp>
#include <opencrab2/highlevel/party.hpp>
#include <opencrab2/highlevel/telemetry.hpp>
#include <opencrab2/highlevel/tiny_decoder.hpp>

#include <atomic>
#include <functional>
#include <string>
#include <vector>

namespace opencrab2 {
namespace highlevel {

struct SessionTrace {
    std::string trace_id = "S0-stable";
    std::string family = "S0";
    std::string split = "test";
    std::vector<TracePoint> points;
    std::string trace_hash() const;  // sha256 of the canonical point list
    Value to_value() const;
    static SessionTrace stable(const std::string& id, double bandwidth_bps, double one_way_delay_s);
};

struct SessionConfig {
    std::string model = "tiny-decoder";
    int n_layers = 1;
    std::vector<int> prompt_ids;
    int n_generate = 3;
    std::string policy = "B4_opencrab";
    std::string trace_id = "S0-stable";
    int seed = 0;
    NumericContract contract = default_contract();
    std::string profile_id = "SH2P-TrustedOffline-v1";
    int length_bucket = 32;
    TinyConfig tiny;
    TinyPlan initial_plan;
    std::string session_id = "sess-0";
    std::string setup_id = "setup-0";
    std::string model_version = "model-0";
    std::string dealer_mode = "ISOLATED_RECIPIENT_BUNDLES";
    Bytes telemetry_key;
    std::string run_id = "run-0";
    int repeat = 0;
    std::string workload_scope = "full-model";
    std::string code_commit = "cpp-mirror-untracked";
    ControllerConfig controller;
    double refill_bytes_per_s = 1e9;
    double max_bandwidth_bps = 1e12;
    int probe_samples = 3;          // >= 1 (TELEMETRY_PARAMS otherwise)
    size_t probe_burst = 65536;     // >= 1 byte
    // Optional: called on P1 with (epoch, token) for every released token (the
    // sidecar streams them to the browser page). Never called on P0.
    std::function<void(int, int)> on_token;

    // --- cancellation and resumption (paper, Section 4.3) ------------------------
    // Declared cancellation: this party raises its cancel flag at the epoch
    // boundary that follows epoch cancel_after_epoch (-1 = never). A flag raised
    // after the last epoch has no effect: the session completed.
    int cancel_after_epoch = -1;
    // Asynchronous cancel request, read at every epoch boundary (may be null).
    const std::atomic<bool>* cancel_flag = nullptr;
    // Continue the session bound to session_id from its public state file.
    bool resume = false;
    // Public state file; "" = <ledger journal path>.state.json.
    std::string state_path;
    // The (model_version, client_key_domain) pair versions the installed objects.
    std::string client_key_domain = "client-key-domain-0";
    // true (the default): the party's share randomness is seeded from the OS
    // CSPRNG (fail closed when it is unavailable) and `seed` is ignored. false:
    // derived from the public `seed` (deterministic tests only). share_input
    // sends value minus mask, so a peer that knows the seed recomputes the mask
    // stream and reads the inputs: such a run is never security evidence. Its
    // record carries security_eligible = false, status DIAGNOSTIC and the note
    // PARTY_RNG_SEEDED (see session_eligibility below).
    bool party_rng_from_os = true;
    // Declared public operand bounds of the k-ary sites in real units (the magnitude
    // condition, paper, Section 5.3): |logit| <= logits_bound, |attention score| <= score_bound. A plan
    // whose arity does not fit the bound is listed as KARY_MAGNITUDE and never chosen.
    double logits_bound = 512.0;
    double score_bound = 512.0;
};

// Runs the full epoch loop for one party and returns the evidence record
// (also appended as one JSON line to out_path when out_path is not empty).
// weights_or_null: P0's model, reference_tokens: P1's expected greedy tokens
// (output_match), link: the in-process dealer link (material counters), or null.
// A declared cancellation is NOT a failure: the record comes back with
// cancelled = true and completed = false.
Expected<Value> run_session_party(int party, const SessionConfig& cfg, Channel& channel, DealerClient& dealer,
                                  MaterialLedger& ledger, const FittedCostModel& costmodel, const SessionTrace& trace,
                                  const TinyWeights* weights_or_null, const std::vector<int>* reference_tokens,
                                  const std::string& out_path, InProcessDealerLink* link = nullptr);

// Public session state (resumption). The file is one JSON object:
//   {schema: "opencrab2-session-state-v1", party, session_id, setup_id, model_version,
//    numeric_contract_hash, profile_id, epochs_done, plan, installed: {object: version},
//    generated_count, prompt_len, n_generate, generation, policy, cancelled, completed,
//    cancel_epoch [, generated_tokens (P1 only: its own released outputs)]}
std::string session_state_path(const SessionConfig& cfg, const MaterialLedger& ledger);
// Codes: SESSION_STATE_MISSING (no file), SESSION_STATE_CORRUPT (unreadable or not this schema).
Expected<Value> read_session_state(const std::string& path);

// Eligibility and status of a session record (paper, Section 3, evidence classes; one rule for
// both languages). security_eligible is true only when ALL of the following hold, default false:
//   the run completed, was not cancelled, did not fail, aborts == 0, diagnostic_reveals == 0;
//   dealer_mode is ISOLATED_RECIPIENT_BUNDLES and the DealerClient is eligible (it ingested only
//   bundles of a dealer that declares isolated, OS keyed generation);
//   the party share randomness is OS keyed (SessionConfig::party_rng_from_os);
//   the dealer is not an in-process link (a dealer in the address space of a party is not an
//   isolated third process).
// status is DIAGNOSTIC when the CONFIGURATION is not the secure one (dealer mode or dealer
// randomness, seeded party randomness, in-process dealer) and MEASURED_NATIVE otherwise: a
// cancelled or aborted run of a secure configuration stays a native measurement that is simply
// not eligible. `notes` lists one stable code per unmet condition (empty when eligible):
// RUN_NOT_COMPLETED, RUN_CANCELLED, RUN_FAILED, ABORTS, DIAGNOSTIC_REVEALS,
// DEALER_MODE_NOT_ISOLATED, DEALER_NOT_ELIGIBLE, PARTY_RNG_SEEDED, DEALER_IN_PROCESS.
struct SessionEligibilityInputs {
    bool completed = false;
    bool cancelled = false;
    bool failed = false;
    int64_t aborts = 0;
    int64_t diagnostic_reveals = 0;
    std::string dealer_mode;        // SessionConfig::dealer_mode
    bool dealer_eligible = false;   // DealerClient::security_eligible()
    bool dealer_insecure_seen = false;  // DealerClient::insecure_material_seen()
    bool party_rng_from_os = false;
    bool in_process_dealer = false;
};
struct SessionEligibility {
    bool eligible = false;
    std::string status;             // MEASURED_NATIVE or DIAGNOSTIC
    std::string dealer_mode;        // what the record states: ISOLATED only without insecure material
    std::vector<std::string> notes;
};
SessionEligibility session_eligibility(const SessionEligibilityInputs& in);

// Evidence schema helpers (schema opencrab-native-evidence-v2).
// run_session_party RETURNS the analysis record: the schema fields (including "schema") flat, the
// token list in "tokens", analysis fields next to them. The line it APPENDS to out_path is
// conform_evidence_record(record): exactly the shape of python/opencrab2/experiments/schema.py
// (schema fields flat, tokens as a count, every other field under extra, extra.language "cpp").
// validate_evidence_record accepts both shapes and applies the eligibility rule of the schema:
// security_eligible requires dealer_mode ISOLATED_RECIPIENT_BUNDLES, diagnostic_reveals == 0,
// aborts == 0 and status MEASURED_NATIVE (SCHEMA_ELIGIBILITY). A record that carries "extra"
// (the file shape) must not carry any other unknown top-level field (SCHEMA_UNKNOWN_FIELD).
extern const char* const kEvidenceSchemaId;   // "opencrab-native-evidence-v2"
const std::vector<std::string>& evidence_record_fields();
const std::vector<std::string>& evidence_optional_fields();
Value conform_evidence_record(const Value& record);
Expected<Unit> validate_evidence_record(const Value& record);
Expected<Unit> append_jsonl(const std::string& path, const Value& record);
uint64_t peak_rss_bytes();  // 0 when unavailable

}  // namespace highlevel
}  // namespace opencrab2
