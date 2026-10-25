// OpenCrab v2 C++17 mirror: two-party session test.
//
// Runs a 1-layer random-weight tiny decoder session (prefill of a 4-token
// private prompt, then 2 decode epochs) in-process over a loopback socket pair
// with a shaped link (stable trace) for the policies B0_fixed:<plan>,
// B2_remaining_plus_install and B4_opencrab, with an in-process dealer link
// that delivers recipient-bound bundles to both parties. Validates the evidence
// record fields (evidence.hpp) on both parties, the released tokens
// against the plaintext greedy reference, the symmetric byte / round counters,
// the ledger state after the run, and the weight file round trip.
//
// Cancellation and resumption (paper, Section 4.3): a declared cancellation
// after epoch 1 stops both parties at the
// next epoch boundary, burns RESERVED and untouched material through the
// ledger and leaves no RESERVED id in the journal; a resume under the same
// session_id continues from the public state and completes with the reference
// tokens (warm: same ledger and client objects; cold: journals replayed into
// new objects); a resume with a different model_version is refused with
// OBJECT_VERSION_MISMATCH and releases nothing. Also: the two-phase commit
// digest covers the object versions (a party with another client key domain
// aborts with COMMIT_DIGEST_MISMATCH), probe_samples = 0 is refused with
// TELEMETRY_PARAMS, and a plan whose arity violates the declared magnitude
// bound is never executed (KARY_MAGNITUDE).
//
// Prints "PASS <n> checks" and exits 0, or prints the failing check and exits 1.
#include <opencrab2/highlevel/session.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <set>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace opencrab2;
using namespace opencrab2::highlevel;

static int g_checks = 0;

#define CHECK(cond, msg)                                                                  \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__);                  \
            return 1;                                                                     \
        }                                                                                 \
        ++g_checks;                                                                       \
    } while (0)

#define CHECK_OK(expr, msg)                                                               \
    do {                                                                                  \
        auto&& oc2_r = (expr);                                                            \
        if (!oc2_r.ok) {                                                                  \
            std::fprintf(stderr, "FAIL: %s: %s (line %d)\n", msg, oc2_r.code.c_str(), __LINE__); \
            return 1;                                                                     \
        }                                                                                 \
        ++g_checks;                                                                       \
    } while (0)

static std::string temp_path(const std::string& name, const std::string& ext) {
    const char* base = std::getenv("TEMP");
    if (!base) base = std::getenv("TMPDIR");
    if (!base) base = "/tmp";
    auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::string(base) + "/oc2_session_" + name + "_" + std::to_string(static_cast<long long>(stamp)) + ext;
}

static bool has_prefix(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }

static bool write_text(const std::string& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << text;
    return static_cast<bool>(out);
}

// One dealer, one in-process link and one journal per party; the ledger and
// client objects live across several runs of the same session (cancel, then
// resume) unless reopen() replaces them with new objects over the same journals
// (the cold resume of a restarted process).
struct Harness {
    std::shared_ptr<Dealer> dealer;
    std::unique_ptr<InProcessDealerLink> link;
    SessionTrace trace;
    std::string journal[2];
    std::unique_ptr<MaterialLedger> ledger[2];
    std::unique_ptr<DealerClient> client[2];
    Bytes key[2];
    Value binding;
    std::string contract_hash;

    // os_dealer: key the dealer from the OS CSPRNG instead of the fixed test seed. The harness
    // dealer stays an in-process link either way, so no run of this file is security evidence.
    Expected<Unit> init(const SessionConfig& cfg, const std::string& tag, bool os_dealer = false) {
        DealerConfig dc;
        dc.setup_id = cfg.setup_id;
        dc.model_version = cfg.model_version;
        dc.numeric_contract_hash = cfg.contract.contract_hash();
        dc.mode = cfg.dealer_mode;
        contract_hash = dc.numeric_contract_hash;
        key[0] = Bytes(32, 0x31);
        key[1] = Bytes(32, 0x32);
        dealer = std::make_shared<Dealer>(dc, os_dealer ? ll::Rng::from_os_or_abort() : ll::Rng::from_seed(99), key[0], key[1]);
        link = std::make_unique<InProcessDealerLink>(dealer);
        trace = SessionTrace::stable("S0-stable-200M-0.5ms", 200e6, 0.0005);
        binding = dict({{"session_id", cfg.session_id}, {"setup_id", cfg.setup_id}, {"model_version", cfg.model_version},
                        {"numeric_contract_hash", dc.numeric_contract_hash}, {"profile_id", cfg.profile_id}});
        for (int p = 0; p < 2; ++p) {
            journal[p] = temp_path(tag + "_p" + std::to_string(p), ".jsonl");
            ledger[p] = std::make_unique<MaterialLedger>(journal[p], binding);
            OC2_CHECK(ledger[p]->open());
            client[p] = std::make_unique<DealerClient>(p, *ledger[p], key[p], contract_hash);
            link->attach(*client[p]);
        }
        return ok_unit();
    }

    // New ledger and client objects over the same journals: what a restarted process holds.
    Expected<Unit> reopen() {
        for (int p = 0; p < 2; ++p) {
            client[p].reset();
            ledger[p].reset();
            ledger[p] = std::make_unique<MaterialLedger>(journal[p], binding);
            OC2_TRY(rec, ledger[p]->recover());
            (void)rec;
            client[p] = std::make_unique<DealerClient>(p, *ledger[p], key[p], contract_hash);
            link->attach(*client[p]);
        }
        return ok_unit();
    }

    // Runs both parties over a new loopback pair; a failing party closes its channel (fail closed).
    Expected<Unit> run(const SessionConfig& cfg0, const SessionConfig& cfg1, const FittedCostModel& cm, const TinyWeights& W,
                       const std::vector<int>& reference, const std::string& evidence, Expected<Value> out[2]) {
        double origin = now_s();
        auto sh0 = std::make_shared<WallClockShaper>(trace.points, origin);
        auto sh1 = std::make_shared<WallClockShaper>(trace.points, origin);
        OC2_TRY(pair, loopback_pair(sh0, sh1));
        std::unique_ptr<Channel> ch[2] = {std::move(pair.first), std::move(pair.second)};
        auto body = [&](int p) {
            const SessionConfig& cfg = p == 0 ? cfg0 : cfg1;
            out[p] = run_session_party(p, cfg, *ch[p], *client[p], *ledger[p], cm, trace, p == 0 ? &W : nullptr,
                                       p == 1 ? &reference : nullptr, evidence, link.get());
            if (!out[p].ok) ch[p]->close();
        };
        std::thread t0(body, 0), t1(body, 1);
        t0.join();
        t1.join();
        return ok_unit();
    }

    // Read-only audit of a journal file: the number of entries that are still RESERVED. (A scratch
    // ledger with another binding can no longer replay the journal: LEDGER_BINDING_MISMATCH.)
    int64_t reserved_in_journal(int p, int64_t* spent = nullptr) const {
        auto rec = MaterialLedger::audit_file(journal[p]);
        if (!rec.ok) return -1;
        if (spent) *spent = rec.value.get_int("spent").value;
        return rec.value.get_int("reserved").value;
    }

    void cleanup(const SessionConfig& cfg) {
        for (int p = 0; p < 2; ++p) {
            std::string state = ledger[p] ? session_state_path(cfg, *ledger[p]) : journal[p] + ".state.json";
            client[p].reset();
            ledger[p].reset();
            std::remove(journal[p].c_str());
            std::remove(state.c_str());
            std::remove((state + ".tmp").c_str());
        }
    }
};

static int check_record(const Value& rec, const std::string& policy, int party, const std::vector<int>& reference,
                        const std::string& contract_hash) {
    CHECK_OK(validate_evidence_record(rec), "record schema");
    CHECK(rec.get_string("policy_id").value == policy, "policy_id");
    CHECK(rec.get_string("numeric_contract_hash").value == contract_hash, "contract hash");
    CHECK(rec.get_string("profile_id").value == "SH2P-TrustedOffline-v1", "profile id");
    // Eligibility, default false. This harness is NOT the secure configuration: the dealer is keyed from
    // the public constant 99 (its bundles carry insecure = true whatever the mode string says), it
    // lives in the process of both parties, and the party masks derive from the public session
    // seed. A completed run must therefore say DIAGNOSTIC and not eligible, and name every reason.
    CHECK(rec.get_string("schema").value == "opencrab-native-evidence-v2", "schema id");
    CHECK(rec.get_string("dealer_mode").value == "INSECURE_TEST_ONLY_SHARED_SEED", "a seeded dealer is recorded as insecure");
    CHECK(!rec.get("security_eligible")->as_bool().value, "a seeded in-process harness run is never security eligible");
    CHECK(rec.get_string("status").value == "DIAGNOSTIC", "status DIAGNOSTIC");
    CHECK(rec.get_string("party_rng").value == "seed", "party randomness from the public seed is stated");
    {
        std::set<std::string> notes;
        for (const auto& n : *rec.get("eligibility_notes")->as_list().value) notes.insert(n.as_string().value);
        CHECK(notes == std::set<std::string>({"DEALER_NOT_ELIGIBLE", "PARTY_RNG_SEEDED", "DEALER_IN_PROCESS"}),
              "every unmet eligibility condition is named");
    }
    CHECK(rec.get_int("aborts").value == 0, "no aborts");
    CHECK(rec.get_int("diagnostic_reveals").value == 0, "no diagnostic reveals");
    CHECK(rec.get_int("bytes").value > 0 && rec.get_int("rounds").value > 0, "bytes and rounds counted");
    CHECK(rec.get_number("T_cold_s").value >= rec.get_number("T_warm_s").value && rec.get_number("T_warm_s").value >= 0,
          "T_cold >= T_warm >= 0");
    CHECK(rec.get_number("T_online_s").value > 0 && rec.get_number("T_online_s").value <= rec.get_number("T_cold_s").value,
          "0 < T_online <= T_cold");
    CHECK(rec.get_int("material_consumed").value > 0 && rec.get_int("material_transferred").value >= rec.get_int("material_consumed").value,
          "material transferred >= consumed > 0");
    CHECK(rec.get_int("material_generated").value >= rec.get_int("material_transferred").value, "material generated >= transferred");
    CHECK(rec.get_int("material_burned").value == 0, "nothing burned in a clean run");
    CHECK(rec.get_string("clock_basis").value == "wall-clock-exogenous", "clock basis");
    CHECK(rec.get_string("trace_hash").value.size() == 64, "trace hash present");
    CHECK(rec.get_string("initial_state_hash").value.size() == 64, "initial state hash present");
    CHECK(rec.get("completed")->as_bool().value && !rec.get("cancelled")->as_bool().value && !rec.get("resumed")->as_bool().value,
          "completed, not cancelled, not resumed");
    CHECK(rec.get_int("reserved_after_run").value == 0 && rec.get_int("epochs_done").value == 3 && rec.get_int("generation").value == 0,
          "no RESERVED id, three epochs done, generation 0");
    CHECK(rec.get("epochs")->as_list().value->size() == 3, "three epochs recorded");
    const List& eps = *rec.get("epochs")->as_list().value;
    CHECK(eps[0].get_int("n_rows").value == 4 && eps[1].get_int("n_rows").value == 1, "prefill then decode rows");
    CHECK(eps[2].get_int("kv_len").value == 6, "KV length grows with the epochs");
    for (const Value& e : eps) {
        CHECK(e.has("decision_digest") && e.get_string("decision_digest").value.size() == 64, "decision digest committed");
        CHECK(e.get("frame")->has("bandwidth_bps") && e.get("decision")->has("chosen"), "frame and decision recorded");
        CHECK(!e.get("recovery_gate")->as_bool().value, "no recovery gate in an uninterrupted run");
    }
    if (party == 1) {
        CHECK(rec.get("peak_rss_client")->is_int() && rec.get("peak_rss_server")->is_none(), "P1 reports client RSS only");
        const List& toks = *rec.get("tokens")->as_list().value;
        CHECK(toks.size() == 3, "three released tokens");
        std::vector<int> got;
        for (const Value& t : toks) got.push_back(static_cast<int>(t.as_int().value));
        CHECK(got == reference, "released tokens equal the plaintext greedy reference");
        CHECK(rec.get("output_match")->as_bool().value, "output_match true");
    } else {
        CHECK(rec.get("peak_rss_server")->is_int() && rec.get("peak_rss_client")->is_none(), "P0 reports server RSS only");
        CHECK(rec.get("tokens")->as_list().value->empty(), "P0 learns no token");
        CHECK(rec.get("output_match")->is_none(), "P0 has no output_match");
    }
    return 0;
}

static std::vector<int> tokens_of(const Value& rec) {
    std::vector<int> got;
    for (const Value& t : *rec.get("tokens")->as_list().value) got.push_back(static_cast<int>(t.as_int().value));
    return got;
}

// ---------------------------------------------------------------------------
// cancellation and resumption
// ---------------------------------------------------------------------------
static int test_cancel_resume(const TinyWeights& W, const SessionConfig& base, const std::vector<int>& reference, const FittedCostModel& cm,
                              const std::string& evidence, bool cold) {
    SessionConfig cfg = base;
    cfg.policy = "B2_remaining_plus_install";
    cfg.session_id = cold ? "sess-cancel-cold" : "sess-cancel-warm";
    cfg.run_id = "run-cancel";
    Harness h;
    CHECK_OK(h.init(cfg, cold ? "cc" : "cw"), "harness");
    // Material that the session never consumes: one bundle RESERVED for a gate that will not run
    // and one untouched AVAILABLE bundle. A cancellation must burn both through the ledger.
    std::string reserved_id[2], idle_id[2];
    {
        auto r2f = h.dealer->make_r2f(4, 68720050177ULL);
        auto tr = h.dealer->make_trunc(3, 5, 40);
        CHECK(r2f.ok && tr.ok, "extra bundles");
        for (int p = 0; p < 2; ++p) {
            CHECK_OK(h.client[p]->ingest(p == 0 ? r2f.value.for0 : r2f.value.for1), "ingest r2f");
            CHECK_OK(h.client[p]->ingest(p == 0 ? tr.value.for0 : tr.value.for1), "ingest trunc");
            reserved_id[p] = r2f.value.for0.header.get_string("material_id").value;
            idle_id[p] = tr.value.for0.header.get_string("material_id").value;
            CHECK_OK(h.ledger[p]->reserve({reserved_id[p]}, "epoch-99", 99), "reserve for a gate that never runs");
            CHECK(h.client[p]->reserved_ids().size() == 1, "one RESERVED id before the run");
        }
    }
    // --- run 1: P1 declares a cancellation after epoch 1 (cold variant: after epoch 0); P0 follows
    const int cancel_after = cold ? 0 : 1;
    SessionConfig c0 = cfg, c1 = cfg;
    c1.cancel_after_epoch = cancel_after;
    Expected<Value> out[2];
    CHECK_OK(h.run(c0, c1, cm, W, reference, evidence, out), "cancelled run");
    for (int p = 0; p < 2; ++p) {
        if (!out[p].ok) std::fprintf(stderr, "party %d: %s\n", p, out[p].code.c_str());
        CHECK_OK(out[p], "a declared cancellation is not a failure");
        const Value& r = out[p].value;
        CHECK_OK(validate_evidence_record(r), "cancelled record validates");
        CHECK(r.get("cancelled")->as_bool().value && !r.get("completed")->as_bool().value, "cancelled, not completed");
        CHECK(r.get_int("cancel_epoch").value == cancel_after + 1 && r.get_int("epochs_done").value == cancel_after + 1,
              "the epoch after the declared one is not started (both parties)");
        CHECK(!r.get("security_eligible")->as_bool().value && r.get_string("status").value == "DIAGNOSTIC", "a cancelled run is not eligible");
        CHECK(r.get("epochs")->as_list().value->size() == static_cast<size_t>(cancel_after + 1), "epochs executed before the cancellation");
        CHECK(r.get_int("aborts").value == 0 && r.get_string("failure").value.empty(), "cancellation is neither an abort nor a failure");
        // no RESERVED material is left: record, client view, ledger state, journal replay
        CHECK(r.get_int("reserved_after_run").value == 0 && h.client[p]->reserved_ids().empty(), "cancel after epoch leaves no RESERVED material");
        CHECK(h.ledger[p]->state(reserved_id[p]).value == "BURNED", "the RESERVED bundle was burned through the ledger");
        CHECK(h.ledger[p]->state(idle_id[p]).value == "BURNED", "the untouched AVAILABLE bundle was burned as safe waste");
        const Value* br = r.get("burned_outstanding");
        CHECK(br && br->is_dict() && br->get_int("reserved_ids").value == 1 && br->get_int("available_ids").value >= 1 &&
                  br->get_int("bytes_burned").value > 0,
              "burn report: one RESERVED id, the untouched bundles, their bytes");
        CHECK(r.get_int("material_burned").value >= br->get_int("bytes_burned").value && r.get_int("material_burned").value > 0,
              "material_burned counts the safe waste");
        CHECK(h.reserved_in_journal(p) == 0, "journal replay finds no RESERVED entry after the cancellation");
        CHECK(h.client[p]->pooled_count("beaver") == 0 && h.client[p]->pooled_count("kary_select") == 0, "one-time pools are empty after the cancellation");
        // public state
        auto st = read_session_state(session_state_path(cfg, *h.ledger[p]));
        CHECK_OK(st, "public state written");
        CHECK(st.value.get_int("epochs_done").value == cancel_after + 1 && st.value.get("cancelled")->as_bool().value &&
                  !st.value.get("completed")->as_bool().value && st.value.get_string("session_id").value == cfg.session_id &&
                  st.value.get_string("model_version").value == cfg.model_version,
              "public state: epochs done, cancelled, binding");
        CHECK(st.value.has("generated_tokens") == (p == 1), "only P1's state holds its released tokens");
    }
    {
        std::vector<int> got = tokens_of(out[1].value);
        std::vector<int> want(reference.begin(), reference.begin() + cancel_after + 1);
        CHECK(got == want && tokens_of(out[0].value).empty(), "tokens released before the cancellation equal the reference prefix");
        CHECK(out[0].value.get_int("bytes").value == out[1].value.get_int("bytes").value &&
                  out[0].value.get_int("rounds").value == out[1].value.get_int("rounds").value,
              "both parties count the same bytes and rounds up to the cancellation");
    }
    // --- refused resumes: a different model_version (both parties, then P0 only), then a party that does not resume
    {
        SessionConfig r0 = cfg, r1 = cfg;
        r0.resume = r1.resume = true;
        r0.model_version = r1.model_version = "another-model-version";
        Expected<Value> bad[2];
        CHECK_OK(h.run(r0, r1, cm, W, reference, evidence, bad), "refused resume run");
        for (int p = 0; p < 2; ++p)
            CHECK(!bad[p].ok && has_prefix(bad[p].code, "OBJECT_VERSION_MISMATCH"), "resume with a different model_version is refused (both parties)");
        SessionConfig q0 = cfg, q1 = cfg;
        q0.resume = q1.resume = true;
        q0.model_version = "p0-swapped-the-model";
        Expected<Value> bad2[2];
        CHECK_OK(h.run(q0, q1, cm, W, reference, evidence, bad2), "refused resume run (P0 only)");
        for (int p = 0; p < 2; ++p)
            CHECK(!bad2[p].ok && has_prefix(bad2[p].code, "OBJECT_VERSION_MISMATCH"), "a model_version that differs between the parties is refused on both");
        SessionConfig s0 = cfg, s1 = cfg;
        s1.resume = true;
        Expected<Value> bad3[2];
        CHECK_OK(h.run(s0, s1, cm, W, reference, evidence, bad3), "resume against a fresh start");
        for (int p = 0; p < 2; ++p)
            CHECK(!bad3[p].ok && has_prefix(bad3[p].code, "SESSION_BINDING_MISMATCH"), "one party resuming against a fresh start is refused");
        for (int p = 0; p < 2; ++p) {
            auto st = read_session_state(session_state_path(cfg, *h.ledger[p]));
            CHECK(st.ok && st.value.get_int("epochs_done").value == cancel_after + 1 && st.value.get_int("generation").value == 0,
                  "refused attempts leave the public state untouched");
            CHECK(h.reserved_in_journal(p) == 0, "refused attempts leave no RESERVED entry");
        }
    }
    // --- one party behind the other (warm variant): P0 lost its last state write. The parties
    // continue at the SMALLER recorded epoch; P1 re-derives the token it already holds and
    // compares it with its record before anything is advanced.
    int64_t expect_resume_epoch = cancel_after + 1, expect_generation = 1;
    if (!cold) {
        const std::string sp0 = session_state_path(cfg, *h.ledger[0]), sp1 = session_state_path(cfg, *h.ledger[1]);
        auto s0 = read_session_state(sp0), s1 = read_session_state(sp1);
        CHECK(s0.ok && s1.ok, "public states readable");
        Value p0_behind = s0.value;
        p0_behind.set("epochs_done", static_cast<int64_t>(cancel_after));
        p0_behind.set("generated_count", static_cast<int64_t>(cancel_after));
        CHECK(write_text(sp0, to_json(p0_behind).value), "P0 state one epoch behind");
        // (i) P1's record of that epoch is corrupted: the re-derived token differs and P1 stops
        Value p1_bad = s1.value;
        List toks = *p1_bad.get("generated_tokens")->as_list().value;
        toks[static_cast<size_t>(cancel_after)] = Value(static_cast<int64_t>((toks[static_cast<size_t>(cancel_after)].as_int().value + 1) % 32));
        p1_bad.set("generated_tokens", Value(toks));
        CHECK(write_text(sp1, to_json(p1_bad).value), "P1 state with a corrupted recorded token");
        SessionConfig m0 = cfg, m1 = cfg;
        m0.resume = m1.resume = true;
        Expected<Value> mm[2];
        CHECK_OK(h.run(m0, m1, cm, W, reference, evidence, mm), "resume over a corrupted record");
        CHECK(!mm[1].ok && has_prefix(mm[1].code, "RESUME_OUTPUT_MISMATCH"), "P1 refuses a re-derived token that differs from its record");
        CHECK(!mm[0].ok, "P0 stops with its peer");
        for (int p = 0; p < 2; ++p) CHECK(h.reserved_in_journal(p) == 0 && h.client[p]->reserved_ids().empty(), "the failed attempt leaves no RESERVED material");
        auto s1_after = read_session_state(sp1);
        CHECK(s1_after.ok && s1_after.value.get_int("generation").value == 0, "P1 did not advance its state on the mismatch");
        // (ii) the true record restored, P0 behind again: the resume below starts one epoch earlier
        auto s0_after = read_session_state(sp0);
        CHECK(s0_after.ok && s0_after.value.get_int("generation").value == 1, "P0 recorded the generation of the failed attempt");
        Value p0_again = s0_after.value;
        p0_again.set("epochs_done", static_cast<int64_t>(cancel_after));
        p0_again.set("generated_count", static_cast<int64_t>(cancel_after));
        // the plan in force before that epoch (here: the declared initial plan); P1, which is ahead,
        // must adopt it so that both parties plan from the same incumbent
        p0_again.set("plan", dict({{"embedding", "generic"}, {"linear", "generic"}, {"argmax_k", static_cast<int64_t>(2)},
                                   {"max_k", static_cast<int64_t>(2)}}));
        CHECK(write_text(sp0, to_json(p0_again).value), "P0 state one epoch behind again");
        CHECK(write_text(sp1, to_json(s1.value).value), "P1 state restored");
        expect_resume_epoch = cancel_after;   // min(P0: cancel_after, P1: cancel_after + 1)
        expect_generation = 2;                // max(P0: 1 + 1, P1: 0 + 1)
    }
    // --- resume: same session_id, continues from the public state and completes
    if (cold) CHECK_OK(h.reopen(), "cold resume: new ledger and client objects over the same journals");
    {
        SessionConfig r0 = cfg, r1 = cfg;
        r0.resume = r1.resume = true;
        r0.run_id = r1.run_id = "run-resume";
        Expected<Value> res[2];
        CHECK_OK(h.run(r0, r1, cm, W, reference, evidence, res), "resumed run");
        for (int p = 0; p < 2; ++p) {
            if (!res[p].ok) std::fprintf(stderr, "party %d resume: %s\n", p, res[p].code.c_str());
            CHECK_OK(res[p], "resume completes");
            const Value& r = res[p].value;
            CHECK_OK(validate_evidence_record(r), "resumed record validates");
            CHECK(r.get("resumed")->as_bool().value && r.get("completed")->as_bool().value && !r.get("cancelled")->as_bool().value,
                  "resumed and completed");
            CHECK(r.get_int("resume_epoch").value == expect_resume_epoch && r.get_int("generation").value == expect_generation &&
                      r.get_int("epochs_done").value == 3,
                  "continues at the smaller recorded epoch under the next generation");
            const List& eps = *r.get("epochs")->as_list().value;
            CHECK(eps.size() == static_cast<size_t>(3 - expect_resume_epoch), "only the remaining epochs run");
            CHECK(eps[0].get("recovery_gate")->as_bool().value && eps[0].get_int("n_rows").value == 4 + expect_resume_epoch &&
                      eps[0].get_int("epoch").value == expect_resume_epoch,
                  "first resumed epoch is the recovery gate over prompt plus released tokens");
            for (size_t i = 1; i < eps.size(); ++i)
                CHECK(!eps[i].get("recovery_gate")->as_bool().value && eps[i].get_int("n_rows").value == 1, "later resumed epochs decode one row");
            CHECK(!r.get("security_eligible")->as_bool().value && r.get_string("status").value == "DIAGNOSTIC",
                  "a resumed run of the seeded in-process harness completes but is not eligible");
            CHECK(r.get_int("reserved_after_run").value == 0 && h.reserved_in_journal(p) == 0, "no RESERVED material after the resumed run");
            auto st = read_session_state(session_state_path(cfg, *h.ledger[p]));
            CHECK(st.ok && st.value.get("completed")->as_bool().value && st.value.get_int("epochs_done").value == 3 &&
                      st.value.get_int("generation").value == expect_generation && !st.value.get("cancelled")->as_bool().value,
                  "public state: completed under the resumed generation");
        }
        CHECK(tokens_of(res[1].value) == reference && res[1].value.get("output_match")->as_bool().value, "resume completes with the reference tokens");
        {
            const List& e0 = *res[0].value.get("epochs")->as_list().value;
            const List& e1 = *res[1].value.get("epochs")->as_list().value;
            bool same = e0.size() == e1.size();
            for (size_t i = 0; same && i < e0.size(); ++i)
                same = e0[i].get_string("decision_digest").value == e1[i].get_string("decision_digest").value;
            CHECK(same, "resumed run: both parties committed the same decision digests (same incumbent, same object versions)");
            CHECK(res[0].value.get_string("initial_state_hash").value == res[1].value.get_string("initial_state_hash").value,
                  "resumed run: identical initial state on both parties");
        }
        CHECK(tokens_of(res[0].value).empty(), "P0 learns no token in a resumed run");
        CHECK(res[0].value.get_int("bytes").value == res[1].value.get_int("bytes").value, "resumed run: symmetric byte counters");
        // nothing is left to resume
        Expected<Value> done[2];
        CHECK_OK(h.run(r0, r1, cm, W, reference, evidence, done), "resume of a completed session");
        for (int p = 0; p < 2; ++p) CHECK(!done[p].ok && has_prefix(done[p].code, "SESSION_COMPLETED"), "a completed session cannot be resumed");
    }
    h.cleanup(cfg);
    return 0;
}

// ---------------------------------------------------------------------------
// fail-closed paths of the session loop
// ---------------------------------------------------------------------------
static int test_session_refusals(const TinyWeights& W, const SessionConfig& base, const std::vector<int>& reference, const FittedCostModel& cm,
                                 const std::string& evidence) {
    // resume without a public state
    {
        SessionConfig cfg = base;
        cfg.session_id = "sess-no-state";
        cfg.resume = true;
        Harness h;
        CHECK_OK(h.init(cfg, "ns"), "harness");
        Expected<Value> out[2];
        CHECK_OK(h.run(cfg, cfg, cm, W, reference, evidence, out), "run");
        for (int p = 0; p < 2; ++p) CHECK(!out[p].ok && has_prefix(out[p].code, "SESSION_STATE_MISSING"), "resume without a public state is refused");
        h.cleanup(cfg);
    }
    // the commit digest covers the object versions: another client key domain on P0 is caught in phase 1
    {
        SessionConfig cfg = base;
        cfg.session_id = "sess-digest";
        cfg.policy = "B2_remaining_plus_install";
        Harness h;
        CHECK_OK(h.init(cfg, "dg"), "harness");
        SessionConfig c0 = cfg, c1 = cfg;
        c0.client_key_domain = "another-key-domain";
        Expected<Value> out[2];
        CHECK_OK(h.run(c0, c1, cm, W, reference, evidence, out), "run");
        for (int p = 0; p < 2; ++p) {
            CHECK(!out[p].ok && has_prefix(out[p].code, "COMMIT_DIGEST_MISMATCH"), "a differing object version aborts the two-phase commit");
            CHECK(h.client[p]->reserved_ids().empty() && h.reserved_in_journal(p) == 0, "an aborted commit leaves no RESERVED material");
        }
        h.cleanup(cfg);
    }
    // the declared initial plans are public state: a difference is caught in the hello
    {
        SessionConfig cfg = base;
        cfg.session_id = "sess-plans";
        Harness h;
        CHECK_OK(h.init(cfg, "pl"), "harness");
        SessionConfig c0 = cfg, c1 = cfg;
        c1.initial_plan = TinyPlan{"generic", "generic", 4, 2};
        Expected<Value> out[2];
        CHECK_OK(h.run(c0, c1, cm, W, reference, evidence, out), "run");
        for (int p = 0; p < 2; ++p) CHECK(!out[p].ok && has_prefix(out[p].code, "SESSION_BINDING_MISMATCH"), "differing initial plans are refused");
        SessionConfig d0 = cfg, d1 = cfg;
        d1.n_generate = 2;
        CHECK_OK(h.run(d0, d1, cm, W, reference, evidence, out), "run");
        for (int p = 0; p < 2; ++p) CHECK(!out[p].ok && has_prefix(out[p].code, "SESSION_BINDING_MISMATCH"), "differing token budgets are refused");
        h.cleanup(cfg);
    }
    // the ledger binding is checked against the declared model version before anything runs
    {
        SessionConfig cfg = base;
        cfg.session_id = "sess-ledger-binding";
        Harness h;
        CHECK_OK(h.init(cfg, "lb"), "harness");
        SessionConfig other = cfg;
        other.model_version = "model-version-of-another-ledger";
        Expected<Value> out[2];
        CHECK_OK(h.run(other, other, cm, W, reference, evidence, out), "run");
        for (int p = 0; p < 2; ++p) CHECK(!out[p].ok && has_prefix(out[p].code, "OBJECT_VERSION_MISMATCH"), "ledger bound to another model_version is refused");
        h.cleanup(cfg);
    }
    // probe parameters: validated on both parties before any message (the unchecked code crashed)
    {
        SessionConfig cfg = base;
        cfg.session_id = "sess-probe";
        cfg.probe_samples = 0;
        Harness h;
        CHECK_OK(h.init(cfg, "pb"), "harness");
        Expected<Value> out[2];
        CHECK_OK(h.run(cfg, cfg, cm, W, reference, evidence, out), "run");
        for (int p = 0; p < 2; ++p) CHECK(!out[p].ok && has_prefix(out[p].code, "TELEMETRY_PARAMS"), "probe_samples = 0 is refused");
        cfg.probe_samples = 3;
        cfg.probe_burst = 0;
        CHECK_OK(h.run(cfg, cfg, cm, W, reference, evidence, out), "run");
        for (int p = 0; p < 2; ++p) CHECK(!out[p].ok && has_prefix(out[p].code, "TELEMETRY_PARAMS"), "probe_burst = 0 is refused");
        h.cleanup(cfg);
    }
    // the magnitude condition: a plan whose arity does not fit the declared logits bound is never executed
    {
        SessionConfig cfg = base;
        cfg.session_id = "sess-kary";
        cfg.n_generate = 1;
        cfg.logits_bound = 1500.0;   // differences up to 3000: beyond the fused node of 8, inside the node of 4
        TinyPlan fixed8{"generic", "generic", 8, 2};
        cfg.policy = "B0_fixed:" + fixed8.key();
        Harness h;
        CHECK_OK(h.init(cfg, "ky"), "harness");
        std::vector<int> ref1(reference.begin(), reference.begin() + 1);
        Expected<Value> out[2];
        CHECK_OK(h.run(cfg, cfg, cm, W, ref1, evidence, out), "run");
        for (int p = 0; p < 2; ++p) {
            if (!out[p].ok) std::fprintf(stderr, "party %d: %s\n", p, out[p].code.c_str());
            CHECK_OK(out[p], "the session runs on the admitted incumbent");
            CHECK(out[p].value.get("final_plan")->get_int("argmax_k").value == 2, "the fixed plan with argmax_k = 8 is not admitted and not installed");
            CHECK(out[p].value.get_int("rejected_candidates").value >= 8, "the argmax_k = 8 plans are counted as rejected");
            const Value& d0 = *(*out[p].value.get("epochs")->as_list().value)[0].get("decision");
            CHECK(d0.get_int("n_infeasible").value >= 8, "the decision record lists the rejected candidates");
            bool named = false;
            for (const Value& rj : *d0.get("rejected")->as_list().value) named = named || rj.get_string("reason").value == "KARY_MAGNITUDE";
            CHECK(named, "rejected with the stable code KARY_MAGNITUDE");
        }
        CHECK(tokens_of(out[1].value) == ref1, "token of the admitted plan equals the reference");
        h.cleanup(cfg);
        // an incumbent that violates the bound is refused outright
        SessionConfig bad = base;
        bad.session_id = "sess-kary-incumbent";
        bad.n_generate = 1;
        bad.logits_bound = 1500.0;
        bad.initial_plan = TinyPlan{"generic", "generic", 8, 2};
        bad.policy = "B2_remaining_plus_install";
        Harness h2;
        CHECK_OK(h2.init(bad, "kz"), "harness");
        CHECK_OK(h2.run(bad, bad, cm, W, ref1, evidence, out), "run");
        for (int p = 0; p < 2; ++p) CHECK(!out[p].ok && has_prefix(out[p].code, "KARY_MAGNITUDE"), "an inadmissible incumbent is never executed");
        h2.cleanup(bad);
    }
    return 0;
}

int main() {
    TinyConfig tc;
    tc.n_layers = 1;
    tc.d_model = 16;
    tc.n_head = 2;
    tc.vocab = 32;
    tc.max_pos = 32;
    tc.d_ff = 64;
    TinyWeights W = TinyWeights::random(tc, 2026);
    // weight file round trip (the exported-weights path)
    std::string wpath = temp_path("weights", ".bin");
    CHECK_OK(W.save(wpath), "save weights");
    auto W2 = TinyWeights::load(wpath);
    CHECK_OK(W2, "load weights");
    CHECK(W2.value.digest() == W.digest() && W2.value.cfg.d_ff == 64 && W2.value.w_lm == W.w_lm, "weight file round trip");
    Bytes corrupt = W.serialize();
    corrupt.resize(corrupt.size() - 4);
    CHECK(!TinyWeights::deserialize(corrupt).ok, "truncated weight file refused");
    std::remove(wpath.c_str());

    std::vector<int> prompt = {3, 17, 8, 25};
    TinyReference ref(W);
    std::vector<int> reference = ref.generate(prompt, 3);
    CHECK(reference.size() == 3, "reference tokens");

    SessionConfig base;
    base.model = "tiny-decoder";
    base.n_layers = 1;
    base.prompt_ids = prompt;
    base.n_generate = 3;
    base.tiny = tc;
    base.initial_plan = TinyPlan{"generic", "generic", 2, 2};
    base.session_id = "sess-two-party";
    base.setup_id = "setup-two-party";
    base.model_version = W.digest();
    base.telemetry_key = Bytes(32, 0x77);
    base.seed = 5;
    base.party_rng_from_os = false;   // deterministic fixtures: the masks derive from the public seed (DIAGNOSTIC)
    CHECK(SessionConfig().party_rng_from_os, "the library default is OS keyed party randomness");
    base.controller = ControllerConfig();
    base.controller.hysteresis_s = 0.001;
    std::string contract_hash = base.contract.contract_hash();
    // The calibrated model of this tree (results/calibration/fitted_model.json) when it is found
    // from the working directory, the HISTORICAL prior otherwise; the shared formula constants likewise.
    auto lf = load_formulas("");
    CHECK_OK(lf, "shared formulas readable when present");
    auto dm = default_model("");
    CHECK_OK(dm, "cost model");
    FittedCostModel cm = dm.value.model;
    for (const std::string& name : components()) CHECK(cm.has(name), "the session cost model prices every component");
    std::printf("cost model: %s (%s); formulas: %s\n", dm.value.status.c_str(), dm.value.source.c_str(),
                lf.value.found ? lf.value.path.c_str() : "builtin");
    std::string evidence = temp_path("evidence", ".jsonl");

    TinyPlan fixed{"wmask", "wmask", 8, 4};
    std::vector<std::string> policies = {"B0_fixed:" + fixed.key(), "B2_remaining_plus_install", "B4_opencrab", "B4_opencrab+planner:exact"};
    std::vector<std::string> journals;
    int n_records = 0;
    for (const auto& pol : policies) {
        std::printf("session policy %s\n", pol.c_str());
        SessionConfig cfg = base;
        cfg.policy = pol;
        cfg.run_id = "run-" + pol.substr(0, pol.find(':'));
        Harness h;
        CHECK_OK(h.init(cfg, pol.substr(0, 2) + std::to_string(n_records)), ("harness " + pol).c_str());
        Expected<Value> out[2];
        CHECK_OK(h.run(cfg, cfg, cm, W, reference, evidence, out), ("run " + pol).c_str());
        for (int p = 0; p < 2; ++p)
            if (!out[p].ok) std::fprintf(stderr, "party %d failed: %s\n", p, out[p].code.c_str());
        for (int p = 0; p < 2; ++p) {
            CHECK_OK(out[p], ("party record " + pol).c_str());
            if (check_record(out[p].value, pol, p, reference, contract_hash)) return 1;
            ++n_records;
        }
        const Value& r0 = out[0].value;
        const Value& r1 = out[1].value;
        CHECK(r0.get_int("bytes").value == r1.get_int("bytes").value, "both parties count the same bytes");
        CHECK(r0.get_int("rounds").value == r1.get_int("rounds").value, "both parties count the same rounds");
        CHECK(r0.get_string("initial_state_hash").value == r1.get_string("initial_state_hash").value, "identical initial state");
        CHECK(r0.get_int("initializations").value == r1.get_int("initializations").value &&
                  r0.get_int("post_init_switches").value == r1.get_int("post_init_switches").value,
              "identical switch history");
        {
            const List& e0 = *r0.get("epochs")->as_list().value;
            const List& e1 = *r1.get("epochs")->as_list().value;
            bool same = e0.size() == e1.size();
            for (size_t i = 0; same && i < e0.size(); ++i)
                same = e0[i].get_string("decision_digest").value == e1[i].get_string("decision_digest").value;
            CHECK(same, "both parties committed the same decision digests");
        }
        if (pol.rfind("B0_fixed", 0) == 0) {
            CHECK(r1.get("final_plan")->get_string("linear").value == "wmask" && r1.get("final_plan")->get_int("argmax_k").value == 8,
                  "fixed plan installed and kept");
            CHECK(r1.get_int("initializations").value == 5, "fixed plan installs wte and the 4 layer matrices at epoch 0");
            CHECK(r1.get_int("post_init_switches").value == 0, "fixed plan never switches");
        } else {
            auto fp = TinyPlan::from_json(value_to_json(*r1.get("final_plan")));
            CHECK(fp.ok, "adaptive policy ends on a valid plan");
            CHECK(r1.get_int("rejected_candidates").value >= 0, "rejected candidates counted");
        }
        std::printf("  bytes=%lld rounds=%lld T_cold=%.3f T_online=%.3f switches=%lld final=%s\n",
                    static_cast<long long>(r1.get_int("bytes").value), static_cast<long long>(r1.get_int("rounds").value),
                    r1.get_number("T_cold_s").value, r1.get_number("T_online_s").value,
                    static_cast<long long>(r1.get_int("post_init_switches").value),
                    to_json(*r1.get("final_plan")).value.c_str());
        // ledgers: replaying every journal must find no RESERVED material
        for (int p = 0; p < 2; ++p) {
            int64_t spent = 0;
            CHECK(h.reserved_in_journal(p, &spent) == 0 && spent > 0, "journal consistent");
            auto st = read_session_state(session_state_path(cfg, *h.ledger[p]));
            CHECK(st.ok && st.value.get("completed")->as_bool().value && st.value.get_int("epochs_done").value == 3, "public state says completed");
        }
        h.cleanup(cfg);
    }
    // evidence file: one parseable line per party and policy, in the schema shape of schema.py
    {
        std::ifstream in(evidence);
        std::string line;
        int n = 0;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            auto parsed = parse_json(line);
            CHECK_OK(parsed, "evidence line parses");
            const Value& v = parsed.value;
            CHECK_OK(validate_evidence_record(v), "evidence line validates");
            CHECK(v.get_string("schema").value == "opencrab-native-evidence-v2", "the line carries the schema id");
            const int64_t line_party = v.get("extra")->get_int("party").value;
            CHECK(v.get("tokens")->is_int() && v.get_int("tokens").value == (line_party == 1 ? 3 : 0),
                  "tokens is a count on disk (3 released to P1, none to P0)");
            CHECK(v.get("extra")->is_dict() && v.get("extra")->get("tokens_list") && v.get("extra")->get_string("language").value == "cpp",
                  "analysis fields live under extra");
            CHECK(!v.has("epochs") && !v.has("party") && v.get("extra")->has("epochs") && v.get("extra")->has("party"),
                  "no unknown top-level field");
            for (const auto& e : *std::get_if<Dict>(&v.v)) {
                const auto& req = evidence_record_fields();
                const auto& opt = evidence_optional_fields();
                CHECK(std::find(req.begin(), req.end(), e.key) != req.end() || std::find(opt.begin(), opt.end(), e.key) != opt.end(),
                      "every top-level key is a schema field");
            }
            ++n;
        }
        CHECK(n == n_records && n == 8, "eight evidence records written (4 policies x 2 parties)");
    }
    // the validator applies the eligibility rule of the schema
    {
        std::ifstream in(evidence);
        std::string line;
        std::getline(in, line);
        Value v = parse_json(line).value;
        v.set("security_eligible", true);
        CHECK(has_prefix(validate_evidence_record(v).code, "SCHEMA_ELIGIBILITY"), "an eligibility claim with an insecure dealer mode is refused");
        v.set("dealer_mode", "ISOLATED_RECIPIENT_BUNDLES");
        CHECK(has_prefix(validate_evidence_record(v).code, "SCHEMA_ELIGIBILITY"), "an eligibility claim with status DIAGNOSTIC is refused");
        v.set("status", "MEASURED_NATIVE");
        CHECK_OK(validate_evidence_record(v), "a consistent eligible record validates");
        v.set("aborts", static_cast<int64_t>(1));
        CHECK(has_prefix(validate_evidence_record(v).code, "SCHEMA_ELIGIBILITY"), "an eligibility claim with aborts is refused");
        v.set("aborts", static_cast<int64_t>(0));
        v.set("diagnostic_reveals", static_cast<int64_t>(1));
        CHECK(has_prefix(validate_evidence_record(v).code, "SCHEMA_ELIGIBILITY"), "an eligibility claim with reveals is refused");
        v.set("diagnostic_reveals", static_cast<int64_t>(0));
        v.set("stray", "x");
        CHECK(has_prefix(validate_evidence_record(v).code, "SCHEMA_UNKNOWN_FIELD"), "an unknown top-level field is refused");
        Value w = parse_json(line).value;
        w.set("schema", "something-else");
        CHECK(has_prefix(validate_evidence_record(w).code, "SCHEMA_BAD_VALUE"), "another schema id is refused");
    }
    // an evidence file that ends in the fragment of a dead append: the next record starts on a new line
    {
        const std::string frag = temp_path("fragment", ".jsonl");
        CHECK(write_text(frag, "{\"a\":1}\n{\"b\":{\"c\":3}"), "fixture with a torn last line");
        CHECK_OK(append_jsonl(frag, dict({{"d", static_cast<int64_t>(4)}})), "append after a fragment");
        std::ifstream in(frag, std::ios::binary);
        std::string all((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        CHECK(all == "{\"a\":1}\n{\"b\":{\"c\":3}\n{\"d\":4}\n", "the record is not glued onto the fragment");
        in.close();
        std::remove(frag.c_str());
    }
    // the eligibility rule itself (one rule for both languages)
    {
        SessionEligibilityInputs good;
        good.completed = true;
        good.dealer_mode = "ISOLATED_RECIPIENT_BUNDLES";
        good.dealer_eligible = true;
        good.party_rng_from_os = true;
        SessionEligibility e = session_eligibility(good);
        CHECK(e.eligible && e.status == "MEASURED_NATIVE" && e.notes.empty() && e.dealer_mode == "ISOLATED_RECIPIENT_BUNDLES",
              "the secure configuration, completed: eligible");
        CHECK(!session_eligibility(SessionEligibilityInputs()).eligible, "default inputs are not eligible");
        auto one = [&](void (*edit)(SessionEligibilityInputs&), const char* note, const char* status) {
            SessionEligibilityInputs in = good;
            edit(in);
            SessionEligibility r = session_eligibility(in);
            return !r.eligible && r.status == status && std::find(r.notes.begin(), r.notes.end(), note) != r.notes.end();
        };
        CHECK(one([](SessionEligibilityInputs& i) { i.party_rng_from_os = false; }, "PARTY_RNG_SEEDED", "DIAGNOSTIC"), "seeded party rng");
        CHECK(one([](SessionEligibilityInputs& i) { i.in_process_dealer = true; }, "DEALER_IN_PROCESS", "DIAGNOSTIC"), "in-process dealer");
        CHECK(one([](SessionEligibilityInputs& i) { i.dealer_eligible = false; }, "DEALER_NOT_ELIGIBLE", "DIAGNOSTIC"), "dealer without the declaration");
        CHECK(one([](SessionEligibilityInputs& i) { i.dealer_insecure_seen = true; i.dealer_eligible = false; }, "DEALER_NOT_ELIGIBLE", "DIAGNOSTIC"),
              "insecure material ingested");
        CHECK(one([](SessionEligibilityInputs& i) { i.dealer_mode = "INSECURE_TEST_ONLY_SHARED_SEED"; }, "DEALER_MODE_NOT_ISOLATED", "DIAGNOSTIC"),
              "insecure mode");
        // run conditions leave the status of a secure configuration at MEASURED_NATIVE
        CHECK(one([](SessionEligibilityInputs& i) { i.completed = false; i.cancelled = true; }, "RUN_CANCELLED", "MEASURED_NATIVE"), "cancelled");
        CHECK(one([](SessionEligibilityInputs& i) { i.completed = false; i.failed = true; i.aborts = 1; }, "RUN_FAILED", "MEASURED_NATIVE"), "failed");
        CHECK(one([](SessionEligibilityInputs& i) { i.aborts = 1; }, "ABORTS", "MEASURED_NATIVE"), "aborts");
        CHECK(one([](SessionEligibilityInputs& i) { i.diagnostic_reveals = 1; }, "DIAGNOSTIC_REVEALS", "MEASURED_NATIVE"), "reveals");
        SessionEligibilityInputs seen = good;
        seen.dealer_insecure_seen = true;
        CHECK(session_eligibility(seen).dealer_mode == "INSECURE_TEST_ONLY_SHARED_SEED", "the record never says ISOLATED after insecure material");
    }
    // An OS keyed dealer and OS keyed party randomness, still behind the in-process link: the only
    // unmet condition is the missing process isolation, and the run stays DIAGNOSTIC.
    {
        SessionConfig cfg = base;
        cfg.policy = "B0_fixed:" + fixed.key();
        cfg.run_id = "run-os-rng";
        cfg.session_id = "sess-os-rng";
        cfg.party_rng_from_os = true;
        Harness h;
        CHECK_OK(h.init(cfg, "osrng", true), "harness with an OS keyed dealer");
        Expected<Value> out[2];
        CHECK_OK(h.run(cfg, cfg, cm, W, reference, "", out), "run with OS randomness");
        for (int p = 0; p < 2; ++p) {
            CHECK_OK(out[p], "party record (OS randomness)");
            const Value& r = out[p].value;
            CHECK(h.client[p]->security_eligible() && !h.client[p]->insecure_material_seen(), "the client saw only eligible bundles");
            CHECK(r.get_string("dealer_mode").value == "ISOLATED_RECIPIENT_BUNDLES" && r.get_string("party_rng").value == "os_csprng",
                  "isolated mode and OS party randomness recorded");
            const List& notes = *r.get("eligibility_notes")->as_list().value;
            CHECK(notes.size() == 1 && notes[0].as_string().value == "DEALER_IN_PROCESS", "the in-process dealer is the only unmet condition");
            CHECK(!r.get("security_eligible")->as_bool().value && r.get_string("status").value == "DIAGNOSTIC",
                  "an in-process dealer is never eligible");
            if (p == 1) CHECK(r.get("output_match")->as_bool().value, "tokens match the reference under OS randomness");
        }
        h.cleanup(cfg);
    }
    // a seeded dealer in ISOLATED mode labels its bundles insecure (dealer.hpp)
    {
        DealerConfig dc;
        dc.numeric_contract_hash = contract_hash;
        Dealer seeded(dc, ll::Rng::from_seed(7), Bytes(32, 1), Bytes(32, 2));
        Dealer keyed(dc, ll::Rng::from_os_or_abort(), Bytes(32, 1), Bytes(32, 2));
        CHECK(seeded.eligible_mode() && !seeded.os_seeded() && seeded.insecure(), "seeded dealer: insecure whatever the mode");
        CHECK(keyed.os_seeded() && !keyed.insecure(), "OS keyed isolated dealer: eligible");
        auto bs = seeded.make_beaver(4), bk = keyed.make_beaver(4);
        CHECK(bs.ok && bk.ok, "bundles");
        CHECK(bs.value.for0.header.get("insecure")->as_bool().value && !bk.value.for0.header.get("insecure")->as_bool().value,
              "the header label follows the randomness, not the mode alone");
        CHECK(has_prefix(keyed.make_vector_share("v", {1, 2, 3}).code, "DEALER_PLAINTEXT_REFUSED"), "an eligible dealer refuses a plaintext vector");
        CHECK(seeded.make_vector_share("v", {1, 2, 3}).ok, "the insecure harness dealer still serves it");
        DealerConfig di = dc;
        di.mode = "INSECURE_TEST_ONLY_SHARED_SEED";
        Dealer ins(di, ll::Rng::from_os_or_abort(), Bytes(32, 1), Bytes(32, 2));
        CHECK(ins.insecure(), "the insecure mode is never upgraded by OS randomness");
    }
    std::printf("cancel after epoch 1, refused resumes, warm resume\n");
    if (test_cancel_resume(W, base, reference, cm, evidence, false)) return 1;
    std::printf("cancel after epoch 0, refused resumes, cold resume\n");
    if (test_cancel_resume(W, base, reference, cm, evidence, true)) return 1;
    std::printf("fail-closed paths\n");
    if (test_session_refusals(W, base, reference, cm, evidence)) return 1;
    std::remove(evidence.c_str());
    reset_formula_constants();
    std::printf("PASS %d checks\n", g_checks);
    return 0;
}
