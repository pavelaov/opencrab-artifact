// Implementation of session.hpp: the epoch loop (hello, readiness, cancellation at epoch
// boundaries, resumption from the public state), the public state file and the evidence record.
#include <opencrab2/highlevel/session.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <mutex>
#include <set>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#endif

namespace opencrab2 {
namespace highlevel {

// ---------------------------------------------------------------------------
// trace
// ---------------------------------------------------------------------------
Value SessionTrace::to_value() const {
    List pts;
    for (const auto& p : points)
        pts.push_back(Value(List{Value(p.elapsed_s), Value(p.state.bandwidth_bps), Value(p.state.one_way_delay_s)}));
    return dict({{"trace_id", trace_id}, {"family", family}, {"split", split}, {"points", Value(pts)}});
}

std::string SessionTrace::trace_hash() const {
    auto cj = canonical_json(to_value());
    return cj.ok ? sha256_hex(cj.value) : std::string("unhashable");
}

SessionTrace SessionTrace::stable(const std::string& id, double bandwidth_bps, double one_way_delay_s) {
    SessionTrace t;
    t.trace_id = id;
    t.family = "S0";
    t.points.push_back(TracePoint{0.0, LinkState{bandwidth_bps, one_way_delay_s}});
    return t;
}

// ---------------------------------------------------------------------------
// evidence helpers
// ---------------------------------------------------------------------------
const char* const kEvidenceSchemaId = "opencrab-native-evidence-v2";

SessionEligibility session_eligibility(const SessionEligibilityInputs& in) {
    SessionEligibility out;
    const bool mode_isolated = in.dealer_mode == "ISOLATED_RECIPIENT_BUNDLES";
    // run conditions: the configuration may still be the secure one
    if (in.failed) out.notes.push_back("RUN_FAILED");
    if (in.cancelled) out.notes.push_back("RUN_CANCELLED");
    if (!in.completed) out.notes.push_back("RUN_NOT_COMPLETED");
    if (in.aborts != 0) out.notes.push_back("ABORTS");
    if (in.diagnostic_reveals != 0) out.notes.push_back("DIAGNOSTIC_REVEALS");
    // configuration conditions: any of them makes the record DIAGNOSTIC
    bool config_ok = true;
    if (!mode_isolated) { out.notes.push_back("DEALER_MODE_NOT_ISOLATED"); config_ok = false; }
    if (in.dealer_insecure_seen || (in.completed && !in.dealer_eligible)) {
        // insecure material was ingested, or a completed run cannot show one eligible bundle
        out.notes.push_back("DEALER_NOT_ELIGIBLE");
        config_ok = false;
    } else if (!in.dealer_eligible) {
        out.notes.push_back("DEALER_NOT_ELIGIBLE");   // nothing ingested yet: not eligible, not a config defect
    }
    if (!in.party_rng_from_os) { out.notes.push_back("PARTY_RNG_SEEDED"); config_ok = false; }
    if (in.in_process_dealer) { out.notes.push_back("DEALER_IN_PROCESS"); config_ok = false; }
    out.eligible = out.notes.empty();
    out.status = config_ok ? "MEASURED_NATIVE" : "DIAGNOSTIC";
    out.dealer_mode = (mode_isolated && !in.dealer_insecure_seen) ? "ISOLATED_RECIPIENT_BUNDLES"
                                                                   : (mode_isolated ? "INSECURE_TEST_ONLY_SHARED_SEED" : in.dealer_mode);
    return out;
}

const std::vector<std::string>& evidence_optional_fields() {
    static const std::vector<std::string> f = {"record_kind", "seed", "family", "variant", "split", "T", "n_layers",
                                               "created_utc", "host", "label", "extra"};
    return f;
}

// The file shape of a record: schema fields flat, tokens as a count, the rest under extra.
Value conform_evidence_record(const Value& record) {
    const Dict* d = std::get_if<Dict>(&record.v);
    if (!d) return record;
    const auto& req = evidence_record_fields();
    const auto& opt = evidence_optional_fields();
    auto in = [](const std::vector<std::string>& l, const std::string& k) {
        return std::find(l.begin(), l.end(), k) != l.end();
    };
    Value out = dict({});
    Value extra = dict({});
    if (const Value* e = record.get("extra")) {
        if (const Dict* ed = std::get_if<Dict>(&e->v))
            for (const auto& kv : *ed) extra.set(kv.key, kv.val);
    }
    for (const auto& kv : *d) {
        if (kv.key == "extra") continue;
        if (kv.key == "tokens") {
            if (const List* l = std::get_if<List>(&kv.val.v)) {
                out.set("tokens", static_cast<int64_t>(l->size()));
                extra.set("tokens_list", kv.val);
            } else {
                out.set("tokens", kv.val);
            }
            continue;
        }
        if (in(req, kv.key) || in(opt, kv.key)) out.set(kv.key, kv.val);
        else extra.set(kv.key, kv.val);
    }
    out.set("schema", kEvidenceSchemaId);
    extra.set("language", "cpp");
    out.set("extra", extra);
    return out;
}

const std::vector<std::string>& evidence_record_fields() {
    static const std::vector<std::string> f = {
        "schema", "status", "security_eligible", "code_commit", "profile_id", "numeric_contract_hash", "dealer_mode",
        "diagnostic_reveals", "workload_scope", "model_revision", "trace_id", "trace_hash", "clock_basis", "policy_id",
        "initial_state_hash", "run_id", "repeat", "initializations", "post_init_switches", "rejected_candidates",
        "aborts", "T_cold_s", "T_warm_s", "T_online_s", "bytes", "rounds", "material_generated", "material_transferred",
        "material_consumed", "material_burned", "peak_rss_client", "peak_rss_server", "tokens", "output_match",
        "raw_measurement_path"};
    return f;
}

Expected<Unit> validate_evidence_record(const Value& record) {
    if (!record.is_dict()) return fail("SCHEMA: record must be a dict");
    for (const auto& k : evidence_record_fields())
        if (!record.has(k)) return fail("SCHEMA_MISSING: " + k);
    static const std::set<std::string> statuses = {"MEASURED_NATIVE", "EMULATED_NUMERIC", "MODELED", "EXTERNAL_REPORTED",
                                                   "DIAGNOSTIC", "HISTORICAL"};
    OC2_TRY(schema_id, record.get_string("schema"));
    if (schema_id != kEvidenceSchemaId) return fail("SCHEMA_BAD_VALUE: schema must be " + std::string(kEvidenceSchemaId));
    const bool file_shape = record.has("extra");
    if (file_shape) {
        if (!record.get("extra")->is_dict()) return fail("SCHEMA_TYPE: extra must be a dict");
        const auto& req = evidence_record_fields();
        const auto& opt = evidence_optional_fields();
        for (const auto& e : *std::get_if<Dict>(&record.v)) {
            if (std::find(req.begin(), req.end(), e.key) == req.end() && std::find(opt.begin(), opt.end(), e.key) == opt.end())
                return fail("SCHEMA_UNKNOWN_FIELD: " + e.key);
        }
    }
    OC2_TRY(st, record.get_string("status"));
    if (!statuses.count(st)) return fail("SCHEMA_STATUS: " + st);
    if (!record.get("security_eligible")->is_bool()) return fail("SCHEMA_TYPE: security_eligible must be bool");
    for (const char* k : {"diagnostic_reveals", "repeat", "initializations", "post_init_switches", "rejected_candidates",
                          "aborts", "bytes", "rounds", "material_generated", "material_transferred", "material_consumed",
                          "material_burned"})
        if (!record.get(k)->is_int()) return fail(std::string("SCHEMA_TYPE: ") + k + " must be int");
    for (const char* k : {"diagnostic_reveals", "repeat", "initializations", "post_init_switches", "rejected_candidates",
                          "aborts", "bytes", "rounds", "material_generated", "material_transferred", "material_consumed",
                          "material_burned"})
        if (record.get_int(k).value < 0) return fail(std::string("SCHEMA_TYPE: ") + k + " must be non-negative");
    for (const char* k : {"T_cold_s", "T_warm_s", "T_online_s"}) {
        const Value* v = record.get(k);
        if (!(v->is_double() || v->is_int() || v->is_none())) return fail(std::string("SCHEMA_TYPE: ") + k + " must be a number or null");
    }
    OC2_TRY(dealer_mode, record.get_string("dealer_mode"));
    if (dealer_mode != "ISOLATED_RECIPIENT_BUNDLES" && dealer_mode != "INSECURE_TEST_ONLY_SHARED_SEED" && dealer_mode != "NONE")
        return fail("SCHEMA_BAD_VALUE: dealer_mode " + dealer_mode);
    // The eligibility rule of the schema (the same four conditions as schema.validate).
    if (record.get("security_eligible")->as_bool().value) {
        if (dealer_mode != "ISOLATED_RECIPIENT_BUNDLES")
            return fail("SCHEMA_ELIGIBILITY: security_eligible requires dealer_mode ISOLATED_RECIPIENT_BUNDLES");
        if (record.get_int("diagnostic_reveals").value != 0)
            return fail("SCHEMA_ELIGIBILITY: security_eligible requires diagnostic_reveals == 0");
        if (st != "MEASURED_NATIVE") return fail("SCHEMA_ELIGIBILITY: security_eligible requires status MEASURED_NATIVE");
        if (record.get_int("aborts").value != 0) return fail("SCHEMA_ELIGIBILITY: security_eligible requires aborts == 0");
    }
    OC2_TRY(cb, record.get_string("clock_basis"));
    if (cb != "wall-clock-exogenous") return fail("SCHEMA_CLOCK_BASIS");
    OC2_TRY(ws, record.get_string("workload_scope"));
    if (ws != "full-model" && ws != "prefix-4L" && ws != "prefix-1L" && ws != "primitive" && ws != "web")
        return fail("SCHEMA_WORKLOAD_SCOPE: " + ws);
    const Value* tk = record.get("tokens");
    if (file_shape) {
        if (!tk->is_int() || tk->as_int().value < 0) return fail("SCHEMA_TYPE: tokens must be a non-negative count in the file shape");
    } else if (!(tk->is_list() || tk->is_int())) {
        return fail("SCHEMA_TYPE: tokens must be a list or a count");
    }
    const Value* om = record.get("output_match");
    if (!(om->is_bool() || om->is_none())) return fail("SCHEMA_TYPE: output_match must be bool or null");
    OC2_TRY(ch, record.get_string("numeric_contract_hash"));
    if (ch.size() != 64) return fail("SCHEMA_TYPE: numeric_contract_hash must be a sha256 hex");
    return ok_unit();
}

Expected<Unit> append_jsonl(const std::string& path, const Value& record) {
    // One unbuffered write per line under a process-wide lock: two parties in
    // one process (tests) or two processes appending to the same file never
    // interleave bytes of a record.
    static std::mutex mu;
    std::lock_guard<std::mutex> g(mu);
    OC2_TRY(text, to_json(record));
    // A file that does not end with a newline holds the fragment of an append that died: start on
    // a new line so that this record is never glued onto it.
    if (std::FILE* rf = std::fopen(path.c_str(), "rb")) {
        if (std::fseek(rf, -1, SEEK_END) == 0) {
            const int last = std::fgetc(rf);
            if (last != EOF && last != '\n') text.insert(text.begin(), '\n');
        }
        std::fclose(rf);
    }
    std::FILE* f = std::fopen(path.c_str(), "ab");
    if (!f) return fail("EVIDENCE_IO: cannot open " + path);
    std::setvbuf(f, nullptr, _IONBF, 0);
    text.push_back('\n');
    size_t n = std::fwrite(text.data(), 1, text.size(), f);
    std::fflush(f);
#ifdef _WIN32
    _commit(_fileno(f));
#else
    fsync(fileno(f));
#endif
    std::fclose(f);
    if (n != text.size()) return fail("EVIDENCE_IO: short write");
    return ok_unit();
}

uint64_t peak_rss_bytes() {
#ifdef _WIN32
    struct Counters {
        DWORD cb;
        DWORD PageFaultCount;
        SIZE_T PeakWorkingSetSize;
        SIZE_T WorkingSetSize;
        SIZE_T QuotaPeakPagedPoolUsage;
        SIZE_T QuotaPagedPoolUsage;
        SIZE_T QuotaPeakNonPagedPoolUsage;
        SIZE_T QuotaNonPagedPoolUsage;
        SIZE_T PagefileUsage;
        SIZE_T PeakPagefileUsage;
    };
    typedef BOOL(WINAPI * Fn)(HANDLE, Counters*, DWORD);
    static Fn fn = []() -> Fn {
        HMODULE k = GetModuleHandleW(L"kernel32.dll");
        if (!k) return nullptr;
        return reinterpret_cast<Fn>(reinterpret_cast<void*>(GetProcAddress(k, "K32GetProcessMemoryInfo")));
    }();
    if (!fn) return 0;
    Counters c;
    c.cb = sizeof(c);
    if (!fn(GetCurrentProcess(), &c, sizeof(c))) return 0;
    return static_cast<uint64_t>(c.PeakWorkingSetSize);
#else
    std::ifstream in("/proc/self/status");
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("VmHWM:", 0) == 0) {
            unsigned long long kb = 0;
            if (std::sscanf(line.c_str() + 6, "%llu", &kb) != 1) return 0;
            return static_cast<uint64_t>(kb) * 1024;
        }
    }
    return 0;
#endif
}

// ---------------------------------------------------------------------------
// public session state (resumption)
// ---------------------------------------------------------------------------
namespace {

const char* const kStateSchema = "opencrab2-session-state-v1";

// Atomic replace: the state is written to <path>.tmp, flushed to disk and renamed over the
// previous file, so a crash leaves either the old or the new state, never a torn one.
Expected<Unit> write_state_file(const std::string& path, const Value& state) {
    OC2_TRY(text, to_json(state));
    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return fail("SESSION_STATE_IO: cannot open " + tmp);
    text.push_back('\n');
    const size_t n = std::fwrite(text.data(), 1, text.size(), f);
    std::fflush(f);
#ifdef _WIN32
    _commit(_fileno(f));
#else
    fsync(fileno(f));
#endif
    std::fclose(f);
    if (n != text.size()) return fail("SESSION_STATE_IO: short write");
#ifdef _WIN32
    if (!MoveFileExA(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        return fail("SESSION_STATE_IO: cannot replace " + path);
#else
    if (std::rename(tmp.c_str(), path.c_str()) != 0) return fail("SESSION_STATE_IO: cannot replace " + path);
#endif
    return ok_unit();
}

}  // namespace

std::string session_state_path(const SessionConfig& cfg, const MaterialLedger& ledger) {
    return cfg.state_path.empty() ? ledger.journal_path() + ".state.json" : cfg.state_path;
}

Expected<Value> read_session_state(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return fail("SESSION_STATE_MISSING: no public state at " + path);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    auto parsed = parse_json(text);
    if (!parsed.ok || !parsed.value.is_dict()) return fail("SESSION_STATE_CORRUPT: " + path + " is not a JSON object");
    auto schema = parsed.value.get_string("schema");
    if (!schema.ok || schema.value != kStateSchema) return fail("SESSION_STATE_CORRUPT: " + path + " does not carry " + kStateSchema);
    for (const char* k : {"session_id", "model_version", "numeric_contract_hash", "profile_id"})
        if (!parsed.value.get_string(k).ok) return fail(std::string("SESSION_STATE_CORRUPT: missing ") + k);
    for (const char* k : {"epochs_done", "n_generate", "prompt_len", "generation"})
        if (!parsed.value.get(k) || !parsed.value.get(k)->is_int()) return fail(std::string("SESSION_STATE_CORRUPT: missing ") + k);
    return ok(std::move(parsed.value));
}

// ---------------------------------------------------------------------------
// session loop
// ---------------------------------------------------------------------------
namespace {

Json version_json(const SessionConfig& cfg, const std::string& contract_hash) {
    Json v = Json::object();
    v.set("model_version", cfg.model_version);
    v.set("contract_hash", contract_hash);
    return v;
}

std::vector<std::string> installed_ids(Party& party, TinyDecoder& dec) {
    std::vector<std::string> out;
    TinyPlan all;
    all.embedding = "wmask";
    all.linear = "wmask";
    for (const auto& o : dec.required_objects(all))
        if (party.installed(o)) out.push_back(o);
    return out;
}

// Installed objects WITH their versions ({object: {model_version, client_key_domain}}): only the
// objects that are installed under the party's current version are listed.
Value installed_versions(Party& party, TinyDecoder& dec) {
    Value all = party.installed_objects();
    Value out = dict({});
    for (const auto& o : installed_ids(party, dec)) {
        const Value* v = all.get(o);
        out.set(o, v ? *v : party.version());
    }
    return out;
}

Value plan_value(const TinyPlan& p) {
    return dict({{"embedding", p.embedding}, {"linear", p.linear}, {"argmax_k", static_cast<int64_t>(p.argmax_k)},
                 {"max_k", static_cast<int64_t>(p.max_k)}});
}

Value decision_value(const Decision& d) {
    int64_t infeasible = 0;
    List rejected;
    for (const Json& rec : d.candidates) {
        if (rec.has("feasible") && !rec.get("feasible").as_bool()) {
            infeasible += 1;
            rejected.push_back(dict({{"plan", json_to_value(rec.get("plan"))}, {"reason", rec.get("reason").as_string()}}));
        }
    }
    return dict({{"policy", d.policy},
                 {"chosen", json_to_value(d.chosen)},
                 {"switched", d.switched},
                 {"reason", d.reason},
                 {"q_keep", d.q_keep},
                 {"q_best", d.q_best},
                 {"lower_bound_gain", d.lower_bound_gain},
                 {"u_diff", d.u_diff},
                 {"n_candidates", static_cast<int64_t>(d.candidates.size())},
                 {"n_infeasible", infeasible},
                 {"rejected", Value(rejected)}});
}

std::string code_head(const std::string& code) {
    const std::string::size_type p = code.find(':');
    return p == std::string::npos ? code : code.substr(0, p);
}

// The public binding both parties must agree on (commit digest, hello).
Value binding_value(const SessionConfig& cfg, const std::string& contract_hash) {
    return dict({{"session_id", cfg.session_id},
                 {"setup_id", cfg.setup_id},
                 {"model_version", cfg.model_version},
                 {"numeric_contract_hash", contract_hash},
                 {"profile_id", cfg.profile_id}});
}

// Compares the ledger binding with the declared session binding (only the keys the ledger
// carries). A model_version difference is the versioning failure OBJECT_VERSION_MISMATCH.
std::string ledger_binding_error(const MaterialLedger& ledger, const SessionConfig& cfg, const std::string& contract_hash) {
    const Value& b = ledger.binding();
    auto differs = [&](const char* key, const std::string& mine) {
        const Value* v = b.get(key);
        if (!v) return false;
        auto s = v->as_string();
        return !s.ok || s.value != mine;
    };
    if (differs("model_version", cfg.model_version)) return "OBJECT_VERSION_MISMATCH: the ledger is bound to another model_version";
    if (differs("numeric_contract_hash", contract_hash)) return "CONTRACT_MISMATCH: the ledger is bound to another numeric contract";
    if (differs("profile_id", cfg.profile_id)) return "PROFILE_MISMATCH: the ledger is bound to another profile";
    if (differs("session_id", cfg.session_id) || differs("setup_id", cfg.setup_id))
        return "SESSION_BINDING_MISMATCH: the ledger is bound to another session or setup";
    return "";
}

}  // namespace

Expected<Value> run_session_party(int party, const SessionConfig& cfg, Channel& channel, DealerClient& dealer,
                                  MaterialLedger& ledger, const FittedCostModel& costmodel, const SessionTrace& trace,
                                  const TinyWeights* weights_or_null, const std::vector<int>* reference_tokens,
                                  const std::string& out_path, InProcessDealerLink* link) {
    // ---- public parameters (identical checks on both parties, before any message) ----------
    if (party != 0 && party != 1) return fail("SESSION_PARTY");
    if (cfg.prompt_ids.empty() && party == 1) return fail("SESSION_PROMPT: P1 needs a prompt");
    if (cfg.n_generate < 1) return fail("SESSION_EPOCHS: n_generate must be >= 1");
    if (cfg.telemetry_key.size() < 16) return fail("TELEMETRY_KEY: key must be at least 16 bytes");
    if (cfg.probe_samples < 1 || cfg.probe_samples > PROBE_MAX_SAMPLES)
        return fail("TELEMETRY_PARAMS: probe_samples must be in [1, " + std::to_string(PROBE_MAX_SAMPLES) + "]");
    if (cfg.probe_burst < 1 || cfg.probe_burst > PROBE_MAX_BURST_BYTES) return fail("TELEMETRY_PARAMS: probe_burst must be in [1, 2^28] bytes");
    if (cfg.cancel_after_epoch < -1) return fail("SESSION_PARAMS: cancel_after_epoch must be >= -1");
    const std::string contract_hash = cfg.contract.contract_hash();
    const std::string state_path = session_state_path(cfg, ledger);
    const double t_start = now_s();
    uint64_t bytes0 = channel.stats.bytes_sent + channel.stats.bytes_recv;
    uint64_t rounds0 = channel.stats.rounds;
    double gen0 = link ? link->generation_seconds() : 0.0;

    // ---- local durable state against the declared binding ----------------------------------
    // Nothing is sent yet: the outcome travels in the readiness exchange below, so that both
    // parties stop at the same protocol step with a stable code.
    std::string local_error = ledger_binding_error(ledger, cfg, contract_hash);
    int64_t state_epochs_done = 0, state_generation = -1, state_prompt_len = -1;
    std::vector<int> prior_tokens;      // P1: the tokens released before this run
    std::vector<int> known_tokens;      // P1: every token its state recorded (may reach beyond the resume epoch)
    TinyPlan plan = cfg.initial_plan;
    if (cfg.resume) {
        auto st = read_session_state(state_path);
        if (!st.ok) {
            if (local_error.empty()) local_error = st.code;
        } else {
            const Value& s = st.value;
            const std::string s_session = s.get_string("session_id").value, s_model = s.get_string("model_version").value;
            if (local_error.empty() && s_session != cfg.session_id)
                local_error = "SESSION_BINDING_MISMATCH: the public state belongs to another session_id";
            if (local_error.empty() && s_model != cfg.model_version)
                local_error = "OBJECT_VERSION_MISMATCH: the resumed session was bound to another model_version";
            if (local_error.empty() && s.get_string("numeric_contract_hash").value != contract_hash)
                local_error = "CONTRACT_MISMATCH: the resumed session used another numeric contract";
            if (local_error.empty() && s.get_string("profile_id").value != cfg.profile_id)
                local_error = "PROFILE_MISMATCH: the resumed session used another profile";
            if (local_error.empty() && s.get_int("n_generate").value != cfg.n_generate)
                local_error = "SESSION_BINDING_MISMATCH: the resumed session declared another token budget";
            const Value* completed_v = s.get("completed");
            if (local_error.empty() && completed_v && completed_v->is_bool() && completed_v->as_bool().value)
                local_error = "SESSION_COMPLETED: nothing left to resume";
            state_epochs_done = s.get_int("epochs_done").value;
            state_generation = s.get_int("generation").value;
            state_prompt_len = s.get_int("prompt_len").value;
            if (local_error.empty() && (state_epochs_done < 0 || state_epochs_done >= cfg.n_generate || state_generation < 0))
                local_error = "SESSION_STATE_CORRUPT: epoch or generation out of range";
            if (local_error.empty() && party == 1 && state_prompt_len != static_cast<int64_t>(cfg.prompt_ids.size()))
                local_error = "SESSION_BINDING_MISMATCH: the resumed session declared another prompt length";
            if (local_error.empty()) {
                if (const Value* pv = s.get("plan")) {
                    auto pr = TinyPlan::from_json(value_to_json(*pv));
                    if (pr.ok) plan = pr.value;
                    else local_error = "SESSION_STATE_CORRUPT: plan";
                }
            }
            if (local_error.empty() && party == 1) {
                const Value* tv = s.get("generated_tokens");
                if (!tv || !tv->is_list()) {
                    local_error = "SESSION_STATE_CORRUPT: P1 state lacks its released tokens";
                } else {
                    for (const Value& t : *tv->as_list().value) {
                        auto ti = t.as_int();
                        if (!ti.ok || ti.value < 0 || static_cast<uint64_t>(ti.value) >= cfg.tiny.vocab) {
                            local_error = "SESSION_STATE_CORRUPT: token outside the vocabulary";
                            break;
                        }
                        prior_tokens.push_back(static_cast<int>(ti.value));
                    }
                    if (local_error.empty() && static_cast<int64_t>(prior_tokens.size()) < state_epochs_done)
                        local_error = "SESSION_STATE_CORRUPT: fewer tokens than epochs";
                }
            }
        }
    }

    Value record = dict({});
    List epochs;
    List tokens_v;
    std::vector<int> tokens;
    int64_t initializations = 0, post_init_switches = 0, rejected = 0, aborts = 0;
    double T_online = 0.0, setup_s = 0.0, sensing_s = 0.0, planning_s = 0.0, install_s = 0.0;
    bool completed = false, cancelled = false, resumed = false;
    int64_t cancel_epoch = -1, start_epoch = 0, generation = 0, epochs_done = 0;
    std::string failure;
    Value burn_report = Value(None{});
    size_t prompt_len = party == 1 ? cfg.prompt_ids.size() : 0;

    // ---- hello: the public binding ------------------------------------------------------------
    std::string tiny_digest;
    {
        auto cj = canonical_json(cfg.tiny.to_value());
        tiny_digest = cj.ok ? sha256_hex(cj.value) : "unhashable";
    }
    const int64_t my_generation = cfg.resume ? state_generation + 1 : 0;
    {
        Value hello = dict({{"party", static_cast<int64_t>(party)},
                            {"session_id", cfg.session_id},
                            {"setup_id", cfg.setup_id},
                            {"model_version", cfg.model_version},
                            {"numeric_contract_hash", contract_hash},
                            {"profile_id", cfg.profile_id},
                            {"n_generate", static_cast<int64_t>(cfg.n_generate)},
                            {"policy", cfg.policy},
                            {"model_shape", tiny_digest},
                            {"resume", cfg.resume},
                            {"resume_epoch", state_epochs_done},
                            {"generation", my_generation},
                            {"plan", plan_value(plan)},
                            {"prompt_len", party == 1 ? static_cast<int64_t>(cfg.prompt_ids.size()) : static_cast<int64_t>(-1)}});
        auto ex = channel.exchange(hello);
        if (!ex.ok) {
            failure = ex.code;
        } else {
            const Value& peer = ex.value;
            auto peer_str = [&](const char* k) {
                auto s = peer.get_string(k);
                return s.ok ? s.value : std::string(1, static_cast<char>(1)) + "missing";
            };
            auto peer_int = [&](const char* k, int64_t missing) {
                const Value* v = peer.get(k);
                return v && v->is_int() ? v->as_int().value : missing;
            };
            const Value* pr = peer.get("resume");
            if (!peer.is_dict() || peer_int("party", -1) != 1 - party) failure = "SESSION_HANDSHAKE: malformed hello from the peer";
            else if (peer_str("model_version") != cfg.model_version)
                failure = "OBJECT_VERSION_MISMATCH: the peer declares another model_version";
            else if (peer_str("numeric_contract_hash") != contract_hash) failure = "CONTRACT_MISMATCH: the peer declares another numeric contract";
            else if (peer_str("profile_id") != cfg.profile_id) failure = "PROFILE_MISMATCH: the peer declares another profile";
            else if (peer_str("session_id") != cfg.session_id || peer_str("setup_id") != cfg.setup_id)
                failure = "SESSION_BINDING_MISMATCH: session_id or setup_id differs";
            else if (peer_int("n_generate", -1) != cfg.n_generate) failure = "SESSION_BINDING_MISMATCH: token budget differs";
            else if (peer_str("policy") != cfg.policy) failure = "SESSION_BINDING_MISMATCH: policy differs";
            else if (peer_str("model_shape") != tiny_digest) failure = "SESSION_BINDING_MISMATCH: public model shape differs";
            else if (!pr || !pr->is_bool() || pr->as_bool().value != cfg.resume)
                failure = "SESSION_BINDING_MISMATCH: one party resumes, the other starts";
            else {
                if (party == 0) {
                    const int64_t pl = peer_int("prompt_len", -1);
                    if (pl < 1) failure = "SESSION_PROMPT: zero prompt length";
                    else if (cfg.resume && local_error.empty() && state_prompt_len != pl)
                        local_error = "SESSION_BINDING_MISMATCH: the resumed session declared another prompt length";
                    else prompt_len = static_cast<size_t>(pl);
                }
                // The incumbent plan is public state both parties must share. Fresh start: the declared
                // initial plans must be equal. Resume: the parties continue at the smaller recorded
                // epoch, so the plan in force is the one recorded by the party that is behind (its
                // state was written right before that epoch); at equal epochs the records must agree.
                const int64_t peer_epoch = std::max<int64_t>(peer_int("resume_epoch", 0), 0);
                TinyPlan peer_plan;
                bool have_peer_plan = false;
                if (const Value* pp = peer.get("plan")) {
                    auto ppr = TinyPlan::from_json(value_to_json(*pp));
                    if (ppr.ok) {
                        peer_plan = ppr.value;
                        have_peer_plan = true;
                    }
                }
                if (failure.empty() && !have_peer_plan) failure = "SESSION_HANDSHAKE: malformed plan in the hello of the peer";
                if (failure.empty()) {
                    const bool same = peer_plan.key() == plan.key();
                    if (!cfg.resume || peer_epoch == state_epochs_done) {
                        if (!same && local_error.empty()) failure = "SESSION_BINDING_MISMATCH: the incumbent plans differ";
                    } else if (peer_epoch < state_epochs_done) {
                        plan = peer_plan;
                    }
                }
                if (failure.empty() && cfg.resume) {
                    // continue at the smaller recorded epoch, under the larger proposed generation
                    start_epoch = std::min<int64_t>(state_epochs_done, peer_epoch);
                    generation = std::max<int64_t>(my_generation, peer_int("generation", 0));
                }
            }
        }
    }
    // ---- ready: each party's own durable state ---------------------------------------------------
    if (failure.empty()) {
        Value ready = dict({{"ready", local_error.empty()}, {"code", code_head(local_error)}});
        auto ex = channel.exchange(ready);
        if (!local_error.empty()) {
            failure = local_error;   // this party's own refusal is what it reports, whatever the peer did
        } else if (!ex.ok) {
            failure = ex.code;
        } else {
            const Value* pr = ex.value.get("ready");
            if (!pr || !pr->is_bool()) failure = "SESSION_HANDSHAKE: malformed readiness from the peer";
            else if (!pr->as_bool().value) {
                auto pc = ex.value.get_string("code");
                failure = "SESSION_PEER_REFUSED: " + (pc.ok ? code_head(pc.value) : std::string("UNKNOWN"));
            }
        }
    }
    if (failure.empty() && cfg.resume) {
        resumed = true;
        epochs_done = start_epoch;
        if (party == 1) {
            known_tokens = prior_tokens;                             // kept to cross-check what is re-derived
            prior_tokens.resize(static_cast<size_t>(start_epoch));   // a later token is re-derived, never trusted
            tokens = prior_tokens;
            for (int t : tokens) tokens_v.push_back(Value(static_cast<int64_t>(t)));
        }
        // A new generation: nothing pooled before the interruption is used again (the two pools may
        // not be aligned any more), the masks included; new bundles are requested on demand.
        auto br = dealer.burn_outstanding("RESUME", true);
        if (!br.ok) failure = br.code;
        dealer.set_generation(generation);
    }

    // ---- party, decoder, policy -----------------------------------------------------------------
    ReleasePolicy release(1, {"token", "class"}, cfg.n_generate - static_cast<int>(start_epoch));
    ll::Rng rng = ll::Rng::from_seed(static_cast<uint64_t>(cfg.seed) * 7919ULL + static_cast<uint64_t>(party) + 1ULL);
    if (cfg.party_rng_from_os) {
        Status os = ll::Rng::from_os(rng);
        if (!os.ok && failure.empty()) failure = os.code.empty() ? std::string("RNG_OS_UNAVAILABLE") : os.code;
    }
    Party p(party, channel, dealer, cfg.contract, release, rng, cfg.model_version, cfg.client_key_domain);
    TinyDecoder dec(p, weights_or_null, cfg.tiny, plan);
    p.set_gate("setup", -1);
    auto policy_r = make_policy(cfg.policy, cfg.controller);
    if (!policy_r.ok && failure.empty()) failure = policy_r.code;
    TelemetryVerifier verifier(cfg.telemetry_key, start_epoch, cfg.max_bandwidth_bps, 10.0);

    auto public_state = [&]() {
        Value st = dict({{"schema", kStateSchema},
                         {"party", static_cast<int64_t>(party)},
                         {"session_id", cfg.session_id},
                         {"setup_id", cfg.setup_id},
                         {"model_version", cfg.model_version},
                         {"numeric_contract_hash", contract_hash},
                         {"profile_id", cfg.profile_id},
                         {"epochs_done", epochs_done},
                         {"plan", plan_value(plan)},
                         {"installed", installed_versions(p, dec)},
                         {"generated_count", epochs_done},
                         {"prompt_len", static_cast<int64_t>(prompt_len)},
                         {"n_generate", static_cast<int64_t>(cfg.n_generate)},
                         {"generation", generation},
                         {"policy", cfg.policy},
                         {"cancelled", cancelled},
                         {"cancel_epoch", cancel_epoch},
                         {"completed", completed}});
        if (party == 1) st.set("generated_tokens", Value(tokens_v));   // P1's own outputs, kept on P1 only
        return st;
    };

    // ---- setup: server-private vectors, generic weight shares, LM head install (T_cold) ----------
    if (failure.empty()) {
        double t0 = now_s();
        auto st = dec.setup();
        if (!st.ok) failure = st.code;
        else setup_s = now_s() - t0;
    }
    // Initial state hash (after setup): incumbent, installed objects with versions, binding, epoch.
    std::string initial_state_hash;
    {
        Value st = dict({{"incumbent", plan_value(plan)}, {"installed", installed_versions(p, dec)},
                         {"binding", binding_value(cfg, contract_hash)}, {"epoch", start_epoch}, {"generation", generation}});
        auto cj = canonical_json(st);
        initial_state_hash = cj.ok ? sha256_hex(cj.value) : "unhashable";
    }

    int prev_token = (party == 1 && !tokens.empty()) ? tokens.back() : -1;
    bool local_cancel = false;   // raised by cancel_after_epoch at the end of that epoch
    for (int e = static_cast<int>(start_epoch); e < cfg.n_generate && failure.empty(); ++e) {
        // 0. declared cancellation status (public)
        {
            const bool mine = local_cancel || (cfg.cancel_flag && cfg.cancel_flag->load());
            auto ex = channel.exchange(dict({{"cancel", mine}, {"epoch", static_cast<int64_t>(e)}, {"generation", generation}}));
            if (!ex.ok) { failure = ex.code; break; }
            const Value* pc = ex.value.get("cancel");
            const Value* pe = ex.value.get("epoch");
            const Value* pg = ex.value.get("generation");
            if (!pc || !pc->is_bool() || !pe || !pe->is_int() || !pg || !pg->is_int()) { failure = "SESSION_HANDSHAKE: malformed epoch status"; break; }
            if (pe->as_int().value != e || pg->as_int().value != generation) { failure = "SESSION_BINDING_MISMATCH: epoch or generation differs"; break; }
            if (mine || pc->as_bool().value) {
                // The epoch is not started. RESERVED ids and untouched one-time bundles are burned
                // through the ledger (safe waste); the session may be resumed under the same id.
                auto br = dealer.burn_outstanding("CANCEL");
                if (!br.ok) { failure = br.code; break; }
                burn_report = br.value.to_value();
                cancelled = true;
                cancel_epoch = e;
                break;
            }
        }
        const bool recovery = resumed && e == static_cast<int>(start_epoch) && e > 0;   // KV cache is rebuilt
        Value ep = dict({{"epoch", static_cast<int64_t>(e)}});
        std::string gate = "epoch-" + std::to_string(e);
        p.set_gate(gate, e);
        size_t n_rows = (e == 0 || recovery) ? prompt_len + static_cast<size_t>(e) : 1;
        size_t kv_len = prompt_len + static_cast<size_t>(e);
        // 1. sense
        double ts0 = now_s();
        auto fr = sense_epoch(channel, party, e, e, now_s() - t_start, verifier, cfg.telemetry_key, cfg.max_bandwidth_bps,
                              cfg.probe_samples, cfg.probe_burst);
        if (!fr.ok) { failure = fr.code; break; }
        TelemetryFrame frame = fr.value;
        sensing_s += now_s() - ts0;
        Env env = frame.to_env();
        policy_r.value->observe(frame.to_sample());
        // 2. candidates
        double tp0 = now_s();
        if (link) {
            auto dr = link->drain(party, dealer);
            if (!dr.ok) { failure = dr.code; break; }
        }
        SessionState state;
        state.incumbent = plan.to_json();
        for (const auto& o : installed_ids(p, dec)) state.installed[o] = version_json(cfg, contract_hash);
        // stock in the cost-model vocabulary: wmask_mask is the per-call weight-mask material
        for (const char* mt : {"beaver", "kary_select", "drelu", "trunc_faithful", "ring_to_field", "matrix_triple"}) {
            auto s = ledger.stock(mt);
            state.stock[mt] = {static_cast<int64_t>(s.first), static_cast<int64_t>(s.second)};
        }
        {
            auto s = ledger.stock("wmask_call");
            state.stock["wmask_mask"] = {static_cast<int64_t>(s.first), static_cast<int64_t>(s.second)};
        }
        state.kv_bucket = cfg.length_bucket;
        state.remaining_epochs = cfg.n_generate - e;
        state.session_id = cfg.session_id;
        state.model_version = cfg.model_version;
        state.contract_hash = contract_hash;
        Shapes shapes;
        shapes.n_layers = static_cast<int>(cfg.tiny.n_layers);
        shapes.d_model = static_cast<int>(cfg.tiny.d_model);
        shapes.d_ff = static_cast<int>(cfg.tiny.d_ff);
        shapes.n_head = static_cast<int>(cfg.tiny.n_head);
        shapes.vocab = static_cast<int>(cfg.tiny.vocab);
        shapes.n_rows = static_cast<int>(n_rows);
        shapes.kv_len = static_cast<int>(kv_len);
        shapes.refill_bytes_per_s = cfg.refill_bytes_per_s;
        shapes.logits_bound = cfg.logits_bound;
        shapes.score_bound = cfg.score_bound;
        // The horizon is priced step by step: this epoch on `shapes`, every later epoch u as one decode
        // row over its true key count prompt_len + u, the length convention of the current epoch (the
        // tiny decoder executes the true length; the Python session rounds every entry, the current
        // one included, up to its length bucket).
        const std::vector<Shapes> schedule = horizon_schedule(shapes, e, cfg.n_generate, static_cast<int>(prompt_len), 1);
        auto cands = enumerate_candidates(state, Catalog::full(), costmodel, env, shapes, cfg.contract, schedule);
        if (!cands.ok) { failure = cands.code; break; }
        for (const auto& c : cands.value)
            if (!c.feasible) rejected += 1;
        // The incumbent itself must be admitted: a plan that violates the magnitude bound of its
        // k-ary sites is never executed (paper, Section 5.3), not even as "keep".
        if (!cands.value.empty() && !cands.value[0].feasible && cands.value[0].reason == "KARY_MAGNITUDE") {
            failure = cands.value[0].detail.empty() ? std::string("KARY_MAGNITUDE: the incumbent plan is not admitted") : cands.value[0].detail;
            break;
        }
        // 3. decide
        auto dec_r = policy_r.value->decide(state, cands.value, env, e);
        if (!dec_r.ok) { failure = dec_r.code; break; }
        Decision decision = dec_r.value;
        auto chosen_r = TinyPlan::from_json(decision.chosen);
        if (!chosen_r.ok) { failure = chosen_r.code; break; }
        TinyPlan chosen = chosen_r.value;
        {
            // admission of the plan that is about to execute (fixed policies choose without the filter)
            Status ks = kary_feasibility(chosen.to_json(), shapes, cfg.contract);
            if (!ks.ok) { failure = ks.code; break; }
        }
        planning_s += now_s() - tp0;
        // 4. two-phase commit of the decision digest: plan, object versions, epoch, session binding
        {
            List targets;
            for (const auto& o : dec.required_objects(chosen)) targets.push_back(Value(o));
            auto fd = frame.digest();
            Value dg = dict({{"epoch", static_cast<int64_t>(e)},
                             {"generation", generation},
                             {"plan", plan_value(chosen)},
                             {"objects", installed_versions(p, dec)},
                             {"target_objects", Value(targets)},
                             {"object_version", p.version()},
                             {"frame", fd.ok ? fd.value : std::string("")},
                             {"policy", cfg.policy},
                             {"binding", binding_value(cfg, contract_hash)}});
            auto cj = canonical_json(dg);
            if (!cj.ok) { failure = cj.code; break; }
            std::string digest = sha256_hex(cj.value);
            auto ex1 = channel.exchange(Value(digest));
            if (!ex1.ok) { failure = ex1.code; break; }
            auto peer1 = ex1.value.as_string();
            if (!peer1.ok || peer1.value != digest) { aborts += 1; failure = "COMMIT_DIGEST_MISMATCH: phase 1"; break; }
            auto ex2 = channel.exchange(Value("COMMIT:" + digest));
            if (!ex2.ok) { failure = ex2.code; break; }
            auto peer2 = ex2.value.as_string();
            if (!peer2.ok || peer2.value != "COMMIT:" + digest) { aborts += 1; failure = "COMMIT_DIGEST_MISMATCH: phase 2"; break; }
            ep.set("decision_digest", digest);
        }
        // 5. transactional install
        double ti0 = now_s();
        auto inst = dec.install_for_plan(chosen);
        if (!inst.ok) { failure = inst.code; break; }
        install_s += now_s() - ti0;
        int64_t n_inst = inst.value.get_int("count").ok ? inst.value.get_int("count").value : 0;
        bool switched = chosen.key() != plan.key();
        if (e == static_cast<int>(start_epoch)) initializations += n_inst;
        else if (switched) post_init_switches += 1;
        plan = chosen;
        // 6. execute
        double gen_before = link ? link->generation_seconds() : 0.0;
        double tx0 = now_s();
        // epoch e >= 1 feeds the token released at epoch e-1 at position prompt_len + e - 1;
        // the KV cache then holds kv_len = prompt_len + e keys. The recovery gate of a resumed
        // session re-runs the prefill over the prompt plus the tokens released so far.
        Expected<int> tok = Expected<int>::failure("SESSION_INTERNAL");
        if (e == 0) {
            tok = dec.prefill(party == 1 ? &cfg.prompt_ids : nullptr, prompt_len);
        } else if (recovery) {
            std::vector<int> ids;
            if (party == 1) {
                ids = cfg.prompt_ids;
                ids.insert(ids.end(), tokens.begin(), tokens.end());
            }
            tok = dec.prefill(party == 1 ? &ids : nullptr, prompt_len + static_cast<size_t>(e));
        } else {
            tok = dec.decode_step(prev_token, kv_len - 1);
        }
        double exec_wall = now_s() - tx0;
        double gen_delta = link ? link->generation_seconds() - gen_before : 0.0;
        if (!tok.ok) { failure = tok.code; break; }
        T_online += std::max(0.0, exec_wall - gen_delta);
        if (party == 1) {
            // A token this party already held from before the interruption is re-derived, not
            // trusted: the outputs are deterministic, so a different value means the model, the
            // prompt or the state changed underneath the session (fail closed, state not advanced).
            if (static_cast<size_t>(e) < known_tokens.size() && known_tokens[static_cast<size_t>(e)] != tok.value) {
                failure = "RESUME_OUTPUT_MISMATCH: the re-derived token of epoch " + std::to_string(e) + " differs from the recorded one";
                break;
            }
            tokens.push_back(tok.value);
            tokens_v.push_back(Value(static_cast<int64_t>(tok.value)));
            prev_token = tok.value;
            if (cfg.on_token) cfg.on_token(e, tok.value);
        }
        ep.set("n_rows", static_cast<int64_t>(n_rows));
        ep.set("kv_len", static_cast<int64_t>(kv_len));
        ep.set("recovery_gate", recovery);
        ep.set("frame", frame.to_value());
        ep.set("decision", decision_value(decision));
        ep.set("install", inst.value);
        ep.set("switched", switched);
        ep.set("exec_wall_s", exec_wall);
        ep.set("dealer_generation_s", gen_delta);
        ep.set("plan", plan_value(plan));
        ep.set("bytes_total", static_cast<int64_t>(channel.stats.bytes_sent + channel.stats.bytes_recv - bytes0));
        ep.set("rounds_total", static_cast<int64_t>(channel.stats.rounds - rounds0));
        epochs.push_back(ep);
        // 7. public state after the epoch
        epochs_done = e + 1;
        if (e + 1 == cfg.n_generate) completed = true;
        auto ws = write_state_file(state_path, public_state());
        if (!ws.ok) { failure = ws.code; break; }
        if (cfg.cancel_after_epoch == e) local_cancel = true;   // raised at the next boundary
    }
    if (!failure.empty() && aborts == 0) aborts += 1;
    if (!failure.empty()) {
        // Uncertainty (paper, Section 4.3): after a failure the two pools may not be aligned any more. Whatever is
        // RESERVED or untouched is burned through the ledger; the original code is what is reported.
        auto br = dealer.burn_outstanding("ABORT");
        if (br.ok) burn_report = br.value.to_value();
    }
    if (cancelled) {
        auto ws = write_state_file(state_path, public_state());
        if (!ws.ok && failure.empty()) failure = ws.code;
    }

    double T_cold = now_s() - t_start;
    double T_warm = std::max(0.0, T_cold - setup_s);
    // Eligibility (default false; session.hpp session_eligibility). The public session seed,
    // a seeded or in-process dealer and a mode string alone never make a run security evidence.
    SessionEligibilityInputs elig_in;
    elig_in.completed = completed;
    elig_in.cancelled = cancelled;
    elig_in.failed = !failure.empty();
    elig_in.aborts = aborts;
    elig_in.diagnostic_reveals = 0;
    elig_in.dealer_mode = cfg.dealer_mode;
    elig_in.dealer_eligible = dealer.security_eligible();
    elig_in.dealer_insecure_seen = dealer.insecure_material_seen();
    elig_in.party_rng_from_os = cfg.party_rng_from_os;
    elig_in.in_process_dealer = (link != nullptr);
    const SessionEligibility elig = session_eligibility(elig_in);
    const bool eligible = elig.eligible;
    uint64_t rss = peak_rss_bytes();
    Value output_match = Value(None{});
    if (party == 1 && reference_tokens && completed) output_match = Value(*reference_tokens == tokens);

    record.set("schema", kEvidenceSchemaId);
    record.set("status", elig.status);
    record.set("security_eligible", eligible);
    record.set("code_commit", cfg.code_commit);
    record.set("profile_id", cfg.profile_id);
    record.set("numeric_contract_hash", contract_hash);
    record.set("dealer_mode", elig.dealer_mode);
    record.set("diagnostic_reveals", static_cast<int64_t>(0));
    record.set("workload_scope", cfg.workload_scope);
    record.set("model_revision", cfg.model_version);
    record.set("trace_id", trace.trace_id);
    record.set("trace_hash", trace.trace_hash());
    record.set("clock_basis", "wall-clock-exogenous");
    record.set("policy_id", cfg.policy);
    record.set("initial_state_hash", initial_state_hash);
    record.set("run_id", cfg.run_id);
    record.set("repeat", static_cast<int64_t>(cfg.repeat));
    record.set("initializations", initializations);
    record.set("post_init_switches", post_init_switches);
    record.set("rejected_candidates", rejected);
    record.set("aborts", aborts);
    record.set("T_cold_s", T_cold);
    record.set("T_warm_s", T_warm);
    record.set("T_online_s", T_online);
    record.set("bytes", static_cast<int64_t>(channel.stats.bytes_sent + channel.stats.bytes_recv - bytes0));
    record.set("rounds", static_cast<int64_t>(channel.stats.rounds - rounds0));
    record.set("material_generated", static_cast<int64_t>(link ? link->bytes_generated() : dealer.bytes_ingested()));
    record.set("material_transferred", static_cast<int64_t>(dealer.bytes_ingested()));
    record.set("material_consumed", static_cast<int64_t>(dealer.bytes_consumed()));
    record.set("material_burned", static_cast<int64_t>(ledger.bytes_burned()));
    record.set("peak_rss_client", party == 1 ? Value(static_cast<int64_t>(rss)) : Value(None{}));
    record.set("peak_rss_server", party == 0 ? Value(static_cast<int64_t>(rss)) : Value(None{}));
    record.set("tokens", Value(tokens_v));
    record.set("output_match", output_match);
    record.set("raw_measurement_path", out_path);
    // extras (not part of the binding field list, kept for analysis)
    record.set("party", static_cast<int64_t>(party));
    record.set("session_id", cfg.session_id);
    record.set("setup_id", cfg.setup_id);
    record.set("model", cfg.model);
    record.set("model_config", cfg.tiny.to_value());
    record.set("prompt_len", static_cast<int64_t>(prompt_len));
    record.set("n_generate", static_cast<int64_t>(cfg.n_generate));
    record.set("setup_s", setup_s);
    record.set("sensing_s", sensing_s);
    record.set("planning_s", planning_s);
    record.set("install_s", install_s);
    record.set("dealer_generation_s", link ? link->generation_seconds() - gen0 : 0.0);
    record.set("shaping_delay_s", channel.stats.shaping_delay_s);
    record.set("completed", completed);
    record.set("cancelled", cancelled);
    record.set("cancel_epoch", cancel_epoch >= 0 ? Value(cancel_epoch) : Value(None{}));
    record.set("resumed", resumed);
    record.set("resume_epoch", resumed ? Value(start_epoch) : Value(None{}));
    record.set("generation", generation);
    record.set("epochs_done", epochs_done);
    record.set("burned_outstanding", burn_report);
    record.set("reserved_after_run", static_cast<int64_t>(dealer.reserved_ids().size()));
    record.set("state_path", state_path);
    record.set("party_rng", cfg.party_rng_from_os ? "os_csprng" : "seed");
    {
        List notes;
        for (const auto& n : elig.notes) notes.push_back(Value(n));
        record.set("eligibility_notes", Value(notes));
    }
    record.set("failure", failure);
    record.set("final_plan", plan_value(plan));
    record.set("epochs", Value(epochs));
    record.set("gadget_stats", p.stats_value());
    record.set("release_policy", p.release().to_value());
    // The line on disk has the schema shape (tokens as a count, analysis fields under extra) and
    // is validated before it is written; the returned record keeps the flat analysis view.
    if (!out_path.empty()) {
        const Value line = conform_evidence_record(record);
        OC2_CHECK(validate_evidence_record(line));
        OC2_CHECK(append_jsonl(out_path, line));
    }
    if (!failure.empty()) return fail(failure);
    return ok(std::move(record));
}

}  // namespace highlevel
}  // namespace opencrab2
