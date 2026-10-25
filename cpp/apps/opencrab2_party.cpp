// opencrab2_party: direct two-party runner of the highlevel session loop
// (paper, Section 4.2, SessionControl; the counterpart of the Python run_sessions).
//
//   opencrab2_party --party 0 --listen 127.0.0.1:9000 --dealer host:port --keys keys.json [options]
//   opencrab2_party --party 1 --peer   127.0.0.1:9000 --dealer host:port --keys keys.json [options]
//   opencrab2_party --reference (--weights FILE | --seed N [shape options]) (--prompt-ids a,b,c | --prompt-len N) --generate T
//   opencrab2_party --make-weights FILE --seed N [shape options]
//
// What runs: highlevel::run_session_party with the tiny decoder (random or
// exported weights), the controller and the baseline policies, the material
// ledger (journal on disk) and material from the isolated dealer process, over
// one TCP channel with the binding wire codec. Per epoch: sense (fixed-size
// probe), candidates, decide, two-phase commit, transactional install,
// execute; the token is released to P1 only.
//
// Options (public session parameters; P0 announces them in a SETUP frame and
// P1 refuses any value that contradicts its own command line):
//   --policy ID            B4_opencrab (default), B0_fixed:<plan-json>, B1_naive,
//                          B2_remaining_plus_install, B3_planner_plus_B2, B4_no_install,
//                          B4_no_hyst, B4_no_unc, delayed:<k>, ...
//   --trace FILE           experiments trace JSON {points: [[elapsed_s, bandwidth_bps,
//                          one_way_delay_s, cpu_share, stock_refill_bps], ...]} as written by
//                          python -m opencrab2.experiments.traces / traces.make_trace. Both
//                          parties load the same file (hash checked in SETUP) and shape every
//                          message with the wall-clock shaper indexed by the shared origin, so a
//                          change of the trace applies in the middle of a transfer. Families:
//                          S0 to S3 shape bandwidth and delay; S4 adds the public CPU load of the
//                          cpu_share column (on --contention-party, default 0); S5 sets the
//                          declared material refill rate; S6 sets the length bucket; S7 events
//                          are recorded and driven by --cancel-after-epoch / --resume; S8
//                          initial_installed selects the initial plan (wte -> embedding wmask,
//                          every layer object -> linear wmask). Without --trace the link is
//                          unshaped (trace id "unshaped").
//   --layers N --d-model D --heads H --vocab V --max-pos P --d-ff F
//                          tiny decoder shape (P0; defaults 1, 16, 2, 32, max(64, prompt + T), 4 D)
//   --weights FILE         P0: OC2TDW01 weights; without it the model is drawn from --seed
//   --weights-out FILE     P0: save the model that is used
//   --prompt-len N         public prompt length (default 4)
//   --prompt-ids a,b,c     P1: the private prompt; without it a test prompt is drawn from --seed
//   --generate T           tokens to generate = epochs (default 3)
//   --seed N               PUBLIC experiment seed for the random model and the test prompt. It is
//                          never protocol randomness: each party draws its session seed from the
//                          OS CSPRNG and keeps it private.
//   --reference-tokens a,b P1: the expected greedy tokens (from --reference, computed by whoever
//                          holds both the model and the prompt); sets output_match
//   --verify               P1 test mode: derive the model from --seed and compute the reference
//                          locally. The model is then not private; the record says so.
//   --plan JSON            initial plan {"embedding","linear","argmax_k","max_k"}
//   --hysteresis S         controller hysteresis (default 0.5)
//   --costmodel FILE       fitted cost model JSON (default: the HISTORICAL prior, recorded)
//   --cancel-after-epoch K declared cancellation after epoch K (RESERVED material is burned)
//   --resume               resume the session bound to --session-id; together with
//                          --cancel-after-epoch: cancel, then resume in the same process
//   --session-id ID --journal-dir DIR --run-id ID --repeat K --out FILE
//   --contention-party 0|1|both|none   who runs the public CPU load of an S4 trace
//   --no-shaping           ignore bandwidth and delay of the trace (diagnostic)
//   --allow-insecure-dealer  accept a dealer that is not ISOLATED_RECIPIENT_BUNDLES (DIAGNOSTIC record)
//   --setup-id ID --model-version V --profile P --contract NAME | --contract-hash HEX
//                          the binding of the dealer bundles. --model-version defaults to the one
//                          the dealer announces (its bundles are bound to it); a different value
//                          is refused before the session starts.
//
// Evidence: one line per party appended to --out in the schema of
// python/opencrab2/experiments/schema.py (evidence.hpp), also printed as
// {"event":"party_record",...}. security_eligible follows the evidence classes
// of the paper (Section 3) as evaluated by the session loop (isolated dealer, no diagnostic
// reveal, completed run) and is additionally withdrawn here, with a note in
// extra.eligibility_notes, when the dealer is not isolated, when the model
// comes from the public seed, when P1 ran with --verify, when shaping was
// switched off, or when the session build cannot seed the party share
// randomness from the OS CSPRNG (PARTY_RNG_SEEDED, see session_bridge.hpp;
// the apps always request SessionConfig::party_rng_from_os).
//
// Timing boundaries (session.hpp) are those of the session loop: T_cold from the
// session start (setup, sharing, LM head install, sensing, planning, installs,
// execution, release), T_warm without the declared reusable setup, T_online
// the protocol execution intervals. Excluded here: process start, weight
// loading, dealer and peer connect, the SETUP frame, the record write.
//
// Exit status: 0 when the session completed (or was cancelled as requested)
// and output_match is not false; 1 on a runtime failure; 2 on a usage error.
// Logs carry ids, sizes and timings only.
#include <opencrab2/highlevel/session.hpp>
#include <opencrab2/platform/dealer_service.hpp>
#include <opencrab2/platform/evidence.hpp>
#include <opencrab2/platform/session_bridge.hpp>
#include <opencrab2/platform/shaper.hpp>
#include <opencrab2/platform/support.hpp>
#include <opencrab2/platform/tcp.hpp>

#include <cctype>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using namespace opencrab2::platform;
using opencrab2::Expected;
namespace hl = opencrab2::highlevel;

namespace {

const char* const kCodeCommit = OPENCRAB2_CODE_COMMIT_LITERAL;

int usage(const char* prog) {
  std::fprintf(stderr,
               "usage: %s --party 0|1 (--listen host:port | --peer host:port) --dealer host:port --keys keys.json\n"
               "          [--policy ID] [--trace FILE] [--layers N] [--d-model D] [--heads H] [--vocab V] [--max-pos P] [--d-ff F]\n"
               "          [--weights FILE] [--weights-out FILE] [--prompt-len N] [--prompt-ids a,b,c] [--generate T] [--seed N]\n"
               "          [--reference-tokens a,b,c] [--verify] [--plan JSON] [--hysteresis S] [--costmodel FILE]\n"
               "          [--cancel-after-epoch K] [--resume] [--session-id ID] [--journal-dir DIR] [--run-id ID] [--repeat K]\n"
               "          [--out FILE] [--contention-party 0|1|both|none] [--no-shaping] [--allow-insecure-dealer]\n"
               "          [--setup-id ID] [--model-version V] [--profile P] [--contract NAME | --contract-hash HEX]\n"
               "       %s --reference (--weights FILE | --seed N [shape]) (--prompt-ids a,b,c | --prompt-len N) --generate T\n"
               "       %s --make-weights FILE --seed N [shape]\n",
               prog, prog, prog);
  return 2;
}

const std::vector<std::string>& known_options() {
  static const std::vector<std::string> k = {
      "party", "listen", "peer", "dealer", "keys", "policy", "trace", "layers", "d-model", "heads", "vocab", "max-pos",
      "d-ff", "weights", "weights-out", "prompt-len", "prompt-ids", "generate", "seed", "reference-tokens", "verify",
      "plan", "hysteresis", "costmodel", "cancel-after-epoch", "resume", "session-id", "journal-dir", "run-id", "repeat",
      "out", "contention-party", "no-shaping", "allow-insecure-dealer", "setup-id", "model-version", "profile",
      "contract", "contract-hash", "reference", "make-weights", "length-bucket", "help", "h"};
  return k;
}

// "1,2,3" -> ints; fails on anything else.
Expected<std::vector<int>> parse_int_list(const std::string& s) {
  std::vector<int> out;
  for (const auto& part : detail::split(s, ',')) {
    std::string p = detail::trim(part);
    if (p.empty()) continue;
    for (char c : p) {
      if (!std::isdigit(static_cast<unsigned char>(c))) return make_fail<std::vector<int>>("PARTY_ARGS: not an integer list");
    }
    if (p.size() > 9) return make_fail<std::vector<int>>("PARTY_ARGS: integer too large");
    out.push_back(static_cast<int>(std::strtol(p.c_str(), nullptr, 10)));
  }
  return make_ok(std::move(out));
}

std::string safe_name(const std::string& s) {
  std::string out;
  for (char c : s) out.push_back((std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_') ? c : '_');
  return out;
}

std::string temp_dir() {
  const char* base = std::getenv("TEMP");
  if (!base) base = std::getenv("TMPDIR");
  if (!base) base = "/tmp";
  return base;
}

// Tiny decoder shape from the command line (P0, --reference, --make-weights).
hl::TinyConfig tiny_from_args(const cli::Args& args, int64_t prompt_len, int64_t generate) {
  hl::TinyConfig tc;
  tc.n_layers = static_cast<uint32_t>(args.get_int("layers", 1));
  tc.d_model = static_cast<uint32_t>(args.get_int("d-model", 16));
  tc.n_head = static_cast<uint32_t>(args.get_int("heads", 2));
  tc.vocab = static_cast<uint32_t>(args.get_int("vocab", 32));
  int64_t need = prompt_len + generate + 1;
  tc.max_pos = static_cast<uint32_t>(args.get_int("max-pos", need > 64 ? need : 64));
  tc.d_ff = static_cast<uint32_t>(args.get_int("d-ff", 4 * static_cast<int64_t>(tc.d_model)));
  return tc;
}

bool tiny_valid(const hl::TinyConfig& tc) {
  return tc.n_layers >= 1 && tc.n_layers <= 64 && tc.d_model >= 2 && tc.d_model <= 4096 && tc.n_head >= 1 &&
         tc.d_model % tc.n_head == 0 && tc.vocab >= 2 && tc.vocab <= (1u << 20) && tc.max_pos >= 2 && tc.max_pos <= (1u << 16) &&
         tc.d_ff >= 1 && tc.d_ff <= 16384;
}

// The test prompt of a public seed (P1 and --reference use the same rule).
std::vector<int> seeded_prompt(uint64_t seed, int64_t prompt_len, uint32_t vocab) {
  opencrab2::lowlevel::Rng rng = opencrab2::lowlevel::Rng::from_seed(seed ^ 0x70726f6d7074ULL);
  std::vector<int> out;
  for (int64_t i = 0; i < prompt_len; ++i) out.push_back(static_cast<int>(rng.below(vocab)));
  return out;
}

hl::Value int_list_value(const std::vector<int>& v) {
  hl::List l;
  for (int x : v) l.push_back(hl::Value(static_cast<int64_t>(x)));
  return hl::Value(std::move(l));
}

int64_t dict_int(const hl::Value& d, const char* key, int64_t def) {
  auto v = d.get_int(key);
  return v.ok ? v.value : def;
}

std::string dict_str(const hl::Value& d, const char* key, const std::string& def) {
  auto v = d.get_string(key);
  return v.ok ? v.value : def;
}

bool dict_bool(const hl::Value& d, const char* key, bool def) {
  const hl::Value* v = d.get(key);
  if (!v || !v->is_bool()) return def;
  return v->as_bool().value;
}

double dict_num(const hl::Value& d, const char* key, double def) {
  auto v = d.get_number(key);
  return v.ok ? v.value : def;
}

// First word of a failure code ("CANCELLED: detail" -> "CANCELLED").
std::string code_head(const std::string& code) {
  size_t p = code.find(':');
  return p == std::string::npos ? code : code.substr(0, p);
}

bool is_cancel_code(const std::string& code) {
  const std::string h = code_head(code);
  return h == "CANCELLED" || h == "SESSION_CANCELLED" || h == "CANCELED";
}

// True when a run ended by the declared cancellation (either reporting style).
bool run_was_cancelled(const Expected<hl::Value>& run) {
  if (!run.ok) return is_cancel_code(run.code);
  if (dict_bool(run.value, "cancelled", false)) return true;
  const hl::Value* e = run.value.get("extra");
  return e && dict_bool(*e, "cancelled", false);
}

void print_event(const char* event, int party, const hl::Value& body) {
  hl::Value o = hl::dict({{"event", event}, {"party", static_cast<int64_t>(party)}});
  if (const hl::Dict* d = std::get_if<hl::Dict>(&body.v)) {
    for (const auto& e : *d) o.set(e.key, e.val);
  }
  auto text = hl::canonical_json(o);
  std::printf("%s\n", text.ok ? text.value.c_str() : "{\"event\":\"unprintable\"}");
  std::fflush(stdout);
}

int reference_mode(const cli::Args& args) {
  int64_t generate = args.get_int("generate", 3);
  int64_t prompt_len = args.get_int("prompt-len", 4);
  uint64_t seed = static_cast<uint64_t>(args.get_int("seed", 0));
  if (generate < 1 || generate > 4096 || prompt_len < 1 || prompt_len > 4096) {
    std::fprintf(stderr, "PARTY_ARGS: --generate and --prompt-len must be in [1, 4096]\n");
    return 2;
  }
  hl::TinyWeights W;
  std::string source = "public_seed";
  if (args.has("weights")) {
    auto w = hl::TinyWeights::load(args.get("weights"));
    if (!w.ok) {
      std::fprintf(stderr, "%s: cannot load weights\n", w.code.c_str());
      return 1;
    }
    W = std::move(w.value);
    source = "file";
  } else {
    hl::TinyConfig tc = tiny_from_args(args, prompt_len, generate);
    if (!tiny_valid(tc)) {
      std::fprintf(stderr, "PARTY_ARGS: tiny decoder shape out of range\n");
      return 2;
    }
    W = hl::TinyWeights::random(tc, seed);
  }
  std::vector<int> prompt;
  if (args.has("prompt-ids")) {
    auto p = parse_int_list(args.get("prompt-ids"));
    if (!p.ok || p.value.empty()) {
      std::fprintf(stderr, "PARTY_ARGS: --prompt-ids\n");
      return 2;
    }
    prompt = p.value;
  } else {
    prompt = seeded_prompt(seed, prompt_len, W.cfg.vocab);
  }
  for (int t : prompt) {
    if (t < 0 || static_cast<uint32_t>(t) >= W.cfg.vocab) {
      std::fprintf(stderr, "PARTY_ARGS: prompt id outside the vocabulary\n");
      return 2;
    }
  }
  if (prompt.size() + static_cast<size_t>(generate) > W.cfg.max_pos) {
    std::fprintf(stderr, "SESSION_PROMPT_TOO_LONG: prompt + generate exceeds max_pos\n");
    return 2;
  }
  hl::TinyReference ref(W);
  std::vector<int> toks = ref.generate(prompt, static_cast<int>(generate));
  // This mode is run by whoever holds both the model and the prompt (the experimenter).
  print_event("reference", -1,
              hl::dict({{"tokens", int_list_value(toks)},
                        {"prompt_len", static_cast<int64_t>(prompt.size())},
                        {"model_digest", W.digest()},
                        {"model_source", source},
                        {"model_config", W.cfg.to_value()}}));
  return 0;
}

int make_weights_mode(const cli::Args& args) {
  const std::string path = args.get("make-weights");
  if (path.empty()) return 2;
  hl::TinyConfig tc = tiny_from_args(args, args.get_int("prompt-len", 4), args.get_int("generate", 3));
  if (!tiny_valid(tc)) {
    std::fprintf(stderr, "PARTY_ARGS: tiny decoder shape out of range\n");
    return 2;
  }
  hl::TinyWeights W = hl::TinyWeights::random(tc, static_cast<uint64_t>(args.get_int("seed", 0)));
  auto sv = W.save(path);
  if (!sv.ok) {
    std::fprintf(stderr, "%s: cannot write the weights file\n", sv.code.c_str());
    return 1;
  }
  print_event("weights_written", -1, hl::dict({{"model_digest", W.digest()}, {"model_config", W.cfg.to_value()}}));
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
#ifndef _WIN32
  std::signal(SIGPIPE, SIG_IGN);  // a peer that went away is an error return of the write, not the end of the process
#endif
  cli::Args args(argc, argv);
  if (args.flag("help") || args.flag("h")) return usage(argv[0]);
  {
    auto unknown = args.unknown_keys(known_options());
    if (!unknown.empty()) {
      std::fprintf(stderr, "PARTY_ARGS: unknown option --%s\n", unknown[0].c_str());
      return usage(argv[0]);
    }
  }
  if (args.has("reference")) return reference_mode(args);
  if (args.has("make-weights")) return make_weights_mode(args);

  const int party = static_cast<int>(args.get_int("party", -1));
  if (party != 0 && party != 1) return usage(argv[0]);
  auto dhp = parse_host_port(args.get("dealer", ""));
  const std::string keys_path = args.get("keys", "");
  if (!dhp.ok || keys_path.empty()) {
    std::fprintf(stderr, "PARTY_ARGS: --dealer host:port and --keys are required\n");
    return usage(argv[0]);
  }
  auto keys = load_keys_file(keys_path);
  if (!keys.ok) {
    std::fprintf(stderr, "%s: keys file\n", keys.code.c_str());
    return 2;
  }
  auto chash = cli::contract_hash_from_args(args);
  if (!chash.ok) {
    std::fprintf(stderr, "%s: --contract\n", chash.code.c_str());
    return 2;
  }
  const hl::NumericContract contract = hl::default_contract();
  if (chash.value != contract.contract_hash()) {
    std::fprintf(stderr, "CONTRACT_UNSUPPORTED: the tiny decoder session runs under the default contract only\n");
    return 2;
  }
  const std::string setup_id = args.get("setup-id", "setup-0");
  const std::string model_version_arg = args.get("model-version", "");
  const std::string profile_id = args.get("profile", evidence::kProfileId);
  const std::string out_path = args.get("out", "");
  const std::string journal_dir = args.get("journal-dir", temp_dir());
  const int64_t repeat = args.get_int("repeat", 0);
  const int64_t generate = args.get_int("generate", 3);
  const int64_t prompt_len_arg = args.get_int("prompt-len", 4);
  const uint64_t public_seed = static_cast<uint64_t>(args.get_int("seed", 0));
  const int64_t cancel_after = args.has("cancel-after-epoch") ? args.get_int("cancel-after-epoch", -1) : -1;
  const bool resume = args.flag("resume");
  const bool no_shaping = args.flag("no-shaping");
  const bool allow_insecure = args.flag("allow-insecure-dealer");
  const bool verify = args.flag("verify");
  const std::string contention_party = args.get("contention-party", "0");
  if (generate < 1 || generate > 4096 || prompt_len_arg < 1 || prompt_len_arg > 4096) {
    std::fprintf(stderr, "PARTY_ARGS: --generate and --prompt-len must be in [1, 4096]\n");
    return 2;
  }
  if (args.has("cancel-after-epoch") && (cancel_after < 0 || cancel_after >= generate)) {
    std::fprintf(stderr, "PARTY_ARGS: --cancel-after-epoch must be in [0, generate)\n");
    return 2;
  }
  if (contention_party != "0" && contention_party != "1" && contention_party != "both" && contention_party != "none") {
    std::fprintf(stderr, "PARTY_ARGS: --contention-party is 0, 1, both or none\n");
    return 2;
  }
  if (verify && party != 1) {
    std::fprintf(stderr, "PARTY_ARGS: --verify is a P1 test mode\n");
    return 2;
  }

  // ---- public trace (each party loads the same file; the hash is checked in SETUP) ----
  Trace trace = make_constant_trace(0.0, 0.0, 0.0, 1.0, "unshaped");
  if (args.has("trace")) {
    auto t = load_trace_file(args.get("trace"));
    if (!t.ok) {
      std::fprintf(stderr, "%s: --trace\n", t.code.c_str());
      return 2;
    }
    trace = std::move(t.value);
    if (trace.trace_id.empty()) trace.trace_id = "trace-" + trace.trace_hash.substr(0, 12);
  }

  // ---- policy, plan, controller, cost model (public) ----
  std::string policy = args.get("policy", "B4_opencrab");
  hl::ControllerConfig controller;
  controller.hysteresis_s = args.get_double("hysteresis", controller.hysteresis_s);
  {
    auto pol = hl::make_policy(policy, controller);
    if (!pol.ok) {
      std::fprintf(stderr, "%s: --policy\n", pol.code.c_str());
      return 2;
    }
  }
  hl::TinyPlan initial_plan;
  std::string plan_source = "default";
  if (args.has("plan")) {
    auto pj = hl::Json::parse(args.get("plan"));
    auto pp = pj.ok ? hl::TinyPlan::from_json(pj.value) : Expected<hl::TinyPlan>::failure(pj.code);
    if (!pp.ok) {
      std::fprintf(stderr, "%s: --plan\n", pp.code.c_str());
      return 2;
    }
    initial_plan = pp.value;
    plan_source = "command_line";
  }
  hl::FittedCostModel costmodel = hl::historical_prior_model();
  std::string costmodel_source = "HISTORICAL_PRIOR";
  if (args.has("costmodel")) {
    auto cm = hl::FittedCostModel::load(args.get("costmodel"));
    if (!cm.ok) {
      std::fprintf(stderr, "%s: --costmodel (fail closed, no fallback to the prior)\n", cm.code.c_str());
      return 2;
    }
    costmodel = std::move(cm.value);
    costmodel_source = "FITTED_FILE";
  }
  // identical decisions need identical cost models: the digest is checked in SETUP
  const std::string costmodel_digest = hl::json_sha256_hex(costmodel.to_json());

  // ---- P0: the model ----
  std::shared_ptr<hl::TinyWeights> weights;
  std::string model_source = "none";
  if (party == 0) {
    weights = std::make_shared<hl::TinyWeights>();
    if (args.has("weights")) {
      auto w = hl::TinyWeights::load(args.get("weights"));
      if (!w.ok) {
        std::fprintf(stderr, "%s: cannot load --weights\n", w.code.c_str());
        return 2;
      }
      *weights = std::move(w.value);
      model_source = "file";
    } else {
      hl::TinyConfig tc = tiny_from_args(args, prompt_len_arg, generate);
      if (!tiny_valid(tc)) {
        std::fprintf(stderr, "PARTY_ARGS: tiny decoder shape out of range\n");
        return 2;
      }
      *weights = hl::TinyWeights::random(tc, public_seed);
      model_source = "public_seed";
      std::fprintf(stderr, "[opencrab2-party] model drawn from the public --seed: verification run, the weights are not private\n");
    }
    if (args.has("weights-out")) {
      auto sv = weights->save(args.get("weights-out"));
      if (!sv.ok) {
        std::fprintf(stderr, "%s: cannot write --weights-out\n", sv.code.c_str());
        return 2;
      }
    }
  }

  // ---- 1. dealer (isolated third process) ----
  auto dconn = DealerConnection::connect(dhp.value.first, dhp.value.second, party, setup_id, chash.value, keys.value[party],
                                         30.0, 200);
  if (!dconn.ok) {
    std::fprintf(stderr, "%s: dealer unreachable\n", dconn.code.c_str());
    return 1;
  }
  auto dealer = std::make_shared<DealerConnection>(std::move(dconn.value));
  if (!dealer->isolated() && !allow_insecure) {
    std::fprintf(stderr, "DEALER_NOT_ISOLATED: the dealer announced mode '%s'; pass --allow-insecure-dealer for a DIAGNOSTIC run\n",
                 dealer->dealer_mode().c_str());
    dealer->close();
    return 1;
  }
  const std::string dealer_mode = dealer->isolated() ? "ISOLATED_RECIPIENT_BUNDLES" : "INSECURE_TEST_ONLY_SHARED_SEED";
  // Every bundle header carries the dealer's model_version and profile; the ledger binds them.
  if (!model_version_arg.empty() && !dealer->model_version().empty() && model_version_arg != dealer->model_version()) {
    std::fprintf(stderr, "VERSION_MISMATCH: --model-version differs from the dealer's model_version\n");
    dealer->close();
    return 1;
  }
  if (!dealer->profile_id().empty() && dealer->profile_id() != profile_id) {
    std::fprintf(stderr, "PROFILE_MISMATCH: --profile differs from the dealer's profile_id\n");
    dealer->close();
    return 1;
  }
  const std::string bound_model_version =
      !model_version_arg.empty() ? model_version_arg : (!dealer->model_version().empty() ? dealer->model_version() : std::string("model-0"));

  // ---- 2. peer channel (highlevel::Channel over TCP, unshaped until SETUP is done) ----
  std::unique_ptr<hl::Channel> ch;
  if (party == 0) {
    auto lhp = parse_host_port(args.get("listen", "127.0.0.1:0"));
    if (!lhp.ok) return usage(argv[0]);
    auto l = hl::listen_socket(lhp.value.first, lhp.value.second);
    if (!l.ok) {
      std::fprintf(stderr, "%s: cannot bind --listen\n", l.code.c_str());
      return 1;
    }
    print_event("party0_listening", 0, hl::dict({{"port", static_cast<int64_t>(l.value.port)}}));
    auto c = hl::accept_channel(l.value, 0, nullptr);
    if (!c.ok) {
      std::fprintf(stderr, "%s: no peer connected\n", c.code.c_str());
      return 1;
    }
    ch = std::move(c.value);
  } else {
    auto php = parse_host_port(args.get("peer", ""));
    if (!php.ok) {
      std::fprintf(stderr, "PARTY_ARGS: --peer host:port is required for party 1\n");
      return 2;
    }
    auto c = hl::dial_channel(php.value.first, php.value.second, 1, nullptr, 3000);
    if (!c.ok) {
      std::fprintf(stderr, "%s: cannot reach the peer\n", c.code.c_str());
      return 1;
    }
    ch = std::move(c.value);
  }

  // ---- 3. SETUP: public session parameters, P0 -> P1, then P1's verdict ----
  hl::SessionConfig cfg;
  double origin_unix = 0.0;
  bool shaping = !no_shaping;
  if (party == 0) {
    cfg.tiny = weights->cfg;
    cfg.session_id = args.get("session-id", "sess-" + detail::hex_encode(detail::os_random_bytes(8)));
    cfg.model_version = bound_model_version;
    cfg.model = "tiny-decoder@" + weights->digest().substr(0, 16);
    std::vector<uint8_t> tk = detail::os_random_bytes(32);
    cfg.telemetry_key = hl::Bytes(tk.begin(), tk.end());
    cfg.run_id = args.get("run-id", "party-" + cfg.session_id);
    origin_unix = wall_s() + 0.25;
    if (static_cast<uint64_t>(prompt_len_arg + generate) > cfg.tiny.max_pos) {
      std::fprintf(stderr, "SESSION_PROMPT_TOO_LONG: prompt-len + generate exceeds max_pos\n");
      return 2;
    }
    hl::Value setup = hl::dict({{"type", "SETUP"},
                                {"session_id", cfg.session_id},
                                {"run_id", cfg.run_id},
                                {"origin_unix_s", origin_unix},
                                {"tiny", cfg.tiny.to_value()},
                                {"model", cfg.model},
                                {"model_version", cfg.model_version},
                                {"model_source", model_source},
                                {"numeric_contract_hash", chash.value},
                                {"profile_id", profile_id},
                                {"setup_id", setup_id},
                                {"n_generate", generate},
                                {"prompt_len", prompt_len_arg},
                                {"policy", policy},
                                {"plan", initial_plan.to_json().dump()},
                                {"plan_source", plan_source},
                                {"controller", controller.to_json().dump()},
                                {"trace_id", trace.trace_id},
                                {"trace_hash", trace.trace_hash},
                                {"shaping", shaping},
                                {"telemetry_key", hl::Value(cfg.telemetry_key)},
                                {"dealer_mode", dealer_mode},
                                {"cancel_after_epoch", cancel_after},
                                {"resume", resume},
                                {"costmodel", costmodel_source},
                                {"costmodel_digest", costmodel_digest}});
    auto s = ch->send(setup);
    auto r = s.ok ? ch->recv() : Expected<hl::Value>::failure(s.code);
    if (!r.ok || dict_str(r.value, "type", "") != "SETUP_OK") {
      std::fprintf(stderr, "%s: setup refused by P1\n", r.ok ? dict_str(r.value, "code", "PARTY_PROTOCOL").c_str() : r.code.c_str());
      return 1;
    }
  } else {
    auto r = ch->recv();
    if (!r.ok || dict_str(r.value, "type", "") != "SETUP") {
      std::fprintf(stderr, "%s: no SETUP from P0\n", r.ok ? "PARTY_PROTOCOL" : r.code.c_str());
      return 1;
    }
    const hl::Value& su = r.value;
    std::string bad;
    const hl::Value* tiny = su.get("tiny");
    auto tkey = su.get_bytes("telemetry_key");
    if (!tiny || !tiny->is_dict() || !tkey.ok || tkey.value.size() < 16) bad = "SETUP_MALFORMED";
    if (bad.empty()) {
      cfg.tiny.n_layers = static_cast<uint32_t>(dict_int(*tiny, "n_layers", 0));
      cfg.tiny.d_model = static_cast<uint32_t>(dict_int(*tiny, "d_model", 0));
      cfg.tiny.n_head = static_cast<uint32_t>(dict_int(*tiny, "n_head", 0));
      cfg.tiny.vocab = static_cast<uint32_t>(dict_int(*tiny, "vocab", 0));
      cfg.tiny.max_pos = static_cast<uint32_t>(dict_int(*tiny, "max_pos", 0));
      cfg.tiny.d_ff = static_cast<uint32_t>(dict_int(*tiny, "d_ff", 0));
      if (!tiny_valid(cfg.tiny)) bad = "SETUP_MALFORMED";
    }
    // every public parameter P1 was given on its own command line must agree
    if (bad.empty() && dict_str(su, "numeric_contract_hash", "") != chash.value) bad = "CONTRACT_MISMATCH";
    if (bad.empty() && dict_str(su, "profile_id", "") != profile_id) bad = "PROFILE_MISMATCH";
    if (bad.empty() && dict_str(su, "setup_id", "") != setup_id) bad = "SETUP_MISMATCH";
    if (bad.empty() && dict_str(su, "model_version", "") != bound_model_version) bad = "VERSION_MISMATCH";
    if (bad.empty() && dict_str(su, "trace_hash", "") != trace.trace_hash) bad = "TRACE_MISMATCH";
    if (bad.empty() && dict_str(su, "trace_id", "") != trace.trace_id) bad = "TRACE_MISMATCH";  // same points, other metadata
    if (bad.empty() && dict_int(su, "n_generate", -1) != generate) bad = "SHAPE_MISMATCH";
    if (bad.empty() && dict_int(su, "prompt_len", -1) != prompt_len_arg) bad = "SHAPE_MISMATCH";
    if (bad.empty() && args.has("policy") && dict_str(su, "policy", "") != policy) bad = "POLICY_MISMATCH";
    if (bad.empty() && dict_bool(su, "shaping", true) != shaping) bad = "SHAPING_MISMATCH";
    if (bad.empty() && dict_str(su, "dealer_mode", "") != dealer_mode) bad = "DEALER_MODE_MISMATCH";
    if (bad.empty() && dict_int(su, "cancel_after_epoch", -1) != cancel_after) bad = "CANCEL_MISMATCH";
    if (bad.empty() && dict_bool(su, "resume", false) != resume) bad = "RESUME_MISMATCH";
    if (bad.empty() && dict_str(su, "costmodel_digest", "") != costmodel_digest) bad = "COSTMODEL_MISMATCH";
    if (bad.empty() && args.has("session-id") && dict_str(su, "session_id", "") != args.get("session-id")) bad = "SESSION_MISMATCH";
    if (bad.empty() && args.has("layers") && dict_int(*tiny, "n_layers", -1) != args.get_int("layers", -2)) bad = "SHAPE_MISMATCH";
    if (bad.empty() && args.has("vocab") && dict_int(*tiny, "vocab", -1) != args.get_int("vocab", -2)) bad = "SHAPE_MISMATCH";
    if (bad.empty() && args.has("d-model") && dict_int(*tiny, "d_model", -1) != args.get_int("d-model", -2)) bad = "SHAPE_MISMATCH";
    origin_unix = dict_num(su, "origin_unix_s", 0.0);
    if (bad.empty() && (!(origin_unix > 0.0) || origin_unix > wall_s() + 3600.0 || origin_unix < wall_s() - 3600.0))
      bad = "SETUP_MALFORMED";
    if (bad.empty() && static_cast<uint64_t>(prompt_len_arg + generate) > cfg.tiny.max_pos) bad = "SESSION_PROMPT_TOO_LONG";
    hl::ControllerConfig cc;
    hl::TinyPlan pl;
    if (bad.empty()) {
      auto cj = hl::Json::parse(dict_str(su, "controller", "{}"));
      auto ccr = cj.ok ? hl::ControllerConfig::from_json(cj.value) : Expected<hl::ControllerConfig>::failure(cj.code);
      auto pj = hl::Json::parse(dict_str(su, "plan", "{}"));
      auto plr = pj.ok ? hl::TinyPlan::from_json(pj.value) : Expected<hl::TinyPlan>::failure(pj.code);
      if (!ccr.ok || !plr.ok) {
        bad = "SETUP_MALFORMED";
      } else {
        cc = ccr.value;
        pl = plr.value;
      }
    }
    if (bad.empty()) {
      auto pol = hl::make_policy(dict_str(su, "policy", ""), cc);
      if (!pol.ok) bad = "POLICY_UNKNOWN";
    }
    hl::Value reply = bad.empty() ? hl::dict({{"type", "SETUP_OK"}}) : hl::dict({{"type", "REJECT"}, {"code", bad}});
    auto s = ch->send(reply);
    ch->clear_pending();  // the SETUP pair is one round on both sides
    ch->count_round();
    if (!bad.empty() || !s.ok) {
      std::fprintf(stderr, "%s: refusing P0's setup\n", bad.empty() ? s.code.c_str() : bad.c_str());
      return 1;
    }
    policy = dict_str(su, "policy", policy);
    controller = cc;
    initial_plan = pl;
    plan_source = dict_str(su, "plan_source", "default");
    model_source = dict_str(su, "model_source", "none");
    cfg.session_id = dict_str(su, "session_id", "");
    cfg.run_id = dict_str(su, "run_id", "party-" + cfg.session_id);
    cfg.model = dict_str(su, "model", "tiny-decoder");
    cfg.model_version = dict_str(su, "model_version", "");
    cfg.telemetry_key = tkey.value;
  }

  // ---- 4. session configuration ----
  cfg.n_layers = static_cast<int>(cfg.tiny.n_layers);
  cfg.n_generate = static_cast<int>(generate);
  cfg.policy = policy;
  cfg.trace_id = trace.trace_id;
  cfg.seed = private_session_seed();  // private party randomness seed, never sent, never recorded
  cfg.contract = contract;
  cfg.profile_id = profile_id;
  cfg.length_bucket = static_cast<int>(args.get_int("length-bucket", trace.length_bucket > 0 ? trace.length_bucket : 32));
  cfg.initial_plan = initial_plan;
  cfg.setup_id = setup_id;
  cfg.dealer_mode = dealer_mode;
  cfg.repeat = static_cast<int>(repeat);
  cfg.workload_scope = "full-model";
  cfg.code_commit = kCodeCommit;
  cfg.controller = controller;
  if (trace.max_stock_refill_bps() > 0) cfg.refill_bytes_per_s = trace.max_stock_refill_bps();  // S5: declared refill
  // S8: the declared installed objects select the initial plan (both parties read the same trace)
  if (plan_source == "default" && !trace.initial_installed.empty()) {
    auto has = [&](const std::string& o) {
      for (const auto& s : trace.initial_installed) {
        if (s == o) return true;
      }
      return false;
    };
    if (has("wte")) cfg.initial_plan.embedding = "wmask";
    bool all_layers = true;
    for (uint32_t l = 0; l < cfg.tiny.n_layers; ++l) {
      for (const char* s : {"qkv", "out", "fc", "proj"}) all_layers = all_layers && has("h" + std::to_string(l) + "." + s);
    }
    if (all_layers) cfg.initial_plan.linear = "wmask";
    plan_source = "trace_initial_installed";
  }
  const bool os_rng = session_caps::request_os_party_rng(cfg);
  const std::string state_path = journal_dir + "/oc2_party_" + safe_name(cfg.session_id) + "_p" + std::to_string(party) + ".state.json";
  session_caps::set_state_path(cfg, state_path, 0);

  // P1's private prompt and the expected tokens
  std::vector<int> reference;
  const std::vector<int>* ref_ptr = nullptr;
  std::string prompt_source = "none";
  bool weights_disclosed = false;
  if (party == 1) {
    if (args.has("prompt-ids")) {
      auto p = parse_int_list(args.get("prompt-ids"));
      if (!p.ok || static_cast<int64_t>(p.value.size()) != prompt_len_arg) {
        std::fprintf(stderr, "PARTY_ARGS: --prompt-ids must list --prompt-len token ids\n");
        return 2;
      }
      cfg.prompt_ids = p.value;
      prompt_source = "command_line";
    } else {
      cfg.prompt_ids = seeded_prompt(public_seed, prompt_len_arg, cfg.tiny.vocab);
      prompt_source = "public_seed";
    }
    for (int t : cfg.prompt_ids) {
      if (t < 0 || static_cast<uint32_t>(t) >= cfg.tiny.vocab) {
        std::fprintf(stderr, "PARTY_ARGS: prompt id outside the vocabulary\n");
        return 2;
      }
    }
    if (args.has("reference-tokens")) {
      auto rt = parse_int_list(args.get("reference-tokens"));
      if (!rt.ok || static_cast<int64_t>(rt.value.size()) != generate) {
        std::fprintf(stderr, "PARTY_ARGS: --reference-tokens must list --generate tokens\n");
        return 2;
      }
      reference = rt.value;
      ref_ptr = &reference;
    } else if (verify) {
      if (model_source != "public_seed") {
        std::fprintf(stderr, "VERIFY_UNAVAILABLE: --verify needs a model drawn from the public --seed (P0 without --weights)\n");
        return 1;
      }
      hl::TinyWeights W = hl::TinyWeights::random(cfg.tiny, public_seed);
      if ("tiny-decoder@" + W.digest().substr(0, 16) != cfg.model) {
        std::fprintf(stderr, "VERIFY_WEIGHTS_MISMATCH: the model of --seed differs from P0's model\n");
        return 1;
      }
      hl::TinyReference ref(W);
      reference = ref.generate(cfg.prompt_ids, cfg.n_generate);
      ref_ptr = &reference;
      weights_disclosed = true;
    }
  }

  // ---- 5. ledger, dealer client, shaping, public CPU load ----
  hl::Value binding = hl::dict({{"session_id", cfg.session_id},
                                {"setup_id", cfg.setup_id},
                                {"model_version", cfg.model_version},
                                {"numeric_contract_hash", contract.contract_hash()},
                                {"profile_id", cfg.profile_id}});
  const std::string journal = journal_dir + "/oc2_party_" + safe_name(cfg.session_id) + "_p" + std::to_string(party) + ".jsonl";
  if (!resume) std::remove(journal.c_str());
  hl::MaterialLedger ledger(journal, binding);
  auto lo = ledger.open();
  if (!lo.ok) {
    std::fprintf(stderr, "%s: cannot open the ledger journal\n", lo.code.c_str());
    return 1;
  }
  if (resume) {
    // cold resume: open() replayed the journal; RESERVED entries are burned (paper, Section 4.3).
    // This also holds for a resumed run that is cancelled again (--cancel-after).
    auto rec = ledger.recover();
    if (!rec.ok) {
      std::fprintf(stderr, "%s: cannot recover the ledger journal\n", rec.code.c_str());
      return 1;
    }
  }
  hl::DealerClient client(party, ledger, hl::Bytes(keys.value[party].begin(), keys.value[party].end()), contract.contract_hash());
  auto conn_mu = std::make_shared<std::mutex>();
  attach_dealer_connection(client, dealer, conn_mu, contract.logical_bits);
  if (shaping) ch->set_shaper(make_channel_shaper(trace, origin_unix));
  ContentionLoad load;
  const bool contention = trace.has_contention() && (contention_party == "both" || contention_party == std::to_string(party));
  if (contention) load.start(trace, origin_unix, 1);
  hl::SessionTrace strace = to_session_trace(trace);

  // ---- 6. the session (the loop writes its raw line to a private file) ----
  const std::string raw_path = journal_dir + "/oc2_party_" + safe_name(cfg.session_id) + "_p" + std::to_string(party) + ".record.jsonl";
  std::remove(raw_path.c_str());
  auto run_once = [&]() {
    return hl::run_session_party(party, cfg, *ch, client, ledger, costmodel, strace, weights.get(), ref_ptr, raw_path, nullptr);
  };
  Expected<hl::Value> run = Expected<hl::Value>::failure("SESSION_NOT_RUN");
  hl::Value cancel_summary = hl::Value(hl::None{});
  bool cancel_honoured = false;
  bool resumed = false;
  if (cancel_after >= 0) {
    if (!session_caps::set_cancel_after_epoch(cfg, static_cast<int>(cancel_after), 0)) {
      std::fprintf(stderr, "SESSION_CANCEL_UNSUPPORTED: this build's highlevel session exposes no cancel_after_epoch\n");
      load.stop();
      dealer->close();
      return 2;
    }
    if (resume && !session_caps::set_resume(cfg, false, 0)) {
      std::fprintf(stderr, "SESSION_RESUME_UNSUPPORTED: this build's highlevel session exposes no resume\n");
      load.stop();
      dealer->close();
      return 2;
    }
    run = run_once();
    cancel_honoured = run_was_cancelled(run);
    cancel_summary = hl::dict({{"cancel_after_epoch", cancel_after},
                               {"cancelled", cancel_honoured},
                               {"code", run.ok ? std::string() : run.code},
                               {"material_burned_at_cancel", static_cast<int64_t>(ledger.bytes_burned())},
                               {"material_consumed_at_cancel", static_cast<int64_t>(client.bytes_consumed())}});
    if (resume && cancel_honoured) {
      // warm resume in the same process: same session id, same ledger, new bundles on demand
      session_caps::set_cancel_after_epoch(cfg, -1, 0);
      session_caps::set_resume(cfg, true, 0);
      std::remove(raw_path.c_str());
      run = run_once();
      resumed = true;
    }
  } else {
    if (resume) {
      if (!session_caps::set_resume(cfg, true, 0)) {
        std::fprintf(stderr, "SESSION_RESUME_UNSUPPORTED: this build's highlevel session exposes no resume\n");
        load.stop();
        dealer->close();
        return 2;
      }
      resumed = true;
    }
    run = run_once();
  }
  load.stop();

  // ---- 7. evidence record ----
  hl::Value raw;
  if (run.ok) {
    raw = run.value;
  } else {
    std::string last;
    if (std::FILE* f = std::fopen(raw_path.c_str(), "rb")) {
      std::string all;
      char buf[65536];
      size_t n;
      while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) all.append(buf, n);
      std::fclose(f);
      for (const auto& line : detail::split(all, '\n')) {
        if (!detail::trim(line).empty()) last = detail::trim(line);
      }
    }
    auto parsed = last.empty() ? Expected<hl::Value>::failure("RECORD_IO") : hl::parse_json(last);
    if (parsed.ok && parsed.value.is_dict()) {
      raw = parsed.value;
    } else {
      raw = evidence::new_record(kCodeCommit);
      raw.set("status", "DIAGNOSTIC");
      raw.set("clock_basis", "wall-clock-exogenous");
      raw.set("workload_scope", "full-model");
      raw.set("policy_id", policy);
      raw.set("run_id", cfg.run_id);
      raw.set("aborts", static_cast<int64_t>(1));
      raw.set("completed", false);
      raw.set("failure", run.code);
    }
  }
  std::remove(raw_path.c_str());
  SessionRecordContext rc;
  rc.record_kind = "session";
  rc.code_commit = kCodeCommit;
  rc.implementation = "opencrab2_party";
  rc.trace_id = trace.trace_id;
  rc.trace_hash = trace.trace_hash;
  rc.trace_family = trace.family;
  rc.trace_variant = trace.variant;
  rc.trace_split = trace.split;
  rc.raw_measurement_path = out_path;
  rc.dealer_mode = dealer_mode;
  rc.party_rng_os_seeded = os_rng;
  rc.weights_disclosed_to_p1 = weights_disclosed;
  rc.model_from_public_seed = model_source == "public_seed";
  rc.contention = contention;
  hl::Value rec = finalize_session_record(raw, rc);
  rec.set("seed", static_cast<int64_t>(public_seed));  // the PUBLIC experiment seed (model / test prompt), not party randomness
  rec.set("n_layers", static_cast<int64_t>(cfg.tiny.n_layers));
  rec.set("T", generate);
  rec.set("label", "tiny-decoder-session");
  rec.set("host", evidence::host_info());
  rec.set("repeat", repeat);
  evidence::set_extra(rec, "party", hl::Value(static_cast<int64_t>(party)));
  evidence::set_extra(rec, "model_source", hl::Value(model_source));
  evidence::set_extra(rec, "prompt_source", hl::Value(prompt_source));
  evidence::set_extra(rec, "plan_source", hl::Value(plan_source));
  evidence::set_extra(rec, "costmodel_source", hl::Value(costmodel_source));
  evidence::set_extra(rec, "costmodel_digest", hl::Value(costmodel_digest));
  evidence::set_extra(rec, "shaping", hl::Value(shaping));
  evidence::set_extra(rec, "shaping_origin_unix_s", hl::Value(origin_unix));
  evidence::set_extra(rec, "trace_points", hl::Value(static_cast<int64_t>(trace.points.size())));
  evidence::set_extra(rec, "trace_hash_verified", hl::Value(trace.hash_verified));
  evidence::set_extra(rec, "length_bucket", hl::Value(static_cast<int64_t>(cfg.length_bucket)));
  evidence::set_extra(rec, "refill_bytes_per_s", hl::Value(cfg.refill_bytes_per_s));
  evidence::set_extra(rec, "resumed", hl::Value(resumed));
  if (!cancel_summary.is_none()) evidence::set_extra(rec, "cancel", cancel_summary);
  if (!trace.events.empty()) {
    hl::List ev;
    for (const auto& e : trace.events) ev.push_back(hl::Value(hl::List{hl::Value(e.elapsed_s), hl::Value(e.kind)}));
    evidence::set_extra(rec, "trace_events", hl::Value(std::move(ev)));
  }
  if (!trace.initial_installed.empty()) {
    hl::List inst;
    for (const auto& s : trace.initial_installed) inst.push_back(hl::Value(s));
    evidence::set_extra(rec, "trace_initial_installed", hl::Value(std::move(inst)));
  }
  if (!shaping && args.has("trace"))
    evidence::withdraw_eligibility(rec, "SHAPING_DISABLED: the trace was loaded but the link was not shaped", true);
  evidence::enforce(rec);
  auto valid = evidence::validate(rec);
  if (!valid.ok) {
    std::fprintf(stderr, "%s: the session record does not conform to the evidence schema\n", valid.code.c_str());
    dealer->close();
    return 1;
  }
  if (!out_path.empty()) {
    auto ap = evidence::append(out_path, rec);
    if (!ap.ok) {
      std::fprintf(stderr, "%s: cannot append the record\n", ap.code.c_str());
      dealer->close();
      return 1;
    }
  }
  print_event("party_record", party, hl::dict({{"record", rec}}));
  dealer->close();
  ch->close();

  // ---- 8. exit status ----
  const hl::Value* om = rec.get("output_match");
  const bool mismatch = om && om->is_bool() && !om->as_bool().value;
  bool success = run.ok;
  if (cancel_after >= 0 && !resume) success = cancel_honoured;  // a cancel-only run ends by the declared cancellation
  if (!success) std::fprintf(stderr, "%s: session did not complete\n", run.ok ? "SESSION_NOT_CANCELLED" : run.code.c_str());
  if (mismatch) std::fprintf(stderr, "OUTPUT_MISMATCH: released tokens differ from the reference\n");
  return (success && !mismatch) ? 0 : 1;
}
