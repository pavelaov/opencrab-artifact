// opencrab2_server: the P0 model server (WebSocket /session; paper, Section 3, design).
//
//   opencrab2_server --ws 127.0.0.1:8765 --dealer host:port --keys keys.json
//       [--contract NAME | --contract-hash HEX] [--model-version V] [--profile P] [--setup-id ID]
//       [--ttl S] [--max-queue N] [--handler tiny|demo|ack] [--demo-n 64] [--demo-seed N]
//       [--weights FILE | --tiny-seed N [--tiny-layers 1 --tiny-d 16 --tiny-heads 2 --tiny-vocab 32
//        --tiny-pos 64 --tiny-ff 64]] [--weights-out PATH] [--out evidence.jsonl] [--journal-dir DIR]
//       [--trace FILE | --trace-id ID --trace-bw BPS --trace-delay S | --no-shaping] [--hysteresis S]
//       [--costmodel FILE] [--allow-insecure-dealer] [--origin O]... [--max-seconds S] [--port-file PATH]
//
// Handlers:
//   tiny  (default) the highlevel session loop with the tiny decoder
//         (session_bridge.hpp): one run_session_party(0) thread per WebSocket
//         session, material from the isolated dealer, policies and plan
//         switching of paper, Section 4, one evidence line per
//         session appended to --out in the schema of
//         python/opencrab2/experiments/schema.py. --trace FILE (an
//         experiments/traces.py file) shapes the link of every session with
//         the wall-clock shaper; P0 announces the public trace to P1. A dealer
//         that is not ISOLATED_RECIPIENT_BUNDLES is refused unless
//         --allow-insecure-dealer is given. Weights come from --weights (exported
//         OC2TDW01 file) or are drawn deterministically from --tiny-seed and
//         optionally saved with --weights-out (verification runs).
//   demo  DemoBeaverP0: a real two-round semi-honest elementwise product with
//         dealer Beaver triples; P0's private vector w is drawn from the OS
//         CSPRNG, or derived from --demo-seed for reproducible verification
//         runs (test mode, printed as such).
//   ack   transport only: every MSG is acknowledged, nothing is computed.
// Prints {"event":"server_ready",...}.
#include <opencrab2/platform/dealer_service.hpp>
#include <opencrab2/platform/evidence.hpp>
#include <opencrab2/platform/model_server.hpp>
#include <opencrab2/platform/session_bridge.hpp>
#include <opencrab2/platform/support.hpp>

#include <atomic>
#include <csignal>
#include <cstdio>
#include <fstream>
#include <memory>

using namespace opencrab2::platform;

namespace {
std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

int usage(const char* prog) {
  std::fprintf(stderr,
               "usage: %s --ws host:port --dealer host:port --keys keys.json [--contract NAME | --contract-hash HEX]\n"
               "          [--model-version V] [--profile P] [--setup-id ID] [--ttl S] [--max-queue N]\n"
               "          [--handler tiny|demo|ack] [--demo-n 64] [--demo-seed N] [--origin O]... [--max-seconds S]\n"
               "          [--weights FILE | --tiny-seed N] [--weights-out PATH] [--out evidence.jsonl] [--journal-dir DIR]\n"
               "          [--trace FILE | --trace-id ID --trace-bw BPS --trace-delay S | --no-shaping] [--hysteresis S]\n"
               "          [--allow-insecure-dealer]\n",
               prog);
  return 2;
}
}  // namespace

int main(int argc, char** argv) {
#ifndef _WIN32
  std::signal(SIGPIPE, SIG_IGN);  // a peer that went away is an error return of the write, not the end of the process
#endif
  cli::Args args(argc, argv);
  if (args.flag("help") || args.flag("h")) return usage(argv[0]);
  auto ws = parse_host_port(args.get("ws", "127.0.0.1:8765"));
  if (!ws.ok) {
    std::fprintf(stderr, "%s: --ws\n", ws.code.c_str());
    return 2;
  }
  ModelServerConfig cfg;
  cfg.host = ws.value.first;
  cfg.port = ws.value.second;
  cfg.model_version = args.get("model-version", "gpt2-4L");
  cfg.profile_id = args.get("profile", "SH2P-TrustedOffline-v1");
  cfg.setup_id = args.get("setup-id", "setup-0");
  {
    auto chash = cli::contract_hash_from_args(args);
    if (!chash.ok) {
      std::fprintf(stderr, "%s: --contract\n", chash.code.c_str());
      return 2;
    }
    cfg.numeric_contract_hash = chash.value;
  }
  cfg.session_ttl_s = args.get_double("ttl", 300.0);
  cfg.max_queue_frames = static_cast<size_t>(args.get_int("max-queue", 64));
  cfg.origin_allowlist = args.get_all("origin");
  std::string handler_name = args.get("handler", "tiny");
  double max_seconds = args.get_double("max-seconds", 0.0);

  ProtocolHandler handler;
  std::unique_ptr<DemoBeaverP0> demo;
  std::shared_ptr<DealerConnection> dealer;
  std::string model_digest;
  if (handler_name == "ack") {
    handler = make_ack_handler();
  } else if (handler_name == "tiny") {
    std::string keys_path = args.get("keys", "");
    auto dhp = parse_host_port(args.get("dealer", ""));
    if (keys_path.empty() || !dhp.ok) {
      std::fprintf(stderr, "SERVER_ARGS: --dealer host:port and --keys are required for the tiny handler\n");
      return usage(argv[0]);
    }
    auto keys = load_keys_file(keys_path);
    if (!keys.ok) {
      std::fprintf(stderr, "%s: keys file %s\n", keys.code.c_str(), keys_path.c_str());
      return 2;
    }
    // P0's model: exported weights or a deterministic random model (public seed, verification runs)
    auto weights = std::make_shared<opencrab2::highlevel::TinyWeights>();
    const bool model_from_public_seed = !args.has("weights");
    if (args.has("weights")) {
      auto w = opencrab2::highlevel::TinyWeights::load(args.get("weights", ""));
      if (!w.ok) {
        std::fprintf(stderr, "%s: cannot load weights %s\n", w.code.c_str(), args.get("weights", "").c_str());
        return 2;
      }
      *weights = std::move(w.value);
    } else {
      opencrab2::highlevel::TinyConfig tc;
      tc.n_layers = static_cast<uint32_t>(args.get_int("tiny-layers", 1));
      tc.d_model = static_cast<uint32_t>(args.get_int("tiny-d", 16));
      tc.n_head = static_cast<uint32_t>(args.get_int("tiny-heads", 2));
      tc.vocab = static_cast<uint32_t>(args.get_int("tiny-vocab", 32));
      tc.max_pos = static_cast<uint32_t>(args.get_int("tiny-pos", 64));
      tc.d_ff = static_cast<uint32_t>(args.get_int("tiny-ff", 64));
      uint64_t seed = static_cast<uint64_t>(args.get_int("tiny-seed", 2026));
      *weights = opencrab2::highlevel::TinyWeights::random(tc, seed);
      std::fprintf(stderr, "[opencrab2-server] random tiny decoder weights from --tiny-seed %llu (public seed: verification run)\n",
                   static_cast<unsigned long long>(seed));
    }
    model_digest = weights->digest();
    if (args.has("weights-out")) {
      auto sv = weights->save(args.get("weights-out", ""));
      if (!sv.ok) {
        std::fprintf(stderr, "%s: cannot write %s\n", sv.code.c_str(), args.get("weights-out", "").c_str());
        return 2;
      }
    }
    auto conn = DealerConnection::connect(dhp.value.first, dhp.value.second, 0, cfg.setup_id, cfg.numeric_contract_hash,
                                          keys.value[0], 30.0, 100);
    if (!conn.ok) {
      std::fprintf(stderr, "%s: cannot reach the dealer at %s\n", conn.code.c_str(), args.get("dealer", "").c_str());
      return 1;
    }
    dealer = std::make_shared<DealerConnection>(std::move(conn.value));
    TinySessionOptions opts;
    opts.out_path = args.get("out", "");
    opts.journal_dir = args.get("journal-dir", "");
    opts.trace_id = args.get("trace-id", opts.trace_id);
    opts.trace_bandwidth_bps = args.get_double("trace-bw", opts.trace_bandwidth_bps);
    opts.trace_one_way_delay_s = args.get_double("trace-delay", opts.trace_one_way_delay_s);
    opts.trace_file = args.get("trace", "");
    if (!opts.trace_file.empty()) {
      auto t = load_trace_file(opts.trace_file);
      if (!t.ok) {
        std::fprintf(stderr, "%s: --trace %s\n", t.code.c_str(), opts.trace_file.c_str());
        return 2;
      }
    }
    opts.shape = !args.flag("no-shaping");
    opts.costmodel_path = args.get("costmodel", "");  // public; sent to P1 at setup (default: the HISTORICAL prior)
    if (!opts.costmodel_path.empty()) {
      auto cm = opencrab2::highlevel::FittedCostModel::load(opts.costmodel_path);
      if (!cm.ok) {
        std::fprintf(stderr, "%s: --costmodel %s (fail closed, no fallback to the prior)\n", cm.code.c_str(), opts.costmodel_path.c_str());
        return 2;
      }
    }
    opts.allow_insecure_dealer = args.flag("allow-insecure-dealer");
    opts.model_from_public_seed = model_from_public_seed;  // such a record is DIAGNOSTIC (the weights are not private)
    opts.code_commit = OPENCRAB2_CODE_COMMIT_LITERAL;
    opts.implementation = "opencrab2_server";
    opts.controller.hysteresis_s = args.get_double("hysteresis", opts.controller.hysteresis_s);
    if (!dealer->isolated() && !opts.allow_insecure_dealer) {
      std::fprintf(stderr, "DEALER_NOT_ISOLATED: the dealer announced mode '%s'; pass --allow-insecure-dealer for DIAGNOSTIC runs\n",
                   dealer->dealer_mode().c_str());
      return 1;
    }
    opts.on_record = [](const opencrab2::highlevel::Value& rec) {
      auto cj = opencrab2::highlevel::canonical_json(rec);
      if (cj.ok) {
        std::printf("{\"event\":\"session_record\",\"party\":0,\"record\":%s}\n", cj.value.c_str());
        std::fflush(stdout);
      }
    };
    handler = make_tiny_session_handler(weights, dealer, keys.value[0], cfg, opts);
  } else if (handler_name == "demo") {
    std::string keys_path = args.get("keys", "");
    auto dhp = parse_host_port(args.get("dealer", ""));
    if (keys_path.empty() || !dhp.ok) {
      std::fprintf(stderr, "SERVER_ARGS: --dealer host:port and --keys are required for the demo handler\n");
      return usage(argv[0]);
    }
    auto keys = load_keys_file(keys_path);
    if (!keys.ok) {
      std::fprintf(stderr, "%s: keys file %s\n", keys.code.c_str(), keys_path.c_str());
      return 2;
    }
    auto conn = DealerConnection::connect(dhp.value.first, dhp.value.second, 0, cfg.setup_id, cfg.numeric_contract_hash,
                                          keys.value[0], 30.0, 100);
    if (!conn.ok) {
      std::fprintf(stderr, "%s: cannot reach the dealer at %s\n", conn.code.c_str(), args.get("dealer", "").c_str());
      return 1;
    }
    dealer = std::make_shared<DealerConnection>(std::move(conn.value));
    size_t n = static_cast<size_t>(args.get_int("demo-n", 64));
    std::vector<uint64_t> w(n);
    if (args.has("demo-seed")) {
      detail::ChaCha20 rng = detail::ChaCha20::from_seed(static_cast<uint64_t>(args.get_int("demo-seed", 0)));
      rng.fill_u64(w.data(), n);
      std::fprintf(stderr, "[opencrab2-server] TEST MODE: demo vector w derived from --demo-seed (public)\n");
    } else {
      detail::ChaCha20 rng = detail::ChaCha20::from_os();
      rng.fill_u64(w.data(), n);
    }
    demo.reset(new DemoBeaverP0(dealer, std::move(w)));
    handler = demo->as_handler();
  } else {
    std::fprintf(stderr, "SERVER_ARGS: unknown handler %s\n", handler_name.c_str());
    return 2;
  }

  ModelServer server(cfg, handler);
  auto st = server.start();
  if (!st.ok) {
    std::fprintf(stderr, "%s: cannot start the model server on %s:%u\n", st.code.c_str(), cfg.host.c_str(),
                 static_cast<unsigned>(cfg.port));
    return 1;
  }
  std::string port_file = args.get("port-file", "");
  if (!port_file.empty()) {
    std::ofstream pf(port_file, std::ios::trunc);
    pf << server.port() << "\n";
  }
  json::Value ready = json::Value::object();
  ready.set("event", json::Value::string("server_ready"));
  ready.set("host", json::Value::string(cfg.host));
  ready.set("port", json::Value::integer(server.port()));
  ready.set("endpoint", json::Value::string("/session"));
  ready.set("handler", json::Value::string(handler_name));
  ready.set("model_version", json::Value::string(cfg.model_version));
  ready.set("numeric_contract_hash", json::Value::string(cfg.numeric_contract_hash));
  ready.set("profile_id", json::Value::string(cfg.profile_id));
  ready.set("ttl_s", json::Value::number(cfg.session_ttl_s));
  ready.set("max_queue_frames", json::Value::integer(static_cast<int64_t>(cfg.max_queue_frames)));
  if (dealer) ready.set("dealer_mode", json::Value::string(dealer->dealer_mode()));
  if (!model_digest.empty()) ready.set("model_digest", json::Value::string(model_digest));
  std::printf("%s\n", ready.dump(false).c_str());
  std::fflush(stdout);

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  double t0 = monotonic_s();
  while (!g_stop) {
    sleep_s(0.1);
    if (max_seconds > 0 && monotonic_s() - t0 >= max_seconds) break;
  }
  json::Value done = server.stats_json();
  done.set("event", json::Value::string("server_stopped"));
  server.stop();
  if (dealer) dealer->close();
  std::printf("%s\n", done.dump(false).c_str());
  return 0;
}
