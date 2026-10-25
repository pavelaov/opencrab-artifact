// opencrab2_sidecar: the P1 local sidecar (paper, Section 3, design).
//
//   opencrab2_sidecar --http 127.0.0.1:8787 --server ws://127.0.0.1:8765/session
//       --dealer host:port --keys keys.json [--origin O]... [--token T] [--token-file PATH]
//       [--static-dir DIR] [--engine tiny|demo|stub] [--contract NAME | --contract-hash HEX]
//       [--model-version V] [--profile P] [--setup-id ID] [--demo-n 64] [--length-bucket 32]
//       [--out evidence.jsonl] [--policy B4_opencrab] [--journal-dir DIR] [--verify-weights FILE]
//       [--allow-insecure-dealer] [--max-seconds S] [--port-file PATH]
//
// Engines: tiny (default) runs the highlevel tiny-decoder session of
// session_bridge.hpp against a server started with --handler tiny: the prompt
// bytes become token ids inside this process, every released token is streamed
// as an SSE event, and one evidence line per generation is appended to --out.
// --verify-weights FILE (test mode) loads the server's weights file and sets
// output_match in the record; the record is then DIAGNOSTIC because the
// weights were disclosed to P1. demo is the Beaver product exercise, stub
// needs no network. GET /v1/status carries the web metrics of paper,
// Section 7.4 (TTFT cold and warm, inter-token latency, completion, cancel
// waste, rejection counters); the final sidecar_stopped line repeats them.
//
// The sidecar binds 127.0.0.1 only. Every /v1 request needs an allowed Origin
// (when present) and the per-launch bearer token, which is printed once in the
// launch URL fragment and optionally written to --token-file (created with
// mode 0600 on POSIX). The static page is served from --static-dir (default: the
// python browser_client directory when it can be found from the working
// directory), read only, or a built-in page otherwise.
#include <opencrab2/platform/dealer_service.hpp>
#include <opencrab2/platform/evidence.hpp>
#include <opencrab2/platform/session_bridge.hpp>
#include <opencrab2/platform/sidecar.hpp>
#include <opencrab2/platform/support.hpp>

#include <atomic>
#include <csignal>
#include <cstdio>
#include <fstream>

using namespace opencrab2::platform;

namespace {
std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

int usage(const char* prog) {
  std::fprintf(stderr,
               "usage: %s --http 127.0.0.1:8787 --server ws://host:port/session --dealer host:port --keys keys.json\n"
               "          [--origin O]... [--token T] [--token-file PATH] [--static-dir DIR] [--engine tiny|demo|stub]\n"
               "          [--out evidence.jsonl] [--policy ID] [--journal-dir DIR] [--verify-weights FILE]\n"
               "          [--allow-insecure-dealer]\n"
               "          [--contract NAME | --contract-hash HEX] [--model-version V] [--profile P] [--setup-id ID]\n"
               "          [--demo-n 64] [--length-bucket 32] [--max-seconds S] [--port-file PATH]\n",
               prog);
  return 2;
}

std::string find_static_dir() {
  const char* candidates[] = {"python/opencrab2/platform/browser_client", "../python/opencrab2/platform/browser_client",
                              "../../python/opencrab2/platform/browser_client",
                              "../../../python/opencrab2/platform/browser_client"};
  for (const char* c : candidates) {
    std::ifstream probe(std::string(c) + "/index.html");
    if (probe) return c;
  }
  return "";
}
}  // namespace

int main(int argc, char** argv) {
#ifndef _WIN32
  std::signal(SIGPIPE, SIG_IGN);  // a peer that went away is an error return of the write, not the end of the process
#endif
  cli::Args args(argc, argv);
  if (args.flag("help") || args.flag("h")) return usage(argv[0]);
  auto hp = parse_host_port(args.get("http", "127.0.0.1:8787"));
  if (!hp.ok) {
    std::fprintf(stderr, "%s: --http\n", hp.code.c_str());
    return 2;
  }
  SidecarConfig cfg;
  cfg.host = hp.value.first;
  cfg.port = hp.value.second;
  cfg.origin_allowlist = args.get_all("origin");
  cfg.token = args.get("token", "");
  cfg.server_ws_url = args.get("server", "ws://127.0.0.1:8765/session");
  cfg.setup_id = args.get("setup-id", "setup-0");
  cfg.model_version = args.get("model-version", "gpt2-4L");
  cfg.profile_id = args.get("profile", "SH2P-TrustedOffline-v1");
  {
    auto chash = cli::contract_hash_from_args(args);
    if (!chash.ok) {
      std::fprintf(stderr, "%s: --contract\n", chash.code.c_str());
      return 2;
    }
    cfg.numeric_contract_hash = chash.value;
  }
  cfg.length_bucket = args.get_int("length-bucket", 32);
  cfg.demo_vector_len = static_cast<size_t>(args.get_int("demo-n", 64));
  cfg.static_dir = args.has("static-dir") ? args.get("static-dir", "") : find_static_dir();
  std::string engine_name = args.get("engine", "tiny");
  double max_seconds = args.get_double("max-seconds", 0.0);

  ClientEngine engine;
  if (engine_name == "stub") {
    engine = make_stub_engine(0.05);
  } else if (engine_name == "tiny") {
    auto dhp = parse_host_port(args.get("dealer", ""));
    std::string keys_path = args.get("keys", "");
    if (!dhp.ok || keys_path.empty()) {
      std::fprintf(stderr, "SIDECAR_ARGS: --dealer host:port and --keys are required for the tiny engine\n");
      return usage(argv[0]);
    }
    auto keys = load_keys_file(keys_path);
    if (!keys.ok) {
      std::fprintf(stderr, "%s: keys file %s\n", keys.code.c_str(), keys_path.c_str());
      return 2;
    }
    cfg.dealer_host = dhp.value.first;
    cfg.dealer_port = dhp.value.second;
    cfg.dealer_key = keys.value[1];
    TinySessionOptions opts;
    opts.out_path = args.get("out", "");
    opts.policy = args.get("policy", "B4_opencrab");
    opts.journal_dir = args.get("journal-dir", "");
    opts.verify_weights_path = args.get("verify-weights", "");
    opts.allow_insecure_dealer = args.flag("allow-insecure-dealer");
    opts.code_commit = OPENCRAB2_CODE_COMMIT_LITERAL;
    opts.implementation = "opencrab2_sidecar";
    opts.on_record = [](const opencrab2::highlevel::Value& rec) {
      auto cj = opencrab2::highlevel::canonical_json(rec);
      if (cj.ok) {
        std::printf("{\"event\":\"session_record\",\"party\":1,\"record\":%s}\n", cj.value.c_str());
        std::fflush(stdout);
      }
    };
    if (!opts.verify_weights_path.empty())
      std::fprintf(stderr, "[opencrab2-sidecar] TEST MODE: --verify-weights discloses the model weights to P1 (output_match only)\n");
    engine = make_tiny_session_engine(opts);
  } else if (engine_name == "demo") {
    auto dhp = parse_host_port(args.get("dealer", ""));
    std::string keys_path = args.get("keys", "");
    if (!dhp.ok || keys_path.empty()) {
      std::fprintf(stderr, "SIDECAR_ARGS: --dealer host:port and --keys are required for the demo engine\n");
      return usage(argv[0]);
    }
    auto keys = load_keys_file(keys_path);
    if (!keys.ok) {
      std::fprintf(stderr, "%s: keys file %s\n", keys.code.c_str(), keys_path.c_str());
      return 2;
    }
    cfg.dealer_host = dhp.value.first;
    cfg.dealer_port = dhp.value.second;
    cfg.dealer_key = keys.value[1];
    engine = make_demo_beaver_engine();
  } else {
    std::fprintf(stderr, "SIDECAR_ARGS: unknown engine %s\n", engine_name.c_str());
    return 2;
  }

  Sidecar sidecar(cfg, engine);
  auto st = sidecar.start();
  if (!st.ok) {
    std::fprintf(stderr, "%s: cannot start the sidecar on %s:%u\n", st.code.c_str(), cfg.host.c_str(),
                 static_cast<unsigned>(cfg.port));
    return 1;
  }
  std::string token_file = args.get("token-file", "");
  if (!token_file.empty()) {
    if (!detail::write_private_file(token_file, sidecar.token() + "\n")) {
      std::fprintf(stderr, "SIDECAR_IO: cannot write the token file %s\n", token_file.c_str());
      return 1;
    }
  }
  std::string port_file = args.get("port-file", "");
  if (!port_file.empty()) {
    std::ofstream pf(port_file, std::ios::trunc);
    pf << sidecar.port() << "\n";
  }
  json::Value ready = json::Value::object();
  ready.set("event", json::Value::string("sidecar_ready"));
  ready.set("host", json::Value::string(cfg.host));
  ready.set("port", json::Value::integer(sidecar.port()));
  ready.set("engine", json::Value::string(engine_name));
  ready.set("server", json::Value::string(cfg.server_ws_url));
  ready.set("static_dir", json::Value::string(cfg.static_dir.empty() ? "(built-in page)" : cfg.static_dir));
  json::Value origins = json::Value::array();
  for (const auto& o : sidecar.config().origin_allowlist) origins.push(json::Value::string(o));
  ready.set("origins", origins);
  ready.set("launch_url", json::Value::string(sidecar.launch_url()));
  std::printf("%s\n", ready.dump(false).c_str());
  std::fflush(stdout);

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  double t0 = monotonic_s();
  while (!g_stop) {
    sleep_s(0.1);
    if (max_seconds > 0 && monotonic_s() - t0 >= max_seconds) break;
  }
  Sidecar::Stats s = sidecar.stats();
  json::Value web_metrics = sidecar.metrics_json();
  sidecar.stop();
  json::Value done = json::Value::object();
  done.set("metrics", web_metrics);
  done.set("event", json::Value::string("sidecar_stopped"));
  done.set("requests", json::Value::integer(static_cast<int64_t>(s.requests)));
  done.set("rejected_origin", json::Value::integer(static_cast<int64_t>(s.rejected_origin)));
  done.set("rejected_auth", json::Value::integer(static_cast<int64_t>(s.rejected_auth)));
  done.set("jobs_started", json::Value::integer(static_cast<int64_t>(s.jobs_started)));
  done.set("jobs_done", json::Value::integer(static_cast<int64_t>(s.jobs_done)));
  done.set("jobs_cancelled", json::Value::integer(static_cast<int64_t>(s.jobs_cancelled)));
  done.set("jobs_error", json::Value::integer(static_cast<int64_t>(s.jobs_error)));
  std::printf("%s\n", done.dump(false).c_str());
  return 0;
}
