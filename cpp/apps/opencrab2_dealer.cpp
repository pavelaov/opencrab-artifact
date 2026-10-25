// opencrab2_dealer: the isolated dealer process (paper, Appendix B, DealerDelivery).
//
//   opencrab2_dealer --listen 127.0.0.1:0 --keys keys.json [--gen-keys]
//       [--contract NC-v2-f16-l40 | --contract-hash HEX] [--setup-id ID]
//       [--model-version V] [--profile SH2P-TrustedOffline-v1]
//       [--mode ISOLATED_RECIPIENT_BUNDLES | INSECURE_TEST_ONLY_SHARED_SEED --insecure-seed N]
//       [--refill-bytes-per-s N] [--io-timeout S] [--max-seconds S] [--port-file PATH]
//
// Prints one JSON line {"event":"dealer_ready",...} on stdout once bound (a
// launcher parses it), then serves until SIGINT or --max-seconds. Logs carry
// ids, counts and sizes only. The master randomness is the lowlevel CSPRNG
// (opencrab2::lowlevel::Rng seeded by BCryptGenRandom on Windows and
// /dev/urandom elsewhere; the process aborts when the OS source is missing)
// unless the insecure test mode is selected explicitly; bundles produced in
// that mode carry insecure=true and never make an eligible record. The ready
// line states the randomness source.
#include <opencrab2/platform/dealer_service.hpp>
#include <opencrab2/platform/support.hpp>
#include <opencrab2/platform/tcp.hpp>

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
               "usage: %s --listen host:port --keys keys.json [--gen-keys] [--contract NAME | --contract-hash HEX]\n"
               "          [--setup-id ID] [--model-version V] [--profile P] [--mode M] [--insecure-seed N]\n"
               "          [--refill-bytes-per-s N] [--io-timeout S] [--max-seconds S] [--port-file PATH]\n",
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
  std::string listen = args.get("listen", "127.0.0.1:0");
  std::string keys_path = args.get("keys", "");
  if (keys_path.empty()) {
    std::fprintf(stderr, "DEALER_ARGS: --keys is required\n");
    return usage(argv[0]);
  }
  if (args.flag("gen-keys")) {
    std::ifstream probe(keys_path);
    if (!probe) {
      auto w = write_random_keys_file(keys_path);
      if (!w.ok) {
        std::fprintf(stderr, "%s: cannot write %s\n", w.code.c_str(), keys_path.c_str());
        return 2;
      }
      std::fprintf(stderr, "[opencrab2-dealer] wrote fresh per-recipient keys to %s\n", keys_path.c_str());
    }
  }
  auto keys = load_keys_file(keys_path);
  if (!keys.ok) {
    std::fprintf(stderr, "%s: keys file %s\n", keys.code.c_str(), keys_path.c_str());
    return 2;
  }
  auto hp = parse_host_port(listen);
  if (!hp.ok) {
    std::fprintf(stderr, "%s: --listen %s\n", hp.code.c_str(), listen.c_str());
    return 2;
  }
  DealerConfig cfg;
  cfg.recipient_keys = keys.value;
  cfg.setup_id = args.get("setup-id", "setup-0");
  cfg.model_version = args.get("model-version", "gpt2-4L");
  cfg.profile_id = args.get("profile", "SH2P-TrustedOffline-v1");
  cfg.mode = args.get("mode", "ISOLATED_RECIPIENT_BUNDLES");
  if (cfg.mode != "ISOLATED_RECIPIENT_BUNDLES" && cfg.mode != "INSECURE_TEST_ONLY_SHARED_SEED") {
    std::fprintf(stderr, "DEALER_MODE: unknown mode %s\n", cfg.mode.c_str());
    return 2;
  }
  if (args.has("insecure-seed")) {
    if (cfg.mode != "INSECURE_TEST_ONLY_SHARED_SEED") {
      std::fprintf(stderr, "DEALER_MODE: --insecure-seed is admitted only with --mode INSECURE_TEST_ONLY_SHARED_SEED\n");
      return 2;
    }
    cfg.test_seed = static_cast<uint64_t>(args.get_int("insecure-seed", 0));
  }
  auto chash = cli::contract_hash_from_args(args);
  if (!chash.ok) {
    std::fprintf(stderr, "%s: --contract %s\n", chash.code.c_str(), args.get("contract", "").c_str());
    return 2;
  }
  cfg.numeric_contract_hash = chash.value;
  cfg.refill_bytes_per_s = static_cast<uint64_t>(args.get_int("refill-bytes-per-s", 0));
  cfg.io_timeout_s = args.get_double("io-timeout", 30.0);
  double max_seconds = args.get_double("max-seconds", 0.0);

  auto listener = TcpListener::bind(hp.value.first, hp.value.second, 16);
  if (!listener.ok) {
    std::fprintf(stderr, "%s: cannot bind %s\n", listener.code.c_str(), listen.c_str());
    return 1;
  }
  DealerService service(cfg, std::move(listener.value));
  auto st = service.start();
  if (!st.ok) {
    std::fprintf(stderr, "%s: cannot start dealer\n", st.code.c_str());
    return 1;
  }
  std::string port_file = args.get("port-file", "");
  if (!port_file.empty()) {
    std::ofstream pf(port_file, std::ios::trunc);
    pf << service.port() << "\n";
  }
  json::Value ready = json::Value::object();
  ready.set("event", json::Value::string("dealer_ready"));
  ready.set("host", json::Value::string(hp.value.first));
  ready.set("port", json::Value::integer(service.port()));
  ready.set("mode", json::Value::string(cfg.mode));
  ready.set("randomness", json::Value::string(cfg.mode == "ISOLATED_RECIPIENT_BUNDLES"
                                                  ? "os_csprng (lowlevel::Rng: BCryptGenRandom or /dev/urandom seeding ChaCha20)"
                                                  : "INSECURE test seed"));
  ready.set("setup_id", json::Value::string(cfg.setup_id));
  ready.set("model_version", json::Value::string(cfg.model_version));
  ready.set("numeric_contract_hash", json::Value::string(cfg.numeric_contract_hash));
  ready.set("lowlevel_material", json::Value::boolean(DealerService::lowlevel_material_available()));
  json::Value types = json::Value::array();
  for (const auto& t : service.supported_types()) types.push(json::Value::string(t));
  ready.set("material_types", types);
  std::printf("%s\n", ready.dump(false).c_str());
  std::fflush(stdout);

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  double t0 = monotonic_s();
  while (!g_stop) {
    sleep_s(0.1);
    if (max_seconds > 0 && monotonic_s() - t0 >= max_seconds) break;
  }
  DealerService::Stats s = service.stats();
  service.stop();
  json::Value done = json::Value::object();
  done.set("event", json::Value::string("dealer_stopped"));
  done.set("connections", json::Value::integer(static_cast<int64_t>(s.connections)));
  done.set("bundles_sent", json::Value::integer(static_cast<int64_t>(s.bundles_sent)));
  done.set("bytes_sent", json::Value::integer(static_cast<int64_t>(s.bytes_sent)));
  done.set("pairs_generated", json::Value::integer(static_cast<int64_t>(s.pairs_generated)));
  done.set("auth_failures", json::Value::integer(static_cast<int64_t>(s.auth_failures)));
  done.set("requests_rejected", json::Value::integer(static_cast<int64_t>(s.requests_rejected)));
  std::printf("%s\n", done.dump(false).c_str());
  return 0;
}
