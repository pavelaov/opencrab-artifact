// opencrab2/platform/evidence.cpp (see evidence.hpp)
#include <opencrab2/platform/evidence.hpp>
#include <opencrab2/platform/tcp.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <system_error>
#include <thread>

#include <fcntl.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace opencrab2 {
namespace platform {
namespace evidence {

namespace hl = opencrab2::highlevel;

const char* const kSchemaId = "opencrab-native-evidence-v2";
const char* const kProfileId = "SH2P-TrustedOffline-v1";

// ---------------------------------------------------------------------------
// vocabulary (schema.py)
// ---------------------------------------------------------------------------
const std::vector<std::string>& record_fields() {
  static const std::vector<std::string> f = {
      "schema", "status", "security_eligible", "code_commit", "profile_id", "numeric_contract_hash",
      "dealer_mode", "diagnostic_reveals", "workload_scope", "model_revision", "trace_id", "trace_hash",
      "clock_basis", "policy_id", "initial_state_hash", "run_id", "repeat", "initializations",
      "post_init_switches", "rejected_candidates", "aborts", "T_cold_s", "T_warm_s", "T_online_s", "bytes",
      "rounds", "material_generated", "material_transferred", "material_consumed", "material_burned",
      "peak_rss_client", "peak_rss_server", "tokens", "output_match", "raw_measurement_path"};
  return f;
}

const std::vector<std::string>& optional_fields() {
  static const std::vector<std::string> f = {"record_kind", "seed", "family", "variant", "split", "T",
                                             "n_layers", "created_utc", "host", "label", "extra"};
  return f;
}

const std::vector<std::string>& statuses() {
  static const std::vector<std::string> f = {"MEASURED_NATIVE", "EMULATED_NUMERIC", "MODELED", "EXTERNAL_REPORTED",
                                             "DIAGNOSTIC", "HISTORICAL", "ESTIMATED"};
  return f;
}

const std::vector<std::string>& workload_scopes() {
  static const std::vector<std::string> f = {"full-model", "prefix-4L", "prefix-1L", "primitive", "web"};
  return f;
}

const std::vector<std::string>& dealer_modes() {
  static const std::vector<std::string> f = {"ISOLATED_RECIPIENT_BUNDLES", "INSECURE_TEST_ONLY_SHARED_SEED", "NONE"};
  return f;
}

const std::vector<std::string>& clock_bases() {
  static const std::vector<std::string> f = {"wall-clock-exogenous", "modeled-epoch", "process-perf-counter"};
  return f;
}

const std::vector<std::string>& record_kinds() {
  static const std::vector<std::string> f = {"session", "primitive", "calibration", "simulation", "factorial", "web"};
  return f;
}

namespace {

bool in_list(const std::vector<std::string>& l, const std::string& s) { return std::find(l.begin(), l.end(), s) != l.end(); }

const std::vector<std::string>& int_fields() {
  static const std::vector<std::string> f = {"diagnostic_reveals", "repeat", "initializations", "post_init_switches",
                                             "rejected_candidates", "aborts", "bytes", "rounds", "material_generated",
                                             "material_transferred", "material_consumed", "material_burned", "tokens"};
  return f;
}

const std::vector<std::string>& str_fields() {
  static const std::vector<std::string> f = {"schema", "status", "code_commit", "profile_id", "numeric_contract_hash",
                                             "dealer_mode", "workload_scope", "model_revision", "trace_id", "trace_hash",
                                             "clock_basis", "policy_id", "initial_state_hash", "run_id",
                                             "raw_measurement_path"};
  return f;
}

// JSON serialisable without NaN: no bytes, no ndarray, every float finite.
bool json_clean(const hl::Value& v, std::string& why) {
  if (v.is_bytes() || v.is_array()) {
    why = "bytes and arrays are not JSON values";
    return false;
  }
  if (const double* d = std::get_if<double>(&v.v)) {
    if (!std::isfinite(*d)) {
      why = "non-finite float";
      return false;
    }
    return true;
  }
  if (const hl::List* l = std::get_if<hl::List>(&v.v)) {
    for (const auto& e : *l) {
      if (!json_clean(e, why)) return false;
    }
    return true;
  }
  if (const hl::Dict* d = std::get_if<hl::Dict>(&v.v)) {
    for (const auto& e : *d) {
      if (!json_clean(e.val, why)) return false;
    }
  }
  return true;
}

// Makes any value JSON-clean: non-finite floats become null, bytes and arrays their sizes.
hl::Value sanitize(const hl::Value& v) {
  if (const hl::Bytes* b = std::get_if<hl::Bytes>(&v.v)) return hl::dict({{"__bytes__", static_cast<int64_t>(b->size())}});
  if (const hl::NdArray* a = std::get_if<hl::NdArray>(&v.v))
    return hl::dict({{"__ndarray__", static_cast<int64_t>(a->data.size())}});
  if (const double* d = std::get_if<double>(&v.v)) return std::isfinite(*d) ? hl::Value(*d) : hl::Value(hl::None{});
  if (const hl::List* l = std::get_if<hl::List>(&v.v)) {
    hl::List out;
    out.reserve(l->size());
    for (const auto& e : *l) out.push_back(sanitize(e));
    return hl::Value(std::move(out));
  }
  if (const hl::Dict* d = std::get_if<hl::Dict>(&v.v)) {
    hl::Dict out;
    out.reserve(d->size());
    for (const auto& e : *d) out.push_back(hl::DictEntry{e.key, sanitize(e.val)});
    return hl::Value(std::move(out));
  }
  return v;
}

std::string str_of(const hl::Value& rec, const char* key) {
  const hl::Value* v = rec.get(key);
  if (!v) return "";
  auto s = v->as_string();
  return s.ok ? s.value : std::string();
}

int64_t int_of(const hl::Value& rec, const char* key, int64_t def) {
  const hl::Value* v = rec.get(key);
  if (!v || !v->is_int()) return def;
  return v->as_int().value;
}

bool bool_of(const hl::Value& rec, const char* key, bool def) {
  const hl::Value* v = rec.get(key);
  if (!v || !v->is_bool()) return def;
  return v->as_bool().value;
}

Expected<bool> bad(const std::string& code, const std::string& detail) { return make_fail<bool>(code + ": " + detail); }

}  // namespace

// ---------------------------------------------------------------------------
// construction
// ---------------------------------------------------------------------------
std::string utc_now() {
  std::time_t t = std::time(nullptr);
  std::tm tm_utc{};
#ifdef _WIN32
  gmtime_s(&tm_utc, &t);
#else
  gmtime_r(&t, &tm_utc);
#endif
  char buf[32];
  std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
  return buf;
}

hl::Value host_info() {
#ifdef _WIN32
  const char* os = "windows";
#elif defined(__APPLE__)
  const char* os = "darwin";
#else
  const char* os = "linux";
#endif
#if defined(__clang__)
  std::string compiler = std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
  std::string compiler = std::string("gcc ") + __VERSION__;
#else
  std::string compiler = "unknown";
#endif
  return hl::dict({{"os", os},
                   {"hardware_threads", static_cast<int64_t>(std::thread::hardware_concurrency())},
                   {"compiler", compiler},
                   {"pointer_bits", static_cast<int64_t>(sizeof(void*) * 8)}});
}

hl::Value new_record(const std::string& code_commit) {
  const std::string run_id = "run-" + detail::hex_encode(detail::os_random_bytes(8));
  hl::Value r = hl::dict({});
  r.set("schema", kSchemaId);
  r.set("status", "MODELED");
  r.set("security_eligible", false);
  r.set("code_commit", code_commit.empty() ? std::string("cpp-tree-unknown") : code_commit);
  r.set("profile_id", kProfileId);
  r.set("numeric_contract_hash", contract::hash());
  r.set("dealer_mode", "NONE");
  r.set("diagnostic_reveals", static_cast<int64_t>(0));
  r.set("workload_scope", "primitive");
  r.set("model_revision", "n/a");
  r.set("trace_id", "none");
  r.set("trace_hash", "");
  r.set("clock_basis", "modeled-epoch");
  r.set("policy_id", "none");
  r.set("initial_state_hash", "");
  r.set("run_id", run_id);
  for (const char* k : {"repeat", "initializations", "post_init_switches", "rejected_candidates", "aborts"})
    r.set(k, static_cast<int64_t>(0));
  for (const char* k : {"T_cold_s", "T_warm_s", "T_online_s"}) r.set(k, hl::Value(hl::None{}));
  for (const char* k : {"bytes", "rounds", "material_generated", "material_transferred", "material_consumed",
                        "material_burned"})
    r.set(k, static_cast<int64_t>(0));
  r.set("peak_rss_client", hl::Value(hl::None{}));
  r.set("peak_rss_server", hl::Value(hl::None{}));
  r.set("tokens", static_cast<int64_t>(0));
  r.set("output_match", hl::Value(hl::None{}));
  r.set("raw_measurement_path", "");
  r.set("created_utc", utc_now());
  r.set("extra", hl::dict({{"language", "cpp"}}));
  return r;
}

void set_extra(hl::Value& record, const std::string& key, hl::Value value) {
  hl::Value extra = hl::dict({});
  if (const hl::Value* e = record.get("extra")) {
    if (e->is_dict()) extra = *e;
  }
  extra.set(key, sanitize(value));
  record.set("extra", std::move(extra));
}

hl::Value conform(const hl::Value& record, const std::string& record_kind, const std::string& code_commit) {
  hl::Value out = new_record(code_commit);
  hl::Value extra = hl::dict({});
  bool had_created = false;
  if (const hl::Dict* d = std::get_if<hl::Dict>(&record.v)) {
    if (const hl::Value* e = record.get("extra")) {
      if (const hl::Dict* ed = std::get_if<hl::Dict>(&e->v)) {
        for (const auto& kv : *ed) extra.set(kv.key, kv.val);
      }
    }
    for (const auto& kv : *d) {
      if (kv.key == "extra") continue;
      if (kv.key == "tokens") {
        if (const hl::List* l = std::get_if<hl::List>(&kv.val.v)) {
          out.set("tokens", static_cast<int64_t>(l->size()));
          extra.set("tokens_list", kv.val);
        } else if (kv.val.is_int()) {
          out.set("tokens", kv.val);
        } else {
          extra.set("tokens_raw", kv.val);
        }
        continue;
      }
      if (kv.key == "created_utc") had_created = true;
      if (in_list(record_fields(), kv.key) || in_list(optional_fields(), kv.key)) {
        out.set(kv.key, kv.val);
      } else {
        extra.set(kv.key, kv.val);
      }
    }
  }
  out.set("schema", kSchemaId);
  if (!in_list(record_kinds(), str_of(out, "record_kind"))) out.set("record_kind", record_kind);
  if (str_of(out, "code_commit").empty()) out.set("code_commit", code_commit.empty() ? std::string("cpp-tree-unknown") : code_commit);
  if (!had_created) out.set("created_utc", utc_now());
  // timings: a non-finite value is "not measured"
  for (const char* k : {"T_cold_s", "T_warm_s", "T_online_s"}) {
    if (const hl::Value* v = out.get(k)) {
      if (const double* dv = std::get_if<double>(&v->v)) {
        if (!std::isfinite(*dv) || *dv < 0.0) out.set(k, hl::Value(hl::None{}));
      }
    }
  }
  extra.set("language", "cpp");
  out.set("extra", sanitize(extra));
  return out;
}

void withdraw_eligibility(hl::Value& record, const std::string& reason, bool diagnostic) {
  record.set("security_eligible", false);
  if (diagnostic && str_of(record, "status") == "MEASURED_NATIVE") record.set("status", "DIAGNOSTIC");
  hl::List notes;
  if (const hl::Value* e = record.get("extra")) {
    if (const hl::Value* n = e->get("eligibility_notes")) {
      if (const hl::List* l = std::get_if<hl::List>(&n->v)) notes = *l;
    }
  }
  notes.push_back(hl::Value(reason));
  set_extra(record, "eligibility_notes", hl::Value(std::move(notes)));
}

std::vector<std::string> enforce(hl::Value& record) {
  std::vector<std::string> reasons;
  const std::string mode = str_of(record, "dealer_mode");
  if (mode == "INSECURE_TEST_ONLY_SHARED_SEED") {
    // Evidence class: any record produced with the shared seed is DIAGNOSTIC and not eligible.
    if (str_of(record, "status") != "DIAGNOSTIC" || bool_of(record, "security_eligible", false)) {
      reasons.push_back("SHARED_SEED_DEALER: dealer_mode INSECURE_TEST_ONLY_SHARED_SEED makes the record DIAGNOSTIC");
      record.set("status", "DIAGNOSTIC");
      record.set("security_eligible", false);
    }
  }
  if (bool_of(record, "security_eligible", false)) {
    std::string why;
    if (mode != "ISOLATED_RECIPIENT_BUNDLES") why = "ELIGIBILITY_DEALER_MODE: dealer_mode is not ISOLATED_RECIPIENT_BUNDLES";
    else if (int_of(record, "diagnostic_reveals", 1) != 0) why = "ELIGIBILITY_DIAGNOSTIC_REVEALS: diagnostic_reveals != 0";
    else if (str_of(record, "status") != "MEASURED_NATIVE") why = "ELIGIBILITY_STATUS: status is not MEASURED_NATIVE";
    else if (int_of(record, "aborts", 1) != 0) why = "ELIGIBILITY_ABORTS: the run did not complete (aborts != 0)";
    if (!why.empty()) {
      reasons.push_back(why);
      record.set("security_eligible", false);
    }
  }
  if (!reasons.empty()) {
    hl::List notes;
    if (const hl::Value* e = record.get("extra")) {
      if (const hl::Value* n = e->get("eligibility_notes")) {
        if (const hl::List* l = std::get_if<hl::List>(&n->v)) notes = *l;
      }
    }
    for (const auto& r : reasons) notes.push_back(hl::Value(r));
    set_extra(record, "eligibility_notes", hl::Value(std::move(notes)));
  }
  return reasons;
}

// ---------------------------------------------------------------------------
// validation (schema.validate)
// ---------------------------------------------------------------------------
Expected<bool> validate(const hl::Value& record) {
  const hl::Dict* d = std::get_if<hl::Dict>(&record.v);
  if (!d) return bad("SCHEMA_BAD_TYPE", "a record must be a mapping");
  for (const auto& f : record_fields()) {
    if (!record.has(f)) return bad("SCHEMA_MISSING_FIELD", "record lacks " + f);
  }
  for (const auto& kv : *d) {
    if (!in_list(record_fields(), kv.key) && !in_list(optional_fields(), kv.key))
      return bad("SCHEMA_UNKNOWN_FIELD", kv.key + " is not a record field");
  }
  for (const auto& f : str_fields()) {
    if (!record.get(f)->is_string()) return bad("SCHEMA_BAD_TYPE", f + " must be a string");
  }
  if (str_of(record, "schema") != kSchemaId) return bad("SCHEMA_BAD_VALUE", std::string("schema must be ") + kSchemaId);
  const std::string status = str_of(record, "status");
  if (!in_list(statuses(), status)) return bad("SCHEMA_BAD_STATUS", status + " is not a status");
  if (!in_list(workload_scopes(), str_of(record, "workload_scope")))
    return bad("SCHEMA_BAD_SCOPE", str_of(record, "workload_scope") + " is not a workload scope");
  if (!in_list(dealer_modes(), str_of(record, "dealer_mode")))
    return bad("SCHEMA_BAD_VALUE", "dealer_mode " + str_of(record, "dealer_mode") + " is not a dealer mode");
  if (!in_list(clock_bases(), str_of(record, "clock_basis")))
    return bad("SCHEMA_BAD_VALUE", "clock_basis " + str_of(record, "clock_basis") + " is not a clock basis");
  if (str_of(record, "run_id").empty()) return bad("SCHEMA_BAD_VALUE", "run_id must be non-empty");
  if (!record.get("security_eligible")->is_bool()) return bad("SCHEMA_BAD_TYPE", "security_eligible must be a bool");
  for (const auto& f : int_fields()) {
    const hl::Value* v = record.get(f);
    if (!v->is_int() || v->as_int().value < 0) return bad("SCHEMA_BAD_TYPE", f + " must be a non-negative int");
  }
  for (const char* f : {"T_cold_s", "T_warm_s", "T_online_s"}) {
    const hl::Value* v = record.get(f);
    if (v->is_none()) continue;
    if (v->is_int()) {
      if (v->as_int().value < 0) return bad("SCHEMA_BAD_TYPE", std::string(f) + " must be None or a finite non-negative number");
      continue;
    }
    const double* dv = std::get_if<double>(&v->v);
    if (!dv || !std::isfinite(*dv) || *dv < 0.0)
      return bad("SCHEMA_BAD_TYPE", std::string(f) + " must be None or a finite non-negative number");
  }
  for (const char* f : {"peak_rss_client", "peak_rss_server"}) {
    const hl::Value* v = record.get(f);
    if (v->is_none()) continue;
    if (!v->is_int() || v->as_int().value < 0) return bad("SCHEMA_BAD_TYPE", std::string(f) + " must be None or a non-negative int");
  }
  const hl::Value* om = record.get("output_match");
  if (!om->is_none() && !om->is_bool()) return bad("SCHEMA_BAD_TYPE", "output_match must be None or a bool");
  if (const hl::Value* rk = record.get("record_kind")) {
    auto s = rk->as_string();
    if (!s.ok || !in_list(record_kinds(), s.value)) return bad("SCHEMA_BAD_VALUE", "record_kind is not a record kind");
  }
  if (const hl::Value* ex = record.get("extra")) {
    if (!ex->is_dict()) return bad("SCHEMA_BAD_TYPE", "extra must be a mapping");
  }
  if (bool_of(record, "security_eligible", false)) {
    if (str_of(record, "dealer_mode") != "ISOLATED_RECIPIENT_BUNDLES")
      return bad("SCHEMA_ELIGIBILITY", "security_eligible requires dealer_mode ISOLATED_RECIPIENT_BUNDLES");
    if (int_of(record, "diagnostic_reveals", 1) != 0) return bad("SCHEMA_ELIGIBILITY", "security_eligible requires diagnostic_reveals == 0");
    if (status != "MEASURED_NATIVE") return bad("SCHEMA_ELIGIBILITY", "security_eligible requires status MEASURED_NATIVE");
    if (int_of(record, "aborts", 1) != 0) return bad("SCHEMA_ELIGIBILITY", "security_eligible requires a completed run (aborts == 0)");
  }
  std::string why;
  if (!json_clean(record, why)) return bad("SCHEMA_BAD_VALUE", "record is not JSON serialisable without NaN (" + why + ")");
  return make_ok(true);
}

// ---------------------------------------------------------------------------
// locked append (schema.locked_append_lines)
// ---------------------------------------------------------------------------
namespace {

constexpr double kLockTimeoutS = 30.0;
constexpr double kLockStaleS = 120.0;

bool try_create_lock(const std::string& lock) {
#ifdef _WIN32
  int fd = ::_open(lock.c_str(), _O_CREAT | _O_EXCL | _O_WRONLY, _S_IREAD | _S_IWRITE);
  if (fd < 0) return false;
  ::_close(fd);
#else
  int fd = ::open(lock.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0644);
  if (fd < 0) return false;
  ::close(fd);
#endif
  return true;
}

double lock_age_s(const std::string& lock) {
  struct stat st;
  if (::stat(lock.c_str(), &st) != 0) return -1.0;
  return std::difftime(std::time(nullptr), st.st_mtime);
}

}  // namespace

Expected<bool> append(const std::string& path, const hl::Value& record) {
  auto v = validate(record);
  if (!v.ok) return v;
  auto text = hl::canonical_json(record);
  if (!text.ok) return make_fail<bool>("SCHEMA_BAD_VALUE: " + text.code);
  std::string line = text.value;
  line.push_back('\n');
  const std::string lock = path + ".lock";
  const double deadline = monotonic_s() + kLockTimeoutS;
  while (!try_create_lock(lock)) {
    if (lock_age_s(lock) > kLockStaleS) {
      std::remove(lock.c_str());  // left behind by a crashed process
      continue;
    }
    if (monotonic_s() > deadline) return make_fail<bool>("RECORD_LOCK_TIMEOUT: could not acquire " + lock);
    sleep_s(0.005);
  }
  bool ok = false;
  // A file that does not end with a newline holds the fragment of an append that died: start on
  // a new line so that this record is never glued onto it (schema.locked_append_lines does the same).
  if (std::FILE* rf = std::fopen(path.c_str(), "rb")) {
    if (std::fseek(rf, -1, SEEK_END) == 0) {
      const int last = std::fgetc(rf);
      if (last != EOF && last != '\n') line.insert(line.begin(), '\n');
    }
    std::fclose(rf);
  }
  std::FILE* f = std::fopen(path.c_str(), "ab");
  if (f) {
    const size_t n = std::fwrite(line.data(), 1, line.size(), f);
    std::fflush(f);
#ifdef _WIN32
    ::_commit(::_fileno(f));
#else
    ::fsync(::fileno(f));
#endif
    ok = (std::fclose(f) == 0) && n == line.size();
  }
  std::remove(lock.c_str());
  if (!ok) return make_fail<bool>("RECORD_IO: cannot append to " + path);
  return make_ok(true);
}

std::string record_path(const std::string& path) {
  if (path.empty()) return path;
  const std::filesystem::path given(path);
  if (!given.has_root_directory()) return given.generic_string();  // relative: as given, forward slashes
  std::error_code ec;
  const std::filesystem::path abs = std::filesystem::absolute(given, ec).lexically_normal();
  if (ec) return given.filename().generic_string();
  // 1. relative to the data root of the run, when the file lies inside it
  const char* env = std::getenv("OPENCRAB2_RESULTS_DIR");
  if (env != nullptr && *env != '\0') {
    std::filesystem::path root = std::filesystem::absolute(std::filesystem::path(env), ec).lexically_normal();
    if (!ec && !root.has_filename() && root.has_relative_path()) root = root.parent_path();   // a trailing separator
    if (!ec) {
      const std::filesystem::path rel = abs.lexically_relative(root);
      if (!rel.empty() && rel != "." && *rel.begin() != "..") return rel.generic_string();
    }
  }
  // 2. cut at the deepest directory above the file named like a subdirectory of a data root
  std::vector<std::filesystem::path> parts(abs.begin(), abs.end());
  for (size_t i = parts.size() > 1 ? parts.size() - 1 : 0; i-- > 0;) {
    const std::string name = parts[i].generic_string();
    if (name != "raw" && name != "calibration" && name != "processed" && name != "tables") continue;
    std::filesystem::path rel;
    for (size_t k = i; k < parts.size(); ++k) rel /= parts[k];
    return rel.generic_string();
  }
  // 3. the file name alone
  return abs.filename().generic_string();
}

}  // namespace evidence
}  // namespace platform
}  // namespace opencrab2
