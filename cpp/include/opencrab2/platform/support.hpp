// opencrab2/platform/support.hpp
//
// Support header of the platform layer. It carries NO cryptographic, codec or
// JSON implementation of its own any more: each of those has exactly one
// implementation in the tree and this header only gives it the spelling the
// platform sources use.
//
//   what                          the one implementation
//   ----------------------------  ---------------------------------------------
//   SHA-1, SHA-256, HMAC-SHA256   opencrab2::lowlevel::Sha1 / Sha256 / hmac_sha256
//   base64, hex                   opencrab2::lowlevel::base64_* / to_hex / from_hex
//   OS randomness, CSPRNG stream  opencrab2::lowlevel::os_random_bytes / Rng
//                                 (BCryptGenRandom or /dev/urandom seeding ChaCha20)
//   wire codec                    opencrab2::highlevel::encode_value / decode_value
//   JSON text (parse, canonical)  opencrab2::highlevel::Json::parse / Json::dump
//   numeric contract hash         opencrab2::highlevel::NumericContract::contract_hash
//   channel statistics            opencrab2::highlevel::ChannelStats
//
// What stays here because it has no other owner: the small result helpers
// (make_ok / make_fail / forward_fail), the two value containers the platform
// sources are written against (json::Value, wire::Value: plain trees with
// accessors, converted to the highlevel types for every encode, decode, parse
// and dump), ASCII helpers and the command line parser of the apps.
//
// Codes surfaced through these adapters: HEX_LENGTH, HEX_ALPHABET,
// BASE64_LENGTH, BASE64_PADDING, BASE64_ALPHABET (lowlevel); JSON_PARSE,
// JSON_DEPTH (highlevel Json); CODEC_TRUNCATED, CODEC_DEPTH, CODEC_DTYPE,
// CODEC_SHAPE, CODEC_TAG, CODEC_TRAILING, CODEC_MALFORMED (highlevel codec);
// WIRE_TRUNCATED, WIRE_BAD_LENGTH (the 8-byte frame prefix check done here);
// CONTRACT_UNKNOWN. A failing OS CSPRNG aborts the process (fail closed, never
// a weak default).
//
// Tested (tests/test_platform.cpp): the FIPS / RFC vectors through these
// spellings, byte equality of wire::encode with the reference Python bytes,
// canonical JSON equality with Python json.dumps(sort_keys=True,
// separators=(",", ":")), nesting-depth rejection of both decoders, and the
// contract hashes of both contracts.
//
// No em dashes anywhere in this file (the tests of the tree enforce it).
#pragma once

#include <opencrab2/expected.hpp>
#include <opencrab2/highlevel/channel.hpp>
#include <opencrab2/highlevel/codec.hpp>
#include <opencrab2/highlevel/contract.hpp>
#include <opencrab2/highlevel/json.hpp>
#include <opencrab2/lowlevel/base64.hpp>
#include <opencrab2/lowlevel/csprng.hpp>
#include <opencrab2/lowlevel/hmac.hpp>
#include <opencrab2/lowlevel/sha1.hpp>
#include <opencrab2/lowlevel/sha256.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace opencrab2 {
namespace platform {

// Result helpers that only touch the three public members of Expected<T>.
template <class T>
inline Expected<T> make_ok(T v) {
  Expected<T> e;
  e.ok = true;
  e.value = std::move(v);
  e.code.clear();
  return e;
}

template <class T>
inline Expected<T> make_fail(std::string code) {
  Expected<T> e;
  e.ok = false;
  e.code = std::move(code);
  return e;
}

template <class T, class U>
inline Expected<T> forward_fail(const Expected<U>& from) {
  return make_fail<T>(from.code);
}

// One statistics type for every channel of the tree (the Python ChannelStats).
using ChannelStats = highlevel::ChannelStats;

namespace detail {

// ---------------------------------------------------------------------------
// Hex, constant time compare, OS randomness (lowlevel)
// ---------------------------------------------------------------------------
inline std::string hex_encode(const uint8_t* p, size_t n) { return lowlevel::to_hex(p, n); }
inline std::string hex_encode(const std::vector<uint8_t>& v) { return lowlevel::to_hex(v); }
inline Expected<std::vector<uint8_t>> hex_decode(const std::string& s) { return lowlevel::from_hex(s); }
inline int hex_nibble(char c) { return lowlevel::hex_value(c); }  // -1 for a non-hex character

// Constant time for equal lengths; a length mismatch returns false at once,
// which leaks only the (public) length.
inline bool constant_time_equal(const uint8_t* a, const uint8_t* b, size_t n) {
  return lowlevel::constant_time_equal(a, b, n);
}
inline bool constant_time_equal(const std::string& a, const std::string& b) {
  if (a.size() != b.size()) return false;
  return lowlevel::constant_time_equal(reinterpret_cast<const uint8_t*>(a.data()),
                                       reinterpret_cast<const uint8_t*>(b.data()), a.size());
}

// n bytes of the operating system CSPRNG (BCryptGenRandom / /dev/urandom).
// Aborts when the OS source is unavailable: no caller may continue with
// predictable nonces, tokens or keys.
inline std::vector<uint8_t> os_random_bytes(size_t n) {
  std::vector<uint8_t> out(n);
  const Status st = lowlevel::os_random_bytes(out);
  if (!st.ok) {
    std::fprintf(stderr, "%s: operating system CSPRNG not available\n", st.code.c_str());
    std::abort();
  }
  return out;
}

// ---------------------------------------------------------------------------
// Hashes, HMAC, base64 (lowlevel)
// ---------------------------------------------------------------------------
inline std::array<uint8_t, 20> sha1(const uint8_t* data, size_t len) { return lowlevel::Sha1::digest(data, len); }
inline std::array<uint8_t, 20> sha1(const std::string& s) { return lowlevel::Sha1::digest(s); }

inline std::array<uint8_t, 32> sha256(const uint8_t* data, size_t len) { return lowlevel::Sha256::digest(data, len); }
inline std::array<uint8_t, 32> sha256(const std::string& s) { return lowlevel::Sha256::digest(s); }
inline std::array<uint8_t, 32> sha256(const std::vector<uint8_t>& v) { return lowlevel::Sha256::digest(v.data(), v.size()); }

inline std::array<uint8_t, 32> hmac_sha256(const uint8_t* key, size_t keylen, const uint8_t* msg, size_t msglen) {
  return lowlevel::hmac_sha256(key, keylen, msg, msglen);
}
inline std::array<uint8_t, 32> hmac_sha256(const std::vector<uint8_t>& key, const std::vector<uint8_t>& msg) {
  return lowlevel::hmac_sha256(key.data(), key.size(), msg.data(), msg.size());
}
inline std::array<uint8_t, 32> hmac_sha256(const std::vector<uint8_t>& key, const std::string& msg) {
  return lowlevel::hmac_sha256(key, msg);
}
inline std::string hmac_sha256_hex(const std::vector<uint8_t>& key, const std::string& msg) {
  return lowlevel::to_hex(lowlevel::hmac_sha256(key, msg));
}

inline std::string base64_encode(const uint8_t* p, size_t n) { return lowlevel::base64_encode(p, n); }
inline std::string base64_encode(const std::vector<uint8_t>& v) { return lowlevel::base64_encode(v); }
inline Expected<std::vector<uint8_t>> base64_decode(const std::string& s) { return lowlevel::base64_decode(s); }

// ---------------------------------------------------------------------------
// The platform's CSPRNG stream: the lowlevel Rng (ChaCha20, RFC 7539) with the
// two bulk helpers the platform sources use. A default-constructed stream is
// seeded from the OS (a forgotten initialisation can never yield a predictable
// stream); from_seed exists for INSECURE_TEST_ONLY_SHARED_SEED and tests.
// ---------------------------------------------------------------------------
class ChaCha20 : public lowlevel::Rng {
 public:
  ChaCha20() : lowlevel::Rng(lowlevel::Rng::from_os_or_abort()) {}
  explicit ChaCha20(const lowlevel::Rng& r) : lowlevel::Rng(r) {}
  static ChaCha20 from_os() { return ChaCha20(lowlevel::Rng::from_os_or_abort()); }
  static ChaCha20 from_seed(uint64_t seed) { return ChaCha20(lowlevel::Rng::from_seed(seed)); }
  void fill_u64(uint64_t* p, size_t n) {
    for (size_t i = 0; i < n; ++i) p[i] = next_u64();
  }
  std::vector<uint8_t> bytes(size_t n) {
    std::vector<uint8_t> v(n);
    fill(v.data(), n);
    return v;
  }
};

// ---------------------------------------------------------------------------
// ASCII helpers (header names, scheme/host comparisons, option lists)
// ---------------------------------------------------------------------------
inline std::string ascii_lower(std::string s) {
  for (auto& c : s) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  }
  return s;
}

inline std::string trim(const std::string& s) {
  size_t a = 0, b = s.size();
  while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r' || s[a] == '\n')) ++a;
  while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r' || s[b - 1] == '\n')) --b;
  return s.substr(a, b - a);
}

inline std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  size_t start = 0;
  for (size_t i = 0; i <= s.size(); ++i) {
    if (i == s.size() || s[i] == sep) {
      out.push_back(s.substr(start, i - start));
      start = i + 1;
    }
  }
  return out;
}

// Writes `text` to a file that holds a secret (recipient keys, the bearer
// token). On POSIX the file is created with mode 0600 and an existing file is
// set to 0600 before the first byte is written, so no other user can read it
// whatever the umask is; on Windows it gets the access control list of its
// directory. Returns false when the file cannot be created or written in full.
bool write_private_file(const std::string& path, const std::string& text);

}  // namespace detail

// ---------------------------------------------------------------------------
// json::Value: the JSON tree the platform sources build and read. It is a
// plain container; the text form is produced and parsed by the highlevel Json
// type (support.cpp), so dump() is byte-identical to Python's
// json.dumps(obj, sort_keys=True, separators=(",", ":")) with ensure_ascii,
// floats in Python repr form, and parse() carries the highlevel depth limit.
// ---------------------------------------------------------------------------
namespace json {

class Value {
 public:
  enum class Kind { Null, Bool, Number, String, Array, Object };
  Kind kind = Kind::Null;
  bool b = false;
  double num = 0.0;
  bool is_int = false;
  int64_t inum = 0;
  std::string str;
  std::vector<Value> arr;
  std::vector<std::pair<std::string, Value>> obj;

  Value() = default;
  static Value null() { return Value(); }
  static Value boolean(bool v) {
    Value x;
    x.kind = Kind::Bool;
    x.b = v;
    return x;
  }
  static Value number(double v) {
    Value x;
    x.kind = Kind::Number;
    x.num = v;
    x.is_int = false;
    return x;
  }
  static Value integer(int64_t v) {
    Value x;
    x.kind = Kind::Number;
    x.inum = v;
    x.num = static_cast<double>(v);
    x.is_int = true;
    return x;
  }
  static Value string(std::string v) {
    Value x;
    x.kind = Kind::String;
    x.str = std::move(v);
    return x;
  }
  static Value array() {
    Value x;
    x.kind = Kind::Array;
    return x;
  }
  static Value object() {
    Value x;
    x.kind = Kind::Object;
    return x;
  }

  bool is_null() const { return kind == Kind::Null; }
  bool is_bool() const { return kind == Kind::Bool; }
  bool is_number() const { return kind == Kind::Number; }
  bool is_string() const { return kind == Kind::String; }
  bool is_array() const { return kind == Kind::Array; }
  bool is_object() const { return kind == Kind::Object; }

  double as_double() const { return is_int ? static_cast<double>(inum) : num; }
  int64_t as_int() const { return is_int ? inum : static_cast<int64_t>(num); }

  const Value* find(const std::string& key) const {
    if (kind != Kind::Object) return nullptr;
    for (const auto& kv : obj) {
      if (kv.first == key) return &kv.second;
    }
    return nullptr;
  }

  Value& set(const std::string& key, Value v) {
    if (kind != Kind::Object) {
      kind = Kind::Object;
      obj.clear();
    }
    for (auto& kv : obj) {
      if (kv.first == key) {
        kv.second = std::move(v);
        return kv.second;
      }
    }
    obj.emplace_back(key, std::move(v));
    return obj.back().second;
  }

  Value& push(Value v) {
    if (kind != Kind::Array) {
      kind = Kind::Array;
      arr.clear();
    }
    arr.push_back(std::move(v));
    return arr.back();
  }

  std::string str_or(const std::string& key, const std::string& def) const {
    const Value* v = find(key);
    return (v && v->is_string()) ? v->str : def;
  }
  int64_t int_or(const std::string& key, int64_t def) const {
    const Value* v = find(key);
    return (v && v->is_number()) ? v->as_int() : def;
  }
  double double_or(const std::string& key, double def) const {
    const Value* v = find(key);
    return (v && v->is_number()) ? v->as_double() : def;
  }
  bool bool_or(const std::string& key, bool def) const {
    const Value* v = find(key);
    return (v && v->is_bool()) ? v->b : def;
  }

  // Conversions to and from the highlevel Json (objects become key-sorted).
  highlevel::Json to_highlevel() const;
  static Value from_highlevel(const highlevel::Json& j);

  // Compact JSON text through highlevel::Json::dump(): sorted keys, no
  // spaces, ASCII only. The flag is kept for source compatibility; both
  // spellings produce the canonical text.
  std::string dump(bool canonical = true) const;
};

// highlevel::Json::parse; codes JSON_PARSE and JSON_DEPTH (without detail).
Expected<Value> parse(const std::string& text);

}  // namespace json

// ---------------------------------------------------------------------------
// wire::Value: the typed value of the wire codec (codec.hpp)
// in the shape the platform sources use. encode / decode convert to
// highlevel::Value and call the one codec implementation (codec.hpp), so the
// bytes are those of the Python Channel and peer frames meet the highlevel
// nesting-depth limit (CODEC_DEPTH).
// ---------------------------------------------------------------------------
namespace wire {

enum class Tag : uint8_t { Bytes = 1, NdArray = 2, Dict = 3, Int = 4, Float = 5, Str = 6, List = 7, None = 8, Bool = 9 };
enum class DType : uint8_t { U64 = 1, I64 = 2, F64 = 3, U8 = 4, I32 = 5 };

inline size_t dtype_size(DType d) { return highlevel::dtype_size(static_cast<highlevel::DType>(d)); }

struct NdArray {
  DType dtype = DType::U64;
  std::vector<uint64_t> shape;
  std::vector<uint8_t> data;  // little-endian element bytes

  uint64_t count() const {
    uint64_t n = 1;
    for (auto d : shape) n *= d;
    return shape.empty() ? 1 : n;
  }
  static NdArray from_u64(const std::vector<uint64_t>& v, std::vector<uint64_t> shape = {});
  std::vector<uint64_t> to_u64() const;
};

struct Value {
  Tag tag = Tag::None;
  std::vector<uint8_t> bytes;
  NdArray array;
  std::vector<std::pair<std::string, Value>> dict;
  int64_t i = 0;
  double f = 0.0;
  std::string s;
  std::vector<Value> list;
  bool b = false;

  static Value none() { return Value(); }
  static Value from_bytes(std::vector<uint8_t> v) {
    Value x;
    x.tag = Tag::Bytes;
    x.bytes = std::move(v);
    return x;
  }
  static Value from_array(NdArray a) {
    Value x;
    x.tag = Tag::NdArray;
    x.array = std::move(a);
    return x;
  }
  static Value from_u64_vector(const std::vector<uint64_t>& v) { return from_array(NdArray::from_u64(v)); }
  static Value from_int(int64_t v) {
    Value x;
    x.tag = Tag::Int;
    x.i = v;
    return x;
  }
  static Value from_float(double v) {
    Value x;
    x.tag = Tag::Float;
    x.f = v;
    return x;
  }
  static Value from_str(std::string v) {
    Value x;
    x.tag = Tag::Str;
    x.s = std::move(v);
    return x;
  }
  static Value from_bool(bool v) {
    Value x;
    x.tag = Tag::Bool;
    x.b = v;
    return x;
  }
  static Value from_list(std::vector<Value> v) {
    Value x;
    x.tag = Tag::List;
    x.list = std::move(v);
    return x;
  }
  static Value make_dict() {
    Value x;
    x.tag = Tag::Dict;
    return x;
  }

  bool is_dict() const { return tag == Tag::Dict; }
  bool is_str() const { return tag == Tag::Str; }
  bool is_int() const { return tag == Tag::Int; }
  bool is_bytes() const { return tag == Tag::Bytes; }
  bool is_array() const { return tag == Tag::NdArray; }
  bool is_list() const { return tag == Tag::List; }
  bool is_none() const { return tag == Tag::None; }
  bool is_bool() const { return tag == Tag::Bool; }
  bool is_float() const { return tag == Tag::Float; }

  Value& set(const std::string& key, Value v) {
    tag = Tag::Dict;
    for (auto& kv : dict) {
      if (kv.first == key) {
        kv.second = std::move(v);
        return kv.second;
      }
    }
    dict.emplace_back(key, std::move(v));
    return dict.back().second;
  }
  const Value* get(const std::string& key) const {
    if (tag != Tag::Dict) return nullptr;
    for (const auto& kv : dict) {
      if (kv.first == key) return &kv.second;
    }
    return nullptr;
  }
  bool has(const std::string& key) const { return get(key) != nullptr; }
  std::string str_or(const std::string& key, const std::string& def) const {
    const Value* v = get(key);
    return (v && v->tag == Tag::Str) ? v->s : def;
  }
  int64_t int_or(const std::string& key, int64_t def) const {
    const Value* v = get(key);
    return (v && v->tag == Tag::Int) ? v->i : def;
  }
  double float_or(const std::string& key, double def) const {
    const Value* v = get(key);
    if (!v) return def;
    if (v->tag == Tag::Float) return v->f;
    if (v->tag == Tag::Int) return static_cast<double>(v->i);
    return def;
  }
  bool bool_or(const std::string& key, bool def) const {
    const Value* v = get(key);
    return (v && v->tag == Tag::Bool) ? v->b : def;
  }

  // Dict/list/str/int/float/bool/none values convert to JSON for logging.
  // Bytes and arrays are represented by their sizes only, never their
  // contents (no payload bytes in logs).
  json::Value to_json() const;
  static Value from_json(const json::Value& j);

  // Structural conversions to and from the highlevel typed value (no bytes
  // are re-encoded; dict order is kept).
  highlevel::Value to_highlevel() const;
  static Value from_highlevel(const highlevel::Value& v);
};

// Payload bytes (no length prefix): highlevel::encode_value.
std::vector<uint8_t> encode(const Value& v);
// Payload with the 8-byte big-endian length prefix: highlevel::encode_frame.
std::vector<uint8_t> frame(const Value& v);
// highlevel::decode_value; codes CODEC_* (without detail).
Expected<Value> decode(const uint8_t* p, size_t n);
Expected<Value> decode(const std::vector<uint8_t>& v);
// Strips and validates the 8-byte prefix (WIRE_TRUNCATED, WIRE_BAD_LENGTH), then decode.
Expected<Value> decode_frame(const std::vector<uint8_t>& f);

}  // namespace wire

// ---------------------------------------------------------------------------
// Numeric contract hash by contract name, from
// highlevel::NumericContract. An unknown name fails closed (CONTRACT_UNKNOWN);
// the unchecked spelling returns an empty string for it, which matches no
// binding.
// ---------------------------------------------------------------------------
namespace contract {

inline Expected<std::string> hash_checked(const std::string& name) {
  auto c = highlevel::contract_by_name(name);
  if (!c.ok) return make_fail<std::string>(c.code.empty() ? std::string("CONTRACT_UNKNOWN") : c.code);
  return make_ok(c.value.contract_hash());
}

inline std::string hash(const std::string& name = "NC-v2-f16-l40") {
  auto h = hash_checked(name);
  return h.ok ? h.value : std::string();
}

}  // namespace contract

// ---------------------------------------------------------------------------
// Hand-rolled argument parser for the apps: --key value, --key=value, --flag,
// repeated keys collect into a list, positionals kept in order.
// ---------------------------------------------------------------------------
namespace cli {

class Args {
 public:
  Args(int argc, char** argv) {
    if (argc > 0) prog_ = argv[0];
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a.size() > 2 && a.compare(0, 2, "--") == 0) {
        std::string key = a.substr(2);
        std::string val;
        size_t eq = key.find('=');
        if (eq != std::string::npos) {
          val = key.substr(eq + 1);
          key = key.substr(0, eq);
          kv_.emplace_back(key, val);
        } else if (i + 1 < argc && std::string(argv[i + 1]).compare(0, 2, "--") != 0) {
          kv_.emplace_back(key, argv[++i]);
        } else {
          kv_.emplace_back(key, "");
          flags_.push_back(key);
        }
      } else {
        positional_.push_back(a);
      }
    }
  }
  bool has(const std::string& key) const {
    for (const auto& kv : kv_) {
      if (kv.first == key) return true;
    }
    return false;
  }
  std::string get(const std::string& key, const std::string& def = "") const {
    std::string out = def;
    for (const auto& kv : kv_) {
      if (kv.first == key) out = kv.second;
    }
    return out;
  }
  std::vector<std::string> get_all(const std::string& key) const {
    std::vector<std::string> out;
    for (const auto& kv : kv_) {
      if (kv.first == key && !kv.second.empty()) out.push_back(kv.second);
    }
    return out;
  }
  int64_t get_int(const std::string& key, int64_t def) const {
    if (!has(key)) return def;
    std::string v = get(key);
    if (v.empty()) return def;
    return std::strtoll(v.c_str(), nullptr, 10);
  }
  double get_double(const std::string& key, double def) const {
    if (!has(key)) return def;
    std::string v = get(key);
    if (v.empty()) return def;
    return std::strtod(v.c_str(), nullptr);
  }
  bool flag(const std::string& key) const {
    for (const auto& f : flags_) {
      if (f == key) return true;
    }
    std::string v = get(key, "");
    return v == "1" || v == "true" || v == "yes";
  }
  const std::vector<std::string>& positional() const { return positional_; }
  const std::string& prog() const { return prog_; }
  // Keys given on the command line that are not in `known` (typo guard of the apps).
  std::vector<std::string> unknown_keys(const std::vector<std::string>& known) const {
    std::vector<std::string> out;
    for (const auto& kv : kv_) {
      bool found = false;
      for (const auto& k : known) {
        if (k == kv.first) found = true;
      }
      if (!found) out.push_back(kv.first);
    }
    return out;
  }

 private:
  std::string prog_;
  std::vector<std::pair<std::string, std::string>> kv_;
  std::vector<std::string> flags_;
  std::vector<std::string> positional_;
};

// The contract hash selected by --contract-hash HEX or --contract NAME
// (default NC-v2-f16-l40); CONTRACT_UNKNOWN for an unknown name.
inline Expected<std::string> contract_hash_from_args(const Args& args) {
  if (args.has("contract-hash") && !args.get("contract-hash").empty()) return make_ok(args.get("contract-hash"));
  return contract::hash_checked(args.get("contract", "NC-v2-f16-l40"));
}

}  // namespace cli
}  // namespace platform
}  // namespace opencrab2
