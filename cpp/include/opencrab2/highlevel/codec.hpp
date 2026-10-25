// OpenCrab v2 C++17 mirror: wire codec (the bytes of python/opencrab2/highlevel/channel.py), typed Value,
// canonical JSON (Python json.dumps(sort_keys=True, separators=(",", ":"),
// ensure_ascii=False)), and the byte-level helpers (hex, SHA-256, HMAC-SHA256)
// that the high-level layer needs for digests, bundle tags and telemetry tags.
//
// The wire codec is byte-binding with the Python Channel:
//   frame   = 8-byte big-endian length L, then L payload bytes
//   payload = typed value with tags 0x01..0x09 (see encode_value / decode_value)
//
// Fail-closed convention: every function that can fail returns
// opencrab2::Expected<T> {bool ok; T value; std::string code;} with a stable
// upper-case code. The tiny Expected struct lives in the lowlevel layer
// (include/opencrab2/expected.hpp); it is included when present and a
// byte-identical local definition is used otherwise.
//
// SHA-256 and HMAC-SHA256 delegate to the lowlevel layer's implementations
// (opencrab2/lowlevel/sha256.hpp, hmac.hpp); the FIPS / RFC vectors are checked
// again in tests/test_engine.cpp through the wrappers below. The bridge
// value_to_json / json_to_value converts between this layer's wire Value and
// the opencrab2::highlevel::Json type of the decision layer (json.hpp).
#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <variant>
#include <array>

#include <opencrab2/expected.hpp>
#include <opencrab2/highlevel/json.hpp>

namespace opencrab2 {

// A failure that converts to Expected<T> for any T (keeps call sites short).
struct Failure {
    std::string code;
    template <class T>
    operator Expected<T>() const {
        Expected<T> e;
        e.ok = false;
        e.code = code;
        return e;
    }
};

template <class T>
inline Expected<T> ok(T v) {
    Expected<T> e;
    e.ok = true;
    e.value = std::move(v);
    return e;
}

inline Expected<Unit> ok_unit() { return ok(Unit{}); }

inline Failure fail(std::string code) { return Failure{std::move(code)}; }

// Propagation macros. OC2_TRY(x, expr) binds x to the value of a successful
// Expected or returns the failure to the caller.
#define OC2_TRY(var, expr)                                              \
    auto var##_oc2res = (expr);                                         \
    if (!var##_oc2res.ok) return ::opencrab2::Failure{var##_oc2res.code}; \
    auto var = std::move(var##_oc2res.value)

#define OC2_CHECK(expr)                                                 \
    do {                                                                \
        auto oc2_chk = (expr);                                          \
        if (!oc2_chk.ok) return ::opencrab2::Failure{oc2_chk.code};     \
    } while (0)

namespace highlevel {

using Bytes = std::vector<uint8_t>;

// ndarray dtype codes of the wire codec.
enum class DType : uint8_t { u64 = 1, i64 = 2, f64 = 3, u8 = 4, i32 = 5 };

inline size_t dtype_size(DType d) {
    switch (d) {
        case DType::u64: return 8;
        case DType::i64: return 8;
        case DType::f64: return 8;
        case DType::u8: return 1;
        case DType::i32: return 4;
    }
    return 0;
}

struct NdArray {
    DType dtype = DType::u64;
    std::vector<uint64_t> dims;
    Bytes data;  // raw little-endian element data

    size_t numel() const {
        size_t n = 1;
        for (auto d : dims) n *= static_cast<size_t>(d);
        return dims.empty() ? 1 : n;
    }
    // Typed views (copying). The caller checks dtype.
    std::vector<uint64_t> as_u64() const;
    std::vector<double> as_f64() const;
    static NdArray from_u64(const std::vector<uint64_t>& v, std::vector<uint64_t> dims);
    static NdArray from_f64(const std::vector<double>& v, std::vector<uint64_t> dims);
    static NdArray from_u8(const Bytes& v, std::vector<uint64_t> dims);
};

struct Value;
struct DictEntry;
struct None {};
using Dict = std::vector<DictEntry>;  // insertion ordered, like a Python dict
using List = std::vector<Value>;

struct Value {
    std::variant<None, bool, int64_t, double, std::string, Bytes, NdArray, Dict, List> v;

    Value();
    ~Value();
    Value(const Value&);
    Value(Value&&) noexcept;
    Value& operator=(const Value&);
    Value& operator=(Value&&) noexcept;

    Value(None);
    Value(bool b);
    Value(int64_t i);
    Value(int i);
    Value(uint64_t u);  // stored as int64 (two's complement); use NdArray for ring data
    Value(double d);
    Value(const char* s);
    Value(std::string s);
    Value(Bytes b);
    Value(NdArray a);
    Value(Dict d);
    Value(List l);

    bool is_none() const { return std::holds_alternative<None>(v); }
    bool is_bool() const { return std::holds_alternative<bool>(v); }
    bool is_int() const { return std::holds_alternative<int64_t>(v); }
    bool is_double() const { return std::holds_alternative<double>(v); }
    bool is_string() const { return std::holds_alternative<std::string>(v); }
    bool is_bytes() const { return std::holds_alternative<Bytes>(v); }
    bool is_array() const { return std::holds_alternative<NdArray>(v); }
    bool is_dict() const { return std::holds_alternative<Dict>(v); }
    bool is_list() const { return std::holds_alternative<List>(v); }

    // Dict helpers (nullptr when absent or not a dict).
    const Value* get(const std::string& key) const;
    Value* get(const std::string& key);
    void set(const std::string& key, Value val);  // replaces or appends
    bool has(const std::string& key) const { return get(key) != nullptr; }

    // Typed access with fail-closed semantics.
    Expected<int64_t> as_int() const;
    Expected<double> as_number() const;  // int or double
    Expected<std::string> as_string() const;
    Expected<bool> as_bool() const;
    Expected<Bytes> as_bytes() const;
    Expected<NdArray> as_array() const;
    Expected<const Dict*> as_dict() const;
    Expected<const List*> as_list() const;
    Expected<int64_t> get_int(const std::string& key) const;
    Expected<double> get_number(const std::string& key) const;
    Expected<std::string> get_string(const std::string& key) const;
    Expected<NdArray> get_array(const std::string& key) const;
    Expected<Bytes> get_bytes(const std::string& key) const;
};

struct DictEntry {
    std::string key;
    Value val;
};

// Convenience builders.
Value dict(std::initializer_list<std::pair<const char*, Value>> items);
Value u64_array(const std::vector<uint64_t>& v, std::vector<uint64_t> dims);
Value f64_array(const std::vector<double>& v, std::vector<uint64_t> dims);

// --- wire codec ---------------------------------------------------------------
// Decoder limits (fail closed, stable codes). Both decoders of this layer recurse once per
// container, so peer-controlled or disk-controlled input must not choose the recursion depth.
//   CODEC_MAX_DEPTH       dict / list nesting accepted by decode_value; deeper input fails with
//                         CODEC_DEPTH (a 250 KB frame of nested lists crashed the first version).
//   JSON_PARSE_MAX_DEPTH  object / array nesting accepted by parse_json; deeper input fails with
//                         JSON_DEPTH. Lines written by this layer nest at most 5 levels.
// The frame length cap lives in Channel (CHANNEL_FRAME_TOO_LARGE, channel.hpp).
constexpr int CODEC_MAX_DEPTH = 64;
// Array limits of decode_value (the Python decoder has the same two): at most 16 dimensions, each
// at most 2^40 (CODEC_SHAPE otherwise, also for an array without elements). decode_value further
// refuses a bool byte other than 0 or 1 and a str or dict key that is not strict UTF-8
// (CODEC_MALFORMED), so a frame that decodes re-encodes to the same bytes in both languages.
constexpr int CODEC_MAX_NDIM = 16;
constexpr uint64_t CODEC_MAX_DIM = static_cast<uint64_t>(1) << 40;
constexpr int JSON_PARSE_MAX_DEPTH = 64;

// Encodes a typed value (payload only, no length prefix).
Bytes encode_value(const Value& v);
void encode_value_into(const Value& v, Bytes& out);
// Decodes exactly one typed value from [data, data+len). Code CODEC_* on failure
// (CODEC_TRUNCATED, CODEC_TAG, CODEC_DTYPE, CODEC_SHAPE, CODEC_DEPTH, CODEC_TRAILING).
Expected<Value> decode_value(const uint8_t* data, size_t len);
Expected<Value> decode_value(const Bytes& b);
// Encodes a full frame: 8-byte big-endian length + payload.
Bytes encode_frame(const Value& v);

// Big-endian helpers.
void put_u16be(Bytes& out, uint16_t v);
void put_u32be(Bytes& out, uint32_t v);
void put_u64be(Bytes& out, uint64_t v);
void put_u64le(Bytes& out, uint64_t v);
uint64_t get_u64be(const uint8_t* p);
uint32_t get_u32be(const uint8_t* p);
uint16_t get_u16be(const uint8_t* p);
uint64_t get_u64le(const uint8_t* p);

// --- canonical JSON -----------------------------------------------------------
// Python json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
// allow_nan=False). Floats use Python's repr (shortest round trip). Bytes and
// ndarrays are not canonicalisable (code JSON_NOT_CANONICAL), matching the
// Python hmac_auth module which rejects them.
Expected<std::string> canonical_json(const Value& v);
// Python repr() of a finite double ("1.5", "1e-05", "100.0", "1e+16").
std::string python_float_repr(double d);
// Minimal JSON parser for the journal / records written by this layer (objects,
// arrays, strings, numbers, true/false/null). Code JSON_PARSE on failure, JSON_DEPTH
// when arrays / objects nest deeper than JSON_PARSE_MAX_DEPTH.
Expected<Value> parse_json(const std::string& text);
// Non-canonical pretty-free JSON emitter that keeps dict insertion order (used
// for evidence records and journal lines; canonical_json is used for tags).
Expected<std::string> to_json(const Value& v);

// Bridge to the decision layer's Json. bytes become lowercase hex strings,
// u64/i64/i32 ndarrays become lists of ints (two's complement for u64), f64
// ndarrays lists of doubles, u8 ndarrays lists of ints; dict order is dropped
// (Json objects are sorted). json_to_value maps objects to dicts, arrays to
// lists, numbers to int64 / double.
Json value_to_json(const Value& v);
Value json_to_value(const Json& j);

// --- hex / hashing ------------------------------------------------------------
std::string to_hex(const uint8_t* p, size_t n);
std::string to_hex(const Bytes& b);
Expected<Bytes> from_hex(const std::string& h);

// SHA-256 of a byte string (FIPS 180-4).
std::array<uint8_t, 32> sha256(const uint8_t* data, size_t len);
std::array<uint8_t, 32> sha256(const Bytes& b);
std::string sha256_hex(const Bytes& b);
// sha256_hex(const std::string&) is provided by json.hpp.
// HMAC-SHA256 (RFC 2104); hex output is lowercase.
std::array<uint8_t, 32> hmac_sha256(const uint8_t* key, size_t key_len, const uint8_t* msg, size_t msg_len);
std::string hmac_sha256_hex(const Bytes& key, const Bytes& msg);
// Constant-time comparison of two equal-length byte strings.
bool constant_time_equal(const uint8_t* a, const uint8_t* b, size_t n);

// HMAC tag over canonical JSON (mirrors opencrab2.lowlevel.hmac_auth.tag).
Expected<std::string> hmac_tag(const Bytes& key, const Value& obj);
Expected<bool> hmac_verify(const Bytes& key, const Value& obj, const std::string& tag_hex);
// Header + payload framing: len(header) 8-byte BE || header || payload.
Expected<std::string> hmac_tag_with_payload(const Bytes& key, const Value& header, const Bytes& payload);
Expected<bool> hmac_verify_with_payload(const Bytes& key, const Value& header, const Bytes& payload,
                                        const std::string& tag_hex);

// Monotonic wall clock in seconds (steady clock) and Unix time in seconds.
double now_s();
double unix_time_s();

}  // namespace highlevel
}  // namespace opencrab2
