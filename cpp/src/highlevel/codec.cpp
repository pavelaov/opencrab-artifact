// Implementation of codec.hpp: typed Value, wire codec, canonical JSON,
// SHA-256, HMAC-SHA256. See the header for the contracts.
#include <opencrab2/highlevel/codec.hpp>
#include <opencrab2/lowlevel/hmac.hpp>
#include <opencrab2/lowlevel/sha256.hpp>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <system_error>

namespace opencrab2 {
namespace highlevel {

// ---------------------------------------------------------------------------
// NdArray
// ---------------------------------------------------------------------------
std::vector<uint64_t> NdArray::as_u64() const {
    std::vector<uint64_t> out;
    size_t es = dtype_size(dtype);
    if (es == 0) return out;
    size_t n = data.size() / es;
    out.resize(n);
    for (size_t i = 0; i < n; ++i) {
        const uint8_t* p = data.data() + i * es;
        uint64_t v = 0;
        for (size_t k = 0; k < es; ++k) v |= static_cast<uint64_t>(p[k]) << (8 * k);
        if (dtype == DType::i32) v = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(v)));
        out[i] = v;
    }
    return out;
}

std::vector<double> NdArray::as_f64() const {
    std::vector<double> out;
    if (dtype != DType::f64) return out;
    size_t n = data.size() / 8;
    out.resize(n);
    for (size_t i = 0; i < n; ++i) {
        uint64_t bits = get_u64le(data.data() + 8 * i);
        double d;
        std::memcpy(&d, &bits, 8);
        out[i] = d;
    }
    return out;
}

NdArray NdArray::from_u64(const std::vector<uint64_t>& v, std::vector<uint64_t> dims) {
    NdArray a;
    a.dtype = DType::u64;
    a.dims = std::move(dims);
    a.data.reserve(v.size() * 8);
    for (uint64_t x : v) put_u64le(a.data, x);
    return a;
}

NdArray NdArray::from_f64(const std::vector<double>& v, std::vector<uint64_t> dims) {
    NdArray a;
    a.dtype = DType::f64;
    a.dims = std::move(dims);
    a.data.reserve(v.size() * 8);
    for (double d : v) {
        uint64_t bits;
        std::memcpy(&bits, &d, 8);
        put_u64le(a.data, bits);
    }
    return a;
}

NdArray NdArray::from_u8(const Bytes& v, std::vector<uint64_t> dims) {
    NdArray a;
    a.dtype = DType::u8;
    a.dims = std::move(dims);
    a.data = v;
    return a;
}

// ---------------------------------------------------------------------------
// Value special members and constructors
// ---------------------------------------------------------------------------
Value::Value() : v(None{}) {}
Value::~Value() = default;
Value::Value(const Value&) = default;
Value::Value(Value&&) noexcept = default;
Value& Value::operator=(const Value&) = default;
Value& Value::operator=(Value&&) noexcept = default;
Value::Value(None) : v(None{}) {}
Value::Value(bool b) : v(b) {}
Value::Value(int64_t i) : v(i) {}
Value::Value(int i) : v(static_cast<int64_t>(i)) {}
Value::Value(uint64_t u) : v(static_cast<int64_t>(u)) {}
Value::Value(double d) : v(d) {}
Value::Value(const char* s) : v(std::string(s)) {}
Value::Value(std::string s) : v(std::move(s)) {}
Value::Value(Bytes b) : v(std::move(b)) {}
Value::Value(NdArray a) : v(std::move(a)) {}
Value::Value(Dict d) : v(std::move(d)) {}
Value::Value(List l) : v(std::move(l)) {}

const Value* Value::get(const std::string& key) const {
    const Dict* d = std::get_if<Dict>(&v);
    if (!d) return nullptr;
    for (const auto& e : *d)
        if (e.key == key) return &e.val;
    return nullptr;
}

Value* Value::get(const std::string& key) {
    Dict* d = std::get_if<Dict>(&v);
    if (!d) return nullptr;
    for (auto& e : *d)
        if (e.key == key) return &e.val;
    return nullptr;
}

void Value::set(const std::string& key, Value val) {
    if (!is_dict()) v = Dict{};
    Dict& d = std::get<Dict>(v);
    for (auto& e : d) {
        if (e.key == key) {
            e.val = std::move(val);
            return;
        }
    }
    d.push_back(DictEntry{key, std::move(val)});
}

Expected<int64_t> Value::as_int() const {
    if (const int64_t* p = std::get_if<int64_t>(&v)) return ok(*p);
    if (const bool* b = std::get_if<bool>(&v)) return ok<int64_t>(*b ? 1 : 0);
    return fail("VALUE_TYPE: expected int");
}

Expected<double> Value::as_number() const {
    if (const double* p = std::get_if<double>(&v)) return ok(*p);
    if (const int64_t* p = std::get_if<int64_t>(&v)) return ok(static_cast<double>(*p));
    return fail("VALUE_TYPE: expected number");
}

Expected<std::string> Value::as_string() const {
    if (const std::string* p = std::get_if<std::string>(&v)) return ok(*p);
    return fail("VALUE_TYPE: expected str");
}

Expected<bool> Value::as_bool() const {
    if (const bool* p = std::get_if<bool>(&v)) return ok(*p);
    return fail("VALUE_TYPE: expected bool");
}

Expected<Bytes> Value::as_bytes() const {
    if (const Bytes* p = std::get_if<Bytes>(&v)) return ok(*p);
    return fail("VALUE_TYPE: expected bytes");
}

Expected<NdArray> Value::as_array() const {
    if (const NdArray* p = std::get_if<NdArray>(&v)) return ok(*p);
    return fail("VALUE_TYPE: expected ndarray");
}

Expected<const Dict*> Value::as_dict() const {
    if (const Dict* p = std::get_if<Dict>(&v)) return ok(p);
    return fail("VALUE_TYPE: expected dict");
}

Expected<const List*> Value::as_list() const {
    if (const List* p = std::get_if<List>(&v)) return ok(p);
    return fail("VALUE_TYPE: expected list");
}

Expected<int64_t> Value::get_int(const std::string& key) const {
    const Value* p = get(key);
    if (!p) return fail("VALUE_MISSING: " + key);
    return p->as_int();
}

Expected<double> Value::get_number(const std::string& key) const {
    const Value* p = get(key);
    if (!p) return fail("VALUE_MISSING: " + key);
    return p->as_number();
}

Expected<std::string> Value::get_string(const std::string& key) const {
    const Value* p = get(key);
    if (!p) return fail("VALUE_MISSING: " + key);
    return p->as_string();
}

Expected<NdArray> Value::get_array(const std::string& key) const {
    const Value* p = get(key);
    if (!p) return fail("VALUE_MISSING: " + key);
    return p->as_array();
}

Expected<Bytes> Value::get_bytes(const std::string& key) const {
    const Value* p = get(key);
    if (!p) return fail("VALUE_MISSING: " + key);
    return p->as_bytes();
}

Value dict(std::initializer_list<std::pair<const char*, Value>> items) {
    Dict d;
    for (const auto& it : items) d.push_back(DictEntry{it.first, it.second});
    return Value(std::move(d));
}

Value u64_array(const std::vector<uint64_t>& v, std::vector<uint64_t> dims) {
    return Value(NdArray::from_u64(v, std::move(dims)));
}

Value f64_array(const std::vector<double>& v, std::vector<uint64_t> dims) {
    return Value(NdArray::from_f64(v, std::move(dims)));
}

// ---------------------------------------------------------------------------
// big-endian / little-endian helpers
// ---------------------------------------------------------------------------
void put_u16be(Bytes& out, uint16_t v) {
    out.push_back(static_cast<uint8_t>(v >> 8));
    out.push_back(static_cast<uint8_t>(v));
}
void put_u32be(Bytes& out, uint32_t v) {
    for (int i = 3; i >= 0; --i) out.push_back(static_cast<uint8_t>(v >> (8 * i)));
}
void put_u64be(Bytes& out, uint64_t v) {
    for (int i = 7; i >= 0; --i) out.push_back(static_cast<uint8_t>(v >> (8 * i)));
}
void put_u64le(Bytes& out, uint64_t v) {
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<uint8_t>(v >> (8 * i)));
}
uint64_t get_u64be(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v = (v << 8) | p[i];
    return v;
}
uint32_t get_u32be(const uint8_t* p) {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v = (v << 8) | p[i];
    return v;
}
uint16_t get_u16be(const uint8_t* p) { return static_cast<uint16_t>((p[0] << 8) | p[1]); }
uint64_t get_u64le(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}

// ---------------------------------------------------------------------------
// wire codec
// ---------------------------------------------------------------------------
namespace {

struct Encoder {
    Bytes& out;
    void enc(const Value& val) {
        std::visit(
            [this](const auto& x) { this->put(x); }, val.v);
    }
    void put(const None&) { out.push_back(0x08); }
    void put(bool b) {
        out.push_back(0x09);
        out.push_back(b ? 1 : 0);
    }
    void put(int64_t i) {
        out.push_back(0x04);
        put_u64be(out, static_cast<uint64_t>(i));
    }
    void put(double d) {
        out.push_back(0x05);
        uint64_t bits;
        std::memcpy(&bits, &d, 8);
        put_u64be(out, bits);
    }
    void put(const std::string& s) {
        out.push_back(0x06);
        put_u32be(out, static_cast<uint32_t>(s.size()));
        out.insert(out.end(), s.begin(), s.end());
    }
    void put(const Bytes& b) {
        out.push_back(0x01);
        put_u32be(out, static_cast<uint32_t>(b.size()));
        out.insert(out.end(), b.begin(), b.end());
    }
    void put(const NdArray& a) {
        out.push_back(0x02);
        out.push_back(static_cast<uint8_t>(a.dtype));
        out.push_back(static_cast<uint8_t>(a.dims.size()));
        for (uint64_t d : a.dims) put_u64be(out, d);
        out.insert(out.end(), a.data.begin(), a.data.end());
    }
    void put(const Dict& d) {
        out.push_back(0x03);
        put_u32be(out, static_cast<uint32_t>(d.size()));
        for (const auto& e : d) {
            put_u16be(out, static_cast<uint16_t>(e.key.size()));
            out.insert(out.end(), e.key.begin(), e.key.end());
            enc(e.val);
        }
    }
    void put(const List& l) {
        out.push_back(0x07);
        put_u32be(out, static_cast<uint32_t>(l.size()));
        for (const auto& e : l) enc(e);
    }
};

// The decoder recurses once per container (dict 0x03, list 0x07). Without a limit a 250 KB
// peer frame of 50000 nested one-element lists overflows the stack of the receiving party
// (tests/test_ledger.cpp reproduces the frame). CODEC_MAX_DEPTH containers may be nested; one more fails
// closed with CODEC_DEPTH before any recursion happens.
struct Decoder {
    const uint8_t* p;
    const uint8_t* end;
    std::string err;
    int depth = 0;   // containers currently open

    bool need(size_t n) {
        if (static_cast<size_t>(end - p) < n) {
            err = "CODEC_TRUNCATED";
            return false;
        }
        return true;
    }

    bool dec(Value& out) {
        if (!need(1)) return false;
        uint8_t tag = *p++;
        const bool container = (tag == 0x03 || tag == 0x07);
        if (container) {
            if (depth >= CODEC_MAX_DEPTH) {
                err = "CODEC_DEPTH: nesting deeper than " + std::to_string(CODEC_MAX_DEPTH);
                return false;
            }
            ++depth;
        }
        const bool r = dec_body(tag, out);
        if (container) --depth;
        return r;
    }

    // Strict UTF-8 (RFC 3629): no overlong forms, no surrogates, nothing above U+10FFFF. This is
    // what the Python decoder accepts (bytes.decode("utf-8")), so both sides refuse the same frames.
    static bool valid_utf8(const uint8_t* s, size_t n) {
        size_t i = 0;
        while (i < n) {
            const uint8_t c = s[i];
            if (c < 0x80) {
                ++i;
                continue;
            }
            size_t len;
            uint32_t cp, min;
            if ((c & 0xE0) == 0xC0) { len = 2; cp = c & 0x1F; min = 0x80; }
            else if ((c & 0xF0) == 0xE0) { len = 3; cp = c & 0x0F; min = 0x800; }
            else if ((c & 0xF8) == 0xF0) { len = 4; cp = c & 0x07; min = 0x10000; }
            else return false;
            if (n - i < len) return false;
            for (size_t k = 1; k < len; ++k) {
                if ((s[i + k] & 0xC0) != 0x80) return false;
                cp = (cp << 6) | (s[i + k] & 0x3F);
            }
            if (cp < min || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return false;
            i += len;
        }
        return true;
    }

    bool dec_body(uint8_t tag, Value& out) {
        switch (tag) {
            case 0x01: {
                if (!need(4)) return false;
                uint32_t n = get_u32be(p);
                p += 4;
                if (!need(n)) return false;
                out = Value(Bytes(p, p + n));
                p += n;
                return true;
            }
            case 0x02: {
                if (!need(2)) return false;
                NdArray a;
                uint8_t dt = p[0];
                uint8_t nd = p[1];
                p += 2;
                if (dt < 1 || dt > 5) {
                    err = "CODEC_DTYPE";
                    return false;
                }
                a.dtype = static_cast<DType>(dt);
                if (nd > CODEC_MAX_NDIM) {
                    err = "CODEC_SHAPE: ndim above " + std::to_string(CODEC_MAX_NDIM);
                    return false;
                }
                if (!need(static_cast<size_t>(nd) * 8)) return false;
                size_t numel = 1;
                for (int i = 0; i < nd; ++i) {
                    uint64_t d = get_u64be(p);
                    p += 8;
                    if (d > CODEC_MAX_DIM) {
                        // also for an array without elements: (0, 2^63) is not a shape either side can hold
                        err = "CODEC_SHAPE: dimension above 2^40";
                        return false;
                    }
                    a.dims.push_back(d);
                    if (d != 0 && numel > (static_cast<size_t>(1) << 40) / d) {
                        err = "CODEC_SHAPE";
                        return false;
                    }
                    numel *= static_cast<size_t>(d);
                }
                size_t nbytes = numel * dtype_size(a.dtype);
                if (!need(nbytes)) return false;
                a.data.assign(p, p + nbytes);
                p += nbytes;
                out = Value(std::move(a));
                return true;
            }
            case 0x03: {
                if (!need(4)) return false;
                uint32_t n = get_u32be(p);
                p += 4;
                Dict d;
                d.reserve(std::min<uint32_t>(n, 1024));
                for (uint32_t i = 0; i < n; ++i) {
                    if (!need(2)) return false;
                    uint16_t kl = get_u16be(p);
                    p += 2;
                    if (!need(kl)) return false;
                    if (!valid_utf8(p, kl)) {
                        err = "CODEC_MALFORMED: dict key is not valid UTF-8";
                        return false;
                    }
                    std::string key(reinterpret_cast<const char*>(p), kl);
                    p += kl;
                    Value val;
                    if (!dec(val)) return false;
                    d.push_back(DictEntry{std::move(key), std::move(val)});
                }
                out = Value(std::move(d));
                return true;
            }
            case 0x04: {
                if (!need(8)) return false;
                out = Value(static_cast<int64_t>(get_u64be(p)));
                p += 8;
                return true;
            }
            case 0x05: {
                if (!need(8)) return false;
                uint64_t bits = get_u64be(p);
                p += 8;
                double d;
                std::memcpy(&d, &bits, 8);
                out = Value(d);
                return true;
            }
            case 0x06: {
                if (!need(4)) return false;
                uint32_t n = get_u32be(p);
                p += 4;
                if (!need(n)) return false;
                if (!valid_utf8(p, n)) {
                    err = "CODEC_MALFORMED: str is not valid UTF-8";
                    return false;
                }
                out = Value(std::string(reinterpret_cast<const char*>(p), n));
                p += n;
                return true;
            }
            case 0x07: {
                if (!need(4)) return false;
                uint32_t n = get_u32be(p);
                p += 4;
                List l;
                l.reserve(std::min<uint32_t>(n, 1024));
                for (uint32_t i = 0; i < n; ++i) {
                    Value val;
                    if (!dec(val)) return false;
                    l.push_back(std::move(val));
                }
                out = Value(std::move(l));
                return true;
            }
            case 0x08:
                out = Value(None{});
                return true;
            case 0x09: {
                if (!need(1)) return false;
                if (*p > 1) {
                    // a canonical bool is 0 or 1: any other byte would re-encode to different bytes
                    err = "CODEC_MALFORMED: bool byte must be 0 or 1";
                    return false;
                }
                out = Value(static_cast<bool>(*p++ != 0));
                return true;
            }
            default:
                err = "CODEC_TAG";
                return false;
        }
    }
};

}  // namespace

void encode_value_into(const Value& v, Bytes& out) {
    Encoder e{out};
    e.enc(v);
}

Bytes encode_value(const Value& v) {
    Bytes out;
    encode_value_into(v, out);
    return out;
}

Expected<Value> decode_value(const uint8_t* data, size_t len) {
    Decoder d{data, data + len, "", 0};
    Value out;
    if (!d.dec(out)) return fail(d.err.empty() ? std::string("CODEC_MALFORMED") : d.err);
    if (d.p != d.end) return fail("CODEC_TRAILING");
    return ok(std::move(out));
}

Expected<Value> decode_value(const Bytes& b) { return decode_value(b.data(), b.size()); }

Bytes encode_frame(const Value& v) {
    Bytes payload = encode_value(v);
    Bytes out;
    out.reserve(payload.size() + 8);
    put_u64be(out, static_cast<uint64_t>(payload.size()));
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

// ---------------------------------------------------------------------------
// canonical JSON
// ---------------------------------------------------------------------------
std::string python_float_repr(double d) {
    if (d == 0.0) return std::signbit(d) ? "-0.0" : "0.0";
    char buf[64];
    auto r = std::to_chars(buf, buf + sizeof(buf), d, std::chars_format::scientific);
    std::string s(buf, r.ptr);
    bool neg = false;
    if (!s.empty() && s[0] == '-') {
        neg = true;
        s.erase(0, 1);
    }
    size_t epos = s.find('e');
    std::string mant = s.substr(0, epos);
    int exp10 = std::atoi(s.c_str() + epos + 1);
    std::string digits;
    for (char c : mant)
        if (c != '.') digits.push_back(c);
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
    int ndigits = static_cast<int>(digits.size());
    int decpt = exp10 + 1;
    std::string out;
    if (decpt > -4 && decpt <= 16) {
        if (decpt <= 0) {
            out = "0." + std::string(static_cast<size_t>(-decpt), '0') + digits;
        } else if (decpt >= ndigits) {
            out = digits + std::string(static_cast<size_t>(decpt - ndigits), '0') + ".0";
        } else {
            out = digits.substr(0, static_cast<size_t>(decpt)) + "." + digits.substr(static_cast<size_t>(decpt));
        }
    } else {
        out = digits.substr(0, 1);
        if (ndigits > 1) out += "." + digits.substr(1);
        char eb[16];
        std::snprintf(eb, sizeof(eb), "e%c%02d", exp10 < 0 ? '-' : '+', std::abs(exp10));
        out += eb;
    }
    return neg ? "-" + out : out;
}

namespace {

void json_escape_string(const std::string& s, std::string& out) {
    out.push_back('"');
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (c < 0x20) {
                    char b[8];
                    std::snprintf(b, sizeof(b), "\\u%04x", c);
                    out += b;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    out.push_back('"');
}

bool emit_json(const Value& v, std::string& out, bool sorted, std::string& err) {
    if (v.is_none()) {
        out += "null";
        return true;
    }
    if (const bool* b = std::get_if<bool>(&v.v)) {
        out += *b ? "true" : "false";
        return true;
    }
    if (const int64_t* i = std::get_if<int64_t>(&v.v)) {
        out += std::to_string(*i);
        return true;
    }
    if (const double* d = std::get_if<double>(&v.v)) {
        if (!std::isfinite(*d)) {
            err = "JSON_NOT_CANONICAL: non-finite float";
            return false;
        }
        out += python_float_repr(*d);
        return true;
    }
    if (const std::string* s = std::get_if<std::string>(&v.v)) {
        json_escape_string(*s, out);
        return true;
    }
    if (const List* l = std::get_if<List>(&v.v)) {
        out.push_back('[');
        for (size_t i = 0; i < l->size(); ++i) {
            if (i) out.push_back(',');
            if (!emit_json((*l)[i], out, sorted, err)) return false;
        }
        out.push_back(']');
        return true;
    }
    if (const Dict* d = std::get_if<Dict>(&v.v)) {
        std::vector<const DictEntry*> entries;
        entries.reserve(d->size());
        for (const auto& e : *d) entries.push_back(&e);
        if (sorted)
            std::sort(entries.begin(), entries.end(),
                      [](const DictEntry* a, const DictEntry* b) { return a->key < b->key; });
        out.push_back('{');
        for (size_t i = 0; i < entries.size(); ++i) {
            if (i) out.push_back(',');
            json_escape_string(entries[i]->key, out);
            out.push_back(':');
            if (!emit_json(entries[i]->val, out, sorted, err)) return false;
        }
        out.push_back('}');
        return true;
    }
    err = "JSON_NOT_CANONICAL: bytes and ndarray values are not canonicalisable";
    return false;
}

}  // namespace

Expected<std::string> canonical_json(const Value& v) {
    std::string out, err;
    if (!emit_json(v, out, true, err)) return fail(err);
    return ok(std::move(out));
}

Expected<std::string> to_json(const Value& v) {
    std::string out, err;
    if (!emit_json(v, out, false, err)) return fail(err);
    return ok(std::move(out));
}

namespace {

// The parser recurses once per array or object. JSON_PARSE_MAX_DEPTH containers may be nested;
// one more fails closed with JSON_DEPTH (a tampered or corrupted journal line must not overflow
// the stack of a restarting party).
struct JsonParser {
    const char* p;
    const char* end;
    std::string err;
    int depth = 0;

    void ws() {
        while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p;
    }
    bool parse(Value& out) {
        ws();
        if (p >= end) return bad("JSON_PARSE: unexpected end");
        char c = *p;
        if (c == '{' || c == '[') {
            if (depth >= JSON_PARSE_MAX_DEPTH) return bad("JSON_DEPTH: nesting deeper than " + std::to_string(JSON_PARSE_MAX_DEPTH));
            ++depth;
            bool r = c == '{' ? parse_object(out) : parse_array(out);
            --depth;
            return r;
        }
        if (c == '"') {
            std::string s;
            if (!parse_string(s)) return false;
            out = Value(std::move(s));
            return true;
        }
        if (c == 't' && match("true")) { out = Value(true); return true; }
        if (c == 'f' && match("false")) { out = Value(false); return true; }
        if (c == 'n' && match("null")) { out = Value(None{}); return true; }
        return parse_number(out);
    }
    bool bad(const std::string& e) {
        err = e;
        return false;
    }
    bool match(const char* lit) {
        size_t n = std::strlen(lit);
        if (static_cast<size_t>(end - p) >= n && std::strncmp(p, lit, n) == 0) {
            p += n;
            return true;
        }
        return false;
    }
    bool parse_number(Value& out) {
        const char* start = p;
        bool is_float = false;
        if (p < end && (*p == '-' || *p == '+')) ++p;
        while (p < end && (std::isdigit(static_cast<unsigned char>(*p)) || *p == '.' || *p == 'e' || *p == 'E' ||
                           *p == '-' || *p == '+')) {
            if (*p == '.' || *p == 'e' || *p == 'E') is_float = true;
            ++p;
        }
        if (p == start) return bad("JSON_PARSE: bad token");
        std::string tok(start, p);
        if (!is_float) {
            long long v = 0;
            auto r = std::from_chars(tok.data(), tok.data() + tok.size(), v);
            if (r.ec == std::errc()) {
                out = Value(static_cast<int64_t>(v));
                return true;
            }
        }
        double d = std::strtod(tok.c_str(), nullptr);
        out = Value(d);
        return true;
    }
    static void append_utf8(std::string& s, uint32_t cp) {
        if (cp < 0x80) {
            s.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            s.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }
    bool parse_string(std::string& s) {
        ++p;  // opening quote
        while (p < end && *p != '"') {
            if (*p == '\\') {
                ++p;
                if (p >= end) return bad("JSON_PARSE: bad escape");
                char e = *p++;
                switch (e) {
                    case '"': s.push_back('"'); break;
                    case '\\': s.push_back('\\'); break;
                    case '/': s.push_back('/'); break;
                    case 'n': s.push_back('\n'); break;
                    case 'r': s.push_back('\r'); break;
                    case 't': s.push_back('\t'); break;
                    case 'b': s.push_back('\b'); break;
                    case 'f': s.push_back('\f'); break;
                    case 'u': {
                        if (end - p < 4) return bad("JSON_PARSE: bad unicode escape");
                        uint32_t cp = static_cast<uint32_t>(std::strtoul(std::string(p, p + 4).c_str(), nullptr, 16));
                        p += 4;
                        if (cp >= 0xD800 && cp <= 0xDBFF && end - p >= 6 && p[0] == '\\' && p[1] == 'u') {
                            uint32_t lo = static_cast<uint32_t>(std::strtoul(std::string(p + 2, p + 6).c_str(), nullptr, 16));
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            p += 6;
                        }
                        append_utf8(s, cp);
                        break;
                    }
                    default: return bad("JSON_PARSE: bad escape");
                }
            } else {
                s.push_back(*p++);
            }
        }
        if (p >= end) return bad("JSON_PARSE: unterminated string");
        ++p;
        return true;
    }
    bool parse_array(Value& out) {
        ++p;
        List l;
        ws();
        if (p < end && *p == ']') {
            ++p;
            out = Value(std::move(l));
            return true;
        }
        while (true) {
            Value item;
            if (!parse(item)) return false;
            l.push_back(std::move(item));
            ws();
            if (p < end && *p == ',') {
                ++p;
                continue;
            }
            if (p < end && *p == ']') {
                ++p;
                break;
            }
            return bad("JSON_PARSE: expected , or ]");
        }
        out = Value(std::move(l));
        return true;
    }
    bool parse_object(Value& out) {
        ++p;
        Dict d;
        ws();
        if (p < end && *p == '}') {
            ++p;
            out = Value(std::move(d));
            return true;
        }
        while (true) {
            ws();
            if (p >= end || *p != '"') return bad("JSON_PARSE: expected key");
            std::string key;
            if (!parse_string(key)) return false;
            ws();
            if (p >= end || *p != ':') return bad("JSON_PARSE: expected :");
            ++p;
            Value val;
            if (!parse(val)) return false;
            d.push_back(DictEntry{std::move(key), std::move(val)});
            ws();
            if (p < end && *p == ',') {
                ++p;
                continue;
            }
            if (p < end && *p == '}') {
                ++p;
                break;
            }
            return bad("JSON_PARSE: expected , or }");
        }
        out = Value(std::move(d));
        return true;
    }
};

}  // namespace

Expected<Value> parse_json(const std::string& text) {
    JsonParser jp{text.data(), text.data() + text.size(), "", 0};
    Value out;
    if (!jp.parse(out)) return fail(jp.err);
    jp.ws();
    if (jp.p != jp.end) return fail("JSON_PARSE: trailing characters");
    return ok(std::move(out));
}

// ---------------------------------------------------------------------------
// hex
// ---------------------------------------------------------------------------
std::string to_hex(const uint8_t* p, size_t n) {
    static const char* digits = "0123456789abcdef";
    std::string s;
    s.reserve(2 * n);
    for (size_t i = 0; i < n; ++i) {
        s.push_back(digits[p[i] >> 4]);
        s.push_back(digits[p[i] & 15]);
    }
    return s;
}

std::string to_hex(const Bytes& b) { return to_hex(b.data(), b.size()); }

Expected<Bytes> from_hex(const std::string& h) {
    if (h.size() % 2) return fail("HEX_LENGTH");
    Bytes out(h.size() / 2);
    auto val = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    for (size_t i = 0; i < out.size(); ++i) {
        int hi = val(h[2 * i]), lo = val(h[2 * i + 1]);
        if (hi < 0 || lo < 0) return fail("HEX_DIGIT");
        out[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return ok(std::move(out));
}

// ---------------------------------------------------------------------------
// SHA-256 / HMAC-SHA256: delegated to the lowlevel layer (FIPS 180-4, RFC 2104)
// ---------------------------------------------------------------------------
std::array<uint8_t, 32> sha256(const uint8_t* data, size_t len) { return lowlevel::Sha256::digest(data, len); }

std::array<uint8_t, 32> sha256(const Bytes& b) { return sha256(b.data(), b.size()); }

std::string sha256_hex(const Bytes& b) {
    auto d = sha256(b);
    return to_hex(d.data(), d.size());
}

std::array<uint8_t, 32> hmac_sha256(const uint8_t* key, size_t key_len, const uint8_t* msg, size_t msg_len) {
    return lowlevel::hmac_sha256(key, key_len, msg, msg_len);
}

std::string hmac_sha256_hex(const Bytes& key, const Bytes& msg) {
    auto d = hmac_sha256(key.data(), key.size(), msg.data(), msg.size());
    return to_hex(d.data(), d.size());
}

bool constant_time_equal(const uint8_t* a, const uint8_t* b, size_t n) {
    uint8_t acc = 0;
    for (size_t i = 0; i < n; ++i) acc |= static_cast<uint8_t>(a[i] ^ b[i]);
    return acc == 0;
}

Expected<std::string> hmac_tag(const Bytes& key, const Value& obj) {
    if (key.empty()) return fail("HMAC_KEY: key must not be empty");
    OC2_TRY(cj, canonical_json(obj));
    Bytes msg(cj.begin(), cj.end());
    return ok(hmac_sha256_hex(key, msg));
}

Expected<bool> hmac_verify(const Bytes& key, const Value& obj, const std::string& tag_hex) {
    if (key.empty()) return fail("HMAC_KEY: key must not be empty");
    OC2_TRY(cj, canonical_json(obj));
    auto expected = hmac_sha256(key.data(), key.size(), reinterpret_cast<const uint8_t*>(cj.data()), cj.size());
    if (tag_hex.size() != 64) return ok(false);
    auto given = from_hex(tag_hex);
    if (!given.ok) return ok(false);
    return ok(constant_time_equal(expected.data(), given.value.data(), 32));
}

namespace {
Expected<Bytes> mac_input(const Value& header, const Bytes& payload) {
    OC2_TRY(cj, canonical_json(header));
    Bytes out;
    out.reserve(8 + cj.size() + payload.size());
    put_u64be(out, static_cast<uint64_t>(cj.size()));
    out.insert(out.end(), cj.begin(), cj.end());
    out.insert(out.end(), payload.begin(), payload.end());
    return ok(std::move(out));
}
}  // namespace

Expected<std::string> hmac_tag_with_payload(const Bytes& key, const Value& header, const Bytes& payload) {
    if (key.empty()) return fail("HMAC_KEY: key must not be empty");
    OC2_TRY(mi, mac_input(header, payload));
    return ok(hmac_sha256_hex(key, mi));
}

Expected<bool> hmac_verify_with_payload(const Bytes& key, const Value& header, const Bytes& payload,
                                        const std::string& tag_hex) {
    if (key.empty()) return fail("HMAC_KEY: key must not be empty");
    OC2_TRY(mi, mac_input(header, payload));
    auto expected = hmac_sha256(key.data(), key.size(), mi.data(), mi.size());
    if (tag_hex.size() != 64) return ok(false);
    auto given = from_hex(tag_hex);
    if (!given.ok) return ok(false);
    return ok(constant_time_equal(expected.data(), given.value.data(), 32));
}

// ---------------------------------------------------------------------------
// Json bridge
// ---------------------------------------------------------------------------
Json value_to_json(const Value& v) {
    if (v.is_none()) return Json();
    if (const bool* b = std::get_if<bool>(&v.v)) return Json(*b);
    if (const int64_t* i = std::get_if<int64_t>(&v.v)) return Json(static_cast<long long>(*i));
    if (const double* d = std::get_if<double>(&v.v)) return Json(*d);
    if (const std::string* s = std::get_if<std::string>(&v.v)) return Json(*s);
    if (const Bytes* b = std::get_if<Bytes>(&v.v)) return Json(to_hex(*b));
    if (const NdArray* a = std::get_if<NdArray>(&v.v)) {
        Json::Array arr;
        if (a->dtype == DType::f64) {
            for (double d : a->as_f64()) arr.push_back(Json(d));
        } else {
            for (uint64_t u : a->as_u64()) arr.push_back(Json(static_cast<long long>(static_cast<int64_t>(u))));
        }
        return Json(std::move(arr));
    }
    if (const List* l = std::get_if<List>(&v.v)) {
        Json::Array arr;
        for (const auto& e : *l) arr.push_back(value_to_json(e));
        return Json(std::move(arr));
    }
    if (const Dict* d = std::get_if<Dict>(&v.v)) {
        Json::Object obj;
        for (const auto& e : *d) obj.emplace_back(e.key, value_to_json(e.val));
        return Json(std::move(obj));
    }
    return Json();
}

Value json_to_value(const Json& j) {
    switch (j.type()) {
        case Json::Type::Null: return Value(None{});
        case Json::Type::Bool: return Value(j.as_bool());
        case Json::Type::Int: return Value(static_cast<int64_t>(j.as_int()));
        case Json::Type::Double: return Value(j.as_double());
        case Json::Type::String: return Value(j.as_string());
        case Json::Type::Array: {
            List l;
            for (const auto& e : j.as_array()) l.push_back(json_to_value(e));
            return Value(std::move(l));
        }
        case Json::Type::Object: {
            Dict d;
            for (const auto& kv : j.as_object()) d.push_back(DictEntry{kv.first, json_to_value(kv.second)});
            return Value(std::move(d));
        }
    }
    return Value(None{});
}

double now_s() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

double unix_time_s() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(system_clock::now().time_since_epoch()).count();
}

}  // namespace highlevel
}  // namespace opencrab2
