// opencrab2/platform/support.cpp (see support.hpp)
//
// Conversions between the platform value containers and the highlevel types.
// Every byte of JSON text and of the wire codec is produced and consumed by
// the highlevel implementations (json.cpp, codec.cpp); nothing here parses or
// serializes on its own. Also the writer of files that hold a secret.
#include <opencrab2/platform/support.hpp>

#ifdef _WIN32
#include <fstream>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace opencrab2 {
namespace platform {

namespace hl = opencrab2::highlevel;

// ---------------------------------------------------------------------------
// files that hold a secret
// ---------------------------------------------------------------------------
namespace detail {

bool write_private_file(const std::string& path, const std::string& text) {
#ifdef _WIN32
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) return false;
  out.write(text.data(), static_cast<std::streamsize>(text.size()));
  out.flush();
  return static_cast<bool>(out);
#else
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
  if (fd < 0) return false;
  // the mode argument of open() applies to a new file only: tighten one that existed
  bool ok = ::fchmod(fd, 0600) == 0;
  size_t done = 0;
  while (ok && done < text.size()) {
    const ssize_t n = ::write(fd, text.data() + done, text.size() - done);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) ok = false;
    else done += static_cast<size_t>(n);
  }
  if (::close(fd) != 0) ok = false;
  return ok;
#endif
}

}  // namespace detail

// ---------------------------------------------------------------------------
// json::Value <-> highlevel::Json
// ---------------------------------------------------------------------------
namespace json {

hl::Json Value::to_highlevel() const {
  switch (kind) {
    case Kind::Null:
      return hl::Json();
    case Kind::Bool:
      return hl::Json(b);
    case Kind::Number:
      if (is_int) return hl::Json(static_cast<long long>(inum));
      return hl::Json(num);
    case Kind::String:
      return hl::Json(str);
    case Kind::Array: {
      hl::Json a = hl::Json::array();
      for (const auto& v : arr) a.push_back(v.to_highlevel());
      return a;
    }
    case Kind::Object: {
      hl::Json o = hl::Json::object();
      for (const auto& kv : obj) o.set(kv.first, kv.second.to_highlevel());
      return o;
    }
  }
  return hl::Json();
}

Value Value::from_highlevel(const hl::Json& j) {
  switch (j.type()) {
    case hl::Json::Type::Null:
      return Value::null();
    case hl::Json::Type::Bool:
      return Value::boolean(j.as_bool());
    case hl::Json::Type::Int:
      return Value::integer(static_cast<int64_t>(j.as_int()));
    case hl::Json::Type::Double:
      return Value::number(j.as_double());
    case hl::Json::Type::String:
      return Value::string(j.as_string());
    case hl::Json::Type::Array: {
      Value a = Value::array();
      for (const auto& v : j.as_array()) a.arr.push_back(from_highlevel(v));
      return a;
    }
    case hl::Json::Type::Object: {
      Value o = Value::object();
      for (const auto& kv : j.as_object()) o.obj.emplace_back(kv.first, from_highlevel(kv.second));
      return o;
    }
  }
  return Value::null();
}

std::string Value::dump(bool) const { return to_highlevel().dump(); }

Expected<Value> parse(const std::string& text) {
  auto j = hl::Json::parse(text);
  if (!j.ok) return make_fail<Value>(j.code.empty() ? std::string("JSON_PARSE") : j.code_prefix());
  return make_ok(Value::from_highlevel(j.value));
}

}  // namespace json

// ---------------------------------------------------------------------------
// wire::Value <-> highlevel::Value and the codec entry points
// ---------------------------------------------------------------------------
namespace wire {

NdArray NdArray::from_u64(const std::vector<uint64_t>& v, std::vector<uint64_t> shape) {
  hl::NdArray h = hl::NdArray::from_u64(v, shape.empty() ? std::vector<uint64_t>{v.size()} : std::move(shape));
  NdArray a;
  a.dtype = DType::U64;
  a.shape = std::move(h.dims);
  a.data = std::move(h.data);
  return a;
}

std::vector<uint64_t> NdArray::to_u64() const {
  std::vector<uint64_t> v;
  if (dtype != DType::U64 && dtype != DType::I64) return v;
  hl::NdArray h;
  h.dtype = hl::DType::u64;
  h.dims = {data.size() / 8};
  h.data.assign(data.begin(), data.begin() + static_cast<std::ptrdiff_t>((data.size() / 8) * 8));
  return h.as_u64();
}

json::Value Value::to_json() const {
  switch (tag) {
    case Tag::None:
      return json::Value::null();
    case Tag::Bool:
      return json::Value::boolean(b);
    case Tag::Int:
      return json::Value::integer(i);
    case Tag::Float:
      return json::Value::number(f);
    case Tag::Str:
      return json::Value::string(s);
    case Tag::Bytes: {
      json::Value o = json::Value::object();
      o.set("__bytes__", json::Value::integer(static_cast<int64_t>(bytes.size())));
      return o;
    }
    case Tag::NdArray: {
      json::Value o = json::Value::object();
      o.set("__ndarray__", json::Value::integer(static_cast<int64_t>(array.data.size())));
      return o;
    }
    case Tag::List: {
      json::Value a = json::Value::array();
      for (const auto& v : list) a.push(v.to_json());
      return a;
    }
    case Tag::Dict: {
      json::Value o = json::Value::object();
      for (const auto& kv : dict) o.set(kv.first, kv.second.to_json());
      return o;
    }
  }
  return json::Value::null();
}

Value Value::from_json(const json::Value& j) {
  switch (j.kind) {
    case json::Value::Kind::Null:
      return none();
    case json::Value::Kind::Bool:
      return from_bool(j.b);
    case json::Value::Kind::Number:
      return j.is_int ? from_int(j.inum) : from_float(j.num);
    case json::Value::Kind::String:
      return from_str(j.str);
    case json::Value::Kind::Array: {
      std::vector<Value> items;
      for (const auto& it : j.arr) items.push_back(from_json(it));
      return from_list(std::move(items));
    }
    case json::Value::Kind::Object: {
      Value d = make_dict();
      for (const auto& kv : j.obj) d.set(kv.first, from_json(kv.second));
      return d;
    }
  }
  return none();
}

hl::Value Value::to_highlevel() const {
  switch (tag) {
    case Tag::None:
      return hl::Value(hl::None{});
    case Tag::Bool:
      return hl::Value(b);
    case Tag::Int:
      return hl::Value(static_cast<int64_t>(i));
    case Tag::Float:
      return hl::Value(f);
    case Tag::Str:
      return hl::Value(s);
    case Tag::Bytes:
      return hl::Value(hl::Bytes(bytes.begin(), bytes.end()));
    case Tag::NdArray: {
      hl::NdArray a;
      a.dtype = static_cast<hl::DType>(array.dtype);
      a.dims = array.shape;
      a.data = array.data;
      return hl::Value(std::move(a));
    }
    case Tag::List: {
      hl::List l;
      l.reserve(list.size());
      for (const auto& v : list) l.push_back(v.to_highlevel());
      return hl::Value(std::move(l));
    }
    case Tag::Dict: {
      hl::Dict d;
      d.reserve(dict.size());
      for (const auto& kv : dict) d.push_back(hl::DictEntry{kv.first, kv.second.to_highlevel()});
      return hl::Value(std::move(d));
    }
  }
  return hl::Value(hl::None{});
}

Value Value::from_highlevel(const hl::Value& v) {
  if (v.is_none()) return none();
  if (const bool* b = std::get_if<bool>(&v.v)) return from_bool(*b);
  if (const int64_t* i = std::get_if<int64_t>(&v.v)) return from_int(*i);
  if (const double* d = std::get_if<double>(&v.v)) return from_float(*d);
  if (const std::string* s = std::get_if<std::string>(&v.v)) return from_str(*s);
  if (const hl::Bytes* by = std::get_if<hl::Bytes>(&v.v)) return from_bytes(std::vector<uint8_t>(by->begin(), by->end()));
  if (const hl::NdArray* a = std::get_if<hl::NdArray>(&v.v)) {
    NdArray out;
    out.dtype = static_cast<DType>(a->dtype);
    out.shape = a->dims;
    out.data = a->data;
    return from_array(std::move(out));
  }
  if (const hl::List* l = std::get_if<hl::List>(&v.v)) {
    std::vector<Value> items;
    items.reserve(l->size());
    for (const auto& e : *l) items.push_back(from_highlevel(e));
    return from_list(std::move(items));
  }
  if (const hl::Dict* d = std::get_if<hl::Dict>(&v.v)) {
    Value out = make_dict();
    out.dict.reserve(d->size());
    for (const auto& e : *d) out.dict.emplace_back(e.key, from_highlevel(e.val));
    return out;
  }
  return none();
}

std::vector<uint8_t> encode(const Value& v) { return hl::encode_value(v.to_highlevel()); }

std::vector<uint8_t> frame(const Value& v) { return hl::encode_frame(v.to_highlevel()); }

Expected<Value> decode(const uint8_t* p, size_t n) {
  auto d = hl::decode_value(p, n);
  if (!d.ok) return make_fail<Value>(d.code.empty() ? std::string("CODEC_MALFORMED") : d.code_prefix());
  return make_ok(Value::from_highlevel(d.value));
}

Expected<Value> decode(const std::vector<uint8_t>& v) { return decode(v.data(), v.size()); }

Expected<Value> decode_frame(const std::vector<uint8_t>& f) {
  if (f.size() < 8) return make_fail<Value>("WIRE_TRUNCATED");
  const uint64_t len = hl::get_u64be(f.data());
  if (len != f.size() - 8) return make_fail<Value>("WIRE_BAD_LENGTH");
  return decode(f.data() + 8, f.size() - 8);
}

}  // namespace wire
}  // namespace platform
}  // namespace opencrab2
