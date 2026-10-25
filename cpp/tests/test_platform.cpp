// tests/test_platform.cpp: unit and loopback tests of the platform layer.
//
// Prints "PASS <n> checks" and exits 0, or prints every failing check and
// exits 1. Everything runs in process on 127.0.0.1 ephemeral ports; no
// external service, no network beyond loopback, deterministic dealer seed
// (INSECURE_TEST_ONLY_SHARED_SEED) so that failures reproduce.
#include <opencrab2/highlevel/dealer.hpp>
#include <opencrab2/highlevel/ledger.hpp>
#include <opencrab2/platform/dealer_service.hpp>
#include <opencrab2/platform/evidence.hpp>
#include <opencrab2/platform/http.hpp>
#include <opencrab2/platform/model_server.hpp>
#include <opencrab2/platform/session_bridge.hpp>
#include <opencrab2/platform/shaper.hpp>
#include <opencrab2/platform/sidecar.hpp>
#include <opencrab2/platform/support.hpp>
#include <opencrab2/platform/tcp.hpp>
#include <opencrab2/platform/websocket.hpp>

#if __has_include(<opencrab2/lowlevel/trunc.hpp>)
#include <opencrab2/lowlevel/trunc.hpp>
#define TEST_HAVE_LOWLEVEL 1
#endif

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <thread>

#ifndef _WIN32
#include <sys/stat.h>
#endif

using namespace opencrab2::platform;
using opencrab2::Expected;
namespace hl = opencrab2::highlevel;
namespace ll = opencrab2::lowlevel;

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond)                                                      \
  do {                                                                   \
    if (cond) {                                                          \
      ++g_pass;                                                          \
    } else {                                                             \
      ++g_fail;                                                          \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
    }                                                                    \
  } while (0)

#define CHECK_EQ_STR(a, b)                                                                              \
  do {                                                                                                  \
    std::string _a = (a), _b = (b);                                                                     \
    if (_a == _b) {                                                                                     \
      ++g_pass;                                                                                         \
    } else {                                                                                            \
      ++g_fail;                                                                                         \
      std::printf("FAIL %s:%d: %s == %s\n  got:  %s\n  want: %s\n", __FILE__, __LINE__, #a, #b, _a.c_str(), _b.c_str()); \
    }                                                                                                   \
  } while (0)

#define REQUIRE(cond)                                                        \
  do {                                                                       \
    if (cond) {                                                              \
      ++g_pass;                                                              \
    } else {                                                                 \
      ++g_fail;                                                              \
      std::printf("FAIL (fatal) %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
      return;                                                                \
    }                                                                        \
  } while (0)

namespace {

// A file of this process in the temp directory (TEMP, TMPDIR or /tmp): the test writes nothing
// into the working directory and concurrent runs never share a file.
std::string temp_path(const std::string& name) {
  const char* base = std::getenv("TEMP");
  if (!base) base = std::getenv("TMPDIR");
  if (!base) base = "/tmp";
  static const long long stamp = static_cast<long long>(std::chrono::steady_clock::now().time_since_epoch().count());
  return std::string(base) + "/oc2_platform_" + std::to_string(stamp) + "_" + name;
}

std::string hex(const std::vector<uint8_t>& v) { return detail::hex_encode(v); }
std::vector<uint8_t> bytes_of(const std::string& s) { return std::vector<uint8_t>(s.begin(), s.end()); }

// ---------------------------------------------------------------------------
void test_crypto() {
  CHECK_EQ_STR(detail::hex_encode(detail::sha1(std::string("abc")).data(), 20), "a9993e364706816aba3e25717850c26c9cd0d89d");
  CHECK_EQ_STR(detail::hex_encode(detail::sha1(std::string("")).data(), 20), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
  CHECK_EQ_STR(detail::hex_encode(detail::sha256(std::string("abc")).data(), 32),
               "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  CHECK_EQ_STR(detail::hex_encode(detail::sha256(std::string(1000, 'a')).data(), 32),
               "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3");
  // RFC 4231 test case 2
  std::vector<uint8_t> key(20, 0x0b);
  auto t = detail::hmac_sha256(key.data(), key.size(), reinterpret_cast<const uint8_t*>("Hi There"), 8);
  CHECK_EQ_STR(detail::hex_encode(t.data(), 32), "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7");
  // RFC 4648 section 10
  CHECK_EQ_STR(detail::base64_encode(bytes_of("")), "");
  CHECK_EQ_STR(detail::base64_encode(bytes_of("f")), "Zg==");
  CHECK_EQ_STR(detail::base64_encode(bytes_of("fo")), "Zm8=");
  CHECK_EQ_STR(detail::base64_encode(bytes_of("foo")), "Zm9v");
  CHECK_EQ_STR(detail::base64_encode(bytes_of("foob")), "Zm9vYg==");
  CHECK_EQ_STR(detail::base64_encode(bytes_of("fooba")), "Zm9vYmE=");
  CHECK_EQ_STR(detail::base64_encode(bytes_of("foobar")), "Zm9vYmFy");
  auto dec = detail::base64_decode("Zm9vYmE=");
  CHECK(dec.ok && std::string(dec.value.begin(), dec.value.end()) == "fooba");
  CHECK(!detail::base64_decode("Zm9").ok);
  CHECK(!detail::base64_decode("Zm9v*mE=").ok);
  // RFC 7539 section 2.3.2: the block function behind the platform CSPRNG stream (lowlevel)
  uint8_t k[32];
  for (int i = 0; i < 32; ++i) k[i] = static_cast<uint8_t>(i);
  uint8_t n[12] = {0, 0, 0, 9, 0, 0, 0, 0x4a, 0, 0, 0, 0};
  uint8_t blk[64];
  ll::chacha20_block(k, 1, n, blk);
  CHECK_EQ_STR(detail::hex_encode(blk, 64),
               "10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4e"
               "d2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e");
  // one implementation each: the platform spellings ARE the lowlevel functions
  CHECK(detail::sha1(std::string("abc")) == ll::Sha1::digest(std::string("abc")));
  CHECK(detail::sha256(std::string("abc")) == ll::Sha256::digest(std::string("abc")));
  CHECK(detail::hmac_sha256(key, std::string("Hi There")) == t);
  CHECK_EQ_STR(detail::hmac_sha256_hex(key, "Hi There"), "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7");
  CHECK_EQ_STR(detail::base64_encode(reinterpret_cast<const uint8_t*>("foob"), 4), "Zm9vYg==");
  CHECK_EQ_STR(detail::base64_decode("Zm9").code, "BASE64_LENGTH");
  // the CSPRNG stream of the platform is the lowlevel Rng
  {
    detail::ChaCha20 a = detail::ChaCha20::from_seed(5);
    ll::Rng b = ll::Rng::from_seed(5);
    bool same = true;
    for (int i = 0; i < 64; ++i) same = same && a.next_u64() == b.next_u64();
    CHECK(same);
    std::vector<uint64_t> u(8);
    detail::ChaCha20 c = detail::ChaCha20::from_seed(5);
    c.fill_u64(u.data(), u.size());
    ll::Rng d = ll::Rng::from_seed(5);
    CHECK(u[0] == d.next_u64() && u[1] == d.next_u64());
    // a default-constructed stream and from_os are OS seeded: two of them never agree
    detail::ChaCha20 o1, o2, o3 = detail::ChaCha20::from_os();
    uint64_t v1 = o1.next_u64(), v2 = o2.next_u64(), v3 = o3.next_u64();
    CHECK(v1 != v2 && v2 != v3 && v1 != v3);
    CHECK(detail::ChaCha20::from_seed(5).bytes(16) == detail::ChaCha20::from_seed(5).bytes(16));
  }
  auto hx = detail::hex_decode("00ff10");
  CHECK(hx.ok && hx.value.size() == 3 && hx.value[1] == 0xff);
  CHECK(!detail::hex_decode("0g").ok);
  CHECK_EQ_STR(detail::hex_decode("0g").code, "HEX_ALPHABET");
  CHECK_EQ_STR(detail::hex_decode("0").code, "HEX_LENGTH");
  CHECK(detail::hex_nibble('f') == 15 && detail::hex_nibble('g') == -1);
  CHECK(detail::constant_time_equal(std::string("abc"), std::string("abc")));
  CHECK(!detail::constant_time_equal(std::string("abc"), std::string("abd")));
  CHECK(!detail::constant_time_equal(std::string("abc"), std::string("ab")));
  auto r1 = detail::os_random_bytes(32), r2 = detail::os_random_bytes(32);
  CHECK(r1 != r2);
}

// ---------------------------------------------------------------------------
void test_json() {
  auto j = json::parse(std::string("{" "\x22" "b\x22" ":[1,2.5,\x22" "x\x5cu00e9\x5cud83d\x5cude00\x22" "]," "\x22" "a\x22" ":{\x22" "z\x22" ":true,\x22" "y\x22" ":null}," "\x22" "n\x22" ":-12}"));
  REQUIRE(j.ok);
  // byte-identical to Python json.dumps(obj, sort_keys=True, separators=(",", ":"))
  CHECK_EQ_STR(j.value.dump(true), std::string("{" "\x22" "a\x22" ":{\x22" "y\x22" ":null,\x22" "z\x22" ":true}," "\x22" "b\x22" ":[1,2.5,\x22" "x\x5cu00e9\x5cud83d\x5cude00\x22" "]," "\x22" "n\x22" ":-12}"));
  CHECK(j.value.int_or("n", 0) == -12);
  CHECK(j.value.find("b")->arr.size() == 3);
  CHECK(!json::parse("{").ok);
  CHECK(!json::parse("[1,]").ok);
  CHECK(!json::parse("{\"a\":1} x").ok);
  CHECK(!json::parse("\"abc").ok);
  CHECK(json::parse("  [ ]  ").ok);
  json::Value o = json::Value::object();
  o.set("k", json::Value::string("v\"\\\n"));
  o.set("f", json::Value::number(0.1));
  o.set("i", json::Value::integer(-5));
  CHECK_EQ_STR(o.dump(true), R"({"f":0.1,"i":-5,"k":"v\"\\\n"})");
  // one JSON implementation: the text is highlevel::Json's, both dump spellings are canonical
  CHECK_EQ_STR(o.dump(false), o.dump(true));
  CHECK_EQ_STR(o.dump(), o.to_highlevel().dump());
  CHECK_EQ_STR(hl::format_double_repr(100.0), "100.0");
  CHECK_EQ_STR(hl::format_double_repr(1e-05), "1e-05");
  CHECK_EQ_STR(json::parse("{").code, "JSON_PARSE");
  // nesting depth fails closed in the shared parser (a deep document cannot exhaust the stack)
  std::string deep(100000, '[');
  CHECK_EQ_STR(json::parse(deep).code, "JSON_DEPTH");
  // conversions keep types: an integer stays an integer, a float a float
  auto back = json::Value::from_highlevel(j.value.to_highlevel());
  CHECK_EQ_STR(back.dump(), j.value.dump());
  CHECK(back.find("n")->is_int && !back.find("b")->arr[1].is_int);
  json::Value nonfinite = json::Value::object();
  nonfinite.set("x", json::Value::number(1.5));
  CHECK_EQ_STR(nonfinite.dump(), R"({"x":1.5})");
}

// ---------------------------------------------------------------------------
void test_wire() {
  wire::Value dv = wire::Value::make_dict();
  dv.set("type", wire::Value::from_str("HELLO"));
  dv.set("n", wire::Value::from_int(-5));
  dv.set("f", wire::Value::from_float(1.5));
  dv.set("arr", wire::Value::from_u64_vector({1, 2, 3}));
  dv.set("ok", wire::Value::from_bool(true));
  dv.set("none", wire::Value::none());
  dv.set("raw", wire::Value::from_bytes({9, 8, 7}));
  auto fr = wire::frame(dv);
  // reference bytes produced by an independent Python encoder of the wire codec
  CHECK_EQ_STR(hex(fr),
               "000000000000006f0300000007000474797065060000000548454c4c4f00016e04fffffffffffffffb0001660"
               "53ff80000000000000003617272020101000000000000000301000000000000000200000000000000030000000"
               "000000000026f6b090100046e6f6e650800037261770100000003090807");
  auto back = wire::decode_frame(fr);
  REQUIRE(back.ok);
  CHECK_EQ_STR(back.value.str_or("type", ""), "HELLO");
  CHECK(back.value.int_or("n", 0) == -5);
  CHECK(back.value.float_or("f", 0) == 1.5);
  CHECK(back.value.bool_or("ok", false));
  CHECK(back.value.get("none") && back.value.get("none")->is_none());
  CHECK(back.value.get("arr")->array.to_u64() == std::vector<uint64_t>({1, 2, 3}));
  CHECK(back.value.get("raw")->bytes == std::vector<uint8_t>({9, 8, 7}));
  // tags: none = 0x08, bool = 0x09 (codec.hpp)
  CHECK(wire::encode(wire::Value::none()) == std::vector<uint8_t>({0x08}));
  CHECK(wire::encode(wire::Value::from_bool(false)) == std::vector<uint8_t>({0x09, 0x00}));
  // truncation and bad tags fail closed
  std::vector<uint8_t> body = wire::encode(dv);
  CHECK(!wire::decode(body.data(), body.size() - 1).ok);
  std::vector<uint8_t> bad = {0x0c};
  CHECK(!wire::decode(bad).ok);
  std::vector<uint8_t> short_frame = {0, 0, 0, 0, 0, 0, 0, 5, 0x08};
  CHECK(!wire::decode_frame(short_frame).ok);
  // json view hides payload bytes
  CHECK(back.value.to_json().dump().find("090807") == std::string::npos);
  // one codec implementation: the platform bytes are the highlevel encoder's bytes
  CHECK(wire::encode(dv) == hl::encode_value(dv.to_highlevel()));
  CHECK(wire::frame(dv) == hl::encode_frame(dv.to_highlevel()));
  auto hv = hl::decode_value(body);
  REQUIRE(hv.ok);
  CHECK(wire::encode(wire::Value::from_highlevel(hv.value)) == body);
  // stable codes of the shared decoder and of the frame prefix check
  CHECK_EQ_STR(wire::decode(body.data(), body.size() - 1).code, "CODEC_TRUNCATED");
  CHECK_EQ_STR(wire::decode(bad).code, "CODEC_TAG");
  CHECK_EQ_STR(wire::decode_frame(short_frame).code, "WIRE_BAD_LENGTH");
  CHECK_EQ_STR(wire::decode_frame({0, 0, 0}).code, "WIRE_TRUNCATED");
  std::vector<uint8_t> trailing = body;
  trailing.push_back(0x08);
  CHECK_EQ_STR(wire::decode(trailing).code, "CODEC_TRAILING");
  // a peer frame of 50000 nested lists (250 KB) is refused, not recursed into
  {
    std::vector<uint8_t> nested;
    for (int i = 0; i < 50000; ++i) {
      nested.push_back(0x07);
      nested.push_back(0);
      nested.push_back(0);
      nested.push_back(0);
      nested.push_back(1);
    }
    nested.push_back(0x08);
    CHECK_EQ_STR(wire::decode(nested).code, "CODEC_DEPTH");
    // the same frame through the frame entry point and inside a dict value
    std::vector<uint8_t> framed(8, 0);
    for (int i = 0; i < 8; ++i) framed[static_cast<size_t>(i)] = static_cast<uint8_t>(static_cast<uint64_t>(nested.size()) >> (8 * (7 - i)));
    framed.insert(framed.end(), nested.begin(), nested.end());
    CHECK_EQ_STR(wire::decode_frame(framed).code, "CODEC_DEPTH");
  }
}

// ---------------------------------------------------------------------------
void test_contract() {
  CHECK_EQ_STR(contract::hash(), "9c357c649b5264717a658c4edd57fcb149ef354c8759809088f8ee20896f4b88");
  CHECK_EQ_STR(contract::hash("NC-v2-f16-l46-smollm"), "91e1988b7bd81214b77b25e39a5a50880663c93d3a03f042172118dfeeeb769f");
  // the hash is the highlevel contract's; an unknown name fails closed
  CHECK_EQ_STR(contract::hash(), hl::default_contract().contract_hash());
  CHECK_EQ_STR(contract::hash("NC-v2-f16-l46-smollm"), hl::smollm_contract().contract_hash());
  CHECK(!contract::hash_checked("NC-nope").ok);
  CHECK_EQ_STR(contract::hash("NC-nope"), "");
}

// ---------------------------------------------------------------------------
void test_tcp() {
  auto hp = parse_host_port("127.0.0.1:8787");
  CHECK(hp.ok && hp.value.first == "127.0.0.1" && hp.value.second == 8787);
  hp = parse_host_port("[::1]:80");
  CHECK(hp.ok && hp.value.first == "::1" && hp.value.second == 80);
  hp = parse_host_port("9000");
  CHECK(hp.ok && hp.value.first == "127.0.0.1" && hp.value.second == 9000);
  CHECK(!parse_host_port("host:99999").ok);
  CHECK(!parse_host_port("host:").ok);
  CHECK(!parse_host_port("").ok);

  auto l = TcpListener::bind("127.0.0.1", 0, 4);
  REQUIRE(l.ok);
  CHECK(l.value.port() != 0);
  auto to = l.value.accept(0.05);
  CHECK(!to.ok && to.code == "TCP_TIMEOUT");
  uint16_t port = l.value.port();
  std::thread srv([&l]() {
    auto c = l.value.accept(5.0);
    if (!c.ok) return;
    auto f = c.value.recv_frame();
    if (!f.ok) return;
    std::vector<uint8_t> echo = f.value;
    echo.push_back(0xee);
    c.value.send_frame(echo);
    // then stay silent so the client's recv times out
    sleep_s(0.3);
  });
  auto c = TcpSocket::connect("127.0.0.1", port, 2.0);
  REQUIRE(c.ok);
  std::vector<uint8_t> msg = {1, 2, 3};
  CHECK(c.value.send_frame(msg).ok);
  auto f = c.value.recv_frame();
  CHECK(f.ok && f.value == std::vector<uint8_t>({1, 2, 3, 0xee}));
  CHECK(c.value.set_timeout(0.05).ok);
  auto f2 = c.value.recv_frame();
  CHECK(!f2.ok && f2.code == "TCP_TIMEOUT");
  srv.join();
  CHECK(!TcpSocket::connect("127.0.0.1", 1, 0.3).ok);  // nothing listens on port 1
  // A refused connection is TCP_REFUSED, not TCP_TIMEOUT: a port that was bound a moment ago
  // and is closed again answers the SYN with a reset. Timed (non-blocking) and blocking path.
  uint16_t closed_port = 0;
  {
    auto tmp = TcpListener::bind("127.0.0.1", 0, 1);
    REQUIRE(tmp.ok);
    closed_port = tmp.value.port();
  }
  double r0 = monotonic_s();
  auto refused = TcpSocket::connect("127.0.0.1", closed_port, 5.0);
  double refused_s = monotonic_s() - r0;
  CHECK_EQ_STR(refused.code, "TCP_REFUSED");
  CHECK(!refused.ok && refused_s < 1.5);  // reported at once on loopback, far inside the 5 s timeout
  auto refused_blocking = TcpSocket::connect("127.0.0.1", closed_port, 0.0);
  CHECK_EQ_STR(refused_blocking.code, "TCP_REFUSED");
  // retries still end with the refusal and stay bounded
  r0 = monotonic_s();
  auto refused_retry = TcpSocket::connect("127.0.0.1", closed_port, 2.0, 3, 0.01);
  CHECK_EQ_STR(refused_retry.code, "TCP_REFUSED");
  CHECK(monotonic_s() - r0 < 3.0);
  // the WebSocket client bounds its connect phase by the caller's timeout and reports the refusal
  r0 = monotonic_s();
  auto ws_refused = websocket_connect("127.0.0.1", closed_port, "/session", "", {}, 0.5);
  CHECK_EQ_STR(ws_refused.code, "TCP_REFUSED");
  CHECK(monotonic_s() - r0 < 2.0);
}

// ---------------------------------------------------------------------------
void test_http() {
  auto r = parse_http_request("GET /v1/stream?job_id=job-1&x=a%20b HTTP/1.1\r\nHost: h\r\nOrigin: http://127.0.0.1:1\r\n\r\n");
  REQUIRE(r.ok);
  CHECK_EQ_STR(r.value.method, "GET");
  CHECK_EQ_STR(r.value.path, "/v1/stream");
  CHECK_EQ_STR(r.value.query_param("job_id", ""), "job-1");
  CHECK_EQ_STR(r.value.query_param("x", ""), "a b");
  CHECK_EQ_STR(r.value.header_or("origin", ""), "http://127.0.0.1:1");
  auto p = parse_http_request("POST /v1/generate HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: 13\r\n\r\n{\"prompt\":1}\n");
  REQUIRE(p.ok);
  CHECK_EQ_STR(p.value.body, "{\"prompt\":1}\n");
  auto inc = parse_http_request("POST /x HTTP/1.1\r\nContent-Length: 10\r\n\r\nabc");
  CHECK(!inc.ok && inc.code == "HTTP_INCOMPLETE");
  auto ch = parse_http_request("POST /x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n");
  CHECK(!ch.ok && ch.code == "HTTP_UNSUPPORTED");
  CHECK(!parse_http_request("GET /x\r\n\r\n").ok);
  HttpRequestParser small(64, 1024);
  std::string big = "GET / HTTP/1.1\r\nX: " + std::string(200, 'a') + "\r\n\r\n";
  auto tl = small.feed(reinterpret_cast<const uint8_t*>(big.data()), big.size());
  CHECK(!tl.ok && tl.code == "HTTP_TOO_LARGE");
  // incremental feed
  HttpRequestParser inc2;
  std::string whole = "GET /a HTTP/1.1\r\nHost: x\r\n\r\n";
  bool complete_before_end = false;
  for (size_t i = 0; i < whole.size(); ++i) {
    auto fr = inc2.feed(reinterpret_cast<const uint8_t*>(whole.data() + i), 1);
    CHECK(fr.ok);
    if (fr.value && i + 1 < whole.size()) complete_before_end = true;
  }
  CHECK(!complete_before_end && inc2.complete());

  HttpResponse resp = HttpResponse::json(201, json::Value::object());
  std::string ser = resp.serialize();
  CHECK(ser.find("HTTP/1.1 201 Created\r\n") == 0);
  CHECK(ser.find("Content-Length: 2\r\n") != std::string::npos);
  CHECK(ser.find("Connection: close\r\n") != std::string::npos);
  CHECK(ser.find("\r\n\r\n{}") != std::string::npos);

  HttpRequest req;
  req.headers.emplace_back("origin", "HTTP://127.0.0.1:8787/");
  CHECK(origin_allowed(req, {"http://127.0.0.1:8787"}, false));
  CHECK(!origin_allowed(req, {"http://127.0.0.1:8788"}, false));
  HttpRequest none;
  CHECK(origin_allowed(none, {"http://a"}, true));
  CHECK(!origin_allowed(none, {"http://a"}, false));
  HttpRequest nul;
  nul.headers.emplace_back("origin", "null");
  CHECK(!origin_allowed(nul, {"http://a"}, true));
  HttpRequest auth;
  auth.headers.emplace_back("authorization", "Bearer  abc123 ");
  CHECK(bearer_token_ok(auth, "abc123"));
  CHECK(!bearer_token_ok(auth, "abc124"));
  CHECK(!bearer_token_ok(auth, ""));
  CHECK(!bearer_token_ok(none, "abc123"));
  CHECK_EQ_STR(SseStream::format_event("token", "a\nb", "7"), "id: 7\nevent: token\ndata: a\ndata: b\n\n");

  // loopback server
  auto l = TcpListener::bind("127.0.0.1", 0, 4);
  REQUIRE(l.ok);
  HttpServer server(std::move(l.value), [](HttpServer::Exchange& ex) {
    if (ex.request.path == "/hello") {
      json::Value o = json::Value::object();
      o.set("m", json::Value::string(ex.request.method));
      o.set("len", json::Value::integer(static_cast<int64_t>(ex.request.body.size())));
      ex.respond(HttpResponse::json(200, o));
    } else {
      ex.respond(HttpResponse::error(404, "NOT_FOUND"));
    }
  });
  REQUIRE(server.start().ok);
  HttpRequest q;
  q.method = "POST";
  q.target = "/hello";
  q.body = "12345";
  auto res = http_request("127.0.0.1", server.port(), q, 5.0);
  REQUIRE(res.ok);
  CHECK(res.value.status == 200);
  auto body = json::parse(res.value.body);
  CHECK(body.ok && body.value.str_or("m", "") == "POST" && body.value.int_or("len", 0) == 5);
  q.method = "GET";
  q.target = "/missing";
  q.body.clear();
  auto nf = http_request("127.0.0.1", server.port(), q, 5.0);
  CHECK(nf.ok && nf.value.status == 404);
  // malformed request gets 400 and the server survives
  auto raw = TcpSocket::connect("127.0.0.1", server.port(), 2.0);
  REQUIRE(raw.ok);
  raw.value.send_all(std::string("BOGUS\r\n\r\n"));
  auto bad = read_http_response(raw.value, 2.0);
  CHECK(bad.ok && bad.value.status == 400);
  auto again = http_request("127.0.0.1", server.port(), q, 5.0);
  CHECK(again.ok && again.value.status == 404);
  CHECK(server.connections_total() >= 3);
  server.stop();
}

// ---------------------------------------------------------------------------
void test_websocket() {
  // RFC 6455 section 1.3
  CHECK_EQ_STR(websocket_accept_key("dGhlIHNhbXBsZSBub25jZQ=="), "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");
  auto hs = parse_http_request(
      "GET /session HTTP/1.1\r\nHost: s\r\nUpgrade: websocket\r\nConnection: keep-alive, Upgrade\r\n"
      "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n");
  REQUIRE(hs.ok);
  CHECK(is_websocket_upgrade(hs.value));
  auto resp = websocket_handshake_response(hs.value);
  REQUIRE(resp.ok);
  CHECK(resp.value.status == 101);
  CHECK_EQ_STR(*resp.value.header("Sec-WebSocket-Accept"), "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");
  HttpRequest v12 = hs.value;
  for (auto& kv : v12.headers) {
    if (kv.first == "sec-websocket-version") kv.second = "12";
  }
  CHECK(websocket_handshake_response(v12).code == "WS_BAD_VERSION");
  HttpRequest nokey = hs.value;
  nokey.headers.erase(nokey.headers.begin() + 3);  // Sec-WebSocket-Key
  CHECK(!websocket_handshake_response(nokey).ok);

  // frame codec: 7-bit, 16-bit and 64-bit lengths, masked and unmasked
  uint8_t mk[4] = {0x37, 0xfa, 0x21, 0x3d};
  for (size_t n : {size_t(5), size_t(300), size_t(70000)}) {
    std::vector<uint8_t> payload(n);
    for (size_t i = 0; i < n; ++i) payload[i] = static_cast<uint8_t>(i * 7 + 1);
    for (bool masked : {false, true}) {
      std::vector<uint8_t> f = ws_encode_frame(WsOpcode::Binary, payload.data(), n, true, masked, mk);
      auto h = ws_decode_header(f.data(), f.size());
      REQUIRE(h.ok);
      CHECK(h.value.fin && h.value.opcode == WsOpcode::Binary && h.value.masked == masked && h.value.payload_len == n);
      size_t expect_hdr = (n < 126 ? 2 : (n <= 0xFFFF ? 4 : 10)) + (masked ? 4 : 0);
      CHECK(h.value.header_len == expect_hdr);
      std::vector<uint8_t> body(f.begin() + static_cast<long>(h.value.header_len), f.end());
      if (masked) {
        CHECK(body != payload);  // scrambled on the wire
        ws_apply_mask(body.data(), body.size(), h.value.mask);
      }
      CHECK(body == payload);
    }
  }
  // RFC 6455 section 5.7: single-frame masked text "Hello"
  uint8_t hello_mask[4] = {0x37, 0xfa, 0x21, 0x3d};
  std::vector<uint8_t> hello = ws_encode_frame(WsOpcode::Text, reinterpret_cast<const uint8_t*>("Hello"), 5, true, true, hello_mask);
  CHECK_EQ_STR(hex(hello), "818537fa213d7f9f4d5158");
  uint8_t bad_rsv[2] = {0xC2, 0x00};
  CHECK(ws_decode_header(bad_rsv, 2).code == "WS_PROTOCOL_ERROR");
  uint8_t bad_op[2] = {0x83, 0x00};
  CHECK(ws_decode_header(bad_op, 2).code == "WS_PROTOCOL_ERROR");
  uint8_t ctl_big[4] = {0x89, 0x7E, 0x00, 0x80};
  CHECK(ws_decode_header(ctl_big, 4).code == "WS_PROTOCOL_ERROR");
  uint8_t partial[1] = {0x82};
  CHECK(ws_decode_header(partial, 1).code == "WS_INCOMPLETE");

  // loopback echo server over HttpServer (handler hijacks the socket)
  auto l = TcpListener::bind("127.0.0.1", 0, 4);
  REQUIRE(l.ok);
  std::atomic<int> server_msgs{0};
  std::atomic<int> server_pings{0};
  HttpServer server(std::move(l.value), [&](HttpServer::Exchange& ex) {
    auto ws = websocket_accept(std::move(ex.socket), ex.request);
    ex.hijacked = true;
    ex.responded = true;
    if (!ws.ok) return;
    if (ex.request.path == "/channel") {
      WebSocketChannel ch(std::move(ws.value), 0);
      wire::Value mine = wire::Value::make_dict();
      mine.set("from", wire::Value::from_str("server"));
      auto got = ch.exchange(mine);
      if (got.ok) server_msgs++;
      ch.close(1000);
      return;
    }
    while (true) {
      auto m = ws.value.recv_message();
      if (!m.ok || m.value.is_close()) break;
      server_msgs++;
      if (m.value.is_text()) {
        ws.value.send_text(m.value.text());
      } else {
        ws.value.send_binary(m.value.payload);
      }
    }
    server_pings += static_cast<int>(ws.value.stats().pings_answered);
  });
  REQUIRE(server.start().ok);
  uint16_t port = server.port();
  auto ws = websocket_connect("127.0.0.1", port, "/echo", "http://127.0.0.1", {}, 5.0);
  REQUIRE(ws.ok);
  CHECK(ws.value.is_client());
  std::vector<uint8_t> payload(1000);
  for (size_t i = 0; i < payload.size(); ++i) payload[i] = static_cast<uint8_t>(i);
  CHECK(ws.value.send_binary(payload).ok);
  auto echo = ws.value.recv_message();
  CHECK(echo.ok && echo.value.is_binary() && echo.value.payload == payload);
  CHECK(ws.value.send_text("hi there").ok);
  auto echo2 = ws.value.recv_message();
  CHECK(echo2.ok && echo2.value.is_text() && echo2.value.text() == "hi there");
  // ping is answered with a pong carrying the same payload
  CHECK(ws.value.send_ping({1, 2, 3}).ok);
  auto pong = ws.value.recv_frame();
  CHECK(pong.ok && pong.value.opcode == WsOpcode::Pong && pong.value.payload == std::vector<uint8_t>({1, 2, 3}));
  // fragmentation: two raw masked fragments are reassembled by the server
  {
    uint8_t k1[4] = {1, 2, 3, 4}, k2[4] = {5, 6, 7, 8};
    std::vector<uint8_t> a(200, 0xaa), b(300, 0xbb);
    std::vector<uint8_t> f1 = ws_encode_frame(WsOpcode::Binary, a.data(), a.size(), false, true, k1);
    std::vector<uint8_t> f2 = ws_encode_frame(WsOpcode::Continuation, b.data(), b.size(), true, true, k2);
    CHECK(ws.value.socket().send_all(f1).ok);
    CHECK(ws.value.socket().send_all(f2).ok);
    auto re = ws.value.recv_message();
    std::vector<uint8_t> want = a;
    want.insert(want.end(), b.begin(), b.end());
    CHECK(re.ok && re.value.payload == want);
  }
  // an unmasked client frame is a protocol error for the server (connection drops)
  {
    auto ws2 = websocket_connect("127.0.0.1", port, "/echo", "", {}, 5.0);
    REQUIRE(ws2.ok);
    uint8_t zero[4] = {0, 0, 0, 0};
    std::vector<uint8_t> raw = ws_encode_frame(WsOpcode::Binary, payload.data(), 10, true, false, zero);
    CHECK(ws2.value.socket().send_all(raw).ok);
    ws2.value.set_timeout(2.0);
    auto r = ws2.value.recv_message();
    CHECK(!r.ok);  // server closed without echoing
  }
  // close handshake
  CHECK(ws.value.send_close(1000, "bye").ok);
  auto cl = ws.value.recv_message();
  CHECK(cl.ok && cl.value.is_close() && cl.value.close_code == 1000);
  CHECK(!ws.value.recv_message().ok);
  WsStats st = ws.value.stats();
  // sent: binary, text, ping, close (the two raw fragments bypassed the counter);
  // received: 2 echoes, pong, the reassembled echo (one frame), close
  CHECK(st.frames_sent == 4 && st.frames_recv == 5);
  // channel adapter: wire-codec exchange with round and byte accounting
  {
    auto wsc = websocket_connect("127.0.0.1", port, "/channel", "", {}, 5.0);
    REQUIRE(wsc.ok);
    WebSocketChannel ch(std::move(wsc.value), 1);
    wire::Value mine = wire::Value::make_dict();
    mine.set("from", wire::Value::from_str("client"));
    mine.set("x", wire::Value::from_u64_vector({7, 8, 9}));
    auto got = ch.exchange(mine);
    REQUIRE(got.ok);
    CHECK_EQ_STR(got.value.str_or("from", ""), "server");
    CHECK(ch.stats.rounds == 1 && ch.stats.messages_sent == 1 && ch.stats.messages_recv == 1);
    CHECK(ch.stats.bytes_sent == wire::frame(mine).size());
    auto closed = ch.recv();
    CHECK(!closed.ok && closed.code == "WS_CLOSED");
    CHECK(ch.stats_json().int_or("rounds", 0) == 1);
  }
  sleep_s(0.1);
  CHECK(server_msgs.load() >= 4);
  server.stop();
  CHECK(server_pings.load() >= 1);
  auto u = parse_ws_url("ws://127.0.0.1:8765/session");
  CHECK(u.ok && u.value.host == "127.0.0.1" && u.value.port == 8765 && u.value.path == "/session");
  CHECK(!parse_ws_url("wss://x/").ok);
  CHECK(!parse_ws_url("http://x/").ok);
}

// ---------------------------------------------------------------------------
void test_shaper() {
  auto t = parse_trace_json(R"({"trace_id":"S1-ramp","trace_hash":"","family":"S1","split":"test","seed":3,
      "points":[[0.0, 8000, 0.01, 1.0, 0], [1.0, 16000, 0.02, 0.5, 0], [5.0, 16000, 0.02, 0.5, 0]]})");
  REQUIRE(t.ok);
  CHECK(t.value.points.size() == 3);
  CHECK(!t.value.trace_hash.empty());
  CHECK(t.value.at(0.5).bandwidth_bps == 8000);
  CHECK(t.value.at(1.0).bandwidth_bps == 16000);
  CHECK(t.value.at(100.0).cpu_share == 0.5);
  CHECK(t.value.at(-1.0).bandwidth_bps == 8000);
  CHECK(!parse_trace_json("{}").ok);
  CHECK(parse_trace_json(R"({"points":[]})").code == "TRACE_EMPTY");
  CHECK(parse_trace_json(R"({"points":[[1,1,0],[0,1,0]]})").code == "TRACE_UNSORTED");
  CHECK(parse_trace_json(R"({"points":[[0,1,0,2.0]]})").code == "TRACE_MALFORMED");
  CHECK(!load_trace_file("does-not-exist.json").ok);
  double origin = 1000.0;
  WallClockShaper sh(t.value, origin);
  // 2000 bytes = 16000 bits from t=0: 8000 bits in the first second at 8 kbps,
  // the remaining 8000 bits take 0.5 s at 16 kbps: 1.5 s serialization + the one-way delay of
  // the state in force when the last byte leaves (the second segment, 20 ms)
  double d = sh.delay_for(2000, origin);
  CHECK(d > 1.519 && d < 1.521);
  CHECK(sh.serialization_s(2000, origin) > 1.499 && sh.serialization_s(2000, origin) < 1.501);
  double d2 = sh.delay_for(2000, origin + 1.0);  // 16000 bits at 16 kbps = 1 s + 20 ms
  CHECK(d2 > 1.019 && d2 < 1.021);
  CHECK(sh.state_at(origin + 0.5).one_way_delay_s == 0.01);
  Trace unlimited = make_constant_trace(0.0, 0.03, 10.0);
  WallClockShaper su(unlimited, origin);
  CHECK(su.delay_for(1 << 20, origin) == 0.03);
  CHECK(su.state_at(origin).unlimited());
  // apply() sleeps for the shaped delay (1 Mbps, 12500 bytes = 0.1 s)
  Trace fast = make_constant_trace(1e6, 0.0, 10.0);
  WallClockShaper sf(fast, wall_s());
  double a = monotonic_s();
  double applied = sf.apply(12500);
  double b = monotonic_s();
  CHECK(applied > 0.09 && applied < 0.12);
  CHECK(b - a > 0.08);
  CHECK(sf.messages() == 1 && sf.total_delay_s() == applied);
  // back-to-back messages queue behind each other
  double a2 = monotonic_s();
  sf.apply(12500);
  sf.apply(12500);
  CHECK(monotonic_s() - a2 > 0.18);
  TraceDriver drv(t.value, wall_s() - 2.0);  // in the cpu_share 0.5 segment
  drv.start_contention(1);
  sleep_s(0.1);
  CHECK(drv.contention_running());
  CHECK(drv.elapsed_s() > 2.0);
  drv.stop();
  CHECK(!drv.contention_running());
  CHECK(t.value.to_json().find("points")->arr.size() == 3);

  // A file written by python -m opencrab2.experiments.traces (make_trace("S3", "two-changes", 4, 12.0)):
  // the declared trace_hash is the Python points_hash and is verified against the points.
  const std::string py_trace =
      R"JSON({"trace_id":"S3:two-changes:seed4:d12","trace_hash":"b5f3798a1106f21374cb36270f5023698d1fd88143c41c088e9bb75aa2454204",)JSON"
      R"JSON("family":"S3","variant":"two-changes","seed":4,"split":"test","duration_s":12.0,)JSON"
      R"JSON("points":[[0.0,1000000000.0,0.00025,1.0,0.0],[3.87185,50000000.0,0.01,1.0,0.0],[7.141812,100000000.0,0.02,1.0,0.0],)JSON"
      R"JSON([8.697714,10000000.0,0.07,1.0,0.0],[10.713593,100000000.0,0.02,1.0,0.0]],"events":[],"initial_installed":[],"length_bucket":0})JSON";
  auto py = parse_trace_json(py_trace);
  REQUIRE(py.ok);
  CHECK(py.value.hash_verified);
  CHECK_EQ_STR(py.value.trace_hash, "b5f3798a1106f21374cb36270f5023698d1fd88143c41c088e9bb75aa2454204");
  CHECK_EQ_STR(py.value.compute_hash(), py.value.trace_hash);  // the C++ hash of five-column rows is the Python hash
  CHECK_EQ_STR(py.value.family, "S3");
  CHECK(py.value.points.size() == 5 && py.value.declared_duration_s == 12.0 && !py.value.has_contention());
  CHECK(py.value.at(9.0).bandwidth_bps == 10000000.0 && py.value.at(9.0).one_way_delay_s == 0.07);  // the spike
  {
    std::string tampered = py_trace;
    size_t pos = tampered.find("50000000.0");
    REQUIRE(pos != std::string::npos);
    tampered.replace(pos, 10, "50000001.0");
    CHECK_EQ_STR(parse_trace_json(tampered).code, "TRACE_HASH_MISMATCH");
  }
  // S7 events and S8 installed objects are parsed as public metadata; S5 refill; S4 contention flag
  auto s7 = parse_trace_json(
      R"JSON({"trace_id":"S7:cancel-mid-resume:seed2:d20","trace_hash":"e0cd702ab2a43b63d178df5c40f3d4331c6c64bc6ced0efed58040d2ebce8db5",)JSON"
      R"JSON("points":[[0.0,100000000.0,0.02,1.0,0.0]],"events":[[10.186146,"cancel"],[13.186146,"resume"]],)JSON"
      R"JSON("initial_installed":["wte","lm_head"],"length_bucket":64,"mem_limit_bytes":2147483648})JSON");
  REQUIRE(s7.ok);
  CHECK(s7.value.events.size() == 2 && s7.value.events[0].kind == "cancel" && s7.value.events[1].elapsed_s == 13.186146);
  CHECK(s7.value.initial_installed.size() == 2 && s7.value.initial_installed[0] == "wte");
  CHECK(s7.value.length_bucket == 64 && s7.value.mem_limit_bytes == 2147483648LL);
  CHECK(parse_trace_json(R"({"points":[[0,1,0]],"events":[[1.0,"explode"]]})").code == "TRACE_MALFORMED");
  auto s5 = parse_trace_json(R"({"points":[[0.0,100000000.0,0.02,0.5,10000000.0]],"initial_stock_bytes":1073741824})");
  REQUIRE(s5.ok);
  CHECK(s5.value.max_stock_refill_bps() == 10000000.0 && s5.value.initial_stock_bytes == 1073741824 && s5.value.has_contention());
  // ONE token bucket: the trace-aware adapter returns exactly what the highlevel shaper computes
  {
    const double o = 5000.0;
    WallClockShaper adapter(py.value, o);
    bool same = true;
    for (size_t nbytes : {size_t(100), size_t(65536), size_t(50000000), size_t(400000000)}) {
      for (double at : {0.0, 3.5, 3.87, 8.0, 8.69, 11.0, 30.0}) {
        hl::WallClockShaper fresh(py.value.link_points(), o);
        same = same && adapter.delay_for(nbytes, o + at) == fresh.delay_for(nbytes, o + at);
      }
    }
    CHECK(same);
    // 50 MB starting at t = 3.5 crosses the 1 Gbit/s -> 50 Mbit/s change: 0.37185 s carry 46.48 MB,
    // the rest 3.51875 MB at 50 Mbit/s takes 0.563 s; plus the one-way delay of the state in force
    // when the LAST byte leaves (the 50 Mbit/s segment), the shared rule of channel.hpp
    const double ser_want = 0.37185 + (50000000.0 * 8 - 0.37185 * 1e9) / 50e6;
    const double owd_end = adapter.state_at(o + 3.5 + ser_want).one_way_delay_s;
    const double owd_start = adapter.state_at(o + 3.5).one_way_delay_s;
    CHECK(owd_start == 0.00025 && owd_end != owd_start);
    double cross = adapter.delay_for(50000000, o + 3.5);
    CHECK(std::fabs(cross - (ser_want + owd_end)) < 1e-6);
    CHECK(std::fabs(adapter.serialization_s(50000000, o + 3.5) - ser_want) < 1e-6);
    CHECK(std::fabs(adapter.serialization_s(50000000, o + 3.5) - (cross - owd_end)) < 1e-9);
  }
  // the highlevel shaper of a Channel: Unix origin converted to this process's monotonic clock
  {
    const double origin_unix = wall_s() - 4.0;  // elapsed is 4.0 now: the 50 Mbit/s segment
    auto chs = make_channel_shaper(py.value, origin_unix);
    hl::LinkState now_state = chs->state_at(hl::now_s());
    CHECK(now_state.bandwidth_bps == 50000000.0 && now_state.one_way_delay_s == 0.01);
    CHECK(chs->state_at(hl::now_s() + 5.0).bandwidth_bps == 10000000.0);  // elapsed 9.0: the spike
    CHECK(std::fabs((hl::now_s() - chs->origin_wall_s()) - 4.0) < 0.05);
  }
}

// ---------------------------------------------------------------------------
struct DealerFixture {
  std::unique_ptr<DealerService> svc;
  std::vector<uint8_t> key0 = std::vector<uint8_t>(32, 0x11);
  std::vector<uint8_t> key1 = std::vector<uint8_t>(32, 0x22);
  std::string contract = contract::hash();
  std::string setup = "setup-test";
  uint16_t port = 0;
  bool start(uint64_t seed = 5, const std::string& mode = "INSECURE_TEST_ONLY_SHARED_SEED") {
    auto l = TcpListener::bind("127.0.0.1", 0, 8);
    if (!l.ok) return false;
    DealerConfig cfg;
    cfg.mode = mode;
    cfg.test_seed = seed;
    cfg.recipient_keys[0] = key0;
    cfg.recipient_keys[1] = key1;
    cfg.numeric_contract_hash = contract;
    cfg.setup_id = setup;
    cfg.model_version = "gpt2-4L";
    svc.reset(new DealerService(cfg, std::move(l.value)));
    port = svc->port();
    return svc->start().ok;
  }
  Expected<DealerConnection> connect(int recipient, std::vector<uint8_t> key = {}) {
    if (key.empty()) key = recipient == 0 ? key0 : key1;
    return DealerConnection::connect("127.0.0.1", port, recipient, setup, contract, key, 5.0, 20);
  }
};

void test_dealer() {
  DealerFixture fx;
  REQUIRE(fx.start());
  auto c0 = fx.connect(0);
  auto c1 = fx.connect(1);
  REQUIRE(c0.ok && c1.ok);
  CHECK_EQ_STR(c0.value.dealer_mode(), "INSECURE_TEST_ONLY_SHARED_SEED");
  wire::Value params = wire::Value::make_dict();
  params.set("shape", wire::Value::from_list({wire::Value::from_int(8)}));
  auto b0 = c0.value.request("beaver", params, 1, "g1");
  auto b1 = c1.value.request("beaver", params, 1, "g1");
  REQUIRE(b0.ok && b1.ok);
  CHECK_EQ_STR(b0.value.material_id(), b1.value.material_id());
  CHECK_EQ_STR(b0.value.material_type(), "beaver");
  CHECK(b0.value.count() == 8);
  CHECK(b0.value.header.int_or("recipient", -1) == 0 && b1.value.header.int_or("recipient", -1) == 1);
  CHECK(b0.value.header.bool_or("insecure", false));
  CHECK_EQ_STR(b0.value.header.str_or("dealer_mode", ""), "INSECURE_TEST_ONLY_SHARED_SEED");
  CHECK(b0.value.header.get("params") && b0.value.header.get("params")->is_dict());
  auto parts = b0.value.parts();
  CHECK(parts.size() == 3 && parts[0] == "a" && parts[1] == "b" && parts[2] == "c");
  CHECK_EQ_STR(c0.value.model_version(), "gpt2-4L");  // the binding the dealer announces at CHALLENGE
  CHECK_EQ_STR(c0.value.profile_id(), "SH2P-TrustedOffline-v1");
  CHECK(!c0.value.isolated());
  // header params carry the binding spelling of the Python dealer (shape, elements)
  {
    const wire::Value* hp = b0.value.header.get("params");
    REQUIRE(hp && hp->is_dict());
    CHECK(hp->get("shape") && hp->get("shape")->is_list() && hp->get("shape")->list.size() == 1 && hp->get("shape")->list[0].i == 8);
    CHECK(hp->int_or("elements", 0) == 8);
  }
  auto h0 = demo::parse_beaver_half(b0.value, 8), h1 = demo::parse_beaver_half(b1.value, 8);
  REQUIRE(h0.ok && h1.ok);
  bool triple_ok = true;
  for (size_t i = 0; i < 8; ++i) {
    uint64_t a = h0.value.a[i] + h1.value.a[i], b = h0.value.b[i] + h1.value.b[i], c = h0.value.c[i] + h1.value.c[i];
    if (c != a * b) triple_ok = false;
  }
  CHECK(triple_ok);  // test-only cross check of the two halves
  CHECK(h0.value.a != h1.value.a);
  // tag framing: HMAC over len(header) BE || canonical header || payload, computed and
  // verified by the ONE implementation of the tree (highlevel::hmac_tag_with_payload)
  CHECK_EQ_STR(b0.value.tag, header_tag(fx.key0, b0.value.header, b0.value.payload));
  CHECK(mac_input(b0.value.header, b0.value.payload).size() == 8 + b0.value.header.to_json().dump(true).size() + b0.value.payload.size());
  {
    auto t = hl::hmac_tag_with_payload(fx.key0, b0.value.header.to_highlevel(), b0.value.payload);
    CHECK(t.ok && t.value == b0.value.tag);
    auto v = hl::hmac_verify_with_payload(fx.key0, b0.value.header.to_highlevel(), b0.value.payload, b0.value.tag);
    CHECK(v.ok && v.value);
    auto w = hl::hmac_verify_with_payload(fx.key1, b0.value.header.to_highlevel(), b0.value.payload, b0.value.tag);
    CHECK(w.ok && !w.value);
  }
  // the highlevel DealerClient ingests the service bundle unchanged (tag, recipient, binding, payload layout)
  {
    const std::string journal = temp_path("ledger.jsonl");
    std::remove(journal.c_str());
    {
      hl::MaterialLedger ledger(journal, hl::dict({{"setup_id", fx.setup}, {"numeric_contract_hash", fx.contract},
                                                  {"model_version", "gpt2-4L"}, {"profile_id", "SH2P-TrustedOffline-v1"}}));
      REQUIRE(ledger.open().ok);
      hl::DealerClient client(0, ledger, fx.key0, fx.contract);
      auto ing = client.ingest(b0.value.header.to_highlevel(), b0.value.payload, b0.value.tag);
      if (!ing.ok) std::printf("  ingest: %s\n", ing.code.c_str());
      CHECK(ing.ok);
      CHECK(!client.security_eligible());  // a shared-seed bundle marks the client not eligible
      CHECK(client.ingest(b0.value.header.to_highlevel(), b0.value.payload, b0.value.tag).code_prefix() == "MATERIAL_REUSE");
      auto triple = client.take_beaver(8);
      CHECK(triple.ok && triple.value.a == h0.value.a && triple.value.c == h0.value.c);
      // a bundle of the other recipient is refused by the same client
      CHECK(!client.ingest(b1.value.header.to_highlevel(), b1.value.payload, b1.value.tag).ok);
    }
    std::remove(journal.c_str());
  }
  std::set<std::string> seen;
  CHECK(DealerConnection::verify_bundle(b0.value, 0, fx.setup, fx.contract, fx.key0, seen).ok);
  CHECK(DealerConnection::verify_bundle(b0.value, 0, fx.setup, fx.contract, fx.key0, seen).code == "MATERIAL_REUSE");
  std::set<std::string> seen2;
  CHECK(DealerConnection::verify_bundle(b0.value, 1, fx.setup, fx.contract, fx.key1, seen2).code == "BUNDLE_BAD_TAG");
  CHECK(DealerConnection::verify_bundle(b0.value, 1, fx.setup, fx.contract, fx.key0, seen2).code == "BUNDLE_WRONG_RECIPIENT");
  Bundle tampered = b0.value;
  tampered.payload[0] ^= 1;
  CHECK(DealerConnection::verify_bundle(tampered, 0, fx.setup, fx.contract, fx.key0, seen2).code == "BUNDLE_BAD_TAG");
  Bundle other_contract = b0.value;
  other_contract.header.set("numeric_contract_hash", wire::Value::from_str("00"));
  CHECK(!DealerConnection::verify_bundle(other_contract, 0, fx.setup, fx.contract, fx.key0, seen2).ok);
  // unsupported type keeps the session alive
  auto bad = c0.value.request("no_such_material", wire::Value::make_dict(), 1, "");
  CHECK(!bad.ok && bad.code == "MATERIAL_TYPE_UNSUPPORTED");
  auto again = c0.value.request("beaver", params, 1, "g2");
  CHECK(again.ok);
  CHECK(fx.svc->stats().pending_halves == 1);  // the peer's half waits
  CHECK(c1.value.request("beaver", params, 1, "g2").ok);
  CHECK(fx.svc->stats().pending_halves == 0);
  // matrix triple C = A B
  wire::Value mp = wire::Value::make_dict();
  mp.set("n", wire::Value::from_int(3));
  mp.set("k", wire::Value::from_int(4));
  mp.set("m", wire::Value::from_int(2));
  auto m0 = c0.value.request("matrix_triple", mp, 1, "mt"), m1 = c1.value.request("matrix_triple", mp, 1, "mt");
  REQUIRE(m0.ok && m1.ok);
  {
    auto P0 = le_to_u64(m0.value.payload.data(), m0.value.payload.size());
    auto P1 = le_to_u64(m1.value.payload.data(), m1.value.payload.size());
    REQUIRE(P0.size() == 12 + 8 + 6 && P1.size() == P0.size());
    std::vector<uint64_t> A(12), B(8), C(6), Cc;
    for (size_t i = 0; i < 12; ++i) A[i] = P0[i] + P1[i];
    for (size_t i = 0; i < 8; ++i) B[i] = P0[12 + i] + P1[12 + i];
    for (size_t i = 0; i < 6; ++i) C[i] = P0[20 + i] + P1[20 + i];
    ring_matmul(A, 3, 4, B, 2, Cc);
    CHECK(Cc == C);
    auto pr = m0.value.parts();
    CHECK(pr.size() == 3 && pr[0] == "A");
    // both spellings in the header: the binding shape_a / shape_b and the n, k, m the C++ client reads
    const wire::Value* hp = m0.value.header.get("params");
    REQUIRE(hp && hp->is_dict());
    CHECK(hp->int_or("n", 0) == 3 && hp->int_or("k", 0) == 4 && hp->int_or("m", 0) == 2);
    const wire::Value* sa = hp->get("shape_a");
    const wire::Value* sb = hp->get("shape_b");
    CHECK(sa && sa->is_list() && sa->list.size() == 2 && sa->list[0].i == 3 && sa->list[1].i == 4);
    CHECK(sb && sb->is_list() && sb->list.size() == 2 && sb->list[0].i == 4 && sb->list[1].i == 2);
  }
  // the Python spelling of the request is accepted and yields the same header keys
  {
    wire::Value py = wire::Value::make_dict();
    py.set("shape_a", wire::Value::from_list({wire::Value::from_int(2), wire::Value::from_int(3)}));
    py.set("shape_b", wire::Value::from_list({wire::Value::from_int(3), wire::Value::from_int(5)}));
    auto q0 = c0.value.request("matrix_triple", py, 1, "mt-py"), q1 = c1.value.request("matrix_triple", py, 1, "mt-py");
    REQUIRE(q0.ok && q1.ok);
    CHECK(q0.value.header.get("params")->int_or("n", 0) == 2 && q0.value.header.get("params")->int_or("m", 0) == 5);
    CHECK(q0.value.payload.size() == 8 * (6 + 15 + 10));
  }
  // wmask: persistent B then per-call (A, C = A B)
  wire::Value wm = wire::Value::make_dict();
  wm.set("mat_id", wire::Value::from_str("lm_head"));
  wm.set("d", wire::Value::from_int(4));
  wm.set("m", wire::Value::from_int(3));
  auto w0 = c0.value.request("wmask_mask", wm, 1, ""), w1 = c1.value.request("wmask_mask", wm, 1, "");
  REQUIRE(w0.ok && w1.ok);
  std::vector<uint64_t> B(12);
  {
    auto B0 = le_to_u64(w0.value.payload.data(), w0.value.payload.size());
    auto B1 = le_to_u64(w1.value.payload.data(), w1.value.payload.size());
    for (size_t i = 0; i < 12; ++i) B[i] = B0[i] + B1[i];
  }
  wire::Value wc = wire::Value::make_dict();
  wc.set("mat_id", wire::Value::from_str("lm_head"));
  wc.set("n_rows", wire::Value::from_int(2));
  auto cA0 = c0.value.request("wmask_call", wc, 1, ""), cA1 = c1.value.request("wmask_call", wc, 1, "");
  REQUIRE(cA0.ok && cA1.ok);
  {
    auto Q0 = le_to_u64(cA0.value.payload.data(), cA0.value.payload.size());
    auto Q1 = le_to_u64(cA1.value.payload.data(), cA1.value.payload.size());
    REQUIRE(Q0.size() == 8 + 6);
    std::vector<uint64_t> A(8), C(6), Cc;
    for (size_t i = 0; i < 8; ++i) A[i] = Q0[i] + Q1[i];
    for (size_t i = 0; i < 6; ++i) C[i] = Q0[8 + i] + Q1[8 + i];
    ring_matmul(A, 2, 4, B, 3, Cc);
    CHECK(Cc == C);
  }
  wire::Value wc_unknown = wire::Value::make_dict();
  wc_unknown.set("mat_id", wire::Value::from_str("nope"));
  wc_unknown.set("n_rows", wire::Value::from_int(2));
  CHECK(c0.value.request("wmask_call", wc_unknown, 1, "").code == "WMASK_NOT_INSTALLED");
  // vector share without values is a zero sharing
  wire::Value vs = wire::Value::make_dict();
  vs.set("name", wire::Value::from_str("bias"));
  vs.set("size", wire::Value::from_int(5));
  auto v0 = c0.value.request("vector_share", vs, 1, ""), v1 = c1.value.request("vector_share", vs, 1, "");
  REQUIRE(v0.ok && v1.ok);
  {
    auto s0 = le_to_u64(v0.value.payload.data(), v0.value.payload.size());
    auto s1 = le_to_u64(v1.value.payload.data(), v1.value.payload.size());
    bool zero = s0.size() == 5;
    for (size_t i = 0; i < 5 && zero; ++i) zero = (s0[i] + s1[i]) == 0;
    CHECK(zero);
    CHECK(v0.value.header.get("params")->int_or("size", 0) == 5);
  }
  // a header never carries material: a request with an array keeps only its size in params,
  // and the header stays taggable by the highlevel canonical JSON
  {
    wire::Value pv = wire::Value::make_dict();
    pv.set("name", wire::Value::from_str("secret_vec"));
    pv.set("values", wire::Value::from_u64_vector({0x1111111111111111ULL, 0x2222222222222222ULL, 0x3333333333333333ULL}));
    auto x0 = c0.value.request("vector_share", pv, 1, "vs-values");
    REQUIRE(x0.ok);
    const wire::Value* hv = x0.value.header.get("params")->get("values");
    CHECK(hv && hv->is_dict() && hv->int_or("__ndarray__", 0) == 24);
    std::vector<uint8_t> enc = wire::encode(x0.value.header);
    std::string as_text(enc.begin(), enc.end());
    CHECK(as_text.find(std::string(8, '\x11')) == std::string::npos);  // no element bytes in the header
    CHECK(hl::canonical_json(x0.value.header.to_highlevel()).ok);
    wire::Value nested = wire::Value::make_dict();
    nested.set("blob", wire::Value::from_bytes({1, 2, 3, 4}));
    wire::Value pub = public_params(nested);
    CHECK(pub.get("blob") && pub.get("blob")->int_or("__bytes__", 0) == 4);
    // the peer's half is requested with a zero vector of the same size (it does not know the values)
    wire::Value peer = wire::Value::make_dict();
    peer.set("name", wire::Value::from_str("secret_vec"));
    peer.set("values", wire::Value::from_u64_vector({0, 0, 0}));
    CHECK(c1.value.request("vector_share", peer, 1, "vs-values").ok);
  }
#ifdef TEST_HAVE_LOWLEVEL
  if (DealerService::lowlevel_material_available()) {
    wire::Value tp = wire::Value::make_dict();
    tp.set("M", wire::Value::from_int(4));
    tp.set("f", wire::Value::from_int(16));
    tp.set("L", wire::Value::from_int(40));
    auto t0 = c0.value.request("trunc_faithful", tp, 1, "tr"), t1 = c1.value.request("trunc_faithful", tp, 1, "tr");
    REQUIRE(t0.ok && t1.ok);
    CHECK(t0.value.count() == 4);
    auto mat0 = opencrab2::lowlevel::trunc::TruncMaterial::from_bytes(t0.value.payload);
    auto mat1 = opencrab2::lowlevel::trunc::TruncMaterial::from_bytes(t1.value.payload);
    CHECK(mat0.ok && mat1.ok);
    if (mat0.ok && mat1.ok) {
      CHECK(mat0.value.M() == 4 && mat0.value.f == 16 && mat0.value.party == 0 && mat1.value.party == 1);
      // the gate works on the delivered halves: x = -13 * 2^16 truncated by 16 bits is -13
      std::vector<uint64_t> x0(4), x1(4);
      for (size_t i = 0; i < 4; ++i) {
        uint64_t xv = static_cast<uint64_t>(static_cast<int64_t>(-13) * 65536);
        x0[i] = 0x1234567890abcdefULL + i;
        x1[i] = xv - x0[i];
      }
      // trunc_open_share applies P0's public offset itself (gate step 1)
      auto y0 = opencrab2::lowlevel::trunc::trunc_open_share(x0, mat0.value, 0);
      auto y1 = opencrab2::lowlevel::trunc::trunc_open_share(x1, mat1.value, 1);
      REQUIRE(y0.ok && y1.ok);
      std::vector<uint64_t> y(4);
      for (size_t i = 0; i < 4; ++i) y[i] = y0.value[i] + y1.value[i];
      auto z0 = opencrab2::lowlevel::trunc::trunc_faithful_eval(mat0.value, y, 0);
      auto z1 = opencrab2::lowlevel::trunc::trunc_faithful_eval(mat1.value, y, 1);
      REQUIRE(z0.ok && z1.ok);
      bool all = true;
      for (size_t i = 0; i < 4; ++i) all = all && (static_cast<int64_t>(z0.value[i] + z1.value[i]) == -13);
      CHECK(all);
    }
    CHECK(fx.svc->supports("ring_to_field") && fx.svc->supports("drelu"));
    CHECK(t0.value.header.get("params")->int_or("M", 0) == 4 && t0.value.header.get("params")->int_or("L", 0) == 40);
    // the toy bridge needs the full ring as logical width (L = 64); anything wider is refused
    wire::Value t64 = wire::Value::make_dict();
    t64.set("M", wire::Value::from_int(2));
    t64.set("f", wire::Value::from_int(48));
    t64.set("L", wire::Value::from_int(64));
    auto b64 = c0.value.request("trunc_faithful", t64, 1, "tr64");
    CHECK(b64.ok && c1.value.request("trunc_faithful", t64, 1, "tr64").ok);
    t64.set("L", wire::Value::from_int(65));
    CHECK(c0.value.request("trunc_faithful", t64, 1, "tr65").code == "MATERIAL_BAD_PARAMS");
    // DReLU material in both payload widths, with the header naming P
    for (bool with_product : {false, true}) {
      wire::Value dp = wire::Value::make_dict();
      dp.set("M", wire::Value::from_int(3));
      dp.set("n_bits", wire::Value::from_int(32));
      dp.set("with_product", wire::Value::from_bool(with_product));
      std::string gate = with_product ? "dr2" : "dr1";
      auto d0 = c0.value.request("drelu", dp, 1, gate), d1 = c1.value.request("drelu", dp, 1, gate);
      REQUIRE(d0.ok && d1.ok);
      CHECK(d0.value.header.get("params")->int_or("P", 0) == (with_product ? 2 : 1));
      auto dm0 = opencrab2::lowlevel::drelu::DreluMaterial::from_bytes(d0.value.payload);
      auto dm1 = opencrab2::lowlevel::drelu::DreluMaterial::from_bytes(d1.value.payload);
      REQUIRE(dm0.ok && dm1.ok);
      CHECK(dm0.value.M() == 3 && dm0.value.with_product() == with_product && dm1.value.party == 1);
      // the DCF keys inside the container are in the binding layout:
      // re-serialising the parsed material in that layout reproduces the payload byte for byte
      CHECK(dm0.value.to_bytes(opencrab2::lowlevel::dcf::WireLayout::Binding) == d0.value.payload);
      CHECK(dm0.value.to_bytes(opencrab2::lowlevel::dcf::WireLayout::PythonV2) != d0.value.payload);
    }
  }
#endif
  // authentication failures
  auto wrong_key = fx.connect(1, std::vector<uint8_t>(32, 0x99));
  CHECK(!wrong_key.ok && wrong_key.code == "AUTH_FAILED");
  auto wrong_setup = DealerConnection::connect("127.0.0.1", fx.port, 0, "other-setup", fx.contract, fx.key0, 5.0, 5);
  CHECK(!wrong_setup.ok && wrong_setup.code == "SETUP_MISMATCH");
  auto wrong_contract = DealerConnection::connect("127.0.0.1", fx.port, 0, fx.setup, "00", fx.key0, 5.0, 5);
  CHECK(!wrong_contract.ok && wrong_contract.code == "CONTRACT_MISMATCH");
  auto wrong_recipient = DealerConnection::connect("127.0.0.1", fx.port, 2, fx.setup, fx.contract, fx.key0, 5.0, 5);
  CHECK(!wrong_recipient.ok && wrong_recipient.code == "DEALER_UNKNOWN_RECIPIENT");
  DealerService::Stats s = fx.svc->stats();
  CHECK(s.auth_failures >= 2);
  CHECK(s.bundles_sent >= 10);
  CHECK(s.pending_halves == 0);
  c0.value.close();
  c1.value.close();
  fx.svc->stop();
  // Master randomness: two isolated dealers draw from the OS seeded
  // lowlevel CSPRNG and never produce the same material; two dealers with the same test seed do.
  {
    auto payload_of = [&](uint64_t seed, const std::string& mode) -> std::vector<uint8_t> {
      DealerFixture f;
      if (!f.start(seed, mode)) return {};
      auto c = f.connect(0);
      if (!c.ok) return {};
      wire::Value prm = wire::Value::make_dict();
      prm.set("shape", wire::Value::from_list({wire::Value::from_int(16)}));
      auto b = c.value.request("beaver", prm, 1, "rand");
      bool iso = c.value.isolated();
      bool insecure = b.ok && b.value.header.bool_or("insecure", true);
      c.value.close();
      f.svc->stop();
      if (!b.ok || iso != (mode == "ISOLATED_RECIPIENT_BUNDLES") || insecure == iso) return {};
      return b.value.payload;
    };
    auto i1 = payload_of(5, "ISOLATED_RECIPIENT_BUNDLES"), i2 = payload_of(5, "ISOLATED_RECIPIENT_BUNDLES");
    auto s1 = payload_of(5, "INSECURE_TEST_ONLY_SHARED_SEED"), s2 = payload_of(5, "INSECURE_TEST_ONLY_SHARED_SEED");
    auto s3 = payload_of(6, "INSECURE_TEST_ONLY_SHARED_SEED");
    CHECK(i1.size() == 16 * 3 * 8 && i2.size() == i1.size() && s1.size() == i1.size());
    CHECK(i1 != i2);          // OS seeded: the test seed is ignored in the isolated mode
    CHECK(s1 == s2);          // the shared seed reproduces
    CHECK(s1 != s3 && i1 != s1);
  }
  // The isolated dealer never accepts a plaintext vector (it would learn a model value): the
  // {name, values} form of vector_share is refused with a stable code and the connection stays
  // usable; the weight-blind {name, size} form (a zero sharing) is still served.
  {
    DealerFixture f;
    REQUIRE(f.start(5, "ISOLATED_RECIPIENT_BUNDLES"));
    auto c = f.connect(0);
    REQUIRE(c.ok);
    wire::Value pv = wire::Value::make_dict();
    pv.set("name", wire::Value::from_str("bias"));
    pv.set("values", wire::Value::from_u64_vector({1, 2, 3}));
    auto refused = c.value.request("vector_share", pv, 1, "vs-plain");
    CHECK(!refused.ok);
    CHECK_EQ_STR(refused.code, "DEALER_PLAINTEXT_REFUSED");
    wire::Value blind = wire::Value::make_dict();
    blind.set("name", wire::Value::from_str("bias"));
    blind.set("size", wire::Value::from_int(3));
    auto zero = c.value.request("vector_share", blind, 1, "vs-blind");
    CHECK(zero.ok && !zero.value.header.bool_or("insecure", true));
    wire::Value prm = wire::Value::make_dict();
    prm.set("shape", wire::Value::from_list({wire::Value::from_int(4)}));
    CHECK(c.value.request("beaver", prm, 1, "after-refusal").ok);
    c.value.close();
    f.svc->stop();
  }
  // keys file round trip
  std::string kp = temp_path("keys.json");
  CHECK(write_random_keys_file(kp).ok);
  auto keys = load_keys_file(kp);
  CHECK(keys.ok && keys.value.size() == 2 && keys.value[0] != keys.value[1]);
  // the recipient keys are readable by the owner only (POSIX mode 0600), also when the file existed with a wider mode
  bool owner_only = true;
#ifndef _WIN32
  {
    struct stat st;
    owner_only = ::stat(kp.c_str(), &st) == 0 && (st.st_mode & 0777) == 0600;
    owner_only = owner_only && ::chmod(kp.c_str(), 0644) == 0 && write_random_keys_file(kp).ok;
    owner_only = owner_only && ::stat(kp.c_str(), &st) == 0 && (st.st_mode & 0777) == 0600;
  }
#endif
  CHECK(owner_only);
  CHECK(detail::write_private_file(kp, "token\n"));
  {
    std::ifstream in(kp, std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    CHECK(text == "token\n");  // truncated and replaced, written without translation
  }
  CHECK(!detail::write_private_file(temp_path("no-such-dir") + "/keys.json", "x"));
  std::remove(kp.c_str());
  CHECK(!load_keys_file(kp).ok);
}

// ---------------------------------------------------------------------------
// WebSocket client helpers for the model server tests
struct WsClient {
  std::unique_ptr<WebSocketChannel> ch;
  bool connect(uint16_t port) {
    auto ws = websocket_connect("127.0.0.1", port, "/session", "", {}, 5.0);
    if (!ws.ok) return false;
    ws.value.set_timeout(5.0);
    ch.reset(new WebSocketChannel(std::move(ws.value), 1));
    return true;
  }
  Expected<wire::Value> hello(const std::string& sid, const std::string& version, const std::string& contract,
                              const std::vector<uint8_t>& nonce, const std::string& profile = "SH2P-TrustedOffline-v1",
                              const wire::Value& length_bucket = wire::Value::from_int(32), bool send_length_bucket = true) {
    wire::Value h = wire::Value::make_dict();
    h.set("type", wire::Value::from_str("HELLO"));
    h.set("session_id", wire::Value::from_str(sid));
    h.set("model_version", wire::Value::from_str(version));
    h.set("numeric_contract_hash", wire::Value::from_str(contract));
    h.set("profile_id", wire::Value::from_str(profile));
    h.set("nonce", wire::Value::from_bytes(nonce));
    if (send_length_bucket) h.set("length_bucket", length_bucket);
    auto s = ch->send(h);
    if (!s.ok) return forward_fail<wire::Value>(s);
    return ch->recv();
  }
  Expected<wire::Value> msg(const std::string& gate, int64_t seq, const wire::Value& payload) {
    wire::Value m = wire::Value::make_dict();
    m.set("type", wire::Value::from_str("MSG"));
    m.set("gate_id", wire::Value::from_str(gate));
    m.set("seq", wire::Value::from_int(seq));
    m.set("payload", payload);
    auto s = ch->send(m);
    if (!s.ok) return forward_fail<wire::Value>(s);
    return ch->recv();
  }
};

// Named requests (REQ with request_id, the form the Python client speaks):
// `count` bundles of ONE item each, delivered one per REQ with `remaining` counting down, the two
// recipients paired by the request id.
void test_dealer_named_requests() {
  DealerFixture fx;
  REQUIRE(fx.start());
  auto req = [](int recipient, const std::string& type, wire::Value params, int64_t count) {
    MaterialRequest r;
    r.material_type = type;
    r.params = std::move(params);
    r.count = count;
    r.gate_id = "*";
    r.recipient = recipient;
    return r;
  };
  wire::Value mt = wire::Value::make_dict();
  mt.set("shape_a", wire::Value::from_list({wire::Value::from_int(2), wire::Value::from_int(3)}));
  mt.set("shape_b", wire::Value::from_list({wire::Value::from_int(3), wire::Value::from_int(4)}));
  const size_t one_triple = 8 * (2 * 3 + 3 * 4 + 2 * 4);
  int64_t rem0 = -1, rem1 = -1;
  std::vector<Bundle> got0, got1;
  for (int i = 0; i < 3; ++i) {
    auto b = fx.svc->produce_named(req(0, "matrix_triple", mt, 3), "sess:mt:0", rem0);
    REQUIRE(b.ok);
    CHECK(rem0 == 2 - i);
    CHECK(b.value.payload.size() == one_triple && b.value.count() == 1);   // one triple per bundle, not three
    got0.push_back(b.value);
  }
  CHECK(fx.svc->stats().pending_halves == 3);                              // the peer's three halves wait
  // recipient 0 drained the request: a fourth REQ is refused while the peer still has halves pending
  CHECK_EQ_STR(fx.svc->produce_named(req(0, "matrix_triple", mt, 3), "sess:mt:0", rem0).code, "DEALER_ALREADY_DELIVERED");
  // the same id with other parameters is a conflict (count, params or type)
  CHECK_EQ_STR(fx.svc->produce_named(req(1, "matrix_triple", mt, 2), "sess:mt:0", rem1).code, "DEALER_REQUEST_CONFLICT");
  CHECK_EQ_STR(fx.svc->produce_named(req(1, "beaver", mt, 3), "sess:mt:0", rem1).code, "DEALER_REQUEST_CONFLICT");
  for (int i = 0; i < 3; ++i) {
    auto b = fx.svc->produce_named(req(1, "matrix_triple", mt, 3), "sess:mt:0", rem1);
    REQUIRE(b.ok);
    CHECK(rem1 == 2 - i);
    got1.push_back(b.value);
  }
  CHECK(fx.svc->stats().pending_halves == 0);                              // drained by both: forgotten
  for (int i = 0; i < 3; ++i) {
    CHECK_EQ_STR(got0[i].material_id(), got1[i].material_id());            // same pair, same order
    CHECK(got0[i].header.int_or("recipient", -1) == 0 && got1[i].header.int_or("recipient", -1) == 1);
    CHECK_EQ_STR(got0[i].tag, header_tag(fx.key0, got0[i].header, got0[i].payload));
    std::vector<uint64_t> s0 = le_to_u64(got0[i].payload.data(), got0[i].payload.size());
    std::vector<uint64_t> s1 = le_to_u64(got1[i].payload.data(), got1[i].payload.size());
    std::vector<uint64_t> A(6), B(12), C(8), want;
    for (size_t j = 0; j < 6; ++j) A[j] = s0[j] + s1[j];
    for (size_t j = 0; j < 12; ++j) B[j] = s0[6 + j] + s1[6 + j];
    for (size_t j = 0; j < 8; ++j) C[j] = s0[18 + j] + s1[18 + j];
    ring_matmul(A, 2, 3, B, 4, want);
    CHECK(C == want);                                                       // test-only cross check: C = A B
  }
  CHECK(got0[0].material_id() != got0[1].material_id() && got0[1].material_id() != got0[2].material_id());
  // Beaver-like material with a shape: count bundles of prod(shape) elements each
  wire::Value bp = wire::Value::make_dict();
  bp.set("shape", wire::Value::from_list({wire::Value::from_int(5)}));
  auto e0 = fx.svc->produce_named(req(0, "beaver", bp, 2), "sess:beaver:0", rem0);
  REQUIRE(e0.ok);
  CHECK(rem0 == 1 && e0.value.count() == 5 && e0.value.payload.size() == 3 * 5 * 8);
  // DCF-based material with M: count batches of M gates, one batch per bundle
  wire::Value dp = wire::Value::make_dict();
  dp.set("M", wire::Value::from_int(4));
  dp.set("n_bits", wire::Value::from_int(32));
  dp.set("with_product", wire::Value::from_bool(false));
  auto d0 = fx.svc->produce_named(req(0, "drelu", dp, 2), "sess:drelu1:0", rem0);
  if (DealerService::lowlevel_material_available()) {
    REQUIRE(d0.ok);
    CHECK(rem0 == 1 && d0.value.count() == 4);
    auto d1 = fx.svc->produce_named(req(1, "drelu", dp, 2), "sess:drelu1:0", rem1);
    REQUIRE(d1.ok);
    CHECK_EQ_STR(d0.value.material_id(), d1.value.material_id());
  } else {
    CHECK_EQ_STR(d0.code, "MATERIAL_TYPE_UNSUPPORTED");
  }
  // refusals: empty id, non-positive count, unknown type (nothing is stored for a refused request)
  const uint64_t before = fx.svc->stats().pending_halves;
  CHECK_EQ_STR(fx.svc->produce_named(req(0, "beaver", bp, 1), "", rem0).code, "MATERIAL_BAD_PARAMS");
  CHECK_EQ_STR(fx.svc->produce_named(req(0, "beaver", bp, 0), "x", rem0).code, "MATERIAL_BAD_PARAMS");
  CHECK_EQ_STR(fx.svc->produce_named(req(0, "no_such_material", bp, 1), "y", rem0).code, "MATERIAL_TYPE_UNSUPPORTED");
  CHECK_EQ_STR(fx.svc->produce_named(req(2, "beaver", bp, 1), "z", rem0).code, "DEALER_UNKNOWN_RECIPIENT");
  CHECK(fx.svc->stats().pending_halves == before);
  // over the wire: the unnamed form (C++ client) still answers one bundle holding all count items
  auto c0 = fx.connect(0);
  REQUIRE(c0.ok);
  auto all = c0.value.request("matrix_triple", mt, 3, "g-unnamed");
  REQUIRE(all.ok);
  CHECK(all.value.count() == 3 && all.value.payload.size() == 3 * one_triple);
}

void test_model_server() {
  const std::string contract_h = contract::hash();
  ModelServerConfig cfg;
  cfg.host = "127.0.0.1";
  cfg.port = 0;
  cfg.model_version = "gpt2-4L";
  cfg.numeric_contract_hash = contract_h;
  cfg.session_ttl_s = 30.0;
  cfg.max_queue_frames = 4;
  std::vector<std::string> logs;
  std::mutex log_mu;
  cfg.log = [&](const std::string& line) {
    std::lock_guard<std::mutex> lk(log_mu);
    logs.push_back(line);
  };
  ModelServer server(cfg, make_ack_handler());
  REQUIRE(server.start().ok);
  uint16_t port = server.port();
  std::vector<uint8_t> nonce = {1, 2, 3, 4, 5, 6, 7, 8};

  // healthz over plain HTTP
  HttpRequest hq;
  hq.method = "GET";
  hq.target = "/healthz";
  auto hz = http_request("127.0.0.1", port, hq, 5.0);
  CHECK(hz.ok && hz.value.status == 200);
  hq.target = "/session";
  auto no_upgrade = http_request("127.0.0.1", port, hq, 5.0);
  CHECK(no_upgrade.ok && no_upgrade.value.status == 426);

  auto describe = [](const Expected<wire::Value>& r) {
    return r.ok ? "ok:" + r.value.to_json().dump() : "fail:" + r.code;
  };
  // version mismatch
  {
    WsClient c;
    REQUIRE(c.connect(port));
    auto r = c.hello("s-bad", "gpt2-12L", contract_h, nonce);
    bool rejected = r.ok && r.value.str_or("type", "") == "REJECT" && r.value.str_or("code", "") == "VERSION_MISMATCH";
    if (!rejected) std::printf("  version mismatch reply: %s\n", describe(r).c_str());
    CHECK(rejected);
    auto cl = c.ch->recv();
    if (cl.ok) std::printf("  after REJECT expected close, got: %s\n", describe(cl).c_str());
    CHECK(!cl.ok && (cl.code == "WS_CLOSED" || cl.code == "TCP_CLOSED" || cl.code == "TCP_RECV_FAILED"));
  }
  // contract mismatch
  {
    WsClient c;
    REQUIRE(c.connect(port));
    auto r = c.hello("s-bad2", "gpt2-4L", "deadbeef", nonce);
    bool rejected = r.ok && r.value.str_or("code", "") == "CONTRACT_MISMATCH";
    if (!rejected) std::printf("  contract mismatch reply: %s\n", describe(r).c_str());
    CHECK(rejected);
  }
  // MSG before HELLO
  {
    WsClient c;
    REQUIRE(c.connect(port));
    auto r = c.msg("g", 0, wire::Value::from_int(1));
    CHECK(r.ok && r.value.str_or("code", "") == "NOT_BOUND");
  }
  // happy path with replay and seq reuse
  WsClient a;
  REQUIRE(a.connect(port));
  auto ok = a.hello("s-1", "gpt2-4L", contract_h, nonce);
  REQUIRE(ok.ok);
  CHECK_EQ_STR(ok.value.str_or("type", ""), "HELLO_OK");
  CHECK(ok.value.int_or("max_queue_frames", 0) == 4);
  wire::Value payload = wire::Value::make_dict();
  payload.set("share", wire::Value::from_u64_vector({11, 22, 33}));
  auto r1 = a.msg("attn.0", 0, payload);
  REQUIRE(r1.ok);
  CHECK_EQ_STR(r1.value.str_or("type", ""), "MSG");
  CHECK(r1.value.int_or("seq", -1) == 0);
  CHECK(r1.value.get("payload") && r1.value.get("payload")->bool_or("ack", false));
  std::vector<uint8_t> first_bytes = wire::frame(r1.value);
  auto r1b = a.msg("attn.0", 0, payload);  // duplicate seq, same input: byte-identical replay
  REQUIRE(r1b.ok);
  CHECK(wire::frame(r1b.value) == first_bytes);
  CHECK(server.stats().duplicate_seq_retransmits == 1);
  auto r2 = a.msg("attn.0", 1, payload);
  CHECK(r2.ok && r2.value.int_or("seq", -1) == 1);
  // duplicate session while attached
  {
    WsClient d;
    REQUIRE(d.connect(port));
    auto r = d.hello("s-1", "gpt2-4L", contract_h, nonce);
    CHECK(r.ok && r.value.str_or("code", "") == "DUPLICATE_SESSION");
  }
  // plaintext guard
  {
    WsClient p;
    REQUIRE(p.connect(port));
    REQUIRE(p.hello("s-plain", "gpt2-4L", contract_h, nonce).ok);
    wire::Value txt = wire::Value::make_dict();
    txt.set("prompt", wire::Value::from_str("the quick brown fox"));
    auto r = p.msg("g", 0, txt);
    CHECK(r.ok && r.value.str_or("code", "") == "PLAINTEXT_NOT_ACCEPTED");
    CHECK(server.stats().plaintext_rejections == 1);
  }
  // same seq, different input is a protocol violation
  wire::Value other = wire::Value::make_dict();
  other.set("share", wire::Value::from_u64_vector({99, 22, 33}));
  auto viol = a.msg("attn.0", 1, other);
  CHECK(viol.ok && viol.value.str_or("code", "") == "SEQ_REUSED_DIFFERENT_INPUT");
  CHECK(server.stats().seq_violations == 1);
  sleep_s(0.15);  // let the server detach the session
  // resume with the right nonce, then wrong nonce
  {
    WsClient r;
    REQUIRE(r.connect(port));
    wire::Value res = wire::Value::make_dict();
    res.set("type", wire::Value::from_str("RESUME"));
    res.set("session_id", wire::Value::from_str("s-1"));
    res.set("nonce", wire::Value::from_bytes(nonce));
    REQUIRE(r.ch->send(res).ok);
    auto rr = r.ch->recv();
    REQUIRE(rr.ok);
    CHECK_EQ_STR(rr.value.str_or("type", ""), "RESUME_OK");
    const wire::Value* last = rr.value.get("last_seq");
    CHECK(last && last->int_or("attn.0", -1) == 1);
    // the retransmission log survives the reconnect
    auto again = r.msg("attn.0", 0, payload);
    CHECK(again.ok && wire::frame(again.value) == first_bytes);
    wire::Value cancel = wire::Value::make_dict();
    cancel.set("type", wire::Value::from_str("CANCEL"));
    REQUIRE(r.ch->send(cancel).ok);
    auto co = r.ch->recv();
    CHECK(co.ok && co.value.str_or("type", "") == "CANCEL_OK");
    r.ch->close(1000);
  }
  sleep_s(0.15);
  {
    WsClient r;
    REQUIRE(r.connect(port));
    wire::Value res = wire::Value::make_dict();
    res.set("type", wire::Value::from_str("RESUME"));
    res.set("session_id", wire::Value::from_str("s-1"));
    res.set("nonce", wire::Value::from_bytes({9, 9, 9}));
    REQUIRE(r.ch->send(res).ok);
    auto rr = r.ch->recv();
    CHECK(rr.ok && rr.value.str_or("code", "") == "NONCE_MISMATCH");
    WsClient u;
    REQUIRE(u.connect(port));
    res.set("session_id", wire::Value::from_str("never-seen"));
    REQUIRE(u.ch->send(res).ok);
    auto ru = u.ch->recv();
    CHECK(ru.ok && ru.value.str_or("code", "") == "SESSION_UNKNOWN");
  }
  // logs carry no payload bytes
  {
    std::lock_guard<std::mutex> lk(log_mu);
    bool clean = true;
    for (const auto& l : logs) {
      if (l.find("quick brown") != std::string::npos) clean = false;
    }
    CHECK(clean && logs.size() > 5);
  }
  ModelServer::Stats s = server.stats();
  CHECK(s.sessions_opened == 2 && s.sessions_resumed == 1 && s.rejected_version == 1 && s.rejected_contract == 1);
  CHECK(s.rejected_duplicate_session == 1 && s.rejected_nonce == 1 && s.rejected_unknown_session == 1 && s.cancels == 1);
  CHECK(server.live_sessions() == 2);
  CHECK(s.rejected_profile == 0 && s.rejected_length_bucket == 0);
  server.stop();

  // PROFILE_MISMATCH at HELLO and the public length bucket (own server: clean counters)
  {
    ModelServerConfig c4 = cfg;
    c4.log = [](const std::string&) {};
    std::atomic<int64_t> seen_bucket{-1};
    ModelServer srv(c4, [&seen_bucket](Session& session, const std::string&, int64_t, const wire::Value&) {
      seen_bucket = session.binding.length_bucket;
      wire::Value r = wire::Value::make_dict();
      r.set("ack", wire::Value::from_bool(true));
      return make_ok(std::move(r));
    });
    REQUIRE(srv.start().ok);
    auto expect_reject = [&](const char* sid, const std::string& profile, const wire::Value& bucket, bool send_bucket,
                             const char* code) {
      WsClient c;
      if (!c.connect(srv.port())) return false;
      auto r = c.hello(sid, "gpt2-4L", contract_h, nonce, profile, bucket, send_bucket);
      bool rejected = r.ok && r.value.str_or("type", "") == "REJECT" && r.value.str_or("code", "") == code;
      if (!rejected) std::printf("  %s: expected REJECT %s, got %s\n", sid, code, describe(r).c_str());
      auto cl = c.ch->recv();  // the server closes after a HELLO reject
      return rejected && !cl.ok;
    };
    // another profile is refused before anything else is bound
    CHECK(expect_reject("p-1", "MAL2P-Dishonest-v9", wire::Value::from_int(32), true, "PROFILE_MISMATCH"));
    CHECK(expect_reject("p-2", "", wire::Value::from_int(32), true, "PROFILE_MISMATCH"));
    CHECK(srv.stats().rejected_profile == 2);
    CHECK(srv.stats().sessions_opened == 0 && srv.live_sessions() == 0);
    // version and contract are checked before the profile (first mismatch wins)
    {
      WsClient c;
      REQUIRE(c.connect(srv.port()));
      auto r = c.hello("p-3", "gpt2-12L", contract_h, nonce, "MAL2P-Dishonest-v9");
      CHECK(r.ok && r.value.str_or("code", "") == "VERSION_MISMATCH");
      CHECK(srv.stats().rejected_profile == 2 && srv.stats().rejected_version == 1);
    }
    // length_bucket: must be an integer in [1, 2^20]; missing, non-integer, zero, negative, too large are refused
    CHECK(expect_reject("lb-missing", "SH2P-TrustedOffline-v1", wire::Value::none(), false, "BAD_LENGTH_BUCKET"));
    CHECK(expect_reject("lb-str", "SH2P-TrustedOffline-v1", wire::Value::from_str("64"), true, "BAD_LENGTH_BUCKET"));
    CHECK(expect_reject("lb-float", "SH2P-TrustedOffline-v1", wire::Value::from_float(64.0), true, "BAD_LENGTH_BUCKET"));
    CHECK(expect_reject("lb-none", "SH2P-TrustedOffline-v1", wire::Value::none(), true, "BAD_LENGTH_BUCKET"));
    CHECK(expect_reject("lb-zero", "SH2P-TrustedOffline-v1", wire::Value::from_int(0), true, "BAD_LENGTH_BUCKET"));
    CHECK(expect_reject("lb-neg", "SH2P-TrustedOffline-v1", wire::Value::from_int(-64), true, "BAD_LENGTH_BUCKET"));
    CHECK(expect_reject("lb-huge", "SH2P-TrustedOffline-v1", wire::Value::from_int((1 << 20) + 1), true, "BAD_LENGTH_BUCKET"));
    CHECK(srv.stats().rejected_length_bucket == 7 && srv.stats().sessions_opened == 0);
    // a valid bucket is echoed in HELLO_OK, bound to the session and visible to the handler
    for (int64_t bucket : {int64_t(1), int64_t(256), int64_t(1) << 20}) {
      WsClient c;
      REQUIRE(c.connect(srv.port()));
      std::string sid = "lb-ok-" + std::to_string(bucket);
      auto r = c.hello(sid, "gpt2-4L", contract_h, nonce, "SH2P-TrustedOffline-v1", wire::Value::from_int(bucket));
      REQUIRE(r.ok);
      CHECK_EQ_STR(r.value.str_or("type", ""), "HELLO_OK");
      CHECK(r.value.int_or("length_bucket", -1) == bucket);
      auto m = c.msg("g", 0, wire::Value::from_int(1));
      CHECK(m.ok && m.value.str_or("type", "") == "MSG");
      CHECK(seen_bucket.load() == bucket);
      c.ch->close(1000);
    }
    CHECK(srv.stats().sessions_opened == 3 && srv.stats().rejected_length_bucket == 7);
    CHECK(srv.stats_json().int_or("rejected_profile", -1) == 2 && srv.stats_json().int_or("rejected_length_bucket", -1) == 7);
    srv.stop();
  }

  // TTL expiry with a short TTL server
  {
    ModelServerConfig c2 = cfg;
    c2.session_ttl_s = 0.4;
    c2.log = [](const std::string&) {};
    ModelServer short_srv(c2, make_ack_handler());
    REQUIRE(short_srv.start().ok);
    WsClient c;
    REQUIRE(c.connect(short_srv.port()));
    REQUIRE(c.hello("s-ttl", "gpt2-4L", contract_h, nonce).ok);
    c.ch->close(1000);
    sleep_s(0.7);
    WsClient r;
    REQUIRE(r.connect(short_srv.port()));
    wire::Value res = wire::Value::make_dict();
    res.set("type", wire::Value::from_str("RESUME"));
    res.set("session_id", wire::Value::from_str("s-ttl"));
    res.set("nonce", wire::Value::from_bytes(nonce));
    REQUIRE(r.ch->send(res).ok);
    auto rr = r.ch->recv();
    REQUIRE(rr.ok);
    std::string code = rr.value.str_or("code", "");
    CHECK(code == "SESSION_EXPIRED" || code == "SESSION_UNKNOWN");
    // an attached but idle connection is closed after the TTL
    WsClient idle;
    REQUIRE(idle.connect(short_srv.port()));
    REQUIRE(idle.hello("s-idle", "gpt2-4L", contract_h, nonce).ok);
    idle.ch->ws().set_timeout(3.0);
    auto closed = idle.ch->recv();
    CHECK(closed.ok ? closed.value.str_or("code", "") == "SESSION_EXPIRED" : closed.code == "WS_CLOSED");
    short_srv.stop();
  }

  // backpressure: slow handler, bounded queue of 6 (pause at 4, resume at 1)
  {
    ModelServerConfig c3 = cfg;
    c3.max_queue_frames = 6;
    c3.log = [](const std::string&) {};
    ModelServer slow(c3, [](Session&, const std::string&, int64_t, const wire::Value&) {
      sleep_s(0.03);
      wire::Value r = wire::Value::make_dict();
      r.set("ack", wire::Value::from_bool(true));
      return make_ok(std::move(r));
    });
    REQUIRE(slow.start().ok);
    WsClient c;
    REQUIRE(c.connect(slow.port()));
    REQUIRE(c.hello("s-flow", "gpt2-4L", contract_h, nonce).ok);
    const int burst = 6;
    for (int i = 0; i < burst; ++i) {
      wire::Value m = wire::Value::make_dict();
      m.set("type", wire::Value::from_str("MSG"));
      m.set("gate_id", wire::Value::from_str("g"));
      m.set("seq", wire::Value::from_int(i));
      m.set("payload", wire::Value::from_int(i));
      REQUIRE(c.ch->send(m).ok);
    }
    int msgs = 0, pauses = 0, resumes = 0;
    while (msgs < burst) {
      auto f = c.ch->recv();
      REQUIRE(f.ok);
      std::string type = f.value.str_or("type", "");
      if (type == "MSG") ++msgs;
      if (type == "FLOW") (f.value.bool_or("pause", false) ? ++pauses : ++resumes);
    }
    CHECK(pauses >= 1);
    CHECK(resumes >= 1);
    CHECK(slow.stats().flow_pause_frames >= 1);
    // ignoring FLOW long enough is a violation: the server closes
    for (int i = 0; i < 40; ++i) {
      wire::Value m = wire::Value::make_dict();
      m.set("type", wire::Value::from_str("MSG"));
      m.set("gate_id", wire::Value::from_str("h"));
      m.set("seq", wire::Value::from_int(i));
      m.set("payload", wire::Value::from_int(i));
      if (!c.ch->send(m).ok) break;
    }
    bool violation_seen = false;
    for (int i = 0; i < 200 && !violation_seen; ++i) {
      auto f = c.ch->recv();
      if (!f.ok) {
        violation_seen = (slow.stats().flow_violations >= 1);
        break;
      }
      if (f.value.str_or("type", "") == "REJECT" && f.value.str_or("code", "") == "FLOW_VIOLATION") violation_seen = true;
    }
    CHECK(violation_seen);
    slow.stop();
  }
}

// ---------------------------------------------------------------------------
struct SseEvent {
  std::string name;
  std::string data;
};

// Reads a complete SSE response (until the server closes) and splits events.
std::vector<SseEvent> read_sse(TcpSocket& sock, double timeout_s) {
  sock.set_timeout(timeout_s);
  std::string buf;
  uint8_t tmp[4096];
  while (true) {
    auto r = sock.recv_some(tmp, sizeof tmp);
    if (!r.ok || r.value == 0) break;
    buf.append(reinterpret_cast<const char*>(tmp), r.value);
  }
  std::vector<SseEvent> out;
  size_t head = buf.find("\r\n\r\n");
  if (head == std::string::npos) return out;
  std::string body = buf.substr(head + 4);
  size_t pos = 0;
  while (pos < body.size()) {
    size_t end = body.find("\n\n", pos);
    if (end == std::string::npos) break;
    std::string block = body.substr(pos, end - pos);
    pos = end + 2;
    SseEvent ev;
    for (const auto& line : detail::split(block, '\n')) {
      if (line.compare(0, 7, "event: ") == 0) ev.name = line.substr(7);
      if (line.compare(0, 6, "data: ") == 0) ev.data += line.substr(6);
    }
    if (!ev.name.empty() || !ev.data.empty()) out.push_back(ev);
  }
  return out;
}

HttpRequest api_request(const std::string& method, const std::string& target, const std::string& token,
                        const std::string& origin, const std::string& body = "") {
  HttpRequest q;
  q.method = method;
  q.target = target;
  if (!token.empty()) q.headers.emplace_back("Authorization", "Bearer " + token);
  if (!origin.empty()) q.headers.emplace_back("Origin", origin);
  if (!body.empty()) q.headers.emplace_back("Content-Type", "application/json");
  q.body = body;
  return q;
}

void test_sidecar() {
  SidecarConfig cfg;
  cfg.host = "127.0.0.1";
  cfg.port = 0;
  cfg.sse_heartbeat_s = 0.2;
  cfg.log = [](const std::string&) {};
  Sidecar sc(cfg, make_stub_engine(0.02));
  REQUIRE(sc.start().ok);
  uint16_t port = sc.port();
  std::string good_origin = "http://127.0.0.1:" + std::to_string(port);
  std::string token = sc.token();
  CHECK(token.size() == 64);
  CHECK(sc.launch_url().find("#token=" + token) != std::string::npos);

  auto r = http_request("127.0.0.1", port, api_request("GET", "/v1/status", "", ""), 5.0);
  CHECK(r.ok && r.value.status == 401);
  CHECK(r.ok && r.value.header("WWW-Authenticate") != nullptr);
  r = http_request("127.0.0.1", port, api_request("GET", "/v1/status", token, "http://evil.example"), 5.0);
  CHECK(r.ok && r.value.status == 403);
  r = http_request("127.0.0.1", port, api_request("GET", "/v1/status", "", "http://evil.example"), 5.0);
  CHECK(r.ok && r.value.status == 403);  // origin is checked before the token
  r = http_request("127.0.0.1", port, api_request("GET", "/v1/status", "wrong" + token.substr(5), good_origin), 5.0);
  CHECK(r.ok && r.value.status == 401);
  r = http_request("127.0.0.1", port, api_request("GET", "/v1/status", token, good_origin), 5.0);
  REQUIRE(r.ok && r.value.status == 200);
  auto st = json::parse(r.value.body);
  CHECK(st.ok && st.value.bool_or("token_required", false) && st.value.int_or("active_jobs", -1) == 0);
  CHECK(r.value.header("Access-Control-Allow-Origin") && *r.value.header("Access-Control-Allow-Origin") == good_origin);
  r = http_request("127.0.0.1", port, api_request("GET", "/v1/status", token, ""), 5.0);
  CHECK(r.ok && r.value.status == 200);  // non-browser client without Origin, token suffices
  r = http_request("127.0.0.1", port, api_request("OPTIONS", "/v1/generate", "", good_origin), 5.0);
  CHECK(r.ok && r.value.status == 204 && r.value.header("Access-Control-Allow-Headers") != nullptr);
  r = http_request("127.0.0.1", port, api_request("GET", "/v1/nothing", token, good_origin), 5.0);
  CHECK(r.ok && r.value.status == 404);
  r = http_request("127.0.0.1", port, api_request("POST", "/v1/generate", token, good_origin, "{not json"), 5.0);
  CHECK(r.ok && r.value.status == 400);

  // generate + stream
  r = http_request("127.0.0.1", port,
                   api_request("POST", "/v1/generate", token, good_origin, R"({"prompt":"secret prompt","n_tokens":5})"), 5.0);
  REQUIRE(r.ok && r.value.status == 202);
  auto gen = json::parse(r.value.body);
  REQUIRE(gen.ok);
  std::string job = gen.value.str_or("job_id", "");
  CHECK(!job.empty() && !gen.value.str_or("session_id", "").empty());
  {
    auto sock = TcpSocket::connect("127.0.0.1", port, 5.0);
    REQUIRE(sock.ok);
    sock.value.send_all(serialize_request(api_request("GET", "/v1/stream?job_id=" + job, token, good_origin),
                                          "127.0.0.1:" + std::to_string(port)));
    auto events = read_sse(sock.value, 5.0);
    int tokens = 0;
    bool done = false;
    for (const auto& e : events) {
      if (e.name == "token") {
        ++tokens;
        auto j = json::parse(e.data);
        CHECK(j.ok && j.value.str_or("engine", "") == "stub");
      }
      if (e.name == "done") done = true;
    }
    CHECK(tokens == 5);
    CHECK(done);
  }
  auto js = sc.job_status(job);
  CHECK(js.ok && js.value.state == "done" && js.value.tokens_emitted == 5 && js.value.ttft_s >= 0);
  // the stream of a finished job replays its tokens
  {
    auto sock = TcpSocket::connect("127.0.0.1", port, 5.0);
    REQUIRE(sock.ok);
    sock.value.send_all(serialize_request(api_request("GET", "/v1/stream?job_id=" + job, token, good_origin),
                                          "127.0.0.1:" + std::to_string(port)));
    auto events = read_sse(sock.value, 5.0);
    CHECK(events.size() == 6);
  }
  r = http_request("127.0.0.1", port, api_request("GET", "/v1/stream?job_id=nope", token, good_origin), 5.0);
  CHECK(r.ok && r.value.status == 404);
  // the stream endpoint is protected too
  r = http_request("127.0.0.1", port, api_request("GET", "/v1/stream?job_id=" + job, "", good_origin), 5.0);
  CHECK(r.ok && r.value.status == 401);

  // cancel a long job
  r = http_request("127.0.0.1", port,
                   api_request("POST", "/v1/generate", token, good_origin, R"({"prompt":"p","n_tokens":200})"), 5.0);
  REQUIRE(r.ok && r.value.status == 202);
  std::string job2 = json::parse(r.value.body).value.str_or("job_id", "");
  sleep_s(0.1);
  r = http_request("127.0.0.1", port, api_request("POST", "/v1/cancel", token, good_origin, "{\"job_id\":\"" + job2 + "\"}"), 5.0);
  CHECK(r.ok && r.value.status == 200);
  auto w = sc.wait_job(job2, 5.0);
  CHECK(w.ok && w.value.state == "cancelled" && w.value.tokens_emitted < 200 && w.value.tokens_emitted > 0);
  r = http_request("127.0.0.1", port, api_request("POST", "/v1/cancel", token, good_origin, "{\"job_id\":\"zzz\"}"), 5.0);
  CHECK(r.ok && r.value.status == 404);

  // static page (built-in) and traversal guard
  r = http_request("127.0.0.1", port, api_request("GET", "/", "", ""), 5.0);
  CHECK(r.ok && r.value.status == 200 && r.value.body.find("<!doctype html>") == 0);
  CHECK(r.ok && r.value.header("Content-Security-Policy") != nullptr);
  r = http_request("127.0.0.1", port, api_request("GET", "/../keys.json", "", ""), 5.0);
  CHECK(r.ok && (r.value.status == 403 || r.value.status == 404));
  r = http_request("127.0.0.1", port, api_request("GET", "/app.js", "", ""), 5.0);
  CHECK(r.ok && r.value.status == 404);
  // web metrics in GET /v1/status: TTFT, inter-token latency, completion,
  // cancel waste and the rejection counters (job 1 completed with 5 tokens, job 2 was cancelled)
  r = http_request("127.0.0.1", port, api_request("GET", "/v1/status", token, good_origin), 5.0);
  REQUIRE(r.ok && r.value.status == 200);
  {
    auto sj = json::parse(r.value.body);
    REQUIRE(sj.ok);
    const json::Value* m = sj.value.find("metrics");
    REQUIRE(m && m->is_object());
    const json::Value* ttft = m->find("ttft_s");
    const json::Value* itl = m->find("inter_token_s");
    const json::Value* comp = m->find("completion");
    const json::Value* comp_s = m->find("completion_s");
    const json::Value* waste = m->find("cancel_waste");
    const json::Value* rej = m->find("rejections");
    REQUIRE(ttft && itl && comp && comp_s && waste && rej);
    CHECK(ttft->int_or("n", 0) == 2 && ttft->double_or("mean", -1) > 0.0 && ttft->double_or("max", -1) >= ttft->double_or("p50", 1e9));
    CHECK(ttft->double_or("cold_s", -1) > 0.0);                 // the first job of this launch
    CHECK(ttft->int_or("warm_n", 0) == 1 && ttft->double_or("warm_mean_s", -1) > 0.0);
    CHECK(itl->int_or("n", 0) >= 4);                            // 4 gaps of job 1 plus those of job 2
    CHECK(itl->double_or("mean", -1) > 0.005 && itl->double_or("mean", -1) < 1.0);  // the stub emits every 20 ms
    CHECK(itl->double_or("max", -1) >= itl->double_or("p50", 1e9));
    CHECK(comp->int_or("finished", 0) == 2 && comp->int_or("completed", 0) == 1 && comp->int_or("cancelled", 0) == 1);
    CHECK(comp->int_or("error", -1) == 0 && comp->double_or("rate", -1) == 0.5);
    CHECK(comp_s->int_or("n", 0) == 1 && comp_s->double_or("mean", -1) > 0.05);
    CHECK(waste->int_or("jobs", 0) == 1 && waste->double_or("latency_s_max", -1) >= 0.0 && waste->double_or("wall_s", -1) > 0.0);
    CHECK(waste->int_or("tokens_after_cancel", -1) >= 0 && waste->int_or("tokens_after_cancel", -1) <= 2);
    CHECK(rej->int_or("origin", -1) == 2 && rej->int_or("auth", -1) == 3 && rej->int_or("version_mismatch", -1) == 0);
    // per job: the same quantities
    const json::Value* jobs = sj.value.find("jobs");
    REQUIRE(jobs && jobs->is_array() && jobs->arr.size() == 2);
    for (const auto& jb : jobs->arr) {
      const json::Value* jit = jb.find("inter_token_s");
      const json::Value* jc = jb.find("cancel");
      REQUIRE(jit && jc);
      if (jb.str_or("job_id", "") == job) {
        CHECK(jb.bool_or("completed", false) && jb.bool_or("cold", false) && jb.double_or("completion_s", -1) > 0.05);
        CHECK(jit->int_or("n", 0) == 4 && jit->double_or("p50", -1) > 0.005 && !jc->bool_or("requested", true));
      } else {
        CHECK(!jb.bool_or("completed", true) && !jb.bool_or("cold", true));
        CHECK(jc->bool_or("requested", false) && jc->double_or("latency_s", -1) >= 0.0);
      }
    }
    // the status never echoes a prompt
    CHECK(r.value.body.find("secret prompt") == std::string::npos);
  }
  Sidecar::Stats s = sc.stats();
  CHECK(s.rejected_origin == 2 && s.rejected_auth == 3 && s.jobs_started == 2 && s.jobs_done == 1 && s.jobs_cancelled == 1);
  sc.stop();
  // engine-reported counters and the server-side rejection codes
  {
    SidecarConfig c2 = cfg;
    std::atomic<bool> flag_seen{false};
    Sidecar se(c2, [&flag_seen](JobContext& ctx) -> Expected<bool> {
      ctx.set_engine_name("probe");
      if (ctx.prompt() == "reject-version") return make_fail<bool>("VERSION_MISMATCH");
      if (ctx.prompt() == "reject-profile") return make_fail<bool>("PROFILE_MISMATCH");
      if (ctx.prompt() == "reject-stale") return make_fail<bool>("SESSION_EXPIRED");
      if (ctx.prompt() == "reject-contract") return make_fail<bool>("CONTRACT_MISMATCH: detail");
      // a job that is cancelled after two tokens and reports its material counters
      ctx.emit_token(1, "a");
      ctx.emit_token(2, "b");
      while (!ctx.cancelled()) sleep_s(0.005);
      flag_seen = ctx.cancel_flag() != nullptr && ctx.cancel_flag()->load();
      ctx.report_metric("material_transferred_bytes", 10000.0);
      ctx.report_metric("material_consumed_bytes", 6000.0);
      ctx.report_metric("material_burned_bytes", 4096.0);
      ctx.report_metric("not_finite", std::nan(""));  // dropped
      return make_ok(true);
    });
    REQUIRE(se.start().ok);
    std::string origin2 = "http://127.0.0.1:" + std::to_string(se.port());
    auto submit = [&](const std::string& prompt) {
      auto g = http_request("127.0.0.1", se.port(),
                            api_request("POST", "/v1/generate", se.token(), origin2, "{\"prompt\":\"" + prompt + "\",\"n_tokens\":4}"), 5.0);
      return g.ok && g.value.status == 202 ? json::parse(g.value.body).value.str_or("job_id", "") : std::string();
    };
    for (const char* pr : {"reject-version", "reject-profile", "reject-stale", "reject-contract"}) {
      std::string id = submit(pr);
      REQUIRE(!id.empty());
      auto w2 = se.wait_job(id, 5.0);
      CHECK(w2.ok && w2.value.state == "error");
    }
    std::string cj = submit("cancel-me");
    REQUIRE(!cj.empty());
    for (int i = 0; i < 200; ++i) {
      auto st2 = se.job_status(cj);
      if (st2.ok && st2.value.tokens_emitted == 2) break;
      sleep_s(0.005);
    }
    CHECK(se.cancel_job(cj).ok);
    auto wc = se.wait_job(cj, 5.0);
    CHECK(wc.ok && wc.value.state == "cancelled" && wc.value.cancel_requested_wall_s > 0 && wc.value.cancel_latency_s() >= 0.0);
    CHECK(wc.ok && wc.value.metrics.double_or("material_burned_bytes", 0) == 4096.0 && wc.value.metrics.find("not_finite") == nullptr);
    CHECK(flag_seen.load());  // the engine saw the cancel request through the flag as well
    Sidecar::Stats s2 = se.stats();
    CHECK(s2.rejected_version == 1 && s2.rejected_profile == 1 && s2.rejected_stale == 1 && s2.rejected_contract == 1);
    json::Value m2 = se.metrics_json();
    CHECK(m2.find("rejections")->int_or("version_mismatch", 0) == 1 && m2.find("rejections")->int_or("profile_mismatch", 0) == 1);
    CHECK(m2.find("rejections")->int_or("stale_session", 0) == 1 && m2.find("rejections")->int_or("contract_mismatch", 0) == 1);
    CHECK(m2.find("cancel_waste")->int_or("jobs", 0) == 1 && m2.find("cancel_waste")->int_or("material_burned_bytes", 0) == 4096);
    CHECK(m2.find("cancel_waste")->int_or("material_unused_bytes", 0) == 4000);
    CHECK(m2.find("completion")->int_or("error", 0) == 4 && m2.find("completion")->double_or("rate", -1) == 0.0);
    se.stop();
  }
  // the sidecar refuses to bind anything but loopback
  SidecarConfig bad = cfg;
  bad.host = "0.0.0.0";
  Sidecar sb(bad, make_stub_engine());
  CHECK(sb.start().code == "SIDECAR_MUST_BIND_LOOPBACK");
}

// ---------------------------------------------------------------------------
// dealer + model server (DemoBeaverP0) + sidecar (demo engine): the released
// products equal x * w for the prompt-derived x and the server's w.
void test_demo_end_to_end() {
  DealerFixture fx;
  REQUIRE(fx.start(77));
  const size_t n = 8;
  std::vector<uint64_t> w(n);
  for (size_t i = 0; i < n; ++i) w[i] = 1000003ULL * (i + 1) + 0x9e3779b97f4a7c15ULL;
  auto dconn = fx.connect(0);
  REQUIRE(dconn.ok);
  auto dealer0 = std::make_shared<DealerConnection>(std::move(dconn.value));
  DemoBeaverP0 p0(dealer0, w);
  ModelServerConfig mcfg;
  mcfg.port = 0;
  mcfg.model_version = "gpt2-4L";
  mcfg.numeric_contract_hash = fx.contract;
  mcfg.log = [](const std::string&) {};
  ModelServer server(mcfg, p0.as_handler());
  REQUIRE(server.start().ok);
  SidecarConfig scfg;
  scfg.port = 0;
  scfg.server_ws_url = "ws://127.0.0.1:" + std::to_string(server.port()) + "/session";
  scfg.dealer_host = "127.0.0.1";
  scfg.dealer_port = fx.port;
  scfg.dealer_key = fx.key1;
  scfg.setup_id = fx.setup;
  scfg.numeric_contract_hash = fx.contract;
  scfg.model_version = "gpt2-4L";
  scfg.demo_vector_len = n;
  scfg.log = [](const std::string&) {};
  Sidecar sc(scfg, make_demo_beaver_engine());
  REQUIRE(sc.start().ok);
  std::string origin = "http://127.0.0.1:" + std::to_string(sc.port());
  const std::string prompt = "abc";
  auto r = http_request("127.0.0.1", sc.port(),
                        api_request("POST", "/v1/generate", sc.token(), origin,
                                    "{\"prompt\":\"" + prompt + "\",\"n_tokens\":3,\"session_id\":\"e2e-1\"}"),
                        5.0);
  REQUIRE(r.ok && r.value.status == 202);
  std::string job = json::parse(r.value.body).value.str_or("job_id", "");
  auto done = sc.wait_job(job, 20.0);
  REQUIRE(done.ok);
  CHECK_EQ_STR(done.value.state, "done");
  CHECK_EQ_STR(done.value.error_code, "");
  CHECK_EQ_STR(done.value.engine, "demo_beaver");
  CHECK_EQ_STR(done.value.binding.str_or("session_id", ""), "e2e-1");
  auto toks = sc.job_tokens(job);
  REQUIRE(toks.ok && toks.value.size() == 3);
  std::vector<uint64_t> x(n);
  for (size_t i = 0; i < n; ++i) x[i] = static_cast<uint8_t>(prompt[i % prompt.size()]);
  for (size_t k = 0; k < 3; ++k) {
    uint64_t z = x[k % n] * w[k % n];
    CHECK(toks.value[k].token_id == static_cast<int64_t>(z % 50257u));
  }
  ModelServer::Stats ms = server.stats();
  // HELLO + 3 steps x (share_input, beaver_open)
  CHECK(ms.sessions_opened == 1 && ms.frames_in == 7 && ms.plaintext_rejections == 0);
  DealerService::Stats ds = fx.svc->stats();
  CHECK(ds.pairs_generated == 3 && ds.bundles_sent == 6 && ds.pending_halves == 0);
  sc.stop();
  server.stop();
  dealer0->close();
  fx.svc->stop();
}

// ---------------------------------------------------------------------------
// evidence schema (python/opencrab2/experiments/schema.py) as the apps write it
void test_evidence() {
  CHECK(evidence::record_fields().size() == 35 && evidence::record_fields().front() == "schema");
  CHECK(evidence::optional_fields().size() == 11);
  hl::Value r = evidence::new_record("cpp-tree-test");
  CHECK(evidence::validate(r).ok);
  CHECK(r.get("extra") && r.get("extra")->get("language") && r.get("extra")->get("language")->as_string().value == "cpp");
  CHECK(!r.get("security_eligible")->as_bool().value);
  auto code_of = [](hl::Value rec) { return evidence::validate(rec).code_prefix(); };
  auto with = [&](const char* key, hl::Value v) {
    hl::Value c = r;
    c.set(key, std::move(v));
    return c;
  };
  // every rejection of schema.validate, with the Python codes
  {
    hl::Dict d = *std::get_if<hl::Dict>(&r.v);
    hl::Dict without;
    for (const auto& e : d) {
      if (e.key != "status") without.push_back(e);
    }
    CHECK_EQ_STR(code_of(hl::Value(without)), "SCHEMA_MISSING_FIELD");
  }
  CHECK_EQ_STR(code_of(with("party", hl::Value(static_cast<int64_t>(1)))), "SCHEMA_UNKNOWN_FIELD");
  CHECK_EQ_STR(code_of(with("schema", hl::Value("other"))), "SCHEMA_BAD_VALUE");
  CHECK_EQ_STR(code_of(with("status", hl::Value("MEASURED"))), "SCHEMA_BAD_STATUS");
  CHECK_EQ_STR(code_of(with("workload_scope", hl::Value("toy"))), "SCHEMA_BAD_SCOPE");
  CHECK_EQ_STR(code_of(with("dealer_mode", hl::Value("SHARED"))), "SCHEMA_BAD_VALUE");
  CHECK_EQ_STR(code_of(with("clock_basis", hl::Value("steady-clock"))), "SCHEMA_BAD_VALUE");
  CHECK_EQ_STR(code_of(with("run_id", hl::Value(""))), "SCHEMA_BAD_VALUE");
  CHECK_EQ_STR(code_of(with("policy_id", hl::Value(static_cast<int64_t>(4)))), "SCHEMA_BAD_TYPE");
  CHECK_EQ_STR(code_of(with("security_eligible", hl::Value("yes"))), "SCHEMA_BAD_TYPE");
  CHECK_EQ_STR(code_of(with("bytes", hl::Value(static_cast<int64_t>(-1)))), "SCHEMA_BAD_TYPE");
  CHECK_EQ_STR(code_of(with("rounds", hl::Value(1.5))), "SCHEMA_BAD_TYPE");
  CHECK_EQ_STR(code_of(with("tokens", hl::Value(hl::List{hl::Value(static_cast<int64_t>(3))}))), "SCHEMA_BAD_TYPE");  // a count, not a list
  CHECK_EQ_STR(code_of(with("T_online_s", hl::Value(-0.5))), "SCHEMA_BAD_TYPE");
  CHECK_EQ_STR(code_of(with("T_online_s", hl::Value(std::nan("")))), "SCHEMA_BAD_TYPE");
  CHECK_EQ_STR(code_of(with("peak_rss_client", hl::Value(1.0))), "SCHEMA_BAD_TYPE");
  CHECK_EQ_STR(code_of(with("output_match", hl::Value(static_cast<int64_t>(1)))), "SCHEMA_BAD_TYPE");
  CHECK_EQ_STR(code_of(with("record_kind", hl::Value("bench"))), "SCHEMA_BAD_VALUE");
  CHECK_EQ_STR(code_of(with("extra", hl::Value("x"))), "SCHEMA_BAD_TYPE");
  CHECK_EQ_STR(code_of(with("extra", hl::dict({{"x", std::nan("")}}))), "SCHEMA_BAD_VALUE");
  CHECK_EQ_STR(code_of(with("extra", hl::dict({{"x", hl::Value(hl::Bytes{1, 2})}}))), "SCHEMA_BAD_VALUE");
  CHECK(evidence::validate(with("T_online_s", hl::Value(0.25))).ok && evidence::validate(with("T_cold_s", hl::Value(static_cast<int64_t>(2)))).ok);
  // eligibility needs the isolated dealer, no diagnostic reveal, a native measurement and a completed run
  hl::Value e = r;
  e.set("security_eligible", true);
  CHECK_EQ_STR(code_of(e), "SCHEMA_ELIGIBILITY");  // dealer_mode NONE
  e.set("dealer_mode", "ISOLATED_RECIPIENT_BUNDLES");
  CHECK_EQ_STR(code_of(e), "SCHEMA_ELIGIBILITY");  // status MODELED
  e.set("status", "MEASURED_NATIVE");
  CHECK(evidence::validate(e).ok);
  CHECK_EQ_STR(code_of([&]() { hl::Value c = e; c.set("diagnostic_reveals", static_cast<int64_t>(1)); return c; }()), "SCHEMA_ELIGIBILITY");
  CHECK_EQ_STR(code_of([&]() { hl::Value c = e; c.set("aborts", static_cast<int64_t>(1)); return c; }()), "SCHEMA_ELIGIBILITY");
  CHECK_EQ_STR(code_of([&]() { hl::Value c = e; c.set("dealer_mode", "INSECURE_TEST_ONLY_SHARED_SEED"); return c; }()), "SCHEMA_ELIGIBILITY");
  // enforce: a shared-seed record is DIAGNOSTIC and not eligible, whatever the caller claimed
  {
    hl::Value c = e;
    c.set("dealer_mode", "INSECURE_TEST_ONLY_SHARED_SEED");
    auto reasons = evidence::enforce(c);
    CHECK(reasons.size() == 1 && c.get("status")->as_string().value == "DIAGNOSTIC" && !c.get("security_eligible")->as_bool().value);
    CHECK(evidence::validate(c).ok && c.get("extra")->get("eligibility_notes")->is_list());
    hl::Value ok_rec = e;
    CHECK(evidence::enforce(ok_rec).empty() && ok_rec.get("security_eligible")->as_bool().value);
    hl::Value aborted = e;
    aborted.set("aborts", static_cast<int64_t>(2));
    CHECK(evidence::enforce(aborted).size() == 1 && !aborted.get("security_eligible")->as_bool().value && evidence::validate(aborted).ok);
    hl::Value w = e;
    evidence::withdraw_eligibility(w, "TEST_REASON: detail", true);
    CHECK(!w.get("security_eligible")->as_bool().value && w.get("status")->as_string().value == "DIAGNOSTIC" && evidence::validate(w).ok);
  }
  // conform: a record of the highlevel session loop (flat analysis fields, token list) into the schema shape
  hl::Value session = hl::dict({{"status", "MEASURED_NATIVE"},
                                {"security_eligible", true},
                                {"code_commit", "cpp-mirror-untracked"},
                                {"profile_id", "SH2P-TrustedOffline-v1"},
                                {"numeric_contract_hash", contract::hash()},
                                {"dealer_mode", "ISOLATED_RECIPIENT_BUNDLES"},
                                {"diagnostic_reveals", static_cast<int64_t>(0)},
                                {"workload_scope", "full-model"},
                                {"model_revision", "tiny"},
                                {"trace_id", "S0"},
                                {"trace_hash", "abc"},
                                {"clock_basis", "wall-clock-exogenous"},
                                {"policy_id", "B4_opencrab"},
                                {"initial_state_hash", "h"},
                                {"run_id", "run-1"},
                                {"repeat", static_cast<int64_t>(0)},
                                {"initializations", static_cast<int64_t>(1)},
                                {"post_init_switches", static_cast<int64_t>(0)},
                                {"rejected_candidates", static_cast<int64_t>(0)},
                                {"aborts", static_cast<int64_t>(0)},
                                {"T_cold_s", 1.5},
                                {"T_warm_s", 1.25},
                                {"T_online_s", 1.0},
                                {"bytes", static_cast<int64_t>(1000)},
                                {"rounds", static_cast<int64_t>(10)},
                                {"material_generated", static_cast<int64_t>(5)},
                                {"material_transferred", static_cast<int64_t>(5)},
                                {"material_consumed", static_cast<int64_t>(4)},
                                {"material_burned", static_cast<int64_t>(0)},
                                {"peak_rss_client", static_cast<int64_t>(123)},
                                {"peak_rss_server", hl::Value(hl::None{})},
                                {"tokens", hl::Value(hl::List{hl::Value(static_cast<int64_t>(7)), hl::Value(static_cast<int64_t>(9))})},
                                {"output_match", true},
                                {"raw_measurement_path", "x.jsonl"},
                                {"party", static_cast<int64_t>(1)},
                                {"completed", true},
                                {"failure", ""},
                                {"seed", static_cast<int64_t>(424242)},
                                {"shaping_delay_s", std::nan("")},
                                {"epochs", hl::Value(hl::List{hl::dict({{"epoch", static_cast<int64_t>(0)}})})}});
  CHECK_EQ_STR(evidence::validate(session).code_prefix(), "SCHEMA_MISSING_FIELD");  // no schema tag, extra fields flat
  hl::Value conf = evidence::conform(session, "session", "cpp-tree-test");
  CHECK(evidence::validate(conf).ok);
  CHECK(conf.get("tokens")->as_int().value == 2 && conf.get("extra")->get("tokens_list")->is_list());
  CHECK(conf.get("extra")->get("party")->as_int().value == 1 && conf.get("extra")->get("epochs")->is_list());
  CHECK(conf.get("extra")->get("shaping_delay_s")->is_none());  // non-finite becomes null
  CHECK(conf.get("record_kind")->as_string().value == "session" && conf.get("schema")->as_string().value == std::string(evidence::kSchemaId));
  CHECK(conf.get("code_commit")->as_string().value == "cpp-mirror-untracked" && conf.get("security_eligible")->as_bool().value);
  // finalize: the app context decides what the record may claim; the private seed never appears
  {
    SessionRecordContext rc;
    rc.code_commit = "cpp-tree-test";
    rc.implementation = "test_platform";
    rc.trace_id = "S3:two-changes:seed4:d12";
    rc.trace_hash = "b5f3798a1106f21374cb36270f5023698d1fd88143c41c088e9bb75aa2454204";
    rc.trace_family = "S3";
    rc.dealer_mode = "ISOLATED_RECIPIENT_BUNDLES";
    rc.raw_measurement_path = "out.jsonl";
    rc.party_rng_os_seeded = true;
    hl::Value fin = finalize_session_record(session, rc);
    CHECK(evidence::validate(fin).ok && fin.get("security_eligible")->as_bool().value);
    CHECK(fin.get("trace_hash")->as_string().value == rc.trace_hash && fin.get("extra")->get("session_trace_digest")->as_string().value == "abc");
    CHECK(fin.get("code_commit")->as_string().value == "cpp-tree-test" && fin.get("family")->as_string().value == "S3");
    CHECK(fin.get("seed") == nullptr && fin.get("extra")->get("seed") == nullptr);
    CHECK(fin.get("extra")->get("party_rng")->as_string().value == "os_csprng");
    rc.party_rng_os_seeded = false;
    hl::Value seeded = finalize_session_record(session, rc);
    CHECK(evidence::validate(seeded).ok && !seeded.get("security_eligible")->as_bool().value);
    CHECK(seeded.get("status")->as_string().value == "DIAGNOSTIC");
    auto notes = seeded.get("extra")->get("eligibility_notes")->as_list();
    CHECK(notes.ok && notes.value->size() == 1 && (*notes.value)[0].as_string().value.compare(0, 16, "PARTY_RNG_SEEDED") == 0);
    rc.party_rng_os_seeded = true;
    rc.weights_disclosed_to_p1 = true;
    CHECK(!finalize_session_record(session, rc).get("security_eligible")->as_bool().value);
    rc.weights_disclosed_to_p1 = false;
    rc.dealer_mode = "INSECURE_TEST_ONLY_SHARED_SEED";
    hl::Value shared = finalize_session_record(session, rc);
    CHECK(evidence::validate(shared).ok && !shared.get("security_eligible")->as_bool().value &&
          shared.get("status")->as_string().value == "DIAGNOSTIC");
    // the record names its evidence file relative to the results root, never by the place of the tree
    rc.raw_measurement_path = "/srv/tree/results/raw/cpp_sessions.jsonl";
    CHECK_EQ_STR(finalize_session_record(session, rc).get("raw_measurement_path")->as_string().value, "raw/cpp_sessions.jsonl");
  }
  // record_path (the rule of schema.relative_raw_path): relative to $OPENCRAB2_RESULTS_DIR, else cut at the
  // deepest raw/calibration/processed/tables directory above the file, else the file name; relative paths as given
  {
    const char* saved_env = std::getenv("OPENCRAB2_RESULTS_DIR");
    const std::string saved = saved_env ? saved_env : "";
    auto set_root = [](const std::string& v) {
#ifdef _WIN32
      ::_putenv_s("OPENCRAB2_RESULTS_DIR", v.c_str());   // "" removes the variable
#else
      if (v.empty()) ::unsetenv("OPENCRAB2_RESULTS_DIR"); else ::setenv("OPENCRAB2_RESULTS_DIR", v.c_str(), 1);
#endif
    };
    set_root("");
    CHECK_EQ_STR(evidence::record_path("/srv/tree/results/raw/cpp_bench.jsonl"), "raw/cpp_bench.jsonl");
    CHECK_EQ_STR(evidence::record_path("/srv/results/old/tree/results/raw/pilot/web.jsonl"), "raw/pilot/web.jsonl");
    CHECK_EQ_STR(evidence::record_path("/srv/tree/results/raw/../raw/cpp_bench.jsonl"), "raw/cpp_bench.jsonl");
    CHECK_EQ_STR(evidence::record_path("/home/u/results/opencrab-artifact/runs/full/raw/cpp_bench.jsonl"), "raw/cpp_bench.jsonl");
    CHECK_EQ_STR(evidence::record_path("/srv/raw/cell/raw"), "raw/cell/raw");   // the file itself is never the anchor
    CHECK_EQ_STR(evidence::record_path("/srv/tree/output/results"), "results");
    CHECK_EQ_STR(evidence::record_path("/srv/scratch/run1/evidence.jsonl"), "evidence.jsonl");
    CHECK_EQ_STR(evidence::record_path("raw/cpp_bench.jsonl"), "raw/cpp_bench.jsonl");
    CHECK_EQ_STR(evidence::record_path(""), "");
    set_root("/srv/runs/full/");
    CHECK_EQ_STR(evidence::record_path("/srv/runs/full/logs/bench.jsonl"), "logs/bench.jsonl");
    CHECK_EQ_STR(evidence::record_path("/srv/runs/full/raw/cpp_bench.jsonl"), "raw/cpp_bench.jsonl");
    CHECK_EQ_STR(evidence::record_path("/srv/runs/fullx/out.jsonl"), "out.jsonl");
    set_root("");
    bool native_separators = true;  // the backslash is a separator on Windows only
#ifdef _WIN32
    native_separators = evidence::record_path("C:\\tree\\results\\raw\\cpp_bench.jsonl") == "raw/cpp_bench.jsonl" &&
                        evidence::record_path("C:\\scratch\\run1\\evidence.jsonl") == "evidence.jsonl" &&
                        evidence::record_path("raw\\cpp_bench.jsonl") == "raw/cpp_bench.jsonl";
#endif
    CHECK(native_separators);
    set_root(saved);
  }
  // append: validated, one line per record, safe under concurrent writers (lock file protocol of schema.py)
  {
    const std::string path = temp_path("evidence.jsonl");
    std::remove(path.c_str());
    std::remove((path + ".lock").c_str());
    CHECK_EQ_STR(evidence::append(path, session).code_prefix(), "SCHEMA_MISSING_FIELD");  // nothing is written
    {
      std::ifstream probe(path);
      CHECK(!probe.good());
    }
    std::vector<std::thread> writers;
    std::atomic<int> failures{0};
    for (int t = 0; t < 4; ++t) {
      writers.emplace_back([&, t]() {
        for (int i = 0; i < 20; ++i) {
          hl::Value c = conf;
          c.set("repeat", static_cast<int64_t>(t * 100 + i));
          if (!evidence::append(path, c).ok) failures++;
        }
      });
    }
    for (auto& w : writers) w.join();
    CHECK(failures.load() == 0);
    std::ifstream in(path);
    std::string line;
    int lines = 0, parsed = 0;
    while (std::getline(in, line)) {
      ++lines;
      auto pj = hl::parse_json(line);
      if (pj.ok && evidence::validate(pj.value).ok) ++parsed;
    }
    in.close();
    CHECK(lines == 80 && parsed == 80);
    std::ifstream lock_left(path + ".lock");
    CHECK(!lock_left.good());
    std::remove(path.c_str());
  }
  CHECK(evidence::utc_now().size() == 20 && evidence::utc_now().back() == 'Z');
  CHECK(std::string(OPENCRAB2_CODE_COMMIT_LITERAL).compare(0, 9, "cpp-tree-") == 0);
}

// ---------------------------------------------------------------------------
// session helpers of the apps: capability probes, private seeds, trace conversion
struct CapsFull {
  int cancel_after_epoch = -1;
  bool resume = false;
  std::string state_path;
  const std::atomic<bool>* cancel_flag = nullptr;
  bool deterministic_rng = true;
};
struct CapsNone {
  int seed = 0;
};

void test_session_helpers() {
  CapsFull full;
  std::atomic<bool> flag{false};
  CHECK(session_caps::set_cancel_after_epoch(full, 3, 0) && full.cancel_after_epoch == 3);
  CHECK(session_caps::set_resume(full, true, 0) && full.resume);
  CHECK(session_caps::set_state_path(full, std::string("s.json"), 0) && full.state_path == "s.json");
  CHECK(session_caps::set_cancel_flag(full, static_cast<const std::atomic<bool>*>(&flag), 0) && full.cancel_flag == &flag);
  CHECK(session_caps::request_os_party_rng(full) && !full.deterministic_rng);
  CapsNone none;
  CHECK(!session_caps::set_cancel_after_epoch(none, 3, 0));
  CHECK(!session_caps::set_resume(none, true, 0));
  CHECK(!session_caps::set_state_path(none, std::string("s.json"), 0));
  CHECK(!session_caps::set_cancel_flag(none, static_cast<const std::atomic<bool>*>(&flag), 0));
  CHECK(!session_caps::request_os_party_rng(none) && none.seed == 0);
  // what this build's highlevel session offers (reported, the apps fail closed on a missing one)
  {
    hl::SessionConfig cfg;
    bool can_cancel = session_caps::set_cancel_after_epoch(cfg, 1, 0);
    bool can_resume = session_caps::set_resume(cfg, false, 0);
    bool os_rng = session_caps::request_os_party_rng(cfg);
    std::printf("  session interface: cancel_after_epoch=%d resume=%d os_party_rng=%d\n", can_cancel ? 1 : 0, can_resume ? 1 : 0,
                os_rng ? 1 : 0);
    CHECK(true);
  }
  // private session seeds: 31 bits from the OS, never equal in practice
  {
    int a = private_session_seed(), b = private_session_seed(), c = private_session_seed();
    CHECK(a >= 0 && b >= 0 && c >= 0);
    CHECK(!(a == b && b == c));
  }
  Trace t = make_constant_trace(1e6, 0.01, 5.0, 1.0, "const-1M");
  hl::SessionTrace st = to_session_trace(t);
  CHECK(st.trace_id == "const-1M" && st.family == "S0" && st.points.size() == 2);
  CHECK(st.points[0].state.bandwidth_bps == 1e6 && st.points[0].state.one_way_delay_s == 0.01);
  // conversions between the two value views are structural and lossless
  wire::Value w = wire::Value::make_dict();
  w.set("a", wire::Value::from_u64_vector({1, 2, 3}));
  w.set("b", wire::Value::from_list({wire::Value::from_bool(true), wire::Value::none(), wire::Value::from_float(2.5)}));
  auto h = to_highlevel(w);
  REQUIRE(h.ok);
  auto back = to_wire(h.value);
  REQUIRE(back.ok);
  CHECK(wire::encode(back.value) == wire::encode(w));
}

}  // namespace

int main() {
  auto init = tcp_global_init();
  if (!init.ok) {
    std::printf("FAIL %s\n", init.code.c_str());
    return 1;
  }
  test_crypto();
  test_json();
  test_wire();
  test_contract();
  test_tcp();
  test_http();
  test_websocket();
  test_shaper();
  test_dealer();
  test_dealer_named_requests();
  test_model_server();
  test_sidecar();
  test_demo_end_to_end();
  test_evidence();
  test_session_helpers();
  if (g_fail) {
    std::printf("FAIL %d checks (%d passed)\n", g_fail, g_pass);
    return 1;
  }
  std::printf("PASS %d checks\n", g_pass);
  return 0;
}
