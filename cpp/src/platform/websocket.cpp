// opencrab2/platform/websocket.cpp (see websocket.hpp)
#include <opencrab2/platform/websocket.hpp>

#include <cstring>

namespace opencrab2 {
namespace platform {

const char* const kWebSocketGuid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

std::string websocket_accept_key(const std::string& client_key) {
  std::string s = detail::trim(client_key) + kWebSocketGuid;
  auto h = detail::sha1(s);
  return detail::base64_encode(h.data(), h.size());
}

std::string websocket_random_client_key() {
  auto b = detail::os_random_bytes(16);
  return detail::base64_encode(b.data(), b.size());
}

namespace {
bool token_list_contains(const std::string& header_value, const std::string& token_lower) {
  for (const auto& part : detail::split(header_value, ',')) {
    if (detail::ascii_lower(detail::trim(part)) == token_lower) return true;
  }
  return false;
}
}  // namespace

bool is_websocket_upgrade(const HttpRequest& req) {
  const std::string* up = req.header("upgrade");
  const std::string* conn = req.header("connection");
  return up && detail::ascii_lower(detail::trim(*up)) == "websocket" && conn && token_list_contains(*conn, "upgrade");
}

Expected<HttpResponse> websocket_handshake_response(const HttpRequest& req) {
  if (req.method != "GET") return make_fail<HttpResponse>("WS_BAD_HANDSHAKE");
  if (!is_websocket_upgrade(req)) return make_fail<HttpResponse>("WS_BAD_HANDSHAKE");
  const std::string* ver = req.header("sec-websocket-version");
  if (!ver || detail::trim(*ver) != "13") return make_fail<HttpResponse>("WS_BAD_VERSION");
  const std::string* key = req.header("sec-websocket-key");
  if (!key) return make_fail<HttpResponse>("WS_BAD_HANDSHAKE");
  auto raw = detail::base64_decode(detail::trim(*key));
  if (!raw.ok || raw.value.size() != 16) return make_fail<HttpResponse>("WS_BAD_HANDSHAKE");
  HttpResponse r;
  r.status = 101;
  r.reason = "Switching Protocols";
  r.set_header("Upgrade", "websocket");
  r.set_header("Connection", "Upgrade");
  r.set_header("Sec-WebSocket-Accept", websocket_accept_key(*key));
  r.set_header("Content-Length", "0");
  return make_ok(std::move(r));
}

// ---------------------------------------------------------------------------
// Frame codec
// ---------------------------------------------------------------------------
void ws_apply_mask(uint8_t* p, size_t n, const uint8_t key[4]) {
  for (size_t i = 0; i < n; ++i) p[i] ^= key[i & 3];
}

std::vector<uint8_t> ws_encode_frame(WsOpcode op, const uint8_t* payload, size_t n, bool fin, bool mask,
                                     const uint8_t mask_key[4]) {
  std::vector<uint8_t> out;
  out.reserve(n + 14);
  out.push_back(static_cast<uint8_t>((fin ? 0x80 : 0x00) | (static_cast<uint8_t>(op) & 0x0F)));
  uint8_t mbit = mask ? 0x80 : 0x00;
  if (n < 126) {
    out.push_back(static_cast<uint8_t>(mbit | n));
  } else if (n <= 0xFFFF) {
    out.push_back(static_cast<uint8_t>(mbit | 126));
    out.push_back(static_cast<uint8_t>(n >> 8));
    out.push_back(static_cast<uint8_t>(n));
  } else {
    out.push_back(static_cast<uint8_t>(mbit | 127));
    uint64_t len = n;
    for (int i = 7; i >= 0; --i) out.push_back(static_cast<uint8_t>(len >> (8 * i)));
  }
  if (mask) {
    out.insert(out.end(), mask_key, mask_key + 4);
    size_t start = out.size();
    out.insert(out.end(), payload, payload + n);
    ws_apply_mask(out.data() + start, n, mask_key);
  } else {
    out.insert(out.end(), payload, payload + n);
  }
  return out;
}

Expected<WsFrameHeader> ws_decode_header(const uint8_t* p, size_t n) {
  if (n < 2) return make_fail<WsFrameHeader>("WS_INCOMPLETE");
  WsFrameHeader h;
  h.fin = (p[0] & 0x80) != 0;
  uint8_t rsv = p[0] & 0x70;
  if (rsv != 0) return make_fail<WsFrameHeader>("WS_PROTOCOL_ERROR");
  uint8_t op = p[0] & 0x0F;
  if (!(op <= 2 || (op >= 8 && op <= 10))) return make_fail<WsFrameHeader>("WS_PROTOCOL_ERROR");
  h.opcode = static_cast<WsOpcode>(op);
  h.masked = (p[1] & 0x80) != 0;
  uint8_t len7 = p[1] & 0x7F;
  size_t pos = 2;
  if (len7 < 126) {
    h.payload_len = len7;
  } else if (len7 == 126) {
    if (n < 4) return make_fail<WsFrameHeader>("WS_INCOMPLETE");
    h.payload_len = (static_cast<uint64_t>(p[2]) << 8) | p[3];
    pos = 4;
    if (h.payload_len < 126) return make_fail<WsFrameHeader>("WS_PROTOCOL_ERROR");
  } else {
    if (n < 10) return make_fail<WsFrameHeader>("WS_INCOMPLETE");
    uint64_t len = 0;
    for (int i = 0; i < 8; ++i) len = (len << 8) | p[2 + i];
    if (len >> 63) return make_fail<WsFrameHeader>("WS_PROTOCOL_ERROR");
    if (len <= 0xFFFF) return make_fail<WsFrameHeader>("WS_PROTOCOL_ERROR");
    h.payload_len = len;
    pos = 10;
  }
  if (h.masked) {
    if (n < pos + 4) return make_fail<WsFrameHeader>("WS_INCOMPLETE");
    std::memcpy(h.mask, p + pos, 4);
    pos += 4;
  }
  if (op >= 8) {
    if (!h.fin) return make_fail<WsFrameHeader>("WS_PROTOCOL_ERROR");
    if (h.payload_len > 125) return make_fail<WsFrameHeader>("WS_PROTOCOL_ERROR");
  }
  h.header_len = pos;
  return make_ok(h);
}

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------
WebSocket::WebSocket(TcpSocket&& sock, bool is_client)
    : sock_(std::move(sock)), client_(is_client), mask_rng_(detail::ChaCha20::from_os()) {}

WebSocket::WebSocket(WebSocket&& o) noexcept
    : sock_(std::move(o.sock_)),
      client_(o.client_),
      closed_(o.closed_.load()),
      close_sent_(o.close_sent_.load()),
      close_recv_(o.close_recv_.load()),
      max_message_(o.max_message_),
      stats_(o.stats_),
      mask_rng_(o.mask_rng_) {
  o.closed_ = true;
}

WebSocket& WebSocket::operator=(WebSocket&& o) noexcept {
  if (this != &o) {
    sock_ = std::move(o.sock_);
    client_ = o.client_;
    closed_ = o.closed_.load();
    close_sent_ = o.close_sent_.load();
    close_recv_ = o.close_recv_.load();
    max_message_ = o.max_message_;
    stats_ = o.stats_;
    mask_rng_ = o.mask_rng_;
    o.closed_ = true;
  }
  return *this;
}

WebSocket::~WebSocket() { close(); }

void WebSocket::close() {
  closed_ = true;
  sock_.close();
}

WsStats WebSocket::stats() const {
  std::lock_guard<std::mutex> lk(stats_mu_);
  return stats_;
}

void WebSocket::count_sent(size_t bytes) {
  std::lock_guard<std::mutex> lk(stats_mu_);
  stats_.frames_sent++;
  stats_.bytes_sent += bytes;
}

void WebSocket::count_recv(size_t bytes) {
  std::lock_guard<std::mutex> lk(stats_mu_);
  stats_.frames_recv++;
  stats_.bytes_recv += bytes;
}

Expected<bool> WebSocket::send_frame(WsOpcode op, const uint8_t* p, size_t n, bool fin) {
  if (!sock_.valid() || closed_) return make_fail<bool>("WS_CLOSED");
  std::lock_guard<std::mutex> lk(send_mu_);
  uint8_t key[4] = {0, 0, 0, 0};
  if (client_) mask_rng_.fill(key, 4);
  std::vector<uint8_t> frame = ws_encode_frame(op, p, n, fin, client_, key);
  auto r = sock_.send_all(frame);
  if (!r.ok) return forward_fail<bool>(r);
  count_sent(frame.size());
  return make_ok(true);
}

Expected<bool> WebSocket::send_binary(const uint8_t* p, size_t n) { return send_frame(WsOpcode::Binary, p, n, true); }

Expected<bool> WebSocket::send_text(const std::string& s) {
  return send_frame(WsOpcode::Text, reinterpret_cast<const uint8_t*>(s.data()), s.size(), true);
}

Expected<bool> WebSocket::send_ping(const std::vector<uint8_t>& payload) {
  if (payload.size() > 125) return make_fail<bool>("WS_TOO_LARGE");
  return send_frame(WsOpcode::Ping, payload.data(), payload.size(), true);
}

Expected<bool> WebSocket::send_pong(const std::vector<uint8_t>& payload) {
  if (payload.size() > 125) return make_fail<bool>("WS_TOO_LARGE");
  return send_frame(WsOpcode::Pong, payload.data(), payload.size(), true);
}

Expected<bool> WebSocket::send_close(uint16_t code, const std::string& reason) {
  if (close_sent_.exchange(true)) return make_ok(true);
  std::vector<uint8_t> body;
  body.push_back(static_cast<uint8_t>(code >> 8));
  body.push_back(static_cast<uint8_t>(code));
  std::string r = reason.substr(0, 123);
  body.insert(body.end(), r.begin(), r.end());
  return send_frame(WsOpcode::Close, body.data(), body.size(), true);
}

Expected<WsMessage> WebSocket::recv_frame() {
  if (!sock_.valid() || closed_) return make_fail<WsMessage>("WS_CLOSED");
  uint8_t hdr[14];
  size_t have = 0;
  auto r = sock_.recv_exact(hdr, 2);
  if (!r.ok) return forward_fail<WsMessage>(r);
  have = 2;
  Expected<WsFrameHeader> h = ws_decode_header(hdr, have);
  while (!h.ok && h.code == "WS_INCOMPLETE") {
    // need more header bytes: extended length and/or mask
    uint8_t len7 = hdr[1] & 0x7F;
    bool masked = (hdr[1] & 0x80) != 0;
    size_t need = 2 + (len7 == 126 ? 2 : (len7 == 127 ? 8 : 0)) + (masked ? 4 : 0);
    if (need <= have) return make_fail<WsMessage>("WS_PROTOCOL_ERROR");
    auto r2 = sock_.recv_exact(hdr + have, need - have);
    if (!r2.ok) return forward_fail<WsMessage>(r2);
    have = need;
    h = ws_decode_header(hdr, have);
  }
  if (!h.ok) {
    closed_ = true;
    return forward_fail<WsMessage>(h);
  }
  // Masking rule: client frames masked, server frames unmasked.
  if (client_ && h.value.masked) {
    closed_ = true;
    return make_fail<WsMessage>("WS_PROTOCOL_ERROR");
  }
  if (!client_ && !h.value.masked) {
    closed_ = true;
    return make_fail<WsMessage>("WS_PROTOCOL_ERROR");
  }
  if (h.value.payload_len > max_message_) {
    closed_ = true;
    return make_fail<WsMessage>("WS_TOO_LARGE");
  }
  WsMessage m;
  m.opcode = h.value.opcode;
  m.payload.resize(static_cast<size_t>(h.value.payload_len));
  if (!m.payload.empty()) {
    auto r3 = sock_.recv_exact(m.payload.data(), m.payload.size());
    if (!r3.ok) return forward_fail<WsMessage>(r3);
  }
  if (h.value.masked) ws_apply_mask(m.payload.data(), m.payload.size(), h.value.mask);
  count_recv(h.value.header_len + m.payload.size());
  if (m.opcode == WsOpcode::Close) {
    if (m.payload.size() >= 2) {
      m.close_code = static_cast<uint16_t>((m.payload[0] << 8) | m.payload[1]);
      m.close_reason.assign(m.payload.begin() + 2, m.payload.end());
    } else if (m.payload.size() == 1) {
      closed_ = true;
      return make_fail<WsMessage>("WS_PROTOCOL_ERROR");
    } else {
      m.close_code = 1005;
    }
  }
  m.fin = h.value.fin;
  return make_ok(std::move(m));
}

Expected<WsMessage> WebSocket::recv_message() {
  if (close_recv_) return make_fail<WsMessage>("WS_CLOSED");
  WsMessage assembled;
  bool in_progress = false;
  while (true) {
    auto f = recv_frame();
    if (!f.ok) return f;
    WsMessage& fr = f.value;
    switch (fr.opcode) {
      case WsOpcode::Ping: {
        auto p = send_pong(fr.payload);
        if (!p.ok) return forward_fail<WsMessage>(p);
        {
          std::lock_guard<std::mutex> lk(stats_mu_);
          stats_.pings_answered++;
        }
        continue;
      }
      case WsOpcode::Pong:
        continue;
      case WsOpcode::Close: {
        close_recv_ = true;
        if (!close_sent_) {
          uint16_t code = fr.close_code == 1005 ? 1000 : fr.close_code;
          send_close(code, "");
        }
        return make_ok(std::move(fr));  // carries the peer's close code
      }
      case WsOpcode::Text:
      case WsOpcode::Binary: {
        if (in_progress) {
          closed_ = true;
          return make_fail<WsMessage>("WS_PROTOCOL_ERROR");
        }
        if (fr.fin) return make_ok(std::move(fr));
        assembled.opcode = fr.opcode;
        assembled.payload = std::move(fr.payload);
        in_progress = true;
        continue;
      }
      case WsOpcode::Continuation: {
        if (!in_progress) {
          closed_ = true;
          return make_fail<WsMessage>("WS_PROTOCOL_ERROR");
        }
        if (assembled.payload.size() + fr.payload.size() > max_message_) {
          closed_ = true;
          return make_fail<WsMessage>("WS_TOO_LARGE");
        }
        assembled.payload.insert(assembled.payload.end(), fr.payload.begin(), fr.payload.end());
        if (fr.fin) {
          assembled.fin = true;
          return make_ok(std::move(assembled));
        }
        continue;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Accept / connect
// ---------------------------------------------------------------------------
Expected<WebSocket> websocket_accept(TcpSocket&& sock, const HttpRequest& req) {
  auto resp = websocket_handshake_response(req);
  if (!resp.ok) {
    HttpResponse err = HttpResponse::error(resp.code == "WS_BAD_VERSION" ? 426 : 400, resp.code);
    if (resp.code == "WS_BAD_VERSION") err.set_header("Sec-WebSocket-Version", "13");
    sock.send_all(err.serialize());
    return forward_fail<WebSocket>(resp);
  }
  auto w = sock.send_all(resp.value.serialize());
  if (!w.ok) return forward_fail<WebSocket>(w);
  sock.set_timeout(0);
  return make_ok(WebSocket(std::move(sock), false));
}

Expected<WsUrl> parse_ws_url(const std::string& url) {
  std::string u = detail::trim(url);
  WsUrl out;
  std::string rest;
  if (u.compare(0, 5, "ws://") == 0) {
    rest = u.substr(5);
    out.port = 80;
  } else if (u.compare(0, 6, "wss://") == 0) {
    return make_fail<WsUrl>("WS_BAD_URL");  // TLS not implemented
  } else {
    return make_fail<WsUrl>("WS_BAD_URL");
  }
  size_t slash = rest.find('/');
  std::string hostport = slash == std::string::npos ? rest : rest.substr(0, slash);
  out.path = slash == std::string::npos ? "/" : rest.substr(slash);
  if (hostport.empty()) return make_fail<WsUrl>("WS_BAD_URL");
  if (hostport.find(':') != std::string::npos) {
    auto hp = parse_host_port(hostport);
    if (!hp.ok) return make_fail<WsUrl>("WS_BAD_URL");
    out.host = hp.value.first;
    out.port = hp.value.second;
  } else {
    out.host = hostport;
  }
  return make_ok(out);
}

Expected<WebSocket> websocket_connect(const std::string& host, uint16_t port, const std::string& path,
                                      const std::string& origin,
                                      const std::vector<std::pair<std::string, std::string>>& extra_headers,
                                      double timeout_s) {
  // The server may still be starting: a refused connection is retried, but the
  // whole connect phase never exceeds timeout_s (the retry count scales with
  // the caller's deadline instead of a fixed 50 attempts).
  const double deadline = monotonic_s() + (timeout_s > 0 ? timeout_s : 10.0);
  Expected<TcpSocket> s = make_fail<TcpSocket>("TCP_CONNECT_FAILED");
  while (true) {
    double remaining = deadline - monotonic_s();
    if (remaining < 0.05) remaining = 0.05;
    s = TcpSocket::connect(host, port, remaining, 1, 0.0);
    if (s.ok || s.code == "TCP_RESOLVE_FAILED" || s.code == "TCP_INIT_FAILED") break;
    if (monotonic_s() + 0.05 >= deadline) break;
    sleep_s(0.05);
  }
  if (!s.ok) return forward_fail<WebSocket>(s);
  s.value.set_timeout(timeout_s);
  std::string key = websocket_random_client_key();
  HttpRequest req;
  req.method = "GET";
  req.target = path.empty() ? "/" : path;
  req.headers.emplace_back("Upgrade", "websocket");
  req.headers.emplace_back("Connection", "Upgrade");
  req.headers.emplace_back("Sec-WebSocket-Key", key);
  req.headers.emplace_back("Sec-WebSocket-Version", "13");
  if (!origin.empty()) req.headers.emplace_back("Origin", origin);
  for (const auto& kv : extra_headers) req.headers.emplace_back(kv.first, kv.second);
  std::string raw = serialize_request(req, host + ":" + std::to_string(port));
  // serialize_request appends Connection: close only when absent; we set Upgrade above.
  auto w = s.value.send_all(raw);
  if (!w.ok) return forward_fail<WebSocket>(w);
  auto resp = read_http_response(s.value, timeout_s);
  if (!resp.ok) return forward_fail<WebSocket>(resp);
  if (resp.value.status != 101) return make_fail<WebSocket>("WS_BAD_HANDSHAKE");
  const std::string* acc = resp.value.header("sec-websocket-accept");
  if (!acc || detail::trim(*acc) != websocket_accept_key(key)) return make_fail<WebSocket>("WS_BAD_ACCEPT");
  s.value.set_timeout(0);
  return make_ok(WebSocket(std::move(s.value), true));
}

Expected<WebSocket> websocket_connect_url(const std::string& url, const std::string& origin, double timeout_s) {
  auto u = parse_ws_url(url);
  if (!u.ok) return forward_fail<WebSocket>(u);
  return websocket_connect(u.value.host, u.value.port, u.value.path, origin, {}, timeout_s);
}

// ---------------------------------------------------------------------------
// WebSocketChannel
// ---------------------------------------------------------------------------
WebSocketChannel::WebSocketChannel(WebSocket&& ws, int party, WallClockShaper* shaper)
    : ws_(std::move(ws)), party_(party), shaper_(shaper) {}

Expected<bool> WebSocketChannel::send(const wire::Value& v) {
  std::vector<uint8_t> f = wire::frame(v);
  if (shaper_) stats.shaping_delay_s += shaper_->apply(f.size());
  auto r = ws_.send_binary(f);
  if (!r.ok) return r;
  stats.bytes_sent += f.size();
  stats.messages_sent++;
  return make_ok(true);
}

Expected<wire::Value> WebSocketChannel::recv() {
  while (true) {
    auto m = ws_.recv_message();
    if (!m.ok) return forward_fail<wire::Value>(m);
    if (m.value.is_close()) return make_fail<wire::Value>("WS_CLOSED");
    if (!m.value.is_binary()) continue;  // text frames are not protocol messages
    stats.bytes_recv += m.value.payload.size();
    stats.messages_recv++;
    return wire::decode_frame(m.value.payload);
  }
}

Expected<wire::Value> WebSocketChannel::exchange(const wire::Value& v) {
  stats.rounds++;
  if (party_ == 0) {
    auto s = send(v);
    if (!s.ok) return forward_fail<wire::Value>(s);
    return recv();
  }
  auto other = recv();
  if (!other.ok) return other;
  auto s = send(v);
  if (!s.ok) return forward_fail<wire::Value>(s);
  return other;
}

Expected<bool> WebSocketChannel::close(uint16_t code) {
  auto r = ws_.send_close(code, "");
  ws_.close();
  return r;
}

json::Value WebSocketChannel::stats_json() const {
  json::Value o = json::Value::object();
  o.set("bytes_sent", json::Value::integer(static_cast<int64_t>(stats.bytes_sent)));
  o.set("bytes_recv", json::Value::integer(static_cast<int64_t>(stats.bytes_recv)));
  o.set("bytes_total", json::Value::integer(static_cast<int64_t>(stats.bytes_sent + stats.bytes_recv)));
  o.set("messages_sent", json::Value::integer(static_cast<int64_t>(stats.messages_sent)));
  o.set("messages_recv", json::Value::integer(static_cast<int64_t>(stats.messages_recv)));
  o.set("rounds", json::Value::integer(static_cast<int64_t>(stats.rounds)));
  o.set("shaping_delay_s", json::Value::number(stats.shaping_delay_s));
  WsStats w = ws_.stats();
  o.set("ws_bytes_sent", json::Value::integer(static_cast<int64_t>(w.bytes_sent)));
  o.set("ws_bytes_recv", json::Value::integer(static_cast<int64_t>(w.bytes_recv)));
  return o;
}

}  // namespace platform
}  // namespace opencrab2
