// opencrab2/platform/http.cpp (see http.hpp)
#include <opencrab2/platform/http.hpp>

#include <chrono>
#include <cstdlib>
#include <ctime>

namespace opencrab2 {
namespace platform {

// ---------------------------------------------------------------------------
// HttpRequest / HttpResponse
// ---------------------------------------------------------------------------
const std::string* HttpRequest::header(const std::string& lower_name) const {
  for (const auto& kv : headers) {
    if (kv.first == lower_name) return &kv.second;
  }
  return nullptr;
}

std::string HttpRequest::header_or(const std::string& lower_name, const std::string& def) const {
  const std::string* v = header(lower_name);
  return v ? *v : def;
}

std::vector<std::pair<std::string, std::string>> HttpRequest::query_params() const {
  std::vector<std::pair<std::string, std::string>> out;
  if (query.empty()) return out;
  for (const auto& part : detail::split(query, '&')) {
    if (part.empty()) continue;
    size_t eq = part.find('=');
    if (eq == std::string::npos) {
      out.emplace_back(url_decode(part), "");
    } else {
      out.emplace_back(url_decode(part.substr(0, eq)), url_decode(part.substr(eq + 1)));
    }
  }
  return out;
}

std::string HttpRequest::query_param(const std::string& name, const std::string& def) const {
  for (const auto& kv : query_params()) {
    if (kv.first == name) return kv.second;
  }
  return def;
}

HttpResponse& HttpResponse::set_header(const std::string& name, const std::string& value) {
  std::string lname = detail::ascii_lower(name);
  for (auto& kv : headers) {
    if (detail::ascii_lower(kv.first) == lname) {
      kv.second = value;
      return *this;
    }
  }
  headers.emplace_back(name, value);
  return *this;
}

const std::string* HttpResponse::header(const std::string& name) const {
  std::string lname = detail::ascii_lower(name);
  for (const auto& kv : headers) {
    if (detail::ascii_lower(kv.first) == lname) return &kv.second;
  }
  return nullptr;
}

std::string HttpResponse::serialize() const {
  std::string out = "HTTP/1.1 " + std::to_string(status) + " " + (reason.empty() ? http_reason_phrase(status) : reason) + "\r\n";
  bool has_len = header("content-length") != nullptr;
  bool has_conn = header("connection") != nullptr;
  bool has_date = header("date") != nullptr;
  for (const auto& kv : headers) out += kv.first + ": " + kv.second + "\r\n";
  if (!has_len) out += "Content-Length: " + std::to_string(body.size()) + "\r\n";
  if (!has_conn) out += "Connection: close\r\n";
  if (!has_date) out += "Date: " + http_date_now() + "\r\n";
  out += "\r\n";
  out += body;
  return out;
}

HttpResponse HttpResponse::text(int status, const std::string& body, const std::string& content_type) {
  HttpResponse r;
  r.status = status;
  r.body = body;
  r.set_header("Content-Type", content_type);
  return r;
}

HttpResponse HttpResponse::json(int status, const json::Value& v) {
  HttpResponse r;
  r.status = status;
  r.body = v.dump(false);
  r.set_header("Content-Type", "application/json");
  return r;
}

HttpResponse HttpResponse::error(int status, const std::string& code) {
  json::Value o = json::Value::object();
  o.set("error", json::Value::string(code));
  o.set("status", json::Value::integer(status));
  return json(status, o);
}

std::string http_reason_phrase(int status) {
  switch (status) {
    case 100: return "Continue";
    case 101: return "Switching Protocols";
    case 200: return "OK";
    case 201: return "Created";
    case 202: return "Accepted";
    case 204: return "No Content";
    case 304: return "Not Modified";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 403: return "Forbidden";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 408: return "Request Timeout";
    case 409: return "Conflict";
    case 410: return "Gone";
    case 411: return "Length Required";
    case 413: return "Payload Too Large";
    case 415: return "Unsupported Media Type";
    case 426: return "Upgrade Required";
    case 429: return "Too Many Requests";
    case 500: return "Internal Server Error";
    case 501: return "Not Implemented";
    case 503: return "Service Unavailable";
    default: return "Status";
  }
}

std::string url_decode(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '%' && i + 2 < s.size()) {
      int hi = detail::hex_nibble(s[i + 1]);
      int lo = detail::hex_nibble(s[i + 2]);
      if (hi >= 0 && lo >= 0) {
        out.push_back(static_cast<char>((hi << 4) | lo));
        i += 2;
        continue;
      }
    }
    if (s[i] == '+') {
      out.push_back(' ');
    } else {
      out.push_back(s[i]);
    }
  }
  return out;
}

std::string http_date_now() {
  std::time_t t = std::time(nullptr);
  std::tm tmv;
#ifdef _WIN32
  gmtime_s(&tmv, &t);
#else
  gmtime_r(&t, &tmv);
#endif
  char buf[64];
  std::strftime(buf, sizeof buf, "%a, %d %b %Y %H:%M:%S GMT", &tmv);
  return buf;
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------
void HttpRequestParser::reset() {
  buf_.clear();
  head_done_ = false;
  complete_ = false;
  body_start_ = 0;
  content_length_ = 0;
  req_ = HttpRequest();
}

Expected<bool> HttpRequestParser::try_parse_head() {
  size_t end = buf_.find("\r\n\r\n");
  if (end == std::string::npos) {
    if (buf_.size() > max_header_) return make_fail<bool>("HTTP_TOO_LARGE");
    return make_ok(false);
  }
  if (end > max_header_) return make_fail<bool>("HTTP_TOO_LARGE");
  std::string head = buf_.substr(0, end);
  body_start_ = end + 4;
  std::vector<std::string> lines = detail::split(head, '\n');
  if (lines.empty()) return make_fail<bool>("HTTP_MALFORMED");
  std::string request_line = detail::trim(lines[0]);
  std::vector<std::string> parts = detail::split(request_line, ' ');
  if (parts.size() != 3) return make_fail<bool>("HTTP_MALFORMED");
  req_.method = parts[0];
  req_.target = parts[1];
  req_.version = parts[2];
  if (req_.version != "HTTP/1.1" && req_.version != "HTTP/1.0") return make_fail<bool>("HTTP_UNSUPPORTED");
  if (req_.method.empty() || req_.target.empty()) return make_fail<bool>("HTTP_MALFORMED");
  size_t q = req_.target.find('?');
  if (q == std::string::npos) {
    req_.path = url_decode(req_.target);
    req_.query.clear();
  } else {
    req_.path = url_decode(req_.target.substr(0, q));
    req_.query = req_.target.substr(q + 1);
  }
  for (size_t i = 1; i < lines.size(); ++i) {
    std::string line = detail::trim(lines[i]);
    if (line.empty()) continue;
    size_t colon = line.find(':');
    if (colon == std::string::npos || colon == 0) return make_fail<bool>("HTTP_MALFORMED");
    std::string name = detail::ascii_lower(detail::trim(line.substr(0, colon)));
    std::string value = detail::trim(line.substr(colon + 1));
    req_.headers.emplace_back(name, value);
  }
  if (req_.has_header("transfer-encoding")) return make_fail<bool>("HTTP_UNSUPPORTED");
  const std::string* cl = req_.header("content-length");
  if (cl) {
    if (cl->empty()) return make_fail<bool>("HTTP_MALFORMED");
    for (char c : *cl) {
      if (c < '0' || c > '9') return make_fail<bool>("HTTP_MALFORMED");
    }
    if (cl->size() > 18) return make_fail<bool>("HTTP_TOO_LARGE");
    unsigned long long v = std::strtoull(cl->c_str(), nullptr, 10);
    if (v > max_body_) return make_fail<bool>("HTTP_TOO_LARGE");
    content_length_ = static_cast<size_t>(v);
  }
  head_done_ = true;
  return make_ok(true);
}

Expected<bool> HttpRequestParser::feed(const uint8_t* p, size_t n) {
  if (complete_) return make_ok(true);
  buf_.append(reinterpret_cast<const char*>(p), n);
  if (!head_done_) {
    auto r = try_parse_head();
    if (!r.ok) return r;
    if (!r.value) return make_ok(false);
  }
  if (buf_.size() - body_start_ >= content_length_) {
    req_.body = buf_.substr(body_start_, content_length_);
    complete_ = true;
    return make_ok(true);
  }
  return make_ok(false);
}

Expected<HttpRequest> HttpRequestParser::take() {
  if (!complete_) return make_fail<HttpRequest>("HTTP_INCOMPLETE");
  HttpRequest out = std::move(req_);
  reset();
  return make_ok(std::move(out));
}

Expected<HttpRequest> parse_http_request(const std::string& raw) {
  HttpRequestParser p;
  auto r = p.feed(reinterpret_cast<const uint8_t*>(raw.data()), raw.size());
  if (!r.ok) return forward_fail<HttpRequest>(r);
  if (!r.value) return make_fail<HttpRequest>("HTTP_INCOMPLETE");
  return p.take();
}

Expected<HttpRequest> read_http_request(TcpSocket& sock, size_t max_body_bytes, double timeout_s) {
  HttpRequestParser p(64 * 1024, max_body_bytes);
  sock.set_timeout(timeout_s);
  uint8_t buf[8192];
  while (true) {
    auto r = sock.recv_some(buf, sizeof buf);
    if (!r.ok) return forward_fail<HttpRequest>(r);
    if (r.value == 0) return make_fail<HttpRequest>("HTTP_INCOMPLETE");
    auto f = p.feed(buf, r.value);
    if (!f.ok) return forward_fail<HttpRequest>(f);
    if (f.value) return p.take();
  }
}

Expected<HttpResponse> read_http_response(TcpSocket& sock, double timeout_s) {
  sock.set_timeout(timeout_s);
  std::string buf;
  uint8_t tmp[8192];
  size_t head_end = std::string::npos;
  while (head_end == std::string::npos) {
    auto r = sock.recv_some(tmp, sizeof tmp);
    if (!r.ok) return forward_fail<HttpResponse>(r);
    if (r.value == 0) return make_fail<HttpResponse>("HTTP_INCOMPLETE");
    buf.append(reinterpret_cast<const char*>(tmp), r.value);
    head_end = buf.find("\r\n\r\n");
    if (buf.size() > 64 * 1024 && head_end == std::string::npos) return make_fail<HttpResponse>("HTTP_TOO_LARGE");
  }
  HttpResponse resp;
  std::vector<std::string> lines = detail::split(buf.substr(0, head_end), '\n');
  if (lines.empty()) return make_fail<HttpResponse>("HTTP_MALFORMED");
  std::string status_line = detail::trim(lines[0]);
  if (status_line.size() < 12 || status_line.compare(0, 5, "HTTP/") != 0) return make_fail<HttpResponse>("HTTP_MALFORMED");
  size_t sp = status_line.find(' ');
  if (sp == std::string::npos || sp + 4 > status_line.size()) return make_fail<HttpResponse>("HTTP_MALFORMED");
  resp.status = std::atoi(status_line.substr(sp + 1, 3).c_str());
  if (resp.status < 100 || resp.status > 599) return make_fail<HttpResponse>("HTTP_BAD_STATUS");
  if (sp + 5 < status_line.size()) resp.reason = status_line.substr(sp + 5);
  for (size_t i = 1; i < lines.size(); ++i) {
    std::string line = detail::trim(lines[i]);
    if (line.empty()) continue;
    size_t colon = line.find(':');
    if (colon == std::string::npos) return make_fail<HttpResponse>("HTTP_MALFORMED");
    resp.headers.emplace_back(detail::trim(line.substr(0, colon)), detail::trim(line.substr(colon + 1)));
  }
  std::string body = buf.substr(head_end + 4);
  const std::string* cl = resp.header("content-length");
  const std::string* ct = resp.header("content-type");
  bool sse = ct && detail::ascii_lower(*ct).find("text/event-stream") != std::string::npos;
  if (cl) {
    size_t want = static_cast<size_t>(std::strtoull(cl->c_str(), nullptr, 10));
    while (body.size() < want) {
      auto r = sock.recv_some(tmp, sizeof tmp);
      if (!r.ok) return forward_fail<HttpResponse>(r);
      if (r.value == 0) break;
      body.append(reinterpret_cast<const char*>(tmp), r.value);
    }
    body.resize(want < body.size() ? want : body.size());
  } else if (!sse && resp.status != 101 && resp.status != 204) {
    while (true) {
      auto r = sock.recv_some(tmp, sizeof tmp);
      if (!r.ok) break;
      if (r.value == 0) break;
      body.append(reinterpret_cast<const char*>(tmp), r.value);
    }
  }
  resp.body = body;
  return make_ok(std::move(resp));
}

std::string serialize_request(const HttpRequest& req, const std::string& host_header) {
  std::string target = req.target.empty() ? (req.path + (req.query.empty() ? "" : "?" + req.query)) : req.target;
  std::string out = req.method + " " + target + " HTTP/1.1\r\n";
  bool has_host = false, has_len = false, has_conn = false;
  for (const auto& kv : req.headers) {
    std::string l = detail::ascii_lower(kv.first);
    if (l == "host") has_host = true;
    if (l == "content-length") has_len = true;
    if (l == "connection") has_conn = true;
    out += kv.first + ": " + kv.second + "\r\n";
  }
  if (!has_host) out += "Host: " + host_header + "\r\n";
  if (!has_len && (!req.body.empty() || req.method == "POST" || req.method == "PUT"))
    out += "Content-Length: " + std::to_string(req.body.size()) + "\r\n";
  if (!has_conn) out += "Connection: close\r\n";
  out += "\r\n";
  out += req.body;
  return out;
}

Expected<HttpResponse> http_request(const std::string& host, uint16_t port, const HttpRequest& req, double timeout_s) {
  auto s = TcpSocket::connect(host, port, timeout_s);
  if (!s.ok) return forward_fail<HttpResponse>(s);
  s.value.set_timeout(timeout_s);
  auto w = s.value.send_all(serialize_request(req, host + ":" + std::to_string(port)));
  if (!w.ok) return forward_fail<HttpResponse>(w);
  return read_http_response(s.value, timeout_s);
}

// ---------------------------------------------------------------------------
// Origin / bearer
// ---------------------------------------------------------------------------
namespace {
std::string normalize_origin(const std::string& o) {
  std::string t = detail::trim(o);
  size_t scheme_end = t.find("://");
  if (scheme_end == std::string::npos) return detail::ascii_lower(t);
  std::string scheme = detail::ascii_lower(t.substr(0, scheme_end));
  std::string rest = t.substr(scheme_end + 3);
  size_t slash = rest.find('/');
  if (slash != std::string::npos) rest = rest.substr(0, slash);
  rest = detail::ascii_lower(rest);
  if (scheme == "http" && rest.size() > 3 && rest.compare(rest.size() - 3, 3, ":80") == 0) rest.resize(rest.size() - 3);
  if (scheme == "https" && rest.size() > 4 && rest.compare(rest.size() - 4, 4, ":443") == 0) rest.resize(rest.size() - 4);
  return scheme + "://" + rest;
}
}  // namespace

bool origin_allowed(const HttpRequest& req, const std::vector<std::string>& allowlist, bool allow_missing) {
  const std::string* origin = req.header("origin");
  if (origin == nullptr || origin->empty()) return allow_missing;
  if (*origin == "null") return false;
  std::string norm = normalize_origin(*origin);
  for (const auto& a : allowlist) {
    if (normalize_origin(a) == norm) return true;
  }
  return false;
}

bool bearer_token_ok(const HttpRequest& req, const std::string& token) {
  if (token.empty()) return false;
  const std::string* auth = req.header("authorization");
  if (!auth) return false;
  std::string v = detail::trim(*auth);
  if (v.size() < 7) return false;
  if (detail::ascii_lower(v.substr(0, 7)) != "bearer ") return false;
  std::string presented = detail::trim(v.substr(7));
  return detail::constant_time_equal(presented, token);
}

// ---------------------------------------------------------------------------
// SSE
// ---------------------------------------------------------------------------
Expected<bool> SseStream::begin(const std::vector<std::pair<std::string, std::string>>& extra_headers) {
  if (started_) return make_ok(true);
  std::string head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nX-Accel-Buffering: no\r\n";
  for (const auto& kv : extra_headers) head += kv.first + ": " + kv.second + "\r\n";
  head += "\r\n";
  auto r = sock_.send_all(head);
  if (!r.ok) return forward_fail<bool>(r);
  started_ = true;
  return make_ok(true);
}

std::string SseStream::format_event(const std::string& name, const std::string& data, const std::string& id) {
  std::string out;
  if (!id.empty()) out += "id: " + id + "\n";
  if (!name.empty()) out += "event: " + name + "\n";
  // data may contain newlines: each line gets its own "data:" field
  size_t start = 0;
  while (true) {
    size_t nl = data.find('\n', start);
    if (nl == std::string::npos) {
      out += "data: " + data.substr(start) + "\n";
      break;
    }
    out += "data: " + data.substr(start, nl - start) + "\n";
    start = nl + 1;
  }
  out += "\n";
  return out;
}

Expected<bool> SseStream::event(const std::string& name, const std::string& data, const std::string& id) {
  if (!started_) {
    auto b = begin();
    if (!b.ok) return b;
  }
  auto r = sock_.send_all(format_event(name, data, id));
  if (!r.ok) return forward_fail<bool>(r);
  return make_ok(true);
}

Expected<bool> SseStream::comment(const std::string& text) {
  if (!started_) {
    auto b = begin();
    if (!b.ok) return b;
  }
  auto r = sock_.send_all(": " + text + "\n\n");
  if (!r.ok) return forward_fail<bool>(r);
  return make_ok(true);
}

// ---------------------------------------------------------------------------
// HttpServer
// ---------------------------------------------------------------------------
Expected<bool> HttpServer::Exchange::respond(const HttpResponse& resp) {
  if (responded) return make_fail<bool>("HTTP_ALREADY_RESPONDED");
  responded = true;
  auto r = socket.send_all(resp.serialize());
  if (!r.ok) return forward_fail<bool>(r);
  return make_ok(true);
}

HttpServer::HttpServer(TcpListener&& listener, Handler handler)
    : listener_(std::move(listener)), handler_(std::move(handler)) {}

HttpServer::~HttpServer() { stop(); }

Expected<bool> HttpServer::start() {
  if (!listener_.valid()) return make_fail<bool>("TCP_INVALID_SOCKET");
  if (started_.exchange(true)) return make_ok(true);
  stopping_ = false;
  accept_thread_ = std::thread([this]() { accept_loop(); });
  return make_ok(true);
}

void HttpServer::stop() {
  if (!started_) return;
  stopping_ = true;
  listener_.close();
  {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& w : live_) {
      if (auto s = w.lock()) s->shutdown_both();
    }
  }
  if (accept_thread_.joinable()) accept_thread_.join();
  std::vector<std::thread> workers;
  {
    std::lock_guard<std::mutex> lk(mu_);
    workers.swap(workers_);
  }
  for (auto& t : workers) {
    if (t.joinable()) t.join();
  }
  started_ = false;
}

size_t HttpServer::connections_active() const {
  std::lock_guard<std::mutex> lk(mu_);
  size_t n = 0;
  for (const auto& w : live_) {
    if (!w.expired()) ++n;
  }
  return n;
}

void HttpServer::accept_loop() {
  while (!stopping_) {
    auto c = listener_.accept(0.2);
    if (!c.ok) {
      if (c.code == "TCP_TIMEOUT") continue;
      if (stopping_) break;
      sleep_s(0.01);
      continue;
    }
    connections_total_++;
    auto sock = std::make_shared<TcpSocket>(std::move(c.value));
    std::lock_guard<std::mutex> lk(mu_);
    // reap finished workers (their sockets expired)
    size_t keep = 0;
    for (size_t i = 0; i < live_.size(); ++i) {
      if (!live_[i].expired()) {
        live_[keep] = live_[i];
        if (keep != i) workers_[keep] = std::move(workers_[i]);
        ++keep;
      } else {
        if (workers_[i].joinable()) workers_[i].join();
      }
    }
    live_.resize(keep);
    workers_.resize(keep);
    live_.push_back(sock);
    workers_.emplace_back([this, sock]() { serve(sock); });
  }
}

void HttpServer::serve(std::shared_ptr<TcpSocket> sock) {
  auto req = read_http_request(*sock, max_body_, request_timeout_s_);
  if (!req.ok) {
    int status = 400;
    if (req.code == "HTTP_TOO_LARGE") status = 413;
    if (req.code == "HTTP_UNSUPPORTED") status = 501;  // e.g. Transfer-Encoding: not implemented, never guessed
    if (req.code == "TCP_TIMEOUT") status = 408;
    if (req.code != "TCP_CLOSED" && req.code != "HTTP_INCOMPLETE") {
      sock->send_all(HttpResponse::error(status, req.code).serialize());
    }
    sock->close();
    return;
  }
  Exchange ex(std::move(req.value), *sock, sock->peer(), stopping_);
  handler_(ex);
  if (!ex.hijacked) {
    if (!ex.responded) sock->send_all(HttpResponse::error(500, "HANDLER_NO_RESPONSE").serialize());
    // half close first: the response drains before the FIN and a client that
    // is still writing does not get a reset that discards the response
    sock->shutdown_send();
    sock->close();
  }
}

}  // namespace platform
}  // namespace opencrab2
