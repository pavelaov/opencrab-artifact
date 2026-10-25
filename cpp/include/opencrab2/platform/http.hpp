// opencrab2/platform/http.hpp
//
// Minimal HTTP/1.1 for the sidecar and the model server: a request parser
// (request line, headers, Content-Length body; chunked transfer coding is
// rejected with HTTP_UNSUPPORTED), a responder, Server-Sent Events over a
// persistent response, Origin and Authorization checks, a tiny blocking
// client for tests, and a thread-per-connection server (one request per
// connection, "Connection: close"), whose handler may hijack the socket for
// WebSocket upgrades or SSE.
//
// Codes: HTTP_MALFORMED, HTTP_TOO_LARGE, HTTP_UNSUPPORTED, HTTP_INCOMPLETE,
// HTTP_BAD_STATUS, plus the TCP_* codes of tcp.hpp.
//
// Tested: parser on well formed, truncated, oversized and chunked input;
// serialize/parse round trip; origin and bearer checks; SSE framing; loopback
// server. Design limits: no pipelining, no keep-alive, no HTTP/2, no TLS (the
// sidecar binds 127.0.0.1 and the model server is expected behind a TLS
// terminator in deployment).
#pragma once

#include <opencrab2/platform/support.hpp>
#include <opencrab2/platform/tcp.hpp>

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace opencrab2 {
namespace platform {

struct HttpRequest {
  std::string method;
  std::string target;   // raw request target
  std::string path;     // target without the query string, percent-decoded
  std::string query;    // raw query string (no leading '?')
  std::string version;  // "HTTP/1.1"
  std::vector<std::pair<std::string, std::string>> headers;  // names lower-cased, values trimmed
  std::string body;

  const std::string* header(const std::string& lower_name) const;
  std::string header_or(const std::string& lower_name, const std::string& def) const;
  bool has_header(const std::string& lower_name) const { return header(lower_name) != nullptr; }
  std::vector<std::pair<std::string, std::string>> query_params() const;  // percent-decoded
  std::string query_param(const std::string& name, const std::string& def) const;
};

struct HttpResponse {
  int status = 200;
  std::string reason;  // empty -> standard phrase
  std::vector<std::pair<std::string, std::string>> headers;
  std::string body;

  HttpResponse& set_header(const std::string& name, const std::string& value);  // replace, case-insensitive
  const std::string* header(const std::string& name) const;                     // case-insensitive
  std::string serialize() const;  // adds Content-Length when absent; Connection: close when absent

  static HttpResponse text(int status, const std::string& body,
                           const std::string& content_type = "text/plain; charset=utf-8");
  static HttpResponse json(int status, const json::Value& v);
  static HttpResponse error(int status, const std::string& code);  // JSON {"error": code}
};

std::string http_reason_phrase(int status);
std::string url_decode(const std::string& s);
std::string http_date_now();

class HttpRequestParser {
 public:
  explicit HttpRequestParser(size_t max_header_bytes = 64 * 1024, size_t max_body_bytes = 16 * 1024 * 1024)
      : max_header_(max_header_bytes), max_body_(max_body_bytes) {}
  // Returns true once a complete request is buffered. Fails closed on
  // malformed or oversized input; after a failure the parser must be reset.
  Expected<bool> feed(const uint8_t* p, size_t n);
  bool complete() const { return complete_; }
  Expected<HttpRequest> take();
  void reset();

 private:
  Expected<bool> try_parse_head();
  size_t max_header_;
  size_t max_body_;
  std::string buf_;
  bool head_done_ = false;
  bool complete_ = false;
  size_t body_start_ = 0;
  size_t content_length_ = 0;
  HttpRequest req_;
};

// Whole request in memory (tests and small clients).
Expected<HttpRequest> parse_http_request(const std::string& raw);
// Blocking read of one request from a socket.
Expected<HttpRequest> read_http_request(TcpSocket& sock, size_t max_body_bytes = 16 * 1024 * 1024,
                                        double timeout_s = 15.0);
// Blocking read of one response (status line, headers, Content-Length body or
// read-to-EOF when absent and not an SSE stream).
Expected<HttpResponse> read_http_response(TcpSocket& sock, double timeout_s = 15.0);
// Convenience client: connect, send, read one response, close.
Expected<HttpResponse> http_request(const std::string& host, uint16_t port, const HttpRequest& req,
                                    double timeout_s = 15.0);
std::string serialize_request(const HttpRequest& req, const std::string& host_header);

// Origin allowlist: exact string match on scheme://host[:port] after lower-casing
// scheme and host. A missing Origin header is accepted only when allow_missing.
bool origin_allowed(const HttpRequest& req, const std::vector<std::string>& allowlist, bool allow_missing);
// "Authorization: Bearer <token>" compared in constant time.
bool bearer_token_ok(const HttpRequest& req, const std::string& token);

// Server-Sent Events writer on a hijacked socket.
class SseStream {
 public:
  explicit SseStream(TcpSocket& sock) : sock_(sock) {}
  Expected<bool> begin(const std::vector<std::pair<std::string, std::string>>& extra_headers = {});
  Expected<bool> event(const std::string& name, const std::string& data, const std::string& id = "");
  Expected<bool> comment(const std::string& text);  // keep-alive ": text"
  bool started() const { return started_; }
  static std::string format_event(const std::string& name, const std::string& data, const std::string& id);

 private:
  TcpSocket& sock_;
  bool started_ = false;
};

class HttpServer {
 public:
  struct Exchange {
    HttpRequest request;
    TcpSocket& socket;
    std::string peer;
    bool hijacked = false;   // handler took the socket (WebSocket, SSE)
    bool responded = false;
    const std::atomic<bool>& stopping;
    Exchange(HttpRequest r, TcpSocket& s, std::string p, const std::atomic<bool>& st)
        : request(std::move(r)), socket(s), peer(std::move(p)), stopping(st) {}
    Expected<bool> respond(const HttpResponse& resp);
  };
  using Handler = std::function<void(Exchange&)>;

  HttpServer(TcpListener&& listener, Handler handler);
  ~HttpServer();
  HttpServer(const HttpServer&) = delete;
  HttpServer& operator=(const HttpServer&) = delete;

  Expected<bool> start();
  void stop();
  uint16_t port() const { return listener_.port(); }
  std::string host() const { return listener_.host(); }
  size_t connections_total() const { return connections_total_.load(); }
  size_t connections_active() const;
  const std::atomic<bool>& stopping_flag() const { return stopping_; }
  void set_request_timeout(double s) { request_timeout_s_ = s; }
  void set_max_body(size_t n) { max_body_ = n; }

 private:
  void accept_loop();
  void serve(std::shared_ptr<TcpSocket> sock);
  TcpListener listener_;
  Handler handler_;
  std::thread accept_thread_;
  std::atomic<bool> stopping_{false};
  std::atomic<bool> started_{false};
  std::atomic<size_t> connections_total_{0};
  double request_timeout_s_ = 15.0;
  size_t max_body_ = 16 * 1024 * 1024;
  mutable std::mutex mu_;
  std::vector<std::thread> workers_;
  std::vector<std::weak_ptr<TcpSocket>> live_;
};

}  // namespace platform
}  // namespace opencrab2
