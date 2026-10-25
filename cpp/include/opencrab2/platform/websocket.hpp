// opencrab2/platform/websocket.hpp
//
// RFC 6455 WebSocket server and client on top of tcp.hpp and http.hpp:
// opening handshake (SHA-1 + base64 accept key), frame codec with masking
// (client frames masked, server frames unmasked, both directions enforced),
// fragmentation reassembly, ping/pong, close handshake, size limits. The
// WebSocketChannel adapter exposes the send/recv/exchange/stats interface of
// opencrab2::highlevel::Channel over the wire codec of codec.hpp: one
// binary WebSocket message carries exactly one wire frame (8-byte big-endian
// length prefix followed by the typed payload), so a Python Channel peer and
// a C++ WebSocketChannel peer exchange identical bytes.
//
// Codes: WS_BAD_HANDSHAKE, WS_BAD_VERSION, WS_BAD_ACCEPT, WS_PROTOCOL_ERROR,
// WS_INCOMPLETE, WS_TOO_LARGE, WS_CLOSED, WS_UNEXPECTED_OPCODE, WS_BAD_URL,
// plus TCP_* and HTTP_* codes.
//
// Tested: RFC 6455 section 1.3 accept-key vector, frame codec round trip for
// 7/16/64-bit lengths with and without mask, loopback echo over a masked
// client connection, ping/pong, close handshake, channel exchange round trip
// and byte counters. Design limits: no permessage-deflate, no subprotocol
// negotiation, no TLS.
#pragma once

#include <opencrab2/platform/http.hpp>
#include <opencrab2/platform/shaper.hpp>
#include <opencrab2/platform/support.hpp>
#include <opencrab2/platform/tcp.hpp>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace opencrab2 {
namespace platform {

extern const char* const kWebSocketGuid;  // 258EAFA5-E914-47DA-95CA-C5AB0DC85B11

// base64(sha1(key + GUID)); RFC 6455 section 1.3 vector:
// "dGhlIHNhbXBsZSBub25jZQ==" -> "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
std::string websocket_accept_key(const std::string& client_key);
std::string websocket_random_client_key();  // base64 of 16 CSPRNG bytes

bool is_websocket_upgrade(const HttpRequest& req);
// Validates Upgrade/Connection/Sec-WebSocket-Version(13)/Sec-WebSocket-Key
// and returns the 101 response to send (not sent here).
Expected<HttpResponse> websocket_handshake_response(const HttpRequest& req);

enum class WsOpcode : uint8_t { Continuation = 0x0, Text = 0x1, Binary = 0x2, Close = 0x8, Ping = 0x9, Pong = 0xA };

struct WsFrameHeader {
  bool fin = true;
  WsOpcode opcode = WsOpcode::Binary;
  bool masked = false;
  uint64_t payload_len = 0;
  uint8_t mask[4] = {0, 0, 0, 0};
  size_t header_len = 0;  // bytes consumed by the header
};

// Pure frame codec (unit tested without sockets).
std::vector<uint8_t> ws_encode_frame(WsOpcode op, const uint8_t* payload, size_t n, bool fin, bool mask,
                                     const uint8_t mask_key[4]);
Expected<WsFrameHeader> ws_decode_header(const uint8_t* p, size_t n);  // WS_INCOMPLETE when more bytes are needed
void ws_apply_mask(uint8_t* p, size_t n, const uint8_t key[4]);

struct WsMessage {
  WsOpcode opcode = WsOpcode::Binary;
  std::vector<uint8_t> payload;
  bool fin = true;  // per frame (recv_frame); always true for reassembled messages
  uint16_t close_code = 0;
  std::string close_reason;
  bool is_text() const { return opcode == WsOpcode::Text; }
  bool is_binary() const { return opcode == WsOpcode::Binary; }
  bool is_close() const { return opcode == WsOpcode::Close; }
  std::string text() const { return std::string(payload.begin(), payload.end()); }
};

struct WsStats {
  uint64_t frames_sent = 0;
  uint64_t frames_recv = 0;
  uint64_t bytes_sent = 0;  // on the wire, including framing
  uint64_t bytes_recv = 0;
  uint64_t pings_answered = 0;
};

class WebSocket {
 public:
  WebSocket() = default;
  WebSocket(TcpSocket&& sock, bool is_client);
  WebSocket(WebSocket&& o) noexcept;
  WebSocket& operator=(WebSocket&& o) noexcept;
  WebSocket(const WebSocket&) = delete;
  WebSocket& operator=(const WebSocket&) = delete;
  ~WebSocket();

  bool valid() const { return sock_.valid() && !closed_.load(); }
  bool is_client() const { return client_; }

  Expected<bool> send_binary(const uint8_t* p, size_t n);
  Expected<bool> send_binary(const std::vector<uint8_t>& v) { return send_binary(v.data(), v.size()); }
  Expected<bool> send_text(const std::string& s);
  Expected<bool> send_ping(const std::vector<uint8_t>& payload = {});
  Expected<bool> send_pong(const std::vector<uint8_t>& payload = {});
  Expected<bool> send_close(uint16_t code = 1000, const std::string& reason = "");

  // Next data message (Text or Binary) with fragments reassembled; pings are
  // answered internally; a Close frame is answered (once) and returned as a
  // message with opcode Close; after that every call fails with WS_CLOSED.
  Expected<WsMessage> recv_message();
  // One raw frame, control frames included (tests, diagnostics).
  Expected<WsMessage> recv_frame();

  void set_max_message_bytes(size_t n) { max_message_ = n; }
  Expected<bool> set_timeout(double seconds) { return sock_.set_timeout(seconds); }
  WsStats stats() const;  // snapshot; one thread may send while another receives
  TcpSocket& socket() { return sock_; }
  bool close_received() const { return close_recv_.load(); }
  void close();  // closes the TCP socket; send_close first for a clean shutdown

 private:
  Expected<bool> send_frame(WsOpcode op, const uint8_t* p, size_t n, bool fin);
  void count_sent(size_t bytes);
  void count_recv(size_t bytes);
  TcpSocket sock_;
  bool client_ = false;
  std::atomic<bool> closed_{false};
  std::atomic<bool> close_sent_{false};
  std::atomic<bool> close_recv_{false};
  size_t max_message_ = static_cast<size_t>(64) << 20;
  WsStats stats_;
  mutable std::mutex stats_mu_;
  detail::ChaCha20 mask_rng_;
  std::mutex send_mu_;
};

// Server side: validates the upgrade request, sends 101 (or an error
// response, then fails), returns the socket wrapped as a server WebSocket.
Expected<WebSocket> websocket_accept(TcpSocket&& sock, const HttpRequest& req);
// Client side.
Expected<WebSocket> websocket_connect(const std::string& host, uint16_t port, const std::string& path,
                                      const std::string& origin = "",
                                      const std::vector<std::pair<std::string, std::string>>& extra_headers = {},
                                      double timeout_s = 10.0);
// ws://host:port/path
Expected<WebSocket> websocket_connect_url(const std::string& url, const std::string& origin = "", double timeout_s = 10.0);
struct WsUrl {
  std::string host;
  uint16_t port = 80;
  std::string path = "/";
};
Expected<WsUrl> parse_ws_url(const std::string& url);

// Frame-level channel adapter over one WebSocket: one binary message carries
// one wire frame. It serves the control dialogue of the model server (HELLO,
// MSG, RESUME, CANCEL, FLOW) and counts with opencrab2::highlevel::ChannelStats.
// The highlevel::Channel that the session loop runs on is built on top of it
// in session_bridge.hpp (WsSessionChannel derives from highlevel::Channel).
class WebSocketChannel {
 public:
  WebSocketChannel(WebSocket&& ws, int party, WallClockShaper* shaper = nullptr);
  Expected<bool> send(const wire::Value& v);
  Expected<wire::Value> recv();
  // P0 sends then receives, P1 receives then sends; counts one round.
  Expected<wire::Value> exchange(const wire::Value& v);
  // A one-directional message that still forms a protocol round (request /
  // response against a server, reveal_to).
  void count_round() { stats.rounds++; }
  ChannelStats stats;
  int party() const { return party_; }
  void set_shaper(WallClockShaper* s) { shaper_ = s; }
  WebSocket& ws() { return ws_; }
  Expected<bool> close(uint16_t code = 1000);
  json::Value stats_json() const;

 private:
  WebSocket ws_;
  int party_;
  WallClockShaper* shaper_;
};

}  // namespace platform
}  // namespace opencrab2
