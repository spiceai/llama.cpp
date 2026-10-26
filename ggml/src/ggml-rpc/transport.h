#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

struct socket_t;
typedef std::shared_ptr<socket_t> socket_ptr;

static constexpr size_t MAX_CHUNK_SIZE = 1024ull * 1024ull * 1024ull; // 1 GiB
static constexpr size_t RPC_CONN_CAPS_SIZE = 24;

struct socket_t {
    ~socket_t();

    bool send_data(const void * data, size_t size);
    // Like send_data, but over RDMA it returns once the message is posted: its completion is waited for when this
    // connection next needs the send queue or buffer, or by its next send_data, which also reports a failure of it.
    // The data may change as soon as it returns. Elsewhere it is send_data.
    bool send_data_deferred(const void * data, size_t size);
    bool recv_data(void * data, size_t size);
    // Must be called at every message boundary: the RDMA transport coalesces
    // writes into fixed-size frames and posts the trailing partial frame only
    // here. No-op on TCP.
    bool flush();
    // Ends the connection in both directions, so that a send or receive another thread is blocked in returns false.
    // The socket stays open until it is destroyed.
    void shutdown();

    socket_ptr accept();

    void get_caps(uint8_t * local_caps);
    void update_caps(const uint8_t * remote_caps);
    const char * transport_name() const;
    // Waits up to timeout_ms for the peer to close the connection, without reading from it; true once it has.
    // Only where detects_peer_close() (Linux, POLLRDHUP); elsewhere this sleeps and returns false.
    bool wait_peer_closed(int timeout_ms);
    static bool detects_peer_close();

    // backlog: how many connections may wait for accept()
    static socket_ptr create_server(const char * host, int port, int backlog = 1);
    static socket_ptr connect(const char * host, int port);

private:
    struct impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
void rpc_transport_shutdown();
