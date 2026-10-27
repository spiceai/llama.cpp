// The TCP transport under the conditions the mesh comm puts it in:
// - a connection shut down while another thread is blocked sending on it (rpc_comm_async::release ends a mesh link the
//   exchange thread may be sending on) makes the send fail, and must not end the process with SIGPIPE;
// - accept(timeout) and set_recv_timeout bound the waits of the mesh's initialization.

#include "ggml-rpc/transport.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <unistd.h>
#include <vector>

static socket_ptr listen_on_free_port(int & port) {
    for (int k = 0; k < 200; k++) {
        port = 40000 + (int) ((getpid() + k) % 20000);
        socket_ptr srv = socket_t::create_server("127.0.0.1", port);
        if (srv != nullptr) {
            return srv;
        }
    }
    return nullptr;
}

static double elapsed_ms(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int main() {
    // one line at a time, so a run that dies still shows how far it got
    setvbuf(stdout, nullptr, _IOLBF, 0);
    if (!rpc_transport_init()) {
        fprintf(stderr, "the transport did not initialize\n");
        return 2;
    }
    int port = 0;
    socket_ptr srv = listen_on_free_port(port);
    if (srv == nullptr) {
        fprintf(stderr, "no free port to listen on\n");
        return 2;
    }
    int failures = 0;

    // nothing connects: accept(timeout) returns nullptr once the time is up
    {
        const auto t0 = std::chrono::steady_clock::now();
        socket_ptr none = srv->accept(200);
        const double ms = elapsed_ms(t0);
        const bool ok = none == nullptr && ms >= 150 && ms < 5000;
        printf("accept(200 ms) with no connection: %s after %.0f ms: %s\n", none == nullptr ? "nullptr" : "a socket", ms,
               ok ? "ok" : "FAILED");
        failures += ok ? 0 : 1;
    }

    socket_ptr a = socket_t::connect("127.0.0.1", port);
    socket_ptr b = a != nullptr ? srv->accept(5000) : nullptr;
    if (a == nullptr || b == nullptr) {
        fprintf(stderr, "could not connect over loopback on port %d\n", port);
        return 2;
    }

    // the peer sends nothing: a receive with a 200 ms timeout fails once the time is up
    {
        uint32_t word = 0;
        const auto t0 = std::chrono::steady_clock::now();
        const bool set = a->set_recv_timeout(200);
        const bool got = set && a->recv_data(&word, sizeof(word));
        const double ms = elapsed_ms(t0);
        const bool ok = set && !got && ms >= 150 && ms < 5000 && a->set_recv_timeout(0);
        printf("recv with a 200 ms timeout and nothing sent: %s after %.0f ms: %s\n", got ? "returned data" : "failed", ms,
               ok ? "ok" : "FAILED");
        failures += ok ? 0 : 1;
    }

    // the peer reads nothing, so the send blocks once the socket buffers are full; shutting the socket down from this
    // thread must make it fail rather than raise SIGPIPE
    {
        std::vector<char> data(64u << 20, 1);
        std::atomic<int> sent{-1};
        std::thread sender([&] { sent = a->send_data(data.data(), data.size()) ? 1 : 0; });
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        a->shutdown();
        sender.join();
        const bool ok = sent == 0;
        printf("send blocked on a full buffer, then the socket shut down: send_data returned %s: %s\n",
               sent == 1 ? "true" : "false", ok ? "ok" : "FAILED");
        failures += ok ? 0 : 1;
    }

    if (failures != 0) {
        fprintf(stderr, "%d failures\n", failures);
        return 1;
    }
    return 0;
}
