// RPC_CMD_COPY_TENSOR must not write outside its destination tensor (CVE-2026-137567).
//
// The server checks a tensor's data against its buffer only when the tensor has elements, so a destination with none
// may point anywhere; before the fix copy_tensor checked only that the copy ends before the end of the destination's
// buffer, so a client that learned buffer addresses with BUFFER_GET_BASE could write the source's bytes anywhere below
// it. Here the destination claims one buffer and points into another below it: the server must refuse the copy and
// close the connection, instead of writing into the other buffer. A copy into a destination that holds it still works.
//
// The client speaks the wire protocol itself, as an attacker would, so the structs below mirror ggml-rpc.cpp's.

#include "ggml-backend.h"
#include "ggml-rpc.h"
#include "ggml.h"
#include "ggml-rpc/transport.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#pragma pack(push, 1)
struct wire_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    char     name[GGML_MAX_NAME];
    int32_t  use_count;
};
struct hello_req   { uint8_t conn_caps[RPC_CONN_CAPS_SIZE]; };
struct hello_rsp   { uint8_t major, minor, patch, flags; uint8_t conn_caps[RPC_CONN_CAPS_SIZE]; };
struct alloc_req   { uint32_t device; uint64_t size; };
struct alloc_rsp   { uint64_t remote_ptr; uint64_t remote_size; };
struct base_req    { uint64_t remote_ptr; };
struct base_rsp    { uint64_t base_ptr; };
struct clear_req   { uint64_t remote_ptr; uint8_t value; };
struct get_req     { wire_tensor tensor; uint64_t offset; uint64_t size; };
struct copy_req    { wire_tensor src; wire_tensor dst; };
struct copy_rsp    { uint8_t result; };
#pragma pack(pop)

// the command numbers of ggml-rpc.cpp's rpc_cmd
enum : uint8_t { CMD_ALLOC_BUFFER = 0, CMD_BUFFER_GET_BASE = 3, CMD_BUFFER_CLEAR = 5, CMD_GET_TENSOR = 8, CMD_COPY_TENSOR = 9, CMD_HELLO = 14 };

static constexpr uint64_t BUF_SIZE = 4096;
static constexpr uint32_t N_FLOATS = 64;

static bool send_cmd(const socket_ptr & sock, uint8_t cmd, const void * msg, uint64_t size) {
    return sock->send_data(&cmd, 1) && sock->send_data(&size, sizeof(size)) && sock->send_data(msg, size) && sock->flush();
}

static bool recv_rsp(const socket_ptr & sock, void * msg, uint64_t size) {
    uint64_t got = 0;
    return sock->recv_data(&got, sizeof(got)) && got == size && sock->recv_data(msg, size);
}

template <typename Req, typename Rsp>
static bool call(const socket_ptr & sock, uint8_t cmd, const Req & req, Rsp & rsp) {
    return send_cmd(sock, cmd, &req, sizeof(req)) && recv_rsp(sock, &rsp, sizeof(rsp));
}

static wire_tensor f32_tensor(uint64_t buffer, uint64_t data, uint32_t n) {
    wire_tensor t = {};
    t.id     = data;
    t.type   = GGML_TYPE_F32;
    t.buffer = buffer;
    t.ne[0]  = n;
    t.nb[0]  = sizeof(float);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        t.ne[i] = 1;
        t.nb[i] = n * sizeof(float);
    }
    t.data = data;
    return t;
}

struct session {
    socket_ptr sock;
    alloc_rsp  lo, hi;  // two buffers, lo the one at the lower address
    uint64_t   lo_base = 0, hi_base = 0;
};

// connects, says hello, allocates two buffers, fills the lower with 0 and the higher with 0xAB
static bool open_session(int port, session & s) {
    for (int k = 0; k < 100 && s.sock == nullptr; k++) {
        s.sock = socket_t::connect("127.0.0.1", port);
        if (s.sock == nullptr) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    if (s.sock == nullptr) {
        return false;
    }
    hello_req hreq = {};
    hello_rsp hrsp = {};
    if (!call(s.sock, CMD_HELLO, hreq, hrsp) || hrsp.major != RPC_PROTO_MAJOR_VERSION) {
        return false;
    }
    alloc_rsp a, b;
    base_rsp  ab, bb;
    if (!call(s.sock, CMD_ALLOC_BUFFER, alloc_req{0, BUF_SIZE}, a) || !call(s.sock, CMD_ALLOC_BUFFER, alloc_req{0, BUF_SIZE}, b) ||
        a.remote_ptr == 0 || b.remote_ptr == 0 ||
        !call(s.sock, CMD_BUFFER_GET_BASE, base_req{a.remote_ptr}, ab) || !call(s.sock, CMD_BUFFER_GET_BASE, base_req{b.remote_ptr}, bb)) {
        return false;
    }
    const bool a_lower = ab.base_ptr < bb.base_ptr;
    s.lo      = a_lower ? a : b;
    s.hi      = a_lower ? b : a;
    s.lo_base = a_lower ? ab.base_ptr : bb.base_ptr;
    s.hi_base = a_lower ? bb.base_ptr : ab.base_ptr;
    clear_req clo = {s.lo.remote_ptr, 0x00};
    clear_req chi = {s.hi.remote_ptr, 0xAB};
    return send_cmd(s.sock, CMD_BUFFER_CLEAR, &clo, sizeof(clo)) && send_cmd(s.sock, CMD_BUFFER_CLEAR, &chi, sizeof(chi));
}

// the first N_FLOATS floats of the lower buffer
static bool read_lo(session & s, std::vector<uint8_t> & out) {
    get_req req = {f32_tensor(s.lo.remote_ptr, s.lo_base, N_FLOATS), 0, N_FLOATS * sizeof(float)};
    out.assign(N_FLOATS * sizeof(float), 0);
    return send_cmd(s.sock, CMD_GET_TENSOR, &req, sizeof(req)) && recv_rsp(s.sock, out.data(), out.size());
}

int main() {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    ggml_backend_load_all();
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu == nullptr || !rpc_transport_init()) {
        fprintf(stderr, "no CPU device or transport\n");
        return 2;
    }
    // a free port: bind it, release it, and let the server take it
    int port = 0;
    for (int k = 0; k < 200 && port == 0; k++) {
        int p = 40000 + (int) ((std::chrono::steady_clock::now().time_since_epoch().count() + k) % 20000);
        if (socket_t::create_server("127.0.0.1", p) != nullptr) {
            port = p;
        }
    }
    if (port == 0) {
        fprintf(stderr, "no free port\n");
        return 2;
    }
    const std::string endpoint = "127.0.0.1:" + std::to_string(port);
    std::thread([endpoint, cpu]() mutable {
        ggml_backend_rpc_start_server(endpoint.c_str(), nullptr, 1, 1, &cpu);
    }).detach();

    int failures = 0;

    // control: a copy into a destination that holds it is done
    {
        session s;
        if (!open_session(port, s)) {
            fprintf(stderr, "could not set up the first session\n");
            return 2;
        }
        copy_req req = {f32_tensor(s.hi.remote_ptr, s.hi_base, N_FLOATS), f32_tensor(s.lo.remote_ptr, s.lo_base, N_FLOATS)};
        copy_rsp rsp = {};
        std::vector<uint8_t> lo;
        const bool ok = call(s.sock, CMD_COPY_TENSOR, req, rsp) && rsp.result == 1 && read_lo(s, lo) &&
                        std::all_of(lo.begin(), lo.end(), [](uint8_t v) { return v == 0xAB; });
        printf("copy into a destination that holds it: %s\n", ok ? "ok" : "FAILED");
        failures += ok ? 0 : 1;
    }

    // a destination with no elements that claims the higher buffer but points into the lower one
    {
        session s;
        if (!open_session(port, s)) {
            fprintf(stderr, "could not set up the second session\n");
            return 2;
        }
        copy_req req = {f32_tensor(s.hi.remote_ptr, s.hi_base, N_FLOATS), f32_tensor(s.hi.remote_ptr, s.lo_base, 0)};
        copy_rsp rsp = {};
        const bool done = call(s.sock, CMD_COPY_TENSOR, req, rsp);
        std::vector<uint8_t> lo;
        const bool wrote = done && read_lo(s, lo) && std::any_of(lo.begin(), lo.end(), [](uint8_t v) { return v != 0; });
        printf("copy into a zero-element destination outside its buffer: %s%s: %s\n",
               done ? "done" : "refused (connection closed)", wrote ? ", wrote into another buffer" : "",
               done ? "FAILED" : "ok");
        failures += done ? 1 : 0;
    }

    printf("%s\n", failures == 0 ? "ok" : "FAILED");
    // the server thread runs until the process ends
    fflush(stdout);
    std::_Exit(failures == 0 ? 0 : 1);
}
