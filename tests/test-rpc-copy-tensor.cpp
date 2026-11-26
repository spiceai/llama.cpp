// The RPC server's commands that read or write a client-given region of a tensor must keep it inside the tensor and
// its buffer, whatever the client sends.
//
// - COPY_TENSOR (CVE-2026-137567): the server checks a tensor's data against its buffer only when the tensor has
//   elements, so a destination with none may point anywhere; the copy must fit in the destination itself.
// - SET_TENSOR, SET_TENSOR_MULTI, SET_TENSOR_HASH and GET_TENSOR checked `data + offset` against the buffer, a sum
//   that wraps: an offset near 2^64 reached bytes before the tensor. A region in the buffer but past the tensor passed
//   the check and then failed ggml_backend_tensor_set/get's assert, which ended the server.
// - MEMSET_TENSOR, SET_TENSOR_2D and GET_TENSOR_2D already checked both bounds without wrapping; they are covered so
//   that they keep doing so.
//
// Each case runs against its own server, in a child process, so that one the request ends is told apart from one that
// refuses it (and closes the connection) and lives on. The client speaks the wire protocol itself, as an attacker
// would, so the structs below mirror ggml-rpc.cpp's.

#include "ggml-backend.h"
#include "ggml-rpc.h"
#include "ggml.h"
#include "ggml-rpc/transport.h"

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

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
struct memset_req  { wire_tensor tensor; uint64_t offset; uint64_t size; uint8_t value; };
struct hash_req    { wire_tensor tensor; uint64_t offset; uint64_t hash; };
struct hash_rsp    { uint8_t result; };
struct get_req     { wire_tensor tensor; uint64_t offset; uint64_t size; };
struct get_2d_req  { wire_tensor tensor; uint64_t offset; uint64_t size; uint64_t n_copies; uint64_t stride; };
struct copy_req    { wire_tensor src; wire_tensor dst; };
struct copy_rsp    { uint8_t result; };
#pragma pack(pop)

// the command numbers of ggml-rpc.cpp's rpc_cmd
enum : uint8_t {
    CMD_ALLOC_BUFFER = 0, CMD_BUFFER_GET_BASE = 3, CMD_BUFFER_CLEAR = 5, CMD_SET_TENSOR = 6, CMD_SET_TENSOR_HASH = 7,
    CMD_GET_TENSOR = 8, CMD_COPY_TENSOR = 9, CMD_HELLO = 14, CMD_MEMSET_TENSOR = 17, CMD_SET_TENSOR_2D = 18,
    CMD_GET_TENSOR_2D = 19, CMD_SET_TENSOR_MULTI = 23,
};

static constexpr uint64_t BUF_SIZE   = 4096;
static constexpr uint32_t N_FLOATS   = 64;
static constexpr uint64_t T_BYTES    = N_FLOATS * sizeof(float);  // 256
static constexpr uint64_t CACHE_HASH = 0x5eed5eed5eed5eedULL;

static bool send_all(const socket_ptr & sock, const void * msg, uint64_t size) {
    return sock->send_data(msg, size);
}

static bool send_cmd(const socket_ptr & sock, uint8_t cmd, const void * msg, uint64_t size) {
    return sock->send_data(&cmd, 1) && sock->send_data(&size, sizeof(size)) && send_all(sock, msg, size) && sock->flush();
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

static socket_ptr connect_to(int port) {
    for (int k = 0; k < 100; k++) {
        socket_ptr sock = socket_t::connect("127.0.0.1", port);
        if (sock != nullptr) {
            hello_req hreq = {};
            hello_rsp hrsp = {};
            if (call(sock, CMD_HELLO, hreq, hrsp) && hrsp.major == RPC_PROTO_MAJOR_VERSION) {
                return sock;
            }
            return nullptr;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return nullptr;
}

// connects, allocates two buffers, fills the lower with 0x00 and the higher with 0xAB
static bool open_session(int port, session & s) {
    s.sock = connect_to(port);
    if (s.sock == nullptr) {
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

// the lower buffer's first `n` bytes, or false when the connection is gone
static bool read_lo(session & s, std::vector<uint8_t> & out, uint64_t n = T_BYTES) {
    get_req req = {f32_tensor(s.lo.remote_ptr, s.lo_base, (uint32_t) (n / sizeof(float))), 0, n};
    out.assign(n, 0);
    return send_cmd(s.sock, CMD_GET_TENSOR, &req, sizeof(req)) && recv_rsp(s.sock, out.data(), out.size());
}

static bool all_are(const std::vector<uint8_t> & v, uint8_t x) {
    return std::all_of(v.begin(), v.end(), [x](uint8_t b) { return b == x; });
}

// SET_TENSOR's request: | rpc_tensor | cache_flag | offset | data |
static std::vector<uint8_t> set_tensor_msg(const wire_tensor & t, uint64_t offset, uint64_t size, uint8_t value) {
    std::vector<uint8_t> msg(sizeof(t) + 1 + sizeof(offset) + size, value);
    memcpy(msg.data(), &t, sizeof(t));
    msg[sizeof(t)] = 0;
    memcpy(msg.data() + sizeof(t) + 1, &offset, sizeof(offset));
    return msg;
}

enum class outcome { done, refused, server_ended, setup_failed };

static const char * outcome_name(outcome o) {
    switch (o) {
        case outcome::done:         return "done";
        case outcome::refused:      return "refused (connection closed)";
        case outcome::server_ended: return "the server ended";
        default:                    return "setup failed";
    }
}

static int free_port() {
    for (int k = 0; k < 200; k++) {
        int p = 40000 + (int) ((std::chrono::steady_clock::now().time_since_epoch().count() / 1000 + k * 7919) % 20000);
        if (socket_t::create_server("127.0.0.1", p) != nullptr) {
            return p;
        }
    }
    return 0;
}

static std::string g_cache_dir;

// runs `body` against a fresh server in a child process; `body` returns whether the request was done (true) or the
// connection closed under it (false), and fills `note` with what it saw
static outcome run_case(const std::function<bool(session &, std::string &)> & body, std::string & note) {
    const int port = free_port();
    if (port == 0) {
        return outcome::setup_failed;
    }
    const pid_t pid = fork();
    if (pid == 0) {
        ggml_backend_load_all();
        ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        const std::string endpoint = "127.0.0.1:" + std::to_string(port);
        ggml_backend_rpc_start_server(endpoint.c_str(), g_cache_dir.c_str(), 1, 1, &cpu);
        _exit(3);
    }
    session s;
    outcome result = outcome::setup_failed;
    if (open_session(port, s)) {
        const bool done = body(s, note);
        s.sock.reset();
        if (done) {
            result = outcome::done;
        } else {
            // a server that refused the request serves the next client; one that ended does not
            result = connect_to(port) != nullptr ? outcome::refused : outcome::server_ended;
        }
    }
    kill(pid, SIGKILL);
    waitpid(pid, nullptr, 0);
    return result;
}

static int failures = 0;

static void expect(const char * what, outcome got, outcome want, const std::string & note) {
    const bool ok = got == want;
    printf("%s: %s%s%s: %s\n", what, outcome_name(got), note.empty() ? "" : ", ", note.c_str(), ok ? "ok" : "FAILED");
    failures += ok ? 0 : 1;
}

int main() {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    if (!rpc_transport_init()) {
        fprintf(stderr, "no transport\n");
        return 2;
    }
    // SET_TENSOR_HASH reads the data from the server's cache directory, by the name of its hash
    char tmpl[] = "/tmp/test-rpc-region-XXXXXX";
    if (mkdtemp(tmpl) == nullptr) {
        fprintf(stderr, "no temporary directory\n");
        return 2;
    }
    g_cache_dir = tmpl;
    char hash_name[17];
    snprintf(hash_name, sizeof(hash_name), "%016llx", (unsigned long long) CACHE_HASH);
    {
        std::ofstream f(g_cache_dir + "/" + hash_name, std::ios::binary);
        std::vector<char> data(T_BYTES, (char) 0xCD);
        f.write(data.data(), (std::streamsize) data.size());
    }

    const uint64_t WRAP = 0 - T_BYTES;  // an offset that wraps data + offset back by one tensor
    std::string note;

    // controls: requests inside the tensor are done
    note.clear();
    expect("COPY_TENSOR into a destination that holds it", run_case([](session & s, std::string & n) {
        copy_req req = {f32_tensor(s.hi.remote_ptr, s.hi_base, N_FLOATS), f32_tensor(s.lo.remote_ptr, s.lo_base, N_FLOATS)};
        copy_rsp rsp = {};
        std::vector<uint8_t> lo;
        if (!call(s.sock, CMD_COPY_TENSOR, req, rsp) || !read_lo(s, lo)) return false;
        n = all_are(lo, 0xAB) ? "copied" : "wrong bytes";
        return true;
    }, note), outcome::done, note);

    note.clear();
    expect("SET_TENSOR inside the tensor", run_case([](session & s, std::string & n) {
        auto msg = set_tensor_msg(f32_tensor(s.lo.remote_ptr, s.lo_base, N_FLOATS), 0, T_BYTES, 0xCD);
        std::vector<uint8_t> lo;
        if (!send_cmd(s.sock, CMD_SET_TENSOR, msg.data(), msg.size()) || !read_lo(s, lo)) return false;
        n = all_are(lo, 0xCD) ? "written" : "wrong bytes";
        return true;
    }, note), outcome::done, note);

    // COPY_TENSOR into a zero-element destination that claims the higher buffer but points into the lower one
    note.clear();
    expect("COPY_TENSOR into a zero-element destination outside its buffer", run_case([](session & s, std::string & n) {
        copy_req req = {f32_tensor(s.hi.remote_ptr, s.hi_base, N_FLOATS), f32_tensor(s.hi.remote_ptr, s.lo_base, 0)};
        copy_rsp rsp = {};
        std::vector<uint8_t> lo;
        if (!call(s.sock, CMD_COPY_TENSOR, req, rsp)) return false;
        n = read_lo(s, lo) && !all_are(lo, 0x00) ? "wrote into another buffer" : "";
        return true;
    }, note), outcome::refused, note);

    // the commands whose `data + offset` wrapped: a tensor one tensor into the buffer, an offset back by one tensor
    note.clear();
    expect("SET_TENSOR with an offset that wraps before the tensor", run_case([WRAP](session & s, std::string & n) {
        auto msg = set_tensor_msg(f32_tensor(s.lo.remote_ptr, s.lo_base + T_BYTES, N_FLOATS), WRAP, T_BYTES, 0xCD);
        std::vector<uint8_t> lo;
        if (!send_cmd(s.sock, CMD_SET_TENSOR, msg.data(), msg.size()) || !read_lo(s, lo)) return false;
        n = all_are(lo, 0xCD) ? "wrote before the tensor" : "";
        return true;
    }, note), outcome::refused, note);

    note.clear();
    expect("SET_TENSOR_MULTI with an offset that wraps before the tensor", run_case([WRAP](session & s, std::string & n) {
        auto one = set_tensor_msg(f32_tensor(s.lo.remote_ptr, s.lo_base + T_BYTES, N_FLOATS), WRAP, T_BYTES, 0xCD);
        std::vector<uint8_t> msg(sizeof(uint32_t) + sizeof(uint64_t) + one.size());
        const uint32_t n_writes = 1;
        const uint64_t size     = one.size();
        memcpy(msg.data(), &n_writes, sizeof(n_writes));
        memcpy(msg.data() + sizeof(n_writes), &size, sizeof(size));
        memcpy(msg.data() + sizeof(n_writes) + sizeof(size), one.data(), one.size());
        std::vector<uint8_t> lo;
        if (!send_cmd(s.sock, CMD_SET_TENSOR_MULTI, msg.data(), msg.size()) || !read_lo(s, lo)) return false;
        n = all_are(lo, 0xCD) ? "wrote before the tensor" : "";
        return true;
    }, note), outcome::refused, note);

    note.clear();
    expect("SET_TENSOR_HASH with an offset that wraps before the tensor", run_case([WRAP](session & s, std::string & n) {
        hash_req req = {f32_tensor(s.lo.remote_ptr, s.lo_base + T_BYTES, N_FLOATS), WRAP, CACHE_HASH};
        hash_rsp rsp = {};
        std::vector<uint8_t> lo;
        if (!call(s.sock, CMD_SET_TENSOR_HASH, req, rsp) || !read_lo(s, lo)) return false;
        n = rsp.result == 1 && all_are(lo, 0xCD) ? "wrote before the tensor" : "cache miss?";
        return true;
    }, note), outcome::refused, note);

    note.clear();
    expect("GET_TENSOR with an offset that wraps before the tensor", run_case([WRAP](session & s, std::string & n) {
        get_req req = {f32_tensor(s.lo.remote_ptr, s.lo_base + T_BYTES, N_FLOATS), WRAP, T_BYTES};
        std::vector<uint8_t> out(T_BYTES);
        if (!send_cmd(s.sock, CMD_GET_TENSOR, &req, sizeof(req)) || !recv_rsp(s.sock, out.data(), out.size())) return false;
        n = "read before the tensor";
        return true;
    }, note), outcome::refused, note);

    // in the buffer but past the tensor: ggml_backend_tensor_set/get assert on it, which must not be reached
    const char * past[] = {"SET_TENSOR", "SET_TENSOR_MULTI", "SET_TENSOR_HASH", "GET_TENSOR"};
    for (int k = 0; k < 4; k++) {
        note.clear();
        const std::string what = std::string(past[k]) + " past the end of the tensor, inside its buffer";
        expect(what.c_str(), run_case([k](session & s, std::string &) {
            const wire_tensor t = f32_tensor(s.lo.remote_ptr, s.lo_base, N_FLOATS);
            std::vector<uint8_t> lo;
            if (k == 0) {
                auto msg = set_tensor_msg(t, T_BYTES, 16, 0xCD);
                return send_cmd(s.sock, CMD_SET_TENSOR, msg.data(), msg.size()) && read_lo(s, lo);
            }
            if (k == 1) {
                auto one = set_tensor_msg(t, T_BYTES, 16, 0xCD);
                std::vector<uint8_t> msg(sizeof(uint32_t) + sizeof(uint64_t) + one.size());
                const uint32_t n_writes = 1;
                const uint64_t size     = one.size();
                memcpy(msg.data(), &n_writes, sizeof(n_writes));
                memcpy(msg.data() + sizeof(n_writes), &size, sizeof(size));
                memcpy(msg.data() + sizeof(n_writes) + sizeof(size), one.data(), one.size());
                return send_cmd(s.sock, CMD_SET_TENSOR_MULTI, msg.data(), msg.size()) && read_lo(s, lo);
            }
            if (k == 2) {
                hash_req req = {t, 16, CACHE_HASH};
                hash_rsp rsp = {};
                return call(s.sock, CMD_SET_TENSOR_HASH, req, rsp) && read_lo(s, lo);
            }
            get_req req = {t, T_BYTES, 16};
            std::vector<uint8_t> out(16);
            return send_cmd(s.sock, CMD_GET_TENSOR, &req, sizeof(req)) && recv_rsp(s.sock, out.data(), out.size());
        }, note), outcome::refused, note);
    }

    // already checked without wrapping, on both bounds: past the tensor, and before it by a wrapping offset
    const char * checked[] = {"MEMSET_TENSOR", "SET_TENSOR_2D", "GET_TENSOR_2D"};
    for (int k = 0; k < 3; k++) {
        for (int wrap = 0; wrap < 2; wrap++) {
            note.clear();
            const std::string what = std::string(checked[k]) + (wrap ? " with an offset that wraps before the tensor" : " past the end of the tensor");
            expect(what.c_str(), run_case([k, wrap, WRAP](session & s, std::string &) {
                const wire_tensor t = f32_tensor(s.lo.remote_ptr, s.lo_base + (wrap ? T_BYTES : 0), N_FLOATS);
                const uint64_t offset = wrap ? WRAP : T_BYTES;
                const uint64_t size   = wrap ? T_BYTES : 16;
                std::vector<uint8_t> lo;
                if (k == 0) {
                    memset_req req = {t, offset, size, 0xCD};
                    return send_cmd(s.sock, CMD_MEMSET_TENSOR, &req, sizeof(req)) && read_lo(s, lo);
                }
                if (k == 1) {
                    std::vector<uint8_t> msg(sizeof(t) + 4 * sizeof(uint64_t) + size, 0xCD);
                    const uint64_t header[4] = {offset, size, 1, 0};
                    memcpy(msg.data(), &t, sizeof(t));
                    memcpy(msg.data() + sizeof(t), header, sizeof(header));
                    return send_cmd(s.sock, CMD_SET_TENSOR_2D, msg.data(), msg.size()) && read_lo(s, lo);
                }
                get_2d_req req = {t, offset, size, 1, 0};
                std::vector<uint8_t> out(size);
                return send_cmd(s.sock, CMD_GET_TENSOR_2D, &req, sizeof(req)) && recv_rsp(s.sock, out.data(), out.size());
            }, note), outcome::refused, note);
        }
    }

    std::remove((g_cache_dir + "/" + hash_name).c_str());
    rmdir(g_cache_dir.c_str());
    printf("%s\n", failures == 0 ? "ok" : "FAILED");
    return failures == 0 ? 0 : 1;
}
