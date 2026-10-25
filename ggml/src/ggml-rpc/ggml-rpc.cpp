#include "ggml-rpc.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "transport.h"
#include "log.h"

#include <array>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <optional>
#include <string>
#include <thread>
#include <vector>
#include <queue>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <atomic>
#if !defined(_WIN32)
#  include <arpa/inet.h>
#  include <ifaddrs.h>
#  include <net/if.h>
#  include <netinet/in.h>
#endif

namespace fs = std::filesystem;

// macro for nicer error messages on server crash
#define RPC_STATUS_ASSERT(x) if (!(x)) GGML_ABORT("Remote RPC server crashed or returned malformed response")

// all RPC structures must be packed
#pragma pack(push, 1)
// ggml_tensor is serialized into rpc_tensor
struct rpc_tensor {
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
    char name[GGML_MAX_NAME];

    int32_t use_count;
};

static_assert(sizeof(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

// RPC commands
enum rpc_cmd {
    RPC_CMD_ALLOC_BUFFER = 0,
    RPC_CMD_GET_ALIGNMENT,
    RPC_CMD_GET_MAX_SIZE,
    RPC_CMD_BUFFER_GET_BASE,
    RPC_CMD_FREE_BUFFER,
    RPC_CMD_BUFFER_CLEAR,
    RPC_CMD_SET_TENSOR,
    RPC_CMD_SET_TENSOR_HASH,
    RPC_CMD_GET_TENSOR,
    RPC_CMD_COPY_TENSOR,
    RPC_CMD_GRAPH_COMPUTE,
    RPC_CMD_GET_DEVICE_MEMORY,
    RPC_CMD_INIT_TENSOR,
    RPC_CMD_GET_ALLOC_SIZE,
    RPC_CMD_HELLO,
    RPC_CMD_DEVICE_COUNT,
    RPC_CMD_GRAPH_RECOMPUTE,
    RPC_CMD_MEMSET_TENSOR,
    RPC_CMD_SET_TENSOR_2D,
    RPC_CMD_GET_TENSOR_2D,
    RPC_CMD_COMM_INIT,
    RPC_CMD_COMM_ALLREDUCE,
    RPC_CMD_COMM_FREE,
    RPC_CMD_NONE,
    RPC_CMD_COUNT,
};

static_assert(RPC_CMD_HELLO == 14, "RPC_CMD_HELLO must be always 14");

static const char * rpc_cmd_name(enum rpc_cmd cmd) {
    switch (cmd) {
        case RPC_CMD_ALLOC_BUFFER:      return "ALLOC_BUFFER";
        case RPC_CMD_GET_ALIGNMENT:     return "GET_ALIGNMENT";
        case RPC_CMD_GET_MAX_SIZE:      return "GET_MAX_SIZE";
        case RPC_CMD_BUFFER_GET_BASE:   return "BUFFER_GET_BASE";
        case RPC_CMD_FREE_BUFFER:       return "FREE_BUFFER";
        case RPC_CMD_BUFFER_CLEAR:      return "BUFFER_CLEAR";
        case RPC_CMD_SET_TENSOR:        return "SET_TENSOR";
        case RPC_CMD_SET_TENSOR_HASH:   return "SET_TENSOR_HASH";
        case RPC_CMD_GET_TENSOR:        return "GET_TENSOR";
        case RPC_CMD_COPY_TENSOR:       return "COPY_TENSOR";
        case RPC_CMD_GRAPH_COMPUTE:     return "GRAPH_COMPUTE";
        case RPC_CMD_GET_DEVICE_MEMORY: return "GET_DEVICE_MEMORY";
        case RPC_CMD_INIT_TENSOR:       return "INIT_TENSOR";
        case RPC_CMD_GET_ALLOC_SIZE:    return "GET_ALLOC_SIZE";
        case RPC_CMD_HELLO:             return "HELLO";
        case RPC_CMD_DEVICE_COUNT:      return "DEVICE_COUNT";
        case RPC_CMD_GRAPH_RECOMPUTE:   return "GRAPH_RECOMPUTE";
        case RPC_CMD_MEMSET_TENSOR:     return "MEMSET_TENSOR";
        case RPC_CMD_SET_TENSOR_2D:     return "SET_TENSOR_2D";
        case RPC_CMD_GET_TENSOR_2D:     return "GET_TENSOR_2D";
        case RPC_CMD_COMM_INIT:         return "COMM_INIT";
        case RPC_CMD_COMM_ALLREDUCE:    return "COMM_ALLREDUCE";
        case RPC_CMD_COMM_FREE:         return "COMM_FREE";
        case RPC_CMD_NONE:              return "NONE";
        default:                        return "UNKNOWN";
    }
}

// Try RPC_CMD_SET_TENSOR_HASH first when data size is larger than this threshold
const size_t HASH_THRESHOLD = 10 * 1024 * 1024;

// Maximum number of graphs cached per device; client and server must use the same value
// so that both sides clear their caches at the same point in the message stream
const size_t GRAPH_CACHE_MAX = 1024;

struct rpc_msg_hello_req {
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

struct rpc_msg_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t padding;
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

struct rpc_msg_device_count_rsp {
    uint32_t device_count;
};

struct rpc_msg_get_alloc_size_req {
    uint32_t   device;
    rpc_tensor tensor;
    rpc_tensor srcs[GGML_MAX_SRC];
};

struct rpc_msg_get_alloc_size_rsp {
    uint64_t alloc_size;
};

struct rpc_msg_init_tensor_req {
    rpc_tensor tensor;
};

struct rpc_msg_alloc_buffer_req {
    uint32_t device;
    uint64_t size;
};

struct rpc_msg_alloc_buffer_rsp {
    uint64_t remote_ptr;
    uint64_t remote_size;
};

struct rpc_msg_get_alignment_req {
    uint32_t device;
};

struct rpc_msg_get_alignment_rsp {
    uint64_t alignment;
};

struct rpc_msg_get_max_size_req {
    uint32_t device;
};

struct rpc_msg_get_max_size_rsp {
    uint64_t max_size;
};

struct rpc_msg_buffer_get_base_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_get_base_rsp {
    uint64_t base_ptr;
};

struct rpc_msg_free_buffer_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_clear_req {
    uint64_t remote_ptr;
    uint8_t value;
};

struct rpc_msg_memset_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint8_t value;
};

struct rpc_msg_set_tensor_hash_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t hash;
};

struct rpc_msg_set_tensor_hash_rsp {
    uint8_t result;
};

struct rpc_msg_get_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
};

struct rpc_msg_copy_tensor_req {
    rpc_tensor src;
    rpc_tensor dst;
};

struct rpc_msg_copy_tensor_rsp {
    uint8_t result;
};

struct rpc_msg_get_device_memory_req {
    uint32_t device;
};

struct rpc_msg_get_device_memory_rsp {
    uint64_t free_mem;
    uint64_t total_mem;
};

struct rpc_msg_graph_recompute_req {
    uint32_t device;
    uint64_t uid;
};

struct rpc_msg_get_tensor_2d_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint64_t n_copies;
    uint64_t stride;
};

// RPC_COMM_FLAG_*: how the client asks both ranks to reduce, so the two servers always agree
static constexpr uint32_t RPC_COMM_FLAG_NCCL     = 1u; // through NCCL when both servers can (GGML_RPC_COMM_NCCL=1)
static constexpr uint32_t RPC_COMM_FLAG_WIRE_F32 = 2u; // host path: every partial as F32 (GGML_RPC_COMM_WIRE_F32=1)
static constexpr uint32_t RPC_COMM_FLAG_MESH     = 4u; // two ranks through the mesh, as more always go (GGML_RPC_COMM_MESH=1)

// the most ranks one comm joins; more than two connect every pair of ranks directly (a full mesh)
static constexpr uint32_t RPC_COMM_MAX_WORLD = 8;

struct rpc_msg_comm_init_req {
    uint32_t device;
    uint32_t rank;
    uint32_t world;
    uint32_t port;      // two ranks: rank 0 listens on it; more: rank r listens on port + r
    char     host[64];  // two ranks, rank 1: rank 0's host
    uint32_t flags;     // RPC_COMM_FLAG_*, the same for every rank; sent only when one is set or world > 2
    char     hosts[RPC_COMM_MAX_WORLD][64]; // world > 2: hosts[j] is where this rank reaches rank j, for each j < rank
};

// a COMM_INIT with no RPC_COMM_FLAG_* set is upstream's request, without the flags, so that a client that sets no
// flag and a server without them (or the reverse) still agree on its size; the hosts follow only for world > 2
static constexpr size_t RPC_COMM_INIT_REQ_SIZE_NO_FLAGS = offsetof(rpc_msg_comm_init_req, flags);
static constexpr size_t RPC_COMM_INIT_REQ_SIZE_PAIR     = offsetof(rpc_msg_comm_init_req, hosts);
static_assert(RPC_COMM_INIT_REQ_SIZE_NO_FLAGS == 80, "COMM_INIT without flags must stay upstream's 80 bytes");

// world > 2: what precedes each partial on a mesh link, so that a rank about to sum another all-reduce's partial, or one
// of another size, fails instead
struct rpc_comm_part_hdr {
    uint64_t seq;    // the sender's all-reduce count, 1 for its first
    uint64_t nbytes; // the F32 bytes that follow
};
static_assert(sizeof(rpc_comm_part_hdr) % sizeof(float) == 0, "the partial after the header must stay F32-aligned");

// world > 2: partials up to this size fit every transport's buffering (TCP's send buffer, RDMA's 6 MiB receive ring), so
// a rank sends to every peer before receiving; larger ones go pair by pair
static constexpr size_t RPC_COMM_MESH_EAGER_BYTES = 1024 * 1024;

struct rpc_msg_comm_init_rsp {
    uint8_t ok;
};

struct rpc_msg_comm_allreduce_req {
    uint32_t   device;
    rpc_tensor tensor;
};

struct rpc_msg_comm_free_req {
    uint32_t device;
};

#pragma pack(pop)

// RPC data structures

static ggml_guid_t ggml_backend_rpc_guid() {
    static ggml_guid guid = {0x99, 0x68, 0x5b, 0x6c, 0xd2, 0x83, 0x3d, 0x24, 0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03};
    return &guid;
}

struct ggml_backend_rpc_device_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    std::string description;
};

struct ggml_backend_rpc_buffer_type_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    size_t      alignment;
    size_t      max_size;
};

class rpc_dispatcher;
struct ggml_backend_rpc_context {
    std::shared_ptr<rpc_dispatcher> dispatcher;
    std::string                     endpoint;
    uint32_t                        device;
    std::string                     name;
};

struct ggml_backend_rpc_buffer_context {
    std::shared_ptr<rpc_dispatcher>   dispatcher;
    void                            * base_ptr;
    uint64_t                          remote_ptr;
};

// RPC helper functions

// Computes FNV-1a hash of the data
static uint64_t fnv_hash(const uint8_t * data, size_t len, uint64_t hash = 0xcbf29ce484222325ULL) {
    const uint64_t fnv_prime = 0x100000001b3ULL;

    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    return hash;
}

static bool send_msg(socket_ptr sock, const void * msg, size_t msg_size) {
    if (!sock->send_data(&msg_size, sizeof(msg_size))) {
        return false;
    }
    if (!sock->send_data(msg, msg_size)) {
        return false;
    }
    return sock->flush();
}

static bool recv_msg(socket_ptr sock, void * msg, size_t msg_size) {
    uint64_t size;
    if (!sock->recv_data(&size, sizeof(size))) {
        return false;
    }
    if (size != msg_size) {
        return false;
    }
    return sock->recv_data(msg, msg_size);
}

// COMM_INIT comes in three sizes: with the hosts of a mesh, with the flags, or upstream's request without them
static bool recv_comm_init_req(socket_ptr sock, rpc_msg_comm_init_req & request) {
    uint64_t size;
    if (!sock->recv_data(&size, sizeof(size))) {
        return false;
    }
    if (size != sizeof(request) && size != RPC_COMM_INIT_REQ_SIZE_PAIR && size != RPC_COMM_INIT_REQ_SIZE_NO_FLAGS) {
        return false;
    }
    request = {};
    return sock->recv_data(&request, size);
}

static bool recv_msg(socket_ptr sock, std::vector<uint8_t> & input) {
    uint64_t size;
    if (!sock->recv_data(&size, sizeof(size))) {
        return false;
    }
    try {
        input.resize(size);
    } catch (const std::bad_alloc & e) {
        LOG_ERROR("Failed to allocate input buffer of size %" PRIu64 "\n", size);
        return false;
    }
    return sock->recv_data(input.data(), size);
}

static bool parse_endpoint(const std::string & endpoint, std::string & host, int & port) {
    size_t pos = endpoint.find(':');
    if (pos == std::string::npos) {
        return false;
    }
    host = endpoint.substr(0, pos);
    try {
        port = std::stoi(endpoint.substr(pos + 1));
    } catch (...) {
        return false;
    }
    return true;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// No response
static bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size) {
    uint8_t cmd_byte = cmd;
    if (!sock->send_data(&cmd_byte, sizeof(cmd_byte))) {
        return false;
    }
    if (!sock->send_data(&input_size, sizeof(input_size))) {
        return false;
    }
    if (!sock->send_data(input, input_size)) {
        return false;
    }
    return sock->flush();
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
static bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size, void * output, size_t output_size) {
    if (!send_rpc_cmd(sock, cmd, input, input_size)) {
        return false;
    }
    uint64_t out_size;
    if (!sock->recv_data(&out_size, sizeof(out_size))) {
        return false;
    }
    if (out_size != output_size) {
        return false;
    }
    if (!sock->recv_data(output, output_size)) {
        return false;
    }
    return true;
}

// RPC client-side implementation

// with busy spinning on, the dispatcher still blocks on its queue after this long without commands
static constexpr auto RPC_BUSY_SPIN_IDLE_TIME = std::chrono::milliseconds(100);

static inline void rpc_cpu_relax() {
#if defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
    __asm__ volatile("yield" ::: "memory");
#else
    std::this_thread::yield();
#endif
}

// Performs HELLO handshake with transport auto-negotiation.
// Advertises local capabilities via conn_caps; if the server responds with
// matching capabilities, the socket is upgraded transparently.
static bool negotiate_hello(const std::shared_ptr<socket_t> & sock) {
    rpc_msg_hello_req request = {};
    rpc_msg_hello_rsp response = {};

    sock->get_caps(request.conn_caps);

    bool status = send_rpc_cmd(sock, RPC_CMD_HELLO, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);

    if (response.major != RPC_PROTO_MAJOR_VERSION || response.minor > RPC_PROTO_MINOR_VERSION) {
        LOG_ERROR("RPC server version mismatch: %d.%d.%d\n",
                       response.major, response.minor, response.patch);
        return false;
    }

    sock->update_caps(response.conn_caps);
    LOG_DBG("[%s] handshake ok: server version %d.%d.%d, transport: %s\n",
            __func__, response.major, response.minor, response.patch, sock->transport_name());
    return true;
}

template <typename T>
class message_queue {
public:
    message_queue() {}

    bool push(const T &value) {
        std::unique_lock<std::mutex> lock(mutex);
        if (interrupted) {
            return false;
        }
        queue.push(value);
        cvar.notify_all();
        return true;
    }

    bool pop(T* out) {
        std::unique_lock<std::mutex> lock(mutex);
        cvar.wait(lock, [this] { return !queue.empty() || interrupted; });
        if (interrupted) {
            return false;
        }
        *out = queue.front();
        queue.pop();
        return true;
    }

    bool try_pop(T* out) {
        std::unique_lock<std::mutex> lock(mutex);
        if (interrupted || queue.empty()) {
            return false;
        }
        *out = queue.front();
        queue.pop();
        return true;
    }

    void interrupt() {
        std::unique_lock<std::mutex> lock(mutex);
        interrupted = true;
        lock.unlock();
        cvar.notify_all();
    }

private:
    bool interrupted = false;
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cvar;
};

class rpc_dispatcher {
public:
    rpc_dispatcher() {
    }

    void send(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size);
    void send(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size, void * output, size_t output_size);
    void send_async(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size);
    void send_async(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size, void * output, size_t output_size);

    ggml_backend_event_t event_new(ggml_backend_dev_t dev);
    void event_free(ggml_backend_event_t event);
    void event_synchronize(ggml_backend_event_t event);
    void event_record(ggml_backend_event_t event);
    void synchronize();
    void busy_spin_acquire();
    void busy_spin_release();
    void graph_compute(uint32_t device, const ggml_cgraph * cgraph);

    void start(const std::string & endpoint);
    void work();

    ~rpc_dispatcher();

private:
    struct rpc_msg {
        rpc_cmd                       cmd;
        std::shared_ptr<const void>   input;
        size_t                        input_size;
        void                        * output;
        size_t                        output_size;
        std::promise<void>            completion;
    };
    using rpc_msg_ptr   = std::shared_ptr<rpc_msg>;
    using rpc_msg_queue = message_queue<rpc_msg_ptr>;
    struct rpc_event {
        rpc_msg_ptr              msg;
        std::shared_future<void> sf;
    };
    std::mutex graph_mutex;
    std::unordered_map<uint32_t, std::unordered_set<uint64_t>> graph_uids;
    rpc_msg_queue    queue;
    socket_ptr       sock;
    std::atomic_uint busy_spin_users = 0;
    std::atomic_bool running;
    std::thread      thread;
};

static void rpc_dispatcher_trampoline(rpc_dispatcher * dispatcher)
{
    dispatcher->work();
}

void rpc_dispatcher::send(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size) {
    auto msg = std::make_shared<rpc_msg>();
    msg->cmd = cmd;
    msg->input = input;
    msg->input_size = input_size;
    msg->output = nullptr;
    msg->output_size = 0;
    GGML_ASSERT(queue.push(msg));
    auto future = msg->completion.get_future();
    future.wait();
}

void rpc_dispatcher::send_async(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size) {
    auto msg = std::make_shared<rpc_msg>();
    msg->cmd = cmd;
    msg->input = input;
    msg->input_size = input_size;
    msg->output = nullptr;
    msg->output_size = 0;
    GGML_ASSERT(queue.push(msg));
}

void rpc_dispatcher::send(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size, void * output, size_t output_size) {
    auto msg = std::make_shared<rpc_msg>();
    msg->cmd = cmd;
    msg->input = input;
    msg->input_size = input_size;
    msg->output = output;
    msg->output_size = output_size;
    GGML_ASSERT(queue.push(msg));
    auto future = msg->completion.get_future();
    future.wait();
}

void rpc_dispatcher::send_async(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size, void * output, size_t output_size) {
    auto msg = std::make_shared<rpc_msg>();
    msg->cmd = cmd;
    msg->input = input;
    msg->input_size = input_size;
    msg->output = output;
    msg->output_size = output_size;
    GGML_ASSERT(queue.push(msg));
}

ggml_backend_event_t rpc_dispatcher::event_new(ggml_backend_dev_t dev) {
    rpc_event * ev = new rpc_event;
    ev->msg = std::make_shared<rpc_msg>();
    ev->msg->cmd = RPC_CMD_NONE;
    ev->sf = ev->msg->completion.get_future().share();
    GGML_ASSERT(queue.push(ev->msg));
    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ ev,
    };
}

void rpc_dispatcher::event_free(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    delete ev;
}

void rpc_dispatcher::event_synchronize(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->sf.wait();
}

void rpc_dispatcher::event_record(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->msg = std::make_shared<rpc_msg>();
    ev->msg->cmd = RPC_CMD_NONE;
    ev->sf = ev->msg->completion.get_future().share();
    GGML_ASSERT(queue.push(ev->msg));
}

void rpc_dispatcher::synchronize() {
    // to ensure all messages are processed, submit dummy message and wait for it to complete
    auto msg = std::make_shared<rpc_msg>();
    msg->cmd = RPC_CMD_NONE;
    GGML_ASSERT(queue.push(msg));
    msg->completion.get_future().wait();
}

void rpc_dispatcher::busy_spin_acquire() {
    busy_spin_users.fetch_add(1, std::memory_order_relaxed);
}

void rpc_dispatcher::busy_spin_release() {
    const unsigned previous = busy_spin_users.fetch_sub(1, std::memory_order_relaxed);
    GGML_ASSERT(previous > 0);
}

void rpc_dispatcher::start(const std::string & endpoint) {
    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        GGML_ABORT("Failed to parse endpoint: %s\n", endpoint.c_str());
    }
    if (!rpc_transport_init()) {
        GGML_ABORT("RPC transport initialization failed\n");
    }

    sock = socket_t::connect(host.c_str(), port);
    if (sock == nullptr) {
        GGML_ABORT("Failed to connect to %s\n", endpoint.c_str());
    }
    if (!negotiate_hello(sock)) {
        GGML_ABORT("RPC handshake failed for %s\n", endpoint.c_str());
    }
    LOG_DBG("[%s] connected to %s\n", __func__, endpoint.c_str());
    running = true;
    thread = std::thread(rpc_dispatcher_trampoline, this);
}

void rpc_dispatcher::work() {
    auto last_cmd = std::chrono::steady_clock::now();
    while (running) {
        rpc_msg_ptr msg_ptr;
        // spin only while commands keep coming, so an idle dispatcher does not keep a core busy
        const bool spin = busy_spin_users.load(std::memory_order_relaxed) != 0 &&
                          std::chrono::steady_clock::now() - last_cmd < RPC_BUSY_SPIN_IDLE_TIME;
        if (spin) {
            if (!queue.try_pop(&msg_ptr)) {
                rpc_cpu_relax();
                continue;
            }
        } else if (!queue.pop(&msg_ptr)) {
            break;
        }
        if (msg_ptr->cmd != RPC_CMD_NONE) {
            LOG_DBG2("[%s] %s (in: %s, out: %s)\n", __func__, rpc_cmd_name(msg_ptr->cmd),
                     rpc_log_format_size(msg_ptr->input_size).c_str(), rpc_log_format_size(msg_ptr->output_size).c_str());
            const bool trace = rpc_log_debug_level() >= 3;
            const auto t0 = std::chrono::steady_clock::now();
            if (msg_ptr->output) {
                bool status = send_rpc_cmd(sock, msg_ptr->cmd, msg_ptr->input.get(), msg_ptr->input_size, msg_ptr->output, msg_ptr->output_size);
                RPC_STATUS_ASSERT(status);
            } else {
                bool status = send_rpc_cmd(sock, msg_ptr->cmd, msg_ptr->input.get(), msg_ptr->input_size);
                RPC_STATUS_ASSERT(status);
            }
            if (trace) {
                const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - t0;
                LOG_DBG3("[%s] %s done in %.3f ms\n", __func__, rpc_cmd_name(msg_ptr->cmd), elapsed.count());
            }
        } else {
            LOG_DBG3("[%s] sync barrier\n", __func__);
        }
        msg_ptr->completion.set_value();
        last_cmd = std::chrono::steady_clock::now();
    }
}

rpc_dispatcher::~rpc_dispatcher() {
    running = false;
    queue.interrupt();
    sock = nullptr;
    if (thread.joinable()) {
        thread.join();
    }
}

static std::shared_ptr<rpc_dispatcher> get_dispatcher(const std::string & endpoint) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static std::unordered_map<std::string, std::weak_ptr<rpc_dispatcher>> dispatchers;

    auto it = dispatchers.find(endpoint);
    if (it != dispatchers.end()) {
        if (auto dispatcher = it->second.lock()) {
            return dispatcher;
        }
    }

    auto dispatcher = std::make_shared<rpc_dispatcher>();
    dispatcher->start(endpoint);
    dispatchers[endpoint] = dispatcher;
    return dispatcher;
}

static void ggml_backend_rpc_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_msg_free_buffer_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request->remote_ptr);
    ctx->dispatcher->send(RPC_CMD_FREE_BUFFER, request, sizeof(*request));
    delete ctx;
}

static void * ggml_backend_rpc_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    if (ctx->base_ptr != nullptr) {
        return ctx->base_ptr;
    }
    auto request = std::make_shared<rpc_msg_buffer_get_base_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, ctx->remote_ptr);
    rpc_msg_buffer_get_base_rsp response;
    ctx->dispatcher->send(RPC_CMD_BUFFER_GET_BASE, request, sizeof(*request), &response, sizeof(response));
    ctx->base_ptr = reinterpret_cast<void *>(response.base_ptr);
    return ctx->base_ptr;
}

static bool ggml_backend_buffer_is_rpc(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_rpc_buffer_free_buffer;
}

static rpc_tensor serialize_tensor(const ggml_tensor * tensor, const rpc_dispatcher * dispatcher = nullptr) {
    rpc_tensor result;
    if (!tensor) {
        memset(&result, 0, sizeof(result));
        return result;
    }

    result.id = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    if (tensor->buffer && ggml_backend_buffer_is_rpc(tensor->buffer)) {
        ggml_backend_buffer_t buffer = tensor->buffer;
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        // ref: https://github.com/ggml-org/llama.cpp/pull/26500
        if (ctx != nullptr && (dispatcher == nullptr || ctx->dispatcher.get() == dispatcher)) {
            result.buffer = ctx->remote_ptr;
            result.data = reinterpret_cast<uint64_t>(tensor->data);
        } else {
            result.buffer = 0;
            result.data = 0;
        }
    } else {
        result.buffer = 0;
        result.data   = 0;
    }
    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = tensor->nb[i];
    }
    result.op = tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;
    for (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;

    // Avoid sending uninitialized data over the wire
    memset(result.name, 0, sizeof(result.name));
    result.use_count = 0;

    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    return result;
}

static enum ggml_status ggml_backend_rpc_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;

    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    if (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
        LOG_DBG2("[%s] tensor: %s\n", __func__, rpc_log_format_tensor(tensor).c_str());
        auto request = std::make_shared<rpc_msg_init_tensor_req>();
        request->tensor = serialize_tensor(tensor);
        ctx->dispatcher->send(RPC_CMD_INIT_TENSOR, request, sizeof(*request));
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_rpc_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_msg_memset_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size   = size;
    request->value  = value;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, value: %u\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str(), value);
    ctx->dispatcher->send(RPC_CMD_MEMSET_TENSOR, request, sizeof(*request));
}

// input serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes)
static std::shared_ptr<uint8_t> serialize_set_tensor(const rpc_tensor & rpc_tensor, uint8_t cache_flag, uint64_t offset, const void * data, size_t size, size_t & input_size) {
    input_size = sizeof(rpc_tensor) + sizeof(cache_flag) + sizeof(offset) + size;
    uint8_t * input = new uint8_t[input_size]();
    uint8_t * p = input;
    memcpy(p, &rpc_tensor, sizeof(rpc_tensor)); p += sizeof(rpc_tensor);
    memcpy(p, &cache_flag, sizeof(cache_flag)); p += sizeof(cache_flag);
    memcpy(p, &offset,     sizeof(offset));     p += sizeof(offset);
    memcpy(p, data, size);
    return std::shared_ptr<uint8_t>(input, std::default_delete<uint8_t[]>());
}

// the hash cache is meant for weights, so that a model reload can skip re-sending them.
// compute-buffer inputs (the activations ggml_backend_sched copies between backends) must not
// take this path, otherwise with `rpc-server -c` every ubatch above the threshold is written
// to the cache directory and later served from there.
static bool rpc_use_hash_cache(const ggml_tensor * tensor, size_t size) {
    return size > HASH_THRESHOLD && tensor->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

// GGML_RPC_SET_ASYNC=1: a buffer write returns once it is queued; the request owns a copy of the data and the
// server handles commands in order, so a later graph compute or read on that server sees it. A write into a weight
// buffer still waits: model loading writes the whole model that way, and nothing else would bound the queued copies
static bool rpc_set_async(ggml_backend_buffer_t buffer) {
    static const bool v = [] {
        const char * e = std::getenv("GGML_RPC_SET_ASYNC");
        return e != nullptr && std::atoi(e) != 0;
    }();
    return v && ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

static void ggml_backend_rpc_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    if (rpc_use_hash_cache(tensor, size)) {
        auto request = std::make_shared<rpc_msg_set_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->offset = offset;
        request->hash = fnv_hash((const uint8_t*)data, size);
        rpc_msg_set_tensor_hash_rsp response;
        ctx->dispatcher->send(RPC_CMD_SET_TENSOR_HASH, request, sizeof(*request), &response, sizeof(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.result ? "hit" : "miss");
        if (response.result) {
            // the server has the same data, no need to send it
            return;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    size_t input_size;
    auto input = serialize_set_tensor(rpc_tensor, cache_flag, offset, data, size, input_size);
    if (rpc_set_async(buffer)) {
        ctx->dispatcher->send_async(RPC_CMD_SET_TENSOR, input, input_size);
    } else {
        ctx->dispatcher->send(RPC_CMD_SET_TENSOR, input, input_size);
    }
}

static void ggml_backend_rpc_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), offset, rpc_log_format_size(size).c_str(), n_copies, stride_tensor);
    // input serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    size_t input_size = sizeof(rpc_tensor) + 4*sizeof(uint64_t) + size*n_copies;
    uint8_t * input = new uint8_t[input_size]();
    uint8_t * dest = input;
    memcpy(dest, &rpc_tensor, sizeof(rpc_tensor));
    dest += sizeof(rpc_tensor);
    uint64_t header[4] = { offset, size, n_copies, stride_tensor };
    memcpy(dest, header, sizeof(header));
    dest += sizeof(header);
    for (size_t i = 0; i < n_copies; i++) {
        memcpy(dest + i*size, (const char *)data + i*stride_data, size);
    }
    std::shared_ptr<uint8_t> input_ptr(input, std::default_delete<uint8_t[]>());
    if (rpc_set_async(buffer)) {
        ctx->dispatcher->send_async(RPC_CMD_SET_TENSOR_2D, input_ptr, input_size);
    } else {
        ctx->dispatcher->send(RPC_CMD_SET_TENSOR_2D, input_ptr, input_size);
    }
}

static void ggml_backend_rpc_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), offset, rpc_log_format_size(size).c_str(), n_copies, stride_tensor);
    auto request = std::make_shared<rpc_msg_get_tensor_2d_req>();
    request->tensor   = serialize_tensor(tensor);
    request->offset   = offset;
    request->size     = size;
    request->n_copies = n_copies;
    request->stride   = stride_tensor;
    if (stride_data == size) {
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, sizeof(*request), data, size*n_copies);
    } else {
        std::vector<uint8_t> packed(size*n_copies);
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, sizeof(*request), packed.data(), packed.size());
        for (size_t i = 0; i < n_copies; i++) {
            memcpy((char *)data + i*stride_data, packed.data() + i*size, size);
        }
    }
}

static void ggml_backend_rpc_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    auto request = std::make_shared<rpc_msg_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size = size;
    ctx->dispatcher->send(RPC_CMD_GET_TENSOR, request, sizeof(*request), data, size);
}

static bool ggml_backend_rpc_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_rpc(src->buffer)) {
        // check if src and dst are on the same server
        ggml_backend_buffer_t src_buffer = src->buffer;
        ggml_backend_rpc_buffer_context * src_ctx = (ggml_backend_rpc_buffer_context *)src_buffer->context;
        ggml_backend_buffer_t dst_buffer = dst->buffer;
        ggml_backend_rpc_buffer_context * dst_ctx = (ggml_backend_rpc_buffer_context *)dst_buffer->context;
        if (src_ctx->dispatcher != dst_ctx->dispatcher) {
            LOG_DBG2("[%s] src and dst are on different servers, falling back to a host copy\n", __func__);
            return false;
        }
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());
        auto request = std::make_shared<rpc_msg_copy_tensor_req>();
        request->src = serialize_tensor(src);
        request->dst = serialize_tensor(dst);
        rpc_msg_copy_tensor_rsp response;
        ctx->dispatcher->send(RPC_CMD_COPY_TENSOR, request, sizeof(*request), &response, sizeof(response));
        return response.result;
    }
    return false;
}

static void ggml_backend_rpc_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_msg_buffer_clear_req>();
    request->remote_ptr = ctx->remote_ptr;
    request->value = value;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request->remote_ptr, request->value);
    ctx->dispatcher->send(RPC_CMD_BUFFER_CLEAR, request, sizeof(*request));
}

static ggml_backend_buffer_i ggml_backend_rpc_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_rpc_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_rpc_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_rpc_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_rpc_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_rpc_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_rpc_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_rpc_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_rpc_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_rpc_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_rpc_buffer_clear,
    /* .reset           = */ NULL,
};

static const char * ggml_backend_rpc_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_rpc_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    auto request = std::make_shared<rpc_msg_alloc_buffer_req>();
    request->device = buft_ctx->device;
    request->size = size;
    rpc_msg_alloc_buffer_rsp response;

    auto dispatcher = get_dispatcher(buft_ctx->endpoint);
    dispatcher->send(RPC_CMD_ALLOC_BUFFER, request, sizeof(*request), &response, sizeof(response));
    if (response.remote_ptr != 0) {
        LOG_DBG("[%s] endpoint: %s, device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_size(request->size).c_str(),
                response.remote_ptr, rpc_log_format_size(response.remote_size).c_str());
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft,
            ggml_backend_rpc_buffer_interface,
            new ggml_backend_rpc_buffer_context{dispatcher, nullptr, response.remote_ptr},
            response.remote_size);
        return buffer;
    } else {
        LOG_WARN("[%s] endpoint: %s, device: %u, size: %s -> failed\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_size(request->size).c_str());
        return nullptr;
    }
}

static size_t get_alignment(const std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    auto request = std::make_shared<rpc_msg_get_alignment_req>();
    request->device = device;
    rpc_msg_get_alignment_rsp response;
    dispatcher->send(RPC_CMD_GET_ALIGNMENT, request, sizeof(*request), &response, sizeof(response));
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, device, rpc_log_format_size(response.alignment).c_str());
    return response.alignment;
}

static size_t ggml_backend_rpc_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->alignment;
}

static size_t get_max_size(const std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    auto request = std::make_shared<rpc_msg_get_max_size_req>();
    request->device = device;
    rpc_msg_get_max_size_rsp response;
    dispatcher->send(RPC_CMD_GET_MAX_SIZE, request, sizeof(*request), &response, sizeof(response));
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, device, rpc_log_format_size(response.max_size).c_str());
    return response.max_size;
}

static size_t ggml_backend_rpc_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->max_size;
}

static size_t ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    // should we query the remote server for the actual size
    bool rpc_get = false;

    // See comments in init_tensor.
    rpc_get |= ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr);

    // [TAG_ALLOC_SIZE_EXPAND]
    // ops that may require additional memory for fleeting data on certain backends
    // ref: https://github.com/ggml-org/llama.cpp/pull/15966
    rpc_get |= ggml_op_alloc_size_may_expand(tensor->op);

    if (rpc_get) {
        ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;

        // the reported size must never be below ggml_nbytes: rpc_tensor stores nb[] as uint32_t,
        // so a stride over 4 GiB is truncated on the wire and the remote size comes back too small
        // TODO: change rpc_tensor nb to 64-bit int
        const size_t min_size = ggml_nbytes(tensor);

        // Cache key for calls to read the alloc_size.
        // We deliberately exclude src tensor dimensions from the key because:
        // 1. For CPU backends, alloc_size = ggml_nbytes(output) regardless of src shapes
        // 2. For GPU backends, the reservation graph uses max dimensions, so the
        //    cached value from reservation is always >= any subsequent request
        // 3. Including src dims causes cache misses per-ubatch (e.g. growing KV cache)
        //    which blocks the main thread behind in-flight GRAPH_COMPUTE commands
        struct alloc_size_cache_key {
            uint32_t device;
            uint32_t type;
            uint32_t op;
            int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
            uint32_t ne[GGML_MAX_DIMS];
            uint64_t nb[GGML_MAX_DIMS];
        };

        alloc_size_cache_key key = {};
        key.device = buft_ctx->device;
        key.type = tensor->type;
        key.op = tensor->op;
        memcpy(key.op_params, tensor->op_params, sizeof(key.op_params));
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            key.ne[i] = (uint32_t)tensor->ne[i];
            key.nb[i] = (uint64_t)tensor->nb[i];
        }

        uint64_t cache_hash = fnv_hash((const uint8_t *)&key, sizeof(key));
        cache_hash = fnv_hash((const uint8_t *)buft_ctx->endpoint.data(), buft_ctx->endpoint.size(), cache_hash);

        // alloc sizes are immutable for a given tensor configuration
        static std::mutex cache_mutex;
        static std::unordered_map<uint64_t, size_t> cache;

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            auto it = cache.find(cache_hash);
            if (it != cache.end()) {
                LOG_DBG3("[%s] cache hit: %s -> %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                         rpc_log_format_size(it->second).c_str());
                return std::max<size_t>(it->second, min_size);
            }
        }

        auto request = std::make_shared<rpc_msg_get_alloc_size_req>();
        request->device = buft_ctx->device;
        request->tensor = serialize_tensor(tensor);

        // .get_alloc_size could be a function of the tensor's srcs, so we must serialize them as well
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            request->srcs[i] = serialize_tensor(tensor->src[i]);
        }

        LOG_DBG2("[%s] cache miss: %s, querying server\n", __func__, rpc_log_format_tensor(tensor).c_str());
        rpc_msg_get_alloc_size_rsp response;
        auto dispatcher = get_dispatcher(buft_ctx->endpoint);
        dispatcher->send(RPC_CMD_GET_ALLOC_SIZE, request, sizeof(*request), &response, sizeof(response));

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            cache[cache_hash] = response.alloc_size;
        }

        LOG_DBG2("[%s] %s -> alloc_size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 rpc_log_format_size(response.alloc_size).c_str());

        return std::max<size_t>(response.alloc_size, min_size);
    }

    return ggml_nbytes(tensor);
}

static ggml_backend_buffer_type_i ggml_backend_rpc_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_rpc_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_rpc_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ NULL,
    /* .get_alignment       = */ ggml_backend_rpc_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_rpc_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_rpc_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ NULL,
    /* .is_host             = */ NULL,
};

static const char * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    return rpc_ctx->name.c_str();
}

static void ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    delete rpc_ctx;
    delete backend;
}

static void ggml_backend_rpc_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    if (rpc_use_hash_cache(tensor, size)) {
        auto request = std::make_shared<rpc_msg_set_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->offset = offset;
        request->hash = fnv_hash((const uint8_t*)data, size);
        rpc_msg_set_tensor_hash_rsp response;
        // TODO: make this async
        ctx->dispatcher->send(RPC_CMD_SET_TENSOR_HASH, request, sizeof(*request), &response, sizeof(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.result ? "hit" : "miss");
        if (response.result) {
            // the server has the same data, no need to send it
            return;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    size_t input_size;
    auto input = serialize_set_tensor(rpc_tensor, cache_flag, offset, data, size, input_size);
    ctx->dispatcher->send_async(RPC_CMD_SET_TENSOR, input, input_size);
}

static void ggml_backend_rpc_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    auto request = std::make_shared<rpc_msg_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size = size;
    ctx->dispatcher->send_async(RPC_CMD_GET_TENSOR, request, sizeof(*request), data, size);
}

static void ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->synchronize();
}

static void add_tensor(ggml_tensor * tensor, const ggml_cgraph * cgraph, const rpc_dispatcher * dispatcher, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & visited) {
    if (tensor == nullptr) {
        return;
    }
    if (visited.find(tensor) != visited.end()) {
        return;
    }
    visited.insert(tensor);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        add_tensor(tensor->src[i], cgraph, dispatcher, tensors, visited);
    }
    add_tensor(tensor->view_src, cgraph, dispatcher, tensors, visited);
    rpc_tensor result = serialize_tensor(tensor, dispatcher);
    const size_t hash_pos = ggml_hash_find(&cgraph->visited_hash_set, tensor);
    if (hash_pos != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
        result.use_count = cgraph->use_counts[hash_pos];
    }
    tensors.push_back(result);
}

static uint8_t * serialize_graph(uint32_t device, const ggml_cgraph * cgraph, const rpc_dispatcher * dispatcher, size_t * output_size) {
    uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> visited;
    for (uint32_t i = 0; i < n_nodes; i++) {
        add_tensor(cgraph->nodes[i], cgraph, dispatcher, tensors, visited);
    }
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    *output_size = 2*sizeof(uint32_t) + sizeof(uint64_t) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t) + n_tensors * sizeof(rpc_tensor);
    uint8_t * output = new uint8_t[*output_size]();
    uint8_t * dest = output;
    memcpy(dest, &device, sizeof(device));
    dest += sizeof(device);
    memcpy(dest, &cgraph->uid, sizeof(cgraph->uid));
    dest += sizeof(cgraph->uid);
    memcpy(dest, &n_nodes, sizeof(n_nodes));
    dest += sizeof(n_nodes);
    for (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(dest + i * sizeof(uint64_t), &cgraph->nodes[i], sizeof(uint64_t));
    }
    dest += n_nodes * sizeof(uint64_t);
    memcpy(dest, &n_tensors, sizeof(n_tensors));
    dest += sizeof(n_tensors);
    rpc_tensor * out_tensors = (rpc_tensor *)dest;
    memcpy(out_tensors, tensors.data(), n_tensors * sizeof(rpc_tensor));
    return output;
}

void rpc_dispatcher::graph_compute(uint32_t device, const ggml_cgraph * cgraph) {
    std::lock_guard<std::mutex> lock(graph_mutex);
    GGML_ASSERT(cgraph->n_nodes > 0);
    auto & device_graph_uids = graph_uids[device];
    bool reuse = cgraph->uid != 0 && device_graph_uids.count(cgraph->uid) > 0;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, reuse: %s\n", __func__, device, cgraph->uid, cgraph->n_nodes, reuse ? "yes" : "no");
    if (reuse) {
        auto request = std::make_shared<rpc_msg_graph_recompute_req>();
        request->device = device;
        request->uid    = cgraph->uid;
        send_async(RPC_CMD_GRAPH_RECOMPUTE, request, sizeof(*request));
    } else {
        if (cgraph->uid != 0) {
            if (device_graph_uids.size() >= GRAPH_CACHE_MAX) {
                LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                        __func__, device, device_graph_uids.size());
                device_graph_uids.clear();
            }
            device_graph_uids.insert(cgraph->uid);
        }
        size_t input_size = 0;
        uint8_t * input = serialize_graph(device, cgraph, this, &input_size);
        LOG_DBG2("[%s] serialized graph: %s\n", __func__, rpc_log_format_size(input_size).c_str());
        std::shared_ptr<uint8_t> input_ptr(input, std::default_delete<uint8_t[]>());
        send_async(RPC_CMD_GRAPH_COMPUTE, input_ptr, input_size);
    }
}

static enum ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->graph_compute(rpc_ctx->device, cgraph);
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_rpc_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->event_record(event);
}

static void ggml_backend_rpc_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    // this is noop for RPC as we have a single stream
    GGML_UNUSED(backend);
    GGML_UNUSED(event);
}

static void ggml_backend_rpc_set_tensor_2d_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
    GGML_UNUSED(backend);
}

static void ggml_backend_rpc_get_tensor_2d_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_tensor_get_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
    GGML_UNUSED(backend);
}

static ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .set_tensor_async        = */ ggml_backend_rpc_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_rpc_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_rpc_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_rpc_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_rpc_graph_compute,
    /* .event_record            = */ ggml_backend_rpc_event_record,
    /* .event_wait              = */ ggml_backend_rpc_event_wait,
    /* .graph_optimize          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::string buft_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    // NOTE: buffer types are allocated and never freed; this is by design
    static std::unordered_map<std::string, ggml_backend_buffer_type_t> buft_map;
    auto it = buft_map.find(buft_name);
    if (it != buft_map.end()) {
        return it->second;
    }
    auto dispatcher = get_dispatcher(endpoint);
    size_t alignment = get_alignment(dispatcher, device);
    size_t max_size = get_max_size(dispatcher, device);
    ggml_backend_rpc_buffer_type_context * buft_ctx = new ggml_backend_rpc_buffer_type_context {
        /* .endpoint  = */ endpoint,
        /* .device    = */ device,
        /* .name      = */ buft_name,
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_buffer_type_t buft = new ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_rpc_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ buft_ctx
    };
    buft_map[buft_name] = buft;
    LOG_DBG("[%s] created %s (alignment: %s, max_size: %s)\n", __func__, buft_name.c_str(),
            rpc_log_format_size(alignment).c_str(), rpc_log_format_size(max_size).c_str());
    return buft;
}

ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device) {
    std::string dev_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    auto dispatcher = get_dispatcher(endpoint);
    ggml_backend_rpc_context * ctx = new ggml_backend_rpc_context {
        /* .dispatcher = */ dispatcher,
        /* .endpoint   = */ endpoint,
        /* .device     = */ device,
        /* .name       = */ dev_name,
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    LOG_DBG("[%s] %s\n", __func__, dev_name.c_str());
    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_rpc_guid(),
        /* .iface   = */ ggml_backend_rpc_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ ctx
    };
    return backend;
}

bool ggml_backend_is_rpc(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total) {
    auto dispatcher = get_dispatcher(endpoint);
    auto request = std::make_shared<rpc_msg_get_device_memory_req>();
    request->device = device;
    rpc_msg_get_device_memory_rsp response;
    dispatcher->send(RPC_CMD_GET_DEVICE_MEMORY, request, sizeof(*request), &response, sizeof(response));
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, device,
             rpc_log_format_size(response.free_mem).c_str(), rpc_log_format_size(response.total_mem).c_str());
    *free = response.free_mem;
    *total = response.total_mem;
}

// RPC server-side implementation

struct rpc_comm_async;
struct rpc_comm_async_deleter {
    void operator()(rpc_comm_async * async) const;
};

class rpc_server {
public:
    rpc_server(std::vector<ggml_backend_t> all_backends, const char * cache_dir)
        : backends(std::move(all_backends)), cache_dir(cache_dir) {
        stored_graphs.resize(backends.size());
        comm_states.resize(backends.size());
    }
    ~rpc_server();

    void hello(rpc_msg_hello_rsp & response);
    bool alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response);
    bool get_alignment(const rpc_msg_get_alignment_req & request, rpc_msg_get_alignment_rsp & response);
    bool get_max_size(const rpc_msg_get_max_size_req & request, rpc_msg_get_max_size_rsp & response);
    bool buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response);
    bool free_buffer(const rpc_msg_free_buffer_req & request);
    bool buffer_clear(const rpc_msg_buffer_clear_req & request);
    bool memset_tensor(const rpc_msg_memset_tensor_req & request);
    bool set_tensor(const std::vector<uint8_t> & input);
    bool set_tensor_2d(const std::vector<uint8_t> & input);
    bool set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response);
    bool get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response);
    bool get_tensor_2d(const rpc_msg_get_tensor_2d_req & request, std::vector<uint8_t> & response);
    bool copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response);
    bool graph_compute(const std::vector<uint8_t> & input);
    bool graph_recompute(const rpc_msg_graph_recompute_req & request);
    bool comm_init(const rpc_msg_comm_init_req & request, rpc_msg_comm_init_rsp & response);
    bool comm_allreduce(const rpc_msg_comm_allreduce_req & request);
    bool comm_free(const rpc_msg_comm_free_req & request);
    bool init_tensor(const rpc_msg_init_tensor_req & request);
    bool get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response);
    bool get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response);

    struct stored_graph {
        std::vector<uint8_t>   buffer;
        ggml_cgraph          * graph;
    };

private:
    void sync_all_backends();
    bool get_cached_file(uint64_t hash, std::vector<uint8_t> & data);
    ggml_tensor * deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor);
    ggml_tensor * create_node(uint64_t id,
                              struct ggml_context * ctx,
                              const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map);


    // pairwise allreduce over a direct connection to the peer server
    struct comm_state {
        socket_ptr              peer;
        uint32_t                rank = 0;
        uint32_t                world = 0;
        ggml_backend_buffer_ptr scratch;
        size_t                  scratch_size = 0;
        std::vector<uint8_t>    send_buf;
        std::vector<uint8_t>    recv_buf;
        bool                    wire_f32 = false;              // send every host-path partial as F32 (RPC_COMM_FLAG_WIRE_F32)
        void *                  nccl = nullptr;                // NCCL communicator with the peer (RPC_COMM_FLAG_NCCL)
        bool                  (*nccl_allreduce)(ggml_backend_t, void *, ggml_tensor *) = nullptr;
        void                  (*nccl_free)(void *) = nullptr;   // COMM_FREE: both ranks free the communicator together
        void                  (*nccl_abort)(void *) = nullptr;  // disconnect without COMM_FREE: the peer may not take part
        // world > 2: a direct connection to every other rank (peers[rank] stays empty), each rank's partial behind its
        // rpc_comm_part_hdr, the sum, and how many all-reduces this rank has run
        std::vector<socket_ptr>           peers;
        std::vector<std::vector<uint8_t>> parts;
        std::vector<float>                sum;
        uint64_t                          seq = 0;
        // GGML_RPC_COMM_ASYNC: the all-reduces queued on the backend's stream, exchanged by a thread of their own
        std::unique_ptr<struct rpc_comm_async, rpc_comm_async_deleter> async;
    };

    bool comm_init_mesh(const rpc_msg_comm_init_req & request, comm_state & state);
    bool comm_allreduce_mesh(comm_state & state, ggml_backend_t backend, ggml_tensor * t_dst);
    bool comm_allreduce_mesh_async(comm_state & state, ggml_backend_t backend, ggml_tensor * t_dst);

    std::vector<ggml_backend_t> backends;
    const char * cache_dir;
    std::unordered_set<ggml_backend_buffer_t> buffers;
    // computed graphs cached per backend, keyed by uid
    std::vector<std::unordered_map<uint64_t, stored_graph>> stored_graphs;
    std::vector<comm_state> comm_states;
};

void rpc_server::hello(rpc_msg_hello_rsp & response) {
    response.major = RPC_PROTO_MAJOR_VERSION;
    response.minor = RPC_PROTO_MINOR_VERSION;
    response.patch = RPC_PROTO_PATCH_VERSION;
    LOG_DBG("[%s] version: %d.%d.%d\n", __func__, response.major, response.minor, response.patch);
}

bool rpc_server::get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft;
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead()*(1 + GGML_MAX_SRC),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (request.srcs[i].id != 0) {
            tensor->src[i] = deserialize_tensor(ctx, &request.srcs[i]);
        }
    }

    LOG_DBG2("[%s] device: %u, tensor: %s, buffer: %p, data: %p\n", __func__, dev_id, rpc_log_format_tensor(tensor).c_str(),
             (void*)tensor->buffer, tensor->data);
    if (tensor->buffer == nullptr) {
        //No buffer allocated.
        buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    } else {
        buft = tensor->buffer->buft;
    }

    response.alloc_size = ggml_backend_buft_get_alloc_size(buft, tensor);

    return true;
}

bool rpc_server::alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, request.size);
    response.remote_ptr = 0;
    response.remote_size = 0;
    if (buffer != nullptr) {
        response.remote_ptr = reinterpret_cast<uint64_t>(buffer);
        response.remote_size = buffer->size;
        LOG_DBG("[%s] device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, dev_id, rpc_log_format_size(request.size).c_str(), response.remote_ptr,
                rpc_log_format_size(response.remote_size).c_str());
        buffers.insert(buffer);
    } else {
        LOG_WARN("[%s] device: %u, size: %s -> failed\n", __func__, dev_id, rpc_log_format_size(request.size).c_str());
    }
    return true;
}

bool rpc_server::get_alignment(const rpc_msg_get_alignment_req & request, rpc_msg_get_alignment_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, dev_id, rpc_log_format_size(alignment).c_str());
    response.alignment = alignment;
    return true;
}

bool rpc_server::get_max_size(const rpc_msg_get_max_size_req & request, rpc_msg_get_max_size_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t max_size = ggml_backend_buft_get_max_size(buft);
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, dev_id, rpc_log_format_size(max_size).c_str());
    response.max_size = max_size;
    return true;
}

bool rpc_server::buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response) {
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    void * base = ggml_backend_buffer_get_base(buffer);
    response.base_ptr = reinterpret_cast<uint64_t>(base);
    return true;
}

bool rpc_server::free_buffer(const rpc_msg_free_buffer_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    // Discard all cached graphs to avoid use-after-free in graph_recompute,
    // since their nodes may hold pointers to the buffer being freed. The client
    // will have to re-serialize its graphs from now on.
    for (auto & sgs : stored_graphs) {
        for (auto & sg : sgs) {
            sg.second.graph = nullptr;
        }
    }
    ggml_backend_buffer_free(buffer);
    buffers.erase(buffer);
    return true;
}

bool rpc_server::buffer_clear(const rpc_msg_buffer_clear_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_clear(buffer, request.value);
    return true;
}

bool rpc_server::memset_tensor(const rpc_msg_memset_tensor_req & request) {
    sync_all_backends();
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }

    const uint64_t tensor_size = ggml_nbytes(tensor);
    if (request.offset > tensor_size || request.size > tensor_size - request.offset) {
        LOG_ERROR("[%s] tensor region (offset=%" PRIu64 ", size=%" PRIu64 ") out of tensor bounds [0, %" PRIu64 ")\n",
                       __func__, request.offset, request.size, tensor_size);
        return false;
    }

    const uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
    const uint64_t buffer_size = ggml_backend_buffer_get_size(tensor->buffer);
    if (request.tensor.data < buffer_start) {
        LOG_ERROR("[%s] tensor data before buffer start\n", __func__);
        return false;
    }
    const uint64_t data_offset = request.tensor.data - buffer_start;
    if (data_offset > buffer_size ||
        request.offset > buffer_size - data_offset ||
        request.size > buffer_size - data_offset - request.offset) {
        LOG_ERROR("[%s] tensor region out of buffer bounds\n", __func__);
        return false;
    }
    if (tensor->buffer->iface.memset_tensor == nullptr) {
        LOG_ERROR("[%s] memset not implemented by backend buffer\n", __func__);
        return false;
    }

    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, value: %u\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (void *) tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(request.size).c_str(), request.value);
    ggml_backend_tensor_memset(tensor, request.value, request.offset, request.size);
    return true;
}

ggml_tensor * rpc_server::deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor) {
    // Validate tensor type before using it
    if (tensor->type >= GGML_TYPE_COUNT) {
        LOG_ERROR("[%s] invalid tensor type received: %u\n", __func__, tensor->type);
        return nullptr;
    }

    // Fix: Prevent division by zero if blck_size is 0 (e.g., deprecated types)
    if (ggml_blck_size((enum ggml_type)tensor->type) == 0) {
        LOG_ERROR("[%s] invalid tensor type received (blck_size is 0): %u\n", __func__, tensor->type);
        return nullptr;
    }

    ggml_tensor * result = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

    // ggml_new_tensor_4d might fail if dimensions are invalid, although less likely to crash than invalid type
    if (result == nullptr) {
        LOG_ERROR("[%s] ggml_new_tensor_4d failed for type %u\n", __func__, tensor->type);
        return nullptr;
    }

    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    if (result->buffer && buffers.find(result->buffer) == buffers.end()) {
        result->buffer = nullptr;
    }

    if (result->buffer && ggml_nelements(result) > 0) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_size = (uint64_t) ggml_nbytes(result);
        uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(result->buffer);
        uint64_t buffer_size = (uint64_t) ggml_backend_buffer_get_size(result->buffer);
        if (tensor->data + tensor_size < tensor->data ||
            tensor->data < buffer_start || tensor->data + tensor_size > buffer_start + buffer_size) {
            LOG_ERROR("[%s] tensor '%s' (op %s, type %s, ne [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]) "
                      "data [0x%" PRIx64 ", 0x%" PRIx64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, tensor->name, ggml_op_name((ggml_op) tensor->op), ggml_type_name(result->type),
                      result->ne[0], result->ne[1], result->ne[2], result->ne[3],
                      tensor->data, tensor->data + tensor_size, buffer_start, buffer_start + buffer_size);
            return nullptr;
        }
    }

    result->op = (ggml_op) tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data = reinterpret_cast<void *>(tensor->data);
    ggml_set_name(result, tensor->name);
    return result;
}


bool rpc_server::set_tensor(const std::vector<uint8_t> & input) {
    sync_all_backends();
    // serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes) |
    uint8_t  cache_flag;
    uint64_t offset;
    const size_t header_size = sizeof(rpc_tensor) + sizeof(cache_flag) + sizeof(offset);
    if (input.size() < header_size) {
        return false;
    }
    const rpc_tensor * in_tensor = (const rpc_tensor *)input.data();
    memcpy(&cache_flag, input.data() + sizeof(rpc_tensor), sizeof(cache_flag));
    memcpy(&offset,     input.data() + sizeof(rpc_tensor) + sizeof(cache_flag), sizeof(offset));
    const size_t size = input.size() - header_size;

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (void*)tensor->buffer, tensor->data, offset,
            rpc_log_format_size(size).c_str());

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || size > (p1 - in_tensor->data - offset)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu) out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, in_tensor->data, offset, size, p0, p1);
            return false;
        }
    }

    const void * data = input.data() + header_size;
    if (cache_dir && cache_flag) {
        uint64_t hash = fnv_hash((const uint8_t*)data, size);
        char hash_str[17];
        snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, hash);
        // save to cache_dir/hash_str
        fs::path cache_file = fs::path(cache_dir) / hash_str;
        std::ofstream ofs(cache_file, std::ios::binary);
        ofs.write((const char *)data, size);
        LOG_INFO("[%s] saved to '%s'\n", __func__, cache_file.string().c_str());
    }
    ggml_backend_tensor_set(tensor, data, offset, size);
    return true;
}

bool rpc_server::set_tensor_2d(const std::vector<uint8_t> & input) {
    sync_all_backends();
    // serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    if (input.size() < sizeof(rpc_tensor) + 4*sizeof(uint64_t)) {
        return false;
    }
    const rpc_tensor * in_tensor = (const rpc_tensor *)input.data();
    uint64_t header[4];
    memcpy(header, input.data() + sizeof(rpc_tensor), sizeof(header));
    const uint64_t offset   = header[0];
    const uint64_t size     = header[1];
    const uint64_t n_copies = header[2];
    const uint64_t stride   = header[3];

    const uint64_t data_size = input.size() - sizeof(rpc_tensor) - 4*sizeof(uint64_t);
    if (n_copies == 0 || size == 0 || size > data_size / n_copies || size * n_copies != data_size) {
        return false;
    }

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (void*)tensor->buffer, tensor->data, offset,
            rpc_log_format_size(size).c_str(), n_copies, stride);

    // sanitize tensor->data
    {
        if (stride != 0 && n_copies - 1 > (UINT64_MAX - size) / stride) {
            return false;
        }
        const uint64_t span = (n_copies - 1)*stride + size;
        const uint64_t p0 = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
        const uint64_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (in_tensor->data < p0 || in_tensor->data > p1 || offset > p1 - in_tensor->data || span > p1 - in_tensor->data - offset) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, in_tensor->data, offset, span, p0, p1);
            return false;
        }
        if (offset > ggml_nbytes(tensor) || span > ggml_nbytes(tensor) - offset) {
            LOG_ERROR("[%s] tensor write region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, offset, span, ggml_nbytes(tensor));
            return false;
        }
    }

    const void * data = input.data() + sizeof(rpc_tensor) + 4*sizeof(uint64_t);
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride, size);
    return true;
}

bool rpc_server::get_cached_file(uint64_t hash, std::vector<uint8_t> & data) {
    if (!cache_dir) {
        return false;
    }
    char hash_str[17];
    snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, hash);
    fs::path cache_file = fs::path(cache_dir) / hash_str;
    std::error_code ec;
    if (!fs::exists(cache_file, ec)) {
        return false;
    }
    std::ifstream ifs(cache_file, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    data.resize(size);
    ifs.read((char *)data.data(), size);
    return true;
}

bool rpc_server::set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response)
{
    sync_all_backends();
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }

    std::vector<uint8_t> cached_file;
    if (!get_cached_file(request.hash, cached_file)) {
        LOG_DBG2("[%s] tensor: %s, offset: %" PRIu64 ", hash: 0x%" PRIx64 ", cache miss\n",
                __func__, rpc_log_format_tensor(tensor).c_str(), request.offset, request.hash);
        response.result = 0;
        return true;
    }
    size_t size = cached_file.size();
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, hash: 0x%" PRIx64 ", cache hit\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (void*)tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(size).c_str(), request.hash);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data + request.offset < p0
         || request.tensor.data + request.offset >= p1
         || size > (p1 - request.tensor.data - request.offset)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu, hash=0x%" PRIx64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, request.tensor.data, request.offset, size, request.hash, p0, p1);
            return false;
        }
    }
    ggml_backend_tensor_set(tensor, cached_file.data(), request.offset, size);
    response.result = 1;
    return true;
}

bool rpc_server::init_tensor(const rpc_msg_init_tensor_req & request) {
    sync_all_backends();
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        return false;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p\n", __func__, rpc_log_format_tensor(tensor).c_str(),
            (void*)tensor->buffer, tensor->data);
    // Call the backend's buffer_init_tensor function
    ggml_backend_buffer_t buffer = tensor->buffer;
    if (buffer && buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    } else {
        if (!buffer) {
            LOG_ERROR("Tensor with null buffer passed to init_tensor function\n");
        }
    }

    if (tensor->extra != nullptr) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
        return false;
    }

    return true;
}

bool rpc_server::get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), (void*)tensor->buffer, tensor->data, request.offset,
             rpc_log_format_size(request.size).c_str());

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data + request.offset < p0 ||
            request.tensor.data + request.offset >= p1 ||
            request.size > (p1 - request.tensor.data - request.offset)) {
                LOG_ERROR("[%s] requested tensor region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                               __func__, request.tensor.data, request.offset, request.size, p0, p1);
                return false;
        }
    }

    response.resize(request.size, 0);
    ggml_backend_tensor_get(tensor, response.data(), request.offset, request.size);
    return true;
}

bool rpc_server::get_tensor_2d(const rpc_msg_get_tensor_2d_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (void*)tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(request.size).c_str(), request.n_copies, request.stride);

    // sanitize tensor->data
    {
        if (request.n_copies == 0 || request.size == 0 || request.size > UINT64_MAX / request.n_copies) {
            return false;
        }
        if (request.stride != 0 && request.n_copies - 1 > (UINT64_MAX - request.size) / request.stride) {
            return false;
        }
        const uint64_t span = (request.n_copies - 1)*request.stride + request.size;
        const uint64_t p0 = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
        const uint64_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data < p0 || request.tensor.data > p1 || request.offset > p1 - request.tensor.data ||
                span > p1 - request.tensor.data - request.offset) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, request.tensor.data, request.offset, span, p0, p1);
            return false;
        }
        if (request.offset > ggml_nbytes(tensor) || span > ggml_nbytes(tensor) - request.offset) {
            LOG_ERROR("[%s] tensor read region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, request.offset, span, ggml_nbytes(tensor));
            return false;
        }
    }

    response.resize(request.size * request.n_copies, 0);
    ggml_backend_tensor_get_2d(tensor, response.data(), request.offset, request.size, request.n_copies, request.stride, request.size);
    return true;
}

bool rpc_server::copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response) {
    sync_all_backends();
    struct ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
    if (src == nullptr || dst == nullptr || src->buffer == nullptr || dst->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensors\n", __func__);
        return false;
    }

    uint64_t src_size   = (uint64_t) ggml_nbytes(src);
    uint64_t dst_data   = (uint64_t) dst->data;
    uint64_t dst_base   = (uint64_t) ggml_backend_buffer_get_base(dst->buffer);
    uint64_t dst_buf_sz = (uint64_t) ggml_backend_buffer_get_size(dst->buffer);

    if (dst_data + src_size > dst_base + dst_buf_sz) {
        LOG_ERROR("[%s] out-of-bounds write in rpc_server::copy_tensor:\n"
                         "    write range : [0x%" PRIx64 ", 0x%" PRIx64 "]\n"
                         "    buffer base: [0x%" PRIx64 ", 0x%" PRIx64 "]\n",
                         __func__,
                         dst_data,
                         dst_data + src_size,
                         dst_base,
                         dst_base + dst_buf_sz);
        return false;
    }

    LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());

    response.result = ggml_backend_buffer_copy_tensor(src, dst);
    return true;
}

ggml_tensor * rpc_server::create_node(uint64_t id,
                                      struct ggml_context * ctx,
                                      const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map) {
    if (tensor_map.find(id) != tensor_map.end()) {
        return tensor_map[id];
    }
    // Safely find the tensor pointer
    auto it_ptr = tensor_ptrs.find(id);
    if (it_ptr == tensor_ptrs.end()) {
        return nullptr;
    }
    const rpc_tensor * tensor = it_ptr->second;

    struct ggml_tensor * result = deserialize_tensor(ctx, tensor);
    if (result == nullptr) {
        return nullptr;
    }
    if (result->buffer == nullptr && result->data != nullptr) {
        LOG_ERROR("[%s] invalid data ptr\n", __func__);
        return nullptr;
    }
    tensor_map[id] = result;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        // Check if the source ID is 0 before calling create_node recursively
        if (tensor->src[i] == 0) {
            result->src[i] = nullptr;
        } else {
            result->src[i] = create_node(tensor->src[i], ctx, tensor_ptrs, tensor_map);
            // If the recursive call failed for a non-zero ID, propagate the error
            if (result->src[i] == nullptr) {
                LOG_ERROR("[%s] failed to create source node %d (src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                               __func__, i, tensor->src[i], id);
                // Must return nullptr to signal failure up the call stack
                return nullptr;
            }
        }
    }

    // Handle view_src similarly
    if (tensor->view_src == 0) {
        result->view_src = nullptr;
    } else {
        result->view_src = create_node(tensor->view_src, ctx, tensor_ptrs, tensor_map);
        // If the recursive call failed for a non-zero ID, propagate the error
        if (result->view_src == nullptr) {
            LOG_ERROR("[%s] failed to create view_src node (view_src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                           __func__, tensor->view_src, id);
            // Must return nullptr to signal failure up the call stack
            return nullptr;
        }
    }
    result->view_offs = tensor->view_offs;
    return result;
}

bool rpc_server::graph_compute(const std::vector<uint8_t> & input) {
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    if (input.size() < 2*sizeof(uint32_t) + sizeof(uint64_t)) {
        return false;
    }
    const uint8_t * src = input.data();
    uint32_t device;
    memcpy(&device, src, sizeof(device));
    src += sizeof(device);
    if (device >= backends.size()) {
        return false;
    }
    uint64_t uid;
    memcpy(&uid, src, sizeof(uid));
    src += sizeof(uid);
    uint32_t n_nodes;
    memcpy(&n_nodes, src, sizeof(n_nodes));
    src += sizeof(n_nodes);
    if (input.size() < 2*sizeof(uint32_t) + sizeof(uint64_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t)) {
        return false;
    }
    const uint64_t * nodes = (const uint64_t *)src;
    src += n_nodes*sizeof(uint64_t);
    uint32_t n_tensors;
    memcpy(&n_tensors, src, sizeof(n_tensors));
    src += sizeof(n_tensors);
    if (input.size() < 2*sizeof(uint32_t) + sizeof(uint64_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t) + n_tensors*sizeof(rpc_tensor)) {
        return false;
    }
    const rpc_tensor * tensors = (const rpc_tensor *)src;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, n_tensors: %u\n", __func__, device, uid, n_nodes, n_tensors);

    // graphs with uid == 0 are not cached, see GRAPH_CACHE_MAX for the eviction policy
    if (uid != 0 && stored_graphs[device].size() >= GRAPH_CACHE_MAX) {
        LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                __func__, device, stored_graphs[device].size());
        stored_graphs[device].clear();
    }
    stored_graph sg_tmp;
    stored_graph & sg = uid != 0 ? stored_graphs[device][uid] : sg_tmp;

    size_t buf_size = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, false);
    if (sg.buffer.size() < buf_size) {
        sg.buffer.resize(buf_size);
    }
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ sg.buffer.data(),
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, false);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, const rpc_tensor*> tensor_ptrs;
    tensor_ptrs.reserve(n_tensors);
    for (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs.emplace(tensors[i].id, &tensors[i]);
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    tensor_map.reserve(n_nodes);
    // the compute itself is asynchronous, so building the graph is the only server-side cost
    // observable here - the round trip is timed on the client
    const auto t_deserialize = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], sizeof(id));
        graph->nodes[i] = create_node(id, ctx, tensor_ptrs, tensor_map);
        // Check if create_node failed for a *non-zero* ID.
        // If id was 0, create_node returning nullptr is expected.
        // If id was non-zero and create_node returned nullptr, it indicates a deserialization error.
        if (graph->nodes[i] == nullptr && id != 0) {
            LOG_ERROR("[%s] failed to create graph node %d (id=%" PRId64 ")\n", __func__, i, id);
            return false;
        }
        if (graph->nodes[i] != nullptr) {
            LOG_DBG3("[%s] node %u: id=%" PRId64 ", %s\n", __func__, i, id, rpc_log_format_tensor(graph->nodes[i]).c_str());
            const size_t hash_pos = ggml_hash_insert(&graph->visited_hash_set, graph->nodes[i]);
            graph->use_counts[hash_pos] = tensor_ptrs.at(id)->use_count;
        }
    }
    const std::chrono::duration<double, std::milli> deserialize_ms = std::chrono::steady_clock::now() - t_deserialize;

    ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    sg.graph = graph;
    LOG_DBG3("[%s] device: %u, graph build: %.3f ms\n", __func__, device, deserialize_ms.count());
    return true;
}

bool rpc_server::graph_recompute(const rpc_msg_graph_recompute_req & request) {
    uint32_t device = request.device;
    if (device >= backends.size()) {
        return false;
    }
    auto it = stored_graphs[device].find(request.uid);
    if (it == stored_graphs[device].end() || it->second.graph == nullptr) {
        LOG_ERROR("[%s] device: %u, graph with uid %" PRIu64 " not found\n", __func__, device, request.uid);
        return false;
    }
    ggml_cgraph * graph = it->second.graph;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 "\n", __func__, device, request.uid);
    ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    return true;
}

// graph compute is asynchronous; commands that read or write buffer data synchronize first
void rpc_server::sync_all_backends() {
    for (ggml_backend_t backend : backends) {
        ggml_backend_synchronize(backend);
    }
}

// The comm link between two servers uses the same caps negotiation as the client HELLO,
// so it gets the same transport upgrades (e.g. RDMA).
// NCCL between the two servers of the pairwise comm, through the CUDA backend's exports
typedef bool   (*rpc_nccl_unique_id_t)(void * out, size_t size);
typedef void * (*rpc_nccl_comm_init_rank_t)(ggml_backend_t backend, const void * id, int nranks, int rank);
typedef void   (*rpc_nccl_comm_free_t)(void * comm);
typedef void   (*rpc_nccl_comm_abort_t)(void * comm);
typedef bool   (*rpc_nccl_allreduce_t)(ggml_backend_t backend, void * comm, ggml_tensor * t);

static void * rpc_backend_proc(ggml_backend_t backend, const char * name) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = dev != nullptr ? ggml_backend_dev_backend_reg(dev) : nullptr;
    return reg != nullptr ? ggml_backend_reg_get_proc_address(reg, name) : nullptr;
}

bool rpc_server::comm_init(const rpc_msg_comm_init_req & request, rpc_msg_comm_init_rsp & response) {
    response.ok = 0;
    if (request.device >= backends.size() || request.world < 2 || request.world > RPC_COMM_MAX_WORLD ||
            request.rank >= request.world) {
        return true;
    }
    comm_state & state = comm_states[request.device];
    if (state.peer != nullptr || !state.peers.empty()) {
        response.ok = 1;
        return true;
    }
    if (request.world > 2 || (request.flags & RPC_COMM_FLAG_MESH) != 0) {
        response.ok = comm_init_mesh(request, state) ? 1 : 0;
        return true;
    }
    uint8_t local_caps[RPC_CONN_CAPS_SIZE] = {};
    uint8_t remote_caps[RPC_CONN_CAPS_SIZE] = {};
    if (request.rank == 0) {
        socket_ptr srv = socket_t::create_server("0.0.0.0", request.port);
        if (srv == nullptr) {
            LOG_ERROR("[%s] failed to listen on comm port %u\n", __func__, request.port);
            return true;
        }
        state.peer = srv->accept();
        if (state.peer == nullptr) {
            return true;
        }
        if (!state.peer->recv_data(remote_caps, sizeof(remote_caps))) {
            state.peer = nullptr;
            return true;
        }
        state.peer->get_caps(local_caps);
        if (!state.peer->send_data(local_caps, sizeof(local_caps))) {
            state.peer = nullptr;
            return true;
        }
        state.peer->update_caps(remote_caps);
    } else {
        const std::string host(request.host, strnlen(request.host, sizeof(request.host)));
        // rank 0 may not be listening yet, retry for a few seconds
        for (int i = 0; i < 100 && state.peer == nullptr; i++) {
            state.peer = socket_t::connect(host.c_str(), request.port);
            if (state.peer == nullptr) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
        if (state.peer == nullptr) {
            LOG_ERROR("[%s] failed to connect to peer %s:%u\n", __func__, host.c_str(), request.port);
            return true;
        }
        state.peer->get_caps(local_caps);
        if (!state.peer->send_data(local_caps, sizeof(local_caps)) ||
            !state.peer->recv_data(remote_caps, sizeof(remote_caps))) {
            state.peer = nullptr;
            return true;
        }
        state.peer->update_caps(remote_caps);
    }
    state.rank     = request.rank;
    state.world    = request.world;
    state.wire_f32 = (request.flags & RPC_COMM_FLAG_WIRE_F32) != 0;

    // RPC_COMM_FLAG_NCCL: reduce on the device with NCCL, queued on the backend's stream, instead of copying each
    // partial to the host and exchanging it over the peer connection. The client sends the same flags to both ranks,
    // so both take this branch, and they agree on whether NCCL is available before initializing it, because
    // ncclCommInitRank waits for every rank to join.
    if ((request.flags & RPC_COMM_FLAG_NCCL) != 0) {
        ggml_backend_t backend = backends[request.device];
        auto unique_id = (rpc_nccl_unique_id_t)      rpc_backend_proc(backend, "ggml_backend_cuda_nccl_unique_id");
        auto init_rank = (rpc_nccl_comm_init_rank_t) rpc_backend_proc(backend, "ggml_backend_cuda_nccl_comm_init_rank");
        state.nccl_allreduce = (rpc_nccl_allreduce_t)  rpc_backend_proc(backend, "ggml_backend_cuda_nccl_allreduce");
        state.nccl_free      = (rpc_nccl_comm_free_t)  rpc_backend_proc(backend, "ggml_backend_cuda_nccl_comm_free");
        state.nccl_abort     = (rpc_nccl_comm_abort_t) rpc_backend_proc(backend, "ggml_backend_cuda_nccl_comm_abort");
        uint8_t ok = unique_id != nullptr && init_rank != nullptr && state.nccl_allreduce != nullptr && state.nccl_free != nullptr &&
                     state.nccl_abort != nullptr;
        uint8_t peer_ok = 0;
        uint8_t id[128] = {};
        bool sent;
        if (request.rank == 0) {
            ok = ok && unique_id(id, sizeof(id));
            sent = state.peer->send_data(&ok, 1) && state.peer->send_data(id, sizeof(id)) && state.peer->flush() &&
                   state.peer->recv_data(&peer_ok, 1);
        } else {
            sent = state.peer->recv_data(&peer_ok, 1) && state.peer->recv_data(id, sizeof(id)) &&
                   state.peer->send_data(&ok, 1) && state.peer->flush();
        }
        if (!sent) {
            // the peer connection failed mid-handshake, so the ranks may now disagree: this COMM_INIT fails
            state.peer = nullptr;
            return true;
        }
        if (ok && peer_ok) {
            state.nccl = init_rank(backend, id, (int) request.world, (int) request.rank);
            // ncclCommInitRank can succeed on one rank and fail on the other, so the ranks agree on its result as
            // well: unless both have a communicator, neither keeps one, and both copy through the host
            uint8_t inited = state.nccl != nullptr;
            uint8_t peer_inited = 0;
            const bool agreed = request.rank == 0 ?
                state.peer->send_data(&inited, 1) && state.peer->flush() && state.peer->recv_data(&peer_inited, 1) :
                state.peer->recv_data(&peer_inited, 1) && state.peer->send_data(&inited, 1) && state.peer->flush();
            if ((!agreed || !peer_inited) && state.nccl != nullptr) {
                state.nccl_abort(state.nccl);
                state.nccl = nullptr;
            }
            if (!agreed) {
                state.peer = nullptr;
                return true;
            }
        }
        LOG_INFO("[%s] device %u NCCL all-reduce %s\n", __func__, request.device, state.nccl != nullptr ? "on" : "unavailable, copying through the host");
    }
    LOG_INFO("[%s] device %u joined pairwise comm as rank %u\n", __func__, request.device, request.rank);
    response.ok = 1;
    return true;
}

bool rpc_server::comm_allreduce(const rpc_msg_comm_allreduce_req & request) {
    if (request.device >= backends.size()) {
        return false;
    }
    comm_state & state = comm_states[request.device];
    if (state.peer == nullptr && state.peers.empty()) {
        LOG_ERROR("[%s] no communicator for device %u\n", __func__, request.device);
        return false;
    }
    ggml_backend_t backend = backends[request.device];

    size_t ctx_size = 16*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(8, false);
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * t_dst = deserialize_tensor(ctx, &request.tensor);
    if (t_dst == nullptr || t_dst->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    const size_t  nbytes = ggml_nbytes(t_dst);
    const int64_t ne     = ggml_nelements(t_dst);
    if (nbytes == 0) {
        return true;
    }
    if (!state.peers.empty()) {
        return comm_allreduce_mesh(state, backend, t_dst);
    }
    if (state.nccl != nullptr) {
        // NCCL reduces ggml_nelements floats in place, so it takes only a dense F32 tensor whose floats are exactly the
        // ggml_nbytes that deserialize_tensor kept inside the buffer; for a huge ne, ggml_nbytes wraps and is smaller
        if (t_dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(t_dst) || nbytes / sizeof(float) != (size_t) ne) {
            LOG_ERROR("[%s] NCCL all-reduce needs a contiguous F32 tensor whose size matches its element count, got %s with %" PRId64 " elements in %zu bytes\n",
                      __func__, ggml_type_name(t_dst->type), ne, nbytes);
            return false;
        }
        return state.nccl_allreduce(backend, state.nccl, t_dst);
    }
    // reduce large partials in bf16 to halve the wire bytes; small (decode-sized) ones
    // stay f32 since the extra casts and sync cost more than the bytes saved
    const bool   wire_bf16  = !state.wire_f32 && t_dst->type == GGML_TYPE_F32 && ne >= 32768;
    const size_t wire_bytes = wire_bf16 ? (size_t) ne*2 : nbytes;
    const size_t need       = wire_bf16 ? 2*nbytes : nbytes;
    if (state.scratch_size < need) {
        state.scratch.reset(ggml_backend_alloc_buffer(backend, need));
        state.scratch_size = need;
    }
    char * scratch_base = (char *) ggml_backend_buffer_get_base(state.scratch.get());
    state.send_buf.resize(wire_bytes);
    state.recv_buf.resize(wire_bytes);

    auto new_scratch_tensor = [&](ggml_type type, size_t offset) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
        t->buffer = state.scratch.get();
        t->data   = scratch_base + offset;
        return t;
    };
    auto new_cpy_node = [&](ggml_tensor * src, ggml_tensor * dst) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, dst->type, dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
        t->op     = GGML_OP_CPY;
        t->src[0] = src;
        t->src[1] = dst;
        t->buffer = dst->buffer;
        t->data   = dst->data;
        t->flags |= GGML_TENSOR_FLAG_COMPUTE;
        return t;
    };
    auto compute_nodes = [&](ggml_tensor * n0, ggml_tensor * n1) {
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 2, false);
        graph->nodes[0] = n0;
        graph->nodes[1] = n1;
        graph->n_nodes  = n1 != nullptr ? 2 : 1;
        ggml_status status = ggml_backend_graph_compute_async(backend, graph);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    };

    // wait for the pending subgraph that produced this partial
    ggml_backend_synchronize(backend);

    ggml_tensor * t_wire_send = nullptr;
    ggml_tensor * t_wire_recv = nullptr;
    if (wire_bf16) {
        t_wire_send = new_scratch_tensor(GGML_TYPE_BF16, 0);
        t_wire_recv = new_scratch_tensor(GGML_TYPE_BF16, ne*2);
        compute_nodes(new_cpy_node(t_dst, t_wire_send), nullptr);
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(t_wire_send, state.send_buf.data(), 0, wire_bytes);
    } else {
        ggml_backend_tensor_get(t_dst, state.send_buf.data(), 0, wire_bytes);
    }

    // rank 0 sends first, rank 1 receives first, so large payloads cannot deadlock
    if (state.rank == 0) {
        if (!state.peer->send_data(state.send_buf.data(), wire_bytes) || !state.peer->flush() ||
            !state.peer->recv_data(state.recv_buf.data(), wire_bytes)) {
            return false;
        }
    } else {
        if (!state.peer->recv_data(state.recv_buf.data(), wire_bytes) ||
            !state.peer->send_data(state.send_buf.data(), wire_bytes) || !state.peer->flush()) {
            return false;
        }
    }

    ggml_tensor * t_peer = new_scratch_tensor(t_dst->type, wire_bf16 ? (size_t) ne*4 : 0);
    ggml_tensor * t_cast = nullptr;
    if (wire_bf16) {
        ggml_backend_tensor_set(t_wire_recv, state.recv_buf.data(), 0, wire_bytes);
        t_cast = new_cpy_node(t_wire_recv, t_peer);
    } else {
        ggml_backend_tensor_set(t_peer, state.recv_buf.data(), 0, wire_bytes);
    }

    ggml_tensor * t_red = ggml_new_tensor_4d(ctx, t_dst->type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
    t_red->op     = GGML_OP_ADD;
    t_red->src[0] = t_dst;
    t_red->src[1] = t_peer;
    t_red->buffer = t_dst->buffer;
    t_red->data   = t_dst->data;
    t_red->flags |= GGML_TENSOR_FLAG_COMPUTE;

    if (t_cast != nullptr) {
        compute_nodes(t_cast, t_red);
    } else {
        compute_nodes(t_red, nullptr);
    }
    return true;
}

// More than two ranks: every pair of ranks gets its own connection, so a partial reaches each peer in one hop over the
// link the two share (three Sparks cabled as a triangle have a link per pair). Rank r listens on port + r and connects
// to each lower rank j at hosts[j], port + j, naming itself first so that the listener knows which rank arrived. Each
// link negotiates its transport like the pairwise one, so a host on an RDMA link's subnet gets RDMA.
// world > 2: a rank's IPv4 links, which a lower rank sends to a rank that connects to it, so that the two can move to the
// fastest link they share: the address the client gave may route over a slower one, or over loopback identities that
// RDMA cannot use (RDMA needs the address of the cable's own port)
struct rpc_comm_link {
    uint32_t addr;  // network order
    uint32_t mask;  // network order
    uint32_t speed; // Mbit/s, 0 when not known
};

// the first word a connecting rank sends: its rank, with this bit when it connects over a link already chosen
static constexpr uint32_t RPC_COMM_CHOSEN_LINK = 0x80000000u;

static std::vector<rpc_comm_link> rpc_comm_local_links() {
    std::vector<rpc_comm_link> links;
#if !defined(_WIN32)
    struct ifaddrs * ifs = nullptr;
    if (getifaddrs(&ifs) != 0) {
        return links;
    }
    for (struct ifaddrs * it = ifs; it != nullptr; it = it->ifa_next) {
        if (it->ifa_addr == nullptr || it->ifa_netmask == nullptr || it->ifa_addr->sa_family != AF_INET ||
                (it->ifa_flags & IFF_LOOPBACK) != 0 || (it->ifa_flags & IFF_UP) == 0 || (it->ifa_flags & IFF_RUNNING) == 0) {
            continue;
        }
        rpc_comm_link link = {};
        link.addr = ((const struct sockaddr_in *) it->ifa_addr)->sin_addr.s_addr;
        link.mask = ((const struct sockaddr_in *) it->ifa_netmask)->sin_addr.s_addr;
        if (link.mask == 0xffffffffu) {
            continue; // a /32 is an identity, not a link to a peer
        }
        std::ifstream speed_file(std::string("/sys/class/net/") + it->ifa_name + "/speed");
        long speed = 0;
        if (speed_file >> speed && speed > 0) {
            link.speed = (uint32_t) speed;
        }
        links.push_back(link);
    }
    freeifaddrs(ifs);
#endif
    return links;
}

// the peer's address on the fastest subnet both ranks are on; an address both have (an identical bridge on every
// node, such as Docker's) is no link between them
static std::string rpc_comm_best_link(const std::vector<rpc_comm_link> & ours, const std::vector<rpc_comm_link> & theirs) {
    std::string best;
#if !defined(_WIN32)
    bool found = false;
    uint32_t best_speed = 0;
    for (const auto & t : theirs) {
        for (const auto & o : ours) {
            const bool same_subnet = (t.addr & o.mask) == (o.addr & o.mask) && (t.addr & t.mask) == (o.addr & t.mask);
            if (t.addr == o.addr || !same_subnet) {
                continue;
            }
            const uint32_t speed = std::min(t.speed, o.speed);
            if (!found || speed > best_speed) {
                char text[INET_ADDRSTRLEN] = {};
                struct in_addr in = {};
                in.s_addr = t.addr;
                if (inet_ntop(AF_INET, &in, text, sizeof(text)) != nullptr) {
                    found = true;
                    best_speed = speed;
                    best = text;
                }
            }
        }
    }
#endif
    return best;
}

// Sends this rank's partial (`nbytes` at `own`) to every peer and receives theirs into `parts`, then sums all of them
// in rank order into `sum`, so every rank holds the same bits. Each partial goes behind its rpc_comm_part_hdr, as a
// separate message: the RDMA transport delivers messages whole, so a read must take exactly what one send wrote.
static bool rpc_comm_mesh_exchange(const std::vector<socket_ptr> & peers, uint32_t rank, uint32_t world, uint64_t seq,
                                   size_t nbytes, const uint8_t * own, std::vector<std::vector<uint8_t>> & parts, float * sum) {
    const rpc_comm_part_hdr hdr = { seq, (uint64_t) nbytes };
    for (auto & part : parts) {
        if (part.size() < nbytes) {
            part.resize(nbytes);
        }
    }
    // the header first: a peer on another all-reduce, or with a partial of another size, fails here instead of being
    // read as this partial
    auto recv_part = [&](uint32_t j) {
        rpc_comm_part_hdr got;
        if (!peers[j]->recv_data(&got, sizeof(got))) {
            return false;
        }
        if (got.seq != hdr.seq || got.nbytes != hdr.nbytes) {
            LOG_ERROR("[%s] rank %u expected all-reduce %" PRIu64 " of %" PRIu64 " bytes from rank %u, got %" PRIu64
                      " of %" PRIu64 " bytes\n", __func__, rank, hdr.seq, hdr.nbytes, j, got.seq, got.nbytes);
            return false;
        }
        return peers[j]->recv_data(parts[j].data(), nbytes);
    };
    auto send_part = [&](uint32_t j) {
        return peers[j]->send_data(&hdr, sizeof(hdr)) && peers[j]->send_data(own, nbytes) && peers[j]->flush();
    };
    auto lost = [&](uint32_t j) {
        LOG_ERROR("[%s] rank %u lost rank %u during all-reduce %" PRIu64 "\n", __func__, rank, j, hdr.seq);
        return false;
    };
    if (nbytes <= RPC_COMM_MESH_EAGER_BYTES) {
        for (uint32_t j = 0; j < world; j++) {
            if (j != rank && !send_part(j)) {
                return lost(j);
            }
        }
        for (uint32_t j = 0; j < world; j++) {
            if (j != rank && !recv_part(j)) {
                return lost(j);
            }
        }
    } else {
        // pair by pair in rank order, the lower rank of each pair sending first: the unfinished pair with the lowest
        // ranks always has both of its ranks waiting on it, so the exchange cannot deadlock
        for (uint32_t j = 0; j < world; j++) {
            if (j == rank) {
                continue;
            }
            const bool ok = rank < j ? send_part(j) && recv_part(j) : recv_part(j) && send_part(j);
            if (!ok) {
                return lost(j);
            }
        }
    }
    const int64_t ne = (int64_t) (nbytes / sizeof(float));
    for (uint32_t j = 0; j < world; j++) {
        const float * part = (const float *) (j == rank ? own : parts[j].data());
        if (j == 0) {
            memcpy(sum, part, nbytes);
            continue;
        }
        for (int64_t k = 0; k < ne; k++) {
            sum[k] += part[k];
        }
    }
    return true;
}

// GGML_RPC_COMM_ASYNC=1 (or flag:<path>, on while the file exists): the backend's stream copies each partial to host
// memory, signals `ready` and waits on `done`; this thread exchanges and sums the partials and signals `done`, so the
// server thread never waits for the stream and the next subgraph is already queued behind the wait. Only for a backend
// that exports ggml_backend_cuda_comm_stage (CUDA) and whose device can wait on host memory.
typedef void * (*rpc_comm_host_alloc_t)(size_t size);
typedef void   (*rpc_comm_host_free_t)(void * ptr);
typedef bool   (*rpc_comm_can_stage_t)(ggml_backend_t backend, uint32_t * ready, uint32_t * done);
typedef bool   (*rpc_comm_stage_t)(ggml_backend_t backend, ggml_tensor * t, void * out, const void * in,
                                   uint32_t * ready, uint32_t * done, uint32_t seq);

struct rpc_comm_async {
    static constexpr uint32_t RING = 256;         // all-reduces that may be queued ahead of the exchange

    rpc_comm_host_free_t host_free = nullptr;
    rpc_comm_stage_t     stage     = nullptr;
    ggml_backend_t       backend   = nullptr;
    uint32_t * flags    = nullptr;                // [0] ready, [1] done: host memory the stream writes and waits on
    uint8_t  * out      = nullptr;                // this rank's partial, copied out by the stream
    float    * in       = nullptr;                // the sum, copied back by the stream
    size_t     capacity = 0;
    void *   (*host_alloc)(size_t) = nullptr;

    // jobs: the server thread queues, the exchange thread takes them in order
    struct job { uint32_t flag; uint64_t seq; size_t nbytes; };
    job                   jobs[RING] = {};
    std::atomic<uint32_t> queued{0};              // the flag of the last queued all-reduce
    std::atomic<uint32_t> processed{0};           // the flag of the last one whose `done` was signalled
    std::atomic<bool>     stop{false};
    std::atomic<bool>     broken{false};          // an exchange failed: the client must be dropped
    uint32_t              next_flag = 0;
    std::thread           thread;

    // peers and their buffers, used by the exchange thread only while jobs are pending
    const std::vector<socket_ptr> * peers = nullptr;
    std::vector<std::vector<uint8_t>> * parts = nullptr;
    uint32_t rank = 0, world = 0;

    ~rpc_comm_async() {
        // all-reduces still queued on the stream wait on `done`: the thread releases them, without exchanging, before
        // it stops, so the stream drains instead of waiting forever
        broken.store(true, std::memory_order_release);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (thread.joinable() && !idle() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        stop.store(true, std::memory_order_release);
        if (thread.joinable()) {
            thread.join();
        }
        if (host_free != nullptr) {
            host_free(flags);
            host_free(out);
            host_free(in);
        }
    }

    bool idle() const {
        return processed.load(std::memory_order_acquire) == queued.load(std::memory_order_acquire);
    }

    void run() {
        uint32_t spins = 0;
        while (!stop.load(std::memory_order_acquire)) {
            const uint32_t last = processed.load(std::memory_order_relaxed);
            if (last == queued.load(std::memory_order_acquire)) {
                // spin while all-reduces keep coming, then back off, so an idle model does not hold a core
                if (++spins > 20000) {
                    std::this_thread::sleep_for(std::chrono::microseconds(spins > 200000 ? 1000 : 20));
                }
                continue;
            }
            spins = 0;
            const job & j = jobs[(last + 1) % RING];
            // wait for the stream to copy the partial out
            while (__atomic_load_n(&flags[0], __ATOMIC_ACQUIRE) < j.flag) {
                if (stop.load(std::memory_order_acquire)) {
                    return;
                }
            }
            if (!broken.load(std::memory_order_relaxed) &&
                !rpc_comm_mesh_exchange(*peers, rank, world, j.seq, j.nbytes, out, *parts, in)) {
                // the stream is released either way, so it drains; the server drops the client before using any result
                broken.store(true, std::memory_order_release);
            }
            __atomic_store_n(&flags[1], j.flag, __ATOMIC_RELEASE);
            processed.store(j.flag, std::memory_order_release);
        }
    }
};

void rpc_comm_async_deleter::operator()(rpc_comm_async * async) const {
    delete async;
}

static bool rpc_comm_async_enabled() {
    static const char * env = std::getenv("GGML_RPC_COMM_ASYNC");
    if (env == nullptr) {
        return false;
    }
    if (strncmp(env, "flag:", 5) == 0) {
        return std::filesystem::exists(env + 5);
    }
    return std::atoi(env) != 0;
}

bool rpc_server::comm_init_mesh(const rpc_msg_comm_init_req & request, comm_state & state) {
    const uint32_t rank  = request.rank;
    const uint32_t world = request.world;
    std::vector<socket_ptr> peers(world);
    socket_ptr srv;
    if (rank + 1 < world) {
        // every higher rank may connect at once
        srv = socket_t::create_server("0.0.0.0", (int) (request.port + rank), (int) world);
        if (srv == nullptr) {
            LOG_ERROR("[%s] rank %u failed to listen on comm port %u\n", __func__, rank, request.port + rank);
            return false;
        }
    }
    uint8_t local_caps[RPC_CONN_CAPS_SIZE] = {};
    uint8_t remote_caps[RPC_CONN_CAPS_SIZE] = {};
    const std::vector<rpc_comm_link> our_links = rpc_comm_local_links();
    for (uint32_t j = 0; j < rank; j++) {
        std::string host(request.hosts[j], strnlen(request.hosts[j], sizeof(request.hosts[j])));
        socket_ptr peer;
        // the first connection, at the host the client gave, learns the lower rank's links; when another link the two
        // share is faster, the rank connects again over it, telling the first connection it moves
        bool chosen = false;
        // the lower rank may not be listening yet, or may drop a connection before the handshake ends: the whole
        // handshake is retried for a few seconds, and a rank that reconnects replaces its first connection there
        for (int i = 0; i < 100 && peer == nullptr; i++) {
            if (i > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            socket_ptr attempt = socket_t::connect(host.c_str(), (int) (request.port + j));
            if (attempt == nullptr) {
                continue;
            }
            const uint32_t word = rank | (chosen ? RPC_COMM_CHOSEN_LINK : 0u);
            if (!attempt->send_data(&word, sizeof(word))) {
                continue;
            }
            if (!chosen) {
                uint32_t n_links = 0;
                if (!attempt->recv_data(&n_links, sizeof(n_links)) || n_links > 64) {
                    continue;
                }
                std::vector<rpc_comm_link> their_links(n_links);
                if (n_links > 0 && !attempt->recv_data(their_links.data(), n_links * sizeof(rpc_comm_link))) {
                    continue;
                }
                const std::string best = rpc_comm_best_link(our_links, their_links);
                const uint8_t move = !best.empty() && best != host;
                if (!attempt->send_data(&move, sizeof(move))) {
                    continue;
                }
                chosen = true;
                if (move) {
                    LOG_INFO("[%s] device %u: rank %u moves to rank %u's link at %s (given %s)\n", __func__, request.device,
                             rank, j, best.c_str(), host.c_str());
                    host = best;
                    i = 0;
                    continue;
                }
            }
            attempt->get_caps(local_caps);
            if (!attempt->send_data(local_caps, sizeof(local_caps)) || !attempt->recv_data(remote_caps, sizeof(remote_caps))) {
                LOG_DBG("[%s] rank %u: the handshake with rank %u failed, retrying\n", __func__, rank, j);
                continue;
            }
            attempt->update_caps(remote_caps);
            peer = attempt;
        }
        if (peer == nullptr) {
            LOG_ERROR("[%s] rank %u failed to connect to rank %u at %s:%u\n", __func__, rank, j, host.c_str(),
                      request.port + j);
            return false;
        }
        LOG_INFO("[%s] device %u: rank %u reaches rank %u at %s over %s\n", __func__, request.device, rank, j,
                 host.c_str(), peer->transport_name());
        peers[j] = peer;
    }
    for (uint32_t joined = 0; joined < world - 1 - rank; ) {
        socket_ptr peer = srv->accept();
        if (peer == nullptr) {
            LOG_ERROR("[%s] rank %u failed to accept a higher rank\n", __func__, rank);
            return false;
        }
        uint32_t word = 0;
        if (!peer->recv_data(&word, sizeof(word))) {
            LOG_DBG("[%s] rank %u: a connection closed before naming its rank\n", __func__, rank);
            continue;
        }
        const uint32_t peer_rank = word & ~RPC_COMM_CHOSEN_LINK;
        if (peer_rank <= rank || peer_rank >= world) {
            LOG_ERROR("[%s] rank %u refused a connection naming rank %u of %u\n", __func__, rank, peer_rank, world);
            return false;
        }
        if ((word & RPC_COMM_CHOSEN_LINK) == 0) {
            // a first connection learns our links; when the rank moves to another one, this connection is done
            const uint32_t n_links = (uint32_t) our_links.size();
            uint8_t move = 0;
            if (!peer->send_data(&n_links, sizeof(n_links)) ||
                (n_links > 0 && !peer->send_data(our_links.data(), n_links * sizeof(rpc_comm_link))) ||
                !peer->recv_data(&move, sizeof(move))) {
                LOG_DBG("[%s] rank %u: rank %u closed while learning the links\n", __func__, rank, peer_rank);
                continue;
            }
            if (move) {
                continue;
            }
        }
        if (!peer->recv_data(remote_caps, sizeof(remote_caps))) {
            LOG_DBG("[%s] rank %u: rank %u closed during the handshake\n", __func__, rank, peer_rank);
            continue;
        }
        peer->get_caps(local_caps);
        if (!peer->send_data(local_caps, sizeof(local_caps))) {
            LOG_DBG("[%s] rank %u: rank %u closed during the handshake\n", __func__, rank, peer_rank);
            continue;
        }
        peer->update_caps(remote_caps);
        LOG_INFO("[%s] device %u: rank %u accepted rank %u over %s\n", __func__, request.device, rank, peer_rank,
                 peer->transport_name());
        if (peers[peer_rank] == nullptr) {
            joined++;
        }
        peers[peer_rank] = peer;
    }
    if ((request.flags & RPC_COMM_FLAG_NCCL) != 0) {
        LOG_INFO("[%s] device %u: NCCL all-reduce takes two ranks, so %u ranks reduce through the host\n", __func__,
                 request.device, world);
    }
    state.peers    = std::move(peers);
    state.rank     = rank;
    state.world    = world;
    state.wire_f32 = true;
    state.seq      = 0;
    state.parts.assign(world, {});
    LOG_INFO("[%s] device %u joined a %u-rank mesh comm as rank %u\n", __func__, request.device, world, rank);
    return true;
}

// Every rank sends its partial to every peer and sums all of them in rank order, ((p0 + p1) + p2) + ..., so that the
// ranks hold the same bits afterwards: the mirrored activations each rank computes from the sum must not drift apart.
// The wire is always F32, since a partial rounded for the wire would have to be rounded on its own rank as well.
bool rpc_server::comm_allreduce_mesh(comm_state & state, ggml_backend_t backend, ggml_tensor * t_dst) {
    const size_t  nbytes = ggml_nbytes(t_dst);
    const int64_t ne     = ggml_nelements(t_dst);
    if (t_dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(t_dst) || nbytes / sizeof(float) != (size_t) ne) {
        LOG_ERROR("[%s] the mesh all-reduce needs a contiguous F32 tensor whose size matches its element count, got %s "
                  "with %" PRId64 " elements in %zu bytes\n", __func__, ggml_type_name(t_dst->type), ne, nbytes);
        return false;
    }
    if (state.async != nullptr && state.async->broken.load(std::memory_order_acquire)) {
        return false;
    }
    if (rpc_comm_async_enabled() && comm_allreduce_mesh_async(state, backend, t_dst)) {
        return true;
    }
    if (state.async != nullptr && state.async->broken.load(std::memory_order_acquire)) {
        return false;
    }
    // wait for the pending subgraph that produced this partial, and for any all-reduce still being exchanged off the
    // stream, which uses the same connections
    ggml_backend_synchronize(backend);
    while (state.async != nullptr && !state.async->idle()) {
        std::this_thread::yield();
    }
    state.seq++;
    std::vector<uint8_t> & own = state.parts[state.rank];
    if (own.size() < nbytes) {
        own.resize(nbytes);
    }
    state.sum.resize((size_t) ne);
    ggml_backend_tensor_get(t_dst, own.data(), 0, nbytes);
    if (!rpc_comm_mesh_exchange(state.peers, state.rank, state.world, state.seq, nbytes, own.data(), state.parts,
                                state.sum.data())) {
        return false;
    }
    ggml_backend_tensor_set(t_dst, state.sum.data(), 0, nbytes);
    return true;
}

// Queues the all-reduce on the backend's stream and returns at once; false when the backend cannot, leaving the
// all-reduce to the synchronous path
bool rpc_server::comm_allreduce_mesh_async(comm_state & state, ggml_backend_t backend, ggml_tensor * t_dst) {
    if (state.async == nullptr) {
        auto host_alloc = (rpc_comm_host_alloc_t) rpc_backend_proc(backend, "ggml_backend_cuda_comm_host_alloc");
        auto host_free  = (rpc_comm_host_free_t)  rpc_backend_proc(backend, "ggml_backend_cuda_comm_host_free");
        auto can_stage  = (rpc_comm_can_stage_t)  rpc_backend_proc(backend, "ggml_backend_cuda_comm_can_stage");
        auto stage      = (rpc_comm_stage_t)      rpc_backend_proc(backend, "ggml_backend_cuda_comm_stage");
        if (host_alloc == nullptr || host_free == nullptr || can_stage == nullptr || stage == nullptr) {
            return false;
        }
        // no queued work may still run on the stream while it is probed
        ggml_backend_synchronize(backend);
        std::unique_ptr<rpc_comm_async, rpc_comm_async_deleter> async(new rpc_comm_async());
        async->host_alloc = host_alloc;
        async->host_free  = host_free;
        async->stage      = stage;
        async->backend    = backend;
        async->flags      = (uint32_t *) host_alloc(2 * sizeof(uint32_t));
        if (async->flags == nullptr || !can_stage(backend, &async->flags[0], &async->flags[1])) {
            LOG_WARN("[%s] the device cannot queue all-reduces on its stream; they wait for it instead\n", __func__);
            host_free(async->flags);
            async->flags = nullptr;
            async->host_free = nullptr;
            return false;
        }
        async->peers = &state.peers;
        async->parts = &state.parts;
        async->rank  = state.rank;
        async->world = state.world;
        rpc_comm_async * a = async.get();
        async->thread = std::thread([a] { a->run(); });
        state.async = std::move(async);
        LOG_INFO("[%s] rank %u queues its all-reduces on the device's stream\n", __func__, state.rank);
    }
    rpc_comm_async & a = *state.async;
    if (a.flags == nullptr) {
        return false;
    }
    const size_t nbytes = ggml_nbytes(t_dst);
    // a larger partial, or a flag about to wrap: drain the stream and the exchange first
    if (nbytes > a.capacity || a.next_flag >= 0xfffffff0u) {
        ggml_backend_synchronize(backend);
        while (!a.idle()) {
            std::this_thread::yield();
        }
        if (nbytes > a.capacity) {
            a.host_free(a.out);
            a.host_free(a.in);
            const size_t capacity = std::max(nbytes, 2 * a.capacity);
            a.out = (uint8_t *) a.host_alloc(capacity);
            a.in  = (float *)   a.host_alloc(capacity);
            if (a.out == nullptr || a.in == nullptr) {
                LOG_ERROR("[%s] could not allocate %zu bytes of host memory for the all-reduce\n", __func__, capacity);
                a.broken.store(true, std::memory_order_release);
                return false;
            }
            a.capacity = capacity;
        }
        if (a.next_flag >= 0xfffffff0u) {
            __atomic_store_n(&a.flags[0], 0u, __ATOMIC_RELEASE);
            __atomic_store_n(&a.flags[1], 0u, __ATOMIC_RELEASE);
            a.next_flag = 0;
            a.queued.store(0, std::memory_order_release);
            a.processed.store(0, std::memory_order_release);
        }
    }
    // no more than RING all-reduces ahead of the exchange
    while (a.next_flag - a.processed.load(std::memory_order_acquire) >= rpc_comm_async::RING - 1) {
        std::this_thread::yield();
    }
    const uint32_t flag = ++a.next_flag;
    state.seq++;
    a.jobs[flag % rpc_comm_async::RING] = { flag, state.seq, nbytes };
    if (!a.stage(backend, t_dst, a.out, a.in, &a.flags[0], &a.flags[1], flag)) {
        LOG_ERROR("[%s] could not queue all-reduce %" PRIu64 " on the device's stream\n", __func__, state.seq);
        a.broken.store(true, std::memory_order_release);
        return false;
    }
    a.queued.store(flag, std::memory_order_release);
    return true;
}

bool rpc_server::comm_free(const rpc_msg_comm_free_req & request) {
    if (request.device >= backends.size()) {
        return false;
    }
    comm_state & state = comm_states[request.device];
    if (state.async != nullptr) {
        // the stream and the exchange drain before the connections they use close
        ggml_backend_synchronize(backends[request.device]);
        state.async.reset();
    }
    if (!state.peers.empty()) {
        LOG_INFO("[%s] device %u: rank %u of the %u-rank mesh ran %" PRIu64 " all-reduces\n", __func__, request.device,
                 state.rank, state.world, state.seq);
    }
    if (state.nccl != nullptr) {
        state.nccl_free(state.nccl);
    }
    comm_states[request.device] = comm_state();
    return true;
}

bool rpc_server::get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    size_t free, total;
    ggml_backend_dev_t dev = ggml_backend_get_device(backends[dev_id]);
    ggml_backend_dev_memory(dev, &free, &total);
    response.free_mem = free;
    response.total_mem = total;
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, dev_id,
             rpc_log_format_size(response.free_mem).c_str(), rpc_log_format_size(response.total_mem).c_str());
    return true;
}

rpc_server::~rpc_server() {
    // a client that disconnects without COMM_FREE still leaves no NCCL communicator behind; abort it rather than
    // destroy it, since the client may have sent a collective to this rank only, which the peer never joins
    for (size_t device = 0; device < comm_states.size(); device++) {
        comm_state & state = comm_states[device];
        if (state.nccl != nullptr && state.nccl_abort != nullptr) {
            state.nccl_abort(state.nccl);
            state.nccl = nullptr;
            LOG_INFO("[%s] device %zu: aborted the NCCL communicator the client left open\n", __func__, device);
        }
    }
    for (auto buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
}

static void rpc_serve_client(const std::vector<ggml_backend_t> & backends, const char * cache_dir,
                             socket_ptr sock) {
    rpc_server server(backends, cache_dir);
    uint8_t cmd;
    if (!sock->recv_data(&cmd, 1)) {
        return;
    }
    if (cmd != RPC_CMD_HELLO) {
        LOG_ERROR("Expected HELLO command, update client\n");
        return;
    }

    // Read input_size and validate protocol version
    uint64_t hello_input_size;
    if (!sock->recv_data(&hello_input_size, sizeof(hello_input_size))) {
        return;
    }

    if (hello_input_size != sizeof(rpc_msg_hello_req)) {
        LOG_ERROR("HELLO request size mismatch (%zu vs %zu) — client needs upgrade to protocol v%d.x\n",
                       (size_t)hello_input_size, sizeof(rpc_msg_hello_req), RPC_PROTO_MAJOR_VERSION);
        return;
    }

    rpc_msg_hello_req req = {};
    if (!sock->recv_data(&req, sizeof(req))) {
        return;
    }

    rpc_msg_hello_rsp rsp = {};
    server.hello(rsp);
    // Advertise server transport capabilities based on client's caps
    sock->get_caps(rsp.conn_caps);
    if (!send_msg(sock, &rsp, sizeof(rsp))) {
        return;
    }

    // Activate transport upgrade using client's caps
    sock->update_caps(req.conn_caps);
    LOG_DBG("[%s] client connected, transport: %s\n", __func__, sock->transport_name());
    while (true) {
        if (!sock->recv_data(&cmd, 1)) {
            break;
        }
        if (cmd >= RPC_CMD_COUNT) {
            // fail fast if the command is invalid
            LOG_ERROR("Unknown command: %d\n", cmd);
            break;
        }
        LOG_DBG2("[%s] %s\n", __func__, rpc_cmd_name((enum rpc_cmd)cmd));
        switch (cmd) {
            case RPC_CMD_HELLO: {
                // HELLO command is handled above
                return;
            }
            case RPC_CMD_DEVICE_COUNT: {
                if (!recv_msg(sock, nullptr, 0)) {
                    return;
                }
                rpc_msg_device_count_rsp response;
                response.device_count = backends.size();
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_ALLOC_BUFFER: {
                rpc_msg_alloc_buffer_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_alloc_buffer_rsp response;
                if (!server.alloc_buffer(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_ALLOC_SIZE: {
                rpc_msg_get_alloc_size_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_alloc_size_rsp response;
                if (!server.get_alloc_size(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_ALIGNMENT: {
                rpc_msg_get_alignment_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_alignment_rsp response;
                if (!server.get_alignment(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_MAX_SIZE: {
                rpc_msg_get_max_size_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_max_size_rsp response;
                if (!server.get_max_size(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_BUFFER_GET_BASE: {
                rpc_msg_buffer_get_base_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_buffer_get_base_rsp response;
                if (!server.buffer_get_base(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_FREE_BUFFER: {
                rpc_msg_free_buffer_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.free_buffer(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_BUFFER_CLEAR: {
                rpc_msg_buffer_clear_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.buffer_clear(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_MEMSET_TENSOR: {
                rpc_msg_memset_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.memset_tensor(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.set_tensor(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_2D: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.set_tensor_2d(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_TENSOR_2D: {
                rpc_msg_get_tensor_2d_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                std::vector<uint8_t> response;
                if (!server.get_tensor_2d(request, response)) {
                    return;
                }
                if (!send_msg(sock, response.data(), response.size())) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_HASH: {
                rpc_msg_set_tensor_hash_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_set_tensor_hash_rsp response;
                if (!server.set_tensor_hash(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_INIT_TENSOR: {
                rpc_msg_init_tensor_req request;
                if (!recv_msg(sock, &request,sizeof(request))) {
                    return;
                }
                if (!server.init_tensor(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_TENSOR: {
                rpc_msg_get_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                std::vector<uint8_t> response;
                if (!server.get_tensor(request, response)) {
                    return;
                }
                if (!send_msg(sock, response.data(), response.size())) {
                    return;
                }
                break;
            }
            case RPC_CMD_COPY_TENSOR: {
                rpc_msg_copy_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_copy_tensor_rsp response;
                if (!server.copy_tensor(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_COMPUTE: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.graph_compute(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_RECOMPUTE: {
                rpc_msg_graph_recompute_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.graph_recompute(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_COMM_INIT: {
                rpc_msg_comm_init_req request;
                if (!recv_comm_init_req(sock, request)) {
                    return;
                }
                rpc_msg_comm_init_rsp response;
                if (!server.comm_init(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_COMM_ALLREDUCE: {
                rpc_msg_comm_allreduce_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.comm_allreduce(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_COMM_FREE: {
                rpc_msg_comm_free_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.comm_free(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_DEVICE_MEMORY: {
                rpc_msg_get_device_memory_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_device_memory_rsp response;
                if (!server.get_device_memory(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            default: {
                LOG_ERROR("Unhandled command: %s\n", rpc_cmd_name((enum rpc_cmd)cmd));
                return;
            }
        }
    }
}

void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                   size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices) {
    if (n_devices == 0 || devices == nullptr) {
        LOG_ERROR("Invalid arguments to ggml_backend_rpc_start_server\n");
        return;
    }
    std::vector<ggml_backend_t> backends;
    LOG_INFO("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_VERSION,
        RPC_PROTO_MINOR_VERSION,
        RPC_PROTO_PATCH_VERSION);
    LOG_INFO("  endpoint       : %s\n", endpoint);
    LOG_INFO("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    LOG_INFO("Devices:\n");
    for (size_t i = 0; i < n_devices; i++) {
        auto dev = devices[i];
        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);
        LOG_INFO("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                 total / 1024 / 1024, free / 1024 / 1024);
        auto backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            LOG_ERROR("Failed to create backend for device %s\n", dev->iface.get_name(dev));
            return;
        }
        backends.push_back(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg) {
            auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (ggml_backend_set_n_threads_fn) {
                ggml_backend_set_n_threads_fn(backend, n_threads);
            }
        }
    }

    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        return;
    }

#ifdef GGML_RPC_RDMA
    LOG_INFO("  transport      : TCP (RDMA auto-negotiate enabled)\n");
#else
    LOG_INFO("  transport      : TCP\n");
#endif // GGML_RPC_RDMA
    if (!rpc_transport_init()) {
        LOG_ERROR("Failed to initialize RPC transport\n");
        return;
    }
    auto server_socket = socket_t::create_server(host.c_str(), port);
    if (server_socket == nullptr) {
        LOG_ERROR("Failed to create server socket\n");
        return;
    }
    while (true) {
        auto client_socket = server_socket->accept();
        if (client_socket == nullptr) {
            LOG_ERROR("Failed to accept client connection\n");
            return;
        }
        LOG_INFO("Accepted client connection\n");
        rpc_serve_client(backends, cache_dir, client_socket);
        LOG_INFO("Client connection closed\n");
    }
    rpc_transport_shutdown();
    for (auto backend : backends) {
        ggml_backend_free(backend);
    }
}

static const char * ggml_backend_rpc_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ctx->name.c_str();
}

static const char * ggml_backend_rpc_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ctx->description.c_str();
}

static void ggml_backend_rpc_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ggml_backend_rpc_get_device_memory(ctx->endpoint.c_str(), ctx->device, free, total);
}

static enum ggml_backend_dev_type ggml_backend_rpc_device_get_type(ggml_backend_dev_t dev) {
    // TODO: obtain value from the server
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_rpc_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rpc_device_get_name(dev);
    props->description = ggml_backend_rpc_device_get_description(dev);
    props->type        = ggml_backend_rpc_device_get_type(dev);
    ggml_backend_rpc_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ true,
        /* .mmap_support          = */ true,
    };
}

static ggml_backend_t ggml_backend_rpc_device_init(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ggml_backend_rpc_init(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_rpc_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ggml_backend_rpc_buffer_type(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(dev);
}

static bool ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    //TODO: call the remote backend and cache the results
    return true;
}

static bool ggml_backend_rpc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (!buft || buft->iface.get_name != ggml_backend_rpc_buffer_type_name) {
        return false;
    }
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *)dev->context;
    return buft_ctx->endpoint == dev_ctx->endpoint && buft_ctx->device == dev_ctx->device;
}

static ggml_backend_event_t ggml_backend_rpc_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    return dispatcher->event_new(dev);
}

static void ggml_backend_rpc_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_free(event);
}

static void ggml_backend_rpc_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_synchronize(event);
}

static const struct ggml_backend_device_i ggml_backend_rpc_device_i = {
    /* .get_name             = */ ggml_backend_rpc_device_get_name,
    /* .get_description      = */ ggml_backend_rpc_device_get_description,
    /* .get_memory           = */ ggml_backend_rpc_device_get_memory,
    /* .get_type             = */ ggml_backend_rpc_device_get_type,
    /* .get_props            = */ ggml_backend_rpc_device_get_props,
    /* .init_backend         = */ ggml_backend_rpc_device_init,
    /* .get_buffer_type      = */ ggml_backend_rpc_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_rpc_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rpc_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ ggml_backend_rpc_device_event_new,
    /* .event_free           = */ ggml_backend_rpc_device_event_free,
    /* .event_synchronize    = */ ggml_backend_rpc_device_event_synchronize,
};

// backend reg interface

struct ggml_backend_rpc_reg_context {
    std::string                     name;
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_rpc_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    return ctx ? ctx->name.c_str() : "RPC";
}

static size_t ggml_backend_rpc_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    return ctx ? ctx->devices.size() : 0;
}

static ggml_backend_dev_t ggml_backend_rpc_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    if (ctx == nullptr) {
        GGML_ABORT("The RPC backend does not have enumerated devices - use ggml_backend_rpc_add_server instead");
    } else {
        GGML_ASSERT(index < ctx->devices.size());
        return ctx->devices[index];
    }
}

// Pairwise allreduce between two RPC servers over a direct server-to-server connection.
// The client only sends fire-and-forget COMM_ALLREDUCE commands; the tensor data is
// exchanged between the servers and never passes through the client.
struct ggml_backend_rpc_comm_context {
    struct rank_info {
        std::string                     endpoint;
        uint32_t                        device;
        std::shared_ptr<rpc_dispatcher> dispatcher;
    };
    std::vector<rank_info> ranks;
};

static void ggml_backend_rpc_comm_free(void * comm_ctx_v) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    if (comm_ctx == nullptr) {
        return;
    }
    for (const auto & rank : comm_ctx->ranks) {
        auto request = std::make_shared<rpc_msg_comm_free_req>();
        request->device = rank.device;
        rank.dispatcher->send(RPC_CMD_COMM_FREE, request, sizeof(*request));
        rank.dispatcher->busy_spin_release();
    }
    delete comm_ctx;
}

// rank i reaches rank j (j < i) of a mesh at mesh_hosts[i][j]: rank j's serving host, unless GGML_RPC_COMM_HOSTS names
// another as a comma-separated "i:j=host" list, e.g. the address rank j has on the link the two share
static bool rpc_comm_mesh_hosts(const std::vector<std::string> & endpoints, std::vector<std::vector<std::string>> & mesh_hosts) {
    const size_t world = endpoints.size();
    mesh_hosts.assign(world, std::vector<std::string>(world));
    for (size_t j = 0; j < world; j++) {
        std::string host;
        int port;
        if (!parse_endpoint(endpoints[j], host, port)) {
            return false;
        }
        for (size_t i = j + 1; i < world; i++) {
            mesh_hosts[i][j] = host;
        }
    }
    if (const char * env = std::getenv("GGML_RPC_COMM_HOSTS")) {
        std::string list(env);
        size_t start = 0;
        while (start < list.size()) {
            size_t end = list.find(',', start);
            if (end == std::string::npos) {
                end = list.size();
            }
            const std::string entry = list.substr(start, end - start);
            start = end + 1;
            if (entry.empty()) {
                continue;
            }
            unsigned i = 0, j = 0;
            int consumed = 0;
            if (sscanf(entry.c_str(), "%u:%u=%n", &i, &j, &consumed) != 2 || consumed == 0 || i >= world || j >= i ||
                    (size_t) consumed >= entry.size()) {
                LOG_WARN("%s: GGML_RPC_COMM_HOSTS entry '%s' is not i:j=host with j < i < %zu\n", __func__, entry.c_str(), world);
                return false;
            }
            mesh_hosts[i][j] = entry.substr((size_t) consumed);
        }
    }
    for (size_t i = 0; i < world; i++) {
        for (size_t j = 0; j < i; j++) {
            if (mesh_hosts[i][j].empty() || mesh_hosts[i][j].size() >= sizeof(rpc_msg_comm_init_req::hosts[0])) {
                return false;
            }
        }
    }
    return true;
}

static void * ggml_backend_rpc_comm_init(ggml_backend_t * backends, size_t n_backends) {
    if (n_backends < 2 || n_backends > RPC_COMM_MAX_WORLD || std::getenv("GGML_RPC_NO_COMM") != nullptr) {
        return nullptr;
    }
    std::vector<ggml_backend_rpc_comm_context::rank_info> ranks;
    ranks.reserve(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        if (!ggml_backend_is_rpc(backends[i])) {
            return nullptr;
        }
        ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backends[i]->context;
        // one rank per endpoint: a server processes its socket sequentially, so a second
        // COMM_INIT on the same connection would deadlock behind the first
        for (const auto & rank : ranks) {
            if (rank.endpoint == rpc_ctx->endpoint) {
                LOG_WARN("%s: multiple ranks on endpoint %s are not supported\n", __func__, rpc_ctx->endpoint.c_str());
                return nullptr;
            }
        }
        ranks.push_back({rpc_ctx->endpoint, rpc_ctx->device, rpc_ctx->dispatcher});
    }

    // TODO: simplify this logic
    // rank 1 connects to rank 0 on its serving host; endpoints must be mutually reachable
    // (e.g. do not bind the servers to 127.0.0.1 when they run on different machines)
    std::string host0;
    int port0;
    if (!parse_endpoint(ranks[0].endpoint, host0, port0)) {
        return nullptr;
    }
    const uint32_t comm_port = (uint32_t) port0 + 1000;
    // GGML_RPC_COMM_NCCL=1 / GGML_RPC_COMM_WIRE_F32=1 here, on the client, decide for both ranks
    auto env_on = [](const char * name) {
        const char * v = std::getenv(name);
        return v != nullptr && std::atoi(v) != 0;
    };
    const bool     mesh       = n_backends > 2 || env_on("GGML_RPC_COMM_MESH");
    const uint32_t comm_flags = (env_on("GGML_RPC_COMM_NCCL")     ? RPC_COMM_FLAG_NCCL     : 0u) |
                                (env_on("GGML_RPC_COMM_WIRE_F32") ? RPC_COMM_FLAG_WIRE_F32 : 0u) |
                                (mesh                             ? RPC_COMM_FLAG_MESH     : 0u);
    // rank 1 reaches rank 0 at this address instead, e.g. rank 0 on the link the two share
    if (const char * comm_host = std::getenv("GGML_RPC_COMM_HOST")) {
        host0 = comm_host;
    }
    if (host0.size() >= 64) {
        return nullptr;
    }
    std::vector<std::vector<std::string>> mesh_hosts;
    if (mesh) {
        std::vector<std::string> endpoints;
        for (const auto & rank : ranks) {
            endpoints.push_back(rank.endpoint);
        }
        if (!rpc_comm_mesh_hosts(endpoints, mesh_hosts)) {
            return nullptr;
        }
    }

    for (const auto & rank : ranks) {
        rank.dispatcher->busy_spin_acquire();
    }

    // Send all init requests before reading any response: rank 0 blocks in accept
    // until rank 1 has connected.
    std::vector<rpc_msg_comm_init_rsp> responses(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        auto request = std::make_shared<rpc_msg_comm_init_req>();
        request->device = ranks[i].device;
        request->rank   = (uint32_t) i;
        request->world  = (uint32_t) n_backends;
        request->port   = comm_port;
        request->flags  = comm_flags;
        if (i > 0) {
            memcpy(request->host, host0.c_str(), host0.size());
        }
        for (size_t j = 0; mesh && j < i; j++) {
            memcpy(request->hosts[j], mesh_hosts[i][j].c_str(), mesh_hosts[i][j].size());
        }
        // with no flag set this is upstream's COMM_INIT, which a server without the flags also reads; two ranks never
        // send the mesh hosts, so a server without the mesh still reads a pairwise request with flags
        const size_t request_size = mesh ? sizeof(*request) :
                                    comm_flags != 0 ? RPC_COMM_INIT_REQ_SIZE_PAIR : RPC_COMM_INIT_REQ_SIZE_NO_FLAGS;
        ranks[i].dispatcher->send_async(RPC_CMD_COMM_INIT, request, request_size, &responses[i], sizeof(responses[i]));
    }
    for (size_t i = 0; i < n_backends; i++) {
        ranks[i].dispatcher->synchronize();
    }
    bool ok = true;
    for (size_t i = 0; i < n_backends; i++) {
        if (!responses[i].ok) {
            LOG_WARN("%s: rank %zu (%s) failed to initialize\n", __func__, i, ranks[i].endpoint.c_str());
            ok = false;
        }
    }
    if (!ok) {
        for (const auto & rank : ranks) {
            rank.dispatcher->busy_spin_release();
        }
        return nullptr;
    }
    if (!mesh) {
        LOG_INFO("%s: pairwise communicator initialized (%s <-> %s)\n", __func__,
                 ranks[0].endpoint.c_str(), ranks[1].endpoint.c_str());
    } else {
        LOG_INFO("%s: %zu-rank mesh communicator initialized\n", __func__, n_backends);
    }
    return new ggml_backend_rpc_comm_context{std::move(ranks)};
}

static bool ggml_backend_rpc_comm_allreduce_tensor(void * comm_ctx_v, ggml_tensor ** tensors) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    if (comm_ctx == nullptr) {
        return false;
    }
    const size_t n_ranks = comm_ctx->ranks.size();
    const int64_t ne = ggml_nelements(tensors[0]);
    if (ne == 0) {
        return true;
    }
    for (size_t i = 0; i < n_ranks; i++) {
        if (tensors[i] == nullptr || tensors[i]->type != GGML_TYPE_F32 || ggml_nelements(tensors[i]) != ne ||
                !ggml_is_contiguously_allocated(tensors[i]) ||
                tensors[i]->buffer == nullptr || !ggml_backend_buffer_is_rpc(tensors[i]->buffer)) {
            return false;
        }
        // a rank with a disabled node has garbage in its partial and must contribute zeros,
        // which only the fallback path handles
        if ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            return false;
        }
    }
    for (size_t i = 0; i < n_ranks; i++) {
        auto request = std::make_shared<rpc_msg_comm_allreduce_req>();
        request->device = comm_ctx->ranks[i].device;
        request->tensor = serialize_tensor(tensors[i]);
        comm_ctx->ranks[i].dispatcher->send_async(RPC_CMD_COMM_ALLREDUCE, request, sizeof(*request));
    }
    return true;
}

static void * ggml_backend_rpc_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_rpc_add_server") == 0) {
        return (void *)ggml_backend_rpc_add_server;
    }
    if (std::strcmp(name, "ggml_backend_rpc_start_server") == 0) {
        return (void *)ggml_backend_rpc_start_server;
    }
    if (std::strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *)ggml_backend_rpc_comm_init;
    }
    if (std::strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *)ggml_backend_rpc_comm_free;
    }
    if (std::strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *)ggml_backend_rpc_comm_allreduce_tensor;
    }
    return NULL;

    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i ggml_backend_rpc_reg_i = {
    /* .get_name         = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count = */ ggml_backend_rpc_reg_get_device_count,
    /* .get_device       = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_reg(void) {
    static struct ggml_backend_reg ggml_backend_rpc_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_rpc_reg;
}

static uint32_t ggml_backend_rpc_get_device_count(const char * endpoint) {
    auto dispatcher = get_dispatcher(endpoint);
    rpc_msg_device_count_rsp response;
    dispatcher->send(RPC_CMD_DEVICE_COUNT, nullptr, 0, &response, sizeof(response));
    return response.device_count;
}

static const ggml_backend_reg_i ggml_backend_rpc_reg_interface = {
    /* .get_name          = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count  = */ ggml_backend_rpc_reg_get_device_count,
    /* .get_device        = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint) {
    static std::unordered_map<std::string, ggml_backend_reg_t> reg_map;
    static std::mutex mutex;
    static uint32_t dev_id = 0;
    std::lock_guard<std::mutex> lock(mutex);
    if (reg_map.find(endpoint) != reg_map.end()) {
        LOG_DBG3("[%s] endpoint %s already registered\n", __func__, endpoint);
        return reg_map[endpoint];
    }
    uint32_t dev_count = ggml_backend_rpc_get_device_count(endpoint);
    LOG_DBG2("[%s] endpoint: %s, device count: %u\n", __func__, endpoint, dev_count);
    if (dev_count == 0) {
        return nullptr;
    }
    ggml_backend_rpc_reg_context * ctx = new ggml_backend_rpc_reg_context;
    ctx->name = "RPC[" + std::string(endpoint) + "]";
    for (uint32_t ind = 0; ind < dev_count; ind++) {
        std::string dev_name = "RPC" + std::to_string(dev_id);
        std::string dev_desc = std::string(endpoint);
        ggml_backend_rpc_device_context * dev_ctx = new ggml_backend_rpc_device_context {
            /* .endpoint    = */    endpoint,
            /* .device      = */    ind,
            /* .name        = */    dev_name,
            /* .description = */    dev_desc,
        };

        ggml_backend_dev_t dev = new ggml_backend_device {
            /* .iface   = */ ggml_backend_rpc_device_i,
            /* .reg     = */ ggml_backend_rpc_reg(),
            /* .context = */ dev_ctx,
        };
        ctx->devices.push_back(dev);
        dev_id++;
    }
    ggml_backend_reg_t reg = new ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_interface,
        /* .context     = */ ctx
    };
    reg_map[endpoint] = reg;
    return reg;
}


GGML_BACKEND_DL_IMPL(ggml_backend_rpc_reg)
