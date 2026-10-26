// A meta backend keeps one graph split per compute buffer (graph slots): the split of the graph it computed last is
// the active one and the splits of graphs from other compute buffers are parked under their buffer. A compute buffer
// that is freed tells every live meta backend to forget the split kept for it, and the thread freeing it can be
// another context's compute thread: a model unloaded while a second model decodes, or a drafter's scheduler recreated
// while the target computes. This test runs that interleaving on a meta device over the CPU device: one meta backend
// alternates graphs from two compute buffers, parking and activating their splits, while a second meta backend on
// another thread allocates, computes and frees graphs of its own. Under ThreadSanitizer it fails when the free's
// notification reaches a backend's slot state without the synchronization its compute path uses; on every build it
// checks that both backends keep computing the right values while buffers come and go.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

static ggml_backend_meta_split_state mirrored_split_state(const ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
}

// out = (a + b) * scale, allocated in a compute buffer of its own the way a scheduler allocates the graph it computes
struct slot_graph {
    ggml_context * ctx    = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_cgraph *  gf     = nullptr;
    ggml_tensor *  a      = nullptr;
    ggml_tensor *  b      = nullptr;
    ggml_tensor *  out    = nullptr;
    float          scale  = 0.0f;

    slot_graph(ggml_backend_t backend, uint64_t uid, float scale, int64_t n) : scale(scale) {
        ggml_init_params params = {
            /* .mem_size   = */ ggml_tensor_overhead()*8 + ggml_graph_overhead(),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ctx = ggml_init(params);
        a   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        b   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        ggml_set_input(a);
        ggml_set_input(b);
        out = ggml_scale(ctx, ggml_add(ctx, a, b), scale);
        ggml_set_output(out);
        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        gf->uid = uid;
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, gf));
    }

    ~slot_graph() {
        ggml_gallocr_free(galloc); // frees the compute buffer, which every live meta backend is told to forget
        ggml_free(ctx);
    }

    slot_graph(const slot_graph &) = delete;
    slot_graph & operator=(const slot_graph &) = delete;

    // computes the graph for inputs filled with va and vb and checks every output element is (va + vb) * scale
    bool compute_and_check(ggml_backend_t backend, float va, float vb, const char * who) {
        const std::vector<float> xa(ggml_nelements(a), va);
        const std::vector<float> xb(ggml_nelements(b), vb);
        ggml_backend_tensor_set(a, xa.data(), 0, ggml_nbytes(a));
        ggml_backend_tensor_set(b, xb.data(), 0, ggml_nbytes(b));

        const ggml_status status = ggml_backend_graph_compute(backend, gf);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: graph compute failed with status %d\n", who, (int) status);
            return false;
        }

        std::vector<float> y(ggml_nelements(out));
        ggml_backend_tensor_get(out, y.data(), 0, ggml_nbytes(out));
        const float expected = (va + vb) * scale;
        for (size_t i = 0; i < y.size(); i++) {
            if (y[i] != expected) {
                fprintf(stderr, "%s: out[%zu] = %f, expected %f\n", who, i, y[i], expected);
                return false;
            }
        }
        return true;
    }
};

int main() {
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(cpu_dev != nullptr);

    ggml_backend_dev_t simple_devs[1] = { cpu_dev };
    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(simple_devs, 1, mirrored_split_state, nullptr);
    GGML_ASSERT(meta_dev != nullptr);

    // two meta backends of one device, as two models, or a model and its drafter, hold
    ggml_backend_t backend = ggml_backend_dev_init(meta_dev, nullptr);
    ggml_backend_t other   = ggml_backend_dev_init(meta_dev, nullptr);
    GGML_ASSERT(backend != nullptr && other != nullptr);
    ggml_backend_cpu_set_n_threads(ggml_backend_meta_simple_backend(backend, 0), 1);
    ggml_backend_cpu_set_n_threads(ggml_backend_meta_simple_backend(other,   0), 1);

    constexpr int      n_rounds_min       = 1000;    // rounds this backend computes at least
    constexpr long     n_other_graphs_min = 50;      // compute buffers the other backend must free meanwhile
    constexpr uint64_t uid_other0         = 1000000; // the other backend's graphs take uids of their own, one per graph
    const auto         t_limit            = std::chrono::steady_clock::now() + std::chrono::seconds(30);

    std::atomic<bool> done{false};
    std::atomic<bool> other_ok{true};
    std::atomic<long> other_graphs{0};

    // the other context: allocates a graph in a fresh compute buffer, computes it, and frees it, until told to stop
    std::thread other_thread([&]() {
        for (uint64_t i = 0; !done.load(std::memory_order_relaxed); i++) {
            slot_graph g(other, uid_other0 + i, 1.0f, 32);
            if (!g.compute_and_check(other, 1.0f, 1.0f, "other")) {
                other_ok.store(false);
                return;
            }
            other_graphs.fetch_add(1, std::memory_order_relaxed);
        }
    });

    // this context: graphs from two compute buffers, alternating, so each compute parks one split and activates the other
    slot_graph g1(backend, 1, 0.5f, 64);
    slot_graph g2(backend, 2, 2.0f, 64);
    bool ok     = true;
    int  rounds = 0;
    while (ok && std::chrono::steady_clock::now() < t_limit &&
            (rounds < n_rounds_min || other_graphs.load(std::memory_order_relaxed) < n_other_graphs_min)) {
        ok = g1.compute_and_check(backend, 1.0f, 2.0f, "slot 1") &&
             g2.compute_and_check(backend, 3.0f, 4.0f, "slot 2");
        rounds++;
    }

    done.store(true);
    other_thread.join();

    if (!ok || !other_ok.load()) {
        fprintf(stderr, "FAIL: wrong results while compute buffers were freed (%d rounds, %ld graphs freed by the other backend)\n",
            rounds, other_graphs.load());
        return EXIT_FAILURE;
    }
    if (other_graphs.load() < n_other_graphs_min) {
        fprintf(stderr, "FAIL: the other backend freed %ld compute buffers while this one computed %d rounds, fewer than the %ld the test needs\n",
            other_graphs.load(), rounds, n_other_graphs_min);
        return EXIT_FAILURE;
    }
    printf("OK: %d rounds over two compute buffers while the other backend allocated and freed %ld graphs\n",
        rounds, other_graphs.load());

    ggml_backend_free(other);
    ggml_backend_free(backend);
    return EXIT_SUCCESS;
}
