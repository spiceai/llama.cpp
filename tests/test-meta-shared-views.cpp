// Two meta backends whose graphs view the same buffer: a model's weights, viewed by the model's own graph and by its
// drafter's (a DSpark drafter multiplies a rows view of the target's output head). A meta buffer keeps the device tensors
// of views made of its tensors in two compute containers that rotate as graphs are rebuilt: a graph's views go into the
// current one, and each rebuild resets the other one for the next graph. A backend reuses the split it built for a graph
// it computes again (same uid), so that split points at its views' device tensors in the container of its time. Another
// backend's two rebuilds reset that container and give its memory to the other backend's views.
//
// The test: backend A computes out = x * view(s, 0..n), then backend B allocates and computes two fresh graphs, each
// out = x * view(s, n..2n), and backend A computes its graph again without allocating it, as a scheduler reuses the graph
// it allocated last. Every compute must give x * s[0..n) on A and x * s[n..2n) on B. A reused split that still points
// into the reset container reads B's view, s[n..2n), instead of its own. It runs twice: A computing its whole graph, and
// A computing only the product, the view being a source made in an earlier part of the graph (a scheduler computes a
// graph in splits, whose graphs have no leafs and need not hold the views their nodes read).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static ggml_backend_meta_split_state mirrored_split_state(const ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
}

static constexpr int64_t N = 64;

// out = x * view(s, offset, N), in a compute buffer of its own, allocated once the way a scheduler allocates a graph
struct view_graph {
    ggml_context * ctx    = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_cgraph *  gf     = nullptr;
    ggml_tensor *  x      = nullptr;
    ggml_tensor *  out    = nullptr;
    int64_t        offset = 0;
    ggml_cgraph    part   = {};      // the product alone, when computed as a split of gf
    bool           split  = false;

    view_graph(ggml_backend_t backend, ggml_tensor * s, int64_t offset, uint64_t uid, bool split = false)
        : offset(offset), split(split) {
        ggml_init_params params = {
            /* .mem_size   = */ ggml_tensor_overhead()*8 + ggml_graph_overhead(),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ctx = ggml_init(params);
        x   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
        ggml_set_input(x);
        out = ggml_mul(ctx, x, ggml_view_1d(ctx, s, N, offset*ggml_element_size(s)));
        ggml_set_output(out);
        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        gf->uid = uid;
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, gf));
        if (split) {
            // nodes: the view, then the product; the split holds the product alone, as the scheduler's splits do
            GGML_ASSERT(ggml_graph_n_nodes(gf) == 2 && ggml_graph_node(gf, 0)->op == GGML_OP_VIEW && ggml_graph_node(gf, 1) == out);
            // what ggml_graph_view (not exported) returns for nodes [1, 2)
            part           = *gf;
            part.size      = 0;
            part.n_nodes   = 1;
            part.n_leafs   = 0;
            part.nodes     = gf->nodes + 1;
            part.grads     = nullptr;
            part.grad_accs = nullptr;
            part.leafs     = nullptr;
            part.uid       = uid;
        }
    }

    ~view_graph() {
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
    }

    view_graph(const view_graph &) = delete;
    view_graph & operator=(const view_graph &) = delete;

    // computes the graph for x[i] = i + 1 and checks out[i] = x[i] * s[offset + i], where s[k] = 1000 + k
    bool compute_and_check(ggml_backend_t backend, const char * who) {
        std::vector<float> xv(N);
        for (int64_t i = 0; i < N; i++) {
            xv[i] = (float) (i + 1);
        }
        ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
        if (ggml_backend_graph_compute(backend, split ? &part : gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: graph compute failed\n", who);
            return false;
        }
        std::vector<float> y(N);
        ggml_backend_tensor_get(out, y.data(), 0, ggml_nbytes(out));
        for (int64_t i = 0; i < N; i++) {
            const float want = xv[i] * (float) (1000 + offset + i);
            if (y[i] != want) {
                fprintf(stderr, "%s: out[%lld] = %.1f, want %.1f (x * s[%lld]); x * s[%lld] would be %.1f\n", who,
                    (long long) i, y[i], want, (long long) (offset + i), (long long) ((offset + N) % (2*N) + i),
                    xv[i] * (float) (1000 + (offset + N) % (2*N) + i));
                return false;
            }
        }
        return true;
    }
};

int main() {
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(cpu_dev != nullptr);

    // two copies of the CPU device, as a tensor split over two devices
    ggml_backend_dev_t simple_devs[2] = { cpu_dev, cpu_dev };
    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(simple_devs, 2, mirrored_split_state, nullptr);
    GGML_ASSERT(meta_dev != nullptr);

    // two meta backends of one device: a model's context, and its drafter's
    ggml_backend_t a = ggml_backend_dev_init(meta_dev, nullptr);
    ggml_backend_t b = ggml_backend_dev_init(meta_dev, nullptr);
    GGML_ASSERT(a != nullptr && b != nullptr);

    // the shared weights: s[k] = 1000 + k, 2N values, mirrored on both devices
    ggml_init_params wparams = {
        /* .mem_size   = */ ggml_tensor_overhead()*2,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx_w = ggml_init(wparams);
    ggml_tensor * s = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, 2*N);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(ctx_w, a);
    GGML_ASSERT(wbuf != nullptr);
    std::vector<float> sv(2*N);
    for (int64_t k = 0; k < 2*N; k++) {
        sv[k] = (float) (1000 + k);
    }
    ggml_backend_tensor_set(s, sv.data(), 0, ggml_nbytes(s));

    bool all_ok = true;
    for (const bool split : {false, true}) {
        const char * how = split ? "split" : "whole graph";
        const uint64_t uid_a = split ? 2 : 1, uid_b0 = split ? 2000 : 1000;
        view_graph ga(a, s, 0, uid_a, split);
        bool ok = ga.compute_and_check(a, "A, first compute");

        // the drafter's graphs: each one fresh, so each is allocated and its split built, as a drafter alternating
        // between two graphs with one graph slot does
        for (int i = 0; ok && i < 2; i++) {
            view_graph gb(b, s, N, uid_b0 + (uint64_t) i);
            ok = gb.compute_and_check(b, i == 0 ? "B, first graph" : "B, second graph");
        }

        // A computes the graph it allocated last again, without allocating it
        ok = ok && ga.compute_and_check(a, "A, the same graph again after B's two graphs");

        // and keeps doing so while B keeps building graphs
        for (int i = 2; ok && i < 8; i++) {
            view_graph gb(b, s, N, uid_b0 + (uint64_t) i);
            ok = gb.compute_and_check(b, "B, a later graph") && ga.compute_and_check(a, "A, again");
        }
        printf("%s: A computing its %s\n", ok ? "ok" : "FAILED", how);
        all_ok = all_ok && ok;
    }

    ggml_backend_buffer_free(wbuf);
    ggml_free(ctx_w);
    ggml_backend_free(b);
    ggml_backend_free(a);

    if (!all_ok) {
        fprintf(stderr, "FAIL: a graph reused after another backend's graphs rotated a shared buffer's views computed with the wrong view\n");
        return EXIT_FAILURE;
    }
    printf("OK: graphs reused across another backend's rebuilds compute with their own views\n");
    return EXIT_SUCCESS;
}
