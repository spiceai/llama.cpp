// ggml_backend_meta_gather must reproduce the split tensor exactly, whatever all-reduce the devices use: a gather moves
// values (logits, token ids) rather than summing partial results, so a reduction that rounds them (to BF16, say) changes
// them. On a meta device over two copies of one device (TEST_DEVICE, default CPU; on CUDA, two devices reduce through the
// CUDA internal all-reduce unless GGML_CUDA_ALLREDUCE says otherwise), a product split by rows is gathered and compared
// bit for bit with the same product of a mirrored copy of the matrix: logits-like values, and integer values that BF16
// cannot represent (token ids above 100000).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static const int64_t K      = 64;
static const int64_t V      = 129280; // DeepSeek-V4.1's vocabulary
static const int64_t NCOLS  = 3;
static const int     N_DEVS = 2;

// the matrices named "*split" are split by rows (axis 1) evenly across the devices, every other tensor is mirrored
static ggml_backend_meta_split_state split_state(const ggml_tensor * tensor, void * userdata) {
    (void) userdata;
    const std::string name = tensor->name;
    if (name.size() >= 5 && name.compare(name.size() - 5, 5, "split") == 0) {
        ggml_backend_meta_split_state ss = {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
        for (int j = 0; j < N_DEVS; j++) {
            ss.ne[j] = tensor->ne[1] / N_DEVS + (j < tensor->ne[1] % N_DEVS ? 1 : 0);
        }
        return ss;
    }
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    ggml_backend_load_all();
    const char * dev_name = getenv("TEST_DEVICE");
    ggml_backend_dev_t dev = dev_name ? ggml_backend_dev_by_name(dev_name) : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(dev != nullptr);
    const char * ar  = getenv("GGML_CUDA_ALLREDUCE");
    const char * thr = getenv("GGML_CUDA_AR_BF16_THRESHOLD");
    printf("device %s x %d, GGML_CUDA_ALLREDUCE=%s, GGML_CUDA_AR_BF16_THRESHOLD=%s\n", ggml_backend_dev_name(dev), N_DEVS,
        ar ? ar : "(unset)", thr ? thr : "(unset)");

    std::vector<ggml_backend_dev_t> devs(N_DEVS, dev);
    ggml_backend_t backend = ggml_backend_dev_init(ggml_backend_meta_device(devs.data(), devs.size(), split_state, nullptr), nullptr);
    GGML_ASSERT(backend != nullptr);

    // logits-like rows, and rows that put token id 100001 + r in column 0 of row r (x's column 0 is the unit vector e_0)
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 0.05f);
    std::vector<float> wl(K*V), wi(K*V, 0.0f), xf(K*NCOLS);
    for (auto & v : wl) {
        v = nd(rng);
    }
    for (int64_t r = 0; r < V; r++) {
        wi[r*K + 0] = (float) (100001 + r);
        wi[r*K + 1] = 0.5f;
    }
    for (int64_t c = 0; c < NCOLS; c++) {
        for (int64_t i = 0; i < K; i++) {
            xf[c*K + i] = c == 0 ? (i == 0 ? 1.0f : 0.0f) : nd(rng) * 20.0f;
        }
    }

    ggml_init_params wp = { ggml_tensor_overhead()*8, nullptr, true };
    ggml_context * ctx_w = ggml_init(wp);
    ggml_tensor * w_logits_split = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, K, V); ggml_set_name(w_logits_split, "w_logits_split");
    ggml_tensor * w_logits_whole = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, K, V); ggml_set_name(w_logits_whole, "w_logits_whole");
    ggml_tensor * w_ids_split    = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, K, V); ggml_set_name(w_ids_split,    "w_ids_split");
    ggml_tensor * w_ids_whole    = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, K, V); ggml_set_name(w_ids_whole,    "w_ids_whole");
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    GGML_ASSERT(wbuf != nullptr);
    ggml_backend_tensor_set(w_logits_split, wl.data(), 0, ggml_nbytes(w_logits_split));
    ggml_backend_tensor_set(w_logits_whole, wl.data(), 0, ggml_nbytes(w_logits_whole));
    ggml_backend_tensor_set(w_ids_split,    wi.data(), 0, ggml_nbytes(w_ids_split));
    ggml_backend_tensor_set(w_ids_whole,    wi.data(), 0, ggml_nbytes(w_ids_whole));

    ggml_init_params gp = { ggml_tensor_overhead()*32 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, NCOLS);
    ggml_set_input(x);
    struct pair { const char * what; ggml_tensor * gathered; ggml_tensor * whole; };
    pair pairs[2] = {
        { "logits",    ggml_backend_meta_gather(ctx, ggml_mul_mat(ctx, w_logits_split, x)), ggml_mul_mat(ctx, w_logits_whole, x) },
        { "token ids", ggml_backend_meta_gather(ctx, ggml_mul_mat(ctx, w_ids_split,    x)), ggml_mul_mat(ctx, w_ids_whole,    x) },
    };
    ggml_cgraph * gf = ggml_new_graph(ctx);
    for (const pair & p : pairs) {
        ggml_set_output(p.gathered);
        ggml_set_output(p.whole);
        ggml_build_forward_expand(gf, p.gathered);
        ggml_build_forward_expand(gf, p.whole);
    }
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, gf));
    ggml_backend_tensor_set(x, xf.data(), 0, ggml_nbytes(x));
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    int n_fail = 0;
    for (const pair & p : pairs) {
        std::vector<float> g(V*NCOLS), w(V*NCOLS);
        ggml_backend_tensor_get(p.gathered, g.data(), 0, ggml_nbytes(p.gathered));
        ggml_backend_tensor_get(p.whole,    w.data(), 0, ggml_nbytes(p.whole));
        int64_t n_diff = 0, first = -1;
        for (int64_t i = 0; i < V*NCOLS; i++) {
            uint32_t a, b;
            memcpy(&a, &g[i], 4);
            memcpy(&b, &w[i], 4);
            if (a != b) {
                n_diff++;
                first = first < 0 ? i : first;
            }
        }
        n_fail += n_diff != 0;
        printf("  %-9s gathered differs from the whole product in %lld of %lld values", p.what, (long long) n_diff, (long long) (V*NCOLS));
        if (first >= 0) {
            printf(" (first at %lld: %.9g, want %.9g)", (long long) first, g[first], w[first]);
        }
        if (p.whole == pairs[1].whole) {
            // row r of column 0 is token id 100001 + r: show a few as gathered
            printf("; ids at rows 0, 64640, 129279: %.0f %.0f %.0f (want %.0f %.0f %.0f)", g[0], g[64640], g[V - 1], w[0], w[64640], w[V - 1]);
        }
        printf(" %s\n", n_diff == 0 ? "ok" : "FAIL");
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_buffer_free(wbuf);
    ggml_free(ctx_w);
    ggml_backend_free(backend);
    printf("%s\n", n_fail == 0 ? "OK" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
