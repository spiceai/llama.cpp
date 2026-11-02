// RPC answers every op as supported without asking the server, and a server sent an op its backend cannot run aborts
// (Metal: "unsupported op"). A row split across RPC devices (the DSpark drafter's head under a tensor split) may take each
// row's argmax through GGML_OP_ARGMAX_PART and GGML_OP_ARGMAX_MERGE only where every device runs them, so RPC answers for
// those two from the server's HELLO, which says whether every backend the server serves runs them.
//
// Over two servers on device argv[3], both on this host: each RPC device, and a meta device over the two, must answer for
// both ops as device argv[3] answers here. Then the product of a matrix split by rows across the two devices is reduced to
// each column's pick the way the drafter decides (dspark_vocab_parts): through ARGMAX_PART and ARGMAX_MERGE when the meta
// device runs them, otherwise by ggml_argmax over the whole product. Either way the picks must be the first max of each
// column of the whole product, and the path the run took is printed.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-rpc.h"
#include "ggml.h"

#include <cstdio>
#include <random>
#include <string>
#include <vector>

static const int64_t K       = 64;
static const int64_t V       = 4096;
static const int64_t NCOLS   = 3;
static const int     N_PARTS = 2;

static ggml_backend_meta_split_state mirrored_split_state(const ggml_tensor * tensor, void * userdata) {
    (void) tensor;
    (void) userdata;
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
}

// the first largest value of x, which holds no NaN
static int32_t first_max(const float * x, int64_t n) {
    int64_t best = 0;
    for (int64_t i = 1; i < n; i++) {
        if (x[i] > x[best]) {
            best = i;
        }
    }
    return (int32_t) best;
}

// whether dev runs ARGMAX_PART and ARGMAX_MERGE over the columns of an F32 [V, NCOLS] split N_PARTS ways
static bool runs_argmax_split(ggml_backend_dev_t dev) {
    ggml_init_params params = { 3*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * part  = ggml_argmax_part(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, V, NCOLS), N_PARTS);
    ggml_tensor * merge = ggml_argmax_merge(ctx, part);
    const bool ok = ggml_backend_dev_supports_op(dev, part) && ggml_backend_dev_supports_op(dev, merge);
    ggml_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <endpoint> <endpoint> <the device both servers run>\n", argv[0]);
        return 2;
    }
    setvbuf(stdout, nullptr, _IONBF, 0);
    ggml_backend_load_all();

    ggml_backend_dev_t local = ggml_backend_dev_by_name(argv[3]);
    if (local == nullptr) {
        printf("device %s: not on this host\nFAILED\n", argv[3]);
        return 1;
    }
    const bool want = runs_argmax_split(local);
    printf("%s on this host: ARGMAX_PART and ARGMAX_MERGE %s\n", argv[3], want ? "supported" : "not supported");

    std::vector<ggml_backend_dev_t> devs;
    for (int i = 1; i <= 2; i++) {
        ggml_backend_reg_t reg = ggml_backend_rpc_add_server(argv[i]);
        GGML_ASSERT(reg != nullptr && ggml_backend_reg_dev_count(reg) > 0);
        devs.push_back(ggml_backend_reg_dev_get(reg, 0));
    }
    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(devs.data(), devs.size(), mirrored_split_state, nullptr);
    GGML_ASSERT(meta_dev != nullptr);

    int n_fail = 0;
    const std::string meta_name = std::string("meta device over ") + ggml_backend_dev_name(devs[0]) + " and " + ggml_backend_dev_name(devs[1]);
    for (ggml_backend_dev_t dev : { devs[0], devs[1], meta_dev }) {
        const bool got = runs_argmax_split(dev);
        n_fail += got != want;
        printf("  %s: %s %s\n", dev == meta_dev ? meta_name.c_str() : (std::string(ggml_backend_dev_name(dev)) + " (" +
            ggml_backend_dev_description(dev) + ")").c_str(), got ? "supported" : "not supported", got == want ? "ok" : "FAIL");
    }
    const bool split = runs_argmax_split(meta_dev);

    ggml_backend_t backend = ggml_backend_dev_init(meta_dev, nullptr);
    GGML_ASSERT(backend != nullptr);

    // a BF16 matrix, like the drafter's Markov transition: its rows' products do not depend on which rows a device computes
    std::mt19937 rng(17);
    std::normal_distribution<float> nd(0.0f, 0.05f);
    std::vector<float> wf(K*V), xf(K*NCOLS);
    for (auto & a : wf) {
        a = nd(rng);
    }
    for (auto & a : xf) {
        a = nd(rng) * 20.0f;
    }
    ggml_init_params wparams = { ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx_w = ggml_init(wparams);
    ggml_tensor * w = ggml_new_tensor_2d(ctx_w, GGML_TYPE_BF16, K, V);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    GGML_ASSERT(wbuf != nullptr);
    std::vector<uint8_t> wq(ggml_nbytes(w));
    ggml_quantize_chunk(GGML_TYPE_BF16, wf.data(), wq.data(), 0, V, K, nullptr);
    ggml_backend_tensor_set(w, wq.data(), 0, wq.size());

    ggml_init_params params = { ggml_tensor_overhead()*16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, NCOLS);
    ggml_set_input(x);
    ggml_tensor * whole = ggml_mul_mat(ctx, w, x); // mirrored: every device computes every row
    ggml_tensor * pick  = nullptr;
    if (split) {
        const int64_t parts[N_PARTS] = { V/2, V - V/2 };
        ggml_tensor * rows = ggml_mul_mat(ctx, ggml_backend_meta_rows(ctx, w, N_PARTS, parts), x);
        pick = ggml_argmax_merge(ctx, ggml_backend_meta_gather(ctx, ggml_argmax_part(ctx, rows, N_PARTS)));
    } else {
        pick = ggml_argmax(ctx, whole);
    }
    ggml_set_output(whole);
    ggml_set_output(pick);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, whole);
    ggml_build_forward_expand(gf, pick);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, gf));
    ggml_backend_tensor_set(x, xf.data(), 0, ggml_nbytes(x));
    printf("computing each column's pick %s\n", split ? "from ARGMAX_PART and ARGMAX_MERGE over the head split by rows" :
        "by ggml_argmax over the whole head");
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    std::vector<float>   yw(V*NCOLS);
    std::vector<int32_t> pk(NCOLS);
    ggml_backend_tensor_get(whole, yw.data(), 0, ggml_nbytes(whole));
    ggml_backend_tensor_get(pick,  pk.data(), 0, ggml_nbytes(pick));
    for (int64_t c = 0; c < NCOLS; c++) {
        const int32_t ref = first_max(&yw[c*V], V);
        n_fail += pk[c] != ref;
        printf("  col %lld: pick %d, first max of the whole head %d %s\n", (long long) c, pk[c], ref, pk[c] == ref ? "ok" : "FAIL");
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_buffer_free(wbuf);
    ggml_free(ctx_w);
    ggml_backend_free(backend);
    printf("head %s: %s\n", split ? "split by rows" : "whole", n_fail == 0 ? "OK" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
