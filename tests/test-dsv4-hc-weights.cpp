// ggml_dsv4_hc_weights, which src/models/deepseek4.cpp uses with LLAMA_DSV4_HC_WEIGHTS=1, must equal bit for bit the
// separate MUL, ADD, SIGMOID, SCALE_BIAS / SCALE and DSV4_HC_COMB nodes that build_hc_mixes uses without it.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static uint64_t splitmix64(uint64_t & state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

// uniform in [lo, hi), the same on every platform
static float rand_f32(uint64_t & state, float lo, float hi) {
    return lo + (hi - lo) * (float) (splitmix64(state) >> 40) / (float) (1ull << 24);
}

static ggml_tensor * view_1d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t i0) {
    return ggml_view_1d(ctx, t, ne0, i0*ggml_element_size(t));
}

static ggml_tensor * view_2d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t ne1, int64_t i0) {
    return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], i0*ggml_element_size(t));
}

// returns how many of the fused op's values differ from the separate nodes' bit for bit
static int64_t compare(ggml_backend_t backend, int64_t n_tokens, uint64_t seed) {
    const int64_t hc     = 4;
    const float   eps    = 1e-6f;
    const int32_t n_iter = 20;

    ggml_init_params params = {
        /*.mem_size   =*/ 32*ggml_tensor_overhead() + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);

    ggml_tensor * mixes = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (2 + hc)*hc, n_tokens);
    ggml_tensor * scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
    ggml_tensor * base  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (2 + hc)*hc);

    ggml_tensor * weights = ggml_dsv4_hc_weights(ctx, mixes, scale, base, eps, n_iter, eps, 2.0f);

    // the nodes build_hc_mixes builds without LLAMA_DSV4_HC_WEIGHTS, with the fused comb that the switch requires
    ggml_tensor * pre = view_2d(ctx, mixes, hc, n_tokens, 0);
    pre = ggml_add(ctx, ggml_mul(ctx, pre, view_1d(ctx, scale, 1, 0)), view_1d(ctx, base, hc, 0));
    pre = ggml_scale_bias(ctx, ggml_sigmoid(ctx, pre), 1.0f, eps);

    ggml_tensor * post = view_2d(ctx, mixes, hc, n_tokens, hc);
    post = ggml_add(ctx, ggml_mul(ctx, post, view_1d(ctx, scale, 1, 1)), view_1d(ctx, base, hc, hc));
    post = ggml_scale(ctx, ggml_sigmoid(ctx, post), 2.0f);

    ggml_tensor * comb = ggml_dsv4_hc_comb(ctx, mixes, scale, base, eps, n_iter);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, weights);
    ggml_build_forward_expand(gf, pre);
    ggml_build_forward_expand(gf, post);
    ggml_build_forward_expand(gf, comb);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buf != nullptr);

    // mixes wide enough to reach both tails of the sigmoid, where a rounding difference in its input shows most
    std::vector<float> v_mixes(ggml_nelements(mixes));
    std::vector<float> v_scale(ggml_nelements(scale));
    std::vector<float> v_base(ggml_nelements(base));
    for (float & v : v_mixes) { v = rand_f32(seed, -6.0f,  6.0f);  }
    for (float & v : v_scale) { v = rand_f32(seed,  0.25f, 1.25f); }
    for (float & v : v_base)  { v = rand_f32(seed, -2.0f,  2.0f);  }
    ggml_backend_tensor_set(mixes, v_mixes.data(), 0, ggml_nbytes(mixes));
    ggml_backend_tensor_set(scale, v_scale.data(), 0, ggml_nbytes(scale));
    ggml_backend_tensor_set(base,  v_base.data(),  0, ggml_nbytes(base));

    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    std::vector<float> v_weights(ggml_nelements(weights));
    std::vector<float> v_pre(ggml_nelements(pre));
    std::vector<float> v_post(ggml_nelements(post));
    std::vector<float> v_comb(ggml_nelements(comb));
    ggml_backend_tensor_get(weights, v_weights.data(), 0, ggml_nbytes(weights));
    ggml_backend_tensor_get(pre,     v_pre.data(),     0, ggml_nbytes(pre));
    ggml_backend_tensor_get(post,    v_post.data(),    0, ggml_nbytes(post));
    ggml_backend_tensor_get(comb,    v_comb.data(),    0, ggml_nbytes(comb));

    int64_t n_diff = 0;
    auto check = [&](const char * name, int64_t t, int64_t i, float fused, float separate) {
        if (memcmp(&fused, &separate, sizeof(float)) != 0) {
            if (n_diff == 0) {
                printf("  first difference: %s[%" PRId64 "] of token %" PRId64 ": fused %.9g, separate nodes %.9g\n",
                       name, i, t, fused, separate);
            }
            n_diff++;
        }
    };
    for (int64_t t = 0; t < n_tokens; t++) {
        const float * w = v_weights.data() + t*(2 + hc)*hc;
        for (int64_t h = 0; h < hc; h++) {
            check("pre",  t, h, w[h],      v_pre[t*hc + h]);
            check("post", t, h, w[hc + h], v_post[t*hc + h]);
        }
        for (int64_t i = 0; i < hc*hc; i++) {
            check("comb", t, i, w[2*hc + i], v_comb[t*hc*hc + i]);
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return n_diff;
}

int main(void) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    GGML_ASSERT(backend != nullptr);

    int n_failed = 0;
    for (int64_t n_tokens : { 1, 17, 257, 4096 }) {
        for (uint64_t seed : { 1, 2, 3 }) {
            const int64_t n_diff = compare(backend, n_tokens, seed);
            printf("n_tokens = %4" PRId64 ", seed = %" PRIu64 ": %" PRId64 " of %" PRId64 " values differ\n",
                   n_tokens, seed, n_diff, n_tokens*24);
            n_failed += n_diff != 0;
        }
    }

    ggml_backend_free(backend);

    if (n_failed != 0) {
        printf("FAIL: %d of 12 cases differ from the separate nodes\n", n_failed);
        return 1;
    }
    printf("OK\n");
    return 0;
}
