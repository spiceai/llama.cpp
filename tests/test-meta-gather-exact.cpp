// ggml_backend_meta_gather must reproduce the split tensor exactly, whatever all-reduce the devices use: a gather moves
// values (logits, token ids) rather than summing partial results, so a reduction that rounds them (to BF16, say) changes
// them. On a meta device over two devices, tensors split by rows are gathered and compared with their slices as each
// device holds them, read back: two products (logits-like values, and integer values that BF16 cannot represent, token
// ids above 100000) bit for bit, and a directly populated tensor of signed zeros and values BF16 cannot represent by
// value. A gather is exact in value, but a -0.0 comes back as +0.0, the sum of -0.0 and the other device's zero padding.
//
// It runs on two copies of every CPU and GPU device found, or of TEST_DEVICE alone when that is set. On CUDA the
// reduction is GGML_CUDA_ALLREDUCE's, or the platform default (NCCL on Linux when the build has it, else the internal
// all-reduce); the registered tests pin GGML_CUDA_AR_BF16_THRESHOLD=1 so BF16 compression applies to every F32 tensor.
// With GGML_CUDA_ALLREDUCE=nccl a CUDA device runs only beside a second GPU (NCCL takes one rank per GPU) in a build
// with NCCL, and is skipped otherwise.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
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
// the directly populated tensor: ZC x ZR, split by rows, 65536 elements, above the direct NCCL all-reduce's 32768
static const int64_t ZC     = 4;
static const int64_t ZR     = 16384;

// the tensors named "*split" are split by rows (axis 1) evenly across the devices, every other tensor is mirrored
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

// the first op of the test's graph that dev cannot run (a product, the gather's pad, or the add the meta backend
// reduces with), or nullptr when it runs them all: a GPU without the kernels this product needs is skipped
static const char * unsupported_op(ggml_backend_dev_t dev) {
    ggml_init_params p = { ggml_tensor_overhead()*16, nullptr, true };
    ggml_context * ctx = ggml_init(p);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, V);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, NCOLS);
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    ggml_tensor * z = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ZC, ZR);
    const char * missing = nullptr;
    for (ggml_tensor * t : { y, ggml_backend_meta_gather(ctx, y), ggml_add(ctx, y, y), ggml_backend_meta_gather(ctx, z), ggml_add(ctx, z, z) }) {
        if (!ggml_backend_dev_supports_op(dev, t)) {
            missing = ggml_op_desc(t);
            break;
        }
    }
    ggml_free(ctx);
    return missing;
}

static bool is_cuda(ggml_backend_dev_t dev) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    return reg != nullptr && strcmp(ggml_backend_reg_name(reg), "CUDA") == 0;
}

// the number of gathers that differ from their slices on a meta device over dev0 and dev1
static int test_devices(ggml_backend_dev_t dev0, ggml_backend_dev_t dev1) {
    const char * ar  = getenv("GGML_CUDA_ALLREDUCE");
    const char * thr = getenv("GGML_CUDA_AR_BF16_THRESHOLD");
    printf("devices %s + %s, GGML_CUDA_ALLREDUCE=%s, GGML_CUDA_AR_BF16_THRESHOLD=%s\n", ggml_backend_dev_name(dev0),
        ggml_backend_dev_name(dev1), ar ? ar : "(unset)", thr ? thr : "(unset)");

    std::vector<ggml_backend_dev_t> devs = { dev0, dev1 };
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
    // signed zeros, among values BF16 cannot represent (a quarter-step from 0.25 up to 16383.75, and 100001 up)
    std::vector<float> zf(ZC*ZR);
    for (int64_t i = 0; i < ZC*ZR; i++) {
        switch (i % 4) {
            case 0:  zf[i] = -0.0f;                           break;
            case 1:  zf[i] =  0.0f;                           break;
            case 2:  zf[i] =  (float) i * 0.25f + 0.25f;      break;
            default: zf[i] = -(float) (100001 + i);           break;
        }
    }

    ggml_init_params wp = { ggml_tensor_overhead()*8, nullptr, true };
    ggml_context * ctx_w = ggml_init(wp);
    ggml_tensor * w_logits_split = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, K, V);   ggml_set_name(w_logits_split, "w_logits_split");
    ggml_tensor * w_ids_split    = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, K, V);   ggml_set_name(w_ids_split,    "w_ids_split");
    ggml_tensor * z_split        = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, ZC, ZR); ggml_set_name(z_split,        "z_split");
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    GGML_ASSERT(wbuf != nullptr);
    ggml_backend_tensor_set(w_logits_split, wl.data(), 0, ggml_nbytes(w_logits_split));
    ggml_backend_tensor_set(w_ids_split,    wi.data(), 0, ggml_nbytes(w_ids_split));
    ggml_backend_tensor_set(z_split,        zf.data(), 0, ggml_nbytes(z_split));

    ggml_init_params gp = { ggml_tensor_overhead()*32 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, NCOLS);
    ggml_set_input(x);
    // each product is split by rows; read back, it is each device's slice as that device computed it
    struct pair { const char * what; ggml_tensor * split; ggml_tensor * gathered; };
    ggml_tensor * y_logits = ggml_mul_mat(ctx, w_logits_split, x);
    ggml_tensor * y_ids    = ggml_mul_mat(ctx, w_ids_split,    x);
    pair pairs[2] = {
        { "logits",    y_logits, ggml_backend_meta_gather(ctx, y_logits) },
        { "token ids", y_ids,    ggml_backend_meta_gather(ctx, y_ids)    },
    };
    ggml_tensor * z_gathered = ggml_backend_meta_gather(ctx, z_split);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    for (const pair & p : pairs) {
        ggml_set_output(p.split);
        ggml_set_output(p.gathered);
        ggml_build_forward_expand(gf, p.gathered);
    }
    ggml_set_output(z_gathered);
    ggml_build_forward_expand(gf, z_gathered);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, gf));
    ggml_backend_tensor_set(x, xf.data(), 0, ggml_nbytes(x));
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    int n_fail = 0;
    for (const pair & p : pairs) {
        std::vector<float> g(V*NCOLS), s(V*NCOLS);
        ggml_backend_tensor_get(p.gathered, g.data(), 0, ggml_nbytes(p.gathered));
        ggml_backend_tensor_get(p.split,    s.data(), 0, ggml_nbytes(p.split));
        int64_t n_diff = 0, first = -1;
        for (int64_t i = 0; i < V*NCOLS; i++) {
            uint32_t a, b;
            memcpy(&a, &g[i], 4);
            memcpy(&b, &s[i], 4);
            if (a != b) {
                n_diff++;
                first = first < 0 ? i : first;
            }
        }
        n_fail += n_diff != 0;
        printf("  %-9s gathered differs from its slices in %lld of %lld values", p.what, (long long) n_diff, (long long) (V*NCOLS));
        if (first >= 0) {
            printf(" (first at %lld: %.9g, want %.9g)", (long long) first, g[first], s[first]);
        }
        if (p.split == y_ids) {
            // row r of column 0 is token id 100001 + r: show a few as gathered
            printf("; ids at rows 0, 64640, 129279: %.0f %.0f %.0f (want %.0f %.0f %.0f)", g[0], g[64640], g[V - 1], s[0], s[64640], s[V - 1]);
        }
        printf(" %s\n", n_diff == 0 ? "ok" : "FAIL");
    }
    {
        std::vector<float> g(ZC*ZR);
        ggml_backend_tensor_get(z_gathered, g.data(), 0, ggml_nbytes(z_gathered));
        int64_t n_diff = 0, n_neg_zero = 0, n_now_pos = 0;
        for (int64_t i = 0; i < ZC*ZR; i++) {
            n_diff += !(g[i] == zf[i]); // -0.0 == +0.0
            if (zf[i] == 0.0f && std::signbit(zf[i])) {
                n_neg_zero++;
                n_now_pos += !std::signbit(g[i]);
            }
        }
        n_fail += n_diff != 0;
        printf("  %-9s gathered differs from the source in value in %lld of %lld values; %lld of %lld -0.0 came back +0.0 %s\n",
            "direct", (long long) n_diff, (long long) (ZC*ZR), (long long) n_now_pos, (long long) n_neg_zero, n_diff == 0 ? "ok" : "FAIL");
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_buffer_free(wbuf);
    ggml_free(ctx_w);
    ggml_backend_free(backend);
    return n_fail;
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    ggml_backend_load_all();
    std::vector<ggml_backend_dev_t> devs;
    const char * dev_name = getenv("TEST_DEVICE");
    if (dev_name != nullptr) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(dev_name);
        if (dev == nullptr) {
            printf("TEST_DEVICE=%s: no such device\nFAILED\n", dev_name);
            return 1;
        }
        devs.push_back(dev);
    } else {
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
            // an accelerator (BLAS, say) runs only beside the CPU, not as a device of its own
            if (type == GGML_BACKEND_DEVICE_TYPE_CPU || type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                devs.push_back(dev);
            }
        }
    }
    const char * ar   = getenv("GGML_CUDA_ALLREDUCE");
    const bool   nccl = ar != nullptr && strcmp(ar, "nccl") == 0;
    int n_fail = 0, n_run = 0;
    bool nccl_done = false;
    for (ggml_backend_dev_t dev : devs) {
        if (const char * op = unsupported_op(dev)) {
            printf("device %s: skipped, it cannot run %s\n", ggml_backend_dev_name(dev), op);
            continue;
        }
        ggml_backend_dev_t dev1 = dev;
        if (nccl && is_cuda(dev)) {
            if (nccl_done) {
                continue;
            }
            nccl_done = true;
            if (ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_cuda_nccl_allreduce") == nullptr) {
                printf("device %s: skipped, GGML_CUDA_ALLREDUCE=nccl and this build has no NCCL\n", ggml_backend_dev_name(dev));
                continue;
            }
            dev1 = nullptr;
            for (size_t i = 0; i < ggml_backend_dev_count() && dev1 == nullptr; i++) {
                ggml_backend_dev_t other = ggml_backend_dev_get(i);
                if (other != dev && is_cuda(other) && ggml_backend_dev_type(other) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                    dev1 = other;
                }
            }
            if (dev1 == nullptr) {
                printf("device %s: skipped, GGML_CUDA_ALLREDUCE=nccl needs a second GPU (NCCL takes one rank per GPU)\n", ggml_backend_dev_name(dev));
                continue;
            }
        }
        n_fail += test_devices(dev, dev1);
        n_run++;
    }
    // TEST_DEVICE names the device to test, so skipping it is a failure
    n_fail += dev_name != nullptr && n_run == 0;
    printf("%d of %zu devices run: %s\n", n_run, devs.size(), n_fail == 0 ? "OK" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
