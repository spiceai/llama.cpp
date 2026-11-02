// ggml_argmax_part and ggml_argmax_merge, and a product with ggml_backend_meta_rows, on a meta device over n copies of one
// device (TEST_DEVICE, default CPU): each copy computes its rows of the product of a whole (mirrored) matrix, and the argmax
// of each column comes from one (value, index) pair per copy. Checks that the gathered product equals the product of the
// whole matrix bit for bit (a row's dot product does not depend on which rows a device computes), and that the picks
// are the first max of each column as the greedy sampler takes it: the earliest index on a tie, 0 when the first value
// is a NaN, NaNs skipped otherwise. Ties straddle the devices' rows, from duplicated rows of the matrix.

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
#include <tuple>
#include <vector>

static ggml_backend_meta_split_state mirrored_split_state(const ggml_tensor * tensor, void * userdata) {
    (void) tensor;
    (void) userdata;
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
}

// the greedy sampler's pick (common_logits_argmax): 0 when x[0] is a NaN, else the first largest value that is not one
static int32_t ref_argmax(const float * x, int64_t n) {
    if (n == 0 || std::isnan(x[0])) {
        return 0;
    }
    float m = -INFINITY;
    for (int64_t i = 0; i < n; i++) {
        m = x[i] > m ? x[i] : m;
    }
    for (int64_t i = 0; i < n; i++) {
        if (x[i] >= m) {
            return (int32_t) i;
        }
    }
    return 0;
}

static std::vector<int64_t> even_parts(int64_t n, int n_parts) {
    std::vector<int64_t> parts(n_parts);
    int64_t low = 0;
    for (int j = 0; j + 1 < n_parts; j++) {
        const int64_t high = n * (j + 1) / n_parts;
        parts[j] = high - low;
        low = high;
    }
    parts[n_parts - 1] = n - low;
    return parts;
}

// rows of logits computed directly, through argmax_part and argmax_merge, on one backend (n_parts pairs, one written)
static int test_pairs(ggml_backend_t backend, const char * name) {
    const int64_t n = 1000;
    std::vector<std::vector<float>> rows;
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd;
    auto random_row = [&]() {
        std::vector<float> r(n);
        for (auto & v : r) {
            v = nd(rng);
        }
        return r;
    };
    {
        auto r = random_row();
        rows.push_back(r);
    }
    {
        auto r = random_row(); // tie, the later one first in memory order of threads
        r[700] = 9.0f; r[31] = 9.0f; r[999] = 9.0f;
        rows.push_back(r);
    }
    {
        auto r = random_row(); // NaN first: token 0
        r[0] = NAN; r[5] = 50.0f;
        rows.push_back(r);
    }
    {
        auto r = random_row(); // NaNs elsewhere are skipped
        r[3] = NAN; r[4] = 7.0f; r[900] = NAN;
        rows.push_back(r);
    }
    {
        std::vector<float> r(n, -INFINITY); // all -inf: token 0
        rows.push_back(r);
    }
    {
        std::vector<float> r(n, -INFINITY); // -inf with NaNs after the first value
        r[1] = NAN; r[2] = NAN;
        rows.push_back(r);
    }
    {
        std::vector<float> r(n, -INFINITY); // the first value that is not a NaN, after NaNs that are not first
        r[0] = -INFINITY; r[10] = -1e30f; r[11] = -1e30f;
        rows.push_back(r);
    }
    {
        std::vector<float> r(n, 0.0f); // signed zeros tie
        r[0] = -0.0f; r[1] = 0.0f;
        rows.push_back(r);
    }

    const int64_t n_rows  = (int64_t) rows.size();
    const int     n_parts = 3;

    ggml_init_params params = { ggml_tensor_overhead()*16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n_rows);
    ggml_set_input(x);
    ggml_tensor * pairs = ggml_argmax_part(ctx, x, n_parts);
    ggml_tensor * pick  = ggml_argmax_merge(ctx, pairs);
    ggml_set_output(pairs);
    ggml_set_output(pick);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, pick);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, gf));

    for (int64_t r = 0; r < n_rows; r++) {
        ggml_backend_tensor_set(x, rows[r].data(), r*x->nb[1], n*sizeof(float));
    }
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    std::vector<int32_t> got(n_rows);
    std::vector<float>   got_pairs(2*n_parts*n_rows);
    ggml_backend_tensor_get(pick, got.data(), 0, ggml_nbytes(pick));
    ggml_backend_tensor_get(pairs, got_pairs.data(), 0, ggml_nbytes(pairs));

    int n_fail = 0;
    for (int64_t r = 0; r < n_rows; r++) {
        const int32_t want = ref_argmax(rows[r].data(), n);
        const bool    ok   = got[r] == want;
        n_fail += ok ? 0 : 1;
        printf("  %s pairs row %lld: pick %d, want %d, pair (%g, %g), others (%g, %g) %s\n", name, (long long) r, got[r], want,
            got_pairs[r*2*n_parts + 0], got_pairs[r*2*n_parts + 1], got_pairs[r*2*n_parts + 2], got_pairs[r*2*n_parts + 3],
            ok ? "ok" : "FAIL");
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return n_fail;
}

// a product with a rows view on a meta device of parts.size() copies: gathered, it equals the whole product bit for bit;
// its picks, plain and after a bias like the DSpark Markov head adds, equal the first max of each column of the whole
// product. negative: the bias puts every logit far below 0, so a pair a device left unwritten (an empty share of the rows)
// would win. The graph then runs again on the same buffers with other inputs, so a pair left from the first run would show.
static int test_meta(ggml_backend_dev_t dev, ggml_type type, int64_t k, int64_t v, int64_t ncols, const std::vector<int64_t> & parts,
        bool negative) {
    const int n_parts = (int) parts.size();
    std::vector<ggml_backend_dev_t> devs(n_parts, dev);
    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(devs.data(), devs.size(), mirrored_split_state, nullptr);
    GGML_ASSERT(meta_dev != nullptr);
    ggml_backend_t backend = ggml_backend_dev_init(meta_dev, nullptr);
    GGML_ASSERT(backend != nullptr);

    std::string layout;
    for (int64_t p : parts) {
        layout += (layout.empty() ? "" : "/") + std::to_string(p);
    }

    // weights: row r of w is v's column r of logits; duplicate rows straddling the devices tie
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 0.05f);
    std::vector<float> wf(k*v);
    for (auto & x : wf) {
        x = nd(rng);
    }
    std::vector<float> xf(k*ncols);
    for (auto & x : xf) {
        x = nd(rng) * 20.0f;
    }
    // rows that win: aligned with column c's activations, scaled, then copied across the end of the first device with rows
    const int64_t cut = parts[0] > 0 ? parts[0] : parts[1];
    GGML_ASSERT(cut > 518 && cut + 1 < v);
    const int64_t r0 = cut - 1;                  // the last row of the first device with rows
    const int64_t r1 = cut + 1;                  // a row of the next device
    const int64_t r2 = v - 1;                    // the last row of the last device with rows
    const int64_t r3 = cut - 1 - 517;            // an earlier row of r0's device, read by another thread and warp than r0
    for (int64_t i = 0; i < k; i++) {
        wf[r0*k + i] = xf[0*k + i] * 0.2f;
    }
    memcpy(&wf[r1*k], &wf[r0*k], k*sizeof(float));
    memcpy(&wf[r2*k], &wf[r0*k], k*sizeof(float));
    memcpy(&wf[r3*k], &wf[r0*k], k*sizeof(float));

    // a bias like the Markov transition's, a BF16 product split by the same rows; its tied rows are copies too
    const int64_t kb = 256;
    std::vector<float> w2f(kb*v);
    for (auto & x : w2f) {
        x = negative ? 1.0f : nd(rng);
    }
    memcpy(&w2f[r1*kb], &w2f[r0*kb], kb*sizeof(float));
    memcpy(&w2f[r2*kb], &w2f[r0*kb], kb*sizeof(float));
    memcpy(&w2f[r3*kb], &w2f[r0*kb], kb*sizeof(float));
    auto bias_input = [&](float scale) {
        std::vector<float> xbf(kb*ncols);
        for (auto & x : xbf) {
            x = negative ? -1000.0f : nd(rng) * scale;
        }
        return xbf;
    };
    const std::vector<float> xbf = bias_input(5.0f);

    ggml_init_params wparams = { ggml_tensor_overhead()*4, nullptr, true };
    ggml_context * ctx_w = ggml_init(wparams);
    ggml_tensor * w  = ggml_new_tensor_2d(ctx_w, type, k, v);
    ggml_tensor * w2 = ggml_new_tensor_2d(ctx_w, GGML_TYPE_BF16, kb, v);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    GGML_ASSERT(wbuf != nullptr);
    for (auto [t, src, kk] : {std::tuple{w, &wf, k}, std::tuple{w2, &w2f, kb}}) {
        std::vector<uint8_t> wq(ggml_row_size(t->type, kk) * v);
        if (t->type == GGML_TYPE_F32) {
            memcpy(wq.data(), src->data(), wq.size());
        } else {
            ggml_quantize_chunk(t->type, src->data(), wq.data(), 0, v, kk, nullptr);
        }
        ggml_backend_tensor_set(t, wq.data(), 0, wq.size());
    }

    ggml_init_params params = { ggml_tensor_overhead()*32 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, ncols);
    ggml_set_input(x);
    ggml_tensor * xb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kb, ncols);
    ggml_set_input(xb);

    ggml_tensor * whole  = ggml_mul_mat(ctx, w, x);                                           // mirrored
    ggml_tensor * split  = ggml_mul_mat(ctx, ggml_backend_meta_rows(ctx, w, n_parts, parts.data()), x); // split by rows
    ggml_tensor * gather = ggml_backend_meta_gather(ctx, split);
    ggml_tensor * pick   = ggml_argmax_merge(ctx, ggml_backend_meta_gather(ctx, ggml_argmax_part(ctx, split, n_parts)));
    ggml_tensor * col    = ggml_add(ctx, split, ggml_mul_mat(ctx, ggml_backend_meta_rows(ctx, w2, n_parts, parts.data()), xb));
    ggml_tensor * pick_b = ggml_argmax_merge(ctx, ggml_backend_meta_gather(ctx, ggml_argmax_part(ctx, col, n_parts)));
    ggml_tensor * whole_b = ggml_add(ctx, whole, ggml_mul_mat(ctx, w2, xb));
    ggml_tensor * plain   = ggml_argmax(ctx, whole); // the op the draft chain takes its argmax with when the head is whole
    for (ggml_tensor * t : {x, xb, whole, gather, pick, pick_b, whole_b, plain}) {
        ggml_set_output(t); // the inputs too: the graph runs twice on one allocation
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    for (ggml_tensor * t : {whole, gather, pick, pick_b, whole_b, plain}) {
        ggml_build_forward_expand(gf, t);
    }
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, gf));

    int n_fail = 0;
    for (int run = 0; run < 2; run++) {
        // the second run: other activations, on the same buffers
        std::vector<float> xr = xf;
        std::vector<float> xbr = xbf;
        if (run == 1) {
            for (auto & a : xr) {
                a = nd(rng) * 20.0f;
            }
            xbr = bias_input(7.0f);
        }
        ggml_backend_tensor_set(x,  xr.data(),  0, ggml_nbytes(x));
        ggml_backend_tensor_set(xb, xbr.data(), 0, ggml_nbytes(xb));
        GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

        std::vector<float> yw(v*ncols), yg(v*ncols), ywb(v*ncols);
        std::vector<int32_t> pk(ncols), pkb(ncols), pl(ncols);
        ggml_backend_tensor_get(plain,   pl.data(),  0, ggml_nbytes(plain));
        ggml_backend_tensor_get(whole,   yw.data(),  0, ggml_nbytes(whole));
        ggml_backend_tensor_get(gather,  yg.data(),  0, ggml_nbytes(gather));
        ggml_backend_tensor_get(whole_b, ywb.data(), 0, ggml_nbytes(whole_b));
        ggml_backend_tensor_get(pick,    pk.data(),  0, ggml_nbytes(pick));
        ggml_backend_tensor_get(pick_b,  pkb.data(), 0, ggml_nbytes(pick_b));

        int64_t n_diff = 0;
        for (int64_t i = 0; i < v*ncols; i++) {
            uint32_t a, b;
            memcpy(&a, &yw[i], 4);
            memcpy(&b, &yg[i], 4);
            n_diff += a != b;
        }
        n_fail += n_diff != 0;
        for (int64_t c = 0; c < ncols; c++) {
            const int32_t want   = ref_argmax(&yw[c*v], v);
            const int32_t want_b = ref_argmax(&ywb[c*v], v);
            const bool ok = pk[c] == want && pkb[c] == want_b;
            n_fail += !ok;
            printf("  %s %s k=%lld v=%lld parts=%s%s run %d col %lld: pick %d want %d (tie rows %lld %lld %lld %lld, values %.9g %.9g %.9g %.9g), with bias pick %d want %d; ggml_argmax of the whole product %d %s\n",
                ggml_backend_dev_name(dev), ggml_type_name(type), (long long) k, (long long) v, layout.c_str(), negative ? " negative" : "",
                run, (long long) c, pk[c], want, (long long) r3, (long long) r0, (long long) r1, (long long) r2,
                yw[c*v + r3], yw[c*v + r0], yw[c*v + r1], yw[c*v + r2], pkb[c], want_b, pl[c], ok ? "ok" : "FAIL");
        }
        printf("  %s %s k=%lld v=%lld ncols=%lld parts=%s%s run %d: gathered product differs from the whole product in %lld of %lld values %s\n",
            ggml_backend_dev_name(dev), ggml_type_name(type), (long long) k, (long long) v, (long long) ncols, layout.c_str(),
            negative ? " negative" : "", run, (long long) n_diff, (long long) (v*ncols), n_diff == 0 ? "ok" : "FAIL");
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_buffer_free(wbuf);
    ggml_free(ctx_w);
    ggml_backend_free(backend);
    return n_fail;
}

int main() {
    // copies of one CUDA device: the two-device all-reduce pipeline assumes two devices, so reduce through the meta
    // backend's butterfly as a meta device over RPC devices does
    setenv("GGML_CUDA_ALLREDUCE", "none", 0);
    ggml_backend_load_all();

    const char * dev_name = getenv("TEST_DEVICE");
    ggml_backend_dev_t dev = dev_name ? ggml_backend_dev_by_name(dev_name) : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(dev != nullptr);
    printf("device %s\n", ggml_backend_dev_name(dev));

    int n_fail = 0;
    {
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        n_fail += test_pairs(backend, ggml_backend_dev_name(dev));
        ggml_backend_free(backend);
    }
    // the DeepSeek-V4.1 DSpark shapes: the Q6_K output head (3 draft columns) and the BF16 Markov transition (1 column)
    const int64_t v = getenv("TEST_VOCAB") ? atoll(getenv("TEST_VOCAB")) : 129280;
    for (int n_parts : {2, 3}) {
        const std::vector<int64_t> parts = even_parts(v, n_parts);
        n_fail += test_meta(dev, GGML_TYPE_Q6_K, 5120, v, 3, parts, false);
        n_fail += test_meta(dev, GGML_TYPE_BF16, 256,  v, 1, parts, false);
        n_fail += test_meta(dev, GGML_TYPE_F32,  256,  v, 2, parts, false);
    }
    // a device with no rows (a tensor split that gives it none), first and last, under negative logits
    {
        const std::vector<int64_t> p2 = even_parts(v, 2);
        for (const std::vector<int64_t> & parts : {std::vector<int64_t>{0, p2[0], p2[1]}, std::vector<int64_t>{p2[0], p2[1], 0}}) {
            n_fail += test_meta(dev, GGML_TYPE_Q6_K, 5120, v, 3, parts, true);
            n_fail += test_meta(dev, GGML_TYPE_BF16, 256,  v, 1, parts, true);
        }
    }
    printf("%s\n", n_fail == 0 ? "OK" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
