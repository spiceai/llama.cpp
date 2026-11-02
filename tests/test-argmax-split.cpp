// ggml_argmax_part and ggml_argmax_merge, and a product with ggml_backend_meta_rows, on a meta device over n copies of one
// device (TEST_DEVICE, default CPU): each copy computes its rows of the product of a whole (mirrored) matrix, and the argmax
// of each column comes from one (value, index) pair per copy. Checks that the gathered product equals the product of the
// whole matrix bit for bit (a row's dot product does not depend on which rows a device computes), and that the picks
// are the first max of each column as the greedy sampler takes it: the earliest index on a tie, 0 when the first value
// is a NaN, NaNs skipped otherwise, including one at the start of a later device's rows. Ties straddle the devices' rows, from duplicated rows of the matrix. Two copies of a
// CUDA device reduce through the CUDA internal all-reduce, three through the meta backend's butterfly.

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
#include <utility>
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

// the pair ARGMAX_PART writes first for a whole row (origin 0): (NaN, 0) when the first value is a NaN, else the first
// largest value that is not one and its index, as stored (a -0.0 stays -0.0), or (-inf, -1) when every value is a NaN;
// every other pair is (-inf, -1)
static std::pair<float, float> ref_pair(const float * x, int64_t n) {
    if (n > 0 && std::isnan(x[0])) {
        return { NAN, 0.0f };
    }
    int64_t best = -1;
    for (int64_t i = 0; i < n; i++) {
        if (best < 0 ? !std::isnan(x[i]) : x[i] > x[best]) {
            best = i;
        }
    }
    return best < 0 ? std::pair<float, float>{ -INFINITY, -1.0f } : std::pair<float, float>{ x[best], (float) best };
}

// a and b are the same float, bit for bit, any NaN matching any NaN
static bool same_float(float a, float b) {
    if (std::isnan(a) || std::isnan(b)) {
        return std::isnan(a) && std::isnan(b);
    }
    uint32_t ua, ub;
    memcpy(&ua, &a, 4);
    memcpy(&ub, &b, 4);
    return ua == ub;
}

// rows of logits computed directly, through argmax_part and argmax_merge, on one backend (n_parts pairs, one written):
// the pick and every pair, padding included, against the reference
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
        const std::pair<float, float> want_pair = ref_pair(rows[r].data(), n);
        bool pairs_ok = same_float(got_pairs[r*2*n_parts + 0], want_pair.first) && same_float(got_pairs[r*2*n_parts + 1], want_pair.second);
        for (int k = 1; k < n_parts; k++) {
            pairs_ok = pairs_ok && same_float(got_pairs[r*2*n_parts + 2*k], -INFINITY) && same_float(got_pairs[r*2*n_parts + 2*k + 1], -1.0f);
        }
        const bool    ok   = got[r] == want && pairs_ok;
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
    ggml_tensor * gather_b = ggml_backend_meta_gather(ctx, col); // the biased logits as the devices computed them
    ggml_tensor * plain   = ggml_argmax(ctx, whole); // the op the draft chain takes its argmax with when the head is whole
    for (ggml_tensor * t : {x, xb, whole, gather, pick, pick_b, whole_b, gather_b, plain}) {
        ggml_set_output(t); // the inputs too: the graph runs twice on one allocation
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    for (ggml_tensor * t : {whole, gather, pick, pick_b, whole_b, gather_b, plain}) {
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

        std::vector<float> yw(v*ncols), yg(v*ncols), ywb(v*ncols), ygb(v*ncols);
        std::vector<int32_t> pk(ncols), pkb(ncols), pl(ncols);
        ggml_backend_tensor_get(plain,   pl.data(),  0, ggml_nbytes(plain));
        ggml_backend_tensor_get(whole,   yw.data(),  0, ggml_nbytes(whole));
        ggml_backend_tensor_get(gather,  yg.data(),  0, ggml_nbytes(gather));
        ggml_backend_tensor_get(whole_b, ywb.data(), 0, ggml_nbytes(whole_b));
        ggml_backend_tensor_get(gather_b, ygb.data(), 0, ggml_nbytes(gather_b));
        ggml_backend_tensor_get(pick,    pk.data(),  0, ggml_nbytes(pick));
        ggml_backend_tensor_get(pick_b,  pkb.data(), 0, ggml_nbytes(pick_b));

        int64_t n_diff = 0;
        for (int64_t i = 0; i < v*ncols; i++) {
            uint32_t a, b;
            memcpy(&a, &yw[i], 4);
            memcpy(&b, &yg[i], 4);
            n_diff += a != b;
        }
        // the CPU's F32 product can depend on how many rows a device computes (on CI's x86 runners, split 1365/1365/1366,
        // 6376 of 8192 values differ in the last bits), so there the picks are checked against the first max of what the
        // devices computed, and the difference is reported, not failed. Every other product must equal the whole one.
        const bool row_invariant = !(type == GGML_TYPE_F32 && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU);
        n_fail += row_invariant && n_diff != 0;
        for (int64_t c = 0; c < ncols; c++) {
            const int32_t want   = ref_argmax(row_invariant ? &yw[c*v]  : &yg[c*v],  v);
            const int32_t want_b = ref_argmax(row_invariant ? &ywb[c*v] : &ygb[c*v], v);
            const bool ok = pk[c] == want && pkb[c] == want_b;
            n_fail += !ok;
            printf("  %s %s k=%lld v=%lld parts=%s%s run %d col %lld: pick %d want %d (tie rows %lld %lld %lld %lld, values %.9g %.9g %.9g %.9g), with bias pick %d want %d; ggml_argmax of the whole product %d %s\n",
                ggml_backend_dev_name(dev), ggml_type_name(type), (long long) k, (long long) v, layout.c_str(), negative ? " negative" : "",
                run, (long long) c, pk[c], want, (long long) r3, (long long) r0, (long long) r1, (long long) r2,
                yw[c*v + r3], yw[c*v + r0], yw[c*v + r1], yw[c*v + r2], pkb[c], want_b, pl[c], ok ? "ok" : "FAIL");
        }
        printf("  %s %s k=%lld v=%lld ncols=%lld parts=%s%s run %d: gathered product differs from the whole product in %lld of %lld values %s\n",
            ggml_backend_dev_name(dev), ggml_type_name(type), (long long) k, (long long) v, (long long) ncols, layout.c_str(),
            negative ? " negative" : "", run, (long long) n_diff, (long long) (v*ncols),
            n_diff == 0 ? "ok" : row_invariant ? "FAIL" : "(reported only: this backend's F32 product depends on the rows)");
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_buffer_free(wbuf);
    ggml_free(ctx_w);
    ggml_backend_free(backend);
    return n_fail;
}

// NaNs on the split path: a product with a rows view whose rows are given values (row r of w is value r times e_0, and
// x = e_0), so a NaN can sit at the start of a later device's slice. Only the row's first value decides token 0; a NaN
// at a later slice's start is skipped like any other. Each case sets the weights and runs the graph again.
static int test_meta_nan(ggml_backend_dev_t dev, const std::vector<int64_t> & parts) {
    const int     n_parts = (int) parts.size();
    const int64_t k = 32, v = 1000;
    std::vector<ggml_backend_dev_t> devs(n_parts, dev);
    ggml_backend_t backend = ggml_backend_dev_init(ggml_backend_meta_device(devs.data(), devs.size(), mirrored_split_state, nullptr), nullptr);
    GGML_ASSERT(backend != nullptr);
    std::string layout;
    for (int64_t p : parts) {
        layout += (layout.empty() ? "" : "/") + std::to_string(p);
    }

    struct nan_case { std::string what; std::vector<float> vals; };
    std::vector<nan_case> cases;
    {
        std::vector<float> r(v, -1.0f);
        r[0] = NAN; r[v - 1] = 100.0f;
        cases.push_back({ "NaN at row 0, max at the last row", r });
    }
    int64_t start = 0;
    for (int j = 0; j + 1 < n_parts; j++) {
        start += parts[j];
        const std::string at = "NaN at row " + std::to_string(start) + " (device " + std::to_string(j + 1) + "'s first row)";
        {
            std::vector<float> r(v, -1.0f);
            r[start] = NAN; r[v - 1] = 100.0f;
            cases.push_back({ at + ", max at the last row", r });
        }
        {
            std::vector<float> r(v, -1.0f);
            r[start] = NAN; r[10] = 100.0f;
            cases.push_back({ at + ", max at row 10", r });
        }
        {
            std::vector<float> r(v, -INFINITY);
            r[start] = NAN;
            cases.push_back({ at + ", every other row -inf", r });
        }
    }

    ggml_init_params wp = { ggml_tensor_overhead()*2, nullptr, true };
    ggml_context * ctx_w = ggml_init(wp);
    ggml_tensor * w = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, k, v);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    GGML_ASSERT(wbuf != nullptr);

    ggml_init_params gp = { ggml_tensor_overhead()*16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, 1);
    ggml_set_input(x);
    ggml_tensor * split = ggml_mul_mat(ctx, ggml_backend_meta_rows(ctx, w, n_parts, parts.data()), x);
    ggml_tensor * pick  = ggml_argmax_merge(ctx, ggml_backend_meta_gather(ctx, ggml_argmax_part(ctx, split, n_parts)));
    ggml_set_output(x);
    ggml_set_output(pick);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, pick);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, gf));

    std::vector<float> xf(k, 0.0f);
    xf[0] = 1.0f;
    int n_fail = 0;
    for (const nan_case & c : cases) {
        std::vector<float> wf(k*v, 0.0f);
        for (int64_t r = 0; r < v; r++) {
            wf[r*k] = c.vals[r];
        }
        ggml_backend_tensor_set(w, wf.data(), 0, ggml_nbytes(w));
        ggml_backend_tensor_set(x, xf.data(), 0, ggml_nbytes(x));
        GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
        int32_t got = -1;
        ggml_backend_tensor_get(pick, &got, 0, sizeof(got));
        const int32_t want = ref_argmax(c.vals.data(), v);
        n_fail += got != want;
        printf("  %s nan parts=%s, %s: pick %d, want %d %s\n", ggml_backend_dev_name(dev), layout.c_str(), c.what.c_str(), got, want,
            got == want ? "ok" : "FAIL");
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_buffer_free(wbuf);
    ggml_free(ctx_w);
    ggml_backend_free(backend);
    return n_fail;
}

// a rows view of a mirrored matrix that is itself a view at a row offset: each device's rows start after the offset
static int test_meta_offset_view(ggml_backend_dev_t dev, const std::vector<int64_t> & parts) {
    const int     n_parts = (int) parts.size();
    const int64_t k = 32, v = 1000, off = 100, ncols = 2;
    std::vector<ggml_backend_dev_t> devs(n_parts, dev);
    ggml_backend_t backend = ggml_backend_dev_init(ggml_backend_meta_device(devs.data(), devs.size(), mirrored_split_state, nullptr), nullptr);
    GGML_ASSERT(backend != nullptr);

    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.0f, 0.05f);
    std::vector<float> wf(k*(v + off)), xf(k*ncols);
    for (auto & a : wf) {
        a = nd(rng);
    }
    for (auto & a : xf) {
        a = nd(rng) * 20.0f;
    }
    ggml_init_params wp = { ggml_tensor_overhead()*2, nullptr, true };
    ggml_context * ctx_w = ggml_init(wp);
    ggml_tensor * w = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, k, v + off);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    GGML_ASSERT(wbuf != nullptr);
    ggml_backend_tensor_set(w, wf.data(), 0, ggml_nbytes(w));

    ggml_init_params gp = { ggml_tensor_overhead()*16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, ncols);
    ggml_set_input(x);
    ggml_tensor * sub   = ggml_view_2d(ctx, w, k, v, w->nb[1], off*w->nb[1]); // rows off .. off + v - 1
    ggml_tensor * split = ggml_mul_mat(ctx, ggml_backend_meta_rows(ctx, sub, n_parts, parts.data()), x);
    ggml_tensor * gath  = ggml_backend_meta_gather(ctx, split);
    ggml_tensor * pick  = ggml_argmax_merge(ctx, ggml_backend_meta_gather(ctx, ggml_argmax_part(ctx, split, n_parts)));
    ggml_set_output(x);
    ggml_set_output(gath);
    ggml_set_output(pick);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, gath);
    ggml_build_forward_expand(gf, pick);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, gf));
    ggml_backend_tensor_set(x, xf.data(), 0, ggml_nbytes(x));
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    std::vector<float>   yg(v*ncols);
    std::vector<int32_t> pk(ncols);
    ggml_backend_tensor_get(gath, yg.data(), 0, ggml_nbytes(gath));
    ggml_backend_tensor_get(pick, pk.data(), 0, ggml_nbytes(pick));
    // the reference: row r of the view is row off + r of w, summed in the same order as the CPU's dot product
    int n_fail = 0;
    int64_t n_far = 0;
    for (int64_t c = 0; c < ncols; c++) {
        std::vector<float> ref(v);
        for (int64_t r = 0; r < v; r++) {
            double acc = 0.0;
            for (int64_t i = 0; i < k; i++) {
                acc += (double) wf[(off + r)*k + i] * xf[c*k + i];
            }
            ref[r] = (float) acc;
            n_far += std::fabs(yg[c*v + r] - ref[r]) > 1e-4f * (1.0f + std::fabs(ref[r]));
        }
        const int32_t want = ref_argmax(&yg[c*v], v);
        const int32_t want_ref = ref_argmax(ref.data(), v);
        n_fail += pk[c] != want || want != want_ref;
        printf("  %s offset view of %lld rows at row %lld, parts %lld/%lld%s col %lld: pick %d, want %d (rows of w from the offset: %d) %s\n",
            ggml_backend_dev_name(dev), (long long) v, (long long) off, (long long) parts[0], (long long) parts[1],
            n_parts > 2 ? "/..." : "", (long long) c, pk[c], want, want_ref, pk[c] == want && want == want_ref ? "ok" : "FAIL");
    }
    n_fail += n_far != 0;
    printf("  %s offset view: %lld of %lld gathered values far from rows %lld.. of w %s\n", ggml_backend_dev_name(dev),
        (long long) n_far, (long long) (v*ncols), (long long) off, n_far == 0 ? "ok" : "FAIL");

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_buffer_free(wbuf);
    ggml_free(ctx_w);
    ggml_backend_free(backend);
    return n_fail;
}

// tensors named "*rsplit" are split by whole rows (axis 1) evenly across the devices, everything else is mirrored
static ggml_backend_meta_split_state rows_split_state(const ggml_tensor * tensor, void * userdata) {
    const int n_parts = *(const int *) userdata;
    const std::string name = tensor->name;
    if (name.size() >= 6 && name.compare(name.size() - 6, 6, "rsplit") == 0) {
        ggml_backend_meta_split_state ss = {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
        for (int j = 0; j < n_parts; j++) {
            ss.ne[j] = tensor->ne[1] / n_parts + (j < tensor->ne[1] % n_parts ? 1 : 0);
        }
        return ss;
    }
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
}

// whole rows split across the devices (axis 1): each device takes the argmax of its own rows, and the picks stay with
// the rows, split along the picks' only axis
static int test_meta_whole_rows(ggml_backend_dev_t dev, int n_parts) {
    const int64_t n = 1000, rows = 6;
    std::vector<ggml_backend_dev_t> devs(n_parts, dev);
    int ud = n_parts;
    ggml_backend_t backend = ggml_backend_dev_init(ggml_backend_meta_device(devs.data(), devs.size(), rows_split_state, &ud), nullptr);
    GGML_ASSERT(backend != nullptr);

    std::mt19937 rng(9);
    std::normal_distribution<float> nd;
    std::vector<float> xf(n*rows);
    for (auto & a : xf) {
        a = nd(rng);
    }
    xf[0*n + 0]   = NAN;                        // row 0: a first NaN, token 0
    xf[1*n + 3]   = NAN;  xf[1*n + 700] = 9.0f; // row 1: a NaN after the first value is skipped
    xf[2*n + 31]  = 9.0f; xf[2*n + 999] = 9.0f; // row 2: a tie, the first wins
    xf[3*n + 0]   = NAN;  xf[3*n + 10]  = 9.0f; // row 3, the first row of the second device: a first NaN, token 0
    xf[4*n + 500] = 50.0f;
    ggml_init_params wp = { ggml_tensor_overhead()*2, nullptr, true };
    ggml_context * ctx_w = ggml_init(wp);
    ggml_tensor * x = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, n, rows);
    ggml_set_name(x, "x_rsplit");
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    GGML_ASSERT(wbuf != nullptr);
    ggml_backend_tensor_set(x, xf.data(), 0, ggml_nbytes(x));

    ggml_init_params gp = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_tensor * pick = ggml_argmax_merge(ctx, ggml_argmax_part(ctx, x, n_parts));
    ggml_set_output(pick);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, pick);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, gf));
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    std::vector<int32_t> pk(rows);
    ggml_backend_tensor_get(pick, pk.data(), 0, ggml_nbytes(pick));
    int n_fail = 0;
    std::string got, want;
    for (int64_t r = 0; r < rows; r++) {
        const int32_t w = ref_argmax(&xf[r*n], n);
        n_fail += pk[r] != w;
        got  += " " + std::to_string(pk[r]);
        want += " " + std::to_string(w);
    }
    printf("  %s whole rows split %d ways: picks%s, want%s %s\n", ggml_backend_dev_name(dev), n_parts, got.c_str(), want.c_str(),
        n_fail == 0 ? "ok" : "FAIL");

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_buffer_free(wbuf);
    ggml_free(ctx_w);
    ggml_backend_free(backend);
    return n_fail;
}

// The DSpark Markov head's graph: block_drafts positions of n_blocks sequences, base = head x hidden [V, n_blocks*block],
// position i of every block a strided view of base, plus that position's Markov bias, and its pick from the pairs. With one
// block the head is split by rows (as the drafter does); with more, the views would step over the other blocks' rows of a
// row-split tensor, which the meta backend cannot split, so the drafter keeps the head whole: built here on the whole matrix
// of the same meta device. Either way the picks must equal the first max of each column of the whole computation.
static int test_markov_blocks(ggml_backend_dev_t dev, int64_t v, int64_t n_blocks, int n_parts) {
    const int64_t k = 256, kb = 64, block = 3, n_tok = n_blocks*block;
    const bool split = n_blocks == 1;
    std::vector<ggml_backend_dev_t> devs(n_parts, dev);
    ggml_backend_t backend = ggml_backend_dev_init(ggml_backend_meta_device(devs.data(), devs.size(), mirrored_split_state, nullptr), nullptr);
    GGML_ASSERT(backend != nullptr);
    const std::vector<int64_t> parts = even_parts(v, n_parts);

    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 0.05f);
    std::vector<float> wf(k*v), w2f(kb*v), xf(k*n_tok), xbf(kb*n_tok);
    for (auto * vec : {&wf, &w2f}) {
        for (auto & a : *vec) {
            a = nd(rng);
        }
    }
    for (auto & a : xf) {
        a = nd(rng) * 20.0f;
    }
    for (auto & a : xbf) {
        a = nd(rng) * 5.0f;
    }

    ggml_init_params wp = { ggml_tensor_overhead()*4, nullptr, true };
    ggml_context * ctx_w = ggml_init(wp);
    ggml_tensor * w  = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, k, v);
    ggml_tensor * w2 = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, kb, v);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    ggml_backend_tensor_set(w,  wf.data(),  0, ggml_nbytes(w));
    ggml_backend_tensor_set(w2, w2f.data(), 0, ggml_nbytes(w2));

    ggml_init_params gp = { ggml_tensor_overhead()*128 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k,  n_tok);    ggml_set_input(x);
    ggml_tensor * xb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kb, n_blocks, block); ggml_set_input(xb);
    ggml_tensor * base  = ggml_mul_mat(ctx, split ? ggml_backend_meta_rows(ctx, w, n_parts, parts.data()) : w, x);
    ggml_tensor * basew = ggml_mul_mat(ctx, w, x);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    std::vector<ggml_tensor *> picks, cols;
    for (int64_t i = 0; i < block; i++) {
        ggml_tensor * xb_i   = ggml_view_2d(ctx, xb, kb, n_blocks, xb->nb[1], i*xb->nb[2]);
        ggml_tensor * bias   = ggml_mul_mat(ctx, split ? ggml_backend_meta_rows(ctx, w2, n_parts, parts.data()) : w2, xb_i);
        ggml_tensor * base_i = ggml_view_2d(ctx, base, v, n_blocks, block*base->nb[1], i*base->nb[1]);
        ggml_tensor * col    = ggml_add(ctx, base_i, bias);
        ggml_tensor * pick   = ggml_argmax_merge(ctx, ggml_backend_meta_gather(ctx, ggml_argmax_part(ctx, col, n_parts)));
        ggml_tensor * colw   = ggml_add(ctx, ggml_cont(ctx, ggml_view_2d(ctx, basew, v, n_blocks, block*basew->nb[1], i*basew->nb[1])),
                                         ggml_mul_mat(ctx, w2, xb_i));
        for (ggml_tensor * t : {pick, colw}) {
            ggml_set_output(t);
            ggml_build_forward_expand(gf, t);
        }
        picks.push_back(pick);
        cols.push_back(colw);
    }
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, gf));
    ggml_backend_tensor_set(x,  xf.data(),  0, ggml_nbytes(x));
    ggml_backend_tensor_set(xb, xbf.data(), 0, ggml_nbytes(xb));
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    int n_fail = 0;
    std::string got, want;
    for (int64_t i = 0; i < block; i++) {
        std::vector<int32_t> pk(n_blocks);
        std::vector<float> cw(v*n_blocks);
        ggml_backend_tensor_get(picks[i], pk.data(), 0, ggml_nbytes(picks[i]));
        ggml_backend_tensor_get(cols[i],  cw.data(), 0, ggml_nbytes(cols[i]));
        for (int64_t b = 0; b < n_blocks; b++) {
            const int32_t ref = ref_argmax(&cw[b*v], v);
            n_fail += pk[b] != ref;
            got  += " " + std::to_string(pk[b]);
            want += " " + std::to_string(ref);
        }
    }
    printf("  %s markov head, %lld block(s) of %lld positions, head %s over %d devices: picks%s, want%s %s\n",
        ggml_backend_dev_name(dev), (long long) n_blocks, (long long) block, split ? "split by rows" : "whole", n_parts,
        got.c_str(), want.c_str(), n_fail == 0 ? "ok" : "FAIL");

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_buffer_free(wbuf);
    ggml_free(ctx_w);
    ggml_backend_free(backend);
    return n_fail;
}

int main() {
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
    // the DeepSeek-V4.1 DSpark shapes: the Q6_K output head (3 draft columns) and the BF16 Markov transition (1 column).
    // TEST_VOCAB=129280 runs them at the drafter's vocabulary (several GiB on the CPU); the default keeps the test small
    const int64_t v = getenv("TEST_VOCAB") ? atoll(getenv("TEST_VOCAB")) : 4096;
    printf("vocabulary %lld\n", (long long) v);
    for (int n_parts : {2, 3}) {
        n_fail += test_meta_nan(dev, even_parts(1000, n_parts));
        n_fail += test_meta_offset_view(dev, even_parts(1000, n_parts));
        n_fail += test_meta_whole_rows(dev, n_parts);
    }
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
    // the Markov head's graph: one sequence split by rows, two sequences on the whole head
    for (int n_parts : {2, 3}) {
        n_fail += test_markov_blocks(dev, v, 1, n_parts);
        n_fail += test_markov_blocks(dev, v, 2, n_parts);
    }
    printf("%s\n", n_fail == 0 ? "OK" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
