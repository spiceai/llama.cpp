// A meta backend's async read must return what its synchronous read returns: a tensor split in several segments is
// read segment by segment from every device, whole or from a row at an offset, as llama.cpp reads one sequence's cells
// of a recurrent state. A device can hold an empty slice of every tensor in a buffer (a split that gives it no share of
// them); its part of that buffer is then a zero-sized dummy whose tensors have no data, and a read must skip it as the
// synchronous read does. This test splits tensors across two slices of the CPU device, writes them, and checks every
// async read against the bytes written.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Segments along the split axis, each split between the two devices: {device 0, device 1} elements per segment.
struct test_split {
    ggml_backend_meta_split_axis axis;
    std::vector<std::pair<int64_t, int64_t>> segments;
};

static ggml_backend_meta_split_state split_of(const ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    const test_split * split = (const test_split *) userdata;
    ggml_backend_meta_split_state state = {};
    state.axis       = split->axis;
    state.n_segments = split->segments.size();
    for (size_t s = 0; s < split->segments.size(); s++) {
        state.ne[2*s + 0] = split->segments[s].first;
        state.ne[2*s + 1] = split->segments[s].second;
        state.nr[s]       = 1;
    }
    return state;
}

// Writes a tensor of shape ne split as `split`, then reads it whole and from each row, with the synchronous read and
// then the async one, and checks that every read returns the bytes written.
static bool check(const char * name, const test_split & split, int64_t ne0, int64_t ne1, int64_t ne2) {
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_dev_t simple_devs[2] = { cpu_dev, cpu_dev };
    ggml_backend_dev_t meta_dev = ggml_backend_meta_device(simple_devs, 2, split_of, (void *) &split);
    ggml_backend_t backend = ggml_backend_dev_init(meta_dev, nullptr);
    GGML_ASSERT(backend != nullptr);

    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*2,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, ne2);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(meta_dev));
    GGML_ASSERT(buf != nullptr);

    std::vector<float> written(ggml_nelements(t));
    for (size_t i = 0; i < written.size(); i++) {
        written[i] = (float) i;
    }
    ggml_backend_tensor_set(t, written.data(), 0, ggml_nbytes(t));

    bool ok = true;
    const int     row_dim    = split.axis + 1;
    const size_t  row_stride = t->nb[row_dim];
    const int64_t n_rows     = t->ne[row_dim];

    // the whole tensor, then each row at its offset: synchronously, then asynchronously
    for (const bool async : { false, true }) {
        const char * how = async ? "async" : "sync";
        auto get = [&](void * dst, size_t offset, size_t size) {
            if (async) {
                ggml_backend_tensor_get_async(backend, t, dst, offset, size);
                ggml_backend_synchronize(backend);
            } else {
                ggml_backend_tensor_get(t, dst, offset, size);
            }
        };
        std::vector<float> read(written.size(), -1.0f);
        get(read.data(), 0, ggml_nbytes(t));
        if (memcmp(read.data(), written.data(), ggml_nbytes(t)) != 0) {
            fprintf(stderr, "%s: the whole tensor %s read differs from what was written\n", name, how);
            ok = false;
        }
        for (int64_t r = 0; r < n_rows; r++) {
            std::vector<float> row(row_stride/sizeof(float), -1.0f);
            get(row.data(), r*row_stride, row_stride);
            if (memcmp(row.data(), (const char *) written.data() + r*row_stride, row_stride) != 0) {
                fprintf(stderr, "%s: row %lld %s read differs from what was written\n", name, (long long) r, how);
                ok = false;
            }
        }
        printf("%s: %s read %s\n", name, how, ok ? "matches" : "differs");
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return ok;
}

int main() {
    GGML_ASSERT(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU) != nullptr);

    // rows along dim 1, segments interleaved across the two devices within each row
    const test_split axis0      = { GGML_BACKEND_SPLIT_AXIS_0, { {3, 1}, {2, 2} } };
    // device 1 holds an empty slice of every tensor in the buffer, so its part is a dummy with no data
    const test_split axis0_zero = { GGML_BACKEND_SPLIT_AXIS_0, { {4, 0}, {4, 0} } };
    // rows along dim 2, segments of whole dim-0 rows
    const test_split axis1      = { GGML_BACKEND_SPLIT_AXIS_1, { {2, 1}, {1, 2} } };
    const test_split axis1_zero = { GGML_BACKEND_SPLIT_AXIS_1, { {3, 0}, {3, 0} } };

    bool ok = true;
    ok = check("axis 0, two segments",                       axis0,      8, 4, 1) && ok;
    ok = check("axis 0, two segments, device 1 empty",       axis0_zero, 8, 4, 1) && ok;
    ok = check("axis 1, two segments",                       axis1,      4, 6, 3) && ok;
    ok = check("axis 1, two segments, device 1 empty",       axis1_zero, 4, 6, 3) && ok;
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
