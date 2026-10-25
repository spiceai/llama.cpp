// The all-reduce between RPC servers, over every endpoint given (a mesh when there are more than two) and then over the
// first two (the pairwise comm): every rank must end with the same bits, the F32 sum of the partials in rank order, for
// a single float, a decode-sized partial, one above the mesh's eager threshold, and one larger than a socket's buffers.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-rpc.h"
#include "ggml.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int run(const std::vector<const char *> & endpoints, uint32_t seed) {
    const size_t world = endpoints.size();
    std::vector<ggml_backend_t> backends;
    for (const char * endpoint : endpoints) {
        ggml_backend_t backend = ggml_backend_rpc_init(endpoint, 0);
        GGML_ASSERT(backend != nullptr);
        backends.push_back(backend);
    }
    ggml_backend_reg_t reg = ggml_backend_rpc_reg();
    auto comm_init      = (ggml_backend_comm_init_t)             ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_init");
    auto comm_allreduce = (ggml_backend_comm_allreduce_tensor_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_tensor");
    auto comm_free      = (ggml_backend_comm_free_t)             ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_free");
    GGML_ASSERT(comm_init != nullptr && comm_allreduce != nullptr && comm_free != nullptr);

    void * comm = comm_init(backends.data(), world);
    if (comm == nullptr) {
        fprintf(stderr, "world %zu: the comm did not initialize\n", world);
        return 1;
    }

    const int64_t sizes[] = { 1, 7168, 300007, 4000037 };
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    int failures = 0;
    for (const int64_t ne : sizes) {
        std::vector<ggml_context *>         ctxs(world);
        std::vector<ggml_tensor *>          tensors(world);
        std::vector<ggml_backend_buffer_t>  buffers(world);
        std::vector<std::vector<float>>     parts(world, std::vector<float>(ne));
        for (size_t r = 0; r < world; r++) {
            ggml_init_params params = {
                /* .mem_size   = */ ggml_tensor_overhead(),
                /* .mem_buffer = */ nullptr,
                /* .no_alloc   = */ true,
            };
            ctxs[r] = ggml_init(params);
            GGML_ASSERT(ctxs[r] != nullptr);
            tensors[r] = ggml_new_tensor_1d(ctxs[r], GGML_TYPE_F32, ne);
            // the comm reduces only partials a graph computed
            tensors[r]->flags |= GGML_TENSOR_FLAG_COMPUTE;
            buffers[r] = ggml_backend_alloc_ctx_tensors(ctxs[r], backends[r]);
            GGML_ASSERT(buffers[r] != nullptr);
            for (int64_t k = 0; k < ne; k++) {
                parts[r][k] = dist(rng);
            }
            ggml_backend_tensor_set(tensors[r], parts[r].data(), 0, ne*sizeof(float));
        }

        // the rank-order sum, computed here rather than by any server
        std::vector<float> want(ne);
        for (int64_t k = 0; k < ne; k++) {
            float sum = parts[0][k];
            for (size_t r = 1; r < world; r++) {
                sum += parts[r][k];
            }
            want[k] = sum;
        }

        if (!comm_allreduce(comm, tensors.data())) {
            fprintf(stderr, "world %zu, %" PRId64 " floats: the comm refused the all-reduce\n", world, ne);
            failures++;
        } else {
            std::vector<float> got(ne);
            for (size_t r = 0; r < world; r++) {
                ggml_backend_tensor_get(tensors[r], got.data(), 0, ne*sizeof(float));
                int64_t first = -1, wrong = 0;
                for (int64_t k = 0; k < ne; k++) {
                    if (memcmp(&got[k], &want[k], sizeof(float)) != 0) {
                        first = first < 0 ? k : first;
                        wrong++;
                    }
                }
                if (wrong != 0) {
                    fprintf(stderr, "world %zu, %" PRId64 " floats, rank %zu: %" PRId64 " floats differ from the rank-order "
                            "sum, first at %" PRId64 " (%.9g, want %.9g)\n", world, ne, r, wrong, first,
                            (double) got[first], (double) want[first]);
                    failures++;
                }
            }
            printf("world %zu, %" PRId64 " floats: %s\n", world, ne, failures == 0 ? "ok" : "FAILED");
        }

        for (size_t r = 0; r < world; r++) {
            ggml_backend_buffer_free(buffers[r]);
            ggml_free(ctxs[r]);
        }
    }

    comm_free(comm);
    for (ggml_backend_t backend : backends) {
        ggml_backend_free(backend);
    }
    return failures;
}

// TEST_RPC_COMM_MESH_BENCH=<n>[:<floats>]: instead of the checks, time n all-reduces of a partial (7168 floats by
// default) over all the endpoints and print the microseconds per all-reduce
static int bench(const std::vector<const char *> & endpoints, const char * spec) {
    long n = 0, ne = 7168;
    if (sscanf(spec, "%ld:%ld", &n, &ne) < 1 || n <= 0 || ne <= 0) {
        fprintf(stderr, "TEST_RPC_COMM_MESH_BENCH must be <count>[:<floats>]\n");
        return 2;
    }
    const size_t world = endpoints.size();
    std::vector<ggml_backend_t> backends;
    for (const char * endpoint : endpoints) {
        backends.push_back(ggml_backend_rpc_init(endpoint, 0));
        GGML_ASSERT(backends.back() != nullptr);
    }
    ggml_backend_reg_t reg = ggml_backend_rpc_reg();
    auto comm_init      = (ggml_backend_comm_init_t)             ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_init");
    auto comm_allreduce = (ggml_backend_comm_allreduce_tensor_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_tensor");
    auto comm_free      = (ggml_backend_comm_free_t)             ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_free");
    void * comm = comm_init(backends.data(), world);
    GGML_ASSERT(comm != nullptr);
    std::vector<ggml_context *> ctxs(world);
    std::vector<ggml_tensor *> tensors(world);
    std::vector<ggml_backend_buffer_t> buffers(world);
    std::vector<float> ones(ne, 1.0f);
    for (size_t r = 0; r < world; r++) {
        ggml_init_params params = { ggml_tensor_overhead(), nullptr, true };
        ctxs[r] = ggml_init(params);
        tensors[r] = ggml_new_tensor_1d(ctxs[r], GGML_TYPE_F32, ne);
        tensors[r]->flags |= GGML_TENSOR_FLAG_COMPUTE;
        buffers[r] = ggml_backend_alloc_ctx_tensors(ctxs[r], backends[r]);
        ggml_backend_tensor_set(tensors[r], ones.data(), 0, ne*sizeof(float));
    }
    float probe = 0.0f;
    for (int round = 0; round < 2; round++) {
        const int64_t t0 = ggml_time_us();
        for (long i = 0; i < n; i++) {
            GGML_ASSERT(comm_allreduce(comm, tensors.data()));
        }
        // the all-reduces are queued on every rank; reading one element back waits for all of them on that rank
        ggml_backend_tensor_get(tensors[world - 1], &probe, 0, sizeof(float));
        ggml_backend_tensor_get(tensors[0], &probe, 0, sizeof(float));
        const int64_t t1 = ggml_time_us();
        printf("%s: %zu ranks, %ld floats: %.1f us per all-reduce over %ld\n", round == 0 ? "warmup" : "timed", world, ne,
               (double) (t1 - t0) / (double) n, n);
    }
    for (size_t r = 0; r < world; r++) {
        ggml_backend_buffer_free(buffers[r]);
        ggml_free(ctxs[r]);
    }
    comm_free(comm);
    for (ggml_backend_t backend : backends) {
        ggml_backend_free(backend);
    }
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <endpoint> <endpoint> [<endpoint>...]\n", argv[0]);
        return 2;
    }
    // one line at a time, so a run that hangs still shows how far it got
    setvbuf(stdout, nullptr, _IOLBF, 0);
    ggml_backend_load_all();
    ggml_time_init();
    std::vector<const char *> endpoints(argv + 1, argv + argc);
    if (const char * spec = getenv("TEST_RPC_COMM_MESH_BENCH")) {
        return bench(endpoints, spec);
    }
    if (endpoints.size() < 3) {
        fprintf(stderr, "the checks need at least three endpoints\n");
        return 2;
    }
    int failures = run(endpoints, 1234);
    // by default the pairwise comm sends a large partial as BF16 and adds it to its own F32 partial, so the two ranks
    // hold different sums; with an F32 wire both hold the exact sum
    setenv("GGML_RPC_COMM_WIRE_F32", "1", 1);
    failures += run({ endpoints[0], endpoints[1] }, 5678);
    if (failures != 0) {
        fprintf(stderr, "%d failures\n", failures);
        return 1;
    }
    return 0;
}
