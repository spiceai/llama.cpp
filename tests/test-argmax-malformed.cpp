// ARGMAX_PART and ARGMAX_MERGE nodes built without ggml_argmax_part and ggml_argmax_merge, as an RPC server deserializes
// a client's graph: a destination that does not fit what the kernel writes, or of the wrong type, or a source of the wrong
// layout. The CPU kernel must stop at an assertion before it writes anything. Each case computes in a child process on the
// CPU backend, on tensors whose data lives in memory shared with this process: the destination is followed by canaries, so
// a write past it shows here even when the child aborted. A malformed case passes when the child aborted (SIGABRT) with
// every canary intact. Two well-formed nodes must compute their picks with the canaries intact, so the harness is seen to
// tell a normal run from an abort.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef _WIN32
#include <csignal>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#ifndef _WIN32

static const size_t REGION   = 4096; // bytes shared with the child
static const size_t DST_OFFS = 1024; // the source before it, the destination and its canaries from here on
static const float  CANARY   = 12345.0f;

struct node_case {
    const char * what;
    ggml_op      op;
    ggml_type    src_type;
    int64_t      src_ne[3];
    ggml_type    dst_type;
    int64_t      dst_ne[2];
    bool         valid;
};

static bool same_bits(float a, float b) {
    return memcmp(&a, &b, sizeof(float)) == 0;
}

static int run_case(const node_case & c, uint8_t * region) {
    ggml_init_params params = { ggml_tensor_overhead()*4 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * src = ggml_new_tensor_3d(ctx, c.src_type, c.src_ne[0], c.src_ne[1], c.src_ne[2]);
    ggml_tensor * dst = ggml_new_tensor_2d(ctx, c.dst_type, c.dst_ne[0], c.dst_ne[1]);
    // what an RPC server's deserialized node holds: the op and its source, with no constructor's checks
    dst->op     = c.op;
    dst->src[0] = src; // op params zero: ARGMAX_PART's origin 0
    src->data = region;
    dst->data = region + DST_OFFS;
    GGML_ASSERT(ggml_nbytes(src) <= DST_OFFS);

    // sources: values 0, 1, 2, ... along each row (a part's pick is the last); pairs (k, 10*k) (a merge's pick is the last)
    float * x = (float *) region;
    for (int64_t i = 0; i < (int64_t) (ggml_nbytes(src) / sizeof(float)); i++) {
        const int64_t i0 = i % c.src_ne[0];
        x[i] = c.op == GGML_OP_ARGMAX_MERGE ? (i0 % 2 == 0 ? (float) (i0/2) : (float) (10*(i0/2))) : (float) i0;
    }
    float * canaries = (float *) (region + DST_OFFS);
    for (size_t i = 0; i < (REGION - DST_OFFS) / sizeof(float); i++) {
        canaries[i] = CANARY;
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    fflush(stdout);
    const pid_t pid = fork();
    GGML_ASSERT(pid >= 0);
    if (pid == 0) {
        setenv("GGML_NO_BACKTRACE", "1", 1);
        ggml_backend_t backend = ggml_backend_cpu_init();
        const ggml_status st = ggml_backend_graph_compute(backend, gf);
        _exit(st == GGML_STATUS_SUCCESS ? 0 : 3);
    }
    int status = 0;
    GGML_ASSERT(waitpid(pid, &status, 0) == pid);
    const bool aborted  = WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
    const bool computed = WIFEXITED(status) && WEXITSTATUS(status) == 0;

    // a canary after the destination that changed is a write past it
    const size_t dst_end = DST_OFFS + ggml_nbytes(dst);
    int64_t n_over = 0, first_over = -1;
    for (size_t offs = dst_end; offs + sizeof(float) <= REGION; offs += sizeof(float)) {
        float v;
        memcpy(&v, region + offs, sizeof(v));
        if (!same_bits(v, CANARY)) {
            n_over++;
            first_over = first_over < 0 ? (int64_t) (offs - dst_end) : first_over;
        }
    }

    bool ok;
    std::string got;
    if (c.valid) {
        // the picks of the well-formed nodes: ARGMAX_PART (value, index) = (n - 1, n - 1) then (-inf, -1) pairs; ARGMAX_MERGE
        // the index of the last pair
        bool picks = true;
        for (int64_t r = 0; r < c.dst_ne[c.op == GGML_OP_ARGMAX_MERGE ? 0 : 1]; r++) {
            if (c.op == GGML_OP_ARGMAX_PART) {
                const float * y = (const float *) (region + DST_OFFS) + r*c.dst_ne[0];
                const float n1  = (float) (c.src_ne[0] - 1);
                picks = picks && same_bits(y[0], n1) && same_bits(y[1], n1);
                for (int64_t k = 2; k < c.dst_ne[0]; k += 2) {
                    picks = picks && same_bits(y[k], -INFINITY) && same_bits(y[k + 1], -1.0f);
                }
            } else {
                const int32_t * y = (const int32_t *) (region + DST_OFFS);
                picks = picks && y[r] == (int32_t) (10*(c.src_ne[0]/2 - 1));
            }
        }
        ok = computed && n_over == 0 && picks;
        got = std::string(computed ? "computed" : "did not compute") + (picks ? ", picks right" : ", picks WRONG");
    } else {
        ok = aborted && n_over == 0;
        got = aborted ? "refused (SIGABRT)" : computed ? "computed, NOT refused" : "ended otherwise";
    }
    printf("  %-13s %-52s %s; %lld canaries after the destination overwritten", ggml_op_name(c.op), c.what, got.c_str(),
        (long long) n_over);
    if (first_over >= 0) {
        printf(" (first %lld bytes past it)", (long long) first_over);
    }
    printf(" %s\n", ok ? "ok" : "FAIL");

    ggml_free(ctx);
    return ok ? 0 : 1;
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    uint8_t * region = (uint8_t *) mmap(nullptr, REGION, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANON, -1, 0);
    GGML_ASSERT(region != MAP_FAILED);

    const node_case cases[] = {
        { "well-formed: F32 [4,2] -> F32 pairs [4,2]",           GGML_OP_ARGMAX_PART,  GGML_TYPE_F32, {4, 2, 1}, GGML_TYPE_F32, {4, 2}, true  },
        { "well-formed: F32 pairs [4,2] -> I32 [2]",             GGML_OP_ARGMAX_MERGE, GGML_TYPE_F32, {4, 2, 1}, GGML_TYPE_I32, {2, 1}, true  },
        { "F32 [1,1] -> F32 [1,1]: a destination of one value",   GGML_OP_ARGMAX_PART,  GGML_TYPE_F32, {1, 1, 1}, GGML_TYPE_F32, {1, 1}, false },
        { "F32 [4,2] -> F32 [2,1]: fewer destination rows",       GGML_OP_ARGMAX_PART,  GGML_TYPE_F32, {4, 2, 1}, GGML_TYPE_F32, {2, 1}, false },
        { "F32 [4,1] -> F32 [3,1]: an odd destination width",     GGML_OP_ARGMAX_PART,  GGML_TYPE_F32, {4, 1, 1}, GGML_TYPE_F32, {3, 1}, false },
        { "F32 [4,1] -> I32 [2,1]: an I32 destination",           GGML_OP_ARGMAX_PART,  GGML_TYPE_F32, {4, 1, 1}, GGML_TYPE_I32, {2, 1}, false },
        { "F32 [4,1,2] -> F32 [2,1]: a source that is no matrix", GGML_OP_ARGMAX_PART,  GGML_TYPE_F32, {4, 1, 2}, GGML_TYPE_F32, {2, 1}, false },
        { "F32 pairs [2,2] -> I32 [1]: one pick for two rows",    GGML_OP_ARGMAX_MERGE, GGML_TYPE_F32, {2, 2, 1}, GGML_TYPE_I32, {1, 1}, false },
        { "F32 pairs [3,1] -> I32 [1]: an odd pairs width",       GGML_OP_ARGMAX_MERGE, GGML_TYPE_F32, {3, 1, 1}, GGML_TYPE_I32, {1, 1}, false },
        { "F32 pairs [2,1] -> F32 [1]: an F32 destination",       GGML_OP_ARGMAX_MERGE, GGML_TYPE_F32, {2, 1, 1}, GGML_TYPE_F32, {1, 1}, false },
    };
    int n_fail = 0;
    for (const node_case & c : cases) {
        n_fail += run_case(c, region);
    }
    munmap(region, REGION);
    printf("%s\n", n_fail == 0 ? "OK" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}

#else

int main() {
    printf("skipped: the cases run in child processes, which this test forks\n");
    return 0;
}

#endif
