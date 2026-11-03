// The NCCL wire two RPC servers agree on (ggml-rpc/nccl-wire.h), for every ordered pair of ranks. A rank is a server and
// the backend it loads, each from before the BF16 wire was negotiated ("pre-wish"), from before gathers were kept exact
// on it ("old"), or this build ("new"), with or without GGML_NCCL_WIRE_BF16: nine kinds of rank, two wishes, every pair
// in both rank orders.
//
// A pre-wish server sends only whether it has a communicator, and never sets the wire. An old server sends
// RPC_NCCL_WIRE_BF16 for its wish, enables the wire when the peer sent that bit too, and calls the backend's
// ggml_backend_cuda_nccl_comm_set_wire_bf16. A new server runs nccl-wire.h, as ggml-rpc.cpp does. A pre-wish backend
// exports none of the wire's functions. An old backend rounds gathers on the BF16 wire; a new one rounds them after
// set_wire_bf16 and keeps them F32 after set_wire_bf16_exact, which only it exports.
//
// The backends here model the CUDA backend's communicator state (ggml_cuda_nccl_comm in ggml-cuda.cu): the CUDA setters'
// own effect needs an NCCL communicator, so it is unverified by this test. What it does check of the real CUDA backend,
// when the build has one with NCCL, is that it exports both setters, which a new server reads as the capability.
//
// NCCL sums one datatype per collective, so both ranks must pick the same wire and, on the BF16 wire, treat gathers the
// same: a pair that disagrees hangs or corrupts results. The wire is BF16 only when both ranks send the same wish bit,
// with gathers kept F32 only when both ranks' servers and backends are new.

#include "../ggml/src/ggml-rpc/nccl-wire.h"
#include "ggml-backend.h"

#include <cstdio>
#include <string>

struct fake_comm {
    bool wire_bf16     = false;
    bool gathers_exact = false;
};

static bool wanted_yes(void) { return true; }
static bool wanted_no(void)  { return false; }

// an old backend's set_wire_bf16, and a new backend's: the BF16 wire, gathers rounded
static void set_wire_bf16(void * comm, bool on) {
    ((fake_comm *) comm)->wire_bf16     = on;
    ((fake_comm *) comm)->gathers_exact = false;
}

// a new backend's set_wire_bf16_exact: the BF16 wire, gathers F32
static void set_wire_bf16_exact(void * comm, bool on) {
    ((fake_comm *) comm)->wire_bf16     = on;
    ((fake_comm *) comm)->gathers_exact = on;
}

enum build { PRE_WISH, OLD, NEW };

struct rank_cfg {
    build server;
    build backend;
    bool  wish;
};

static rpc_nccl_wire_procs procs_of(const rank_cfg & r) {
    if (r.backend == PRE_WISH) {
        return { nullptr, nullptr, nullptr };
    }
    return { r.wish ? wanted_yes : wanted_no, set_wire_bf16, r.backend == NEW ? set_wire_bf16_exact : nullptr };
}

// the byte a rank sends
static uint8_t word_of(const rank_cfg & r) {
    const rpc_nccl_wire_procs procs = procs_of(r);
    switch (r.server) {
        case PRE_WISH: return RPC_NCCL_INITED;
        case OLD:      return RPC_NCCL_INITED | (rpc_nccl_wire_wanted(true, procs) ? RPC_NCCL_WIRE_BF16 : 0);
        default:       return rpc_nccl_word(true, rpc_nccl_wire_wanted(true, procs), procs);
    }
}

// the communicator a rank ends with, from its byte and its peer's
static fake_comm comm_of(const rank_cfg & r, uint8_t self, uint8_t peer) {
    const rpc_nccl_wire_procs procs = procs_of(r);
    fake_comm comm;
    switch (r.server) {
        case PRE_WISH:
            break;
        case OLD:
            // the old server never looks up set_wire_bf16_exact
            if (procs.set_wire != nullptr) {
                procs.set_wire(&comm, rpc_nccl_wire_wanted(true, procs) && (peer & RPC_NCCL_WIRE_BF16) != 0);
            }
            break;
        default:
            rpc_nccl_wire_set(&comm, procs, rpc_nccl_wire_agree(self, peer));
            break;
    }
    return comm;
}

static std::string name_of(const rank_cfg & r) {
    static const char * builds[3] = { "pre-wish", "old", "new" };
    return std::string(builds[r.server]) + " server, " + builds[r.backend] + " backend" + (r.wish ? " +bf16" : "");
}

// the CUDA backend of this build, when it has NCCL, must export both setters: a new server reads the exact one as the
// capability, and without it a pair of new ranks would agree on the legacy wire and round gathers
static int check_cuda_exports() {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
    if (reg == nullptr || ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_nccl_comm_init_rank") == nullptr) {
        printf("CUDA backend with NCCL: none in this build, export check skipped\n");
        return 0;
    }
    int n_fail = 0;
    for (const char * name : { "ggml_backend_cuda_nccl_wire_bf16_wanted", "ggml_backend_cuda_nccl_comm_set_wire_bf16",
                               "ggml_backend_cuda_nccl_comm_set_wire_bf16_exact" }) {
        const bool ok = ggml_backend_reg_get_proc_address(reg, name) != nullptr;
        n_fail += !ok;
        printf("CUDA backend exports %s: %s\n", name, ok ? "ok" : "FAIL");
    }
    return n_fail;
}

int main() {
    ggml_backend_load_all();
    int n_fail = 0, n_cases = 0;
    for (int a = 0; a < 18; a++) {
        for (int b = 0; b < 18; b++) {
            const rank_cfg r0 = { (build) (a % 3), (build) (a / 3 % 3), a / 9 != 0 };
            const rank_cfg r1 = { (build) (b % 3), (build) (b / 3 % 3), b / 9 != 0 };
            const uint8_t   w0 = word_of(r0), w1 = word_of(r1);
            const fake_comm c0 = comm_of(r0, w0, w1);
            const fake_comm c1 = comm_of(r1, w1, w0);
            // on the BF16 wire, a gather travels as F32 on a rank whose communicator keeps gathers exact
            const bool gathers_agree = !c0.wire_bf16 || c0.gathers_exact == c1.gathers_exact;
            // a rank asks for the wire when its server and its backend can and GGML_NCCL_WIRE_BF16 is set, and keeps
            // gathers exact on it only when both are new
            const bool asks0     = r0.wish && r0.server != PRE_WISH && r0.backend != PRE_WISH;
            const bool asks1     = r1.wish && r1.server != PRE_WISH && r1.backend != PRE_WISH;
            const bool exact0    = r0.server == NEW && r0.backend == NEW;
            const bool exact1    = r1.server == NEW && r1.backend == NEW;
            const bool want_bf16  = asks0 && asks1 && exact0 == exact1;
            const bool want_exact = want_bf16 && exact0;
            const bool ok = c0.wire_bf16 == c1.wire_bf16 && gathers_agree && c0.wire_bf16 == want_bf16 &&
                            (!c0.wire_bf16 || c0.gathers_exact == want_exact);
            n_fail += !ok;
            n_cases++;
            printf("rank 0 %s, rank 1 %s: wire %s / %s, gathers %s / %s%s %s\n", name_of(r0).c_str(), name_of(r1).c_str(),
                c0.wire_bf16 ? "BF16" : "F32", c1.wire_bf16 ? "BF16" : "F32",
                !c0.wire_bf16 || c0.gathers_exact ? "F32" : "BF16", !c1.wire_bf16 || c1.gathers_exact ? "F32" : "BF16",
                gathers_agree ? "" : ", gathers disagree", ok ? "ok" : "FAIL");
        }
    }
    printf("%d of %d pairs agree%s\n", n_cases - n_fail, n_cases, n_fail == 0 ? "" : ": FAILED");
    n_fail += check_cuda_exports();
    printf("%s\n", n_fail == 0 ? "OK" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
