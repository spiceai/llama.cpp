// The NCCL wire two RPC servers agree on (ggml-rpc/nccl-wire.h), for every ordered pair of ranks. A rank is a server and
// the backend it loads, each from before gathers were kept exact on the BF16 wire ("old") or this build ("new"), with or
// without GGML_NCCL_WIRE_BF16: four kinds of rank, two wishes, every pair in both rank orders.
//
// An old server sends RPC_NCCL_WIRE_BF16 for its wish, enables the wire when the peer sent that bit too, and calls the
// backend's ggml_backend_cuda_nccl_comm_set_wire_bf16. A new server runs nccl-wire.h, as ggml-rpc.cpp does. An old
// backend rounds gathers on the BF16 wire; a new one rounds them after set_wire_bf16 and keeps them F32 after
// set_wire_bf16_exact, which only it exports. The backends here model the CUDA backend's communicator state
// (ggml_cuda_nccl_comm in ggml-cuda.cu); the CUDA functions themselves need an NCCL build.
//
// NCCL sums one datatype per collective, so both ranks must pick the same wire and, on the BF16 wire, treat gathers the
// same: a pair that disagrees hangs or corrupts results. The wire is BF16 only when both ranks send the same wish bit,
// with gathers kept F32 only when both ranks' servers and backends are new.

#include "../ggml/src/ggml-rpc/nccl-wire.h"

#include <cstdio>

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

struct rank_cfg {
    bool new_server;
    bool new_backend;
    bool wish;
};

static rpc_nccl_wire_procs procs_of(const rank_cfg & r) {
    return { r.wish ? wanted_yes : wanted_no, set_wire_bf16, r.new_backend ? set_wire_bf16_exact : nullptr };
}

// the byte a rank sends
static uint8_t word_of(const rank_cfg & r) {
    const rpc_nccl_wire_procs procs = procs_of(r);
    const bool wire_wanted = rpc_nccl_wire_wanted(true, procs);
    if (r.new_server) {
        return rpc_nccl_word(true, wire_wanted, procs);
    }
    return RPC_NCCL_INITED | (wire_wanted ? RPC_NCCL_WIRE_BF16 : 0);
}

// the communicator a rank ends with, from its byte and its peer's
static fake_comm comm_of(const rank_cfg & r, uint8_t self, uint8_t peer) {
    const rpc_nccl_wire_procs procs = procs_of(r);
    fake_comm comm;
    if (r.new_server) {
        rpc_nccl_wire_set(&comm, procs, rpc_nccl_wire_agree(self, peer));
    } else {
        // the old server never looks up set_wire_bf16_exact
        procs.set_wire(&comm, rpc_nccl_wire_wanted(true, procs) && (peer & RPC_NCCL_WIRE_BF16) != 0);
    }
    return comm;
}

static const char * name_of(const rank_cfg & r) {
    static const char * names[2][2][2] = {
        { { "old server, old backend",       "old server, old backend +bf16" },
          { "old server, new backend",       "old server, new backend +bf16" } },
        { { "new server, old backend",       "new server, old backend +bf16" },
          { "new server, new backend",       "new server, new backend +bf16" } },
    };
    return names[r.new_server][r.new_backend][r.wish];
}

int main() {
    int n_fail = 0, n_cases = 0;
    for (int a = 0; a < 8; a++) {
        for (int b = 0; b < 8; b++) {
            const rank_cfg r0 = { (a & 1) != 0, (a & 2) != 0, (a & 4) != 0 };
            const rank_cfg r1 = { (b & 1) != 0, (b & 2) != 0, (b & 4) != 0 };
            const uint8_t   w0 = word_of(r0), w1 = word_of(r1);
            const fake_comm c0 = comm_of(r0, w0, w1);
            const fake_comm c1 = comm_of(r1, w1, w0);
            // on the BF16 wire, a gather travels as F32 on a rank whose communicator keeps gathers exact
            const bool gathers_agree = !c0.wire_bf16 || c0.gathers_exact == c1.gathers_exact;
            // a rank keeps gathers exact on the wire only when its server and its backend are both new
            const bool exact0    = r0.new_server && r0.new_backend;
            const bool exact1    = r1.new_server && r1.new_backend;
            const bool want_bf16 = r0.wish && r1.wish && exact0 == exact1;
            const bool want_exact = want_bf16 && exact0;
            const bool ok = c0.wire_bf16 == c1.wire_bf16 && gathers_agree && c0.wire_bf16 == want_bf16 &&
                            (!c0.wire_bf16 || c0.gathers_exact == want_exact);
            n_fail += !ok;
            n_cases++;
            printf("rank 0 %s, rank 1 %s: wire %s / %s, gathers %s / %s%s %s\n", name_of(r0), name_of(r1),
                c0.wire_bf16 ? "BF16" : "F32", c1.wire_bf16 ? "BF16" : "F32",
                !c0.wire_bf16 || c0.gathers_exact ? "F32" : "BF16", !c1.wire_bf16 || c1.gathers_exact ? "F32" : "BF16",
                gathers_agree ? "" : ", gathers disagree", ok ? "ok" : "FAIL");
        }
    }
    printf("%d of %d pairs %s\n", n_cases - n_fail, n_cases, n_fail == 0 ? "agree: OK" : "agree: FAILED");
    return n_fail == 0 ? 0 : 1;
}
