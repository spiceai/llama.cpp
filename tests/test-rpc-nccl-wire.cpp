// The NCCL wire two RPC servers agree on (ggml-rpc/nccl-wire.h), for every pair of ranks: each rank a build from before
// gathers were kept exact on the BF16 wire ("old": sends RPC_NCCL_WIRE_BF16 for its wish and enables the wire when both
// ranks set that bit, rounding gathers too) or this build ("new"), each with or without GGML_NCCL_WIRE_BF16, in both rank
// orders. NCCL sums one datatype per collective, so both ranks must pick the same wire and, on the BF16 wire, treat
// gathers the same: a pair that disagrees hangs or corrupts results. This build enables the BF16 wire only when both ranks
// are new and both ask for it.

#include "../ggml/src/ggml-rpc/nccl-wire.h"

#include <cstdio>

struct rank_cfg {
    bool new_build;
    bool wish;
};

// the byte a rank sends
static uint8_t word_of(const rank_cfg & r) {
    if (r.new_build) {
        return rpc_nccl_word(true, r.wish);
    }
    return RPC_NCCL_INITED | (r.wish ? RPC_NCCL_WIRE_BF16 : 0);
}

// the wire a rank picks from its byte and its peer's
static bool bf16_of(const rank_cfg & r, uint8_t self, uint8_t peer) {
    if (r.new_build) {
        return rpc_nccl_wire_bf16(self, peer);
    }
    return r.wish && (peer & RPC_NCCL_WIRE_BF16) != 0;
}

int main() {
    int n_fail = 0, n_cases = 0;
    for (int a = 0; a < 4; a++) {
        for (int b = 0; b < 4; b++) {
            const rank_cfg ranks[2] = { {(a & 1) != 0, (a & 2) != 0}, {(b & 1) != 0, (b & 2) != 0} };
            for (int first = 0; first < 2; first++) {          // rank 0 is either one
                const rank_cfg & r0 = ranks[first];
                const rank_cfg & r1 = ranks[1 - first];
                const uint8_t w0 = word_of(r0), w1 = word_of(r1);
                const bool bf16_0 = bf16_of(r0, w0, w1);
                const bool bf16_1 = bf16_of(r1, w1, w0);
                // on the BF16 wire a new rank keeps gathers F32 and an old one rounds them
                const bool gathers_agree = !bf16_0 || r0.new_build == r1.new_build;
                const bool both_new_wish = r0.new_build && r1.new_build && r0.wish && r1.wish;
                const bool any_new       = r0.new_build || r1.new_build;
                // an old pair keeps its old behaviour; any pair with a new rank runs BF16 only when both are new and ask
                const bool want_bf16 = any_new ? both_new_wish : (r0.wish && r1.wish);
                const bool ok = bf16_0 == bf16_1 && gathers_agree && bf16_0 == want_bf16;
                n_fail += !ok;
                n_cases++;
                printf("rank 0 %s%s, rank 1 %s%s: wire %s / %s%s %s\n",
                    r0.new_build ? "new" : "old", r0.wish ? " +bf16" : "", r1.new_build ? "new" : "old", r1.wish ? " +bf16" : "",
                    bf16_0 ? "BF16" : "F32", bf16_1 ? "BF16" : "F32", gathers_agree ? "" : ", gathers disagree", ok ? "ok" : "FAIL");
            }
        }
    }
    printf("%d of %d pairs %s\n", n_cases - n_fail, n_cases, n_fail == 0 ? "agree: OK" : "agree: FAILED");
    return n_fail == 0 ? 0 : 1;
}
