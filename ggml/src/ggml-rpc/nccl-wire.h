#pragma once

// The byte the two servers of an RPC pair exchange once each has tried to initialize its NCCL communicator, the wire
// the pair agrees on from the two bytes, and how a server sets it on its communicator. NCCL sums one datatype per
// collective, so the two ranks must agree on the wire for every all-reduce, gathers (ggml_is_meta_gather) included.

#include <cstdint>

static constexpr uint8_t RPC_NCCL_INITED          = 1u; // this rank has a communicator
// asks for the BF16 wire (GGML_NCCL_WIRE_BF16) with every large F32 reduction on it, gathers included: what a server
// sends whose backend cannot keep gathers F32 on the wire, and what every server sent before gathers were kept exact
static constexpr uint8_t RPC_NCCL_WIRE_BF16       = 2u;
// asks for the BF16 wire with gathers kept F32: sent only when this rank's backend exports
// ggml_backend_cuda_nccl_comm_set_wire_bf16_exact, so the capability is the backend's, not the server's
static constexpr uint8_t RPC_NCCL_WIRE_BF16_EXACT = 4u;

typedef bool (*rpc_nccl_wire_bf16_wanted_t)(void);
typedef void (*rpc_nccl_comm_set_wire_bf16_t)(void * comm, bool on);

// what this rank's backend exports for the BF16 wire (ggml_backend_reg_get_proc_address); each may be null
struct rpc_nccl_wire_procs {
    rpc_nccl_wire_bf16_wanted_t   wanted;    // ggml_backend_cuda_nccl_wire_bf16_wanted: GGML_NCCL_WIRE_BF16 is set
    rpc_nccl_comm_set_wire_bf16_t set_wire;  // ggml_backend_cuda_nccl_comm_set_wire_bf16: the BF16 wire, gathers rounded
    rpc_nccl_comm_set_wire_bf16_t set_exact; // ggml_backend_cuda_nccl_comm_set_wire_bf16_exact: the BF16 wire, gathers F32
};

enum rpc_nccl_wire {
    RPC_NCCL_WIRE_AGREED_F32,        // every reduction F32
    RPC_NCCL_WIRE_AGREED_BF16,       // large F32 reductions BF16, gathers included (both ranks round them)
    RPC_NCCL_WIRE_AGREED_BF16_EXACT, // large F32 reductions BF16, gathers F32 on both ranks
};

// this rank asks for the BF16 wire: it has a communicator, and its backend can set the wire and was asked to
static inline bool rpc_nccl_wire_wanted(bool inited, const rpc_nccl_wire_procs & procs) {
    return inited && procs.wanted != nullptr && procs.set_wire != nullptr && procs.wanted();
}

// this rank's byte: its communicator, its wish for the BF16 wire, and whether its backend keeps gathers exact on it
static inline uint8_t rpc_nccl_word(bool inited, bool wire_wanted, const rpc_nccl_wire_procs & procs) {
    if (!inited) {
        return 0;
    }
    if (!wire_wanted) {
        return RPC_NCCL_INITED;
    }
    return RPC_NCCL_INITED | (procs.set_exact != nullptr ? RPC_NCCL_WIRE_BF16_EXACT : RPC_NCCL_WIRE_BF16);
}

// the wire of the pair: BF16 only when both ranks send the same wish bit, F32 otherwise. A server from before gathers
// were kept exact sends RPC_NCCL_WIRE_BF16 for its wish, enables BF16 when the peer sent it too, and rounds gathers on
// it, so this rule gives both ranks of every pair the same wire and the same gathers
static inline rpc_nccl_wire rpc_nccl_wire_agree(uint8_t word, uint8_t peer_word) {
    if (!(word & RPC_NCCL_INITED) || !(peer_word & RPC_NCCL_INITED)) {
        return RPC_NCCL_WIRE_AGREED_F32;
    }
    if ((word & RPC_NCCL_WIRE_BF16_EXACT) && (peer_word & RPC_NCCL_WIRE_BF16_EXACT)) {
        return RPC_NCCL_WIRE_AGREED_BF16_EXACT;
    }
    if ((word & RPC_NCCL_WIRE_BF16) && (peer_word & RPC_NCCL_WIRE_BF16)) {
        return RPC_NCCL_WIRE_AGREED_BF16;
    }
    return RPC_NCCL_WIRE_AGREED_F32;
}

// sets the agreed wire on this rank's communicator. RPC_NCCL_WIRE_AGREED_BF16_EXACT needs this rank's
// RPC_NCCL_WIRE_BF16_EXACT, which rpc_nccl_word sends only when set_exact exists
static inline void rpc_nccl_wire_set(void * comm, const rpc_nccl_wire_procs & procs, rpc_nccl_wire wire) {
    if (wire == RPC_NCCL_WIRE_AGREED_BF16_EXACT && procs.set_exact != nullptr) {
        procs.set_exact(comm, true);
    } else if (procs.set_wire != nullptr) {
        procs.set_wire(comm, wire != RPC_NCCL_WIRE_AGREED_F32);
    }
}
