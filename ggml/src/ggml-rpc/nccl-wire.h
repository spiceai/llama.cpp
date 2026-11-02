#pragma once

// The byte the two servers of an RPC pair exchange once each has tried to initialize its NCCL communicator, and the
// wire the pair agrees on from the two bytes. NCCL sums one datatype per collective, so the two ranks must agree on the
// wire for every all-reduce, gathers (ggml_is_meta_gather) included.

#include <cstdint>

static constexpr uint8_t RPC_NCCL_INITED          = 1u; // this rank has a communicator
// asks for the BF16 wire with every large F32 reduction on it, gathers included: what a server sent before gathers were
// kept exact. Read for the warning, never sent: a rank that sends it rounds gathers that this rank keeps F32
static constexpr uint8_t RPC_NCCL_WIRE_BF16       = 2u;
// asks for the BF16 wire (GGML_NCCL_WIRE_BF16) with gathers kept F32, the only wish this rank sends
static constexpr uint8_t RPC_NCCL_WIRE_BF16_EXACT = 4u;

// this rank's byte: its communicator, and its wish for the BF16 wire
static inline uint8_t rpc_nccl_word(bool inited, bool wire_wanted) {
    return (inited ? RPC_NCCL_INITED : 0) | (inited && wire_wanted ? RPC_NCCL_WIRE_BF16_EXACT : 0);
}

// whether the pair runs the BF16 wire: only when both ranks ask for it with gathers kept F32. A rank that predates the
// exception never sets RPC_NCCL_WIRE_BF16_EXACT, and reads only RPC_NCCL_WIRE_BF16, which this rank never sends, so a
// pair of the two keeps the F32 wire on both ranks
static inline bool rpc_nccl_wire_bf16(uint8_t word, uint8_t peer_word) {
    return (word & RPC_NCCL_INITED) && (peer_word & RPC_NCCL_INITED) &&
        (word & RPC_NCCL_WIRE_BF16_EXACT) && (peer_word & RPC_NCCL_WIRE_BF16_EXACT);
}
