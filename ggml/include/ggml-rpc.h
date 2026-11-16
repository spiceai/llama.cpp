#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// Ops are sent by number, and the handshake compares only the major (and the minor against the client's), never the
// patch, so peers that differ in their ops or commands must differ in the major to refuse each other at connect rather
// than run each other's ops as others. This branch's ops (GGML_OP_ARGMAX_PART, GGML_OP_ARGMAX_MERGE) and commands differ
// from upstream's, so its major is upstream's (10 at v1.0.0) plus 100, and never equals one upstream sends.
#define RPC_PROTO_MAJOR_VERSION    110
#define RPC_PROTO_MINOR_VERSION    0
#define RPC_PROTO_PATCH_VERSION    0

#ifdef  __cplusplus
static_assert(GGML_OP_COUNT == 101, "GGML_OP_COUNT has changed - update RPC_PROTO_MAJOR_VERSION (the handshake ignores the patch)");
#endif

#define GGML_RPC_MAX_SERVERS       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device);

GGML_BACKEND_API bool ggml_backend_is_rpc(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device);

GGML_BACKEND_API void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total);

GGML_BACKEND_API void ggml_backend_rpc_start_server(
        const char * endpoint,
        const char * cache_dir,
        size_t n_threads,
        size_t n_devices,
        ggml_backend_dev_t * devices);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_reg(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint);

#ifdef  __cplusplus
}
#endif
