#include <algorithm>
#include <cstdint>

#include "argmax.cuh"
#include "common.cuh"
#include "sum.cuh"

static __global__ void argmax_f32(const float * __restrict__ x, int32_t * __restrict__ dst, const int64_t ncols) {
    const int64_t row = blockIdx.x;

    float maxval = -FLT_MAX;
    int   argmax = -1;
    const float * rowx = x + row * ncols;

    for (int32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        const float val = rowx[col];
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }

#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
        const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }

    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (n_warps > 1) {
        constexpr int    max_warps = 1024 / WARP_SIZE;
        __shared__ float shared_maxval[max_warps];
        __shared__ int   shared_argmax[max_warps];
        if (lane_id == 0) {
            shared_maxval[warp_id] = maxval;
            shared_argmax[warp_id] = argmax;
        }

        __syncthreads();

        if (warp_id == 0) {
            if (lane_id < n_warps) {
                maxval = shared_maxval[lane_id];
                argmax = shared_argmax[lane_id];
            }
#pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
                const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
                if (val > maxval) {
                    maxval = val;
                    argmax = col;
                }
            }
        }
    }

    if (warp_id == 0 && lane_id == 0) {
        dst[row] = argmax;
    }
}

void ggml_cuda_argmax(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const float * src0_d = (const float *) src0->data;
    int32_t     * dst_d  = (int32_t     *) dst->data;

    cudaStream_t stream = ctx.stream();

    const int64_t num_blocks = nrows;
    const int64_t num_threads = std::min<int64_t>(1024, (ne00 + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
    const dim3 blocks_dim(num_threads, 1, 1);
    const dim3 blocks_num(num_blocks, 1, 1);

    argmax_f32<<<blocks_num, blocks_dim, 0, stream>>>(src0_d, dst_d, ne00);
}

// the later of two (value, index) candidates wins only with a larger value, or an equal one at a smaller index; an index
// below 0 marks no candidate
static __device__ __forceinline__ void argmax_part_combine(float & v, int & i, const float v_other, const int i_other) {
    if (i_other >= 0 && (i < 0 || v_other > v || (v_other == v && i_other < i))) {
        v = v_other;
        i = i_other;
    }
}

static __global__ void argmax_part_f32(const float * __restrict__ x, float * __restrict__ dst, const int64_t ncols,
        const int64_t stride_x, const int64_t n_dst, const int origin) {
    const int64_t row  = blockIdx.x;
    const float * rowx = x + row*stride_x;
    float       * rowd = dst + row*n_dst;

    for (int64_t k = 2 + threadIdx.x; k < n_dst; k += blockDim.x) {
        rowd[k] = (k % 2) == 0 ? -INFINITY : -1.0f;
    }

    // the greedy sampler picks token 0 when the row's first logit is a NaN
    if (origin == 0 && ncols > 0 && isnan(rowx[0])) {
        if (threadIdx.x == 0) {
            rowd[0] = NAN;
            rowd[1] = 0.0f;
        }
        return;
    }

    // each thread keeps the first of its values that is not a NaN and then only a strictly larger one, so its first max
    float best_v = -INFINITY;
    int   best_i = -1;
    for (int64_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        const float val = rowx[col];
        if (best_i < 0 ? !isnan(val) : val > best_v) {
            best_v = val;
            best_i = (int) col;
        }
    }

#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const float v_other = __shfl_xor_sync(0xFFFFFFFF, best_v, offset, WARP_SIZE);
        const int   i_other = __shfl_xor_sync(0xFFFFFFFF, best_i, offset, WARP_SIZE);
        argmax_part_combine(best_v, best_i, v_other, i_other);
    }

    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (n_warps > 1) {
        constexpr int    max_warps = 1024 / WARP_SIZE;
        __shared__ float shared_v[max_warps];
        __shared__ int   shared_i[max_warps];
        if (lane_id == 0) {
            shared_v[warp_id] = best_v;
            shared_i[warp_id] = best_i;
        }

        __syncthreads();

        if (warp_id == 0) {
            best_v = lane_id < n_warps ? shared_v[lane_id] : -INFINITY;
            best_i = lane_id < n_warps ? shared_i[lane_id] : -1;
#pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                const float v_other = __shfl_xor_sync(0xFFFFFFFF, best_v, offset, WARP_SIZE);
                const int   i_other = __shfl_xor_sync(0xFFFFFFFF, best_i, offset, WARP_SIZE);
                argmax_part_combine(best_v, best_i, v_other, i_other);
            }
        }
    }

    if (threadIdx.x == 0) {
        rowd[0] = best_v;
        rowd[1] = best_i < 0 ? -1.0f : (float) (best_i + origin);
    }
}

void ggml_cuda_argmax_part(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = src0->ne[1];

    const int64_t num_threads = std::min<int64_t>(1024, std::max<int64_t>(WARP_SIZE, (ne00 + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE));
    const dim3 blocks_dim(num_threads, 1, 1);
    const dim3 blocks_num(nrows, 1, 1);

    argmax_part_f32<<<blocks_num, blocks_dim, 0, ctx.stream()>>>((const float *) src0->data, (float *) dst->data, ne00,
        src0->nb[1] / sizeof(float), dst->ne[0], ggml_get_op_params_i32(dst, 0));
}

static __global__ void argmax_merge_f32(const float * __restrict__ x, int32_t * __restrict__ dst, const int64_t n_pairs,
        const int64_t stride_x, const int64_t nrows) {
    const int64_t row = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= nrows) {
        return;
    }
    const float * rowx = x + row*stride_x;

    float best_v = -INFINITY;
    int   best_i = -1;
    for (int64_t k = 0; k < n_pairs; k++) {
        const float v = rowx[2*k + 0];
        const int   i = (int) rowx[2*k + 1];
        if (i < 0) {
            continue;
        }
        if (isnan(v)) {
            best_i = i;
            break;
        }
        if (best_i < 0 || v > best_v) {
            best_v = v;
            best_i = i;
        }
    }
    dst[row] = best_i < 0 ? 0 : best_i;
}

void ggml_cuda_argmax_merge(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t nrows = src0->ne[1];
    const int     block = 128;

    argmax_merge_f32<<<(nrows + block - 1) / block, block, 0, ctx.stream()>>>((const float *) src0->data, (int32_t *) dst->data,
        src0->ne[0] / 2, src0->nb[1] / sizeof(float), nrows);
}
