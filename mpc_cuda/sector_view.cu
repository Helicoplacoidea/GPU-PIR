#include <cstdint>
#include <cuda_runtime.h>

#include "pir_config.h"
#include "../mpc_keys/uint128_type.h"

// Multi-SM conditional inner product over a packed DPF selection vector.
// The database is partitioned into contiguous row ranges, one block per range.
// Only rows whose selection bit is set are read, and each block emits one
// partial XOR which a small second kernel reduces.

namespace
{
__device__ __forceinline__ uint128_t shfl_down_128(uint128_t value, int delta)
{
    return uint128_t(__shfl_down_sync(0xffffffffu, value.get_high(), delta),
                     __shfl_down_sync(0xffffffffu, value.get_low(), delta));
}

__device__ __forceinline__ void warp_xor_reduce(uint128_t (&acc)[entry_size])
{
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
    {
#pragma unroll
        for (int j = 0; j < entry_size; ++j)
            acc[j] ^= shfl_down_128(acc[j], delta);
    }
}

__device__ __forceinline__ void store_block_partial(uint128_t (&acc)[entry_size],
                                                    uint128_t *smem,
                                                    uint128_t *partials)
{
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    warp_xor_reduce(acc);

    if (lane == 0)
    {
#pragma unroll
        for (int j = 0; j < entry_size; ++j)
            smem[entry_size * warp + j] = acc[j];
    }
    __syncthreads();

    if (tid < entry_size)
    {
        const int warps = blockDim.x >> 5;
        uint128_t total(0, 0);
        for (int w = 0; w < warps; ++w)
            total ^= smem[entry_size * w + tid];
        partials[(uint64_t)blockIdx.x * entry_size + tid] = total;
    }
}

__device__ __forceinline__ bool selected(const uint128_t *se, uint64_t row)
{
    return se[row >> 7].get_bit(127 - static_cast<int>(row & 127));
}
} // namespace

__global__ __launch_bounds__(256, 1)
void sector_mul_red_row_kernel(const uint128_t *__restrict__ se,
                               const uint128_t *__restrict__ db,
                               uint128_t *partials,
                               uint64_t rows, uint64_t per_block)
{
    __shared__ uint128_t smem[8 * entry_size];

    const int tid = threadIdx.x;
    const uint64_t start = (uint64_t)blockIdx.x * per_block;
    const uint64_t end = (start + per_block < rows) ? start + per_block : rows;

    uint128_t acc[entry_size];
#pragma unroll
    for (int j = 0; j < entry_size; ++j)
        acc[j] = uint128_t(0, 0);

    if (start < rows)
    {
        for (uint64_t row = start + tid; row < end; row += blockDim.x)
        {
            if (selected(se, row))
            {
#pragma unroll
                for (int j = 0; j < entry_size; ++j)
                    acc[j] ^= db[row * entry_size + j];
            }
        }
    }
    store_block_partial(acc, smem, partials);
}

// Two consecutive rows per warp: lanes 0..15 cover the 16 uint128 components
// of one row, lanes 16..31 the components of the next row.  A selected row is
// read as a single coalesced 256-byte range; unselected rows issue no load.
__global__ __launch_bounds__(256, 1)
void sector_mul_red_kernel(const uint128_t *__restrict__ se,
                           const uint128_t *__restrict__ db,
                           uint128_t *partials,
                           uint64_t rows, uint64_t per_block)
{
    __shared__ uint128_t smem[8 * entry_size];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps = blockDim.x >> 5;
    const int row_in_pair = lane >> 4;
    const int component = lane & 15;

    const uint64_t start = (uint64_t)blockIdx.x * per_block;
    const uint64_t end = (start + per_block < rows) ? start + per_block : rows;

    uint128_t acc(0, 0);
    if (start < rows)
    {
        const uint64_t count = end - start;
        const uint64_t pairs = (count + 1) >> 1;
        for (uint64_t pair = warp; pair < pairs; pair += warps)
        {
            const uint64_t row = start + 2 * pair + row_in_pair;
            if (row < end && selected(se, row))
                acc ^= db[row * entry_size + component];
        }
    }

    acc ^= shfl_down_128(acc, 16);
    if (lane < entry_size)
        smem[entry_size * warp + lane] = acc;
    __syncthreads();

    if (tid < entry_size)
    {
        uint128_t total(0, 0);
        for (int w = 0; w < warps; ++w)
            total ^= smem[entry_size * w + tid];
        partials[(uint64_t)blockIdx.x * entry_size + tid] = total;
    }
}

// Four lanes per row, 64 contiguous bytes per lane (two 32-byte sectors):
// eight rows per warp and four independent accumulators per thread.
__global__ __launch_bounds__(256, 1)
void sector_mul_red4_kernel(const uint128_t *__restrict__ se,
                            const uint128_t *__restrict__ db,
                            uint128_t *partials,
                            uint64_t rows, uint64_t per_block)
{
    __shared__ uint128_t smem[8 * entry_size];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps = blockDim.x >> 5;
    const int sub = lane & 3;
    const int row_in_group = lane >> 2;

    const uint64_t start = (uint64_t)blockIdx.x * per_block;
    const uint64_t end = (start + per_block < rows) ? start + per_block : rows;

    uint128_t acc[4];
#pragma unroll
    for (int c = 0; c < 4; ++c)
        acc[c] = uint128_t(0, 0);

    if (start < rows)
    {
        for (uint64_t row = start + 8 * warp + row_in_group; row < end; row += 8 * warps)
        {
            if (selected(se, row))
            {
                const uint128_t *src = db + row * entry_size + sub * 4;
#pragma unroll
                for (int c = 0; c < 4; ++c)
                    acc[c] ^= src[c];
            }
        }
    }

    for (int delta = 16; delta >= 4; delta >>= 1)
    {
#pragma unroll
        for (int c = 0; c < 4; ++c)
            acc[c] ^= shfl_down_128(acc[c], delta);
    }

    if (lane < 4)
    {
#pragma unroll
        for (int c = 0; c < 4; ++c)
            smem[entry_size * warp + sub * 4 + c] = acc[c];
    }
    __syncthreads();

    if (tid < entry_size)
    {
        uint128_t total(0, 0);
        for (int w = 0; w < warps; ++w)
            total ^= smem[entry_size * w + tid];
        partials[(uint64_t)blockIdx.x * entry_size + tid] = total;
    }
}

// Column-major layout: component j of all rows is contiguous.  A warp reading
// component j for consecutive rows is fully coalesced, but a selected row's
// sixteen components are separated by `rows` elements.  At 50% row density a
// 32-byte sector usually holds one selected and one unselected row, so this
// layout transfers ~1.5x the selected payload instead of the row-major 1.0x.
__global__ __launch_bounds__(256, 1)
void sector_col_mul_red_kernel(const uint128_t *__restrict__ se,
                               const uint128_t *__restrict__ db,
                               uint128_t *partials,
                               uint64_t rows, uint64_t per_block)
{
    __shared__ uint128_t smem[8 * entry_size];

    const int tid = threadIdx.x;
    const uint64_t start = (uint64_t)blockIdx.x * per_block;
    const uint64_t end = (start + per_block < rows) ? start + per_block : rows;

    uint128_t acc[entry_size];
#pragma unroll
    for (int j = 0; j < entry_size; ++j)
        acc[j] = uint128_t(0, 0);

    if (start < rows)
    {
        for (uint64_t row = start + tid; row < end; row += blockDim.x)
        {
            if (selected(se, row))
            {
#pragma unroll
                for (int j = 0; j < entry_size; ++j)
                    acc[j] ^= db[(uint64_t)j * rows + row];
            }
        }
    }
    store_block_partial(acc, smem, partials);
}

__global__ void sector_reduce_partials_kernel(const uint128_t *__restrict__ partials,
                                              uint128_t *__restrict__ out,
                                              int partitions)
{
    __shared__ uint128_t smem[16][entry_size];

    const int tid = threadIdx.x;
    const int component = tid & 15;
    const int group = tid >> 4;
    const int groups = blockDim.x >> 4;

    uint128_t value(0, 0);
    for (int p = group; p < partitions; p += groups)
        value ^= partials[(uint64_t)p * entry_size + component];

    smem[group][component] = value;
    __syncthreads();

    if (tid < entry_size)
    {
        uint128_t total(0, 0);
        for (int g = 0; g < groups; ++g)
            total ^= smem[g][tid];
        out[tid] = total;
    }
}
