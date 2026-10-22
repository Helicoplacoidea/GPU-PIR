#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

#include "../mpc_keys/uint128_type.h"

// One-pass batch-tiled inner product.
//
// Each block owns a contiguous DB range and processes it in tiles of
// kBatchTileEntries entries.  Every tile is loaded to shared memory once and
// consumed by all `batch` queries (one thread per query).  A 128-bit se word
// covers exactly one tile, so no per-entry se loads are needed.
//
// partials layout: [block][query][entry_component]

namespace
{
constexpr int kBatchTileEntries = 128;

__device__ __forceinline__ bool batch_word_bit(const uint128_t &w, int bit)
{
    return (bit >= 64) ? bool((w.get_high() >> (bit - 64)) & 1ULL)
                       : bool((w.get_low() >> bit) & 1ULL);
}

template <bool Async>
__device__ __forceinline__ void batch_load_tile(uint128_t *dst, const uint128_t *db,
                                                uint64_t tile, int tid, int nthr)
{
    for (int idx = tid; idx < kBatchTileEntries * entry_size; idx += nthr)
    {
        if constexpr (Async)
            __pipeline_memcpy_async(&dst[idx], &db[tile * entry_size + idx], sizeof(uint128_t));
        else
            dst[idx] = db[tile * entry_size + idx];
    }
}

__device__ __forceinline__ void batch_accumulate_tile(const uint128_t *tile_smem,
                                                      const uint128_t *se,
                                                      uint128_t *acc,
                                                      int query,
                                                      uint64_t blocks_per_query,
                                                      uint64_t tile)
{
    const uint128_t sw = se[(uint64_t)query * blocks_per_query + (tile >> 7)];

#pragma unroll 4
    for (int e = 0; e < kBatchTileEntries; ++e)
    {
        if (batch_word_bit(sw, 127 - e))
        {
#pragma unroll
            for (int j = 0; j < entry_size; ++j)
                acc[j] ^= tile_smem[e * entry_size + j];
        }
    }
}

template <bool Async>
__device__ __forceinline__ void batch_mul_red_body(uint128_t *smem,
                                                   const uint128_t *se,
                                                   const uint128_t *db,
                                                   uint128_t *partials,
                                                   int n, uint64_t per_block, int batch)
{
    const int query = threadIdx.x;
    if (query >= batch)
        return;

    const uint64_t entries = 1ULL << n;
    const uint64_t start = (uint64_t)blockIdx.x * per_block;
    if (start >= entries)
        return;
    const uint64_t end = (start + per_block < entries) ? (start + per_block) : entries;
    const uint64_t blocks_per_query = entries >> 7;

    uint128_t acc[entry_size];
#pragma unroll
    for (int j = 0; j < entry_size; ++j)
        acc[j] = uint128_t(0, 0);

    if constexpr (Async)
    {
        batch_load_tile<true>(smem, db, start, threadIdx.x, blockDim.x);
        __pipeline_commit();

        int buf = 0;
        for (uint64_t tile = start; tile < end; tile += kBatchTileEntries)
        {
            const uint64_t next = tile + kBatchTileEntries;
            const int next_buf = buf ^ 1;

            if (next < end)
            {
                batch_load_tile<true>(smem + (size_t)next_buf * kBatchTileEntries * entry_size,
                                      db, next, threadIdx.x, blockDim.x);
                __pipeline_commit();
                __pipeline_wait_prior(1);
            }
            else
            {
                __pipeline_wait_prior(0);
            }
            __syncthreads();

            batch_accumulate_tile(smem + (size_t)buf * kBatchTileEntries * entry_size,
                                  se, acc, query, blocks_per_query, tile);
            __syncthreads();
            buf = next_buf;
        }
    }
    else
    {
        for (uint64_t tile = start; tile < end; tile += kBatchTileEntries)
        {
            batch_load_tile<false>(smem, db, tile, threadIdx.x, blockDim.x);
            __syncthreads();
            batch_accumulate_tile(smem, se, acc, query, blocks_per_query, tile);
            __syncthreads();
        }
    }

    const uint64_t base = ((uint64_t)blockIdx.x * batch + query) * entry_size;
#pragma unroll
    for (int j = 0; j < entry_size; ++j)
        partials[base + j] = acc[j];
}
} // namespace

__global__ __launch_bounds__(512, 1)
void BatchMulRedSync(const uint128_t *se,
                     const uint128_t *db,
                     uint128_t *partials,
                     int n, uint64_t per_block, int batch)
{
    extern __shared__ uint128_t smem[];
    batch_mul_red_body<false>(smem, se, db, partials, n, per_block, batch);
}

__global__ __launch_bounds__(512, 1)
void BatchMulRedAsync(const uint128_t *se,
                      const uint128_t *db,
                      uint128_t *partials,
                      int n, uint64_t per_block, int batch)
{
    extern __shared__ uint128_t smem[];
    batch_mul_red_body<true>(smem, se, db, partials, n, per_block, batch);
}

__global__ void BatchReducePartials(const uint128_t *partials,
                                    uint128_t *out,
                                    int P, int batch)
{
    const int query = blockIdx.x;
    const int j = threadIdx.x;
    if (query >= batch || j >= entry_size)
        return;

    uint128_t v(0, 0);
    for (int p = 0; p < P; ++p)
        v ^= partials[((uint64_t)p * batch + query) * entry_size + j];
    out[query * entry_size + j] = v;
}
