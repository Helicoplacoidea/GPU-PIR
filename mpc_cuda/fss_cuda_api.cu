#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "../mpc_keys/uint128_type.h"
#include "pir_context.h"
#include "fss_cuda_launch.h"

namespace
{
    [[noreturn]] void fail_cuda_call(cudaError_t error, const char *expr, const char *file, int line)
    {
        std::fprintf(stderr, "CUDA error at %s:%d for %s: %s\n", file, line, expr, cudaGetErrorString(error));
        std::abort();
    }

    void check_last_kernel(const char *expr, const char *file, int line)
    {
        const cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            fail_cuda_call(error, expr, file, line);
        }
    }

#define CUDA_CHECK(expr)                                                   \
    do                                                                     \
    {                                                                      \
        const cudaError_t cuda_check_error__ = (expr);                     \
        if (cuda_check_error__ != cudaSuccess)                             \
        {                                                                  \
            fail_cuda_call(cuda_check_error__, #expr, __FILE__, __LINE__); \
        }                                                                  \
    } while (0)

#define CUDA_KERNEL_CHECK(...)                               \
    do                                                       \
    {                                                        \
        __VA_ARGS__;                                         \
        check_last_kernel(#__VA_ARGS__, __FILE__, __LINE__); \
    } while (0)

    size_t key_record_size(int maxlayer)
    {
        return static_cast<size_t>(1 + 16 + 1 + 18 * maxlayer + 16);
    }

    size_t key_buffer_size(int batch_size, int maxlayer)
    {
        return static_cast<size_t>(batch_size) * key_record_size(maxlayer);
    }

    void print_allocated_cuda_memory(size_t total_cuda_malloc)
    {
        std::cout << "CUDA memory initialized successfully!" << std::endl;
        std::cout << "Total allocated CUDA memory (excluding database): "
                  << total_cuda_malloc / (1024.0 * 1024.0) << " MB" << std::endl;
    }

    void free_device_ptr(void *&ptr)
    {
        if (ptr != nullptr)
        {
            CUDA_CHECK(cudaFree(ptr));
            ptr = nullptr;
        }
    }

    template <typename T>
    T *device_ptr(void *ptr)
    {
        return static_cast<T *>(ptr);
    }

    constexpr int kPipelineReductionBlocks = 256;
    constexpr int kGraphFixedN = 512;
    constexpr int kGraphFixedDepth = 20;
    constexpr int kGraphSplitDepth = 9;

    struct PirPipelineGraphN20N512State
    {
        bool initialized = false;
        uint8_t *key_pinned = nullptr;
        uint8_t *d_key = nullptr;
        uint128_t *res_pinned = nullptr;
        cudaStream_t stream = nullptr;
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t graph_exec = nullptr;
    };

    PirPipelineGraphN20N512State &graph_state()
    {
        static PirPipelineGraphN20N512State state;
        return state;
    }
} // namespace

extern "C" void cudaDPFkeygen(uint8_t *k0, uint8_t *k1, uint64_t *alpha, int n, int maxlayer, int batch_size)
{
    uint8_t *k0_device;
    CUDA_CHECK(cudaMalloc(&k0_device, batch_size * key_record_size(maxlayer)));

    uint8_t *k1_device;
    CUDA_CHECK(cudaMalloc(&k1_device, batch_size * key_record_size(maxlayer)));

    uint64_t *alpha_device;
    CUDA_CHECK(cudaMalloc(&alpha_device, batch_size * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(alpha_device, alpha, batch_size * sizeof(uint64_t), cudaMemcpyHostToDevice));

    uint32_t *aes_key;
    CUDA_CHECK(cudaMalloc(&aes_key, 4 * (14 + 1) * sizeof(uint32_t)));
    CUDA_KERNEL_CHECK(fss_genaeskey_kernel<<<1, 1>>>(aes_key));

    const int threads = batch_size > 256 ? 256 : batch_size;
    const int blocks = (batch_size + threads - 1) / threads;
    CUDA_KERNEL_CHECK(dpf_gen_kernel<<<blocks, threads>>>(aes_key, alpha_device, n, k0_device, k1_device, batch_size, maxlayer));

    CUDA_CHECK(cudaMemcpy(k0, k0_device, batch_size * key_record_size(maxlayer), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(k1, k1_device, batch_size * key_record_size(maxlayer), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(k0_device));
    CUDA_CHECK(cudaFree(k1_device));
    CUDA_CHECK(cudaFree(alpha_device));
    CUDA_CHECK(cudaFree(aes_key));
}

extern "C++" PirContext init_pir(int N, int n, uint128_t *db)
{
    PirContext ctx{};
    const size_t maxlayer = n - 7;
    const size_t db_size = (size_t(1) << n) * entry_size * sizeof(uint128_t);

    size_t total_cuda_malloc = 0;

    uint128_t *d_db;
    CUDA_CHECK(cudaMalloc(&d_db, db_size));
    CUDA_CHECK(cudaMemcpy(d_db, db, db_size, cudaMemcpyHostToDevice));
    ctx.db = d_db;

    uint128_t *d_output;
    const size_t output_size = (size_t)N * entry_size * sizeof(uint128_t);
    CUDA_CHECK(cudaMalloc(&d_output, output_size));
    total_cuda_malloc += output_size;
    ctx.output = d_output;

    uint8_t *d_key;
    const size_t key_size = key_buffer_size(N, static_cast<int>(maxlayer));
    CUDA_CHECK(cudaMalloc(&d_key, key_size));
    total_cuda_malloc += key_size;
    ctx.key = d_key;

    uint128_t *d_s;
    uint128_t *d_s_intermediate;
    uint128_t *d_s_res;
    uint128_t *d_se;
    uint32_t *d_t;
    uint32_t *d_t_intermediate;
    uint32_t *d_t_res;

    const size_t size_d_s = N * sizeof(uint128_t);
    const size_t size_d_t = N * sizeof(uint32_t);
    const size_t size_d_s_intermediate = N * (size_t(1) << 10) * sizeof(uint128_t);
    const size_t size_d_t_intermediate = N * (size_t(1) << 10) * sizeof(uint32_t);
    const size_t size_d_s_res = N * (size_t(1) << maxlayer) * sizeof(uint128_t);
    const size_t size_d_t_res = N * (size_t(1) << maxlayer) * sizeof(uint32_t);
    const size_t size_d_se = N * (size_t(1) << maxlayer) * sizeof(uint128_t);

    CUDA_CHECK(cudaMalloc(&d_s, size_d_s));
    CUDA_CHECK(cudaMalloc(&d_t, size_d_t));
    CUDA_CHECK(cudaMalloc(&d_s_intermediate, size_d_s_intermediate));
    CUDA_CHECK(cudaMalloc(&d_t_intermediate, size_d_t_intermediate));
    CUDA_CHECK(cudaMalloc(&d_s_res, size_d_s_res));
    CUDA_CHECK(cudaMalloc(&d_t_res, size_d_t_res));
    CUDA_CHECK(cudaMalloc(&d_se, size_d_se));

    total_cuda_malloc += size_d_s + size_d_t + size_d_s_intermediate + size_d_t_intermediate + size_d_s_res + size_d_t_res + size_d_se;

    ctx.seed = d_s;
    ctx.bit = d_t;
    ctx.seed_intermediate = d_s_intermediate;
    ctx.bit_intermediate = d_t_intermediate;
    ctx.seed_result = d_s_res;
    ctx.bit_result = d_t_res;
    ctx.selection = d_se;

    uint32_t *aes_key;
    const size_t aes_key_size = 4 * (14 + 1) * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(&aes_key, aes_key_size));
    total_cuda_malloc += aes_key_size;
    CUDA_KERNEL_CHECK(fss_genaeskey_kernel<<<1, 1>>>(aes_key));
    ctx.aes_key = aes_key;

    print_allocated_cuda_memory(total_cuda_malloc);
    return ctx;
}

extern "C++" PirBatchContext init_pir_batch(int N, int n, uint128_t *db)
{
    PirBatchContext ctx{};
    const size_t maxlayer = n - 7;
    const size_t db_size = (size_t(1) << n) * entry_size * sizeof(uint128_t);

    size_t total_cuda_malloc = 0;

    uint128_t *d_db;
    CUDA_CHECK(cudaMalloc(&d_db, db_size));
    CUDA_CHECK(cudaMemcpy(d_db, db, db_size, cudaMemcpyHostToDevice));
    ctx.db = d_db;

    uint8_t *d_key;
    const size_t key_size = key_buffer_size(N, static_cast<int>(maxlayer));
    CUDA_CHECK(cudaMalloc(&d_key, key_size));
    total_cuda_malloc += key_size;
    ctx.key = d_key;

    uint128_t *d_s;
    uint128_t *d_s_intermediate;
    uint128_t *d_s_res;
    uint128_t *d_se;
    uint32_t *d_t;
    uint32_t *d_t_intermediate;
    uint32_t *d_t_res;

    const size_t size_d_s = N * sizeof(uint128_t);
    const size_t size_d_t = N * sizeof(uint32_t);
    const size_t size_d_s_intermediate = N * (size_t(1) << 10) * sizeof(uint128_t);
    const size_t size_d_t_intermediate = N * (size_t(1) << 10) * sizeof(uint32_t);
    const size_t size_d_s_res = N * (size_t(1) << maxlayer) * sizeof(uint128_t);
    const size_t size_d_t_res = N * (size_t(1) << maxlayer) * sizeof(uint32_t);
    const size_t size_d_se = N * (size_t(1) << maxlayer) * sizeof(uint128_t);

    CUDA_CHECK(cudaMalloc(&d_s, size_d_s));
    CUDA_CHECK(cudaMalloc(&d_t, size_d_t));
    CUDA_CHECK(cudaMalloc(&d_s_intermediate, size_d_s_intermediate));
    CUDA_CHECK(cudaMalloc(&d_t_intermediate, size_d_t_intermediate));
    CUDA_CHECK(cudaMalloc(&d_s_res, size_d_s_res));
    CUDA_CHECK(cudaMalloc(&d_t_res, size_d_t_res));
    CUDA_CHECK(cudaMalloc(&d_se, size_d_se));

    total_cuda_malloc += size_d_s + size_d_t + size_d_s_intermediate + size_d_t_intermediate + size_d_s_res + size_d_t_res + size_d_se;

    ctx.seed = d_s;
    ctx.bit = d_t;
    ctx.seed_intermediate = d_s_intermediate;
    ctx.bit_intermediate = d_t_intermediate;
    ctx.seed_result = d_s_res;
    ctx.bit_result = d_t_res;
    ctx.selection = d_se;

    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp device_prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));
    const int blocks = device_prop.multiProcessorCount > 0 ? device_prop.multiProcessorCount : 1;
    ctx.blocks = blocks;

    uint128_t *d_partials;
    uint128_t *d_output;
    const size_t partial_size = (size_t)blocks * N * entry_size * sizeof(uint128_t);
    const size_t output_size = (size_t)N * entry_size * sizeof(uint128_t);
    CUDA_CHECK(cudaMalloc(&d_partials, partial_size));
    CUDA_CHECK(cudaMemset(d_partials, 0, partial_size));
    CUDA_CHECK(cudaMalloc(&d_output, output_size));
    ctx.partials = d_partials;
    ctx.output = d_output;
    total_cuda_malloc += partial_size + output_size;

    const size_t batch_smem_async = 2 * 128 * entry_size * sizeof(uint128_t);
    if (batch_smem_async > device_prop.sharedMemPerBlockOptin)
    {
        std::fprintf(stderr,
                     "BatchMulRedAsync needs %zu bytes of shared memory but device limit is %zu\n",
                     batch_smem_async, device_prop.sharedMemPerBlockOptin);
        std::abort();
    }
    CUDA_CHECK(cudaFuncSetAttribute(BatchMulRedAsync,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(batch_smem_async)));

    uint32_t *aes_key;
    const size_t aes_key_size = 4 * (14 + 1) * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(&aes_key, aes_key_size));
    total_cuda_malloc += aes_key_size;
    CUDA_KERNEL_CHECK(fss_genaeskey_kernel<<<1, 1>>>(aes_key));
    ctx.aes_key = aes_key;

    print_allocated_cuda_memory(total_cuda_malloc);
    return ctx;
}

extern "C++" PirPipelineContext init_pir_pipeline(int N, int n, uint128_t *db)
{
    PirPipelineContext ctx{};
    const size_t maxlayer = n - 7;
    const size_t db_size = (size_t(1) << n) * entry_size * sizeof(uint128_t);

    size_t total_cuda_malloc = 0;

    uint128_t *d_db;
    CUDA_CHECK(cudaMalloc(&d_db, db_size));
    CUDA_CHECK(cudaMemcpy(d_db, db, db_size, cudaMemcpyHostToDevice));
    ctx.db = d_db;

    uint128_t *d_partials;
    uint128_t *d_output;
    const size_t size_partials = entry_size * kPipelineReductionBlocks * sizeof(uint128_t);
    const size_t size_output = entry_size * sizeof(uint128_t);

    CUDA_CHECK(cudaMalloc(&d_partials, size_partials));
    CUDA_CHECK(cudaMalloc(&d_output, size_output));
    total_cuda_malloc += size_partials + size_output;
    ctx.partials = d_partials;
    ctx.output = d_output;

    uint128_t *d_s;
    uint128_t *d_s_intermediate;
    uint128_t *d_s_res;
    uint128_t *d_se;
    uint32_t *d_t;
    uint32_t *d_t_intermediate;
    uint32_t *d_t_res;

    const size_t size_d_s = sizeof(uint128_t);
    const size_t size_d_t = sizeof(uint32_t);
    const size_t size_d_s_intermediate = (size_t(1) << 10) * sizeof(uint128_t);
    const size_t size_d_t_intermediate = (size_t(1) << 10) * sizeof(uint32_t);
    const size_t size_d_s_res = (size_t(1) << maxlayer) * sizeof(uint128_t);
    const size_t size_d_t_res = (size_t(1) << maxlayer) * sizeof(uint32_t);
    const size_t size_d_se = (size_t(1) << maxlayer) * sizeof(uint128_t);

    CUDA_CHECK(cudaMalloc(&d_s, size_d_s));
    CUDA_CHECK(cudaMalloc(&d_t, size_d_t));
    CUDA_CHECK(cudaMalloc(&d_s_intermediate, size_d_s_intermediate));
    CUDA_CHECK(cudaMalloc(&d_t_intermediate, size_d_t_intermediate));
    CUDA_CHECK(cudaMalloc(&d_s_res, size_d_s_res));
    CUDA_CHECK(cudaMalloc(&d_t_res, size_d_t_res));
    CUDA_CHECK(cudaMalloc(&d_se, size_d_se));

    total_cuda_malloc += size_d_s + size_d_t + size_d_s_intermediate + size_d_t_intermediate + size_d_s_res + size_d_t_res + size_d_se;

    ctx.seed = d_s;
    ctx.bit = d_t;
    ctx.seed_intermediate = d_s_intermediate;
    ctx.bit_intermediate = d_t_intermediate;
    ctx.seed_result = d_s_res;
    ctx.bit_result = d_t_res;
    ctx.selection = d_se;

    uint32_t *aes_key;
    const size_t aes_key_size = 4 * (14 + 1) * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(&aes_key, aes_key_size));
    total_cuda_malloc += aes_key_size;
    CUDA_KERNEL_CHECK(fss_genaeskey_kernel<<<1, 1>>>(aes_key));
    ctx.aes_key = aes_key;

    print_allocated_cuda_memory(total_cuda_malloc);
    return ctx;
}

extern "C++" PirLutContext init_pir_LUT(int N, int n, uint32_t *db)
{
    PirLutContext ctx{};
    const size_t maxlayer = n - 7;
    const int chunk_size = 1 << (n - 8);

    uint32_t *d_db;
    const uint64_t db_size = (uint64_t(1) << n) * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(&d_db, db_size));
    CUDA_CHECK(cudaMemcpy(d_db, db, db_size, cudaMemcpyHostToDevice));
    ctx.db = d_db;

    uint32_t *d_input;
    uint32_t *d_intermediate;
    uint32_t *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, N * chunk_size * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_intermediate, N * chunk_size / 256 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(uint32_t)));
    ctx.input = d_input;
    ctx.intermediate = d_intermediate;
    ctx.output = d_output;

    uint128_t *d_s;
    uint128_t *d_s_intermediate;
    uint128_t *d_s_res;
    uint128_t *d_se;
    uint32_t *d_t;
    uint32_t *d_t_intermediate;
    uint32_t *d_t_res;

    CUDA_CHECK(cudaMalloc(&d_s, N * sizeof(uint128_t)));
    CUDA_CHECK(cudaMalloc(&d_t, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_s_intermediate, N * (size_t(1) << 10) * sizeof(uint128_t)));
    CUDA_CHECK(cudaMalloc(&d_t_intermediate, N * (size_t(1) << 10) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_s_res, N * 2 * (size_t(1) << maxlayer) * sizeof(uint128_t)));
    CUDA_CHECK(cudaMalloc(&d_t_res, N * 2 * (size_t(1) << maxlayer) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_se, N * (size_t(1) << maxlayer) * sizeof(uint128_t)));

    ctx.seed = d_s;
    ctx.bit = d_t;
    ctx.seed_intermediate = d_s_intermediate;
    ctx.bit_intermediate = d_t_intermediate;
    ctx.seed_result = d_s_res;
    ctx.bit_result = d_t_res;
    ctx.selection = d_se;

    uint32_t *aes_key;
    CUDA_CHECK(cudaMalloc(&aes_key, 4 * (14 + 1) * sizeof(uint32_t)));
    CUDA_KERNEL_CHECK(fss_genaeskey_kernel<<<1, 1>>>(aes_key));
    ctx.aes_key = aes_key;

    std::cout << "CUDA memory initialized successfully!" << std::endl;
    return ctx;
}

void free_cuda_memory(PirContext &ctx)
{
    free_device_ptr(ctx.db);
    free_device_ptr(ctx.output);
    free_device_ptr(ctx.key);
    free_device_ptr(ctx.seed);
    free_device_ptr(ctx.bit);
    free_device_ptr(ctx.seed_intermediate);
    free_device_ptr(ctx.bit_intermediate);
    free_device_ptr(ctx.seed_result);
    free_device_ptr(ctx.bit_result);
    free_device_ptr(ctx.selection);
    free_device_ptr(ctx.aes_key);
    std::cout << "CUDA memory freed successfully!" << std::endl;
}

void free_cuda_memory(PirBatchContext &ctx)
{
    free_device_ptr(ctx.db);
    free_device_ptr(ctx.partials);
    free_device_ptr(ctx.output);
    free_device_ptr(ctx.key);
    free_device_ptr(ctx.seed);
    free_device_ptr(ctx.bit);
    free_device_ptr(ctx.seed_intermediate);
    free_device_ptr(ctx.bit_intermediate);
    free_device_ptr(ctx.seed_result);
    free_device_ptr(ctx.bit_result);
    free_device_ptr(ctx.selection);
    free_device_ptr(ctx.aes_key);
    std::cout << "CUDA memory freed successfully!" << std::endl;
}

void free_cuda_memory(PirPipelineContext &ctx)
{
    free_device_ptr(ctx.db);
    free_device_ptr(ctx.partials);
    free_device_ptr(ctx.output);
    free_device_ptr(ctx.seed);
    free_device_ptr(ctx.bit);
    free_device_ptr(ctx.seed_intermediate);
    free_device_ptr(ctx.bit_intermediate);
    free_device_ptr(ctx.seed_result);
    free_device_ptr(ctx.bit_result);
    free_device_ptr(ctx.selection);
    free_device_ptr(ctx.aes_key);
    std::cout << "CUDA memory freed successfully!" << std::endl;
}

void free_cuda_memory(PirLutContext &ctx)
{
    free_device_ptr(ctx.db);
    free_device_ptr(ctx.input);
    free_device_ptr(ctx.intermediate);
    free_device_ptr(ctx.output);
    free_device_ptr(ctx.seed);
    free_device_ptr(ctx.bit);
    free_device_ptr(ctx.seed_intermediate);
    free_device_ptr(ctx.bit_intermediate);
    free_device_ptr(ctx.seed_result);
    free_device_ptr(ctx.bit_result);
    free_device_ptr(ctx.selection);
    free_device_ptr(ctx.aes_key);
    std::cout << "CUDA memory freed successfully!" << std::endl;
}

extern "C++" uint128_t *test_dpf_pir(uint8_t *key, const PirContext &ctx, int N)
{
    const int n = key[0];
    const int maxlayer = n - 7;
    const size_t key_size = key_buffer_size(N, maxlayer);

    uint128_t *d_db = device_ptr<uint128_t>(ctx.db);
    uint128_t *d_output = device_ptr<uint128_t>(ctx.output);
    uint8_t *d_key = device_ptr<uint8_t>(ctx.key);
    uint128_t *d_s = device_ptr<uint128_t>(ctx.seed);
    uint32_t *d_t = device_ptr<uint32_t>(ctx.bit);
    uint128_t *d_s_intermediate = device_ptr<uint128_t>(ctx.seed_intermediate);
    uint32_t *d_t_intermediate = device_ptr<uint32_t>(ctx.bit_intermediate);
    uint128_t *d_s_res = device_ptr<uint128_t>(ctx.seed_result);
    uint32_t *d_t_res = device_ptr<uint32_t>(ctx.bit_result);
    uint128_t *d_se = device_ptr<uint128_t>(ctx.selection);
    uint32_t *aes_key = device_ptr<uint32_t>(ctx.aes_key);

    uint128_t *res = static_cast<uint128_t *>(malloc(N * sizeof(uint128_t) * entry_size));
    std::memset(res, 0, N * entry_size * sizeof(uint128_t));

    CUDA_CHECK(cudaMemcpy(d_key, key, key_size, cudaMemcpyHostToDevice));
    CUDA_KERNEL_CHECK(seed_copy<<<(N + 255) / 256, 256>>>(d_s, d_t, d_key, maxlayer, N));
    CUDA_CHECK(cudaDeviceSynchronize());

    int threads_per_block;
    int blocks_per_grid;
    const int r = 9;

    if (maxlayer <= r)
    {
        CUDA_KERNEL_CHECK(EvalAll_BlockEvaluation<<<N, (1 << maxlayer) / 2>>>(aes_key, maxlayer, 0, d_key, d_s, d_t, d_s_res, d_t_res));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    else
    {
        CUDA_KERNEL_CHECK(EvalAll_BlockEvaluation<<<N, (1 << (maxlayer - r)) / 2>>>(aes_key, maxlayer - r, 0, d_key, d_s, d_t, d_s_intermediate, d_t_intermediate));
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_KERNEL_CHECK(EvalAll_BlockEvaluation<<<N * (1 << (maxlayer - r)), (1 << r) / 2>>>(aes_key, r, maxlayer - r, d_key, d_s_intermediate, d_t_intermediate, d_s_res, d_t_res));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    threads_per_block = std::min(1 << maxlayer, 256);
    blocks_per_grid = (1 << maxlayer) / threads_per_block;
    CUDA_KERNEL_CHECK(EvalAll_SeGeneration<<<N * blocks_per_grid, threads_per_block>>>(d_key, d_s_res, d_t_res, d_se));
    CUDA_CHECK(cudaDeviceSynchronize());

    threads_per_block = 256;
    CUDA_KERNEL_CHECK(MultiplicationReduction_grid_row<<<N, threads_per_block / 2>>>(d_key, d_se, d_db, d_output, 1, N));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(res, d_output, N * sizeof(uint128_t) * entry_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    return res;
}

extern "C++" uint128_t *test_dpf_pir_batch(uint8_t *key, const PirBatchContext &ctx, int N, int mode)
{
    const int n = key[0];
    const int maxlayer = n - 7;
    const size_t key_size = key_buffer_size(N, maxlayer);

    if (N <= 0 || N > 512)
    {
        std::fprintf(stderr, "test_dpf_pir_batch supports 1 <= batch_size <= 512\n");
        std::abort();
    }
    if (mode != 0 && mode != 1)
    {
        std::fprintf(stderr, "test_dpf_pir_batch mode must be 0 (sync) or 1 (async)\n");
        std::abort();
    }
    if (ctx.partials == nullptr || ctx.output == nullptr || ctx.blocks <= 0)
    {
        std::fprintf(stderr, "PirBatchContext was not initialized for batch-tiled PIR\n");
        std::abort();
    }

    uint128_t *d_db = device_ptr<uint128_t>(ctx.db);
    uint8_t *d_key = device_ptr<uint8_t>(ctx.key);
    uint128_t *d_s = device_ptr<uint128_t>(ctx.seed);
    uint32_t *d_t = device_ptr<uint32_t>(ctx.bit);
    uint128_t *d_s_intermediate = device_ptr<uint128_t>(ctx.seed_intermediate);
    uint32_t *d_t_intermediate = device_ptr<uint32_t>(ctx.bit_intermediate);
    uint128_t *d_s_res = device_ptr<uint128_t>(ctx.seed_result);
    uint32_t *d_t_res = device_ptr<uint32_t>(ctx.bit_result);
    uint128_t *d_se = device_ptr<uint128_t>(ctx.selection);
    uint32_t *aes_key = device_ptr<uint32_t>(ctx.aes_key);
    uint128_t *d_partials = device_ptr<uint128_t>(ctx.partials);
    uint128_t *d_output = device_ptr<uint128_t>(ctx.output);

    uint128_t *res = static_cast<uint128_t *>(malloc(N * sizeof(uint128_t) * entry_size));
    std::memset(res, 0, N * entry_size * sizeof(uint128_t));

    CUDA_CHECK(cudaMemcpy(d_key, key, key_size, cudaMemcpyHostToDevice));
    CUDA_KERNEL_CHECK(seed_copy<<<(N + 255) / 256, 256>>>(d_s, d_t, d_key, maxlayer, N));
    CUDA_CHECK(cudaDeviceSynchronize());

    const int r = 9;
    if (maxlayer <= r)
    {
        CUDA_KERNEL_CHECK(EvalAll_BlockEvaluation<<<N, (1 << maxlayer) / 2>>>(aes_key, maxlayer, 0, d_key, d_s, d_t, d_s_res, d_t_res));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    else
    {
        CUDA_KERNEL_CHECK(EvalAll_BlockEvaluation<<<N, (1 << (maxlayer - r)) / 2>>>(aes_key, maxlayer - r, 0, d_key, d_s, d_t, d_s_intermediate, d_t_intermediate));
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_KERNEL_CHECK(EvalAll_BlockEvaluation<<<N * (1 << (maxlayer - r)), (1 << r) / 2>>>(aes_key, r, maxlayer - r, d_key, d_s_intermediate, d_t_intermediate, d_s_res, d_t_res));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    const int se_threads = std::min(1 << maxlayer, 256);
    const int se_blocks_per_query = (1 << maxlayer) / se_threads;
    CUDA_KERNEL_CHECK(EvalAll_SeGeneration<<<N * se_blocks_per_query, se_threads>>>(d_key, d_s_res, d_t_res, d_se));
    CUDA_CHECK(cudaDeviceSynchronize());

    const int P = ctx.blocks;
    const uint64_t entries = 1ULL << n;
    const uint64_t per_block = ((entries + P - 1) / P + 127) / 128 * 128;
    const size_t smem_sync = 128 * entry_size * sizeof(uint128_t);
    const size_t smem_async = 2 * smem_sync;

    if (mode == 0)
    {
        CUDA_KERNEL_CHECK(BatchMulRedSync<<<P, N, smem_sync>>>(d_se, d_db, d_partials, n, per_block, N));
    }
    else
    {
        CUDA_KERNEL_CHECK(BatchMulRedAsync<<<P, N, smem_async>>>(d_se, d_db, d_partials, n, per_block, N));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_KERNEL_CHECK(BatchReducePartials<<<N, entry_size>>>(d_partials, d_output, P, N));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(res, d_output, N * sizeof(uint128_t) * entry_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    return res;
}

extern "C++" PirStreamContext init_streams_and_events()
{
    PirStreamContext handles{};

    cudaStream_t *streams = new cudaStream_t[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i)
    {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }
    handles.streams = streams;

    cudaEvent_t(*events)[NUM_STREAMS] = new cudaEvent_t[NUM_CHUNKS][NUM_STREAMS];
    for (int chunk = 0; chunk < NUM_CHUNKS; ++chunk)
    {
        for (int stream = 0; stream < NUM_STREAMS; ++stream)
        {
            CUDA_CHECK(cudaEventCreate(&events[chunk][stream]));
        }
    }
    handles.events = events;

    std::cout << "CUDA streams and events initialized!" << std::endl;
    return handles;
}

extern "C++" void cleanup_streams_and_events(PirStreamContext &handles)
{
    cudaStream_t *streams = static_cast<cudaStream_t *>(handles.streams);
    if (streams != nullptr)
    {
        for (int i = 0; i < NUM_STREAMS; ++i)
        {
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
        }
        delete[] streams;
        handles.streams = nullptr;
    }

    cudaEvent_t(*events)[NUM_STREAMS] = static_cast<cudaEvent_t(*)[NUM_STREAMS]>(handles.events);
    if (events != nullptr)
    {
        for (int chunk = 0; chunk < NUM_CHUNKS; ++chunk)
        {
            for (int stream = 0; stream < NUM_STREAMS; ++stream)
            {
                CUDA_CHECK(cudaEventDestroy(events[chunk][stream]));
            }
        }
        delete[] events;
        handles.events = nullptr;
    }

    std::cout << "CUDA streams and events cleaned up!" << std::endl;
}

extern "C++" uint128_t *test_dpf_pir_pipeline(uint8_t *key, const PirPipelineContext &ctx, const PirStreamContext &handles, int N)
{
    const int n = key[0];
    const int maxlayer = n - 7;
    const size_t key_size = key_buffer_size(N, maxlayer);

    uint128_t *d_db = device_ptr<uint128_t>(ctx.db);
    uint128_t *d_partials = device_ptr<uint128_t>(ctx.partials);
    uint128_t *d_output = device_ptr<uint128_t>(ctx.output);
    uint128_t *d_s = device_ptr<uint128_t>(ctx.seed);
    uint32_t *d_t = device_ptr<uint32_t>(ctx.bit);
    uint128_t *d_s_intermediate = device_ptr<uint128_t>(ctx.seed_intermediate);
    uint32_t *d_t_intermediate = device_ptr<uint32_t>(ctx.bit_intermediate);
    uint128_t *d_s_res = device_ptr<uint128_t>(ctx.seed_result);
    uint32_t *d_t_res = device_ptr<uint32_t>(ctx.bit_result);
    uint128_t *d_se = device_ptr<uint128_t>(ctx.selection);
    uint32_t *aes_key = device_ptr<uint32_t>(ctx.aes_key);

    uint8_t *key_pinned;
    CUDA_CHECK(cudaMallocHost(&key_pinned, key_size));
    std::memcpy(key_pinned, key, key_size);

    uint8_t *d_key;
    CUDA_CHECK(cudaMalloc(&d_key, key_size));

    uint128_t *res;
    CUDA_CHECK(cudaMallocHost(&res, entry_size * N * sizeof(uint128_t)));

    cudaStream_t *streams = static_cast<cudaStream_t *>(handles.streams);
    cudaEvent_t(*events)[NUM_STREAMS] = static_cast<cudaEvent_t(*)[NUM_STREAMS]>(handles.events);

    int threads_per_block;
    int blocks_per_grid;

#ifdef USE_L2_CACHE
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    const size_t size = std::min(static_cast<size_t>(prop.l2CacheSize * 0.75),
                                 static_cast<size_t>(prop.persistingL2CacheMaxSize));
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size));

    const size_t window_size = std::min((uint64_t)prop.accessPolicyMaxWindowSize, (((uint64_t)1 << n) * entry_size * 16));

    cudaStreamAttrValue stream_attribute;
    stream_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void *>(d_db);
    stream_attribute.accessPolicyWindow.num_bytes = window_size;
    stream_attribute.accessPolicyWindow.hitRatio = window_size / ((double)((uint64_t)1 << n) * entry_size * 16);
    stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

    CUDA_CHECK(cudaStreamSetAttribute(streams[4], cudaStreamAttributeAccessPolicyWindow, &stream_attribute));
#endif

    for (int chunk = 0; chunk < N; ++chunk)
    {
        unsigned char *d_key_offset = d_key + chunk * key_record_size(maxlayer);
        const uint8_t *host_key_offset = key_pinned + chunk * key_record_size(maxlayer);

        if (maxlayer <= 9)
        {
            if (chunk > 0)
            {
                CUDA_CHECK(cudaStreamWaitEvent(streams[1], events[chunk - 1][3], 0));
            }
            CUDA_CHECK(cudaMemcpyAsync(d_s, host_key_offset + 1, 16, cudaMemcpyHostToDevice, streams[1]));
            CUDA_CHECK(cudaMemcpyAsync(d_t, host_key_offset + 17, 1, cudaMemcpyHostToDevice, streams[1]));
            CUDA_CHECK(cudaMemcpyAsync(d_key_offset, host_key_offset, key_record_size(maxlayer), cudaMemcpyHostToDevice, streams[1]));

            CUDA_KERNEL_CHECK(EvalAll_BlockEvaluation<<<1, (1 << maxlayer) / 2, 0, streams[1]>>>(aes_key, maxlayer, 0, d_key_offset, d_s, d_t, d_s_res, d_t_res));
            CUDA_CHECK(cudaEventRecord(events[chunk][1], streams[1]));
        }
        else
        {
            if (chunk > 0)
            {
                CUDA_CHECK(cudaStreamWaitEvent(streams[1], events[chunk - 1][2], 0));
            }
            CUDA_CHECK(cudaMemcpyAsync(d_s, host_key_offset + 1, 16, cudaMemcpyHostToDevice, streams[1]));
            CUDA_CHECK(cudaMemcpyAsync(d_t, host_key_offset + 17, 1, cudaMemcpyHostToDevice, streams[1]));
            CUDA_CHECK(cudaMemcpyAsync(d_key_offset, host_key_offset, key_record_size(maxlayer), cudaMemcpyHostToDevice, streams[1]));

            CUDA_KERNEL_CHECK(EvalAll_BlockEvaluation<<<1, (1 << (maxlayer - 9)) / 2, 0, streams[1]>>>(aes_key, maxlayer - 9, 0, d_key_offset, d_s, d_t, d_s_intermediate, d_t_intermediate));
            CUDA_CHECK(cudaEventRecord(events[chunk][1], streams[1]));

            if (chunk > 0)
            {
                CUDA_CHECK(cudaStreamWaitEvent(streams[2], events[chunk - 1][3], 0));
            }
            CUDA_CHECK(cudaStreamWaitEvent(streams[2], events[chunk][1], 0));
            CUDA_KERNEL_CHECK(EvalAll_BlockEvaluation<<<1 << (maxlayer - 9), (1 << 9) / 2, 0, streams[2]>>>(aes_key, 9, maxlayer - 9, d_key_offset, d_s_intermediate, d_t_intermediate, d_s_res, d_t_res));
            CUDA_CHECK(cudaEventRecord(events[chunk][2], streams[2]));
        }

        threads_per_block = std::min(1 << maxlayer, 256);
        blocks_per_grid = (1 << maxlayer) / threads_per_block;

        if (chunk > 0)
        {
            CUDA_CHECK(cudaStreamWaitEvent(streams[3], events[chunk - 1][4], 0));
        }
        if (maxlayer <= 9)
        {
            CUDA_CHECK(cudaStreamWaitEvent(streams[3], events[chunk][1], 0));
        }
        else
        {
            CUDA_CHECK(cudaStreamWaitEvent(streams[3], events[chunk][2], 0));
        }
        CUDA_KERNEL_CHECK(EvalAll_SeGeneration<<<blocks_per_grid, threads_per_block, 0, streams[3]>>>(d_key_offset, d_s_res, d_t_res, d_se));
        CUDA_CHECK(cudaEventRecord(events[chunk][3], streams[3]));

        threads_per_block = 256;
        blocks_per_grid = (1 << n) / threads_per_block;
        if (chunk > 0)
        {
            CUDA_CHECK(cudaStreamWaitEvent(streams[4], events[chunk - 1][5], 0));
        }
        CUDA_CHECK(cudaStreamWaitEvent(streams[4], events[chunk][3], 0));

        CUDA_KERNEL_CHECK(MultiplicationReduction_grid_row<<<kPipelineReductionBlocks, threads_per_block / 2, 0, streams[4]>>>(d_key_offset, d_se, d_db, d_partials, 1, 1));
        CUDA_CHECK(cudaEventRecord(events[chunk][4], streams[4]));

        CUDA_CHECK(cudaStreamWaitEvent(streams[5], events[chunk][4], 0));
        CUDA_KERNEL_CHECK(BlockReduction<<<1, kPipelineReductionBlocks, 0, streams[5]>>>(d_partials, d_output));
        CUDA_CHECK(cudaEventRecord(events[chunk][5], streams[5]));
        CUDA_CHECK(cudaMemcpyAsync(res + entry_size * chunk, d_output, entry_size * sizeof(uint128_t), cudaMemcpyDeviceToHost, streams[5]));
    }
#ifdef L2_CACHE
    CUDA_CHECK(cudaCtxResetPersistingL2Cache());
#endif
    for (int i = 0; i < NUM_STREAMS; ++i)
    {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

    CUDA_CHECK(cudaFree(d_key));
    CUDA_CHECK(cudaFreeHost(key_pinned));
    return res;
}

extern "C++" uint128_t *test_dpf_pir_pipeline_graph_n20_N512(uint8_t *key, const PirPipelineContext &ctx, const PirStreamContext &handles)
{
    const int n = key[0];
    if (n != kGraphFixedDepth)
    {
        std::fprintf(stderr, "test_dpf_pir_pipeline_graph_n20_N512 expects n=20, got n=%d\n", n);
        std::abort();
    }
    const int maxlayer = n - 7;
    if (maxlayer <= kGraphSplitDepth)
    {
        std::fprintf(stderr, "test_dpf_pir_pipeline_graph_n20_N512 expects maxlayer>9, got maxlayer=%d\n", maxlayer);
        std::abort();
    }

    PirPipelineGraphN20N512State &state = graph_state();
    const size_t record_size = key_record_size(maxlayer);
    const size_t key_size = key_buffer_size(kGraphFixedN, maxlayer);
    cudaStream_t *streams = static_cast<cudaStream_t *>(handles.streams);
    cudaEvent_t(*events)[NUM_STREAMS] = static_cast<cudaEvent_t(*)[NUM_STREAMS]>(handles.events);
    if (streams == nullptr || events == nullptr)
    {
        std::fprintf(stderr, "test_dpf_pir_pipeline_graph_n20_N512 expects initialized streams/events handles\n");
        std::abort();
    }

    if (!state.initialized)
    {
        uint128_t *d_db = device_ptr<uint128_t>(ctx.db);
        uint128_t *d_partials = device_ptr<uint128_t>(ctx.partials);
        uint128_t *d_output = device_ptr<uint128_t>(ctx.output);
        uint128_t *d_s = device_ptr<uint128_t>(ctx.seed);
        uint32_t *d_t = device_ptr<uint32_t>(ctx.bit);
        uint128_t *d_s_intermediate = device_ptr<uint128_t>(ctx.seed_intermediate);
        uint32_t *d_t_intermediate = device_ptr<uint32_t>(ctx.bit_intermediate);
        uint128_t *d_s_res = device_ptr<uint128_t>(ctx.seed_result);
        uint32_t *d_t_res = device_ptr<uint32_t>(ctx.bit_result);
        uint128_t *d_se = device_ptr<uint128_t>(ctx.selection);
        uint32_t *aes_key = device_ptr<uint32_t>(ctx.aes_key);

        CUDA_CHECK(cudaMallocHost(&state.key_pinned, key_size));
        CUDA_CHECK(cudaMalloc(&state.d_key, key_size));
        CUDA_CHECK(cudaMallocHost(&state.res_pinned, entry_size * kGraphFixedN * sizeof(uint128_t)));
        state.stream = streams[1];

        CUDA_CHECK(cudaStreamBeginCapture(state.stream, cudaStreamCaptureModeGlobal));
        for (int chunk = 0; chunk < kGraphFixedN; ++chunk)
        {
            unsigned char *d_key_offset = state.d_key + chunk * record_size;
            const uint8_t *host_key_offset = state.key_pinned + chunk * record_size;

            if (chunk > 0)
            {
                CUDA_CHECK(cudaStreamWaitEvent(streams[1], events[chunk - 1][2], 0));
            }
            CUDA_CHECK(cudaMemcpyAsync(d_s, host_key_offset + 1, 16, cudaMemcpyHostToDevice, streams[1]));
            CUDA_CHECK(cudaMemcpyAsync(d_t, host_key_offset + 17, 1, cudaMemcpyHostToDevice, streams[1]));
            CUDA_CHECK(cudaMemcpyAsync(d_key_offset, host_key_offset, record_size, cudaMemcpyHostToDevice, streams[1]));

            CUDA_KERNEL_CHECK(EvalAll_BlockEvaluation<<<1, (1 << (maxlayer - kGraphSplitDepth)) / 2, 0, streams[1]>>>(
                aes_key, maxlayer - kGraphSplitDepth, 0, d_key_offset, d_s, d_t, d_s_intermediate, d_t_intermediate));
            CUDA_CHECK(cudaEventRecord(events[chunk][1], streams[1]));

            if (chunk > 0)
            {
                CUDA_CHECK(cudaStreamWaitEvent(streams[2], events[chunk - 1][3], 0));
            }
            CUDA_CHECK(cudaStreamWaitEvent(streams[2], events[chunk][1], 0));
            CUDA_KERNEL_CHECK(EvalAll_BlockEvaluation<<<1 << (maxlayer - kGraphSplitDepth), (1 << kGraphSplitDepth) / 2, 0, streams[2]>>>(
                aes_key, kGraphSplitDepth, maxlayer - kGraphSplitDepth, d_key_offset, d_s_intermediate, d_t_intermediate, d_s_res, d_t_res));
            CUDA_CHECK(cudaEventRecord(events[chunk][2], streams[2]));

            if (chunk > 0)
            {
                CUDA_CHECK(cudaStreamWaitEvent(streams[3], events[chunk - 1][4], 0));
            }
            CUDA_CHECK(cudaStreamWaitEvent(streams[3], events[chunk][2], 0));
            CUDA_KERNEL_CHECK(EvalAll_SeGeneration<<<(1 << maxlayer) / 256, 256, 0, streams[3]>>>(d_key_offset, d_s_res, d_t_res, d_se));
            CUDA_CHECK(cudaEventRecord(events[chunk][3], streams[3]));

            if (chunk > 0)
            {
                CUDA_CHECK(cudaStreamWaitEvent(streams[4], events[chunk - 1][5], 0));
            }
            CUDA_CHECK(cudaStreamWaitEvent(streams[4], events[chunk][3], 0));
            CUDA_KERNEL_CHECK(MultiplicationReduction_grid_row<<<256, 128, 0, streams[4]>>>(d_key_offset, d_se, d_db, d_partials, 1, 1));
            CUDA_CHECK(cudaEventRecord(events[chunk][4], streams[4]));

            CUDA_CHECK(cudaStreamWaitEvent(streams[5], events[chunk][4], 0));
            CUDA_KERNEL_CHECK(BlockReduction<<<1, 256, 0, streams[5]>>>(d_partials, d_output));
            CUDA_CHECK(cudaMemcpyAsync(state.res_pinned + entry_size * chunk, d_output, entry_size * sizeof(uint128_t), cudaMemcpyDeviceToHost, streams[5]));
            CUDA_CHECK(cudaEventRecord(events[chunk][5], streams[5]));
        }
        CUDA_CHECK(cudaStreamWaitEvent(state.stream, events[kGraphFixedN - 1][5], 0));
        CUDA_CHECK(cudaStreamEndCapture(state.stream, &state.graph));
        CUDA_CHECK(cudaGraphInstantiate(&state.graph_exec, state.graph, nullptr, nullptr, 0));
        state.initialized = true;
    }
    else if (state.stream != streams[1])
    {
        std::fprintf(stderr, "test_dpf_pir_pipeline_graph_n20_N512 requires stable stream handles across launches\n");
        std::abort();
    }

    std::memcpy(state.key_pinned, key, key_size);
    CUDA_CHECK(cudaGraphLaunch(state.graph_exec, state.stream));
    CUDA_CHECK(cudaStreamSynchronize(state.stream));
    return state.res_pinned;
}

extern "C++" uint32_t *test_dpf_pir_LUT(uint8_t *key, const PirLutContext &ctx, int N)
{
    const int n = key[0];
    const int maxlayer = n - 7;
    const size_t key_size = key_buffer_size(N, maxlayer);

    uint32_t *d_db = device_ptr<uint32_t>(ctx.db);
    uint32_t *d_input = device_ptr<uint32_t>(ctx.input);
    uint32_t *d_intermediate = device_ptr<uint32_t>(ctx.intermediate);
    uint32_t *d_output = device_ptr<uint32_t>(ctx.output);
    uint128_t *d_s = device_ptr<uint128_t>(ctx.seed);
    uint32_t *d_t = device_ptr<uint32_t>(ctx.bit);
    uint128_t *d_s_intermediate = device_ptr<uint128_t>(ctx.seed_intermediate);
    uint32_t *d_t_intermediate = device_ptr<uint32_t>(ctx.bit_intermediate);
    uint128_t *d_s_res = device_ptr<uint128_t>(ctx.seed_result);
    uint32_t *d_t_res = device_ptr<uint32_t>(ctx.bit_result);
    uint128_t *d_se = device_ptr<uint128_t>(ctx.selection);
    uint32_t *aes_key = device_ptr<uint32_t>(ctx.aes_key);

    uint8_t *key_pinned;
    CUDA_CHECK(cudaMallocHost(&key_pinned, key_size));
    std::memcpy(key_pinned, key, key_size);

    uint8_t *d_key;
    CUDA_CHECK(cudaMalloc(&d_key, key_size));
    CUDA_CHECK(cudaMemcpy(d_key, key_pinned, key_size, cudaMemcpyHostToDevice));

    uint32_t *res;
    CUDA_CHECK(cudaMallocHost(&res, N * sizeof(uint32_t)));

    CUDA_KERNEL_CHECK(seed_copy<<<(N + 255) / 256, 256>>>(d_s, d_t, d_key, maxlayer, N));
    CUDA_CHECK(cudaDeviceSynchronize());

    int threads_per_block;
    int blocks_per_grid;

    if (maxlayer <= 9)
    {
        CUDA_KERNEL_CHECK(EvalAll_BlockEvaluation<<<N, (1 << maxlayer) / 2>>>(aes_key, maxlayer, 0, d_key, d_s, d_t, d_s_res, d_t_res));
        CUDA_CHECK(cudaDeviceSynchronize());

        threads_per_block = std::min(1 << maxlayer, 256);
        blocks_per_grid = (1 << maxlayer) / threads_per_block;
        CUDA_KERNEL_CHECK(EvalAll_SeGeneration<<<N * blocks_per_grid, threads_per_block>>>(d_key, d_s_res, d_t_res, d_se));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    else if (maxlayer <= 18)
    {
        CUDA_KERNEL_CHECK(EvalAll_BlockEvaluation<<<N, (1 << (maxlayer - 9)) / 2>>>(aes_key, maxlayer - 9, 0, d_key, d_s, d_t, d_s_intermediate, d_t_intermediate));
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_KERNEL_CHECK(EvalAll_BlockEvaluation<<<N * (1 << (maxlayer - 9)), (1 << 9) / 2>>>(aes_key, 9, maxlayer - 9, d_key, d_s_intermediate, d_t_intermediate, d_s_res, d_t_res));
        CUDA_CHECK(cudaDeviceSynchronize());

        threads_per_block = std::min(1 << maxlayer, 256);
        blocks_per_grid = (1 << maxlayer) / threads_per_block;
        CUDA_KERNEL_CHECK(EvalAll_SeGeneration<<<N * blocks_per_grid, threads_per_block>>>(d_key, d_s_res, d_t_res, d_se));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    else
    {
        CUDA_KERNEL_CHECK(EvalAll_BlockEvaluation<<<1, (1 << 9) / 2>>>(aes_key, 9, 0, d_key, d_s, d_t, d_s_intermediate, d_t_intermediate));
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_KERNEL_CHECK(EvalAll_BlockEvaluation<<<1 << 9, (1 << 9) / 2>>>(aes_key, 9, 9, d_key, d_s_intermediate, d_t_intermediate, d_s_res + ((1 << 18) - 1), d_t_res + ((1 << 18) - 1)));
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int layer = 19; layer <= maxlayer; ++layer)
        {
            const int threads_per_layer = 1 << (layer - 1);
            threads_per_block = std::min(threads_per_layer, 256);
            blocks_per_grid = (threads_per_layer + threads_per_block - 1) / threads_per_block;
            CUDA_KERNEL_CHECK(EvalAll_LBLEvaluation<<<blocks_per_grid, threads_per_block>>>(aes_key, layer, maxlayer, 1, d_key, d_s_res, d_t_res));
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        threads_per_block = std::min(1 << maxlayer, 256);
        blocks_per_grid = (1 << maxlayer) / threads_per_block;
        CUDA_KERNEL_CHECK(EVAL_Pack_last_layer_gense<<<blocks_per_grid, threads_per_block>>>(d_key, d_s_res, d_t_res, d_se));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    threads_per_block = 256;
    blocks_per_grid = (1 << n) / threads_per_block;
    blocks_per_grid = std::min(blocks_per_grid, 768);

    CUDA_KERNEL_CHECK(MultiplicationReduction_LUT_new<<<N * blocks_per_grid, threads_per_block>>>(d_key, d_se, d_db, d_input, 1, N));
    CUDA_CHECK(cudaDeviceSynchronize());

    if (n == 8)
    {
        CUDA_CHECK(cudaMemcpy(res, d_input, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    }

    if (blocks_per_grid > 256)
    {
        blocks_per_grid = (blocks_per_grid + 255) / 256;
        CUDA_KERNEL_CHECK(BlockReduction_LUT<<<N * blocks_per_grid, 256>>>(d_input, d_intermediate));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_KERNEL_CHECK(BlockReduction_LUT<<<N, blocks_per_grid>>>(d_intermediate, d_output));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    else
    {
        CUDA_KERNEL_CHECK(BlockReduction_LUT<<<N, blocks_per_grid>>>(d_input, d_output));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaMemcpy(res, d_output, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_key));
    CUDA_CHECK(cudaFreeHost(key_pinned));
    return res;
}
