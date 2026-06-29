// bench_lut_only.cpp — ONLY the DPF-PIR LUT benchmark.

#include "../mpc_cuda/mpc_core.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "pir_test_utils.h"

namespace
{
constexpr int kBenchmarkIterations = 10;

void print_benchmark_result(std::chrono::duration<double> elapsed, int batch_size)
{
    const double average_ms = elapsed.count() * 1000.0 / kBenchmarkIterations;
    const double throughput = (static_cast<double>(kBenchmarkIterations) / elapsed.count()) * batch_size;
    std::cout << "DPF-PIR LUT Time taken: " << average_ms << " ms\n";
    std::cout << "Throughput: " << throughput << " pirs/s\n";
}

bool benchmark_pir_lut(int n, int batch_size)
{
    if (!validate_packed_dpf_depth(n, "bench_pir_LUT") ||
        !validate_lut_batch(n, batch_size, "bench_pir_LUT"))
    {
        return false;
    }

    const int maxlayer = n - 7;
    std::vector<uint32_t> db = make_lut_db(n);
    std::vector<uint64_t> queries = make_constant_queries(batch_size, 4);
    std::vector<uint8_t> k0(batch_size * dpf_key_bytes(n));
    std::vector<uint8_t> k1(batch_size * dpf_key_bytes(n));

    cudaDPFkeygen(k0.data(), k1.data(), queries.data(), n, maxlayer, batch_size);
    PirLutContext ctx = init_pir_LUT(batch_size, n, db.data());

    uint32_t *verify0 = test_dpf_pir_LUT(k0.data(), ctx, batch_size);
    uint32_t *verify1 = test_dpf_pir_LUT(k1.data(), ctx, batch_size);

    std::string error;
    const bool verified = verify_lut_response(verify0, verify1, db.data(), queries.data(), batch_size, &error);
    cudaFreeHost(verify0);
    cudaFreeHost(verify1);
    if (!verified)
    {
        std::cerr << "[FAIL] bench_pir_LUT smoke check: " << error << '\n';
        free_cuda_memory(ctx);
        return false;
    }

    uint32_t *warmup = test_dpf_pir_LUT(k0.data(), ctx, batch_size);
    cudaFreeHost(warmup);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kBenchmarkIterations; ++i)
    {
        uint32_t *result = test_dpf_pir_LUT(k0.data(), ctx, batch_size);
        cudaFreeHost(result);
    }
    auto end = std::chrono::high_resolution_clock::now();

    free_cuda_memory(ctx);
    print_benchmark_result(end - start, batch_size);
    return true;
}
} // namespace

int main(int argc, char **argv)
{
    std::srand(1);

    int n = 20;
    int batch_size = 512;
    std::string error;
    if (!parse_cli_overrides(argc, argv, n, batch_size, &n, &batch_size, &error))
    {
        std::cerr << "[FAIL] " << error << '\n';
        return 1;
    }

    select_best_gpu();
    if (!has_cuda_device())
    {
        std::cout << "[SKIP] bench_lut_only requires a CUDA-capable device\n";
        return 0;
    }

    return benchmark_pir_lut(n, batch_size) ? 0 : 1;
}
