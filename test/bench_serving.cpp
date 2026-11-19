#include "pir_test_utils.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace
{
double percentile(std::vector<double> values, double fraction)
{
    std::sort(values.begin(), values.end());
    const double position = (values.size() - 1) * fraction;
    const size_t low = static_cast<size_t>(position);
    const size_t high = std::min(low + 1, values.size() - 1);
    return values[low] + (values[high] - values[low]) * (position - low);
}

double elapsed_ms(std::chrono::steady_clock::time_point start,
                  std::chrono::steady_clock::time_point stop)
{
    return std::chrono::duration<double, std::milli>(stop - start).count();
}
} // namespace

int main(int argc, char **argv)
{
    int n = 20;
    int batch = 512;
    int low_samples = 31;
    int batch_samples = 5;
    if (argc > 1)
        n = std::atoi(argv[1]);
    if (argc > 2)
        batch = std::atoi(argv[2]);
    if (argc > 3)
        low_samples = std::atoi(argv[3]);
    if (argc > 4)
        batch_samples = std::atoi(argv[4]);

    if (n < 8 || n > 24 || batch < 1 || batch > 512 ||
        batch > (1 << n) || low_samples < 3 || batch_samples < 1)
    {
        std::fprintf(stderr, "usage: %s [n=8..24] [batch<=512 and <=2^n] "
                             "[low_samples>=3] [batch_samples>=1]\n", argv[0]);
        return 2;
    }
    if (!has_cuda_device())
    {
        std::printf("SKIP bench_serving: no available CUDA GPU\n");
        return 0;
    }

    cudaDeviceProp prop{};
    cudaError_t error = cudaGetDeviceProperties(&prop, 0);
    if (error != cudaSuccess)
    {
        std::fprintf(stderr, "cudaGetDeviceProperties: %s\n", cudaGetErrorString(error));
        return 1;
    }
    std::printf("DEVICE %s SM=%d n=%d batch=%d\n",
                prop.name, prop.multiProcessorCount, n, batch);

    std::srand(1);
    std::vector<uint128_t> db = make_random_pir_db(n);

    const uint64_t query = (uint64_t{1} << n) / 3;
    std::vector<uint64_t> single_query(1, query);
    std::vector<uint8_t> single_k0(dpf_key_bytes(n));
    std::vector<uint8_t> single_k1(dpf_key_bytes(n));
    cudaDPFkeygen(single_k0.data(), single_k1.data(), single_query.data(), n, n - 7, 1);

    PirContext low_latency = init_pir(1, n, db.data());
    uint128_t *verify0 = test_dpf_pir(single_k0.data(), low_latency, 1);
    uint128_t *verify1 = test_dpf_pir(single_k1.data(), low_latency, 1);
    std::string error_text;
    if (!verify_pir_response(verify0, verify1, db.data(), single_query.data(), 1, &error_text))
    {
        std::fprintf(stderr, "FAIL low-latency verification: %s\n", error_text.c_str());
        return 1;
    }
    free(verify0);
    free(verify1);

    for (int i = 0; i < 5; ++i)
        free(test_dpf_pir(single_k0.data(), low_latency, 1));

    std::vector<double> single_latency;
    single_latency.reserve(low_samples);
    for (int i = 0; i < low_samples; ++i)
    {
        const auto start = std::chrono::steady_clock::now();
        uint128_t *result = test_dpf_pir(single_k0.data(), low_latency, 1);
        const auto stop = std::chrono::steady_clock::now();
        free(result);
        single_latency.push_back(elapsed_ms(start, stop));
    }
    const double single_p50 = percentile(single_latency, 0.5);
    const double single_p95 = percentile(single_latency, 0.95);
    const double single_p99 = percentile(single_latency, 0.99);
    std::printf("LOW_LATENCY n=%d samples=%d p50_ms=%.4f p95_ms=%.4f p99_ms=%.4f "
                "min_ms=%.4f max_ms=%.4f\n",
                n, low_samples, single_p50, single_p95, single_p99,
                *std::min_element(single_latency.begin(), single_latency.end()),
                *std::max_element(single_latency.begin(), single_latency.end()));
    free_cuda_memory(low_latency);

    std::vector<uint64_t> queries = make_descending_queries(n, batch);
    std::vector<uint8_t> batch_k0(batch * dpf_key_bytes(n));
    std::vector<uint8_t> batch_k1(batch * dpf_key_bytes(n));
    cudaDPFkeygen(batch_k0.data(), batch_k1.data(), queries.data(), n, n - 7, batch);

    pir_batch_context high_throughput = init_pir_batch(batch, n, db.data());
    double async_p50 = 0;
    double async_p99 = 0;
    for (int mode = 0; mode < 2; ++mode)
    {
        uint128_t *r0 = test_dpf_pir_batch(batch_k0.data(), high_throughput, batch, mode);
        uint128_t *r1 = test_dpf_pir_batch(batch_k1.data(), high_throughput, batch, mode);
        if (r0 == nullptr || r1 == nullptr ||
            !verify_pir_response(r0, r1, db.data(), queries.data(), batch, &error_text))
        {
            std::fprintf(stderr, "FAIL batch verification (mode %d): %s\n",
                         mode, error_text.c_str());
            return 1;
        }
        free(r0);
        free(r1);

        for (int i = 0; i < 2; ++i)
            free(test_dpf_pir_batch(batch_k0.data(), high_throughput, batch, mode));

        std::vector<double> totals;
        totals.reserve(batch_samples);
        for (int i = 0; i < batch_samples; ++i)
        {
            const auto start = std::chrono::steady_clock::now();
            uint128_t *result = test_dpf_pir_batch(batch_k0.data(), high_throughput, batch, mode);
            const auto stop = std::chrono::steady_clock::now();
            free(result);
            totals.push_back(elapsed_ms(start, stop));
        }
        const double median = percentile(totals, 0.5);
        const double p99 = percentile(totals, 0.99);
        const double throughput = batch / (median / 1000.0);
        std::printf("THROUGHPUT n=%d B=%d mode=%s samples=%d median_ms=%.3f p99_ms=%.3f "
                    "pirs_per_s=%.1f per_request_us=%.2f\n",
                    n, batch, mode == 0 ? "sync" : "async", batch_samples,
                    median, p99, throughput, median * 1000.0 / batch);
        if (mode == 1)
        {
            async_p50 = median;
            async_p99 = p99;
        }
    }
    free_cuda_memory(high_throughput);

    const double single_throughput = 1000.0 / single_p50;
    const double async_throughput = batch / (async_p50 / 1000.0);
    std::printf("COMPARISON n=%d single_p50_ms=%.4f single_p99_ms=%.4f "
                "batch_async_p50_ms=%.3f batch_async_p99_ms=%.3f "
                "async_per_request_us=%.2f throughput_ratio=%.1f\n",
                n, single_p50, single_p99, async_p50, async_p99,
                async_p50 * 1000.0 / batch, async_throughput / single_throughput);
    return 0;
}
