#include "../mpc_keys/uint128_type.h"
#include "../mpc_cuda/pir_config.h"
#include "../mpc_cuda/fss_cuda_launch.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void cudaDPFkeygen(uint8_t *k0, uint8_t *k1, uint64_t *alpha,
                              int n, int maxlayer, int batch_size);

namespace
{
void check_cuda(cudaError_t error, const char *call, int line)
{
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(call) + " at line " +
                                 std::to_string(line) + ": " + cudaGetErrorString(error));
}

#define CUDA_CHECK(call) check_cuda((call), #call, __LINE__)

template <typename T>
struct DeviceBuffer
{
    T *data = nullptr;
    size_t count = 0;

    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t n) { allocate(n); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    ~DeviceBuffer()
    {
        if (data != nullptr)
            cudaFree(data);
    }

    void allocate(size_t n)
    {
        count = n;
        CUDA_CHECK(cudaMalloc(&data, n * sizeof(T)));
    }
};

size_t key_bytes(int n) { return 1 + 16 + 1 + 18 * static_cast<size_t>(n - 7) + 16; }

struct Variant
{
    std::string name;
    std::function<void(const uint128_t *)> launch;
    std::vector<double> times_ms;
};

double percentile(std::vector<double> values, double fraction)
{
    std::sort(values.begin(), values.end());
    const double position = (values.size() - 1) * fraction;
    const size_t low = static_cast<size_t>(position);
    const size_t high = std::min(low + 1, values.size() - 1);
    return values[low] + (values[high] - values[low]) * (position - low);
}

struct EvalWorkspace
{
    int maxlayer;
    size_t key_size;
    DeviceBuffer<uint8_t> key;
    DeviceBuffer<uint128_t> seed, seed_intermediate, selection;
    DeviceBuffer<uint32_t> bit, bit_intermediate;

    explicit EvalWorkspace(int n)
        : maxlayer(n - 7), key_size(key_bytes(n)),
          key(key_size), seed(1), seed_intermediate(size_t{1} << 10),
          selection(size_t{1} << maxlayer), bit(1), bit_intermediate(size_t{1} << 10)
    {
    }

    void evaluate(const std::vector<uint8_t> &host_key, const uint32_t *aes_key)
    {
        CUDA_CHECK(cudaMemcpy(key.data, host_key.data(), key_size, cudaMemcpyHostToDevice));
        seed_copy<<<1, 256>>>(seed.data, bit.data, key.data, maxlayer, 1);
        if (maxlayer <= 9)
        {
            EvalAll_BlockEvaluation_Se<<<1, (1 << maxlayer) / 2>>>(
                const_cast<uint32_t *>(aes_key), maxlayer, 0, key.data,
                seed.data, bit.data, selection.data);
        }
        else
        {
            EvalAll_BlockEvaluation<<<1, (1 << (maxlayer - 9)) / 2>>>(
                const_cast<uint32_t *>(aes_key), maxlayer - 9, 0, key.data,
                seed.data, bit.data, seed_intermediate.data, bit_intermediate.data);
            EvalAll_BlockEvaluation_Se<<<1 << (maxlayer - 9), 256>>>(
                const_cast<uint32_t *>(aes_key), 9, maxlayer - 9, key.data,
                seed_intermediate.data, bit_intermediate.data, selection.data);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
};

double selection_density(const uint128_t *device_selection, size_t words)
{
    std::vector<uint128_t> host(words);
    CUDA_CHECK(cudaMemcpy(host.data(), device_selection, words * sizeof(uint128_t),
                          cudaMemcpyDeviceToHost));
    size_t set = 0;
    for (const uint128_t &word : host)
        set += __builtin_popcountll(word.get_high()) + __builtin_popcountll(word.get_low());
    return static_cast<double>(set) / static_cast<double>(words * 128);
}

void run_case(int n, int samples)
{
    const int maxlayer = n - 7;
    const uint64_t rows = uint64_t{1} << n;
    const size_t words = size_t{1} << maxlayer;
    const size_t components = rows * entry_size;

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int sms = prop.multiProcessorCount;

    std::printf("DEVICE %s SM=%d n=%d rows=%llu words=%zu samples=%d\n",
                prop.name, sms, n, (unsigned long long)rows, words, samples);

    std::mt19937_64 rng(0x5EC7026ULL + n);
    std::vector<uint128_t> host_db(components);
    std::vector<uint128_t> host_db_col(components);
    for (uint64_t row = 0; row < rows; ++row)
    {
        for (int j = 0; j < entry_size; ++j)
        {
            const uint128_t value(rng(), rng());
            host_db[row * entry_size + j] = value;
            host_db_col[(uint64_t)j * rows + row] = value;
        }
    }

    DeviceBuffer<uint128_t> db(components);
    DeviceBuffer<uint128_t> db_col(components);
    CUDA_CHECK(cudaMemcpy(db.data, host_db.data(), components * sizeof(uint128_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db_col.data, host_db_col.data(), components * sizeof(uint128_t),
                          cudaMemcpyHostToDevice));
    std::vector<uint128_t>().swap(host_db_col);

    const int partitions = std::max(1, std::min<int>(sms, rows / k_batch_tile_entries));
    DeviceBuffer<uint128_t> partials(size_t{512} * entry_size);
    DeviceBuffer<uint128_t> output(entry_size);
    DeviceBuffer<uint32_t> aes_key(4 * (14 + 1));
    DeviceBuffer<uint8_t> meta(1);
    const uint8_t meta_value = static_cast<uint8_t>(n);
    CUDA_CHECK(cudaMemcpy(meta.data, &meta_value, 1, cudaMemcpyHostToDevice));
    fss_genaeskey_kernel<<<1, 1>>>(aes_key.data);
    CUDA_CHECK(cudaDeviceSynchronize());

    EvalWorkspace party[2] = {EvalWorkspace(n), EvalWorkspace(n)};
    std::vector<uint8_t> host_keys[2] = {
        std::vector<uint8_t>(key_bytes(n)), std::vector<uint8_t>(key_bytes(n))};

    const size_t sync_smem = k_batch_tile_entries * entry_size * sizeof(uint128_t);
    const size_t async_smem = 2 * sync_smem;
    CUDA_CHECK(cudaFuncSetAttribute(batch_mul_red_async_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(async_smem)));

    auto make_variants = [&]() {
        std::vector<Variant> variants;
        variants.push_back({"native_1sm", [&](const uint128_t *se) {
            MultiplicationReduction_grid_row<<<1, 128>>>(
                meta.data, const_cast<uint128_t *>(se), db.data, output.data, 1, 1);
        }});

        const int sweep[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
        for (int p : sweep)
        {
            const uint64_t per_block = (rows + p - 1) / p;
            variants.push_back({"multi_row_p" + std::to_string(p),
                                [&, p, per_block](const uint128_t *se) {
                sector_mul_red_row_kernel<<<p, 256>>>(
                    se, db.data, partials.data, rows, per_block);
                sector_reduce_partials_kernel<<<1, 256>>>(partials.data, output.data, p);
            }});
            variants.push_back({"multi_sector_p" + std::to_string(p),
                                [&, p, per_block](const uint128_t *se) {
                sector_mul_red_kernel<<<p, 256>>>(
                    se, db.data, partials.data, rows, per_block);
                sector_reduce_partials_kernel<<<1, 256>>>(partials.data, output.data, p);
            }});
            variants.push_back({"multi_sector4_p" + std::to_string(p),
                                [&, p, per_block](const uint128_t *se) {
                sector_mul_red4_kernel<<<p, 256>>>(
                    se, db.data, partials.data, rows, per_block);
                sector_reduce_partials_kernel<<<1, 256>>>(partials.data, output.data, p);
            }});
            variants.push_back({"multi_col_p" + std::to_string(p),
                                [&, p, per_block](const uint128_t *se) {
                sector_col_mul_red_kernel<<<p, 256>>>(
                    se, db_col.data, partials.data, rows, per_block);
                sector_reduce_partials_kernel<<<1, 256>>>(partials.data, output.data, p);
            }});
        }

        const uint64_t batch_per_block =
            ((rows + partitions - 1) / partitions + k_batch_tile_entries - 1) /
            k_batch_tile_entries * k_batch_tile_entries;
        variants.push_back({"shared_sync_b1", [&, batch_per_block](const uint128_t *se) {
            batch_mul_red_sync_kernel<<<partitions, 1, sync_smem>>>(
                se, db.data, partials.data, n, batch_per_block, 1);
            batch_reduce_partials_kernel<<<1, entry_size>>>(
                partials.data, output.data, partitions, 1);
        }});
        variants.push_back({"shared_async_b1", [&, batch_per_block](const uint128_t *se) {
            batch_mul_red_async_kernel<<<partitions, 1, async_smem>>>(
                se, db.data, partials.data, n, batch_per_block, 1);
            batch_reduce_partials_kernel<<<1, entry_size>>>(
                partials.data, output.data, partitions, 1);
        }});
        return variants;
    };

    std::vector<Variant> variants = make_variants();
    uint64_t alphas[] = {0, rows - 1, rng() & (rows - 1)};

    for (uint64_t alpha : alphas)
    {
        cudaDPFkeygen(host_keys[0].data(), host_keys[1].data(), &alpha, n, maxlayer, 1);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int p = 0; p < 2; ++p)
            party[p].evaluate(host_keys[p], aes_key.data);

        uint128_t reference[2][entry_size];
        for (int p = 0; p < 2; ++p)
        {
            variants.front().launch(party[p].selection.data);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(reference[p], output.data, sizeof(reference[p]),
                                  cudaMemcpyDeviceToHost));
        }
        for (int j = 0; j < entry_size; ++j)
        {
            if ((reference[0][j] ^ reference[1][j]) != host_db[alpha * entry_size + j])
                throw std::runtime_error("native reconstruction failed");
        }

        for (const Variant &variant : variants)
        {
            for (int p = 0; p < 2; ++p)
            {
                uint128_t result[entry_size];
                variant.launch(party[p].selection.data);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(result, output.data, sizeof(result),
                                      cudaMemcpyDeviceToHost));
                for (int j = 0; j < entry_size; ++j)
                {
                    if (result[j] != reference[p][j])
                        throw std::runtime_error("share check failed for " + variant.name +
                                                 " alpha=" + std::to_string(alpha) +
                                                 " party=" + std::to_string(p) +
                                                 " component=" + std::to_string(j));
                }
            }
        }
    }

    const double density = selection_density(party[0].selection.data, words);
    std::printf("PARTY0_SELECTION_DENSITY %.6f\n", density);

    {
        for (int i = 0; i < 5; ++i)
            party[0].evaluate(host_keys[0], aes_key.data);
        std::vector<double> eval_ms;
        for (int i = 0; i < samples; ++i)
        {
            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));
            CUDA_CHECK(cudaEventRecord(start));
            party[0].evaluate(host_keys[0], aes_key.data);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float elapsed = 0;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
            eval_ms.push_back(elapsed);
        }
        std::printf("SELECTION_EVAL n=%d median_ms=%.6f p10_ms=%.6f p90_ms=%.6f\n",
                    n, percentile(eval_ms, 0.5), percentile(eval_ms, 0.1),
                    percentile(eval_ms, 0.9));
    }

    const uint128_t *timed_selection = party[0].selection.data;
    for (Variant &variant : variants)
    {
        for (int i = 0; i < 3; ++i)
            variant.launch(timed_selection);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto measure = [&](int reps) {
            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));
            CUDA_CHECK(cudaEventRecord(start));
            for (int i = 0; i < reps; ++i)
                variant.launch(timed_selection);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float elapsed = 0;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
            return elapsed / reps;
        };

        const double estimate = measure(1);
        const int reps = std::max(1, std::min(200, static_cast<int>(2.0 / estimate) + 1));
        variant.times_ms.reserve(samples);
        for (int s = 0; s < samples; ++s)
            variant.times_ms.push_back(measure(reps));
    }

    const Variant &native = variants.front();
    const double native_median = percentile(native.times_ms, 0.5);
    for (const Variant &variant : variants)
    {
        const double median = percentile(variant.times_ms, 0.5);
        std::printf("RESULT n=%d variant=%s median_ms=%.6f min_ms=%.6f "
                    "p10_ms=%.6f p90_ms=%.6f speedup_vs_native=%.3f density=%.4f\n",
                    n, variant.name.c_str(), median,
                    *std::min_element(variant.times_ms.begin(), variant.times_ms.end()),
                    percentile(variant.times_ms, 0.1), percentile(variant.times_ms, 0.9),
                    native_median / median, density);
        for (double value : variant.times_ms)
            std::printf("SAMPLE,%d,%s,%.6f\n", n, variant.name.c_str(), value);
    }
}
} // namespace

int main(int argc, char **argv)
{
    try
    {
        int n = 20;
        int samples = 31;
        if (argc > 1)
            n = std::atoi(argv[1]);
        if (argc > 2)
            samples = std::atoi(argv[2]);
        if (n < 8 || n > 24 || samples < 3)
        {
            std::fprintf(stderr, "usage: %s [n=8..24] [samples>=3]\n", argv[0]);
            return 2;
        }

        int devices = 0;
        const cudaError_t status = cudaGetDeviceCount(&devices);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
            (status == cudaSuccess && devices == 0))
        {
            std::printf("SKIP bench_sector_view: no available CUDA GPU (%s)\n",
                        cudaGetErrorString(status));
            return 77;
        }
        CUDA_CHECK(status);
        run_case(n, samples);
        CUDA_CHECK(cudaDeviceSynchronize());
        return 0;
    }
    catch (const std::exception &error)
    {
        std::fprintf(stderr, "FAIL bench_sector_view: %s\n", error.what());
        return 1;
    }
}
