#include "../mpc_keys/uint128_type.h"
#include "../mpc_cuda/pir_config.h"
#include "../mpc_cuda/fss_cuda_launch.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <openssl/aes.h>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void cudaDPFkeygen(uint8_t *k0, uint8_t *k1, uint64_t *alpha,
                              int n, int maxlayer, int batch_size);

namespace
{
bool cleanup_failed = false;

void check_cuda(cudaError_t error, const char *call, int line)
{
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(call) + " at line " +
                                 std::to_string(line) + ": " + cudaGetErrorString(error));
}

#define CUDA_CHECK(call) check_cuda((call), #call, __LINE__)
#define CUDA_LAUNCH(...)                        \
    do                                          \
    {                                           \
        __VA_ARGS__;                            \
        CUDA_CHECK(cudaGetLastError());         \
        CUDA_CHECK(cudaDeviceSynchronize());    \
    } while (0)

template <typename T>
struct DeviceBuffer
{
    T *data = nullptr;
    const size_t bytes;

    explicit DeviceBuffer(size_t count) : bytes(count * sizeof(T))
    {
        CUDA_CHECK(cudaMalloc(&data, bytes));
    }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    ~DeviceBuffer()
    {
        const cudaError_t error = cudaFree(data);
        if (error != cudaSuccess)
        {
            std::fprintf(stderr, "cudaFree: %s\n", cudaGetErrorString(error));
            cleanup_failed = true;
        }
    }
};

std::vector<uint128_t> cpu_selection(const std::vector<uint8_t> &keys, int n, int batch_size)
{
    AES_KEY aes;
    uint128_t aes_seed(597349, 121379);
    if (AES_set_encrypt_key(aes_seed.get_bytes(), 128, &aes) != 0)
        throw std::runtime_error("CPU AES key setup failed");

    const int depth = n - 7;
    const size_t words_per_query = size_t{1} << depth;
    std::vector<uint128_t> selection(batch_size * words_per_query);
    for (int q = 0; q < batch_size; ++q)
    {
        const uint8_t *key = keys.data() + size_t(q) * (34 + 18 * depth);
        uint128_t root;
        std::memcpy(&root, key + 1, 16);
        std::vector<uint128_t> seeds(1, root);
        std::vector<unsigned> bits(1, key[17]);
        for (int layer = 0; layer < depth; ++layer)
        {
            const uint8_t *correction = key + 18 * (layer + 1);
            uint128_t correction_seed;
            std::memcpy(&correction_seed, correction, 16);
            std::vector<uint128_t> next_seeds(2 * seeds.size());
            std::vector<unsigned> next_bits(2 * bits.size());
            for (size_t i = 0; i < seeds.size(); ++i)
            {
                const uint128_t clean(seeds[i].get_high(), seeds[i].get_low() & ~uint64_t{1});
                for (unsigned side = 0; side < 2; ++side)
                {
                    uint128_t input(clean.get_high(), clean.get_low() | side);
                    uint128_t child;
                    AES_encrypt(input.get_bytes(), child.get_bytes(), &aes);
                    child ^= clean;
                    // PRG_cuda's mutated input flips only the left child's control bit.
                    unsigned bit = child.get_lsb() ^ (side == 0);
                    child.set_lsb_zero();
                    if (bits[i])
                    {
                        child ^= correction_seed;
                        bit ^= correction[16 + side];
                    }
                    next_seeds[2 * i + side] = child;
                    next_bits[2 * i + side] = bit;
                }
            }
            seeds.swap(next_seeds);
            bits.swap(next_bits);
        }
        uint128_t final_correction;
        std::memcpy(&final_correction, key + 18 * (depth + 1), 16);
        for (size_t i = 0; i < words_per_query; ++i)
            selection[size_t(q) * words_per_query + i] = bits[i] ? seeds[i] ^ final_correction : seeds[i];
    }
    return selection;
}

void run_case(int n, int batch_size)
{
    constexpr int r = 9;
    const int maxlayer = n - 7;
    const int prefix = std::max(0, maxlayer - r);
    const int final_depth = maxlayer - prefix;
    const size_t words_per_query = size_t{1} << maxlayer;
    const size_t words = batch_size * words_per_query;
    const size_t key_bytes = size_t(batch_size) * (34 + 18 * maxlayer);
    const int final_blocks = batch_size * (1 << prefix);
    const int final_threads = (1 << final_depth) / 2;
    const int se_threads = static_cast<int>(std::min(words_per_query, size_t{256}));
    const int se_blocks = static_cast<int>(words / se_threads);
    const uint64_t rows = uint64_t{1} << n;
    const uint64_t boundary = maxlayer > r ? uint64_t{1} << (r + 7) : rows / 2;
    const std::vector<uint64_t> edges = {
        0, 63, 64, 127, 128, rows - 1,
        boundary - 1, boundary, boundary + 1,
        rows / 2 - 1, rows / 2, rows / 2 + 1};

    DeviceBuffer<uint32_t> aes_key(4 * (14 + 1));
    DeviceBuffer<uint8_t> key(key_bytes);
    DeviceBuffer<uint128_t> seed(batch_size), intermediate(size_t(batch_size) << prefix);
    DeviceBuffer<uint32_t> bit(batch_size), intermediate_bit(size_t(batch_size) << prefix);
    DeviceBuffer<uint128_t> leaf(words), selection(words);
    DeviceBuffer<uint32_t> leaf_bit(words);
    std::vector<uint8_t> keys[2] = {std::vector<uint8_t>(key_bytes), std::vector<uint8_t>(key_bytes)};
    std::vector<uint64_t> queries(batch_size);
    std::vector<uint128_t> reference(words);
    std::vector<uint128_t> shares[2] = {std::vector<uint128_t>(words), std::vector<uint128_t>(words)};
    std::vector<uint32_t> leaf_bits(words);
    std::mt19937_64 rng(0x20261110ULL + n * 257 + batch_size);
    CUDA_LAUNCH(fss_genaeskey_kernel<<<1, 1>>>(aes_key.data));

    for (int repeat = 0; repeat < 3; ++repeat)
    {
        // Singleton batches sweep every edge without changing their launch shape.
        const int passes = batch_size == 1 ? static_cast<int>(edges.size()) + 2 : 1;
        for (int pass = 0; pass < passes; ++pass)
        {
            const uint64_t repeated_query = rng() & (rows - 1);
            for (int q = 0; q < batch_size; ++q)
            {
                if (batch_size == 1)
                {
                    if (pass < static_cast<int>(edges.size()))
                        queries[q] = edges[(pass + repeat) % edges.size()];
                    else if (pass == static_cast<int>(edges.size()))
                        queries[q] = repeated_query;
                    // The last singleton pass repeats the previous query with fresh keys.
                }
                else if (repeat == 0 && q < static_cast<int>(edges.size()))
                    queries[q] = edges[q];
                else if (repeat == 2)
                    queries[q] = repeated_query;
                else if (repeat == 1 && q % 4 == 3)
                    queries[q] = queries[q - 1];
                else
                    queries[q] = rng() & (rows - 1);
            }

            const std::string context = "n=" + std::to_string(n) + " B=" +
                std::to_string(batch_size) + " repeat=" + std::to_string(repeat) +
                " pass=" + std::to_string(pass);
            cudaDPFkeygen(keys[0].data(), keys[1].data(), queries.data(), n, maxlayer, batch_size);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            bool saw_leaf_bit[2] = {false, false};

            for (int party = 0; party < 2; ++party)
            {
                CUDA_CHECK(cudaMemcpy(key.data, keys[party].data(), key.bytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemset(seed.data, 0xa5, seed.bytes));
                CUDA_CHECK(cudaMemset(bit.data, 0xff, bit.bytes));
                CUDA_CHECK(cudaMemset(leaf.data, 0xa5, leaf.bytes));
                CUDA_CHECK(cudaMemset(leaf_bit.data, 0xff, leaf_bit.bytes));
                CUDA_LAUNCH(seed_copy<<<(batch_size + 255) / 256, 256>>>(
                    seed.data, bit.data, key.data, maxlayer, batch_size));

                uint128_t *input_seed = seed.data;
                uint32_t *input_bit = bit.data;
                if (prefix != 0)
                {
                    CUDA_CHECK(cudaMemset(intermediate.data, 0xa5, intermediate.bytes));
                    CUDA_CHECK(cudaMemset(intermediate_bit.data, 0xff, intermediate_bit.bytes));
                    CUDA_LAUNCH(EvalAll_BlockEvaluation<<<batch_size, (1 << prefix) / 2>>>(
                        aes_key.data, prefix, 0, key.data, seed.data, bit.data,
                        intermediate.data, intermediate_bit.data));
                    input_seed = intermediate.data;
                    input_bit = intermediate_bit.data;
                }

                CUDA_LAUNCH(EvalAll_BlockEvaluation<<<final_blocks, final_threads>>>(
                    aes_key.data, final_depth, prefix, key.data, input_seed, input_bit,
                    leaf.data, leaf_bit.data));
                CUDA_CHECK(cudaMemcpy(leaf_bits.data(), leaf_bit.data, leaf_bit.bytes, cudaMemcpyDeviceToHost));
                for (uint32_t value : leaf_bits)
                {
                    if (value > 1)
                        throw std::runtime_error(context + " invalid/unwritten leaf bit, party=" + std::to_string(party));
                    saw_leaf_bit[value] = true;
                }
                CUDA_CHECK(cudaMemset(selection.data, 0x3c, selection.bytes));
                CUDA_LAUNCH(EvalAll_SeGeneration<<<se_blocks, se_threads>>>(
                    key.data, leaf.data, leaf_bit.data, selection.data));
                CUDA_CHECK(cudaMemcpy(reference.data(), selection.data, selection.bytes, cudaMemcpyDeviceToHost));

                CUDA_CHECK(cudaMemset(selection.data, 0xc3, selection.bytes));
                CUDA_LAUNCH(EvalAll_BlockEvaluation_Se<<<final_blocks, final_threads>>>(
                    aes_key.data, final_depth, prefix, key.data, input_seed, input_bit, selection.data));
                CUDA_CHECK(cudaMemcpy(shares[party].data(), selection.data, selection.bytes, cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < words; ++i)
                {
                    if (shares[party][i] != reference[i])
                        throw std::runtime_error(context + " selection mismatch, party=" +
                            std::to_string(party) + " query=" + std::to_string(i / words_per_query) +
                            " word=" + std::to_string(i % words_per_query));
                }
                if (n <= 12)
                {
                    const auto cpu = cpu_selection(keys[party], n, batch_size);
                    for (size_t i = 0; i < words; ++i)
                    {
                        if (shares[party][i] != cpu[i])
                            throw std::runtime_error(context + " CPU selection mismatch, party=" +
                                std::to_string(party) + " query=" + std::to_string(i / words_per_query) +
                                " word=" + std::to_string(i % words_per_query));
                    }
                }
            }

            if (!saw_leaf_bit[0] || !saw_leaf_bit[1])
                throw std::runtime_error(context + " did not exercise both leaf correction bits");
            for (int q = 0; q < batch_size; ++q)
            {
                const unsigned bit_index = static_cast<unsigned>(queries[q] & 127);
                const uint64_t mask = uint64_t{1} << (63 - (bit_index & 63));
                const uint128_t selected = bit_index < 64 ? uint128_t(mask, 0) : uint128_t(0, mask);
                for (size_t word = 0; word < words_per_query; ++word)
                {
                    const size_t i = q * words_per_query + word;
                    const uint128_t expected = word == (queries[q] >> 7) ? selected : uint128_t(0, 0);
                    if ((shares[0][i] ^ shares[1][i]) != expected)
                        throw std::runtime_error(context + " non-one-hot XOR, query=" +
                            std::to_string(q) + " row=" + std::to_string(queries[q]) +
                            " word=" + std::to_string(word));
                }
            }
        }
    }
    std::printf("PASS n=%d B=%d (%s, final threads=%d)\n", n, batch_size,
                prefix ? "split r=9" : "single stage", final_threads);
}
} // namespace

int main()
{
    try
    {
        int devices = 0;
        const cudaError_t status = cudaGetDeviceCount(&devices);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
            (status == cudaSuccess && devices == 0))
        {
            std::printf("SKIP test_fused_eval: no available CUDA GPU (%s)\n", cudaGetErrorString(status));
            return 77;
        }
        CUDA_CHECK(status);
        const int cases[][2] = {{8, 1}, {9, 33}, {12, 31}, {16, 33}, {17, 129}, {20, 33}};
        for (const auto &test_case : cases)
            run_case(test_case[0], test_case[1]);
        CUDA_CHECK(cudaDeviceSynchronize());
        return cleanup_failed ? 1 : 0;
    }
    catch (const std::exception &error)
    {
        std::fprintf(stderr, "FAIL test_fused_eval: %s\n", error.what());
        return 1;
    }
}
