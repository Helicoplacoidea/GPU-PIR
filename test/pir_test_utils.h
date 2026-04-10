#pragma once

#include "../mpc_cuda/mpc_core.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

inline uint64_t generate_random_uint64()
{
    const uint64_t high = static_cast<uint64_t>(rand());
    const uint64_t low = static_cast<uint64_t>(rand());
    return (high << 32) | low;
}

inline bool validate_packed_dpf_depth(int n, const char *label)
{
    if (n < 8)
    {
        std::cerr << "[FAIL] " << label << ": n should be larger than 7\n";
        return false;
    }
    return true;
}

inline bool is_valid_lut_batch(int n, int batch_size)
{
    return n < 25 || batch_size <= 1;
}

inline bool validate_lut_batch(int n, int batch_size, const char *label)
{
    if (!is_valid_lut_batch(n, batch_size))
    {
        std::cerr << "[FAIL] " << label << ": batch_size should be 1 when n >= 25\n";
        return false;
    }
    return true;
}

inline size_t dpf_key_bytes(int n)
{
    return 1 + 16 + 1 + 18 * static_cast<size_t>(n - 7) + 16;
}

inline std::vector<uint128_t> make_random_pir_db(int n)
{
    const size_t total_entries = size_t{1} << n;
    std::vector<uint128_t> db(total_entries * entry_size);
    for (size_t i = 0; i < db.size(); ++i)
    {
        db[i] = uint128_t(generate_random_uint64(), generate_random_uint64());
    }
    return db;
}

inline std::vector<uint32_t> make_lut_db(int n)
{
    const size_t total_entries = size_t{1} << n;
    std::vector<uint32_t> db(total_entries);
    for (size_t i = 0; i < total_entries; ++i)
    {
        db[i] = static_cast<uint32_t>(i);
    }
    return db;
}

inline std::vector<uint64_t> make_descending_queries(int n, int batch_size)
{
    const uint64_t total_entries = 1ULL << n;
    std::vector<uint64_t> queries(batch_size);
    for (int i = 0; i < batch_size; ++i)
    {
        queries[i] = total_entries - static_cast<uint64_t>(i) - 1;
    }
    return queries;
}

inline std::vector<uint64_t> make_sequential_queries(int batch_size)
{
    std::vector<uint64_t> queries(batch_size);
    for (int i = 0; i < batch_size; ++i)
    {
        queries[i] = static_cast<uint64_t>(i);
    }
    return queries;
}

inline std::vector<uint64_t> make_constant_queries(int batch_size, uint64_t value)
{
    return std::vector<uint64_t>(batch_size, value);
}

inline bool verify_pir_response(const uint128_t *lhs,
                                const uint128_t *rhs,
                                const uint128_t *db,
                                const uint64_t *queries,
                                int batch_size,
                                std::string *error)
{
    for (int batch = 0; batch < batch_size; ++batch)
    {
        for (int entry = 0; entry < entry_size; ++entry)
        {
            const int offset = batch * entry_size + entry;
            const uint128_t combined = lhs[offset] ^ rhs[offset];
            const uint128_t expected = db[queries[batch] * entry_size + entry];
            if (combined != expected)
            {
                if (error != nullptr)
                {
                    std::ostringstream oss;
                    oss << "mismatch at batch " << batch << ", entry " << entry;
                    *error = oss.str();
                }
                return false;
            }
        }
    }
    return true;
}

inline bool verify_lut_response(const uint32_t *lhs,
                                const uint32_t *rhs,
                                const uint32_t *db,
                                const uint64_t *queries,
                                int batch_size,
                                std::string *error)
{
    for (int batch = 0; batch < batch_size; ++batch)
    {
        const uint32_t combined = lhs[batch] ^ rhs[batch];
        const uint32_t expected = db[queries[batch]];
        if (combined != expected)
        {
            if (error != nullptr)
            {
                std::ostringstream oss;
                oss << "mismatch at batch " << batch;
                *error = oss.str();
            }
            return false;
        }
    }
    return true;
}

inline bool parse_best_gpu_from_nvidia_smi_output(const std::string &output,
                                                  int *gpu_id,
                                                  int *free_mem_mb)
{
    std::istringstream stream(output);
    std::string line;
    int best_gpu = -1;
    int best_free_mem = std::numeric_limits<int>::min();
    int idx = 0;

    while (std::getline(stream, line))
    {
        if (line.empty())
        {
            continue;
        }

        std::istringstream line_stream(line);
        int parsed_free_mem = 0;
        if (!(line_stream >> parsed_free_mem))
        {
            return false;
        }

        if (parsed_free_mem > best_free_mem)
        {
            best_free_mem = parsed_free_mem;
            best_gpu = idx;
        }
        ++idx;
    }

    if (best_gpu < 0)
    {
        return false;
    }

    if (gpu_id != nullptr)
    {
        *gpu_id = best_gpu;
    }
    if (free_mem_mb != nullptr)
    {
        *free_mem_mb = best_free_mem;
    }
    return true;
}

inline bool select_best_gpu()
{
    FILE *pipe = popen("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits", "r");
    if (pipe == nullptr)
    {
        std::cerr << "[WARN] Failed to run nvidia-smi, using the default CUDA device\n";
        return false;
    }

    char buffer[128];
    std::string output;

    while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
    {
        output += buffer;
    }

    pclose(pipe);

    int gpu_id = -1;
    int max_free_mem = -1;
    if (!parse_best_gpu_from_nvidia_smi_output(output, &gpu_id, &max_free_mem))
    {
        std::cerr << "[WARN] Unable to parse nvidia-smi output, using the default CUDA device\n";
        return false;
    }

    std::ostringstream ss;
    ss << gpu_id;
    setenv("CUDA_VISIBLE_DEVICES", ss.str().c_str(), 1);
    std::cout << "[INFO] Selected GPU " << gpu_id << " (free memory: " << max_free_mem << " MB)\n";
    return true;
}

inline bool has_cuda_device()
{
    int device_count = 0;
    const cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count <= 0)
    {
        std::cerr << "[WARN] No CUDA-capable device is available\n";
        return false;
    }
    return true;
}

inline bool parse_cli_overrides(int argc,
                                const char *const argv[],
                                int default_n,
                                int default_batch_size,
                                int *n,
                                int *batch_size,
                                std::string *error)
{
    if (argc != 1 && argc != 3)
    {
        if (error != nullptr)
        {
            std::ostringstream oss;
            oss << "Usage: " << argv[0] << " [n batch_size]";
            *error = oss.str();
        }
        return false;
    }

    int parsed_n = default_n;
    int parsed_batch_size = default_batch_size;
    if (argc == 3)
    {
        try
        {
            parsed_n = std::stoi(argv[1]);
            parsed_batch_size = std::stoi(argv[2]);
        }
        catch (const std::exception &)
        {
            if (error != nullptr)
            {
                std::ostringstream oss;
                oss << "Usage: " << argv[0] << " [n batch_size]";
                *error = oss.str();
            }
            return false;
        }
    }

    if (parsed_n <= 0 || parsed_batch_size <= 0)
    {
        if (error != nullptr)
        {
            std::ostringstream oss;
            oss << "Usage: " << argv[0] << " [n batch_size]";
            *error = oss.str();
        }
        return false;
    }

    if (n != nullptr)
    {
        *n = parsed_n;
    }
    if (batch_size != nullptr)
    {
        *batch_size = parsed_batch_size;
    }
    return true;
}
