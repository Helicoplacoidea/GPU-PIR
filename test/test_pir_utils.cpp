#include "../mpc_cuda/mpc_core.h"

#include <iostream>
#include <string>
#include <vector>

#include "pir_test_utils.h"

namespace
{
bool verify_uint128_result_reports_mismatch()
{
    const std::vector<uint128_t> lhs = {uint128_t(0, 1), uint128_t(0, 2)};
    const std::vector<uint128_t> rhs = {uint128_t(0, 0), uint128_t(0, 0)};
    const std::vector<uint128_t> db = {uint128_t(0, 1), uint128_t(0, 3)};
    const std::vector<uint64_t> queries = {0};

    std::string error;
    const bool ok = verify_pir_response(lhs.data(), rhs.data(), db.data(), queries.data(), 1, &error);
    return !ok && error.find("batch 0") != std::string::npos && error.find("entry 1") != std::string::npos;
}

bool verify_lut_limit_rule()
{
    return !is_valid_lut_batch(25, 2) && is_valid_lut_batch(24, 2) && is_valid_lut_batch(25, 1);
}

bool verify_gpu_parser_rejects_non_numeric_output()
{
    const std::string output =
        "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.\n";
    int gpu_id = -1;
    int free_mem = -1;
    return !parse_best_gpu_from_nvidia_smi_output(output, &gpu_id, &free_mem);
}

bool verify_cli_override_parser()
{
    {
        const char *argv[] = {"test_pir"};
        int n = 0;
        int batch_size = 0;
        std::string error;
        if (!parse_cli_overrides(1, argv, 24, 512, &n, &batch_size, &error))
        {
            return false;
        }
        if (n != 24 || batch_size != 512)
        {
            return false;
        }
    }

    {
        const char *argv[] = {"test_pir", "20", "128"};
        int n = 0;
        int batch_size = 0;
        std::string error;
        if (!parse_cli_overrides(3, argv, 24, 512, &n, &batch_size, &error))
        {
            return false;
        }
        if (n != 20 || batch_size != 128)
        {
            return false;
        }
    }

    {
        const char *argv[] = {"test_pir", "abc", "128"};
        int n = 0;
        int batch_size = 0;
        std::string error;
        if (parse_cli_overrides(3, argv, 24, 512, &n, &batch_size, &error))
        {
            return false;
        }
        if (error.find("Usage") == std::string::npos)
        {
            return false;
        }
    }

    return true;
}
} // namespace

int main()
{
    if (!verify_uint128_result_reports_mismatch())
    {
        std::cerr << "uint128 verification helper did not report mismatch correctly\n";
        return 1;
    }

    if (!verify_lut_limit_rule())
    {
        std::cerr << "LUT batch validation helper returned unexpected result\n";
        return 1;
    }

    if (!verify_gpu_parser_rejects_non_numeric_output())
    {
        std::cerr << "GPU parser accepted invalid nvidia-smi output\n";
        return 1;
    }

    if (!verify_cli_override_parser())
    {
        std::cerr << "CLI override parser returned unexpected result\n";
        return 1;
    }

    return 0;
}
