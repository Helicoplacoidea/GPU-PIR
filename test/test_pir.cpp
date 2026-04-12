#include "../mpc_cuda/mpc_core.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "pir_test_utils.h"

namespace
{
    bool run_pir_test(int n, int batch_size)
    {
        if (!validate_packed_dpf_depth(n, "test_pir"))
        {
            return false;
        }

        const int maxlayer = n - 7;
        std::vector<uint128_t> db = make_random_pir_db(n);
        std::vector<uint64_t> queries = make_descending_queries(n, batch_size);
        std::vector<uint8_t> k0(batch_size * dpf_key_bytes(n));
        std::vector<uint8_t> k1(batch_size * dpf_key_bytes(n));

        cudaDPFkeygen(k0.data(), k1.data(), queries.data(), n, maxlayer, batch_size);
        PirContext ctx = init_pir(batch_size, n, db.data());

        uint128_t *res0 = test_dpf_pir(k0.data(), ctx, batch_size);
        uint128_t *res1 = test_dpf_pir(k1.data(), ctx, batch_size);

        std::string error;
        const bool ok = verify_pir_response(res0, res1, db.data(), queries.data(), batch_size, &error);

        free(res0);
        free(res1);
        free_cuda_memory(ctx);

        if (!ok)
        {
            std::cerr << "[FAIL] test_pir: " << error << '\n';
            return false;
        }

        std::cout << "[PASS] test_pir\n";
        return true;
    }

    bool run_pir_pipeline_test(int n, int batch_size)
    {
        if (!validate_packed_dpf_depth(n, "test_pir_pipeline"))
        {
            return false;
        }

        const int maxlayer = n - 7;
        std::vector<uint128_t> db = make_random_pir_db(n);
        std::vector<uint64_t> queries = make_descending_queries(n, batch_size);
        std::vector<uint8_t> k0(batch_size * dpf_key_bytes(n));
        std::vector<uint8_t> k1(batch_size * dpf_key_bytes(n));

        cudaDPFkeygen(k0.data(), k1.data(), queries.data(), n, maxlayer, batch_size);
        PirPipelineContext ctx = init_pir_pipeline(batch_size, n, db.data());
        PirStreamContext handles = init_streams_and_events();

        uint128_t *res0 = test_dpf_pir_pipeline(k0.data(), ctx, handles, batch_size);
        uint128_t *res1 = test_dpf_pir_pipeline(k1.data(), ctx, handles, batch_size);

        std::string error;
        const bool ok = verify_pir_response(res0, res1, db.data(), queries.data(), batch_size, &error);

        cudaFreeHost(res0);
        cudaFreeHost(res1);
        free_cuda_memory(ctx);
        cleanup_streams_and_events(handles);

        if (!ok)
        {
            std::cerr << "[FAIL] test_pir_pipeline: " << error << '\n';
            return false;
        }

        std::cout << "[PASS] test_pir_pipeline\n";
        return true;
    }

    bool run_pir_lut_test(int n, int batch_size)
    {
        if (!validate_packed_dpf_depth(n, "test_pir_LUT") ||
            !validate_lut_batch(n, batch_size, "test_pir_LUT"))
        {
            return false;
        }

        const int maxlayer = n - 7;
        std::vector<uint32_t> db = make_lut_db(n);
        std::vector<uint64_t> queries = make_constant_queries(batch_size, 3);
        std::vector<uint8_t> k0(batch_size * dpf_key_bytes(n));
        std::vector<uint8_t> k1(batch_size * dpf_key_bytes(n));

        cudaDPFkeygen(k0.data(), k1.data(), queries.data(), n, maxlayer, batch_size);
        PirLutContext ctx = init_pir_LUT(batch_size, n, db.data());

        uint32_t *res0 = test_dpf_pir_LUT(k0.data(), ctx, batch_size);
        uint32_t *res1 = test_dpf_pir_LUT(k1.data(), ctx, batch_size);

        std::string error;
        const bool ok = verify_lut_response(res0, res1, db.data(), queries.data(), batch_size, &error);

        cudaFreeHost(res0);
        cudaFreeHost(res1);
        free_cuda_memory(ctx);

        if (!ok)
        {
            std::cerr << "[FAIL] test_pir_LUT: " << error << '\n';
            return false;
        }

        std::cout << "[PASS] test_pir_LUT\n";
        return true;
    }
} // namespace

int main(int argc, char **argv)
{
    std::srand(1);

    int n = 24;
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
        std::cout << "[SKIP] test_pir requires a CUDA-capable device\n";
        return 0;
    }

    bool ok = true;
    ok = run_pir_test(n, batch_size) && ok;
    ok = run_pir_pipeline_test(n, batch_size) && ok;
    ok = run_pir_lut_test(n, batch_size) && ok;

    return ok ? 0 : 1;
}
