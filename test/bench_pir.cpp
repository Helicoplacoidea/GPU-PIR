#include "../mpc_cuda/mpc_core.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <stdlib.h>

uint64_t generate_random_uint64()
{
    uint64_t high = (uint64_t)rand();
    uint64_t low = (uint64_t)rand();
    return (high << 32) | low;
}

void bench_evalAll_cpu(int n)
{
    uint128_t *db = (uint128_t *)malloc(entry_size * (1 << n) * sizeof(uint128_t));
    for (uint64_t i = 0; i < (1 << n) * entry_size; i++)
    {
        db[i] = uint128_t(generate_random_uint64(), generate_random_uint64());
    }

    uint64_t userkey1 = 597349;
    uint64_t userkey2 = 121379;
    block userkey = makeBlock(userkey1, userkey2);

    AES_KEY key_host;
    AES_set_encrypt_key(userkey, &key_host);

    AES_Generator prg;

    int maxlayer = n - 7;
    if (maxlayer < 0)
    {
        printf("n should be larger than 7\n");
        return;
    }
    uint128_t a = uint128_t(0, 65536);
    // uint64_t a = 65535;
    unsigned char *k0;
    unsigned char *k1;
    k0 = (unsigned char *)malloc(1 + 16 + 1 + 18 * maxlayer + 16);
    k1 = (unsigned char *)malloc(1 + 16 + 1 + 18 * maxlayer + 16);
    GEN_Pack(&prg, &key_host, a, n, &k0, &k1);

    uint128_t *pack_res0;
    pack_res0 = (uint128_t *)malloc(sizeof(uint128_t) * entry_size);
    // pack_res1 = (uint128_t *)malloc(sizeof(uint128_t) * entry_size);
    // pack_res = (uint128_t *)malloc(sizeof(uint128_t) * entry_size);

    std::vector<uint8_t> se0, se1, res;
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < 100; i++)
    {
        db[i].print_uint128("db:", db[i]);

        se0 = EvalFull(&key_host, k0, n);
        // se1 = EvalFull(&key_host, k1, n);

        for (int i = 0; i < entry_size; i++)
        {
            pack_res0[i] = uint128_t(0, 0);
            // pack_res1[i] = uint128_t(0, 0);
        }
        for (int i = 0; i < (1 << n); i++)
        {
            for (int j = 0; j < entry_size; j++)
            {
                pack_res0[j] ^= db[i * entry_size + j].select((se0[i / 8] >> (7 - i % 8)) & 1);
                // pack_res1[j] ^= db[i * entry_size + j].select((se1[i / 8] >> (7 - i % 8)) & 1);
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    printf("DPF-PIR Time taken: %f milliseconds\n", elapsed.count() * 1000 / 100);
    return;
}

void bench_pir(int n, int batch_size)
{
    uint128_t *db = (uint128_t *)malloc(entry_size * (1 << n) * sizeof(uint128_t));
    for (uint64_t i = 0; i < (1 << n) * entry_size; i++)
    {
        db[i] = uint128_t(generate_random_uint64(), generate_random_uint64());
    }
    int maxlayer = n - 7;
    if (maxlayer < 0)
    {
        printf("n should be larger than 7\n");
        return;
    }
    // uint64_t a = 5;
    uint64_t *a = (uint64_t *)malloc(sizeof(uint64_t) * batch_size);
    for (int i = 0; i < batch_size; i++)
    {
        a[i] = i;
    }

    unsigned char *k0;
    unsigned char *k1;
    k0 = (unsigned char *)malloc(batch_size * (1 + 16 + 1 + 18 * maxlayer + 16));
    k1 = (unsigned char *)malloc(batch_size * (1 + 16 + 1 + 18 * maxlayer + 16));

    cudaDPFkeygen(k0, k1, a, n, maxlayer, batch_size);

    uint128_t *pack_res0;
    pack_res0 = (uint128_t *)malloc(sizeof(uint128_t) * entry_size * batch_size);

    std::vector<void *> d_ptrs = init_pir(batch_size, n, db);

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < 10; i++)
    {
        pack_res0 = test_dpf_pir(k0, d_ptrs, batch_size);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    printf("DPF-PIR Time taken: %f ms\n", elapsed.count() * 1000 / 10);
    printf("Throughput: %.2f pirs/s\n", 10 / elapsed.count() * batch_size);

    free_cuda_memory(d_ptrs);

    return;
}

void bench_pir_pipeline(int n, int batch_size)
{
    uint128_t *db = (uint128_t *)malloc(entry_size * (1 << n) * sizeof(uint128_t));
    for (uint64_t i = 0; i < (1 << n) * entry_size; i++)
    {
        db[i] = uint128_t(generate_random_uint64(), generate_random_uint64());
    }
    int maxlayer = n - 7;
    if (maxlayer < 0)
    {
        printf("n should be larger than 7\n");
        return;
    }
    // uint64_t a = 5;
    uint64_t *a = (uint64_t *)malloc(sizeof(uint64_t) * batch_size);
    for (int i = 0; i < batch_size; i++)
    {
        a[i] = (1 << n) - i - 1;
    }

    unsigned char *k0;
    unsigned char *k1;
    k0 = (unsigned char *)malloc(batch_size * (1 + 16 + 1 + 18 * maxlayer + 16));
    k1 = (unsigned char *)malloc(batch_size * (1 + 16 + 1 + 18 * maxlayer + 16));

    cudaDPFkeygen(k0, k1, a, n, maxlayer, batch_size);
    std::vector<void *> d_ptrs = init_pir_pipeline(batch_size, n, db);
    std::vector<void *> handles = init_streams_and_events();

    uint128_t *pack_res;
    pack_res = (uint128_t *)malloc(sizeof(uint128_t) * entry_size * batch_size);

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < 10; i++)
    {
        pack_res = test_dpf_pir_pipeline(k0, d_ptrs, handles, batch_size);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    printf("DPF-PIR Time taken: %f ms\n", elapsed.count() * 1000 / 10);
    printf("Throughput: %.2f pirs/s\n", 10 / elapsed.count() * batch_size);

    free_cuda_memory(d_ptrs);
    cleanup_streams_and_events(handles);

    return;
}

void bench_pir_LUT(int n, int batch_size)
{
    uint64_t total = (1ULL << n);
    uint32_t *db = (uint32_t *)malloc(total * sizeof(uint32_t));
    if (!db)
    {
        perror("malloc failed");
        exit(1);
    }

    for (uint64_t i = 0; i < (1ULL << n); i++)
    {
        db[i] = i;
    }
    int maxlayer = n - 7;
    if (maxlayer < 0)
    {
        printf("n should be larger than 7\n");
        return;
    }

    // uint64_t a = 5;
    uint64_t *a = (uint64_t *)malloc(sizeof(uint64_t) * batch_size);
    for (int i = 0; i < batch_size; i++)
    {
        a[i] = 4;
    }

    // for (int i = entry_size - 1; i >= 0; i--)
    // {
    //     db[0 * entry_size + i].print_uint128("", db[3 * entry_size + i]);
    // }
    // printf("\n");

    unsigned char *k0;
    unsigned char *k1;
    k0 = (unsigned char *)malloc(batch_size * (1 + 16 + 1 + 18 * maxlayer + 16));
    k1 = (unsigned char *)malloc(batch_size * (1 + 16 + 1 + 18 * maxlayer + 16));

    cudaDPFkeygen(k0, k1, a, n, maxlayer, batch_size);
    std::vector<void *> d_ptrs = init_pir_LUT(batch_size, n, db);
    // std::vector<void *> handles = init_streams_and_events();

    uint32_t *pack_res;
    pack_res = (uint32_t *)malloc(sizeof(uint32_t) * batch_size);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++)
        pack_res = test_dpf_pir_LUT(k0, d_ptrs, batch_size);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    printf("DPF-PIR Time taken: %f ms\n", elapsed.count() * 1000 / 10);
    printf("Throughput: %.2f pirs/s\n", 10 / elapsed.count() * batch_size);

    // free_cuda_memory(d_ptrs);
    // cleanup_streams_and_events(handles);
    free(db);
    // free(pack_res0);
    // free(k0);
    // free(k1);
    // free(a);
}

void select_best_gpu()
{
    FILE *pipe = popen(
        "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits", "r");
    if (!pipe)
    {
        std::cerr << "Failed to run nvidia-smi\n";
        return;
    }

    std::string line;
    char buffer[128];
    int gpu_id = -1;
    int max_free_mem = -1;
    int idx = 0;

    while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
    {
        int free_mem = std::stoi(buffer);
        if (free_mem > max_free_mem)
        {
            max_free_mem = free_mem;
            gpu_id = idx;
        }
        idx++;
    }

    pclose(pipe);

    if (gpu_id >= 0)
    {
        std::ostringstream ss;
        ss << gpu_id;
        setenv("CUDA_VISIBLE_DEVICES", ss.str().c_str(), 1);
        std::cout << "[INFO] Selected GPU " << gpu_id
                  << " (free memory: " << max_free_mem << " MB)" << std::endl;
    }
    else
    {
        std::cerr << "[WARN] No GPU found via nvidia-smi\n";
    }
}

int main(int argc, char **argv)
{
    select_best_gpu();
    size_t N = 22;
    size_t batch = 512;

    // if (N >= 25)
    // {
    //     if (batch > 1)
    //     {
    //         printf("batch_size should be 1 when n >= 25\n");
    //         return 0;
    //     }
    // }

    bench_pir(N, batch);
    bench_pir_pipeline(N, batch);
    bench_pir_LUT(N, batch);

    return 0;
}
