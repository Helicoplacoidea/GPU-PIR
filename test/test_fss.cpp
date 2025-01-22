#include "../mpc_cuda/mpc_core.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <stdlib.h>

#define entry_size 16

uint64_t generate_random_uint64()
{
    uint64_t high = (uint64_t)rand();
    uint64_t low = (uint64_t)rand();
    return (high << 32) | low;
}

void test_evalAll(int n)
{
    int maxlayer = n - 7;
    uint8_t *k0, *k1;
    uint64_t a = 3;

    k0 = (unsigned char *)malloc(1 + 16 + 1 + 18 * maxlayer + 16);
    k1 = (unsigned char *)malloc(1 + 16 + 1 + 18 * maxlayer + 16);

    cudaDPFkeygen(k0, k1, &a, maxlayer + 7, maxlayer, 1);

    uint128_t *pack_res0, *pack_res1, *pack_res;
    pack_res0 = (uint128_t *)malloc(sizeof(uint128_t) * (1 << maxlayer));
    pack_res1 = (uint128_t *)malloc(sizeof(uint128_t) * (1 << maxlayer));
    pack_res = (uint128_t *)malloc(sizeof(uint128_t) * (1 << maxlayer));

    pack_res0 = test_dpf(k0);
    pack_res1 = test_dpf(k1);
    for (int i = 0; i < (1 << maxlayer); i++)
    {
        pack_res[i] = pack_res0[i] ^ pack_res1[i];
        // pack_res0[i].print_uint128("dpf_res0:", pack_res0[i]);
        // pack_res1[i].print_uint128("dpf_res1:", pack_res1[i]);
        pack_res[i].print_uint128("dpf_res:", pack_res[i]);
    }
}

void test_pir(int n, int batch_size)
{
    uint128_t *db = (uint128_t *)malloc(entry_size * (1 << n) * sizeof(uint128_t));
    for (uint64_t i = 0; i < (1 << n) * entry_size; i++)
    {
        db[i] = uint128_t(generate_random_uint64(), generate_random_uint64());
    }
    int maxlayer = n - 7;
    // uint64_t a = 5;
    uint64_t *a = (uint64_t *)malloc(sizeof(uint64_t) * batch_size);
    for (int i = 0; i < batch_size; i++)
    {
        a[i] = i;
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

    uint128_t *pack_res0, *pack_res1, *pack_res;
    pack_res0 = (uint128_t *)malloc(sizeof(uint128_t) * entry_size * batch_size);
    pack_res1 = (uint128_t *)malloc(sizeof(uint128_t) * entry_size * batch_size);
    pack_res = (uint128_t *)malloc(sizeof(uint128_t) * entry_size * batch_size);

    pack_res0 = test_dpf_pir(k0, db, batch_size);
    pack_res1 = test_dpf_pir(k1, db, batch_size);

    // printf("\n");

    for (int j = 0; j < batch_size; j++)
        for (int i = entry_size - 1; i >= 0; i--)
        {
            pack_res[i + j * entry_size] = pack_res1[i + j * entry_size] ^ pack_res0[i + j * entry_size];
            // pack_res[i].print_uint128("res:", pack_res[i + j * entry_size]);
            if (pack_res[j * entry_size + i] != db[a[j] * entry_size + i])
            {
                printf("Error at %d pir", j);
                break;
            }
        }
}

void test_pir_pipeline(int n, int batch_size)
{
    uint128_t *db = (uint128_t *)malloc(entry_size * (1 << n) * sizeof(uint128_t));
    for (uint64_t i = 0; i < (1 << n) * entry_size; i++)
    {
        db[i] = uint128_t(generate_random_uint64(), generate_random_uint64());
    }
    int maxlayer = n - 7;
    // uint64_t a = 5;
    uint64_t *a = (uint64_t *)malloc(sizeof(uint64_t) * batch_size);
    for (int i = 0; i < batch_size; i++)
    {
        a[i] = i;
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

    uint128_t *pack_res0, *pack_res1, *pack_res;
    pack_res0 = (uint128_t *)malloc(sizeof(uint128_t) * entry_size * batch_size);
    pack_res1 = (uint128_t *)malloc(sizeof(uint128_t) * entry_size * batch_size);
    pack_res = (uint128_t *)malloc(sizeof(uint128_t) * entry_size * batch_size);

    pack_res0 = test_dpf_pir_pipeline(k0, db, batch_size);
    pack_res1 = test_dpf_pir_pipeline(k1, db, batch_size);

    printf("\n");

    for (int j = 0; j < batch_size; j++)
        for (int i = entry_size - 1; i >= 0; i--)
        {
            pack_res[i + j * entry_size] = pack_res1[i + j * entry_size] ^ pack_res0[i + j * entry_size];
            // pack_res[i].print_uint128("res:", pack_res0[i + j * entry_size]);
            if (pack_res[j * entry_size + i] != db[a[j] * entry_size + i])
            {
                printf("Error at %d pir", j);
                break;
            }
        }
}

// 测试代码
int main()
{

    // uint64_t userkey1 = 597349;
    // uint64_t userkey2 = 121379;
    // block userkey = makeBlock(userkey1, userkey2);
    // uint64_t plaintext1 = 597349;
    // uint64_t plaintext2 = 121379;
    // uint128_t plaintext(plaintext1, plaintext2);
    // uint128_t ciphertext;
    // AES_KEY key_host;
    // AES_set_encrypt_key(userkey, &key_host);
    // // AES_ecb_encrypt(plaintext.get_bytes(), ciphertext.get_bytes(), &key_host,1);

    // uint128_t ciphertext_2;
    // int bit1, bit2;
    // Double_PRG(&key_host, plaintext, ciphertext, ciphertext_2, bit1, bit2);
    // plaintext.print_uint128("plaintext = ", plaintext);
    // ciphertext.print_uint128("ciphertext1 = ", ciphertext);
    // ciphertext_2.print_uint128("ciphertext2 = ", ciphertext_2);
    // printf("bit1 = %d, bit2 = %d\n", bit1, bit2);

    // AES_Generator prg;
    // uint128_t output1, output2;
    // uint8_t *k0, *k1;
    // auto start = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < 30520; i++)
    // {
    //     fss_gen(&prg, &key_host, uint128_t(0, 5), 64, &k0, &k1);
    // }
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed = end - start;
    // printf("Keygen Time taken: %f milliseconds\n", elapsed.count() * 1000);

    // start = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < 30520; i++)
    // {
    //     output1 = dcf_eval(&key_host, k0, uint128_t(0, i));
    //     // output2 = dcf_eval(&key_host, k1, uint128_t(0, i));
    //     // printf("output1 = %lu, output2 = %lu\n", output1.get_low(), output2.get_low());
    //     // uint128_t res = output1 ^ output2;
    //     // res.print_uint128("res = ", res);
    // }
    // end = std::chrono::high_resolution_clock::now();
    // elapsed = end - start;
    // printf("CPU Time taken: %f milliseconds\n", elapsed.count() * 1000);

    printf("===============================================\n");

    cudaWarmup(512, 0);
    // int N = 30520;
    // int maxlayer = 64;
    // DCF_Keys dcf_k0, dcf_k1;
    // dcf_k0 = (DCF_Keys)malloc(N * (1 + 16 + 1 + 18 * maxlayer + 16));
    // dcf_k1 = (DCF_Keys)malloc(N * (1 + 16 + 1 + 18 * maxlayer + 16));
    // uint64_t *alpha = new uint64_t[N];
    // uint64_t *alpha2 = new uint64_t[N];
    // for (int i = 0; i < N; i++)
    // {
    //     alpha[i] = 64;
    //     alpha2[i] = i;
    // }

    // start = std::chrono::high_resolution_clock::now();
    // cudafsskeygen(dcf_k0, dcf_k1, alpha, N, 64, maxlayer);
    // end = std::chrono::high_resolution_clock::now();
    // elapsed = end - start;
    // printf("Keygen CUDA Time taken: %f milliseconds\n", elapsed.count() * 1000);

    // bool *res1 = new bool[N];
    // bool *res2 = new bool[N];
    // start = std::chrono::high_resolution_clock::now();
    // cudafsseval(res1, dcf_k0, alpha2, N, maxlayer, 0);
    // end = std::chrono::high_resolution_clock::now();
    // elapsed = end - start;
    // printf("Eval CUDA Time taken: %f milliseconds\n", elapsed.count() * 1000);

    // cudafsseval(res2, dcf_k1, alpha2, N, maxlayer, 1);

    // start = std::chrono::high_resolution_clock::now();
    // uint32_t final_res = 0;
    // for (int i = 0; i < N; i++)
    // {
    //     final_res += (uint32_t)(res1[i] ^ res2[i]);
    //     if (res1[i] ^ res2[i] ^ (i >= 64))
    //     {
    //         printf("Error at i = %d, res1[i]^res2[i] = %d, i>=64 = %d\n", i, res1[i] ^ res2[i], i >= 64);
    //     }
    // }
    // printf("final_res = %d\n", final_res);
    // end = std::chrono::high_resolution_clock::now();
    // elapsed = end - start;
    // printf("Final res reduction Time taken: %f milliseconds\n", elapsed.count() * 1000);

    // free(dcf_k0);
    // free(dcf_k1);

    // int maxlayer = 9;
    // // // uint128_t a = uint128_t(0, 128);
    // uint64_t a = 65535;
    // unsigned char *k0;
    // unsigned char *k1;
    // k0 = (unsigned char *)malloc(1 + 16 + 1 + 18 * maxlayer + 16);
    // k1 = (unsigned char *)malloc(1 + 16 + 1 + 18 * maxlayer + 16);
    // // // GEN_Pack(&prg, &key_host, a, 20, &k0, &k1);

    // // // uint128_t dpf_res0, dpf_res1, dpf_res;
    // // // uint128_t x = uint128_t(0, 130);
    // // // dpf_res0 = EVAL_Pack(&key_host, k0, x);
    // // // dpf_res1 = EVAL_Pack(&key_host, k1, x);
    // // // dpf_res0.print_uint128("dpf_res0:", dpf_res0);
    // // // dpf_res1.print_uint128("dpf_res1:", dpf_res1);
    // // // dpf_res = dpf_res0 ^ dpf_res1;
    // // // dpf_res.print_uint128("dpf_res:", dpf_res);

    // cudaDPFkeygen(k0, k1, &a, maxlayer + 7, maxlayer, 1);

    // uint128_t *pack_res0, *pack_res1, *pack_res;
    // pack_res0 = (uint128_t *)malloc(sizeof(uint128_t) * (1 << maxlayer));
    // pack_res1 = (uint128_t *)malloc(sizeof(uint128_t) * (1 << maxlayer));
    // pack_res = (uint128_t *)malloc(sizeof(uint128_t) * (1 << maxlayer));

    // pack_res0 = test_dpf(k0);
    // pack_res1 = test_dpf(k1);
    // for (int i = 0; i < 512; i++)
    // {
    //     pack_res[i] = pack_res0[i] ^ pack_res1[i];
    //     // pack_res0[i].print_uint128("dpf_res0:", pack_res0[i]);
    //     // pack_res1[i].print_uint128("dpf_res1:", pack_res1[i]);
    //     pack_res[i].print_uint128("dpf_res:", pack_res[i]);
    // }
    int a = 20, b = 1024;
    scanf("%d %d", &a, &b);

    test_pir(a, b);
    test_pir_pipeline(a, b);

    // delete[] alpha;
    // delete[] res1;
    // delete[] res2;

    return 0;
}
