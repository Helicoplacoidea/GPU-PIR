#pragma once

#include <iostream>
#include <vector>
#include "emp-tool/emp-tool.h"
#include "../mpc_keys/aes_prg_host.h"
#include "../mpc_keys/fss_keygen.h"
using namespace emp;

typedef unsigned char *DCF_Keys;

extern "C++" uint128_t *test_dpf_pir(uint8_t *key, std::vector<void *> d_ptrs, int N);
extern "C++" uint128_t *test_dpf_pir_pipeline(uint8_t *key, std::vector<void *> d_ptrs, std::vector<void *> handles, int N);
extern "C++" uint32_t *test_dpf_pir_LUT(uint8_t *key, std::vector<void *> d_ptrs, int N);
extern "C++" std::vector<void *> init_pir(int N, int n, uint128_t *db);
extern "C++" std::vector<void *> init_pir_pipeline(int N, int n, uint128_t *db);
extern "C++" std::vector<void *> init_pir_LUT(int N, int n, uint32_t *db);

extern "C++" std::vector<void *> init_streams_and_events();
extern "C++" void cleanup_streams_and_events(std::vector<void *> &handles);

void free_cuda_memory(std::vector<void *> &d_ptrs);

extern "C" void cudaDPFkeygen(uint8_t *k0, uint8_t *k1, uint64_t *alpha, int n, int maxlayer, int batch_size);
