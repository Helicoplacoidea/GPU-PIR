#pragma once

#include <Eigen/Dense>
#include <iostream>
#include "emp-tool/emp-tool.h"
#include "../mpc_keys/aes_prg_host.h"
#include "../mpc_keys/fss_keygen.h"
using namespace emp;
using namespace Eigen;

typedef unsigned char *DCF_Keys;
// CUDA 函数声明
extern "C" void cudaWarmup(int size, int party);
extern "C" int test_dcf();
extern "C" uint128_t *test_dpf(uint8_t *key);
extern "C" uint128_t *test_dpf_pir(uint8_t *key, uint128_t *db, int N);
extern "C" uint128_t *test_dpf_pir_pipeline(uint8_t *key, uint128_t *db, int N);

extern "C" void cudafsseval(bool *res, DCF_Keys key, uint64_t *value, int N, int maxlayer, int party);
extern "C" void cudafsskeygen(DCF_Keys k0, DCF_Keys k1, uint64_t *alpha, int N, int n, int maxlayer);
extern "C" void cudaDPFkeygen(uint8_t *k0, uint8_t *k1, uint64_t *alpha, int n, int maxlayer, int batch_size);
extern "C" void cudamsbkeygen(DCF_Keys k0, DCF_Keys k1, int64_t *random0, int64_t *random1, bool *r_msb0, bool *r_msb1, int N, int maxlayer);
extern "C" void cudamsbeval(bool *res, DCF_Keys k, int64_t *value, bool *r_msb, int N, int maxlayer, int party);