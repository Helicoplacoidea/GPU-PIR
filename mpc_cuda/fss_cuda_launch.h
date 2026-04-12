#pragma once

#include <stddef.h>
#include <stdint.h>

#include "../mpc_keys/uint128_type.h"

typedef unsigned char *DCF_Keys;

__global__ void fss_genaeskey_kernel(uint32_t key[4 * (14 + 1)]);
__global__ void dpf_gen_kernel(uint32_t *key, uint64_t *alpha, int n, DCF_Keys k0, DCF_Keys k1, int N, int maxlayer);
__global__ void EvalAll_LBLEvaluation(uint32_t *key, int layer, int maxlayer, int N, unsigned char *k, uint128_t *s, uint32_t *t);
__global__ void EvalAll_BlockEvaluation(uint32_t *key, int layer_len, int layer, unsigned char *k, uint128_t *s, uint32_t *t, uint128_t *res_s, uint32_t *res_t);
__global__ void EvalAll_lastlayer(unsigned char *k, uint128_t *s, uint32_t *t, uint128_t *res);
__global__ void MultiplicationReduction_grid_col(unsigned char *k, uint128_t *se, const uint128_t *d_db, uint128_t *blocksum_cuda, size_t block_num, int N);
__global__ void MultiplicationReduction_grid_row(unsigned char *k, uint128_t *se, const uint128_t *d_db, uint128_t *blocksum_cuda, size_t block_num, int N);
__global__ void MultiplicationReduction_LUT_new(unsigned char *k, uint128_t *se, const uint32_t *d_db, uint32_t *blocksum_cuda, size_t block_num, int N);
__global__ void MultiplicationReduction_LUT(unsigned char *k, uint128_t *se, const uint32_t *d_db, uint32_t *blocksum_cuda, size_t block_num);
__global__ void EVAL_Pack_last_layer_gense(unsigned char *k, uint128_t *s, uint32_t *t, uint128_t *se);
__global__ void EvalAll_SeGeneration(unsigned char *k, uint128_t *s, uint32_t *t, uint128_t *se);
__global__ void BlockReduction(uint128_t *blocksum_cuda, uint128_t *k_res);
__global__ void BlockReduction_LUT(uint32_t *blocksum_cuda, uint32_t *k_res);
__global__ void seed_copy(uint128_t *d_s, uint32_t *d_t, uint8_t *key, int maxlayer, int N);
