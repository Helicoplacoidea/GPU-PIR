#pragma once

#include <iostream>
#include "pir_context.h"
#include "pir_config.h"
#include "emp-tool/emp-tool.h"
#include "../mpc_keys/aes_prg_host.h"
#include "../mpc_keys/fss_keygen.h"

// struct PirContext / pir_batch_context are defined in pir_context.h.
// struct PirPipelineContext / PirLutContext / PirStreamContext are defined in pir_context.h.

extern "C++" uint128_t *test_dpf_pir(uint8_t *key, const PirContext &ctx, int N);
// mode: 0 = batch_mul_red_sync_kernel, 1 = batch_mul_red_async_kernel
extern "C++" uint128_t *test_dpf_pir_batch(uint8_t *key, const pir_batch_context &ctx, int batch_size, int mode);
extern "C++" uint128_t *test_dpf_pir_pipeline(uint8_t *key, const PirPipelineContext &ctx, const PirStreamContext &handles, int N);
extern "C++" uint32_t *test_dpf_pir_LUT(uint8_t *key, const PirLutContext &ctx, int N);
extern "C++" PirContext init_pir(int N, int n, uint128_t *db);
extern "C++" pir_batch_context init_pir_batch(int batch_size, int db_bits, uint128_t *db);
extern "C++" PirPipelineContext init_pir_pipeline(int N, int n, uint128_t *db);
extern "C++" PirLutContext init_pir_LUT(int N, int n, uint32_t *db);

extern "C++" PirStreamContext init_streams_and_events();
extern "C++" void cleanup_streams_and_events(PirStreamContext &handles);

void free_cuda_memory(PirContext &ctx);
void free_cuda_memory(pir_batch_context &ctx);
void free_cuda_memory(PirPipelineContext &ctx);
void free_cuda_memory(PirLutContext &ctx);

extern "C" void cudaDPFkeygen(uint8_t *k0, uint8_t *k1, uint64_t *alpha, int n, int maxlayer, int batch_size);
