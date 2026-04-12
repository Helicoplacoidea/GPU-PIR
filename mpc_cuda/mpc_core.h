#pragma once

#include <iostream>
#include "pir_context.h"
#include "emp-tool/emp-tool.h"
#include "../mpc_keys/aes_prg_host.h"
#include "../mpc_keys/fss_keygen.h"
using namespace emp;

// struct PirContext is defined in pir_context.h.
// struct PirPipelineContext is defined in pir_context.h.
// struct PirLutContext is defined in pir_context.h.
// struct PirStreamContext is defined in pir_context.h.

extern "C++" uint128_t *test_dpf_pir(uint8_t *key, const PirContext &ctx, int N);
extern "C++" uint128_t *test_dpf_pir_pipeline(uint8_t *key, const PirPipelineContext &ctx, const PirStreamContext &handles, int N);
extern "C++" uint32_t *test_dpf_pir_LUT(uint8_t *key, const PirLutContext &ctx, int N);
extern "C++" PirContext init_pir(int N, int n, uint128_t *db);
extern "C++" PirPipelineContext init_pir_pipeline(int N, int n, uint128_t *db);
extern "C++" PirLutContext init_pir_LUT(int N, int n, uint32_t *db);

extern "C++" PirStreamContext init_streams_and_events();
extern "C++" void cleanup_streams_and_events(PirStreamContext &handles);

void free_cuda_memory(PirContext &ctx);
void free_cuda_memory(PirPipelineContext &ctx);
void free_cuda_memory(PirLutContext &ctx);

extern "C" void cudaDPFkeygen(uint8_t *k0, uint8_t *k1, uint64_t *alpha, int n, int maxlayer, int batch_size);
