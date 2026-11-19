#pragma once

struct PirContext
{
    void *db = nullptr;
    void *output = nullptr;
    void *partials = nullptr; // [num_partitions][entry_size], single-query path
    int num_partitions = 0;
    void *key = nullptr;
    void *seed = nullptr;
    void *bit = nullptr;
    void *seed_intermediate = nullptr;
    void *bit_intermediate = nullptr;
    void *selection = nullptr;
    void *aes_key = nullptr;
};

struct pir_batch_context
{
    void *db = nullptr;
    void *partials = nullptr; // [num_blocks][batch_size][entry_size]
    void *output = nullptr;   // [batch_size][entry_size]
    void *key = nullptr;
    void *seed = nullptr;
    void *bit = nullptr;
    void *seed_intermediate = nullptr;
    void *bit_intermediate = nullptr;
    void *selection = nullptr;
    void *aes_key = nullptr;
    int num_blocks = 0;
};

struct PirPipelineContext
{
    void *db = nullptr;
    void *partials = nullptr;
    void *output = nullptr;
    void *seed = nullptr;
    void *bit = nullptr;
    void *seed_intermediate = nullptr;
    void *bit_intermediate = nullptr;
    void *seed_result = nullptr;
    void *bit_result = nullptr;
    void *selection = nullptr;
    void *aes_key = nullptr;
};

struct PirLutContext
{
    void *db = nullptr;
    void *input = nullptr;
    void *intermediate = nullptr;
    void *output = nullptr;
    void *seed = nullptr;
    void *bit = nullptr;
    void *seed_intermediate = nullptr;
    void *bit_intermediate = nullptr;
    void *seed_result = nullptr;
    void *bit_result = nullptr;
    void *selection = nullptr;
    void *aes_key = nullptr;
};

struct PirStreamContext
{
    void *streams = nullptr;
    void *events = nullptr;
};
