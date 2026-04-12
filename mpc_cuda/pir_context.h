#pragma once

typedef unsigned char *DCF_Keys;

struct PirContext
{
    void *db = nullptr;
    void *blocksum = nullptr;
    void *key = nullptr;
    void *seed = nullptr;
    void *bit = nullptr;
    void *seed_intermediate = nullptr;
    void *bit_intermediate = nullptr;
    void *seed_result = nullptr;
    void *bit_result = nullptr;
    void *selection = nullptr;
    void *aes_key = nullptr;
};

struct PirPipelineContext
{
    void *db = nullptr;
    void *input = nullptr;
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
