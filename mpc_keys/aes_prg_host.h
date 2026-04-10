#ifndef AES_PRG_HOST_H
#define AES_PRG_HOST_H

#include "uint128_type.h"
#include <emp-tool/emp-tool.h>
#include <openssl/rand.h>
using namespace emp;

class AES_Generator
{
private:
    AES_KEY key;
    block counter;

    static block init_counter()
    {
        uint64_t high, low;
        RAND_bytes(reinterpret_cast<unsigned char *>(&high), sizeof(high));
        RAND_bytes(reinterpret_cast<unsigned char *>(&low), sizeof(low));
        return makeBlock(high, low);
    }

public:
    AES_Generator() : counter(init_counter())
    {
        uint64_t userkey1 = 597349;
        uint64_t userkey2 = 121379;
        block key_block = makeBlock(userkey1, userkey2);
        AES_set_encrypt_key(key_block, &key);
    }

    uint128_t random()
    {
        block result;
        result = counter;
        AES_ecb_encrypt_blks(&result, 1, &key);
        counter += makeBlock(1, 0);
        int64_t *v64val = (int64_t *)&result;
        return uint128_t(v64val[1], v64val[0]);
    }
};

#endif // AES_PRG_HOST_H