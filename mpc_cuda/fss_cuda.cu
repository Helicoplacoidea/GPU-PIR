#include <stdio.h>

#include <string.h>
#include <vector>
#include <iostream>
#include "aes_cuda.h"
#include "../mpc_keys/uint128_type.h"
#include "aes_prg_device.h"
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

// #define entry_size 2

typedef unsigned char *DCF_Keys;

__device__ const uint64_t select_vector_table[256] = {
	0x8000000000000000ULL, 0x0000000000000000UL, // 0
	0x4000000000000000ULL, 0x0000000000000000UL, // 1
	0x2000000000000000ULL, 0x0000000000000000UL, // 2
	0x1000000000000000ULL, 0x0000000000000000UL, // 3
	0x0800000000000000ULL, 0x0000000000000000UL, // 4
	0x0400000000000000ULL, 0x0000000000000000UL, // 5
	0x0200000000000000ULL, 0x0000000000000000UL, // 6
	0x0100000000000000ULL, 0x0000000000000000UL, // 7
	0x0080000000000000ULL, 0x0000000000000000UL, // 8
	0x0040000000000000ULL, 0x0000000000000000UL, // 9
	0x0020000000000000ULL, 0x0000000000000000UL, // 10
	0x0010000000000000ULL, 0x0000000000000000UL, // 11
	0x0008000000000000ULL, 0x0000000000000000UL, // 12
	0x0004000000000000ULL, 0x0000000000000000UL, // 13
	0x0002000000000000ULL, 0x0000000000000000UL, // 14
	0x0001000000000000ULL, 0x0000000000000000UL, // 15
	0x0000800000000000ULL, 0x0000000000000000UL, // 16
	0x0000400000000000ULL, 0x0000000000000000UL, // 17
	0x0000200000000000ULL, 0x0000000000000000UL, // 18
	0x0000100000000000ULL, 0x0000000000000000UL, // 19
	0x0000080000000000ULL, 0x0000000000000000UL, // 20
	0x0000040000000000ULL, 0x0000000000000000UL, // 21
	0x0000020000000000ULL, 0x0000000000000000UL, // 22
	0x0000010000000000ULL, 0x0000000000000000UL, // 23
	0x0000008000000000ULL, 0x0000000000000000UL, // 24
	0x0000004000000000ULL, 0x0000000000000000UL, // 25
	0x0000002000000000ULL, 0x0000000000000000UL, // 26
	0x0000001000000000ULL, 0x0000000000000000UL, // 27
	0x0000000800000000ULL, 0x0000000000000000UL, // 28
	0x0000000400000000ULL, 0x0000000000000000UL, // 29
	0x0000000200000000ULL, 0x0000000000000000UL, // 30
	0x0000000100000000ULL, 0x0000000000000000UL, // 31
	0x0000000080000000ULL, 0x0000000000000000UL, // 32
	0x0000000040000000ULL, 0x0000000000000000UL, // 33
	0x0000000020000000ULL, 0x0000000000000000UL, // 34
	0x0000000010000000ULL, 0x0000000000000000UL, // 35
	0x0000000008000000ULL, 0x0000000000000000UL, // 36
	0x0000000004000000ULL, 0x0000000000000000UL, // 37
	0x0000000002000000ULL, 0x0000000000000000UL, // 38
	0x0000000001000000ULL, 0x0000000000000000UL, // 39
	0x0000000000800000ULL, 0x0000000000000000UL, // 40
	0x0000000000400000ULL, 0x0000000000000000UL, // 41
	0x0000000000200000ULL, 0x0000000000000000UL, // 42
	0x0000000000100000ULL, 0x0000000000000000UL, // 43
	0x0000000000080000ULL, 0x0000000000000000UL, // 44
	0x0000000000040000ULL, 0x0000000000000000UL, // 45
	0x0000000000020000ULL, 0x0000000000000000UL, // 46
	0x0000000000010000ULL, 0x0000000000000000UL, // 47
	0x0000000000008000ULL, 0x0000000000000000UL, // 48
	0x0000000000004000ULL, 0x0000000000000000UL, // 49
	0x0000000000002000ULL, 0x0000000000000000UL, // 50
	0x0000000000001000ULL, 0x0000000000000000UL, // 51
	0x0000000000000800ULL, 0x0000000000000000UL, // 52
	0x0000000000000400ULL, 0x0000000000000000UL, // 53
	0x0000000000000200ULL, 0x0000000000000000UL, // 54
	0x0000000000000100ULL, 0x0000000000000000UL, // 55
	0x0000000000000080ULL, 0x0000000000000000UL, // 56
	0x0000000000000040ULL, 0x0000000000000000UL, // 57
	0x0000000000000020ULL, 0x0000000000000000UL, // 58
	0x0000000000000010ULL, 0x0000000000000000UL, // 59
	0x0000000000000008ULL, 0x0000000000000000UL, // 60
	0x0000000000000004ULL, 0x0000000000000000UL, // 61
	0x0000000000000002ULL, 0x0000000000000000UL, // 62
	0x0000000000000001ULL, 0x0000000000000000UL, // 63
	0x0000000000000000ULL, 0x8000000000000000UL, // 64
	0x0000000000000000ULL, 0x4000000000000000UL, // 65
	0x0000000000000000ULL, 0x2000000000000000UL, // 66
	0x0000000000000000ULL, 0x1000000000000000UL, // 67
	0x0000000000000000ULL, 0x0800000000000000UL, // 68
	0x0000000000000000ULL, 0x0400000000000000UL, // 69
	0x0000000000000000ULL, 0x0200000000000000UL, // 70
	0x0000000000000000ULL, 0x0100000000000000UL, // 71
	0x0000000000000000ULL, 0x0080000000000000UL, // 72
	0x0000000000000000ULL, 0x0040000000000000UL, // 73
	0x0000000000000000ULL, 0x0020000000000000UL, // 74
	0x0000000000000000ULL, 0x0010000000000000UL, // 75
	0x0000000000000000ULL, 0x0008000000000000UL, // 76
	0x0000000000000000ULL, 0x0004000000000000UL, // 77
	0x0000000000000000ULL, 0x0002000000000000UL, // 78
	0x0000000000000000ULL, 0x0001000000000000UL, // 79
	0x0000000000000000ULL, 0x0000800000000000UL, // 80
	0x0000000000000000ULL, 0x0000400000000000UL, // 81
	0x0000000000000000ULL, 0x0000200000000000UL, // 82
	0x0000000000000000ULL, 0x0000100000000000UL, // 83
	0x0000000000000000ULL, 0x0000080000000000UL, // 84
	0x0000000000000000ULL, 0x0000040000000000UL, // 85
	0x0000000000000000ULL, 0x0000020000000000UL, // 86
	0x0000000000000000ULL, 0x0000010000000000UL, // 87
	0x0000000000000000ULL, 0x0000008000000000UL, // 88
	0x0000000000000000ULL, 0x0000004000000000UL, // 89
	0x0000000000000000ULL, 0x0000002000000000UL, // 90
	0x0000000000000000ULL, 0x0000001000000000UL, // 91
	0x0000000000000000ULL, 0x0000000800000000UL, // 92
	0x0000000000000000ULL, 0x0000000400000000UL, // 93
	0x0000000000000000ULL, 0x0000000200000000UL, // 94
	0x0000000000000000ULL, 0x0000000100000000UL, // 95
	0x0000000000000000ULL, 0x0000000080000000UL, // 96
	0x0000000000000000ULL, 0x0000000040000000UL, // 97
	0x0000000000000000ULL, 0x0000000020000000UL, // 98
	0x0000000000000000ULL, 0x0000000010000000UL, // 99
	0x0000000000000000ULL, 0x0000000008000000UL, // 100
	0x0000000000000000ULL, 0x0000000004000000UL, // 101
	0x0000000000000000ULL, 0x0000000002000000UL, // 102
	0x0000000000000000ULL, 0x0000000001000000UL, // 103
	0x0000000000000000ULL, 0x0000000000800000UL, // 104
	0x0000000000000000ULL, 0x0000000000400000UL, // 105
	0x0000000000000000ULL, 0x0000000000200000UL, // 106
	0x0000000000000000ULL, 0x0000000000100000UL, // 107
	0x0000000000000000ULL, 0x0000000000080000UL, // 108
	0x0000000000000000ULL, 0x0000000000040000UL, // 109
	0x0000000000000000ULL, 0x0000000000020000UL, // 110
	0x0000000000000000ULL, 0x0000000000010000UL, // 111
	0x0000000000000000ULL, 0x0000000000008000UL, // 112
	0x0000000000000000ULL, 0x0000000000004000UL, // 113
	0x0000000000000000ULL, 0x0000000000002000UL, // 114
	0x0000000000000000ULL, 0x0000000000001000UL, // 115
	0x0000000000000000ULL, 0x0000000000000800UL, // 116
	0x0000000000000000ULL, 0x0000000000000400UL, // 117
	0x0000000000000000ULL, 0x0000000000000200UL, // 118
	0x0000000000000000ULL, 0x0000000000000100UL, // 119
	0x0000000000000000ULL, 0x0000000000000080UL, // 120
	0x0000000000000000ULL, 0x0000000000000040UL, // 121
	0x0000000000000000ULL, 0x0000000000000020UL, // 122
	0x0000000000000000ULL, 0x0000000000000010UL, // 123
	0x0000000000000000ULL, 0x0000000000000008UL, // 124
	0x0000000000000000ULL, 0x0000000000000004UL, // 125
	0x0000000000000000ULL, 0x0000000000000002UL, // 126
	0x0000000000000000ULL, 0x0000000000000001ULL // 127
};

// 简单的CUDA核函数
__global__ void warmupKernel(int *data, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		data[idx] += 1;
	}
}

__device__ void PRG_cuda(uint32_t *key, uint128_t input, uint128_t &output1, uint128_t &output2, int &bit1, int &bit2)
{
	input = input.set_lsb_zero();

	uint128_t stash[2];
	stash[0] = input;
	stash[1] = input.reverse_lsb();

	AES_encrypt_cu(stash[0].get_bytes(), stash[0].get_bytes(), key);
	AES_encrypt_cu(stash[1].get_bytes(), stash[1].get_bytes(), key);

	stash[0] = stash[0] ^ input;
	stash[1] = stash[1] ^ input;
	stash[1] = stash[1].reverse_lsb();

	bit1 = stash[0].get_lsb();
	bit2 = stash[1].get_lsb();

	output1 = stash[0].set_lsb_zero();
	output2 = stash[1].set_lsb_zero();
}

__device__ void fss_gen_device(AES_Generator_device *prg, uint32_t *key, uint128_t alpha, int n, DCF_Keys k0, DCF_Keys k1)
{
	// int maxlayer = n - 7;
	int maxlayer = n;
	const int MAX_LAYER = 64;

	uint128_t s[MAX_LAYER + 1][2];
	int t[MAX_LAYER + 1][2];
	uint128_t sCW[MAX_LAYER];
	int tCW[MAX_LAYER][2];

	s[0][0] = prg->random();
	s[0][1] = prg->random();
	t[0][0] = s[0][0].get_lsb();
	t[0][1] = t[0][0] ^ 1;
	s[0][0] = s[0][0].set_lsb_zero();
	s[0][1] = s[0][1].set_lsb_zero();

	int i;
	uint128_t s0[2], s1[2]; // 0=L,1=R
#define LEFT 0
#define RIGHT 1
	int t0[2], t1[2];
	for (i = 1; i <= maxlayer; i++)
	{
		PRG_cuda(key, s[i - 1][0], s0[LEFT], s0[RIGHT], t0[LEFT], t0[RIGHT]);
		PRG_cuda(key, s[i - 1][1], s1[LEFT], s1[RIGHT], t1[LEFT], t1[RIGHT]);

		int keep, lose;
		// int alphabit = getbit(alpha, n, i);
		int alphabit = alpha.get_bit(n - i);
		if (alphabit == 0)
		{
			keep = LEFT;
			lose = RIGHT;
		}
		else
		{
			keep = RIGHT;
			lose = LEFT;
		}

		sCW[i - 1] = s0[lose] ^ s1[lose];

		tCW[i - 1][LEFT] = t0[LEFT] ^ t1[LEFT] ^ alphabit ^ 1;
		tCW[i - 1][RIGHT] = t0[RIGHT] ^ t1[RIGHT] ^ alphabit;

		if (t[i - 1][0] == 1)
		{
			s[i][0] = s0[keep] ^ sCW[i - 1];
			t[i][0] = t0[keep] ^ tCW[i - 1][keep];
		}
		else
		{
			s[i][0] = s0[keep];
			t[i][0] = t0[keep];
		}

		if (t[i - 1][1] == 1)
		{
			s[i][1] = s1[keep] ^ sCW[i - 1];
			t[i][1] = t1[keep] ^ tCW[i - 1][keep];
		}
		else
		{
			s[i][1] = s1[keep];
			t[i][1] = t1[keep];
		}
	}

	uint128_t finalblock(0, 1);
	finalblock = finalblock ^ s[maxlayer][0];
	finalblock = finalblock ^ s[maxlayer][1];
	// finalblock.print_uint128("finalblock = ", finalblock);

	// unsigned char *buff0;
	// unsigned char *buff1;
	// buff0 = (unsigned char*) malloc(1 + 16 + 1 + 18 * maxlayer + 16);
	// buff1 = (unsigned char*) malloc(1 + 16 + 1 + 18 * maxlayer + 16);

	// if(buff0 == NULL || buff1 == NULL){
	// 	printf("Memory allocation failed\n");
	// 	return;
	// }

	k0[0] = n;
	memcpy(&k0[1], &s[0][0], 16);
	k0[17] = t[0][0];
	for (i = 1; i <= maxlayer; i++)
	{
		memcpy(&k0[18 * i], &sCW[i - 1], 16);
		k0[18 * i + 16] = tCW[i - 1][0];
		k0[18 * i + 17] = tCW[i - 1][1];
	}
	memcpy(&k0[18 * maxlayer + 18], &finalblock, 16);

	k1[0] = n;
	memcpy(&k1[18], &k0[18], 18 * (maxlayer));
	memcpy(&k1[1], &s[0][1], 16);
	k1[17] = t[0][1];
	memcpy(&k1[18 * maxlayer + 18], &finalblock, 16);

	// memcpy(k0, buff0, 1 + 16 + 1 + 18 * maxlayer + 16);
	// memcpy(k1, buff1, 1 + 16 + 1 + 18 * maxlayer + 16);
	// free(buff0);
	// free(buff1);
}

__device__ void dpf_gen_device(AES_Generator_device *prg, uint32_t *key, uint128_t alpha, int n, uint8_t *k0, uint8_t *k1)
{
	int maxlayer = n - 7;
	// int maxlayer = n;
	const int MAX_LAYER = 64;

	uint128_t s[MAX_LAYER + 1][2];
	int t[MAX_LAYER + 1][2];
	uint128_t sCW[MAX_LAYER];
	int tCW[MAX_LAYER][2];

	s[0][0] = prg->random();
	s[0][1] = prg->random();
	t[0][0] = s[0][0].get_lsb();
	t[0][1] = t[0][0] ^ 1;
	s[0][0] = s[0][0].set_lsb_zero();
	s[0][1] = s[0][1].set_lsb_zero();

	int i;
	uint128_t s0[2], s1[2]; // 0=L,1=R
#define LEFT 0
#define RIGHT 1
	int t0[2], t1[2];
	for (i = 1; i <= maxlayer; i++)
	{
		PRG_cuda(key, s[i - 1][0], s0[LEFT], s0[RIGHT], t0[LEFT], t0[RIGHT]);
		PRG_cuda(key, s[i - 1][1], s1[LEFT], s1[RIGHT], t1[LEFT], t1[RIGHT]);

		int keep, lose;
		// int alphabit = getbit(alpha, n, i);
		int alphabit = alpha.get_bit(n - i);
		if (alphabit == 0)
		{
			keep = LEFT;
			lose = RIGHT;
		}
		else
		{
			keep = RIGHT;
			lose = LEFT;
		}

		sCW[i - 1] = s0[lose] ^ s1[lose];

		tCW[i - 1][LEFT] = t0[LEFT] ^ t1[LEFT] ^ alphabit ^ 1;
		tCW[i - 1][RIGHT] = t0[RIGHT] ^ t1[RIGHT] ^ alphabit;

		if (t[i - 1][0] == 1)
		{
			s[i][0] = s0[keep] ^ sCW[i - 1];
			t[i][0] = t0[keep] ^ tCW[i - 1][keep];
		}
		else
		{
			s[i][0] = s0[keep];
			t[i][0] = t0[keep];
		}

		if (t[i - 1][1] == 1)
		{
			s[i][1] = s1[keep] ^ sCW[i - 1];
			t[i][1] = t1[keep] ^ tCW[i - 1][keep];
		}
		else
		{
			s[i][1] = s1[keep];
			t[i][1] = t1[keep];
		}
	}

	uint128_t finalblock;
	finalblock = uint128_t(select_vector_table[(alpha.get_low() & 127) * 2], select_vector_table[(alpha.get_low() & 127) * 2 + 1]);
	finalblock = finalblock ^ s[maxlayer][0];
	finalblock = finalblock ^ s[maxlayer][1];

	// finalblock.print_uint128("finalblock = ", finalblock);

	// unsigned char *buff0;
	// unsigned char *buff1;
	// buff0 = (unsigned char *)malloc(1 + 16 + 1 + 18 * maxlayer + 16);
	// buff1 = (unsigned char *)malloc(1 + 16 + 1 + 18 * maxlayer + 16);

	// if (buff0 == NULL || buff1 == NULL)
	// {
	// 	printf("Memory allocation failed\n");
	// 	return;
	// }

	k0[0] = n;
	memcpy(&k0[1], &s[0][0], 16);
	k0[17] = t[0][0];
	for (i = 1; i <= maxlayer; i++)
	{
		memcpy(&k0[18 * i], &sCW[i - 1], 16);
		k0[18 * i + 16] = tCW[i - 1][0];
		k0[18 * i + 17] = tCW[i - 1][1];
	}
	memcpy(&k0[18 * maxlayer + 18], &finalblock, 16);

	k1[0] = n;
	memcpy(&k1[18], &k0[18], 18 * (maxlayer));
	memcpy(&k1[1], &s[0][1], 16);
	k1[17] = t[0][1];
	memcpy(&k1[18 * maxlayer + 18], &finalblock, 16);

	// memcpy(k0, buff0, 1 + 16 + 1 + 18 * maxlayer + 16);
	// memcpy(k1, buff1, 1 + 16 + 1 + 18 * maxlayer + 16);
	// free(buff0);
	// free(buff1);
}

__device__ uint128_t dcf_eval_device(uint32_t *key, DCF_Keys k, uint128_t x)
{
	int n = k[0];
	int maxlayer = n;
	const int MAX_LAYER = 64;

	uint128_t s[MAX_LAYER + 1];
	int t[MAX_LAYER + 1];
	uint128_t sCW[MAX_LAYER];
	int tCW[MAX_LAYER][2];
	uint128_t finalblock;

	memcpy(&s[0], &k[1], 16);
	t[0] = k[17];

	int i;
	for (i = 1; i <= maxlayer; i++)
	{
		memcpy(&sCW[i - 1], &k[18 * i], 16);
		tCW[i - 1][0] = k[18 * i + 16];
		tCW[i - 1][1] = k[18 * i + 17];
	}

	memcpy(&finalblock, &k[18 * (maxlayer + 1)], 16);

	uint128_t sL, sR;
	uint128_t res(0, 0);
	int tL, tR;

	// first layer
	PRG_cuda(key, s[0], sL, sR, tL, tR);

	sL = sL ^ sCW[0].select(t[0]);
	sR = sR ^ sCW[0].select(t[0]);
	tL = tL ^ (tCW[0][0] * t[0]);
	tR = tR ^ (tCW[0][1] * t[0]);

	int xbit = x.get_bit(n - 1);

	s[1] = sR.select(xbit) ^ sL.select((1 - xbit));
	t[1] = tR * xbit + tL * (1 - xbit);

	res = res ^ uint128_t(0, xbit * t[0]);

	for (i = 2; i <= maxlayer; i++)
	{
		PRG_cuda(key, s[i - 1], sL, sR, tL, tR);

		sL = sL ^ sCW[i - 1].select(t[i - 1]);
		sR = sR ^ sCW[i - 1].select(t[i - 1]);
		tL = tL ^ (tCW[i - 1][0] * t[i - 1]);
		tR = tR ^ (tCW[i - 1][1] * t[i - 1]);

		int xbit = x.get_bit(n - i);
		s[i] = sR.select(xbit) ^ sL.select((1 - xbit));
		t[i] = tR * xbit + tL * (1 - xbit);

		int xbit_last = x.get_bit(n - i + 1);
		int changed = (xbit_last * (1 - xbit)) | ((1 - xbit_last) * xbit);
		res = res ^ uint128_t(0, changed * t[i - 1]);
	}
	xbit = 1 - x.get_bit(0);
	res = res ^ uint128_t(0, t[maxlayer] * xbit);
	return res;
}

__inline__ __device__ void
warpReduce(uint128_t *result)
{
#pragma unroll
	for (int offset = (warpSize >> 1); offset > 0; offset >>= 1)
	{
		for (int i = 0; i < entry_size; i++)
		{
			// printf("hello");
			uint64_t high, low;
			high = result[i].get_high() ^ __shfl_down_sync(0xFFFFFFFF, result[i].get_high(), offset);
			low = result[i].get_low() ^ __shfl_down_sync(0xFFFFFFFF, result[i].get_low(), offset);
			result[i] = uint128_t(high, low);
		}
	}
}

__inline__ __device__ uint32_t
warpReduce_LUT(uint32_t result)
{
#if __CUDA_ARCH__ >= 800
	result = __reduce_xor_sync(0xFFFFFFFF, result);
#else
#pragma unroll
	for (int offset = (warpSize >> 1); offset > 0; offset >>= 1)
	{
		result ^= __shfl_down_sync(0xFFFFFFFF, result, offset);
	}
#endif
	return result;
}
__global__ void fss_genaeskey_kernel(uint32_t key[4 * (14 + 1)])
{
	uint64_t userkey1 = 597349;
	uint64_t userkey2 = 121379;
	uint128_t userkey(userkey1, userkey2);

	if (AES_set_encrypt_key_cu(userkey.get_bytes(), 128, key) != 0)
	{
		printf("Key expansion failed!\n");
		return;
	}
}

__global__ void fss_gen_kernel(uint32_t key[4 * (14 + 1)], uint64_t *alpha, int n, DCF_Keys k0, DCF_Keys k1, int N, int maxlayer)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= N)
		return;
	uint32_t expanded_key[4 * (14 + 1)];
	memcpy(expanded_key, key, 4 * (14 + 1) * sizeof(uint32_t));
	AES_Generator_device prg;
	unsigned char *k0_local;
	unsigned char *k1_local;
	uint128_t alpha_tid(0, alpha[tid]);
	k0_local = (unsigned char *)(k0 + tid * (1 + 16 + 1 + 18 * maxlayer + 16));
	k1_local = (unsigned char *)(k1 + tid * (1 + 16 + 1 + 18 * maxlayer + 16));
	fss_gen_device(&prg, expanded_key, alpha_tid, n, k0_local, k1_local);
}

__global__ void dpf_gen_kernel(uint32_t key[4 * (14 + 1)], uint64_t *alpha, int n, DCF_Keys k0, DCF_Keys k1, int N, int maxlayer)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= N)
		return;
	uint32_t expanded_key[4 * (14 + 1)];
	memcpy(expanded_key, key, 4 * (14 + 1) * sizeof(uint32_t));
	AES_Generator_device prg;
	unsigned char *k0_local;
	unsigned char *k1_local;
	uint128_t alpha_tid(0, alpha[tid]);
	k0_local = (unsigned char *)(k0 + tid * (1 + 16 + 1 + 18 * maxlayer + 16));
	k1_local = (unsigned char *)(k1 + tid * (1 + 16 + 1 + 18 * maxlayer + 16));
	dpf_gen_device(&prg, expanded_key, alpha_tid, n, k0_local, k1_local);
}

__global__ void fss_eval_kernel(bool *res, uint32_t key[4 * (14 + 1)], uint64_t *alpha, int n, DCF_Keys k, int N, int maxlayer)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= N)
		return;
	uint32_t expanded_key[4 * (14 + 1)];
	memcpy(expanded_key, key, 4 * (14 + 1) * sizeof(uint32_t));
	unsigned char *k_local = (unsigned char *)(k + tid * (1 + 16 + 1 + 18 * maxlayer + 16));
	uint128_t alpha_tid(0, alpha[tid]);
	res[tid] = dcf_eval_device(expanded_key, k_local, alpha_tid).get_lsb();
}

__global__ void aes_test_kernel(int N, DCF_Keys k0, DCF_Keys k1)
{
	// 测试密钥 (16字节 = 128位)
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= N)
		return;

	uint64_t userkey1 = 597349;
	uint64_t userkey2 = 121379;
	uint128_t userkey(userkey1, userkey2);

	// 扩展密钥
	uint32_t expanded_key[4 * (14 + 1)]; // AES-128需要11组轮密钥
	if (AES_set_encrypt_key_cu(userkey.get_bytes(), 128, expanded_key) != 0)
	{
		printf("Key expansion failed!\n");
		return;
	}
	AES_Generator_device prg;
	uint64_t random = prg.random().get_low();
	uint64_t random2 = prg.random().get_low();
	uint128_t output1, output2;

	int maxlayer = 64;
	unsigned char *k0_local;
	unsigned char *k1_local;

	k0_local = (unsigned char *)(k0 + tid * (1 + 16 + 1 + 18 * maxlayer + 16));
	k1_local = (unsigned char *)(k1 + tid * (1 + 16 + 1 + 18 * maxlayer + 16));
	fss_gen_device(&prg, expanded_key, uint128_t(0, random), 64, k0_local, k1_local);

	output1 = dcf_eval_device(expanded_key, k0_local, uint128_t(0, random2));
	output2 = dcf_eval_device(expanded_key, k1_local, uint128_t(0, random2));
	uint128_t res = output1 ^ output2;
	printf("random < random2 = %s\n", (random < random2) == res.get_lsb() ? "success" : "failed");
}

__global__ void fss_msb_keygen_kernel(uint32_t key[4 * (14 + 1)], DCF_Keys k0, DCF_Keys k1, int64_t *random0, int64_t *random1, bool *r_msb0, bool *r_msb1, int N, int maxlayer)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= N)
		return;
	uint32_t expanded_key[4 * (14 + 1)];
	memcpy(expanded_key, key, 4 * (14 + 1) * sizeof(uint32_t));
	AES_Generator_device prg;
	unsigned char *k0_local;
	unsigned char *k1_local;
	k0_local = (unsigned char *)(k0 + tid * (1 + 16 + 1 + 18 * maxlayer + 16));
	k1_local = (unsigned char *)(k1 + tid * (1 + 16 + 1 + 18 * maxlayer + 16));
	uint64_t random0_local = prg.random().get_low();
	uint64_t random1_local = prg.random().get_low();
	uint64_t random_local = random0_local + random1_local;
	// printf("random_local = %lx\n", random_local);
	uint64_t r_prime = ((uint64_t)1 << 63);
	// printf("r_prime = %lx\n", r_prime);
	r_prime = r_prime - (random_local << 1 >> 1);
	// printf("r_prime = %lx\n", r_prime);
	uint128_t random_tid(0, r_prime);
	fss_gen_device(&prg, expanded_key, random_tid, 64, k0_local, k1_local);
	r_msb0[tid] = prg.random().get_lsb();
	r_msb1[tid] = (random_local >> 63) != r_msb0[tid];
	random0[tid] = (int64_t)random0_local;
	random1[tid] = (int64_t)random1_local;
}

__global__ void fss_msb_eval_kernel(bool *res, uint32_t key[4 * (14 + 1)], DCF_Keys k, int64_t *value, bool *r_msb, int N, int maxlayer, int select)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= N)
		return;
	uint32_t expanded_key[4 * (14 + 1)];
	memcpy(expanded_key, key, 4 * (14 + 1) * sizeof(uint32_t));
	unsigned char *k_local = (unsigned char *)(k + tid * (1 + 16 + 1 + 18 * maxlayer + 16));
	// printf("value[tid] = %lx\n", value[tid]);
	uint64_t value_tid = ((uint64_t)value[tid] << 1) >> 1;
	// printf("value_tid = %lx\n", value_tid);
	uint128_t value_tid_128(0, value_tid);
	bool res_local = dcf_eval_device(expanded_key, k_local, value_tid_128).get_lsb();
	// printf("res_local = %d\n", res_local);
	res_local = res_local != r_msb[tid];
	// printf("res_local = %d\n", res_local);
	bool value_msb = ((uint64_t)value[tid]) >> 63;
	// printf("value_msb = %d\n", value_msb);
	res_local = res_local != select * value_msb;
	// printf("res[tid] = %d\n", res_local);
	res[tid] = res_local;
}

__global__ void EvalAll_LBLEvaluation(uint32_t *key, int layer, int maxlayer, int N, unsigned char *k, uint128_t *s, uint32_t *t)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int stride = 1 << (layer - 1); // layer stride
	int batch_id = idx / stride;
	idx %= stride;
	uint128_t sCW;
	int tCW[2];
	unsigned char *k_local = (unsigned char *)(k + batch_id * (1 + 16 + 1 + 18 * maxlayer + 16));
	uint128_t *s_local = (uint128_t *)(s + batch_id * (1 << maxlayer) * 2);
	uint32_t *t_local = (uint32_t *)(t + batch_id * (1 << maxlayer) * 2);

	memcpy(&sCW, &k_local[18 * layer], 16);
	tCW[0] = k_local[18 * layer + 16];
	tCW[1] = k_local[18 * layer + 17];

	uint128_t sL, sR;
	int tL, tR;

	PRG_cuda(key, s_local[idx + stride - 1], sL, sR, tL, tR);
	// if(idx == 0){
	// 	s_local->print_uint128("s_local[idx + stride - 1] = ", s_local[idx + stride - 1]);
	// }

	if (t_local[idx + stride - 1] == 1)
	{
		sL = sL ^ sCW;
		sR = sR ^ sCW;
		tL = tL ^ tCW[0];
		tR = tR ^ tCW[1];
	}

	s_local[(idx + stride - 1) * 2 + 1] = sL;
	s_local[(idx + stride - 1) * 2 + 2] = sR;
	t_local[(idx + stride - 1) * 2 + 1] = tL;
	t_local[(idx + stride - 1) * 2 + 2] = tR;
}

__global__ void EvalAll_BlockEvaluation(uint32_t *key, int layer_len, int layer, unsigned char *k, uint128_t *s, uint32_t *t, uint128_t *res_s, uint32_t *res_t)
{

	__shared__ uint128_t shared_s[512 * 2];
	__shared__ uint8_t shared_t[512 * 2];
	__shared__ uint128_t shared_sCW[10];
	__shared__ uint8_t shared_tCW[10][2];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	auto block = cooperative_groups::this_thread_block();
	// if (tid >= (1 << layer_len))
	// 	return;

	int Stride = (1 << (layer_len + layer)) / 2; // layer stride
	int batch_id = idx / Stride;
	// if (idx == 0)
	// 	printf("batch_id = %d, Stride = %d, layer_len = %d, layer = %d\n", batch_id, Stride, layer_len, layer);

	idx %= Stride;
	int maxlayer = k[0] - 7;
	unsigned char *k_local = (unsigned char *)(k + batch_id * (1 + 16 + 1 + 18 * maxlayer + 16));
	uint128_t *s_local = (uint128_t *)(s + batch_id * (1 << layer));
	uint32_t *t_local = (uint32_t *)(t + batch_id * (1 << layer));
	uint128_t *res_s_local = (uint128_t *)(res_s + batch_id * (1 << (layer_len + layer)));
	uint32_t *res_t_local = (uint32_t *)(res_t + batch_id * (1 << (layer_len + layer)));

	if (tid == 0)
	{
		// cooperative_groups::memcpy_async(block, shared_s, s_local + blockIdx.x % (1 << layer), sizeof(uint128_t));
		// cooperative_groups::memcpy_async(block, shared_t, (uint8_t *)(t_local + blockIdx.x % (1 << layer)), sizeof(uint8_t));
		shared_s[0] = s_local[blockIdx.x % (1 << layer)];
		shared_t[0] = t_local[blockIdx.x % (1 << layer)];
	}
	if (tid < layer_len)
	{

		memcpy(&shared_sCW[tid], &k_local[18 * (tid + 1 + layer)], 16);
		shared_tCW[tid][0] = k_local[18 * (tid + 1 + layer) + 16];
		shared_tCW[tid][1] = k_local[18 * (tid + 1 + layer) + 17];
	}
	// for (int batch = 0; batch < layer_len; batch++)
	// {
	// 	cooperative_groups::memcpy_async(block, (unsigned char *)&shared_sCW[batch], &k_local[18 * (batch + 1 + layer)], 16);
	// 	// cooperative_groups::memcpy_async(block, (unsigned char *)&shared_tCW[batch][0], &k_local[18 * (batch + 1 + layer) + 16], 1);
	// 	// cooperative_groups::memcpy_async(block, (unsigned char *)&shared_tCW[batch][1], &k_local[18 * (batch + 1 + layer) + 17], 1);

	// 	block.sync();
	// }

	__syncthreads();

	uint128_t sCW;
	int tCW[2];

	for (int i = 0; i < layer_len; i++)
	{
		int stride = 1 << (i); // layer stride
		if (tid < stride)
		{
			// memcpy(&sCW, &k[18 * (i + 1 + layer)], 16);
			// tCW[0] = k[18 * (i + 1 + layer) + 16];
			// tCW[1] = k[18 * (i + 1 + layer) + 17];
			sCW = shared_sCW[i];
			tCW[0] = shared_tCW[i][0];
			tCW[1] = shared_tCW[i][1];

			uint128_t sL, sR;
			int tL, tR;
			// shared_s[tid + stride - 1].print_uint128("s", shared_s[tid + stride - 1]);

			PRG_cuda(key, shared_s[tid + stride - 1], sL, sR, tL, tR);
			// sL.print_uint128("sL:", sL);
			// sR.print_uint128("sR:", sR);

			if (shared_t[tid + stride - 1] == 1)
			{
				sL = sL ^ sCW;
				sR = sR ^ sCW;
				tL = tL ^ tCW[0];
				tR = tR ^ tCW[1];
			}

			if (i == layer_len - 1)
			{
				res_s_local[idx * 2] = sL;
				res_s_local[idx * 2 + 1] = sR;

				res_t_local[idx * 2] = tL;
				res_t_local[idx * 2 + 1] = tR;
			}
			else
			{
				shared_s[(tid + stride - 1) * 2 + 1] = sL;
				shared_s[(tid + stride - 1) * 2 + 2] = sR;

				shared_t[(tid + stride - 1) * 2 + 1] = tL;
				shared_t[(tid + stride - 1) * 2 + 2] = tR;
			}
		}
		__syncthreads();
	}

	// res_s_local[idx] = shared_s[(tid + (1 << layer_len) - 1)];
	// res_t_local[idx] = shared_t[tid + (1 << layer_len) - 1];
	// __syncthreads();
}

__global__ void EvalAll_BlockEvaluation_nobankconflict(uint32_t *key, int layer_len, int layer, unsigned char *k, uint128_t *s, uint32_t *t, uint128_t *res_s, uint32_t *res_t)
{

	// __shared__ uint128_t shared_s[512 * 2];
	__shared__ uint32_t shared_s[4 * 1024];
	__shared__ uint8_t shared_t[512 * 2];
	__shared__ uint128_t shared_sCW[10];
	__shared__ uint8_t shared_tCW[10][2];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	auto block = cooperative_groups::this_thread_block();
	const int wid = tid / warpSize;
	const int lid = tid & 31;
	if (tid >= (1 << layer_len))
		return;

	int Stride = (1 << (layer_len + layer)) / 2; // layer stride
	int batch_id = idx / Stride;
	// if (idx == 0)
	// 	printf("batch_id = %d, Stride = %d, layer_len = %d, layer = %d\n", batch_id, Stride, layer_len, layer);

	idx %= Stride;
	int maxlayer = k[0] - 7;
	unsigned char *k_local = (unsigned char *)(k + batch_id * (1 + 16 + 1 + 18 * maxlayer + 16));
	uint128_t *s_local = (uint128_t *)(s + batch_id * (1 << layer));
	uint32_t *t_local = (uint32_t *)(t + batch_id * (1 << layer));
	uint128_t *res_s_local = (uint128_t *)(res_s + batch_id * (1 << (layer_len + layer)));
	uint32_t *res_t_local = (uint32_t *)(res_t + batch_id * (1 << (layer_len + layer)));

	if (tid == 0)
	{

		shared_s[0] = s_local[blockIdx.x % (1 << layer)].get_low();
		shared_s[32] = (s_local[blockIdx.x % (1 << layer)].get_low() >> 32);
		shared_s[64] = s_local[blockIdx.x % (1 << layer)].get_high();
		shared_s[96] = (s_local[blockIdx.x % (1 << layer)].get_high() >> 32);
		shared_t[0] = t_local[blockIdx.x % (1 << layer)];
	}
	if (tid < layer_len)
	{
		memcpy(&shared_sCW[tid], &k_local[18 * (tid + 1 + layer)], 16);
		shared_tCW[tid][0] = k_local[18 * (tid + 1 + layer) + 16];
		shared_tCW[tid][1] = k_local[18 * (tid + 1 + layer) + 17];
	}
	__syncthreads();

	uint128_t sCW;
	int tCW[2];

	for (int i = 0; i < layer_len; i++)
	{
		int stride = 1 << (i); // layer stride
		int cnt = stride >> 4;
		if (tid < stride)
		{
			sCW = shared_sCW[i];
			tCW[0] = shared_tCW[i][0];
			tCW[1] = shared_tCW[i][1];

			uint128_t sL, sR;
			uint128_t ss;
			if (stride < 32)
				ss = uint128_t(uint64_t(shared_s[64 + tid + stride - 1]) | (uint64_t(shared_s[96 + tid + stride - 1]) << 32),
							   uint64_t(shared_s[tid + stride - 1]) | (uint64_t(shared_s[32 + tid + stride - 1]) << 32));
			else if (stride == 32)
				ss = uint128_t(uint64_t(shared_s[4 * 32 + 64 + tid]) | (uint64_t(shared_s[4 * 32 + 96 + tid]) << 32),
							   uint64_t(shared_s[4 * 32 + tid]) | (uint64_t(shared_s[4 * 32 + 32 + tid]) << 32));
			else
				ss = uint128_t(uint64_t(shared_s[cnt * 2 * 32 + 64 + wid * 128 + lid]) | (uint64_t(shared_s[cnt * 2 * 32 + 96 + wid * 128 + lid]) << 32),
							   uint64_t(shared_s[cnt * 2 * 32 + wid * 128 + lid]) | (uint64_t(shared_s[cnt * 2 * 32 + 32 + wid * 128 + lid]) << 32));

			int tL, tR;

			PRG_cuda(key, ss, sL, sR, tL, tR);

			if (shared_t[tid + stride - 1] == 1)
			{
				sL = sL ^ sCW;
				sR = sR ^ sCW;
				tL = tL ^ tCW[0];
				tR = tR ^ tCW[1];
			}

			if (i == layer_len - 1)
			{
				res_s_local[idx * 2] = sL;
				res_s_local[idx * 2 + 1] = sR;

				res_t_local[idx * 2] = tL;
				res_t_local[idx * 2 + 1] = tR;
			}
			else
			{
				if (stride < 16)
				{
					shared_s[(tid + stride - 1) * 2 + 1] = sL.get_low();
					shared_s[32 + (tid + stride - 1) * 2 + 1] = sL.get_low() >> 32;
					shared_s[64 + (tid + stride - 1) * 2 + 1] = sL.get_high();
					shared_s[96 + (tid + stride - 1) * 2 + 1] = (sL.get_high() >> 32);
					shared_s[(tid + stride - 1) * 2 + 2] = sR.get_low();
					shared_s[32 + (tid + stride - 1) * 2 + 2] = sR.get_low() >> 32;
					shared_s[64 + (tid + stride - 1) * 2 + 2] = sR.get_high();
					shared_s[96 + (tid + stride - 1) * 2 + 2] = (sR.get_high() >> 32);
					shared_t[(tid + stride - 1) * 2 + 1] = tL;
					shared_t[(tid + stride - 1) * 2 + 2] = tR;
				}
				else if (stride == 16)
				{
					shared_s[cnt * 4 * 32 + tid * 2] = sL.get_low();
					shared_s[32 + cnt * 4 * 32 + tid * 2] = sL.get_low() >> 32;
					shared_s[64 + cnt * 4 * 32 + tid * 2] = sL.get_high();
					shared_s[96 + cnt * 4 * 32 + tid * 2] = (sL.get_high() >> 32);
					shared_s[cnt * 4 * 32 + tid * 2 + 1] = sR.get_low();
					shared_s[32 + cnt * 4 * 32 + tid * 2 + 1] = sR.get_low() >> 32;
					shared_s[64 + cnt * 4 * 32 + tid * 2 + 1] = sR.get_high();
					shared_s[96 + cnt * 4 * 32 + tid * 2 + 1] = (sR.get_high() >> 32);
					shared_t[(tid + stride - 1) * 2 + 1] = tL;
					shared_t[(tid + stride - 1) * 2 + 2] = tR;
				}
				else
				{
					shared_s[cnt * 4 * 32 + wid * 256 + lid] = sL.get_low();
					shared_s[32 + cnt * 4 * 32 + wid * 256 + lid] = sL.get_low() >> 32;
					shared_s[64 + cnt * 4 * 32 + wid * 256 + lid] = sL.get_high();
					shared_s[96 + cnt * 4 * 32 + wid * 256 + lid] = (sL.get_high() >> 32);
					shared_s[cnt * 4 * 32 + wid * 256 + lid + 128] = sR.get_low();
					shared_s[32 + cnt * 4 * 32 + wid * 256 + lid + 128] = sR.get_low() >> 32;
					shared_s[64 + cnt * 4 * 32 + wid * 256 + lid + 128] = sR.get_high();
					shared_s[96 + cnt * 4 * 32 + wid * 256 + lid + 128] = (sR.get_high() >> 32);
					shared_t[(tid + stride - 1) * 2 + 1] = tL;
					shared_t[(tid + stride - 1) * 2 + 2] = tR;
				}
			}
		}
		__syncthreads();
	}
}

__global__ void EvalAll_lastlayer(unsigned char *k, uint128_t *s, uint32_t *t, uint128_t *res)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= (1 << (k[0] - 7)))
		return;
	uint128_t finalblock;
	memcpy(&finalblock, &k[18 * (k[0] - 7 + 1)], 16);

	uint128_t temp = s[idx + (1 << (k[0] - 7)) - 1];
	res[idx] = temp ^ finalblock.select(t[idx + (1 << (k[0] - 7)) - 1]);
	// res[idx].print_uint128("res[idx]:", res[idx]);
}

__global__ void MultiplicationReduction(unsigned char *k, uint128_t *se, const uint128_t *d_db, uint128_t *blocksum_cuda, size_t block_num)
{
	__shared__ uint128_t sdata[8 * entry_size];
	__shared__ uint128_t s_se[2];

	uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	const int tid = threadIdx.x;
	const int lid = tid & 31;
	const int wid = tid >> 5;
	int n = k[0];

	int stride = (1 << (n)) >> ((int)log2f((float)block_num)); // layer stride
	int batch_id = idx >> ((int)log2f((float)stride));
	idx = idx & (stride - 1);

	uint128_t *se_local = (uint128_t *)(se + batch_id * (1 << (n - 7)));

	if (tid == 0 || tid == blockDim.x >> 1)
	{
		s_se[tid >> 7] = se_local[idx >> 7];
	}
	__syncthreads();

	uint128_t temp_0[entry_size];

	if (s_se[tid >> 7].get_bit(127 - idx & 127) != 0)
	{
		for (int j = 0; j < entry_size; j++)
		{
			// temp_0[j] = d_db[idx + j * stride];
			temp_0[j] = d_db[idx * entry_size + j];
		}
		// temp_0[j] = uint128_t(0, 0);
	}
	else
	{
		for (int j = 0; j < entry_size; j++)
			temp_0[j] = uint128_t(0, 0);
	}
	// for (int j = 0; j < entry_size; j++)
	// {
	// 	temp_0[j] = uint128_t(d_db[idx + j * stride].get_high() * s_se[tid >> 7].get_bit(127 - (idx & 127)),
	// 						  d_db[idx + j * stride].get_low() * s_se[tid >> 7].get_bit(127 - (idx & 127)));
	// }
	__syncthreads();

	warpReduce(temp_0);

	if (lid == 0)
	{
		for (int j = 0; j < entry_size; j++)
		{
			sdata[8 * j + wid] = temp_0[j];
		}
	}
	__syncthreads();

	for (int j = 0; j < entry_size; j++)
	{
		temp_0[j] = (tid < (blockDim.x >> 5)) ? sdata[8 * j + lid] : uint128_t(0, 0);
	}

	if (wid == 0)
		warpReduce(temp_0);

	// write res to global mem
	if (tid == 0)
	{
		for (int i = 0; i < entry_size; i++)
		{
			blocksum_cuda[(blockIdx.x) * block_num * entry_size + i] = temp_0[i];
		}
	}
}

__global__ void MultiplicationReduction_col(unsigned char *k, uint128_t *se, const uint128_t *d_db, uint128_t *blocksum_cuda, size_t block_num)
{
	__shared__ uint128_t sdata[8 * entry_size];
	__shared__ uint128_t s_se[2];

	uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	const int tid = threadIdx.x;
	const int lid = tid & 31;
	const int wid = tid >> 5;
	int n = k[0];

	int stride = (1 << (n)) >> ((int)log2f((float)block_num)); // layer stride
	int batch_id = idx >> ((int)log2f((float)stride));
	idx = idx & (stride - 1);

	uint128_t *se_local = (uint128_t *)(se + batch_id * (1 << (n - 7)));

	if (tid == 0 || tid == blockDim.x >> 1)
	{
		s_se[tid >> 7] = se_local[idx >> 7];
	}
	__syncthreads();

	uint128_t temp_0[entry_size];

	for (int j = 0; j < entry_size; j++)
	{
		uint128_t db = d_db[idx + j * stride];
		temp_0[j] = uint128_t(db.get_high() * s_se[tid >> 7].get_bit(127 - (idx & 127)),
							  db.get_low() * s_se[tid >> 7].get_bit(127 - (idx & 127)));
	}
	__syncthreads();

	warpReduce(temp_0);

	if (lid == 0)
	{
		for (int j = 0; j < entry_size; j++)
		{
			sdata[8 * j + wid] = temp_0[j];
		}
	}
	__syncthreads();

	for (int j = 0; j < entry_size; j++)
	{
		temp_0[j] = (tid < (blockDim.x >> 5)) ? sdata[8 * j + lid] : uint128_t(0, 0);
	}

	if (wid == 0)
		warpReduce(temp_0);

	// write res to global mem
	if (tid == 0)
	{
		for (int i = 0; i < entry_size; i++)
		{
			blocksum_cuda[(blockIdx.x) * block_num * entry_size + i] = temp_0[i];
		}
	}
}

__global__ void MultiplicationReduction_grid_col(unsigned char *k, uint128_t *se, const uint128_t *d_db, uint128_t *blocksum_cuda, size_t block_num, int N)
{
	__shared__ uint128_t sdata[8 * entry_size];

	uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	const int tid = threadIdx.x;
	const int lid = tid & 31;
	const int wid = tid / warpSize;
	int n = k[0];

	const uint64_t totalThreads = gridDim.x * blockDim.x;
	const uint64_t threads_per_batch = totalThreads / N;

	int batch_id = idx / threads_per_batch;
	uint64_t local_idx = idx % threads_per_batch;

	uint128_t *se_local = se + batch_id * (1 << (n - 7));

	uint128_t temp_0[entry_size];
	for (uint64_t i = local_idx; i < (1ULL << n); i += threads_per_batch)
	{
		for (int j = 0; j < entry_size; j++)
		{
			uint128_t db = d_db[i + (1ULL << n) * j];
			temp_0[j] ^= uint128_t(db.get_high() * se_local[i >> 7].get_bit(127 - (i & 127)),
								   db.get_low() * se_local[i >> 7].get_bit(127 - (i & 127)));
		}
	}

	__syncthreads();

	// if (idx == 0)
	// 	for (int i = 0; i < 16; i++)
	// 		temp_0[i].print_uint128("", temp_0[i]);

	warpReduce(temp_0);

	if (lid == 0)
	{
		for (int j = 0; j < entry_size; j++)
		{
			sdata[8 * j + wid] = temp_0[j];
		}
	}
	__syncthreads();

	for (int j = 0; j < entry_size; j++)
	{
		temp_0[j] = (tid < (blockDim.x >> 5)) ? sdata[8 * j + lid] : uint128_t(0, 0);
	}

	if (wid == 0)
		warpReduce(temp_0);

	// write res to global mem
	if (tid == 0)
	{
		for (int i = 0; i < entry_size; i++)
		{
			blocksum_cuda[(blockIdx.x) * block_num * entry_size + i] = temp_0[i];
		}
	}
}

__global__ void MultiplicationReduction_grid_row(unsigned char *k, uint128_t *se, const uint128_t *d_db, uint128_t *blocksum_cuda, size_t block_num, int N)
{
	__shared__ uint128_t sdata[8 * entry_size];

	uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	const int tid = threadIdx.x;
	const int lid = tid & 31;
	const int wid = tid / warpSize;
	int n = k[0];

	const uint64_t totalThreads = gridDim.x * blockDim.x;
	const uint64_t threads_per_batch = totalThreads / N;

	int batch_id = idx / threads_per_batch;
	uint64_t local_idx = idx % threads_per_batch;

	uint128_t *se_local = se + batch_id * (1 << (n - 7));

	uint128_t temp_0[entry_size];
	for (uint64_t i = local_idx; i < (1ULL << n); i += threads_per_batch)
	{
		if (se_local[i >> 7].get_bit(127 - (i & 127)))
			for (int j = 0; j < entry_size; j++)
				temp_0[j] ^= d_db[i * entry_size + j];
	}

	__syncthreads();

	// if (idx == 0)
	// 	for (int i = 0; i < 16; i++)
	// 		temp_0[i].print_uint128("", temp_0[i]);

	warpReduce(temp_0);

	if (lid == 0)
	{
		for (int j = 0; j < entry_size; j++)
		{
			sdata[8 * j + wid] = temp_0[j];
		}
	}
	__syncthreads();

	for (int j = 0; j < entry_size; j++)
	{
		temp_0[j] = (tid < (blockDim.x >> 5)) ? sdata[8 * j + lid] : uint128_t(0, 0);
	}

	if (wid == 0)
		warpReduce(temp_0);

	// write res to global mem
	if (tid == 0)
	{
		for (int i = 0; i < entry_size; i++)
		{
			blocksum_cuda[(blockIdx.x) * block_num * entry_size + i] = temp_0[i];
		}
	}
}

__global__ void MultiplicationReduction_LUT_new(unsigned char *k, uint128_t *se, const uint32_t *d_db, uint32_t *blocksum_cuda, size_t block_num, int N)
{
	__shared__ uint32_t sdata[8];

	uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	const int tid = threadIdx.x;
	const int lid = tid & 31;
	const int wid = tid / warpSize;
	int n = k[0];

	const uint64_t totalThreads = gridDim.x * blockDim.x;
	const uint64_t threads_per_batch = totalThreads / N;

	int batch_id = idx / threads_per_batch;
	uint64_t local_idx = idx % threads_per_batch;

	uint128_t *se_local = se + batch_id * (1 << (n - 7));

	uint32_t temp_0 = 0;
	for (uint64_t i = local_idx; i < (1ULL << n); i += threads_per_batch)
	{
		// temp_0 ^= d_db[i] * se_local[i >> 7].get_bit(127 - (i & 127));
		if (se_local[i >> 7].get_bit(127 - i & 127) != 0)
		{
			temp_0 ^= d_db[i];
		}
	}

	__syncthreads();

	// if (idx == 0)
	// 	for (int i = 0; i < 16; i++)
	// 		temp_0[i].print_uint128("", temp_0[i]);

	temp_0 = warpReduce_LUT(temp_0);

	if (lid == 0)
	{
		sdata[wid] = temp_0;
	}
	__syncthreads();

	temp_0 = (tid < (blockDim.x >> 5)) ? sdata[lid] : 0;

	if (wid == 0)
		temp_0 = warpReduce_LUT(temp_0);

	// write res to global mem
	if (tid == 0)
	{
		blocksum_cuda[(blockIdx.x) * block_num] = temp_0;
		// atomicXor(blocksum_cuda + batch_id, temp_0);
	}
}

__global__ void MultiplicationReduction_LUT(unsigned char *k, uint128_t *se, const uint32_t *d_db, uint32_t *blocksum_cuda, size_t block_num)
{
	__shared__ uint32_t sdata[8];
	__shared__ uint128_t s_se[2];

	uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	const int tid = threadIdx.x;
	const int lid = tid & 31;
	const int wid = tid / warpSize;
	int n = k[0];

	uint64_t stride = (1ULL << (n)) >> ((int)log2f((float)block_num)); // layer stride
	int batch_id = idx >> (n - (int)log2f((float)block_num));

	idx = idx & (stride - 1);

	uint128_t *se_local = (uint128_t *)(se + batch_id * (1 << (n - 7)));

	// if (tid == 0 || tid == blockDim.x >> 1)
	// {
	// 	s_se[tid >> 7] = se_local[idx >> 7];
	// }
	// __syncthreads();

	uint32_t temp_0;

	// if (se_local[idx >> 7].get_bit(127 - idx & 127) != 0)
	// {
	// 	temp_0 = d_db[idx];
	// }
	// else
	// {
	// 	temp_0 = 0;
	// }

	temp_0 = d_db[idx] * se_local[idx >> 7].get_bit(127 - (idx & 127));

	// __syncthreads();
	// if (idx == 0)
	// 	for (int i = 0; i < 16; i++)
	// 		temp_0[i].print_uint128("", temp_0[i]);

	temp_0 = warpReduce_LUT(temp_0);

	if (lid == 0)
	{
		sdata[wid] = temp_0;
	}
	__syncthreads();

	temp_0 = (tid < (blockDim.x >> 5)) ? sdata[lid] : 0;

	if (wid == 0)
		temp_0 = warpReduce_LUT(temp_0);

	// write res to global mem
	if (tid == 0)
	{
		blocksum_cuda[(blockIdx.x) * block_num] = temp_0;
	}
}

__global__ void EVAL_Pack_last_layer_gense(unsigned char *k, uint128_t *s, uint32_t *t, uint128_t *se)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int n = k[0];
	int maxlayer = n - 7;

	int stride = 1 << (maxlayer); // layer stride
	int batch_id = idx / stride;

	idx %= stride;

	unsigned char *k_local = (unsigned char *)(k + batch_id * (1 + 16 + 1 + 18 * maxlayer + 16));
	uint128_t *s_local = (uint128_t *)(s + batch_id * (1 << maxlayer) * 2);
	uint32_t *t_local = (uint32_t *)(t + batch_id * (1 << maxlayer) * 2);
	uint128_t *se_local = (uint128_t *)(se + batch_id * (1 << maxlayer));

	// if (idx < (1 << maxlayer))
	// {
	uint128_t finalblock, res;
	memcpy(&finalblock, &k_local[18 * (maxlayer + 1)], 16);

	se_local[idx] = s_local[idx + (1 << maxlayer) - 1] ^ finalblock.select(t_local[idx + (1 << maxlayer) - 1]);
	// }
	__syncthreads();

	// if (idx == 0)
	// 	printf("%d: %d  ", i, se_local[i]);
}

__global__ void EvalAll_SeGeneration(unsigned char *k, uint128_t *s, uint32_t *t, uint128_t *se)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int n = k[0];
	int maxlayer = n - 7;

	int stride = 1 << (maxlayer); // layer stride
	int batch_id = idx / stride;

	idx %= stride;

	unsigned char *k_local = (unsigned char *)(k + batch_id * (1 + 16 + 1 + 18 * maxlayer + 16));
	uint128_t *s_local = (uint128_t *)(s + batch_id * (1 << maxlayer));
	uint32_t *t_local = (uint32_t *)(t + batch_id * (1 << maxlayer));
	uint128_t *se_local = (uint128_t *)(se + batch_id * (1 << maxlayer));

	// if (idx < (1 << maxlayer))
	// {
	uint128_t finalblock, res;
	memcpy(&finalblock, &k_local[18 * (maxlayer + 1)], 16);

	se_local[idx] = s_local[idx] ^ finalblock.select(t_local[idx]);
	// }
}

__global__ void BlockReduction(uint128_t *blocksum_cuda, uint128_t *k_res)
{
	__shared__ uint128_t sdata[entry_size * 8];

	const int tid = threadIdx.x;
	const int lid = tid & 31;
	const int wid = tid / warpSize;
	uint128_t temp[entry_size];

#pragma unroll
	for (int i = 0; i < entry_size; i++)
	{
		temp[i] = blocksum_cuda[(blockDim.x * blockIdx.x + tid) * entry_size + i];
	}

	// reduce in each warp
	warpReduce(temp);

	if (lid == 0)
	{
		for (int i = 0; i < entry_size; i++)
		{
			// sdata[(tid >> 5) * entry_size + i] = temp[i];
			sdata[8 * i + wid] = temp[i];
		}
	}
	__syncthreads();

	int warps = (blockDim.x > 32) ? (blockDim.x >> 5) : 1;
	for (int i = 0; i < entry_size; i++)
	{
		// temp[i] = (tid < warps) ? sdata[lid * entry_size + i] : uint128_t(0, 0);
		temp[i] = (tid < warps) ? sdata[8 * i + lid] : uint128_t(0, 0);
	}

	if (wid == 0)
		warpReduce(temp);

	if (tid == 0)
	{
		for (int i = 0; i < entry_size; i++)
		{
			k_res[blockIdx.x * entry_size + i] = temp[i];
		}
	}
}

__global__ void BlockReduction_LUT(uint32_t *blocksum_cuda, uint32_t *k_res)
{
	__shared__ uint32_t sdata[8];

	const int tid = threadIdx.x;
	const int lid = tid & 31;
	const int wid = tid / warpSize;
	uint32_t temp;

	temp = blocksum_cuda[blockDim.x * blockIdx.x + tid];

	// reduce in each warp
	temp = warpReduce_LUT(temp);

	if (lid == 0)
	{
		// sdata[(tid >> 5) * entry_size + i] = temp[i];
		sdata[wid] = temp;
	}
	__syncthreads();

	int warps = (blockDim.x > 32) ? (blockDim.x >> 5) : 1;
	// temp[i] = (tid < warps) ? sdata[lid * entry_size + i] : uint128_t(0, 0);
	temp = (tid < warps) ? sdata[lid] : 0;

	if (wid == 0)
		temp = warpReduce_LUT(temp);

	if (tid == 0)
	{
		k_res[blockIdx.x] = temp;
	}
}

__global__ void seed_copy(uint128_t *d_s, uint32_t *d_t, uint8_t *key, int maxlayer, int N)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < N)
	{
		memcpy(d_s + idx, &key[1 + idx * (1 + 16 + 1 + 18 * maxlayer + 16)], 16);
		memcpy(d_t + idx, &key[17 + idx * (1 + 16 + 1 + 18 * maxlayer + 16)], 1);
	}
}

// check GPU's mem
void check_mem_usage()
{
	size_t free_memory, total_memory;
	cudaError_t err = cudaMemGetInfo(&free_memory, &total_memory);

	if (err != cudaSuccess)
	{
		printf("Error: %s\n", cudaGetErrorString(err));
		return;
	}

	printf("Free memory: %zu bytes\n", free_memory);
	printf("Total memory: %zu bytes\n", total_memory);
	printf("Used memory: %zu bytes\n", total_memory - free_memory);
}

extern "C" void cudamsbkeygen(DCF_Keys k0, DCF_Keys k1, int64_t *random0, int64_t *random1, bool *r_msb0, bool *r_msb1, int N, int maxlayer)
{

	DCF_Keys k0_device;
	cudaMalloc(&k0_device, N * (1 + 16 + 1 + 18 * maxlayer + 16));

	DCF_Keys k1_device;
	cudaMalloc(&k1_device, N * (1 + 16 + 1 + 18 * maxlayer + 16));

	int64_t *random0_device;
	cudaMalloc(&random0_device, N * sizeof(int64_t));
	cudaMemcpy(random0_device, random0, N * sizeof(int64_t), cudaMemcpyHostToDevice);

	int64_t *random1_device;
	cudaMalloc(&random1_device, N * sizeof(int64_t));
	cudaMemcpy(random1_device, random1, N * sizeof(int64_t), cudaMemcpyHostToDevice);

	bool *r_msb0_device;
	cudaMalloc(&r_msb0_device, N * sizeof(bool));
	cudaMemcpy(r_msb0_device, r_msb0, N * sizeof(bool), cudaMemcpyHostToDevice);

	bool *r_msb1_device;
	cudaMalloc(&r_msb1_device, N * sizeof(bool));
	cudaMemcpy(r_msb1_device, r_msb1, N * sizeof(bool), cudaMemcpyHostToDevice);

	uint32_t *aes_key;
	cudaMalloc(&aes_key, 4 * (14 + 1) * sizeof(uint32_t));
	fss_genaeskey_kernel<<<1, 1>>>(aes_key);

	int threads = N > 256 ? 256 : N;
	int blocks = (N + threads - 1) / threads;
	fss_msb_keygen_kernel<<<blocks, threads>>>(aes_key, k0_device, k1_device, random0_device, random1_device, r_msb0_device, r_msb1_device, N, maxlayer);

	cudaMemcpy(k0, k0_device, N * (1 + 16 + 1 + 18 * maxlayer + 16), cudaMemcpyDeviceToHost);
	cudaMemcpy(k1, k1_device, N * (1 + 16 + 1 + 18 * maxlayer + 16), cudaMemcpyDeviceToHost);
	cudaMemcpy(random0, random0_device, N * sizeof(int64_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(random1, random1_device, N * sizeof(int64_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(r_msb0, r_msb0_device, N * sizeof(bool), cudaMemcpyDeviceToHost);
	cudaMemcpy(r_msb1, r_msb1_device, N * sizeof(bool), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
	cudaFree(k0_device);
	cudaFree(k1_device);
	cudaFree(random0_device);
	cudaFree(random1_device);
	cudaFree(r_msb0_device);
	cudaFree(r_msb1_device);
	cudaFree(aes_key);
}

extern "C" void cudamsbeval(bool *res, DCF_Keys k, int64_t *value, bool *r_msb, int N, int maxlayer, int party)
{

	// move MSB_keys to device
	DCF_Keys k_device;
	cudaMalloc(&k_device, N * (1 + 16 + 1 + 18 * maxlayer + 16));
	cudaMemcpy(k_device, k, N * (1 + 16 + 1 + 18 * maxlayer + 16), cudaMemcpyHostToDevice);

	int64_t *value_device;
	cudaMalloc(&value_device, N * sizeof(int64_t));
	cudaMemcpy(value_device, value, N * sizeof(int64_t), cudaMemcpyHostToDevice);

	bool *r_msb_device;
	cudaMalloc(&r_msb_device, N * sizeof(bool));
	cudaMemcpy(r_msb_device, r_msb, N * sizeof(bool), cudaMemcpyHostToDevice);

	bool *res_device;
	cudaMalloc(&res_device, N * sizeof(bool));

	uint32_t *aes_key;
	cudaMalloc(&aes_key, 4 * (14 + 1) * sizeof(uint32_t));
	fss_genaeskey_kernel<<<1, 1>>>(aes_key);

	int threads = N > 512 ? 512 : N;
	int blocks = (N + threads - 1) / threads;
	int select = party == 1 ? 1 : 0;

	// cudaEvent_t start1, stop1;
	// cudaEventCreate(&start1);
	// cudaEventCreate(&stop1);

	// cudaEventRecord(start1);
	fss_msb_eval_kernel<<<blocks, threads>>>(res_device, aes_key, k_device, value_device, r_msb_device, N, maxlayer, select);
	// cudaEventRecord(stop1);

	// cudaEventSynchronize(stop1);
	// float time1 = 0;
	// cudaEventElapsedTime(&time1, start1, stop1);
	// printf("msb eval Kernel Time taken: %.3f ms\n", time1);

	cudaMemcpy(res, res_device, N * sizeof(bool), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
	cudaFree(k_device);
	cudaFree(value_device);
	cudaFree(r_msb_device);
	cudaFree(res_device);
	cudaFree(aes_key);
}

extern "C" void cudafsskeygen(DCF_Keys k0, DCF_Keys k1, uint64_t *alpha, int N, int n, int maxlayer)
{

	DCF_Keys k0_device;
	cudaMalloc(&k0_device, N * (1 + 16 + 1 + 18 * maxlayer + 16));

	DCF_Keys k1_device;
	cudaMalloc(&k1_device, N * (1 + 16 + 1 + 18 * maxlayer + 16));

	uint64_t *alpha_device;
	cudaMalloc(&alpha_device, N * sizeof(uint64_t));
	cudaMemcpy(alpha_device, alpha, N * sizeof(uint64_t), cudaMemcpyHostToDevice);

	uint32_t *aes_key;
	cudaMalloc(&aes_key, 4 * (14 + 1) * sizeof(uint32_t));
	fss_genaeskey_kernel<<<1, 1>>>(aes_key);

	int threads = N > 256 ? 256 : N;
	int blocks = (N + threads - 1) / threads;
	fss_gen_kernel<<<blocks, threads>>>(aes_key, alpha_device, n, k0_device, k1_device, N, maxlayer);

	cudaMemcpy(k0, k0_device, N * (1 + 16 + 1 + 18 * maxlayer + 16), cudaMemcpyDeviceToHost);
	cudaMemcpy(k1, k1_device, N * (1 + 16 + 1 + 18 * maxlayer + 16), cudaMemcpyDeviceToHost);

	cudaFree(k0_device);
	cudaFree(k1_device);
	cudaFree(alpha_device);
	cudaFree(aes_key);
}

extern "C" void cudaDPFkeygen(uint8_t *k0, uint8_t *k1, uint64_t *alpha, int n, int maxlayer, int batch_size)
{

	uint8_t *k0_device;
	cudaMalloc(&k0_device, batch_size * (1 + 16 + 1 + 18 * maxlayer + 16));

	uint8_t *k1_device;
	cudaMalloc(&k1_device, batch_size * (1 + 16 + 1 + 18 * maxlayer + 16));

	uint64_t *alpha_device;
	cudaMalloc(&alpha_device, batch_size * sizeof(uint64_t));
	cudaMemcpy(alpha_device, alpha, batch_size * sizeof(uint64_t), cudaMemcpyHostToDevice);

	uint32_t *aes_key;
	cudaMalloc(&aes_key, 4 * (14 + 1) * sizeof(uint32_t));
	fss_genaeskey_kernel<<<1, 1>>>(aes_key);

	int threads = batch_size > 256 ? 256 : batch_size;
	int blocks = (batch_size + threads - 1) / threads;
	dpf_gen_kernel<<<blocks, threads>>>(aes_key, alpha_device, n, k0_device, k1_device, batch_size, maxlayer);

	cudaMemcpy(k0, k0_device, batch_size * (1 + 16 + 1 + 18 * maxlayer + 16), cudaMemcpyDeviceToHost);
	cudaMemcpy(k1, k1_device, batch_size * (1 + 16 + 1 + 18 * maxlayer + 16), cudaMemcpyDeviceToHost);

	cudaFree(k0_device);
	cudaFree(k1_device);
	cudaFree(alpha_device);
	cudaFree(aes_key);
}

extern "C" void cudafsseval(bool *res, DCF_Keys key, uint64_t *value, int N, int maxlayer, int party)
{

	DCF_Keys key_device;
	cudaMalloc(&key_device, N * (1 + 16 + 1 + 18 * maxlayer + 16));
	cudaMemcpy(key_device, key, N * (1 + 16 + 1 + 18 * maxlayer + 16), cudaMemcpyHostToDevice);

	uint64_t *value_device;
	cudaMalloc(&value_device, N * sizeof(uint64_t));
	cudaMemcpy(value_device, value, N * sizeof(uint64_t), cudaMemcpyHostToDevice);

	bool *res_device;
	cudaMalloc(&res_device, N * sizeof(bool));

	uint32_t *aes_key;
	cudaMalloc(&aes_key, 4 * (14 + 1) * sizeof(uint32_t));
	fss_genaeskey_kernel<<<1, 1>>>(aes_key);

	int threads = N > 256 ? 256 : N;
	int blocks = (N + threads - 1) / threads;
	fss_eval_kernel<<<blocks, threads>>>(res_device, aes_key, value_device, 64, key_device, N, maxlayer);

	cudaMemcpy(res, res_device, N * sizeof(bool), cudaMemcpyDeviceToHost);

	cudaFree(key_device);
	cudaFree(value_device);
}

extern "C" int test_dcf()
{
	DCF_Keys k0;
	DCF_Keys k1;
	int maxlayer = 64;
	int N = 1000;
	cudaMalloc(&k0, N * (1 + 16 + 1 + 18 * maxlayer + 16));
	cudaMalloc(&k1, N * (1 + 16 + 1 + 18 * maxlayer + 16));

	bool *res1;
	cudaMalloc(&res1, N * sizeof(bool));
	bool *res2;
	cudaMalloc(&res2, N * sizeof(bool));

	bool *res1_host;
	cudaMallocHost(&res1_host, N * sizeof(bool));
	bool *res2_host;
	cudaMallocHost(&res2_host, N * sizeof(bool));

	// uint128_t alpha1 = uint128_t(0, 1);
	// uint128_t alpha2 = uint128_t(0, 2);

	uint64_t *alpha1_host;
	cudaMallocHost(&alpha1_host, N * sizeof(uint64_t));
	uint64_t *alpha2_host;
	cudaMallocHost(&alpha2_host, N * sizeof(uint64_t));
	for (int i = 0; i < N; i++)
	{
		alpha1_host[i] = i + 1;
		alpha2_host[i] = i + 3;
	}

	uint64_t *value1;
	cudaMalloc(&value1, N * sizeof(uint64_t));
	uint64_t *value2;
	cudaMalloc(&value2, N * sizeof(uint64_t));
	cudaMemcpy(value1, alpha1_host, N * sizeof(uint64_t), cudaMemcpyHostToDevice);
	cudaMemcpy(value2, alpha2_host, N * sizeof(uint64_t), cudaMemcpyHostToDevice);

	int threads = N > 256 ? 256 : N;
	int blocks = (N + threads - 1) / threads;
	uint32_t *aes_key;
	cudaMalloc(&aes_key, 4 * (14 + 1) * sizeof(uint32_t));
	fss_genaeskey_kernel<<<1, 1>>>(aes_key);

	cudaEvent_t start1, stop1, start2, stop2, start3, stop3;
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);
	cudaEventCreate(&start3);
	cudaEventCreate(&stop3);

	cudaEventRecord(start1);
	fss_gen_kernel<<<blocks, threads>>>(aes_key, value1, 64, k0, k1, N, maxlayer);
	cudaEventRecord(stop1);

	cudaEventRecord(start2);
	fss_eval_kernel<<<blocks, threads>>>(res1, aes_key, value2, 64, k0, N, maxlayer);
	cudaMemcpy(res1_host, res1, N * sizeof(bool), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop2);

	cudaEventRecord(start3);
	fss_eval_kernel<<<blocks, threads>>>(res2, aes_key, value2, 64, k1, N, maxlayer);
	cudaMemcpy(res2_host, res2, N * sizeof(bool), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop3);

	cudaEventSynchronize(stop1);
	cudaEventSynchronize(stop2);
	cudaEventSynchronize(stop3);

	float time1 = 0, time2 = 0, time3 = 0;
	cudaEventElapsedTime(&time1, start1, stop1);
	cudaEventElapsedTime(&time2, start2, stop2);
	cudaEventElapsedTime(&time3, start3, stop3);

	printf("fss_gen_kernel time: %.3f ms\n", time1);
	printf("First fss_eval_kernel time: %.3f ms\n", time2);
	printf("Second fss_eval_kernel time: %.3f ms\n", time3);
	cudaEventDestroy(start1);
	cudaEventDestroy(stop1);
	cudaEventDestroy(start2);
	cudaEventDestroy(stop2);
	cudaEventDestroy(start3);
	cudaEventDestroy(stop3);

	for (int i = 0; i < N; i++)
	{
		bool res = res1_host[i] ^ res2_host[i];
		printf("res = %d\n", res);
	}

	cudaDeviceSynchronize();

	cudaFree(k0);
	cudaFree(k1);
	cudaFree(res1);
	cudaFree(res2);
	// cudaFree(alpha1_host);
	// cudaFree(alpha2_host);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		return -1;
	}

	return 0;
}

extern "C" uint128_t *test_dpf_LBL(uint8_t *key)
{
	int n = key[0];
	int maxlayer = n - 7;

	uint8_t *d_key;
	uint128_t *d_s;
	uint32_t *d_t;
	uint128_t *eval_res = (uint128_t *)malloc(sizeof(uint128_t) * (1 << maxlayer));
	uint128_t *d_eval_res;

	cudaMalloc(&d_key, (1 + 16 + 1 + 18 * maxlayer + 16) * sizeof(uint8_t));
	cudaMemcpy(d_key, key, (1 + 16 + 1 + 18 * maxlayer + 16) * sizeof(uint8_t), cudaMemcpyHostToDevice);
	cudaMalloc(&d_s, (1 << maxlayer) * 2 * sizeof(uint128_t));
	cudaMalloc(&d_t, (1 << maxlayer) * 2 * sizeof(uint32_t));
	cudaMallocManaged(&d_eval_res, (1 << maxlayer) * 2 * sizeof(uint128_t));

	cudaMemcpy(d_s, &key[1], 16, cudaMemcpyHostToDevice);
	cudaMemcpy(d_t, &key[17], 1, cudaMemcpyHostToDevice);

	uint32_t *aes_key;
	cudaMalloc(&aes_key, 4 * (14 + 1) * sizeof(uint32_t));
	fss_genaeskey_kernel<<<1, 1>>>(aes_key);

	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	int threads_per_block;
	int blocks_per_grid;
	for (int i = 0; i < 100; i++)
	{

		for (int layer = 1; layer <= maxlayer; layer++)
		{
			int threads_per_layer = 1 << (layer - 1);
			threads_per_block = min(threads_per_layer, 256);
			blocks_per_grid = (threads_per_layer + threads_per_block - 1) / threads_per_block;

			EvalAll_LBLEvaluation<<<blocks_per_grid, threads_per_block>>>(aes_key, layer, maxlayer, 1, d_key, d_s, d_t);
			cudaDeviceSynchronize();
		}
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("Full Domain Evalutaion takes: %.2f us\n", milliseconds / 100 * 1000);

	threads_per_block = min(1 << (maxlayer), 256);
	blocks_per_grid = (1 << maxlayer) / threads_per_block;
	EvalAll_lastlayer<<<blocks_per_grid, threads_per_block>>>(d_key, d_s, d_t, d_eval_res);
	cudaDeviceSynchronize();

	memcpy(eval_res, d_eval_res, sizeof(uint128_t) * (1 << maxlayer));

	cudaFree(d_key);
	cudaFree(d_s);
	cudaFree(d_t);
	cudaFree(d_eval_res);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		// return -1;
	}

	return eval_res;
}

extern "C" uint128_t *test_dpf_block(uint8_t *key, int N)
{
	int n = key[0];
	int maxlayer = n - 7;

	uint8_t *d_key;
	uint128_t *d_s;
	uint32_t *d_t;
	uint128_t *eval_res = (uint128_t *)malloc(sizeof(uint128_t) * (1 << maxlayer) * N);
	uint128_t *d_eval_res;

	cudaMalloc(&d_key, N * (1 + 16 + 1 + 18 * maxlayer + 16) * sizeof(uint8_t));
	cudaMemcpy(d_key, key, N * (1 + 16 + 1 + 18 * maxlayer + 16) * sizeof(uint8_t), cudaMemcpyHostToDevice);
	cudaMalloc(&d_s, N * (1 << maxlayer) * 2 * sizeof(uint128_t));
	cudaMalloc(&d_t, N * (1 << maxlayer) * 2 * sizeof(uint32_t));
	cudaMallocManaged(&d_eval_res, N * (1 << maxlayer) * 2 * sizeof(uint128_t));

	seed_copy<<<(N + 255) / 256, 256>>>(d_s, d_t, d_key, maxlayer, N);
	cudaDeviceSynchronize();

	uint32_t *aes_key;
	cudaMalloc(&aes_key, 4 * (14 + 1) * sizeof(uint32_t));
	fss_genaeskey_kernel<<<1, 1>>>(aes_key);

	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	int threads_per_block;
	int blocks_per_grid;
	int r = 9;
	for (int i = 0; i < 100; i++)
	{

		if (maxlayer <= 9)
		{
			EvalAll_BlockEvaluation<<<N * 1, (1 << maxlayer) / 2>>>(aes_key, maxlayer, 0, d_key, d_s, d_t, d_s + (1 << maxlayer) - 1, d_t + (1 << maxlayer) - 1);
			cudaDeviceSynchronize();
		}
		else if (maxlayer <= 18)
		{
			EvalAll_BlockEvaluation<<<N * 1, (1 << (maxlayer - r)) / 2>>>(aes_key, maxlayer - r, 0, d_key, d_s, d_t, d_s + (1 << (maxlayer - r) - 1), d_t + (1 << (maxlayer - r) - 1));
			cudaDeviceSynchronize();
			EvalAll_BlockEvaluation<<<N * (1 << (maxlayer - r)), (1 << r) / 2>>>(aes_key, r, maxlayer - r, d_key, d_s + (1 << (maxlayer - r) - 1), d_t + (1 << (maxlayer - r) - 1), d_s + (1 << maxlayer) - 1, d_t + (1 << maxlayer) - 1);
			cudaDeviceSynchronize();
		}
		else if (maxlayer <= 27)
		{
			EvalAll_BlockEvaluation<<<N * 1, (1 << (maxlayer - 2 * r)) / 2>>>(aes_key, maxlayer - 2 * r, 0, d_key,
																			  d_s, d_t, d_s + (1 << (maxlayer - 2 * r) - 1), d_t + (1 << (maxlayer - 2 * r) - 1));
			cudaDeviceSynchronize();
			EvalAll_BlockEvaluation<<<N * (1 << (maxlayer - 2 * r)), (1 << r) / 2>>>(aes_key, r, maxlayer - 2 * r, d_key,
																					 d_s + (1 << (maxlayer - 2 * r) - 1), d_t + (1 << (maxlayer - 2 * r) - 1), d_s + (1 << (maxlayer - r)) - 1, d_t + (1 << (maxlayer - r)) - 1);
			cudaDeviceSynchronize();
			EvalAll_BlockEvaluation<<<N * (1 << (maxlayer - r)), (1 << r) / 2>>>(aes_key, r, maxlayer - r, d_key,
																				 d_s + (1 << (maxlayer - r) - 1), d_t + (1 << (maxlayer - r) - 1), d_s + (1 << (maxlayer)) - 1, d_t + (1 << (maxlayer)) - 1);
			cudaDeviceSynchronize();
		}
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("Full Domain Evalutaion takes: %.2f us", milliseconds / 100 * 1000);

	threads_per_block = min(1 << (maxlayer), 256);
	blocks_per_grid = (1 << maxlayer) / threads_per_block;
	EvalAll_lastlayer<<<blocks_per_grid, threads_per_block>>>(d_key, d_s, d_t, d_eval_res);
	cudaDeviceSynchronize();

	memcpy(eval_res, d_eval_res, sizeof(uint128_t) * (1 << maxlayer));

	cudaFree(d_key);
	cudaFree(d_s);
	cudaFree(d_t);
	cudaFree(d_eval_res);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		// return -1;
	}

	return eval_res;
}

extern "C++" std::vector<void *> init_pir(int N, int n, uint128_t *db)
{
	check_mem_usage();

	std::vector<void *> d_ptrs;
	size_t maxlayer = n - 7;

	uint128_t *d_db;
	size_t db_size = (1 << n) * entry_size * sizeof(uint128_t);
	cudaMalloc(&d_db, db_size);
	cudaMemcpy(d_db, db, db_size, cudaMemcpyHostToDevice);
	d_ptrs.push_back(d_db);

	size_t total_cuda_malloc = 0;

	uint128_t *blocksum_cuda;
	uint128_t *d_k_res;

	size_t blocksum_size = N * ((1 << n) / 256) * sizeof(uint128_t) * entry_size;
	size_t kres_size;
	if ((1 << n) > 65536)
	{
		kres_size = N * ((1 << n) / 65536) * sizeof(uint128_t) * entry_size;
	}
	else
	{
		kres_size = N * sizeof(uint128_t) * entry_size;
	}

	cudaMalloc(&blocksum_cuda, blocksum_size);
	total_cuda_malloc += blocksum_size;

	cudaMalloc(&d_k_res, kres_size);
	total_cuda_malloc += kres_size;

	d_ptrs.push_back(blocksum_cuda);
	d_ptrs.push_back(d_k_res);

	uint8_t *d_key;
	size_t key_size = N * (1 + 16 + 1 + 18 * maxlayer + 16) * sizeof(uint8_t);
	cudaMalloc(&d_key, key_size);
	total_cuda_malloc += key_size;
	d_ptrs.push_back(d_key);

	uint128_t *d_s, *d_s_intermediate, *d_s_res;
	uint32_t *d_t, *d_t_intermediate, *d_t_res;
	uint128_t *d_se;

	size_t size_d_s = N * sizeof(uint128_t);
	size_t size_d_t = N * sizeof(uint32_t);
	size_t size_d_s_intermediate = N * (1 << 10) * sizeof(uint128_t);
	size_t size_d_t_intermediate = N * (1 << 10) * sizeof(uint32_t);
	size_t size_d_s_res = N * (1 << maxlayer) * sizeof(uint128_t);
	size_t size_d_t_res = N * (1 << maxlayer) * sizeof(uint32_t);
	size_t size_d_se = N * (1 << maxlayer) * sizeof(uint128_t);

	cudaMalloc(&d_s, size_d_s);
	total_cuda_malloc += size_d_s;
	cudaMalloc(&d_t, size_d_t);
	total_cuda_malloc += size_d_t;
	cudaMalloc(&d_s_intermediate, size_d_s_intermediate);
	total_cuda_malloc += size_d_s_intermediate;
	cudaMalloc(&d_t_intermediate, size_d_t_intermediate);
	total_cuda_malloc += size_d_t_intermediate;
	cudaMalloc(&d_s_res, size_d_s_res);
	total_cuda_malloc += size_d_s_res;
	cudaMalloc(&d_t_res, size_d_t_res);
	total_cuda_malloc += size_d_t_res;
	cudaMalloc(&d_se, size_d_se);
	total_cuda_malloc += size_d_se;

	d_ptrs.push_back(d_s);
	d_ptrs.push_back(d_t);
	d_ptrs.push_back(d_s_intermediate);
	d_ptrs.push_back(d_t_intermediate);
	d_ptrs.push_back(d_s_res);
	d_ptrs.push_back(d_t_res);
	d_ptrs.push_back(d_se);

	uint32_t *aes_key;
	size_t aes_key_size = 4 * (14 + 1) * sizeof(uint32_t);
	cudaMalloc(&aes_key, aes_key_size);
	total_cuda_malloc += aes_key_size;
	fss_genaeskey_kernel<<<1, 1>>>(aes_key);
	d_ptrs.push_back(aes_key);

	std::cout << "CUDA memory initialized successfully!" << std::endl;
	std::cout << "Total allocated CUDA memory (excluding database): "
			  << total_cuda_malloc / (1024.0 * 1024.0) << " MB" << std::endl;

	return d_ptrs;
}

extern "C++" std::vector<void *> init_pir_pipeline(int N, int n, uint128_t *db)
{
	check_mem_usage();

	std::vector<void *> d_ptrs;
	size_t maxlayer = n - 7;
	int CHUNK_SIZE = 1 << (n - 8);

	uint128_t *d_db;
	size_t db_size = (1 << n) * entry_size * sizeof(uint128_t);
	cudaMalloc(&d_db, db_size);
	cudaMemcpy(d_db, db, db_size, cudaMemcpyHostToDevice);
	d_ptrs.push_back(d_db);

	size_t total_cuda_malloc = 0;

	uint128_t *d_input, *d_intermediate8, *d_intermediate16, *d_output;

	size_t size_input = entry_size * CHUNK_SIZE * sizeof(uint128_t);
	size_t size_intermediate8 = entry_size * CHUNK_SIZE / 256 * sizeof(uint128_t);
	size_t size_intermediate16 = entry_size * CHUNK_SIZE / 65536 * sizeof(uint128_t);
	size_t size_output = entry_size * sizeof(uint128_t);

	cudaMalloc(&d_input, size_input);
	total_cuda_malloc += size_input;
	d_ptrs.push_back(d_input);

	cudaMalloc(&d_intermediate8, size_intermediate8);
	total_cuda_malloc += size_intermediate8;
	d_ptrs.push_back(d_intermediate8);

	cudaMalloc(&d_intermediate16, size_intermediate16);
	total_cuda_malloc += size_intermediate16;
	d_ptrs.push_back(d_intermediate16);

	cudaMalloc(&d_output, size_output);
	total_cuda_malloc += size_output;
	d_ptrs.push_back(d_output);

	uint128_t *d_s, *d_s_intermediate, *d_s_res;
	uint32_t *d_t, *d_t_intermediate, *d_t_res;
	uint128_t *d_se;

	size_t size_d_s = sizeof(uint128_t);
	size_t size_d_t = sizeof(uint32_t);
	size_t size_d_s_intermediate = (1 << 10) * sizeof(uint128_t);
	size_t size_d_t_intermediate = (1 << 10) * sizeof(uint32_t);
	size_t size_d_s_res = (1 << maxlayer) * sizeof(uint128_t);
	size_t size_d_t_res = (1 << maxlayer) * sizeof(uint32_t);
	size_t size_d_se = (1 << maxlayer) * sizeof(uint128_t);

	cudaMalloc(&d_s, size_d_s);
	total_cuda_malloc += size_d_s;
	cudaMalloc(&d_t, size_d_t);
	total_cuda_malloc += size_d_t;
	cudaMalloc(&d_s_intermediate, size_d_s_intermediate);
	total_cuda_malloc += size_d_s_intermediate;
	cudaMalloc(&d_t_intermediate, size_d_t_intermediate);
	total_cuda_malloc += size_d_t_intermediate;
	cudaMalloc(&d_s_res, size_d_s_res);
	total_cuda_malloc += size_d_s_res;
	cudaMalloc(&d_t_res, size_d_t_res);
	total_cuda_malloc += size_d_t_res;
	cudaMalloc(&d_se, size_d_se);
	total_cuda_malloc += size_d_se;

	d_ptrs.push_back(d_s);
	d_ptrs.push_back(d_t);
	d_ptrs.push_back(d_s_intermediate);
	d_ptrs.push_back(d_t_intermediate);
	d_ptrs.push_back(d_s_res);
	d_ptrs.push_back(d_t_res);
	d_ptrs.push_back(d_se);

	uint32_t *aes_key;
	size_t aes_key_size = 4 * (14 + 1) * sizeof(uint32_t);
	cudaMalloc(&aes_key, aes_key_size);
	total_cuda_malloc += aes_key_size;
	fss_genaeskey_kernel<<<1, 1>>>(aes_key);
	d_ptrs.push_back(aes_key);

	std::cout << "CUDA memory initialized successfully!" << std::endl;
	std::cout << "Total allocated CUDA memory (excluding database): "
			  << total_cuda_malloc / (1024.0 * 1024.0) << " MB" << std::endl;

	return d_ptrs;
}

extern "C++" std::vector<void *> init_pir_LUT(int N, int n, uint32_t *db)
{
	std::vector<void *> d_ptrs; // 存储设备指针的容器
	size_t maxlayer = n - 7;
	int CHUNK_SIZE = 1 << (n - 8);

	uint32_t *d_db;
	uint64_t db_size = (1ULL << n) * sizeof(uint32_t);
	cudaMalloc(&d_db, db_size);

	cudaMemcpy(d_db, db, (1ULL << n) * sizeof(uint32_t), cudaMemcpyHostToDevice);

	d_ptrs.push_back(d_db);

	uint32_t *d_input, *d_intermediate8, *d_intermediate16, *d_output;
	cudaMalloc(&d_input, N * CHUNK_SIZE * sizeof(uint32_t));
	d_ptrs.push_back(d_input);
	cudaMalloc(&d_intermediate8, N * CHUNK_SIZE / 256 * sizeof(uint32_t));
	d_ptrs.push_back(d_intermediate8);
	cudaMalloc(&d_intermediate16, N * CHUNK_SIZE / 65536 * sizeof(uint32_t));
	d_ptrs.push_back(d_intermediate16);
	cudaMalloc(&d_output, N * sizeof(uint32_t));
	d_ptrs.push_back(d_output);

	uint128_t *d_s, *d_s_intermediate, *d_s_res;
	uint32_t *d_t, *d_t_intermediate, *d_t_res;
	uint128_t *d_se;

	cudaMalloc(&d_s, N * sizeof(uint128_t));
	cudaMalloc(&d_t, N * sizeof(uint32_t));
	cudaMalloc(&d_s_intermediate, N * (1 << 10) * sizeof(uint128_t));
	cudaMalloc(&d_t_intermediate, N * (1 << 10) * sizeof(uint32_t));
	cudaMalloc(&d_s_res, N * 2 * (1 << maxlayer) * sizeof(uint128_t));
	cudaMalloc(&d_t_res, N * 2 * (1 << maxlayer) * sizeof(uint32_t));
	cudaMalloc(&d_se, N * (1 << maxlayer) * sizeof(uint128_t));

	d_ptrs.push_back(d_s);
	d_ptrs.push_back(d_t);
	d_ptrs.push_back(d_s_intermediate);
	d_ptrs.push_back(d_t_intermediate);
	d_ptrs.push_back(d_s_res);
	d_ptrs.push_back(d_t_res);
	d_ptrs.push_back(d_se);

	uint32_t *aes_key;
	cudaMalloc(&aes_key, 4 * (14 + 1) * sizeof(uint32_t));
	fss_genaeskey_kernel<<<1, 1>>>(aes_key);

	d_ptrs.push_back(aes_key);

	// uint8_t *key_pinned;
	// size_t key_size = N * (1 + 16 + 1 + 18 * maxlayer + 16) * sizeof(uint8_t);
	// cudaMallocHost(&key_pinned, key_size);

	// uint32_t *res;
	// cudaMallocHost(&res, N * sizeof(uint32_t));

	// std::vector<void *> d_ptrs_host;
	// d_ptrs_host.push_back(key_pinned);
	// d_ptrs_host.push_back(res);
	std::cout << "CUDA memory initialized successfully!" << std::endl;
	return d_ptrs;
}

void free_cuda_memory(std::vector<void *> &d_ptrs)
{
	for (auto ptr : d_ptrs)
	{
		if (ptr != nullptr)
		{
			cudaFree(ptr);
		}
	}
	d_ptrs.clear();
	std::cout << "CUDA memory freed successfully!" << std::endl;
}

extern "C++" uint128_t *test_dpf_pir(uint8_t *key, std::vector<void *> d_ptrs, int N)
{
	int n = key[0];
	int maxlayer = n - 7;

	uint8_t *d_key;
	uint128_t *d_s, *d_s_intermediate, *d_s_res;
	uint32_t *d_t, *d_t_intermediate, *d_t_res;

	uint128_t *d_db;
	uint128_t *d_se;
	// uint128_t *d_input;
	uint128_t *blocksum_cuda; // save first reduction
	uint128_t *d_k_res;		  // save second reduction

	d_db = static_cast<uint128_t *>(d_ptrs[0]);
	blocksum_cuda = static_cast<uint128_t *>(d_ptrs[1]);
	d_k_res = static_cast<uint128_t *>(d_ptrs[2]);
	d_key = static_cast<uint8_t *>(d_ptrs[3]);
	d_s = static_cast<uint128_t *>(d_ptrs[4]);
	d_t = static_cast<uint32_t *>(d_ptrs[5]);
	d_s_intermediate = static_cast<uint128_t *>(d_ptrs[6]);
	d_t_intermediate = static_cast<uint32_t *>(d_ptrs[7]);
	d_s_res = static_cast<uint128_t *>(d_ptrs[8]);
	d_t_res = static_cast<uint32_t *>(d_ptrs[9]);
	d_se = static_cast<uint128_t *>(d_ptrs[10]);

	uint128_t *res = (uint128_t *)malloc(N * sizeof(uint128_t) * entry_size);
	memset(res, 0, N * entry_size * sizeof(uint128_t));

	uint32_t *aes_key;
	// cudaMalloc(&aes_key, 4 * (14 + 1) * sizeof(uint32_t));
	aes_key = static_cast<uint32_t *>(d_ptrs.back());

	// cudaMalloc(&d_key, N * (1 + 16 + 1 + 18 * maxlayer + 16) * sizeof(uint8_t));
	cudaMemcpy(d_key, key, N * (1 + 16 + 1 + 18 * maxlayer + 16) * sizeof(uint8_t), cudaMemcpyHostToDevice);

	// for (int i = 0; i < N; i++)
	// {
	// 	cudaMemcpy(d_s + i, &key[1 + i * (1 + 16 + 1 + 18 * maxlayer + 16)], 16, cudaMemcpyHostToDevice);
	// 	cudaMemcpy(d_t + i, &key[17 + i * (1 + 16 + 1 + 18 * maxlayer + 16)], 1, cudaMemcpyHostToDevice);
	// }
	seed_copy<<<(N + 255) / 256, 256>>>(d_s, d_t, d_key, maxlayer, N);
	cudaDeviceSynchronize();

	int threads_per_block;
	int blocks_per_grid;

	int r = 9; // 10 layers for the first reduction
	// int maxBlocks;
	// cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocks, EvalAll_BlockEvaluation, (1 << r) / 2, 0);
	// printf("Max active blocks per SM: %d\n", maxBlocks);
	if (maxlayer <= r)
	{
		EvalAll_BlockEvaluation<<<N * 1, (1 << maxlayer) / 2>>>(aes_key, maxlayer, 0, d_key, d_s, d_t, d_s_res, d_t_res);
		cudaDeviceSynchronize();
	}
	else if (maxlayer > r)
	{

		EvalAll_BlockEvaluation<<<N * 1, (1 << (maxlayer - r)) / 2>>>(aes_key, maxlayer - r, 0, d_key, d_s, d_t, d_s_intermediate, d_t_intermediate);
		cudaDeviceSynchronize();

		EvalAll_BlockEvaluation<<<(N * (1 << (maxlayer - r))), (1 << r) / 2>>>(aes_key, r, maxlayer - r, d_key, d_s_intermediate, d_t_intermediate, d_s_res, d_t_res);
		cudaDeviceSynchronize();
	}

	threads_per_block = min(1 << maxlayer, 256);
	blocks_per_grid = (1 << maxlayer) / threads_per_block;

	EvalAll_SeGeneration<<<N * blocks_per_grid, threads_per_block>>>(d_key, d_s_res, d_t_res, d_se);
	cudaDeviceSynchronize();

	// cudaEventRecord(start);
	threads_per_block = 256;
	blocks_per_grid = ((1ULL) << n) / threads_per_block;

	MultiplicationReduction_grid_row<<<N, threads_per_block / 2>>>(d_key, d_se, d_db, blocksum_cuda, 1, N);
	cudaDeviceSynchronize();

	cudaMemcpy(res, blocksum_cuda, N * sizeof(uint128_t) * entry_size, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	// 检查错误
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		// return -1;
	}
	check_mem_usage();

	return res;
}

// #define NUM_STREAMS 10
// #define NUM_CHUNKS 32768
// #define CHUNK_SIZE (1 << 16)

extern "C++" std::vector<void *> init_streams_and_events()
{
	std::vector<void *> handles;

	cudaStream_t *streams = new cudaStream_t[NUM_STREAMS];
	for (int i = 0; i < NUM_STREAMS; i++)
	{
		cudaStreamCreate(&streams[i]);
	}
	handles.push_back(streams);

	cudaEvent_t(*events)[NUM_STREAMS] = new cudaEvent_t[NUM_CHUNKS][NUM_STREAMS];
	for (int chunk = 0; chunk < NUM_CHUNKS; chunk++)
	{
		for (int j = 0; j < NUM_STREAMS; j++)
		{
			cudaEventCreate(&events[chunk][j]);
		}
	}
	handles.push_back(events);

	std::cout << "CUDA streams and events initialized!" << std::endl;
	return handles;
}

extern "C++" void cleanup_streams_and_events(std::vector<void *> &handles)
{
	cudaStream_t *streams = static_cast<cudaStream_t *>(handles[0]);
	for (int i = 0; i < NUM_STREAMS; i++)
	{
		cudaStreamDestroy(streams[i]);
	}
	delete[] streams;

	cudaEvent_t(*events)[NUM_STREAMS] = static_cast<cudaEvent_t(*)[NUM_STREAMS]>(handles[1]);
	for (int chunk = 0; chunk < NUM_CHUNKS; chunk++)
	{
		for (int j = 0; j < NUM_STREAMS; j++)
		{
			cudaEventDestroy(events[chunk][j]);
		}
	}
	delete[] events;

	std::cout << "CUDA streams and events cleaned up!" << std::endl;
}

extern "C++" uint128_t *test_dpf_pir_pipeline(uint8_t *key, std::vector<void *> d_ptrs, std::vector<void *> handles, int N)
{
	int n = key[0];
	int maxlayer = n - 7;
	int CHUNK_SIZE = 1 << (n - 8);

	uint8_t *d_key;
	uint128_t *d_s, *d_s_intermediate, *d_s_res;
	uint32_t *d_t, *d_t_intermediate, *d_t_res;
	uint128_t *d_se;

	uint128_t *d_db;

	d_db = static_cast<uint128_t *>(d_ptrs[0]);

	uint128_t *d_input, *d_intermediate8, *d_intermediate16, *d_output;
	d_input = static_cast<uint128_t *>(d_ptrs[1]);
	d_intermediate8 = static_cast<uint128_t *>(d_ptrs[2]);
	d_intermediate16 = static_cast<uint128_t *>(d_ptrs[3]);
	d_output = static_cast<uint128_t *>(d_ptrs[4]);

	uint8_t *key_pinned;
	size_t key_size = N * (1 + 16 + 1 + 18 * maxlayer + 16) * sizeof(uint8_t);
	cudaMallocHost(&key_pinned, key_size);
	memcpy(key_pinned, key, key_size);

	d_s = static_cast<uint128_t *>(d_ptrs[5]);
	d_t = static_cast<uint32_t *>(d_ptrs[6]);
	d_s_intermediate = static_cast<uint128_t *>(d_ptrs[7]);
	d_t_intermediate = static_cast<uint32_t *>(d_ptrs[8]);
	d_s_res = static_cast<uint128_t *>(d_ptrs[9]);
	d_t_res = static_cast<uint32_t *>(d_ptrs[10]);
	d_se = static_cast<uint128_t *>(d_ptrs[11]);

	uint128_t *res;
	cudaMallocHost(&res, entry_size * N * sizeof(uint128_t));
	// res = static_cast<uint128_t *>(d_ptrs[14]);

	uint32_t *aes_key;
	aes_key = static_cast<uint32_t *>(d_ptrs.back());

	cudaMalloc(&d_key, N * (1 + 16 + 1 + 18 * maxlayer + 16) * sizeof(uint8_t));

	cudaStream_t *streams = static_cast<cudaStream_t *>(handles[0]);
	cudaEvent_t(*events)[28] = static_cast<cudaEvent_t(*)[28]>(handles[1]);

	cudaEvent_t keyReadyEvent;
	cudaEventCreate(&keyReadyEvent);

	int threads_per_block;
	int blocks_per_grid;

#ifdef USE_L2_CACHE
	cudaDeviceProp prop;			   // CUDA device properties variable
	cudaGetDeviceProperties(&prop, 0); // Query GPU properties

	size_t size = min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
	cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size); // set-aside 3/4 of L2 cache for persisting accesses or the max allowed

	// printf("%d\n", size);
	size_t window_size = min((uint64_t)prop.accessPolicyMaxWindowSize, (((uint64_t)(1) << n) * entry_size * 16));
	// printf("%d\n", window_size);

	cudaStreamAttrValue stream_attribute;																		   // Stream level attributes data structure
	stream_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void *>(d_db);								   // Global Memory data pointer
	stream_attribute.accessPolicyWindow.num_bytes = window_size;												   // Number of bytes for persisting accesses.
																												   // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
	stream_attribute.accessPolicyWindow.hitRatio = window_size / ((double)((uint64_t)(1) << n) * entry_size * 16); // Hint for L2 cache hit ratio for persisting accesses in the num_bytes region

	stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting; // Type of access property on cache hit
	stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming; // Type of access property on cache miss.

	// Set the attributes to a CUDA stream of type cudaStream_t
	cudaStreamSetAttribute(streams[4], cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
#endif
	// cudaMemcpy(d_input, db, entry_size * CHUNK_SIZE * sizeof(uint128_t), cudaMemcpyHostToDevice);

	for (int chunk = 0; chunk < N; chunk++)
	{
		unsigned char *d_key_offset = d_key + chunk * (1 + 16 + 1 + 18 * maxlayer + 16);

		if (maxlayer <= 9)
		{
			if (chunk > 0)
				cudaStreamWaitEvent(streams[1], events[chunk - 1][3], 0);
			cudaMemcpyAsync(d_s, &key_pinned[1 + chunk * (1 + 16 + 1 + 18 * maxlayer + 16)], 16, cudaMemcpyHostToDevice, streams[1]);
			cudaMemcpyAsync(d_t, &key_pinned[17 + chunk * (1 + 16 + 1 + 18 * maxlayer + 16)], 1, cudaMemcpyHostToDevice, streams[1]);
			cudaMemcpyAsync(d_key_offset, &key_pinned[chunk * (1 + 16 + 1 + 18 * maxlayer + 16)], (1 + 16 + 1 + 18 * maxlayer + 16) * sizeof(uint8_t), cudaMemcpyHostToDevice, streams[1]);

			EvalAll_BlockEvaluation<<<1, (1 << maxlayer) / 2, 0, streams[1]>>>(aes_key, maxlayer, 0, d_key_offset, d_s, d_t, d_s_res, d_t_res);

			cudaEventRecord(events[chunk][1], streams[1]);
		}
		else if (maxlayer > 9)
		{
			if (chunk > 0)
				cudaStreamWaitEvent(streams[1], events[chunk - 1][2], 0);
			cudaMemcpyAsync(d_s, &key_pinned[1 + chunk * (1 + 16 + 1 + 18 * maxlayer + 16)], 16, cudaMemcpyHostToDevice, streams[1]);
			cudaMemcpyAsync(d_t, &key_pinned[17 + chunk * (1 + 16 + 1 + 18 * maxlayer + 16)], 1, cudaMemcpyHostToDevice, streams[1]);
			cudaMemcpyAsync(d_key_offset, &key_pinned[chunk * (1 + 16 + 1 + 18 * maxlayer + 16)], (1 + 16 + 1 + 18 * maxlayer + 16) * sizeof(uint8_t), cudaMemcpyHostToDevice, streams[1]);

			EvalAll_BlockEvaluation<<<1, (1 << (maxlayer - 9)) / 2, 0, streams[1]>>>(aes_key, maxlayer - 9, 0, d_key_offset, d_s, d_t, d_s_intermediate, d_t_intermediate);
			cudaEventRecord(events[chunk][1], streams[1]);

			if (chunk > 0)
				cudaStreamWaitEvent(streams[2], events[chunk - 1][3], 0);
			cudaStreamWaitEvent(streams[2], events[chunk][1], 0);
			EvalAll_BlockEvaluation<<<(1 << (maxlayer - 9)), (1 << 9) / 2, 0, streams[2]>>>(aes_key, 9, maxlayer - 9, d_key_offset, d_s_intermediate, d_t_intermediate, d_s_res, d_t_res);
			cudaEventRecord(events[chunk][2], streams[2]);
		}
		threads_per_block = min(1 << maxlayer, 256);
		blocks_per_grid = (1 << maxlayer) / threads_per_block;

		if (chunk > 0)
			cudaStreamWaitEvent(streams[3], events[chunk - 1][4], 0);
		if (maxlayer <= 10)
			cudaStreamWaitEvent(streams[3], events[chunk][1], 0);
		else if (maxlayer > 10)
			cudaStreamWaitEvent(streams[3], events[chunk][2], 0);
		EvalAll_SeGeneration<<<blocks_per_grid, threads_per_block, 0, streams[3]>>>(d_key_offset, d_s_res, d_t_res, d_se);
		cudaEventRecord(events[chunk][3], streams[3]);

		threads_per_block = 256;
		blocks_per_grid = (1 << n) / threads_per_block;
		if (chunk > 0)
			cudaStreamWaitEvent(streams[4], events[chunk - 1][5], 0);

		cudaStreamWaitEvent(streams[4], events[chunk][3], 0);

		// MultiplicationReduction<<<blocks_per_grid, threads_per_block, 0, streams[4]>>>(d_key_offset, d_se, d_db, d_input, 1);
		MultiplicationReduction_grid_row<<<256, threads_per_block / 2, 0, streams[4]>>>(d_key_offset, d_se, d_db, d_input, 1, 1);
		// cudaDeviceSynchronize();
		cudaEventRecord(events[chunk][4], streams[4]);

		cudaStreamWaitEvent(streams[5], events[chunk][4], 0);
		// if (chunk > 0)
		// 	cudaStreamWaitEvent(streams[5], events[chunk - 1][6], 0);
		BlockReduction<<<1, 256, 0, streams[5]>>>(d_input, d_output);
		cudaEventRecord(events[chunk][5], streams[5]);
		cudaMemcpyAsync(res + entry_size * chunk * 1, d_output, entry_size * 1 * sizeof(uint128_t), cudaMemcpyDeviceToHost, streams[5]);

		// if (n <= 16)
		// {

		// 	cudaStreamWaitEvent(streams[5], events[chunk][4], 0);
		// 	// if (chunk > 0)
		// 	// 	cudaStreamWaitEvent(streams[5], events[chunk - 1][6], 0);
		// 	BlockReduction<<<1, ((1 << n) + 255) / 256, 0, streams[5]>>>(d_input, d_output);
		// 	cudaEventRecord(events[chunk][5], streams[5]);
		// 	cudaMemcpyAsync(res + entry_size * chunk * 1, d_output, entry_size * 1 * sizeof(uint128_t), cudaMemcpyDeviceToHost, streams[5]);
		// }
		// else if (n > 16 && n <= 24)
		// {

		// 	cudaStreamWaitEvent(streams[5], events[chunk][4], 0);
		// 	// if (chunk > 0)
		// 	// 	cudaStreamWaitEvent(streams[5], events[chunk - 1][6], 0);
		// 	BlockReduction<<<(CHUNK_SIZE + 255) / 256, 256, 0, streams[5]>>>(d_input, d_intermediate8);
		// 	cudaEventRecord(events[chunk][5], streams[5]);

		// 	cudaStreamWaitEvent(streams[6], events[chunk][5], 0);
		// 	BlockReduction<<<1, (CHUNK_SIZE + 255) / 256, 0, streams[6]>>>(d_intermediate8, d_output);

		// 	cudaEventRecord(events[chunk][6], streams[6]);
		// 	cudaMemcpyAsync(res + entry_size * chunk * 1, d_output, entry_size * 1 * sizeof(uint128_t), cudaMemcpyDeviceToHost, streams[6]);
		// }
		// else if (n > 24 && n <= 28)
		// {
		// 	cudaStreamWaitEvent(streams[5], events[chunk][4], 0);
		// 	// if (chunk > 0)
		// 	// 	cudaStreamWaitEvent(streams[5], events[chunk - 1][6], 0);
		// 	BlockReduction<<<(CHUNK_SIZE + 255) / 256, 256, 0, streams[5]>>>(d_input, d_intermediate8);
		// 	cudaEventRecord(events[chunk][5], streams[5]);

		// 	cudaStreamWaitEvent(streams[6], events[chunk][5], 0);
		// 	// if (chunk > 0)
		// 	// 	cudaStreamWaitEvent(streams[6], events[chunk - 1][7], 0);
		// 	BlockReduction<<<(CHUNK_SIZE + 255) / 65536, 256, 0, streams[6]>>>(d_intermediate8, d_intermediate16);
		// 	cudaEventRecord(events[chunk][6], streams[6]);

		// 	cudaStreamWaitEvent(streams[7], events[chunk][6], 0);
		// 	BlockReduction<<<1, (CHUNK_SIZE + 255) / 65536, 0, streams[7]>>>(d_intermediate16, d_output);
		// 	cudaEventRecord(events[chunk][7], streams[7]);

		// 	cudaMemcpyAsync(res + entry_size * chunk * 1, d_output, entry_size * 1 * sizeof(uint128_t), cudaMemcpyDeviceToHost, streams[7]);
		// }
	}
#ifdef L2_CACHE
	cudaCtxResetPersistingL2Cache();
#endif
	for (int i = 0; i < NUM_STREAMS; i++)
		cudaStreamSynchronize(streams[i]);
	check_mem_usage();

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		// return -1;
	}

	return res;
}

extern "C++" uint32_t *test_dpf_pir_LUT(uint8_t *key, std::vector<void *> d_ptrs, int N)
{
	int n = key[0];
	int maxlayer = n - 7;
	int CHUNK_SIZE = 1 << (n - 8);

	uint8_t *d_key;
	uint128_t *d_s, *d_s_intermediate, *d_s_res;
	uint32_t *d_t, *d_t_intermediate, *d_t_res;
	uint128_t *d_se;

	uint32_t *d_db;

	d_db = static_cast<uint32_t *>(d_ptrs[0]);

	uint32_t *d_input, *d_intermediate, *d_intermediate16, *d_output;
	d_input = static_cast<uint32_t *>(d_ptrs[1]);
	d_intermediate = static_cast<uint32_t *>(d_ptrs[2]);
	d_intermediate16 = static_cast<uint32_t *>(d_ptrs[3]);
	d_output = static_cast<uint32_t *>(d_ptrs[4]);

	uint8_t *key_pinned;
	size_t key_size = N * (1 + 16 + 1 + 18 * maxlayer + 16) * sizeof(uint8_t);
	cudaMallocHost(&key_pinned, key_size);
	memcpy(key_pinned, key, key_size);

	d_s = static_cast<uint128_t *>(d_ptrs[5]);
	d_t = static_cast<uint32_t *>(d_ptrs[6]);
	d_s_intermediate = static_cast<uint128_t *>(d_ptrs[7]);
	d_t_intermediate = static_cast<uint32_t *>(d_ptrs[8]);
	d_s_res = static_cast<uint128_t *>(d_ptrs[9]);
	d_t_res = static_cast<uint32_t *>(d_ptrs[10]);
	d_se = static_cast<uint128_t *>(d_ptrs[11]);

	uint32_t *res;
	cudaMallocHost(&res, N * sizeof(uint32_t));
	// res = static_cast<uint128_t *>(d_ptrs[14]);

	uint32_t *aes_key;
	aes_key = static_cast<uint32_t *>(d_ptrs.back());

	cudaMalloc(&d_key, N * (1 + 16 + 1 + 18 * maxlayer + 16) * sizeof(uint8_t));

	cudaMemcpy(d_key, key, N * (1 + 16 + 1 + 18 * maxlayer + 16) * sizeof(uint8_t), cudaMemcpyHostToDevice);

	// for (int i = 0; i < N; i++)
	// {
	// 	cudaMemcpy(d_s + i, &key[1 + i * (1 + 16 + 1 + 18 * maxlayer + 16)], 16, cudaMemcpyHostToDevice);
	// 	cudaMemcpy(d_t + i, &key[17 + i * (1 + 16 + 1 + 18 * maxlayer + 16)], 1, cudaMemcpyHostToDevice);
	// }

	seed_copy<<<(N + 255) / 256, 256>>>(d_s, d_t, d_key, maxlayer, N);
	cudaDeviceSynchronize();

	int threads_per_block;
	int blocks_per_grid;

	if (maxlayer <= 9)
	{
		EvalAll_BlockEvaluation<<<N * 1, (1 << maxlayer) / 2>>>(aes_key, maxlayer, 0, d_key, d_s, d_t, d_s_res, d_t_res);
		cudaDeviceSynchronize();

		threads_per_block = min(1 << maxlayer, 256);
		blocks_per_grid = (1 << maxlayer) / threads_per_block;

		EvalAll_SeGeneration<<<N * blocks_per_grid, threads_per_block>>>(d_key, d_s_res, d_t_res, d_se);
		cudaDeviceSynchronize();
	}
	else if (maxlayer > 9 && maxlayer <= 18)
	{
		EvalAll_BlockEvaluation<<<N * 1, (1 << (maxlayer - 9)) / 2>>>(aes_key, maxlayer - 9, 0, d_key, d_s, d_t, d_s_intermediate, d_t_intermediate);
		cudaDeviceSynchronize();

		EvalAll_BlockEvaluation<<<(N * (1 << (maxlayer - 9))), (1 << 9) / 2>>>(aes_key, 9, maxlayer - 9, d_key, d_s_intermediate, d_t_intermediate, d_s_res, d_t_res);
		cudaDeviceSynchronize();

		threads_per_block = min(1 << maxlayer, 256);
		blocks_per_grid = (1 << maxlayer) / threads_per_block;

		EvalAll_SeGeneration<<<N * blocks_per_grid, threads_per_block>>>(d_key, d_s_res, d_t_res, d_se);
		cudaDeviceSynchronize();
	}
	else if (maxlayer <= 30)
	{
		EvalAll_BlockEvaluation<<<1, (1 << 9) / 2>>>(aes_key, 9, 0, d_key, d_s, d_t, d_s_intermediate, d_t_intermediate);
		cudaDeviceSynchronize();

		EvalAll_BlockEvaluation<<<(1 << 9), (1 << 9) / 2>>>(aes_key, 9, 9, d_key, d_s_intermediate, d_t_intermediate, d_s_res + ((1 << 18) - 1), d_t_res + ((1 << 18) - 1));
		cudaDeviceSynchronize();

		// Eval_multi_layer<<<(1 << (maxlayer - 10)), (1 << 10) / 2>>>(aes_key, 10, maxlayer - 10, d_key, d_s + (1 << (maxlayer - 10) - 1), d_t + (1 << (maxlayer - 10) - 1), d_s + (1 << maxlayer) - 1, d_t + (1 << maxlayer) - 1);
		// cudaDeviceSynchronize();

		for (int layer = 19; layer <= maxlayer; layer++)
		{
			int threads_per_layer = 1 << (layer - 1);
			threads_per_block = min(threads_per_layer, 256);
			blocks_per_grid = (threads_per_layer + threads_per_block - 1) / threads_per_block;

			EvalAll_LBLEvaluation<<<blocks_per_grid, threads_per_block>>>(aes_key, layer, maxlayer, 1, d_key, d_s_res, d_t_res);
			cudaDeviceSynchronize();
		}
		threads_per_block = min(1 << maxlayer, 256);
		blocks_per_grid = (1 << maxlayer) / threads_per_block;

		EVAL_Pack_last_layer_gense<<<blocks_per_grid, threads_per_block>>>(d_key, d_s_res, d_t_res, d_se);
		cudaDeviceSynchronize();
	}

	threads_per_block = 256;
	blocks_per_grid = ((1ULL) << n) / threads_per_block;
	blocks_per_grid = min(blocks_per_grid, 768);

	const int device = 0;

	// printf("Cooperative single-kernel reduce XOR test\n");

	// cudaSetDevice(device);
	// cudaDeviceProp prop;
	// cudaGetDeviceProperties(&prop, device);

	// int grid_size = 0;
	// cudaOccupancyMaxActiveBlocksPerMultiprocessor(&grid_size,
	// 											  EvalAll_BlockEvaluation,
	// 											  threads_per_block, 4397 * sizeof(uint32_t));
	// grid_size *= prop.multiProcessorCount;
	// grid_size = min(grid_size, blocks_per_grid);
	// printf("%d %d\n", prop.multiProcessorCount, grid_size);

	// MultiplicationReduction_LUT<<<N * blocks_per_grid, threads_per_block>>>(d_key, d_se, d_db, d_input, 1);
	// cudaDeviceSynchronize();

	MultiplicationReduction_LUT_new<<<N * blocks_per_grid, threads_per_block>>>(d_key, d_se, d_db, d_input, 1, N);
	cudaDeviceSynchronize();

	if (n == 8)
		cudaMemcpy(res, d_input, N * 1 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

	if (blocks_per_grid > 256)
	{
		blocks_per_grid = (blocks_per_grid + 255) / 256;
		BlockReduction_LUT<<<N * blocks_per_grid, 256>>>(d_input, d_intermediate);
		cudaDeviceSynchronize();
		BlockReduction_LUT<<<N, blocks_per_grid>>>(d_intermediate, d_output);
		cudaDeviceSynchronize();
	}
	else
	{
		BlockReduction_LUT<<<N, blocks_per_grid>>>(d_input, d_output);
		cudaDeviceSynchronize();
	}
	cudaMemcpy(res, d_output, N * 1 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		// return -1;
	}

	return res;
}

extern "C" void cudaWarmup(int size, int party)
{

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	printf("Number of CUDA devices: %d\n", deviceCount);

	int deviceToUse = party;
	if (deviceToUse < deviceCount)
	{
		cudaSetDevice(deviceToUse);
		printf("Using CUDA device: %d\n", deviceToUse);
	}
	else
	{
		printf("Invalid device number.\n");
		return;
	}

	int *d_data;
	size_t bytes = size * sizeof(int);

	cudaMalloc(&d_data, bytes);

	int blockSize = 256;
	int gridSize = (size + blockSize - 1) / blockSize;

	warmupKernel<<<gridSize, blockSize>>>(d_data, size);

	cudaDeviceSynchronize();

	cudaFree(d_data);
}
