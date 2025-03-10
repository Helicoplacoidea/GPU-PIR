#include <stdio.h>
#include <string.h>
#include "aes_cuda.h"
#include "../mpc_keys/uint128_type.h"
#include "aes_prg_device.h"

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
		data[idx] += 1; // 简单的操作，增加每个元素的值
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

	unsigned char *buff0;
	unsigned char *buff1;
	buff0 = (unsigned char *)malloc(1 + 16 + 1 + 18 * maxlayer + 16);
	buff1 = (unsigned char *)malloc(1 + 16 + 1 + 18 * maxlayer + 16);

	if (buff0 == NULL || buff1 == NULL)
	{
		printf("Memory allocation failed\n");
		return;
	}

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
	free(buff0);
	free(buff1);
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

__inline__ __device__ void
warpReduce_4(uint128_t *result)
{
#pragma unroll
	for (int offset = (warpSize >> 1); offset > 1; offset >>= 1)
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

__global__ void fss_genaeskey_kernel(uint32_t key[4 * (14 + 1)])
{
	// 测试密钥 (16字节 = 128位)
	uint64_t userkey1 = 597349;
	uint64_t userkey2 = 121379;
	uint128_t userkey(userkey1, userkey2);

	// 扩展密钥
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

__global__ void EVAL_Pack_layer(uint32_t *key, int layer, int maxlayer, int N, unsigned char *k, uint128_t *s, uint32_t *t)
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

__global__ void Eval_multi_layer(uint32_t *key, int layer_len, int layer, unsigned char *k, uint128_t *s, uint32_t *t, uint128_t *res_s, uint32_t *res_t)
{
	__shared__ uint128_t shared_s[1024 * 2];
	__shared__ uint8_t shared_t[1024 * 2];
	__shared__ uint128_t shared_sCW[10];
	__shared__ uint8_t shared_tCW[10][2];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	if (tid > (1 << layer_len))
		return;

	if (threadIdx.x == 0)
	{
		shared_s[0] = s[blockIdx.x];
		shared_t[0] = t[blockIdx.x];
		for (int i = 0; i < layer_len; i++)
		{
			memcpy(&shared_sCW[i], &k[18 * (i + 1 + layer)], 16);
			shared_tCW[i][0] = k[18 * (i + 1 + layer) + 16];
			shared_tCW[i][1] = k[18 * (i + 1 + layer) + 17];
		}
	}
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

			PRG_cuda(key, shared_s[tid + stride - 1], sL, sR, tL, tR);

			if (shared_t[tid + stride - 1] == 1)
			{
				sL = sL ^ sCW;
				sR = sR ^ sCW;
				tL = tL ^ tCW[0];
				tR = tR ^ tCW[1];
			}

			shared_s[(tid + stride - 1) * 2 + 1] = sL;
			shared_s[(tid + stride - 1) * 2 + 2] = sR;
			shared_t[(tid + stride - 1) * 2 + 1] = tL;
			shared_t[(tid + stride - 1) * 2 + 2] = tR;
		}
		__syncthreads();
	}

	res_s[idx] = shared_s[(tid + (1 << layer_len) - 1)];
	res_t[idx] = shared_t[tid + (1 << layer_len) - 1];
	__syncthreads();
}

__global__ void EVAL_Pack_last_layer(unsigned char *k, uint128_t *s, uint32_t *t, uint128_t *res)
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

__global__ void EVAL_Pack_last_layer_warp(unsigned char *k, uint128_t *s, uint32_t *t, uint128_t *d_db, uint128_t *blocksum_cuda)
{
	__shared__ uint128_t sdata[8 * entry_size];
	bool selector[128];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int tid = threadIdx.x;
	const int lid = tid & 31;
	const int wid = tid / warpSize;
	int n = k[0];
	int maxlayer = n - 7;

	int stride = 1 << (maxlayer); // layer stride
	int batch_id = idx / stride;
	idx %= stride;

	unsigned char *k_local = (unsigned char *)(k + batch_id * (1 + 16 + 1 + 18 * maxlayer + 16));
	uint128_t *s_local = (uint128_t *)(s + batch_id * (1 << maxlayer) * 2);
	uint32_t *t_local = (uint32_t *)(t + batch_id * (1 << maxlayer) * 2);
	// if (idx < (1 << maxlayer))
	// {
	uint128_t finalblock;
	memcpy(&finalblock, &k_local[18 * (maxlayer + 1)], 16);

	// res[idx] = s_local[idx + (1 << maxlayer) - 1] ^ finalblock.select(t_local[idx + (1 << maxlayer) - 1]);
	// }
	__syncthreads();

	for (int i = 0; i < 128; i++)
	{
		selector[i] = (s_local[idx + (1 << maxlayer) - 1] ^ finalblock.select(t_local[idx + (1 << maxlayer) - 1])).get_bit(127 - i);
		// if (idx == 0)
		// 	printf("%d: %d\n", i, selector[i]);
	}
	__syncthreads();

	// if (idx >= 27 * 64 && idx < 28 * 64)
	// {
	// printf("%d:%d\n", idx, selector);
	// }
	// if (idx == 1)
	// 	res[idx].print_uint128("res[1]:", res[idx]);
	uint128_t temp[128][entry_size];
	uint128_t temp_0[entry_size];

	for (int i = 0; i < 128; i++)
		if (selector[i] != 0)
		{
			for (int j = 0; j < entry_size; j++)
				temp[i][j] = d_db[i * entry_size + j];
		}
		else
		{
			for (int j = 0; j < entry_size; j++)
				temp[i][j] = uint128_t(0, 0);
		}
	__syncthreads();

	// if (idx == 0)
	// 	for (int j = 0; j < 64; j++)
	// 		temp[0][j].print_uint128("", temp[0][j]);

	memset(temp_0, 0, entry_size * sizeof(uint128_t));

	// method 1
	for (int i = 0; i < 128; i++)
	{
		for (int j = 0; j < entry_size; j++)
		{
			temp_0[j] ^= temp[i][j];
		}
	}
	warpReduce(temp_0);

	if (lid == 0)
	{
		for (int j = 0; j < entry_size; j++)
		{
			sdata[wid * entry_size + j] = temp_0[j];
		}
	}
	__syncthreads();

	for (int j = 0; j < entry_size; j++)
	{
		temp_0[j] = (tid < (blockDim.x >> 5)) ? sdata[lid * entry_size + j] : uint128_t(0, 0);
	}

	if (wid == 0)
		warpReduce(temp_0);

	// write res to global mem
	if (tid == 0)
	{
		for (int i = 0; i < entry_size; i++)
		{
			blocksum_cuda[(blockIdx.x) * entry_size + i] = temp_0[i];
			// if (blockIdx.x == 1)
			// 	temp[i].print_uint128("", temp[i]);
		}
	}

	// reduce in per warp
	// for (int i = 0; i < 128; i++)
	// 	warpReduce(temp[i]);

	// for (int i = 0; i < 128; i++)
	// {
	// 	if (lid == 0)
	// 	{
	// 		for (int j = 0; j < 64; j++)
	// 		{
	// 			sdata[wid * 64 + j] = temp[i][j];
	// 		}
	// 	}
	// 	__syncthreads();

	// 	for (int j = 0; j < 64; j++)
	// 	{
	// 		temp[i][j] = (tid < (blockDim.x >> 5)) ? sdata[lid * 64 + j] : uint128_t(0, 0);
	// 	}

	// 	if (wid == 0)
	// 		warpReduce(temp[i]);
	// }

	// // write res to global mem
	// if (tid == 0)
	// {
	// 	for (int j = 0; j < 128; j++)
	// 		for (int i = 0; i < 64; i++)
	// 		{
	// 			blocksum_cuda[(blockIdx.x) * 64 * 128 + j * 64 + i] = temp[j][i];
	// 			// if (blockIdx.x == 1)
	// 			// 	temp[i].print_uint128("", temp[i]);
	// 		}
	// }
}

__global__ void EVAL_Pack_last_layer_warp_n(unsigned char *k, uint128_t *se, uint128_t *d_db, uint128_t *blocksum_cuda)
{
	__shared__ uint128_t sdata[8 * entry_size];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int tid = threadIdx.x;
	const int lid = tid & 31;
	const int wid = tid / warpSize;
	int n = k[0];

	int stride = 1 << (n); // layer stride
	int batch_id = idx / stride;
	idx %= stride;

	uint128_t *se_local = (uint128_t *)(se + batch_id * (1 << (n - 7)));

	uint128_t temp_0[entry_size];

	if (se_local[idx / 128].get_bit(127 - idx % 128) != 0)
	{
		for (int j = 0; j < entry_size; j++)
			temp_0[j] = d_db[idx * entry_size + j];
		// temp_0[j] = uint128_t(0, 0);
	}
	else
	{
		for (int j = 0; j < entry_size; j++)
			temp_0[j] = uint128_t(0, 0);
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
			sdata[wid * entry_size + j] = temp_0[j];
		}
	}
	__syncthreads();

	for (int j = 0; j < entry_size; j++)
	{
		temp_0[j] = (tid < (blockDim.x >> 5)) ? sdata[lid * entry_size + j] : uint128_t(0, 0);
	}

	if (wid == 0)
		warpReduce(temp_0);

	// write res to global mem
	if (tid == 0)
	{
		for (int i = 0; i < entry_size; i++)
		{
			blocksum_cuda[(blockIdx.x) * entry_size + i] = temp_0[i];
		}
	}
}

__global__ void EVAL_Pack_last_layer_warp_once(int n, uint128_t *se, uint128_t *d_db, uint128_t *blocksum_cuda)
{
	// 计算线程索引
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int tid = threadIdx.x;										   // 线程索引（块内）
	const int lid = tid & 31;											   // 线程在 warp 中的局部索引
	const int wid = tid / warpSize;										   // warp 索引（块内）
	const int warp_global_id = blockIdx.x * (blockDim.x / warpSize) + wid; // warp 的全局 ID
	// int n = k[0];														   // 获取层数信息

	int stride = 1 << (n);		 // layer stride
	int batch_id = idx / stride; // 批次 ID
	idx %= stride;

	uint128_t *se_local = (uint128_t *)(se + batch_id * (1 << (n - 7))); // 局部 SE 指针

	uint128_t temp_0[entry_size]; // 每个线程的局部数据

	// 根据 `se_local` 初始化数据
	if (se_local[idx / 128].get_bit(127 - idx % 128) != 0)
	{
		for (int j = 0; j < entry_size; j++)
			temp_0[j] = d_db[idx * entry_size + j];
	}
	else
	{
		for (int j = 0; j < entry_size; j++)
			temp_0[j] = uint128_t(0, 0);
	}
	__syncwarp(); // 同步 warp 内线程

	// 进行 warp 级别归约
	warpReduce_4(temp_0);

	// 每个 warp 的线程 0 将归约结果写入 global memory
	if (lid == 0 || lid == 1)
	{
		for (int j = 0; j < entry_size; j++)
		{
			blocksum_cuda[(warp_global_id * 2 + lid) * entry_size + j] = temp_0[j];
		}
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

__global__ void cal_sum_warp(uint128_t *blocksum_cuda, uint128_t *k_res)
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
			sdata[(tid >> 5) * entry_size + i] = temp[i];
		}
	}
	__syncthreads();

	// 	if (tid == 0)
	// 	{
	// #pragma unroll
	// 		for (int i = 1; i < (blockDim.x >> 5); i++)
	// 			for (int j = 0; j < 256; j++)
	// 				sdata[j] ^= sdata[i * 258 + j];
	// 	}
	// 	__syncthreads();
	int warps = (blockDim.x > 32) ? (blockDim.x >> 5) : 1;
	for (int i = 0; i < entry_size; i++)
	{
		temp[i] = (tid < warps) ? sdata[lid * entry_size + i] : uint128_t(0, 0);
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

__global__ void half_reduction_full_stride(uint128_t *input, uint128_t *output)
{
	const int tid = threadIdx.x;
	const int lid = tid & 31;
	const int wid = tid / warpSize;
	uint128_t temp[entry_size];

#pragma unroll
	for (int i = 0; i < entry_size; i++)
	{
		temp[i] = input[(blockDim.x * blockIdx.x + tid) * entry_size + i];
	}

	int offset = 16;
	for (int i = 0; i < entry_size; i++)
	{
		// printf("hello");
		uint64_t high, low;
		high = temp[i].get_high() ^ __shfl_down_sync(0xFFFFFFFF, temp[i].get_high(), offset);
		low = temp[i].get_low() ^ __shfl_down_sync(0xFFFFFFFF, temp[i].get_low(), offset);
		temp[i] = uint128_t(high, low);
	}
	__syncthreads();
	if (lid < 16)
		for (int i = 0; i < entry_size; i++)
		{
			output[((blockIdx.x * blockDim.x / warpSize + wid) * 16 + lid) * entry_size + i] = temp[i];
		}
}

__global__ void half_reduction_double(uint128_t *input, uint128_t *output)
{
	const int tid = threadIdx.x;
	const int lid = tid & 31;
	const int wid = tid / warpSize;
	uint128_t temp[entry_size];

#pragma unroll
	for (int i = 0; i < entry_size; i++)
	{
		temp[i] = input[(blockDim.x * blockIdx.x + tid) * entry_size + i];
	}

	int offset = 16;
	for (int i = 0; i < entry_size; i++)
	{
		// printf("hello");
		uint64_t high, low;
		high = temp[i].get_high() ^ __shfl_down_sync(0xFFFFFFFF, temp[i].get_high(), offset);
		low = temp[i].get_low() ^ __shfl_down_sync(0xFFFFFFFF, temp[i].get_low(), offset);
		temp[i] = uint128_t(high, low);
	}
	offset = 8;
	for (int i = 0; i < entry_size; i++)
	{
		// printf("hello");
		uint64_t high, low;
		high = temp[i].get_high() ^ __shfl_down_sync(0xFFFFFFFF, temp[i].get_high(), offset);
		low = temp[i].get_low() ^ __shfl_down_sync(0xFFFFFFFF, temp[i].get_low(), offset);
		temp[i] = uint128_t(high, low);
	}
	__syncthreads();
	if (lid < 8)
		for (int i = 0; i < entry_size; i++)
		{
			output[((blockIdx.x * blockDim.x / warpSize + wid) * 8 + lid) * entry_size + i] = temp[i];
		}
}

__global__ void half_reduction_8_stride(uint128_t *input, uint128_t *output)
{
	const int tid = threadIdx.x;

	uint128_t temp[entry_size];

#pragma unroll
	for (int i = 0; i < entry_size; i++)
	{
		temp[i] = input[(blockDim.x * blockIdx.x + tid) * entry_size + i];
	}

	int offset = 8;
	for (int i = 0; i < entry_size; i++)
	{
		// printf("hello");
		uint64_t high, low;
		high = temp[i].get_high() ^ __shfl_down_sync(0xFFFFFFFF, temp[i].get_high(), offset);
		low = temp[i].get_low() ^ __shfl_down_sync(0xFFFFFFFF, temp[i].get_low(), offset);
		temp[i] = uint128_t(high, low);
	}
	// if (lid < 16)
	//     for (int i = 0; i < entry_size; i++)
	//     {
	//         output[((blockIdx.x * blockDim.x / warpSize + wid) * 16 + lid) * entry_size + i] = temp[i];
	//     }
	if (tid < 8)
		for (int i = 0; i < entry_size; i++)
		{
			output[tid * entry_size + i] = temp[i];
		}
}

__global__ void half_reduction_4_stride(uint128_t *input, uint128_t *output)
{
	const int tid = threadIdx.x;

	uint128_t temp[entry_size];

#pragma unroll
	for (int i = 0; i < entry_size; i++)
	{
		temp[i] = input[(blockDim.x * blockIdx.x + tid) * entry_size + i];
	}

	int offset = 4;
	for (int i = 0; i < entry_size; i++)
	{
		// printf("hello");
		uint64_t high, low;
		high = temp[i].get_high() ^ __shfl_down_sync(0xFFFFFFFF, temp[i].get_high(), offset);
		low = temp[i].get_low() ^ __shfl_down_sync(0xFFFFFFFF, temp[i].get_low(), offset);
		temp[i] = uint128_t(high, low);
	}
	// if (lid < 16)
	//     for (int i = 0; i < entry_size; i++)
	//     {
	//         output[((blockIdx.x * blockDim.x / warpSize + wid) * 16 + lid) * entry_size + i] = temp[i];
	//     }
	if (tid < 4)
		for (int i = 0; i < entry_size; i++)
		{
			output[tid * entry_size + i] = temp[i];
		}
}

__global__ void half_reduction_2_stride(uint128_t *input, uint128_t *output)
{
	const int tid = threadIdx.x;

	uint128_t temp[entry_size];

#pragma unroll
	for (int i = 0; i < entry_size; i++)
	{
		temp[i] = input[(blockDim.x * blockIdx.x + tid) * entry_size + i];
	}

	int offset = 2;
	for (int i = 0; i < entry_size; i++)
	{
		// printf("hello");
		uint64_t high, low;
		high = temp[i].get_high() ^ __shfl_down_sync(0xFFFFFFFF, temp[i].get_high(), offset);
		low = temp[i].get_low() ^ __shfl_down_sync(0xFFFFFFFF, temp[i].get_low(), offset);
		temp[i] = uint128_t(high, low);
	}
	// if (lid < 16)
	//     for (int i = 0; i < entry_size; i++)
	//     {
	//         output[((blockIdx.x * blockDim.x / warpSize + wid) * 16 + lid) * entry_size + i] = temp[i];
	//     }
	if (tid < 2)
		for (int i = 0; i < entry_size; i++)
		{
			output[tid * entry_size + i] = temp[i];
		}
}

__global__ void half_reduction_1_stride(uint128_t *input, uint128_t *output)
{
	const int tid = threadIdx.x;

	uint128_t temp[entry_size];

#pragma unroll
	for (int i = 0; i < entry_size; i++)
	{
		temp[i] = input[(blockDim.x * blockIdx.x + tid) * entry_size + i];
	}

	int offset = 1;
	for (int i = 0; i < entry_size; i++)
	{
		// printf("hello");
		uint64_t high, low;
		high = temp[i].get_high() ^ __shfl_down_sync(0xFFFFFFFF, temp[i].get_high(), offset);
		low = temp[i].get_low() ^ __shfl_down_sync(0xFFFFFFFF, temp[i].get_low(), offset);
		temp[i] = uint128_t(high, low);
	}
	// if (lid < 16)
	//     for (int i = 0; i < entry_size; i++)
	//     {
	//         output[((blockIdx.x * blockDim.x / warpSize + wid) * 16 + lid) * entry_size + i] = temp[i];
	//     }
	if (tid < 1)
		for (int i = 0; i < entry_size; i++)
		{
			output[tid * entry_size + i] = temp[i];
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

	// 等待GPU完成
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
	// 启动kernel
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

	// 创建CUDA事件
	cudaEvent_t start1, stop1, start2, stop2, start3, stop3;
	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);
	cudaEventCreate(&start3);
	cudaEventCreate(&stop3);

	// 测量第一个kernel: fss_gen_kernel
	cudaEventRecord(start1);
	fss_gen_kernel<<<blocks, threads>>>(aes_key, value1, 64, k0, k1, N, maxlayer);
	cudaEventRecord(stop1);

	// 测量第二个kernel: first fss_eval_kernel
	cudaEventRecord(start2);
	fss_eval_kernel<<<blocks, threads>>>(res1, aes_key, value2, 64, k0, N, maxlayer);
	cudaMemcpy(res1_host, res1, N * sizeof(bool), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop2);

	// 测量第三个kernel: second fss_eval_kernel
	cudaEventRecord(start3);
	fss_eval_kernel<<<blocks, threads>>>(res2, aes_key, value2, 64, k1, N, maxlayer);
	cudaMemcpy(res2_host, res2, N * sizeof(bool), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop3);

	// 同步并获取时间
	cudaEventSynchronize(stop1);
	cudaEventSynchronize(stop2);
	cudaEventSynchronize(stop3);

	float time1 = 0, time2 = 0, time3 = 0;
	cudaEventElapsedTime(&time1, start1, stop1);
	cudaEventElapsedTime(&time2, start2, stop2);
	cudaEventElapsedTime(&time3, start3, stop3);

	// 打印结果
	printf("fss_gen_kernel time: %.3f ms\n", time1);
	printf("First fss_eval_kernel time: %.3f ms\n", time2);
	printf("Second fss_eval_kernel time: %.3f ms\n", time3);

	// 销毁事件
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

	// 等待GPU完成
	cudaDeviceSynchronize();

	cudaFree(k0);
	cudaFree(k1);
	cudaFree(res1);
	cudaFree(res2);
	// cudaFree(alpha1_host);
	// cudaFree(alpha2_host);

	// 检查错误
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		return -1;
	}

	return 0;
}

extern "C" uint128_t *test_dpf(uint8_t *key)
{
	// 启动kernel
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

		// for (int layer = 1; layer <= maxlayer; layer++)
		// {
		// 	int threads_per_layer = 1 << (layer - 1);
		// 	threads_per_block = min(threads_per_layer, 256);
		// 	blocks_per_grid = (threads_per_layer + threads_per_block - 1) / threads_per_block;

		// 	EVAL_Pack_layer<<<blocks_per_grid, threads_per_block>>>(aes_key, layer, maxlayer, 1, d_key, d_s, d_t);
		// 	cudaDeviceSynchronize();
		// }

		if (maxlayer <= 10)
		{
			Eval_multi_layer<<<1, (1 << maxlayer)>>>(aes_key, maxlayer, 0, d_key, d_s, d_t, d_s + (1 << maxlayer) - 1, d_t + (1 << maxlayer) - 1);
			cudaDeviceSynchronize();
		}
		else if (maxlayer <= 20)
		{
			Eval_multi_layer<<<1, (1 << (maxlayer - 10))>>>(aes_key, maxlayer - 10, 0, d_key, d_s, d_t, d_s + (1 << (maxlayer - 10) - 1), d_t + (1 << (maxlayer - 10) - 1));
			cudaDeviceSynchronize();
			Eval_multi_layer<<<(1 << (maxlayer - 10)), (1 << 10)>>>(aes_key, 10, maxlayer - 10, d_key, d_s + (1 << (maxlayer - 10) - 1), d_t + (1 << (maxlayer - 10) - 1), d_s + (1 << maxlayer) - 1, d_t + (1 << maxlayer) - 1);
			cudaDeviceSynchronize();
		}
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("二叉树全域评估完成，耗时: %.2f ms\n", milliseconds);

	threads_per_block = min(1 << (maxlayer), 256);
	blocks_per_grid = (1 << maxlayer) / threads_per_block;
	EVAL_Pack_last_layer<<<blocks_per_grid, threads_per_block>>>(d_key, d_s, d_t, d_eval_res);
	cudaDeviceSynchronize();

	memcpy(eval_res, d_eval_res, sizeof(uint128_t) * (1 << maxlayer));

	cudaFree(d_key);
	cudaFree(d_s);
	cudaFree(d_t);
	cudaFree(d_eval_res);
	// 检查错误
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		// return -1;
	}

	return eval_res;
}

extern "C" uint128_t *test_dpf_pir(uint8_t *key, uint128_t *db, int N)
{
	int deviceCount;
	cudaError_t err = cudaGetDeviceCount(&deviceCount);
	// 使用第一个设备（device 0）
	int deviceId = 0;
	cudaSetDevice(deviceId);
	int maxThreadsPerBlock;

	// 查询每个线程块最大支持的线程数
	err = cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxGridDimX, deviceId);

	printf("Device %d: Maximum threads per block = %d\n", deviceId, maxThreadsPerBlock);
	check_mem_usage();

	// 启动kernel
	int n = key[0];
	int maxlayer = n - 7;

	uint8_t *d_key;
	uint128_t *d_s;
	uint32_t *d_t;
	uint128_t *d_db;
	// uint128_t *eval_res = (uint128_t *)malloc(sizeof(uint128_t) * (1 << maxlayer));
	uint128_t *d_eval_res;
	uint128_t *d_se;
	// uint128_t *d_input;
	uint128_t *blocksum_cuda; // save first reduction
	uint128_t *d_k_res;		  // save second reduction

	cudaMalloc(&d_db, (1 << n) * entry_size * sizeof(uint128_t));
	cudaMemcpy(d_db, db, (1 << n) * entry_size * sizeof(uint128_t), cudaMemcpyHostToDevice);

	// cudaMallocManaged(&d_input, N * ((1 << n)) * sizeof(uint128_t) * entry_size);
	cudaMalloc(&blocksum_cuda, N * ((1 << n) / (256)) * sizeof(uint128_t) * entry_size);
	if ((1 << n) > 65536)
		cudaMalloc(&d_k_res, N * ((1 << n) / 65536) * sizeof(uint128_t) * entry_size);
	else if ((1 << n) > 256 && (1 << n) <= 65536)
		cudaMalloc(&d_k_res, N * sizeof(uint128_t) * entry_size);

	uint128_t *k_res;
	if ((1 << n) > 65536)
		k_res = (uint128_t *)malloc(N * ((1 << n) / 65536) * sizeof(uint128_t) * entry_size);
	else if ((1 << n) > 256 && (1 << n) <= 65536)
		k_res = (uint128_t *)malloc(N * sizeof(uint128_t) * entry_size);

	uint128_t *res = (uint128_t *)malloc(N * sizeof(uint128_t) * entry_size);
	memset(res, 0, N * entry_size * sizeof(uint128_t));

	uint32_t *aes_key;
	cudaMalloc(&aes_key, 4 * (14 + 1) * sizeof(uint32_t));
	fss_genaeskey_kernel<<<1, 1>>>(aes_key);

	cudaEvent_t start, stop;
	float milliseconds = 0;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	cudaMalloc(&d_key, N * (1 + 16 + 1 + 18 * maxlayer + 16) * sizeof(uint8_t));
	cudaMemcpy(d_key, key, N * (1 + 16 + 1 + 18 * maxlayer + 16) * sizeof(uint8_t), cudaMemcpyHostToDevice);
	cudaMalloc(&d_s, N * (1 << maxlayer) * 2 * sizeof(uint128_t));
	cudaMalloc(&d_t, N * (1 << maxlayer) * 2 * sizeof(uint32_t));

	cudaMalloc(&d_se, N * (1 << maxlayer) * sizeof(uint128_t));

	for (int i = 0; i < N; i++)
	{
		cudaMemcpy(d_s + i * (1 << maxlayer) * 2, &key[1 + i * (1 + 16 + 1 + 18 * maxlayer + 16)], 16, cudaMemcpyHostToDevice);
		cudaMemcpy(d_t + i * (1 << maxlayer) * 2, &key[17 + i * (1 + 16 + 1 + 18 * maxlayer + 16)], 1, cudaMemcpyHostToDevice);
	}

	int threads_per_block;
	int blocks_per_grid;
	for (int layer = 1; layer <= maxlayer; layer++)
	{
		int threads_per_layer = 1 << (layer - 1);
		threads_per_block = min(threads_per_layer, 256);
		blocks_per_grid = (threads_per_layer + threads_per_block - 1) / threads_per_block;

		EVAL_Pack_layer<<<N * blocks_per_grid, threads_per_block>>>(aes_key, layer, maxlayer, N, d_key, d_s, d_t);
		cudaDeviceSynchronize();
	}

	threads_per_block = min(1 << maxlayer, 256);
	blocks_per_grid = (1 << maxlayer) / threads_per_block;
	// EVAL_Pack_last_layer_warp<<<N * blocks_per_grid, threads_per_block>>>(d_key, d_s, d_t, d_db, blocksum_cuda);
	// cudaDeviceSynchronize();
	EVAL_Pack_last_layer_gense<<<N * blocks_per_grid, threads_per_block>>>(d_key, d_s, d_t, d_se);
	cudaDeviceSynchronize();

	// cudaEventRecord(stop);
	// cudaEventSynchronize(stop);
	// cudaEventElapsedTime(&milliseconds, start, stop);
	// printf("二叉树全域评估完成，耗时: %.2f ms\n", milliseconds);

	// cudaEventRecord(start);
	threads_per_block = 256;
	blocks_per_grid = (1 << n) / threads_per_block;

	EVAL_Pack_last_layer_warp_n<<<N * blocks_per_grid, threads_per_block>>>(d_key, d_se, d_db, blocksum_cuda);
	cudaDeviceSynchronize();

	// EVAL_Pack_last_layer_warp_n<<<N / 4 * blocks_per_grid, threads_per_block>>>(d_key, d_se + N / 4 * (1 << maxlayer), d_db, blocksum_cuda + N / 4 * ((1 << n) / (256)) * entry_size);
	// cudaDeviceSynchronize();
	// EVAL_Pack_last_layer_warp_n<<<N / 4 * blocks_per_grid, threads_per_block>>>(d_key, d_se + N / 4 * 2 * (1 << maxlayer), d_db, blocksum_cuda + N / 4 * 2 * ((1 << n) / (256)) * entry_size);
	// cudaDeviceSynchronize();
	// EVAL_Pack_last_layer_warp_n<<<N / 4 * blocks_per_grid, threads_per_block>>>(d_key, d_se + N / 4 * 3 * (1 << maxlayer), d_db, blocksum_cuda + N / 4 * 3 * ((1 << n) / (256)) * entry_size);
	// cudaDeviceSynchronize();

	if (n - 16 >= 0)
	{
		int CHUNK_SIZE = 1 << (n - 8);
		uint128_t *d_intermediate8;
		cudaMalloc(&d_intermediate8, N * entry_size * CHUNK_SIZE / 256 * sizeof(uint128_t));

		cal_sum_warp<<<(N * CHUNK_SIZE + 255) / 256, 256>>>(blocksum_cuda, d_intermediate8);
		cudaDeviceSynchronize();

		cal_sum_warp<<<N * 1, (CHUNK_SIZE + 255) / 256>>>(d_intermediate8, d_k_res);
		cudaMemcpy(res, d_k_res, N * sizeof(uint128_t) * entry_size, cudaMemcpyDeviceToHost);
	}
	else if ((n - 8 > 0) && (n - 16 < 0))
	{
		cal_sum_warp<<<N, (1 << (n - 8))>>>(blocksum_cuda, d_k_res);
		cudaDeviceSynchronize();
		cudaMemcpy(res, d_k_res, N * sizeof(uint128_t) * entry_size, cudaMemcpyDeviceToHost);
	}

	cudaDeviceSynchronize();

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("归约完成，耗时: %.2f ms\n", milliseconds);
	printf("Throughput: %.2f ms\n", 1000 / milliseconds * N);

	check_mem_usage();

	cudaFree(d_key);
	cudaFree(d_s);
	cudaFree(d_t);
	cudaFree(d_se);
	cudaFree(blocksum_cuda);
	cudaFree(d_db);
	cudaFree(aes_key);

	free(k_res);
	cudaFree(d_k_res);

	// cudaEventDestroy(start);
	// cudaEventDestroy(stop);

	// 检查错误
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		// return -1;
	}

	return res;
}

// #define NUM_STREAMS 10
// #define NUM_CHUNKS 32768
// #define CHUNK_SIZE (1 << 16)

extern "C" uint128_t *test_dpf_pir_pipeline(uint8_t *key, uint128_t *db, int N)
{
	check_mem_usage();

	// 启动kernel
	int n = key[0];
	int maxlayer = n - 7;
	int CHUNK_SIZE = 1 << (n - 8);

	uint8_t *d_key;
	uint128_t *d_s;
	uint32_t *d_t;
	uint128_t *d_db;
	// uint128_t *eval_res = (uint128_t *)malloc(sizeof(uint128_t) * (1 << maxlayer));

	uint128_t *d_se;

	cudaMalloc(&d_db, (1 << n) * entry_size * sizeof(uint128_t));
	cudaMemcpy(d_db, db, (1 << n) * entry_size * sizeof(uint128_t), cudaMemcpyHostToDevice);

	uint128_t *res;
	cudaMallocHost(&res, entry_size * NUM_CHUNKS * sizeof(uint128_t));

	uint32_t *aes_key;
	cudaMalloc(&aes_key, 4 * (14 + 1) * sizeof(uint32_t));
	fss_genaeskey_kernel<<<1, 1>>>(aes_key);

	uint8_t *key_pinned;
	size_t key_size = N * (1 + 16 + 1 + 18 * maxlayer + 16) * sizeof(uint8_t);
	cudaMallocHost(&key_pinned, key_size); // 分配页锁定内存
	memcpy(key_pinned, key, key_size);	   // 复制原始数据到页锁定内存

	cudaMalloc(&d_key, N * (1 + 16 + 1 + 18 * maxlayer + 16) * sizeof(uint8_t));
	cudaMalloc(&d_s, (1 << maxlayer) * 2 * sizeof(uint128_t));
	cudaMalloc(&d_t, (1 << maxlayer) * 2 * sizeof(uint32_t));

	cudaMalloc(&d_se, (1 << maxlayer) * sizeof(uint128_t));

	cudaStream_t streams[NUM_STREAMS];
	cudaEvent_t events[NUM_CHUNKS][28];
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	for (int i = 0; i < NUM_STREAMS; i++)
		cudaStreamCreate(&streams[i]);

	for (int chunk = 0; chunk < NUM_CHUNKS; chunk++)
		for (int j = 0; j < NUM_STREAMS; j++)
			cudaEventCreate(&events[chunk][j]);

	cudaEvent_t keyReadyEvent;
	cudaEventCreate(&keyReadyEvent);

	// uint128_t *d_input, *d_intermediate1, *d_intermediate2, *d_intermediate3, *d_intermediate4, *d_intermediate5, *d_intermediate6, *d_intermediate7, *d_intermediate8, *d_intermediate9, *d_intermediate10, *d_output;
	// uint128_t *d_intermediate11, *d_intermediate12, *d_intermediate13, *d_intermediate14, *d_intermediate15, *d_intermediate16, *d_intermediate17;
	uint128_t *d_input, *d_intermediate4, *d_intermediate8, *d_intermediate16, *d_output;
	cudaMalloc(&d_input, entry_size * CHUNK_SIZE * sizeof(uint128_t));
	// cudaMalloc(&d_intermediate1, entry_size * CHUNK_SIZE / 2 * sizeof(uint128_t));
	// cudaMalloc(&d_intermediate2, entry_size * CHUNK_SIZE / 4 * sizeof(uint128_t));
	// cudaMalloc(&d_intermediate3, entry_size * CHUNK_SIZE / 8 * sizeof(uint128_t));
	cudaMalloc(&d_intermediate4, entry_size * CHUNK_SIZE / 16 * sizeof(uint128_t));
	// cudaMalloc(&d_intermediate5, entry_size * CHUNK_SIZE / 32 * sizeof(uint128_t));
	// cudaMalloc(&d_intermediate6, entry_size * CHUNK_SIZE / 64 * sizeof(uint128_t));
	// cudaMalloc(&d_intermediate7, entry_size * CHUNK_SIZE / 128 * sizeof(uint128_t));
	cudaMalloc(&d_intermediate8, entry_size * CHUNK_SIZE / 256 * sizeof(uint128_t));
	// cudaMalloc(&d_intermediate9, entry_size * CHUNK_SIZE / 512 * sizeof(uint128_t));
	// cudaMalloc(&d_intermediate10, entry_size * CHUNK_SIZE / 1024 * sizeof(uint128_t));
	// cudaMalloc(&d_intermediate11, entry_size * CHUNK_SIZE / 2048 * sizeof(uint128_t));
	// cudaMalloc(&d_intermediate12, entry_size * CHUNK_SIZE / 4096 * sizeof(uint128_t));
	// cudaMalloc(&d_intermediate13, entry_size * CHUNK_SIZE / 8192 * sizeof(uint128_t));
	// cudaMalloc(&d_intermediate14, entry_size * CHUNK_SIZE / 16384 * sizeof(uint128_t));
	// cudaMalloc(&d_intermediate15, entry_size * CHUNK_SIZE / 32768 * sizeof(uint128_t));
	cudaMalloc(&d_intermediate16, entry_size * CHUNK_SIZE / 65536 * sizeof(uint128_t));
	// cudaMalloc(&d_intermediate17, entry_size * CHUNK_SIZE / 131072 * sizeof(uint128_t));

	cudaMalloc(&d_output, entry_size * sizeof(uint128_t));

	int threads_per_block;
	int blocks_per_grid;

	cudaEventRecord(start);

	// cudaMemcpy(d_input, db, entry_size * CHUNK_SIZE * sizeof(uint128_t), cudaMemcpyHostToDevice);

	// 进行流水线式执行
	for (int chunk = 0; chunk < N; chunk++)
	{

		// uint128_t *d_s_offset = d_s + chunk * (1 << maxlayer) * 2;
		// uint32_t *d_t_offset = d_t + chunk * (1 << maxlayer) * 2;
		unsigned char *d_key_offset = d_key + chunk * (1 + 16 + 1 + 18 * maxlayer + 16);
		// bool *d_se_offset = d_se + chunk * (1 << n);
		// uint128_t *d_input_offset = d_input + entry_size * CHUNK_SIZE * chunk;
		// uint128_t *d_output_offset = d_output + entry_size * chunk;
		// uint128_t *d_intermediate8_offset = d_intermediate8 + entry_size * CHUNK_SIZE / 256 * chunk;

		// for (int layer = 1; layer <= maxlayer; layer++)
		// {
		// 	int threads_per_layer = 1 << (layer - 1);
		// 	threads_per_block = min(threads_per_layer, 256);
		// 	blocks_per_grid = (threads_per_layer + threads_per_block - 1) / threads_per_block;
		// 	if (layer == 1)
		// 	{
		// 		cudaMemcpyAsync(d_key, &key_pinned[chunk * (1 + 16 + 1 + 18 * maxlayer + 16)], (1 + 16 + 1 + 18 * maxlayer + 16) * sizeof(uint8_t), cudaMemcpyHostToDevice, streams[1]);
		// 		cudaMemcpyAsync(d_s, &key_pinned[1 + chunk * (1 + 16 + 1 + 18 * maxlayer + 16)], 16, cudaMemcpyHostToDevice, streams[1]);
		// 		cudaMemcpyAsync(d_t, &key_pinned[17 + chunk * (1 + 16 + 1 + 18 * maxlayer + 16)], 1, cudaMemcpyHostToDevice, streams[1]);
		// 		EVAL_Pack_layer<<<1 * blocks_per_grid, threads_per_block, 0, streams[layer]>>>(aes_key, layer, maxlayer, 1, d_key, d_s, d_t);
		// 		cudaEventRecord(events[chunk][layer], streams[layer]);
		// 	}
		// 	else
		// 	{
		// 		cudaStreamWaitEvent(streams[layer], events[chunk][layer - 1], 0);
		// 		EVAL_Pack_layer<<<1 * blocks_per_grid, threads_per_block, 0, streams[layer]>>>(aes_key, layer, maxlayer, 1, d_key, d_s, d_t);
		// 		cudaEventRecord(events[chunk][layer], streams[layer]);
		// 	}
		// }
		if (maxlayer <= 10)
		{
			if (chunk > 0)
				cudaStreamWaitEvent(streams[1], events[chunk - 1][3], 0);
			cudaMemcpyAsync(d_s, &key_pinned[1 + chunk * (1 + 16 + 1 + 18 * maxlayer + 16)], 16, cudaMemcpyHostToDevice, streams[1]);
			cudaMemcpyAsync(d_t, &key_pinned[17 + chunk * (1 + 16 + 1 + 18 * maxlayer + 16)], 1, cudaMemcpyHostToDevice, streams[1]);
			cudaMemcpyAsync(d_key_offset, &key_pinned[chunk * (1 + 16 + 1 + 18 * maxlayer + 16)], (1 + 16 + 1 + 18 * maxlayer + 16) * sizeof(uint8_t), cudaMemcpyHostToDevice, streams[1]);

			Eval_multi_layer<<<1, (1 << maxlayer), 0, streams[1]>>>(aes_key, maxlayer, 0, d_key_offset, d_s, d_t, d_s + (1 << maxlayer) - 1, d_t + (1 << maxlayer) - 1);

			cudaEventRecord(events[chunk][1], streams[1]);
		}
		else if (maxlayer > 10)
		{
			if (chunk > 0)
				cudaStreamWaitEvent(streams[1], events[chunk - 1][2], 0);
			cudaMemcpyAsync(d_s, &key_pinned[1 + chunk * (1 + 16 + 1 + 18 * maxlayer + 16)], 16, cudaMemcpyHostToDevice, streams[1]);
			cudaMemcpyAsync(d_t, &key_pinned[17 + chunk * (1 + 16 + 1 + 18 * maxlayer + 16)], 1, cudaMemcpyHostToDevice, streams[1]);
			cudaMemcpyAsync(d_key_offset, &key_pinned[chunk * (1 + 16 + 1 + 18 * maxlayer + 16)], (1 + 16 + 1 + 18 * maxlayer + 16) * sizeof(uint8_t), cudaMemcpyHostToDevice, streams[1]);

			Eval_multi_layer<<<1, (1 << (maxlayer - 10)), 0, streams[1]>>>(aes_key, maxlayer - 10, 0, d_key_offset, d_s, d_t, d_s + (1 << (maxlayer - 10) - 1), d_t + (1 << (maxlayer - 10) - 1));
			cudaEventRecord(events[chunk][1], streams[1]);

			if (chunk > 0)
				cudaStreamWaitEvent(streams[2], events[chunk - 1][3], 0);
			cudaStreamWaitEvent(streams[2], events[chunk][1], 0);
			Eval_multi_layer<<<(1 << (maxlayer - 10)), (1 << 10), 0, streams[2]>>>(aes_key, 10, maxlayer - 10, d_key_offset, d_s + (1 << (maxlayer - 10) - 1), d_t + (1 << (maxlayer - 10) - 1), d_s + (1 << maxlayer) - 1, d_t + (1 << maxlayer) - 1);
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
		EVAL_Pack_last_layer_gense<<<blocks_per_grid, threads_per_block, 0, streams[3]>>>(d_key_offset, d_s, d_t, d_se);
		cudaEventRecord(events[chunk][3], streams[3]);
		// cudaStreamSynchronize(streams[3]);
		// 在这里记录事件，确保 d_key 使用完成
		// cudaEventRecord(keyReadyEvent, streams[maxlayer + 1]);

		threads_per_block = 256;
		blocks_per_grid = (1 << n) / threads_per_block;
		if (chunk > 0)
			cudaStreamWaitEvent(streams[4], events[chunk - 1][5], 0);

		cudaStreamWaitEvent(streams[4], events[chunk][3], 0);
		if (n > 16)
		{
			// EVAL_Pack_last_layer_warp_once<<<blocks_per_grid, threads_per_block, 0, streams[maxlayer + 5]>>>(n, d_se, d_db, d_input);
			EVAL_Pack_last_layer_warp_n<<<blocks_per_grid, threads_per_block, 0, streams[4]>>>(d_key_offset, d_se, d_db, d_input);
			// EVAL_Pack_last_layer_warp_n<<<blocks_per_grid / 2, threads_per_block, 0, streams[8]>>>(d_key_offset, d_se + (1 << maxlayer) / 2, d_db + (1 << n) * entry_size / 2, d_input + entry_size * CHUNK_SIZE / 2);
		}
		else if (n <= 16)
			EVAL_Pack_last_layer_warp_n<<<blocks_per_grid, threads_per_block, 0, streams[4]>>>(d_key_offset, d_se, d_db, d_intermediate4);
		cudaEventRecord(events[chunk][4], streams[4]);

		if (n <= 16)
		{

			cudaStreamWaitEvent(streams[5], events[chunk][4], 0);
			// if (chunk > 0)
			// 	cudaStreamWaitEvent(streams[5], events[chunk - 1][6], 0);
			cal_sum_warp<<<1, ((1 << n) + 255) / 256, 0, streams[5]>>>(d_intermediate4, d_output);
			cudaEventRecord(events[chunk][5], streams[5]);
			cudaMemcpyAsync(res + entry_size * chunk * 1, d_output, entry_size * 1 * sizeof(uint128_t), cudaMemcpyDeviceToHost, streams[5]);
		}
		else if (n > 16 && n <= 24)
		{

			cudaStreamWaitEvent(streams[5], events[chunk][4], 0);
			// if (chunk > 0)
			// 	cudaStreamWaitEvent(streams[5], events[chunk - 1][6], 0);
			cal_sum_warp<<<(CHUNK_SIZE + 255) / 256, 256, 0, streams[5]>>>(d_input, d_intermediate8);
			cudaEventRecord(events[chunk][5], streams[5]);

			cudaStreamWaitEvent(streams[6], events[chunk][5], 0);
			cal_sum_warp<<<1, (CHUNK_SIZE + 255) / 256, 0, streams[6]>>>(d_intermediate8, d_output);

			cudaEventRecord(events[chunk][6], streams[6]);
			cudaMemcpyAsync(res + entry_size * chunk * 1, d_output, entry_size * 1 * sizeof(uint128_t), cudaMemcpyDeviceToHost, streams[6]);
		}
		else if (n > 24 && n <= 28)
		{
			cudaStreamWaitEvent(streams[5], events[chunk][4], 0);
			// if (chunk > 0)
			// 	cudaStreamWaitEvent(streams[5], events[chunk - 1][6], 0);
			cal_sum_warp<<<(CHUNK_SIZE + 255) / 256, 256, 0, streams[5]>>>(d_input, d_intermediate8);
			cudaEventRecord(events[chunk][5], streams[5]);

			cudaStreamWaitEvent(streams[6], events[chunk][5], 0);
			// if (chunk > 0)
			// 	cudaStreamWaitEvent(streams[6], events[chunk - 1][7], 0);
			cal_sum_warp<<<(CHUNK_SIZE + 255) / 65536, 256, 0, streams[6]>>>(d_intermediate8, d_intermediate16);
			cudaEventRecord(events[chunk][6], streams[6]);

			cudaStreamWaitEvent(streams[7], events[chunk][6], 0);
			cal_sum_warp<<<1, (CHUNK_SIZE + 255) / 65536, 0, streams[7]>>>(d_intermediate16, d_output);
			cudaEventRecord(events[chunk][7], streams[7]);

			cudaMemcpyAsync(res + entry_size * chunk * 1, d_output, entry_size * 1 * sizeof(uint128_t), cudaMemcpyDeviceToHost, streams[7]);
		}

		// // 流水线执行
		// cudaStreamWaitEvent(streams[5], events[chunk][maxlayer + 2], 0);
		// half_reduction_full_stride<<<(CHUNK_SIZE + 255) / 256, 256, 0, streams[5]>>>(d_input, d_intermediate1);
		// cudaEventRecord(events[chunk][5], streams[5]);

		// // 第二个核函数依赖于第一个核函数的输出
		// cudaStreamWaitEvent(streams[6], events[chunk][5], 0);
		// half_reduction_full_stride<<<(CHUNK_SIZE / 2 + 255) / 256, 256, 0, streams[6]>>>(d_intermediate1, d_intermediate2);
		// cudaEventRecord(events[chunk][6], streams[6]);

		// // 第三个核函数依赖于第二个核函数的输出
		// cudaStreamWaitEvent(streams[maxlayer + 5], events[chunk][6], 0);
		// half_reduction_full_stride<<<(CHUNK_SIZE / 4 + 255) / 256, 256, 0, streams[maxlayer + 5]>>>(d_intermediate2, d_intermediate3);
		// cudaEventRecord(events[chunk][maxlayer + 5], streams[maxlayer + 5]);

		// // 第四个核函数依赖于第三个核函数的输出
		// cudaStreamWaitEvent(streams[7], events[chunk][maxlayer + 5], 0);
		// half_reduction_full_stride<<<(CHUNK_SIZE / 8 + 255) / 256, 256, 0, streams[7]>>>(d_intermediate3, d_intermediate4);
		// cudaEventRecord(events[chunk][7], streams[7]);

		// // 第五个核函数依赖于第四个核函数的输出
		// cudaStreamWaitEvent(streams[maxlayer + 7], events[chunk][7], 0);
		// half_reduction_full_stride<<<(CHUNK_SIZE / 16 + 255) / 256, 256, 0, streams[maxlayer + 7]>>>(d_intermediate4, d_intermediate5);
		// cudaEventRecord(events[chunk][maxlayer + 7], streams[maxlayer + 7]);

		// cudaStreamWaitEvent(streams[maxlayer + 8], events[chunk][maxlayer + 7], 0);
		// half_reduction_full_stride<<<1, 128, 0, streams[maxlayer + 8]>>>(d_intermediate5, d_intermediate6);
		// cudaEventRecord(events[chunk][maxlayer + 8], streams[maxlayer + 8]);

		// cudaStreamWaitEvent(streams[maxlayer + 9], events[chunk][maxlayer + 8], 0);
		// half_reduction_full_stride<<<1, 64, 0, streams[maxlayer + 9]>>>(d_intermediate6, d_intermediate7);
		// cudaEventRecord(events[chunk][maxlayer + 9], streams[maxlayer + 9]);

		// cudaStreamWaitEvent(streams[maxlayer + 10], events[chunk][maxlayer + 9], 0);
		// half_reduction_full_stride<<<1, 32, 0, streams[maxlayer + 10]>>>(d_intermediate7, d_intermediate8);
		// cudaEventRecord(events[chunk][maxlayer + 10], streams[maxlayer + 10]);

		// cudaStreamWaitEvent(streams[maxlayer + 11], events[chunk][maxlayer + 10], 0);
		// half_reduction_8_stride<<<1, 16, 0, streams[maxlayer + 11]>>>(d_intermediate8, d_intermediate9);
		// cudaEventRecord(events[chunk][maxlayer + 11], streams[maxlayer + 11]);

		// cudaStreamWaitEvent(streams[maxlayer + 12], events[chunk][maxlayer + 11], 0);
		// half_reduction_4_stride<<<1, 8, 0, streams[maxlayer + 12]>>>(d_intermediate9, d_intermediate10);
		// cudaEventRecord(events[chunk][maxlayer + 12], streams[maxlayer + 12]);

		// cudaStreamWaitEvent(streams[maxlayer + 13], events[chunk][maxlayer + 12], 0);
		// half_reduction_2_stride<<<1, 4, 0, streams[maxlayer + 13]>>>(d_intermediate10, d_intermediate11);
		// cudaEventRecord(events[chunk][maxlayer + 13], streams[maxlayer + 13]);

		// cudaStreamWaitEvent(streams[maxlayer + 14], events[chunk][maxlayer + 13], 0);
		// half_reduction_1_stride<<<1, 2, 0, streams[maxlayer + 14]>>>(d_intermediate11, d_output);

		// 最后将输出数据从设备复制回主机
		// // 等待 d_key 使用完成后，再传输下一组 d_key
		// cudaStreamWaitEvent(streams[1], keyReadyEvent, 0);
	}

	for (int i = 0; i < NUM_STREAMS; i++)
		cudaStreamSynchronize(streams[i]);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds_pipeline = 0;
	cudaEventElapsedTime(&milliseconds_pipeline, start, stop);
	printf("Pipeline Execution Time: %.2f ms\n", milliseconds_pipeline);
	printf("Throughput: %.2f ms\n", 1000 / milliseconds_pipeline * N);

	check_mem_usage();

	cudaFree(d_key);
	cudaFree(d_s);
	cudaFree(d_t);
	cudaFree(d_se);
	cudaFree(d_db);
	cudaFree(aes_key);
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_intermediate8);
	cudaFree(d_intermediate4);
	cudaFree(d_intermediate16);

	cudaFreeHost(key_pinned);

	for (int chunk = 0; chunk < NUM_CHUNKS; chunk++)
		for (int j = 0; j < NUM_STREAMS; j++)
			cudaEventDestroy(events[chunk][j]);
	for (int i = 0; i < NUM_STREAMS; i++)
		cudaStreamDestroy(streams[i]);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// 检查错误
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		// return -1;
	}

	return res;
}

// CUDA warmup函数
extern "C" void cudaWarmup(int size, int party)
{

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	printf("Number of CUDA devices: %d\n", deviceCount);

	int deviceToUse = party; // 指定要使用的设备编号
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
	// 分配设备内存
	cudaMalloc(&d_data, bytes);

	// 定义CUDA核函数的网格和块大小
	int blockSize = 256;
	int gridSize = (size + blockSize - 1) / blockSize;

	// 启动CUDA核函数
	warmupKernel<<<gridSize, blockSize>>>(d_data, size);

	// 同步设备，确保核函数执行完成
	cudaDeviceSynchronize();

	// 释放设备内存
	cudaFree(d_data);
}
