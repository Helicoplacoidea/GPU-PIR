#include <stdio.h>

#include <string.h>
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
		shared_s[0] = s_local[blockIdx.x % (1 << layer)];
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
