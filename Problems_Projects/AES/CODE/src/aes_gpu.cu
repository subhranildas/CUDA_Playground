#/* =============================================================================
 * aes_gpu.cu
 *
 * CUDA accelerated AES helpers and ECB-mode wrappers.
 *
 * This file contains device-side S-boxes, AES transformation helpers
 * implemented on the GPU, single-block encrypt/decrypt device routines,
 * ECB-mode kernels (one thread per 16-byte block), and host wrapper
 * functions `aes_encrypt_ecb_cuda` and `aes_decrypt_ecb_cuda`.
 *
 * Design:
 * - Device helpers operate on a 4x4 state matrix stored in column-major order.
 * - Host wrappers expand keys on the host, copy expanded keys + data to device,
 *   launch kernels, synchronize and copy results back.
 *
 * Notes:
 * - ECB mode is used for simplicity and correctness testing only. It is not
 *   recommended for production use.
 * - The CPU reference implementation is in `aes_cpu.cpp` and should be used
 *   as the canonical source of truth for correctness.
 ============================================================================ */
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"
#include "aes.h"
using namespace std;

/*==============================================================================
 * Simple CUDA error check (This is something I have taken from the internet)
 * Seemed very useful as I got some segmentation fault during runtime
==============================================================================*/
#define CUDA_CHECK(call)                                          \
	do                                                            \
	{                                                             \
		cudaError_t err = (call);                                 \
		if (err != cudaSuccess)                                   \
		{                                                         \
			fprintf(stderr, "CUDA error %s:%d: %s\n",             \
					__FILE__, __LINE__, cudaGetErrorString(err)); \
			exit(EXIT_FAILURE);                                   \
		}                                                         \
	} while (0)

/* Device-side S-box (Same as CPU side) */
__device__ __constant__ uint8_t d_sbox[256] = {
	// 0	 1	   2   	 3	   4	 5	   6	 7	   8	 9	   A	 B	   C 	 D	   E	 F
	0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
	0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
	0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
	0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
	0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
	0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
	0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
	0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
	0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
	0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
	0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
	0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
	0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
	0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
	0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
	0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16};

/* Device-side inverse S-box (copied from CPU inverse table) */
__device__ __constant__ uint8_t d_inv_sbox[256] = {
	// 0	 1	   2   	 3	   4	 5	   6	 7	   8	 9	   A	 B	   C 	 D	   E	 F
	0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
	0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
	0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
	0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
	0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
	0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
	0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
	0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
	0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
	0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
	0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
	0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
	0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
	0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
	0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
	0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d};

/*==============================================================================
 * GPU Clamp Function
==============================================================================*/

/* =============================================================================
 * d_galois_mul()
 *
 * Multiply two bytes in GF(2^8) using the AES reducing polynomial (0x1B).
 *
 * Parameters:
 *   a, b  - input bytes to multiply in GF(2^8)
 *
 * Returns:
 *   The product a * b computed in GF(2^8).
 *
 * Notes:
 *   - Implemented bitwise to avoid lookup tables on device.
 *   - Marked __forceinline__ to encourage inlining in performance-sensitive
 *     code.
 ============================================================================ */
__device__ __forceinline__ uint8_t d_galois_mul(uint8_t a, uint8_t b)
{
	uint8_t p = 0;
	for (int i = 0; i < 8; i++)
	{
		if (b & 1)
			p ^= a;
		uint8_t hi = a & 0x80;
		a <<= 1;
		if (hi)
			a ^= 0x1B; /* AES reducing poly */
		b >>= 1;
	}
	return p;
}

/* =============================================================================
 * Device Transform Helpers
 *
 * The following device functions implement AES state transformations:
 *   - SubBytes / InvSubBytes
 *   - ShiftRows / InvShiftRows
 *   - MixColumns / InvMixColumns
 *   - AddRoundKey
 *
 * Each function operates on a 4x4 state array in column-major order.
 ============================================================================ */

/* SubBytes: apply the forward S-box to each byte in the state. */
__device__ void d_substitute_bytes(uint8_t state[4][4])
{
	for (int r = 0; r < 4; ++r)
		for (int c = 0; c < 4; ++c)
			state[r][c] = d_sbox[state[r][c]];
}

/* InvSubBytes: apply the inverse S-box to each byte in the state. */
__device__ void d_inv_substitute_bytes(uint8_t state[4][4])
{
	for (int r = 0; r < 4; ++r)
		for (int c = 0; c < 4; ++c)
			state[r][c] = d_inv_sbox[state[r][c]];
}

/* AddRoundKey: XOR the 16-byte round key into the state. */
__device__ void d_add_round_key(uint8_t state[4][4], const uint8_t *round_key)
{
	for (int c = 0; c < 4; ++c)
		for (int r = 0; r < 4; ++r)
			state[r][c] ^= round_key[c * 4 + r];
}

/* ShiftRows:
 *
 * Cyclically rotate each row of the state left by a fixed offset:
 *  - Row 0: 0 bytes
 *  - Row 1: 1 byte
 *  - Row 2: 2 bytes
 *  - Row 3: 3 bytes
 */
__device__ void d_shift_rows(uint8_t state[4][4])
{
	uint8_t tmp;
	/* Row 1: rotate left by 1 */
	tmp = state[1][0];
	state[1][0] = state[1][1];
	state[1][1] = state[1][2];
	state[1][2] = state[1][3];
	state[1][3] = tmp;

	/* Row 2: rotate left by 2 */
	tmp = state[2][0];
	state[2][0] = state[2][2];
	state[2][2] = tmp;
	tmp = state[2][1];
	state[2][1] = state[2][3];
	state[2][3] = tmp;

	/* Row 3: rotate left by 3 (or right by 1) */
	tmp = state[3][3];
	state[3][3] = state[3][2];
	state[3][2] = state[3][1];
	state[3][1] = state[3][0];
	state[3][0] = tmp;
}

/* InvShiftRows:
 *
 * Inverse of ShiftRows: cyclically rotate rows to the right by the same
 * offsets used in ShiftRows. Used during decryption.
 */
__device__ void d_inv_shift_rows(uint8_t state[4][4])
{
	uint8_t tmp;
	/* Row 1: rotate right by 1 */
	tmp = state[1][3];
	state[1][3] = state[1][2];
	state[1][2] = state[1][1];
	state[1][1] = state[1][0];
	state[1][0] = tmp;

	/* Row 2: rotate right by 2 */
	tmp = state[2][0];
	state[2][0] = state[2][2];
	state[2][2] = tmp;
	tmp = state[2][1];
	state[2][1] = state[2][3];
	state[2][3] = tmp;

	/* Row 3: rotate right by 3 (or left by 1) */
	tmp = state[3][0];
	state[3][0] = state[3][1];
	state[3][1] = state[3][2];
	state[3][2] = state[3][3];
	state[3][3] = tmp;
}

/* =============================================================================
 * MixColumns:
 *
 * Transform each column of the state by multiplying it with the fixed
 * AES matrix in GF(2^8):
 *   |02 03 01 01|
 *   |01 02 03 01|
 *   |01 01 02 03|
 *   |03 01 01 02|
 *
 * This mixes bytes within each column and provides diffusion.
 ============================================================================ */
__device__ void d_mix_columns(uint8_t state[4][4])
{
	uint8_t t[4];
	for (int c = 0; c < 4; ++c)
	{
		for (int r = 0; r < 4; ++r)
			t[r] = state[r][c];

		state[0][c] = d_galois_mul(t[0], 2) ^ d_galois_mul(t[1], 3) ^ t[2] ^ t[3];
		state[1][c] = t[0] ^ d_galois_mul(t[1], 2) ^ d_galois_mul(t[2], 3) ^ t[3];
		state[2][c] = t[0] ^ t[1] ^ d_galois_mul(t[2], 2) ^ d_galois_mul(t[3], 3);
		state[3][c] = d_galois_mul(t[0], 3) ^ t[1] ^ t[2] ^ d_galois_mul(t[3], 2);
	}
}

/* =============================================================================
 * InvMixColumns:
 *
 * Apply the inverse MixColumns matrix used during decryption:
 *   |0E 0B 0D 09|
 *   |09 0E 0B 0D|
 *   |0D 09 0E 0B|
 *   |0B 0D 09 0E|
 ============================================================================ */
__device__ void d_inv_mix_columns(uint8_t state[4][4])
{
	uint8_t t[4];
	for (int c = 0; c < 4; ++c)
	{
		for (int r = 0; r < 4; ++r)
			t[r] = state[r][c];

		state[0][c] = d_galois_mul(t[0], 0x0e) ^ d_galois_mul(t[1], 0x0b) ^ d_galois_mul(t[2], 0x0d) ^ d_galois_mul(t[3], 0x09);
		state[1][c] = d_galois_mul(t[0], 0x09) ^ d_galois_mul(t[1], 0x0e) ^ d_galois_mul(t[2], 0x0b) ^ d_galois_mul(t[3], 0x0d);
		state[2][c] = d_galois_mul(t[0], 0x0d) ^ d_galois_mul(t[1], 0x09) ^ d_galois_mul(t[2], 0x0e) ^ d_galois_mul(t[3], 0x0b);
		state[3][c] = d_galois_mul(t[0], 0x0b) ^ d_galois_mul(t[1], 0x0d) ^ d_galois_mul(t[2], 0x09) ^ d_galois_mul(t[3], 0x0e);
	}
}

/* =============================================================================
 * d_cipher_encrypt_block()
 *
 * Device-side single-block AES encryption. Implements the FIPS-197
 * encryption sequence for one 128-bit block using an expanded key
 * stored in device memory.
 *
 * Parameters:
 *   state_in      - pointer to 16 input bytes (plaintext block)
 *   state_out     - pointer to 16 output bytes (ciphertext block)
 *   expanded_key  - pointer to (Nr+1)*16 bytes of round keys in device memory
 *   num_rounds    - number of AES rounds (10/12/14)
 *
 * Notes:
 *   - The expanded_key layout is contiguous round keys: round0, round1, ...
 *   - State is processed in column-major order to match the CPU reference.
 ============================================================================ */
__device__ void d_cipher_encrypt_block(uint8_t state_in[16],
									   uint8_t state_out[16],
									   const uint8_t *expanded_key,
									   uint16_t num_rounds)
{
	/* Load into 4x4 state column-major */
	uint8_t state[4][4];
	for (int r = 0; r < 4; ++r)
		for (int c = 0; c < 4; ++c)
			state[r][c] = state_in[r + 4 * c];

	/* Initial AddRoundKey */
	d_add_round_key(state, expanded_key);

	/* Main rounds */
	for (uint16_t round = 1; round < num_rounds; ++round)
	{
		d_substitute_bytes(state);
		d_shift_rows(state);
		d_mix_columns(state);
		d_add_round_key(state, expanded_key + (AES_BLOCK_SIZE * round));
	}

	/* Final round (no MixColumns) */
	d_substitute_bytes(state);
	d_shift_rows(state);
	d_add_round_key(state, expanded_key + (AES_BLOCK_SIZE * num_rounds));

	/* Store back to state_out */
	for (int r = 0; r < 4; ++r)
		for (int c = 0; c < 4; ++c)
			state_out[r + 4 * c] = state[r][c];
}

/* Device AES single-block decrypt using expanded_key stored in device global
 * memory */
/* =============================================================================
 * d_cipher_decrypt_block()
 *
 * Device-side single-block AES decryption. Implements the inverse AES
 * transformations to convert a single 128-bit ciphertext block back to
 * plaintext using the expanded round keys stored in device memory.
 *
 * Parameters:
 *   state_in      - pointer to 16 input bytes (ciphertext block)
 *   state_out     - pointer to 16 output bytes (plaintext block)
 *   expanded_key  - pointer to (Nr+1)*16 bytes of round keys in device memory
 *   num_rounds    - number of AES rounds (10/12/14)
 *
 * Notes:
 *   - The implementation mirrors the CPU reference's decryption order.
 ============================================================================ */
__device__ void d_cipher_decrypt_block(uint8_t state_in[16],
									   uint8_t state_out[16],
									   const uint8_t *expanded_key,
									   uint16_t num_rounds)
{
	/* Load into 4x4 state column-major */
	uint8_t state[4][4];
	for (int r = 0; r < 4; ++r)
		for (int c = 0; c < 4; ++c)
			state[r][c] = state_in[r + 4 * c];

	/* Initial AddRoundKey with last round key */
	d_add_round_key(state, expanded_key + (AES_BLOCK_SIZE * num_rounds));

	/* Main rounds (Nr-1 down to 1) */
	for (int round = num_rounds - 1; round >= 1; --round)
	{
		d_inv_shift_rows(state);
		d_inv_substitute_bytes(state);
		d_add_round_key(state, expanded_key + (AES_BLOCK_SIZE * round));
		d_inv_mix_columns(state);
	}

	/* Final round (InvShift + InvSub + AddRoundKey(0)) */
	d_inv_shift_rows(state);
	d_inv_substitute_bytes(state);
	d_add_round_key(state, expanded_key);

	/* Store back to state_out */
	for (int r = 0; r < 4; ++r)
		for (int c = 0; c < 4; ++c)
			state_out[r + 4 * c] = state[r][c];
}

/* =============================================================================
 * Kernel: aes_ecb_encrypt_kernel
 *
 * ECB-mode encryption kernel where each GPU thread encrypts exactly one
 * 16-byte AES block. This mapping keeps the kernel simple and is useful
 * for correctness testing. For high throughput, consider larger per-thread
 * workloads or cooperative approaches.
 *
 * Parameters:
 *   d_input        - device pointer to plaintext bytes (length = num_blocks*16)
 *   d_output       - device pointer to output ciphertext bytes
 *   d_expanded_key - device pointer to expanded round keys
 *   num_rounds     - number of AES rounds
 *   num_blocks     - total number of 16-byte blocks to process
 ============================================================================ */
__global__ void aes_ecb_encrypt_kernel(const uint8_t *d_input,
									   uint8_t *d_output,
									   const uint8_t *d_expanded_key,
									   uint16_t num_rounds, size_t num_blocks)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_blocks)
		return;

	/* Pointers to this block */
	const uint8_t *inblk = d_input + idx * AES_BLOCK_SIZE;
	uint8_t outblk[AES_BLOCK_SIZE];

	/* Call device block encrypt */
	d_cipher_encrypt_block((uint8_t *)inblk, outblk, d_expanded_key, num_rounds);

	/* Write result */
	uint8_t *outptr = d_output + idx * AES_BLOCK_SIZE;
#pragma unroll
	for (int i = 0; i < AES_BLOCK_SIZE; ++i)
		outptr[i] = outblk[i];
}

/* =============================================================================
 * Kernel: aes_ecb_decrypt_kernel
 *
 * ECB-mode decryption kernel where each GPU thread decrypts exactly one
 * 16-byte AES block.
 *
 * Parameters are analogous to the encrypt kernel.
 ============================================================================ */
__global__ void aes_ecb_decrypt_kernel(const uint8_t *d_input,
									   uint8_t *d_output,
									   const uint8_t *d_expanded_key,
									   uint16_t num_rounds, size_t num_blocks)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_blocks)
		return;

	const uint8_t *inblk = d_input + idx * AES_BLOCK_SIZE;
	uint8_t outblk[AES_BLOCK_SIZE];

	d_cipher_decrypt_block((uint8_t *)inblk, outblk, d_expanded_key,
						   num_rounds);

	uint8_t *outptr = d_output + idx * AES_BLOCK_SIZE;
#pragma unroll
	for (int i = 0; i < AES_BLOCK_SIZE; ++i)
		outptr[i] = outblk[i];
}

/* =============================================================================
 * aes_encrypt_ecb_cuda()
 *
 * Host wrapper that performs ECB encryption on the GPU.
 *
 * Steps performed:
 *  1. Validate inputs and key size.
 *  2. Expand the AES key on the host using `aes_expand_key()`.
 *  3. Allocate device buffers for input, output and expanded key.
 *  4. Copy plaintext and expanded key to device memory.
 *  5. Launch `aes_ecb_encrypt_kernel` with one thread per 16-byte block.
 *  6. Synchronize and copy ciphertext back to host.
 *  7. Zeroize and free host/device buffers.
 *
 * Parameters:
 *   plaintext   - pointer to input plaintext bytes (length must be multiple of
 *                 16)
 *   length      - number of bytes to encrypt
 *   ciphertext  - output buffer (same length)
 *   key         - AES key bytes
 *   key_size    - AES key size enum (128/192/256)
 *
 * Returns:
 *   AES_SUCCESS on success, or an `aes_error_te` enum value on failure
 *   (memory errors, invalid args, or CUDA failures).
 ============================================================================ */
aes_error_te aes_encrypt_ecb_cuda(const uint8_t *plaintext, size_t length,
								  uint8_t *ciphertext,
								  const uint8_t *key, aes_key_size_te key_size)
{
	if (!plaintext || !ciphertext || !key)
		return AES_ERROR_UNSUPPORTED_KEY_SIZE;

	if (length % AES_BLOCK_SIZE != 0)
		return AES_ERROR_INVALID_INPUT_LENGTH;

	uint16_t num_rounds;
	switch (key_size)
	{
	case AES_KEY_SIZE_128:
		num_rounds = AES_ROUNDS_128;
		break;
	case AES_KEY_SIZE_192:
		num_rounds = AES_ROUNDS_192;
		break;
	case AES_KEY_SIZE_256:
		num_rounds = AES_ROUNDS_256;
		break;
	default:
		return AES_ERROR_UNSUPPORTED_KEY_SIZE;
	}

	size_t num_blocks = length / AES_BLOCK_SIZE;
	size_t expanded_key_size = (size_t)AES_BLOCK_SIZE * (num_rounds + 1);

	/* Expand key on host using existing function */
	uint8_t *h_expanded_key = (uint8_t *)malloc(expanded_key_size);
	if (!h_expanded_key)
		return AES_ERROR_MEMORY_ALLOCATION_FAILED;
	aes_expand_key(h_expanded_key, key, key_size, expanded_key_size);

	/* Device allocations */
	uint8_t *d_input = NULL, *d_output = NULL, *d_expanded_key = NULL;
	cudaError_t cerr;

	cerr = cudaMalloc((void **)&d_input, length);
	if (cerr != cudaSuccess)
	{
		free(h_expanded_key);
		return AES_ERROR_MEMORY_ALLOCATION_FAILED;
	}

	cerr = cudaMalloc((void **)&d_output, length);
	if (cerr != cudaSuccess)
	{
		cudaFree(d_input);
		free(h_expanded_key);
		return AES_ERROR_MEMORY_ALLOCATION_FAILED;
	}

	cerr = cudaMalloc((void **)&d_expanded_key, expanded_key_size);
	if (cerr != cudaSuccess)
	{
		cudaFree(d_input);
		cudaFree(d_output);
		free(h_expanded_key);
		return AES_ERROR_MEMORY_ALLOCATION_FAILED;
	}

	/* Copy plaintext and expanded key to device */
	cerr = cudaMemcpy(d_input, plaintext, length,
					  cudaMemcpyHostToDevice);
	if (cerr != cudaSuccess)
	{
		cudaFree(d_input);
		cudaFree(d_output);
		cudaFree(d_expanded_key);
		free(h_expanded_key);
		return AES_ERROR_MEMORY_ALLOCATION_FAILED;
	}

	cerr = cudaMemcpy(d_expanded_key, h_expanded_key, expanded_key_size,
					  cudaMemcpyHostToDevice);
	if (cerr != cudaSuccess)
	{
		cudaFree(d_input);
		cudaFree(d_output);
		cudaFree(d_expanded_key);
		free(h_expanded_key);
		return AES_ERROR_MEMORY_ALLOCATION_FAILED;
	}

	/* Launch kernel */
	const int threads_per_block = 256;
	int blocks = (int)((num_blocks + threads_per_block - 1) /
					   threads_per_block);

	aes_ecb_encrypt_kernel<<<blocks, threads_per_block>>>(d_input, d_output,
														  d_expanded_key,
														  num_rounds,
														  num_blocks);
	/* Check for launch errors and synchronize to catch runtime errors */
	cerr = cudaGetLastError();
	if (cerr != cudaSuccess)
	{
		cudaFree(d_input);
		cudaFree(d_output);
		cudaFree(d_expanded_key);
		memory_write_zeros(h_expanded_key, expanded_key_size);
		free(h_expanded_key);
		return AES_ERROR_CUDA_FAILURE;
	}

	/* Ensure kernel completed successfully */
	cerr = cudaDeviceSynchronize();
	if (cerr != cudaSuccess)
	{
		cudaFree(d_input);
		cudaFree(d_output);
		cudaFree(d_expanded_key);
		memory_write_zeros(h_expanded_key, expanded_key_size);
		free(h_expanded_key);
		return AES_ERROR_CUDA_FAILURE;
	}

	/* Copy ciphertext back */
	cerr = cudaMemcpy(ciphertext, d_output, length, cudaMemcpyDeviceToHost);
	if (cerr != cudaSuccess)
	{
		cudaFree(d_input);
		cudaFree(d_output);
		cudaFree(d_expanded_key);
		memory_write_zeros(h_expanded_key, expanded_key_size);
		free(h_expanded_key);
		return AES_ERROR_CUDA_FAILURE;
	}

	/* Clean up */
	cudaFree(d_input);
	cudaFree(d_output);

	/* Zeroize and free expanded key on device & host */
	cudaMemset(d_expanded_key, 0, expanded_key_size);
	cudaFree(d_expanded_key);

	memory_write_zeros(h_expanded_key, expanded_key_size);
	free(h_expanded_key);

	return AES_SUCCESS;
}

/* =============================================================================
 * aes_decrypt_ecb_cuda()
 *
 * Host wrapper that performs ECB decryption on the GPU. Mirrors the steps of
 * `aes_encrypt_ecb_cuda()` but launches the decryption kernel.
 *
 * Steps performed:
 *  1. Validate inputs and key size.
 *  2. Expand AES key on host.
 *  3. Allocate and copy ciphertext + expanded key to device memory.
 *  4. Launch `aes_ecb_decrypt_kernel`.
 *  5. Synchronize and copy plaintext back to host.
 *  6. Zeroize and free buffers.
 *
 * Parameters:
 *   ciphertext  - pointer to input ciphertext bytes (length must be multiple
 *                 of 16)
 *   length      - number of bytes to decrypt
 *   plaintext   - output buffer (same length)
 *   key         - AES key bytes
 *   key_size    - AES key size enum (128/192/256)
 *
 * Returns:
 *   AES_SUCCESS on success, or an `aes_error_te` enum on failure.
 ============================================================================ */
aes_error_te aes_decrypt_ecb_cuda(const uint8_t *ciphertext, size_t length,
								  uint8_t *plaintext,
								  const uint8_t *key, aes_key_size_te key_size)
{
	if (!ciphertext || !plaintext || !key)
		return AES_ERROR_UNSUPPORTED_KEY_SIZE;

	if (length % AES_BLOCK_SIZE != 0)
		return AES_ERROR_INVALID_INPUT_LENGTH;

	uint16_t num_rounds;
	switch (key_size)
	{
	case AES_KEY_SIZE_128:
		num_rounds = AES_ROUNDS_128;
		break;
	case AES_KEY_SIZE_192:
		num_rounds = AES_ROUNDS_192;
		break;
	case AES_KEY_SIZE_256:
		num_rounds = AES_ROUNDS_256;
		break;
	default:
		return AES_ERROR_UNSUPPORTED_KEY_SIZE;
	}

	size_t num_blocks = length / AES_BLOCK_SIZE;
	size_t expanded_key_size = (size_t)AES_BLOCK_SIZE * (num_rounds + 1);

	/* Expand key on host */
	uint8_t *h_expanded_key = (uint8_t *)malloc(expanded_key_size);
	if (!h_expanded_key)
		return AES_ERROR_MEMORY_ALLOCATION_FAILED;
	aes_expand_key(h_expanded_key, key, key_size, expanded_key_size);

	/* Device allocations */
	uint8_t *d_input = NULL, *d_output = NULL, *d_expanded_key = NULL;
	cudaError_t cerr;

	cerr = cudaMalloc((void **)&d_input, length);
	if (cerr != cudaSuccess)
	{
		free(h_expanded_key);
		return AES_ERROR_MEMORY_ALLOCATION_FAILED;
	}

	cerr = cudaMalloc((void **)&d_output, length);
	if (cerr != cudaSuccess)
	{
		cudaFree(d_input);
		free(h_expanded_key);
		return AES_ERROR_MEMORY_ALLOCATION_FAILED;
	}

	cerr = cudaMalloc((void **)&d_expanded_key, expanded_key_size);
	if (cerr != cudaSuccess)
	{
		cudaFree(d_input);
		cudaFree(d_output);
		free(h_expanded_key);
		return AES_ERROR_MEMORY_ALLOCATION_FAILED;
	}

	/* Copy ciphertext and expanded key to device */
	cerr = cudaMemcpy(d_input, ciphertext, length, cudaMemcpyHostToDevice);
	if (cerr != cudaSuccess)
	{
		cudaFree(d_input);
		cudaFree(d_output);
		cudaFree(d_expanded_key);
		free(h_expanded_key);
		return AES_ERROR_CUDA_FAILURE;
	}

	cerr = cudaMemcpy(d_expanded_key, h_expanded_key, expanded_key_size,
					  cudaMemcpyHostToDevice);
	if (cerr != cudaSuccess)
	{
		cudaFree(d_input);
		cudaFree(d_output);
		cudaFree(d_expanded_key);
		free(h_expanded_key);
		return AES_ERROR_CUDA_FAILURE;
	}

	/* Launch kernel */
	const int threads_per_block = 256;
	int blocks = (int)((num_blocks + threads_per_block - 1) /
					   threads_per_block);

	aes_ecb_decrypt_kernel<<<blocks, threads_per_block>>>(d_input, d_output,
														  d_expanded_key,
														  num_rounds,
														  num_blocks);
	cerr = cudaGetLastError();
	if (cerr != cudaSuccess)
	{
		cudaFree(d_input);
		cudaFree(d_output);
		cudaFree(d_expanded_key);
		memory_write_zeros(h_expanded_key, expanded_key_size);
		free(h_expanded_key);
		return AES_ERROR_CUDA_FAILURE;
	}

	/* Ensure kernel finished */
	cerr = cudaDeviceSynchronize();
	if (cerr != cudaSuccess)
	{
		cudaFree(d_input);
		cudaFree(d_output);
		cudaFree(d_expanded_key);
		memory_write_zeros(h_expanded_key, expanded_key_size);
		free(h_expanded_key);
		return AES_ERROR_CUDA_FAILURE;
	}

	/* Copy plaintext back */
	cerr = cudaMemcpy(plaintext, d_output, length, cudaMemcpyDeviceToHost);
	if (cerr != cudaSuccess)
	{
		cudaFree(d_input);
		cudaFree(d_output);
		cudaFree(d_expanded_key);
		memory_write_zeros(h_expanded_key, expanded_key_size);
		free(h_expanded_key);
		return AES_ERROR_CUDA_FAILURE;
	}

	/* Clean up */
	cudaFree(d_input);
	cudaFree(d_output);

	cudaMemset(d_expanded_key, 0, expanded_key_size);
	cudaFree(d_expanded_key);

	memory_write_zeros(h_expanded_key, expanded_key_size);
	free(h_expanded_key);

	return AES_SUCCESS;
}