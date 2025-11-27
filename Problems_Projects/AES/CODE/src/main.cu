#include "kernels.h"
#include "utils.h"
#include "aes.h"
#include <cstdio>
#include <cstring>
#include <chrono>

/* =============================================================================
 * Benchmark AES-128 ECB encryption on large multi-block plaintext.
 * Compares CPU vs GPU performance for various data sizes.
 * ============================================================================ */

int main(int argc, char **argv)
{
	(void)argc;
	(void)argv;

	/* AES-128 key */
	const uint8_t key128[16] = {
		0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
		0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};

	printf("========================================\n");
	printf("  AES-128 ECB CPU vs GPU Benchmark\n");
	printf("========================================\n\n");

	/* Test different data sizes */
	const size_t test_sizes[] = {
		1024,			  /* 1 KB (64 blocks) */
		10 * 1024,		  /* 10 KB (640 blocks) */
		100 * 1024,		  /* 100 KB (6400 blocks) */
		1024 * 1024,	  /* 1 MB (65536 blocks) */
		10 * 1024 * 1024, /* 10 MB (655360 blocks) */
		100 * 1024 * 1024 /* 100 MB (6553600 blocks) */
	};
	const int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

	for (int size_idx = 0; size_idx < num_sizes; ++size_idx)
	{
		const size_t data_size = test_sizes[size_idx];
		const size_t num_blocks = data_size / AES_BLOCK_SIZE;

		printf("Test size: %zu bytes (%zu blocks)\n", data_size, num_blocks);

		/* Allocate buffers */
		uint8_t *plaintext = (uint8_t *)malloc(data_size);
		uint8_t *cpu_ciphertext = (uint8_t *)malloc(data_size);
		uint8_t *gpu_ciphertext = (uint8_t *)malloc(data_size);

		if (!plaintext || !cpu_ciphertext || !gpu_ciphertext)
		{
			printf("  Memory allocation failed\n");
			free(plaintext);
			free(cpu_ciphertext);
			free(gpu_ciphertext);
			return 1;
		}

		/* Fill plaintext with pattern */
		for (size_t i = 0; i < data_size; ++i)
		{
			plaintext[i] = (uint8_t)(i & 0xFF);
		}

		/* CPU Benchmark */
		auto cpu_start = std::chrono::high_resolution_clock::now();
		aes_error_te cpu_err = aes_encrypt_ecb(plaintext, data_size, cpu_ciphertext,
											   key128, AES_KEY_SIZE_128);
		auto cpu_end = std::chrono::high_resolution_clock::now();
		auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
			cpu_end - cpu_start);

		if (cpu_err != AES_SUCCESS)
		{
			printf("  CPU encryption failed (err=%d)\n", (int)cpu_err);
			free(plaintext);
			free(cpu_ciphertext);
			free(gpu_ciphertext);
			return 1;
		}

		/* GPU Benchmark */
		auto gpu_start = std::chrono::high_resolution_clock::now();
		aes_error_te gpu_err = aes_encrypt_ecb_cuda(plaintext, data_size, gpu_ciphertext,
													key128, AES_KEY_SIZE_128);
		auto gpu_end = std::chrono::high_resolution_clock::now();
		auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
			gpu_end - gpu_start);

		if (gpu_err != AES_SUCCESS)
		{
			printf("  GPU encryption failed (err=%d)\n", (int)gpu_err);
			free(plaintext);
			free(cpu_ciphertext);
			free(gpu_ciphertext);
			return 1;
		}

		/* Verify CPU and GPU produce same result */
		bool match = (memcmp(cpu_ciphertext, gpu_ciphertext, data_size) == 0);

		/* Calculate throughput (MB/s) */
		double cpu_throughput = (data_size / (1024.0 * 1024.0)) / (cpu_duration.count() / 1000.0);
		double gpu_throughput = (data_size / (1024.0 * 1024.0)) / (gpu_duration.count() / 1000.0);
		double speedup = (double)cpu_duration.count() / gpu_duration.count();

		printf("  CPU time: %lld ms (%.2f MB/s)\n", (long long)cpu_duration.count(), cpu_throughput);
		printf("  GPU time: %lld ms (%.2f MB/s)\n", (long long)gpu_duration.count(), gpu_throughput);
		printf("  Speedup:  %.2fx\n", speedup);
		printf("  Match:    %s\n\n", match ? "YES" : "NO");

		if (!match)
		{
			printf("  ERROR: CPU and GPU results don't match!\n");
			free(plaintext);
			free(cpu_ciphertext);
			free(gpu_ciphertext);
			return 1;
		}

		free(plaintext);
		free(cpu_ciphertext);
		free(gpu_ciphertext);
	}

	printf("========================================\n");
	printf("  Benchmark Complete\n");
	printf("========================================\n");

	return 0;
}
