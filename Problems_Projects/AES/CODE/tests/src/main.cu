#include <cstdio>
#include "aes_test.h"

/* Forward declaration of GPU test function */
extern "C" int aes_gpu_test_ecb();

int main(int argc, char **argv)
{
	(void)argc;
	(void)argv;

	printf("========================================\n");
	printf("   AES ECB CPU & GPU Test Suite\n");
	printf("========================================\n\n");

	/* Run CPU tests */
	int cpu_res = aes_test_ecb();
	printf("\n");

	/* Run GPU tests */
	int gpu_res = aes_gpu_test_ecb();
	printf("\n");

	int total_res = cpu_res + gpu_res;
	if (total_res == 0)
	{
		printf("All AES ECB CPU & GPU tests PASSED\n");
	}
	else
	{
		printf("Some tests FAILED (CPU=%d, GPU=%d)\n", cpu_res, gpu_res);
	}
	return total_res;
}
