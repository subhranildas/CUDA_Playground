#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

__global__ void test01(void)
{
	printf("\nBlock ID: %d --- Thread ID : %d", blockIdx.x, threadIdx.x);
}

int main()
{
	test01<<<1, 8>>>();
	return 0;
}