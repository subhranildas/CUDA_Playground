/* ==================================================================================
# A simple Program to print Block(Thread Block) ID, Thread ID and warp ID
===================================================================================*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

/* Kernel Function : To be executed in GPU not CPU */
__global__ void test_01() {

	int warpID = 0;
	warpID = threadIdx.x / 32;
	/* Print the Blocks and Threads IDs */
	printf("\nThe Block ID is %d --- The Thread ID is %d --- The warp ID %d", blockIdx.x, threadIdx.x, warpID);
}

/* CUDA C always requires a main Function */
int main() {
	/* Kernel_Name <<< Number_of_Blocks, Number_of_Threads_per_Block >>> */
	test_01 << <2, 64 >> > ();
	/* Note: Please note that if the thread per block count is over 1024 then the
			 application will not work as the GPU might be limited to 1024 threads per block.
			 In case we want to do a vector addition of 2048 size then we will need 2 blocks
			 of 1024 threads each to compute the sum in parallel */
			 /* Note: Also note that the number of Blocks that we can use in a application is also
					 limited, the maximum number of blocks we can use in a kernel call is limited
					 by the number of SMs in the GPU and the number of blocks per SM in the GPU
					 If a GPU has 32 blocks per SM and 10 SMs total then we can only use a maximum
					 of 320 Blocks */

					 /* The Below Function is used make the CPU wait until all the GPU operations are executed */
	cudaDeviceSynchronize();
}