/* ==================================================================================
# Vector Addition Example with multiple Blocks for Higher GPU usage and performance
===================================================================================*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define SIZE 2048

/* Kernel Function : To be executed in GPU not CPU */
__global__ void vectorAdd(int* A, int* B, int* C, int size) {

	/* In case the multiple blocks are used to perform the calculation
	   the interator variable must match the position properly, this
	   will be possible if we take the block id into account, the following
	   equation generates i perfectly with SIZE greater than 1024 which is
	   the maximum threads per block count for the GPU */
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	/* Perform the Addition */
	C[i] = A[i] + B[i];
}

/* Function Compare Buffers */
bool compareBuffer(int* a, int* b, int size) {
	for (int i = 0; i < size; i++) {
		if (a[i] != b[i]) {
			return false;
		}
	}
	return true;
}

/* CUDA C always requires a main Function */
int main() {

	/* Declare CPU side and GPU side variables for the Operations */
	int* A, * B, * C, * C_CPU;
	int* d_A, * d_B, * d_C;
	int size = SIZE * sizeof(int);

	/* Variable used to do some Rough Profiling (Cuda Event Variables) */
	cudaEvent_t start, stop;
	/* Create the Cuda Events */
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	/* Allocate Memory for CPU side vaiables */
	A = (int*)malloc(size);
	B = (int*)malloc(size);
	C = (int*)malloc(size);
	C_CPU = (int*)malloc(size);

	/* Allocate memory for GPU side variables */
	cudaMalloc((void**)&d_A, size);
	cudaMalloc((void**)&d_B, size);
	cudaMalloc((void**)&d_C, size);

	/* Fill data in the variabels Host side */
	for (int i = 0; i < SIZE; i++) {
		A[i] = i;
		B[i] = SIZE - i;
	}

	/* Copy data from host side (CPU Memory) to device side (GPU Memory) */
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

	/* GPU Cold start Run for Clean Event times */
	for (int i = 0; i < 3; i++)
	{
		/* GPU Addition */
		vectorAdd << <64, 32 >> > (d_A, d_B, d_C, SIZE);
		cudaDeviceSynchronize(); // Ensure kernel finished
	}

	/* Launch The vectorAdd CUDA Kernel */

	/* Here in below case we can make use of more SMs by increasing the
	   number of blocks and decreasing the number of threads. The minimum
	   number of threads per block is 32, therefore for a vector addition
	   of 2048 length we can take 32 threads per block and 2048/32=64 blocks.
	   This will increase the overall GPU usage and increase performance */

	/* Record Cuda Event */
	cudaEventRecord(start);

	//vectorAdd << <2, 1024 >> > (d_A, d_B, d_C, SIZE);
	vectorAdd << <64, 32 >> > (d_A, d_B, d_C, SIZE);

	/* Halt the CPU thread until all preceding GPU operations (kernels, memory transfers, etc.)
	   are finished on the Entire Device (GPU) */
	cudaDeviceSynchronize();

	/* Record Cuda Event */
	cudaEventRecord(stop);

	/* Copy the results from GPU memory to CPU Memory */
	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

	/* Do the same Operation using the CPU */
	for (int i = 0; i < SIZE; i++) {
		C_CPU[i] = A[i] + B[i];
	}

	/* Calculate time for CUDA Operations */

	/* Waits for Stop Event Completion */
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	/* Print time required for calculation Completion */
	printf("Execution time: %f Ms\n", milliseconds);


	/* Note: The execution time calculated in this way can vary from one round
	   to another, hence this method for profiling is very crude and not recommended.
	   Using a Profiler (Tools provided by NVIDIA) is always a better option */

	printf("%s", compareBuffer(C, C_CPU, SIZE) ? "Calculation Correct !!\n" : "Error !!\n");


	/* Free All the Memories used */
	free(A);
	free(B);
	free(C);
	free(C_CPU);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}