/* ==================================================================================
# A simple Vector Addition Example with CUDA
===================================================================================*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE 1024

/* Kernel Function : To be executed in GPU not CPU */
__global__ void vectorAdd(int* A, int* B, int* C, int size) {

	int i = threadIdx.x;
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
	int * A, * B, * C, * C_CPU;
	int *d_A, *d_B, *d_C;
	int size = SIZE * sizeof(int);

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

	/* Launch The vectorAdd CUDA Kernel */
	vectorAdd <<<1, 1024 >>> (d_A, d_B, d_C, SIZE);

	/* Halt the CPU thread until all preceding GPU operations (kernels, memory transfers, etc.)
	   are finished on the Entire Device (GPU) */
	cudaDeviceSynchronize();

	/* Copy the results from GPU memory to CPU Memory */
	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

	/* Do the same Operation using the CPU */
	for (int i = 0; i < SIZE; i++) {
		C_CPU[i] = A[i] + B[i];
	}

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