/* ==================================================================================
# Vector Addition Example for Abnormally Large Vectors (Chunk-Based Processing)
===================================================================================*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define SIZE                    (1024 * 1024 * 1024)   // 1 Billion elements
#define CHUNK_SIZE              (1024 * 512)          // Process 524,288 elements at a time
#define CHUNK_COUNT             (SIZE / CHUNK_SIZE)   // Number of chunks

/* Kernel Function : Executed on the GPU */
__global__ void vectorAdd(int* A, int* B, int* C, int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        C[i] = A[i] + B[i];
    }
}

/* Compare Buffers Function */
bool compareBuffer(int* a, int* b, int size) {
    for (int i = 0; i < size; i++) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

int main() {

    int* A, * B, * C, * C_CPU;
    int* d_A, * d_B, * d_C;


    /* Allocate CPU - side memory */
    printf("Allocating memory for vectors in CPU ...\n");
    A = (int*)malloc(SIZE * sizeof(int));
    B = (int*)malloc(SIZE * sizeof(int));
    C = (int*)malloc(SIZE * sizeof(int));
    C_CPU = (int*)malloc(SIZE * sizeof(int));

    
    /* Initialize host arrays */
    printf("Filling vectors with value ...\n");
    for (int i = 0; i < SIZE; i++) {
        A[i] = i;
        B[i] = SIZE - i;
    }

    /* Allocate GPU memory for one chunk */
    printf("Allocating memory for Chunk in GPU ...\n");
    cudaMalloc((void**)&d_A, CHUNK_SIZE * sizeof(int));
    cudaMalloc((void**)&d_B, CHUNK_SIZE * sizeof(int));
    cudaMalloc((void**)&d_C, CHUNK_SIZE * sizeof(int));

    printf("Starting vector addition in %d chunks...\n", CHUNK_COUNT);

    /* Process each chunk */
    for (int chunk = 0; chunk < CHUNK_COUNT; chunk++) {

        printf("Performing Addition for Chunk %d ...\n", chunk);

        int offset = chunk * CHUNK_SIZE;

        /* Copy one chunk from host to device */
        cudaMemcpy(d_A, A + offset, CHUNK_SIZE * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B + offset, CHUNK_SIZE * sizeof(int), cudaMemcpyHostToDevice);

        /* Configure kernel launch */
        int threads = 256;
        int blocks = (CHUNK_SIZE + threads - 1) / threads;

        /* Launch kernel for this chunk */
        vectorAdd << <blocks, threads >> > (d_A, d_B, d_C, CHUNK_SIZE);

        /* Wait for GPU to finish this chunk before moving on */
        cudaDeviceSynchronize();

        /* Copy result back to host */
        cudaMemcpy(C + offset, d_C, CHUNK_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    }

    printf("All chunks processed successfully ...\n");

    /* CPU reference computation */
    printf("Calculating using CPU ...\n");
    for (int i = 0; i < SIZE; i++) {
        C_CPU[i] = A[i] + B[i];
    }

    /* Verify results */
    printf("Comparing Results with CPU output ...\n");
    printf("%s\n", compareBuffer(C, C_CPU, SIZE) ? "Calculation Correct !!" : "Error !!");

    /* Cleanup */
    printf("Cleaning Memory Allocations ...\n");
    free(A);
    free(B);
    free(C);
    free(C_CPU);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
