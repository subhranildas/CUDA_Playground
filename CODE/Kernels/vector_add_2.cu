/*
===============================================================================
 CUDA Vector Addition Benchmark
===============================================================================
 This program compares CPU and GPU (1D and 3D grid) implementations of vector
 addition. It measures performance, verifies correctness, and demonstrates
 CUDA grid/block configurations.

 Features:
 - Vector addition on CPU (serial implementation)
 - Vector addition on GPU using:
	(1) 1D grid/block mapping
	(2) 3D grid/block mapping
 - Cross-platform high resolution timing (Windows + Linux)
 - Correctness verification (CPU vs GPU results)
===============================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

#ifdef _WIN32
#include <windows.h> // High-resolution timers on Windows
#endif

/*==============================================================================
 * Configuration constants
 *============================================================================*/

// Vector size (10 million elements)
#define N 10000000

// GPU block sizes
#define BLOCK_SIZE_1D 1024 // Threads per block in 1D kernel
#define BLOCK_SIZE_3D_X 16 // Threads per block in X
#define BLOCK_SIZE_3D_Y 8  // Threads per block in Y
#define BLOCK_SIZE_3D_Z 8  // Threads per block in Z
// Total threads per block in 3D = 16 * 8 * 8 = 1024

/*==============================================================================
 * CPU implementation
 *============================================================================*/

// Serial CPU vector addition
void vector_add_cpu(float *a, float *b, float *c, int n)
{
	for (int i = 0; i < n; i++)
	{
		c[i] = a[i] + b[i];
	}
}

/*==============================================================================
 * GPU kernels
 *============================================================================*/

// 1D CUDA kernel for vector addition
__global__ void vector_add_gpu_1d(float *a, float *b, float *c, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n)
	{
		c[i] = a[i] + b[i];
	}
}

// 3D CUDA kernel for vector addition
__global__ void vector_add_gpu_3d(float *a, float *b, float *c,
								  int nx, int ny, int nz)
{
	// Compute 3D thread index
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	// Flatten 3D index into 1D index
	if (i < nx && j < ny && k < nz)
	{
		int idx = i + j * nx + k * nx * ny;

		if (idx < nx * ny * nz)
		{
			c[idx] = a[idx] + b[idx];
		}
	}
}

/*==============================================================================
 * Utility functions
 *============================================================================*/

// Initialize a vector with random float values in [0,1)
void init_vector(float *vec, int n)
{
	for (int i = 0; i < n; i++)
	{
		vec[i] = (float)rand() / RAND_MAX;
	}
}

/*==============================================================================
 * Cross-platform timing utilities
 *============================================================================*/

#ifdef _WIN32
// Windows: use QueryPerformanceCounter
double get_time()
{
	LARGE_INTEGER freq, counter;
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&counter);
	return (double)counter.QuadPart / (double)freq.QuadPart;
}
#elif __linux__
// Linux: use clock_gettime with CLOCK_MONOTONIC
double get_time()
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#endif

/*==============================================================================
 * Main program
 *============================================================================*/

int main()
{
	// -------------------------------------------------------------------------
	// Host (CPU) and device (GPU) memory pointers
	// -------------------------------------------------------------------------
	float *h_a, *h_b, *h_c_cpu, *h_c_gpu_1d, *h_c_gpu_3d;
	float *d_a, *d_b, *d_c_1d, *d_c_3d;
	size_t size = N * sizeof(float);

	// -------------------------------------------------------------------------
	// Allocate host memory
	// -------------------------------------------------------------------------
	h_a = (float *)malloc(size);
	h_b = (float *)malloc(size);
	h_c_cpu = (float *)malloc(size);
	h_c_gpu_1d = (float *)malloc(size);
	h_c_gpu_3d = (float *)malloc(size);

	// Initialize input vectors with random values
	srand(time(NULL));
	init_vector(h_a, N);
	init_vector(h_b, N);

	// -------------------------------------------------------------------------
	// Allocate device memory
	// -------------------------------------------------------------------------
	cudaMalloc(&d_a, size);
	cudaMalloc(&d_b, size);
	cudaMalloc(&d_c_1d, size);
	cudaMalloc(&d_c_3d, size);

	// Copy input vectors to device
	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

	// -------------------------------------------------------------------------
	// Define grid and block dimensions
	// -------------------------------------------------------------------------

	// 1D configuration
	int num_blocks_1d = (N + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;

	// 3D configuration: N = 100 * 100 * 1000
	int nx = 100, ny = 100, nz = 1000;
	dim3 block_size_3d(BLOCK_SIZE_3D_X, BLOCK_SIZE_3D_Y, BLOCK_SIZE_3D_Z);
	dim3 num_blocks_3d(
		(nx + block_size_3d.x - 1) / block_size_3d.x,
		(ny + block_size_3d.y - 1) / block_size_3d.y,
		(nz + block_size_3d.z - 1) / block_size_3d.z);

	// -------------------------------------------------------------------------
	// Warm-up runs (to stabilize GPU performance)
	// -------------------------------------------------------------------------
	printf("Performing warm-up runs...\n");
	for (int i = 0; i < 3; i++)
	{
		vector_add_cpu(h_a, h_b, h_c_cpu, N);
		vector_add_gpu_1d<<<num_blocks_1d, BLOCK_SIZE_1D>>>(d_a, d_b, d_c_1d, N);
		vector_add_gpu_3d<<<num_blocks_3d, block_size_3d>>>(d_a, d_b, d_c_3d, nx, ny, nz);
		cudaDeviceSynchronize();
	}

	// -------------------------------------------------------------------------
	// Benchmark CPU implementation
	// -------------------------------------------------------------------------
	printf("Benchmarking CPU implementation...\n");
	double cpu_total_time = 0.0;
	for (int i = 0; i < 5; i++)
	{
		double start_time = get_time();
		vector_add_cpu(h_a, h_b, h_c_cpu, N);
		double end_time = get_time();
		cpu_total_time += end_time - start_time;
	}
	double cpu_avg_time = cpu_total_time / 5.0;

	// -------------------------------------------------------------------------
	// Benchmark GPU 1D implementation
	// -------------------------------------------------------------------------
	printf("Benchmarking GPU 1D implementation...\n");
	double gpu_1d_total_time = 0.0;
	for (int i = 0; i < 100; i++)
	{
		cudaMemset(d_c_1d, 0, size); // Clear results
		double start_time = get_time();
		vector_add_gpu_1d<<<num_blocks_1d, BLOCK_SIZE_1D>>>(d_a, d_b, d_c_1d, N);
		cudaDeviceSynchronize();
		double end_time = get_time();
		gpu_1d_total_time += end_time - start_time;
	}
	double gpu_1d_avg_time = gpu_1d_total_time / 100.0;

	// Verify 1D results
	cudaMemcpy(h_c_gpu_1d, d_c_1d, size, cudaMemcpyDeviceToHost);
	bool correct_1d = true;
	for (int i = 0; i < N; i++)
	{
		if (fabs(h_c_cpu[i] - h_c_gpu_1d[i]) > 1e-4)
		{
			correct_1d = false;
			std::cout << i << " cpu: " << h_c_cpu[i]
					  << " != " << h_c_gpu_1d[i] << std::endl;
			break;
		}
	}
	printf("1D Results are %s\n", correct_1d ? "correct" : "incorrect");

	// -------------------------------------------------------------------------
	// Benchmark GPU 3D implementation
	// -------------------------------------------------------------------------
	printf("Benchmarking GPU 3D implementation...\n");
	double gpu_3d_total_time = 0.0;
	for (int i = 0; i < 100; i++)
	{
		cudaMemset(d_c_3d, 0, size); // Clear results
		double start_time = get_time();
		vector_add_gpu_3d<<<num_blocks_3d, block_size_3d>>>(d_a, d_b, d_c_3d,
															nx, ny, nz);
		cudaDeviceSynchronize();
		double end_time = get_time();
		gpu_3d_total_time += end_time - start_time;
	}
	double gpu_3d_avg_time = gpu_3d_total_time / 100.0;

	// Verify 3D results
	cudaMemcpy(h_c_gpu_3d, d_c_3d, size, cudaMemcpyDeviceToHost);
	bool correct_3d = true;
	for (int i = 0; i < N; i++)
	{
		if (fabs(h_c_cpu[i] - h_c_gpu_3d[i]) > 1e-4)
		{
			correct_3d = false;
			std::cout << i << " cpu: " << h_c_cpu[i]
					  << " != " << h_c_gpu_3d[i] << std::endl;
			break;
		}
	}
	printf("3D Results are %s\n", correct_3d ? "correct" : "incorrect");

	// -------------------------------------------------------------------------
	// Print benchmark results
	// -------------------------------------------------------------------------
	printf("CPU average time: %f ms\n", cpu_avg_time * 1000);
	printf("GPU 1D average time: %f ms\n", gpu_1d_avg_time * 1000);
	printf("GPU 3D average time: %f ms\n", gpu_3d_avg_time * 1000);
	printf("Speedup (CPU vs GPU 1D): %fx\n", cpu_avg_time / gpu_1d_avg_time);
	printf("Speedup (CPU vs GPU 3D): %fx\n", cpu_avg_time / gpu_3d_avg_time);
	printf("Speedup (GPU 1D vs GPU 3D): %fx\n",
		   gpu_1d_avg_time / gpu_3d_avg_time);

	// -------------------------------------------------------------------------
	// Cleanup
	// -------------------------------------------------------------------------
	free(h_a);
	free(h_b);
	free(h_c_cpu);
	free(h_c_gpu_1d);
	free(h_c_gpu_3d);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c_1d);
	cudaFree(d_c_3d);

	return 0;
}
