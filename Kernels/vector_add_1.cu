#include <stdio.h>		  // For printf
#include <stdlib.h>		  // For malloc, rand, srand
#include <time.h>		  // For seeding random numbers
#include <cuda_runtime.h> // CUDA runtime API

#ifdef _WIN32
#include <windows.h> // For high-resolution timers on Windows
#endif

/*==============================================================================
 * CONFIGURATION
 *============================================================================*/
#define N 10000000	   // Vector size = 10 million elements
#define BLOCK_SIZE 256 // CUDA block size (number of threads per block)

// Example:
// A = [1, 2, 3, 4, 5]
// B = [6, 7, 8, 9, 10]
// C = A + B = [7, 9, 11, 13, 15]

/*==============================================================================
 * CPU Implementation of Vector Addition
 *============================================================================*/
void vector_add_cpu(float *a, float *b, float *c, int n)
{
	// Simple loop: each element in C is sum of A and B
	for (int i = 0; i < n; i++)
	{
		c[i] = a[i] + b[i];
	}
}

/*==============================================================================
 * CUDA Kernel for Vector Addition
 *============================================================================*/
__global__ void vector_add_gpu(float *a, float *b, float *c, int n)
{
	/* Calculate global thread index */
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	/* Perform addition only if index is in range */
	if (i < n)
	{
		c[i] = a[i] + b[i];
	}
}

/*==============================================================================
 * Initialize a vector with random values (between 0 and 1)
 *============================================================================*/
void init_vector(float *vec, int n)
{
	for (int i = 0; i < n; i++)
	{
		vec[i] = (float)rand() / RAND_MAX;
	}
}

/*==============================================================================
 * Cross-platform timing utility
 *============================================================================*/

#ifdef _WIN32
/* Windows: use QueryPerformanceCounter */
double get_time()
{
	LARGE_INTEGER freq, counter;
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&counter);
	return (double)counter.QuadPart / (double)freq.QuadPart;
}
#elif __linux__
/* Linux: use clock_gettime with CLOCK_MONOTONIC */
double get_time()
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#endif

/*==============================================================================
 * MAIN PROGRAM
 *============================================================================*/
int main()
{
	/* Host pointers (CPU memory) */
	float *h_a, *h_b, *h_c_cpu, *h_c_gpu;

	/* Device pointers (GPU memory) */
	float *d_a, *d_b, *d_c;

	/* Memory size in bytes for N floats */
	size_t size = N * sizeof(float);

	/* Allocate host memory */
	h_a = (float *)malloc(size);
	h_b = (float *)malloc(size);
	h_c_cpu = (float *)malloc(size);
	h_c_gpu = (float *)malloc(size);

	/* Initialize host vectors with random values */
	srand(time(NULL));
	init_vector(h_a, N);
	init_vector(h_b, N);

	/* Allocate device (GPU) memory */
	cudaMalloc(&d_a, size);
	cudaMalloc(&d_b, size);
	cudaMalloc(&d_c, size);

	/* Copy data from host → device */
	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

	/* Compute grid size (number of blocks needed) */
	int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	/*
		Example:
		N = 1024, BLOCK_SIZE = 256
		num_blocks = (1024 + 255) / 256 = 4
	*/

	/* Warm-up (To avoid cold start effects) */
	printf("Performing warm-up runs...\n");
	for (int i = 0; i < 3; i++)
	{
		/* CPU Addition */
		vector_add_cpu(h_a, h_b, h_c_cpu, N);
		/* GPU Addition */
		vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
		cudaDeviceSynchronize(); // Ensure kernel finished
	}

	/* Benchmark CPU implementation */
	printf("Benchmarking CPU implementation...\n");
	double cpu_total_time = 0.0;
	for (int i = 0; i < 20; i++)
	{
		double start_time = get_time();
		vector_add_cpu(h_a, h_b, h_c_cpu, N);
		double end_time = get_time();
		cpu_total_time += end_time - start_time;
	}
	double cpu_avg_time = cpu_total_time / 20.0;

	/* Benchmark GPU implementation */
	printf("Benchmarking GPU implementation...\n");
	double gpu_total_time = 0.0;
	for (int i = 0; i < 20; i++)
	{
		double start_time = get_time();

		/* Launch GPU kernel */
		vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
		cudaDeviceSynchronize(); // Wait for kernel to complete

		double end_time = get_time();
		gpu_total_time += end_time - start_time;
	}
	double gpu_avg_time = gpu_total_time / 20.0;

	/* Print benchmark results */
	printf("CPU average time: %f milliseconds\n", cpu_avg_time * 1000);
	printf("GPU average time: %f milliseconds\n", gpu_avg_time * 1000);
	printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

	/* Verify correctness of GPU result */
	cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);

	bool correct = true;
	for (int i = 0; i < N; i++)
	{
		if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5)
		{
			correct = false;
			break;
		}
	}
	printf("Results are %s\n", correct ? "correct" : "incorrect");

	/* Free allocated memory */
	free(h_a);
	free(h_b);
	free(h_c_cpu);
	free(h_c_gpu);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}
