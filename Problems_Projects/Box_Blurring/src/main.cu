/*==============================================================================
 * Defines Related to image loading/conversion libraries
==============================================================================*/
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

/*==============================================================================
 * Includes Related to image loading/conversion libraries
==============================================================================*/
#include "stb_image.h"
#include "stb_image_write.h"

/*==============================================================================
 * Includes related to standard Cpp libraries and CUDA runtime libraries
==============================================================================*/
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h> // For high-resolution timers on Windows
#endif

using namespace std;

/*==============================================================================
 * Constant Defines for Shared Kernel
==============================================================================*/
#define KERNEL_SIZE 10
#define RADIUS (KERNEL_SIZE / 2)
#define BLOCK_X 16
#define BLOCK_Y 16

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

/*==============================================================================
 * Global variables
==============================================================================*/
double time_cpu = 0;
double time_gpu = 0;
double time_gpu_shared = 0;

int kernel[10][10] = {
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};

/*==============================================================================
 * Cross-platform timing utility
==============================================================================*/
#ifdef _WIN32
double get_time()
{
	LARGE_INTEGER freq, counter;
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&counter);
	return (double)counter.QuadPart / (double)freq.QuadPart;
}
#elif __linux__
double get_time()
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#endif

/*==============================================================================
 * CPU Clamp Function
==============================================================================*/
inline int clamp_cpu(int val, int minVal, int maxVal)
{
	return val < minVal ? minVal : (val > maxVal ? maxVal : val);
}

/*==============================================================================
 * GPU Clamp Function
==============================================================================*/
__device__ inline int clamp_gpu(int val, int minVal, int maxVal)
{
	return max(min(val, maxVal), minVal);
}

__constant__ int d_kernel_const[KERNEL_SIZE * KERNEL_SIZE];

/*==============================================================================
 * Shared Kernel (GPU)
==============================================================================*/
__global__ void applyKernelCUDA_Shared(const unsigned char *__restrict__ src,
									   unsigned char *__restrict__ dst,
									   int width, int height)
{
	/* Shared memory tile with padding for halo (border pixels) */
	__shared__ unsigned char tile[BLOCK_Y + KERNEL_SIZE - 1][BLOCK_X + KERNEL_SIZE - 1];

	/* Global coordinates */
	int gx = blockIdx.x * BLOCK_X + threadIdx.x;
	int gy = blockIdx.y * BLOCK_Y + threadIdx.y;

	/* Local coordinates (inside shared memory) */
	int lx = threadIdx.x + RADIUS;
	int ly = threadIdx.y + RADIUS;

	/* Load main pixel */
	if (gx < width && gy < height)
		tile[ly][lx] = src[gy * width + gx];
	else
		tile[ly][lx] = 0;

	/* Load Halo (border) pixels — left, right, top, bottom */
	if (threadIdx.x < RADIUS)
	{
		int x_left = gx - RADIUS;
		int x_right = gx + BLOCK_X;
		int safe_y = gy;

		/* Left Halo */
		tile[ly][lx - RADIUS] = (x_left >= 0) ? src[safe_y * width + x_left] : 0;

		/* Right Halo */
		if (lx + BLOCK_X < BLOCK_X + KERNEL_SIZE - 1)
			tile[ly][lx + BLOCK_X] = (x_right < width) ? src[safe_y * width + x_right] : 0;
	}

	if (threadIdx.y < RADIUS)
	{
		int y_top = gy - RADIUS;
		int y_bottom = gy + BLOCK_Y;
		int safe_x = gx;

		/* Top Halo */
		tile[ly - RADIUS][lx] = (y_top >= 0) ? src[y_top * width + safe_x] : 0;

		/* Bottom Halo */
		if (ly + BLOCK_Y < BLOCK_Y + KERNEL_SIZE - 1)
			tile[ly + BLOCK_Y][lx] = (y_bottom < height) ? src[y_bottom * width + safe_x] : 0;
	}

	__syncthreads();

	/* Perform Convolution */
	if (gx < width && gy < height)
	{
		int sum = 0;
		int weight_sum = 0;
		for (int ky = 0; ky < KERNEL_SIZE; ky++)
		{
			for (int kx = 0; kx < KERNEL_SIZE; kx++)
			{
				int pixel = tile[ly + ky - RADIUS][lx + kx - RADIUS];
				int weight = d_kernel_const[ky * KERNEL_SIZE + kx];
				sum += pixel * weight;
				weight_sum += weight;
			}
		}
		dst[gy * width + gx] = sum / weight_sum;
	}
}

/*==============================================================================
 * Basic Convolution Kernel (GPU)
==============================================================================*/
__global__ void applyKernelCUDA(const unsigned char *src, unsigned char *dst,
								int width, int height, const int *kernel,
								int kSize)
{
	int offset = kSize / 2;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= offset / 2 && x < width - offset / 2 &&
		y >= offset / 2 && y < height - offset / 2)
	{
		int sum = 0;
		for (int ky = 0; ky < kSize; ky++)
			for (int kx = 0; kx < kSize; kx++)
			{
				int nx = clamp_gpu(x + kx - offset, 0, width - 1);
				int ny = clamp_gpu(y + ky - offset, 0, height - 1);
				sum += src[ny * width + nx] * kernel[ky * kSize + kx];
			}

		int kernelSum = 0;
		for (int i = 0; i < kSize * kSize; i++)
			kernelSum += kernel[i];
		if (kernelSum == 0)
			kernelSum = 1;

		dst[y * width + x] = clamp_gpu(sum / kernelSum, 0, 255);
	}
}

/*==============================================================================
 * Basic Convolution Kernel (CPU)
==============================================================================*/
void applyKernelCPU(const unsigned char *src, unsigned char *dst, int width,
					int height, int kernel[KERNEL_SIZE][KERNEL_SIZE], int kSize)
{
	int offset = kSize / 2;
	for (int y = offset / 2; y < height - offset / 2; y++)
	{
		for (int x = offset / 2; x < width - offset / 2; x++)
		{
			int sum = 0;
			for (int ky = 0; ky < kSize; ky++)
				for (int kx = 0; kx < kSize; kx++)
				{
					int ny = clamp_cpu(y + ky - offset, 0, height - 1);
					int nx = clamp_cpu(x + kx - offset, 0, width - 1);
					sum += src[ny * width + nx] * kernel[ky][kx];
				}

			int kernelSum = 0;
			for (int ky = 0; ky < kSize; ky++)
				for (int kx = 0; kx < kSize; kx++)
					kernelSum += kernel[ky][kx];
			if (kernelSum == 0)
				kernelSum = 1;

			dst[y * width + x] = clamp_cpu(sum / kernelSum, 0, 255);
		}
	}
}

/*==============================================================================
 * Host Main
==============================================================================*/
int main(int argc, char **argv)
{
	/* Use arguments if provided, Use if provided, I'm using default */
	string inputFile = (argc > 1) ? argv[1] : "images/globalProfilePic_LW_GS.png";
	string outputFileGPU = (argc > 2) ? argv[2] : "outputImage/blurred_gpu.png";
	string outputFileGPUShared = (argc > 3) ? argv[3] : "outputImage/blurred_gpu_tile.png";
	string outputFileCPU = (argc > 4) ? argv[4] : "outputImage/blurred_cpu.png";

	int width = 0, height = 0, channels = 0;
	double tStart, tEnd;
	int kSize = KERNEL_SIZE;
	size_t imgSize;
	size_t kernelSize;

	/* Host(CPU) Pointers */
	unsigned char *h_src;

	/* Device(GPU) pointers */
	unsigned char *d_src = nullptr;
	unsigned char *d_dst = nullptr;
	int *d_kernel = nullptr;

	/* Load Image *************************************************************/
	h_src = stbi_load(inputFile.c_str(), &width, &height, &channels, 1);

	/* Check if loading was successful */
	if (!h_src)
	{
		fprintf(stderr, "Failed to load image %s\n", inputFile.c_str());
		return EXIT_FAILURE;
	}

	/* Save image size as number of pixels */
	imgSize = width * height * sizeof(unsigned char);
	kernelSize = kSize * kSize * sizeof(int);

	/* Final Image storage vectors for GPU */
	vector<unsigned char> blurredGPU(width * height, 0);
	/* Final Image storage vectors for CPU */
	vector<unsigned char> blurredCPU(width * height, 0);

	/* GPU memory Allocations */
	CUDA_CHECK(cudaMalloc(&d_src, imgSize));
	CUDA_CHECK(cudaMalloc(&d_dst, imgSize));
	CUDA_CHECK(cudaMalloc(&d_kernel, kernelSize));

	/* CPU Implementation *****************************************************/
	tStart = get_time();

	/* Apply CPU Kernel */
	applyKernelCPU(h_src, blurredCPU.data(), width, height, kernel, kSize);

	tEnd = get_time();
	time_cpu = (tEnd - tStart) * 1000.0;

	cout << "CPU execution time: " << time_cpu << " ms\n";

	/* Write Image from CPU Implementation ************************************/
	if (!stbi_write_png(outputFileCPU.c_str(), width, height, 1,
						(void *)blurredCPU.data(), width))
	{
		fprintf(stderr, "Failed to write PNG %s\n", outputFileCPU.c_str());
	}
	else
	{
		printf("Wrote %s\n", outputFileCPU.c_str());
	}

	/* GPU Implementations ****************************************************/

	/* Copy inputs and kernels */
	CUDA_CHECK(cudaMemcpy(d_src, h_src, imgSize, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_kernel, kernel, kernelSize, cudaMemcpyHostToDevice));

	/* Launch kernels */
	dim3 threadsPerBlock(16, 16);
	dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
				   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

	/* Copy Kernel to Device */
	cudaMemcpyToSymbol(d_kernel, kernel, sizeof(int) * KERNEL_SIZE * KERNEL_SIZE);

	/* Cold Start Run */
	applyKernelCUDA<<<numBlocks, threadsPerBlock>>>(d_src, d_dst, width, height, d_kernel, kSize);

	tStart = get_time();
	applyKernelCUDA<<<numBlocks, threadsPerBlock>>>(d_src, d_dst, width, height, d_kernel, kSize);

	CUDA_CHECK(cudaDeviceSynchronize());
	tEnd = get_time();
	time_gpu = (tEnd - tStart) * 1000.0;
	cout << "CUDA kernel execution time (excluding malloc/copy): " << time_gpu << " ms\n";

	CUDA_CHECK(cudaMemcpy(blurredGPU.data(), d_dst, imgSize, cudaMemcpyDeviceToHost));

	/* Write Image from GPU Implementation ************************************/
	if (!stbi_write_png(outputFileGPU.c_str(), width, height, 1,
						(void *)blurredGPU.data(), width))
	{
		fprintf(stderr, "Failed to write PNG %s\n", outputFileGPU.c_str());
	}
	else
	{
		printf("Wrote %s\n", outputFileGPU.c_str());
	}

	/* Copy Kernel to Device */
	CUDA_CHECK(cudaMemcpyToSymbol(d_kernel_const, kernel, sizeof(int) * KERNEL_SIZE * KERNEL_SIZE));

	/* Configure Launch */
	dim3 threadsPerBlock_a(BLOCK_X, BLOCK_Y);
	dim3 numBlocks_a((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y);

	/* Launch Kernel */
	tStart = get_time();
	/* Launch CUDA Kernel */
	applyKernelCUDA_Shared<<<numBlocks_a, threadsPerBlock_a>>>(d_src, d_dst, width, height);

	/* Wait for Completion */
	CUDA_CHECK(cudaDeviceSynchronize());

	tEnd = get_time();
	time_gpu_shared = (tEnd - tStart) * 1000.0;
	cout << "CUDA kernel execution time with tile (excluding malloc/copy): " << time_gpu_shared << " ms\n";

	CUDA_CHECK(cudaMemcpy(blurredGPU.data(), d_dst, imgSize, cudaMemcpyDeviceToHost));

	/* Write Image from GPU Tile Implementation *******************************/
	if (!stbi_write_png(outputFileGPUShared.c_str(), width, height, 1,
						(void *)blurredGPU.data(), width))
	{
		fprintf(stderr, "Failed to write PNG %s\n", outputFileGPUShared.c_str());
	}
	else
	{
		printf("Wrote %s\n", outputFileGPUShared.c_str());
	}

	/* Cleanup */
	CUDA_CHECK(cudaFree(d_src));
	CUDA_CHECK(cudaFree(d_dst));
	CUDA_CHECK(cudaFree(d_kernel));
	stbi_image_free(h_src);

	cout << "Speedup (CPU/GPU): " << (time_cpu / time_gpu_shared) << "x\n";

	return 0;
}
