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
 * GPU Clamp Function
==============================================================================*/
__device__ inline int clamp_gpu(int val, int low, int high)
{
	return max(low, min(val, high));
}

/*==============================================================================
 * CPU Clamp Function
==============================================================================*/
inline int clamp_cpu(int val, int low, int high)
{
	return max(low, min(val, high));
}

/*==============================================================================
 * Basic Convolution Kernel (GPU)
==============================================================================*/
__global__ void applyKernelCUDA(const unsigned char *src, float *dst,
								int width, int height, const int *kernel,
								int kSize)
{
	/* Kernel application offset */
	int offset = kSize / 2;
	/* x and y in terms of thread and block index */
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= offset && x < width - offset &&
		y >= offset && y < height - offset)
	{
		float sum = 0.0f;
		for (int ky = 0; ky < kSize; ky++)
			for (int kx = 0; kx < kSize; kx++)
			{
				int nx = clamp_gpu(x + kx - offset, 0, width - 1);
				int ny = clamp_gpu(y + ky - offset, 0, height - 1);
				/* The src and kernel are although 2D in nature but in GPU
				implementation we consider them as continuous arrays */
				sum += (float)src[ny * width + nx] *
					   (float)kernel[ky * kSize + kx];
			}

		/* Store the sum at the center */
		dst[y * width + x] = sum;
	}
}

/*==============================================================================
 * Convolution Kernel (CPU)
==============================================================================*/
void applyKernelCPU(const unsigned char *src, float *dst,
					int width, int height, const int *kernel,
					int kSize)
{
	/* Kernel application offset */
	int offset = kSize / 2;

	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{
			if (x >= offset && x < width - offset &&
				y >= offset && y < height - offset)
			{
				float sum = 0.0f;
				for (int ky = 0; ky < kSize; ky++)
				{
					for (int kx = 0; kx < kSize; kx++)
					{
						int nx = clamp_cpu(x + kx - offset, 0, width - 1);
						int ny = clamp_cpu(y + ky - offset, 0, height - 1);
						/* The src and kernel are although 2D in nature but in GPU
						implementation we consider them as continuous arrays */
						sum += src[(y + ky) * width + (x + kx)] * kernel[ky * kSize + kx];
					}
				}

				int kernelSum = 0;
				for (int i = 0; i < kSize * kSize; i++)
					kernelSum += kernel[i];
				if (kernelSum == 0)
					kernelSum = 1;

				/* Store the sum at the center */
				dst[y * width + x] = sum;
			}
		}
	}
}

/*==============================================================================
 * Compute Sobel Magnitude (GPU)
==============================================================================*/
__global__ void sobelMagnitudeKernel_GPU(const float *gxImg,
										 const float *gyImg,
										 unsigned char *dst,
										 int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height)
	{
		float gx = gxImg[y * width + x];
		float gy = gyImg[y * width + x];
		/* This sqrtf function runs in gpu and -use_fast_math flag makes sure
		it uses faster approximations, Check my make file */
		float mag = sqrtf(gx * gx + gy * gy);

		// Clamp to 0..255 and store
		mag = fminf(255.0f, fmaxf(0.0f, mag));
		dst[y * width + x] = (unsigned char)(mag);
	}
}

/*==============================================================================
 * Compute Sobel Magnitude (CPU)
==============================================================================*/
void sobelMagnitudeKernel_CPU(const float *gxImg,
							  const float *gyImg,
							  unsigned char *dst,
							  int width, int height)
{
	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{
			if (x < width && y < height)
			{
				float gx = gxImg[y * width + x];
				float gy = gyImg[y * width + x];
				/* Use math.h square root function */
				float mag = sqrt(gx * gx + gy * gy);

				/* Clamp to 0..255 and store */
				mag = min(255.0f, max(0.0f, mag));
				dst[y * width + x] = (unsigned char)(mag);
			}
		}
	}
}

/*==============================================================================
 * Host Main
==============================================================================*/
int main(int argc, char **argv)
{

	double tStart, tEnd;
	double time_cpu = 0.0;
	double time_gpu = 0.0;

	/* Use arguments if provided, Use if provided, I'm using default */
	string inputFile = (argc > 1) ? argv[1] : "images/globalProfilePic_LW_GS.png";
	string outputFileGPU = (argc > 2) ? argv[2] : "outputImage/edge_gpu.png";
	string outputFileCPU = (argc > 3) ? argv[3] : "outputImage/edge_cpu.png";

	int width = 0, height = 0, channels = 0;
	size_t imgSize;

	/* Host(CPU) Pointers */
	unsigned char *h_src;
	unsigned char *h_out = nullptr;
	float *h_gx = nullptr;
	float *h_gy = nullptr;

	/* Load Image *************************************************************/
	h_src = stbi_load(inputFile.c_str(), &width, &height, &channels, 1);

	/* Check if loading was successful */
	if (!h_src)
	{
		fprintf(stderr, "Failed to load image %s\n", inputFile.c_str());
		return EXIT_FAILURE;
	}

	/* Save image size as number of pixels */
	imgSize = (size_t)width * (size_t)height;

	/* Device(GPU) pointers */
	unsigned char *d_src = nullptr;
	float *d_gx = nullptr;
	float *d_gy = nullptr;
	unsigned char *d_out = nullptr;
	int *d_kx = nullptr;
	int *d_ky = nullptr;

	/* CPU Memory Allocations */
	h_out = (unsigned char *)malloc(imgSize * sizeof(unsigned char));
	h_gx = (float *)malloc(imgSize * sizeof(float));
	h_gy = (float *)malloc(imgSize * sizeof(float));

	/* GPU memory Allocations */
	CUDA_CHECK(cudaMalloc(&d_src, imgSize));
	CUDA_CHECK(cudaMalloc(&d_gx, imgSize * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_gy, imgSize * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_out, imgSize));
	CUDA_CHECK(cudaMalloc(&d_kx, 9 * sizeof(int)));
	CUDA_CHECK(cudaMalloc(&d_ky, 9 * sizeof(int)));

	/* Final Image storage vectors for GPU */
	vector<unsigned char> edgeGPU(imgSize);

	/* Check if image size is zero */
	if (imgSize == 0)
	{
		fprintf(stderr, "Zero-size image\n");
		/* Free image allocation memory */
		stbi_image_free(h_src);
		return EXIT_FAILURE;
	}

	/* Sobel kernels host side */
	int h_sobelX[9] = {-1, 0, 1,
					   -2, 0, 2,
					   -1, 0, 1};

	int h_sobelY[9] = {1, 2, 1,
					   0, 0, 0,
					   -1, -2, -1};

	/* CPU Implementation *****************************************************/

	tStart = get_time();
	/* Run CPU-kernel to apply x kernel */
	applyKernelCPU(h_src, h_gx, width, height, h_sobelX, 3);
	/* Run CPU-kernel to apply y kernel */
	applyKernelCPU(h_src, h_gy, width, height, h_sobelY, 3);
	/* Run CPU-kernel to calculate magnitude of each pixel */
	sobelMagnitudeKernel_CPU(h_gx, h_gy, h_out, width, height);

	tEnd = get_time();
	time_cpu = (tEnd - tStart) * 1000.0;
	printf("CPU execution time: %.4f ms\n", time_cpu);
	/* Write Image from CPU Implementation ************************************/

	/* Write PNG (1 channel) */
	if (!stbi_write_png(outputFileCPU.c_str(), width, height, 1,
						(void *)h_out, width))
	{
		fprintf(stderr, "Failed to write PNG %s\n", outputFileGPU.c_str());
	}
	else
	{
		printf("Wrote %s\n", outputFileCPU.c_str());
	}

	/* GPU Implementation *****************************************************/

	tStart = get_time();
	/* Copy inputs and kernels */
	CUDA_CHECK(cudaMemcpy(d_src, h_src, imgSize, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_kx, h_sobelX, 9 * sizeof(int),
						  cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_ky, h_sobelY, 9 * sizeof(int),
						  cudaMemcpyHostToDevice));

	/* Launch kernels */
	dim3 block(16, 16);
	dim3 grid((width + block.x - 1) / block.x,
			  (height + block.y - 1) / block.y);

	/* Run CUDA-kernel to apply x kernel */
	applyKernelCUDA<<<grid, block>>>(d_src, d_gx, width, height, d_kx, 3);
	CUDA_CHECK(cudaGetLastError());

	/* Run CUDA-kernel to apply y kernel */
	applyKernelCUDA<<<grid, block>>>(d_src, d_gy, width, height, d_ky, 3);
	CUDA_CHECK(cudaGetLastError());

	/* Run CUDA-kernel to calculate magnitude of each pixel */
	sobelMagnitudeKernel_GPU<<<grid, block>>>(d_gx, d_gy, d_out, width, height);
	CUDA_CHECK(cudaGetLastError());

	/* Sync CUDA Device (Wait until all operations are complete) */
	CUDA_CHECK(cudaDeviceSynchronize());

	tEnd = get_time();
	time_gpu = (tEnd - tStart) * 1000.0;
	printf("CUDA kernel execution time (excluding malloc/copy): %.4f ms\n", time_gpu);

	printf("Speedup (CPU/GPU): %.3fx\n", time_cpu / time_gpu);

	/* Copy result back to host buffer (correct order: dst, src) */
	CUDA_CHECK(cudaMemcpy(edgeGPU.data(), d_out, imgSize,
						  cudaMemcpyDeviceToHost));

	/* Write Image from GPU Implementation ************************************/

	/* Write PNG (1 channel) */
	if (!stbi_write_png(outputFileGPU.c_str(), width, height, 1,
						edgeGPU.data(), width))
	{
		fprintf(stderr, "Failed to write PNG %s\n", outputFileGPU.c_str());
	}
	else
	{
		printf("Wrote %s\n", outputFileGPU.c_str());
	}

	/* Cleanup */
	CUDA_CHECK(cudaFree(d_src));
	CUDA_CHECK(cudaFree(d_gx));
	CUDA_CHECK(cudaFree(d_gy));
	CUDA_CHECK(cudaFree(d_out));
	CUDA_CHECK(cudaFree(d_kx));
	CUDA_CHECK(cudaFree(d_ky));
	free(h_out);
	stbi_image_free(h_src);

	return 0;
}
