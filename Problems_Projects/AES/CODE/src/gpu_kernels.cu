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
#include "utils.h"
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
 * GPU Clamp Function
==============================================================================*/
__device__ inline int clamp_gpu(int val, int low, int high)
{
	return max(low, min(val, high));
}

/*==============================================================================
 * Basic Convolution Kernel (GPU)
==============================================================================*/

/*==============================================================================
 * Convolution Kernel (CPU)
==============================================================================*/

/*==============================================================================
 * Compute Sobel Magnitude (GPU)
==============================================================================*/

/*==============================================================================
 * Compute Sobel Magnitude (CPU)
==============================================================================*/
