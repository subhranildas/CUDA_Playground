/* ==================================================================================
# CUDA runtime APIs
===================================================================================*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

int main()
{
	int numberOfDevices;
	cudaDeviceProp deviceProperties;

	cudaGetDeviceCount(&numberOfDevices);
	printf("Number of GPU Devices found: %d\n", numberOfDevices);

	/* Get The CUDA Device Properties (This API will populate the cudaDeviceProp Structure */
	cudaGetDeviceProperties(&deviceProperties, 0);

	/* Print some information from the cudaDeviceProp Structure */
	printf("Cuda Compute Capability: %d.%d\n", deviceProperties.major, deviceProperties.minor);
	printf("\n");
	printf("Total Global Memory in Gbytes: %.2lf\n", ((double)deviceProperties.totalGlobalMem/ (double)(1024*1024*1024)));
	printf("Global memory bus width in bits: %d\n", deviceProperties.memoryBusWidth);
	printf("Size of L2 cache in Mbytes: %.2lf\n", (double)deviceProperties.l2CacheSize / (double)(1024 * 1024));
	printf("Number of multiprocessors(SMs) on device: %d\n", deviceProperties.multiProcessorCount);
	printf("\n");
	printf("Shared memory available per block in bytes: %llu\n", deviceProperties.sharedMemPerBlock);
	printf("32-bit registers available per block: %d\n", deviceProperties.regsPerBlock);
	printf("\n");
	printf("Maximum number of threads per block: %d\n", deviceProperties.maxThreadsPerBlock);
	printf("Maximum resident threads per multiprocessor: %d\n", deviceProperties.maxThreadsPerMultiProcessor);
	printf("Warp size in threads: %d\n", deviceProperties.warpSize);
	printf("\n");
	printf("Maximum size of each dimension of a block: x: %d | y: %d | z: %d\n", deviceProperties.maxThreadsDim[0], deviceProperties.maxThreadsDim[1], deviceProperties.maxThreadsDim[2]);
	printf("Maximum size of each dimension of a grid: x: %d | y: %d | z: %d\n", deviceProperties.maxGridSize[0], deviceProperties.maxGridSize[1], deviceProperties.maxGridSize[2]);

}
