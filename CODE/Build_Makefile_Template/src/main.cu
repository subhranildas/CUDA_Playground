#include "main.h"

/* Kernel Function : To be executed in GPU not CPU */
__global__ void test_01()
{
	/* Print the Blocks and Threads IDs */
	printf("\nThe Block ID is %d --- The Thread ID is %d", blockIdx.x, threadIdx.x);
}

/* CUDA C always requires a main Function */
int main()
{
	test_01<<<1, 1>>>();
}