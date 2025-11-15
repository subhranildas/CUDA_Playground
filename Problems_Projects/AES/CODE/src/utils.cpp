#include "utils.h"

using namespace std;

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