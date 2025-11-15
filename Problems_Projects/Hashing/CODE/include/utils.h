/*==============================================================================
 * Includes related to standard Cpp libraries and CUDA runtime libraries
==============================================================================*/
#include <iostream>
#include <vector>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h> // For high-resolution timers on Windows
#endif

#ifdef _WIN32
double get_time(void);
#elif __linux__
double get_time(void);
#endif