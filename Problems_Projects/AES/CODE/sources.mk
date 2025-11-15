# ==============================================================================
# sources.mk — Lists of all source and header files
# ==============================================================================

# ------------------------------------------------------------------------------
# Source files (add all C and CUDA files here)
# ------------------------------------------------------------------------------

SRCS = src/main.cu \
       src/cpu_kernels.cpp \
	   src/gpu_kernels.cu \
       src/utils.cpp \
	   src/aes.cpp

# ------------------------------------------------------------------------------
# Header files (optional, for dependency tracking or IDE support)
# ------------------------------------------------------------------------------
HEADERS = \
	include/kernels.h \
	include/utils.h \
	include/aes.h