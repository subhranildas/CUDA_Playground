# ==============================================================================
# sources.mk — Lists of all source and header files
# ==============================================================================

# ------------------------------------------------------------------------------
# Source files (add all C and CUDA files here)
# ------------------------------------------------------------------------------

SRCS = src/main.cu \
	src/aes_cpu.cpp \
	src/aes_gpu.cu \
	src/utils.cpp

# ------------------------------------------------------------------------------
# Header files (optional, for dependency tracking or IDE support)
# ------------------------------------------------------------------------------
HEADERS = \
	include/kernels.h \
	include/utils.h \
	include/aes.h \
	include/aes_test.h