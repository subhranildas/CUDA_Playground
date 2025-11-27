# This file is deprecated. Use ../test_sources.mk instead.

# Kept for backwards compatibility with older workflows.
TEST_SRCS = \
    src/test_aes_ecb.cpp \
    src/test_aes_ecb_gpu.cu \
    src/main.cu \
    ../src/aes_cpu.cpp \
    ../src/aes_gpu.cu \
    ../src/utils.cpp

TEST_HEADERS = \
    ../include/aes.h \
    ../include/kernels.h \
    ../include/utils.h \
    include/aes_test.h
