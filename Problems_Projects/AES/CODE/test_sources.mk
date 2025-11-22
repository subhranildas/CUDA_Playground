# test_sources.mk — lists test-specific sources (root-level)
# This file is included by the top-level Makefile and by tests/Makefile

# Test sources: include the test harness, a test `main`, and core implementation
TEST_SRCS = \
    tests/src/test_aes_ecb.cpp \
    tests/src/main.cu \
    ../src/aes_cpu.cpp \
    ../src/aes_gpu.cu \
    ../src/utils.cpp

# Test headers: include all headers from the project's `include/` folder
TEST_HEADERS = \
    include/aes.h \
    include/kernels.h \
    include/utils.h \
    include/aes_test.h
