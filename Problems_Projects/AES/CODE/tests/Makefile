# Tests/Makefile — builds and runs unit tests for the CODE project
# This Makefile is intended to be invoked as: `make -C tests [target]`

include sources.mk

CXX = g++
NVCC = nvcc
CXX_FLAGS = -O2 -I.. -I../include -Iinclude
NVCC_FLAGS = -O2 -I.. -I../include -Iinclude

OBJEXT = o
# If running on Windows / MSYS/MinGW prefer .obj to avoid MSVC `cl` warnings
UNAME_S := $(shell uname -s 2>/dev/null || echo)
ifeq ($(OS),Windows_NT)
	OBJEXT = obj
endif
ifneq (,$(findstring MINGW,$(UNAME_S)))
	OBJEXT = obj
endif

MKDIR = mkdir -p build

# Convert TEST_SRCS entries (which are root-relative `tests/...`) to local paths
TEST_SRCS_LOCAL := $(patsubst tests/%,%,$(TEST_SRCS))

# Separate CUDA and C++ sources
CPP_SRCS := $(filter %.cpp,$(TEST_SRCS_LOCAL))
CU_SRCS  := $(filter %.cu,$(TEST_SRCS_LOCAL))

CPP_OBJS := $(addprefix build/,$(notdir $(CPP_SRCS:.cpp=.${OBJEXT})))
CU_OBJS  := $(addprefix build/,$(notdir $(CU_SRCS:.cu=.${OBJEXT})))

OBJS := $(CPP_OBJS) $(CU_OBJS)

TEST_TARGET = build/tests_out

.PHONY: all run clean

all: $(TEST_TARGET)

$(TEST_TARGET): $(OBJS) | build
	@echo "Linking test executable $(TEST_TARGET)..."
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

# C++ compile rule (test-local C++ sources)
build/%.$(OBJEXT): src/%.cpp | build
	@echo "Compiling C++ test: $<"
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# Also support implementation sources that live in ../src (project root)
build/%.$(OBJEXT): ../src/%.cpp | build
	@echo "Compiling C++ (project) source: $<"
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# CUDA compile rule (test-local .cu)
build/%.$(OBJEXT): src/%.cu | build
	@echo "Compiling CUDA source: $<"
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Also support CUDA sources in ../src
build/%.$(OBJEXT): ../src/%.cu | build
	@echo "Compiling CUDA (project) source: $<"
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

build:
	$(MKDIR)

run: all
	@echo "Running tests..."
	@if [ -x $(TEST_TARGET) ]; then \
		./$(TEST_TARGET); \
	 elif [ -x $(TEST_TARGET).exe ]; then \
		./$(TEST_TARGET).exe; \
	 else \
		echo "No test binary found ($(TEST_TARGET) or $(TEST_TARGET).exe)"; exit 1; \
	fi

clean:
	@echo "Cleaning tests..."
	@rm -rf build $(TEST_TARGET)
