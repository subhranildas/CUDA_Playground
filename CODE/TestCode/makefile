# Compiler
NVCC = nvcc

# Compiler flags
# -g   - Generates debug information for host code
# -G   - Generates debug information for device code
# -O3  - Optimization level 3
NVCC_FLAGS = -O3

# Executable name
TARGET = main

# Source file
SRC = main.cu

# Default target, called when you just run "make"
all: build

# Build the executable
build: $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC)

# Run the executable
run: all
	./$(TARGET)

# Clean up build files
clean:
	rm -f $(TARGET)