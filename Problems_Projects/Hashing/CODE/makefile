# ==============================================================================
# Main Makefile — Modular CUDA/C++ build system
# ==============================================================================

# ------------------------------------------------------------------------------
# Compiler settings
# ------------------------------------------------------------------------------
NVCC        = nvcc                 # CUDA compiler
NVCC_FLAGS  = -O3 -Iinclude -use_fast_math

CXX         = g++                  # Standard C++ compiler
CXX_FLAGS   = -O2 -Iinclude

# ------------------------------------------------------------------------------
# Detect operating system
# ------------------------------------------------------------------------------
ifeq ($(OS),Windows_NT)
    EXEEXT = .exe
    RM = rm -rf
    OBJEXT = obj
    MKDIR = mkdir -p build
    RMDIR = rm -rf
else
    EXEEXT =
    RM = rm -f
    OBJEXT = o
    MKDIR = mkdir -p build
    RMDIR = rm -rf
endif

# ------------------------------------------------------------------------------
# Include project-specific file lists
# ------------------------------------------------------------------------------
include sources.mk

# ------------------------------------------------------------------------------
# Target executable
# ------------------------------------------------------------------------------
TARGET = build/out$(EXEEXT)

# ------------------------------------------------------------------------------
# Convert source files to object files (place in build/)
# ------------------------------------------------------------------------------
OBJS = $(addprefix build/, $(notdir $(SRCS)))
OBJS := $(OBJS:.c=.$(OBJEXT))
OBJS := $(OBJS:.cu=.$(OBJEXT))
OBJS := $(OBJS:.cpp=.$(OBJEXT))

# ------------------------------------------------------------------------------
# Default target
# ------------------------------------------------------------------------------
all: scrub $(TARGET)
	@./$(TARGET) $(ARGS)

# ------------------------------------------------------------------------------
# Compile only (no run)
# ------------------------------------------------------------------------------
compile: scrub $(TARGET)

# ------------------------------------------------------------------------------
# Linking step
# ------------------------------------------------------------------------------
$(TARGET): $(OBJS)
	@echo "Linking $(TARGET)..."
	@$(NVCC) $(NVCC_FLAGS) $^ -o $@ \
	    -Xcompiler=/EHsc -Xcompiler=/MD \
	    -Xlinker /NODEFAULTLIB:LIBCMT

# ------------------------------------------------------------------------------
# Compile CUDA sources
# ------------------------------------------------------------------------------
build/%.$(OBJEXT): src/%.cu | build
	@echo "Compiling CUDA: $<"
	@$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# ------------------------------------------------------------------------------
# Compile C++ sources
# ------------------------------------------------------------------------------
build/%.$(OBJEXT): src/%.cpp | build
	@echo "Compiling C++: $<"
	@$(CXX) $(CXX_FLAGS) -c $< -o $@

# ------------------------------------------------------------------------------
# Compile C sources (optional)
# ------------------------------------------------------------------------------
build/%.$(OBJEXT): src/%.c | build
	@echo "Compiling C: $<"
	@$(CXX) $(CXX_FLAGS) -c $< -o $@

# ------------------------------------------------------------------------------
# Ensure build directory exists
# ------------------------------------------------------------------------------
build:
	@$(MKDIR)

# ------------------------------------------------------------------------------
# Run explicitly
# ------------------------------------------------------------------------------
run:
	@./$(TARGET) $(ARGS)

# ------------------------------------------------------------------------------
# Clean object files and executable
# ------------------------------------------------------------------------------
clean:
	@echo "Cleaning ..."
	@$(RM) build/*$(OBJEXT) $(TARGET)
	@echo "Done."

# ------------------------------------------------------------------------------
# Remove build folder completely
# ------------------------------------------------------------------------------
scrub:
	@echo "Scrubbing ..."
	@$(RMDIR) build
	@echo "Done."

# ------------------------------------------------------------------------------
# Help section
# ------------------------------------------------------------------------------
help:
	@echo "Makefile Help"
	@echo "==============================="
	@echo "Targets:"
	@echo "  make all         : Clean, compile sources, and run program"
	@echo "  make compile     : Clean and compile only"
	@echo "  make run         : Run the built executable"
	@echo "  make clean       : Remove objects and executable"
	@echo "  make scrub       : Remove entire build folder"
	@echo ""
	@echo "Variables:"
	@echo "  ARGS             : Arguments passed to program"
	@echo ""
	@echo "To add new sources or headers, edit 'sources.mk'"
