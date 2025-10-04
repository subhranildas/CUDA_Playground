# ==============================================================================
# Main Makefile — Modular CUDA/C build system
# ==============================================================================

# ------------------------------------------------------------------------------
# Compiler settings
# ------------------------------------------------------------------------------
NVCC        = nvcc          	# CUDA compiler
NVCC_FLAGS  = -O3 -Iinclude 	# Optimization level and include path for headers

CC          = gcc           	# Standard C compiler
CFLAGS      = -O2 -Iinclude  	# Optimization level and include path

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
OBJS = $(SRCS:.c=.$(OBJEXT))
OBJS := $(OBJS:.cu=.$(OBJEXT))
OBJS := $(addprefix build/,$(notdir $(OBJS)))

# ------------------------------------------------------------------------------
# Default target
# ------------------------------------------------------------------------------
all: scrub $(TARGET)
	@$(TARGET) $(ARGS)

# ------------------------------------------------------------------------------
# Compile only (no run)
# ------------------------------------------------------------------------------
compile: scrub $(TARGET)

# ------------------------------------------------------------------------------
# Linking step
# ------------------------------------------------------------------------------
$(TARGET): $(OBJS)
	@echo "Linking $(TARGET)..."
	@$(NVCC) $(NVCC_FLAGS) $^ -o $@

# ------------------------------------------------------------------------------
# Compile CUDA sources
# ------------------------------------------------------------------------------
build/%.$(OBJEXT): src/%.cu | build
	@echo "Compiling CUDA: $<"
	@$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# ------------------------------------------------------------------------------
# Compile C sources
# ------------------------------------------------------------------------------
build/%.$(OBJEXT): src/%.c | build
	@echo "Compiling C: $<"
	@$(CC) $(CFLAGS) -c $< -o $@

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
# Clean build artifacts
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
