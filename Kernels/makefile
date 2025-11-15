# ==============================================================================
# Compiler settings
# ==============================================================================
NVCC        = nvcc          # CUDA compiler
NVCC_FLAGS  = -O3           # Optimization level for CUDA compilation

CC          = gcc           # Standard C compiler
CFLAGS      = -O2           # Optimization level for C compilation

# ==============================================================================
# Detect operating system
# ==============================================================================
ifeq ($(OS),Windows_NT)      # If on Windows
    EXEEXT = .exe            # Executable extension
    RM = rm -rf              # Remove files command (note: will override later for Windows)
    OBJEXT = obj             # Object file extension
    MKDIR = mkdir -p build   # Create build directory
    RMDIR = rm -rf           # Remove directory command
else                          # If on Linux / macOS
    EXEEXT =                 # No extension for Linux
    RM = rm -f               # Remove files command
    OBJEXT = o               # Object file extension
    MKDIR = mkdir -p build   # Create build directory
    RMDIR = rm -rf           # Remove directory command
endif

# ==============================================================================
# Target executable
# ==============================================================================
TARGET      = build/out$(EXEEXT)   # Final executable path in build folder

# ==============================================================================
# Source files and object files
# ==============================================================================
# Default source files (can override with SRCS=)
SRCS      	?= main.cu
# Convert .c sources to .obj/.o
OBJS        = $(SRCS:.c=.$(OBJEXT))
# Convert .cu sources to .obj/.o
OBJS        := $(OBJS:.cu=.$(OBJEXT))
# Put all object files in build/ folder
OBJS        := $(addprefix build/,$(notdir $(OBJS)))

# ==============================================================================
# Default rule: run program
# ==============================================================================
all: scrub $(TARGET)       	# Default 'all' target: scrub + build + run
	@./$(TARGET) $(ARGS)

# ==============================================================================
# Compile only (without running)
# ==============================================================================
compile: scrub $(TARGET)  	# Compile target: cleans + builds but doesn't run

# ==============================================================================
# Link step: create executable from objects
# ==============================================================================
$(TARGET): $(OBJS)
	@$(NVCC) $(NVCC_FLAGS) $^ -o $@

# ==============================================================================
# Compile CUDA files to object files
# ==============================================================================
build/%.$(OBJEXT): %.cu | build  	# Pattern rule for CUDA source files
	@$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# ==============================================================================
# Compile C files to object files
# ==============================================================================
build/%.$(OBJEXT): %.c | build       # Pattern rule for C source files
	@$(CC) $(CFLAGS) -c $< -o $@

# ==============================================================================
# Ensure build directory exists
# ==============================================================================
build:
	@$(MKDIR)

# ==============================================================================
# Run target explicitly
# ==============================================================================
run:
	@./$(TARGET)

# ==============================================================================
# Clean build artifacts (remove objects and executable)
# ==============================================================================
clean:
	@echo "Cleaning ..."
	@$(RM) build/*$(OBJEXT) $(TARGET)
	@echo "Done ..."

# ==============================================================================
# Remove entire build folder
# ==============================================================================
scrub:
	@echo "Scrubbing ..."
	@$(RMDIR) build
	@echo "Done ..."

# ==============================================================================
# Help section
# ==============================================================================
help:
	@echo "Makefile Help"
	@echo "==============================="
	@echo "Default behavior:"
	@echo "  - make all        : Cleans build folder, compiles default source (main.cu), and runs"
	@echo "  - By default, only 'main.cu' is compiled."
	@echo ""
	@echo "To compile multiple source files:"
	@echo "  make all SRCS=\"main.cu utils.c cpu_vector_add.c\""
	@echo "    - Override SRCS variable to include all desired source files"
	@echo "    - This will clean, compile all listed sources, and run the program"
	@echo ""
	@echo "Targets:"
	@echo "  make all       : Clean build folder, compile sources, and run program"
	@echo "  make compile   : Clean build folder and compile sources (do not run)"
	@echo "  make run       : Run the compiled executable (after make all or make compile)"
	@echo "  make clean     : Remove object files and executable but keep build folder"
	@echo "  make scrub     : Remove entire build folder including all object files"
	@echo ""
	@echo "Variables:"
	@echo "  SRCS           : Source files to compile (default: main.cu)"
	@echo "  ARGS           : Command-line arguments to pass to the program"
	@echo ""
	@echo "Example Usage:"
	@echo "  make all ARGS=\"1000000\"                	# Build and run with argument"
	@echo "  make compile SRCS=\"main.cu utils.c\"     	# Compile multiple sources only"
	@echo "  make run ARGS=\"1000000\"                 	# Run the program after compilation"
	@echo "  make clean                                	# Clean objects and executable"
	@echo "  make scrub                                	# Delete build folder entirely"
