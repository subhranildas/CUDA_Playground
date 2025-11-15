# CUDA_Playground

CUDA_Playground is a collection of small, focused experiments around CUDA and GPU
programming, with an emphasis on:

- **Learning CUDA basics** (kernels, grids/blocks, memory transfers)
- **Benchmarking CPU vs GPU** performance
- **Exploring GPU-accelerated primitives** for cryptography and image processing
- **Laying groundwork for GPU-accelerated neural networks**

This repo is meant as a **playground / lab notebook** rather than a single
monolithic application.

---

## Repository Layout

- **Kernels/**  
  Standalone CUDA examples and benchmarks.
  - `main.cu` – minimal kernel that prints block/thread IDs.
  - `vector_add_1.cu` – CPU vs single-kernel GPU vector addition benchmark.
  - `vector_add_2.cu` – more advanced vector addition benchmark comparing
    1D vs 3D grid/block configurations, with timing and correctness checks.
  - `makefile` – cross-platform Makefile for building/running these kernels.

- **Problems_Projects/**  
  Self-contained problem folders exploring GPU-friendly algorithms:
  - `AES/` – notes and (planned) implementation for AES encryption,
    including `Docs/README.md` explaining AES structure and why it maps
    well to GPUs.
  - `Hashing/` – notes and (planned) implementation for SHA-256
    with `Docs/README.md` describing the algorithms and GPU
    considerations.
  - `Box_Blurring/` – image box blur experiment (uses stb image library).
  - `Sobel_Edge/` – Sobel/edge-detection style image processing experiment.

- **VStudio_Projects/**  
  Visual Studio CUDA runtime examples (Windows-centric development setup).

- **Docs/**  
  Reference material and planning documents, e.g. GPU architecture PDFs and
  an NN project plan (`gpu_nn_project_plan.pdf`).

- **Build_Makefile_Template/**  
  Generic Makefile template that can be reused for new CUDA experiments.

---

## Prerequisites

- NVIDIA GPU with CUDA support
- **CUDA Toolkit** installed (for `nvcc` and CUDA runtime headers)
- C/C++ toolchain (e.g. `gcc`/`g++` on Linux, MSVC on Windows)
- (Optional) **Visual Studio** with CUDA integration for the
  `VStudio_Projects` examples

---

## Building & Running the Kernel Examples

### Using the Makefile (Kernels/)

From the `Kernels/` directory:

```bash
make            # cleans, builds default source (main.cu), and runs it
make compile    # cleans and builds, but does not run
make run        # run previously built executable
```

You can override the source list, for example:

```bash
make all SRCS="vector_add_1.cu"
make all SRCS="vector_add_2.cu"
```

Some kernels are simple demos (like `main.cu`), while others are
benchmarks that:

- Allocate large vectors on host and device
- Run CPU and GPU implementations multiple times
- Measure and report timings + speedups
- Verify correctness between CPU and GPU results

### Using Visual Studio (VStudio_Projects/)

On Windows, open the CUDA solution/project under `VStudio_Projects/` in
Visual Studio and build using the standard CUDA project workflow:

1. Open the `.sln` file in Visual Studio.
2. Select the desired configuration (Debug/Release) and platform.
3. Build and run from within the IDE.

---

## Problems & Mini-Projects

The `Problems_Projects/` directory is where more substantial experiments
live. A typical workflow is:

1. Read the `Docs/README.md` in the problem folder (e.g. `AES/Docs`,
   `Hashing/Docs`) for the algorithm/architecture overview.
2. Implement a CPU reference version for correctness.
3. Port the critical path to CUDA kernels.
4. Benchmark CPU vs GPU and iterate on optimizations.

These are in various stages of completion and are intended for
experimentation, not production use.


