# Understanding AES (Advanced Encryption Standard)

## What is AES on a High‑Level

AES is a symmetric‑key block cipher standardized by NIST.

- **Symmetric key** → Same key used for encryption and decryption.
- **Block cipher** → Operates on fixed-size blocks (128 bits).
- **Key sizes** → 128, 192, or 256 bits.
- **Widely used** in HTTPS, VPNs, disk encryption, secure messaging, etc.

# AES — CODE Directory (Quick Guide)

This document describes the `CODE/` folder in this project, how to build and run the AES reference and CUDA implementations, and how to run the unit tests.

**Location:** `Problems_Projects/AES/CODE`

**Purpose:** reference CPU AES implementation, CUDA AES kernels (ECB), and a small test harness.

**Contents (high level)**

- `include/` — public headers (`aes.h`, `utils.h`, `kernels.h`, `aes_test.h`)
- `src/` — implementation sources (`aes_cpu.cpp`, `aes_gpu.cu`, `utils.cpp`, `main.cu`)
- `tests/` — test harness and `Makefile` for running unit tests
- `Makefile`, `sources.mk` — top-level build and source list

---

**Quick Start — Build & Run**

From the project AES `CODE` folder run:

```bash
cd Problems_Projects/AES/CODE
make all      # builds and runs the example target (build/out or build/out.exe)
```

If you only want to compile:

```bash
make compile
```

Clean up build artifacts:

```bash
make clean    # remove objects + executable
make scrub    # remove entire build/ directory
```

Notes:

- Building GPU code requires the CUDA toolkit (`nvcc`). On Windows, `nvcc` may invoke MSVC; ensure Visual Studio Build Tools are available.

---

## Running the Application (Benchmarks)

The main application (`main.cu`) runs AES-128 ECB benchmarks comparing CPU vs GPU performance on various data sizes.

### Step-by-Step Instructions

**1. From the CODE directory, clean previous builds:**

```bash
cd Problems_Projects/AES/CODE
make clean
```

**2. Build the application:**

```bash
make all
```

This compiles all sources and runs the benchmark. The output will show:

- CPU and GPU encryption times for each data size
- Throughput (MB/s) for both implementations
- Speedup factor (CPU time / GPU time)
- Verification that results match

**3. Run the compiled executable directly:**

```bash
./build/out.exe        # Windows/MSVC
# or
./build/out            # Linux/GCC
```

**4. Benchmark test sizes include:**

- 1 KB (64 blocks)
- 10 KB (640 blocks)
- 100 KB (6400 blocks)
- 1 MB (65536 blocks)
- 10 MB (655360 blocks)
- **100 MB (6553600 blocks)**

**Expected output format:**

```
========================================
  AES-128 ECB CPU vs GPU Benchmark
========================================

Test size: 104857600 bytes (6553600 blocks)
  CPU time: 11300 ms (8.85 MB/s)
  GPU time: 296 ms (337.84 MB/s)
  Speedup:  38.18x
  Match:    YES
```

---

## Running Unit Tests

The `tests/` folder contains comprehensive CPU and GPU unit tests for AES-ECB.

### Step-by-Step Instructions

**1. Navigate to the tests directory:**

```bash
cd Problems_Projects/AES/CODE/tests
```

**2. Clean previous test builds:**

```bash
make clean
```

**3. Build the test suite:**

```bash
make all
```

This compiles and links all test sources (CPU tests from `test_aes_ecb.cpp` and GPU tests from `test_aes_ecb_gpu.cu`).

**4. Run the tests:**

```bash
make run
```

Or run the compiled executable directly:

```bash
./build/tests_out.exe   # Windows/MSVC
# or
./build/tests_out       # Linux/GCC
```

**5. Expected test output:**

Both CPU and GPU tests run together and verify:

- AES-128 single-block known-answer tests (FIPS-197 vectors)
- AES-128 multi-block round-trip tests (4 blocks = 64 bytes)
- AES-192 multi-block round-trip tests
- AES-256 multi-block round-trip tests

```
========================================
   AES ECB CPU & GPU Test Suite
========================================

========== CPU AES ECB Tests ==========
[CPU] AES-128 single-block round-trip PASSED
[CPU] AES-128 multi-block round-trip PASSED
[CPU] AES-192 multi-block round-trip PASSED
[CPU] AES-256 multi-block round-trip PASSED
=======================================

========== GPU AES ECB Tests ==========
[GPU] AES-128 single-block round-trip PASSED
[GPU] AES-128 multi-block round-trip PASSED
[GPU] AES-192 multi-block round-trip PASSED
[GPU] AES-256 multi-block round-trip PASSED
=======================================

All AES ECB CPU & GPU tests PASSED
```

**6. Build CPU-only tests (without GPU):**

To test only the CPU implementation:

```bash
# Edit tests/sources.mk and remove the line: src/test_aes_ecb_gpu.cu
nano tests/sources.mk   # remove GPU test file from TEST_SRCS
make clean && make all
```

---

## Common Build Commands Reference

From `CODE/` directory:

| Command        | Purpose                          |
| -------------- | -------------------------------- |
| `make all`     | Build and run the main benchmark |
| `make compile` | Compile without running          |
| `make clean`   | Remove objects and executable    |
| `make scrub`   | Remove entire build directory    |
| `make run`     | Run the compiled executable      |

From `CODE/tests/` directory:

| Command        | Purpose                            |
| -------------- | ---------------------------------- |
| `make all`     | Build and run tests                |
| `make compile` | Compile test sources only          |
| `make clean`   | Remove test objects and executable |
| `make run`     | Run the test executable            |

---

**Quick Start — Build & Run**

From the project AES `CODE` folder run:

```bash
cd Problems_Projects/AES/CODE
make all      # builds and runs the example target (build/out or build/out.exe)
```

If you only want to compile:

```bash
make compile
```

Clean up build artifacts:

```bash
make clean    # remove objects + executable
make scrub    # remove entire build/ directory
```

Notes:

- Building GPU code requires the CUDA toolkit (`nvcc`). On Windows, `nvcc` may invoke MSVC; ensure Visual Studio Build Tools are available.

---

**Running Unit Tests**

Tests are under `CODE/tests/` and have a separate Makefile. They validate AES-ECB using known vectors and multi-block round-trips.

Build and run tests:

```bash
cd Problems_Projects/AES/CODE/tests
make all
make run
# or ./build/tests_out(.exe)
```

The tests compile CUDA sources with `nvcc` and C++ tests with `g++` by default. The `tests/Makefile` handles Windows `.obj` object extensions so `nvcc`/MSVC behave cleanly.

To run CPU-only tests: remove `../src/aes_gpu.cu` from `tests/sources.mk` (or edit the `tests/Makefile`) and rebuild.

---

**Public API (in `include/aes.h`)**

High-level functions you can call from other code:

- `aes_error_te aes_encrypt_ecb(...)` — CPU ECB encrypt (block-aligned input required)
- `aes_error_te aes_decrypt_ecb(...)` — CPU ECB decrypt
- `aes_error_te aes_encrypt_ecb_cuda(...)` — GPU ECB encrypt wrapper
- `aes_error_te aes_decrypt_ecb_cuda(...)` — GPU ECB decrypt wrapper

Error codes: `AES_SUCCESS`, `AES_ERROR_UNSUPPORTED_KEY_SIZE`, `AES_ERROR_MEMORY_ALLOCATION_FAILED`, `AES_ERROR_CUDA_FAILURE`, `AES_ERROR_INVALID_INPUT_LENGTH`.

See `include/aes.h` for full function signatures.

---

**Implementation notes**

- `aes_cpu.cpp` contains a straightforward, readable AES implementation used as the correctness reference.
- `aes_gpu.cu` contains device functions and kernels for AES encrypt/decrypt (ECB), plus host wrappers that expand keys on the host and launch the kernels.
- `main.cu` demonstrates a round-trip: encrypt then decrypt using the CUDA wrappers.

Security note: ECB mode is insecure for most uses — this project uses ECB for simple testing and correctness verification. For real workloads, prefer `CTR`, `GCM` or authenticated modes.
