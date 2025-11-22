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

---

**Troubleshooting**

- nvcc errors: confirm CUDA toolkit is installed and `nvcc` is on PATH. On Windows ensure Visual Studio Build Tools are available.
- Linker errors: make sure `sources.mk` is included by `Makefile` (it is in this project). If you change linkage or add files, run `make clean` before rebuilding.
- If tests fail, run the CPU AES reference (`aes_encrypt_ecb`, `aes_decrypt_ecb`) to isolate whether the issue is in the CUDA path or the reference implementation.

---

**Adding test vectors**

- Add new test cases to `CODE/tests/src/test_aes_ecb.cpp`. The tests use known-answer vectors (KATs) and multi-block round-trips.
