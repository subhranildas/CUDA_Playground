# Understanding AES (Advanced Encryption Standard)

## What is AES on a High‑Level

AES is a symmetric‑key block cipher standardized by NIST.

- **Symmetric key** → Same key used for encryption and decryption.
- **Block cipher** → Operates on fixed-size blocks (128 bits).
- **Key sizes** → 128, 192, or 256 bits.
- **Widely used** in HTTPS, VPNs, disk encryption, secure messaging, etc.

AES is fast, secure, and hardware‑accelerated on most CPUs and GPUs.

## AES Structure at a Glance

AES is built from repeated **Rounds** of transformations applied to a 16‑byte state.

- AES‑128 = 10 rounds
- AES‑192 = 12 rounds
- AES‑256 = 14 rounds

Each round transforms the internal 4×4 byte matrix using:

- **SubBytes** – Nonlinear substitution using S‑box
- **ShiftRows** – Circular shifts of each row
- **MixColumns** – Mixes each column using finite‑field math
- **AddRoundKey** – XOR with round key

> The **final round** does **not** have MixColumns.

## Data Representation

AES stores the 16‑byte block in a 4×4 matrix (called the “state”), column‑major:

```bash
s0  s4  s8  s12
s1  s5  s9  s13
s2  s6  s10 s14
s3  s7  s11 s15
```

## AES Main Steps

### SubBytes

Each byte is replaced with a value from a **fixed 256‑byte lookup table** (the S‑box).
The S‑box is derived mathematically and provides resistance against attacks.

### ShiftRows

Each row is rotated:

- Row 0 → no shift
- Row 1 → shift left by 1
- Row 2 → shift left by 2
- Row 3 → shift left by 3

### MixColumns

Each column is transformed using matrix multiplication in GF(2⁸):

```
[02 03 01 01]   [s0 ]
[01 02 03 01] * [s1 ]
[01 01 02 03]   [s2 ]
[03 01 01 02]   [s3 ]
```

(These multiplications use bitwise operations, not standard integer math.)

### AddRoundKey

Each byte in the state is XORed with one byte from the expanded round key.

---

## Key Expansion (Key Schedule)

AES doesn’t use the same key for each round.
Instead, it expands the key into multiple round keys using:

- **RotWord** – rotate bytes
- **SubWord** – apply S‑box
- **Rcon** – round constant XOR
- Operations differ slightly across 128/192/256‑bit versions.

AES‑128 key expansion outputs **11 round keys** (for 10 rounds + initial).

---

## Full AES‑128 Encryption Process

```bash
Input (16 bytes)
        ↓
AddRoundKey (Round 0)
        ↓
Round 1:
    SubBytes → ShiftRows → MixColumns → AddRoundKey
Round 2:
    SubBytes → ShiftRows → MixColumns → AddRoundKey
...
Round 9:
    SubBytes → ShiftRows → MixColumns → AddRoundKey
Round 10 (Final):
    SubBytes → ShiftRows → AddRoundKey  (no MixColumns)
        ↓
Ciphertext (16 bytes)
```

---

## AES Modes of Operation

AES by itself only encrypts 16 bytes.
Modes allow encryption of arbitrary-length data:

- **ECB** – insecure, do NOT use
- **CBC** – classic mode (requires IV)
- **CTR** – counter mode, parallelizable (good for GPU)
- **GCM** – authenticated encryption (AEAD), modern and widely used
- **XTS** – disk encryption

For GPU projects, **CTR is the easiest** because each block is independent.

---

## Why AES is Good for GPUs

AES is great for GPU acceleration because:

- Each block is 16 bytes → each block fits perfectly in registers
- CTR mode → each block processed independently
- S‑boxes can be stored in shared/constant memory
- Thousands of threads can run one block each
- Good memory‑to-compute ratio

GPU AES can achieve **10×–20× speedups** depending on the mode and device.

---

## Implementation Steps

1. **CPU reference implementation**
   – Ensures correctness before touching GPU code.

2. **AES‑CTR mode (recommended)**
   – Parallelizable, fast.

3. **Device functions for**

   - SubBytes (S‑box lookup)
   - ShiftRows
   - MixColumns
   - AddRoundKey
   - Key schedule (host-side, usually)

4. **Kernel design**

   - Each thread handles one AES block (16 bytes)
   - Counter block generation per thread
   - XOR with plaintext for CTR mode

5. **Validation**

   - Compare CPU vs GPU outputs for multiple test vectors.

6. **Optimization**
   - S‑box + inverse S‑box in shared memory
   - Use constant memory for round keys
   - Unroll loops
   - Avoid branching
   - Experiment with warp‑level primitives

---

## 10. Next Steps

If you want, we can continue with:

- A) **CPU AES‑128 reference implementation (C)**
- B) **CUDA AES‑CTR implementation**
- C) **Project folder + build system + test vectors**
- D) **GPU optimization plan**

Tell me which step you want next.
