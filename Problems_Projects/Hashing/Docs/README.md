## What is SHA-256? (short primer)

SHA-256 is a cryptographic hash function from the SHA-2 family. It takes an arbitrary-length input (message) and deterministically produces a 256-bit (32-byte) digest.

### Properties:

- Deterministic and fixed-size output.
- Avalanche: A small input change drastically changes the output.
- Preimage / second-preimage resistance and collision resistance (practical security assumptions).
- Uses a compressing function over 512-bit blocks. Each 512-bit block is processed through a sequence of 64 rounds using integer bitwise ops (rotates, XORs, additions) and a set of constants.
- Message must be padded (1 bit then zeros then message length) to a multiple of 512 bits.

> The core compression function is relatively small and has internal data dependencies, so the usual high GPU speedup strategy is to run many independent SHA-256 computations in parallel (many messages) rather than trying to parallelize a single hash’s internal rounds across threads.

## What is AES ?

AES (Advanced Encryption Standard) is a symmetric block cipher standardized by NIST. Common facts you should know:

### Properties:

- AES operates on 128-bit blocks.
- Key sizes: AES-128, AES-192, AES-256 (number of rounds = 10, 12, 14 respectively).
- Encryption is a sequence of rounds: SubBytes (S-box), ShiftRows, MixColumns, AddRoundKey. The final round omits MixColumns.
- Uses a key schedule to derive round keys from the master key.

AES is a building block — real systems use modes of operation (ECB, CBC, CTR, GCM, etc.).
For GPU work, CTR (counter) and ECB are the easiest to parallelize because each block is independent;
CBC encrypt is sequential (unless using special techniques), but CBC decrypt is parallelizable.

Implementations vary: table-based (T-tables), S-box + MixColumns math, or bitsliced.
On CPU the fastest often use AES-NI instructions, but GPUs don’t have AES-NI, the logic must be implemented in software.
