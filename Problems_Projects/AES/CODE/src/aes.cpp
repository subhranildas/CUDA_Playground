#include "aes.h"

/* =============================================================================
 * Macro Defines
 * ========================================================================== */
#define BITS_PER_BYTE 8
#define WORD_SIZE 4

/* Galois Field (GF(2^8)) constants for the MixColumns step. */
/* Irreducible polynomial for AES: x^8 <-> x^4 + x^3 + x + 1 */
#define GF_REDUCING_POLYNOMIAL 0x1B
#define GF_MSB_MASK 0x80
/* =============================================================================
 * S-Box Lookup Tables
 * ========================================================================== */

static const uint8_t sbox_lookup[256] = {
	// 0	 1	   2   	 3	   4	 5	   6	 7	   8	 9	   A	 B	   C 	 D	   E	 F
	0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
	0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
	0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
	0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
	0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
	0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
	0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
	0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
	0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
	0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
	0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
	0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
	0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
	0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
	0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
	0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16};

/* =============================================================================
 * Inverse S-Box Lookup Tables
 * ========================================================================== */

static const uint8_t inv_sbox_lookup[256] = {
	// 0     1     2     3     4     5     6     7     8     9     A     B     C     D     E     F
	0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
	0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
	0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
	0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
	0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
	0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
	0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
	0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
	0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
	0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
	0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
	0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
	0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
	0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
	0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
	0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d};

// static const uint8_t Rcon[11] = {
// 	0x00000000UL,
// 	0x01000000UL, 0x02000000UL, 0x04000000UL, 0x08000000UL,
// 	0x10000000UL, 0x20000000UL, 0x40000000UL, 0x80000000UL,
// 	0x1b000000UL, 0x36000000UL};
static const uint8_t rcon[] = {
	0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36,
	0x6c, 0xd8, 0xab, 0x4d, 0x9a, 0x2f, 0x5e, 0xbc, 0x63, 0xc6, 0x97,
	0x35, 0x6a, 0xd4, 0xb3, 0x7d, 0xfa, 0xef, 0xc5, 0x91, 0x39};

/* =============================================================================
 * Private Utility Functions
 * ========================================================================== */

/* =============================================================================
 * Securely overwrite a memory region with zeros.
 * Writes `n` zero-bytes to the buffer pointed to by `ptr`.
 * Uses a volatile pointer to prevent the compiler from optimizing
 * the clearing operation away (important for wiping sensitive data).
 ============================================================================ */
static void memory_write_zeros(void *ptr, unsigned long long n)
{
	if ((ptr == NULL) || (n == 0))
		return;

	/* Typecast to byte Pointer */
	volatile uint8_t *vptr = (volatile uint8_t *)ptr;

	while (n > 0)
	{
		*vptr = 0;
		vptr++;
		n--;
	}
}

/* Rotate a 32-bit value left by 8 bits (1 byte) */
static inline uint32_t rotl8(uint32_t x) { return (x << 8) | (x >> 24); }

/* Rotate a 32-bit value left by 16 bits (2 bytes) */
static inline uint32_t rotl16(uint32_t x) { return (x << 16) | (x >> 16); }

/* Rotate a 32-bit value left by 24 bits (3 bytes) */
static inline uint32_t rotl24(uint32_t x) { return (x << 24) | (x >> 8); }

/* Rotate a 32-bit value right by 8 bits (1 byte) */
static inline uint32_t rotr8(uint32_t x) { return (x >> 8) | (x << 24); }

/* Rotate a 32-bit value right by 16 bits (2 bytes) */
static inline uint32_t rotr16(uint32_t x) { return (x >> 16) | (x << 16); }

/* Rotate a 32-bit value right by 24 bits (3 bytes) */
static inline uint32_t rotr24(uint32_t x) { return (x >> 24) | (x << 8); }

/* =============================================================================
 * Rotate a 4-byte AES word left by 1 byte (8 bits).
 *
 * The AES key schedule repeatedly rotates 32-bit words.
 * This optimized version treats the word as a uint32_t and performs
 * a single 8-bit left rotation:
 *
 *   [b0 b1 b2 b3] → [b1 b2 b3 b0]
 *
 * This is equivalent to:
 *      v = (v << 8) | (v >> 24)
 *
 * Parameters:
 *   word – Pointer to a 4-byte array to be rotated.
 ============================================================================ */

static inline void word_rotate_left(uint8_t *word)
{

	uint32_t *w32 = (uint32_t *)word;
	uint32_t v = *w32;

	v = (v << 8) | (v >> 24); // rotate left by 8 bits

	*w32 = v;
}

/* =============================================================================
 * Perform the AES key schedule core transformation on a 4-byte word.
 *
 * This function is applied once per 128/192/256-bit key block during
 * key expansion. The transformation consists of:
 *
 *   1. RotWord  – rotate the word left by 1 byte.
 *   2. SubWord  – apply the AES S-Box to each byte.
 *   3. XOR with Rcon[iteration] on the first byte.
 *
 * This produces the key-dependent nonlinearity that drives security
 * in the AES key schedule.
 *
 * Parameters:
 *   word       – 4-byte word to be transformed.
 *   iteration  – Rcon index used for the round.
 ============================================================================ */

static void key_schedule_core(uint8_t *word, uint8_t iteration)
{
	word_rotate_left(word);

	for (uint8_t i = 0; i < WORD_SIZE; ++i)
		word[i] = sbox_lookup[word[i]];

	word[0] ^= rcon[iteration];
}

/* =============================================================================
 * Expand the user-provided AES key into the full key schedule.
 *
 * This function generates all round keys required for AES encryption
 * and decryption. AES uses the Rijndael key schedule, which takes the
 * original key (128/192/256 bits) and expands it into multiple 16-byte
 * round keys used in each encryption round.
 *
 * Process overview:
 * 1. Copy the original key into the beginning of expanded_key.
 * 2. Repeatedly generate new 4-byte words until the full expanded
 *    key size is reached.
 * 3. Every key_size bytes, apply the key schedule core:
 *       - Rotate the word left by 1 byte
 *       - Substitute each byte using the S-box
 *       - XOR the first byte with the appropriate Rcon value
 * 4. For AES-256, perform an additional S-box substitution on the
 *    middle word of each 256-bit key block.
 * 5. XOR the transformed temp_word with the word key_size bytes earlier
 *    to form the next word of the expanded key.
 *
 * Parameters:
 *   expanded_key       – Output buffer that receives the full key schedule.
 *   key                – The original AES key provided by the user.
 *   key_size           – Size of the original key (128/192/256 bits).
 *   expanded_key_size  – Total number of bytes to generate.
 ============================================================================ */
void aes_expand_key(uint8_t *expanded_key, const uint8_t *key,
					aes_key_size_te key_size, size_t expanded_key_size)
{
	size_t current_size = (size_t)key_size;
	uint8_t rcon_iteration = 1;
	uint8_t temp_word[WORD_SIZE];

	for (size_t i = 0; i < current_size; i++)
		expanded_key[i] = key[i];

	while (current_size < expanded_key_size)
	{
		for (size_t i = 0; i < WORD_SIZE; i++)
			temp_word[i] = expanded_key[current_size - WORD_SIZE + i];

		if (current_size % (size_t)key_size == 0)
		{
			key_schedule_core(temp_word, rcon_iteration++);
		}

		if (key_size == AES_KEY_SIZE_256 && (current_size % (size_t)key_size) ==
												AES_BLOCK_SIZE)
		{
			for (size_t i = 0; i < WORD_SIZE; i++)
				temp_word[i] = sbox_lookup[temp_word[i]];
		}

		for (size_t i = 0; i < WORD_SIZE; i++)
		{
			expanded_key[current_size] = expanded_key[current_size - (size_t)key_size] ^ temp_word[i];
			current_size++;
		}
	}
}

/* =============================================================================
 * Perform the AES SubBytes step.
 * Each byte in the 4×4 state matrix is replaced using the
 * forward S-box (sbox_lookup), introducing non-linearity.
 ============================================================================ */
static void substitute_bytes(aes_state_t *state)
{
	for (int r = 0; r < AES_STATE_DIM; ++r)
		for (int c = 0; c < AES_STATE_DIM; ++c)
			(*state)[r][c] = sbox_lookup[(*state)[r][c]];
}

/* =============================================================================
 * Perform the AES inverse SubBytes step.
 * Each byte in the state is replaced using the inverse S-box
 * (inv_sbox_lookup), reversing the SubBytes operation during decryption.
 ============================================================================ */
static void inverse_substitute_bytes(aes_state_t *state)
{
	for (int r = 0; r < AES_STATE_DIM; ++r)
		for (int c = 0; c < AES_STATE_DIM; ++c)
			(*state)[r][c] = inv_sbox_lookup[(*state)[r][c]];
}

/* =============================================================================
 * shift_rows()
 *
 * Perform the AES ShiftRows transformation.
 *
 * ShiftRows cyclically rotates each row of the 4×4 AES state to the left by a
 * fixed number of bytes:
 *
 *   Row 0: no shift
 *   Row 1: left shift by 1 byte
 *   Row 2: left shift by 2 bytes
 *   Row 3: left shift by 3 bytes
 *
 * This implementation treats each row as a 32-bit word and applies byte-level
 * rotations (rotl8, rotl16, rotl24), which produces the same transformation as
 * the AES specification but with fewer operations than manual byte swapping.
 *
 * Parameters:
 *   state  Pointer to the AES 4×4 state matrix (modified in place)
 ============================================================================ */

static void shift_rows(aes_state_t *state)
{
	uint32_t r;

	/* Row 1: rotate left by 1 byte */
	memcpy(&r, (*state)[1], 4);
	r = rotl8(r);
	memcpy((*state)[1], &r, 4);

	/* Row 2: rotate left by 2 bytes */
	memcpy(&r, (*state)[2], 4);
	r = rotl16(r);
	memcpy((*state)[2], &r, 4);

	/* Row 3: rotate left by 3 bytes */
	memcpy(&r, (*state)[3], 4);
	r = rotl24(r);
	memcpy((*state)[3], &r, 4);
}

/* =============================================================================
 * inv_shift_rows()
 *
 * Perform the AES Inverse ShiftRows transformation.
 *
 * This is the inverse of the ShiftRows step used during encryption. Each row
 * of the AES state is cyclically rotated to the right:
 *
 *   Row 0: no shift
 *   Row 1: right shift by 1 byte
 *   Row 2: right shift by 2 bytes
 *   Row 3: right shift by 3 bytes
 *
 * The implementation uses 32-bit right rotations (rotr8, rotr16, rotr24)
 * to efficiently apply the inverse permutation.
 *
 * Parameters:
 *   state  Pointer to the AES 4×4 state matrix (modified in place)
 ============================================================================ */

static void inv_shift_rows(aes_state_t *state)
{
	uint32_t r;

	// Row 1: rotate right by 1 byte
	memcpy(&r, (*state)[1], 4);
	r = rotr8(r);
	memcpy((*state)[1], &r, 4);

	// Row 2: rotate right by 2 bytes
	memcpy(&r, (*state)[2], 4);
	r = rotr16(r);
	memcpy((*state)[2], &r, 4);

	// Row 3: rotate right by 3 bytes
	memcpy(&r, (*state)[3], 4);
	r = rotr24(r);
	memcpy((*state)[3], &r, 4);
}

/* =============================================================================
 * add_round_key()
 *
 * XORs the current AES state with a 16-byte round key.
 *
 * This function performs the AddRoundKey step defined in the AES specification.
 * Each byte of the 4×4 state matrix is XORed with the corresponding byte from
 * the round key. The key schedule provides one 128-bit round key per round.
 *
 * Operation:
 *   state[r][c] = state[r][c] XOR round_key[c*4 + r]
 *
 * Notes:
 *  - The state is stored in column-major order (AES standard).
 *  - round_key must point to a 16-byte key slice extracted from the expanded
 *    key.
 *  - This step is the only operation in AES that directly uses the key
 *    material.
 *
 * Parameters:
 *   state     Pointer to the AES state matrix (modified in place)
 *   round_key Pointer to the 16-byte round key for this round
 ============================================================================ */

static uint8_t galois_mul(uint8_t a, uint8_t b)
{
	uint8_t p = 0;
	for (int i = 0; i < BITS_PER_BYTE; i++)
	{
		if (b & 1)
		{
			p ^= a;
		}

		uint8_t hi_bit_set = a & GF_MSB_MASK;
		a <<= 1;

		if (hi_bit_set)
		{
			a ^= GF_REDUCING_POLYNOMIAL;
		}

		b >>= 1;
	}
	return p;
}

/* =============================================================================
 * add_round_key()
 *
 * XORs the current AES state with a 16-byte round key.
 *
 * This function performs the AddRoundKey step defined in the AES specification.
 * Each byte of the 4×4 state matrix is XORed with the corresponding byte from
 * the round key. The key schedule provides one 128-bit round key per round.
 *
 * Operation:
 *   state[r][c] = state[r][c] XOR round_key[c*4 + r]
 *
 * Notes:
 *  - The state is stored in column-major order (AES standard).
 *  - round_key must point to a 16-byte key slice extracted from the expanded
 *    key.
 *  - This step is the only operation in AES that directly uses the key
 *    material.
 *
 * Parameters:
 *   state     Pointer to the AES state matrix (modified in place)
 *   round_key Pointer to the 16-byte round key for this round
 ============================================================================ */

static void add_round_key(aes_state_t *state, const uint8_t *round_key)
{
	for (int c = 0; c < AES_STATE_DIM; c++)
		for (int r = 0; r < AES_STATE_DIM; r++)
			(*state)[r][c] ^= round_key[c * AES_STATE_DIM + r];
}

/* =============================================================================
 * AES MixColumns Matrix (Forward)
 *
 *  | 02  03  01  01 |
 *  | 01  02  03  01 |
 *  | 01  01  02  03 |
 *  | 03  01  01  02 |
 *
 * Each column of the state is multiplied by this matrix in GF(2^8)
 * using the irreducible polynomial: x^8 + x^4 + x^3 + x + 1 (0x11B).
 *
 *
 * AES Inverse MixColumns Matrix (Inverse)
 *
 *  | 0E  0B  0D  09 |
 *  | 09  0E  0B  0D |
 *  | 0D  09  0E  0B |
 *  | 0B  0D  09  0E |
 *
 * These constants correspond to the polynomials:
 *  09 = x^3 + 1
 *  0B = x^3 + x + 1
 *  0D = x^3 + x^2 + 1
 *  0E = x^3 + x^2 + x
 *
 * All multiplications are done in GF(2^8).
 ============================================================================ */

static void mix_columns(aes_state_t *state)
{
	uint8_t t[AES_STATE_DIM];
	for (int c = 0; c < AES_STATE_DIM; c++)
	{
		/* Load the column in 't' */
		for (int r = 0; r < AES_STATE_DIM; r++)
		{
			t[r] = (*state)[r][c];
		}

		(*state)[0][c] = galois_mul(t[0], 2) ^
						 galois_mul(t[1], 3) ^
						 t[2] ^
						 t[3];

		(*state)[1][c] = t[0] ^
						 galois_mul(t[1], 2) ^
						 galois_mul(t[2], 3) ^
						 t[3];

		(*state)[2][c] = t[0] ^
						 t[1] ^
						 galois_mul(t[2], 2) ^
						 galois_mul(t[3], 3);

		(*state)[3][c] = galois_mul(t[0], 3) ^
						 t[1] ^
						 t[2] ^
						 galois_mul(t[3], 2);
	}
}

static void inv_mix_columns(aes_state_t *state)
{
	uint8_t t[AES_STATE_DIM];
	for (int c = 0; c < AES_STATE_DIM; ++c)
	{
		for (int r = 0; r < AES_STATE_DIM; ++r)
			t[r] = (*state)[r][c];

		(*state)[0][c] = galois_mul(t[0], 14) ^
						 galois_mul(t[1], 11) ^
						 galois_mul(t[2], 13) ^
						 galois_mul(t[3], 9);

		(*state)[1][c] = galois_mul(t[0], 9) ^
						 galois_mul(t[1], 14) ^
						 galois_mul(t[2], 11) ^
						 galois_mul(t[3], 13);

		(*state)[2][c] = galois_mul(t[0], 13) ^
						 galois_mul(t[1], 9) ^
						 galois_mul(t[2], 14) ^
						 galois_mul(t[3], 11);

		(*state)[3][c] = galois_mul(t[0], 11) ^
						 galois_mul(t[1], 13) ^
						 galois_mul(t[2], 9) ^
						 galois_mul(t[3], 14);
	}
}

/* =============================================================================
 * cipher_encrypt_block()
 *
 * Perform AES block encryption on a 16-byte state.
 *
 * This function implements the AES encryption procedure defined in FIPS-197.
 * Given an initialized 4×4 state matrix and an expanded key schedule, it
 * applies the following sequence:
 *
 *   1. Initial AddRoundKey
 *
 *   2. For rounds 1 .. (Nr - 1):
 *        - SubBytes        : Non-linear byte substitution using the AES S-Box
 *        - ShiftRows       : Cyclic left shift of each state row
 *        - MixColumns      : Linear mixing transformation on each column
 *        - AddRoundKey     : XOR the state with the round key
 *
 *   3. Final Round (round Nr):
 *        - SubBytes
 *        - ShiftRows
 *        - AddRoundKey     (no MixColumns in the final round)
 *
 * Parameters:
 *   state        Pointer to the 4×4 AES state matrix (modified in place)
 *   expanded_key Pointer to the full expanded key schedule (AES-128/192/256)
 *   num_rounds   Number of AES rounds (10 for AES-128, 12 for AES-192,
 * 										14 for AES-256)
 *
 * The function encrypts exactly one 128-bit block.
 ============================================================================ */

static void cipher_encrypt_block(aes_state_t *state,
								 const uint8_t *expanded_key,
								 uint16_t num_rounds)
{
	/* Start by adding the round keys first */
	add_round_key(state, expanded_key);

	/* Cycle encryp*/
	for (uint16_t round = 1; round < num_rounds; round++)
	{
		/* Substitute the bytes */
		substitute_bytes(state);
		/* Shift Rows */
		shift_rows(state);
		/* Mix Columns */
		mix_columns(state);
		/* Add Round Key */
		add_round_key(state, expanded_key + (AES_BLOCK_SIZE * round));
	}

	/* Final round, mix_column step is not part of it */
	substitute_bytes(state);
	shift_rows(state);
	add_round_key(state, expanded_key + (AES_BLOCK_SIZE * num_rounds));
}