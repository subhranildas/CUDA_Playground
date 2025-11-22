

#ifndef AES_H
#define AES_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/* =============================================================================
 * Macros
 * ========================================================================== */

#define AES_BLOCK_SIZE 16
#define AES_STATE_DIM 4

/* 4*(Nr+1) words, for AES-256 Nr=14 => 4*(14+1)=60 words -> 240 bytes as uint32_t? */
/* We'll store words as uint32_t, so store up to 4*(14+1)=60 words. */

#define MAX_ROUND_KEYS 240

/* Number of rounds for different key sizes */
#define AES_ROUNDS_128 10
#define AES_ROUNDS_192 12
#define AES_ROUNDS_256 14
/* =============================================================================
 * Type defines
 * ========================================================================== */

typedef enum
{
	AES_KEY_SIZE_128 = 16, /**< For 128-bit keys (16 bytes). */
	AES_KEY_SIZE_192 = 24, /**< For 192-bit keys (24 bytes). */
	AES_KEY_SIZE_256 = 32  /**< For 256-bit keys (32 bytes). */
} aes_key_size_te;

typedef enum
{
	AES_SUCCESS = 0,					/**< The operation completed successfully. */
	AES_ERROR_UNSUPPORTED_KEY_SIZE,		/**< The provided key size is not supported. */
	AES_ERROR_MEMORY_ALLOCATION_FAILED, /**< A memory allocation call failed. */
	AES_ERROR_CUDA_FAILURE,				/**< A CUDA runtime call or kernel failed. */
	AES_ERROR_INVALID_INPUT_LENGTH,		/** Input length is not in multiple of 16 bytes */
} aes_error_te;

/* A type definition for the 4x4 byte AES state matrix. */
typedef uint8_t aes_state_t[AES_STATE_DIM][AES_STATE_DIM];

/* =============================================================================
 * Public APIs
 * ========================================================================== */

#ifdef __cplusplus
extern "C"
{
#endif

	void memory_write_zeros(void *ptr, unsigned long long n);

	void aes_expand_key(uint8_t *expanded_key, const uint8_t *key,
						aes_key_size_te key_size, size_t expanded_key_size);

	aes_error_te aes_encrypt_block(const uint8_t *plaintext, uint8_t *ciphertext,
								   const uint8_t *key, aes_key_size_te key_size);

	aes_error_te aes_decrypt_block(const uint8_t *ciphertext, uint8_t *plaintext,
								   const uint8_t *key, aes_key_size_te key_size);

	aes_error_te aes_encrypt_ecb(const uint8_t *plaintext, size_t length,
								 uint8_t *ciphertext,
								 const uint8_t *key, aes_key_size_te key_size);

	aes_error_te aes_decrypt_ecb(const uint8_t *ciphertext, size_t length,
								 uint8_t *plaintext,
								 const uint8_t *key, aes_key_size_te key_size);

	aes_error_te aes_encrypt_cbc(const uint8_t *plaintext, size_t length,
								 uint8_t *ciphertext,
								 const uint8_t *key, aes_key_size_te key_size,
								 const uint8_t iv[AES_BLOCK_SIZE]);

	aes_error_te aes_decrypt_cbc(const uint8_t *ciphertext, size_t length,
								 uint8_t *plaintext,
								 const uint8_t *key, aes_key_size_te key_size,
								 const uint8_t iv[AES_BLOCK_SIZE]);

	aes_error_te aes_crypt_ctr(const uint8_t *input, size_t length,
							   uint8_t *output,
							   const uint8_t *key, aes_key_size_te key_size,
							   const uint8_t nonce_counter[AES_BLOCK_SIZE]);

	aes_error_te aes_encrypt_ecb_cuda(const uint8_t *plaintext, size_t length,
									  uint8_t *ciphertext,
									  const uint8_t *key,
									  aes_key_size_te key_size);

    aes_error_te aes_decrypt_ecb_cuda(const uint8_t *ciphertext, size_t length,
                                       uint8_t *plaintext,
                                       const uint8_t *key, aes_key_size_te key_size);

#ifdef __cplusplus
}
#endif

#endif