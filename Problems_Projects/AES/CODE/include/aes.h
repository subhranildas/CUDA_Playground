

#ifndef AES_H
#define AES_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/* =============================================================================
 * General Use Constants
 * ========================================================================== */

#define AES_BLOCK_SIZE 16
#define AES_STATE_DIM 4

/* 4*(Nr+1) words, for AES-256 Nr=14 => 4*(14+1)=60 words -> 240 bytes as uint32_t? */
/* We'll store words as uint32_t, so store up to 4*(14+1)=60 words. */

#define MAX_ROUND_KEYS 240

typedef enum
{
	AES_KEY_SIZE_128 = 16, /**< For 128-bit keys (16 bytes). */
	AES_KEY_SIZE_192 = 24, /**< For 192-bit keys (24 bytes). */
	AES_KEY_SIZE_256 = 32  /**< For 256-bit keys (32 bytes). */
} aes_key_size_te;

/* A type definition for the 4x4 byte AES state matrix. */
typedef uint8_t aes_state_t[AES_STATE_DIM][AES_STATE_DIM];

#endif