#ifndef _KERNELS_H_
#define _KERNELS_H_

#include <stdint.h>

__global__ void aes_ecb_encrypt_kernel(const uint8_t *d_input, uint8_t *d_output,
									   const uint8_t *d_expanded_key, uint16_t num_rounds,
									   size_t num_blocks);

#endif