#include "aes.h"
#include <cstdio>
#include <cstring>
#include "aes_test.h"

/* AES-128 Known Answer Test (user-provided vector) */
const uint8_t key128[16] = {
    0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
    0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};

const uint8_t key192[24] = {
    0x8e, 0x73, 0xb0, 0xf7, 0xda, 0x0e, 0x64, 0x52,
    0xc8, 0x10, 0xf3, 0x2b, 0x80, 0x90, 0x79, 0xe5,
    0x62, 0xf8, 0xea, 0xd2, 0x52, 0x2c, 0x6b, 0x7b};

const uint8_t key256[32] = {
    0x60, 0x3d, 0xeb, 0x10, 0x15, 0xca, 0x71, 0xbe,
    0x2b, 0x73, 0xae, 0xf0, 0x85, 0x7d, 0x77, 0x81,
    0x1f, 0x35, 0x2c, 0x07, 0x3b, 0x61, 0x08, 0xd7,
    0x2d, 0x98, 0x10, 0xa3, 0x09, 0x14, 0xdf, 0xf4};

const uint8_t plaintext[16] = {
    0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d,
    0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34};

const uint8_t expected_cipher[16] = {
    0x39, 0x25, 0x84, 0x1d, 0x02, 0xdc, 0x09, 0xfb,
    0xdc, 0x11, 0x85, 0x97, 0x19, 0x6a, 0x0b, 0x32};

static void print_hex(const uint8_t *buf, size_t len)
{
    for (size_t i = 0; i < len; ++i)
        printf("%02x", buf[i]);
}

static int aes_test_ecb_key128_singleblock()
{
    /* Storage for Ciphertext */
    uint8_t ciphertext[16];
    /* Storage for decrypted text */
    uint8_t decrypted[16];
    /* Error Storage */
    aes_error_te err;
    bool overall_ok = true;

    /* Perform AES-128 ECB encryption */
    err = aes_encrypt_ecb(plaintext, 16, ciphertext, key128,
                          AES_KEY_SIZE_128);

    if (err != AES_SUCCESS)
    {
        printf("aes_encrypt_block returned error %d\n", (int)err);
        return 2;
    }

    /* Verify encryption matches expected cipher */
    bool enc_ok = (memcmp(ciphertext, expected_cipher, 16) == 0);

    /* Verify decryption also returns the original plaintext */
    err = aes_decrypt_ecb(ciphertext, 16, decrypted, key128,
                          AES_KEY_SIZE_128);

    if (err != AES_SUCCESS)
    {
        printf("aes_decrypt_block returned error %d\n", (int)err);
        return 3;
    }

    /* Check decrypted matches original plaintext */
    bool dec_ok = (memcmp(decrypted, plaintext, 16) == 0);

    if (enc_ok && dec_ok)
    {
        printf("AES-128 single-block round-trip PASSED\n");
    }
    else
    {
        overall_ok = false;
        printf("AES-128 single-block test FAILED\n");
        if (!enc_ok)
        {
            printf("Encryption mismatch:\n");
            printf("Expected:   ");
            print_hex(expected_cipher, 16);
            printf("\n");
            printf("Got:        ");
            print_hex(ciphertext, 16);
            printf("\n");
        }
        if (!dec_ok)
        {
            printf("Decryption mismatch:\n");
            printf("Expected:   ");
            print_hex(plaintext, 16);
            printf("\n");
            printf("Got:        ");
            print_hex(decrypted, 16);
            printf("\n");
        }
    }
    return 0;
}

static int aes_test_ecb_key128_multiblock()
{
    const size_t blocks = 4;
    const size_t len = blocks * AES_BLOCK_SIZE;
    uint8_t multi_plain[64];
    uint8_t multi_cipher[64];
    uint8_t multi_decrypted[64];
    bool overall_ok = true;

    /* Error Storage */
    aes_error_te err;

    for (size_t i = 0; i < len; ++i)
    {
        multi_plain[i] = (uint8_t)(i & 0xFF);
    }

    err = aes_encrypt_ecb(multi_plain, len, multi_cipher, key128,
                          AES_KEY_SIZE_128);

    if (err != AES_SUCCESS)
    {
        printf("AES-128 multi-block encrypt error %d\n", (int)err);
        return 8;
    }
    err = aes_decrypt_ecb(multi_cipher, len, multi_decrypted, key128,
                          AES_KEY_SIZE_128);

    if (err != AES_SUCCESS)
    {
        printf("AES-128 multi-block decrypt error %d\n", (int)err);
        return 9;
    }
    if (memcmp(multi_plain, multi_decrypted, len) == 0)
        printf("AES-128 multi-block round-trip PASSED\n");
    else
    {
        overall_ok = false;
        printf("AES-128 multi-block round-trip FAILED\n");
    }

    return overall_ok ? 0 : 1;
}

static int aes_test_ecb_key192_multiblock()
{
    const size_t blocks = 4;
    const size_t len = blocks * AES_BLOCK_SIZE;
    uint8_t multi_plain[64];
    uint8_t multi_cipher[64];
    uint8_t multi_decrypted[64];
    bool overall_ok = true;

    /* Error Storage */
    aes_error_te err;

    for (size_t i = 0; i < len; ++i)
    {
        multi_plain[i] = (uint8_t)(i & 0xFF);
    }

    err = aes_encrypt_ecb(multi_plain, len, multi_cipher, key192,
                          AES_KEY_SIZE_192);

    if (err != AES_SUCCESS)
    {
        printf("AES-192 multi-block encrypt error %d\n", (int)err);
        return 8;
    }
    err = aes_decrypt_ecb(multi_cipher, len, multi_decrypted, key192,
                          AES_KEY_SIZE_192);

    if (err != AES_SUCCESS)
    {
        printf("AES-192 multi-block decrypt error %d\n", (int)err);
        return 9;
    }
    if (memcmp(multi_plain, multi_decrypted, len) == 0)
        printf("AES-192 multi-block round-trip PASSED\n");
    else
    {
        overall_ok = false;
        printf("AES-192 multi-block round-trip FAILED\n");
    }

    return overall_ok ? 0 : 1;
}

static int aes_test_ecb_key256_multiblock()
{
    const size_t blocks = 4;
    const size_t len = blocks * AES_BLOCK_SIZE;
    uint8_t multi_plain[64];
    uint8_t multi_cipher[64];
    uint8_t multi_decrypted[64];
    bool overall_ok = true;

    /* Error Storage */
    aes_error_te err;

    for (size_t i = 0; i < len; ++i)
    {
        multi_plain[i] = (uint8_t)(i & 0xFF);
    }

    err = aes_encrypt_ecb(multi_plain, len, multi_cipher, key256,
                          AES_KEY_SIZE_256);

    if (err != AES_SUCCESS)
    {
        printf("AES-256 multi-block encrypt error %d\n", (int)err);
        return 8;
    }
    err = aes_decrypt_ecb(multi_cipher, len, multi_decrypted, key256,
                          AES_KEY_SIZE_256);

    if (err != AES_SUCCESS)
    {
        printf("AES-256 multi-block decrypt error %d\n", (int)err);
        return 9;
    }
    if (memcmp(multi_plain, multi_decrypted, len) == 0)
        printf("AES-256 multi-block round-trip PASSED\n");
    else
    {
        overall_ok = false;
        printf("AES-256 multi-block round-trip FAILED\n");
    }

    return overall_ok ? 0 : 1;
}

int aes_test_ecb()
{
    int status = 0;

    status += aes_test_ecb_key128_singleblock();
    status += aes_test_ecb_key128_multiblock();
    status += aes_test_ecb_key192_multiblock();
    status += aes_test_ecb_key256_multiblock();

    return status;
}
