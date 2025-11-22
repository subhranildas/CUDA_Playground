#include <cstdio>
#include "aes_test.h"

int main(int argc, char **argv)
{
    (void)argc; (void)argv;
    int res = aes_test_ecb();
    if (res == 0)
    {
        printf("All AES ECB tests PASSED\n");
    }
    else
    {
        printf("Some AES ECB tests FAILED (code=%d)\n", res);
    }
    return res;
}
