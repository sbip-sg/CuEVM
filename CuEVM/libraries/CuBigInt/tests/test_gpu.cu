#include <gtest/gtest.h>
#include <CuBigInt/bigint.cuh>
#include <cuda_runtime.h>
#include <curand_kernel.h>

class CudaBigIntTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaDeviceReset();
        cudaMalloc(&d_result, sizeof(int32_t));
    }

    void TearDown() override {
        cudaFree(d_result);
        cudaDeviceReset();
    }

    int32_t *d_result;
};

__global__ void RandomizedOperations_kernel(int32_t *result) {
    curandState localState;
    curand_init(1234, 0, 0, &localState);
    bigint a[1], b[1], c[1], d[1], e[20];
    bigint_init(a);
    bigint_init(b);
    bigint_init(c);
    bigint_init(d);
    for (int i = 0; i < 20; i++) bigint_init(e + i);
    *result = 0;
    for (int i = 0; i < 12345; i++) {
        int x = curand(&localState) % 12345;
        int y = curand(&localState) % 12345;
        int shift = curand(&localState) % 1234;
        if (curand(&localState) & 1) x = -x;
        if (curand(&localState) & 1) y = -y;

        bigint_from_int(a, x);
        bigint_from_int(b, y);
        bigint_from_int(e + 0, x + y);
        bigint_from_int(e + 1, x - y);
        bigint_from_int(e + 2, x * y);

        if (y != 0) {
            bigint_from_int(e + 3, x / y);
            bigint_from_int(e + 4, x % y);
        }

        bigint_from_int(e + 5, x);
        bigint_from_int(e + 6, bigint_int_gcd(x, y));

        bigint_cpy(c, a);
        bigint_shift_left(a, a, shift);
        bigint_shift_right(a, a, shift);

        *result |= bigint_cmp(a, c);

        bigint_add(e + 10, a, b);
        bigint_sub(e + 11, a, b);
        bigint_mul(e + 12, a, b);
        bigint_div(e + 13, a, b);
        bigint_mod(e + 14, a, b);
        bigint_from_int(e + 15, x);
        bigint_gcd(e + 16, a, b);

        for (int j = 0; j < 7; j++) {
            if (y == 0 && (j == 3 || j == 4)) continue;
            if (bigint_cmp(e + j, e + j + 10) != 0) {
                printf("i %i, j %i failed for bigints %i, %i\n", i, j, x, y);
            }
            *result |= bigint_cmp(e + j, e + j + 10);
        }
    }
    bigint_free(a);
    bigint_free(b);
    bigint_free(c);
    bigint_free(d);
    for (int i = 0; i < 20; i++) bigint_free(e + i);
}


TEST_F(CudaBigIntTest, RandomizedOperations) {
    int32_t result;
    RandomizedOperations_kernel<<<1, 1>>>(d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&result, d_result, sizeof(int32_t), cudaMemcpyDeviceToHost);
    EXPECT_EQ(result, 0);
}