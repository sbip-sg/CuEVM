#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <CuCrypto/ripemd160.cuh>

__global__ void ripemd160_kernel(const uint8_t* d_data, size_t length, uint8_t* d_hash) {
    CuCrypto::ripemd160::ripemd160(d_data, length, d_hash);
}

TEST(Ripemd160Test, NullAssertionsGPU) {
    uint8_t data[32] = {0}; // Data array filled with zeroes
    uint8_t hash[20];

    // Allocate device memory
    uint8_t* d_data;
    uint8_t* d_hash;
    cudaMalloc(&d_data, sizeof(data));
    cudaMalloc(&d_hash, sizeof(hash));

    // Copy data to device
    cudaMemcpy(d_data, data, sizeof(data), cudaMemcpyHostToDevice);

    // Launch kernel
    ripemd160_kernel<<<1, 1>>>(d_data, 0, d_hash);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(hash, d_hash, sizeof(hash), cudaMemcpyDeviceToHost);

    // Expected hash value for an array of 32 zeroes
    uint8_t expected_hash[20] = {
        0x9c, 0x11, 0x85, 0xa5, 0xc5, 0xe9, 0xfc, 0x54,
        0x61, 0x28, 0x08, 0x97, 0x7e, 0xe8, 0xf5, 0x48,
        0xb2, 0x25, 0x8d, 0x31
    };

    for (int i = 0; i < 20; i++) {
        EXPECT_EQ(hash[i], expected_hash[i]);
    }

    // Free device memory
    cudaFree(d_data);
    cudaFree(d_hash);
}

TEST(Ripemd160Test, BasicAssertionsGPU) {
    uint8_t data[32] = {0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20};
    uint8_t hash[20];

    // Allocate device memory
    uint8_t* d_data;
    uint8_t* d_hash;
    cudaMalloc(&d_data, sizeof(data));
    cudaMalloc(&d_hash, sizeof(hash));

    // Copy data to device
    cudaMemcpy(d_data, data, sizeof(data), cudaMemcpyHostToDevice);

    // Launch kernel
    ripemd160_kernel<<<1, 1>>>(d_data, 32, d_hash);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(hash, d_hash, sizeof(hash), cudaMemcpyDeviceToHost);

    // Expected hash value (example, replace with actual expected value)
    uint8_t expected_hash[20] = {
        0xaf, 0x6c, 0xc0, 0x66, 0xe3, 0x74, 0x5a, 0x4c,
        0x73, 0xc5, 0xa7, 0xf6, 0xe0, 0x1b, 0x85, 0xe7,
        0x26, 0xd6, 0x0b, 0xf3
    };

    for (int i = 0; i < 20; i++) {
        EXPECT_EQ(hash[i], expected_hash[i]);
    }

    // Free device memory
    cudaFree(d_data);
    cudaFree(d_hash);
}


TEST(Ripemd160Test, NullAssertions) {
    uint8_t data[32] = {0}; // Data array filled with zeroes
    uint8_t hash[20];
    CuCrypto::ripemd160::ripemd160(data, 0, hash);

    // Expected hash value for an array of 32 zeroes
    uint8_t expected_hash[20] = {
        0x9c, 0x11, 0x85, 0xa5, 0xc5, 0xe9, 0xfc, 0x54,
        0x61, 0x28, 0x08, 0x97, 0x7e, 0xe8, 0xf5, 0x48,
        0xb2, 0x25, 0x8d, 0x31
    };

    for (int i = 0; i < 20; i++) {
        EXPECT_EQ(hash[i], expected_hash[i]);
    }
}

TEST(Ripemd160Test, BasicAssertions) {
    uint8_t data[32] = {0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20};
    uint8_t hash[20];
    CuCrypto::ripemd160::ripemd160(data, 32, hash);

    // Expected hash value (example, replace with actual expected value)
    uint8_t expected_hash[20] = {
        0xaf, 0x6c, 0xc0, 0x66, 0xe3, 0x74, 0x5a, 0x4c,
        0x73, 0xc5, 0xa7, 0xf6, 0xe0, 0x1b, 0x85, 0xe7,
        0x26, 0xd6, 0x0b, 0xf3
    };

    for (int i = 0; i < 20; i++) {
        EXPECT_EQ(hash[i], expected_hash[i]);
    }
}