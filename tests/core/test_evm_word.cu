
#include <gtest/gtest.h>

#include <CuEVM/core/evm_word.cuh>
#include <CuEVM/utils/error_codes.cuh>
#include <CuEVM/utils/evm_defines.cuh>
#include <CuEVM/utils/evm_utils.cuh>

// Test default constructor and from_uint32_t
TEST(EvmWordTest, DefaultConstructorAndFromUint32) {
    CuEVM::evm_word_t word1;
    word1.from_uint32_t(0x12345678);
    EXPECT_EQ(word1._limbs[0], 0x12345678);
    for (int i = 1; i < CuEVM::cgbn_limbs; ++i) {
        EXPECT_EQ(word1._limbs[i], 0);
    }
}

// Test copy constructor
TEST(EvmWordTest, CopyConstructor) {
    CuEVM::evm_word_t word1;
    word1.from_uint32_t(0x12345678);
    CuEVM::evm_word_t word2(word1);
    EXPECT_EQ(word2, word1);
}

// Test assignment operator
TEST(EvmWordTest, AssignmentOperator) {
    CuEVM::evm_word_t word1;
    word1.from_uint32_t(0x12345678);
    CuEVM::evm_word_t word3;
    word3 = word1;
    EXPECT_EQ(word3, word1);
}

// Test equality operator with evm_word_t
TEST(EvmWordTest, EqualityOperatorWithEvmWord) {
    CuEVM::evm_word_t word1;
    word1.from_uint32_t(0x12345678);
    CuEVM::evm_word_t word2(word1);
    CuEVM::evm_word_t word3;
    word3 = word1;

    EXPECT_EQ(word1, word2);
    EXPECT_EQ(word1, word3);
}

// Test equality operator with uint32_t
TEST(EvmWordTest, EqualityOperatorWithUint32) {
    CuEVM::evm_word_t word1;
    word1.from_uint32_t(0x12345678);

    EXPECT_EQ(word1, 0x12345678);
    EXPECT_EQ(word1 == 0x87654321, 0);
}

// Test from_uint64_t
TEST(EvmWordTest, FromUint64) {
    CuEVM::evm_word_t word4;
    word4.from_uint64_t(0x123456789ABCDEF0);
    EXPECT_EQ(word4._limbs[0], 0x9ABCDEF0);
    EXPECT_EQ(word4._limbs[1], 0x12345678);
    for (int i = 2; i < CuEVM::cgbn_limbs; ++i) {
        EXPECT_EQ(word4._limbs[i], 0);
    }
}

// Test from_hex
TEST(EvmWordTest, FromHex) {
    CuEVM::evm_word_t word4;
    word4.from_uint64_t(0x123456789ABCDEF0);

    CuEVM::evm_word_t word5;
    word5.from_hex("0x123456789ABCDEF0");
    EXPECT_EQ(word5, word4);
}

// Test to_hex
TEST(EvmWordTest, ToHex) {
    CuEVM::evm_word_t word4;
    word4.from_uint64_t(0x123456789ABCDEF0);

    char hex_string[2 + CuEVM::cgbn_limbs * 8 + 1];
    word4.to_hex(hex_string, 0, 2);
    EXPECT_STREQ(hex_string, "0x123456789abcdef0");
}

TEST(EvmWordTest, ToByteArray) {
    CuEVM::evm_word_t word4;
    word4.from_uint64_t(0x123456789ABCDEF0);

    CuEVM::byte_array_t byte_array;
    word4.to_byte_array_t(byte_array);
    EXPECT_EQ(byte_array.data[CuEVM::word_size - 1], 0xF0);
    EXPECT_EQ(byte_array.data[CuEVM::word_size - 2], 0xDE);
    EXPECT_EQ(byte_array.data[CuEVM::word_size - 3], 0xBC);
    EXPECT_EQ(byte_array.data[CuEVM::word_size - 4], 0x9A);
    EXPECT_EQ(byte_array.data[CuEVM::word_size - 5], 0x78);
    EXPECT_EQ(byte_array.data[CuEVM::word_size - 6], 0x56);
    EXPECT_EQ(byte_array.data[CuEVM::word_size - 7], 0x34);
    EXPECT_EQ(byte_array.data[CuEVM::word_size - 8], 0x12);
    EXPECT_EQ(byte_array.data[CuEVM::word_size - 9], 0x00);
}

// // Test to_bit_array_t
TEST(EvmWordTest, ToBitArray) {
    CuEVM::evm_word_t word4;
    word4.from_uint64_t(0x123456789ABCDEF0);

    CuEVM::byte_array_t bit_array;
    word4.to_bit_array_t(bit_array);
    for (int i = 0; i < 32; ++i) {
        EXPECT_EQ(bit_array.data[i], ((0x9ABCDEF0 >> i) & 0x01));
    }
    for (int i = 0; i < 32; ++i) {
        EXPECT_EQ(bit_array.data[32 + i], ((0x12345678 >> i) & 0x01));
    }
    for (int i = 64; i < CuEVM::word_bits; ++i) {
        EXPECT_EQ(bit_array.data[i], 0);
    }
}

// Test from_byte_array_t
TEST(EvmWordTest, FromByteArray) {
    CuEVM::byte_array_t byte_array(CuEVM::word_size);
    byte_array.data[0] = 0x12;
    byte_array.data[1] = 0x34;
    byte_array.data[2] = 0x56;
    byte_array.data[3] = 0x78;
    byte_array.data[4] = 0x9A;
    byte_array.data[5] = 0xBC;
    byte_array.data[6] = 0xDE;
    byte_array.data[7] = 0xF0;

    CuEVM::evm_word_t word;
    int32_t result = word.from_byte_array_t(byte_array);
    EXPECT_EQ(result, ERROR_SUCCESS);
    EXPECT_EQ(word._limbs[0], 0x78563412);
    EXPECT_EQ(word._limbs[1], 0xF0DEBC9A);
    for (int i = 2; i < CuEVM::cgbn_limbs; ++i) {
        EXPECT_EQ(word._limbs[i], 0);
    }
}

// Test for the big-endian version of to_bit_array_t
TEST(EvmWordTest, ToBitArrayBigEndian) {
    CuEVM::evm_word_t word;
    word.from_uint64_t(0x123456789ABCDEF0);

    CuEVM::byte_array_t bit_array;
    word.to_bit_array_t(bit_array, BIG_ENDIAN);

    uint8_t expected_bits[64] = {
        0, 0, 0, 0, 1, 1, 1, 1,  // 0xF0
        0, 1, 1, 1, 1, 0, 1, 1,  // 0xDE
        0, 0, 1, 1, 1, 1, 0, 1,  // 0xBC
        0, 1, 0, 1, 1, 0, 0, 1,  // 0x9A
        0, 0, 0, 1, 1, 1, 1, 0,  // 0x78
        0, 1, 1, 0, 1, 0, 1, 0,  // 0x56
        0, 0, 1, 0, 1, 1, 0, 0,  // 0x34
        0, 1, 0, 0, 1, 0, 0, 0   // 0x12
    };

    for (int i = 0; i < 64; ++i) {
        EXPECT_EQ(bit_array.data[CuEVM::word_bits - 1 - i], expected_bits[i]);
    }
}

TEST(EvmWordTest, FromSizeT) {
    CuEVM::evm_word_t word;
    size_t value = 0x12345678;
    word.from_size_t(value);
    EXPECT_EQ(word._limbs[0], 0x12345678);
    for (int i = 1; i < CuEVM::cgbn_limbs; ++i) {
        EXPECT_EQ(word._limbs[i], 0);
    }
}

TEST(EvmWordTest, Print) {
    CuEVM::evm_word_t word;
    word.from_uint32_t(0x12345678);
    testing::internal::CaptureStdout();
    word.print();
    std::string output = testing::internal::GetCapturedStdout();
    ASSERT_EQ(output,
              "00000000 00000000 00000000 00000000 00000000 00000000 00000000 "
              "12345678 \n");
}

__global__ void test_evm_word_t_kernel(uint32_t count, uint32_t *result) {
    int32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / CuEVM::cgbn_tpi;
    if (instance >= count) return;
    result[instance] = ERROR_SUCCESS;
    // Test default constructor and from_uint32_t
    CuEVM::evm_word_t word1;
    word1.from_uint32_t(0x12345678);
    result[instance] |= (word1._limbs[0] == 0x12345678) ? ERROR_SUCCESS : __LINE__;
    for (int i = 1; i < CuEVM::cgbn_limbs; ++i) {
        result[instance] |= (word1._limbs[i] == 0) ? ERROR_SUCCESS : __LINE__;
    }

    // Test copy constructor
    CuEVM::evm_word_t word2(word1);
    result[instance] |= (word2 == word1) ? ERROR_SUCCESS : __LINE__;

    // Test assignment operator
    CuEVM::evm_word_t word3;
    word3 = word1;
    result[instance] |= (word3 == word1) ? ERROR_SUCCESS : __LINE__;

    // Test equality operator with evm_word_t
    result[instance] |= (word1 == word2) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (word1 == word3) ? ERROR_SUCCESS : __LINE__;

    // Test equality operator with uint32_t
    result[instance] |= (word1 == 0x12345678) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (word1 == 0x87654321) ? __LINE__ : ERROR_SUCCESS;

    // Test from_uint64_t
    CuEVM::evm_word_t word4;
    word4.from_uint64_t(0x123456789ABCDEF0);
    result[instance] |= (word4._limbs[0] == 0x9ABCDEF0) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (word4._limbs[1] == 0x12345678) ? ERROR_SUCCESS : __LINE__;
    for (int i = 2; i < CuEVM::cgbn_limbs; ++i) {
        result[instance] |= (word4._limbs[i] == 0) ? ERROR_SUCCESS : __LINE__;
    }

    // Test to_byte_array_t
    CuEVM::byte_array_t byte_array;
    word4.to_byte_array_t(byte_array);
    result[instance] |= (byte_array.data[CuEVM::word_size - 1] == 0xF0) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (byte_array.data[CuEVM::word_size - 2] == 0xDE) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (byte_array.data[CuEVM::word_size - 3] == 0xBC) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (byte_array.data[CuEVM::word_size - 4] == 0x9A) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (byte_array.data[CuEVM::word_size - 5] == 0x78) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (byte_array.data[CuEVM::word_size - 6] == 0x56) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (byte_array.data[CuEVM::word_size - 7] == 0x34) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (byte_array.data[CuEVM::word_size - 8] == 0x12) ? ERROR_SUCCESS : __LINE__;

    // Test to_bit_array_t
    CuEVM::byte_array_t bit_array;
    word4.to_bit_array_t(bit_array);
    for (int i = 0; i < 32; ++i) {
        result[instance] |= (bit_array.data[i] == ((0x9ABCDEF0 >> i) & 0x01)) ? ERROR_SUCCESS : __LINE__;
    }
    for (int i = 0; i < 32; ++i) {
        result[instance] |= (bit_array.data[32 + i] == ((0x12345678 >> i) & 0x01)) ? ERROR_SUCCESS : __LINE__;
    }
    for (int i = 64; i < CuEVM::word_bits; ++i) {
        result[instance] |= (bit_array.data[i] == 0) ? ERROR_SUCCESS : __LINE__;
    }

    // Test from_byte_array_t
    CuEVM::byte_array_t byte_array2(CuEVM::word_size);
    byte_array2.data[0] = 0x12;
    byte_array2.data[1] = 0x34;
    byte_array2.data[2] = 0x56;
    byte_array2.data[3] = 0x78;
    byte_array2.data[4] = 0x9A;
    byte_array2.data[5] = 0xBC;
    byte_array2.data[6] = 0xDE;
    byte_array2.data[7] = 0xF0;

    CuEVM::evm_word_t word5;
    result[instance] |= (word5.from_byte_array_t(byte_array2) == ERROR_SUCCESS) ? ERROR_SUCCESS : __LINE__;

    result[instance] |= (word5._limbs[0] == 0x78563412) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (word5._limbs[1] == 0xF0DEBC9A) ? ERROR_SUCCESS : __LINE__;
    for (int i = 2; i < CuEVM::cgbn_limbs; ++i) {
        result[instance] |= (word5._limbs[i] == 0) ? ERROR_SUCCESS : __LINE__;
    }

    // Test for the big-endian version of to_bit_array_t
    CuEVM::byte_array_t bit_array2;
    CuEVM::evm_word_t word6;
    word6.from_uint64_t(0x123456789ABCDEF0);
    word6.to_bit_array_t(bit_array2, BIG_ENDIAN);

    uint8_t expected_bits[64] = {
        0, 0, 0, 0, 1, 1, 1, 1,  // 0xF0
        0, 1, 1, 1, 1, 0, 1, 1,  // 0xDE
        0, 0, 1, 1, 1, 1, 0, 1,  // 0xBC
        0, 1, 0, 1, 1, 0, 0, 1,  // 0x9A
        0, 0, 0, 1, 1, 1, 1, 0,  // 0x78
        0, 1, 1, 0, 1, 0, 1, 0,  // 0x56
        0, 0, 1, 0, 1, 1, 0, 0,  // 0x34
        0, 1, 0, 0, 1, 0, 0, 0   // 0x12
    };

    for (int i = 0; i < 64; ++i) {
        result[instance] |= (bit_array2.data[CuEVM::word_bits - 1 - i] == expected_bits[i]) ? ERROR_SUCCESS : __LINE__;
    }

    // Test from_size_t
    CuEVM::evm_word_t word7;
    size_t value = 0x12345678;
    word7.from_size_t(value);
    result[instance] |= (word7._limbs[0] == 0x12345678) ? ERROR_SUCCESS : __LINE__;
    for (int i = 1; i < CuEVM::cgbn_limbs; ++i) {
        result[instance] |= (word7._limbs[i] == 0) ? ERROR_SUCCESS : __LINE__;
    }
}

#ifdef GPU
TEST(EvmWordTest, KernelTests) {
    CUDA_CHECK(cudaDeviceReset());
    uint32_t *d_result;
    cudaMalloc(&d_result, 2 * sizeof(uint32_t));
    test_evm_word_t_kernel<<<2, CuEVM::cgbn_tpi>>>(2, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t *h_result;
    h_result = (uint32_t *)malloc(2 * sizeof(uint32_t));
    CUDA_CHECK(cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 2; i++) {
        EXPECT_EQ(h_result[i], ERROR_SUCCESS);
    }
    free(h_result);
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaDeviceReset());
}
#endif

__global__ void test_evm_word_t_kernel_assign(uint32_t count, uint32_t *result) {
    int32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / CuEVM::cgbn_tpi;
    if (instance >= count) return;
    result[instance] = ERROR_SUCCESS;

    // Test from_size_t
    CuEVM::evm_word_t word7;
    size_t value = 0x12345678;
    word7.from_size_t(value);
    result[instance] |= (word7._limbs[0] == 0x12345678) ? ERROR_SUCCESS : __LINE__;
    for (int i = 1; i < CuEVM::cgbn_limbs; ++i) {
        result[instance] |= (word7._limbs[i] == 0) ? ERROR_SUCCESS : __LINE__;
    }

    // Test constructor with uint32_t
    CuEVM::evm_word_t word8(0x87654321);
    result[instance] |= (word8._limbs[0] == 0x87654321) ? ERROR_SUCCESS : __LINE__;
    for (int i = 1; i < CuEVM::cgbn_limbs; ++i) {
        result[instance] |= (word8._limbs[i] == 0) ? ERROR_SUCCESS : __LINE__;
    }
    // Test assignment operator with uint32_t
    CuEVM::evm_word_t word9;
    word9 = 0xAABBCCDD;
    result[instance] |= (word9._limbs[0] == 0xAABBCCDD) ? ERROR_SUCCESS : __LINE__;
    for (int i = 1; i < CuEVM::cgbn_limbs; ++i) {
        result[instance] |= (word9._limbs[i] == 0) ? ERROR_SUCCESS : __LINE__;
    }
}

#ifdef GPU
TEST(EvmWordTest, KernelTestsAssign) {
    CUDA_CHECK(cudaDeviceReset());
    uint32_t *d_result;
    cudaMalloc(&d_result, 2 * sizeof(uint32_t));
    test_evm_word_t_kernel_assign<<<2, CuEVM::cgbn_tpi>>>(2, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t *h_result;
    h_result = (uint32_t *)malloc(2 * sizeof(uint32_t));
    CUDA_CHECK(cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 2; i++) {
        EXPECT_EQ(h_result[i], ERROR_SUCCESS);
    }
    free(h_result);
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaDeviceReset());
}
#endif

TEST(EvmWordTest, CPUAssign) {
    // Test from_size_t
    CuEVM::evm_word_t word7;
    size_t value = 0x12345678;
    word7.from_size_t(value);
    EXPECT_EQ(word7._limbs[0], 0x12345678);
    for (int i = 1; i < CuEVM::cgbn_limbs; ++i) {
        EXPECT_EQ(word7._limbs[i], 0);
    }

    // Test constructor with uint32_t
    CuEVM::evm_word_t word8(0x87654321);
    EXPECT_EQ(word8._limbs[0], 0x87654321);
    for (int i = 1; i < CuEVM::cgbn_limbs; ++i) {
        EXPECT_EQ(word8._limbs[i], 0);
    }

    // Test assignment operator with uint32_t
    CuEVM::evm_word_t word9;
    word9 = 0xAABBCCDD;
    EXPECT_EQ(word9._limbs[0], 0xAABBCCDD);
    for (int i = 1; i < CuEVM::cgbn_limbs; ++i) {
        EXPECT_EQ(word9._limbs[i], 0);
    }
}