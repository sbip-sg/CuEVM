
#include <gtest/gtest.h>
#include <CuEVM/core/evm_word.cuh>
#include <CuEVM/utils/error_codes.cuh>
#include <CuEVM/utils/evm_utils.cuh>

__global__ void test_evm_word_t_kernel(bool *result) {
    bool all_tests_passed = true;

    // Test default constructor and from_uint32_t
    CuEVM::evm_word_t word1;
    word1.from_uint32_t(0x12345678);
    all_tests_passed &= (word1._limbs[0] == 0x12345678);
    for (int i = 1; i < CuEVM::cgbn_limbs; ++i) {
        all_tests_passed &= (word1._limbs[i] == 0);
    }

    // Test copy constructor
    CuEVM::evm_word_t word2(word1);
    all_tests_passed &= (word2 == word1);

    // Test assignment operator
    CuEVM::evm_word_t word3;
    word3 = word1;
    all_tests_passed &= (word3 == word1);

    // Test equality operator with evm_word_t
    all_tests_passed &= (word1 == word2);
    all_tests_passed &= (word1 == word3);

    // Test equality operator with uint32_t
    all_tests_passed &= (word1 == 0x12345678);
    all_tests_passed &= !(word1 == 0x87654321);

    // Test from_uint64_t
    CuEVM::evm_word_t word4;
    word4.from_uint64_t(0x123456789ABCDEF0);
    all_tests_passed &= (word4._limbs[0] == 0x9ABCDEF0);
    all_tests_passed &= (word4._limbs[1] == 0x12345678);
    for (int i = 2; i < CuEVM::cgbn_limbs; ++i) {
        all_tests_passed &= (word4._limbs[i] == 0);
    }

    // Test from_hex
    CuEVM::evm_word_t word5;
    word5.from_hex("0x123456789ABCDEF0");
    all_tests_passed &= (word5 == word4);

    // Test to_hex
    char hex_string[2 + CuEVM::cgbn_limbs * 8 + 1];
    word4.to_hex(hex_string, 0, 2);
    all_tests_passed &= (strcmp(hex_string, "0x123456789abcdef0") == 0);

    // Test to_byte_array_t
    CuEVM::byte_array_t byte_array;
    word4.to_byte_array_t(&byte_array);
    all_tests_passed &= (byte_array.data[0] == 0x12);
    all_tests_passed &= (byte_array.data[1] == 0x34);
    all_tests_passed &= (byte_array.data[2] == 0x56);
    all_tests_passed &= (byte_array.data[3] == 0x78);
    all_tests_passed &= (byte_array.data[4] == 0x9A);
    all_tests_passed &= (byte_array.data[5] == 0xBC);
    all_tests_passed &= (byte_array.data[6] == 0xDE);
    all_tests_passed &= (byte_array.data[7] == 0xF0);

    // Test to_bit_array_t
    CuEVM::byte_array_t bit_array;
    word4.to_bit_array_t(&bit_array);
    for (int i = 0; i < 32; ++i) {
        all_tests_passed &= (bit_array.data[i] == ((0x12345678 >> i) & 0x01));
    }
    for (int i = 0; i < 32; ++i) {
        all_tests_passed &= (bit_array.data[32 + i] == ((0x9ABCDEF0 >> i) & 0x01));
    }

    *result = all_tests_passed;
}

TEST(EvmWordTest, KernelTests) {
    bool *d_result;
    bool h_result = false;
    cudaMalloc(&d_result, sizeof(bool));
    test_evm_word_t_kernel<<<1, 1>>>(d_result);
    cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    ASSERT_TRUE(h_result);
}

TEST(EvmWordTest, FromSizeT) {
    CuEVM::evm_word_t word;
    size_t value = 0x12345678;
    word.from_size_t(value);
    ASSERT_EQ(word._limbs[0], 0x12345678);
    for (int i = 1; i < CuEVM::cgbn_limbs; ++i) {
        ASSERT_EQ(word._limbs[i], 0);
    }
}

TEST(EvmWordTest, Print) {
    CuEVM::evm_word_t word;
    word.from_uint32_t(0x12345678);
    testing::internal::CaptureStdout();
    word.print();
    std::string output = testing::internal::GetCapturedStdout();
    ASSERT_EQ(output, "00000000 00000000 00000000 12345678 \n");
}
