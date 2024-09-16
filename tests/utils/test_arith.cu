#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <CuEVM/utils/arith.cuh>
#include <CuEVM/utils/error_codes.cuh>
#include <CuEVM/utils/evm_utils.cuh>

using namespace CuEVM;

__global__ void kernel_test_cgbn_set_memory(uint32_t *result) {
    ArithEnv arithEnv(cgbn_no_checks);

    __SHARED_MEMORY__ uint8_t *src;
    __ONE_GPU_THREAD_BEGIN__
    src = new uint8_t[32]{0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,
                          0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,
                          0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,
                          0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0};
    __ONE_GPU_THREAD_END__
    bn_t dst;
    cgbn_set_memory(arithEnv.env, dst, src, 32);
    *result = cgbn_get_ui32(arithEnv.env, dst);
    __ONE_GPU_THREAD_BEGIN__
    delete src;
    __ONE_GPU_THREAD_END__
}

TEST(ArithTests, TestCgbnSetMemoryCPU) {
    ArithEnv arithEnv(cgbn_no_checks, 0);

    uint8_t src[32] = {0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,
                       0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,
                       0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,
                       0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0};
    bn_t dst;
    cgbn_set_memory(arithEnv.env, dst, src, 32);

    uint32_t result;
    result = cgbn_get_ui32(arithEnv.env, dst);
    EXPECT_EQ(result, 0x9ABCDEF0);
}

#ifdef GPU
TEST(ArithTests, TestCgbnSetMemory) {
    uint32_t *d_result;
    uint32_t h_result;
    CUDA_CHECK(cudaDeviceReset());
    cudaMalloc(&d_result, sizeof(uint32_t));

    kernel_test_cgbn_set_memory<<<1, CuEVM::cgbn_tpi>>>(d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    EXPECT_EQ(h_result, 0x9ABCDEF0);
    cudaFree(d_result);
    CUDA_CHECK(cudaDeviceReset());
}
#endif

TEST(ArithTests, TestCgbnSetSizeTCPU) {
    ArithEnv arithEnv(cgbn_no_checks, 0);

    size_t src = 0x123456789ABCDEF0;
    bn_t dst;
    cgbn_set_size_t(arithEnv.env, dst, src);
    uint64_t result;
    cgbn_get_uint64_t(arithEnv.env, result, dst);
    EXPECT_EQ(result, 0x123456789ABCDEF0);
}

__global__ void kernel_test_cgbn_set_size_t(uint64_t *result) {
    ArithEnv arithEnv(cgbn_no_checks);

    size_t src = 0x123456789ABCDEF0;
    bn_t dst;
    cgbn_set_size_t(arithEnv.env, dst, src);
    cgbn_get_uint64_t(arithEnv.env, *result, dst);
}

#ifdef GPU
TEST(ArithTests, TestCgbnSetSizeT) {
    uint64_t *d_result;
    uint64_t h_result;
    CUDA_CHECK(cudaDeviceReset());
    cudaMalloc(&d_result, sizeof(uint64_t));

    kernel_test_cgbn_set_size_t<<<1, CuEVM::cgbn_tpi>>>(d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    EXPECT_EQ(h_result, 0x123456789ABCDEF0);
    cudaFree(d_result);
    CUDA_CHECK(cudaDeviceReset());
}
#endif

TEST(ArithTests, TestCgbnGetSizeTCPU) {
    ArithEnv arithEnv(cgbn_no_checks, 0);

    bn_t src;
    size_t expected = 0x123456789ABCDEF0;
    cgbn_set_size_t(arithEnv.env, src, expected);
    size_t result;
    EXPECT_EQ(cgbn_get_size_t(arithEnv.env, result, src), ERROR_SUCCESS);
    EXPECT_EQ(result, 0x123456789ABCDEF0);
}

__global__ void kernel_test_cgbn_get_size_t(size_t *result,
                                            int32_t *error_code) {
    ArithEnv arithEnv(cgbn_no_checks);

    bn_t src;
    size_t expected = 0x123456789ABCDEF0;
    cgbn_set_size_t(arithEnv.env, src, expected);

    *error_code = cgbn_get_size_t(arithEnv.env, *result, src);
}

#ifdef GPU
TEST(ArithTests, TestCgbnGetSizeT) {
    size_t *d_result;
    int32_t *d_error_code;
    size_t h_result;
    int32_t h_error_code;
    CUDA_CHECK(cudaDeviceReset());
    cudaMalloc(&d_result, sizeof(size_t));
    cudaMalloc(&d_error_code, sizeof(int32_t));

    kernel_test_cgbn_get_size_t<<<1, CuEVM::cgbn_tpi>>>(d_result, d_error_code);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_error_code, d_error_code, sizeof(int32_t),
               cudaMemcpyDeviceToHost);

    EXPECT_EQ(h_result, 0x123456789ABCDEF0);
    EXPECT_EQ(h_error_code, ERROR_SUCCESS);
    cudaFree(d_result);
    cudaFree(d_error_code);
    CUDA_CHECK(cudaDeviceReset());
}
#endif

TEST(ArithTests, TestCgbnGetUint64TCPU) {
    ArithEnv arithEnv(cgbn_no_checks, 0);

    bn_t src;
    uint64_t expected = 0x123456789ABCDEF0;
    cgbn_set_size_t(arithEnv.env, src, expected);
    uint64_t result;
    EXPECT_EQ(cgbn_get_uint64_t(arithEnv.env, result, src), ERROR_SUCCESS);
    EXPECT_EQ(result, 0x123456789ABCDEF0);
}

__global__ void kernel_test_cgbn_get_uint64_t(uint64_t *result,
                                              int32_t *error_code) {
    ArithEnv arithEnv(cgbn_no_checks);

    bn_t src;
    uint64_t expected = 0x123456789ABCDEF0;
    cgbn_set_size_t(arithEnv.env, src, expected);

    *error_code = cgbn_get_uint64_t(arithEnv.env, *result, src);
}

#ifdef GPU
TEST(ArithTests, TestCgbnGetUint64T) {
    uint64_t *d_result;
    int32_t *d_error_code;
    uint64_t h_result;
    int32_t h_error_code;
    CUDA_CHECK(cudaDeviceReset());
    cudaMalloc(&d_result, sizeof(uint64_t));
    cudaMalloc(&d_error_code, sizeof(int32_t));

    kernel_test_cgbn_get_uint64_t<<<1, CuEVM::cgbn_tpi>>>(d_result,
                                                          d_error_code);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_error_code, d_error_code, sizeof(int32_t),
               cudaMemcpyDeviceToHost);

    EXPECT_EQ(h_result, 0x123456789ABCDEF0);
    EXPECT_EQ(h_error_code, ERROR_SUCCESS);
    cudaFree(d_result);
    cudaFree(d_error_code);
    CUDA_CHECK(cudaDeviceReset());
}
#endif

TEST(ArithTests, TestCgbnGetUint32TCPU) {
    ArithEnv arithEnv(cgbn_no_checks, 0);

    bn_t src;
    uint32_t expected = 0x12345678;
    cgbn_set_ui32(arithEnv.env, src, expected);

    uint32_t result;
    EXPECT_EQ(cgbn_get_uint32_t(arithEnv.env, result, src), ERROR_SUCCESS);
    EXPECT_EQ(result, 0x12345678);
}

__global__ void kernel_test_cgbn_get_uint32_t(uint32_t *result,
                                              int32_t *error_code) {
    ArithEnv arithEnv(cgbn_no_checks);

    bn_t src;
    uint32_t expected = 0x12345678;
    cgbn_set_ui32(arithEnv.env, src, expected);

    *error_code = cgbn_get_uint32_t(arithEnv.env, *result, src);
}

#ifdef GPU
TEST(ArithTests, TestCgbnGetUint32T) {
    uint32_t *d_result;
    int32_t *d_error_code;
    uint32_t h_result;
    int32_t h_error_code;
    CUDA_CHECK(cudaDeviceReset());
    cudaMalloc(&d_result, sizeof(uint32_t));
    cudaMalloc(&d_error_code, sizeof(int32_t));

    kernel_test_cgbn_get_uint32_t<<<1, CuEVM::cgbn_tpi>>>(d_result,
                                                          d_error_code);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_error_code, d_error_code, sizeof(int32_t),
               cudaMemcpyDeviceToHost);

    EXPECT_EQ(h_result, 0x12345678);
    EXPECT_EQ(h_error_code, ERROR_SUCCESS);
    cudaFree(d_result);
    cudaFree(d_error_code);
    CUDA_CHECK(cudaDeviceReset());
}
#endif

TEST(ArithTests, TestCgbnSetByteArrayTCPU) {
    ArithEnv arithEnv(cgbn_no_checks, 0);
    uint8_t data[32] = {0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,
                        0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,
                        0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,
                        0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0};

    byte_array_t byte_array(data, 32);
    bn_t out;

    EXPECT_EQ(cgbn_set_byte_array_t(arithEnv.env, out, byte_array),
              ERROR_SUCCESS);
    EXPECT_EQ(cgbn_get_ui32(arithEnv.env, out), 0x9ABCDEF0);
}

__global__ void kernel_test_cgbn_set_byte_array_t(uint32_t *result,
                                                  int32_t *error_code) {
    ArithEnv arithEnv(cgbn_no_checks);
    __SHARED_MEMORY__ uint8_t *src;
    __ONE_GPU_THREAD_BEGIN__
    src = new uint8_t[32]{0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,
                          0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,
                          0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,
                          0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0};
    __ONE_GPU_THREAD_END__

    byte_array_t byte_array(src, 32);
    bn_t out;

    *error_code = cgbn_set_byte_array_t(arithEnv.env, out, byte_array);
    *result = cgbn_get_ui32(arithEnv.env, out);

    __ONE_GPU_THREAD_BEGIN__
    delete src;
    __ONE_GPU_THREAD_END__
}

#ifdef GPU
TEST(ArithTests, TestCgbnSetByteArrayT) {
    uint32_t *d_result;
    int32_t *d_error_code;
    uint32_t h_result;
    int32_t h_error_code;
    CUDA_CHECK(cudaDeviceReset());
    cudaMalloc(&d_result, sizeof(uint32_t));
    cudaMalloc(&d_error_code, sizeof(int32_t));

    kernel_test_cgbn_set_byte_array_t<<<1, CuEVM::cgbn_tpi>>>(d_result,
                                                              d_error_code);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_error_code, d_error_code, sizeof(int32_t),
               cudaMemcpyDeviceToHost);

    EXPECT_EQ(h_result, 0x9ABCDEF0);
    EXPECT_EQ(h_error_code, ERROR_SUCCESS);
    cudaFree(d_result);
    cudaFree(d_error_code);
    CUDA_CHECK(cudaDeviceReset());
}
#endif

TEST(ArithTests, TestGetSubByteArrayTCPU) {
    ArithEnv arithEnv(cgbn_no_checks, 0);
    uint8_t data[8] = {0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0};

    byte_array_t byte_array(data, 8);
    bn_t index, length;
    cgbn_set_ui32(arithEnv.env, index, 2);
    cgbn_set_ui32(arithEnv.env, length, 4);
    byte_array_t out;

    EXPECT_EQ(get_sub_byte_array_t(arithEnv, byte_array, index, length, out),
              ERROR_SUCCESS);
    for (uint32_t i = 0; i < 4; ++i) {
        EXPECT_EQ(out.data[i], data[i + 2]);
    }
}

__global__ void kernel_test_get_sub_byte_array_t(int32_t *error_code,
                                                 uint8_t *result) {
    ArithEnv arithEnv(cgbn_no_checks);

    __SHARED_MEMORY__ uint8_t *data;
    __ONE_GPU_THREAD_BEGIN__
    data = new uint8_t[8]{0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0};
    __ONE_GPU_THREAD_END__

    byte_array_t byte_array(data, 8);
    bn_t index, length;
    cgbn_set_ui32(arithEnv.env, index, 2);
    cgbn_set_ui32(arithEnv.env, length, 4);
    byte_array_t out;

    *error_code =
        get_sub_byte_array_t(arithEnv, byte_array, index, length, out);

    for (int i = 0; i < 4; ++i) {
        result[i] = out.data[i];
    }

    __ONE_GPU_THREAD_BEGIN__
    delete data;
    __ONE_GPU_THREAD_END__
}

#ifdef GPU
TEST(ArithTests, TestGetSubByteArrayT) {
    int32_t *d_error_code;
    uint8_t *d_result;
    int32_t h_error_code;
    uint8_t h_result[4];
    CUDA_CHECK(cudaDeviceReset());
    cudaMalloc(&d_error_code, sizeof(int32_t));
    cudaMalloc(&d_result, 4 * sizeof(uint8_t));

    kernel_test_get_sub_byte_array_t<<<1, CuEVM::cgbn_tpi>>>(d_error_code,
                                                             d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_error_code, d_error_code, sizeof(int32_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_result, d_result, 4 * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    EXPECT_EQ(h_result[0], 0x56);
    EXPECT_EQ(h_result[1], 0x78);
    EXPECT_EQ(h_result[2], 0x9A);
    EXPECT_EQ(h_result[3], 0xBC);
    EXPECT_EQ(h_error_code, ERROR_SUCCESS);
    cudaFree(d_error_code);
    cudaFree(d_result);
    CUDA_CHECK(cudaDeviceReset());
}
#endif

TEST(ArithTests, TestEvmAddressConversionCPU) {
    ArithEnv arithEnv(cgbn_no_checks, 0);

    uint8_t src[32] = {0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,
                       0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,
                       0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,
                       0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0};
    bn_t address;
    cgbn_set_memory(arithEnv.env, address, src, 32);

    evm_address_conversion(arithEnv, address);

    evm_word_t result;
    cgbn_store(arithEnv.env, (cgbn_evm_word_t_ptr)&result, address);

    // address are 20 bytes long, so the first 12 bytes should be 0x00 12 = 3 *
    // 4 bytes
    EXPECT_EQ(result._limbs[7], 0x00);
    EXPECT_EQ(result._limbs[6], 0x00);
    EXPECT_EQ(result._limbs[5], 0x00);
    EXPECT_EQ(result._limbs[4], 0x9ABCDEF0);
    EXPECT_EQ(result._limbs[3], 0x12345678);
    EXPECT_EQ(result._limbs[2], 0x9ABCDEF0);
    EXPECT_EQ(result._limbs[1], 0x12345678);
    EXPECT_EQ(result._limbs[0], 0x9ABCDEF0);
}

__global__ void kernel_test_evm_address_conversion(uint32_t *result) {
    ArithEnv arithEnv(cgbn_no_checks);

    __SHARED_MEMORY__ uint8_t *src;
    __ONE_GPU_THREAD_BEGIN__
    src = new uint8_t[32]{0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,
                          0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,
                          0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0,
                          0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0};
    __ONE_GPU_THREAD_END__
    bn_t address;
    cgbn_set_memory(arithEnv.env, address, src, 32);
    evm_address_conversion(arithEnv, address);

    __SHARED_MEMORY__ evm_word_t result_evm_word;
    cgbn_store(arithEnv.env, (cgbn_evm_word_t_ptr)&result_evm_word, address);
    *result |= result_evm_word._limbs[7] == 0x00 ? ERROR_SUCCESS : __LINE__;
    *result |= result_evm_word._limbs[6] == 0x00 ? ERROR_SUCCESS : __LINE__;
    *result |= result_evm_word._limbs[5] == 0x00 ? ERROR_SUCCESS : __LINE__;
    *result |=
        result_evm_word._limbs[4] == 0x9ABCDEF0 ? ERROR_SUCCESS : __LINE__;
    *result |=
        result_evm_word._limbs[3] == 0x12345678 ? ERROR_SUCCESS : __LINE__;
    *result |=
        result_evm_word._limbs[2] == 0x9ABCDEF0 ? ERROR_SUCCESS : __LINE__;
    *result |=
        result_evm_word._limbs[1] == 0x12345678 ? ERROR_SUCCESS : __LINE__;
    *result |=
        result_evm_word._limbs[0] == 0x9ABCDEF0 ? ERROR_SUCCESS : __LINE__;

    __ONE_GPU_THREAD_BEGIN__
    delete src;
    __ONE_GPU_THREAD_END__
}

#ifdef GPU
TEST(ArithTests, TestEvmAddressConversion) {
    uint32_t *d_result;
    uint32_t h_result;
    CUDA_CHECK(cudaDeviceReset());
    cudaMalloc(&d_result, sizeof(uint32_t));

    kernel_test_evm_address_conversion<<<1, CuEVM::cgbn_tpi>>>(d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    EXPECT_EQ(h_result, ERROR_SUCCESS);
    cudaFree(d_result);
    CUDA_CHECK(cudaDeviceReset());
}
#endif