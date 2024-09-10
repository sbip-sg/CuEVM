#include <gtest/gtest.h>
#include <stdint.h>
#include <stdlib.h>

#include <CuEVM/core/byte_array.cuh>
#include <CuEVM/utils/evm_defines.cuh>
#include <CuEVM/utils/error_codes.cuh>

TEST(ByteArrayTests, ConstructorWithHexString) {
    CuEVM::byte_array_t byteArray("0A1B2C");
    ASSERT_EQ(byteArray.size, 3);
    ASSERT_EQ(byteArray[0], 0x0A);
    ASSERT_EQ(byteArray[1], 0x1B);
    ASSERT_EQ(byteArray[2], 0x2C);
}

TEST(ByteArrayTests, ConstructorWithHexStringAndFixedSize) {
    CuEVM::byte_array_t byteArray("0A1B2C", 5U);
    ASSERT_EQ(byteArray.size, 5);
    ASSERT_EQ(byteArray[0], 0x0A);
    ASSERT_EQ(byteArray[1], 0x1B);
    ASSERT_EQ(byteArray[2], 0x2C);
    ASSERT_EQ(byteArray[3], 0x00);  // Assuming zero padding
    ASSERT_EQ(byteArray[4], 0x00);  // Assuming zero padding
}

TEST(ByteArrayTests, CopyConstructor) {
    CuEVM::byte_array_t byteArray1("0A1B2C");
    CuEVM::byte_array_t byteArray2(byteArray1);
    ASSERT_EQ(byteArray2.size, 3);
    ASSERT_EQ(byteArray2[0], 0x0A);
    ASSERT_EQ(byteArray2[1], 0x1B);
    ASSERT_EQ(byteArray2[2], 0x2C);
}

TEST(ByteArrayTests, AssignmentOperator) {
    CuEVM::byte_array_t byteArray1("0A1B2C");
    CuEVM::byte_array_t byteArray2 = byteArray1;
    ASSERT_EQ(byteArray2.size, 3);
    ASSERT_EQ(byteArray2[0], 0x0A);
    ASSERT_EQ(byteArray2[1], 0x1B);
    ASSERT_EQ(byteArray2[2], 0x2C);
}

TEST(ByteArrayTests, GrowFunction) {
    CuEVM::byte_array_t byteArray("0A1B2C");
    byteArray.grow(5, 1);  // Grow with zero padding
    ASSERT_EQ(byteArray.size, 5);
    ASSERT_EQ(byteArray[0], 0x0A);
    ASSERT_EQ(byteArray[1], 0x1B);
    ASSERT_EQ(byteArray[2], 0x2C);
    ASSERT_EQ(byteArray[3], 0x00);  // Zero padding
    ASSERT_EQ(byteArray[4], 0x00);  // Zero padding
}

TEST(ByteArrayTests, PrintFunction) {
    CuEVM::byte_array_t byteArray("0A1B2C");
    testing::internal::CaptureStdout();
    byteArray.print();
    std::string output = testing::internal::GetCapturedStdout();
    ASSERT_EQ(output,
              "size: 3\ndata: 0a1b2c\n");  // Assuming print outputs the hex
                                           // string followed by a newline
}

TEST(ByteArrayTests, DefaultConstructor) {
    CuEVM::byte_array_t byteArray;
    EXPECT_EQ(byteArray.size, 0);
    EXPECT_EQ(byteArray.data, nullptr);
}

TEST(ByteArrayTests, ConstructorWithSize) {
    CuEVM::byte_array_t byteArray(10);
    EXPECT_EQ(byteArray.size, 10);
    EXPECT_NE(byteArray.data, nullptr);
}

TEST(ByteArrayTests, ConstructorWithData) {
    uint8_t data[5] = {1, 2, 3, 4, 5};
    CuEVM::byte_array_t byteArray(data, 5);
    EXPECT_EQ(byteArray.size, 5);
    EXPECT_NE(byteArray.data, nullptr);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(byteArray.data[i], data[i]);
    }
}

TEST(ByteArrayTests, CopyConstructor2) {
    uint8_t data[5] = {1, 2, 3, 4, 5};
    CuEVM::byte_array_t byteArray1(data, 5);
    CuEVM::byte_array_t byteArray2(byteArray1);
    EXPECT_EQ(byteArray2.size, 5);
    EXPECT_NE(byteArray2.data, nullptr);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(byteArray2.data[i], data[i]);
    }
}

TEST(ByteArrayTests, AssignmentOperator2) {
    uint8_t data1[5] = {1, 2, 3, 4, 5};
    CuEVM::byte_array_t byteArray1(data1, 5);
    CuEVM::byte_array_t byteArray2;
    byteArray2 = byteArray1;
    EXPECT_EQ(byteArray2.size, 5);
    EXPECT_NE(byteArray2.data, nullptr);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(byteArray2.data[i], data1[i]);
    }
}

TEST(ByteArrayTests, Grow) {
    CuEVM::byte_array_t byteArray(5);
    byteArray.grow(10, 1);
    EXPECT_EQ(byteArray.size, 10);
    for (int i = 5; i < 10; ++i) {
        EXPECT_EQ(byteArray.data[i], 0);
    }
}

TEST(ByteArrayTests, ToHex) {
    uint8_t data[3] = {0x12, 0x34, 0x56};
    CuEVM::byte_array_t byteArray(data, 3);
    char* hexString = byteArray.to_hex();
    EXPECT_STREQ(hexString, "0x123456");
    free(hexString);
}

TEST(ByteArrayTests, FromHexSetLE) {
    CuEVM::byte_array_t byteArray(3);
    byteArray.from_hex("3456", LITTLE_ENDIAN, CuEVM::NO_PADDING);
    EXPECT_EQ(byteArray.size, 3);
    EXPECT_EQ(byteArray.data[0], 0x34);
    EXPECT_EQ(byteArray.data[1], 0x56);
    EXPECT_EQ(byteArray.data[2], 0x00);
}

TEST(ByteArrayTests, FromHexSetBE_LP) {
    CuEVM::byte_array_t byteArray(3);
    byteArray.from_hex("3456", BIG_ENDIAN, CuEVM::LEFT_PADDING);
    EXPECT_EQ(byteArray.size, 3);
    EXPECT_EQ(byteArray.data[0], 0x56);
    EXPECT_EQ(byteArray.data[1], 0x34);
    EXPECT_EQ(byteArray.data[2], 0x00);
}
TEST(ByteArrayTests, FromHexSetBE_RP) {
    CuEVM::byte_array_t byteArray(3);
    byteArray.from_hex("3456", BIG_ENDIAN, CuEVM::RIGHT_PADDING);
    EXPECT_EQ(byteArray.size, 3);
    EXPECT_EQ(byteArray.data[0], 0x00);
    EXPECT_EQ(byteArray.data[1], 0x56);
    EXPECT_EQ(byteArray.data[2], 0x34);
}

// Additional tests

TEST(ByteArrayTests, ConstructorWithHexString2) {
    CuEVM::byte_array_t byteArray("123456", LITTLE_ENDIAN, CuEVM::NO_PADDING);
    EXPECT_EQ(byteArray.size, 3);
    EXPECT_EQ(byteArray.data[0], 0x12);
    EXPECT_EQ(byteArray.data[1], 0x34);
    EXPECT_EQ(byteArray.data[2], 0x56);
}

TEST(ByteArrayTests, ConstructorWithHexStringAndSize) {
    CuEVM::byte_array_t byteArray("123456", 4, LITTLE_ENDIAN,
                                  CuEVM::LEFT_PADDING);
    EXPECT_EQ(byteArray.size, 4);
    EXPECT_EQ(byteArray.data[0], 0x12);
    EXPECT_EQ(byteArray.data[1], 0x34);
    EXPECT_EQ(byteArray.data[2], 0x56);
    EXPECT_EQ(byteArray.data[3], 0x00);
}

TEST(ByteArrayTests, PaddedCopyBE) {
    uint8_t data[3] = {0x12, 0x34, 0x56};
    CuEVM::byte_array_t src(data, 3);
    CuEVM::byte_array_t dst(5);
    dst.padded_copy_BE(src);
    EXPECT_EQ(dst.size, 5);
    EXPECT_EQ(dst.data[0], 0x12);
    EXPECT_EQ(dst.data[1], 0x34);
    EXPECT_EQ(dst.data[2], 0x56);
    EXPECT_EQ(dst.data[3], 0x00);
    EXPECT_EQ(dst.data[4], 0x00);
}

TEST(ByteArrayTests, FromHex) {
    CuEVM::byte_array_t byteArray;
    byteArray.from_hex("123456", LITTLE_ENDIAN, CuEVM::NO_PADDING, 0);
    EXPECT_EQ(byteArray.size, 3);
    EXPECT_EQ(byteArray.data[0], 0x12);
    EXPECT_EQ(byteArray.data[1], 0x34);
    EXPECT_EQ(byteArray.data[2], 0x56);
}

TEST(ByteArrayTests, IndexOperator) {
    uint8_t data[3] = {0x12, 0x34, 0x56};
    CuEVM::byte_array_t byteArray(data, 3);
    EXPECT_EQ(byteArray[0], 0x12);
    EXPECT_EQ(byteArray[1], 0x34);
    EXPECT_EQ(byteArray[2], 0x56);
}

TEST(ByteArrayTests, ArrayOfZeroByteArrays) {
    CuEVM::byte_array_t byteArray[5];
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(byteArray[i].size, 0);
        EXPECT_EQ(byteArray[i].data, nullptr);
    }
}

TEST(ByteArrayTests, HasValueFunction) {
    CuEVM::byte_array_t byteArray("0A1B2C");
    EXPECT_EQ(byteArray.has_value(0x0A), ERROR_SUCCESS);
    EXPECT_EQ(byteArray.has_value(0x1B), ERROR_SUCCESS);
    EXPECT_EQ(byteArray.has_value(0x2C), ERROR_SUCCESS);
    EXPECT_EQ(byteArray.has_value(0x00), ERROR_VALUE_NOT_FOUND);
    EXPECT_EQ(byteArray.has_value(0xFF), ERROR_VALUE_NOT_FOUND);
}

TEST(ByteArrayTests, GetCpu) {
    CuEVM::byte_array_t* cpuArray = CuEVM::byte_array_t::get_cpu(5);
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(cpuArray[i].size, 0);
        EXPECT_EQ(cpuArray[i].data, nullptr);
    }
    CuEVM::byte_array_t::cpu_free(cpuArray, 5);
}

TEST(ByteArrayTests, CpuGpuFree) {
    CuEVM::byte_array_t* cpuArray = CuEVM::byte_array_t::get_cpu(2);
    cpuArray[0].grow(3, 1);
    cpuArray[0].data[0] = 0x12;
    cpuArray[0].data[1] = 0x34;
    cpuArray[0].data[2] = 0x56;

    cpuArray[1].grow(2, 1);
    cpuArray[1].data[0] = 0x78;
    cpuArray[1].data[1] = 0x9A;

    CUDA_CHECK(cudaDeviceReset());
    CuEVM::byte_array_t* gpuArray =
        CuEVM::byte_array_t::gpu_from_cpu(cpuArray, 2);

    CuEVM::byte_array_t::gpu_free(gpuArray, 2);

    CuEVM::byte_array_t::cpu_free(cpuArray, 2);
    CUDA_CHECK(cudaDeviceReset());
}

// Additional GPU tests

__global__ void testKernel(CuEVM::byte_array_t* gpuArray, uint32_t count, uint32_t *result) {
    int32_t instance =
        (blockIdx.x * blockDim.x + threadIdx.x) / CuEVM::cgbn_tpi;
    if (instance >= count) return;
    result[instance] = ERROR_SUCCESS;
    if (instance == 0) {
        gpuArray[0].grow(3, 1);
        gpuArray[0].data[0] = 0x12;
        gpuArray[0].data[1] = 0x34;
        gpuArray[0].data[2] = 0x56;
        result[instance] |= (gpuArray[0].has_value(0x12) == ERROR_SUCCESS ? ERROR_SUCCESS : ERROR_VALUE_NOT_FOUND);
        result[instance] |= (gpuArray[0].has_value(0x34) == ERROR_SUCCESS ? ERROR_SUCCESS : ERROR_VALUE_NOT_FOUND);
        result[instance] |= (gpuArray[0].has_value(0x56) == ERROR_SUCCESS ? ERROR_SUCCESS : ERROR_VALUE_NOT_FOUND);
        result[instance] |= (gpuArray[0].has_value(0x00) == ERROR_VALUE_NOT_FOUND ? ERROR_SUCCESS : ERROR_VALUE_NOT_FOUND);
    } else if (instance == 1) {
        gpuArray[1].grow(3, 1);
        gpuArray[1].data[0] = 0x78;
        gpuArray[1].data[1] = 0x9A;
        gpuArray[1].data[2] = 0xBC;
        result[instance] |= (gpuArray[1].has_value(0x78) == ERROR_SUCCESS ? ERROR_SUCCESS : ERROR_VALUE_NOT_FOUND);
        result[instance] |= (gpuArray[1].has_value(0x9A) == ERROR_SUCCESS ? ERROR_SUCCESS : ERROR_VALUE_NOT_FOUND);
        result[instance] |= (gpuArray[1].has_value(0xBC) == ERROR_SUCCESS ? ERROR_SUCCESS : ERROR_VALUE_NOT_FOUND);
        result[instance] |= (gpuArray[1].has_value(0x00) == ERROR_VALUE_NOT_FOUND ? ERROR_SUCCESS : ERROR_VALUE_NOT_FOUND);
    }
}

TEST(ByteArrayTests, GpuKernelTest) {
    CuEVM::byte_array_t* cpuArray = CuEVM::byte_array_t::get_cpu(2);
    CUDA_CHECK(cudaDeviceReset());
    CuEVM::byte_array_t* gpuArray =
        CuEVM::byte_array_t::gpu_from_cpu(cpuArray, 2);
    uint32_t *d_result;
    cudaMalloc(&d_result, 2 * sizeof(uint32_t));
    testKernel<<<2, CuEVM::cgbn_tpi>>>(gpuArray, 2, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CuEVM::byte_array_t* results =
        CuEVM::byte_array_t::cpu_from_gpu(gpuArray, 2);
    CuEVM::byte_array_t* expectedCpuArray = CuEVM::byte_array_t::get_cpu(2);
    expectedCpuArray[0].grow(3, 1);
    expectedCpuArray[0].data[0] = 0x12;
    expectedCpuArray[0].data[1] = 0x34;
    expectedCpuArray[0].data[2] = 0x56;

    expectedCpuArray[1].grow(3, 1);
    expectedCpuArray[1].data[0] = 0x78;
    expectedCpuArray[1].data[1] = 0x9A;
    expectedCpuArray[1].data[2] = 0xBC;

    for (int i = 0; i < 2; ++i) {
        EXPECT_EQ(results[i].size, expectedCpuArray[i].size);
        for (int j = 0; j < results[i].size; ++j) {
            EXPECT_EQ(results[i].data[j], expectedCpuArray[i].data[j]);
        }
    }
    uint32_t *h_result;
    h_result = (uint32_t *)malloc(2 * sizeof(uint32_t));
    CUDA_CHECK(cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 2; i++) {
        EXPECT_EQ(h_result[i], ERROR_SUCCESS);
    }
    free(h_result);
    CUDA_CHECK(cudaFree(d_result));
    CuEVM::byte_array_t::cpu_free(cpuArray, 2);
    CuEVM::byte_array_t::cpu_free(expectedCpuArray, 2);
    CuEVM::byte_array_t::cpu_free(results, 2);
    CUDA_CHECK(cudaDeviceReset());
}