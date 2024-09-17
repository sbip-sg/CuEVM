#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <CuEVM/core/evm_word.cuh>
#include <CuEVM/state/storage_element.cuh>
#include <CuEVM/utils/arith.cuh>
#include <CuEVM/utils/error_codes.cuh>

class StorageElementTest : public ::testing::Test {
   protected:
    CuEVM::ArithEnv arith;
    CuEVM::evm_word_t key, value;
    CuEVM::storage_element_t element;
    CuEVM::bn_t bn_key, bn_value;

    StorageElementTest() : arith(cgbn_no_checks), key(1U), value(2U), element(key, value) {}

    void SetUp() override {
        // Any additional setup can be done here
    }

    void TearDown() override {
        // Any cleanup can be done here
    }
};

TEST_F(StorageElementTest, ParameterizedConstructor) {
    EXPECT_EQ(element.get_key(), key) << "Key should be set correctly";
    EXPECT_EQ(element.get_value(), value) << "Value should be set correctly";
    element.get_value(arith, bn_value);
    element.get_key(arith, bn_key);
    EXPECT_EQ(cgbn_compare_ui32(arith.env, bn_key, 1U), 0) << "Key should be set correctly with ArithEnv";
    EXPECT_EQ(cgbn_compare_ui32(arith.env, bn_value, 2U), 0) << "Value should be set correctly with ArithEnv";
}

TEST_F(StorageElementTest, CopyConstructor) {
    CuEVM::storage_element_t element2(element);
    EXPECT_EQ(element2.get_key(), key) << "Copied key should be the same";
    EXPECT_EQ(element2.get_value(), value) << "Copied value should be the same";
}

TEST_F(StorageElementTest, SetGetValue) {
    CuEVM::evm_word_t new_value = 3;
    element.set_value(new_value);
    EXPECT_EQ(element.get_value(), new_value) << "Value should be set and retrieved correctly";
}

TEST_F(StorageElementTest, SetGetKey) {
    CuEVM::evm_word_t new_key = 4;
    element.set_key(new_key);
    EXPECT_EQ(element.get_key(), new_key) << "Key should be set and retrieved correctly";
}

TEST_F(StorageElementTest, AssignmentOperator) {
    CuEVM::storage_element_t element2;
    element2 = element;
    EXPECT_EQ(element2.get_key(), key) << "Assigned key should be the same";
    EXPECT_EQ(element2.get_value(), value) << "Assigned value should be the same";
}

TEST_F(StorageElementTest, IsZeroValue) {
    EXPECT_EQ(element.is_zero_value(), 0) << "Default value should not be zero";
    element.set_value(0);
    EXPECT_EQ(element.is_zero_value(), 1) << "Zero value should be zero";
}

TEST_F(StorageElementTest, SetGetKeyWithArithEnv) {
    CuEVM::storage_element_t element;
    CuEVM::bn_t key;
    cgbn_set_ui32(arith.env, key, 5);
    element.set_key(arith, key);
    CuEVM::bn_t retrieved_key;
    element.get_key(arith, retrieved_key);
    EXPECT_EQ(cgbn_compare(arith.env, key, retrieved_key), 0)
        << "Key should be set and retrieved correctly with ArithEnv";
}

TEST_F(StorageElementTest, SetGetValueWithArithEnv) {
    CuEVM::storage_element_t element;
    CuEVM::bn_t value;
    cgbn_set_ui32(arith.env, value, 6);
    element.set_value(arith, value);
    CuEVM::bn_t retrieved_value;
    element.get_value(arith, retrieved_value);
    EXPECT_EQ(cgbn_compare(arith.env, value, retrieved_value), 0)
        << "Value should be set and retrieved correctly with ArithEnv";
}

TEST_F(StorageElementTest, HasKey) {
    CuEVM::evm_word_t new_key = 7;
    element.set_key(new_key);
    EXPECT_EQ(element.has_key(new_key), 1) << "Key should be found";
    CuEVM::evm_word_t other_key = 8;
    EXPECT_EQ(element.has_key(other_key), 0) << "Other key should not be found";
}

TEST_F(StorageElementTest, HasKeyWithArithEnv) {
    CuEVM::storage_element_t element;
    CuEVM::bn_t key;
    cgbn_set_ui32(arith.env, key, 9);
    element.set_key(arith, key);
    EXPECT_EQ(element.has_key(arith, key), 1) << "Key should be found with ArithEnv";
    CuEVM::bn_t other_key;
    cgbn_set_ui32(arith.env, other_key, 10);
    EXPECT_EQ(element.has_key(arith, other_key), 0) << "Other key should not be found with ArithEnv";
}

__global__ void test_storage_element_kernel(uint32_t count, uint32_t *result) {
    int32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / CuEVM::cgbn_tpi;
    if (instance >= count) return;
    result[instance] = ERROR_SUCCESS;
    CuEVM::ArithEnv arithEnv(cgbn_no_checks);

    // Test HasKey
    CuEVM::storage_element_t element;
    CuEVM::evm_word_t new_key = 7;
    element.set_key(new_key);
    result[instance] = (element.has_key(new_key) == 1) ? ERROR_SUCCESS : __LINE__;
    CuEVM::evm_word_t other_key = 8;
    result[instance] |= (element.has_key(other_key) == 0) ? ERROR_SUCCESS : __LINE__;

    // Test HasKeyWithArithEnv
    CuEVM::bn_t key;
    cgbn_set_ui32(arithEnv.env, key, 9);
    element.set_key(arithEnv, key);
    result[instance] |= (element.has_key(arithEnv, key) == 1) ? ERROR_SUCCESS : __LINE__;
    CuEVM::bn_t other_key_bn;
    cgbn_set_ui32(arithEnv.env, other_key_bn, 10);
    result[instance] |= (element.has_key(arithEnv, other_key_bn) == 0) ? ERROR_SUCCESS : __LINE__;

    // Test SetValue
    CuEVM::evm_word_t new_value = 3;
    element.set_value(new_value);
    result[instance] |= (element.get_value() == new_value) ? ERROR_SUCCESS : __LINE__;
    CuEVM::bn_t value;
    cgbn_set_ui32(arithEnv.env, value, 6);
    element.set_value(arithEnv, value);
    CuEVM::bn_t retrieved_value;
    element.get_value(arithEnv, retrieved_value);
    result[instance] |= (cgbn_compare(arithEnv.env, value, retrieved_value) == 0) ? ERROR_SUCCESS : __LINE__;

    // Test isZeroValue
    cgbn_set_ui32(arithEnv.env, value, 0);
    element.set_value(arithEnv, value);
    result[instance] |= (element.is_zero_value() == 1) ? ERROR_SUCCESS : __LINE__;
    element.set_value(new_value);
    result[instance] |= (element.is_zero_value() == 0) ? ERROR_SUCCESS : __LINE__;

    // Test SetGetKeyWithArithEnv
    cgbn_set_ui32(arithEnv.env, key, 5);
    element.set_key(arithEnv, key);
    CuEVM::bn_t retrieved_key;
    element.get_key(arithEnv, retrieved_key);
    result[instance] |= (cgbn_compare(arithEnv.env, key, retrieved_key) == 0) ? ERROR_SUCCESS : __LINE__;

    // Test AssignmentOperator
    CuEVM::storage_element_t element2;
    element2 = element;
    result[instance] |= (element2.get_key() == element.get_key()) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (element2.get_value() == element.get_value()) ? ERROR_SUCCESS : __LINE__;

    // Test CopyConstructor
    CuEVM::storage_element_t element3(element);
    result[instance] |= (element3.get_key() == element.get_key()) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (element3.get_value() == element.get_value()) ? ERROR_SUCCESS : __LINE__;

    // Test SetGetValue
    element.set_value(new_value);
    result[instance] |= (element.get_value() == new_value) ? ERROR_SUCCESS : __LINE__;

    // Test SetGetKey
    element.set_key(new_key);
    result[instance] |= (element.get_key() == new_key) ? ERROR_SUCCESS : __LINE__;

    // Test copy constructr with ArithEnv
    CuEVM::storage_element_t element4(element);
    CuEVM::bn_t key4, value4;
    element4.get_key(arithEnv, key4);
    element4.get_value(arithEnv, value4);
    CuEVM::bn_t key5, value5;
    element.get_key(arithEnv, key5);
    element.get_value(arithEnv, value5);
    result[instance] |= (cgbn_compare(arithEnv.env, key4, key5) == 0) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (cgbn_compare(arithEnv.env, value4, value5) == 0) ? ERROR_SUCCESS : __LINE__;
}

#ifdef GPU
TEST_F(StorageElementTest, KernelTestsGPU) {
    CUDA_CHECK(cudaDeviceReset());
    uint32_t *d_result;
    cudaMalloc(&d_result, 2 * sizeof(uint32_t));
    test_storage_element_kernel<<<2, CuEVM::cgbn_tpi>>>(2, d_result);
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