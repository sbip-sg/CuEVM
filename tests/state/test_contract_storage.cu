#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <CuEVM/core/evm_word.cuh>
#include <CuEVM/state/contract_storage.cuh>
#include <CuEVM/state/storage_element.cuh>
#include <CuEVM/utils/arith.cuh>
#include <CuEVM/utils/error_codes.cuh>

class ContractStorageTest : public ::testing::Test {
   protected:
    CuEVM::ArithEnv arith;
    CuEVM::contract_storage_t storage;

    ContractStorageTest() : arith(cgbn_no_checks), storage() {}

    void SetUp() override {
        // Any additional setup can be done here
    }

    void TearDown() override {
        // Any cleanup can be done here
        storage.free();
    }
};

TEST_F(ContractStorageTest, DefaultConstrcutor) {
    EXPECT_EQ(storage.size, 0) << "Size should be 0";
    EXPECT_EQ(storage.capacity, 0) << "Capacity should be 0";
    EXPECT_EQ(storage.storage, nullptr) << "Storage should be nullptr";
}

TEST_F(ContractStorageTest, Free) {
    storage.free();
    EXPECT_EQ(storage.size, 0) << "Size should be 0";
    EXPECT_EQ(storage.capacity, 0) << "Capacity should be 0";
    EXPECT_EQ(storage.storage, nullptr) << "Storage should be nullptr";
}

TEST_F(ContractStorageTest, Clear) {
    storage.size = 1;
    storage.clear();
    EXPECT_EQ(storage.size, 0) << "Size should be 0";
    EXPECT_EQ(storage.capacity, 0) << "Capacity should be 0";
    EXPECT_EQ(storage.storage, nullptr) << "Storage should be nullptr";
}

TEST_F(ContractStorageTest, SetGetValue) {
    CuEVM::bn_t key, value;
    cgbn_set_ui32(arith.env, key, 1);
    cgbn_set_ui32(arith.env, value, 2);
    EXPECT_EQ(storage.set_value(arith, key, value), ERROR_SUCCESS) << "Value should be set correctly";
    EXPECT_EQ(storage.size, 1) << "Size should be 1";
    EXPECT_EQ(storage.capacity, CuEVM::initial_storage_capacity) << "Capacity should be initial_storage_capacity";
    EXPECT_NE(storage.storage, nullptr) << "Storage should not be nullptr";
    EXPECT_EQ(storage.storage[0].get_key(), 1) << "Key should be set correctly";
    EXPECT_EQ(storage.storage[0].get_value(), 2) << "Value should be set correctly";
    CuEVM::bn_t value2;
    EXPECT_EQ(storage.get_value(arith, key, value2), ERROR_SUCCESS) << "Value should be retrieved correctly";
    EXPECT_EQ(cgbn_compare(arith.env, value, value2), 0) << "Value should be retrieved correctly";
}

TEST_F(ContractStorageTest, AssignmentOperator) {
    CuEVM::contract_storage_t storage2;
    CuEVM::bn_t key, value;
    cgbn_set_ui32(arith.env, key, 1);
    cgbn_set_ui32(arith.env, value, 2);
    storage.set_value(arith, key, value);
    storage2 = storage;
    EXPECT_EQ(storage2.size, 1) << "Size should be 1";
    EXPECT_EQ(storage2.capacity, CuEVM::initial_storage_capacity) << "Capacity should be initial_storage_capacity";
    EXPECT_NE(storage2.storage, nullptr) << "Storage should not be nullptr";
    EXPECT_EQ(storage2.storage[0].get_key(), 1) << "Key should be set correctly";
    EXPECT_EQ(storage2.storage[0].get_value(), 2) << "Value should be set correctly";
}

TEST_F(ContractStorageTest, SetGetValueExistingKey) {
    CuEVM::bn_t key, value;
    cgbn_set_ui32(arith.env, key, 1);
    cgbn_set_ui32(arith.env, value, 2);
    storage.set_value(arith, key, value);
    cgbn_set_ui32(arith.env, value, 3);
    EXPECT_EQ(storage.set_value(arith, key, value), ERROR_SUCCESS) << "Value should be set correctly";
    EXPECT_EQ(storage.size, 1) << "Size should be 1";
    EXPECT_EQ(storage.capacity, CuEVM::initial_storage_capacity) << "Capacity should be initial_storage_capacity";
    EXPECT_NE(storage.storage, nullptr) << "Storage should not be nullptr";
    EXPECT_EQ(storage.storage[0].get_key(), 1) << "Key should be set correctly";
    EXPECT_EQ(storage.storage[0].get_value(), 3) << "Value should be set correctly";
    CuEVM::bn_t value2;
    EXPECT_EQ(storage.get_value(arith, key, value2), ERROR_SUCCESS) << "Value should be retrieved correctly";
    EXPECT_EQ(cgbn_compare(arith.env, value, value2), 0) << "Value should be retrieved correctly";
}

TEST_F(ContractStorageTest, SetGetValueNonExistingKey) {
    CuEVM::bn_t key, value;
    cgbn_set_ui32(arith.env, key, 1);
    cgbn_set_ui32(arith.env, value, 2);
    storage.set_value(arith, key, value);
    cgbn_set_ui32(arith.env, key, 3);
    cgbn_set_ui32(arith.env, value, 4);
    EXPECT_EQ(storage.set_value(arith, key, value), ERROR_SUCCESS) << "Value should be set correctly";
    EXPECT_EQ(storage.size, 2) << "Size should be 2";
    EXPECT_EQ(storage.capacity, CuEVM::initial_storage_capacity) << "Capacity should be initial_storage_capacity";
    EXPECT_NE(storage.storage, nullptr) << "Storage should not be nullptr";
    EXPECT_EQ(storage.storage[0].get_key(), 1) << "Key should be set correctly";
    EXPECT_EQ(storage.storage[0].get_value(), 2) << "Value should be set correctly";
    EXPECT_EQ(storage.storage[1].get_key(), 3) << "Key should be set correctly";
    EXPECT_EQ(storage.storage[1].get_value(), 4) << "Value should be set correctly";
    CuEVM::bn_t value2;
    EXPECT_EQ(storage.get_value(arith, key, value2), ERROR_SUCCESS) << "Value should be retrieved correctly";
    EXPECT_EQ(cgbn_compare(arith.env, value, value2), 0) << "Value should be retrieved correctly";
}

TEST_F(ContractStorageTest, SetGetValueExistingKeyWithArithEnv) {
    CuEVM::bn_t key, value;
    cgbn_set_ui32(arith.env, key, 1);
    cgbn_set_ui32(arith.env, value, 2);
    storage.set_value(arith, key, value);
    cgbn_set_ui32(arith.env, value, 3);
    EXPECT_EQ(storage.set_value(arith, key, value), ERROR_SUCCESS) << "Value should be set correctly";
    EXPECT_EQ(storage.size, 1) << "Size should be 1";
    CuEVM::bn_t value2;
    EXPECT_EQ(storage.get_value(arith, key, value2), ERROR_SUCCESS) << "Value should be retrieved correctly";
    EXPECT_EQ(cgbn_compare(arith.env, value, value2), 0) << "Value should be retrieved correctly";
}

TEST_F(ContractStorageTest, SetGetValueNonExistingKeyWithArithEnv) {
    CuEVM::bn_t key, value;
    cgbn_set_ui32(arith.env, key, 1);
    cgbn_set_ui32(arith.env, value, 2);
    storage.set_value(arith, key, value);
    cgbn_set_ui32(arith.env, key, 3);
    cgbn_set_ui32(arith.env, value, 4);
    EXPECT_EQ(storage.set_value(arith, key, value), ERROR_SUCCESS) << "Value should be set correctly";
    EXPECT_EQ(storage.size, 2) << "Size should be 2";
    CuEVM::bn_t value2;
    EXPECT_EQ(storage.get_value(arith, key, value2), ERROR_SUCCESS) << "Value should be retrieved correctly";
    EXPECT_EQ(cgbn_compare(arith.env, value, value2), 0) << "Value should be retrieved correctly";
    cgbn_set_ui32(arith.env, key, 5);
    EXPECT_EQ(storage.get_value(arith, key, value2), ERROR_STORAGE_KEY_NOT_FOUND) << "Value should not be retrieved";
}

TEST_F(ContractStorageTest, UpdateWithDifferentKeys) {
    CuEVM::bn_t key, value;
    cgbn_set_ui32(arith.env, key, 1);
    cgbn_set_ui32(arith.env, value, 2);
    storage.set_value(arith, key, value);
    // create a new contract stroage
    CuEVM::contract_storage_t storage2;
    cgbn_set_ui32(arith.env, key, 2);
    cgbn_set_ui32(arith.env, value, 3);
    storage2.set_value(arith, key, value);
    storage.update(arith, storage2);
    EXPECT_EQ(storage.size, 2) << "Size should be 2";
    CuEVM::bn_t value2;
    cgbn_set_ui32(arith.env, key, 1);
    cgbn_set_ui32(arith.env, value, 2);
    EXPECT_EQ(storage.get_value(arith, key, value2), ERROR_SUCCESS) << "Value should be retrieved correctly";
    EXPECT_EQ(cgbn_compare(arith.env, value, value2), 0) << "Value should be retrieved correctly";
    cgbn_set_ui32(arith.env, key, 2);
    cgbn_set_ui32(arith.env, value, 3);
    EXPECT_EQ(storage.get_value(arith, key, value2), ERROR_SUCCESS) << "Value should be retrieved correctly";
    EXPECT_EQ(cgbn_compare(arith.env, value, value2), 0) << "Value should be retrieved correctly";
}

TEST_F(ContractStorageTest, UpdateWithSameKeys) {
    CuEVM::bn_t key, value;
    cgbn_set_ui32(arith.env, key, 1);
    cgbn_set_ui32(arith.env, value, 2);
    storage.set_value(arith, key, value);
    // create a new contract stroage
    CuEVM::contract_storage_t storage2;
    cgbn_set_ui32(arith.env, key, 1);
    cgbn_set_ui32(arith.env, value, 3);
    storage2.set_value(arith, key, value);
    storage.update(arith, storage2);
    EXPECT_EQ(storage.size, 1) << "Size should be 1";
    CuEVM::bn_t value2;
    cgbn_set_ui32(arith.env, key, 1);
    cgbn_set_ui32(arith.env, value, 3);
    EXPECT_EQ(storage.get_value(arith, key, value2), ERROR_SUCCESS) << "Value should be retrieved correctly";
    EXPECT_EQ(cgbn_compare(arith.env, value, value2), 0) << "Value should be retrieved correctly";
}

TEST_F(ContractStorageTest, UpdateEmptyStorage) {
    CuEVM::bn_t key, value;
    cgbn_set_ui32(arith.env, key, 1);
    cgbn_set_ui32(arith.env, value, 2);
    // create a new contract stroage
    CuEVM::contract_storage_t storage2;
    storage2.set_value(arith, key, value);
    storage.update(arith, storage2);
    EXPECT_EQ(storage.size, 1) << "Size should be 1";
    CuEVM::bn_t value2;
    cgbn_set_ui32(arith.env, key, 1);
    cgbn_set_ui32(arith.env, value, 2);
    EXPECT_EQ(storage.get_value(arith, key, value2), ERROR_SUCCESS) << "Value should be retrieved correctly";
    EXPECT_EQ(cgbn_compare(arith.env, value, value2), 0) << "Value should be retrieved correctly";
}

TEST_F(ContractStorageTest, UpdateEmptyStorageWithEmptyStorage) {
    // create a new contract stroage
    CuEVM::contract_storage_t storage2;
    storage.update(arith, storage2);
    EXPECT_EQ(storage.size, 0) << "Size should be 0";
}

TEST_F(ContractStorageTest, UpdateWithEmptyStorage) {
    CuEVM::bn_t key, value;
    cgbn_set_ui32(arith.env, key, 1);
    cgbn_set_ui32(arith.env, value, 2);
    storage.set_value(arith, key, value);
    // create a new contract stroage
    CuEVM::contract_storage_t storage2;
    storage.update(arith, storage2);
    EXPECT_EQ(storage.size, 1) << "Size should be 1";
    CuEVM::bn_t value2;
    cgbn_set_ui32(arith.env, key, 1);
    cgbn_set_ui32(arith.env, value, 2);
    EXPECT_EQ(storage.get_value(arith, key, value2), ERROR_SUCCESS) << "Value should be retrieved correctly";
    EXPECT_EQ(cgbn_compare(arith.env, value, value2), 0) << "Value should be retrieved correctly";
}

__global__ void test_contract_storage_kernel(CuEVM::contract_storage_t* gpuStorage, uint32_t count, uint32_t* result) {
    int32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / CuEVM::cgbn_tpi;
    if (instance >= count) return;
    result[instance] = ERROR_SUCCESS;

    CuEVM::ArithEnv arith(cgbn_no_checks);
    CuEVM::bn_t key, value, retrieved_value;
    cgbn_set_ui32(arith.env, key, 1);
    cgbn_set_ui32(arith.env, value, 42);

    // Test set and get
    result[instance] |= gpuStorage[instance].set_value(arith, key, value);
    result[instance] |= gpuStorage[instance].get_value(arith, key, retrieved_value);
    result[instance] |= cgbn_get_ui32(arith.env, retrieved_value) == 42 ? ERROR_SUCCESS : __LINE__;
    result[instance] |= gpuStorage[instance].size == 1 ? ERROR_SUCCESS : __LINE__;
    result[instance] |= gpuStorage[instance].capacity == CuEVM::initial_storage_capacity ? ERROR_SUCCESS : __LINE__;
    result[instance] |= gpuStorage[instance].storage != nullptr ? ERROR_SUCCESS : __LINE__;
    result[instance] |= gpuStorage[instance].storage[0].get_key() == 1 ? ERROR_SUCCESS : __LINE__;

    // Test update
    cgbn_set_ui32(arith.env, value, 84);
    result[instance] |= gpuStorage[instance].set_value(arith, key, value);
    result[instance] |= gpuStorage[instance].get_value(arith, key, retrieved_value);
    result[instance] |= cgbn_get_ui32(arith.env, retrieved_value) == 84 ? ERROR_SUCCESS : __LINE__;
    result[instance] |= gpuStorage[instance].size == 1 ? ERROR_SUCCESS : __LINE__;
    result[instance] |= gpuStorage[instance].capacity == CuEVM::initial_storage_capacity ? ERROR_SUCCESS : __LINE__;
    result[instance] |= gpuStorage[instance].storage != nullptr ? ERROR_SUCCESS : __LINE__;
    result[instance] |= gpuStorage[instance].storage[0].get_key() == 1 ? ERROR_SUCCESS : __LINE__;

    // test contract state update
    CuEVM::contract_storage_t storage2;
    cgbn_set_ui32(arith.env, key, 2);
    cgbn_set_ui32(arith.env, value, 3);
    storage2.set_value(arith, key, value);
    gpuStorage[instance].update(arith, storage2);
    result[instance] |= gpuStorage[instance].size == 2 ? ERROR_SUCCESS : __LINE__;
    result[instance] |= gpuStorage[instance].capacity == CuEVM::initial_storage_capacity ? ERROR_SUCCESS : __LINE__;
    result[instance] |= gpuStorage[instance].storage != nullptr ? ERROR_SUCCESS : __LINE__;
    result[instance] |= gpuStorage[instance].storage[0].get_key() == 1 ? ERROR_SUCCESS : __LINE__;
    result[instance] |= gpuStorage[instance].storage[1].get_key() == 2 ? ERROR_SUCCESS : __LINE__;
    result[instance] |= gpuStorage[instance].storage[1].get_value() == 3 ? ERROR_SUCCESS : __LINE__;

    // test contract state update with same key
    cgbn_set_ui32(arith.env, value, 4);
    cgbn_set_ui32(arith.env, key, 1);
    storage2.set_value(arith, key, value);
    gpuStorage[instance].update(arith, storage2);
    result[instance] |= gpuStorage[instance].size == 2 ? ERROR_SUCCESS : __LINE__;
    result[instance] |= gpuStorage[instance].capacity == CuEVM::initial_storage_capacity ? ERROR_SUCCESS : __LINE__;
    result[instance] |= gpuStorage[instance].storage != nullptr ? ERROR_SUCCESS : __LINE__;
    result[instance] |= gpuStorage[instance].storage[0].get_key() == 1 ? ERROR_SUCCESS : __LINE__;
    result[instance] |= gpuStorage[instance].storage[0].get_value() == 4 ? ERROR_SUCCESS : __LINE__;
    result[instance] |= gpuStorage[instance].storage[1].get_key() == 2 ? ERROR_SUCCESS : __LINE__;
    result[instance] |= gpuStorage[instance].storage[1].get_value() == 3 ? ERROR_SUCCESS : __LINE__;
}

#ifdef GPU
// Test contract storage operations on GPU
TEST_F(ContractStorageTest, StorageOperationsGPU) {
    CuEVM::contract_storage_t* cpuStorage = CuEVM::contract_storage_t::get_cpu(2);
    CUDA_CHECK(cudaDeviceReset());
    CuEVM::contract_storage_t* gpuStorage = CuEVM::contract_storage_t::get_gpu_from_cpu(cpuStorage, 2);
    uint32_t* d_result;
    cudaMalloc(&d_result, 2 * sizeof(uint32_t));
    test_contract_storage_kernel<<<2, CuEVM::cgbn_tpi>>>(gpuStorage, 2, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CuEVM::contract_storage_t* results = CuEVM::contract_storage_t::get_cpu_from_gpu(gpuStorage, 2);

    for (int i = 0; i < 2; i++) {
        EXPECT_EQ(results[i].size, 2);
        EXPECT_EQ(results[i].capacity, 2);
        EXPECT_NE(results[i].storage, nullptr);
        EXPECT_EQ(results[i].storage[0].get_key(), 1);
        EXPECT_EQ(results[i].storage[0].get_value(), 4);
        EXPECT_EQ(results[i].storage[1].get_key(), 2);
        EXPECT_EQ(results[i].storage[1].get_value(), 3);
    }

    uint32_t* h_result;
    h_result = (uint32_t*)malloc(2 * sizeof(uint32_t));
    CUDA_CHECK(cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 2; i++) {
        EXPECT_EQ(h_result[i], ERROR_SUCCESS);
    }
    free(h_result);
    CUDA_CHECK(cudaFree(d_result));
    CuEVM::contract_storage_t::cpu_free(cpuStorage, 2);
    CuEVM::contract_storage_t::cpu_free(results, 2);
    CUDA_CHECK(cudaDeviceReset());
}
#endif