#include <gtest/gtest.h>

#include <CuEVM/core/byte_array.cuh>
#include <CuEVM/state/account.cuh>
#include <CuEVM/state/account_flags.cuh>
#include <CuEVM/state/contract_storage.cuh>
#include <CuEVM/utils/arith.cuh>
#include <CuEVM/utils/error_codes.cuh>

using namespace CuEVM;

class AccountTest : public ::testing::Test {
   protected:
    CuEVM::ArithEnv arith;
    CuEVM::account_t empty_account;
    CuEVM::account_t fill_account;

    AccountTest() : arith(cgbn_no_checks), empty_account(), fill_account() {
        fill_account.address = 0x12345678;
        fill_account.balance = 1000;
        fill_account.nonce = 1;
        fill_account.byte_code.grow(3, 1);
        fill_account.byte_code.data[0] = 0xFF;
        fill_account.byte_code.data[1] = 0xFE;
        fill_account.byte_code.data[2] = 0xFD;
        CuEVM::bn_t key, value;
        cgbn_set_ui32(arith.env, key, 1);
        cgbn_set_ui32(arith.env, value, 42);
        fill_account.storage.set_value(arith, key, value);
    }

    void SetUp() override {
        // Initialize any necessary resources here
    }

    void TearDown() override {
        // Clean up any resources here
    }
};

TEST_F(AccountTest, DefaultConstructor) {
    // EXPECT_EQ(account.address, 0U);
    // EXPECT_EQ(account.balance, 0U);
    // EXPECT_EQ(account.nonce, 0U);
    EXPECT_EQ(empty_account.byte_code.size, 0U);
    EXPECT_EQ(empty_account.byte_code.data, nullptr);
    EXPECT_EQ(empty_account.storage.size, 0U);
    EXPECT_EQ(empty_account.storage.capacity, 0U);
    EXPECT_EQ(empty_account.storage.storage, nullptr);
}

TEST_F(AccountTest, CopyConstructor) {
    CuEVM::account_t account2(fill_account);
    EXPECT_EQ(account2.address, fill_account.address);
    EXPECT_EQ(account2.balance, fill_account.balance);
    EXPECT_EQ(account2.nonce, fill_account.nonce);
    EXPECT_EQ(account2.byte_code.size, fill_account.byte_code.size);
    EXPECT_EQ(account2.byte_code.data[0], fill_account.byte_code.data[0]);
    EXPECT_EQ(account2.byte_code.data[1], fill_account.byte_code.data[1]);
    EXPECT_EQ(account2.byte_code.data[2], fill_account.byte_code.data[2]);
    EXPECT_EQ(account2.storage.size, fill_account.storage.size);
    EXPECT_EQ(account2.storage.storage[0].key, fill_account.storage.storage[0].key);
    EXPECT_EQ(account2.storage.storage[0].value, fill_account.storage.storage[0].value);
}

TEST_F(AccountTest, CopyConstructorWithFlags) {
    account_flags_t flags = ACCOUNT_BYTE_CODE_FLAG;
    account_t account2(fill_account, flags);
    EXPECT_EQ(account2.address, fill_account.address);
    EXPECT_EQ(account2.balance, fill_account.balance);
    EXPECT_EQ(account2.nonce, fill_account.nonce);
    EXPECT_EQ(account2.byte_code.size, fill_account.byte_code.size);
    EXPECT_EQ(account2.byte_code.data[0], fill_account.byte_code.data[0]);
    EXPECT_EQ(account2.byte_code.data[1], fill_account.byte_code.data[1]);
    EXPECT_EQ(account2.byte_code.data[2], fill_account.byte_code.data[2]);
    EXPECT_EQ(account2.storage.size, 0U);
}

TEST_F(AccountTest, AssignmentOperator) {
    account_t account2;
    account2 = fill_account;
    EXPECT_EQ(account2.address, fill_account.address);
    EXPECT_EQ(account2.balance, fill_account.balance);
    EXPECT_EQ(account2.nonce, fill_account.nonce);
    EXPECT_EQ(account2.byte_code.size, fill_account.byte_code.size);
    EXPECT_EQ(account2.byte_code.data[0], fill_account.byte_code.data[0]);
    EXPECT_EQ(account2.byte_code.data[1], fill_account.byte_code.data[1]);
    EXPECT_EQ(account2.byte_code.data[2], fill_account.byte_code.data[2]);
    EXPECT_EQ(account2.storage.size, fill_account.storage.size);
    EXPECT_EQ(account2.storage.storage[0].key, fill_account.storage.storage[0].key);
    EXPECT_EQ(account2.storage.storage[0].value, fill_account.storage.storage[0].value);
}

TEST_F(AccountTest, SetAndGetStorageValue) {
    bn_t key, value, retrieved_value;
    cgbn_set_ui32(arith.env, key, 1);
    cgbn_set_ui32(arith.env, value, 42);

    EXPECT_EQ(empty_account.set_storage_value(arith, key, value), ERROR_SUCCESS);
    EXPECT_EQ(empty_account.get_storage_value(arith, key, retrieved_value), ERROR_SUCCESS);
    EXPECT_EQ(cgbn_compare(arith.env, value, retrieved_value), 0);
    EXPECT_EQ(cgbn_get_ui32(arith.env, retrieved_value), 42);
}

TEST_F(AccountTest, SetAndGetAddress) {
    bn_t address;
    cgbn_set_ui32(arith.env, address, 0x12345678);

    empty_account.set_address(arith, address);
    bn_t retrieved_address;
    empty_account.get_address(arith, retrieved_address);
    EXPECT_EQ(cgbn_get_ui32(arith.env, retrieved_address), 0x12345678);
}

TEST_F(AccountTest, SetAndGetBalance) {
    bn_t balance;
    cgbn_set_ui32(arith.env, balance, 1000);

    empty_account.set_balance(arith, balance);
    bn_t retrieved_balance;
    empty_account.get_balance(arith, retrieved_balance);
    EXPECT_EQ(cgbn_get_ui32(arith.env, retrieved_balance), 1000);
}

TEST_F(AccountTest, SetAndGetNonce) {
    bn_t nonce;
    cgbn_set_ui32(arith.env, nonce, 1);

    empty_account.set_nonce(arith, nonce);
    bn_t retrieved_nonce;
    empty_account.get_nonce(arith, retrieved_nonce);
    EXPECT_EQ(cgbn_get_ui32(arith.env, retrieved_nonce), 1);
}

TEST_F(AccountTest, SetAndGetByteCode) {
    byte_array_t byte_code(0U);
    byte_code.grow(3, 1);
    byte_code.data[0] = 0xFF;
    byte_code.data[1] = 0xFE;
    byte_code.data[2] = 0xFD;

    empty_account.set_byte_code(byte_code);
    byte_array_t retrieved_byte_code = empty_account.get_byte_code();
    EXPECT_EQ(retrieved_byte_code.size, 3U);
    EXPECT_EQ(retrieved_byte_code.data[0], 0xFF);
    EXPECT_EQ(retrieved_byte_code.data[1], 0xFE);
    EXPECT_EQ(retrieved_byte_code.data[2], 0xFD);
}

TEST_F(AccountTest, HasAddress) {
    bn_t address;
    cgbn_set_ui32(arith.env, address, 0x12345678);

    empty_account.set_address(arith, address);
    EXPECT_EQ(empty_account.has_address(arith, address), 1);

    evm_word_t address_word(0x12345678);
    EXPECT_EQ(empty_account.has_address(arith, address_word), 1);
}

TEST_F(AccountTest, IsEmpty) {
    empty_account.clear();
    EXPECT_EQ(empty_account.is_empty(), ERROR_SUCCESS);
    bn_t address;
    cgbn_set_ui32(arith.env, address, 0x12345678);
    empty_account.set_address(arith, address);
    EXPECT_EQ(empty_account.is_empty(), ERROR_SUCCESS);
    bn_t nonce;
    cgbn_set_ui32(arith.env, nonce, 1);

    empty_account.set_nonce(arith, nonce);
    EXPECT_EQ(empty_account.is_empty(), ERROR_ACCOUNT_NOT_EMPTY);
}

TEST_F(AccountTest, IsContract) {
    EXPECT_EQ(empty_account.is_contract(), 0);

    byte_array_t byte_code(0U);
    byte_code.grow(3, 1);
    byte_code.data[0] = 0xFF;
    byte_code.data[1] = 0xFE;
    byte_code.data[2] = 0xFD;
    empty_account.set_byte_code(byte_code);
    EXPECT_EQ(empty_account.is_contract(), 1);
}

TEST_F(AccountTest, Clear) {
    bn_t address;
    cgbn_set_ui32(arith.env, address, 0x12345678);
    empty_account.set_address(arith, address);
    byte_array_t byte_code(0U);
    byte_code.grow(3, 1);
    byte_code.data[0] = 0xFF;
    byte_code.data[1] = 0xFE;
    byte_code.data[2] = 0xFD;
    empty_account.set_byte_code(byte_code);
    bn_t key, value;
    cgbn_set_ui32(arith.env, key, 1);
    cgbn_set_ui32(arith.env, value, 42);

    EXPECT_EQ(empty_account.set_storage_value(arith, key, value), ERROR_SUCCESS);
    uint8_t* data_ptr = empty_account.byte_code.data;
    storage_element_t* storage_ptr = empty_account.storage.storage;
    empty_account.clear();
    EXPECT_EQ(empty_account.address, 0U);
    EXPECT_EQ(empty_account.balance, 0U);
    EXPECT_EQ(empty_account.nonce, 0U);
    EXPECT_EQ(empty_account.byte_code.size, 0U);
    EXPECT_EQ(empty_account.byte_code.data, nullptr);
    EXPECT_EQ(empty_account.storage.size, 0U);
    EXPECT_EQ(empty_account.storage.capacity, 0U);
    EXPECT_EQ(empty_account.storage.storage, nullptr);
    delete[] data_ptr;
    delete[] storage_ptr;
}

TEST_F(AccountTest, Update) {
    account_t account2;
    account_flags_t flags = ACCOUNT_ADDRESS_FLAG | ACCOUNT_BALANCE_FLAG | ACCOUNT_NONCE_FLAG;
    account2.update(arith, fill_account, flags);

    EXPECT_EQ(account2.address, fill_account.address);
    EXPECT_EQ(account2.balance, fill_account.balance);
    EXPECT_EQ(account2.nonce, fill_account.nonce);
    EXPECT_EQ(account2.byte_code.size, 0U);
    EXPECT_EQ(account2.byte_code.data, nullptr);
    EXPECT_EQ(account2.storage.size, 0U);
    EXPECT_EQ(account2.storage.capacity, 0U);
    EXPECT_EQ(account2.storage.storage, nullptr);
}

TEST_F(AccountTest, FromJson) {
    // Example JSON object
    const char* json_str = R"(
    {
        "test": {
            "pre" : {
                "0x12345678": {
                    "balance": "0x1000",
                    "nonce": "0x01",
                    "code": "0xFF",
                    "storage": {
                        "0x01": "0x42"
                    }
                }
            }
        }
    })";
    cJSON* json = cJSON_Parse(json_str);
    cJSON* test = cJSON_GetArrayItem(json, 0);
    cJSON* pre = cJSON_GetObjectItemCaseSensitive(test, "pre");
    cJSON* account_json = cJSON_GetArrayItem(pre, 0);

    empty_account.from_json(account_json);

    bn_t address;
    empty_account.get_address(arith, address);
    EXPECT_EQ(cgbn_get_ui32(arith.env, address), 0x12345678);

    bn_t balance;
    empty_account.get_balance(arith, balance);
    EXPECT_EQ(cgbn_get_ui32(arith.env, balance), 0x1000);

    bn_t nonce;
    empty_account.get_nonce(arith, nonce);
    EXPECT_EQ(cgbn_get_ui32(arith.env, nonce), 1);

    EXPECT_EQ(empty_account.byte_code.size, 1U);
    EXPECT_EQ(empty_account.byte_code[0], 0xFF);

    cJSON_Delete(json);
}

TEST_F(AccountTest, ToJson) {
    cJSON* json = fill_account.to_json();
    char* json_str = cJSON_Print(json);

    // Expected JSON string
    const char* expected_json_str =
        "{\n\t\"balance\":\t\"0x00000000000000000000000000000000000000000000000000000000000003e8\",\n\t\"nonce\":"
        "\t\"0x0000000000000000000000000000000000000000000000000000000000000001\",\n\t\"code\":\t\"0xfffefd\","
        "\n\t\"storage\":\t{\n\t\t\"0x1\":\t\"0x2a\"\n\t}\n}";

    EXPECT_STREQ(json_str, expected_json_str);

    cJSON_Delete(json);
    free(json_str);
}

TEST_F(AccountTest, Print) {
    // Redirect stdout to a string stream
    testing::internal::CaptureStdout();

    fill_account.print();
    std::string output = testing::internal::GetCapturedStdout();

    // Expected output
    ASSERT_EQ(output,
              "Account:\n00000000 00000000 00000000 00000000 00000000 00000000 00000000 12345678 \nBalance: 00000000 "
              "00000000 00000000 00000000 00000000 00000000 00000000 000003e8 \nNonce: 00000000 00000000 00000000 "
              "00000000 00000000 00000000 00000000 00000001 \nByte code: size: 3\ndata: fffefd\nStorage: \nStorage "
              "size: 1\nElement 0:\nKey: 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000001 "
              "\nValue: 00000000 00000000 00000000 00000000 00000000 00000000 00000000 0000002a \n");
}

__global__ void test_account_kernel(CuEVM::account_t* gpuAccounts, uint32_t count, uint32_t* result) {
    int32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / CuEVM::cgbn_tpi;
    if (instance >= count) return;
    result[instance] = ERROR_SUCCESS;

    CuEVM::ArithEnv arith(cgbn_no_checks);
    CuEVM::bn_t key, value, retrieved_value;
    cgbn_set_ui32(arith.env, key, 1);
    cgbn_set_ui32(arith.env, value, 42);

    // Test Default Constructor
    CuEVM::account_t empty_account;
    empty_account.free();
    result[instance] |= (empty_account.byte_code.size == 0U) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_account.byte_code.data == nullptr) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_account.storage.size == 0U) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_account.storage.capacity == 0U) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_account.storage.storage == nullptr) ? ERROR_SUCCESS : __LINE__;

    // Test Copy Constructor
    CuEVM::account_t fill_account;
    fill_account.address = 0x12345678;
    fill_account.balance = 1000;
    fill_account.nonce = instance;
    fill_account.byte_code.grow(3, 1);
    fill_account.byte_code.data[0] = 0xFF;
    fill_account.byte_code.data[1] = 0xFE;
    fill_account.byte_code.data[2] = 0xFD;
    fill_account.storage.set_value(arith, key, value);

    CuEVM::account_t account2(fill_account);
    result[instance] |= (account2.address == fill_account.address) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (account2.balance == fill_account.balance) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (account2.nonce == fill_account.nonce) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (account2.byte_code.size == fill_account.byte_code.size) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (account2.byte_code.data[0] == fill_account.byte_code.data[0]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (account2.byte_code.data[1] == fill_account.byte_code.data[1]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (account2.byte_code.data[2] == fill_account.byte_code.data[2]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (account2.storage.size == fill_account.storage.size) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (account2.storage.storage[0].key == fill_account.storage.storage[0].key) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (account2.storage.storage[0].value == fill_account.storage.storage[0].value) ? ERROR_SUCCESS : __LINE__;

    // Test Assignment Operator
    CuEVM::account_t account3;
    account3 = fill_account;
    result[instance] |= (account3.address == fill_account.address) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (account3.balance == fill_account.balance) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (account3.nonce == fill_account.nonce) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (account3.byte_code.size == fill_account.byte_code.size) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (account3.byte_code.data[0] == fill_account.byte_code.data[0]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (account3.byte_code.data[1] == fill_account.byte_code.data[1]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (account3.byte_code.data[2] == fill_account.byte_code.data[2]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (account3.storage.size == fill_account.storage.size) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (account3.storage.storage[0].key == fill_account.storage.storage[0].key) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (account3.storage.storage[0].value == fill_account.storage.storage[0].value) ? ERROR_SUCCESS : __LINE__;

    // Test Set and Get Storage Value
    result[instance] |=
        (empty_account.set_storage_value(arith, key, value) == ERROR_SUCCESS) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (empty_account.get_storage_value(arith, key, retrieved_value) == ERROR_SUCCESS) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (cgbn_compare(arith.env, value, retrieved_value) == 0) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (cgbn_get_ui32(arith.env, retrieved_value) == 42) ? ERROR_SUCCESS : __LINE__;
    empty_account.free();

    // Test Set and Get Address
    CuEVM::bn_t address;
    cgbn_set_ui32(arith.env, address, 0x12345678);
    empty_account.set_address(arith, address);
    CuEVM::bn_t retrieved_address;
    empty_account.get_address(arith, retrieved_address);
    result[instance] |= (cgbn_get_ui32(arith.env, retrieved_address) == 0x12345678) ? ERROR_SUCCESS : __LINE__;
    empty_account.free();

    // Test Set and Get Balance
    CuEVM::bn_t balance;
    cgbn_set_ui32(arith.env, balance, 1000);
    empty_account.set_balance(arith, balance);
    CuEVM::bn_t retrieved_balance;
    empty_account.get_balance(arith, retrieved_balance);
    result[instance] |= (cgbn_get_ui32(arith.env, retrieved_balance) == 1000) ? ERROR_SUCCESS : __LINE__;
    empty_account.free();

    // Test Set and Get Nonce
    CuEVM::bn_t nonce;
    cgbn_set_ui32(arith.env, nonce, 1);
    empty_account.set_nonce(arith, nonce);
    CuEVM::bn_t retrieved_nonce;
    empty_account.get_nonce(arith, retrieved_nonce);
    result[instance] |= (cgbn_get_ui32(arith.env, retrieved_nonce) == 1) ? ERROR_SUCCESS : __LINE__;
    empty_account.free();

    // Test Set and Get Byte Code
    CuEVM::byte_array_t byte_code(0U);
    byte_code.grow(3, 1);
    byte_code.data[0] = 0xFF;
    byte_code.data[1] = 0xFE;
    byte_code.data[2] = 0xFD;
    empty_account.set_byte_code(byte_code);
    CuEVM::byte_array_t retrieved_byte_code = empty_account.get_byte_code();
    result[instance] |= (retrieved_byte_code.size == 3U) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_byte_code.data[0] == 0xFF) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_byte_code.data[1] == 0xFE) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_byte_code.data[2] == 0xFD) ? ERROR_SUCCESS : __LINE__;
    empty_account.free();

    // Test Has Address
    cgbn_set_ui32(arith.env, address, 0x12345678);
    empty_account.set_address(arith, address);
    result[instance] |= (empty_account.has_address(arith, address) == 1) ? ERROR_SUCCESS : __LINE__;
    CuEVM::evm_word_t address_word(0x12345678);
    result[instance] |= (empty_account.has_address(arith, address_word) == 1) ? ERROR_SUCCESS : __LINE__;
    empty_account.free();

    // Test Is Empty
    empty_account.clear();
    result[instance] |= (empty_account.is_empty() == ERROR_SUCCESS) ? ERROR_SUCCESS : __LINE__;
    empty_account.set_address(arith, address);
    result[instance] |= (empty_account.is_empty() == ERROR_SUCCESS) ? ERROR_SUCCESS : __LINE__;
    empty_account.set_nonce(arith, nonce);
    result[instance] |= (empty_account.is_empty() == ERROR_ACCOUNT_NOT_EMPTY) ? ERROR_SUCCESS : __LINE__;

    // Test Is Contract
    result[instance] |= (empty_account.is_contract() == 0) ? ERROR_SUCCESS : __LINE__;
    empty_account.set_byte_code(byte_code);
    result[instance] |= (empty_account.is_contract() == 1) ? ERROR_SUCCESS : __LINE__;
    empty_account.free();

    // Test Clear
    empty_account.set_address(arith, address);
    empty_account.set_byte_code(byte_code);
    result[instance] |=
        (empty_account.set_storage_value(arith, key, value) == ERROR_SUCCESS) ? ERROR_SUCCESS : __LINE__;
    uint8_t* data_ptr = empty_account.byte_code.data;
    storage_element_t* storage_ptr = empty_account.storage.storage;
    empty_account.clear();
    result[instance] |= (empty_account.address == 0U) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_account.balance == 0U) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_account.nonce == 0U) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_account.byte_code.size == 0U) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_account.byte_code.data == nullptr) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_account.storage.size == 0U) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_account.storage.capacity == 0U) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_account.storage.storage == nullptr) ? ERROR_SUCCESS : __LINE__;
    __ONE_GPU_THREAD_BEGIN__
    delete[] data_ptr;
    delete[] storage_ptr;
    __ONE_GPU_THREAD_END__

    // Test Update
    CuEVM::account_t account4;
    CuEVM::account_flags_t flags = ACCOUNT_ADDRESS_FLAG | ACCOUNT_BALANCE_FLAG | ACCOUNT_NONCE_FLAG;
    account4.update(arith, fill_account, flags);
    result[instance] |= (account4.address == fill_account.address) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (account4.balance == fill_account.balance) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (account4.nonce == fill_account.nonce) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (account4.byte_code.size == 0U) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (account4.byte_code.data == nullptr) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (account4.storage.size == 0U) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (account4.storage.capacity == 0U) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (account4.storage.storage == nullptr) ? ERROR_SUCCESS : __LINE__;

    gpuAccounts[instance] = fill_account;

    // verify gpuAccounts
    result[instance] |= (gpuAccounts[instance].address == fill_account.address) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (gpuAccounts[instance].balance == fill_account.balance) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (gpuAccounts[instance].nonce == fill_account.nonce) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (gpuAccounts[instance].byte_code.size == fill_account.byte_code.size) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (gpuAccounts[instance].byte_code.data[0] == fill_account.byte_code.data[0]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (gpuAccounts[instance].byte_code.data[1] == fill_account.byte_code.data[1]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (gpuAccounts[instance].byte_code.data[2] == fill_account.byte_code.data[2]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (gpuAccounts[instance].storage.size == fill_account.storage.size) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (gpuAccounts[instance].storage.storage[0].key == fill_account.storage.storage[0].key)
                            ? ERROR_SUCCESS
                            : __LINE__;
    result[instance] |= (gpuAccounts[instance].storage.storage[0].value == fill_account.storage.storage[0].value)
                            ? ERROR_SUCCESS
                            : __LINE__;
}

#ifdef GPU
// Test account operations on GPU
TEST_F(AccountTest, AccountOperationsGPU) {
    CuEVM::account_t* cpuAccounts = CuEVM::account_t::get_cpu(2);
    CUDA_CHECK(cudaDeviceReset());
    CuEVM::account_t* gpuAccounts = CuEVM::account_t::get_gpu_from_cpu(cpuAccounts, 2);
    uint32_t* d_result;
    cudaMalloc(&d_result, 2 * sizeof(uint32_t));
    test_account_kernel<<<2, CuEVM::cgbn_tpi>>>(gpuAccounts, 2, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CuEVM::account_t* results = CuEVM::account_t::get_cpu_from_gpu(gpuAccounts, 2);

    uint32_t* h_result;
    h_result = (uint32_t*)malloc(2 * sizeof(uint32_t));
    CUDA_CHECK(cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 2; i++) {
        EXPECT_EQ(h_result[i], ERROR_SUCCESS);
    }
    free(h_result);
    CUDA_CHECK(cudaFree(d_result));
    for (int i = 0; i < 2; i++) {
        EXPECT_EQ(results[i].address, 0x12345678);
        EXPECT_EQ(results[i].balance, 1000);
        EXPECT_EQ(results[i].nonce, i);
        EXPECT_EQ(results[i].byte_code.size, 3U);
        EXPECT_EQ(results[i].byte_code.data[0], 0xFF);
        EXPECT_EQ(results[i].byte_code.data[1], 0xFE);
        EXPECT_EQ(results[i].byte_code.data[2], 0xFD);
        EXPECT_EQ(results[i].storage.size, 1U);
        EXPECT_EQ(results[i].storage.storage[0].key, 1);
        EXPECT_EQ(results[i].storage.storage[0].value, 42);
    }
    CuEVM::account_t::free_cpu(cpuAccounts, 2);
    CuEVM::account_t::free_cpu(results, 2);
    CUDA_CHECK(cudaDeviceReset());
}
#endif