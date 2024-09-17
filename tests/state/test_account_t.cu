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
    EXPECT_EQ(empty_account.is_empty(), 1);
    bn_t address;
    cgbn_set_ui32(arith.env, address, 0x12345678);
    empty_account.set_address(arith, address);
    EXPECT_EQ(empty_account.is_empty(), 1);
    bn_t nonce;
    cgbn_set_ui32(arith.env, nonce, 1);

    empty_account.set_nonce(arith, nonce);
    EXPECT_EQ(empty_account.is_empty(), 0);
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
    uint8_t *data_ptr = empty_account.byte_code.data;
    storage_element_t *storage_ptr = empty_account.storage.storage;
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

// TEST_F(AccountTest, Update) {
//     account_t account1;
//     account1.address = evm_word_t(0x12345678);
//     account1.balance = evm_word_t(1000);
//     account1.nonce = evm_word_t(1);
//     account1.byte_code = byte_array_t(10, 0xFF);

//     account_t account2;
//     ArithEnv arith(cgbn_no_checks);
//     account_flags_t flags = ACCOUNT_ADDRESS_FLAG | ACCOUNT_BALANCE_FLAG;
//     account2.update(arith, account1, flags);

//     EXPECT_EQ(account2.address, account1.address);
//     EXPECT_EQ(account2.balance, account1.balance);
//     EXPECT_TRUE(account2.nonce.is_zero());
//     EXPECT_EQ(account2.byte_code.size(), 0U);
//     EXPECT_TRUE(account2.storage.is_empty());
// }

// TEST_F(AccountTest, FromJson) {
//     // Example JSON object
//     const char* json_str = R"({
//         "address": "0x12345678",
//         "balance": "1000",
//         "nonce": "1",
//         "byte_code": "0xFF"
//     })";
//     cJSON* json = cJSON_Parse(json_str);

//     account_t account;
//     account.from_json(json);

//     ArithEnv arith(cgbn_no_checks);
//     bn_t address;
//     account.get_address(arith, address);
//     EXPECT_EQ(cgbn_get_ui32(arith.env, address), 0x12345678);

//     bn_t balance;
//     account.get_balance(arith, balance);
//     EXPECT_EQ(cgbn_get_ui32(arith.env, balance), 1000);

//     bn_t nonce;
//     account.get_nonce(arith, nonce);
//     EXPECT_EQ(cgbn_get_ui32(arith.env, nonce), 1);

//     byte_array_t byte_code = account.get_byte_code();
//     EXPECT_EQ(byte_code.size(), 1U);
//     EXPECT_EQ(byte_code[0], 0xFF);

//     cJSON_Delete(json);
// }

// TEST_F(AccountTest, ToJson) {
//     account_t account;
//     account.address = evm_word_t(0x12345678);
//     account.balance = evm_word_t(1000);
//     account.nonce = evm_word_t(1);
//     account.byte_code = byte_array_t(1, 0xFF);

//     cJSON* json = account.to_json();
//     char* json_str = cJSON_Print(json);

//     // Expected JSON string
//     const char* expected_json_str = R"({
//         "address": "0x12345678",
//         "balance": "1000",
//         "nonce": "1",
//         "byte_code": "0xFF"
//     })";

//     EXPECT_STREQ(json_str, expected_json_str);

//     cJSON_Delete(json);
//     free(json_str);
// }

// TEST_F(AccountTest, Print) {
//     account_t account;
//     account.address = evm_word_t(0x12345678);
//     account.balance = evm_word_t(1000);
//     account.nonce = evm_word_t(1);
//     account.byte_code = byte_array_t(1, 0xFF);

//     // Redirect stdout to a string stream
//     std::stringstream buffer;
//     std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());

//     account.print();

//     // Restore stdout
//     std::cout.rdbuf(old);

//     // Expected output
//     std::string expected_output = "Address: 0x12345678\nBalance: 1000\nNonce: 1\nByte Code: 0xFF\n";

//     EXPECT_EQ(buffer.str(), expected_output);
// }