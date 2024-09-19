#include <gtest/gtest.h>

#include <CuEVM/state/account.cuh>
#include <CuEVM/state/account_flags.cuh>
#include <CuEVM/state/state_access.cuh>
#include <CuEVM/utils/arith.cuh>
#include <CuEVM/utils/error_codes.cuh>

using namespace CuEVM;

class StateAccessTest : public ::testing::Test {
   protected:
    CuEVM::ArithEnv arith;
    CuEVM::state_access_t empty_state;
    CuEVM::state_access_t filled_state;
    CuEVM::account_t account;

    StateAccessTest() : arith(cgbn_no_checks), empty_state(), filled_state(), account() {
        CuEVM::bn_t value;
        cgbn_set_ui32(arith.env, value, 0x12345678);
        account.set_address(arith, value);
        cgbn_set_ui32(arith.env, value, 0x1000);
        account.set_balance(arith, value);
        cgbn_set_ui32(arith.env, value, 0x1);
        account.set_nonce(arith, value);
        CuEVM::byte_array_t byte_code(3);
        byte_code[0] = 0xFF;
        byte_code[1] = 0x00;
        byte_code[2] = 0x11;
        account.set_byte_code(byte_code);
        CuEVM::bn_t key;
        cgbn_set_ui32(arith.env, key, 0x1);
        cgbn_set_ui32(arith.env, value, 0x2);
        account.set_storage_value(arith, key, value);
        filled_state.set_account(arith, account, CuEVM::ACCOUNT_NONE_FLAG);
    }

    void SetUp() override {
        // Initialize any necessary resources here
    }

    void TearDown() override {
        // Clean up any resources here
        empty_state.free();
        account.free();
    }
};

TEST_F(StateAccessTest, DefaultConstructor) {
    EXPECT_EQ(empty_state.no_accounts, 0U);
    EXPECT_EQ(empty_state.accounts, nullptr);
    EXPECT_EQ(empty_state.flags, nullptr);
}

TEST_F(StateAccessTest, CopyConstructor) {
    state_access_t state2(filled_state);
    EXPECT_EQ(state2.no_accounts, filled_state.no_accounts);
    EXPECT_EQ(state2.accounts[0].address, filled_state.accounts[0].address);
    EXPECT_EQ(state2.accounts[0].balance, filled_state.accounts[0].balance);
    EXPECT_EQ(state2.accounts[0].nonce, filled_state.accounts[0].nonce);
    EXPECT_EQ(state2.accounts[0].byte_code.size, filled_state.accounts[0].byte_code.size);
    EXPECT_EQ(state2.accounts[0].byte_code[0], filled_state.accounts[0].byte_code[0]);
    EXPECT_EQ(state2.accounts[0].byte_code[1], filled_state.accounts[0].byte_code[1]);
    EXPECT_EQ(state2.accounts[0].byte_code[2], filled_state.accounts[0].byte_code[2]);
    EXPECT_EQ(state2.accounts[0].storage.size, filled_state.accounts[0].storage.size);
    EXPECT_EQ(state2.accounts[0].storage.capacity, filled_state.accounts[0].storage.capacity);
    EXPECT_EQ(state2.accounts[0].storage.storage[0].key, filled_state.accounts[0].storage.storage[0].key);
    EXPECT_EQ(state2.accounts[0].storage.storage[0].value, filled_state.accounts[0].storage.storage[0].value);
    EXPECT_EQ(state2.flags[0].flags, filled_state.flags[0].flags);
}

TEST_F(StateAccessTest, AssignmentOperator) {
    state_access_t state2;
    state2 = filled_state;
    EXPECT_EQ(state2.no_accounts, filled_state.no_accounts);
    EXPECT_EQ(state2.accounts[0].address, filled_state.accounts[0].address);
    EXPECT_EQ(state2.accounts[0].balance, filled_state.accounts[0].balance);
    EXPECT_EQ(state2.accounts[0].nonce, filled_state.accounts[0].nonce);
    EXPECT_EQ(state2.accounts[0].byte_code.size, filled_state.accounts[0].byte_code.size);
    EXPECT_EQ(state2.accounts[0].byte_code[0], filled_state.accounts[0].byte_code[0]);
    EXPECT_EQ(state2.accounts[0].byte_code[1], filled_state.accounts[0].byte_code[1]);
    EXPECT_EQ(state2.accounts[0].byte_code[2], filled_state.accounts[0].byte_code[2]);
    EXPECT_EQ(state2.accounts[0].storage.size, filled_state.accounts[0].storage.size);
    EXPECT_EQ(state2.accounts[0].storage.capacity, filled_state.accounts[0].storage.capacity);
    EXPECT_EQ(state2.accounts[0].storage.storage[0].key, filled_state.accounts[0].storage.storage[0].key);
    EXPECT_EQ(state2.accounts[0].storage.storage[0].value, filled_state.accounts[0].storage.storage[0].value);
    EXPECT_EQ(state2.flags[0].flags, filled_state.flags[0].flags);
}

TEST_F(StateAccessTest, AddAccount) {
    EXPECT_EQ(empty_state.add_account(account), ERROR_SUCCESS);
    EXPECT_EQ(empty_state.no_accounts, 1U);
    EXPECT_EQ(empty_state.accounts[0].address, account.address);
    EXPECT_EQ(empty_state.accounts[0].balance, account.balance);
    EXPECT_EQ(empty_state.accounts[0].nonce, account.nonce);
    EXPECT_EQ(empty_state.accounts[0].byte_code.size, account.byte_code.size);
    EXPECT_EQ(empty_state.accounts[0].byte_code[0], account.byte_code[0]);
    EXPECT_EQ(empty_state.accounts[0].byte_code[1], account.byte_code[1]);
    EXPECT_EQ(empty_state.accounts[0].byte_code[2], account.byte_code[2]);
    EXPECT_EQ(empty_state.accounts[0].storage.size, account.storage.size);
    EXPECT_EQ(empty_state.accounts[0].storage.capacity, account.storage.capacity);
    EXPECT_EQ(empty_state.accounts[0].storage.storage[0].key, account.storage.storage[0].key);
    EXPECT_EQ(empty_state.accounts[0].storage.storage[0].value, account.storage.storage[0].value);
    EXPECT_EQ(empty_state.flags[0].flags, CuEVM::ACCOUNT_NONE_FLAG);
}

TEST_F(StateAccessTest, SetAccount) {
    empty_state.set_account(arith, account);
    EXPECT_EQ(empty_state.no_accounts, 1U);
    EXPECT_EQ(empty_state.accounts[0].address, account.address);
    EXPECT_EQ(empty_state.accounts[0].balance, account.balance);
    EXPECT_EQ(empty_state.accounts[0].nonce, account.nonce);
    EXPECT_EQ(empty_state.accounts[0].byte_code.size, account.byte_code.size);
    EXPECT_EQ(empty_state.accounts[0].byte_code[0], account.byte_code[0]);
    EXPECT_EQ(empty_state.accounts[0].byte_code[1], account.byte_code[1]);
    EXPECT_EQ(empty_state.accounts[0].byte_code[2], account.byte_code[2]);
    EXPECT_EQ(empty_state.accounts[0].storage.size, account.storage.size);
    EXPECT_EQ(empty_state.accounts[0].storage.capacity, account.storage.capacity);
    EXPECT_EQ(empty_state.accounts[0].storage.storage[0].key, account.storage.storage[0].key);
    EXPECT_EQ(empty_state.accounts[0].storage.storage[0].value, account.storage.storage[0].value);
    EXPECT_EQ(empty_state.flags[0].flags, CuEVM::ACCOUNT_ALL_FLAG);
}

TEST_F(StateAccessTest, GetAccount) {
    CuEVM::account_t retrieved_account;
    CuEVM::bn_t address;
    cgbn_set_ui32(arith.env, address, 0x12345678);
    EXPECT_EQ(filled_state.get_account(arith, address, retrieved_account), ERROR_SUCCESS);
    EXPECT_EQ(retrieved_account.address, account.address);
    EXPECT_EQ(retrieved_account.balance, account.balance);
    EXPECT_EQ(retrieved_account.nonce, account.nonce);
    EXPECT_EQ(retrieved_account.byte_code.size, account.byte_code.size);
    EXPECT_EQ(retrieved_account.byte_code[0], account.byte_code[0]);
    EXPECT_EQ(retrieved_account.byte_code[1], account.byte_code[1]);
    EXPECT_EQ(retrieved_account.byte_code[2], account.byte_code[2]);
    EXPECT_EQ(retrieved_account.storage.size, account.storage.size);
    EXPECT_EQ(retrieved_account.storage.capacity, account.storage.capacity);
    EXPECT_EQ(retrieved_account.storage.storage[0].key, account.storage.storage[0].key);
    EXPECT_EQ(retrieved_account.storage.storage[0].value, account.storage.storage[0].value);

    CuEVM::bn_t non_existent_address;
    cgbn_set_ui32(arith.env, non_existent_address, 0x87654321);
    EXPECT_EQ(filled_state.get_account(arith, non_existent_address, retrieved_account), ERROR_STATE_ADDRESS_NOT_FOUND);

    // Test get account ptr
    CuEVM::account_t* retrieved_account_ptr;
    EXPECT_EQ(filled_state.get_account(arith, address, retrieved_account_ptr), ERROR_SUCCESS);
    EXPECT_EQ(retrieved_account_ptr->address, account.address);
    EXPECT_EQ(retrieved_account_ptr->balance, account.balance);
    EXPECT_EQ(retrieved_account_ptr->nonce, account.nonce);
    EXPECT_EQ(retrieved_account_ptr->byte_code.size, account.byte_code.size);
    EXPECT_EQ(retrieved_account_ptr->byte_code[0], account.byte_code[0]);
    EXPECT_EQ(retrieved_account_ptr->byte_code[1], account.byte_code[1]);
    EXPECT_EQ(retrieved_account_ptr->byte_code[2], account.byte_code[2]);
    EXPECT_EQ(retrieved_account_ptr->storage.size, account.storage.size);
    EXPECT_EQ(retrieved_account_ptr->storage.capacity, account.storage.capacity);
    EXPECT_EQ(retrieved_account_ptr->storage.storage[0].key, account.storage.storage[0].key);
    EXPECT_EQ(retrieved_account_ptr->storage.storage[0].value, account.storage.storage[0].value);
}

TEST_F(StateAccessTest, GetAccountWithFlags) {
    empty_state.set_account(arith, account, CuEVM::ACCOUNT_NONE_FLAG);
    EXPECT_EQ(empty_state.no_accounts, 1U);
    EXPECT_EQ(empty_state.accounts[0].address, account.address);
    EXPECT_EQ(empty_state.accounts[0].balance, account.balance);
    EXPECT_EQ(empty_state.accounts[0].nonce, account.nonce);
    EXPECT_EQ(empty_state.accounts[0].byte_code.size, account.byte_code.size);
    EXPECT_EQ(empty_state.accounts[0].byte_code[0], account.byte_code[0]);
    EXPECT_EQ(empty_state.accounts[0].byte_code[1], account.byte_code[1]);
    EXPECT_EQ(empty_state.accounts[0].byte_code[2], account.byte_code[2]);
    EXPECT_EQ(empty_state.accounts[0].storage.size, account.storage.size);
    EXPECT_EQ(empty_state.accounts[0].storage.capacity, account.storage.capacity);
    EXPECT_EQ(empty_state.accounts[0].storage.storage[0].key, account.storage.storage[0].key);
    EXPECT_EQ(empty_state.accounts[0].storage.storage[0].value, account.storage.storage[0].value);
    EXPECT_EQ(empty_state.flags[0].flags, CuEVM::ACCOUNT_NONE_FLAG);
    CuEVM::bn_t address;
    cgbn_set_ui32(arith.env, address, 0x12345678);
    CuEVM::account_t* retrieved_account_ptr;
    EXPECT_EQ(empty_state.get_account(arith, address, retrieved_account_ptr, CuEVM::ACCOUNT_STORAGE_FLAG),
              ERROR_SUCCESS);
    EXPECT_EQ(retrieved_account_ptr->address, account.address);
    EXPECT_EQ(retrieved_account_ptr->balance, account.balance);
    EXPECT_EQ(retrieved_account_ptr->nonce, account.nonce);
    EXPECT_EQ(retrieved_account_ptr->byte_code.size, account.byte_code.size);
    EXPECT_EQ(retrieved_account_ptr->byte_code[0], account.byte_code[0]);
    EXPECT_EQ(retrieved_account_ptr->byte_code[1], account.byte_code[1]);
    EXPECT_EQ(retrieved_account_ptr->byte_code[2], account.byte_code[2]);
    EXPECT_EQ(retrieved_account_ptr->storage.size, account.storage.size);
    EXPECT_EQ(retrieved_account_ptr->storage.capacity, account.storage.capacity);
    EXPECT_EQ(retrieved_account_ptr->storage.storage[0].key, account.storage.storage[0].key);
    EXPECT_EQ(retrieved_account_ptr->storage.storage[0].value, account.storage.storage[0].value);
    EXPECT_EQ(empty_state.flags[0].flags, CuEVM::ACCOUNT_STORAGE_FLAG);
}

TEST_F(StateAccessTest, HasAccount) {
    CuEVM::bn_t address;
    cgbn_set_ui32(arith.env, address, 0x12345678);
    EXPECT_EQ(filled_state.has_account(arith, address), ERROR_SUCCESS);

    CuEVM::bn_t non_existent_address;
    cgbn_set_ui32(arith.env, non_existent_address, 0x87654321);
    EXPECT_EQ(filled_state.has_account(arith, non_existent_address), ERROR_STATE_ADDRESS_NOT_FOUND);
}

TEST_F(StateAccessTest, UpdateAccount) {
    CuEVM::account_t updated_account;
    updated_account.address = account.address;
    updated_account.balance = 100U;
    updated_account.nonce = 2U;
    updated_account.byte_code.clear();
    updated_account.byte_code.grow(2, 1);
    updated_account.byte_code[0] = 0x11;
    updated_account.byte_code[1] = 0x22;
    updated_account.storage.clear();
    CuEVM::bn_t key, value;
    cgbn_set_ui32(arith.env, key, 0x1);
    cgbn_set_ui32(arith.env, value, 0x3);
    updated_account.set_storage_value(arith, key, value);
    cgbn_set_ui32(arith.env, key, 0x2);
    cgbn_set_ui32(arith.env, value, 0x4);
    updated_account.set_storage_value(arith, key, value);
    EXPECT_EQ(filled_state.update_account(arith, updated_account, CuEVM::ACCOUNT_NONCE_FLAG), ERROR_SUCCESS);

    CuEVM::account_t retrieved_account;
    CuEVM::bn_t address;
    cgbn_set_ui32(arith.env, address, 0x12345678);
    EXPECT_EQ(filled_state.get_account(arith, address, retrieved_account), ERROR_SUCCESS);
    EXPECT_EQ(retrieved_account.address, account.address);
    EXPECT_EQ(retrieved_account.balance, account.balance);
    EXPECT_EQ(retrieved_account.nonce, updated_account.nonce);
    EXPECT_EQ(retrieved_account.byte_code.size, account.byte_code.size);
    EXPECT_EQ(retrieved_account.byte_code[0], account.byte_code[0]);
    EXPECT_EQ(retrieved_account.byte_code[1], account.byte_code[1]);
    EXPECT_EQ(retrieved_account.byte_code[2], account.byte_code[2]);
    EXPECT_EQ(retrieved_account.storage.size, account.storage.size);
    EXPECT_EQ(retrieved_account.storage.capacity, account.storage.capacity);
    EXPECT_EQ(retrieved_account.storage.storage[0].key, account.storage.storage[0].key);
    EXPECT_EQ(retrieved_account.storage.storage[0].value, account.storage.storage[0].value);

    EXPECT_EQ(filled_state.update_account(arith, updated_account, CuEVM::ACCOUNT_BALANCE_FLAG), ERROR_SUCCESS);
    EXPECT_EQ(filled_state.get_account(arith, address, retrieved_account), ERROR_SUCCESS);
    EXPECT_EQ(retrieved_account.address, account.address);
    EXPECT_EQ(retrieved_account.balance, updated_account.balance);
    EXPECT_EQ(retrieved_account.nonce, updated_account.nonce);
    EXPECT_EQ(retrieved_account.byte_code.size, account.byte_code.size);
    EXPECT_EQ(retrieved_account.byte_code[0], account.byte_code[0]);
    EXPECT_EQ(retrieved_account.byte_code[1], account.byte_code[1]);
    EXPECT_EQ(retrieved_account.byte_code[2], account.byte_code[2]);
    EXPECT_EQ(retrieved_account.storage.size, account.storage.size);
    EXPECT_EQ(retrieved_account.storage.capacity, account.storage.capacity);
    EXPECT_EQ(retrieved_account.storage.storage[0].key, account.storage.storage[0].key);
    EXPECT_EQ(retrieved_account.storage.storage[0].value, account.storage.storage[0].value);

    EXPECT_EQ(filled_state.update_account(arith, updated_account, CuEVM::ACCOUNT_BYTE_CODE_FLAG), ERROR_SUCCESS);
    EXPECT_EQ(filled_state.get_account(arith, address, retrieved_account), ERROR_SUCCESS);
    EXPECT_EQ(retrieved_account.address, account.address);
    EXPECT_EQ(retrieved_account.balance, updated_account.balance);
    EXPECT_EQ(retrieved_account.nonce, updated_account.nonce);
    EXPECT_EQ(retrieved_account.byte_code.size, updated_account.byte_code.size);
    EXPECT_EQ(retrieved_account.byte_code[0], updated_account.byte_code[0]);
    EXPECT_EQ(retrieved_account.byte_code[1], updated_account.byte_code[1]);
    EXPECT_EQ(retrieved_account.storage.size, account.storage.size);
    EXPECT_EQ(retrieved_account.storage.capacity, account.storage.capacity);
    EXPECT_EQ(retrieved_account.storage.storage[0].key, account.storage.storage[0].key);
    EXPECT_EQ(retrieved_account.storage.storage[0].value, account.storage.storage[0].value);

    EXPECT_EQ(filled_state.update_account(arith, updated_account, CuEVM::ACCOUNT_STORAGE_FLAG), ERROR_SUCCESS);
    EXPECT_EQ(filled_state.get_account(arith, address, retrieved_account), ERROR_SUCCESS);
    EXPECT_EQ(retrieved_account.address, account.address);
    EXPECT_EQ(retrieved_account.balance, updated_account.balance);
    EXPECT_EQ(retrieved_account.nonce, updated_account.nonce);
    EXPECT_EQ(retrieved_account.byte_code.size, updated_account.byte_code.size);
    EXPECT_EQ(retrieved_account.byte_code[0], updated_account.byte_code[0]);
    EXPECT_EQ(retrieved_account.byte_code[1], updated_account.byte_code[1]);
    EXPECT_EQ(retrieved_account.storage.size, updated_account.storage.size);
    EXPECT_EQ(retrieved_account.storage.capacity, updated_account.storage.capacity);
    EXPECT_EQ(retrieved_account.storage.storage[0].key, updated_account.storage.storage[0].key);
    EXPECT_EQ(retrieved_account.storage.storage[0].value, updated_account.storage.storage[0].value);
    EXPECT_EQ(retrieved_account.storage.storage[1].key, updated_account.storage.storage[1].key);
    EXPECT_EQ(retrieved_account.storage.storage[1].value, updated_account.storage.storage[1].value);
}

TEST_F(StateAccessTest, IsEmptyAccount) {
    CuEVM::bn_t address;
    cgbn_set_ui32(arith.env, address, 0x12345678);
    EXPECT_EQ(filled_state.has_account(arith, address), ERROR_SUCCESS);
    EXPECT_EQ(filled_state.is_empty_account(arith, address), ERROR_ACCOUNT_NOT_EMPTY);
    cgbn_set_ui32(arith.env, address, 0x87654321);
    EXPECT_EQ(filled_state.has_account(arith, address), ERROR_STATE_ADDRESS_NOT_FOUND);
    EXPECT_EQ(filled_state.is_empty_account(arith, address), ERROR_STATE_ADDRESS_NOT_FOUND);
    CuEVM::account_t empty_account;
    empty_account.empty();
    cgbn_store(arith.env, &(empty_account.address), address);
    EXPECT_EQ(filled_state.set_account(arith, empty_account), ERROR_SUCCESS);
    EXPECT_EQ(filled_state.has_account(arith, address), ERROR_SUCCESS);
    EXPECT_EQ(filled_state.is_empty_account(arith, address), ERROR_SUCCESS);
}

TEST_F(StateAccessTest, UpdateState) {
    CuEVM::state_access_t updated_state;
    CuEVM::bn_t address;
    cgbn_set_ui32(arith.env, address, 0x87654321);
    CuEVM::account_t updated_account;
    updated_account.set_address(arith, address);
    updated_account.balance = 100U;
    updated_account.nonce = 2U;
    updated_account.byte_code.clear();
    updated_account.byte_code.grow(2, 1);
    updated_account.byte_code[0] = 0x11;
    updated_account.byte_code[1] = 0x22;
    updated_account.storage.clear();
    CuEVM::bn_t key, value;
    cgbn_set_ui32(arith.env, key, 0x1);
    cgbn_set_ui32(arith.env, value, 0x3);
    updated_account.set_storage_value(arith, key, value);
    cgbn_set_ui32(arith.env, key, 0x2);
    cgbn_set_ui32(arith.env, value, 0x4);
    updated_account.set_storage_value(arith, key, value);
    updated_state.set_account(arith, updated_account, CuEVM::ACCOUNT_ALL_FLAG);
    EXPECT_EQ(filled_state.update(arith, updated_state), ERROR_SUCCESS);
    EXPECT_EQ(filled_state.no_accounts, 2U);

    CuEVM::account_t retrieved_account;
    cgbn_set_ui32(arith.env, address, 0x87654321);
    EXPECT_EQ(filled_state.get_account(arith, address, retrieved_account), ERROR_SUCCESS);
    EXPECT_EQ(retrieved_account.address, updated_account.address);
    EXPECT_EQ(retrieved_account.balance, updated_account.balance);
    EXPECT_EQ(retrieved_account.nonce, updated_account.nonce);
    EXPECT_EQ(retrieved_account.byte_code.size, updated_account.byte_code.size);
    EXPECT_EQ(retrieved_account.byte_code[0], updated_account.byte_code[0]);
    EXPECT_EQ(retrieved_account.byte_code[1], updated_account.byte_code[1]);
    EXPECT_EQ(retrieved_account.storage.size, updated_account.storage.size);
    EXPECT_EQ(retrieved_account.storage.capacity, updated_account.storage.capacity);
    EXPECT_EQ(retrieved_account.storage.storage[0].key, updated_account.storage.storage[0].key);
    EXPECT_EQ(retrieved_account.storage.storage[0].value, updated_account.storage.storage[0].value);
    EXPECT_EQ(retrieved_account.storage.storage[1].key, updated_account.storage.storage[1].key);
    EXPECT_EQ(retrieved_account.storage.storage[1].value, updated_account.storage.storage[1].value);
    EXPECT_EQ(filled_state.flags[1].flags, CuEVM::ACCOUNT_ALL_FLAG);
}

TEST_F(StateAccessTest, ToJson) {
    cJSON* json = filled_state.to_json();
    char* json_str = cJSON_Print(json);
    EXPECT_STREQ(
        json_str,
        "{\n\t\"0x0000000000000000000000000000000012345678\":\t{\n\t\t\"balance\":"
        "\t\"0x0000000000000000000000000000000000000000000000000000000000001000\",\n\t\t\"nonce\":"
        "\t\"0x0000000000000000000000000000000000000000000000000000000000000001\",\n\t\t\"code\":\t\"0xff0011\","
        "\n\t\t\"storage\":\t{\n\t\t\t\"0x1\":\t\"0x2\"\n\t\t},\n\t\t\"flags\":\t\"00000000\"\n\t}\n}");
    cJSON_Delete(json);
    free(json_str);
}

TEST_F(StateAccessTest, Print) {
    // Redirect stdout to a string stream
    testing::internal::CaptureStdout();

    filled_state.print();
    std::string output = testing::internal::GetCapturedStdout();

    // Expected output
    ASSERT_EQ(output,
              "no_accounts: 1\naccounts[0]:\nAccount:\n00000000 00000000 00000000 00000000 00000000 00000000 00000000 "
              "12345678 \nBalance: 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00001000 \nNonce: "
              "00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000001 \nByte code: size: 3\ndata: "
              "ff0011\nStorage: \nStorage size: 1\nElement 0:\nKey: 00000000 00000000 00000000 00000000 00000000 "
              "00000000 00000000 00000001 \nValue: 00000000 00000000 00000000 00000000 00000000 00000000 00000000 "
              "00000002 \nflags[0]:\nAccount flags: 00000000\n");
}

__global__ void test_state_access_kernel(CuEVM::state_access_t* gpu_states, uint32_t* result, uint32_t count) {
    int32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / CuEVM::cgbn_tpi;
    if (instance >= count) return;
    result[instance] = ERROR_SUCCESS;
    CuEVM::ArithEnv arith(cgbn_no_checks);

    CuEVM::state_access_t empty_state;
    CuEVM::state_access_t filled_state;
    CuEVM::account_t account;
    CuEVM::bn_t value;
    cgbn_set_ui32(arith.env, value, 0x12345678);
    account.set_address(arith, value);
    cgbn_set_ui32(arith.env, value, 0x1000);
    account.set_balance(arith, value);
    cgbn_set_ui32(arith.env, value, 0x1);
    account.set_nonce(arith, value);
    CuEVM::byte_array_t byte_code(3);
    byte_code[0] = 0xFF;
    byte_code[1] = 0x00;
    byte_code[2] = 0x11;
    account.set_byte_code(byte_code);
    CuEVM::bn_t key;
    cgbn_set_ui32(arith.env, key, 0x1);
    cgbn_set_ui32(arith.env, value, 0x2);
    account.set_storage_value(arith, key, value);

    filled_state.set_account(arith, account, CuEVM::ACCOUNT_NONE_FLAG);
    // return;

    // Test Default Constructor
    result[instance] = (empty_state.no_accounts == 0U) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_state.accounts == nullptr) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_state.flags == nullptr) ? ERROR_SUCCESS : __LINE__;

    // Test Copy Constructor
    CuEVM::state_access_t state2(filled_state);
    result[instance] |= (state2.no_accounts == filled_state.no_accounts) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (state2.accounts[0].address == filled_state.accounts[0].address) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (state2.accounts[0].balance == filled_state.accounts[0].balance) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (state2.accounts[0].nonce == filled_state.accounts[0].nonce) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (state2.accounts[0].byte_code.size == filled_state.accounts[0].byte_code.size) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (state2.accounts[0].byte_code[0] == filled_state.accounts[0].byte_code[0]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (state2.accounts[0].byte_code[1] == filled_state.accounts[0].byte_code[1]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (state2.accounts[0].byte_code[2] == filled_state.accounts[0].byte_code[2]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (state2.accounts[0].storage.size == filled_state.accounts[0].storage.size) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (state2.accounts[0].storage.capacity == filled_state.accounts[0].storage.capacity) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (state2.accounts[0].storage.storage[0].key == filled_state.accounts[0].storage.storage[0].key)
                            ? ERROR_SUCCESS
                            : __LINE__;
    result[instance] |=
        (state2.accounts[0].storage.storage[0].value == filled_state.accounts[0].storage.storage[0].value)
            ? ERROR_SUCCESS
            : __LINE__;
    result[instance] |= (state2.flags[0].flags == filled_state.flags[0].flags) ? ERROR_SUCCESS : __LINE__;

    // Test Assignment Operator
    CuEVM::state_access_t state3;
    state3 = filled_state;
    result[instance] |= (state3.no_accounts == filled_state.no_accounts) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (state3.accounts[0].address == filled_state.accounts[0].address) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (state3.accounts[0].balance == filled_state.accounts[0].balance) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (state3.accounts[0].nonce == filled_state.accounts[0].nonce) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (state3.accounts[0].byte_code.size == filled_state.accounts[0].byte_code.size) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (state3.accounts[0].byte_code[0] == filled_state.accounts[0].byte_code[0]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (state3.accounts[0].byte_code[1] == filled_state.accounts[0].byte_code[1]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (state3.accounts[0].byte_code[2] == filled_state.accounts[0].byte_code[2]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (state3.accounts[0].storage.size == filled_state.accounts[0].storage.size) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (state3.accounts[0].storage.capacity == filled_state.accounts[0].storage.capacity) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (state3.accounts[0].storage.storage[0].key == filled_state.accounts[0].storage.storage[0].key)
                            ? ERROR_SUCCESS
                            : __LINE__;
    result[instance] |=
        (state3.accounts[0].storage.storage[0].value == filled_state.accounts[0].storage.storage[0].value)
            ? ERROR_SUCCESS
            : __LINE__;
    result[instance] |= (state3.flags[0].flags == filled_state.flags[0].flags) ? ERROR_SUCCESS : __LINE__;

    // Test Add Account
    result[instance] |= (empty_state.add_account(account) == ERROR_SUCCESS) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_state.no_accounts == 1U) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_state.accounts[0].address == account.address) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_state.accounts[0].balance == account.balance) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_state.accounts[0].nonce == account.nonce) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_state.accounts[0].byte_code.size == account.byte_code.size) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_state.accounts[0].byte_code[0] == account.byte_code[0]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_state.accounts[0].byte_code[1] == account.byte_code[1]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_state.accounts[0].byte_code[2] == account.byte_code[2]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_state.accounts[0].storage.size == account.storage.size) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (empty_state.accounts[0].storage.capacity == account.storage.capacity) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (empty_state.accounts[0].storage.storage[0].key == account.storage.storage[0].key) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_state.accounts[0].storage.storage[0].value == account.storage.storage[0].value)
                            ? ERROR_SUCCESS
                            : __LINE__;
    result[instance] |= (empty_state.flags[0].flags == CuEVM::ACCOUNT_NONE_FLAG) ? ERROR_SUCCESS : __LINE__;

    empty_state.free();

    // Test Set Account
    empty_state.set_account(arith, account);
    result[instance] |= (empty_state.no_accounts == 1U) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_state.accounts[0].address == account.address) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_state.accounts[0].balance == account.balance) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_state.accounts[0].nonce == account.nonce) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_state.accounts[0].byte_code.size == account.byte_code.size) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_state.accounts[0].byte_code[0] == account.byte_code[0]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_state.accounts[0].byte_code[1] == account.byte_code[1]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_state.accounts[0].byte_code[2] == account.byte_code[2]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_state.accounts[0].storage.size == account.storage.size) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (empty_state.accounts[0].storage.capacity == account.storage.capacity) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (empty_state.accounts[0].storage.storage[0].key == account.storage.storage[0].key) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_state.accounts[0].storage.storage[0].value == account.storage.storage[0].value)
                            ? ERROR_SUCCESS
                            : __LINE__;
    result[instance] |= (empty_state.flags[0].flags == CuEVM::ACCOUNT_ALL_FLAG) ? ERROR_SUCCESS : __LINE__;

    // Test Get Account
    CuEVM::account_t retrieved_account;
    CuEVM::bn_t address;
    cgbn_set_ui32(arith.env, address, 0x12345678);
    result[instance] |=
        (filled_state.get_account(arith, address, retrieved_account) == ERROR_SUCCESS) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account.address == account.address) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account.balance == account.balance) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account.nonce == account.nonce) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account.byte_code.size == account.byte_code.size) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account.byte_code[0] == account.byte_code[0]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account.byte_code[1] == account.byte_code[1]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account.byte_code[2] == account.byte_code[2]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account.storage.size == account.storage.size) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account.storage.capacity == account.storage.capacity) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (retrieved_account.storage.storage[0].key == account.storage.storage[0].key) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (retrieved_account.storage.storage[0].value == account.storage.storage[0].value) ? ERROR_SUCCESS : __LINE__;
    CuEVM::bn_t non_existent_address;
    cgbn_set_ui32(arith.env, non_existent_address, 0x87654321);
    result[instance] |=
        (filled_state.get_account(arith, non_existent_address, retrieved_account) == ERROR_STATE_ADDRESS_NOT_FOUND)
            ? ERROR_SUCCESS
            : __LINE__;
    CuEVM::account_t* retrieved_account_ptr;
    result[instance] |=
        (filled_state.get_account(arith, address, retrieved_account_ptr) == ERROR_SUCCESS) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account_ptr->address == account.address) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account_ptr->balance == account.balance) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account_ptr->nonce == account.nonce) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account_ptr->byte_code.size == account.byte_code.size) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account_ptr->byte_code[0] == account.byte_code[0]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account_ptr->byte_code[1] == account.byte_code[1]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account_ptr->byte_code[2] == account.byte_code[2]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account_ptr->storage.size == account.storage.size) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (retrieved_account_ptr->storage.capacity == account.storage.capacity) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (retrieved_account_ptr->storage.storage[0].key == account.storage.storage[0].key) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account_ptr->storage.storage[0].value == account.storage.storage[0].value)
                            ? ERROR_SUCCESS
                            : __LINE__;
    // Test Get Account with Flags
    empty_state.free();
    empty_state.set_account(arith, account, CuEVM::ACCOUNT_NONE_FLAG);
    result[instance] |= (empty_state.no_accounts == 1U) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_state.accounts[0].address == account.address) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_state.accounts[0].balance == account.balance) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_state.accounts[0].nonce == account.nonce) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_state.accounts[0].byte_code.size == account.byte_code.size) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_state.accounts[0].byte_code[0] == account.byte_code[0]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_state.accounts[0].byte_code[1] == account.byte_code[1]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_state.accounts[0].byte_code[2] == account.byte_code[2]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_state.accounts[0].storage.size == account.storage.size) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (empty_state.accounts[0].storage.capacity == account.storage.capacity) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_state.flags[0].flags == CuEVM::ACCOUNT_NONE_FLAG) ? ERROR_SUCCESS : __LINE__;

    cgbn_set_ui32(arith.env, address, 0x12345678);
    result[instance] |=
        (empty_state.get_account(arith, address, retrieved_account_ptr, CuEVM::ACCOUNT_STORAGE_FLAG) == ERROR_SUCCESS)
            ? ERROR_SUCCESS
            : __LINE__;
    result[instance] |= (retrieved_account_ptr->address == account.address) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account_ptr->balance == account.balance) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account_ptr->nonce == account.nonce) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account_ptr->byte_code.size == account.byte_code.size) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account_ptr->byte_code[0] == account.byte_code[0]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account_ptr->byte_code[1] == account.byte_code[1]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account_ptr->byte_code[2] == account.byte_code[2]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account_ptr->storage.size == account.storage.size) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (retrieved_account_ptr->storage.capacity == account.storage.capacity) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (retrieved_account_ptr->storage.storage[0].key == account.storage.storage[0].key) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account_ptr->storage.storage[0].value == account.storage.storage[0].value)
                            ? ERROR_SUCCESS
                            : __LINE__;
    result[instance] |= (empty_state.flags[0].flags == CuEVM::ACCOUNT_STORAGE_FLAG) ? ERROR_SUCCESS : __LINE__;

    // Test Has Account
    result[instance] |= (filled_state.has_account(arith, address) == ERROR_SUCCESS) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (filled_state.has_account(arith, non_existent_address) == ERROR_STATE_ADDRESS_NOT_FOUND)
                            ? ERROR_SUCCESS
                            : __LINE__;

    // Test Update Account
    CuEVM::account_t updated_account;
    updated_account.address = account.address;
    updated_account.balance = 0x100U;
    updated_account.nonce = 0x2U;
    updated_account.byte_code.clear();
    updated_account.byte_code.grow(2, 1);
    updated_account.byte_code[0] = 0x11;
    updated_account.byte_code[1] = 0x22;
    updated_account.storage.clear();
    cgbn_set_ui32(arith.env, key, 0x1);
    cgbn_set_ui32(arith.env, value, 0x3);
    updated_account.set_storage_value(arith, key, value);
    cgbn_set_ui32(arith.env, key, 0x2);
    cgbn_set_ui32(arith.env, value, 0x4);
    updated_account.set_storage_value(arith, key, value);
    result[instance] |=
        (filled_state.update_account(arith, updated_account, CuEVM::ACCOUNT_NONCE_FLAG) == ERROR_SUCCESS)
            ? ERROR_SUCCESS
            : __LINE__;
    CuEVM::account_t retrieved_account2;
    cgbn_set_ui32(arith.env, address, 0x12345678);
    result[instance] |=
        (filled_state.get_account(arith, address, retrieved_account2) == ERROR_SUCCESS) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account2.address == account.address) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account2.balance == account.balance) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account2.nonce == updated_account.nonce) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account2.byte_code.size == account.byte_code.size) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account2.byte_code[0] == account.byte_code[0]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account2.byte_code[1] == account.byte_code[1]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account2.byte_code[2] == account.byte_code[2]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account2.storage.size == account.storage.size) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account2.storage.capacity == account.storage.capacity) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (retrieved_account2.storage.storage[0].key == account.storage.storage[0].key) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (retrieved_account2.storage.storage[0].value == account.storage.storage[0].value) ? ERROR_SUCCESS : __LINE__;

    result[instance] |=
        (filled_state.update_account(arith, updated_account, CuEVM::ACCOUNT_BALANCE_FLAG) == ERROR_SUCCESS)
            ? ERROR_SUCCESS
            : __LINE__;
    result[instance] |=
        (filled_state.get_account(arith, address, retrieved_account2) == ERROR_SUCCESS) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account2.address == account.address) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account2.balance == updated_account.balance) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account2.nonce == updated_account.nonce) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account2.byte_code.size == account.byte_code.size) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account2.byte_code[0] == account.byte_code[0]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account2.byte_code[1] == account.byte_code[1]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account2.byte_code[2] == account.byte_code[2]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account2.storage.size == account.storage.size) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account2.storage.capacity == account.storage.capacity) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (retrieved_account2.storage.storage[0].key == account.storage.storage[0].key) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (retrieved_account2.storage.storage[0].value == account.storage.storage[0].value) ? ERROR_SUCCESS : __LINE__;

    result[instance] |=
        (filled_state.update_account(arith, updated_account, CuEVM::ACCOUNT_BYTE_CODE_FLAG) == ERROR_SUCCESS)
            ? ERROR_SUCCESS
            : __LINE__;
    result[instance] |=
        (filled_state.get_account(arith, address, retrieved_account2) == ERROR_SUCCESS) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account2.address == account.address) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account2.balance == updated_account.balance) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account2.nonce == updated_account.nonce) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (retrieved_account2.byte_code.size == updated_account.byte_code.size) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account2.byte_code[0] == updated_account.byte_code[0]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account2.byte_code[1] == updated_account.byte_code[1]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account2.storage.size == account.storage.size) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account2.storage.capacity == account.storage.capacity) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (retrieved_account2.storage.storage[0].key == account.storage.storage[0].key) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (retrieved_account2.storage.storage[0].value == account.storage.storage[0].value) ? ERROR_SUCCESS : __LINE__;

    result[instance] |=
        (filled_state.update_account(arith, updated_account, CuEVM::ACCOUNT_STORAGE_FLAG) == ERROR_SUCCESS)
            ? ERROR_SUCCESS
            : __LINE__;
    result[instance] |=
        (filled_state.get_account(arith, address, retrieved_account2) == ERROR_SUCCESS) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account2.address == account.address) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account2.balance == updated_account.balance) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account2.nonce == updated_account.nonce) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (retrieved_account2.byte_code.size == updated_account.byte_code.size) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account2.byte_code[0] == updated_account.byte_code[0]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account2.byte_code[1] == updated_account.byte_code[1]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account2.storage.size == updated_account.storage.size) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (retrieved_account2.storage.capacity == updated_account.storage.capacity) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (retrieved_account2.storage.storage[0].key == updated_account.storage.storage[0].key)
                            ? ERROR_SUCCESS
                            : __LINE__;
    result[instance] |= (retrieved_account2.storage.storage[0].value == updated_account.storage.storage[0].value)
                            ? ERROR_SUCCESS
                            : __LINE__;
    result[instance] |= (retrieved_account2.storage.storage[1].key == updated_account.storage.storage[1].key)
                            ? ERROR_SUCCESS
                            : __LINE__;
    result[instance] |= (retrieved_account2.storage.storage[1].value == updated_account.storage.storage[1].value)
                            ? ERROR_SUCCESS
                            : __LINE__;

    // Test Is Empty Account
    result[instance] |=
        (filled_state.is_empty_account(arith, address) == ERROR_ACCOUNT_NOT_EMPTY) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (filled_state.is_empty_account(arith, non_existent_address) == ERROR_STATE_ADDRESS_NOT_FOUND)
                            ? ERROR_SUCCESS
                            : __LINE__;
    CuEVM::account_t empty_account;
    empty_account.empty();
    cgbn_store(arith.env, &(empty_account.address), address);
    result[instance] |= (empty_state.set_account(arith, empty_account) == ERROR_SUCCESS) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_state.has_account(arith, address) == ERROR_SUCCESS) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (empty_state.is_empty_account(arith, address) == ERROR_SUCCESS) ? ERROR_SUCCESS : __LINE__;

    // add the empty state to the filled state
    empty_account.set_address(arith, non_existent_address);
    result[instance] |= (filled_state.set_account(arith, empty_account) == ERROR_SUCCESS) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (filled_state.has_account(arith, non_existent_address) == ERROR_SUCCESS) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (filled_state.is_empty_account(arith, non_existent_address) == ERROR_SUCCESS) ? ERROR_SUCCESS : __LINE__;

    // add a new state to the filled state
    CuEVM::account_t new_account;
    new_account.clear();
    cgbn_set_ui32(arith.env, value, 0x87651234);
    new_account.set_address(arith, value);
    cgbn_set_ui32(arith.env, value, 0);
    new_account.set_balance(arith, value);
    cgbn_set_ui32(arith.env, value, 0);
    new_account.set_nonce(arith, value);
    new_account.set_byte_code(byte_code);
    cgbn_set_ui32(arith.env, key, 0x1);
    cgbn_set_ui32(arith.env, value, 0x3);
    new_account.set_storage_value(arith, key, value);
    CuEVM::state_access_t new_state;
    new_state.clear();
    new_state.set_account(arith, new_account, CuEVM::ACCOUNT_ALL_FLAG);
    result[instance] |= (filled_state.update(arith, new_state) == ERROR_SUCCESS) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (filled_state.no_accounts == 3U) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (filled_state.accounts[2].address == new_account.address) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (filled_state.accounts[2].balance == new_account.balance) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (filled_state.accounts[2].nonce == new_account.nonce) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (filled_state.accounts[2].byte_code.size == new_account.byte_code.size) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (filled_state.accounts[2].byte_code[0] == new_account.byte_code[0]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (filled_state.accounts[2].byte_code[1] == new_account.byte_code[1]) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (filled_state.accounts[2].storage.size == new_account.storage.size) ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        (filled_state.accounts[2].storage.capacity == new_account.storage.capacity) ? ERROR_SUCCESS : __LINE__;
    result[instance] |= (filled_state.accounts[2].storage.storage[0].key == new_account.storage.storage[0].key)
                            ? ERROR_SUCCESS
                            : __LINE__;
    result[instance] |= (filled_state.accounts[2].storage.storage[0].value == new_account.storage.storage[0].value)
                            ? ERROR_SUCCESS
                            : __LINE__;
    result[instance] |= (filled_state.flags[2].flags == CuEVM::ACCOUNT_ALL_FLAG) ? ERROR_SUCCESS : __LINE__;

    // send back the fill state
    gpu_states[instance] = filled_state;
}

#ifdef GPU
TEST_F(StateAccessTest, StateAccessOperationsGPU) {
    CuEVM::state_access_t* cpu_states = CuEVM::state_access_t::get_cpu(2);

    CUDA_CHECK(cudaDeviceReset());
    CuEVM::state_access_t* gpu_states = CuEVM::state_access_t::get_gpu_from_cpu(cpu_states, 2);
    uint32_t* d_result;
    cudaMalloc((void**)&d_result, 2 * sizeof(uint32_t));
    test_state_access_kernel<<<2, CuEVM::cgbn_tpi>>>(gpu_states, d_result, 2);
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t* h_result = (uint32_t*)malloc(2 * sizeof(uint32_t));
    CUDA_CHECK(cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 2; i++) {
        EXPECT_EQ(h_result[i], ERROR_SUCCESS);
    }
    free(h_result);
    CUDA_CHECK(cudaFree(d_result));
    CuEVM::state_access_t* filled_states = CuEVM::state_access_t::get_cpu_from_gpu(gpu_states, 2);
    for (int i = 0; i < 2; i++) {
        EXPECT_EQ(filled_states[i].no_accounts, 3U);
        EXPECT_EQ(filled_states[i].accounts[0].address, 0x12345678U);
        EXPECT_EQ(filled_states[i].accounts[0].balance, 0x100U);
        EXPECT_EQ(filled_states[i].accounts[0].nonce, 0x2U);
        EXPECT_EQ(filled_states[i].accounts[0].byte_code.size, 2U);
        EXPECT_EQ(filled_states[i].accounts[0].byte_code[0], 0x11);
        EXPECT_EQ(filled_states[i].accounts[0].byte_code[1], 0x22);
        EXPECT_EQ(filled_states[i].accounts[0].storage.size, 2U);
        EXPECT_EQ(filled_states[i].accounts[0].storage.capacity, 4U);
        EXPECT_EQ(filled_states[i].accounts[0].storage.storage[0].key, 0x1U);
        EXPECT_EQ(filled_states[i].accounts[0].storage.storage[0].value, 0x3U);
        EXPECT_EQ(filled_states[i].accounts[0].storage.storage[1].key, 0x2U);
        EXPECT_EQ(filled_states[i].accounts[0].storage.storage[1].value, 0x4U);
        EXPECT_EQ(filled_states[i].accounts[1].address, 0x87654321U);
        EXPECT_EQ(filled_states[i].accounts[1].balance, 0U);
        EXPECT_EQ(filled_states[i].accounts[1].nonce, 0U);
        EXPECT_EQ(filled_states[i].accounts[1].byte_code.size, 0U);
        EXPECT_EQ(filled_states[i].accounts[1].byte_code.data, nullptr);
        EXPECT_EQ(filled_states[i].accounts[1].storage.size, 0U);
        EXPECT_EQ(filled_states[i].accounts[1].storage.capacity, 0U);
        EXPECT_EQ(filled_states[i].accounts[1].storage.storage, nullptr);
        EXPECT_EQ(filled_states[i].accounts[2].address, 0x87651234U);
        EXPECT_EQ(filled_states[i].accounts[2].balance, 0U);
        EXPECT_EQ(filled_states[i].accounts[2].nonce, 0U);
        EXPECT_EQ(filled_states[i].accounts[2].byte_code.size, 3U);
        EXPECT_EQ(filled_states[i].accounts[2].byte_code[0], 0xFFU);
        EXPECT_EQ(filled_states[i].accounts[2].byte_code[1], 0x00U);
        EXPECT_EQ(filled_states[i].accounts[2].byte_code[2], 0x11U);
        EXPECT_EQ(filled_states[i].accounts[2].storage.size, 1U);
        EXPECT_EQ(filled_states[i].accounts[2].storage.capacity, 4U);
        EXPECT_EQ(filled_states[i].accounts[2].storage.storage[0].key, 0x1U);
        EXPECT_EQ(filled_states[i].accounts[2].storage.storage[0].value, 0x3U);
        EXPECT_EQ(filled_states[i].flags[2].flags, CuEVM::ACCOUNT_ALL_FLAG);
    }
    CuEVM::state_access_t::cpu_free(cpu_states, 2);
    // CuEVM::state_access_t::cpu_free(filled_states, 2);
    CUDA_CHECK(cudaDeviceReset());
}
#endif