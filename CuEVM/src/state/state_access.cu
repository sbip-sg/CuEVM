// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation
// Programme) Author: Stefan-Dan Ciocirlan Date: 2024-09-15
// SPDX-License-Identifier: MIT

#include <CuEVM/state/state_access.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM {
__host__ __device__ void state_access_t::duplicate(
    const state_access_t &other) {
    state_t::duplicate(other);
    if (no_accounts > 0) {
        flags = new CuEVM::account_flags_t[no_accounts];
        std::copy(other.flags, other.flags + no_accounts, flags);
    } else {
        flags = nullptr;
    }
}

__host__ __device__ void state_access_t::free() {
    __ONE_GPU_THREAD_BEGIN__
    if (flags != nullptr && no_accounts > 0) {
        delete[] flags;
    }
    __ONE_GPU_THREAD_END__
    state_t::free();
    clear();
}

__host__ void state_access_t::free_managed() {
    if (flags != nullptr && no_accounts > 0) {
        CUDA_CHECK(cudaFree(flags));
    }
    state_t::free_managed();
    clear();
}

__host__ __device__ void state_access_t::clear() {
    flags = nullptr;
    state_t::clear();
}

__host__ __device__ int32_t state_access_t::get_account(
    ArithEnv &arith, const bn_t &address, CuEVM::account_t &account,
    const CuEVM::account_flags_t flag) {
    uint32_t index = 0;
    if (state_t::get_account_index(arith, address, index) == ERROR_SUCCESS) {
        flags[index].update(flag);
        account = accounts[index];
        return ERROR_SUCCESS;
    }
    return ERROR_STATE_ADDRESS_NOT_FOUND;
}

__host__ __device__ int32_t state_access_t::get_account(
    ArithEnv &arith, const bn_t &address, CuEVM::account_t *&account_ptr,
    const CuEVM::account_flags_t flag) {
    uint32_t index = 0;
    if (state_t::get_account_index(arith, address, index) == ERROR_SUCCESS) {
        flags[index].update(flag);
        account_ptr = &accounts[index];
        return ERROR_SUCCESS;
    }
    return ERROR_STATE_ADDRESS_NOT_FOUND;
}

__host__ __device__ int32_t state_access_t::add_account(
    const CuEVM::account_t &account, const CuEVM::account_flags_t flag) {
    state_t::add_account(account);
    uint32_t index = no_accounts - 1;
    __SHARED_MEMORY__ CuEVM::account_flags_t *tmp_flags;
    __ONE_GPU_THREAD_BEGIN__
    tmp_flags = new CuEVM::account_flags_t[no_accounts];
    memcpy(tmp_flags, flags,
           (no_accounts - 1) * sizeof(CuEVM::account_flags_t));
    if (flags != nullptr) {
        delete[] flags;
    }
    __ONE_GPU_THREAD_END__
    flags = tmp_flags;
    flags[index] = flag;
    return ERROR_SUCCESS;
}

__host__ __device__ int32_t state_access_t::add_duplicate_account(
    CuEVM::account_t *&account_ptr, CuEVM::account_t *&src_account_ptr,
    const CuEVM::account_flags_t flag) {
    CuEVM::account_flags_t no_storage_copy(ACCOUNT_NON_STORAGE_FLAG);
    __SHARED_MEMORY__ CuEVM::account_t *tmp_account_ptr;
    __ONE_GPU_THREAD_BEGIN__
    tmp_account_ptr = new CuEVM::account_t(src_account_ptr, no_storage_copy);
    __ONE_GPU_THREAD_END__
    int32_t error_code = add_account(*tmp_account_ptr, flag);
    account_ptr = &accounts[no_accounts - 1];
    __ONE_GPU_THREAD_BEGIN__
    delete tmp_account_ptr;
    __ONE_GPU_THREAD_END__
    return error_code;
}

__host__ __device__ int32_t state_access_t::add_new_account(
    ArithEnv &arith, const bn_t &address, CuEVM::account_t *&account_ptr,
    const CuEVM::account_flags_t flag) {
    __SHARED_MEMORY__ CuEVM::account_t *tmp_account_ptr;
    __ONE_GPU_THREAD_BEGIN__
    tmp_account_ptr = new CuEVM::account_t();
    __ONE_GPU_THREAD_END__
    tmp_account_ptr->set_address(arith, address);
    int32_t error_code = add_account(*account_ptr, flag);
    account_ptr = &accounts[no_accounts - 1];
    __ONE_GPU_THREAD_BEGIN__
    delete tmp_account_ptr;
    __ONE_GPU_THREAD_END__
    return error_code;
}

__host__ __device__ int32_t
state_access_t::set_account(ArithEnv &arith, const CuEVM::account_t &account,
                            const CuEVM::account_flags_t flag) {
    if (update_account(arith, account, flag) != ERROR_SUCCESS) {
        return add_account(account, flag);
    } else {
        return ERROR_SUCCESS;
    }
}

__host__ __device__ int32_t
state_access_t::update_account(ArithEnv &arith, const CuEVM::account_t &account,
                               const CuEVM::account_flags_t flag) {
    bn_t target_address;
    cgbn_load(arith.env, target_address,
              (cgbn_evm_word_t_ptr) & (account.address));
    uint32_t index = 0;
    if (state_t::get_account_index(arith, target_address, index) ==
        ERROR_SUCCESS) {
        accounts[index].update(arith, account, flag);
        flags[index].update(flag);
        return ERROR_SUCCESS;
    }
    return ERROR_STATE_ADDRESS_NOT_FOUND;
}

__host__ __device__ int32_t
state_access_t::update(ArithEnv &arith, const state_access_t &other) {
    for (uint32_t i = 0; i < other.no_accounts; i++) {
        // if update failed (not exist), add the account
        if (update_account(arith, other.accounts[i], other.flags[i]) !=
            ERROR_SUCCESS) {
            add_account(other.accounts[i], other.flags[i]);
            flags[no_accounts - 1] = other.flags[i];
        }
    }
    return ERROR_SUCCESS;
}

__host__ __device__ void state_access_t::print() {
    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    printf("no_accounts: %u\n", no_accounts);
    for (uint32_t idx = 0; idx < no_accounts; idx++) {
        printf("accounts[%u]:\n", idx);
        accounts[idx].print();
        printf("flags[%u]:\n", idx);
        flags[idx].print();
    }
    __ONE_GPU_THREAD_WOSYNC_END__
}

__host__ cJSON *state_access_t::to_json() {
    cJSON *state_json = nullptr;
    cJSON *account_json = nullptr;
    char *hex_string_ptr = new char[CuEVM::word_size * 2 + 3];
    char *flag_string_ptr = new char[sizeof(uint32_t) * 2 + 3];
    state_json = cJSON_CreateObject();
    for (uint32_t idx = 0; idx < no_accounts; idx++) {
        accounts[idx].address.to_hex(hex_string_ptr, 0, 5);
        account_json = accounts[idx].to_json();
        cJSON_AddStringToObject(account_json, "flags",
                                flags[idx].to_hex(flag_string_ptr));
        cJSON_AddItemToObject(state_json, hex_string_ptr, account_json);
    }
    delete[] hex_string_ptr;
    hex_string_ptr = nullptr;
    delete[] flag_string_ptr;
    flag_string_ptr = nullptr;
    return state_json;
}

__host__ int32_t state_access_t::get_account_index_evm(
    const evm_word_t &address, uint32_t &index) const {
    for (index = 0; index < no_accounts; index++) {
        if (accounts[index].address == address) {
            return ERROR_SUCCESS;
        }
    }
    return ERROR_STATE_ADDRESS_NOT_FOUND;
}

__host__ cJSON *state_access_t::merge_json(const state_t &state1,
                                           const state_access_t &state2) {
    cJSON *state_json = nullptr;
    cJSON *account_json = nullptr;
    cJSON *accounts_json = nullptr;

    state_json = cJSON_CreateObject();
    accounts_json = cJSON_CreateArray();
    cJSON_AddItemToObject(state_json, "accounts", accounts_json);
    uint8_t *writen_accounts;
    writen_accounts = new uint8_t[state2.no_accounts];
    std::fill(writen_accounts, writen_accounts + state2.no_accounts, 0);
    const CuEVM::account_t *account1_ptr = nullptr;
    const CuEVM::account_t *account2_ptr = nullptr;
    uint32_t jdx = 0;
    for (uint32_t idx = 0; idx < state1.no_accounts; idx++) {
        account1_ptr = &(state1.accounts[idx]);
        if (state2.get_account_index_evm(account1_ptr->address, jdx) ==
            ERROR_SUCCESS) {
            account2_ptr = &(state2.accounts[jdx]);
            account_json = CuEVM::account_t::merge_json(
                account1_ptr, account2_ptr, state2.flags[jdx]);
            cJSON_AddItemToArray(accounts_json, account_json);
            writen_accounts[jdx] = 1;
        } else {
            account_json = CuEVM::account_t::merge_json(
                account1_ptr, account2_ptr, ACCOUNT_NONE_FLAG);
            cJSON_AddItemToArray(accounts_json, account_json);
        }
    }
    for (jdx = 0; jdx < state2.no_accounts; jdx++) {
        if (writen_accounts[jdx] == 0) {
            account2_ptr = &(state2.accounts[jdx]);
            account_json = CuEVM::account_t::merge_json(
                account1_ptr, account2_ptr, ACCOUNT_ALL_FLAG);
            cJSON_AddItemToArray(accounts_json, account_json);
        }
    }
    delete[] writen_accounts;
    return state_json;
}
}  // namespace CuEVM