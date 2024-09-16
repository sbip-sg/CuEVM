// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation
// Programme) Author: Stefan-Dan Ciocirlan Data: 2023-11-30
// SPDX-License-Identifier: MIT

#include <CuCrypto/keccak.cuh>
#include <CuEVM/state/state.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM {
__host__ __device__ state_t::~state_t() { free(); }

__host__ __device__ state_t::state_t(const state_t &other) { duplicate(other); }

__host__ __device__ state_t &state_t::operator=(const state_t &other) {
    if (this != &other) {
        free();
        duplicate(other);
    }
    return *this;
}

__host__ __device__ void state_t::free() {
    __ONE_GPU_THREAD_BEGIN__
    if (accounts != nullptr && no_accounts > 0) {
        delete[] accounts;
    }
    __ONE_GPU_THREAD_END__
    clear();
}

__host__ void state_t::free_managed() {
    if (accounts != nullptr && no_accounts > 0) {
        CUDA_CHECK(cudaFree(accounts));
    }
    clear();
}

__host__ __device__ void state_t::clear() {
    accounts = nullptr;
    no_accounts = 0;
}

__host__ __device__ void state_t::duplicate(const state_t &other) {
    __SHARED_MEMORY__ CuEVM::account_t *tmp_accounts;
    no_accounts = other.no_accounts;
    if (no_accounts > 0) {
        __ONE_GPU_THREAD_BEGIN__
        tmp_accounts = new CuEVM::account_t[no_accounts];
        __ONE_GPU_THREAD_END__
        for (uint32_t idx = 0; idx < no_accounts; idx++) {
            tmp_accounts[idx] = other.accounts[idx];
        }
    } else {
        tmp_accounts = nullptr;
    }
    accounts = tmp_accounts;
}

__host__ __device__ int32_t state_t::get_account_index(ArithEnv &arith,
                                                       const bn_t &address,
                                                       uint32_t &index) {
    for (index = 0; index < no_accounts; index++) {
        if (accounts[index].has_address(arith, address)) {
            return ERROR_SUCCESS;
        }
    }
    return ERROR_STATE_ADDRESS_NOT_FOUND;
}

__host__ __device__ int32_t state_t::get_account(ArithEnv &arith,
                                                 const bn_t &address,
                                                 CuEVM::account_t &account) {
    uint32_t index;
    if (get_account_index(arith, address, index) == ERROR_SUCCESS) {
        account = accounts[index];
        return ERROR_SUCCESS;
    }
    return ERROR_STATE_ADDRESS_NOT_FOUND;
}

__host__ __device__ int32_t state_t::get_account(
    ArithEnv &arith, const bn_t &address, CuEVM::account_t *&account_ptr) {
    uint32_t index;
    if (get_account_index(arith, address, index) == ERROR_SUCCESS) {
        account_ptr = &accounts[index];
        return ERROR_SUCCESS;
    }
    return ERROR_STATE_ADDRESS_NOT_FOUND;
}

__host__ __device__ int32_t
state_t::add_account(const CuEVM::account_t &account) {
    __SHARED_MEMORY__ CuEVM::account_t *tmp_accounts;
    __ONE_GPU_THREAD_BEGIN__
    tmp_accounts = new CuEVM::account_t[no_accounts + 1];
    memcpy(tmp_accounts, accounts, no_accounts * sizeof(CuEVM::account_t));
    __ONE_GPU_THREAD_END__
    tmp_accounts[no_accounts].clear();
    tmp_accounts[no_accounts] = account;
    if (accounts != nullptr) {
        for (uint32_t idx = 0; idx < no_accounts; idx++) {
            accounts[idx].byte_code.clear();
            accounts[idx].storage.clear();
        }
        __ONE_GPU_THREAD_BEGIN__
        delete[] accounts;
        __ONE_GPU_THREAD_END__
    }
    accounts = tmp_accounts;
    no_accounts++;
    return ERROR_SUCCESS;
}

__host__ __device__ int32_t
state_t::set_account(ArithEnv &arith, const CuEVM::account_t &account) {
    bn_t target_address;
    cgbn_load(arith.env, target_address,
              (cgbn_evm_word_t_ptr) & (account.address));
    for (uint32_t idx = 0; idx < no_accounts; idx++) {
        if (accounts[idx].has_address(arith, target_address)) {
            accounts[idx] = account;
            return ERROR_SUCCESS;
        }
    }

    return add_account(account);
}

__host__ __device__ int32_t state_t::has_account(ArithEnv &arith,
                                                 const bn_t &address) {
    for (uint32_t idx = 0; idx < no_accounts; idx++) {
        if (accounts[idx].has_address(arith, address)) {
            return 1;
        }
    }
    return 0;
}

__host__ __device__ int32_t
state_t::update_account(ArithEnv &arith, const CuEVM::account_t &account) {
    bn_t target_address;
    cgbn_load(arith.env, target_address,
              (cgbn_evm_word_t_ptr) & (account.address));
    for (uint32_t idx = 0; idx < no_accounts; idx++) {
        if (accounts[idx].has_address(arith, target_address)) {
            accounts[idx].update(arith, account);
            return ERROR_SUCCESS;
        }
    }
    return add_account(account);
}

__host__ int32_t state_t::from_json(const cJSON *state_json, int32_t managed) {
    free();
    // if (!cJSON_IsArray(state_json)) return 0;
    no_accounts = cJSON_GetArraySize(state_json);
    if (no_accounts == 0) return 1;
    if (managed) {
        CUDA_CHECK(cudaMallocManaged((void **)&(accounts),
                                     no_accounts * sizeof(CuEVM::account_t)));
    } else {
        accounts = new CuEVM::account_t[no_accounts];
    }
    uint32_t idx = 0;
    cJSON *account_json;
    cJSON_ArrayForEach(account_json, state_json) {
        accounts[idx++].from_json(account_json, managed);
    }
    return ERROR_SUCCESS;
}

__host__ __device__ void state_t::print() {
    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    printf("no_accounts: %u\n", no_accounts);
    for (uint32_t idx = 0; idx < no_accounts; idx++) {
        printf("accounts[%u]:\n", idx);
        accounts[idx].print();
    }
    __ONE_GPU_THREAD_WOSYNC_END__
}

__host__ cJSON *state_t::to_json() {
    cJSON *state_json = nullptr;
    cJSON *account_json = nullptr;
    char *hex_string_ptr = new char[CuEVM::word_size * 2 + 3];
    state_json = cJSON_CreateObject();
    for (uint32_t idx = 0; idx < no_accounts; idx++) {
        accounts[idx].address.to_hex(hex_string_ptr, 0, 5);
        account_json = accounts[idx].to_json();
        cJSON_AddItemToObject(state_json, hex_string_ptr, account_json);
    }
    delete[] hex_string_ptr;
    hex_string_ptr = nullptr;
    return state_json;
}

}  // namespace CuEVM
