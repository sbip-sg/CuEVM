// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#include <CuEVM/state/state.cuh>
#include <CuEVM/utils/error_codes.cuh>
#include <CuCrypto/keccak.cuh>

namespace CuEVM {
    namespace state {
        __host__ __device__ state_t::~state_t() {
            free();
        }


        __host__ __device__ state_t::state_t(const state_t &other) {
            duplicate(other);
        }


        __host__ state_t::state_t(const cJSON *json, int32_t managed ) {
            no_accounts = 0;
            accounts = nullptr;
            from_json(json, managed);
        }


        __host__ __device__ state_t& state_t::operator=(const state_t &other) {
            if (this != &other) {
                free();
                duplicate(other);
            }
            return *this;
        }

        __host__ __device__ void state_t::duplicate(const state_t &other) {
            no_accounts = other.no_accounts;
            if (no_accounts > 0) {
                accounts = new CuEVM::account::account_t[no_accounts];
                for (uint32_t idx = 0; idx < no_accounts; idx++) {
                    accounts[idx] = other.accounts[idx];
                }
            } else {
                accounts = nullptr;
            }
        }


        __host__ __device__ void state_t::free() {
            if (accounts != nullptr && no_accounts > 0) {
                delete[] accounts;
            }
            no_accounts = 0;
            accounts = nullptr;
        }
        __host__ __device__ int32_t state_t::get_account_index(
                ArithEnv &arith,
                const bn_t &address,
                uint32_t &index) {
            for (index = 0; index < no_accounts; index++) {
                if (accounts[index].has_address(arith, address)) {
                    return ERROR_SUCCESS;
                }
            }
            return ERROR_STATE_ADDRESS_NOT_FOUND;
        }


        __host__ __device__ int32_t state_t::get_account(
            ArithEnv &arith,
            const bn_t &address,
            CuEVM::account::account_t &account
        ) {
            uint32_t index;
            if (get_account_index(arith, address, index) == ERROR_SUCCESS) {
                account = accounts[index];
                return ERROR_SUCCESS;
            }
            return ERROR_STATE_ADDRESS_NOT_FOUND;
        }

        __host__ __device__ int32_t state_t::get_account(
            ArithEnv &arith,
            const bn_t &address,
            CuEVM::account::account_t* &account_ptr) {
            uint32_t index;
            if (get_account_index(arith, address, index) == ERROR_SUCCESS) {
                account_ptr = &accounts[index];
                return ERROR_SUCCESS;
            }
            return ERROR_STATE_ADDRESS_NOT_FOUND;

        }


        __host__ __device__ int32_t state_t::add_account(
            const CuEVM::account::account_t &account
        ) {
            CuEVM::account::account_t *tmp_accounts = new CuEVM::account::account_t[no_accounts + 1];
            std::copy(accounts, accounts + no_accounts, tmp_accounts);
            tmp_accounts[no_accounts] = account;
            if (accounts != nullptr) {
                delete[] accounts;
            }
            accounts = tmp_accounts;
            no_accounts++;
            return ERROR_SUCCESS;
        }


        __host__ __device__ int32_t state_t::set_account(
            ArithEnv &arith,
            const CuEVM::account::account_t &account
        ) {
            bn_t target_address;
            cgbn_load(arith.env, target_address, (cgbn_evm_word_t_ptr) &(account.address));
            for (uint32_t idx = 0; idx < no_accounts; idx++) {
                if (accounts[idx].has_address(arith, target_address)) {
                    accounts[idx] = account;
                    return ERROR_SUCCESS;
                }
            }

            return add_account(account);
        }


        __host__ __device__ int32_t state_t::has_account(
            ArithEnv &arith,
            const bn_t &address
        ) {
            for (uint32_t idx = 0; idx < no_accounts; idx++) {
                if (accounts[idx].has_address(arith, address)) {
                    return 1;
                }
            }
            return 0;
        }


        __host__ __device__ int32_t state_t::update_account(
            ArithEnv &arith,
            const CuEVM::account::account_t &account
        ) {
            bn_t target_address;
            cgbn_load(arith.env, target_address, (cgbn_evm_word_t_ptr) &(account.address));
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
            //if (!cJSON_IsArray(state_json)) return 0;
            no_accounts = cJSON_GetArraySize(state_json);
            if (no_accounts == 0) return 1;
            if (managed) {
                CUDA_CHECK(cudaMallocManaged(
                    (void **)&(accounts),
                    no_accounts * sizeof(CuEVM::account::account_t)
                ));
            } else {
                accounts = new CuEVM::account::account_t[no_accounts];
            }
            uint32_t idx = 0;
            cJSON *account_json;
            cJSON_ArrayForEach(account_json, state_json)
            {
                accounts[idx++].from_json(account_json, managed);
            }
            return 1;
        }


        __host__ __device__ void state_t::print() {
            printf("no_accounts: %lu\n", no_accounts);
            for (uint32_t idx = 0; idx < no_accounts; idx++) {
                printf("accounts[%lu]:\n", idx);
                accounts[idx].print();
            }
        }


        __host__ cJSON* state_t::to_json() {
            cJSON *state_json = nullptr;
            cJSON *account_json = nullptr;
            char *hex_string_ptr = new char[CuEVM::word_size * 2 + 3];
            char *flag_string_ptr = nullptr;
            state_json = cJSON_CreateObject();
            for(uint32_t idx = 0; idx < no_accounts; idx++) {
                accounts[idx].address.to_hex(hex_string_ptr, 0, 5);
                account_json = accounts[idx].to_json();
                cJSON_AddItemToObject(
                    state_json,
                    hex_string_ptr,
                    account_json);
            }
            delete[] hex_string_ptr;
            hex_string_ptr = nullptr;
            return state_json;
        }



        __host__ __device__ void state_access_t::duplicate(const state_access_t &other) {
            state_t::duplicate(other);
            if (no_accounts > 0) {
                flags = new CuEVM::account::account_flags_t[no_accounts];
                std::copy(other.flags, other.flags + no_accounts, flags);
            } else {
                flags = nullptr;
            }
        }

        __host__ __device__ void state_access_t::free() {
            if (flags != nullptr && no_accounts > 0) {
                delete[] flags;
                flags = nullptr;
            }
            state_t::free();
        }

        __host__ __device__ int32_t state_access_t::get_account(
            ArithEnv &arith,
            const bn_t &address,
            CuEVM::account::account_t &account,
            const CuEVM::account::account_flags_t flag) {
            uint32_t index = 0;
            if(state_t::get_account_index(arith, address, index)) {
                flags[index].update(flag);
                account = accounts[index];
                return ERROR_SUCCESS;
            }
            return ERROR_STATE_ADDRESS_NOT_FOUND;
        }


        __host__ __device__ int32_t state_access_t::get_account(
            ArithEnv &arith,
            const bn_t &address,
            CuEVM::account::account_t* &account_ptr,
            const CuEVM::account::account_flags_t flag) {
            uint32_t index = 0;
            if(state_t::get_account_index(arith, address, index) == ERROR_SUCCESS) {
                flags[index].update(flag);
                account_ptr = &accounts[index];
                return ERROR_SUCCESS;
            }
            return ERROR_STATE_ADDRESS_NOT_FOUND;
        }

        __host__ __device__ int32_t state_access_t::add_account(
            const CuEVM::account::account_t &account,
            const CuEVM::account::account_flags_t flag) {
            state_t::add_account(account);
            uint32_t index = no_accounts - 1;
            CuEVM::account::account_flags_t *tmp_flags = new CuEVM::account::account_flags_t[no_accounts];
            std::copy(flags, flags + no_accounts - 1, tmp_flags);
            if (flags != nullptr) {
                delete[] flags;
            }
            flags = tmp_flags;
            flags[index] = flag;
            return ERROR_SUCCESS;
        }

        __host__ __device__ int32_t state_access_t::add_duplicate_account(
            CuEVM::account::account_t* &account_ptr,
            CuEVM::account::account_t* &src_account_ptr,
            const CuEVM::account::account_flags_t flag) {
            CuEVM::account::account_flags_t no_storage_copy(ACCOUNT_NON_STORAGE_FLAG);
            uint32_t index = 0;
            account_ptr = new CuEVM::account::account_t(
                src_account_ptr,
                no_storage_copy);
            return add_account(*account_ptr, flag);
        }

        __host__ __device__ int32_t state_access_t::add_new_account(
            ArithEnv &arith,
            const bn_t &address,
            CuEVM::account::account_t* &account_ptr,
            const CuEVM::account::account_flags_t flag) {
            uint32_t index = 0;
            account_ptr = new CuEVM::account::account_t(
                arith,
                address);
            return add_account(*account_ptr, flag);
        }

        __host__ __device__ int32_t state_access_t::set_account(
            ArithEnv &arith,
            const CuEVM::account::account_t &account,
            const CuEVM::account::account_flags_t flag) {
            uint32_t index = 0;
            if (update_account(arith, account, flag)) {
                return add_account(account, flag);
            } else {
                return ERROR_SUCCESS;
            }
        }

        __host__ __device__ int32_t state_access_t::update_account(
            ArithEnv &arith,
            const CuEVM::account::account_t &account,
            const CuEVM::account::account_flags_t flag) {
            bn_t target_address;
            cgbn_load(arith.env, target_address, (cgbn_evm_word_t_ptr) &(account.address));
            uint32_t index = 0;
            if(state_t::get_account_index(arith, target_address, index) == ERROR_SUCCESS) {
                accounts[index].update(arith, account, flag);
                if (flags != nullptr){
                    flags[index].update(flag);
                }

                return ERROR_SUCCESS;
            }
            return ERROR_STATE_ADDRESS_NOT_FOUND;
        }

        __host__ __device__ int32_t state_access_t::update(
            ArithEnv &arith,
            const state_access_t &other) {
            uint32_t index = 0;
            for (uint32_t i = 0; i < other.no_accounts; i++) {
                // if update failed (not exist), add the account
                if (update_account(arith, other.accounts[i], other.flags[i]) != ERROR_SUCCESS) {
                    add_account(other.accounts[i], index);
                    flags[no_accounts - 1] = other.flags[i];
                }
            }
            return ERROR_SUCCESS;
        }



        __host__ __device__ void state_access_t::print() {
            printf("no_accounts: %lu\n", no_accounts);
            for (uint32_t idx = 0; idx < no_accounts; idx++) {
                printf("accounts[%lu]:\n", idx);
                accounts[idx].print();
                printf("flags[%lu]:\n", idx);
                flags[idx].print();
            }
        }

        __host__ cJSON* state_access_t::to_json() {
            cJSON *state_json = nullptr;
            cJSON *account_json = nullptr;
            char *hex_string_ptr = new char[CuEVM::word_size * 2 + 3];
            char *flag_string_ptr = new char[sizeof(uint32_t) * 2 + 3];
            state_json = cJSON_CreateObject();
            for(uint32_t idx = 0; idx < no_accounts; idx++) {
                accounts[idx].address.to_hex(hex_string_ptr, 0, 5);
                account_json = accounts[idx].to_json();
                cJSON_AddStringToObject(
                    account_json,
                    "flags",
                    flags[idx].to_hex(flag_string_ptr)
                );
                cJSON_AddItemToObject(
                    state_json,
                    hex_string_ptr,
                    account_json);
            }
            delete[] hex_string_ptr;
            hex_string_ptr = nullptr;
            delete[] flag_string_ptr;
            flag_string_ptr = nullptr;
            return state_json;
        }

        __host__ int32_t state_access_t::get_account_index_evm(
            const evm_word_t &address,
            uint32_t &index) const {
            for (index = 0; index < no_accounts; index++) {
                if (accounts[index].address == address) {
                    return ERROR_SUCCESS;
                }
            }
            return ERROR_STATE_ADDRESS_NOT_FOUND;
        }

        __host__ cJSON* state_merge_json(
            const state_t &state1,
            const state_access_t &state2
        ) {
            cJSON *state_json = nullptr;
            cJSON *account_json = nullptr;
            cJSON *accounts_json = nullptr;


            state_json = cJSON_CreateObject();
            accounts_json = cJSON_CreateArray();
            cJSON_AddItemToObject(state_json, "accounts", accounts_json);
            uint8_t *writen_accounts;
            writen_accounts = new uint8_t[state2.no_accounts];
            std::fill(writen_accounts, writen_accounts + state2.no_accounts, 0);
            const CuEVM::account::account_t *account1_ptr = nullptr;
            const CuEVM::account::account_t *account2_ptr = nullptr;
            uint32_t jdx = 0;
            for (uint32_t idx = 0; idx < state1.no_accounts; idx++) {
                account1_ptr = &(state1.accounts[idx]);
                if (state2.get_account_index_evm(account1_ptr->address, jdx) == ERROR_SUCCESS) {
                    account2_ptr = &(state2.accounts[jdx]);
                    account_json = CuEVM::account::account_merge_json(
                        account1_ptr,
                        account2_ptr,
                        state2.flags[jdx]);
                    cJSON_AddItemToArray(accounts_json, account_json);
                    writen_accounts[jdx] = 1;
                } else {
                    account_json = CuEVM::account::account_merge_json(
                        account1_ptr,
                        account2_ptr,
                        ACCOUNT_NONE_FLAG);
                    cJSON_AddItemToArray(accounts_json, account_json);
                }
            }
            for (jdx = 0; jdx < state2.no_accounts; jdx++) {
                if (writen_accounts[jdx] == 0) {
                    account2_ptr = &(state2.accounts[jdx]);
                    account_json = CuEVM::account::account_merge_json(
                        account1_ptr,
                        account2_ptr,
                        ACCOUNT_ALL_FLAG);
                    cJSON_AddItemToArray(accounts_json, account_json);
                }
            }
            delete[] writen_accounts;
            return state_json;
        }
    }
}
