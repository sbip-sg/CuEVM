// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#ifndef _STATE_T_H_
#define _STATE_T_H_

#include "account.cuh"
#include "arith.cuh"
#include <CuCrypto/keccak.cuh>
#include <cjson/cJSON.h>

namespace cuEVM {
    namespace state {
        struct state_t {
            cuEVM::account::account_t *accounts;
            uint32_t no_accounts;

            __host__ __device__ state_t() : accounts(nullptr), no_accounts(0) {}

            __host__ __device__ state_t(cuEVM::account::account_t *accounts, uint32_t no_accounts)
                : accounts(accounts), no_accounts(no_accounts) {}
            
            __host__ __device__ ~state_t();

            __host__ __device__ state_t(const state_t &other);

            __host__ state_t(const cJSON *json, int32_t managed = 0);

            __host__ __device__ state_t &operator=(const state_t &other);

            __host__ __device__ void duplicate(const state_t &other);

            __host__ __device__ void free();

            __host__ __device__ int32_t get_account_index(
                ArithEnv arith,
                const bn_t &address,
                uint32_t &index);

            __host__ __device__ int32_t get_account(
                ArithEnv arith,
                const bn_t &address,
                cuEVM::account::account_t &account);

            __host__ __device__ int32_t get_account(
                ArithEnv arith,
                const bn_t &address,
                cuEVM::account::account_t* &account_ptr);

            __host__ __device__ int32_t add_account(
                const cuEVM::account::account_t &account);

            __host__ __device__ int32_t set_account(
                ArithEnv arith,
                const cuEVM::account::account_t &account);

            __host__ __device__ int32_t has_account(
                ArithEnv arith,
                const bn_t &address);

            __host__ __device__ int32_t update_account(
                ArithEnv arith,
                const cuEVM::account::account_t &account);

            __host__ int32_t from_json(
                const cJSON *state_json,
                int32_t managed = 0);

            __host__ __device__ void print();

            __host__ cJSON *to_json();
        };

        struct state_with_flags_t {
            state_t state;
            cuEVM::account::account_flags_t* flags;

            __host__ __device__ state_with_flags_t() : state(), flags(nullptr) {}

            __host__ __device__ state_with_flags_t(
                const state_t &state,
                cuEVM::account::account_flags_t* flags)
                : state(state), flags(flags) {}
            
            __host__ __device__ state_with_flags_t(const state_with_flags_t &other) {
                free();
                state = other.state;
                if (state.no_accounts > 0) {
                    flags = new cuEVM::account::account_flags_t[state.no_accounts];
                    std::copy(other.flags, other.flags + state.no_accounts, flags);
                } else {
                    flags = nullptr;
                }
            }

            __host__ __device__ ~state_with_flags_t() {
                free();
            }

            __host__ __device__ state_with_flags_t &operator=(const state_with_flags_t &other) {
                free();
                state = other.state;
                if (state.no_accounts > 0) {
                    flags = new cuEVM::account::account_flags_t[state.no_accounts];
                    std::copy(other.flags, other.flags + state.no_accounts, flags);
                } else {
                    flags = nullptr;
                }
                return *this;
            }

            __host__ __device__ void free() {
                if (flags != nullptr && state.no_accounts > 0) {
                    delete[] flags;
                    flags = nullptr;
                }
                state.free();
            }

            __host__ __device__ int32_t get_account(
                ArithEnv &arith,
                const bn_t &address,
                cuEVM::account::account_t &account,
                cuEVM::account::account_flags_t flag = ACCOUNT_NONE_FLAG) {
                uint32_t index = 0;
                if(state.get_account_index(arith, address, index)) {
                    flags[index].update(flag);
                    account = state.accounts[index];
                    return 1;
                }
                return 0;
            }

            __host__ __device__ int32_t set_account(
                ArithEnv &arith,
                const cuEVM::account::account_t &account,
                cuEVM::account::account_flags_t flag = ACCOUNT_NONE_FLAG) {
                uint32_t index = 0;
                bn_t address;
                cgbn_load(arith.env, address, (cgbn_evm_word_t_ptr) &account.address);
                if(state.get_account_index(arith, address, index)) {
                    flags[index].update(flag);
                    state.accounts[index] = account;
                    return 1;
                }
                state.add_account(account);
                index = state.no_accounts - 1;
                flags[index] = flag;
                return 1;
            }

            __host__ __device__ int32_t has_account(
                ArithEnv &arith,
                const bn_t &address) {
                return state.has_account(arith, address);
            }

            __host__ __device__ int32_t update(
                ArithEnv &arith,
                state_with_flags_t &other) {
                for (uint32_t i = 0; i < other.state.no_accounts; i++) {
                    
                }
            }
            
        };

        struct state_access_t : state_t {
            cuEVM::account::account_flags_t* flags;

            __host__ __device__ state_access_t() : state_t(), flags(nullptr) {}

            __host__ __device__ state_access_t(
                cuEVM::account::account_t *accounts,
                uint32_t no_accounts,
                cuEVM::account::account_flags_t* flags)
                : state_t(accounts, no_accounts), flags(flags) {}
            
            __host__ __device__ state_access_t(const state_access_t &other) {
                duplicate(other);
            }

            __host__ __device__ ~state_access_t() {
                free();
            }

            __host__ __device__ state_access_t &operator=(const state_access_t &other) {
                if (this != &other) {
                    free();
                    duplicate(other);
                    return *this;
                }
            }

            __host__ __device__ void duplicate(const state_access_t &other) {
                state_t::duplicate(other);
                if (no_accounts > 0) {
                    flags = new cuEVM::account::account_flags_t[no_accounts];
                    std::copy(other.flags, other.flags + no_accounts, flags);
                } else {
                    flags = nullptr;
                }
            }

            __host__ __device__ void free() {
                if (flags != nullptr && no_accounts > 0) {
                    delete[] flags;
                    flags = nullptr;
                }
                state_t::free();
            }

            __host__ __device__ int32_t get_account(
                ArithEnv &arith,
                const bn_t &address,
                cuEVM::account::account_t &account,
                cuEVM::account::account_flags_t flag = ACCOUNT_NONE_FLAG) {
                uint32_t index = 0;
                if(state_t::get_account_index(arith, address, index)) {
                    flags[index].update(flag);
                    account = accounts[index];
                    return 1;
                }
                return 0;
            }


            __host__ __device__ int32_t get_account(
                ArithEnv &arith,
                const bn_t &address,
                cuEVM::account::account_t* &account_ptr,
                cuEVM::account::account_flags_t flag = ACCOUNT_NONE_FLAG) {
                uint32_t index = 0;
                if(state_t::get_account_index(arith, address, index)) {
                    flags[index].update(flag);
                    account_ptr = &accounts[index];
                    return 1;
                }
                return 0;
            }

            __host__ __device__ int32_t set_account(
                ArithEnv &arith,
                const cuEVM::account::account_t &account,
                cuEVM::account::account_flags_t flag = ACCOUNT_ALL_FLAG) {
                uint32_t index = 0;
                bn_t address;
                cgbn_load(arith.env, address, (cgbn_evm_word_t_ptr) &account.address);
                if(state_t::get_account_index(arith, address, index)) {
                    flags[index].update(flag);
                    accounts[index] = account;
                    return 1;
                }
                state_t::add_account(account);
                index = no_accounts - 1;
                flags[index] = flag;
                return 1;
            }

            __host__ __device__ int32_t update_account(
                ArithEnv &arith,
                const cuEVM::account::account_t &account,
                cuEVM::account::account_flags_t flag = ACCOUNT_ALL_FLAG) {
                bn_t target_address;
                cgbn_load(arith.env, target_address, (cgbn_evm_word_t_ptr) &(account.address));
                uint32_t index = 0;
                if(state_t::get_account_index(arith, target_address, index)) {
                    accounts[index].update(arith, account, flag);
                    flags[index].update(flag);
                    return 1;
                }
                return 0;
            }

            __host__ __device__ int32_t update(
                ArithEnv &arith,
                const state_access_t &other) {
                for (uint32_t i = 0; i < other.no_accounts; i++) {
                    if (update_account(arith, other.accounts[i], other.flags[i]) == 0) {
                        add_account(other.accounts[i]);
                        flags[no_accounts - 1] = other.flags[i];
                    }
                }
                return 1;
            }



            __host__ __device__ void print() {
                printf("no_accounts: %lu\n", no_accounts);
                for (uint32_t idx = 0; idx < no_accounts; idx++) {
                    printf("accounts[%lu]:\n", idx);
                    accounts[idx].print();
                    printf("flags[%lu]:\n", idx);
                    flags[idx].print();
                }
            }

            __host__ cJSON* to_json() {
                cJSON *state_json = nullptr;
                cJSON *account_json = nullptr;
                char *hex_string_ptr = new char[EVM_WORD_SIZE * 2 + 3];
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

        };

    }
}

#endif