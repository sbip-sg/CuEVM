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

            __host__ __device__ void free();

            __host__ __device__ int32_t get_account_index(
                ArithEnv arith,
                const bn_t &address,
                uint32_t &index);

            __host__ __device__ int32_t get_account(
                ArithEnv arith,
                const bn_t &address,
                cuEVM::account::account_t &account);

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

            __host__ __device__ void print(
                cuEVM::account::account_flags_t *flags = nullptr);

            __host__ cJSON *to_json(
                cuEVM::account::account_flags_t *flags = nullptr);
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

            __host__ __device__ void print() {
                state.print(flags);
            }

            __host__ cJSON *to_json() {
                return state.to_json(flags);
            }
            
        };
    }
}

#endif