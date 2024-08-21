// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#ifndef _STATE_T_H_
#define _STATE_T_H_

#include "account.cuh"
#include "../utils/arith.cuh"
#include <cjson/cJSON.h>

namespace cuEVM {
    namespace state {
        /**
         * The state struct.
         * It contains the accounts and the number of accounts.
         */
        struct state_t {
            cuEVM::account::account_t *accounts; /**< The accounts */
            uint32_t no_accounts; /**< The number of accounts */

            /**
             * The default constructor.
             * It initializes the accounts to nullptr and the no_accounts to 0.
             */
            __host__ __device__ state_t() : accounts(nullptr), no_accounts(0) {}

            /**
             * The constructor with the accounts and the number of accounts.
             * @param[in] accounts The accounts.
             * @param[in] no_accounts The number of accounts.
             */
            __host__ __device__ state_t(cuEVM::account::account_t *accounts, uint32_t no_accounts)
                : accounts(accounts), no_accounts(no_accounts) {}
            
            /**
             * The destructor.
             */
            __host__ __device__ ~state_t();

            /**
             * The copy constructor.
             * @param[in] other The other state.
             */
            __host__ __device__ state_t(const state_t &other);

            /**
             * The constructor from the JSON.
             * @param[in] json The JSON.
             * @param[in] managed The managed flag.
             */
            __host__ state_t(const cJSON *json, int32_t managed = 0);

            /**
             * The assignment operator.
             * @param[in] other The other state.
             * @return The current state.
             */
            __host__ __device__ state_t &operator=(const state_t &other);

            /**
             * The duplicate function.
             * @param[in] other The other state.
             */
            __host__ __device__ void duplicate(const state_t &other);

            /**
             * The free function.
             */
            __host__ __device__ void free();

            /**
             * The get account index function.
             * @param[in] arith The arithmetic environment.
             * @param[in] address The address.
             * @param[out] index The index.
             * @return If found 0. otherwise error.
             */
            __host__ __device__ int32_t get_account_index(
                ArithEnv &arith,
                const bn_t &address,
                uint32_t &index);

            /**
             * The get account function.
             * @param[in] arith The arithmetic environment.
             * @param[in] address The address.
             * @param[out] account The account.
             * @return If found 0. otherwise error.
             */
            __host__ __device__ int32_t get_account(
                ArithEnv &arith,
                const bn_t &address,
                cuEVM::account::account_t &account);

            /**
             * The get account function.
             * @param[in] arith The arithmetic environment.
             * @param[in] address The address.
             * @param[out] account_ptr The account pointer.
             * @return If found 0. otherwise error.
             */
            __host__ __device__ int32_t get_account(
                ArithEnv &arith,
                const bn_t &address,
                cuEVM::account::account_t* &account_ptr);

            /**
             * The add account function.
             * @param[in] account The account.
             * @return If added 0. otherwise error.
             */
            __host__ __device__ int32_t add_account(
                const cuEVM::account::account_t &account);

            /**
             * The set account function.
             * @param[in] arith The arithmetic environment.
             * @param[in] account The account.
             * @return If set 0. otherwise error.
             */
            __host__ __device__ int32_t set_account(
                ArithEnv &arith,
                const cuEVM::account::account_t &account);

            /**
             * The has account function.
             * @param[in] arith The arithmetic environment.
             * @param[in] address The address.
             * @return If has 1. otherwise 0.
             */
            __host__ __device__ int32_t has_account(
                ArithEnv &arith,
                const bn_t &address);

            /**
             * The update account function.
             * @param[in] arith The arithmetic environment.
             * @param[in] account The account.
             * @return If updated 0. otherwise error.
             */
            __host__ __device__ int32_t update_account(
                ArithEnv &arith,
                const cuEVM::account::account_t &account);

            /**
             * Get the account from the JSON.
             * @param[in] state_json The state JSON.
             * @param[in] managed The managed flag.
             * @return If successful 1. otherwise 0.
             */
            __host__ int32_t from_json(
                const cJSON *state_json,
                int32_t managed = 0);

            /**
             * The print function.
             * It prints the state.
             */
            __host__ __device__ void print();

            /**
             * The to JSON function.
             * @return The JSON.
             */
            __host__ cJSON *to_json();
        };

        /**
         * The state access struct.
         * It extends the state struct with the flags for each account.
         */
        struct state_access_t : state_t {
            cuEVM::account::account_flags_t* flags; /**< The flags fro each account */

            /**
             * The default constructor.
             */
            __host__ __device__ state_access_t() : state_t(), flags(nullptr) {}

            /**
             * The constructor with the accounts, the number of accounts and the flags.
             * @param[in] accounts The accounts.
             * @param[in] no_accounts The number of accounts.
             * @param[in] flags The flags.
             */
            __host__ __device__ state_access_t(
                cuEVM::account::account_t *accounts,
                uint32_t no_accounts,
                cuEVM::account::account_flags_t* flags)
                : state_t(accounts, no_accounts), flags(flags) {}
            
            /**
             * THe copy constructor.
             * @param[in] other The other state access.
             */
            __host__ __device__ state_access_t(const state_access_t &other) {
                free();
                duplicate(other);
            }

            /**
             * The destructor.
             */
            __host__ __device__ ~state_access_t() {
                free();
            }

            /**
             * The assignment operator.
             * @param[in] other The other state access.
             * @return The current state access.
             */
            __host__ __device__ state_access_t &operator=(const state_access_t &other) {
                if (this != &other) {
                    free();
                    duplicate(other);
                    return *this;
                }
            }

            /**
             * The duplicate function.
             * @param[in] other The other state access.
             */
            __host__ __device__ void duplicate(const state_access_t &other);

            /**
             * The free function.
             */
            __host__ __device__ void free();

            /**
             * The get account function.
             * @param[in] arith The arithmetic environment.
             * @param[in] address The address.
             * @param[out] account The account.
             * @param[in] flag The flag access.
             * @return If found 0. otherwise error.
             */
            __host__ __device__ int32_t get_account(
                ArithEnv &arith,
                const bn_t &address,
                cuEVM::account::account_t &account,
                const cuEVM::account::account_flags_t flag = ACCOUNT_NONE_FLAG);

            /**
             * The get account function.
             * @param[in] arith The arithmetic environment.
             * @param[in] address The address.
             * @param[out] account_ptr The account pointer.
             * @param[in] flag The flag access.
             * @return If found 0. otherwise error.
             */
            __host__ __device__ int32_t get_account(
                ArithEnv &arith,
                const bn_t &address,
                cuEVM::account::account_t* &account_ptr,
                const cuEVM::account::account_flags_t flag = ACCOUNT_NONE_FLAG);

            /**
             * The add account function.
             * @param[in] account The account.
             * @param[in] flag The flag access.
             * @return If added 0. otherwise error.
             */
            __host__ __device__ int32_t add_account(
                const cuEVM::account::account_t &account,
                const cuEVM::account::account_flags_t flag = ACCOUNT_NONE_FLAG);

            /**
             * The add duplicate account function.
             * @param[out] account_ptr The account pointer.
             * @param[in] src_account_ptr The source account pointer.
             * @param[in] flag The flag access.
             * @return If added 0. otherwise error.
             */
            __host__ __device__ int32_t add_duplicate_account(
                cuEVM::account::account_t* &account_ptr,
                cuEVM::account::account_t* &src_account_ptr,
                const cuEVM::account::account_flags_t flag);

            /**
             * The add new account function.
             * @param[in] arith The arithmetic environment.
             * @param[in] address The address.
             * @param[out] account_ptr The account pointer.
             * @param[in] flag The flag access.
             * @return If added 0. otherwise error.
             */
            __host__ __device__ int32_t add_new_account(
                ArithEnv &arith,
                const bn_t &address,
                cuEVM::account::account_t* &account_ptr,
                const cuEVM::account::account_flags_t flag);

            /**
             * The set account function.
             * @param[in] arith The arithmetic environment.
             * @param[in] account The account.
             * @param[in] flag The flag access.
             * @return If added 0. otherwise error.
             */
            __host__ __device__ int32_t set_account(
                ArithEnv &arith,
                const cuEVM::account::account_t &account,
                const cuEVM::account::account_flags_t flag = ACCOUNT_ALL_FLAG);

            /**
             * The update account function.
             * @param[in] arith The arithmetic environment.
             * @param[in] account The account.
             * @param[in] flag The flag access.
             * @return If added 0. otherwise error.
             */
            __host__ __device__ int32_t update_account(
                ArithEnv &arith,
                const cuEVM::account::account_t &account,
                const cuEVM::account::account_flags_t flag = ACCOUNT_ALL_FLAG);

            /**
             * The update state function.
             * @param[in] arith The arithmetic environment.
             * @param[in] other The other state access.
             * @return If added 0. otherwise error.
             */
            __host__ __device__ int32_t update(
                ArithEnv &arith,
                const state_access_t &other);

            /**
             * The print function.
             */
            __host__ __device__ void print();

            /**
             * The to JSON function.
             * @return The JSON.
             */
            __host__ cJSON* to_json();

        };

    }
}

#endif