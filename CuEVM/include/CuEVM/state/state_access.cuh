// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation
// Programme) Author: Stefan-Dan Ciocirlan Date: 2024-09-15
// SPDX-License-Identifier: MIT

#pragma once

#include <CuEVM/state/account.cuh>
#include <CuEVM/state/account_flags.cuh>
#include <CuEVM/state/state.cuh>
#include <CuEVM/utils/arith.cuh>

namespace CuEVM {

/**
 * The state access struct.
 * It extends the state struct with the flags for each account.
 */
struct state_access_t : state_t {
    CuEVM::account_flags_t *flags; /**< The flags fro each account */

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
    // __host__ __device__ state_access_t(CuEVM::account_t *accounts,
    //                                    uint32_t no_accounts,
    //                                    CuEVM::account_flags_t *flags)
    //     : state_t(accounts, no_accounts), flags(flags) {}

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
    __host__ __device__ ~state_access_t() { free(); }

    /**
     * The free function.
     */
    __host__ __device__ void free();

    /**
     * The free managed function.
     */
    __host__ void free_managed();

    /**
     * The clear function.
     */
    __host__ __device__ void clear();
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
     * The get account function.
     * @param[in] arith The arithmetic environment.
     * @param[in] address The address.
     * @param[out] account The account.
     * @param[in] flag The flag access.
     * @return If found 0. otherwise error.
     */
    __host__ __device__ int32_t get_account(ArithEnv &arith, const bn_t &address, CuEVM::account_t &account,
                                            const CuEVM::account_flags_t flag = CuEVM::ACCOUNT_NONE_FLAG);

    /**
     * The get account function.
     * @param[in] arith The arithmetic environment.
     * @param[in] address The address.
     * @param[out] account_ptr The account pointer.
     * @param[in] flag The flag access.
     * @return If found 0. otherwise error.
     */
    __host__ __device__ int32_t get_account(ArithEnv &arith, const bn_t &address, CuEVM::account_t *&account_ptr,
                                            const CuEVM::account_flags_t flag = CuEVM::ACCOUNT_NONE_FLAG);

    /**
     * The get account index function.
     * @param[in] address The address.
     * @param[out] index The index.
     * @return If found 0. otherwise error.
     */
    __host__ int32_t get_account_index_evm(const evm_word_t &address, uint32_t &index) const;

    /**
     * The add account function.
     * @param[in] account The account.
     * @param[in] flag The flag access.
     * @return If added 0. otherwise error.
     */
    __host__ __device__ int32_t add_account(const CuEVM::account_t &account,
                                            const CuEVM::account_flags_t flag = CuEVM::ACCOUNT_NONE_FLAG);

    /**
     * The add duplicate account function.
     * @param[out] account_ptr The account pointer.
     * @param[in] src_account_ptr The source account pointer.
     * @param[in] flag The flag access.
     * @return If added 0. otherwise error.
     */
    __host__ __device__ int32_t add_duplicate_account(CuEVM::account_t *&account_ptr,
                                                      CuEVM::account_t *&src_account_ptr,
                                                      const CuEVM::account_flags_t flag = CuEVM::ACCOUNT_NONE_FLAG);

    /**
     * The add new account function.
     * @param[in] arith The arithmetic environment.
     * @param[in] address The address.
     * @param[out] account_ptr The account pointer.
     * @param[in] flag The flag access.
     * @return If added 0. otherwise error.
     */
    __host__ __device__ int32_t add_new_account(ArithEnv &arith, const bn_t &address, CuEVM::account_t *&account_ptr,
                                                const CuEVM::account_flags_t flag = CuEVM::ACCOUNT_NONE_FLAG);

    /**
     * The set account function.
     * @param[in] arith The arithmetic environment.
     * @param[in] account The account.
     * @param[in] flag The flag access.
     * @return If added 0. otherwise error.
     */
    __host__ __device__ int32_t set_account(ArithEnv &arith, const CuEVM::account_t &account,
                                            const CuEVM::account_flags_t flag = CuEVM::ACCOUNT_ALL_FLAG);

    /**
     * The update account function.
     * @param[in] arith The arithmetic environment.
     * @param[in] account The account.
     * @param[in] flag The flag access.
     * @return If added 0. otherwise error.
     */
    __host__ __device__ int32_t update_account(ArithEnv &arith, const CuEVM::account_t &account,
                                               const CuEVM::account_flags_t flag = CuEVM::ACCOUNT_ALL_FLAG);

    /**
     * The update state function.
     * @param[in] arith The arithmetic environment.
     * @param[in] other The other state access.
     * @return If added 0. otherwise error.
     */
    __host__ __device__ int32_t update(ArithEnv &arith, const state_access_t &other);

    /**
     * The print function.
     */
    __host__ __device__ void print();

    /**
     * The to JSON function.
     * @return The JSON.
     */
    __host__ cJSON *to_json();

    // STATIC FUNCTIONS
    /**
     * The merge of two states, the first state is the base state.
     * @param[in] state1 The first state.
     * @param[in] state2 The second state.
     * @return The merged state in JSON.
     */
    __host__ static cJSON *merge_json(const state_t &state1, const state_access_t &state2);
};

}  // namespace CuEVM