// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation
// Programme) Author: Stefan-Dan Ciocirlan Data: 2024-06-20
// SPDX-License-Identifier: MIT

#pragma once

#include <CuEVM/state/account.cuh>
#include <CuEVM/state/state_access.cuh>
#include <CuEVM/state/world_state.cuh>
#include <CuEVM/utils/arith.cuh>

namespace CuEVM {
/**
 * @brief The AccessState class
 *
 */
class AccessState {
   private:
    state_access_t *_state;   /**< The state access */
    WorldState *_world_state; /**< The world state */

    /**
     * Add an account to the state.
     * @param[in] arith The arithmetic environment.
     * @param[in] address The address of the account.
     * @param[out] account_ptr The pointer to the account.
     * @param[in] flag The account access flags.
     * @return 0 if the account is added successfully, error otherwise.
     */
    __host__ __device__ int32_t add_account(ArithEnv &arith,
                                            const bn_t &address,
                                            CuEVM::account_t *&account_ptr,
                                            const CuEVM::account_flags_t flag);

   public:
    /**
     * The default constructor.
     */
    __host__ __device__ AccessState()
        : _state(nullptr), _world_state(nullptr) {}

    /**
     * The constructor with the state and the world state.
     * @param[in] state The state access.
     * @param[in] world_state The world state.
     */
    __host__ __device__ AccessState(state_access_t *state,
                                    WorldState *world_state)
        : _state(state), _world_state(world_state) {}

    /**
     * The destructor.
     */
    __host__ __device__ ~AccessState() {
        _state = nullptr;
        _world_state = nullptr;
    }

    /**
     * Get an account from the state.
     * @param[in] arith The arithmetic environment.
     * @param[in] address The address of the account.
     * @param[out] account_ptr The pointer to the account.
     * @param[in] flag The account access flags.
     * @return 0 if the account is found, error otherwise.
     */
    __host__ __device__ int32_t get_account(
        ArithEnv &arith, const bn_t &address, CuEVM::account_t *&account_ptr,
        const CuEVM::account_flags_t flag = ACCOUNT_NONE_FLAG);

    /**
     * Get the value from the state.
     * @param[in] arith The arithmetic environment.
     * @param[in] address The address of the account.
     * @param[in] key The key of the storage.
     * @param[out] value The value of the storage.
     * @return 0 if found, error otherwiese.
     */
    __host__ __device__ int32_t get_value(ArithEnv &arith, const bn_t &address,
                                          const bn_t &key, bn_t &value);

    /**
     * Get the value without modifing the state
     * @param[in] arith The arithmetic environment.
     * @param[in] address The address of the account.
     * @param[in] key The key of the storage.
     * @param[out] value The value of the storage.
     * @return 0 if the value is found, error otherwise.
     */
    __host__ __device__ int32_t poke_value(ArithEnv &arith, const bn_t &address,
                                           const bn_t &key, bn_t &value) const;

    /**
     * Get the balance without modifing the state
     * @param[in] arith The arithmetic environment.
     * @param[in] address The address of the account.
     * @return 0 if the value is found, error otherwise.
     */
    __host__ __device__ int32_t poke_balance(ArithEnv &arith, const bn_t &address, bn_t &balance) const;


    /**
     * If an account has beeen accessed, it will be marked as warm.
     * @param[in] arith The arithmetic environment.
     * @param[in] address The address of the account.
     * @return 1 if the account is warm, 0 otherwise.
     */
    __host__ __device__ int32_t is_warm_account(ArithEnv &arith,
                                                const bn_t &address) const;

    /**
     * If a key has been accessed, it will be marked as warm.
     * @param[in] arith The arithmetic environment.
     * @param[in] address The address of the account.
     * @param[in] key The key of the storage.
     * @return 1 if the key is warm, 0 otherwise
     */
    __host__ __device__ int32_t is_warm_key(ArithEnv &arith,
                                            const bn_t &address,
                                            const bn_t &key) const;

    /**
     * If an account does not exist in the world state/deleted
     * @param[in] arith The arithmetic environment.
     * @param[in] address The address of the account.
     * @return 1 if the account is deleted, 0 otherwise.
     */
    __host__ __device__ int32_t is_deleted_account(ArithEnv &arith,
                                                   const bn_t &address) const;
};
}  // namespace CuEVM