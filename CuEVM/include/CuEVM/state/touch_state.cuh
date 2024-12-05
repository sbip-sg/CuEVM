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
 * @brief The TouchState class
 *
 */
class TouchState {
   private:
    state_access_t *_state;          /**< The state access */
    CuEVM::WorldState *_world_state; /**< The world state */
    TouchState *parent;              /**< The parent state */

    /**
     * Add an account to the state.
     * @param[in] arith The arithmetic environment.
     * @param[in] address The address of the account.
     * @param[out] account_ptr The pointer to the account.
     * @param[in] acces_state_flag The account access flags.
     * @return 1 if the account is added successfully, 0 otherwise.
     */
    __host__ __device__ int32_t add_account(ArithEnv &arith, const evm_word_t *address, CuEVM::account_t *&account_ptr,
                                            const CuEVM::account_flags_t acces_state_flag);

   public:
    /**
     * The default constructor.
     */
    __host__ __device__ TouchState() : _state(nullptr), parent(nullptr), _world_state(nullptr) {}

    /**
     * The constructor with the state and the access state.
     * @param[in] state The state access.
     * @param[in] world_state The world state.
     */
    __host__ __device__ TouchState(state_access_t *state, CuEVM::WorldState *world_state)
        : _state(state), _world_state(world_state), parent(nullptr) {}

    /**
     * The constructor with the state, the access state, and the parent state.
     * @param[in] state The state access.
     * @param[in] parent The parent state.
     */
    __host__ __device__ TouchState(state_access_t *state, TouchState *parent)
        : _state(state), _world_state(parent->_world_state), parent(parent) {}

    /**
     * destructor for the touch state
     *
     */
    __host__ __device__ ~TouchState() {
        delete _state;
        clear();
    }

    /**
     * Clear the touch state.
     */
    __host__ __device__ void clear() {
        _state = nullptr;
        _world_state = nullptr;
        parent = nullptr;
    }

    /**
     * the assigment operator
     * @param[in] other The other touch state.
     * @return The reference to the touch state.
     */
    __host__ __device__ TouchState &operator=(const TouchState &other) {
        _state = other._state;
        _world_state = other._world_state;
        parent = other.parent;
        return *this;
    }

    __host__ __device__ state_access_t *get_state() const { return _state; }

    /**
     * The getter for the account given by an address.
     * @param[in] arith The arithmetic environment.
     * @param[in] address The address of the account.
     * @param[out] account_ptr The pointer to the account.
     * @param[in] acces_state_flag The account access flags.
     * @param[in] add_to_state If the account should be force added to the state
     *  Set to true when getting the account to update it.
     * @return 0 if the account is found, error otherwise.
     */
    __host__ __device__ int32_t get_account(ArithEnv &arith, const evm_word_t *address, CuEVM::account_t *&account_ptr,
                                            const CuEVM::account_flags_t acces_state_flag, bool add_to_state = false);

    __host__ __device__ int32_t get_account_index(ArithEnv &arith, const evm_word_t *address, uint32_t &index) const;
    /**
     * If the account given by address is empty
     * @param[in] arith The arithmetic environment.
     * @param[in] address The address of the account.
     * @return true if the account is empty, false otherwise.
     */
    __host__ __device__ bool is_empty_account(ArithEnv &arith, const evm_word_t *address);

    /**
     * @brief Determine if an account is empty and can be created
     *  Different treatment to normal empty account (can have balance)
     * @param arith
     * @param address
     * @return __host__
     */
    __host__ __device__ bool is_empty_account_create(ArithEnv &arith, const evm_word_t *address);

    /**
     * If the account given by address is deleted
     * @param[in] arith The arithmetic environment.
     * @param[in] address The address of the account.
     * @return 1 if the account is deleted, 0 otherwise.
     */
    __host__ __device__ int32_t is_deleted_account(ArithEnv &arith, const evm_word_t *address);

    /**
     * The getter for the balance given by an address.
     * @param[in] arith The arithmetic environment.
     * @param[in] address The address of the account.
     * @param[out] balance The balance of the account.
     * @return error_code, 0 if success
     */
    __host__ __device__ int32_t get_balance(ArithEnv &arith, const evm_word_t *address, bn_t &balance);

    /**
     * The getter for the nonce given by an address.
     * @param[in] arith The arithmetic environment.
     * @param[in] address The address of the account.
     * @param[out] nonce The nonce of the account.
     * @return error_code, 0 if success
     */
    __host__ __device__ int32_t get_nonce(ArithEnv &arith, const evm_word_t *address, bn_t &nonce);

    /**
     * The getter for the code given by an address.
     * @param[in] arith The arithmetic environment.
     * @param[in] address The address of the account.
     * @param[out] byte_code The byte code of the account.
     * @return error_code, 0 if success
     */
    __host__ __device__ int32_t get_code(ArithEnv &arith, const evm_word_t *address, byte_array_t &byte_code);

    /**
     * The getter for the value given by an address and a key.
     * @param[in] arith The arithmetic environment.
     * @param[in] address The address of the account.
     * @param[in] key The key of the storage.
     * @param[out] value The value of the storage.
     * @return 0 if the value is found, error otherwise.
     */
    __host__ __device__ int32_t get_value(ArithEnv &arith, const evm_word_t *address, const bn_t &key, bn_t &value);

    /**
     * The getter for the value of a storage element without modifing the state.
     * @param[in] arith The arithmetic environment.
     * @param[in] address The address of the account.
     * @param[in] key The key of the storage.
     * @param[out] value The value of the storage.
     * @return 0 if the value is found, error otherwise.
     */
    __host__ __device__ int32_t poke_value(ArithEnv &arith, const evm_word_t *address, const bn_t &key,
                                           bn_t &value) const;

    __host__ __device__ int32_t poke_original_value(ArithEnv &arith, const evm_word_t *address, const bn_t &key,
                                                    bn_t &value) const;

    /**
     * The setter for the balance given by an address.
     * @param[in] arith The arithmetic environment.
     * @param[in] address The address of the account.
     * @param[in] balance The balance of the account.
     * @return 0 if the balance is set, error otherwise.
     */
    __host__ __device__ int32_t set_balance(ArithEnv &arith, const evm_word_t *address, const bn_t &balance);

    /**
     * The getter for the balance given by an address without modifing the
     * state.
     * @param[in] arith The arithmetic environment.
     * @param[in] address The address of the account.
     * @param[in] balance The balance of the account.
     * @return 0 if the balance is set, error otherwise.
     */
    __host__ __device__ int32_t poke_balance(ArithEnv &arith, const evm_word_t *address, bn_t &balance) const;
    /**
     * Get the account object without settng it warm
     *
     * @param arith The arithmetic environment.
     * @param address The address of the account.
     * @param account_ptr The pointer to the account.
     * @param include_world_state  If the world state should be included
     * @return 0 if the account is found, error otherwise.
     */
    __host__ __device__ int32_t poke_account(ArithEnv &arith, const evm_word_t *address, CuEVM::account_t *&account_ptr,
                                             bool include_world_state = false) const;

    /**
     * Check if an account is in the warm set
     *
     * @param arith The arithmetic environment.
     * @param address The address of the account.
     * @return true if the account is in the warm set, false otherwise.
     */
    __host__ __device__ bool is_warm_account(ArithEnv &arith, const evm_word_t *address) const;

    /**
     * Check if a key is in the warm set
     *
     * @param arith The arithmetic environment.
     * @param address The address of the account.
     * @param key The key of the storage.
     * @return true if the key is in the warm set, false otherwise.
     */
    __host__ __device__ bool is_warm_key(ArithEnv &arith, const evm_word_t *address, const bn_t &key) const;

    /**
     * Set an account to be warm
     * @param arith The arithmetic environment.
     * @param address The address of the account.
     */
    __host__ __device__ bool set_warm_account(ArithEnv &arith, const evm_word_t *address);

    /**
     * Set a key to be warm
     * @param arith The arithmetic environment.
     * @param address The address of the account.
     * @param key The key of the storage.
     * @param value The value of the storage.
     */
    __host__ __device__ bool set_warm_key(ArithEnv &arith, const evm_word_t *address, const bn_t &key,
                                          const bn_t &value);
    /**
     * The setter for the nonce given by an address.
     * @param[in] arith The arithmetic environment.
     * @param[in] address The address of the account.
     * @param[in] nonce The nonce of the account.
     * @return 0 if the nonce is set, error otherwise.
     */
    __host__ __device__ int32_t set_nonce(ArithEnv &arith, const evm_word_t *address, const bn_t &nonce);

    /**
     * The setter for the code given by an address.
     * @param[in] arith The arithmetic environment.
     * @param[in] address The address of the account.
     * @param[in] byte_code The byte code of the account.
     * @return 0 if the code is set, error otherwise.
     */
    __host__ __device__ int32_t set_code(ArithEnv &arith, const evm_word_t *address, const byte_array_t &byte_code);

    /**
     * The setter for the storage value given by an address, a key, and a
     * value.
     * @param[in] arith The arithmetic environment.
     * @param[in] address The address of the account.
     * @param[in] key The key of the storage.
     * @param[in] value The value of the storage.
     * @return 0 if the storage value is set, error otherwise.
     */
    __host__ __device__ int32_t set_storage_value(ArithEnv &arith, const evm_word_t *address, const bn_t &key,
                                                  const bn_t &value);

    /**
     * Delete an account in the state
     * @param[in] arith The arithmetic environment.
     * @param[in] address The address of the account.
     * @return 0 if the account is deleted, error otherwise.
     */
    __host__ __device__ int32_t delete_account(ArithEnv &arith, const evm_word_t *address);

    /**
     * Mark an account for deletion at the end of the transaction
     * @param[in] arith The arithmetic environment.
     * @param[in] address The address of the account.
     * @return 0 if success, error otherwise.
     */
    __host__ __device__ int32_t mark_for_deletion(ArithEnv &arith, const evm_word_t *address);

    /**
     * Update the touch state.
     * @param[in] arith The arithmetic environment.
     * @param[in] other The other touch state.
     * @return 0 if the touch state is updated, error otherwise.
     */
    __host__ __device__ int32_t update(ArithEnv &arith, TouchState *other);

    // update the final state with the world state and combine storage
    __host__ __device__ int32_t update_world_state(ArithEnv &arith);

    /**
     * Transfer the given value from one account to another.
     * @param[in] arith The arithmetic environment.
     * @param[in] from The address of the account to transfer from.
     * @param[in] to The address of the account to transfer to.
     * @param[in] value The value to transfer.
     * @return 0 if the transfer is successful, error otherwise.
     */
    __host__ __device__ int32_t transfer(ArithEnv &arith, const evm_word_t *from, const evm_word_t *to,
                                         const bn_t &value);

    __host__ __device__ CuEVM::contract_storage_t get_entire_storage(ArithEnv &arith,
                                                                     const uint32_t account_index) const;
    /**
     * print the touch state
     */
    __host__ __device__ void print() const;
};
}  // namespace CuEVM