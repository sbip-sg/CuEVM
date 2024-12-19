// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation
// Programme) Author: Stefan-Dan Ciocirlan Date: 2024-09-15
// SPDX-License-Identifier: MIT

#pragma once

#include <CuEVM/state/account.cuh>
#include <CuEVM/state/account_flags.cuh>
#include <CuEVM/state/state.cuh>
#include <CuEVM/state/state_access.cuh>
#include <CuEVM/utils/arith.cuh>

namespace CuEVM {
// for convenient data transfer between host and device. set a fixed maximum size for the number of addresses to be
// transferred
// todo : optimize this later
constexpr uint32_t worldstate_addresses_size = 32;
constexpr uint32_t worldstate_storage_values_size = 1024;
// heuristic size for the bytecode hex string to keep everything within 1MB
constexpr uint32_t byte_code_hex_size = 32 * max_code_size;
struct serialized_worldstate_data {
    uint32_t no_accounts;
    uint32_t no_storage_elements;
    char addresses[worldstate_addresses_size][43];  // 0x + ... + \0
    char code_hash[worldstate_addresses_size][67]; // 0x + ... + \0
    char balance[worldstate_addresses_size][67];    // 0x + ... + \0
    uint32_t nonce[worldstate_addresses_size];
    uint16_t storage_indexes[worldstate_storage_values_size];
    char storage_keys[worldstate_storage_values_size][67];    // 0x + ... + \0
    char storage_values[worldstate_storage_values_size][67];  // 0x + ... + \0
    // currently dont support copy back the bytecode hex string
    // TODO: use 1 large preallocated buffer for bytecode
    void print();
    void print_json();
};

  struct plain_account {
    uint32_t storage_idx_start; // inclusive
    uint32_t storage_idx_end; //exclusive
    char address[43];  // 0x + ... + \0
    char code_hash[67]; // 0x + ... + \0
    char balance[67];    // 0x + ... + \0
    uint32_t nonce;
  };

  struct plain_storage {
    char key[67];    // 0x + ... + \0
    char value[67];  // 0x + ... + \0
  };

  struct flatten_state{
    uint32_t no_accounts;
    uint32_t no_storage_elements;
    plain_account *accounts;
    plain_storage *storage_elements;
  };

/**
 * The world state classs
 */
class WorldState {
   private:
    state_t *_state; /**< The state with the accounts */
   public:
    /**
     * The default constructor
     */
    __host__ __device__ WorldState() : _state(nullptr) {}
    /**
     * The constructor with a pointer to the state struct
     * @param[in] state pointer to the state
     */
    __host__ __device__ WorldState(state_t *state) : _state(state) {}
    /**
     * The destuctor
     */
    __host__ __device__ ~WorldState() { _state = nullptr; }

    /**
     * Get the account pointer given the address
     * @param[in] arith The arithmetic value to be processed.
     * @param[in] address The address for the account
     * @param[out] account_ptr  The pointer to the account
     * @return if found 0, otherwise error.
     */
    __host__ __device__ int32_t get_account(ArithEnv &arith, const evm_word_t *address, CuEVM::account_t *&account_ptr);
    __host__ __device__ int32_t update(ArithEnv &arith, const CuEVM::state_access_t *state_access_ptr);

    /**
     * Get the value of a storage element
     * @param[in] arith The arithemtic environment.
     * @param[in] address The blockchain address of the account whose storage
     * value is being queried. This parameter specifies the account uniquely.
     * @param[in] key The key corresponding to the storage value within the
     * account's storage to be retrieved.
     * @param[out] value The storage value
     * @return Returns 0 if the storage value for the given account and key was
     * successfully found and value has been updated accordingly. Returns error
     * if the account or the key within the account's storage could not be found
     * and the value is set to 0.
     */
    __host__ __device__ int32_t get_value(ArithEnv &arith, const evm_word_t *address, const bn_t &key, bn_t &value);
  __host__ __device__ void serialize_data(ArithEnv &arith, serialized_worldstate_data *data);
  __host__ __device__ void flatten(ArithEnv &arith, CuEVM::flatten_state *data);
};

}  // namespace CuEVM

extern __managed__ CuEVM::flatten_state *flatten_state_ptr;
