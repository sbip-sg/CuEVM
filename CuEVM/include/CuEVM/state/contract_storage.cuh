// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation
// Programme) Author: Stefan-Dan Ciocirlan Date: 2024-09-15
// SPDX-License-Identifier: MIT

#pragma once

#include <CuEVM/state/storage_element.cuh>
#include <CuEVM/utils/arith.cuh>

namespace CuEVM {

/**
 * The contract storage struct for the EVM
 */
struct contract_storage_t {
    uint32_t size;              /**< The size of the storage */
    uint32_t capacity;          /**< The capacity of the storage */
    storage_element_t *storage; /**< The storage elements */

    /**
     * The default constructor for the contract storage
     */
    __host__ __device__ contract_storage_t()
        : size(0), capacity(0), storage(nullptr) {};

    /**
     * The destructor for the contract storage
     */
    __host__ __device__ ~contract_storage_t();

    /**
     * Free the internal memory of the contract storage
     */
    __host__ __device__ void free();

    /**
     * Free the internal memory of a managed contract storage
     */
    __host__ void free_managed();

    /**
     * Clear the internal memory of the contract storage
     */
    __host__ __device__ void clear();

    /**
     * The assignment operator for the contract storage
     * @param[in] contract_storage The contract storage to assign
     */
    __host__ __device__ contract_storage_t &operator=(
        const contract_storage_t &contract_storage);

    /**
     * The free internal storage
     * @param[in] managed The flag to indicate if the memory is managed
     */
    // __host__ __device__ void free_internals(
    //     int32_t managed = 0);

    /**
     * Get the value for the given key
     * @param[in] arith The arithmetic environment
     * @param[in] key The key to get the value for
     * @param[out] value The value for the given key
     * @return The error code for the operation (0 means success)
     */
    __host__ __device__ int32_t get_value(ArithEnv &arith, const bn_t &key,
                                          bn_t &value) const;

    /**
     * Set the value for the given key
     * @param[in] arith The arithmetic environment
     * @param[in] key The key to set the value for
     * @param[in] value The value for the given key
     * @return The error code for the operation (0 means success)
     */
    __host__ __device__ int32_t set_value(ArithEnv &arith, const bn_t &key,
                                          const bn_t &value);

    /**
     * Update the the current storage with the given storage
     * @param[in] arith The arithmetic environment
     * @param[in] other The contract storage with the updates
     */
    __host__ __device__ void update(ArithEnv &arith,
                                    const contract_storage_t &other);

    /**
     * Get the storage element index for the given key
     * @param[in] key The key to get the index for
     * @param[out] index The index for the given key
     * @return The error code for the operation (0 means success)
     */
    __host__ int32_t has_key(const evm_word_t &key, uint32_t &index) const;

    /**
     * Get the contract stroage from a json object
     * @param[in] contract_storage_json The JSON object for the contract storage
     * @param[in] managed The flag to indicate if the memory is managed
     * @return The error code for the operation (0 means success)
     */
    __host__ int32_t from_json(const cJSON *contract_storage_json,
                               int32_t managed = 0);

    /**
     * Get the JSON object for the contract storage
     * @param[in] pretty If the hex string should be left trimmed of zeros
     * @return The JSON object for the contract storage
     */
    __host__ cJSON *to_json(int32_t pretty = 0) const;

    /**
     * Print the contract storage
     */
    __host__ __device__ void print() const;

    // STATIC FUNCTIONS
    /**
     * Merge two contract storages
     * @param[in] storage1 The first contract storage
     * @param[in] storage2 The second contract storage
     * @param[in] pretty If the hex string should be left trimmed of zeros
     */
    __host__ static cJSON *merge_json(const contract_storage_t &storage1,
                                      const contract_storage_t &storage2,
                                      const int32_t pretty = 0);
    /**
     * Transfer memory from one contract storage to another
     * @param[in] src The source contract storage
     * @param[in] dst The destination contract storage
     */
    __host__ __device__ static void transfer_memory(contract_storage_t &src,
                                                    contract_storage_t &dst);
};
}  // namespace CuEVM