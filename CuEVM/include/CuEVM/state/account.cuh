// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation
// Programme) Author: Stefan-Dan Ciocirlan Date: 2024-09-15
// SPDX-License-Identifier: MIT

#pragma once

#include <CuEVM/core/byte_array.cuh>
#include <CuEVM/state/account_flags.cuh>
#include <CuEVM/state/contract_storage.cuh>
#include <CuEVM/utils/arith.cuh>

namespace CuEVM {
/**
 * The account type.
 */
struct account_t {
    evm_word_t address;                /**< The address of the account (YP: \f$a\f$) */
    evm_word_t balance;                /**< The balance of the account (YP: \f$\sigma[a]_{b}\f$) */
    evm_word_t nonce;                  /**< The nonce of the account (YP: \f$\sigma[a]_{n}\f$) */
    byte_array_t byte_code;            /**< The bytecode of the account (YP: \f$b\f$) */
    CuEVM::contract_storage_t storage; /**< The storage of the account (YP: \f$\sigma[a]_{s}\f$) */

    /**
     * The default constructor for the account data structure.
     */
    __host__ __device__ account_t() : storage(), byte_code(0U) {};

    /**
     * The copy constructor for the account data structure.
     * @param[in] account The account data structure
     */
    __host__ __device__ account_t(const account_t &account);

    /**
     * The copy constructor for the account data structure with flags.
     * @param[in] account The account data structure
     * @param[in] flags The account flags
     */
    __host__ __device__ account_t(const account_t &account, const account_flags_t &flags);

    /**
     * The copy constructor from the account ptr structure with flags.
     * @param[in] account_ptr The account data structure
     * @param[in] flags The account flags
     */
    __host__ __device__ account_t(const account_t *account_ptr, const account_flags_t &flags);

    /**
     * Constructor for an empty account with the given address.
     * @param[in] arith The arithmetical environment
     * @param[in] address The address of the account
     */
    __host__ __device__ account_t(ArithEnv &arith, const bn_t &address);

    /**
     * The destructor for the account data structure.
     */
    __host__ __device__ ~account_t();

    /**
     * The destructor for the account data structure.
     */
    __host__ __device__ void free();

    /**
     * the destructor for managed account data structure.
     */
    __host__ void free_managed();

    /**
     * Clear the account data structure.
     */
    __host__ __device__ void clear();

    /**
     * Assigment operator for the account data structure.
     * @param[in] other The other account data structure
     * @return The current account data structure
     */
    __host__ __device__ account_t &operator=(const account_t &other);

    /**
     * Get the storage value for the given key.
     * @param[in] arith The arithmetical environment
     * @param[in] key The key
     * @param[out] value The value
     * @return If found 0, otherwise error code
     */
    __host__ __device__ int32_t get_storage_value(ArithEnv &arith, const bn_t &key, bn_t &value);
    /**
     * Set the storage value for the given key.
     * @param[in] arith The arithmetical environment
     * @param[in] key The key of the storage
     * @param[in] value The value of the storage
     * @return If set succesfull 0, otherwise error code
     */
    __host__ __device__ int32_t set_storage_value(ArithEnv &arith, const bn_t &key, const bn_t &value);

    /**
     * Get the address of the account.
     * @param[in] arith The arithmetical environment
     * @param[out] address The address of the account
     */
    __host__ __device__ void get_address(ArithEnv &arith, bn_t &address);

    /**
     * Get the balance of the account.
     * @param[in] arith The arithmetical environment
     * @param[out] balance The balance of the account
     */
    __host__ __device__ void get_balance(ArithEnv &arith, bn_t &balance);

    /**
     * Get the nonce of the account.
     * @param[in] arith The arithmetical environment
     * @param[out] nonce The nonce of the account
     */
    __host__ __device__ void get_nonce(ArithEnv &arith, bn_t &nonce);

    /**
     * Get the byte code of the account.
     * @return The byte code of the account
     */
    __host__ __device__ byte_array_t get_byte_code() const;

    /**
     * Set the address of the account.
     * @param[in] arith The arithmetical environment
     * @param[in] address The address of the account
     */
    __host__ __device__ void set_address(ArithEnv &arith, const bn_t &address);

    /**
     * Set the balance of the account.
     * @param[in] arith The arithmetical environment
     * @param[in] balance The balance of the account
     */
    __host__ __device__ void set_balance(ArithEnv &arith, const bn_t &balance);

    /**
     * Set the nonce of the account.
     * @param[in] arith The arithmetical environment
     * @param[in] nonce The nonce of the account
     */
    __host__ __device__ void set_nonce(ArithEnv &arith, const bn_t &nonce);

    /**
     * set the byte code of the account.
     * @param[in] byte_code The byte code of the account
     */
    __host__ __device__ void set_byte_code(const byte_array_t &byte_code);

    /**
     * Verify if the account has the the given address.
     * @param[in] arith The arithmetical environment
     * @param[in] address The address
     * @return If found 1, otherwise 0
     */
    __host__ __device__ int32_t has_address(ArithEnv &arith, const bn_t &address);

    /**
     * Verify if the account has the the given address.
     * @param[in] arith The arithmetical environment
     * @param[in] address The address as evm word
     * @return If found 1, otherwise 0
     */
    __host__ __device__ int32_t has_address(ArithEnv &arith, const evm_word_t &address);

    /**
     * Update the current account with the information
     * from the given account
     * @param[in] arith The arithemetic environment
     * @param[in] other The given account
     * @param[in] flags The flags to indicate which fields should be updated
     */
    __host__ __device__ void update(ArithEnv &arith, const account_t &other,
                                    const account_flags_t &flags = ACCOUNT_ALL_FLAG);

    /**
     * Verify if the account is empty using arithmetical environment.
     * @param[in] arith The arithmetical environment
     * @return If empty 1, otherwise 0
     */
    __host__ __device__ int32_t is_empty(ArithEnv &arith);

    /**
     * Verify if the account is empty.
     * @return If empty 1, otherwise 0
     */
    __host__ __device__ int32_t is_empty();

    /**
     * Verify if the account is a contract.
     * @return If contract 1, otherwise 0
     */
    __host__ __device__ int32_t is_contract();

    /**
     * Make the account completly empty.
     */
    __host__ __device__ void empty();

    /**
     * Setup the account data structure from the json object.
     * @param[in] account_json The json object
     * @param[in] managed The flag to indicate if the memory is managed
     */
    __host__ void from_json(const cJSON *account_json, int32_t managed = false);

    /**
     * Generate a JSON object from the account data structure.
     * @return The JSON object
     */
    __host__ cJSON *to_json() const;

    /**
     * Print the account data structure.
     */
    __host__ __device__ void print();

    /**
     * To copy the account data structures.
     * @param[out] dst The destination account data structure
     * @param[in] src The source account data structure
     */
    __host__ __device__ static void transfer_memory(account_t &dst, account_t &src);

    /**
     * Generate the cpu data structures for the account.
     * @param[in] count The number of instances
     * @return The cpu data structures
     */
    __host__ static account_t *get_cpu(uint32_t count);

    /**
     * Free the cpu data structures for the account.
     * @param[inout] cpu_instances The cpu data structures
     * @param[in] count The number of instances
     */
    __host__ static void free_cpu(account_t *cpu_instances, uint32_t count);

    /**
     * Generate the gpu data structures for the account.
     * @param[in] cpu_instances The cpu data structures
     * @param[in] count The number of instances
     */
    __host__ static account_t *get_gpu_from_cpu(account_t *cpu_instances, uint32_t count);

    /**
     * Free the gpu data structures for the account.
     * @param[inout] gpu_instances The gpu data structures
     * @param[in] count The number of instances
     */
    __host__ static void free_gpu(account_t *gpu_instances, uint32_t count);

    /**
     * Generate the cpu data structures for the account from the gpu data structures.
     * @param[in] gpu_instances The gpu data structures
     * @param[in] count The number of instances
     * @return The cpu data structures
     */
    __host__ static account_t *get_cpu_from_gpu(account_t *gpu_instances, uint32_t count);
    /**
     * The json from two account data structures pointers.
     * @param[in] account1_ptr The first account data structure
     * @param[in] account2_ptr The second account data structure
     * @param[in] flags The account flags representig the fields
     * @return The json object
     */
    __host__ static cJSON *merge_json(const account_t *&account1_ptr, const account_t *&account2_ptr,
                                      const account_flags_t &flags);
};

/**
 * The kernel to copy the account data structures.
 * @param[out] dst The destination account data structure
 * @param[in] src The source account data structure
 * @param[in] count The number of instances
 */
__global__ void account_t_transfer_kernel(account_t *dst_instances, account_t *src_instances, uint32_t count);
}  // namespace CuEVM