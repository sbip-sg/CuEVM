// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation
// Programme) Author: Stefan-Dan Ciocirlan Date: 2024-09-15
// SPDX-License-Identifier: MIT

#pragma once

#include <cjson/cJSON.h>

#include <CuEVM/utils/arith.cuh>

namespace CuEVM {
struct storage_element_t {
    evm_word_t key;   /**< The key of the storage */
    evm_word_t value; /**< The value of the storage for the given key */

    /**
     * The default constructor for the storage element
     * TODO: maybe make 0 and 0 for key and value
     */
    __host__ __device__ storage_element_t() {};

    /**
     * The constructor for the storage element
     * @param[in] key The key of the storage
     * @param[in] value The value of the storage for the given key
     */
    __host__ __device__ storage_element_t(evm_word_t key, evm_word_t value)
        : key(key), value(value) {}

    /**
     * The copy constructor for the storage element
     * @param[in] storage_element The storage element to copy
     */
    __host__ __device__
    storage_element_t(const storage_element_t &storage_element)
        : key(storage_element.key), value(storage_element.value) {}

    /**
     * The constructor for the storage element from a JSON object
     * @param[in] storage_element_json The JSON object for the storage element
     */
    __host__ storage_element_t(const cJSON *storage_element_json);

    /**
     * Set the value of the storage element
     * @param[in] value The value of the storage element (evm_word_t)
     */
    __host__ __device__ void set_value(evm_word_t value) {
        this->value = value;
    }

    /**
     * Set the value of the storage element through bn_t
     * @param[in] arith The arithmetic environment
     * @param[in] value The value of the storage element (bn_t)
     */
    __host__ __device__ void set_value(ArithEnv &arith, const bn_t &value);

    /**
     * Get the value of the storage element
     * @return The value of the storage element (evm_word_t)
     */
    __host__ __device__ evm_word_t get_value() const { return this->value; }

    /**
     * Get the value of the storage element through bn_t
     * @param[in] arith The arithmetic environment
     * @param[out] value The value of the storage element (bn_t)
     */
    __host__ __device__ void get_value(ArithEnv &arith, bn_t &value) const;

    /**
     * Set the key of the storage element
     * @param[in] key The key of the storage element (evm_word_t)
     */
    __host__ __device__ void set_key(evm_word_t key) { this->key = key; }

    /**
     * Set the key of the storage element through bn_t
     * @param[in] arith The arithmetic environment
     * @param[in] key The key of the storage element (bn_t)
     */
    __host__ __device__ void set_key(ArithEnv &arith, const bn_t &key);

    /**
     * Get the key of the storage element
     * @return The key of the storage element (evm_word_t)
     */
    __host__ __device__ evm_word_t get_key() const { return this->key; }

    /**
     * Get the key of the storage element through bn_t
     * @param[in] arith The arithmetic environment
     * @param[out] key The key of the storage element (bn_t)
     */
    __host__ __device__ void get_key(ArithEnv &arith, bn_t &key) const;

    /**
     * The assignment operator for the storage element
     * @param[in] storage_element The storage element to assign
     * @return The storage element assigned
     */
    __host__ __device__ storage_element_t &operator=(
        const storage_element_t &storage_element) {
        this->key = storage_element.key;
        this->value = storage_element.value;
        return *this;
    }

    /**
     * Get if the key given (evm_word_t) is equal to the key of the storage
     * element
     * @param[in] key The key to compare (evm_word_t)
     * @return If the key is equal to the key of the storage element
     */
    __host__ __device__ int32_t has_key(const evm_word_t key) const;

    /**
     * Get if the key given (bn_t) is equal to the key of the storage element
     * @param[in] arith The arithmetic environment
     * @param[in] key The key to compare (bn_t)
     * @return If the key is equal to the key of the storage element
     */
    __host__ __device__ int32_t has_key(ArithEnv &arith, const bn_t &key) const;

    /**
     * Get if the value of the storage element is equal to 0
     * @return If the value of the storage element is equal to 0, 1 if true, 0
     * if false
     */
    __host__ __device__ int32_t is_zero_value() const;

    /**
     * Get if the value of the storage element is equal to 0with the help of the
     * arithmetic environment
     * @param[in] arith The arithmetic environment
     * @return If the value of the storage element is equal to 0, 1 if true, 0
     * if false
     */
    __host__ __device__ int32_t is_zero_value(ArithEnv &arith) const;

    /**
     * Get the storage element from a JSON object
     * @param[in] storage_element_json The JSON object for the storage element
     * @return The error code for the operation (0 means success)
     */
    __host__ int32_t from_json(const cJSON *storage_element_json);

    /**
     * Get the JSON object for the storage element.
     * If the hex string pointer is not provided, the hex string will be
     * allocated on the heap.
     * @param[out] storage_json The JSON object for the storage parent
     * @param[in] key_string_ptr The hex string pointer for the storage element
     * key
     * @param[in] value_string_ptr The hex string pointer for the storage
     * element value
     * @param[in] pretty If the hex string should be left trimmed of zeros
     * @return The error code for the operation (0 means success)
     */
    __host__ int32_t add_to_json(cJSON *storage_json,
                                 char *key_string_ptr = nullptr,
                                 char *value_string_ptr = nullptr,
                                 int32_t pretty = 0) const;

    /**
     * Print the storage element
     */
    __host__ __device__ void print() const;
};
}  // namespace CuEVM