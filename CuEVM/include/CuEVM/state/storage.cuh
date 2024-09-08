#ifndef _CUEVM_STORAGE_H_
#define _CUEVM_STORAGE_H_

#include <CuEVM/utils/arith.cuh>
#include <cjson/cJSON.h>

namespace CuEVM
{
    namespace storage {
        struct storage_element_t
        {
            evm_word_t key; /**< The key of the storage */
            evm_word_t value; /**< The value of the storage for the given key */

            /**
             * The default constructor for the storage element
             */
            __host__ __device__ storage_element_t() = default;

            /**
             * The constructor for the storage element
             * @param[in] key The key of the storage
             * @param[in] value The value of the storage for the given key
             */
            __host__ __device__ storage_element_t(evm_word_t key, evm_word_t value) : key(key), value(value) {}

            /**
             * The copy constructor for the storage element
             * @param[in] storage_element The storage element to copy
             */
            __host__ __device__ storage_element_t(const storage_element_t &storage_element) : key(storage_element.key), value(storage_element.value) {}

            /**
             * The constructor for the storage element from a JSON object
             * @param[in] storage_element_json The JSON object for the storage element
             */
            __host__ storage_element_t(const cJSON *storage_element_json);

            /**
             * Set the value of the storage element
             * @param[in] value The value of the storage element (evm_word_t)
             */
            __host__ __device__ void set_value(evm_word_t value) { this->value = value; }

            /**
             * Set the value of the storage element through bn_t
             * @param[in] arith The arithmetic environment
             * @param[in] value The value of the storage element (bn_t)
             */
            __host__ __device__ void set_value(ArithEnv arith, const bn_t &value);

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
            __host__ __device__ void get_value(ArithEnv arith, bn_t &value) const;

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
            __host__ __device__ void set_key(ArithEnv arith, const bn_t &key);

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
            __host__ __device__ void get_key(ArithEnv arith, bn_t &key) const;

            /**
             * The assignment operator for the storage element
             * @param[in] storage_element The storage element to assign
             * @return The storage element assigned
             */
            __host__ __device__ storage_element_t &operator=(const storage_element_t &storage_element)
            {
                this->key = storage_element.key;
                this->value = storage_element.value;
                return *this;
            }

            /**
             * Get if the key given (evm_word_t) is equal to the key of the storage element
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
            __host__ __device__ int32_t has_key(ArithEnv arith, const bn_t &key) const;

            /**
             * Get if the value of the storage element is equal to 0 
             * @return If the value of the storage element is equal to 0, 1 if true, 0 if false
             */
            __host__ __device__ int32_t is_zero_value() const;

            /**
             * Get if the value of the storage element is equal to 0with the help of the arithmetic environment
             * @param[in] arith The arithmetic environment
             * @return If the value of the storage element is equal to 0, 1 if true, 0 if false
             */
            __host__ __device__ int32_t is_zero_value(ArithEnv arith) const;

            /**
             * Get the storage element from a JSON object
             * @param[in] storage_element_json The JSON object for the storage element
             * @return The error code for the operation (0 means success)
             */
            __host__ int32_t from_json(const cJSON *storage_element_json);

            /**
             * Get the JSON object for the storage element.
             * If the hex string pointer is not provided, the hex string will be allocated on the heap.
             * @param[out] storage_json The JSON object for the storage parent
             * @param[in] key_string_ptr The hex string pointer for the storage element key
             * @param[in] value_string_ptr The hex string pointer for the storage element value
             * @param[in] pretty If the hex string should be left trimmed of zeros
             * @return The error code for the operation (0 means success)
             */
            __host__ int32_t add_to_json(
                cJSON *storage_json,
                char *key_string_ptr = nullptr,
                char *value_string_ptr = nullptr,
                int32_t pretty = 0) const;

            /**
             * Print the storage element
             */
            __host__ __device__ void print() const;

        };

        /**
         * The contract storage struct for the EVM
         */
        struct contract_storage_t {
            uint32_t size; /**< The size of the storage */
            uint32_t capacity; /**< The capacity of the storage */
            storage_element_t *storage; /**< The storage elements */

            /**
             * The default constructor for the contract storage
             */
            __host__ __device__ contract_storage_t() : size(0), capacity(0), storage(nullptr) {};

            /**
             * The constructor for the contract storage from a JSON object
             * @param[in] contract_storage_json The JSON object for the contract storage
             */
            __host__ __device__ contract_storage_t(
                const cJSON *contract_storage_json);
            
            /**
             * The destructor for the contract storage
             */
            __host__ __device__ ~contract_storage_t();

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
            __host__ __device__ int32_t get_value(
                ArithEnv &arith,
                const bn_t &key,
                bn_t &value) const;

            /**
             * Set the value for the given key
             * @param[in] arith The arithmetic environment
             * @param[in] key The key to set the value for
             * @param[in] value The value for the given key
             * @return The error code for the operation (0 means success)
             */
            __host__ __device__ int32_t set_value(
                ArithEnv &arith,
                const bn_t &key,
                const bn_t &value);
            
            /**
             * Update the the current storage with the given storage
             * @param[in] arith The arithmetic environment
             * @param[in] other The contract storage with the updates
             */
            __host__ __device__ void update(
                ArithEnv &arith,
                const contract_storage_t &other);
            
            /**
             * Get the storage element index for the given key
             * @param[in] key The key to get the index for
             * @param[out] index The index for the given key
             * @return The error code for the operation (0 means success)
             */
            __host__ int32_t has_key(
                const evm_word_t &key,
                uint32_t &index) const;

            /**
             * Get the contract stroage from a json object
             * @param[in] contract_storage_json The JSON object for the contract storage
             * @param[in] managed The flag to indicate if the memory is managed
             * @return The error code for the operation (0 means success)
             */
            __host__ int32_t from_json(
                const cJSON *contract_storage_json,
                int32_t managed = 0);

            /**
             * Get the JSON object for the contract storage
             * @param[in] pretty If the hex string should be left trimmed of zeros
             * @return The JSON object for the contract storage
             */
            __host__ cJSON* to_json(
                int32_t pretty = 0) const;

            /**
             * Print the contract storage
             */
            __host__ __device__ void print() const;
        };

        /**
         * Merge two contract storages
         * @param[in] storage1 The first contract storage
         * @param[in] storage2 The second contract storage
         * @param[in] pretty If the hex string should be left trimmed of zeros
         */
        __host__ cJSON* storage_merge_json(
            const contract_storage_t &storage1,
            const contract_storage_t &storage2,
            const int32_t pretty = 0);


    } // namespace storage

} // namespace CuEVM

#endif