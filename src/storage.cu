// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2024-06-20
// SPDX-License-Identifier: MIT
#include "include/storage.cuh"
#include "include/utils.cuh"

namespace cuEVM {
    namespace storage {
        __host__ __device__ storage_element_t::storage_element_t(const cJSON *storage_element_json)
        {
            this->from_json(storage_element_json);
        }

        __host__ __device__ void storage_element_t::set_value(ArithEnv arith, const bn_t &value)
        {
            cgbn_store(arith.env, &(this->value), value);
        }

        __host__ __device__ void storage_element_t::get_value(ArithEnv arith, bn_t &value) const
        {
            cgbn_load(arith.env, value, (cgbn_evm_word_t_ptr) &(this->value));
        }

        __host__ __device__ void storage_element_t::set_key(ArithEnv arith, const bn_t &key)
        {
            cgbn_store(arith.env, &(this->key), key);
        }

        __host__ __device__ void storage_element_t::get_key(ArithEnv arith, bn_t &key) const
        {
            cgbn_load(arith.env, key, (cgbn_evm_word_t_ptr) &(this->key));
        }

        __host__ __device__ int32_t storage_element_t::has_key(evm_word_t key) const
        {
            return (this->key == key);
        }

        __host__ __device__ int32_t storage_element_t::has_key(ArithEnv arith, const bn_t &key) const
        {
            bn_t storage_key;
            cgbn_load(arith.env, storage_key, (cgbn_evm_word_t_ptr) &(this->key));
            return (cgbn_compare(arith.env, storage_key, key) == 0);
        }

        __host__ __device__ int32_t storage_element_t::is_zero_value() const
        {
            return (this->value == 0U);
        }

        __host__ __device__ int32_t storage_element_t::is_zero_value(ArithEnv arith) const
        {
            bn_t storage_value;
            cgbn_load(arith.env, storage_value, (cgbn_evm_word_t_ptr) &(this->value));
            return (cgbn_compare_ui32(arith.env, storage_value, 0U) == 0);
        }

        __host__ int32_t storage_element_t::from_json(const cJSON *storage_element_json)
        {
            key.from_hex(storage_element_json->string);
            value.from_hex(storage_element_json->valuestring);
        }

        __host__ int32_t storage_element_t::add_to_json(
                cJSON *storage_json,
                char *key_string_ptr,
                char *value_string_ptr,
                int32_t pretty) const
        {
            if (cJSON_IsNull(storage_json))
            {
                return 1;
            }
            char *tmp_key_string_ptr = nullptr;
            char *tmp_value_string_ptr = nullptr;
            if (key_string_ptr == nullptr)
            {
                tmp_key_string_ptr = new char[EVM_WORD_SIZE * 2 + 3];
                key_string_ptr = tmp_key_string_ptr;
            }
            if (value_string_ptr == nullptr)
            {
                tmp_value_string_ptr = new char[EVM_WORD_SIZE * 2 + 3];
                value_string_ptr = tmp_value_string_ptr;
            }
            key_string_ptr = key.to_hex(key_string_ptr, pretty);
            value_string_ptr = value.to_hex(value_string_ptr, pretty);
            cJSON_AddStringToObject(storage_json, key_string_ptr, value_string_ptr);
            if (tmp_key_string_ptr != nullptr)
            {
                delete[] tmp_key_string_ptr;
            }
            if (tmp_value_string_ptr != nullptr)
            {
                delete[] tmp_value_string_ptr;
            }
            return 0;
        }

        __host__ __device__ void storage_element_t::print() const
        {
            printf("Key: ");
            key.print();
            printf("Value: ");
            value.print();
        }

        // contract_storage_t

        __host__ __device__ contract_storage_t::contract_storage_t(const cJSON *contract_storage_json)
        {
            this->from_json(contract_storage_json);
        }

        __host__ __device__ contract_storage_t::~contract_storage_t()
        {
            if (storage != nullptr)
            {
                delete[] storage;
                storage = nullptr;
            }
            size = 0;
            capacity = 0;
        }

        __host__ __device__ contract_storage_t& contract_storage_t::operator=(
            const contract_storage_t &other)
        {
            if (this == &other)
            {
                return *this;
            }
            if (size != other.size)
            {
                if (storage != nullptr)
                {
                    delete[] storage;
                    storage = nullptr;
                }
                size = other.size;
                capacity = other.capacity;
                storage = new storage_element_t[capacity];
            }
            std::copy(other.storage, other.storage + other.size, storage);
            return *this;
        }

        __host__ __device__ int32_t contract_storage_t::get_value(
            ArithEnv arith,
            const bn_t &key,
            bn_t &value) const
        {
            uint32_t idx = 0;
            for (idx = 0; idx < this->size; idx++)
            {
                if (this->storage[idx].has_key(arith, key))
                {
                    this->storage[idx].get_value(arith, value);
                    return 1;
                }
            }
            return 0;
        }

        __host__ __device__ int32_t contract_storage_t::set_value(
            ArithEnv arith,
            const bn_t &key,
            const bn_t &value)
        {
            uint32_t idx;
            for (idx = 0; idx < this->size; idx++)
            {
                if (this->storage[idx].has_key(arith, key))
                {
                    this->storage[idx].set_value(arith, value);
                    return 1;
                }
            }
            if (this->size >= this->capacity)
            {
                storage_element_t *new_storage;
                if (this->capacity == 0)
                {
                    new_storage = new storage_element_t[4];
                    this->capacity = 4;
                } else {
                    new_storage = new storage_element_t[this->capacity * 2];
                    this->capacity *= 2;
                    std::copy(this->storage, this->storage + this->size, new_storage);
                }
                delete[] this->storage;
                this->storage = new_storage;
            }
            this->storage[this->size].set_key(arith, key);
            this->storage[this->size].set_value(arith, value);
            this->size++;
            return 1;
        }

        __host__ int32_t contract_storage_t::from_json(
            const cJSON *contract_storage_json,
            int32_t managed)
        {
            if (
                cJSON_IsNull(contract_storage_json) ||
                cJSON_IsInvalid(contract_storage_json) ||
                (!cJSON_IsArray(contract_storage_json))
            )
            {
                return 1;
            }
            this->size = cJSON_GetArraySize(contract_storage_json);
            if (this->size == 0)
            {
                this->capacity = 0;
                this->storage = nullptr;
                return 0;
            }
            this->capacity = 2;
            do
            {
                this->capacity *= 2;
            } while (this->capacity < this->size);
            if (managed)
            {
                CUDA_CHECK(
                    cudaMallocManaged(
                        &this->storage,
                        this->capacity * sizeof(storage_element_t)
                    )
                );
            } else {
                this->storage = new storage_element_t[this->capacity];
            }
            cJSON *element_json = nullptr;
            uint32_t idx = 0;
            cJSON_ArrayForEach(element_json, contract_storage_json)
            {
                this->storage[idx].from_json(element_json);
                idx++;
            }
            return 0;
        }

        __host__ cJSON* contract_storage_t::to_json(int32_t pretty) const
        {
            cJSON *contract_storage_json = cJSON_CreateObject();
            if (this->size == 0)
            {
                return contract_storage_json;
            }
            uint32_t idx = 0;
            char *key_string_ptr = new char[EVM_WORD_SIZE * 2 + 3];
            char *value_string_ptr = new char[EVM_WORD_SIZE * 2 + 3];
            for (idx = 0; idx < this->size; idx++)
            {
                this->storage[idx].add_to_json(
                    contract_storage_json,
                    key_string_ptr,
                    value_string_ptr,
                    pretty);
            }
            delete[] key_string_ptr;
            delete[] value_string_ptr;
            return contract_storage_json;
        }

        __host__ __device__ void contract_storage_t::free_internals(
            int32_t managed)
        {
            if (managed)
            {
                CUDA_CHECK(cudaFree(this->storage));
            } else {
                delete[] this->storage;
            }
            this->storage = nullptr;
        }


    } // namespace storage
} // namespace cuEVM