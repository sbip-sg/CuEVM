// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation
// Programme) Author: Stefan-Dan Ciocirlan Date: 2024-09-15
// SPDX-License-Identifier: MIT

#include <CuEVM/state/storage_element.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM {
__host__ storage_element_t::storage_element_t(
    const cJSON *storage_element_json) {
    this->from_json(storage_element_json);
}

__host__ __device__ void storage_element_t::set_value(ArithEnv &arith,
                                                      const bn_t &value) {
    cgbn_store(arith.env, &(this->value), value);
}

__host__ __device__ void storage_element_t::get_value(ArithEnv &arith,
                                                      bn_t &value) const {
    cgbn_load(arith.env, value, (cgbn_evm_word_t_ptr) & (this->value));
}

__host__ __device__ void storage_element_t::set_key(ArithEnv &arith,
                                                    const bn_t &key) {
    cgbn_store(arith.env, &(this->key), key);
}

__host__ __device__ void storage_element_t::get_key(ArithEnv &arith,
                                                    bn_t &key) const {
    cgbn_load(arith.env, key, (cgbn_evm_word_t_ptr) & (this->key));
}

__host__ __device__ int32_t
storage_element_t::has_key(const evm_word_t key) const {
    return (this->key == key);
}

__host__ __device__ int32_t storage_element_t::has_key(ArithEnv &arith,
                                                       const bn_t &key) const {
    bn_t storage_key;
    cgbn_load(arith.env, storage_key, (cgbn_evm_word_t_ptr) & (this->key));
    return (cgbn_compare(arith.env, storage_key, key) == 0);
}

__host__ __device__ int32_t storage_element_t::is_zero_value() const {
    return (this->value == 0U);
}

__host__ __device__ int32_t
storage_element_t::is_zero_value(ArithEnv &arith) const {
    bn_t storage_value;
    cgbn_load(arith.env, storage_value, (cgbn_evm_word_t_ptr) & (this->value));
    return (cgbn_compare_ui32(arith.env, storage_value, 0U) == 0);
}

__host__ int32_t
storage_element_t::from_json(const cJSON *storage_element_json) {
    uint32_t error_code = ERROR_SUCCESS;
    error_code |= key.from_hex(storage_element_json->string);
    error_code |= value.from_hex(storage_element_json->valuestring);
    return error_code;
}

__host__ int32_t storage_element_t::add_to_json(cJSON *storage_json,
                                                char *key_string_ptr,
                                                char *value_string_ptr,
                                                int32_t pretty) const {
    if (cJSON_IsNull(storage_json)) {
        return 1;
    }
    char *tmp_key_string_ptr = nullptr;
    char *tmp_value_string_ptr = nullptr;
    if (key_string_ptr == nullptr) {
        tmp_key_string_ptr = new char[CuEVM::word_size * 2 + 3];
        key_string_ptr = tmp_key_string_ptr;
    }
    if (value_string_ptr == nullptr) {
        tmp_value_string_ptr = new char[CuEVM::word_size * 2 + 3];
        value_string_ptr = tmp_value_string_ptr;
    }
    key_string_ptr = key.to_hex(key_string_ptr, pretty);
    value_string_ptr = value.to_hex(value_string_ptr, pretty);
    cJSON_AddStringToObject(storage_json, key_string_ptr, value_string_ptr);
    if (tmp_key_string_ptr != nullptr) {
        delete[] tmp_key_string_ptr;
    }
    if (tmp_value_string_ptr != nullptr) {
        delete[] tmp_value_string_ptr;
    }
    return 0;
}

__host__ __device__ void storage_element_t::print() const {
    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    printf("Key: ");
    key.print();
    printf("Value: ");
    value.print();
    __ONE_GPU_THREAD_WOSYNC_END__
}
}  // namespace CuEVM