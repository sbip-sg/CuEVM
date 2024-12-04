
#include <CuEVM/state/storage_element.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM {
__host__ storage_element_t::storage_element_t(const cJSON *storage_element_json) {
    this->from_json(storage_element_json);
}

__host__ __device__ void storage_element_t::set_value(const evm_word_t value) { this->value = value; }

__host__ __device__ void storage_element_t::get_value(evm_word_t value) const { value = this->value; }

__host__ __device__ void storage_element_t::set_key(const evm_word_t key) { this->key = key; }

__host__ __device__ void storage_element_t::get_key(evm_word_t key) const { key = this->key; }

__host__ __device__ int32_t storage_element_t::has_key(const evm_word_t key) const { return (this->key == key); }

__host__ __device__ int32_t storage_element_t::is_zero_value() const { return (this->value == 0U); }

__host__ int32_t storage_element_t::from_json(const cJSON *storage_element_json) {
    uint32_t error_code = ERROR_SUCCESS;
    error_code |= key.from_hex(storage_element_json->string);
    error_code |= value.from_hex(storage_element_json->valuestring);
    return error_code;
}

__host__ int32_t storage_element_t::add_to_json(cJSON *storage_json, char *key_string_ptr, char *value_string_ptr,
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