#include <CuEVM/state/contract_storage.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM {

__host__ __device__ contract_storage_t::~contract_storage_t() { free(); }

__host__ __device__ void contract_storage_t::free() {
    if ((storage != nullptr) && (capacity > 0)) {
        delete[] storage;
    }

    clear();
}

__host__ void contract_storage_t::free_managed() {
    if (storage != nullptr && capacity > 0) {
        CUDA_CHECK(cudaFree(storage));
    }
    clear();
}

__host__ __device__ void contract_storage_t::clear() {
    storage = nullptr;
    size = 0;
    capacity = 0;
}

__host__ __device__ contract_storage_t &contract_storage_t::operator=(const contract_storage_t &other) {
    storage_element_t *tmp_storage;
    if (this == &other) {
        return *this;
    }
    if (capacity != other.capacity) {
        free();
        size = other.size;
        capacity = other.capacity;
        // #ifdef __CUDA_ARCH__
        //         printf("contract_storage_t::operator= idx %d size %d capacity %d\n", threadIdx.x, size, capacity);
        // #endif

        if (capacity > 0) {
            tmp_storage = new storage_element_t[capacity];
        }

        storage = tmp_storage;
    }
    if (other.size > 0) memcpy(storage, other.storage, other.size * sizeof(storage_element_t));
    return *this;
}

__host__ __device__ int32_t contract_storage_t::get_value(const evm_word_t &key, evm_word_t &value) const {
    uint32_t idx = 0;
    // #ifdef __CUDA_ARCH__
    //     printf("contract_storage_t::get_value idx %d size %d capacity %d storage %p\n", threadIdx.x, size, capacity,
    //            storage);
    // #endif
    for (idx = 0; idx < size; idx++) {
        if (storage[idx].has_key(key)) {
            storage[idx].get_value(value);
            return ERROR_SUCCESS;
        }
    }
    return ERROR_STORAGE_KEY_NOT_FOUND;
}

__host__ __device__ int32_t contract_storage_t::set_value(const evm_word_t &key, const evm_word_t &value) {
    uint32_t idx;

    for (idx = 0; idx < size; idx++) {
        if (storage[idx].has_key(key)) {
            storage[idx].set_value(value);
            return ERROR_SUCCESS;
        }
    }

    __SHARED_MEMORY__ storage_element_t *new_storage[CGBN_IBP];
    // #ifdef __CUDA_ARCH__
    //     printf("contract_storage_t::set_value before allocateidx %d size %d capacity %d  storage %p\n", threadIdx.x,
    //     size,
    //            capacity, storage);
    // #endif
    if (size >= capacity) {
        if (capacity == 0) {
            capacity = CuEVM::initial_storage_capacity;
        } else {
            capacity *= 2;
        }
        __ONE_GPU_THREAD_WOSYNC_BEGIN__
        new_storage[INSTANCE_IDX_PER_BLOCK] = new storage_element_t[capacity];
        // printf("allocate new storage %p, capacity %d\n", new_storage, capacity);
        if (size > 0) {
            memcpy(new_storage[INSTANCE_IDX_PER_BLOCK], storage, size * sizeof(storage_element_t));
        }
        delete[] storage;
        __ONE_GPU_THREAD_END__
        storage = new_storage[INSTANCE_IDX_PER_BLOCK];
        // printf("set storage size %d capacity %d  storage %p, new_storage %p\n", size, capacity, storage,
        // new_storage);
    }
    // #ifdef __CUDA_ARCH__
    //     printf("contract_storage_t::set_value idx %d size %d capacity %d, storage %p\n", threadIdx.x, size, capacity,
    //     storage); printf("contract_storage_t::set_value idx %d storage 0 key %d val %d\n", threadIdx.x,
    //     storage[0].key._limbs[0], storage[0].value._limbs[0]);
    // #endif
    // #ifdef __CUDA_ARCH__
    //     printf("contract_storage_t::set_value idx %d size %d capacity %d, storage %p\n", threadIdx.x, size, capacity,
    //            storage);
    // #endif
    storage[size].set_key(key);
    storage[size].set_value(value);
    size++;
    return ERROR_SUCCESS;
}

__host__ __device__ void contract_storage_t::update(const contract_storage_t &other) {
    evm_word_t key, value;
    for (uint32_t idx = 0; idx < other.size; idx++) {
        other.storage[idx].get_key(key);
        other.storage[idx].get_value(value);
        set_value(key, value);
    }
}

__host__ int32_t contract_storage_t::from_json(const cJSON *contract_storage_json, int32_t managed) {
    if (cJSON_IsNull(contract_storage_json) || cJSON_IsInvalid(contract_storage_json) /*||
                                               (!cJSON_IsArray(contract_storage_json))*/
    ) {
        return ERROR_INVALID_JSON;
    }
    size = cJSON_GetArraySize(contract_storage_json);
    if (size == 0) {
        capacity = 0;
        storage = nullptr;
        return ERROR_SUCCESS;
    }
    capacity = CuEVM::initial_storage_capacity / 2;
    do {
        capacity *= 2;
    } while (capacity < size);
    if (managed) {
        CUDA_CHECK(cudaMallocManaged(&storage, capacity * sizeof(storage_element_t)));
    } else {
        storage = new storage_element_t[capacity];
    }
    cJSON *element_json = nullptr;
    uint32_t idx = 0;
    cJSON_ArrayForEach(element_json, contract_storage_json) {
        storage[idx].from_json(element_json);
        idx++;
    }
    return 0;
}

__host__ cJSON *contract_storage_t::to_json(int32_t pretty) const {
    cJSON *contract_storage_json = cJSON_CreateObject();
    if (size == 0) {
        return contract_storage_json;
    }
    uint32_t idx = 0;
    char *key_string_ptr = new char[CuEVM::word_size * 2 + 3];
    char *value_string_ptr = new char[CuEVM::word_size * 2 + 3];
    for (idx = 0; idx < size; idx++) {
        storage[idx].add_to_json(contract_storage_json, key_string_ptr, value_string_ptr, pretty);
    }
    delete[] key_string_ptr;
    delete[] value_string_ptr;
    return contract_storage_json;
}

__host__ __device__ void contract_storage_t::print() const {
    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    printf("Storage size: %u\n", size);
    for (uint32_t idx = 0; idx < size; idx++) {
        printf("Element %u:\n", idx);
        storage[idx].print();
    }
    __ONE_GPU_THREAD_WOSYNC_END__
}

__host__ int32_t contract_storage_t::has_key(const evm_word_t &key, uint32_t &index) const {
    for (index = 0; index < size; index++) {
        if (storage[index].has_key(key)) {
            return ERROR_SUCCESS;
        }
    }
    return ERROR_STORAGE_KEY_NOT_FOUND;
}

__host__ cJSON *contract_storage_t::merge_json(const contract_storage_t &storage1, const contract_storage_t &storage2,
                                               const int32_t pretty) {
    cJSON *storage_json = cJSON_CreateObject();
    uint8_t *written = new uint8_t[storage2.size];
    memset(written, 0, storage2.size);
    char *key_string_ptr = new char[CuEVM::word_size * 2 + 3];
    char *value_string_ptr = new char[CuEVM::word_size * 2 + 3];
    for (uint32_t idx = 0; idx < storage1.size; idx++) {
        uint32_t jdx;
        if (storage2.has_key(storage1.storage[idx].key, jdx) == ERROR_SUCCESS) {
            storage2.storage[jdx].key.to_hex(key_string_ptr, pretty);
            storage2.storage[jdx].value.to_hex(value_string_ptr, pretty);
            written[jdx] = 1;
        } else {
            storage1.storage[idx].key.to_hex(key_string_ptr, pretty);
            storage1.storage[idx].value.to_hex(value_string_ptr, pretty);
        }

        // if value is different than 0
        if (value_string_ptr[2] != '0' || value_string_ptr[3] != '\0') {
            cJSON_AddStringToObject(storage_json, key_string_ptr, value_string_ptr);
        }
    }

    for (uint32_t jdx = 0; jdx < storage2.size; jdx++) {
        if (written[jdx] == 0) {
            storage2.storage[jdx].key.to_hex(key_string_ptr, pretty);
            storage2.storage[jdx].value.to_hex(value_string_ptr, pretty);
            if (value_string_ptr[2] != '0' || value_string_ptr[3] != '\0') {
                cJSON_AddStringToObject(storage_json, key_string_ptr, value_string_ptr);
            }
        }
    }
    delete[] written;
    delete[] key_string_ptr;
    delete[] value_string_ptr;
    return storage_json;
}

}  // namespace CuEVM