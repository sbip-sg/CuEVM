#include <CuEVM/core/memory.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM::memory {
__device__ void evm_memory_t::print() const {
    printf("Memory data: \n");
    printf("Size: %d\n", size);
    printf("Memory cost: %lu\n", memory_cost);
    printf("\n");
    for (uint32_t i = 0; i < size; i++) {
        printf("%02x ", data[i]);
    }
    printf("\n");
}

__device__ void evm_memory_t::increase_memory_cost(gas_t memory_expansion_cost) {
    memory_cost += memory_expansion_cost;
}

__device__ int32_t evm_memory_t::grow(uint32_t new_size) {
    if (new_size > size) {
        new_size = (new_size + 31) / 32 * 32;
        uint8_t *new_data = new uint8_t[new_size];
        if (size > 0) {
            memcpy(new_data, data, size);
        }
        if (data != nullptr) {
            delete[] data;
        }
        data = new_data;
        size = new_size;
        if (data == nullptr) {
            printf("Memory allocation failed, size: %d\n", new_size);
            return ERR_MEMORY_INVALID_ALLOCATION;
        }
    }

    return ERROR_SUCCESS;
}

__device__ int32_t evm_memory_t::get(uint32_t index, uint32_t length, uint8_t *&data_) {
    int32_t error_code = ERROR_SUCCESS;
    if (length == 0) {
        data_ = nullptr;
        return error_code;
    }

    error_code |= grow(index + length);
    if (error_code == ERROR_SUCCESS) {
        data_ = this->data + index;
    } else {
        data_ = nullptr;
    }
    return error_code;
}

__device__ int32_t evm_memory_t::copy(uint32_t index, uint32_t length, uint8_t *&data_) {
    int32_t error_code = ERROR_SUCCESS;
    if (length == 0) {
        data_ = nullptr;
        return error_code;
    }
    if (data_ == nullptr) {
        data_ = new uint8_t[length];
    }
    error_code |= grow(index + length);
    if (error_code == ERROR_SUCCESS) {
        memcpy(data_, this->data + index, length);
    }

    return error_code;
}

__device__ int32_t evm_memory_t::set_zero(const uint32_t index, const uint32_t length) {
    int32_t error_code = ERROR_SUCCESS;
    if (length == 0) {
        return error_code;
    }
    error_code |= grow(index + length);
    if (error_code == ERROR_SUCCESS) {
        memset(this->data + index, 0, length);
    }
    return error_code;
}

__device__ int32_t evm_memory_t::set(uint8_t *data_, uint32_t data_size, const uint32_t index, const uint32_t length) {
    int32_t error_code = ERROR_SUCCESS;
    if (length == 0) {
        return error_code;
    }
    error_code |= grow(index + length);
    if (error_code == ERROR_SUCCESS) {
        if (data_ != nullptr) {
            if (data_size >= length) {
                memcpy(this->data + index, data_, length);
            } else {
                memcpy(this->data + index, data_, data_size);
                memset(this->data + index + data_size, 0, length - data_size);
            }
        }
    }

    return error_code;
}

}  // namespace CuEVM::memory
