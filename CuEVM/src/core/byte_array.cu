#include <CuEVM/core/byte_array.cuh>
#include <CuEVM/utils/error_codes.cuh>
#include <CuEVM/utils/evm_utils.cuh>

namespace CuEVM {
__device__ byte_array_t::byte_array_t(const uint32_t size) : size(size) {
    uint8_t *local_data = nullptr;
    if (size > 0) {
        local_data = new uint8_t[size];
        memset(local_data, 0, size * sizeof(uint8_t));
    }
    data = local_data;
}

__device__ byte_array_t::byte_array_t(uint8_t *data, uint32_t size) : size(size) {
    uint8_t *local_data = nullptr;

    if (size > 0) {
        local_data = new uint8_t[size];
        memcpy(local_data, data, size * sizeof(uint8_t));
    }

    this->data = local_data;
}

__device__ byte_array_t::byte_array_t(const byte_array_t &src_byte_array, uint32_t offset, uint32_t size) : size(size) {
    uint8_t *local_data = nullptr;
    // printf("byte_array_t::byte_array_t %d %d %d %d\n", THREADIDX, THREAD_IDX_PER_INSTANCE, offset, size);
    if (size > 0) {
        local_data = new uint8_t[size];

        if (offset < src_byte_array.size)
            for (uint32_t idx = 0; idx < min(size, src_byte_array.size - offset); idx++) {
                local_data[idx] = src_byte_array.data[offset + idx];
            }
    }
    this->data = local_data;
}

__host__ byte_array_t::byte_array_t(const char *hex_string, int32_t endian, PaddingDirection padding)
    : size(0), data(nullptr) {
    from_hex(hex_string, endian, padding);
}

__host__ byte_array_t::byte_array_t(const char *hex_string, uint32_t size, int32_t endian, PaddingDirection padding)
    : size(size), data(nullptr) {
    from_hex(hex_string, endian, padding);
}

__device__ byte_array_t::~byte_array_t() { free(); }

__host__ __device__ void byte_array_t::free() {
    if ((size > 0) && (data != nullptr)) {
#ifndef __CUDA_ARCH__
        delete[] data;
#endif
        clear();
    }
}
__host__ void byte_array_t::free_managed() {
    if ((size > 0) && (data != nullptr)) {
        CUDA_CHECK(cudaFree(data));
    }
    clear();
}
__host__ __device__ void byte_array_t::clear() {
    data = nullptr;
    size = 0;
}

__device__ byte_array_t::byte_array_t(const byte_array_t &other) : size(other.size) {
    uint8_t *local_data = nullptr;
    if (size > 0) {
        local_data = new uint8_t[size];
        memcpy(local_data, other.data, size * sizeof(uint8_t));
    }
    data = local_data;
}

__device__ byte_array_t &byte_array_t::operator=(const byte_array_t &other) {
    uint8_t *local_data = nullptr;

    if (this != &other) {
        free();  // can do outside, all thread set to null

        if (other.size > 0) {
            // printf("other size !=0 %p \n", tmp_data);
            local_data = new uint8_t[other.size];

            for (uint32_t idx = 0; idx < other.size; idx++) {
                local_data[idx] = other.data[idx];
            }
        }

        data = local_data;
        size = other.size;
    }

    return *this;
}

__device__ int32_t byte_array_t::grow(uint32_t new_size, int32_t zero_padding) {
    // printf("byte_array_t::grow %d %d size %d zero_padding %d, new_size %d, data %p\n", THREADIDX,
    //        THREAD_IDX_PER_INSTANCE, size, zero_padding, new_size, data);
    if (new_size == size) return ERROR_SUCCESS;
    uint8_t *new_data;

    new_data = new uint8_t[new_size];

    if (zero_padding) {
        for (uint32_t idx = 0; idx < new_size; idx++) {
            new_data[idx] = 0;
        }
    }
    //  memset(new_data, 0, new_size * sizeof(uint8_t));
    if (size > 0) {
        for (uint32_t idx = 0; idx < min(new_size, size); idx++) {
            new_data[idx] = data[idx];
        }
        delete[] data;
    }

    data = new_data;
    size = new_size;
    return ERROR_SUCCESS;
}

__device__ uint32_t byte_array_t::has_value(uint8_t value) const {
    uint32_t error_code = ERROR_VALUE_NOT_FOUND;
    uint32_t index;
    for (index = 0; index < size; index++) {
        if (data[index] == value) {
            return ERROR_SUCCESS;
        }
    }
    return error_code;
}

__device__ void byte_array_t::print() const {
    printf("size: %u\n", size);
    printf("data: ");
    for (uint32_t index = 0; index < size; index++) printf("%02x", data[index]);
    printf("\n");
}

__device__ char *byte_array_t::to_hex() const {
    char *hex_string = new char[size * 2 + 3];  // 3 - 0x and \0
    hex_string[0] = '0';
    hex_string[1] = 'x';
    char *tmp_hex_string = (char *)hex_string + 2;
    uint8_t *tmp_data = data;
    for (uint32_t idx = 0; idx < size; idx++) {
        CuEVM::utils::hex_from_byte(tmp_hex_string, *(tmp_data++));
        tmp_hex_string += 2;
    }
    hex_string[size * 2 + 2] = 0;
    return hex_string;
}

__host__ cJSON *byte_array_t::to_json() const {
    cJSON *data_json = cJSON_CreateObject();
    cJSON_AddNumberToObject(data_json, "size", size);
    if (size > 0) {
        char *hex_string = to_hex();
        cJSON_AddStringToObject(data_json, "data", hex_string);
        delete[] hex_string;
    } else {
        cJSON_AddStringToObject(data_json, "data", "0x");
    }
    return data_json;
}

__host__ int32_t byte_array_t::from_hex_set_le(const char *clean_hex_string, int32_t length) {
    if ((length < 0) || ((size * 2) < length)) {
        return 1;
    }
    if (length > 0) {
        char *current_char;
        current_char = (char *)clean_hex_string;
        int32_t index;
        uint8_t *dst_ptr;
        dst_ptr = data;
        for (index = 0; index < ((length + 1) / 2) - 1; index++) {
            *(dst_ptr++) = CuEVM::utils::byte_from_two_hex_char(*(current_char), *(current_char + 1));
            current_char += 2;
        }
        if (length % 2 == 1) {
            *(dst_ptr++) = CuEVM::utils::byte_from_two_hex_char(*current_char++, '0');
        } else {
            *(dst_ptr++) = CuEVM::utils::byte_from_two_hex_char(*(current_char), *(current_char + 1));
            current_char += 2;
        }
    }
    return 0;
}

__host__ int32_t byte_array_t::from_hex_set_be(const char *clean_hex_string, int32_t length, PaddingDirection padding) {
    if ((length < 0) || ((size * 2) < length)) {
        return 1;
    }
    if (length > 0) {
        char *current_char;
        current_char = (char *)clean_hex_string;
        uint8_t *dst_ptr;
        if (padding == PaddingDirection::RIGHT_PADDING) {  // right padding
            dst_ptr = data + size - 1;
        } else if (padding == PaddingDirection::LEFT_PADDING) {  // left padding
            dst_ptr = data + (length + 1) / 2 - 1;
        } else {
            return 1;
        }

        if (length % 2 == 1) {
            *dst_ptr-- = CuEVM::utils::byte_from_two_hex_char('0', *current_char++);
        } else {
            *dst_ptr-- = CuEVM::utils::byte_from_two_hex_char(*(current_char), *(current_char + 1));
            current_char += 2;
        }
        while (*current_char != '\0') {
            *dst_ptr-- = CuEVM::utils::byte_from_two_hex_char(*(current_char), *(current_char + 1));
            current_char += 2;
        }
    }
    return 0;
}

__host__ int32_t byte_array_t::from_hex(const char *hex_string, int32_t endian, PaddingDirection padding) {
    char *tmp_hex_char;
    tmp_hex_char = (char *)hex_string;
    printf("byte array from hex: %s\n", hex_string);
    int32_t length = CuEVM::utils::clean_hex_string(&tmp_hex_char);
    if (length < 0) {
        data = nullptr;
        return ERROR_INVALID_HEX_STRING;
    }
    uint32_t new_size = (size == 0) ? (length + 1) / 2 : size;
    if (size > 0) {
        free();
    }
    size = new_size;
    printf("size: %d\n", size);
    if (size > 0) {
        data = new uint8_t[size];
        memset(data, 0, size * sizeof(uint8_t));

    } else
        data = nullptr;
    int32_t error_code = ERROR_SUCCESS;
    if (endian == LITTLE_ENDIAN) {
        error_code = this->from_hex_set_le(tmp_hex_char, length);
    } else {
        error_code = this->from_hex_set_be(tmp_hex_char, length, padding);
    }
    if (error_code != ERROR_SUCCESS) {
        delete[] data;

        data = nullptr;
        size = 0;
    }
    printf("end of byte array from hex\n");
    for (uint32_t idx = 0; idx < size; idx++) {
        printf("%02x", data[idx]);
    }
    printf("\n");
    return error_code;
}

__device__ int32_t byte_array_t::padded_copy_BE(const byte_array_t src) {
    uint32_t copy_size;
    int32_t size_diff;
    if (src.size == size) {
        size_diff = 0;
        copy_size = src.size;
    } else if (src.size < size) {
        size_diff = 1;
        copy_size = src.size;
    } else {
        size_diff = -1;
        copy_size = size;
    }
    memcpy(data, src.data, copy_size);
    memset(data + copy_size, 0, size - src.size);
    return size_diff;
}

__device__ uint8_t &byte_array_t::operator[](uint32_t index) { return data[index]; }

}  // namespace CuEVM
