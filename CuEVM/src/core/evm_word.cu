#include <CuEVM/core/evm_word.cuh>
#include <CuEVM/utils/error_codes.cuh>
#include <CuEVM/utils/evm_utils.cuh>

namespace CuEVM {
__host__ __device__ evm_word_t::evm_word_t(const evm_word_t &src) {
    /*
  #pragma unroll
  for (int32_t index = 0; index < CuEVM::cgbn_limbs; index++) {
    _limbs[index] = src._limbs[index];
  }
  return *this;*/
    memcpy(_limbs, src._limbs, CuEVM::cgbn_limbs * sizeof(uint32_t));
}

__host__ __device__ evm_word_t::evm_word_t(uint32_t value) : evm_word_t() { this->from_uint32_t(value); }

__host__ __device__ evm_word_t &evm_word_t::operator=(const evm_word_t &src) {
    /*
  #pragma unroll
  for (int32_t index = 0; index < CuEVM::cgbn_limbs; index++) {
    _limbs[index] = src._limbs[index];
  }
  return *this;*/
    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    memcpy(_limbs, src._limbs, CuEVM::cgbn_limbs * sizeof(uint32_t));
    __ONE_GPU_THREAD_END__
    return *this;
}

__host__ __device__ evm_word_t &evm_word_t::operator=(uint32_t value) {
    this->from_uint32_t(value);
    return *this;
}

__host__ __device__ int32_t evm_word_t::operator==(const evm_word_t &other) const {
#pragma unroll
    for (int32_t index = 0; index < CuEVM::cgbn_limbs; index++) {
        if (_limbs[index] != other._limbs[index]) {
            return 0;
        }
    }
    return 1;
}

__host__ __device__ int32_t evm_word_t::operator==(const uint32_t &value) const {
    if (_limbs[0] != value) {
        return 0;
    }
#pragma unroll
    for (int32_t index = 1; index < CuEVM::cgbn_limbs; index++) {
        if (_limbs[index] != 0) {
            return 0;
        }
    }
    return 1;
}

__host__ __device__ int32_t evm_word_t::from_hex(const char *hex_string) {
#ifdef __CUDA_ARCH__

#else
    CuEVM::byte_array_t byte_array(hex_string, CuEVM::word_size, BIG_ENDIAN, CuEVM::PaddingDirection::LEFT_PADDING);
    return from_byte_array_t(byte_array);
#endif
}

__host__ __device__ int32_t evm_word_t::from_byte_array_t(byte_array_t &byte_array, int32_t endian) {
    if (byte_array.size != CuEVM::word_size) {
        return ERROR_BYTE_ARRAY_INVALID_SIZE;
    }
    uint8_t *bytes = nullptr;
    if (endian == LITTLE_ENDIAN) {
        bytes = byte_array.data;
#pragma unroll
        for (uint32_t idx = 0; idx < CuEVM::cgbn_limbs; idx++) {
            _limbs[idx] = (*(bytes++) | *(bytes++) << 8 | *(bytes++) << 16 | *(bytes++) << 24);
        }
    } else if (endian == BIG_ENDIAN) {
        bytes = byte_array.data + CuEVM::word_size - 1;
#pragma unroll
        for (uint32_t idx = 0; idx < CuEVM::cgbn_limbs; idx++) {
            _limbs[idx] = (*(bytes--) | *(bytes--) << 8 | *(bytes--) << 16 | *(bytes--) << 24);
        }
    } else {
        return ERROR_NOT_IMPLEMENTED;
    }
    return ERROR_SUCCESS;
}

__host__ __device__ int32_t evm_word_t::from_size_t(size_t value) {
    if (sizeof(size_t) == sizeof(uint64_t)) {
        return from_uint64_t(value);
    } else if (sizeof(size_t) == sizeof(uint32_t)) {
        return from_uint32_t(value);
    } else {
        return ERROR_NOT_IMPLEMENTED;
    }
}

__host__ __device__ int32_t evm_word_t::from_uint64_t(uint64_t value) {
#pragma unroll
    for (uint32_t idx = 2; idx < CuEVM::cgbn_limbs; idx++) {
        _limbs[idx] = 0;
    }
    _limbs[0] = value & 0xFFFFFFFF;
    _limbs[1] = (value >> 32) & 0xFFFFFFFF;
    return ERROR_SUCCESS;
}

__host__ __device__ int32_t evm_word_t::from_uint32_t(uint32_t value) {
#pragma unroll
    for (uint32_t idx = 1; idx < CuEVM::cgbn_limbs; idx++) {
        _limbs[idx] = 0;
    }
    _limbs[0] = value;
    return ERROR_SUCCESS;
}

__host__ __device__ int32_t evm_word_t_compare(const evm_word_t *a, const evm_word_t *b, uint16_t num_limbs = 8) {
    for (uint16_t i = num_limbs - 1; i >= 0; --i) {
        if (a->_limbs[i] != b->_limbs[i]) {
            return (a->_limbs[i] > b->_limbs[i]) ? 1 : -1;
        }
    }
    return 0;
}

__host__ __device__ void evm_word_t::print() const {
    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    for (uint32_t idx = 0; idx < CuEVM::cgbn_limbs; idx++) {
        printf("%08x ", _limbs[CuEVM::cgbn_limbs - 1 - idx]);
    }
    printf("\n");
    __ONE_GPU_THREAD_WOSYNC_END__
}

__host__ __device__ void evm_word_t::print_as_compact_hex() const {
    int first_non_zero = 0;

    printf("\"0x");

    // Iterate over each limb, starting from the most significant one (_limbs[7] to _limbs[0])
    for (int i = 7; i >= 0; --i) {
        if (_limbs[i] != 0 || first_non_zero) {
            // Print the current limb; use "%x" for the first non-zero limb, and "%08x" for the rest
            if (first_non_zero) {
                printf("%08x", _limbs[i]);  // Pad with zeros after the first non-zero limb
            } else {
                printf("%x", _limbs[i]);  // No padding for the first non-zero limb
                first_non_zero = 1;       // Mark the first non-zero limb
            }
        }
    }

    // If all limbs are zero, print "0"
    if (!first_non_zero) {
        printf("0");
    }

    printf("\"");
}

__host__ __device__ char *evm_word_t::to_hex(char *hex_string, int32_t pretty, uint32_t count) const {
    if (hex_string == nullptr) {
        hex_string = new char[count * 8 + 3];
    }
    hex_string[0] = '0';
    hex_string[1] = 'x';
    for (uint32_t idx = 0; idx < count; idx++) {
        CuEVM::utils::hex_from_byte(hex_string + 2 + idx * 8, (_limbs[count - 1 - idx] >> 24) & 0xFF);
        CuEVM::utils::hex_from_byte(hex_string + 2 + idx * 8 + 2, (_limbs[count - 1 - idx] >> 16) & 0xFF);
        CuEVM::utils::hex_from_byte(hex_string + 2 + idx * 8 + 4, (_limbs[count - 1 - idx] >> 8) & 0xFF);
        CuEVM::utils::hex_from_byte(hex_string + 2 + idx * 8 + 6, _limbs[count - 1 - idx] & 0xFF);
    }
    hex_string[count * 8 + 2] = '\0';
    if (pretty) {
        CuEVM::utils::hex_string_without_leading_zeros(hex_string);
    }
    return hex_string;
}

__host__ __device__ int32_t evm_word_t::to_byte_array_t(byte_array_t &byte_array, int32_t endian) const {
    byte_array.grow(CuEVM::word_size, 1);
    uint8_t *bytes = nullptr;
    if (endian == BIG_ENDIAN) {
        __ONE_GPU_THREAD_WOSYNC_BEGIN__
        bytes = byte_array.data + CuEVM::word_size - 1;
        // todo : Parallel copy
        for (uint32_t idx = 0; idx < CuEVM::cgbn_limbs; idx++) {
            *(bytes--) = _limbs[idx] & 0xFF;
            *(bytes--) = (_limbs[idx] >> 8) & 0xFF;
            *(bytes--) = (_limbs[idx] >> 16) & 0xFF;
            *(bytes--) = (_limbs[idx] >> 24) & 0xFF;
        }
        __ONE_GPU_THREAD_WOSYNC_END__
    } else if (endian == LITTLE_ENDIAN) {
        __ONE_GPU_THREAD_WOSYNC_BEGIN__
        bytes = byte_array.data;
        for (uint32_t idx = 0; idx < CuEVM::cgbn_limbs; idx++) {
            *(bytes++) = (_limbs[idx] >> 24) & 0xFF;
            *(bytes++) = (_limbs[idx] >> 16) & 0xFF;
            *(bytes++) = (_limbs[idx] >> 8) & 0xFF;
            *(bytes++) = _limbs[idx] & 0xFF;
        }
        __ONE_GPU_THREAD_WOSYNC_END__
    } else {
        return ERROR_NOT_IMPLEMENTED;
    }
    return ERROR_SUCCESS;
}

__host__ __device__ int32_t evm_word_t::to_bit_array_t(byte_array_t &bit_array, int32_t endian) const {
    bit_array.grow(CuEVM::word_bits, 1);
    uint8_t *bits = nullptr;
    if (endian == BIG_ENDIAN) {
        bits = bit_array.data;
        for (int32_t idx = CuEVM::cgbn_limbs - 1; idx >= 0; idx--) {
            for (int bit = 31; bit >= 0; bit--) {
                *(bits++) = (uint8_t)((_limbs[idx] >> bit) & 0x01);
                // bit_array.data[ (CuEVM::cgbn_limbs - 1 - idx) * 32  +
                // (31-bit)] = (uint8_t)((_limbs[idx] >> bit) & 0x01);
            }
        }
    } else if (endian == LITTLE_ENDIAN) {
        bits = bit_array.data;
        for (uint32_t idx = 0; idx < CuEVM::cgbn_limbs; idx++) {
            for (int bit = 0; bit < 32; bit++) {
                *(bits++) = (_limbs[idx] >> bit) & 0x01;
            }
        }
    }
    return ERROR_SUCCESS;
}
}  // namespace CuEVM
