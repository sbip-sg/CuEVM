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

__host__ __device__ evm_word_t &evm_word_t::operator=(const evm_word_t &src) {
    /*
  #pragma unroll
  for (int32_t index = 0; index < CuEVM::cgbn_limbs; index++) {
    _limbs[index] = src._limbs[index];
  }
  return *this;*/
    memcpy(_limbs, src._limbs, CuEVM::cgbn_limbs * sizeof(uint32_t));
    return *this;
}

__host__ __device__ int32_t
evm_word_t::operator==(const evm_word_t &other) const {
#pragma unroll
    for (int32_t index = 0; index < CuEVM::cgbn_limbs; index++) {
        if (_limbs[index] != other._limbs[index]) {
            return 0;
        }
    }
    return 1;
}

__host__ __device__ int32_t
evm_word_t::operator==(const uint32_t &value) const {
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

__host__ int32_t evm_word_t::from_hex(const char *hex_string) {
    CuEVM::byte_array_t byte_array(hex_string, CuEVM::word_size, BIG_ENDIAN,
                                   CuEVM::PaddingDirection::LEFT_PADDING);
    return from_byte_array_t(byte_array);
}

__host__ __device__ int32_t
evm_word_t::from_byte_array_t(byte_array_t &byte_array) {
    if (byte_array.size != CuEVM::word_size) {
        return ERROR_BYTE_ARRAY_INVALID_SIZE;
    }
    uint8_t *bytes = byte_array.data;
#pragma unroll
    for (uint32_t idx = 0; idx < CuEVM::cgbn_limbs; idx++) {
        _limbs[idx] = (*(bytes++) | *(bytes++) << 8 | *(bytes++) << 16 |
                       *(bytes++) << 24);
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

__host__ __device__ void evm_word_t::print() const {
    __ONE_GPU_THREAD_BEGIN__
    for (uint32_t idx = 0; idx < CuEVM::cgbn_limbs; idx++)
        printf("%08x ", _limbs[CuEVM::cgbn_limbs - 1 - idx]);
    printf("\n");
    __ONE_GPU_THREAD_END__
}

__host__ char *evm_word_t::to_hex(char *hex_string, int32_t pretty,
                                  uint32_t count) const {
    if (hex_string == nullptr) {
        hex_string = new char[count * 8 + 3];
    }
    hex_string[0] = '0';
    hex_string[1] = 'x';
    for (uint32_t idx = 0; idx < count; idx++) {
        sprintf(hex_string + 2 + idx * 8, "%08x", _limbs[count - 1 - idx]);
    }
    hex_string[count * 8 + 2] = '\0';
    if (pretty) {
        CuEVM::utils::hex_string_without_leading_zeros(hex_string);
    }
    return hex_string;
}

__host__ __device__ byte_array_t *evm_word_t::to_byte_array_t(
    byte_array_t *byte_array) const {
    if (byte_array == nullptr) {
        byte_array = new byte_array_t(CuEVM::word_size);
    }
    uint8_t *bytes = byte_array->data + CuEVM::word_size - 1;
    for (uint32_t idx = 0; idx < CuEVM::cgbn_limbs; idx++) {
        *(bytes--) = _limbs[idx] & 0xFF;
        *(bytes--) = (_limbs[idx] >> 8) & 0xFF;
        *(bytes--) = (_limbs[idx] >> 16) & 0xFF;
        *(bytes--) = (_limbs[idx] >> 24) & 0xFF;
    }
    return byte_array;
}

__host__ __device__ byte_array_t *evm_word_t::to_bit_array_t(
    byte_array_t *bit_array) const {
    if (bit_array == nullptr) {
        bit_array = new byte_array_t(CuEVM::word_bits);
    }
    uint8_t *bytes = bit_array->data + CuEVM::word_bits - 1;
    for (uint32_t idx = 0; idx < CuEVM::cgbn_limbs; idx++) {
        for (int bit = 0; bit < 32; bit++) {
            *(bytes--) = (_limbs[idx] >> bit) & 0x01;
        }
    }
    return bit_array;
}
}  // namespace CuEVM