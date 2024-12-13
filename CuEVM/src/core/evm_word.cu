#include <CuEVM/core/evm_word.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM {
__host__ __device__ evm_word_t::evm_word_t(const evm_word_t &src) {
#pragma unroll
    for (int32_t index = 0; index < CuEVM::cgbn_limbs; index++) {
        words[index] = src.words[index];
    }
}

__host__ __device__ evm_word_t::evm_word_t(uint32_t value) : evm_word_t() { this->from_uint32_t(value); }

__host__ __device__ evm_word_t &evm_word_t::operator=(const evm_word_t &src) { uint256_cpy(this, &src); }

__host__ __device__ evm_word_t &evm_word_t::operator=(uint32_t value) {
    this->from_uint32_t(value);
    return *this;
}

__host__ __device__ int32_t evm_word_t::operator==(const evm_word_t &other) const {
    return uint256_cmp(this, &other) == 0;
}

__host__ __device__ int32_t evm_word_t::from_hex(const char *hex_string) { uint256_from_hex(this, hex_string); }

__device__ int32_t evm_word_t::from_byte_array_t(byte_array_t &byte_array, int32_t endian) {
    uint256_from_bytes(this, byte_array.data, byte_array.size);
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

__host__ __device__ void evm_word_t::set_zero() { uint256_set_zero(this); }

__host__ __device__ uint32_t evm_word_t::get_uint32_t() const { return uint256_get_uint32_t(this); }

__host__ __device__ int32_t evm_word_t::from_uint64_t(uint64_t value) {
#pragma unroll
    for (uint32_t idx = 2; idx < CuEVM::cgbn_limbs; idx++) {
        words[idx] = 0;
    }
    words[0] = value & 0xFFFFFFFF;
    words[1] = (value >> 32) & 0xFFFFFFFF;
    return ERROR_SUCCESS;
}

__host__ __device__ int32_t evm_word_t::from_uint32_t(uint32_t value) { uint256_from_uint32(this, value); }

__host__ __device__ int32_t evm_word_t_compare(const evm_word_t *a, const evm_word_t *b, uint16_t num_limbs = 8) {
    return uint256_cmp(a, b);
}

__host__ __device__ void evm_word_t::print() const {
    for (uint32_t idx = 0; idx < CuEVM::cgbn_limbs; idx++) {
        printf("%08x ", words[CuEVM::cgbn_limbs - 1 - idx]);
    }
    printf("\n");
}

__host__ __device__ char *evm_word_t::to_hex(char *hex_string, int32_t pretty, uint32_t count) const {
    if (hex_string == nullptr) {
        hex_string = new char[count * 8 + 3];
    }
    hex_string[0] = '0';
    hex_string[1] = 'x';
    return uint256_to_hex(&hex_string[2], this);
}

__host__ __device__ char *evm_word_t::address_to_hex(char *hex_string, uint32_t count) const {
    if (hex_string == nullptr) {
        hex_string = new char[count * 5 + 3];
    }
    hex_string[0] = '0';
    hex_string[1] = 'x';
    // for (uint32_t idx = 3; idx < count; idx++) {
    //     CuEVM::utils::hex_from_byte(hex_string + 2 + (idx - 3) * 8, (_limbs[count - 1 - idx] >> 24) & 0xFF);
    //     CuEVM::utils::hex_from_byte(hex_string + 2 + (idx - 3) * 8 + 2, (_limbs[count - 1 - idx] >> 16) & 0xFF);
    //     CuEVM::utils::hex_from_byte(hex_string + 2 + (idx - 3) * 8 + 4, (_limbs[count - 1 - idx] >> 8) & 0xFF);
    //     CuEVM::utils::hex_from_byte(hex_string + 2 + (idx - 3) * 8 + 6, _limbs[count - 1 - idx] & 0xFF);
    // }
    // hex_string[43] = '\0';

    return hex_string;
}

__host__ __device__ int32_t evm_word_t::to_byte_array_t(byte_array_t &byte_array, int32_t endian) const {
    byte_array.grow(CuEVM::word_size, 1);
    uint256_to_bytes(byte_array.data, this, byte_array.size);
}

__host__ __device__ int32_t evm_word_t::to_bit_array_t(byte_array_t &bit_array, int32_t endian) const {
    bit_array.grow(CuEVM::word_bits, 1);
    uint8_t *bits = nullptr;

    // if (endian == BIG_ENDIAN) {
    //     bits = bit_array.data;
    //     for (int32_t idx = CuEVM::cgbn_limbs - 1; idx >= 0; idx--) {
    //         for (int bit = 31; bit >= 0; bit--) {
    //             *(bits++) = (uint8_t)((_limbs[idx] >> bit) & 0x01);
    //         }
    //     }
    // } else if (endian == LITTLE_ENDIAN) {
    //     bits = bit_array.data;
    //     for (uint32_t idx = 0; idx < CuEVM::cgbn_limbs; idx++) {
    //         for (int bit = 0; bit < 32; bit++) {
    //             *(bits++) = (_limbs[idx] >> bit) & 0x01;
    //         }
    //     }
    // }
    return ERROR_SUCCESS;
}
}  // namespace CuEVM
