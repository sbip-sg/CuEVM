#include <CuEVM/core/evm_word.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM {
__host__ __device__ evm_word_t::evm_word_t(const evm_word_t &src) {
    memcpy(words, src.words, sizeof(uint32_t) * uint256_limbs);
}

__host__ __device__ evm_word_t::evm_word_t(uint32_t value) : evm_word_t() { this->from_uint32_t(value); }

__host__ __device__ evm_word_t &evm_word_t::operator=(const evm_word_t &src) {
    uint256_cpy(this, &src);
    return *this;
}

__host__ __device__ evm_word_t &evm_word_t::operator=(uint32_t value) {
    this->from_uint32_t(value);
    return *this;
}

__host__ __device__ int32_t evm_word_t::operator==(const evm_word_t &other) const {
    return uint256_cmp(this, &other) == 0;
}

__host__ int32_t evm_word_t::from_hex(const char *hex_string) {
    uint256_from_hex(this, hex_string);
    return 0;
}

__host__ __device__ int32_t evm_word_t::from_byte_array_t(byte_array_t &byte_array, int32_t endian) {
    uint256_from_bytes(this, byte_array.data, byte_array.size);
    return 0;
}

__host__ __device__ void evm_word_t::set_zero() { uint256_set_zero(this); }

__host__ __device__ uint32_t evm_word_t::get_uint32_t() const { return uint256_get_uint32_t(this); }

__host__ __device__ int32_t evm_word_t::from_uint64_t(uint64_t value) {
#pragma unroll
    for (uint32_t idx = 2; idx < uint256_limbs; idx++) {
        words[idx] = 0;
    }
    words[0] = value & 0xFFFFFFFF;
    words[1] = (value >> 32) & 0xFFFFFFFF;
    return ERROR_SUCCESS;
}

__host__ __device__ int32_t evm_word_t::from_uint32_t(uint32_t value) {
    uint256_from_uint32(this, value);
    return 0;
}

__host__ __device__ int32_t evm_word_t_compare(const evm_word_t *a, const evm_word_t *b, uint16_t num_limbs = 8) {
    return uint256_cmp(a, b);
}

__host__ __device__ void evm_word_t::print() const {
    for (uint32_t idx = 0; idx < uint256_limbs; idx++) {
        printf("%08x ", words[uint256_limbs - 1 - idx]);
    }
    printf("\n");
}

__host__ __device__ char *evm_word_t::to_hex(char *hex_string, int32_t pretty, uint32_t count) const {
    if (hex_string == nullptr) {
        hex_string = new char[count * 8 + 3];
    }

    hex_string[0] = '0';
    hex_string[1] = 'x';
    uint256_to_hex(&hex_string[2], this);
    if (pretty) {
        // remove leading zeros
        uint32_t idx = 2;
        while (hex_string[idx] == '0' && idx < count * 8 + 1) {
            idx++;
        }
        if (idx > 2) {
            char *temp = new char[count * 8 + 3 - idx];

            memcpy(temp, &hex_string[idx], count * 8 + 3 - idx);
            memcpy(&hex_string[2], temp, count * 8 + 3 - idx);
            delete[] temp;
        }
    }
    return hex_string;
}

__host__ __device__ bool evm_word_t::is_precompile() const {
    // address cleaned 160 bits  = 5 limbs
    for (uint8_t idx = 4; idx > 0; idx--) {
        if (words[idx] != 0) {
            return false;
        }
    }
    return words[0] <= CuEVM::no_precompile_contracts && words[0] > 0x00;
}

__host__ __device__ int32_t evm_word_t::to_byte_array_t(byte_array_t &byte_array) const {
    byte_array.grow(CuEVM::word_size);
    uint256_to_bytes(byte_array.data, this, CuEVM::word_size);
    return 0;
}

__host__ __device__ int32_t evm_word_t::to_byte_array_t(uint8_t *byte_array, uint32_t &byte_array_length) const {
    byte_array_length = CuEVM::word_size;
    uint256_to_bytes(byte_array, this, byte_array_length);
    return 0;
}
// for ecc, the bit array is flipped
__host__ __device__ int32_t evm_word_t::to_bit_array_t(uint8_t *bit_array, uint32_t &bit_array_length) const {
    uint8_t *bits = bit_array;
    for (int32_t idx = 0; idx < uint256_limbs; idx++) {
        for (int bit = 0; bit < 32; bit++) {
            *(bits++) = (uint8_t)((words[idx] >> bit) & 0x01);
        }
    }
    bit_array_length = CuEVM::word_bits;
    for (int i = CuEVM::word_bits - 1; i >= 0; i--) {
        if (bit_array[i] == 0)
            bit_array_length--;
        else
            break;
    }

    return ERROR_SUCCESS;
}
}  // namespace CuEVM
