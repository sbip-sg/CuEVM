// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#include "../include/utils/arith.cuh"
#include "../include/utils/evm_utils.cuh"

namespace cuEVM {
  __host__ __device__ evm_word_t::evm_word_t(
    const evm_word_t &src)
  {
      /*
    #pragma unroll
    for (int32_t index = 0; index < cuEVM::cgbn_limbs; index++) {
      _limbs[index] = src._limbs[index];
    }
    return *this;*/
    std::copy(src._limbs, src._limbs + cuEVM::cgbn_limbs, _limbs);
  }

  __host__ __device__ evm_word_t& evm_word_t::operator=(
    const evm_word_t &src) {
      /*
    #pragma unroll
    for (int32_t index = 0; index < cuEVM::cgbn_limbs; index++) {
      _limbs[index] = src._limbs[index];
    }
    return *this;*/
    std::copy(src._limbs, src._limbs + cuEVM::cgbn_limbs, _limbs);
  }

  __host__ __device__ int32_t evm_word_t::operator==(
    const evm_word_t &other) const {
    #pragma unroll
    for (int32_t index = 0; index < cuEVM::cgbn_limbs; index++) {
      if (_limbs[index] != other._limbs[index]) {
        return 0;
      }
    }
    return 1;
  }

  
  __host__ __device__ int32_t evm_word_t::operator==(
    const uint32_t &value) const {
    #pragma unroll
    if (_limbs[0] != value) {
      return 0;
    }
    for (int32_t index = 1; index < cuEVM::cgbn_limbs; index++) {
      if (_limbs[index] != 0) {
        return 0;
      }
    }
    return 1;
  }

  __host__ __device__ int32_t evm_word_t::from_hex(
    const char *hex_string)
  {
    cuEVM::byte_array_t byte_array(
      hex_string,
      cuEVM::word_size,
      BIG_ENDIAN,
      cuEVM::PaddingDirection::LEFT_PADDING
    );
    return from_byte_array_t(byte_array);
  }

  __host__ __device__ int32_t evm_word_t::from_byte_array_t(
    byte_array_t &byte_array)
  {
    uint8_t *bytes = byte_array.data;
    if (bytes == nullptr)
    {
      return 1;
    }
    #pragma unroll
    for (uint32_t idx = 0; idx < cuEVM::cgbn_limbs; idx++)
    {
      _limbs[idx] = (
        *(bytes++) |
        *(bytes++) << 8 |
        *(bytes++) << 16 |
        *(bytes++) << 24
      );
    }
    return 0;
  }

  __host__ __device__ int32_t evm_word_t::from_size_t(
    size_t value)
  {
    if (sizeof(size_t) == sizeof(uint64_t))
    {
      return from_uint64_t(value);
    }
    else if (sizeof(size_t) == sizeof(uint32_t))
    {
      return from_uint32_t(value);
    }
    else
    {
      return 1;
    }
  }

  __host__ __device__ int32_t evm_word_t::from_uint64_t(
    uint64_t value)
  {
    #pragma unroll
    for (uint32_t idx = 2; idx < cuEVM::cgbn_limbs; idx++)
    {
      _limbs[idx] = 0;
    }
    _limbs[0] = value & 0xFFFFFFFF;
    _limbs[1] = (value >> 32) & 0xFFFFFFFF;
  }

  __host__ __device__ int32_t evm_word_t::from_uint32_t(
    uint32_t value)
  {
    #pragma unroll
    for (uint32_t idx = 1; idx < cuEVM::cgbn_limbs; idx++)
    {
      _limbs[idx] = 0;
    }
    _limbs[0] = value;
  }

  __host__ __device__ void evm_word_t::print() const
  {
    for (uint32_t idx = 0; idx < cuEVM::cgbn_limbs; idx++)
      printf("%08x ", _limbs[cuEVM::cgbn_limbs - 1 - idx]);
    printf("\n");
  }
  
  __host__ __device__ char* evm_word_t::to_hex(
    char *hex_string,
    int32_t pretty,
    uint32_t count) const
  {
    if (hex_string == nullptr)
    {
      hex_string = new char[count * 8 + 3];
    }
    hex_string[0] = '0';
    hex_string[1] = 'x';
    for (uint32_t idx = 0; idx < count; idx++)
    {
      sprintf(
        hex_string + 2 + idx * 8,
        "%08x",
        _limbs[count - 1 - idx]
      );
    }
    hex_string[count * 8 + 2] = '\0';
    if (pretty)
    {
      cuEVM::utils::hex_string_without_leading_zeros(hex_string);
    }
    return hex_string;
  }

  __host__ __device__ byte_array_t* evm_word_t::to_byte_array_t(
    byte_array_t *byte_array) const
  {
    if (byte_array == nullptr)
    {
      byte_array = new byte_array_t(cuEVM::word_size);
    }
    uint8_t *bytes = byte_array->data + cuEVM::word_size - 1;
    for (uint32_t idx = 0; idx < cuEVM::cgbn_limbs; idx++)
    {
      *(bytes--) = _limbs[idx] & 0xFF;
      *(bytes--) = (_limbs[idx] >> 8) & 0xFF;
      *(bytes--) = (_limbs[idx] >> 16) & 0xFF;
      *(bytes--) = (_limbs[idx] >> 24) & 0xFF;
    }
    return byte_array;
  }

  __host__ __device__ byte_array_t* evm_word_t::to_bit_array_t(
    byte_array_t *bit_array) const
  {
    if (bit_array == nullptr)
    {
      bit_array = new byte_array_t(cuEVM::word_bits);
    }
    uint8_t *bytes = bit_array->data + cuEVM::word_bits - 1;
    for (uint32_t idx = 0; idx < cuEVM::cgbn_limbs; idx++)
    {
      for (int bit = 0; bit < 32; bit++)
      {
        *(bytes--) = (_limbs[idx] >> bit) & 0x01;
      }
    }
    return bit_array;
  }


  __device__ ArithEnv::ArithEnv(
    cgbn_monitor_t monitor,
    cgbn_error_report_t *report,
    uint32_t instance
  ) : _context(monitor, report, instance),
      env(_context),
      instance(instance)
  {
  }

  __device__ ArithEnv::ArithEnv(
    cgbn_monitor_t monitor
  ) : _context(monitor),
      env(_context),
      instance(0)
  {
  }

  __host__ ArithEnv::ArithEnv(
    cgbn_monitor_t monitor,
    uint32_t instance
  ) : _context(monitor),
      env(_context),
      instance(instance)
  {
  }

  __host__ __device__ ArithEnv::ArithEnv(
    const ArithEnv &env
  ) : _context(env._context),
      env(_context),
      instance(env.instance)
  {
  }

  __host__ __device__ void ArithEnv::cgbn_from_fixed_memory(
    bn_t &dst,
    uint8_t *src,
    size_t size
  )
  {
    cgbn_set_ui32(env, dst, 0);
    for (uint8_t idx = (cuEVM::word_size - size); idx < cuEVM::word_size; idx++)
    {
      cgbn_insert_bits_ui32(
          env,
          dst,
          dst,
          idx * 8,
          8,
          src[cuEVM::word_size - 1 - idx]);
    }
  }

  __host__ __device__ void ArithEnv::cgbn_from_size_t(
    bn_t &dst,
    size_t src) {
    cgbn_set_ui32(env, dst, 0);
    cgbn_insert_bits_ui32(env, dst, dst, 32, 32, (src >> 32));
    cgbn_insert_bits_ui32(env, dst, dst, 0, 32, src);
  }

  __host__ __device__ int32_t ArithEnv::size_t_from_cgbn(
    size_t &dst,
    bn_t &src) {
    bn_t MAX_SIZE_T;
    cgbn_set_ui32(env, MAX_SIZE_T, 1);
    cgbn_shift_left(env, MAX_SIZE_T, MAX_SIZE_T, 64);
    dst = 0;
    dst = cgbn_extract_bits_ui32(env, src, 0, 32);
    dst |= ((size_t)cgbn_extract_bits_ui32(env, src, 32, 32)) << 32;
    if (cgbn_compare(env, src, MAX_SIZE_T) >= 0)
    {
      return 1;
    }
    else
    {
      return 0;
    }
  }

  __host__ __device__ int32_t ArithEnv::uint64_t_from_cgbn(
    uint64_t &dst,
    bn_t &src) {
    bn_t MAX_uint64_t;
    cgbn_set_ui32(env, MAX_uint64_t, 1);
    cgbn_shift_left(env, MAX_uint64_t, MAX_uint64_t, 64);
    dst = 0;
    dst = cgbn_extract_bits_ui32(env, src, 0, 32);
    dst |= ((uint64_t)cgbn_extract_bits_ui32(env, src, 32, 32)) << 32;
    if (cgbn_compare(env, src, MAX_uint64_t) >= 0)
    {
      return 1;
    }
    else
    {
      return 0;
    }
  }

  __host__ __device__ int32_t ArithEnv::uint32_t_from_cgbn(
    uint32_t &dst,
    const bn_t &src) {
    bn_t tmp;
    cgbn_bitwise_mask_and(env, tmp, src, -(cuEVM::word_bits - 32));
    dst = cgbn_get_ui32(env, src);
    return cgbn_compare_ui32(env, tmp, 0);
  }

  __host__ __device__ void ArithEnv::bit_array_from_cgbn_memory(
    uint8_t *dst_array,
    uint32_t &array_length,
    evm_word_t &src_cgbn_mem,
    uint32_t limb_count) {
    uint32_t current_limb;
    uint32_t bitIndex = 0; // Index for each bit in dst_array
    array_length = 0;
    for (uint32_t idx = 0; idx < limb_count; idx++) {
        current_limb = src_cgbn_mem._limbs[limb_count - 1 - idx];
        for (int bit = 31; bit >=0; --bit) { //hardcoded 32 bits per limb
            // Extract each bit from the current limb and store '0' or '1' in dst_array
            dst_array[bitIndex++] = (current_limb & (1U << bit)) ? 1 : 0;
            if (dst_array[bitIndex-1] == 1 && array_length ==0){
              array_length = 256 - (bitIndex - 1);
            }
        }
      }
  }

  __host__ __device__ void ArithEnv::byte_array_from_cgbn_memory(
    uint8_t *dst_array,
    size_t &array_length,
    evm_word_t &src_cgbn_mem,
    size_t limb_count = cuEVM::cgbn_limbs) {
    size_t current_limb;
    array_length = limb_count * 4; // Each limb has 4 bytes

    for (size_t idx = 0; idx < limb_count; idx++) {
        current_limb = src_cgbn_mem._limbs[limb_count - 1 - idx];
        dst_array[idx * 4] = (current_limb >> 24) & 0xFF; // Extract the most significant byte
        dst_array[idx * 4 + 1] = (current_limb >> 16) & 0xFF;
        dst_array[idx * 4 + 2] = (current_limb >> 8) & 0xFF;
        dst_array[idx * 4 + 3] = current_limb & 0xFF; // Extract the least significant byte
    }
  }

  __host__ __device__ void ArithEnv::print_byte_array_as_hex(
    const uint8_t *byte_array,
    uint32_t array_length,
    bool is_address) {
      printf("0x");
      for (uint32_t i = is_address? 12: 0; i < array_length; i++) {
          printf("%02x", byte_array[i]);
      }
      printf("\n");
  }

  __host__ __device__ void ArithEnv::trim_to_uint64(
    bn_t &dst,
    bn_t &src) {
    const uint32_t numbits = 256 - 64;
    cgbn_shift_left(env, dst, src, numbits);
    cgbn_shift_right(env, dst, src, numbits);
  }

  __host__ void ArithEnv::pretty_hex_string_from_cgbn_memory(
    char *dst_hex_string,
    evm_word_t &src_cgbn_mem,
    uint32_t count) {
    dst_hex_string[0] = '0';
    dst_hex_string[1] = 'x';
    int offset = 2; // Start after "0x"

    for (uint32_t idx = 0, first = 1; idx < count; ++idx)
    {
      uint32_t value = src_cgbn_mem._limbs[count - 1 - idx];
      if (value != 0 || !first)
      {
        if (first)
        {
          first = 0; // No longer at the first non-zero value
          offset += sprintf(dst_hex_string + offset, "%x", value);
        }
        else
        {
          offset += sprintf(dst_hex_string + offset, "%08x", value);
        }
      }
    }

    if (offset == 2) // Handle the case where all values are zero
    {
      strcpy(dst_hex_string + offset, "0");
      offset += 1;
    }

    dst_hex_string[offset] = '\0'; // Null-terminate the string
  }

  __host__ __device__ uint8_t* ArithEnv::get_data(
    cuEVM::byte_array_t &data_content,
    bn_t &index,
    bn_t &length,
    size_t &available_size) {
    available_size = 0;
    size_t index_s;
    int32_t overflow = size_t_from_cgbn(index_s, index);
    if (
        (overflow != 0) ||
        (index_s >= data_content.size))
    {
        return nullptr;
    }
    else
    {
        size_t length_s;
        overflow = size_t_from_cgbn(length_s, length);
        if (
            (overflow != 0) ||
            (length_s > data_content.size - index_s))
        {
            available_size = data_content.size - index_s;
            return data_content.data + index_s;
        }
        else
        {
            available_size = length_s;
            return data_content.data + index_s;
        }
    }
  }
}