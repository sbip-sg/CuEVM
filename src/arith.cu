// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#include "include/arith.cuh"
#include "include/utils.h"

namespace cuEVM {
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

  __host__ __device__ void evm_address_conversion(
    ArithEnv &arith,
    bn_t &address) {
    cgbn_bitwise_mask_and(arith.env, address, address, EVM_ADDRESS_BITS);
  }

  __host__ __device__ size_t ArithEnv::memory_from_cgbn(
    uint8_t *dst,
    bn_t &src) {
    for (uint32_t idx = 0; idx < EVM_WORD_SIZE; idx++)
    {
      dst[idx] = cgbn_extract_bits_ui32(env, src, EVM_WORD_BITS - (idx + 1) * 8, 8);
    }
    return EVM_WORD_SIZE;
  }

  __host__ __device__ void ArithEnv::cgbn_from_memory(
    bn_t &dst,
    uint8_t *src) {
    for (uint32_t idx = 0; idx < EVM_WORD_SIZE; idx++)
    {
      cgbn_insert_bits_ui32(env, dst, dst, EVM_WORD_BITS - (idx + 1) * 8, 8, src[idx]);
    }
  }

  __host__ __device__ void ArithEnv::word_from_memory(
    evm_word_t &dst,
    uint8_t *src
  )
  {
    bn_t src_cgbn;
    cgbn_from_memory(src_cgbn, src);
    cgbn_store(env, &dst, src_cgbn);
  }

  __host__ __device__ void ArithEnv::cgbn_from_fixed_memory(
    bn_t &dst,
    uint8_t *src,
    size_t size
  )
  {
    cgbn_set_ui32(env, dst, 0);
    for (uint8_t idx = (EVM_WORD_SIZE - size); idx < EVM_WORD_SIZE; idx++)
    {
      cgbn_insert_bits_ui32(
          env,
          dst,
          dst,
          idx * 8,
          8,
          src[EVM_WORD_SIZE - 1 - idx]);
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
    bn_t &src) {
    bn_t tmp;
    cgbn_bitwise_mask_and(env, tmp, src, -(EVM_WORD_BITS - 32));
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
    size_t limb_count = CGBN_LIMBS) {
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

  __host__ void ArithEnv::hex_string_from_cgbn_memory(
    char *dst_hex_string,
    evm_word_t &src_cgbn_mem,
    uint32_t count) {
    dst_hex_string[0] = '0';
    dst_hex_string[1] = 'x';
    for (uint32_t idx = 0; idx < count; idx++)
    {
      sprintf(
        dst_hex_string + 2 + idx * 8,
        "%08x",
        src_cgbn_mem._limbs[count - 1 - idx]
      );
    }
    dst_hex_string[count * 8 + 2] = '\0';
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

  __host__ __device__ int32_t ArithEnv::cgbn_memory_from_hex_string(
    evm_word_t &dst_cgbn_memory,
    const char *src_hex_string) {
    size_t length;
    char *current_char;
    current_char = (char *)src_hex_string;
    if (
      (src_hex_string[0] == '0') &&
      ((src_hex_string[1] == 'x') || (src_hex_string[1] == 'X'))
    ) {
      current_char += 2; // Skip the "0x" prefix
    }
    for (length = 0; current_char[length] != '\0'; length++)
      ;
    if (length > (2 * EVM_WORD_SIZE)) {
      return 1;
    }
    SHARED_MEMORY uint8_t *byte_array;
    ONE_THREAD_PER_INSTANCE(
      byte_array = new uint8_t[EVM_WORD_SIZE];
      memset(byte_array, 0, EVM_WORD_SIZE);
    )

    if(length > 0) {
      size_t idx;
      for (idx = length; idx > 2; idx -= 2)
      {
        byte_array[EVM_WORD_SIZE - 1 - ((length - idx) / 2)] = byte_array::byte_from_two_hex_char(
          current_char[idx - 2],
          current_char[idx - 1]
        );
      }
      if (idx == 1)
      {
        byte_array[EVM_WORD_SIZE - 1 - ((length-1) / 2)] = byte_array::byte_from_two_hex_char('0', current_char[0]);
      } else { //idx = 2
        byte_array[EVM_WORD_SIZE - 1 - ((length-2) / 2)] = byte_array::byte_from_two_hex_char(current_char[0], current_char[1]);
      }

    }
    word_from_memory(dst_cgbn_memory, byte_array);
    ONE_THREAD_PER_INSTANCE(
      delete[] byte_array;
    )
    return 0;
  }

  __host__ __device__ void ArithEnv::cgbn_memory_from_size_t(
    evm_word_t &dst_cgbn_memory,
    size_t src) {
    bn_t src_cgbn;
    cgbn_from_size_t(src_cgbn, src);
    cgbn_store(env, &dst_cgbn_memory, src_cgbn);
  }

  __host__ __device__ void ArithEnv::print_cgbn_memory(
    evm_word_t &src_cgbn_memory) {
    for (uint32_t idx = 0; idx < CGBN_LIMBS; idx++)
      printf("%08x ", src_cgbn_memory._limbs[CGBN_LIMBS - 1 - idx]);
    printf("\n");
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
        return NULL;
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