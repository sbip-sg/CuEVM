// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#ifndef _ARITH_H_
#define _ARITH_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>

#include "utils.h"

/**
 * The arithmetic environment class is a wrapper around the CGBN library.
 * It provides a context, environment, and instance for the CGBN library.
 * It also provides some utility functions for converting between CGBN and other
 * types.
*/
template <class params>
class arith_env_t
{
public:

  static const uint32_t BITS = params::BITS; /**< The number of the bits for the CGBN */
  static const uint32_t BYTES = params::BITS / 8; /**< The number of the bytes for the CGBN */
  static const uint32_t LIMBS = params::BITS / 32; /**< The number of the limbs for the CGBN */
  static const uint32_t ADDRESS_BYTES = 20; /**< The number of the bytes for the address */

  context_t _context; /**< The CGBN context */
  env_t _env;        /**< The CGBN environment */
  uint32_t _instance; /**< The instance number for the CGBN */

  /**
   * The constructor.  This takes a monitor and error report for the CGBN library,
   * and an instance number. The instance number is used to select the appropriate
   * CGBN instance for the thread.
   * @param[in] monitor The monitor for the CGBN library
   * @param[in] report The error report for the CGBN library
   * @param[in] instance The instance number for the CGBN library
  */
  __device__ __forceinline__ arith_env_t(
    cgbn_monitor_t monitor,
    cgbn_error_report_t *report,
    uint32_t instance
  ) : _context(monitor, report, instance),
      _env(_context),
      _instance(instance)
  {
  }

  /**
   * The constructor.  This takes a monitor for the CGBN library.
   * Used more in unit tests.
   * @param[in] monitor The monitor for the CGBN library
  */
  __device__ __forceinline__ arith_env_t(
    cgbn_monitor_t monitor
  ) : _context(monitor),
      _env(_context),
      _instance(0)
  {
  }

  /**
   * The constructor. This takes a monitor for the CGBN library,
   * and an instance number.
   * @param[in] monitor The monitor for the CGBN library
   * @param[in] instance The instance number for the CGBN library
  */
  __host__ __forceinline__ arith_env_t(
    cgbn_monitor_t monitor,
    uint32_t instance
  ) : _context(monitor),
      _env(_context),
      _instance(instance)
  {
  }

  /**
   * The clone constructor.  This takes an existing arithmetic
   * environment and clones it.
   * @param[in] env The existing arithmetic environment
  */
  __host__ __device__ __forceinline__ arith_env_t(
    const arith_env_t &env
  ) : _context(env._context),
      _env(_context),
      _instance(env._instance)
  {
  }

  __host__ __device__ __forceinline__ void address_conversion(
    bn_t &address
  )
  {
    int32_t address_bits = int32_t(ADDRESS_BYTES) * 8;
    cgbn_bitwise_mask_and(_env, address, address, address_bits);
  }

  /**
   * Allocate a memory byte array with the content of the source byte array
   * and the requested size. The memory byte array is in Big Endian format.
   * NOTE: The memory byte array must be freed by the caller.
   * The memory byte array is padded with zeros if the requested size is greater
   * than the current size.
   * @param[in] src The source byte array
   * @param[in] current_size The current size of the byte array
   * @param[in] request_size The requested size of the byte array
   * @return The memory byte array pointer
  */
  __host__ __device__ __forceinline__ uint8_t* padded_malloc_byte_array(
    const uint8_t *src,
    size_t current_size,
    size_t request_size
  ) {
    SHARED_MEMORY uint8_t *dst;
    ONE_THREAD_PER_INSTANCE(
      dst = new uint8_t[request_size];
      size_t copy_size;
      if (current_size < request_size)
      {
        copy_size = current_size;
      }
      else
      {
        copy_size = request_size;
      }
      if (dst != NULL)
      {
        memcpy(dst, src, copy_size);
        memset(dst + copy_size, 0, request_size - copy_size);
      }
    )
    return dst;
  }


  /**
   * Get a memory byte array from CGBN base type.
   * The memory byte array is in Big Endian format.
   * The memory byte array must be allocated by the caller.
   * @param[out] dst The memory byte array
   * @param[in] src The source CGBN
   * @return The size of the byte array
  */
  __host__ __device__ __forceinline__ size_t memory_from_cgbn(
    uint8_t *dst,
    bn_t &src
  )
  {
    for (uint32_t idx = 0; idx < BYTES; idx++)
    {
      dst[idx] = cgbn_extract_bits_ui32(_env, src, BITS - (idx + 1) * 8, 8);
    }
    return BYTES;
  }

  /**
   * Get a CGBN type from memory byte array.
   * The memory byte array is in Big Endian format.
   * @param[out] dst The destination CGBN
   * @param[in] src The memory byte array
  */
  __host__ __device__ __forceinline__ void cgbn_from_memory(
    bn_t &dst,
    uint8_t *src
  )
  {
    for (uint32_t idx = 0; idx < BYTES; idx++)
    {
      cgbn_insert_bits_ui32(_env, dst, dst, BITS - (idx + 1) * 8, 8, src[idx]);
    }
  }

  /**
   * Get evm_word_t from byte array.
  */
  __host__ __device__ __forceinline__ void word_from_memory(
    evm_word_t &dst,
    uint8_t *src
  )
  {
    bn_t src_cgbn;
    cgbn_from_memory(src_cgbn, src);
    cgbn_store(_env, &dst, src_cgbn);
  }

  /**
   * Get a CGBN type from memory byte array.
   * The memory byte array is in Big Endian format.
   * If the memory byte array is smaller than the CGBN, it fills the
   * remaining bytes with zeros.
   * @param[out] dst The destination CGBN
   * @param[in] src The memory byte array
   * @param[in] size The size of the memory byte array
  */
  __host__ __device__ __forceinline__ void cgbn_from_fixed_memory(
    bn_t &dst,
    uint8_t *src,
    size_t size
  )
  {
    cgbn_set_ui32(_env, dst, 0);
    for (uint8_t idx = (BYTES - size); idx < BYTES; idx++)
    {
      cgbn_insert_bits_ui32(
          _env,
          dst,
          dst,
          idx * 8,
          8,
          src[BYTES - 1 - idx]);
    }
  }

  /**
   * Get a CGBN type from a size_t.
   * @param[out] dst The destination CGBN
   * @param[in] src The size_t
  */
  __host__ __device__ __forceinline__ void cgbn_from_size_t(
    bn_t &dst,
    size_t src
  )
  {
    cgbn_set_ui32(_env, dst, 0);
    cgbn_insert_bits_ui32(_env, dst, dst, 32, 32, (src >> 32));
    cgbn_insert_bits_ui32(_env, dst, dst, 0, 32, src);
  }

  /**
   * Get a size_t from a CGBN type.
   * @param[out] dst The destination size_t
   * @param[in] src The source CGBN
   * @return 1 for overflow, 0 otherwiese
  */
  __host__ __device__ __forceinline__ int32_t size_t_from_cgbn(
    size_t &dst,
    bn_t &src
  )
  {
    bn_t MAX_SIZE_T;
    cgbn_set_ui32(_env, MAX_SIZE_T, 1);
    cgbn_shift_left(_env, MAX_SIZE_T, MAX_SIZE_T, 64);
    dst = 0;
    dst = cgbn_extract_bits_ui32(_env, src, 0, 32);
    dst |= ((size_t)cgbn_extract_bits_ui32(_env, src, 32, 32)) << 32;
    if (cgbn_compare(_env, src, MAX_SIZE_T) >= 0)
    {
      return 1;
    }
    else
    {
      return 0;
    }
  }

  /**
   * Get a uint64_t from a CGBN type.
   * @param[out] dst The destination uint64_t
   * @param[in] src The source CGBN
   * @return 1 for overflow, 0 otherwiese
  */
  __host__ __device__ __forceinline__ int32_t uint64_t_from_cgbn(
    uint64_t &dst,
    bn_t &src
  )
  {
    bn_t MAX_uint64_t;
    cgbn_set_ui32(_env, MAX_uint64_t, 1);
    cgbn_shift_left(_env, MAX_uint64_t, MAX_uint64_t, 64);
    dst = 0;
    dst = cgbn_extract_bits_ui32(_env, src, 0, 32);
    dst |= ((uint64_t)cgbn_extract_bits_ui32(_env, src, 32, 32)) << 32;
    if (cgbn_compare(_env, src, MAX_uint64_t) >= 0)
    {
      return 1;
    }
    else
    {
      return 0;
    }
  }

  /**
   * Get an array of maximum 256 bytes, each having value 1 or 0 indicating bit set or not.
   * Use as utility for Elliptic curve point multiplication
   * @param[out] dst_array
   * @param[out] array_length
   * @param[in] src_cgbn_mem
   * @param limb_count
   */
  __host__ __device__ __forceinline__ void bit_array_from_cgbn_memory(uint8_t *dst_array, uint32_t &array_length, evm_word_t &src_cgbn_mem, uint32_t limb_count = LIMBS) {
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

  /**
   * Get an array of bytes from CGBN memory.
   * Use as utility for address conversion from Public Key point
   * @param[out] dst_array
   * @param[out] array_length
   * @param[in] src_cgbn_mem
   * @param limb_count
   */
  __host__ __device__ __forceinline__ void byte_array_from_cgbn_memory(uint8_t *dst_array, size_t &array_length, evm_word_t &src_cgbn_mem, size_t limb_count = LIMBS) {
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

  /**
   * Print the byte array as hex. Utility for debugging
   *
   * @param byte_array
   * @param array_length
   * @param is_address
   */
  __host__ __device__ __forceinline__ void print_byte_array_as_hex(const uint8_t *byte_array, uint32_t array_length, bool is_address=false) {
      printf("0x");
      for (uint32_t i = is_address? 12: 0; i < array_length; i++) {
          printf("%02x", byte_array[i]);
      }
      printf("\n");
  }

    /**
   * Get a uint64_t from a CGBN type.
   * @param[dst] dst The destination to store the trimmed number
   * @param[src] src The number to trim down to uint64
  */
  __host__ __device__ __forceinline__ void trim_to_uint64(bn_t &dst, bn_t &src)
  {
    const uint32_t numbits = 256 - 64;
    cgbn_shift_left(_env, dst, src, numbits);
    cgbn_shift_right(_env, dst, src, numbits);
  }

  /**
   * Get a hex string from the CGBn memory.
   * The hex string is in Big Endian format.
   * The hex string must be allocated by the caller.
   * @param[out] dst_hex_string The destination hex string
   * @param[in] src_cgbn_mem The source CGBN memory
   * @param[in] count The number of limbs
  */
  __host__ void hex_string_from_cgbn_memory(
    char *dst_hex_string,
    evm_word_t &src_cgbn_mem,
    uint32_t count = LIMBS
  )
  {
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
  /*
    * Get a pretty hex string from the CGBn memory.
    * The hex string is in Big Endian format.
    * The hex string must be allocated by the caller.
    * @param[out] dst_hex_string The destination hex string
    * @param[in] src_cgbn_mem The source CGBN memory
    * @param[in] count The number of limbs
  */
  __host__ void pretty_hex_string_from_cgbn_memory(
    char *dst_hex_string,
    evm_word_t &src_cgbn_mem,
    uint32_t count = LIMBS
  )
  {
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

  __host__ __device__ __forceinline__ uint8_t byte_from_two_hex(
    char high,
    char low
  ) {
    uint8_t byte = 0;
    if (high >= '0' && high <= '9')
    {
      byte = (high - '0') << 4;
    }
    else if (high >= 'a' && high <= 'f')
    {
      byte = (high - 'a' + 10) << 4;
    }
    else if (high >= 'A' && high <= 'F')
    {
      byte = (high - 'A' + 10) << 4;
    }
    if (low >= '0' && low <= '9')
    {
      byte |= (low - '0');
    }
    else if (low >= 'a' && low <= 'f')
    {
      byte |= (low - 'a' + 10);
    }
    else if (low >= 'A' && low <= 'F')
    {
      byte |= (low - 'A' + 10);
    }
    return byte;
  }

  /**
   * Get a CGBN memory from a hex string.
   * The hex string is in Big Endian format.
   * @param[out] dst_cgbn_memory The destination CGBN memory
   * @param[in] src_hex_string The source hex string
   * @return 1 for overflow, 0 otherwiese
  */
  __host__ __device__ int32_t cgbn_memory_from_hex_string(
    evm_word_t &dst_cgbn_memory,
    const char *src_hex_string
  ) {
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
    if (length > (2 * BYTES)) {
      return 1;
    }
    SHARED_MEMORY uint8_t *byte_array;
    ONE_THREAD_PER_INSTANCE(
      byte_array = new uint8_t[BYTES];
      memset(byte_array, 0, BYTES);
    )

    size_t idx;
    for (idx = length; idx > 2; idx -= 2)
    {
      byte_array[BYTES - 1 - ((length - idx) / 2)] = byte_from_two_hex(
        current_char[idx - 2],
        current_char[idx - 1]
      );
    }
    if (idx == 1)
    {
      byte_array[BYTES - 1 - ((length-1) / 2)] = byte_from_two_hex('0', current_char[0]);
    } else { //idx = 2
      byte_array[BYTES - 1 - ((length-2) / 2)] = byte_from_two_hex(current_char[0], current_char[1]);
    }
    /*bn_t tmp;
    cgbn_from_memory(tmp, byte_array);
    printf("cgbn_memory_from_hex_string: ");
    for (uint32_t i = 0; i < BYTES; i++)
    {
      printf("%02x ", byte_array[i]);
    }
    printf("\n");
    cgbn_store(_env, &dst_cgbn_memory, tmp);
    print_cgbn_memory(dst_cgbn_memory);
    */
    word_from_memory(dst_cgbn_memory, byte_array);
    ONE_THREAD_PER_INSTANCE(
      delete[] byte_array;
    )
    return 0;
  }

  /**
   * Set the cgbn memory to the size_t value
   * @param[out] dst_cgbn_memory The destination CGBN memory
   * @param[in] src The source size_t
  */
  __host__ __device__ __forceinline__ void cgbn_memory_from_size_t(
    evm_word_t &dst_cgbn_memory,
    size_t src
  )
  {
    bn_t src_cgbn;
    cgbn_from_size_t(src_cgbn, src);
    cgbn_store(_env, &dst_cgbn_memory, src_cgbn);
  }

  /**
   * Print the CGBN memory in hex string format.
   * The hex string is in Big Endian format.
   * @param[in] src_cgbn_memory The source CGBN memory
  */
  __host__ __device__ __forceinline__ void print_cgbn_memory(
    evm_word_t &src_cgbn_memory
  )
  {
    for (uint32_t idx = 0; idx < LIMBS; idx++)
      printf("%08x ", src_cgbn_memory._limbs[LIMBS - 1 - idx]);
    printf("\n");
  }

  /**
   * Verify if is enough gas for the operation.
   * @param[in] gas_limit The gas limit
   * @param[in] gas_used The gas used
   * @param[inout] error_code The error code
   * @return 1 for enough gas and no previous errors, 0 otherwise
  */
  __host__ __device__ __forceinline__ int32_t has_gas(
    bn_t &gas_limit,
    bn_t &gas_used,
    uint32_t &error_code
  )
  {
    int32_t gas_sign = cgbn_compare(_env, gas_limit, gas_used);
    error_code = (gas_sign < 0) ? ERROR_GAS_LIMIT_EXCEEDED : error_code;
    return (gas_sign >= 0) && (error_code == ERR_NONE);
  }

  /**
   * Compute the max gas call.
   * @param[out] gas_capped The gas capped
   * @param[in] gas_limit The gas limit
   * @param[in] gas_used The gas used
  */
  __host__ __device__ __forceinline__ void max_gas_call(
    bn_t &gas_capped,
    bn_t &gas_limit,
    bn_t &gas_used
  )
  {
      // compute the remaining gas
      bn_t gas_left;
      cgbn_sub(_env, gas_left, gas_limit, gas_used);

      // gas capped = (63/64) * gas_left
      cgbn_div_ui32(_env, gas_capped, gas_left, 64);
      cgbn_sub(_env, gas_capped, gas_left, gas_capped);
  }

  /**
   * Get the data at the given index for the given length.
   * If the index is greater than the data size, it returns NULL.
   * If the length is greater than the data size - index, it returns
   * the data from index to the end of the data and sets the
   * available size to the data size - index. Otherwise, it returns
   * the data from index to index + length and sets the available
   * size to length.
   * @param[in] data_content The data content
   * @param[in] index The index of the code data
   * @param[in] length The length of the code data
   * @param[out] available_size The available size of the code data
  */
  __host__ __device__ __forceinline__ uint8_t *get_data(
    data_content_t &data_content,
    bn_t &index,
    bn_t &length,
    size_t &available_size
  )
  {
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

  /**
   * Add the gas cost for the given length of bytes, but considering
   * evm words.
   * @param[inout] gas_used The gas used
   * @param[in] length The length of the bytes
   * @param[in] gas_per_word The gas per evm word
  */
  __host__ __device__ __forceinline__ void evm_words_gas_cost
  (
    bn_t &gas_used,
    bn_t &length,
    uint32_t gas_per_word
  )
  {
    // gas_used += gas_per_word * emv word count of length
    // length = (length + 31) / 32
    bn_t evm_words_gas;
    cgbn_add_ui32(_env, evm_words_gas, length, BYTES -1);
    cgbn_div_ui32(_env, evm_words_gas, evm_words_gas, BYTES);
    cgbn_mul_ui32(_env, evm_words_gas, evm_words_gas, gas_per_word);
    cgbn_add(_env, gas_used, gas_used, evm_words_gas);
  }

  /**
   * Add the cost for initiliasation code.
   * EIP-3860: https://eips.ethereum.org/EIPS/eip-3860
   * @param[inout] gas_used The gas used
   * @param[in] initcode_length The length of the initcode
  */
  __host__ __device__ __forceinline__ void initcode_cost(
    bn_t &gas_used,
    bn_t &initcode_length
  )
  {
    // gas_used += GAS_INITCODE_WORD_COST * emv word count of initcode
    // length = (initcode_length + 31) / 32
    evm_words_gas_cost(gas_used, initcode_length, GAS_INITCODE_WORD_COST);
  }

  /**
   * Add the cost for keccak hashing.
   * @param[inout] gas_used The gas used
   * @param[in] length The length of the data in bytes
  */
  __host__ __device__ __forceinline__ void keccak_cost(
    bn_t &gas_used,
    bn_t &length
  )
  {
    evm_words_gas_cost(gas_used, length, GAS_KECCAK256_WORD);
  }


  /**
   * Add the cost for memory operation on call data/return data.
   * @param[inout] gas_used The gas used
   * @param[in] length The length of the data in bytes
  */
  __host__ __device__ __forceinline__ void memory_cost(
    bn_t &gas_used,
    bn_t &length
  )
  {
    evm_words_gas_cost(gas_used, length, GAS_MEMORY);
  }


  /**
   * Add the cost for sha256 hashing.
   * @param[inout] gas_used The gas used
   * @param[in] length The length of the data in bytes
  */
  __host__ __device__ __forceinline__ void sha256_cost(
    bn_t &gas_used,
    bn_t &length
  )
  {
    evm_words_gas_cost(gas_used, length, GAS_PRECOMPILE_SHA256_WORD);
  }
  /**
   * Add the dynamic cost for ripemd160 hashing.
   * @param[inout] gas_used The gas used
   * @param[in] length The length of the data in bytes
  */
  __host__ __device__ __forceinline__ void ripemd160_cost(
    bn_t &gas_used,
    bn_t &length
  )
  {
    evm_words_gas_cost(gas_used, length, GAS_PRECOMPILE_RIPEMD160_WORD);
  }

  /**
   * Add the dynamics cost for blake2 hashing.
   * @param[inout] gas_used The gas used
   * @param[in] rounds Number of rounds (big-endian unsigned integer)
   */
  __host__ __device__ __forceinline__ void blake2_cost(bn_t &gas_used, uint32_t rounds) {
      // gas_used += GAS_PRECOMPILE_BLAKE2_ROUND * rounds
      bn_t temp;
      cgbn_set_ui32(_env, temp, rounds);
      cgbn_mul_ui32(_env, temp, temp, GAS_PRECOMPILE_BLAKE2_ROUND);
      cgbn_add(_env, gas_used, gas_used, temp);
  }

  /**
   * Add the pairing cost to the gas used.
   * @param[inout] gas_used The gas used
   * @param[in] data_size The size of the data in bytes
  */
  __host__ __device__ __forceinline__ void ecpairing_cost(
    bn_t &gas_used,
    size_t data_size
  ) {
      // gas_used += GAS_PRECOMPILE_ECPAIRING + data_size/192 * GAS_PRECOMPILE_ECPAIRING_PAIR
      cgbn_add_ui32(_env, gas_used, gas_used, GAS_PRECOMPILE_ECPAIRING);
      bn_t temp;
      cgbn_from_size_t(temp, data_size);
      cgbn_div_ui32(_env, temp, temp, 192);
      cgbn_mul_ui32(_env, temp, temp, GAS_PRECOMPILE_ECPAIRING_PAIR);
      cgbn_add(_env, gas_used, gas_used, temp);
  }

};

/**
 * The arithmetical environment used by the arbitrary length
 * integer library.
 */
typedef arith_env_t<evm_params> arith_t;

#endif
