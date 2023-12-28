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
#include <gmp.h>
#ifndef __CGBN_H__
#define __CGBN_H__
#include <cgbn/cgbn.h>
#endif
#include "data_content.h"

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
  /**
   * The CGBN context type.  This is a template type that takes
   * the number of threads per instance and the
   * parameters class as template parameters.
  */
  typedef cgbn_context_t<params::TPI, params> context_t;
  /**
   * The CGBN environment type. This is a template type that takes the
   * context type as a template parameter. It provides the CGBN functions.
  */
  typedef cgbn_env_t<context_t, params::BITS> env_t;
  /**
   * The CGBN base type for the given number of bit in environment.
  */
  typedef typename env_t::cgbn_t bn_t;
  /**
   * The CGBN wide type with double the given number of bits in environment.
  */
  typedef typename env_t::cgbn_wide_t bn_wide_t;
  /**
   * The EVM word type. also use for store CGBN base type.
  */
  typedef cgbn_mem_t<params::BITS> evm_word_t;
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

  /**
   * Get a CGBN memory from a hex string.
   * The hex string is in Big Endian format.
   * It use the GMP library to convert the hex string to a mpz_t type.
   * @param[out] dst_cgbn_memory The destination CGBN memory
   * @param[in] src_hex_string The source hex string
   * @return 1 for overflow, 0 otherwiese
  */
  __host__ int32_t cgbn_memory_from_hex_string(
    evm_word_t &dst_cgbn_memory,
    char *src_hex_string
  )
  {
    mpz_t value;
    size_t written;
    mpz_init(value);
    if (
      (src_hex_string[0] == '0') &&
      ((src_hex_string[1] == 'x') || (src_hex_string[1] == 'X'))
    )
    {
      mpz_set_str(value, src_hex_string + 2, 16);
    }
    else
    {
      mpz_set_str(value, src_hex_string, 16);
    }
    if (mpz_sizeinbase(value, 2) > BITS)
    {
      return 1;
    }
    mpz_export(
      dst_cgbn_memory._limbs,
      &written,
      -1,
      sizeof(uint32_t),
      0,
      0,
      value
    );
    while (written < LIMBS)
    {
      dst_cgbn_memory._limbs[written++] = 0;
    }
    mpz_clear(value);
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



};

#endif