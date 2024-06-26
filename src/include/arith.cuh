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
#include <CGBN/cgbn.h>
#include "evm_defines.h"
#include "byte_array.cuh"


namespace cuEVM {
#define CGBN_LIMBS ((EVM_WORD_BITS + 31) / 32)
/**
 * The CGBN context type.  This is a template type that takes
 * the number of threads per instance and the
 * parameters class as template parameters.
*/
using context_t = cgbn_context_t<CGBN_TPI, cgbn_default_parameters_t>;

/**
 * The CGBN environment type. This is a template type that takes the
 * context type as a template parameter. It provides the CGBN functions.
*/
using env_t = cgbn_env_t<context_t, EVM_WORD_BITS>;

/**
 * The CGBN base type for the given number of bit in environment.
*/
using bn_t = env_t::cgbn_t;
/**
 * The CGBN wide type with double the given number of bits in environment.
*/
using bn_wide_t = env_t::cgbn_wide_t;
//using evm_word_t =  cgbn_mem_t<EVM_WORD_BITS>;
/**
 * The EVM word type. also use for store CGBN base type.
*/
struct evm_word_t : cgbn_mem_t<EVM_WORD_BITS> {
  /**
   * The default constructor.
   */
  __host__ __device__ evm_word_t() = default;
  /**
   * The copy constructor.
   */
  __host__ __device__ evm_word_t(
    const evm_word_t &src);

  /**
   * Set the evm_word_t from a hex string.
   * The hex string is in Big Endian format.
   * @param[in] hex_string The source hex string
   * @return 1 for error, 0 otherwiese
   */
  __host__ __device__ int32_t from_hex(
    const char *hex_string);
  /**
   * Set the evm_word_t from a byte array.
   * The byte array is in Big Endian format.
   * @param[in] byte_array The source byte array
   * @return 1 for error, 0 otherwiese
   */
  __host__ __device__ int32_t from_byte_array_t(
    byte_array_t &byte_array);
  /**
   * Set the evm_word_t from a size_t.
   * @param[in] value The source size_t
   * @return 1 for error, 0 otherwiese
   */
  __host__ __device__ int32_t from_size_t(
    size_t value);
  /**
   * Set the evm_word_t from a uint64_t.
   * @param[in] value The source uint64_t
   * @return 1 for error, 0 otherwiese
   */
  __host__ __device__ int32_t from_uint64_t(
    uint64_t value);
  /**
   * Set the evm_word_t from a uint32_t.
   * @param[in] value The source uint32_t
   * @return 1 for error, 0 otherwiese
   */
  __host__ __device__ int32_t from_uint32_t(
    uint32_t value);

  /**
   * Print the evm_word_t.
   */
  __host__ __device__ void print();
  /**
   * Get the hex string from the evm_word_t.
   * The hex string is in Big Endian format.
   * The hex string must be freed by the caller.
   * @param[in] count The number of the least significant
   * limbs to convert
   * @return The hex string
   */
  __host__ __device__ char* to_hex(
    uint32_t count = CGBN_LIMBS);
  /**
   * Get the pretty hex string from the evm_word_t.
   * The hex string is in Big Endian format.
   * The hex string must be freed by the caller.
   * The hex string has the leading zeroes removed.
   * @return The hex string
   */
  __host__ __device__ char* to_pretty_hex();
  /**
   * Get the byte array from the evm_word_t.
   * The byte array is in Big Endian format.
   * The byte array must be freed by the caller.
   * @return byte_array The destination byte array
   */
  __host__ __device__ byte_array_t* to_byte_array_t();
  /**
   * Get the bit array from the evm_word_t.
   * The bit array is in Big Endian format.
   * The bit array must be freed by the caller.
   * @param[out] bit_array The destination bit array
   */
  __host__ __device__ byte_array_t* to_bit_array_t();
};


/**
 * The arithmetic environment class is a wrapper around the CGBN library.
 * It provides a context, environment, and instance for the CGBN library.
 * It also provides some utility functions for converting between CGBN and other
 * types.
*/
class ArithEnv
{

private:
  context_t _context; /**< The CGBN context */

public:
  env_t env;        /**< The CGBN environment */
  uint32_t instance; /**< The instance number for the CGBN */

  /**
   * The constructor.  This takes a monitor and error report for the CGBN library,
   * and an instance number. The instance number is used to select the appropriate
   * CGBN instance for the thread.
   * @param[in] monitor The monitor for the CGBN library
   * @param[in] report The error report for the CGBN library
   * @param[in] instance The instance number for the CGBN library
  */
  __device__ ArithEnv(
    cgbn_monitor_t monitor,
    cgbn_error_report_t *report,
    uint32_t instance);

  /**
   * The constructor.  This takes a monitor for the CGBN library.
   * Used more in unit tests.
   * @param[in] monitor The monitor for the CGBN library
  */
  __device__ ArithEnv(
    cgbn_monitor_t monitor
  );

  /**
   * The constructor. This takes a monitor for the CGBN library,
   * and an instance number.
   * @param[in] monitor The monitor for the CGBN library
   * @param[in] instance The instance number for the CGBN library
  */
  __host__ ArithEnv(
    cgbn_monitor_t monitor,
    uint32_t instance
  );

  /**
   * The clone constructor.  This takes an existing arithmetic
   * environment and clones it.
   * @param[in] env The existing arithmetic environment
  */
  __host__ __device__ ArithEnv(
    const ArithEnv &env
  );

  /**
   * Get a CGBN type from memory byte array.
   * The memory byte array is in Big Endian format.
   * If the memory byte array is smaller than the CGBN, it fills the
   * remaining bytes with zeros.
   * @param[out] dst The destination CGBN
   * @param[in] src The memory byte array
   * @param[in] size The size of the memory byte array
  */
  __host__ __device__ void cgbn_from_fixed_memory(
    bn_t &dst,
    uint8_t *src,
    size_t size
  );

  /**
   * Get a CGBN type from a size_t.
   * @param[out] dst The destination CGBN
   * @param[in] src The size_t
  */
  __host__ __device__ void cgbn_from_size_t(
    bn_t &dst,
    size_t src
  );
  /**
   * Get a size_t from a CGBN type.
   * @param[out] dst The destination size_t
   * @param[in] src The source CGBN
   * @return 1 for overflow, 0 otherwiese
  */
  __host__ __device__ int32_t size_t_from_cgbn(
    size_t &dst,
    bn_t &src
  );
  /**
   * Get a uint64_t from a CGBN type.
   * @param[out] dst The destination uint64_t
   * @param[in] src The source CGBN
   * @return 1 for overflow, 0 otherwiese
  */
  __host__ __device__ int32_t uint64_t_from_cgbn(
    uint64_t &dst,
    bn_t &src
  );

  __host__ __device__ int32_t uint32_t_from_cgbn(
    uint32_t &dst,
    bn_t &src);

  /**
   * Get an array of maximum 256 bytes, each having value 1 or 0 indicating bit set or not.
   * Use as utility for Elliptic curve point multiplication
   * @param[out] dst_array
   * @param[out] array_length
   * @param[in] src_cgbn_mem
   * @param limb_count
   */
  __host__ __device__ void bit_array_from_cgbn_memory(
    uint8_t *dst_array,
    uint32_t &array_length,
    evm_word_t &src_cgbn_mem,
    uint32_t limb_count = CGBN_LIMBS);

  /**
   * Get an array of bytes from CGBN memory.
   * Use as utility for address conversion from Public Key point
   * @param[out] dst_array
   * @param[out] array_length
   * @param[in] src_cgbn_mem
   * @param limb_count
   */
  __host__ __device__ void byte_array_from_cgbn_memory(
    uint8_t *dst_array,
    size_t &array_length,
    evm_word_t &src_cgbn_mem,
    size_t limb_count = CGBN_LIMBS);

  /**
   * Print the byte array as hex. Utility for debugging
   *
   * @param byte_array
   * @param array_length
   * @param is_address
   */
  __host__ __device__ void print_byte_array_as_hex(
    const uint8_t *byte_array,
    uint32_t array_length,
    bool is_address=false);

    /**
   * Get a uint64_t from a CGBN type.
   * @param[dst] dst The destination to store the trimmed number
   * @param[src] src The number to trim down to uint64
  */
  __host__ __device__ void trim_to_uint64(bn_t &dst, bn_t &src);
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
  __host__ __device__ uint8_t *get_data(
    byte_array_t &data_content,
    bn_t &index,
    bn_t &length,
    size_t &available_size);
  // temporary need to remove
  __host__ void pretty_hex_string_from_cgbn_memory(
    char *dst_hex_string,
    evm_word_t &src_cgbn_mem,
    uint32_t count);
};


/**
 * Convert and evm word to and address format
 * @param[in] arith The arithmetic envorinment
 * @param[inout] address The address variable
*/
__host__ __device__ void evm_address_conversion(
  ArithEnv &arith,
  bn_t &address
);
/**
 * Convert and evm word to and address format
 * @param[out] dst the destination evm word
 * @param[in] src the source hex string for
 * @return 1 for error, 0 otherwiese
 */
__host__ __device__ int32_t evm_word_t_from_hex_string(
  evm_word_t &dst,
  const char *src_hex_string);
/**
 * Get the hex string from an evm word.
 * The hex string is already allocated. the result is in big endian.
 * @param[out] hex_string The hex string
 * @param[in] evm_word The evm word
 * @param[in] count The number of limbs
 * @return 1 for error, 0 otherwiese
 */
__host__ void hex_string_from_evm_word_t(
  char *hex_string,
  evm_word_t &evm_word,
  uint32_t count = CGBN_LIMBS);
  /**
   * Print the evm word in hex string format.
   * The hex string is in Big Endian format.
   * @param[in] evm_word The source evm word
  */
  __host__ __device__ void print_evm_word_t(
    evm_word_t &evm_word);
}

#endif
