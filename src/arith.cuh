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
};

#endif