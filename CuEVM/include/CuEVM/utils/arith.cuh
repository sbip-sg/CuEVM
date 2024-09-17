// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation
// Programme) Author: Stefan-Dan Ciocirlan Data: 2023-11-30
// SPDX-License-Identifier: MIT

#pragma once

#include <CGBN/cgbn.h>
#include <cuda.h>
#include <stdint.h>

#include <CuEVM/core/byte_array.cuh>
#include <CuEVM/core/evm_word.cuh>
#include <CuEVM/utils/evm_defines.cuh>

namespace CuEVM {
/**
 * The arithmetic environment class is a wrapper around the CGBN library.
 * It provides a context, environment, and instance for the CGBN library.
 * It also provides some utility functions for converting between CGBN and other
 * types.
 */
class ArithEnv {
   private:
    context_t _context; /**< The CGBN context */

   public:
    env_t env;         /**< The CGBN environment */
    uint32_t instance; /**< The instance number for the CGBN */

    /**
     * The constructor.  This takes a monitor and error report for the CGBN
     * library, and an instance number. The instance number is used to select
     * the appropriate CGBN instance for the thread.
     * @param[in] monitor The monitor for the CGBN library
     * @param[in] report The error report for the CGBN library
     * @param[in] instance The instance number for the CGBN library
     */
    __device__ ArithEnv(cgbn_monitor_t monitor, cgbn_error_report_t *report,
                        uint32_t instance);

    /**
     * The constructor.  This takes a monitor for the CGBN library.
     * Used more in unit tests.
     * @param[in] monitor The monitor for the CGBN library
     */
    __device__ ArithEnv(cgbn_monitor_t monitor);

    /**
     * The constructor. This takes a monitor for the CGBN library,
     * and an instance number.
     * @param[in] monitor The monitor for the CGBN library
     * @param[in] instance The instance number for the CGBN library
     */
    __host__ ArithEnv(cgbn_monitor_t monitor, uint32_t instance);

    /**
     * The clone constructor.  This takes an existing arithmetic
     * environment and clones it.
     * @param[in] env The existing arithmetic environment
     */
    __host__ __device__ ArithEnv(const ArithEnv &env);
};
/**
 * Get a CGBN type from memory byte array.
 * The memory byte array is in Big Endian format.
 * If the memory byte array is smaller than the CGBN, it fills the
 * remaining bytes with zeros.
 * @param[out] dst The destination CGBN
 * @param[in] src The memory byte array
 * @param[in] size The size of the memory byte array
 */
__host__ __device__ void cgbn_set_memory(env_t env, bn_t &dst,
                                         const uint8_t *src,
                                         const uint32_t size);

/**
 * Get a CGBN type from a size_t.
 * @param[out] dst The destination CGBN
 * @param[in] src The size_t
 */
__host__ __device__ void cgbn_set_size_t(env_t env, bn_t &dst,
                                         const size_t src);
/**
 * Get a size_t from a CGBN type.
 * @param[out] dst The destination size_t
 * @param[in] src The source CGBN
 * @return 1 for overflow, 0 otherwiese
 */
__host__ __device__ int32_t cgbn_get_size_t(env_t env, size_t &dst,
                                            const bn_t &src);
/**
 * Get a uint64_t from a CGBN type.
 * @param[out] dst The destination uint64_t
 * @param[in] src The source CGBN
 * @return 1 for overflow, 0 otherwiese
 */
__host__ __device__ int32_t cgbn_get_uint64_t(env_t env, uint64_t &dst,
                                              const bn_t &src);

__host__ __device__ int32_t cgbn_get_uint32_t(env_t env, uint32_t &dst,
                                              const bn_t &src);

/**
 * Get the cgbn number from the byte array.
 * @param[in] byte_array The byte array.
 * @param[out] out The cgbn number.
 * @return The Error code. 0 for success, 1 for failure.
 */
__host__ __device__ int32_t
cgbn_set_byte_array_t(env_t env, bn_t &out, const byte_array_t &byte_array);

/**
 * Get the sub byte array from the byte array.
 * @param[in] byte_array The byte array.
 * @param[in] index The index of the sub byte array.
 * @param[in] length The length of the sub byte array.
 * @param[out] out The sub byte array.
 * @return The Error code. 0 for success, 1 for failure.
 */
__host__ __device__ int32_t get_sub_byte_array_t(ArithEnv &arith,
                                                 const byte_array_t &byte_array,
                                                 const bn_t &index,
                                                 const bn_t &length,
                                                 byte_array_t &out);

/**
 * Convert and evm word to and address format
 * @param[in] arith The arithmetic envorinment
 * @param[inout] address The address variable
 */
__host__ __device__ void evm_address_conversion(ArithEnv &arith, bn_t &address);
__host__ __device__ void print_bnt(ArithEnv &arith, const bn_t &bn);
}  // namespace CuEVM
