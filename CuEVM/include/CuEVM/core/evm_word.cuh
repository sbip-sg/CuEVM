// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2024 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation
// Programme) Author: Stefan-Dan Ciocirlan Date: 2024-10-09
// SPDX-License-Identifier: MIT
#pragma once

#include <CGBN/cgbn.h>
#include <cuda.h>
#include <stdint.h>

#include <CuEVM/core/byte_array.cuh>
#include <CuEVM/utils/evm_defines.cuh>

namespace CuEVM {
/**
 * The EVM word type. also use for store CGBN base type.
 */
struct evm_word_t : cgbn_mem_t<CuEVM::word_bits> {
    /**
     * The default constructor.
     */
    __host__ __device__ evm_word_t() {}
    /**
     * The copy constructor.
     */
    __host__ __device__ evm_word_t(const evm_word_t &src);
    /**
     * The constructor from a uint32_t.
     * @param[in] value The source uint32_t
     */
    __host__ __device__ evm_word_t(uint32_t value);

    /**
     * The assignment operator.
     * @param[in] src The source evm_word_t
     */
    __host__ __device__ evm_word_t &operator=(const evm_word_t &src);
    /**
     * The assignment operator for uint32_t.
     * @param[in] value The source uint32_t
     */
    __host__ __device__ evm_word_t &operator=(uint32_t value);

    /**
     * The equality operator.
     * @param[in] other The other evm_word_t
     * @return 1 for equal, 0 otherwise
     */
    __host__ __device__ int32_t operator==(const evm_word_t &other) const;

    /**
     * The equality operator for uint32_t
     * @param[in] value the uint32_t value
     * @return 1 for equal, 0 otherwise
     */
    __host__ __device__ int32_t operator==(const uint32_t &value) const;
    /**
     * Set the evm_word_t from a hex string.
     * The hex string is in Big Endian format.
     * @param[in] hex_string The source hex string
     * @return 0 for success, 1 otherwise
     */
    __host__ int32_t from_hex(const char *hex_string);
    /**
     * Set the evm_word_t from a byte array.
     * The byte array is in Big Endian format.
     * @param[in] byte_array The source byte array
     * @param[in] endian The endian format
     * @return 0 for success, 1 otherwise
     */
    __host__ __device__ int32_t from_byte_array_t(byte_array_t &byte_array, int32_t endian = LITTLE_ENDIAN);
    /**
     * Set the evm_word_t from a size_t.
     * @param[in] value The source size_t
     * @return 0 for success, 1 otherwise
     */
    __host__ __device__ int32_t from_size_t(size_t value);
    /**
     * Set the evm_word_t from a uint64_t.
     * @param[in] value The source uint64_t
     * @return 0 for success, 1 otherwise
     */
    __host__ __device__ int32_t from_uint64_t(uint64_t value);
    /**
     * Set the evm_word_t from a uint32_t.
     * @param[in] value The source uint32_t
     * @return 0 for success, 1 otherwise
     */
    __host__ __device__ int32_t from_uint32_t(uint32_t value);

    /**
     * Print the evm_word_t.
     */
    __host__ __device__ void print() const;

    /**
     * Compare two evm_word_t values.
     * @param[in] a Pointer to the first evm_word_t
     * @param[in] b Pointer to the second evm_word_t
     * @return -1 if a < b, 0 if a == b, 1 if a > b
     */
    __host__ __device__ int32_t evm_word_t_compare(const evm_word_t *a, const evm_word_t *b);

    /**
     * Get the hex string from the evm_word_t.
     * The hex string is in Big Endian format.
     * If the caller does not provide a hex string, it allocates one.
     * Note: The caller must free the hex string.
     * @param[inout] hex_string The destination hex string
     * @param[in] pretty If the hex string should be left trimmed of zeros
     * @param[in] count The number of the least significant
     * limbs to convert
     * @return The hex string
     */
    __host__ __device__ char *to_hex(char *hex_string = nullptr, int32_t pretty = 0,
                                     uint32_t count = CuEVM::cgbn_limbs) const;

    __host__ __device__ void print_as_compact_hex() const;

    /**
     * Get the byte array from the evm_word_t.
     * The byte array is in Big Endian format.
     * If the caller does not provide a byte array, it allocates one.
     * Note: The caller must free the byte array.
     * @param[out] byte_array The destination byte array
     * @param[in] endian The endian format
     * @return 0 for success, 1 otherwise
     */
    __host__ __device__ int32_t to_byte_array_t(byte_array_t &byte_array, int32_t endian = BIG_ENDIAN) const;
    /**
     * Get the bit array from the evm_word_t.
     * The bit array is in Big Endian format.
     * If the caller does not provide a byte array, it allocates one.
     * Note: The caller must free the byte array.
     * @param[out] bit_array The destination bit array
     * @param[in] endian The endian format
     * @return 0 for success, 1 otherwise
     */
    __host__ __device__ int32_t to_bit_array_t(byte_array_t &bit_array, int32_t endian = LITTLE_ENDIAN) const;
};

typedef cgbn_mem_t<CuEVM::word_bits> *cgbn_evm_word_t_ptr;
}  // namespace CuEVM
