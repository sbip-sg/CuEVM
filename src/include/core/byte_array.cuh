// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#ifndef _BYTE_ARRAY_H_
#define _BYTE_ARRAY_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <cjson/cJSON.h>
#include "../utils/arith.cuh"

namespace cuEVM {
  enum PaddingDirection {
    NO_PADDING = 0,
    LEFT_PADDING = 1,
    RIGHT_PADDING = 2
  };
  
  /**
   * The byte array structure.
   * It has the size of the data and a pointer to the data.
  */
  struct byte_array_t
  {
    uint32_t size; /**< The size of the array */
    uint8_t *data; /**< The content of the array */
    /**
     * The default constructor.
     */
    __host__ __device__ byte_array_t() : size(0), data(nullptr) {};
    /**
     * The constructor with the size.
     * @param[in] size The size of the array.
     */
    __host__ __device__ byte_array_t(
      uint32_t size);
    /**
     * The constructor with the data.
     * @param[in] data The data of the array.
     * @param[in] size The size of the array.
     */
    __host__ __device__ byte_array_t(
      uint8_t *data,
      uint32_t size);
    /**
     * The constructor with the hex string.
     * @param[in] hex_string The hex string.
     * @param[in] endian The endian format.
     * @param[in] padding The padding direction.
     */
    __host__ __device__ byte_array_t(
      const char *hex_string,
      int32_t endian = LITTLE_ENDIAN,
      PaddingDirection padding = NO_PADDING);
    /**
     * The constructor with the hex string and a fixed size.
     * @param[in] hex_string The hex string.
     * @param[in] size the fixed size.
     * @param[in] endian The endian format.
     * @param[in] padding The padding direction.
     */
    __host__ __device__ byte_array_t(
      const char *hex_string,
      uint32_t size,
      int32_t endian = LITTLE_ENDIAN,
      PaddingDirection padding = NO_PADDING);
    /**
     * The destructor.
     */
    __host__ __device__ ~byte_array_t();
    /**
     * The copy constructor.
     * @param[in] other The other byte array.
     */
    __host__ __device__ byte_array_t(
      const byte_array_t &other);
    /**
     * The assignment operator.
     */
    __host__ __device__ byte_array_t& operator=(
      const byte_array_t &other);

    /**
     * Grow the size of the byte array with or without zero padding.
     * @param[in] new_size The new size of the byte array.
     * @param[in] zero_padding The zero padding flag.
     * @return The Error code. 0 for success, 1 for failure.
     */
    __host__ __device__ int32_t grow(
      uint32_t new_size,
      int32_t zero_padding = 0);

    /**
     * Print the byte array.
     */
    __host__ __device__ void print() const;
    /**
     * Get the hex string from the byte array.
     * The hex string is allocated on the heap and needs to be freed.
     * @return The hex string.
     */
    __host__ __device__ char* to_hex() const;
    /**
     * Get the json object from the byte array.
     * @return The json object.
     */
    __host__ __device__ cJSON *to_json() const;
    /**
     * Get the byte array from a hex string in Little Endian format.
     * @param[in] clean_hex_string The clean hex string.
     * @param[in] length The length of the clean hex string.
     * @return The Error code. 0 for success, 1 for failure.
     */
    __host__ __device__ int32_t from_hex_set_le(
      const char *clean_hex_string,
      int32_t length);
    /**
     * Get the byte array from a hex string in Big Endian format.
     * @param[in] clean_hex_string The clean hex string.
     * @param[in] length The length of the clean hex string.
     * @param[in] padded The padding direction ( 0 for left padding, 1 for right padding)
     * @return The Error code. 0 for success, 1 for failure.
     */
    __host__ __device__ int32_t from_hex_set_be(
      const char *clean_hex_string,
      int32_t length,
      PaddingDirection padding);
    
    /**
     * Get the byte array from a hex string.
     * @param[in] hex_string The hex string.
     * @param[in] endian The endian format.
     * @param[in] padding The padding direction.
     * @param[in] managed The managed flag.
     * @return The Error code. 0 for success, 1 for failure.
     */
    __host__ __device__ int32_t from_hex(
      const char *hex_string,
      int32_t endian = LITTLE_ENDIAN,
      PaddingDirection padding = NO_PADDING,
      int32_t managed = 0);
    /**
     * Copy the source byte array
     * considering a Big Endian format, the extra size
     * of the byte array will be padded with
     * zeros.
     * @param[in] src The source byte array
     * @return the difference in size -1 <, 0 =, 1 >
     */
    __host__ __device__ int32_t padded_copy_BE(
      const byte_array_t src
    );

    __host__ __device__ int32_t to_bn_t(
      ArithEnv &arith,
      bn_t &out
    ) const;
  };

  namespace byte_array {
    // GPU-CPU interaction
    /**
     * Copy data content between two device memories
     * @param[out] dst_instances the destination memory
     * @param[in] src_instances the source memory
     * @param[in] count the number of instances to copy
    */
    __global__ void transfer_kernel(
        byte_array_t *dst_instances,
        byte_array_t *src_instances,
        uint32_t count);
    /**
     * Get the cpu instances for the return data
     * @param[in] count the number of instances
     * @return the cpu instances
    */
    __host__ byte_array_t *get_cpu(
        uint32_t count);

    /**
     * Free the cpu instances
     * @param[in] cpu_instances the cpu instances
     * @param[in] count the number of instances
    */
    __host__ void cpu_free(
        byte_array_t *cpu_instances,
        uint32_t count);

    /**
     * Get the gpu instances for the return data from the cpu instances
     * @param[in] cpu_instances the cpu instances
     * @param[in] count the number of instances
     * @return the gpu instances
    */
    __host__ byte_array_t *gpu_from_cpu(
        byte_array_t *cpu_instances,
        uint32_t count);

    /**
     * Free the gpu instances
     * @param[in] gpu_instances the gpu instances
     * @param[in] count the number of instances
    */
    __host__ void gpu_free(
        byte_array_t *gpu_instances,
        uint32_t count);

    /**
     * Get the cpu instances from the gpu instances
     * @param[in] gpu_instances the gpu instances
     * @param[in] count the number of instances
     * @return the cpu instances
    */
    __host__ byte_array_t *cpu_from_gpu(
        byte_array_t *gpu_instances,
        uint32_t count);
  }
}

#endif