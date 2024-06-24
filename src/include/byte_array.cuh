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

namespace cuEVM {
  /**
   * The byte array structure.
   * It has the size of the data and a pointer to the data.
  */
  typedef struct
  {
    size_t size; /**< The size of the array */
    uint8_t *data; /**< The content of the array */
  } byte_array_t;

  namespace byte_array {
    /**
     * Get the hex string from a nibble.
     * @param[in] nibble The nibble.
     * @return The hex string.
     */
    __host__ __device__ char hex_from_nibble(const uint8_t nibble);
    /**
     * Get the nibble from a hex string.
     * @param[in] hex The hex string.
     * @return The nibble.
     */
    __host__ __device__ uint8_t nibble_from_hex(const char hex);
    /**
     * Get the byte from two nibbles.
     * @param[in] high The high nibble.
     * @param[in] low The low nibble.
     */
    __host__ __device__ uint8_t byte_from_nibbles(const uint8_t high, const uint8_t low);
    /**
     * Get the hex string from a byte.
     * @param[in] byte The byte.
     * @param[out] dst The destination hex string.
     */
    __host__ __device__ void hex_from_byte(char *dst, const uint8_t byte);

    /**
     * Get the byte from two hex characters.
     * @param[in] high The high hex character.
     * @param[in] low The low hex character.
     * @return The byte.
    */
    __host__ __device__ uint8_t byte_from_two_hex_char(const char high, const char low);
    /**
     * Get the hex string from a byte array.
     * The hex string is allocated on the heap and needs to be freed.
     * @param[in] bytes The byte array.
     * @param[in] count The number of bytes.
     * @return The hex string.
     */
    __host__ __device__ char *hex_from_bytes(
      uint8_t *bytes,
      size_t count);
    /**
     * Print a byte array.
     * @param[in] bytes The byte array.
     * @param[in] count The number of bytes.
    */
    __host__ __device__ void print_bytes(
      uint8_t *bytes,
      size_t count);

    /**
     * Print the data content.
     * @param[in] data_content The data content.
    */
    __host__ __device__ void print_byte_array_t(
      byte_array_t &data_content);
    /**
     * Get the hex string from a data content.
     * The hex string is allocated on the heap and needs to be freed.
     * @param[in] data_content The data content.
     * @return The hex string.
    */
    __host__ char *hex_from_byte_array_t(
      byte_array_t &data_content);

    /**
     * Clean for leading zeroes in hex string
     * @param[inout]  hex_string
    */
    __host__ __device__ void rm_leading_zero_hex_string(
      char *hex_string);
    /**
     * Get the json object from a data content.
     * @param[in] data_content The data content.
     * @return The json object.
     */
    __host__ cJSON *json_from_byte_array_t(
      byte_array_t &data_content);

    /**
     * Copy the source byte array in the destination
     * considering a Big Endian format, the extra size
     * of the destination byte array will be padded with
     * zeros.
     * @param[out] dst The destination byte array
     * @param[in] src The source byte array
     * @return the difference in size -1 <, 0 =, 1 >
     */
    __host__ __device__ int32_t padded_copy_BE(
      const byte_array_t dst,
      const byte_array_t src
    );
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
    __host__ byte_array_t *get_cpu_instances(
        uint32_t count);

    /**
     * Free the cpu instances
     * @param[in] cpu_instances the cpu instances
     * @param[in] count the number of instances
    */
    __host__ void free_cpu_instances(
        byte_array_t *cpu_instances,
        uint32_t count);

    /**
     * Get the gpu instances for the return data from the cpu instances
     * @param[in] cpu_instances the cpu instances
     * @param[in] count the number of instances
     * @return the gpu instances
    */
    __host__ byte_array_t *get_gpu_instances_from_cpu_instances(
        byte_array_t *cpu_instances,
        uint32_t count);

    /**
     * Free the gpu instances
     * @param[in] gpu_instances the gpu instances
     * @param[in] count the number of instances
    */
    __host__ void free_gpu_instances(
        byte_array_t *gpu_instances,
        uint32_t count);

    /**
     * Get the cpu instances from the gpu instances
     * @param[in] gpu_instances the gpu instances
     * @param[in] count the number of instances
     * @return the cpu instances
    */
    __host__ byte_array_t *get_cpu_instances_from_gpu_instances(
        byte_array_t *gpu_instances,
        uint32_t count);
  }
}

#endif