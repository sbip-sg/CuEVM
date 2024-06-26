
#ifndef _UTILS_H_
#define _UTILS_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <cjson/cJSON.h>
#include <CGBN/cgbn.h>
#include <gmp.h>

#ifdef __CUDA_ARCH__
#ifndef MULTIPLE_THREADS_PER_INSTANCE
#define MULTIPLE_THREADS_PER_INSTANCE
#endif
#endif
#ifdef MULTIPLE_THREADS_PER_INSTANCE
#define ONE_THREAD_PER_INSTANCE(X) __syncthreads(); if (threadIdx.x == 0) { X } __syncthreads();
#define SHARED_MEMORY __shared__
#else
#define ONE_THREAD_PER_INSTANCE(X) X
#define SHARED_MEMORY
#endif

#ifdef DEBUG
#define DEBUG_PRINT(fmt, args...) fprintf(stderr, "DEBUG: %s:%d:%s(): " fmt, \
  __FILE__, __LINE__, __func__, ##args)
#else
#define DEBUG_PRINT(fmt, args...) /* Don't do anything in release builds */
#endif

#define CUDA_CHECK(action) cuda_check(action, #action, __FILE__, __LINE__)
#define CGBN_CHECK(report) cgbn_check(report, __FILE__, __LINE__)


__host__ size_t adjusted_length(char** hex_string);
__host__ void hex_to_bytes(const char *hex_string, uint8_t *byte_array, size_t length);
void cuda_check(cudaError_t status, const char *action=NULL, const char *file=NULL, int32_t line=0);
void cgbn_check(cgbn_error_report_t *report, const char *file=NULL, int32_t line=0);
__host__ void from_mpz(uint32_t *words, uint32_t count, mpz_t value);
__host__ void to_mpz(mpz_t r, uint32_t *x, uint32_t count);
__host__ cJSON *get_json_from_file(const char *filepath);

namespace cuEVM {
  namespace utils {
    /**
     * If it is a hex character.
     * @param[in] hex The character.
     * @return 1 if it is a hex character, 0 otherwise.
     */
    __host__ __device__ int32_t is_hex(
      const char hex);
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
     * Check if a character is a hex character.
     * @param[in] hex The character.
     * @return 1 if it is a hex character, 0 otherwise.
     */
    __host__ __device__ int32_t is_hex(const char hex);
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
     * Get the number of bytes oh a string
     * @param[in] hex_string
     * @return the number of bytes
     */
    __host__ __device__ int32_t hex_string_length(
      const char *hex_string);
    /**
     * Clean the hex string from prefix and return the length
     * @param[inout] hex_string
     * @return the length of the hex string
     */
    __host__ __device__ int32_t clean_hex_string(
      char **hex_string);
    /**
     * Remove the leading zeros of an hex string
     * @param[inout] hex_string
     * @return the length of the hex string
     */
    __host__ __device__ int32_t hex_string_without_leading_zeros(
      char *hex_string);
  }
}

#endif