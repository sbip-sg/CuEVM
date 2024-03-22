
#ifndef _UTILS_H_
#define _UTILS_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <cjson/cJSON.h>
#include <bigint.h>

#include "cgbn_wrapper.h"
#include "data_content.h"
#include "opcodes.h"
#include "error_codes.h"
#include "gas_cost.h"

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

#define CUDA_CHECK(action) cuda_check(action, #action, __FILE__, __LINE__)
#define CGBN_CHECK(report) cgbn_check(report, __FILE__, __LINE__)


template<uint32_t tpi, uint32_t bits, uint32_t windowBits, uint32_t stackSize, uint32_t memorySize, uint32_t storageSize>
struct mr_params {
  static constexpr uint32_t TPB = 0;
  static constexpr uint32_t MAX_ROTATION = 4;
  static constexpr uint32_t SHM_LIMIT = 0;
  static constexpr bool CONSTANT_TIME = false;

  static constexpr uint32_t TPI = tpi;
  static constexpr uint32_t BITS = bits;
  static constexpr uint32_t WINDOW_BITS = windowBits;
  static constexpr uint32_t STACK_SIZE = stackSize;
  static constexpr uint32_t MEMORY_SIZE = memorySize;
  static constexpr uint32_t STORAGE_SIZE = storageSize;

  static constexpr uint32_t MAX_CODE_SIZE = 500;
  static constexpr uint32_t MAX_STORAGE_SIZE = 100;
  static constexpr uint32_t PAGE_SIZE = 1024;
};

using utils_params = mr_params<8, 256, 1, 1024, 4096, 50>;
using evm_params = mr_params<8, 256, 1, 1024, 4096, 50>;

// common type defs

/**
 * The CGBN context type.  This is a template type that takes
 * the number of threads per instance and the
 * parameters class as template parameters.
*/
using context_t = cgbn_context_t<evm_params::TPI, evm_params>;

/**
 * The CGBN environment type. This is a template type that takes the
 * context type as a template parameter. It provides the CGBN functions.
*/
using env_t = cgbn_env_t<context_t, evm_params::BITS>;

/**
 * The CGBN base type for the given number of bit in environment.
*/
using bn_t = env_t::cgbn_t;
/**
 * The CGBN wide type with double the given number of bits in environment.
*/
using bn_wide_t = env_t::cgbn_wide_t;
/**
 * The EVM word type. also use for store CGBN base type.
*/
using evm_word_t =  cgbn_mem_t<evm_params::BITS>;


__host__ size_t adjusted_length(char** hex_string);
__host__ void hex_to_bytes(const char *hex_string, uint8_t *byte_array, size_t length);
void cuda_check(cudaError_t status, const char *action=NULL, const char *file=NULL, int32_t line=0);
void cgbn_check(cgbn_error_report_t *report, const char *file=NULL, int32_t line=0);
__host__ void from_mpz(uint32_t *words, uint32_t count, mpz_t value);
__host__ void to_mpz(mpz_t r, uint32_t *x, uint32_t count);
__host__ cJSON *get_json_from_file(const char *filepath);

#include "arith.cuh"

#endif