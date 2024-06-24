
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

#endif