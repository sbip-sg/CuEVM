
#ifndef _UTILS_H_
#define _UTILS_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#ifdef __CUDA_ARCH__
#include <cuda.h>
#endif
#include <gmp.h>
#ifndef __CGBN_H__
#define __CGBN_H__
#include <cgbn/cgbn.h>
#endif
#include <cjson/cJSON.h>
#include "opcodes.h"
#include "error_codes.h"
#include "arith.cuh"
#include "gas_cost.h"

#ifdef __CUDA_ARCH__
#ifndef MULTIPLE_THREADS_PER_INSTANCE
#define MULTIPLE_THREADS_PER_INSTANCE
#endif
#endif
#ifdef MULTIPLE_THREADS_PER_INSTANCE
#define ONE_THREAD_PER_INSTANCE(X) if (threadIdx.x == 0) { X } __syncthreads();
#define SHARED_MEMORY __shared__
#else
#define ONE_THREAD_PER_INSTANCE(X) X
#define SHARED_MEMORY
#endif

__host__ size_t adjusted_length(char** hex_string) {
    if (strncmp(*hex_string, "0x", 2) == 0 || strncmp(*hex_string, "0X", 2) == 0) {
        *hex_string += 2;  // Skip the "0x" prefix
        return (strlen(*hex_string) / 2);
    }
    return (strlen(*hex_string) / 2);
}

__host__ void hex_to_bytes(const char *hex_string, uint8_t *byte_array, size_t length)
{
    for (size_t idx = 0; idx < length; idx += 2)
    {
        sscanf(&hex_string[idx], "%2hhx", &byte_array[idx / 2]);
    }
}

// support routines
void cuda_check(cudaError_t status, const char *action=NULL, const char *file=NULL, int32_t line=0) {
  // check for cuda errors

  if(status!=cudaSuccess) {
    printf("CUDA error occurred: %s\n", cudaGetErrorString(status));
    if(action!=NULL)
      printf("While running %s   (file %s, line %d)\n", action, file, line);
    exit(1);
  }
}

void cgbn_check(cgbn_error_report_t *report, const char *file=NULL, int32_t line=0) {
  // check for cgbn errors

  if(cgbn_error_report_check(report)) {
    printf("\n");
    printf("CGBN error occurred: %s\n", cgbn_error_string(report));

    if(report->_instance!=0xFFFFFFFF) {
      printf("Error reported by instance %d", report->_instance);
      if(report->_blockIdx.x!=0xFFFFFFFF || report->_threadIdx.x!=0xFFFFFFFF)
        printf(", ");
      if(report->_blockIdx.x!=0xFFFFFFFF)
      printf("blockIdx=(%d, %d, %d) ", report->_blockIdx.x, report->_blockIdx.y, report->_blockIdx.z);
      if(report->_threadIdx.x!=0xFFFFFFFF)
        printf("threadIdx=(%d, %d, %d)", report->_threadIdx.x, report->_threadIdx.y, report->_threadIdx.z);
      printf("\n");
    }
    else {
      printf("Error reported by blockIdx=(%d %d %d)", report->_blockIdx.x, report->_blockIdx.y, report->_blockIdx.z);
      printf("threadIdx=(%d %d %d)\n", report->_threadIdx.x, report->_threadIdx.y, report->_threadIdx.z);
    }
    if(file!=NULL)
      printf("file %s, line %d\n", file, line);
    exit(1);
  }
}

#define CUDA_CHECK(action) cuda_check(action, #action, __FILE__, __LINE__)
#define CGBN_CHECK(report) cgbn_check(report, __FILE__, __LINE__)


template<uint32_t tpi, uint32_t bits, uint32_t window_bits, uint32_t stack_size, uint32_t memory_size, uint32_t storage_size>
class mr_params_t {
  public:
  // parameters used by the CGBN context
  static const uint32_t TPB=0;                     // get TPB from blockDim.x  
  static const uint32_t MAX_ROTATION=4;            // good default value
  static const uint32_t SHM_LIMIT=0;               // no shared mem available
  static const bool     CONSTANT_TIME=false;       // constant time implementations aren't available yet

  // parameters used locally in the application
  static const uint32_t TPI=tpi;                   // threads per instance
  static const uint32_t BITS=bits;                 // instance size
  static const uint32_t WINDOW_BITS=window_bits;   // window size
  static const uint32_t STACK_SIZE=stack_size;
  static const uint32_t MEMORY_SIZE=memory_size;          // memory size in bytes
  static const uint32_t STORAGE_SIZE=storage_size;          // memory size in bytes
  static const uint32_t MAX_CODE_SIZE=500;         // total instances official 24576
  static const uint32_t MAX_STORAGE_SIZE=100;        // words per instance
  static const uint32_t PAGE_SIZE=1024;        // words per instance
};

typedef mr_params_t<8, 256, 1, 1024, 4096, 50> utils_params;


typedef struct
{
  size_t size;
  uint8_t *data;
} data_content_t;


__host__ void from_mpz(uint32_t *words, uint32_t count, mpz_t value) {
  size_t written;

  if(mpz_sizeinbase(value, 2)>count*32) {
    fprintf(stderr, "from_mpz failed -- result does not fit\n");
    exit(1);
  }

  mpz_export(words, &written, -1, sizeof(uint32_t), 0, 0, value);
  while(written<count)
    words[written++]=0;
}

__host__ void to_mpz(mpz_t r, uint32_t *x, uint32_t count) {
  mpz_import(r, count, -1, sizeof(uint32_t), 0, 0, x);
}

template<class params>
__host__ __device__ void print_bn(cgbn_mem_t<params::BITS> bn) {
  for(size_t idx=0; idx<params::BITS/32; idx++)
    printf("%08x ", bn._limbs[params::BITS/32 - 1 - idx]);
  printf("\n");
}

__host__ __device__ void print_bytes(uint8_t *bytes, size_t count) {
  for(size_t idx=0; idx<count; idx++)
    printf("%02x", bytes[idx]);
  printf("\n");
}

char *bytes_to_hex(uint8_t *bytes, size_t count) {
  char *hex_string = (char *)malloc(count*2+1);
  char *return_string = (char *)malloc(count*2+1+2);
  for(size_t idx=0; idx<count; idx++)
    sprintf(&hex_string[idx*2], "%02x", bytes[idx]);
  hex_string[count*2]=0;
  strcpy(return_string + 2, hex_string);
  free(hex_string);
  return_string[0]='0';
  return_string[1]='x';
  return return_string;
}

__host__ cJSON *get_json_from_file(const char *filepath) {
    FILE *fp = fopen(filepath, "r");
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *buffer = (char *)malloc(size + 1);
    fread(buffer, 1, size, fp);
    fclose(fp);
    buffer[size] = '\0';
    // parse
    cJSON *root = cJSON_Parse(buffer);
    free(buffer);
    return root;
}

__host__ __device__ uint8_t *expand_memory(uint8_t *memory, size_t current_size, size_t new_size) {
  SHARED_MEMORY uint8_t *new_memory;

  ONE_THREAD_PER_INSTANCE(
    new_memory = NULL;
    if (new_size != 0) {
      new_memory = (uint8_t *)malloc(new_size);
      memset(new_memory, 0, new_size);
      if ((memory != NULL) && (current_size > 0))
        if (current_size > new_size)
          memcpy(new_memory, memory, new_size);
        else
          memcpy(new_memory, memory, current_size);

    }
  )
  return new_memory;
}
#endif