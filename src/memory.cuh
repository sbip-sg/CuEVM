#ifndef _MEMORY_H_
#define _MEMORY_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include "utils.h"

template<class params>
class memory_t {
  public:
  typedef typename arith_env_t<params>::bn_t      bn_t;
  typedef arith_env_t<params>                     arith_t;
  static const size_t                             PAGE_SIZE = params::PAGE_SIZE;

  //memory data structure  
  typedef struct {
    size_t size;
    size_t alocated_size;
    uint8_t *data;
  } memory_data_t;

  //content of the memory
  memory_data_t *_content;
  arith_t       _arith;

  //constructor
  __host__ __device__ __forceinline__ memory_t(arith_t arith, memory_data_t *content) : _arith(arith), _content(content) {
  }

  //get the size of the memory
  __host__ __device__ __forceinline__ size_t size() {
    return _content->size;
  }


  //get the all data of the memory
  __host__ __device__ __forceinline__ uint8_t *get_data() {
    return _content->data;
  }

  
  __host__ __device__ __forceinline__ void allocate_pages(size_t new_size, uint32_t &error_code) {
    if (new_size <= _content->alocated_size) {
      return;
    }
    size_t no_pages = (new_size / PAGE_SIZE) + 1;
    #ifdef __CUDA_ARCH__
    __syncthreads();
    if (threadIdx.x == 0) {
    #endif
      uint8_t *new_data = (uint8_t *)malloc(no_pages * PAGE_SIZE);
      /*
      printf("new_data=%p\n", new_data);
      printf("new_size=%lx\n", new_size);
      printf("no_pages=%lx\n", no_pages);
      printf("PAGE_SIZE=%lx\n", PAGE_SIZE);
      printf("new alloc size=%lx\n", no_pages * PAGE_SIZE);
      printf("old data=%p\n", _content->data);
      printf("old size=%lx\n", _content->size);
      printf("old alloc size=%lx\n", _content->alocated_size);
      */
      if (new_data == NULL) {
        error_code = ERR_MEMORY_INVALID_ALLOCATION;
        return;
      }
      // 0 all the data
      memset(new_data, 0, no_pages * PAGE_SIZE);
      memcpy(new_data, _content->data, _content->size);
      _content->alocated_size = no_pages * PAGE_SIZE;
      free(_content->data);
      _content->data = new_data;
    #ifdef __CUDA_ARCH__
    }
    __syncthreads();
    #endif
  }

  __host__ __device__ __forceinline__ void grow_cost(size_t &new_size, bn_t &gas_cost) {
      size_t memory_size_word = (_content->size + 31) / 32;
      size_t new_memory_size_word = (new_size + 31) / 32;
      size_t memory_cost = (memory_size_word * memory_size_word) / 512 + 3 * memory_size_word;
      size_t new_memory_cost = (new_memory_size_word * new_memory_size_word) / 512 + 3 * new_memory_size_word;
      size_t new_cost = new_memory_cost - memory_cost;
      bn_t new_cost_bn;
      _arith.from_size_t_to_cgbn(new_cost_bn, new_cost);
      cgbn_add(_arith._env, gas_cost, gas_cost, new_cost_bn);
      // because size is always a multiple of 32
      new_size = new_memory_size_word * 32;
  }

  __host__ __device__ __forceinline__ void grow(size_t offset, bn_t &gas_cost, uint32_t &error_code) {
    if (offset > _content->alocated_size) {
      allocate_pages(offset, error_code);
    }
    if (offset > _content->size) {
      grow_cost(offset, gas_cost);
      _content->size = offset;
    }
  }

  //get the data of the memory at a specific index and length
  __host__ __device__ __forceinline__ uint8_t *get(size_t index, size_t length, bn_t &gas_cost, uint32_t &error_code) {
    grow(index + length, gas_cost, error_code);
    return _content->data + index;
  }

  //set the data of the memory at a specific index and length
  __host__ __device__ __forceinline__ void set(uint8_t *data, size_t index, size_t length, bn_t &gas_cost, uint32_t &error_code) {
    grow(index + length, gas_cost, error_code);
    #ifdef __CUDA_ARCH__
    if (threadIdx.x == 0) {
    #endif
      memcpy(_content->data + index, data, length);
    #ifdef __CUDA_ARCH__
    }
    __syncthreads();
    #endif
  }


  // copy the data information
  __host__ __device__ __forceinline__ void copy_info(memory_data_t *dest) {
    #ifdef __CUDA_ARCH__
    __syncthreads();
    if (threadIdx.x == 0) {
    #endif
      dest->alocated_size = _content->alocated_size;
      dest->size = _content->size;
      dest->data = _content->data;
    #ifdef __CUDA_ARCH__
    }
    __syncthreads();
    #endif
  }

  // copy content to another memory
  __host__ __device__ __forceinline__ void copy_content(memory_data_t *dest) {
    /*
    #ifdef __CUDA_ARCH__
    __syncthreads();
    if (threadIdx.x == 0) {
    #endif
    */
      dest->alocated_size = _content->size;
      dest->size = _content->size;
      memcpy(dest->data, _content->data, _content->size);
    /*
    #ifdef __CUDA_ARCH__
    }
    __syncthreads();
    #endif
    */
  }

  __device__ __forceinline__ void free_memory() {
    #ifdef __CUDA_ARCH__
    __syncthreads();
    if (threadIdx.x == 0) {
    #endif
      if(_content->alocated_size>0)
        free(_content->data);
    #ifdef __CUDA_ARCH__
    }
    __syncthreads();
    #endif
    _content->alocated_size = 0;
    _content->size = 0;
    _content->data = NULL;
  }

  // generate the memory content structure info on the host
  __host__ static memory_data_t *get_memories_info(uint32_t count) {
    memory_data_t *cpu_memories = (memory_data_t *)malloc(sizeof(memory_data_t) * count);
    for (uint32_t idx = 0; idx < count; idx++) {
      cpu_memories[idx].size = 0;
      cpu_memories[idx].data = NULL;
    }
    return cpu_memories;
  }

  __host__ static memory_data_t *get_gpu_memories_info(memory_data_t *cpu_memories, uint32_t count) {
    memory_data_t *gpu_memories;
    cudaMalloc((void **)&gpu_memories, sizeof(memory_data_t)*count);
    cudaMemcpy(gpu_memories, cpu_memories, sizeof(memory_data_t)*count, cudaMemcpyHostToDevice);
    return gpu_memories;
  }


  // free the memory content structure info on the host
  __host__ static void free_memories_info(memory_data_t *cpu_memories, uint32_t count) {
    free(cpu_memories);
  }

  
  __host__ static void free_gpu_memories_info(memory_data_t *gpu_memories, uint32_t count) {
    cudaFree(gpu_memories);
  }

  //free the memory content structure on the device from the info from gpu
  __host__ static void free_memory_data(memory_data_t *cpu_memories, uint32_t count) {
    for (uint32_t idx = 0; idx < count; idx++) {
      if(cpu_memories[idx].data!=NULL)
        free(cpu_memories[idx].data);
    }
    free(cpu_memories);
  }
   __host__ static memory_data_t  *get_memories_from_gpu(memory_data_t  *gpu_memories, uint32_t count) {
    memory_data_t  *cpu_memories;
    cpu_memories=(memory_data_t *)malloc(sizeof(memory_data_t) * count);
    cudaMemcpy(cpu_memories, gpu_memories, sizeof(memory_data_t)*count, cudaMemcpyDeviceToHost);


    // 1. alocate the memory for gpu memory as memory which can be addressed by the cpu
    memory_data_t  *tmp_cpu_memories, *new_gpu_memories;
    tmp_cpu_memories=(memory_data_t *)malloc(sizeof(memory_data_t) * count);
    memcpy(tmp_cpu_memories, cpu_memories, sizeof(memory_data_t)*count);
    for(size_t idx=0; idx<count; idx++) {
      if (tmp_cpu_memories[idx].size > 0) {
        cudaMalloc((void **)&tmp_cpu_memories[idx].data, sizeof(uint8_t) * tmp_cpu_memories[idx].size);
      } else {
        tmp_cpu_memories[idx].data = NULL;
      }
      tmp_cpu_memories[idx].alocated_size = tmp_cpu_memories[idx].size;
    }
    cudaMalloc((void **)&new_gpu_memories, sizeof(memory_data_t) * count);
    cudaMemcpy(new_gpu_memories, tmp_cpu_memories, sizeof(memory_data_t)*count, cudaMemcpyHostToDevice);
    free(tmp_cpu_memories);

    // 2. call the kernel to copy the memory between the gpu memories
    kernel_get_memory<params><<<1, count>>>(new_gpu_memories, gpu_memories, count);
    cudaFree(gpu_memories);
    gpu_memories=new_gpu_memories;

    // 3. copy the gpu memories back in the cpu memories
    cudaMemcpy(cpu_memories, gpu_memories, sizeof(memory_data_t)*count, cudaMemcpyDeviceToHost);
    tmp_cpu_memories=(memory_data_t *)malloc(sizeof(memory_data_t) * count);
    memcpy(tmp_cpu_memories, cpu_memories, sizeof(memory_data_t)*count);
    for(size_t idx=0; idx<count; idx++) {
      if (tmp_cpu_memories[idx].size > 0) {
        tmp_cpu_memories[idx].data=(uint8_t *)malloc(sizeof(uint8_t) * tmp_cpu_memories[idx].size);
        cudaMemcpy(tmp_cpu_memories[idx].data, cpu_memories[idx].data, sizeof(uint8_t) * tmp_cpu_memories[idx].size, cudaMemcpyDeviceToHost);
        cudaFree(cpu_memories[idx].data);
      } else {
        tmp_cpu_memories[idx].data = NULL;
      }
    }
    cudaFree(gpu_memories);
    free(cpu_memories);
    cpu_memories=tmp_cpu_memories;
    return cpu_memories;
   }
  
  __host__ __device__ void print() {
    printf("size=%lx\n", _content->size);
    printf("data: ");
    if (_content->size > 0)
      print_bytes(_content->data, _content->size);
    printf("\n");
  }

  __host__ cJSON *to_json() {
    char hex_string[67]="0x";
    char *bytes_string=NULL;
    mpz_t mpz_data_size;
    mpz_init(mpz_data_size);
    cJSON *data_json = cJSON_CreateObject();
    
    mpz_set_ui(mpz_data_size, _content->size >> 32);
    mpz_mul_2exp(mpz_data_size, mpz_data_size, 32);
    mpz_add_ui(mpz_data_size, mpz_data_size, _content->size & 0xffffffff);
    strcpy(hex_string+2, mpz_get_str(NULL, 16, mpz_data_size));
    cJSON_AddStringToObject(data_json, "size", hex_string);

    if (_content->size > 0) {
      bytes_string = bytes_to_hex(_content->data, _content->size);
      cJSON_AddStringToObject(data_json, "data", bytes_string);
      free(bytes_string);
    } else {
      cJSON_AddStringToObject(data_json, "data", "0x");
    }
    
    mpz_clear(mpz_data_size);
    return data_json;
  }
};

template<class params>
__global__ void kernel_get_memory(typename memory_t<params>::memory_data_t *dst_instances, typename memory_t<params>::memory_data_t *src_instances, uint32_t instance_count) {
  uint32_t instance=blockIdx.x*blockDim.x + threadIdx.x;
  typedef memory_t<params>    memory_t;
  typedef arith_env_t<params> arith_t;
  
  if(instance>=instance_count)
    return;

  // setup arithmetic
  arith_t arith(cgbn_report_monitor);
  /*
  printf("GET size=%lu\n", src_instances[instance].size);
  printf("GET data address=%p\n", src_instances[instance].data);
  printf("GET lowestbit=%02x\n", src_instances[instance].data[31]);
  */
  memory_t  memory(arith, &(src_instances[instance]));
  memory.copy_content(&(dst_instances[instance]));
  /*
  printf("GET data address=%p\n", memory._content->data);
  printf("GET lowestbit=%02x\n", memory._content->data[31]);
  printf("GET D data address=%p\n", dst_instances[instance].data);
  printf("GET D lowestbit=%02x\n", dst_instances[instance].data[31]);
  */
  memory.free_memory();
}

#endif