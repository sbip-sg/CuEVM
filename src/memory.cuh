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
  bn_t          _memory_cost;

  //constructor
  __host__ __device__ __forceinline__ memory_t(arith_t arith, memory_data_t *content) : _arith(arith), _content(content) {
    cgbn_set_ui32(_arith._env, _memory_cost, 0);
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
    SHARED_MEMORY uint8_t *new_data;
    ONE_THREAD_PER_INSTANCE(
      new_data = (uint8_t *)malloc(no_pages * PAGE_SIZE);
      if (new_data == NULL) {
        error_code = ERR_MEMORY_INVALID_ALLOCATION;
        return;
      }
      // 0 all the data
      memset(new_data, 0, no_pages * PAGE_SIZE);
      if ( (_content->alocated_size > 0) && (_content->data != NULL)) {
        memcpy(new_data, _content->data, _content->size);
        free(_content->data);
        _content->data = NULL;
        _content->alocated_size = 0;
      }
      _content->alocated_size = no_pages * PAGE_SIZE;
      _content->data = new_data;
    )
  }

  __host__ __device__ __forceinline__ void grow_cost_s(size_t &new_size, bn_t &gas_cost) {
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

  __host__ __device__ __forceinline__ size_t grow_cost(bn_t &index, bn_t &length, bn_t &gas_cost, bn_t &remaining_gas, uint32_t &error_code) {
      bn_t offset;
      int32_t overflow = cgbn_add(_arith._env, offset, index, length);
      // verify if is larger than size_t
      bn_t MAX_SIZE_T;
      cgbn_set_ui32(_arith._env, MAX_SIZE_T, 1);
      cgbn_shift_left(_arith._env, MAX_SIZE_T, MAX_SIZE_T, 64);
      if (cgbn_compare(_arith._env, offset, MAX_SIZE_T) >= 0) {
        overflow = 1;
      }
      // memort_size_word = (last_offset + 31) / 32
      bn_t memory_size_word;
      cgbn_add_ui32(_arith._env, memory_size_word, offset, 31);
      cgbn_div_ui32(_arith._env, memory_size_word, memory_size_word, 32);
      // memory_cost = (memory_size_word * memory_size_word) / 512 + 3 * memory_size_word
      bn_t memory_cost;
      cgbn_mul(_arith._env, memory_cost, memory_size_word, memory_size_word);
      cgbn_div_ui32(_arith._env, memory_cost, memory_cost, 512);
      bn_t tmp;
      cgbn_mul_ui32(_arith._env, tmp, memory_size_word, 3);
      cgbn_add(_arith._env, memory_cost, memory_cost, tmp);
      //  gas_cost = gas_cost + memory_cost - old_memory_cost
      bn_t memory_expansion_cost;
      if (cgbn_compare(_arith._env, memory_cost, _memory_cost) == 1) {
        cgbn_sub(_arith._env, memory_expansion_cost, memory_cost, _memory_cost);
        // set the new memory cost
        cgbn_set(_arith._env, _memory_cost, memory_cost);
      } else {
        cgbn_set_ui32(_arith._env, memory_expansion_cost, 0);
      }
      // size is always a multiple of 32
      cgbn_mul_ui32(_arith._env, offset, memory_size_word, 32);
      // get the new size
      size_t new_size;
      if ( (cgbn_compare(_arith._env, memory_expansion_cost, remaining_gas) == 1) || (overflow != 0) ) {
        error_code = ERR_OUT_OF_GAS;
        new_size = 0;
        if (overflow != 0) {
          cgbn_add_ui32(_arith._env, memory_expansion_cost, remaining_gas, 1);
        }
      } else {
        new_size = _arith.from_cgbn_to_size_t(offset);
      }
      cgbn_add(_arith._env, gas_cost, gas_cost, memory_expansion_cost);
      return new_size;
  }

  __host__ __device__ __forceinline__ void grow(bn_t &index, bn_t &length, bn_t &gas_cost, bn_t &remaining_gas, uint32_t &error_code) {
    size_t offset = grow_cost(index, length, gas_cost, remaining_gas, error_code);
    if ( (error_code == ERR_NONE) && (offset > _content->size) ) {
      _content->size = offset;
      if (offset > _content->alocated_size) {
        allocate_pages(offset, error_code);
      }
    }
  }
  
  __host__ __device__ __forceinline__ void grow_s(size_t index, size_t length, bn_t &gas_cost, uint32_t &error_code) {
    size_t offset = index + length;
    // overflow verification
    if ( (offset < index) || (offset < length) ) {
      // maybe set someway of out of gas
      error_code = ERR_MEMORY_INVALID_ALLOCATION;
      return;
    }
    if (offset > _content->size) {
      grow_cost(offset, gas_cost);
      _content->size = offset;
    }
    if (offset > _content->alocated_size) {
      allocate_pages(offset, error_code);
    }
  }

  //get the data of the memory at a specific index and length
  __host__ __device__ __forceinline__ uint8_t *get_s(size_t index, size_t length, bn_t &gas_cost, uint32_t &error_code) {
    grow(index, length, gas_cost, error_code);
    return _content->data + index;
  }

   __host__ __device__ __forceinline__ uint8_t *get(bn_t &index, bn_t &length, bn_t &gas_cost, bn_t &remaining_gas, uint32_t &error_code) {
    grow(index, length, gas_cost, remaining_gas, error_code);
    size_t index_s = _arith.from_cgbn_to_size_t(index);
    if (error_code == ERR_NONE) {
      return _content->data + index_s;
    } else {
      return NULL;
    }
  }

  //set the data of the memory at a specific index and length
  __host__ __device__ __forceinline__ void set_s(uint8_t *data, size_t index, size_t length, bn_t &gas_cost, uint32_t &error_code) {
    grow(index, length, gas_cost, error_code);
    ONE_THREAD_PER_INSTANCE(
      if ( (data != NULL) && (length > 0) && (error_code == ERR_NONE) )
        memcpy(_content->data + index, data, length);
    )
  }

  __host__ __device__ __forceinline__ void set(uint8_t *data, bn_t &index, bn_t &length, bn_t &gas_cost, bn_t &remaining_gas, uint32_t &error_code) {
    grow(index, length, gas_cost, remaining_gas, error_code);
    size_t index_s = _arith.from_cgbn_to_size_t(index);
    size_t length_s = _arith.from_cgbn_to_size_t(length);
    if ( (data != NULL) && (length_s > 0) && (error_code == ERR_NONE) ) {
      ONE_THREAD_PER_INSTANCE(
        memcpy(_content->data + index_s, data, length_s);
      )
    }
  }


  // copy the data information
  __host__ __device__ __forceinline__ void copy_info(memory_data_t *dest) {
    ONE_THREAD_PER_INSTANCE(
      dest->alocated_size = _content->alocated_size;
      dest->size = _content->size;
      dest->data = _content->data;
    )
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
      if (_content->size > 0) {
        memcpy(dest->data, _content->data, _content->size);
      } else {
        dest->data = NULL;
      }
    /*
    #ifdef __CUDA_ARCH__
    }
    __syncthreads();
    #endif
    */
  }

  __device__ __forceinline__ void free_memory() {
    if( (_content->alocated_size>0) && (_content->data!=NULL)) {
      ONE_THREAD_PER_INSTANCE(
          free(_content->data);
      )

    }
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
      if( (cpu_memories[idx].data!=NULL) && (cpu_memories[idx].alocated_size>0)) {
        free(cpu_memories[idx].data);
        cpu_memories[idx].data=NULL;
        cpu_memories[idx].alocated_size=0;
      }
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
    CUDA_CHECK(cudaDeviceSynchronize());
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
    char *tmp_str;
    mpz_t mpz_data_size;
    mpz_init(mpz_data_size);
    cJSON *data_json = cJSON_CreateObject();
    
    mpz_set_ui(mpz_data_size, _content->size >> 32);
    mpz_mul_2exp(mpz_data_size, mpz_data_size, 32);
    mpz_add_ui(mpz_data_size, mpz_data_size, _content->size & 0xffffffff);
    tmp_str = mpz_get_str(NULL, 16, mpz_data_size);
    strcpy(hex_string+2, tmp_str);
    cJSON_AddStringToObject(data_json, "size", hex_string);
    free(tmp_str);

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
  memory_t  memory(arith, &(src_instances[instance]));
  memory.copy_content(&(dst_instances[instance]));
  memory.free_memory();
}

#endif