#ifndef _GPU_STORAGE_H_
#define _GPU_STORAGE_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <gmp.h>
#ifndef __CGBN_H__
#define __CGBN_H__
#include <cgbn/cgbn.h>
#endif
#include "arith.cuh"

template<class params>
class gpu_fixed_storage_t {
  public:

  typedef typename arith_env_t<params>::bn_t      bn_t;
  

  //storage data structure  
  typedef struct {
    cgbn_mem_t<params::BITS> address[params::STORAGE_SIZE];
    cgbn_mem_t<params::BITS> key[params::STORAGE_SIZE];
    cgbn_mem_t<params::BITS> value[params::STORAGE_SIZE];
  } fixed_storage_data_t;


  //copntent of the storage
  fixed_storage_data_t *_content;
  size_t _size;
  arith_env_t<params>     _arith;
  
  
  //constructor
  __device__ __forceinline__ gpu_storage_t(arith_env_t<params> arith, fixed_storage_data_t *content, size_t size) : _arith(arith), _content(content), _size(size) {
  }


  //get the data of the memory at a specific index and length
  __device__ __forceinline__ uint32_t get(const bn_t &address, const bn_t &key, bn_t &value) {
    bn_t local_address, local_key;
    for (size_t idx; idx<_size, idx++) {
      cgbn_load(_arith._env, local_address, &(_content->address[idx]));
      if (cgbn_compare(_arith._env, local_address, address) == 0) {
        cgbn_load(_arith._env, local_key, &(_content->key[idx]));
        if (cgbn_compare(_arith._env, local_address, address) == 0) {
          cgbn_load(_arith._env, value, &(_content->value[idx]));
          return 0;
        }
      }
    }
    return 1;
  }

  //set the data of the memory at a specific index and length
  __device__ __forceinline__ uint32_t set(const bn_t &address, const bn_t &key, const bn_t &value) {
    bn_t local_address, local_key;
    for (size_t idx; idx<_size, idx++) {
      cgbn_load(_arith._env, local_address, &(_content->address[idx]));
      if (cgbn_compare(_arith._env, local_address, address) == 0) {
        cgbn_load(_arith._env, local_key, &(_content->key[idx]));
        if (cgbn_compare(_arith._env, local_address, address) == 0) {
          cgbn_store(_arith._env, &(_content->value[idx]), value);
          return 0;
        }
      }
    }
    if (_size == params::STORAGE_SIZE)
      return 1;
    cgbn_store(_arith._env, &(_content->address[_size]), address);
    cgbn_store(_arith._env, &(_content->key[_size]), key);
    cgbn_store(_arith._env, &(_content->value[_size]), value);
    _size++;
    return 0;
  }

  // generate the memory content structure info on the host
  __host__ static fixed_storage_data_t *generate_storage_data(uint32_t count) {
    fixed_storage_data_t *cpu_instances = (fixed_storage_data_t *)malloc(sizeof(fixed_storage_data_t) * count);
    for (uint32_t idx = 0; idx < count; idx++) {
      cpu_instances[idx]._size = 0;
    }
    return cpu_instances;
  }

  __host__ static fixed_storage_data_t *generate_gpu_storage_data(fixed_storage_data_t *cpu_instances, uint32_t count) {
    fixed_storage_data_t *gpu_instances;
    cudaMalloc((void **)&gpu_instances, sizeof(fixed_storage_data_t)*count);
    cudaMemcpy(gpu_instances, cpu_instances, sizeof(fixed_storage_data_t)*count, cudaMemcpyHostToDevice);
    return gpu_instances;
  }

  
  __device__ __forceinline__ void copy_storage_data(fixed_storage_data_t *dest) {
    __syncthreads();
    if (threadIdx.x == 0) {
      memcpy(dest, _content, sizeof(fixed_storage_data_t));
    }
    __syncthreads();
  }


  // free the storage structure info on the host
  __host__ static void free_storage_data(memory_data_t *cpu_instances, uint32_t count) {
    free(cpu_instances);
  }
  // free the gpu storage structure info on the host
  __host__ static void free_gpu_storage_data(memory_data_t *gpu_instances, uint32_t count) {
    cudaFree(gpu_instances);
  }

  __host__ static void write_storage(FILE *fp, memory_data_t *cpu_instances, uint32_t count) {
    for(uint32_t idx=0; idx<count; idx++) {
      fprintf(fp, "INSTACE: %08x , STORAGE_SIZE: %lx , STORAGE_DATA: ", idx, cpu_instances[idx]._size);
      for(uint32_t jdx=0; jdx<cpu_instances[idx]._size; jdx++) {
        fprintf(fp, "ADDRESS- ");
        for(uint32_t kdx=0; kdx<params::BITS/32; kdx++) {
          fprintf(fp, "%08x ", cpu_instances[idx]._data[jdx].address[kdx]);
        }
        fprintf(fp, "KEY- ");
        for(uint32_t kdx=0; kdx<params::BITS/32; kdx++) {
          fprintf(fp, "%08x ", cpu_instances[idx]._data[jdx].key[kdx]);
        }
        fprintf(fp, "VALUE- ");
        for(uint32_t kdx=0; kdx<params::BITS/32; kdx++) {
          fprintf(fp, "%08x ", cpu_instances[idx]._data[jdx].value[kdx]);
        }
        fprintf(fp, "; ");
      }
      fprintf(fp, "\n");
    }
  }

};

#endif