// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#ifndef _MEMORY_H_
#define _MEMORY_H_

#include "include/utils.h"

/**
 * The memory class (YP: \f$\mu_{m}\f$).
*/
template <class params>
class memory_t
{
public:
  /**
   * The arithmetical environment used by the arbitrary length
   * integer library.
  */
  typedef arith_env_t<params> arith_t;
  /**
   * The arbitrary length integer type.
  */
  typedef typename arith_t::bn_t bn_t;
  /**
   * The arbitrary length integer type used for the storage.
   * It is defined as the EVM word type.
  */
  typedef cgbn_mem_t<params::BITS> evm_word_t;
  static const size_t PAGE_SIZE = params::PAGE_SIZE;

  /**
   * The memory data structure.
  */
  typedef struct
  {
    size_t size; /**< The size of the memory acceesed by now (YP: \f$32 \dot \mu_{i}\f$)*/
    size_t allocated_size; /**< Internal allocated size do not confuse with size*/
    evm_word_t memory_cost; /**< The memory cost (YP: \f$M(\mu_{i})\f$)*/
    uint8_t *data; /**< The data of the memory*/
  } memory_data_t;

  memory_data_t *_content; /**< The memory data structure*/
  arith_t _arith; /**< The arithmetical environment*/

  /**
   * The constructor of the memory class given the memory data structure.
   * @param[in] arith The arithmetical environment.
   * @param[in] content The memory data structure.
  */
  __host__ __forceinline__ memory_t(
    arith_t arith,
    memory_data_t *content
  ) : _arith(arith),
      _content(content)
  {
  }

  /**
   * The constructor of the memory class given the arithmetical environment.
   * @param arith[in] The arithmetical environment.
  */
  __host__ __device__ __forceinline__ memory_t(
    arith_t arith
  ) : _arith(arith)
  {
    SHARED_MEMORY memory_data_t *content;
    ONE_THREAD_PER_INSTANCE(
      content = new memory_data_t;
      content->size = 0;
      content->allocated_size = 0;
      content->data = NULL;
    )
    _content = content;
    _arith.cgbn_memory_from_size_t(_content->memory_cost, 0);
  }

  /**
   * The destructor of the memory class.
   * Frees the allcoated data and the memory data structure.
  */
  __host__ __device__ __forceinline__ ~memory_t()
  {
    if ((_content->allocated_size > 0) && (_content->data != NULL))
    {
      ONE_THREAD_PER_INSTANCE(
        delete[] _content->data;
      )
      _content->allocated_size = 0;
      _content->size = 0;
      _content->data = NULL;
    }
    _arith.cgbn_memory_from_size_t(_content->memory_cost, 0);
    ONE_THREAD_PER_INSTANCE(
      delete _content;
    )
    _content = NULL;
  }

  /**
   * The size of the memory. (Last offset reached by the memory access) x 32
   * (YP: \f$32 \dot \mu_{i}\f$)
   * @return The size of the memory.
  */
  __host__ __device__ __forceinline__ size_t size()
  {
    return _content->size;
  }

  /**
   * Allocates pages for the given offset if needed.
   * @param[in] new_size The new size of the memory.
   * @param[out] error_code The error code.
  */
  __host__ __device__ __forceinline__ void allocate_pages(
    size_t new_size,
    uint32_t &error_code
  )
  {
    if (new_size <= _content->allocated_size)
    {
      return;
    }
    size_t no_pages = (new_size / PAGE_SIZE) + 1;
    SHARED_MEMORY uint8_t *new_data;
    ONE_THREAD_PER_INSTANCE(
      new_data = new uint8_t[no_pages * PAGE_SIZE];
      if (new_data == NULL) {
        error_code = ERR_MEMORY_INVALID_ALLOCATION;
        return;
      }
      // 0 all the data
      memset(new_data, 0, no_pages * PAGE_SIZE);
      if (
        (_content->allocated_size > 0) &&
        (_content->data != NULL)
      )
      {
        memcpy(new_data, _content->data, _content->allocated_size);
        delete[] _content->data;
        _content->data = NULL;
        _content->allocated_size = 0;
      }
      _content->allocated_size = no_pages * PAGE_SIZE;
      _content->data = new_data;
    )
  }

  /**
   * The memory cost of the memory. (YP: \f$C_{mem}-M(\mu_{i}, index, length)\f$)
   * @param[in] index The index of the memory access.
   * @param[in] length The length of the memory access.
   * @param[out] gas_used The gas cost after.
   * @param[out] error_code The error code.
  */
  __host__ __device__ __forceinline__ void grow_cost(
    bn_t &index,
    bn_t &length,
    bn_t &gas_used,
    uint32_t &error_code
  )
  {
    //
    if (cgbn_compare_ui32(_arith._env, length, 0) > 0)
    {
      bn_t offset;
      int32_t overflow;
      size_t last_offset;
      // first overflow check
      overflow = cgbn_add(_arith._env, offset, index, length);
      overflow = overflow | _arith.size_t_from_cgbn(last_offset, offset);

      bn_t old_memory_cost;
      cgbn_load(_arith._env, old_memory_cost, &(_content->memory_cost));
      // memort_size_word = (offset + 31) / 32
      bn_t memory_size_word;
      cgbn_add_ui32(_arith._env, memory_size_word, offset, 31);
      cgbn_div_ui32(_arith._env, memory_size_word, memory_size_word, 32);
      // memory_cost = (memory_size_word * memory_size_word) / 512 + 3 * memory_size_word
      bn_t memory_cost;
      cgbn_mul(_arith._env, memory_cost, memory_size_word, memory_size_word);
      cgbn_div_ui32(_arith._env, memory_cost, memory_cost, 512);
      bn_t tmp;
      cgbn_mul_ui32(_arith._env, tmp, memory_size_word, GAS_MEMORY);
      cgbn_add(_arith._env, memory_cost, memory_cost, tmp);
      //  gas_used = gas_used + memory_cost - old_memory_cost
      bn_t memory_expansion_cost;
      if (cgbn_compare(_arith._env, memory_cost, old_memory_cost) == 1)
      {
        cgbn_sub(_arith._env, memory_expansion_cost, memory_cost, old_memory_cost);
        // set the new memory cost
        cgbn_store(_arith._env, &(_content->memory_cost), memory_cost);
      }
      else
      {
        cgbn_set_ui32(_arith._env, memory_expansion_cost, 0);
      }
      cgbn_add(_arith._env, gas_used, gas_used, memory_expansion_cost);

      // size is always a multiple of 32
      cgbn_mul_ui32(_arith._env, offset, memory_size_word, 32);
      // get the new size
      overflow = overflow | _arith.size_t_from_cgbn(last_offset, offset);
      if (overflow != 0)
      {
        error_code = ERR_MEMORY_INVALID_OFFSET;
      }
    }
  }

  /**
   * Get the highest offset of the memory access.
   * @param[in] index The index of the memory access.
   * @param[in] length The length of the memory access.
   * @param[out] error_code The error code.
  */
  __host__ __device__ __forceinline__ size_t get_last_offset(
    bn_t &index,
    bn_t &length,
    uint32_t &error_code
  )
  {
    bn_t offset;
    int32_t overflow;
    size_t last_offset;
    // first overflow check
    overflow = cgbn_add(_arith._env, offset, index, length);
    overflow = overflow | _arith.size_t_from_cgbn(last_offset, offset);
    bn_t memory_size_word;
    cgbn_add_ui32(_arith._env, memory_size_word, offset, 31);
    cgbn_div_ui32(_arith._env, memory_size_word, memory_size_word, 32);
    cgbn_mul_ui32(_arith._env, offset, memory_size_word, 32);
    // get the new size
    overflow = overflow | _arith.size_t_from_cgbn(last_offset, offset);
    if (overflow != 0)
    {
      error_code = ERR_MEMORY_INVALID_OFFSET;
    }
    return last_offset;
  }

  /**
   * Increase the memory for the given offset if needed.
   * Offset is computed from the index and the length.
   * @param[in] index The index of the memory access.
   * @param[in] length The length of the memory access.
   * @param[out] error_code The error code.
  */
  __host__ __device__ __forceinline__ void grow(
    bn_t &index,
    bn_t &length,
    uint32_t &error_code
  )
  {
    size_t offset = get_last_offset(index, length, error_code);
    if (
      (error_code == ERR_NONE) &&
      (offset > _content->size)
    )
    {
      if (offset > _content->allocated_size)
      {
        allocate_pages(offset, error_code);
      }
      _content->size = offset;
    }
  }

  /**
   * Get the a pointer to the given memory data.
   * @param[in] index The index of the memory access.
   * @param[in] length The length of the memory access.
   * @param[out] error_code The error code.
  */
  __host__ __device__ __forceinline__ uint8_t *get(
    bn_t &index,
    bn_t &length,
    uint32_t &error_code
  )
  {
    if (cgbn_compare_ui32(_arith._env, length, 0) > 0)
    {
      grow(index, length, error_code);
      size_t index_s;
      if (error_code == ERR_NONE)
      {
        _arith.size_t_from_cgbn(index_s, index);
        return _content->data + index_s;
      }
    }
    return NULL;
  }


  /**
   * Set the given memory data.
   * @param[in] data The data to be set.
   * @param[in] index The index of the memory access.
   * @param[in] length The length of the memory access.
   * @param[out] error_code The error code.
  */
  __host__ __device__ __forceinline__ void set(
    uint8_t *data,
    bn_t &index,
    bn_t &length,
    size_t &available_size,
    uint32_t &error_code
  )
  {
    if (cgbn_compare_ui32(_arith._env, length, 0) > 0)
    {
      size_t index_s;
      grow(index, length, error_code);
      _arith.size_t_from_cgbn(index_s, index);
      if (
        (data != NULL) &&
        (available_size > 0) &&
        (error_code == ERR_NONE)
      )
      {
        ONE_THREAD_PER_INSTANCE(
          memcpy(_content->data + index_s, data, available_size);
        )
      }
    }
  }

  /**
   * Copy the current memory data to the given memory data structure.
   * @param[out] dst The destination memory data structure.
  */
  __host__ __device__ __forceinline__ void to_memory_data_t(memory_data_t &dst)
  {
      // free if any memory is allocated
      if (
        (dst.allocated_size > 0) &&
        (dst.data != NULL)
      )
      {
        ONE_THREAD_PER_INSTANCE(
            delete[] dst.data;
        )
        dst.data = NULL;
        dst.allocated_size = 0;
        dst.size = 0;
        _arith.cgbn_memory_from_size_t(dst.memory_cost, 0);
      }

      dst.size = _content->size;
      dst.allocated_size = _content->size;
      bn_t memory_cost;
      cgbn_load(_arith._env, memory_cost, &(_content->memory_cost));
      cgbn_store(_arith._env, &(dst.memory_cost), memory_cost);
      if (_content->size > 0)
      {
        ONE_THREAD_PER_INSTANCE(
            dst.data = new uint8_t[_content->size];
            memcpy(dst.data, _content->data, _content->size);
        )
      }
      else
      {
        dst.data = NULL;
      }
  }
  
  /**
   * Allocate the CPU memory data structures for the given number of instances.
   * @param[in] count The number of instances.
   * @return The CPU memory data structures.
  */
  __host__ static memory_data_t *get_cpu_instances(
      uint32_t count
  )
  {
      memory_data_t *instances;
      instances = new memory_data_t[count];
      memset(instances, 0, sizeof(memory_data_t) * count);
      return instances;
  }

  /**
   * Free the CPU memory data structures for the given number of instances.
   * @param[in] instances The CPU memory data structures.
   * @param[in] count The number of instances.
  */
  __host__ static void free_cpu_instances(
      memory_data_t *instances,
      uint32_t count
  )
  {
    for (uint32_t idx = 0; idx < count; idx++)
    {
      if (
        (instances[idx].data != NULL) &&
        (instances[idx].allocated_size > 0)
      )
      {
        delete[] instances[idx].data;
        instances[idx].data = NULL;
        instances[idx].allocated_size = 0;
      }
    }
    delete[] instances;
  }

  /**
   * Allocate the GPU memory data structures for the given number of instances.
   * @param[in] cpu_instances The CPU memory data structures.
   * @param[in] count The number of instances.
   * @return The GPU memory data structures.
  */
  __host__ static memory_data_t *get_gpu_instances_from_cpu_instances(
      memory_data_t *cpu_instances,
      uint32_t count
  )
  {
      memory_data_t *gpu_instances;
      memory_data_t *tmp_cpu_instances;
      tmp_cpu_instances = new memory_data_t[count];
      memcpy(tmp_cpu_instances, cpu_instances, sizeof(memory_data_t) * count);
      for (size_t idx = 0; idx < count; idx++)
      {
        if (
          (tmp_cpu_instances[idx].allocated_size > 0) &&
          (tmp_cpu_instances[idx].data != NULL)
        )
        {
          CUDA_CHECK(cudaMalloc(
            (void **)&tmp_cpu_instances[idx].data,
            sizeof(uint8_t) * tmp_cpu_instances[idx].allocated_size
          ));
          CUDA_CHECK(cudaMemcpy(
            tmp_cpu_instances[idx].data,
            cpu_instances[idx].data,
            sizeof(uint8_t) * tmp_cpu_instances[idx].allocated_size,
            cudaMemcpyHostToDevice
          ));
        }
        else
        {
          tmp_cpu_instances[idx].data = NULL;
        }
      }
      CUDA_CHECK(cudaMalloc(
        (void **)&gpu_instances,
        sizeof(memory_data_t) * count
      ));
      CUDA_CHECK(cudaMemcpy(
        gpu_instances,
        tmp_cpu_instances,
        sizeof(memory_data_t) * count, cudaMemcpyHostToDevice
      ));
      delete[] tmp_cpu_instances;
      tmp_cpu_instances = NULL;
      return gpu_instances;
  }

  /**
   * Free the GPU memory data structures for the given number of instances.
   * @param[in] gpu_instances The GPU memory data structures.
   * @param[in] count The number of instances.
  */
  __host__ static void free_gpu_instances(
      memory_data_t *gpu_instances,
      uint32_t count
  )
  {
      memory_data_t *tmp_cpu_instances;
      tmp_cpu_instances = new memory_data_t[count];
      CUDA_CHECK(cudaMemcpy(
        tmp_cpu_instances,
        gpu_instances,
        sizeof(memory_data_t) * count, cudaMemcpyDeviceToHost
      ));
      for (size_t idx = 0; idx < count; idx++)
      {
        if (
          (tmp_cpu_instances[idx].allocated_size > 0) &&
          (tmp_cpu_instances[idx].data != NULL)
        )
        {
          CUDA_CHECK(cudaFree(tmp_cpu_instances[idx].data));
        }
      }
      CUDA_CHECK(cudaFree(gpu_instances));
      delete[] tmp_cpu_instances;
      tmp_cpu_instances = NULL;
  }

  /**
   * Copy the GPU memory data structures to the CPU memory data structures.
   * @param[in] gpu_instances The GPU memory data structures.
   * @param[in] count The number of instances.
   * @return The CPU memory data structures.
  */
  __host__ static memory_data_t *get_cpu_instances_from_gpu_instances(
    memory_data_t *gpu_instances,
    uint32_t count
  )
  {
    memory_data_t *cpu_instances;
    cpu_instances = new memory_data_t[count];
    CUDA_CHECK(cudaMemcpy(
      cpu_instances,
      gpu_instances,
      sizeof(memory_data_t) * count,
      cudaMemcpyDeviceToHost
    ));

    // 1. alocate the memory for gpu memory as memory which can be addressed by the cpu
    memory_data_t *tmp_cpu_instances, *tmp_gpu_instances;
    tmp_cpu_instances = new memory_data_t[count];
    memcpy(
      tmp_cpu_instances,
      cpu_instances,
      sizeof(memory_data_t) * count
    );
    for (size_t idx = 0; idx < count; idx++)
    {
      if (tmp_cpu_instances[idx].size > 0)
      {
        CUDA_CHECK(cudaMalloc(
          (void **)&tmp_cpu_instances[idx].data,
          sizeof(uint8_t) * tmp_cpu_instances[idx].size
        ));
      }
      else
      {
        tmp_cpu_instances[idx].data = NULL;
      }
      tmp_cpu_instances[idx].allocated_size = tmp_cpu_instances[idx].size;
    }
    CUDA_CHECK(cudaMalloc(
      (void **)&tmp_gpu_instances,
      sizeof(memory_data_t) * count
    ));
    CUDA_CHECK(cudaMemcpy(
      tmp_gpu_instances,
      tmp_cpu_instances,
      sizeof(memory_data_t) * count,
      cudaMemcpyHostToDevice
    ));
    delete[] tmp_cpu_instances;
    tmp_cpu_instances = NULL;

    // 2. call the kernel to copy the memory between the gpu memories
    kernel_get_memory<params><<<count, 1>>>(tmp_gpu_instances, gpu_instances, count);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(gpu_instances));
    gpu_instances = tmp_gpu_instances;

    // 3. copy the gpu memories back in the cpu memories
    CUDA_CHECK(cudaMemcpy(
      cpu_instances,
      gpu_instances,
      sizeof(memory_data_t) * count,
      cudaMemcpyDeviceToHost
    ));
    tmp_cpu_instances = new memory_data_t[count];
    memcpy(
      tmp_cpu_instances,
      cpu_instances,
      sizeof(memory_data_t) * count
    );
    for (size_t idx = 0; idx < count; idx++)
    {
      if (tmp_cpu_instances[idx].size > 0)
      {
        tmp_cpu_instances[idx].data = new uint8_t[tmp_cpu_instances[idx].size];
        CUDA_CHECK(cudaMemcpy(
          tmp_cpu_instances[idx].data,
          cpu_instances[idx].data,
          sizeof(uint8_t) * tmp_cpu_instances[idx].size,
          cudaMemcpyDeviceToHost
        ));
      }
      else
      {
        tmp_cpu_instances[idx].data = NULL;
      }
    }
    free_gpu_instances(gpu_instances, count);
    memcpy(
      cpu_instances,
      tmp_cpu_instances,
      sizeof(memory_data_t) * count
    );
    delete[] tmp_cpu_instances;
    tmp_cpu_instances = NULL;
    return cpu_instances;
  }

  /**
   * Print the memory data structure.
   * @param[in] arith The arithmetical environment.
   * @param[in] memory_data The memory data structure.
  */
  __host__ __device__ __forceinline__ static void print_memory_data_t(
    arith_t &arith,
    memory_data_t &memory_data
  )
  {
    printf("size=%lu\n", memory_data.size);
    printf("allocated_size=%lu\n", memory_data.allocated_size);
    printf("memory_cost=");
    arith.print_cgbn_memory(memory_data.memory_cost);
    if (memory_data.size > 0)
      print_bytes(memory_data.data, memory_data.size);
  }

  /**
   * Print the memory data structure.
  */
  __host__ __device__ void print()
  {
    print_memory_data_t(_arith, *_content);
  }

  __host__ static cJSON *json_from_memory_data_t(
    arith_t &arith,
    memory_data_t &memory_data
  )
  {
    cJSON *memory_json = cJSON_CreateObject();
    cJSON_AddNumberToObject(memory_json, "size", memory_data.size);
    cJSON_AddNumberToObject(memory_json, "allocated_size", memory_data.allocated_size);
    char *hex_string_ptr = new char[arith_t::BYTES * 2 + 3];
    arith.hex_string_from_cgbn_memory(hex_string_ptr, memory_data.memory_cost);
    cJSON_AddStringToObject(memory_json, "memory_cost", hex_string_ptr);
    if (memory_data.size > 0)
    {
      char *bytes_string = hex_from_bytes(memory_data.data, memory_data.size);
      cJSON_AddStringToObject(memory_json, "data", bytes_string);
      delete[] bytes_string;
    }
    else
    {
      cJSON_AddStringToObject(memory_json, "data", "0x");
    }
    delete[] hex_string_ptr;
    hex_string_ptr = NULL;
    return memory_json;
  }

  /**
   * Get the memory data structure as a JSON object.
   * @return The JSON object.
  */
  __host__ cJSON *json()
  {
    return json_from_memory_data_t(_arith, *_content);
  }
};

/**
 * The kernel to copy the memory data structures between the GPU memories.
*/
template <class params>
__global__ void kernel_get_memory(
  typename memory_t<params>::memory_data_t *dst_instances,
  typename memory_t<params>::memory_data_t *src_instances,
  uint32_t instance_count
)
{
  uint32_t instance = blockIdx.x * blockDim.x + threadIdx.x;
  if (instance >= instance_count)
    return;


  if (src_instances[instance].size > 0)
  {
    memcpy(
      dst_instances[instance].data,
      src_instances[instance].data,
      src_instances[instance].size * sizeof(uint8_t)
    );
    delete[] src_instances[instance].data;
    src_instances[instance].data = NULL;
  }
}

#endif