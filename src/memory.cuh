#ifndef _MEMORY_H_
#define _MEMORY_H_

#include "utils.h"

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

  // memory data structure
  typedef struct
  {
    size_t size;
    size_t alocated_size;
    evm_word_t memory_cost;
    uint8_t *data;
  } memory_data_t;

  // content of the memory
  memory_data_t *_content;
  arith_t _arith;

  // constructor
  __host__ __forceinline__ memory_t(
    arith_t arith,
    memory_data_t *content
  ) : _arith(arith),
      _content(content)
  {
  }

  __host__ __device__ __forceinline__ memory_t(
    arith_t arith
  ) : _arith(arith)
  {
    SHARED_MEMORY memory_data_t *content;
    ONE_THREAD_PER_INSTANCE(
      content = new memory_data_t;
      content->size = 0;
      content->alocated_size = 0;
      content->data = NULL;
    )
    _content = content;
    _arith.cgbn_memory_from_size_t(_content->memory_cost, 0);
  }

  __host__ __device__ __forceinline__ ~memory_t()
  {
    if ((_content->alocated_size > 0) && (_content->data != NULL))
    {
      ONE_THREAD_PER_INSTANCE(
        delete[] _content->data;
      )
      _content->alocated_size = 0;
      _content->size = 0;
      _content->data = NULL;
      _arith.cgbn_memory_from_size_t(_content->memory_cost, 0);
    }
    ONE_THREAD_PER_INSTANCE(
      delete _content;
    )
    _content = NULL;
  }

  // get the size of the memory
  __host__ __device__ __forceinline__ size_t size()
  {
    return _content->size;
  }

  __host__ __device__ __forceinline__ void allocate_pages(
    size_t new_size,
    uint32_t &error_code
  )
  {
    if (new_size <= _content->alocated_size)
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
        (_content->alocated_size > 0) &&
        (_content->data != NULL)
      )
      {
        memcpy(new_data, _content->data, _content->size);
        delete[] _content->data;
        _content->data = NULL;
        _content->alocated_size = 0;
      }
      _content->alocated_size = no_pages * PAGE_SIZE;
      _content->data = new_data;
    )
  }

  __host__ __device__ __forceinline__ void grow_cost(
    bn_t &index,
    bn_t &length,
    bn_t &gas_cost,
    uint32_t &error_code
  )
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
    //  gas_cost = gas_cost + memory_cost - old_memory_cost
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
    cgbn_add(_arith._env, gas_cost, gas_cost, memory_expansion_cost);

    // size is always a multiple of 32
    cgbn_mul_ui32(_arith._env, offset, memory_size_word, 32);
    // get the new size
    overflow = overflow | _arith.size_t_from_cgbn(last_offset, offset);
    if (overflow != 0)
    {
      error_code = ERR_MEMORY_INVALID_OFFSET;
    }
  }

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
      _content->size = offset;
      if (offset > _content->alocated_size)
      {
        allocate_pages(offset, error_code);
      }
    }
  }

  __host__ __device__ __forceinline__ uint8_t *get(
    bn_t &index,
    bn_t &length,
    uint32_t &error_code
  )
  {
    grow(index, length, error_code);
    size_t index_s;
    if (error_code == ERR_NONE)
    {
      _arith.size_t_from_cgbn(index_s, index);
      return _content->data + index_s;
    }
    else
    {
      return NULL;
    }
  }


  __host__ __device__ __forceinline__ void set(
    uint8_t *data,
    bn_t &index,
    bn_t &length,
    uint32_t &error_code
  )
  {
    size_t index_s;
    size_t length_s;
    grow(index, length, error_code);
    _arith.size_t_from_cgbn(index_s, index);
    _arith.size_t_from_cgbn(length_s, length);
    if (
      (data != NULL) &&
      (length_s > 0) &&
      (error_code == ERR_NONE)
    )
    {
      ONE_THREAD_PER_INSTANCE(
        memcpy(_content->data + index_s, data, length_s);
      )
    }
  }

  __host__ __device__ __forceinline__ void to_memory_data_t(memory_data_t *dest)
  {
      // free if any memory is allocated
      if (
        (dest->alocated_size > 0) &&
        (dest->data != NULL)
      )
      {
        ONE_THREAD_PER_INSTANCE(
            delete[] dest->data;
        )
        dest->data = NULL;
        dest->alocated_size = 0;
        dest->size = 0;
        _arith.cgbn_memory_from_size_t(dest->memory_cost, 0);
      }

      dest->size = _content->size;
      dest->alocated_size = _content->size;
      bn_t memory_cost;
      cgbn_load(_arith._env, memory_cost, &(_content->memory_cost));
      cgbn_store(_arith._env, &(dest->memory_cost), memory_cost);
      if (_content->size > 0)
      {
        ONE_THREAD_PER_INSTANCE(
            dest->data = new uint8_t[_content->size];
            memcpy(dest->data, _content->data, _content->size);
        )
      }
      else
      {
        dest->data = NULL;
      }
  }
  
    __host__ static memory_data_t *get_cpu_instances(
        uint32_t count
    )
    {
        memory_data_t *instances;
        instances = new memory_data_t[count];
        memset(instances, 0, sizeof(memory_data_t) * count);
        return instances;
    }

    __host__ static free_cpu_instances(
        memory_data_t *instances,
        uint32_t count
    )
    {
      for (uint32_t idx = 0; idx < count; idx++)
      {
        if (
          (instances[idx].data != NULL) &&
          (instances[idx].alocated_size > 0)
        )
        {
          delete[] instances[idx].data;
          instances[idx].data = NULL;
          instances[idx].alocated_size = 0;
        }
      }
      delete[] instances;
    }

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
            (tmp_cpu_instances[idx].alocated_size > 0) &&
            (tmp_cpu_instances[idx].data != NULL)
          )
          {
            CUDA_CHECK(cudaMalloc(
              (void **)&tmp_cpu_instances[idx].data,
              sizeof(uint8_t) * tmp_cpu_instances[idx].alocated_size
            ));
            CUDA_CHECK(cudaMemcpy(
              tmp_cpu_instances[idx].data,
              cpu_instances[idx].data,
              sizeof(uint8_t) * tmp_cpu_instances[idx].alocated_size,
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
            (tmp_cpu_instances[idx].alocated_size > 0) &&
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
        if (tmp_cpu_memories[idx].size > 0)
        {
          CUDA_CHECK(cudaMalloc(
            (void **)&tmp_cpu_memories[idx].data,
            sizeof(uint8_t) * tmp_cpu_memories[idx].size
          ));
        }
        else
        {
          tmp_cpu_memories[idx].data = NULL;
        }
        tmp_cpu_memories[idx].alocated_size = tmp_cpu_memories[idx].size;
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
      kernel_get_memory<params><<<1, count>>>(tmp_gpu_instances, gpu_instances, count);
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

  __host__ __device__ void print()
  {
    printf("size=%lu\n", _content->size);
    printf("alocated_size=%lu\n", _content->alocated_size);
    printf("memory_cost=");
    _arith.print_cgbn_memory(_content->memory_cost);
    printf("data: ");
    if (_content->size > 0)
      print_bytes(_content->data, _content->size);
    printf("\n");
  }

  __host__ cJSON *json()
  {
    char *bytes_string=NULL;
    cJSON *data_json = cJSON_CreateObject();

    if (_content->size > 0) {
      bytes_string = bytes_to_hex(_content->data, _content->size);
      cJSON_AddStringToObject(data_json, "data", bytes_string);
      free(bytes_string);
    } else {
      cJSON_AddStringToObject(data_json, "data", "0x");
    }
    
    return data_json;
  }
};

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
  }
}

#endif