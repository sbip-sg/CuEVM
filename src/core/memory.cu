// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT
#include "include/memory.cuh"
#include "include/utils.cuh"
#include "include/error_codes.h"
#include "include/gas_cost.cuh"

namespace cuEVM {
  namespace memory {
    __global__ void transfer_kernel(
      memory_data_t *dst_instances,
      memory_data_t *src_instances,
      uint32_t instance_count
    ) {
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

    __host__ memory_data_t *get_cpu_instances(
        uint32_t count
    )
    {
        memory_data_t *instances;
        instances = new memory_data_t[count];
        memset(instances, 0, sizeof(memory_data_t) * count);
        return instances;
    }

    __host__ void free_cpu_instances(
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

    __host__ memory_data_t *get_gpu_instances_from_cpu_instances(
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

    __host__ void free_gpu_instances(
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

    __host__ memory_data_t *get_cpu_instances_from_gpu_instances(
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
      transfer_kernel<<<count, 1>>>(tmp_gpu_instances, gpu_instances, count);
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

    __host__ __device__ void print_memory_data_t(
      ArithEnv &arith,
      memory_data_t &memory_data
    )
    {
      printf("size=%lu\n", memory_data.size);
      printf("allocated_size=%lu\n", memory_data.allocated_size);
      printf("memory_cost=");
      memory_data.memory_cost.print();
      if (memory_data.size > 0)
        cuEVM::byte_array::print_bytes(memory_data.data, memory_data.size);
    }


    __host__ cJSON *json_from_memory_data_t(
      ArithEnv &arith,
      memory_data_t &memory_data
    )
    {
      cJSON *memory_json = cJSON_CreateObject();
      cJSON_AddNumberToObject(memory_json, "size", memory_data.size);
      cJSON_AddNumberToObject(memory_json, "allocated_size", memory_data.allocated_size);
      char *hex_string_ptr = new char[EVM_WORD_SIZE * 2 + 3];
      memory_data.memory_cost.to_hex(hex_string_ptr);
      cJSON_AddStringToObject(memory_json, "memory_cost", hex_string_ptr);
      if (memory_data.size > 0)
      {
        char *bytes_string = cuEVM::byte_array::hex_from_bytes(memory_data.data, memory_data.size);
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

    __host__ EVMMemory::EVMMemory(
      ArithEnv arith,
      memory_data_t *content
    ) : _arith(arith),
        _content(content)
    {
    }
    __host__ __device__ EVMMemory::EVMMemory(
      ArithEnv arith
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
      _content->memory_cost.from_uint32_t(0);
    }

    __host__ __device__ EVMMemory::~EVMMemory()
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
      _content->memory_cost.from_uint32_t(0);
      ONE_THREAD_PER_INSTANCE(
        delete _content;
      )
      _content = NULL;
    }

    __host__ __device__ size_t EVMMemory::size()
    {
      return _content->size;
    }

    __host__ __device__ void EVMMemory::allocate_pages(
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

    __host__ __device__ size_t EVMMemory::get_last_offset(
      bn_t &index,
      bn_t &length,
      uint32_t &error_code
    )
    {
      bn_t offset;
      int32_t overflow;
      size_t last_offset;
      // first overflow check
      overflow = cgbn_add(_arith.env, offset, index, length);
      overflow = overflow | _arith.size_t_from_cgbn(last_offset, offset);
      bn_t memory_size_word;
      cgbn_add_ui32(_arith.env, memory_size_word, offset, 31);
      cgbn_div_ui32(_arith.env, memory_size_word, memory_size_word, 32);
      cgbn_mul_ui32(_arith.env, offset, memory_size_word, 32);
      // get the new size
      overflow = overflow | _arith.size_t_from_cgbn(last_offset, offset);
      if (overflow != 0)
      {
        error_code = ERR_MEMORY_INVALID_OFFSET;
      }
      return last_offset;
    }

    __host__ __device__ void EVMMemory::grow(
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

    __host__ __device__ uint8_t* EVMMemory::get(
      bn_t &index,
      bn_t &length,
      uint32_t &error_code
    )
    {
      if (cgbn_compare_ui32(_arith.env, length, 0) > 0)
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

    __host__ __device__ void EVMMemory::set(
      uint8_t *data,
      bn_t &index,
      bn_t &length,
      size_t &available_size,
      uint32_t &error_code
    )
    {
      if (cgbn_compare_ui32(_arith.env, length, 0) > 0)
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

    __host__ __device__ void EVMMemory::to_memory_data_t(memory_data_t &dst)
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
          dst.memory_cost.from_uint32_t(0);
        }

        dst.size = _content->size;
        dst.allocated_size = _content->size;
        bn_t memory_cost;
        cgbn_load(_arith.env, memory_cost, &(_content->memory_cost));
        cgbn_store(_arith.env, &(dst.memory_cost), memory_cost);
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

    __host__ __device__ void EVMMemory::print()
    {
      print_memory_data_t(_arith, *_content);
    }

    __host__ cJSON* EVMMemory::json()
    {
      return json_from_memory_data_t(_arith, *_content);
    }

  }
}