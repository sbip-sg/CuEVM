// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2024-06-21
// SPDX-License-Identifier: MIT


#include "include/stack.cuh"
#include "include/utils.cuh"
#include "include/evm_defines.h"
#include "include/error_codes.h"

namespace cuEVM {
  namespace stack {
    __global__ void transfer_kernel(
      stack_data_t *dst,
      stack_data_t *src,
      uint32_t count)
    {
      uint32_t instance = blockIdx.x * blockDim.x + threadIdx.x;
      if (instance >= count)
      {
        return;
      }
      dst[instance].stack_offset = src[instance].stack_offset;
      if (dst[instance].stack_offset > 0)
      {
        memcpy(
            dst[instance].stack_base,
            src[instance].stack_base,
            sizeof(evm_word_t) * src[instance].stack_offset);
        delete[] src[instance].stack_base;
        src[instance].stack_base = NULL;
        src[instance].stack_offset = 0;
      }
    }

    __host__ stack_data_t *get_cpu_instances(
        uint32_t count)
    {
      stack_data_t *cpu_instances = new stack_data_t[count];
      for (uint32_t idx = 0; idx < count; idx++)
      {
        cpu_instances[idx].stack_base = NULL;
        cpu_instances[idx].stack_offset = 0;
      }
      return cpu_instances;
    }

    __host__ void free_cpu_instances(
        stack_data_t *cpu_instances,
        uint32_t count)
    {
      for (int index = 0; index < count; index++)
      {
        if (cpu_instances[index].stack_base != NULL)
        {
          delete[] cpu_instances[index].stack_base;
          cpu_instances[index].stack_base = NULL;
        }
        cpu_instances[index].stack_offset = 0;
      }
      delete[] cpu_instances;
    }

    __host__ stack_data_t *get_gpu_instances_from_cpu_instances(
        stack_data_t *cpu_instances,
        uint32_t count)
    {
      stack_data_t *gpu_instances, *tmp_cpu_instances;
      tmp_cpu_instances = new stack_data_t[count];
      memcpy(
          tmp_cpu_instances,
          cpu_instances,
          sizeof(stack_data_t) * count);
      for (uint32_t idx = 0; idx < count; idx++)
      {
        if (cpu_instances[idx].stack_offset > 0)
        {
          CUDA_CHECK(cudaMalloc(
              (void **)&tmp_cpu_instances[idx].stack_base,
              sizeof(evm_word_t) * cpu_instances[idx].stack_offset));
          CUDA_CHECK(cudaMemcpy(
              tmp_cpu_instances[idx].stack_base,
              cpu_instances[idx].stack_base,
              sizeof(evm_word_t) * cpu_instances[idx].stack_offset,
              cudaMemcpyHostToDevice));
        }
        else
        {
          tmp_cpu_instances[idx].stack_base = NULL;
        }
      }
      CUDA_CHECK(cudaMalloc(
          (void **)&gpu_instances,
          sizeof(stack_data_t) * count));
      CUDA_CHECK(cudaMemcpy(
          gpu_instances,
          tmp_cpu_instances,
          sizeof(stack_data_t) * count,
          cudaMemcpyHostToDevice));
      delete[] tmp_cpu_instances;
      tmp_cpu_instances = NULL;
      return gpu_instances;
    }

    __host__ void free_gpu_instances(
        stack_data_t *gpu_instances,
        uint32_t count)
    {
      stack_data_t *tmp_cpu_instances;
      tmp_cpu_instances = new stack_data_t[count];
      CUDA_CHECK(cudaMemcpy(
          tmp_cpu_instances,
          gpu_instances,
          sizeof(stack_data_t) * count,
          cudaMemcpyDeviceToHost));
      for (uint32_t idx = 0; idx < count; idx++)
      {
        if (tmp_cpu_instances[idx].stack_base != NULL)
          CUDA_CHECK(cudaFree(tmp_cpu_instances[idx].stack_base));
      }
      delete[] tmp_cpu_instances;
      CUDA_CHECK(cudaFree(gpu_instances));
    }

    __host__ stack_data_t *get_cpu_instances_from_gpu_instances(
        stack_data_t *gpu_instances,
        uint32_t count)
    {
      stack_data_t *cpu_instances, *tmp_cpu_instances, *tmp_gpu_instances;
      cpu_instances = new stack_data_t[count];
      tmp_cpu_instances = new stack_data_t[count];
      CUDA_CHECK(cudaMemcpy(
          cpu_instances,
          gpu_instances,
          sizeof(stack_data_t) * count,
          cudaMemcpyDeviceToHost));
      memcpy(
          tmp_cpu_instances,
          cpu_instances,
          sizeof(stack_data_t) * count);
      for (uint32_t idx = 0; idx < count; idx++)
      {
        tmp_cpu_instances[idx].stack_offset = cpu_instances[idx].stack_offset;
        if (cpu_instances[idx].stack_offset > 0)
        {
          CUDA_CHECK(cudaMalloc(
              (void **)&tmp_cpu_instances[idx].stack_base,
              sizeof(evm_word_t) * cpu_instances[idx].stack_offset));
        }
        else
        {
          tmp_cpu_instances[idx].stack_base = NULL;
        }
      }
      CUDA_CHECK(cudaMalloc(
          (void **)&tmp_gpu_instances,
          sizeof(stack_data_t) * count));
      CUDA_CHECK(cudaMemcpy(
          tmp_gpu_instances,
          tmp_cpu_instances,
          sizeof(stack_data_t) * count,
          cudaMemcpyHostToDevice));
      delete[] tmp_cpu_instances;
      tmp_cpu_instances = NULL;
      transfer_kernel<<<count, 1>>>(
          tmp_gpu_instances,
          gpu_instances,
          count);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaFree(gpu_instances));
      gpu_instances = tmp_gpu_instances;

      CUDA_CHECK(cudaMemcpy(
          cpu_instances,
          gpu_instances,
          sizeof(stack_data_t) * count,
          cudaMemcpyDeviceToHost));
      tmp_cpu_instances = new stack_data_t[count];
      memcpy(
          tmp_cpu_instances,
          cpu_instances,
          sizeof(stack_data_t) * count);

      for (uint32_t idx = 0; idx < count; idx++)
      {
        tmp_cpu_instances[idx].stack_offset = cpu_instances[idx].stack_offset;
        if (cpu_instances[idx].stack_offset > 0)
        {
          tmp_cpu_instances[idx].stack_base = new evm_word_t[cpu_instances[idx].stack_offset];
          CUDA_CHECK(cudaMemcpy(
              tmp_cpu_instances[idx].stack_base,
              cpu_instances[idx].stack_base,
              sizeof(evm_word_t) * cpu_instances[idx].stack_offset,
              cudaMemcpyDeviceToHost));
        }
        else
        {
          tmp_cpu_instances[idx].stack_base = NULL;
        }
      }

      memcpy(
          cpu_instances,
          tmp_cpu_instances,
          sizeof(stack_data_t) * count);
      delete[] tmp_cpu_instances;
      tmp_cpu_instances = NULL;
      free_gpu_instances(gpu_instances, count);
      return cpu_instances;
    }

    __host__ __device__ void print_stack_data_t(
        ArithEnv &arith,
        stack_data_t &stack_data)
    {
      printf("Stack size: %d, data:\n", stack_data.stack_offset);
      for (uint32_t idx = 0; idx < stack_data.stack_offset; idx++)
      {
        stack_data.stack_base[idx].print();
      }
    }

    __host__ cJSON *json_from_stack_data_t(
        ArithEnv &arith,
        stack_data_t &stack_data)
    {
      char *hex_string_ptr = new char[EVM_WORD_SIZE * 2 + 3];
      cJSON *stack_json = cJSON_CreateObject();

      cJSON *stack_data_json = cJSON_CreateArray();
      for (uint32_t idx = 0; idx < stack_data.stack_offset; idx++)
      {
        stack_data.stack_base[idx].to_hex(hex_string_ptr);
        cJSON_AddItemToArray(stack_data_json, cJSON_CreateString(hex_string_ptr));
      }
      cJSON_AddItemToObject(stack_json, "data", stack_data_json);
      delete[] hex_string_ptr;
      return stack_json;
    }

    __host__ __device__ EVMStack::EVMStack(
          ArithEnv arith,
          stack_data_t *content) : _arith(arith),
                                  _content(content)
    {
    }

    __host__ __device__ EVMStack::EVMStack(
        ArithEnv arith) : _arith(arith)
    {
      SHARED_MEMORY stack_data_t *content;
      ONE_THREAD_PER_INSTANCE(
          content = new stack_data_t;
          content->stack_base = new evm_word_t[EVM_MAX_STACK_SIZE];
          content->stack_offset = 0;)
      _content = content;
    }
      
    __host__ __device__ EVMStack::~EVMStack()
    {
      ONE_THREAD_PER_INSTANCE(
          if (_content->stack_base != NULL) {
            delete[] _content->stack_base;
            _content->stack_base = NULL;
            _content->stack_offset = 0;
          } if (_content != NULL) {
            delete _content;
          })
      _content = NULL;
    }

    __host__ __device__ uint32_t EVMStack::size()
    {
      return _content->stack_offset;
    }

    __host__ __device__ evm_word_t* EVMStack::top()
    {
      return _content->stack_base + _content->stack_offset;
    }

    __host__ __device__ void EVMStack::push(const bn_t &value, uint32_t &error_code)
    {
      if (size() >= EVM_MAX_STACK_SIZE)
      {
        error_code = ERR_STACK_OVERFLOW;
        return;
      }
      cgbn_store(_arith.env, top(), value);
      _content->stack_offset++;
    }

    __host__ __device__ void EVMStack::pop(bn_t &y, uint32_t &error_code)
    {
      if (size() == 0)
      {
        error_code = ERR_STACK_UNDERFLOW;
        cgbn_set_ui32(_arith.env, y, 0);
        return;
      }
      _content->stack_offset--;
      cgbn_load(_arith.env, y, top());
    }

    __host__ __device__ void EVMStack::pushx(
        uint8_t x,
        uint32_t &error_code,
        uint8_t *src_byte_data,
        uint8_t src_byte_size)
    {
      if (x > 32)
      {
        error_code = ERROR_STACK_INVALID_PUSHX_X;
        return;
      }
      bn_t r;
      cgbn_set_ui32(_arith.env, r, 0);
      for (uint8_t idx = (x - src_byte_size); idx < x; idx++)
      {
        cgbn_insert_bits_ui32(
            _arith.env,
            r,
            r,
            idx * 8,
            8,
            src_byte_data[x - 1 - idx]);
      }
      push(r, error_code);
    }

    __host__ __device__ evm_word_t * EVMStack::get_index(
      uint32_t index,
      uint32_t &error_code
    )
    {
      if (size() < index)
      {
        error_code = ERR_STACK_UNDERFLOW;
        return NULL;
      }
      return _content->stack_base + (size() - index);
    }

    __host__ __device__ void EVMStack::dupx(
        uint8_t x,
        uint32_t &error_code)
    {
      if ((x < 1) || (x > 16))
      {
        error_code = ERROR_STACK_INVALID_DUPX_X;
        return;
      }
      bn_t r;
      evm_word_t *value = get_index(x, error_code);
      if (value == NULL)
      {
        return;
      }
      cgbn_load(_arith.env, r, value);
      push(r, error_code);
    }

    __host__ __device__ void EVMStack::swapx(
      uint32_t index,
      uint32_t &error_code)
    {
      if ((index < 1) || (index > 16))
      {
        error_code = ERR_STACK_INVALID_SIZE;
        return;
      }
      bn_t a, b;
      evm_word_t *value_a = get_index(1, error_code);
      evm_word_t *value_b = get_index(index + 1, error_code);
      if ((value_a == NULL) || (value_b == NULL))
      {
        return;
      }
      cgbn_load(_arith.env, a, value_b);
      cgbn_load(_arith.env, b, value_a);
      cgbn_store(_arith.env, value_a, a);
      cgbn_store(_arith.env, value_b, b);
    }

    __host__ __device__ void EVMStack::to_stack_data_t(
        stack_data_t &dst)
    {
      ONE_THREAD_PER_INSTANCE(
          if (
              (dst.stack_offset > 0) &&
              (dst.stack_base != NULL)) {
            delete[] dst.stack_base;
            dst.stack_base = NULL;
          } dst.stack_offset = _content->stack_offset;
          if (dst.stack_offset == 0) {
            dst.stack_base = NULL;
          } else {
            dst.stack_base = new evm_word_t[dst.stack_offset];
            memcpy(
                dst.stack_base,
                _content->stack_base,
                sizeof(evm_word_t) * dst.stack_offset);
          })
    }
    __host__ __device__ void EVMStack::print(
        bool full)
    {
      printf("Stack size: %d, data:\n", size());
      uint32_t print_size = full ? EVM_MAX_STACK_SIZE : size();
      for (uint32_t idx = 0; idx < print_size; idx++)
      {
        _content->stack_base[idx].print();
      }
    }

      
    /**
     * Generate a JSON object from the stack.
     * @param[in] full If false, prints only active stack elements, otherwise prints the entire stack
    */
    __host__ cJSON * EVMStack::json(bool full)
    {
      char *hex_string_ptr = new char[EVM_WORD_SIZE * 2 + 3];
      cJSON *stack_json = cJSON_CreateObject();

      cJSON *stack_data_json = cJSON_CreateArray();
      uint32_t print_size = full ? EVM_MAX_STACK_SIZE : size();
      for (uint32_t idx = 0; idx < print_size; idx++)
      {
        _content->stack_base[idx].to_hex(hex_string_ptr);
        cJSON_AddItemToArray(stack_data_json, cJSON_CreateString(hex_string_ptr));
      }
      cJSON_AddItemToObject(stack_json, "data", stack_data_json);
      delete[] hex_string_ptr;
      return stack_json;
    }
  }
}

