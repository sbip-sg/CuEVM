// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#include "../include/core/stack.cuh"
#include "../include/utils/error_codes.cuh"

namespace cuEVM
{
  namespace stack {

      __host__ __device__ evm_stack_t::evm_stack_t() : stack_base(nullptr), stack_offset(0), capacity(0) {
      }

      __host__ __device__ evm_stack_t::~evm_stack_t() {
        free();
      }

      __host__ __device__ evm_stack_t::evm_stack_t(const evm_stack_t &other) {
        //free();
        duplicate(other);
      }

      __host__ __device__ void evm_stack_t::free() {
        if (stack_base != nullptr) {
          std::free(stack_base);
        }
        clear();
      }

      __host__ __device__ void evm_stack_t::clear() {
        stack_offset = 0;
        capacity = 0;
        stack_base = nullptr;
      }

      __host__ __device__ evm_stack_t &evm_stack_t::operator=(const evm_stack_t &other) {
        if (this != &other) {
          free();
          duplicate(other);
        }
        return *this;
      }

      __host__ __device__ void evm_stack_t::duplicate(
        const evm_stack_t &other) {
        stack_base = static_cast<evm_word_t*>(
          std::aligned_alloc(
            alligment,
            other.stack_offset * sizeof(evm_word_t)
          )
        );
        if (stack_base != nullptr) {
          std::copy(other.stack_base, other.stack_base + other.stack_offset, stack_base);
          stack_offset = other.stack_offset;
          capacity = other.stack_offset;
        }
      }

      __host__ __device__ int32_t evm_stack_t::grow() {
        capacity = (capacity == 0) ? initial_capacity : capacity * 2;
        if (capacity > max_size) {
          return ERROR_STACK_OVERFLOW;
        }
        evm_word_t *new_stack_base = static_cast<evm_word_t*>(
          std::aligned_alloc(
            alligment,
            capacity * sizeof(evm_word_t)
          )
        );
        if (new_stack_base == nullptr) {
          return ERROR_MEMORY_ALLOCATION_FAILED;
        }
        if (stack_base != nullptr) {
          std::copy(stack_base, stack_base + stack_offset, new_stack_base);
          std::free(stack_base);
        }
        stack_base = new_stack_base;
        return ERROR_SUCCESS;
      }

      __host__ __device__ uint32_t evm_stack_t::size() const {
        return stack_offset;
      }

      __host__ __device__ evm_word_t *evm_stack_t::top() {
        return stack_base + stack_offset;
      }

      __host__ __device__ int32_t evm_stack_t::push(
        ArithEnv &arith,
        const bn_t &value) {
        int32_t error_code = (size() > capacity) ? grow() : ERROR_SUCCESS;
        cgbn_store(arith.env, top(), value);
        stack_offset++;
        return error_code;
      }

      __host__ __device__ int32_t evm_stack_t::pop(
        ArithEnv &arith,
        bn_t &y) {
        if (size() == 0) {
          // TODO: delete maybe?
          cgbn_set_ui32(arith.env, y, 0);
          return ERROR_STACK_UNDERFLOW;
        }
        stack_offset--;
        cgbn_load(arith.env, y, top());
        return ERROR_SUCCESS;
      }

      __host__ __device__ int32_t evm_stack_t::pushx(
        ArithEnv &arith,
        uint8_t x,
        uint8_t *src_byte_data,
        uint8_t src_byte_size) {
        // TODO:: for sure is something more efficient here
        if (x > 32)
        {
          return ERROR_STACK_INVALID_SIZE;
        }
        bn_t r;
        cgbn_set_ui32(arith.env, r, 0);
        for (uint8_t idx = (x - src_byte_size); idx < x; idx++)
        {
          cgbn_insert_bits_ui32(
              arith.env,
              r,
              r,
              idx * 8,
              8,
              src_byte_data[x - 1 - idx]);
        }
        return push(arith, r);
      }

      __host__ __device__ int32_t evm_stack_t::get_index(
        ArithEnv &arith,
        uint32_t index,
        bn_t &y) {
        if (index > size()) {
          return ERROR_STACK_INVALID_INDEX;
        }
        cgbn_load(arith.env, y, stack_base + size() - index);
        return ERROR_SUCCESS;
      }

      __host__ __device__ int32_t evm_stack_t::dupx(
        ArithEnv &arith,
        uint32_t x) {
        bn_t r;
        int32_t error_code = ((x > 16) || (x < 1)) ? ERROR_STACK_INVALID_SIZE : get_index(arith, x, r);
        return error_code | push(arith, r);
      }

      __host__ __device__ int32_t evm_stack_t::swapx(
        ArithEnv &arith,
        uint32_t x) {
        bn_t a, b;
        int32_t error_code = ((x > 16) || (x < 1)) ? ERROR_STACK_INVALID_SIZE : (get_index(arith, 1, a) | get_index(arith, x + 1, b));
        cgbn_store(arith.env, stack_base + size() - x - 1, a);
        cgbn_store(arith.env, stack_base + size() - 1, b);
        return error_code;
      }

      __host__ __device__ void evm_stack_t::print() {
        printf("Stack size: %d, data:\n", size());
        for (uint32_t idx = 0; idx < size(); idx++)
        {
          stack_base[idx].print();
        }
      }

      __host__ cJSON *evm_stack_t::to_json() {
        cJSON *json = cJSON_CreateObject();
        char *hex_string_ptr = new char[cuEVM::word_size * 2 + 3];
        cJSON *stack = cJSON_CreateArray();
        for (uint32_t idx = 0; idx < size(); idx++)
        {
          stack_base[idx].to_hex(hex_string_ptr);
          cJSON_AddItemToArray(stack, cJSON_CreateString(hex_string_ptr));
        }
        cJSON_AddItemToObject(json, "data", stack);
        delete[] hex_string_ptr;
        return json;
      }

    __global__ void transfer_kernel(
      evm_stack_t *dst,
      evm_stack_t *src,
      uint32_t count) {
      uint32_t instance = blockIdx.x * blockDim.x + threadIdx.x;
      if (instance >= count)
      {
        return;
      }
      dst[instance].stack_offset = src[instance].stack_offset;
      std::copy(src[instance].stack_base, src[instance].stack_base + src[instance].stack_offset, dst[instance].stack_base);
      src[instance].free();
    }

    __host__ evm_stack_t *get_cpu(
        uint32_t count) {
      evm_stack_t *instances = new evm_stack_t[count];
      return instances;
    }

    __host__ void cpu_free(
        evm_stack_t *instances,
        uint32_t count) {
      for (uint32_t idx = 0; idx < count; idx++)
      {
        instances[idx].free();
      }
      delete[] instances;
    }

    __host__ evm_stack_t *get_gpu_from_cpu(
        evm_stack_t *cpu_instances,
        uint32_t count) {
      evm_stack_t *gpu_instances, *tmp_gpu_instances;
      tmp_gpu_instances = new evm_stack_t[count];
      std::copy(cpu_instances, cpu_instances + count, tmp_gpu_instances);
      for (uint32_t idx = 0; idx < count; idx++)
      {
        if (cpu_instances[idx].stack_base != nullptr)
        {
          CUDA_CHECK(cudaMalloc(
            &tmp_gpu_instances[idx].stack_base,
            cpu_instances[idx].capacity * sizeof(evm_word_t)));
          CUDA_CHECK(cudaMemcpy(
            tmp_gpu_instances[idx].stack_base,
            cpu_instances[idx].stack_base,
            cpu_instances[idx].stack_offset * sizeof(evm_word_t),
            cudaMemcpyHostToDevice));
        } else {
          tmp_gpu_instances[idx].stack_base = nullptr;
        }
      }
      CUDA_CHECK(cudaMalloc(
        &gpu_instances,
        count * sizeof(evm_stack_t)));
      CUDA_CHECK(cudaMemcpy(
        gpu_instances,
        tmp_gpu_instances,
        count * sizeof(evm_stack_t),
        cudaMemcpyHostToDevice));
      for (uint32_t idx = 0; idx < count; idx++)
      {
        tmp_gpu_instances[idx].clear();
      }
      delete[] tmp_gpu_instances;
      return gpu_instances;
    }

    __host__ void gpu_free(
        evm_stack_t *gpu_instances,
        uint32_t count) {
      evm_stack_t *tmp_gpu_instances = new evm_stack_t[count];
      CUDA_CHECK(cudaMemcpy(
        tmp_gpu_instances,
        gpu_instances,
        count * sizeof(evm_stack_t),
        cudaMemcpyDeviceToHost));
      for (uint32_t idx = 0; idx < count; idx++)
      {
        if (tmp_gpu_instances[idx].stack_base != nullptr)
        {
          CUDA_CHECK(cudaFree(tmp_gpu_instances[idx].stack_base));
        }
          tmp_gpu_instances[idx].clear();
      }
      delete[] tmp_gpu_instances;
      CUDA_CHECK(cudaFree(gpu_instances));
    }

    __host__ evm_stack_t *get_cpu_from_gpu(
        evm_stack_t *gpu_instances,
        uint32_t count) {
      evm_stack_t *cpu_instances = new evm_stack_t[count];
      evm_stack_t *tmp_gpu_instances = new evm_stack_t[count];
      evm_stack_t *tmp_cpu_instances = nullptr;
      CUDA_CHECK(cudaMemcpy(
        cpu_instances,
        gpu_instances,
        count * sizeof(evm_stack_t),
        cudaMemcpyDeviceToHost));
      std::copy(cpu_instances, cpu_instances + count, tmp_cpu_instances);
      for (uint32_t idx = 0; idx < count; idx++)
      {
        if (cpu_instances[idx].stack_offset > 0)
        {
          CUDA_CHECK(cudaMallocHost(
            &tmp_cpu_instances[idx].stack_base,
            cpu_instances[idx].stack_offset * sizeof(evm_word_t)));
        } else {
          tmp_cpu_instances[idx].stack_base = nullptr;
        }
      }
      CUDA_CHECK(cudaMalloc(
        &tmp_gpu_instances,
        count * sizeof(evm_stack_t)));
      CUDA_CHECK(cudaMemcpy(
        tmp_gpu_instances,
        tmp_cpu_instances,
        count * sizeof(evm_stack_t),
        cudaMemcpyHostToDevice));
      for (uint32_t idx = 0; idx < count; idx++) {
        tmp_cpu_instances[idx].clear();
      }
      delete[] tmp_cpu_instances;
      cuEVM::stack::transfer_kernel<<<count, 1>>>(
        tmp_gpu_instances,
        gpu_instances,
        count);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaFree(gpu_instances));
      gpu_instances = tmp_gpu_instances;

      CUDA_CHECK(cudaMemcpy(
          cpu_instances,
          gpu_instances,
          sizeof(evm_stack_t) * count,
          cudaMemcpyDeviceToHost));
      tmp_cpu_instances = new evm_stack_t[count];
      memcpy(
          tmp_cpu_instances,
          cpu_instances,
          sizeof(evm_stack_t) * count);

      for (uint32_t idx = 0; idx < count; idx++)
      {
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

      std::copy(tmp_cpu_instances, tmp_cpu_instances + count, cpu_instances);

      for (uint32_t idx = 0; idx < count; idx++)
      {
        tmp_cpu_instances[idx].clear();
      }
      delete[] tmp_cpu_instances;
      tmp_cpu_instances = NULL;
      cuEVM::stack::gpu_free(gpu_instances, count);
      return cpu_instances;
    }
  } // namespace stack
} // namespace cuEVM
