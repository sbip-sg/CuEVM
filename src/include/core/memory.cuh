// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#ifndef _CUEVM_MEMORY_H_
#define _CUEVM_MEMORY_H_

#include "../utils/cuda_utils.cuh"
#include "../utils/arith.cuh"
#include "byte_array.cuh"

namespace cuEVM {
  namespace memory {
    // to change for making more optimal memory allocation current 1KB
    constexpr CONSTANT uint32_t page_size = 1024U;
    /**
     * The memory data structure.
    */
    struct evm_memory_t {
      cuEVM::byte_array_t data; /**< The data of the memory*/
      evm_word_t memory_cost; /**< The memory cost (YP: \f$M(\mu_{i})\f$)*/
      uint32_t size; /**< The size of the memory acceesed by now (YP: \f$32 \dot \mu_{i}\f$)*/

      /**
       * The default constructor.
       */
      __host__ __device__ evm_memory_t() : data(), size(0) {
        memory_cost.from_uint32_t(0);
      }

      /**
       * the destructor
       */
      __host__ __device__ ~evm_memory_t() {
        memory_cost.from_uint32_t(0);
        size = 0;
      }

      /**
       * Print the memory data structure.
       */
      __host__ __device__ void print() {
        printf("Memory data: \n");
        printf("Size: %d\n", size);
        printf("Memory cost: ");
        memory_cost.print();
        printf("\n");
        data.print();
      }

      /**
       * Get the json object from the memory data structure.
       */
      __host__ cJSON *to_json() {
        cJSON *json = cJSON_CreateObject();
        cJSON_AddItemToObject(json, "size", cJSON_CreateNumber(size));
        char *hex_string_ptr = memory_cost.to_hex();
        cJSON_AddStringToObject(json, "memory_cost", hex_string_ptr);
        delete[] hex_string_ptr;
        cJSON_AddItemToObject(json, "data", data.to_json());
        return json;
      }

      /**
       * Get the size of the memory. (Last offset reached by the memory access) x 32
       * @return The size of the memory.
       */
      __host__ __device__ uint32_t get_size() {
        return size;
      }

      /**
       * Get the memory cost.
       * @param[in] arith The arithmetical environment.
       * @param[out] cost The memory cost.
       */
      __host__ __device__ void get_memory_cost(
        ArithEnv &arith,
        bn_t &cost) {
          cgbn_load(
            arith.env,
            cost,
            (cuEVM::cgbn_evm_word_t_ptr) &memory_cost
          );
      }

      

      /**
       * Allocates pages for the given offset if needed.
       * @param[in] new_size The new size of the memory.
       * @return 0 if success, otherwise the error code.
       */
      __host__ __device__ int32_t allocate_pages(
        uint32_t new_size) {
        if (new_size < data.size) {
          return 0;
        }
        uint32_t new_page_count = (new_size / cuEVM::memory::page_size) + 1;
        return data.grow(new_page_count * cuEVM::memory::page_size);
      }

      /**
       * Get the highest offset of the memory access.
       * @param[in] arith The arithmetical environment.
       * @param[in] index The index of the memory access.
       * @param[in] length The length of the memory access.
       * @param[out] offset The highest offset of the memory access.
       * @return 0 if success, otherwise the error code.
       */
      __host__ __device__ int32_t get_last_offset(
        ArithEnv &arith,
        const bn_t &index,
        const bn_t &length,
        uint32_t &offset) const {

        int32_t overflow = 0;
        bn_t offset_bn;
        overflow = cgbn_add(arith.env, offset_bn, index, length);
        overflow |= arith.uint32_t_from_cgbn(offset, offset_bn);
        bn_t memory_size;
        overflow |= cgbn_add_ui32(arith.env, memory_size, offset_bn, 31);
        cgbn_div_ui32(arith.env, memory_size, memory_size, 32);
        overflow |= cgbn_mul_ui32(arith.env, offset_bn, memory_size, 32);
        overflow |= arith.uint32_t_from_cgbn(offset, offset_bn);
    
        return overflow;
      }

      /**
       * Increase the memory for the given offset if needed.
       * Offset is computed from the index and the length.
       * @param[in] arith The arithmetical environment.
       * @param[in] index The index of the memory access.
       * @param[in] length The length of the memory access.
       * @return 0 if success, otherwise the error code.
       */
      __host__ __device__ int32_t grow(
        ArithEnv &arith,
        const bn_t &index,
        const bn_t &length) {
        uint32_t offset;
        if(get_last_offset(arith, index, length, offset) != 0) {
          return 1;
        }
        if (offset >= size) {
          if(allocate_pages(offset) != 0) {
            return 1;
          }
          size = offset;
        }
        return 0;
      }

      /**
       * Get the a pointer to the given memory data.
       * @param[in] arith The arithmetical environment.
       * @param[in] index The index of the memory access.
       * @param[in] length The length of the memory access.
       * @param[out] data The pointer to the memory data.
       * @return 0 if success, otherwise the error code.
       */
      __host__ __device__ int32_t get(
        ArithEnv &arith,
        const bn_t &index,
        const bn_t &length,
        cuEVM::byte_array_t &data) {
        
        do {
          if (cgbn_compare_ui32(arith.env, length, 0) >= 0) {
            break;
          } 
          if (grow(arith, index, length) != 0) {
            break;
          }
          uint32_t index_u32, length_u32;
          arith.uint32_t_from_cgbn(index_u32, index);
          arith.uint32_t_from_cgbn(length_u32, length);
          data = cuEVM::byte_array_t(this->data.data + index_u32, length_u32);
          return 0;
        } while (0);
        data = cuEVM::byte_array_t();
        return 1;
      }

      /**
       * Set the given memory data. Outside available_size is 0.
       * @param[in] arith The arithmetical environment.
       * @param[in] data The data to be set.
       * @param[in] index The index of the memory access.
       * @param[in] length The length of the memory access.
       * @return 0 if success, otherwise the error code.
       */
      __host__ __device__ int32_t set(
        ArithEnv &arith,
        const cuEVM::byte_array_t &data,
        const bn_t &index,
        const bn_t &length) {
        
        do {
          if (cgbn_compare_ui32(arith.env, length, 0) >= 0) {
            break;
          }
          if (grow(arith, index, length) != 0) {
            break;
          }
          uint32_t index_u32;
          arith.uint32_t_from_cgbn(index_u32, index);
          if (data.size > 0) {
            std::copy(data.data, data.data + data.size, this->data.data + index_u32);
          }
          return 0;
        } while (0);
        return 1;
      }
    };

    __global__ void transfer_kernel(
      evm_memory_t *dst_instances,
      evm_memory_t *src_instances,
      uint32_t instance_count
    ) {
      uint32_t instance = blockIdx.x * blockDim.x + threadIdx.x;
      if (instance >= instance_count)
        return;

      if (src_instances[instance].data.size > 0)
      {
        memcpy(
          dst_instances[instance].data.data,
          src_instances[instance].data.data,
          src_instances[instance].data.size * sizeof(uint8_t)
        );
        delete[] src_instances[instance].data.data;
        src_instances[instance].data.data = nullptr;
      }
    }

    /**
     * Allocate the CPU memory data structures for the given number of instances.
     * @param[in] count The number of instances.
     * @return The CPU memory data structures.
    */
    __host__ evm_memory_t *get_cpu(
        uint32_t count
    ) {
        return new evm_memory_t[count];
    }

    /**
     * Free the CPU memory data structures for the given number of instances.
     * @param[in] instances The CPU memory data structures.
     * @param[in] count The number of instances.
    */
    __host__ void cpu_free(
        evm_memory_t* instances,
        uint32_t count
    ) {
      delete[] instances;
    }

    /**
     * Allocate the GPU memory data structures for the given number of instances.
     * @param[in] cpu_instances The CPU memory data structures.
     * @param[in] count The number of instances.
     * @return The GPU memory data structures.
    */
    __host__ evm_memory_t *get_gpu_from_cpu(
        evm_memory_t *cpu_instances,
        uint32_t count
    ) {
        evm_memory_t *gpu_instances, *tmp_cpu_instances;
        CUDA_CHECK(cudaMalloc((void **)&gpu_instances, count * sizeof(evm_memory_t)));
        tmp_cpu_instances = new evm_memory_t[count];
        std::copy(cpu_instances, cpu_instances + count, tmp_cpu_instances);
        for (uint32_t i = 0; i < count; i++) {
            if (cpu_instances[i].data.size > 0) {
                CUDA_CHECK(cudaMalloc((void **)&tmp_cpu_instances[i].data.data, cpu_instances[i].data.size));
                CUDA_CHECK(cudaMemcpy(tmp_cpu_instances[i].data.data, cpu_instances[i].data.data, cpu_instances[i].data.size, cudaMemcpyHostToDevice));
            }
        }
        CUDA_CHECK(cudaMemcpy(tmp_cpu_instances, tmp_cpu_instances, count * sizeof(evm_memory_t), cudaMemcpyHostToDevice));
        delete[] tmp_cpu_instances;
        return gpu_instances;
    }

    /**
     * Free the GPU memory data structures for the given number of instances.
     * @param[in] gpu_instances The GPU memory data structures.
     * @param[in] count The number of instances.
    */
    __host__ void gpu_free(
        evm_memory_t *gpu_instances,
        uint32_t count
    ) {
        evm_memory_t *tmp_cpu_instances = new evm_memory_t[count];
        CUDA_CHECK(cudaMemcpy(tmp_cpu_instances, gpu_instances, count * sizeof(evm_memory_t), cudaMemcpyDeviceToHost));
        for (uint32_t i = 0; i < count; i++) {
            if (tmp_cpu_instances[i].data.size > 0) {
                CUDA_CHECK(cudaFree(tmp_cpu_instances[i].data.data));
            }
        }
        delete[] tmp_cpu_instances;
        CUDA_CHECK(cudaFree(gpu_instances));
    }

    /**
     * Copy the GPU memory data structures to the CPU memory data structures.
     * @param[in] gpu_instances The GPU memory data structures.
     * @param[in] count The number of instances.
     * @return The CPU memory data structures.
    */
    __host__ evm_memory_t *get_cpu_from_gpu(
      evm_memory_t *gpu_instances,
      uint32_t count
    ) {
      evm_memory_t *cpu_instances = new evm_memory_t[count];
      evm_memory_t *tmp_cpu_instances = new evm_memory_t[count];
      evm_memory_t* tmp_gpu_instances = nullptr;
      CUDA_CHECK(cudaMemcpy(cpu_instances, gpu_instances, count * sizeof(evm_memory_t), cudaMemcpyDeviceToHost));
      std::copy(cpu_instances, cpu_instances + count, tmp_cpu_instances);
      for (uint32_t idx = 0; idx < count; idx++) {
        if (tmp_cpu_instances[idx].size > 0)
        {
          CUDA_CHECK(cudaMalloc(
            (void **)&tmp_cpu_instances[idx].data.data,
            sizeof(uint8_t) * tmp_cpu_instances[idx].data.size
          ));
        }
        else
        {
          tmp_cpu_instances[idx].data.data = nullptr;
        }
        tmp_cpu_instances[idx].size = tmp_cpu_instances[idx].size;
      }
      CUDA_CHECK(cudaMalloc((void **)&tmp_gpu_instances, count * sizeof(evm_memory_t)));
      CUDA_CHECK(cudaMemcpy(tmp_gpu_instances, tmp_cpu_instances, count * sizeof(evm_memory_t), cudaMemcpyHostToDevice));
      delete[] tmp_cpu_instances;
      // 2. call the kernel to copy the memory between the gpu memories
      cuEVM::memory::transfer_kernel<<<count, 1>>>(tmp_gpu_instances, gpu_instances, count);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaFree(gpu_instances));
      gpu_instances = tmp_gpu_instances;

      // 3. copy the gpu memories back in the cpu memories
      CUDA_CHECK(cudaMemcpy(
        cpu_instances,
        gpu_instances,
        sizeof(evm_memory_t) * count,
        cudaMemcpyDeviceToHost
      ));
      tmp_cpu_instances = new evm_memory_t[count];
      std::copy(cpu_instances, cpu_instances + count, tmp_cpu_instances);
      for (size_t idx = 0; idx < count; idx++)
      {
        if (tmp_cpu_instances[idx].data.size > 0)
        {
          tmp_cpu_instances[idx].data.data = new uint8_t[tmp_cpu_instances[idx].data.size];
          CUDA_CHECK(cudaMemcpy(
            tmp_cpu_instances[idx].data.data,
            cpu_instances[idx].data.data,
            sizeof(uint8_t) * tmp_cpu_instances[idx].data.size,
            cudaMemcpyDeviceToHost
          ));
        }
        else
        {
          tmp_cpu_instances[idx].data.data = nullptr;
        }
      }
      gpu_free(gpu_instances, count);
      std::copy(tmp_cpu_instances, tmp_cpu_instances + count, cpu_instances);
      delete[] tmp_cpu_instances;
      tmp_cpu_instances = NULL;
      return cpu_instances;
    }
  }
}

#endif