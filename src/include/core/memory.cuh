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
      __host__ __device__ void print() const;

      /**
       * Get the json object from the memory data structure.
       * @return The json object.
       */
      __host__ cJSON *to_json() const;

      /**
       * Get the size of the memory. (Last offset reached by the memory access) x 32
       * @return The size of the memory.
       */
      __host__ __device__ uint32_t get_size() const {
        return size;
      }

      /**
       * Get the memory cost.
       * @param[in] arith The arithmetical environment.
       * @param[out] cost The memory cost.
       */
      __host__ __device__ void get_memory_cost(
        ArithEnv &arith,
        bn_t &cost) const;
      
      /**
       * Increase the memory cost.
       * @param[in] arith The arithmetical environment.
       * @param[in] memory_expansion_cost The memory expansion cost.
       */
      __host__ __device__ void increase_memory_cost(
        ArithEnv &arith,
        const bn_t &memory_expansion_cost);

      /**
       * Allocates pages for the given offset if needed.
       * @param[in] new_size The new size of the memory.
       * @return 0 if success, otherwise the error code.
       */
      __host__ __device__ int32_t allocate_pages(
        uint32_t new_size);

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
        uint32_t &offset) const;

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
        const bn_t &length);

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

    /**
     * The kernel to copy the memory between the GPU memories.
     * @param[out] dst_instances The destination GPU memory data structures.
     * @param[in] src_instances The source GPU memory data structures.
     * @param[in] instance_count The number of instances.
     */
    __global__ void transfer_kernel(
      evm_memory_t *dst_instances,
      evm_memory_t *src_instances,
      uint32_t instance_count);

    /**
     * Allocate the CPU memory data structures for the given number of instances.
     * @param[in] count The number of instances.
     * @return The CPU memory data structures.
    */
    __host__ evm_memory_t *get_cpu(
        uint32_t count);

    /**
     * Free the CPU memory data structures for the given number of instances.
     * @param[in] instances The CPU memory data structures.
     * @param[in] count The number of instances.
    */
    __host__ void cpu_free(
        evm_memory_t* instances,
        uint32_t count);

    /**
     * Allocate the GPU memory data structures for the given number of instances.
     * @param[in] cpu_instances The CPU memory data structures.
     * @param[in] count The number of instances.
     * @return The GPU memory data structures.
    */
    __host__ evm_memory_t *get_gpu_from_cpu(
        evm_memory_t *cpu_instances,
        uint32_t count);
    /**
     * Free the GPU memory data structures for the given number of instances.
     * @param[in] gpu_instances The GPU memory data structures.
     * @param[in] count The number of instances.
    */
    __host__ void gpu_free(
        evm_memory_t *gpu_instances,
        uint32_t count);

    /**
     * Copy the GPU memory data structures to the CPU memory data structures.
     * @param[in] gpu_instances The GPU memory data structures.
     * @param[in] count The number of instances.
     * @return The CPU memory data structures.
    */
    __host__ evm_memory_t *get_cpu_from_gpu(
      evm_memory_t *gpu_instances,
      uint32_t count);
  }
}

#endif