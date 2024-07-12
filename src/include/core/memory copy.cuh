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
    typedef struct
    {
      size_t size; /**< The size of the memory acceesed by now (YP: \f$32 \dot \mu_{i}\f$)*/
      size_t allocated_size; /**< Internal allocated size do not confuse with size*/
      evm_word_t memory_cost; /**< The memory cost (YP: \f$M(\mu_{i})\f$)*/
      uint8_t *data; /**< The data of the memory*/
    } memory_data_t;

    /**
     * The kernel to copy the memory data structures between the GPU memories.
     * @param[out] dst The destination memory data structure
     * @param[in] src The source memory data structure
     * @param[in] count The number of instances
    */
    __global__ void transfer_kernel(
      memory_data_t *dst_instances,
      memory_data_t *src_instances,
      uint32_t instance_count
    );

    /**
     * Allocate the CPU memory data structures for the given number of instances.
     * @param[in] count The number of instances.
     * @return The CPU memory data structures.
    */
    __host__ memory_data_t *get_cpu_instances(
        uint32_t count
    );

    /**
     * Free the CPU memory data structures for the given number of instances.
     * @param[in] instances The CPU memory data structures.
     * @param[in] count The number of instances.
    */
    __host__ void free_cpu_instances(
        memory_data_t *instances,
        uint32_t count
    );

    /**
     * Allocate the GPU memory data structures for the given number of instances.
     * @param[in] cpu_instances The CPU memory data structures.
     * @param[in] count The number of instances.
     * @return The GPU memory data structures.
    */
    __host__ memory_data_t *get_gpu_instances_from_cpu_instances(
        memory_data_t *cpu_instances,
        uint32_t count
    );

    /**
     * Free the GPU memory data structures for the given number of instances.
     * @param[in] gpu_instances The GPU memory data structures.
     * @param[in] count The number of instances.
    */
    __host__ void free_gpu_instances(
        memory_data_t *gpu_instances,
        uint32_t count
    );

    /**
     * Copy the GPU memory data structures to the CPU memory data structures.
     * @param[in] gpu_instances The GPU memory data structures.
     * @param[in] count The number of instances.
     * @return The CPU memory data structures.
    */
    __host__ memory_data_t *get_cpu_instances_from_gpu_instances(
      memory_data_t *gpu_instances,
      uint32_t count
    );

    /**
     * Print the memory data structure.
     * @param[in] arith The arithmetical environment.
     * @param[in] memory_data The memory data structure.
    */
    __host__ __device__ void print_memory_data_t(
      ArithEnv &arith,
      memory_data_t &memory_data
    );


    /**
     * Get the json object from the memory data structure.
     * @param[in] arith The arithmetical environment.
     * @param[in] memory_data The memory data structure.
     */
    __host__ cJSON *json_from_memory_data_t(
      ArithEnv &arith,
      memory_data_t &memory_data
    );


    /**
     * The memory class (YP: \f$\mu_{m}\f$).
    */
    class EVMMemory
    {
    private:
      ArithEnv _arith; /**< The arithmetical environment*/
      memory_data_t *_content; /**< The memory data structure*/
    public:

      /**
       * The constructor of the memory class given the memory data structure.
       * @param[in] arith The arithmetical environment.
       * @param[in] content The memory data structure.
      */
      __host__ EVMMemory(
        ArithEnv arith,
        memory_data_t *content
      );

      /**
       * The constructor of the memory class given the arithmetical environment.
       * @param arith[in] The arithmetical environment.
      */
      __host__ __device__ EVMMemory(
        ArithEnv arith
      );

      /**
       * The destructor of the memory class.
       * Frees the allcoated data and the memory data structure.
      */
      __host__ __device__ ~EVMMemory();

      /**
       * The size of the memory. (Last offset reached by the memory access) x 32
       * (YP: \f$32 \dot \mu_{i}\f$)
       * @return The size of the memory.
      */
      __host__ __device__ size_t size();

      /**
       * Allocates pages for the given offset if needed.
       * @param[in] new_size The new size of the memory.
       * @param[out] error_code The error code.
      */
      __host__ __device__ void allocate_pages(
        size_t new_size,
        uint32_t &error_code
      );

      /**
       * The memory cost of the memory. (YP: \f$C_{mem}-M(\mu_{i}, index, length)\f$)
       * @param[in] index The index of the memory access.
       * @param[in] length The length of the memory access.
       * @param[out] gas_used The gas cost after.
       * @param[out] error_code The error code.
      */
      __host__ __device__ void grow_cost(
        bn_t &index,
        bn_t &length,
        bn_t &gas_used,
        uint32_t &error_code
      );

      /**
       * Get the highest offset of the memory access.
       * @param[in] index The index of the memory access.
       * @param[in] length The length of the memory access.
       * @param[out] error_code The error code.
      */
      __host__ __device__ size_t get_last_offset(
        bn_t &index,
        bn_t &length,
        uint32_t &error_code
      );

      /**
       * Increase the memory for the given offset if needed.
       * Offset is computed from the index and the length.
       * @param[in] index The index of the memory access.
       * @param[in] length The length of the memory access.
       * @param[out] error_code The error code.
      */
      __host__ __device__ void grow(
        bn_t &index,
        bn_t &length,
        uint32_t &error_code
      );

      /**
       * Get the a pointer to the given memory data.
       * @param[in] index The index of the memory access.
       * @param[in] length The length of the memory access.
       * @param[out] error_code The error code.
      */
      __host__ __device__ uint8_t *get(
        bn_t &index,
        bn_t &length,
        uint32_t &error_code
      );


      /**
       * Set the given memory data. Outside available_size is 0.
       * @param[in] data The data to be set.
       * @param[in] index The index of the memory access.
       * @param[in] length The length of the memory access.
       * @param[in] available_size The available size of the memory.
       * @param[out] error_code The error code.
      */
      __host__ __device__ void set(
        uint8_t *data,
        bn_t &index,
        bn_t &length,
        size_t &available_size,
        uint32_t &error_code
      );

      /**
       * Copy the current memory data to the given memory data structure.
       * @param[out] dst The destination memory data structure.
      */
      __host__ __device__ void to_memory_data_t(
        memory_data_t &dst);
      
      /**
       * Print the memory data structure.
      */
      __host__ __device__ void print();

      /**
       * Get the memory data structure as a JSON object.
       * @return The JSON object.
      */
      __host__ cJSON *json();
    };
  }
}

#endif