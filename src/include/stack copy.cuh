// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#ifndef _STACK_H_
#define _STACK_H_

#include "arith.cuh"

namespace cuEVM
{
  namespace stack {
    /**
     * The stack data structure.
     */
    typedef struct
    {
      evm_word_t *stack_base; /**< The stack YP: (YP: \f$\mu_{s}\f$)*/
      uint32_t stack_offset; /**< The stack offset (YP: \f$|\mu_{s}|\f$)*/
    } stack_data_t;

    /**
     * The kernel to copy the stack data structures.
     * @param[out] dst The destination stack data structure
     * @param[in] src The source stack data structure
     * @param[in] count The number of instances
    */
    __global__ void transfer_kernel(
      stack_data_t *dst,
      stack_data_t *src,
      uint32_t count);
    
    /**
     * Generate the cpu data structures for the stack.
     * @param[in] count The number of instances
     * @return The cpu data structures
    */
    __host__ stack_data_t *get_cpu_instances(
        uint32_t count);

    /**
     * Free the cpu data structures for the stack.
     * @param[in] cpu_instances The cpu data structures
     * @param[in] count The number of instances
    */
    __host__ void free_cpu_instances(
        stack_data_t *cpu_instances,
        uint32_t count);
    /**
     * Generate the gpu data structures for the stack.
     * @param[in] cpu_instances The cpu data structures
     * @param[in] count The number of instances
     * @return The gpu data structures
    */
    __host__ stack_data_t *get_gpu_instances_from_cpu_instances(
        stack_data_t *cpu_instances,
        uint32_t count);
    /**
     * Free the gpu data structures for the stack.
     * @param[in] gpu_instances The gpu data structures
     * @param[in] count The number of instances
    */
    __host__ void free_gpu_instances(
        stack_data_t *gpu_instances,
        uint32_t count);
    /**
     * Generate the cpu data structures for the stack from the gpu data structures.
     * @param[in] gpu_instances The gpu data structures
     * @param[in] count The number of instances
     * @return The cpu data structures
    */
    __host__ stack_data_t *get_cpu_instances_from_gpu_instances(
        stack_data_t *gpu_instances,
        uint32_t count);

    /**
     * Print the stack data structure.
     * @param[in] arith The arithmetical environment
     * @param[in] stack_data The stack data structure
    */
    __host__ __device__ void print_stack_data_t(
        ArithEnv &arith,
        stack_data_t &stack_data);


    /**
     * Generate a JSON object from a stack data structure.
     * @param[in] arith The arithmetical environment
     * @param[in] stack_data The stack data structure
     * @return The JSON object
    */
    __host__ cJSON *json_from_stack_data_t(
        ArithEnv &arith,
        stack_data_t &stack_data);

    /**
     * The stack class (YP: \f$\mu_{s}\f$)
    */
    class EVMStack{
    public:


      stack_data_t *_content; /**< The conent of the stack*/
      ArithEnv _arith; /**< The arithmetical environment*/

      /**
       * The constructor of the stack given the arithmetical environment
       * and the stack data structure.
       * @param[in] arith The arithmetical environment
       * @param[in] content The stack data structure
      */
      __host__ __device__ EVMStack(
          ArithEnv arith,
          stack_data_t *content);

      /**
       * The constructor of the stack given the arithmetical environment.
       * @param[in] arith The arithmetical environment
      */
      __host__ __device__ EVMStack(
          ArithEnv arith);

      /**
       * The destructor of the stack.
      */
      __host__ __device__ ~EVMStack();

      /**
       * Get the size of the stack.
       * @return The size of the stack
      */
      __host__ __device__ uint32_t size();

      /**
       * Get the top of the stack.
       * @return The top of the stack pointer
      */
      __host__ __device__ evm_word_t *top();

      /**
       * Push a value on the stack.
       * @param[in] value The value to be pushed on the stack
       * @param[out] error_code The error code
      */
      __host__ __device__ void push(const bn_t &value, uint32_t &error_code);

      /**
       * Pop a value from the stack.
       * @param[out] y The value popped from the stack
       * @param[out] error_code The error code
      */
      __host__ __device__ void pop(bn_t &y, uint32_t &error_code);

      /**
       * Push a value on the stack given by a byte array.
       * If the size of the byte array is smaller than x,
       * the value is padded with zeros for the least
       * significant bytes.
       * @param[in] x The number of bytes to be pushed on the stack
       * @param[out] error_code The error code
       * @param[in] src_byte_data The byte array
       * @param[in] src_byte_size The size of the byte array
      */
      __host__ __device__ void pushx(
          uint8_t x,
          uint32_t &error_code,
          uint8_t *src_byte_data,
          uint8_t src_byte_size);

      /**
       * Get the pointer to the stack element at a given index from the top.
       * @param[in] index The index from the top of the stack
       * @param[out] error_code The error code
       * @return The pointer to the stack element
      */
      __host__ __device__ evm_word_t *get_index(
        uint32_t index,
        uint32_t &error_code
      );

      /**
       * Duplicate the stack element at a given index from the top,
       * and push it on the stack.
       * @param[in] index The index from the top of the stack
       * @param[out] error_code The error code
      */
      __host__ __device__ void dupx(
          uint8_t x,
          uint32_t &error_code);

      /**
       * Swap the stack element at a given index from the top with the top element.
       * @param[in] index The index from the top of the stack
       * @param[out] error_code The error code
      */
      __host__ __device__ void swapx(
        uint32_t index,
        uint32_t &error_code);

      /**
       * Copy the stack to a stack data structure.
       * @param[out] dst The stack data structure
      */
      __host__ __device__ void to_stack_data_t(
          stack_data_t &dst);

      /**
       * Print the stack.
       * @param[in] full If false, prints only active stack elements, otherwise prints the entire stack
      */
      __host__ __device__ void print(
          bool full = false);

      /**
       * Generate a JSON object from the stack.
       * @param[in] full If false, prints only active stack elements, otherwise prints the entire stack
      */
      __host__ cJSON *json(bool full = false);
    };
  } // namespace stack
} // namespace cuEVM

#endif