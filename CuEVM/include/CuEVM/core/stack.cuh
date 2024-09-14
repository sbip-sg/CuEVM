// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation
// Programme) Author: Stefan-Dan Ciocirlan Data: 2023-11-30
// SPDX-License-Identifier: MIT

#ifndef _CUEVM_STACK_H_
#define _CUEVM_STACK_H_

#include <CuEVM/core/evm_word.cuh>
#include <CuEVM/utils/arith.cuh>
#include <CuEVM/utils/cuda_utils.cuh>
#include <CuEVM/utils/evm_defines.cuh>

namespace CuEVM {
namespace stack {
constexpr CONSTANT uint32_t max_size =
    CuEVM::max_stack_size; /**< The maximum stack size*/
// constexpr CONSTANT uint32_t alligment =
//     sizeof(evm_word_t); /**< The alligment of the stack*/
constexpr CONSTANT uint32_t initial_capacity =
    16U; /**< The initial capacity of the stack can be change for performence
            reasons*/

struct evm_stack_t {
    evm_word_t *stack_base; /**< The stack YP: (YP: \f$\mu_{s}\f$)*/
    uint32_t stack_offset;  /**< The stack offset (YP: \f$|\mu_{s}|\f$)*/
    uint32_t capacity;      /**< The capacity of the stack*/

    /**
     * The default constructor
     */
    __host__ __device__ evm_stack_t();

    /**
     * The destructor
     */
    __host__ __device__ ~evm_stack_t();

    /**
     * The copy constructor
     * @param[in] other The other stack
     */
    __host__ __device__ evm_stack_t(const evm_stack_t &other);

    /**
     * Free the memory
     */
    __host__ __device__ void free();

    /**
     * Clear the content
     */
    __host__ __device__ void clear();

    /**
     * The assignment operator
     * @param[in] other The other stack
     * @return The reference to the stack
     */
    __host__ __device__ evm_stack_t &operator=(const evm_stack_t &other);

    /**
     * Duplicate the stack
     * @param[in] other The other stack
     */
    __host__ __device__ void duplicate(const evm_stack_t &other);

    /**
     * Grow the stack
     * @return 0 if the stack is grown, error code otherwise
     */
    __host__ __device__ int32_t grow();

    /**
     * Get the size of the stack
     * @return The size of the stack
     */
    __host__ __device__ uint32_t size() const;

    /**
     * Get the top of the stack
     * @return The top of the stack pointer
     */
    __host__ __device__ evm_word_t *top();

    /**
     * Push a value to the stack
     * @param[in] arith The arithmetical environment
     * @param[in] value The value to be pushed
     * @return 0 if the value is pushed, error code otherwise
     */
    __host__ __device__ int32_t push(ArithEnv &arith, const bn_t &value);

    /**
     * Pop a value from the stack
     * @param[in] arith The arithmetical environment
     * @param[out] y The value popped from the stack
     * @return 0 if the value is popped, error code otherwise
     */
    __host__ __device__ int32_t pop(ArithEnv &arith, bn_t &y);

    /**
     * Push a value to the stack from a byte array
     * @param[in] arith The arithmetical environment
     * @param[in] x the number of bytes of the value
     * @param[in] src_byte_data The source byte data
     * @param[in] src_byte_size The size of the source byte data
     * @return 0 if the value is pushed, error code otherwise
     */
    __host__ __device__ int32_t pushx(ArithEnv &arith, uint8_t x,
                                      uint8_t *src_byte_data,
                                      uint8_t src_byte_size);

    /**
     * Get the value from the stack at the given index
     * @param[in] arith The arithmetical environment
     * @param[in] index The index of the value
     * @param[out] y The value at the given index
     * @return 0 if the value is popped, error code otherwise
     */
    __host__ __device__ int32_t get_index(ArithEnv &arith, uint32_t index,
                                          bn_t &y);

    /**
     * Duplicvate the value at the given index and push
     * it at the top of the stack.
     * @param[in] arith The arithmetical environment
     * @param[in] x The index of the value
     * @return 0 if the value is duplicated, error code otherwise
     */
    __host__ __device__ int32_t dupx(ArithEnv &arith, uint32_t x);

    /**
     * Swap the values at the given index with the top of the stack
     * @param[in] arith The arithmetical environment
     * @param[in] x The index of the value
     * @return 0 if the value is swapped, error code otherwise
     */
    __host__ __device__ int32_t swapx(ArithEnv &arith, uint32_t x);

    /**
     * Print the stack
     */
    __host__ __device__ void print();

    /**
     * Get the JSON object from the stack
     * @return The JSON object
     */
    __host__ cJSON *to_json();

    // STATIC FUNCTIONS
    /**
     * Geenrat ethe stack cpu instances
     * @param[in] count The number of instances
     * @return The stack cpu instances
     */
    __host__ static evm_stack_t *get_cpu(uint32_t count);

    /**
     * Free the stack cpu instances
     * @param[in] instances The stack cpu instances
     * @param[in] count The number of instances
     */
    __host__ static void cpu_free(evm_stack_t *instances, uint32_t count);

    /**
     * Generate the stack gpu instances from the stack cpu instances
     * @param[in] cpu_instances The stack cpu instances
     * @param[in] count The number of instances
     * @return The stack gpu instances
     */
    __host__ static evm_stack_t *gpu_from_cpu(evm_stack_t *cpu_instances,
                                              uint32_t count);

    /**
     * Free the stack gpu instances
     * @param[in] gpu_instances The stack gpu instances
     * @param[in] count The number of instances
     */
    __host__ static void gpu_free(evm_stack_t *gpu_instances, uint32_t count);

    /**
     * Generate the stack cpu instances from the stack gpu instances
     * @param[in] gpu_instances The stack gpu instances
     * @param[in] count The number of instances
     * @return The stack cpu instances
     */
    __host__ static evm_stack_t *cpu_from_gpu(evm_stack_t *gpu_instances,
                                              uint32_t count);
};

/**
 * The kernel to transfer the stack from the CPU to the GPU
 * @param[in] dst The destination stack
 * @param[in] src The source stack
 * @param[in] count The number of instances
 */
__global__ void transfer_kernel_evm_stack_t(evm_stack_t *dst, evm_stack_t *src,
                                            uint32_t count);

}  // namespace stack
   // Type alias for accessing evm_stack_t directly under the CuEVM namespace
using evm_stack_t = stack::evm_stack_t;
}  // namespace CuEVM

#endif