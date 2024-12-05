#pragma once
#include <CuEVM/core/stack.cuh>

/**
 * The comparison operations.
 * Contains the next operations 10s: Comparison Operations:
 * - LT
 * - GT
 * - SLT
 * - SGT
 * - EQ
 * - ISZERO
 */
namespace CuEVM::operations {
/**
 * The LT operation implementation.
 * Takes two values from the stack, compares them and pushes the result
 * back to the stack.
 * The two values are considered unsigned.
 * The result is 1 if the first value is less than the second value,
 * 0 otherwise.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 */
__host__ __device__ int32_t LT(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack);

/**
 * The GT operation implementation.
 * Takes two values from the stack, compares them and pushes the result
 * back to the stack.
 * The two values are considered unsigned.
 * The result is 1 if the first value is greater than the second value,
 * 0 otherwise.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 */
__host__ __device__ int32_t GT(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack);

/**
 * The SLT operation implementation.
 * Takes two values from the stack, compares them and pushes the result
 * back to the stack.
 * The two values are considered signed.
 * The result is 1 if the first value is less than the second value,
 * 0 otherwise.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 */
__host__ __device__ int32_t SLT(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack);

/**
 * The SGT operation implementation.
 * Takes two values from the stack, compares them and pushes the result
 * back to the stack.
 * The two values are considered signed.
 * The result is 1 if the first value is greater than the second value,
 * 0 otherwise.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 */
__host__ __device__ int32_t SGT(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack);

/**
 * The EQ operation implementation.
 * Takes two values from the stack, compares them and pushes the result
 * back to the stack.
 * The result is 1 if the first value is equal to the second value,
 * 0 otherwise.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 */
__host__ __device__ int32_t EQ(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack);

/**
 * The ISZERO operation implementation.
 * Takes a value from the stack, compares it with zero and pushes the result
 * back to the stack.
 * The result is 1 if the value is equal to zero,
 * 0 otherwise.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 */
__host__ __device__ int32_t ISZERO(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack);
}  // namespace CuEVM::operations
