#pragma once
#include <CuEVM/core/stack.cuh>

/**
 * The bitwise operations class.
 * Contains the next operations 10s: Bitwise Logic Operations:
 * - AND
 * - OR
 * - XOR
 * - NOT
 * - BYTE
 * - SHL
 * - SHR
 * - SAR
 */
namespace CuEVM::operations {
/**
 * The AND operation implementation.
 * Takes two values from the stack, performs a bitwise AND operation
 * and pushes the result back to the stack.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 */
__host__ __device__ int32_t AND(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack);

/**
 * The OR operation implementation.
 * Takes two values from the stack, performs a bitwise OR operation
 * and pushes the result back to the stack.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 */
__host__ __device__ int32_t OR(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack);

/**
 * The XOR operation implementation.
 * Takes two values from the stack, performs a bitwise XOR operation
 * and pushes the result back to the stack.
 * @param[in] arith The arithmetical environment.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 */
__host__ __device__ int32_t XOR(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack);

/**
 * The NOT operation implementation.
 * Takes a value from the stack, performs a bitwise NOT operation
 * and pushes the result back to the stack.
 * Similar operation with XOR with only ones.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 */
__host__ __device__ int32_t NOT(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack);

/**
 * The BYTE operation implementation.
 * Takes two values from the stack. The first value is the index of the byte
 * to be extracted from the second value. The operation pushes the byte
 * back to the stack.
 * If the index is out of range, the operation pushes 0 to the stack.
 * The most significat byte has index 0.
 * @param[in] arith The arithmetical environment.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 */
__host__ __device__ int32_t BYTE(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack);

/**
 * The SHL operation implementation.
 * Takes two values from the stack. The first value is the number of bits
 * to shift the second value to the left. The operation pushes the result
 * back to the stack.
 * If the number of bits is out of range, the operation pushes 0 to the stack.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 */
__host__ __device__ int32_t SHL(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack);

/**
 * The SHR operation implementation.
 * Takes two values from the stack. The first value is the number of bits
 * to shift the second value to the right. The operation pushes the result
 * back to the stack.
 * If the number of bits is out of range, the operation pushes 0 to the stack.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 */
__host__ __device__ int32_t SHR(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack);

/**
 * The SAR operation implementation.
 * Takes two values from the stack. The first value is the number of bits
 * to arithmetic shift the second value to the right.
 * The operation pushes the result back to the stack.
 * If the number of bits is out of range, the operations arithmetic shift
 * with the maximum number of bits.
 * The first value is considered unsigned and the second value is considered
 * signed.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 */
__host__ __device__ int32_t SAR(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack);
}  // namespace CuEVM::operations
