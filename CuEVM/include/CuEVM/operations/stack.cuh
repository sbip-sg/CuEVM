#pragma once
#include <CuEVM/core/stack.cuh>

/**
 * The stack operations class.
 * Contains all the operations that can be performed on the stack.
 * - POP 50: POP
 * - PUSH0 5F: PUSH0
 * - PUSHX 60s & 70s: Push Operations
 * - DUPX 80s: Duplication Operations
 * - SWAPX 90s: Exchange Operations
 */
namespace CuEVM::operations {
/**
 * The POP operation implementation.
 * It pops the top element from the stack.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[out] stack The stack.
 */
__device__ int32_t POP(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack);

/**
 * The PUSH0 operation implementation.
 * Pushes a zero value to the stack.
 * @param[inout] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 */
__device__ int32_t PUSH0(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack);

/**
 * The PUSHX operation implementation.
 * Pushes a value from the bytecode to the stack.
 * If the bytecode is not long enough to provide the value,
 * the operation completes with zero bytes for the least
 * significant bytes.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] pc The program counter.
 * @param[inout] stack The stack.
 * @param[in] byte_code The bytecode.
 * @param[in] opcode The opcode.
 */
__device__ int32_t PUSHX(const gas_t &gas_limit, gas_t &gas_used, uint32_t &pc, CuEVM::evm_stack_t *stack,
                         uint8_t *byte_code, uint32_t byte_code_size, const uint8_t &opcode);

/**
 * The DUPX operation implementation.
 * Duplicates a value from the stack and pushes it back to the stack.
 * The value to be duplicated is given by the opcode.
 * The opcode is in the range 0x80 - 0x8F.
 * The index of the value to be duplicated is given
 * by the opcode - 0x80 + 1.
 * We consider the least 4 significant bits of the opcode + 1.
 * The index is from the top of the stack. 1 is the top of the stack.
 * 2 is the second value from the top of the stack and so on.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 * @param[in] opcode The opcode.
 */
__device__ int32_t DUPX(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack, const uint8_t &opcode);

/**
 * The SWAPX operation implementation.
 * Swaps the top of the stack with a value from the stack.
 * The index of the value to be swapped is given
 * by the opcode - 0x90 + 1.
 * We consider the least 4 significant bits of the opcode + 1.
 * The index is starts from the first value under the top of the stack.
 * 1 is the second value from the top of the stack, 2 is the third value
 * from the top of the stack, 2 is the third value
 * from the top of the stack and so on.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 * @param[in] opcode The opcode.
 */
__device__ int32_t SWAPX(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack, const uint8_t &opcode);
}  // namespace CuEVM::operations
