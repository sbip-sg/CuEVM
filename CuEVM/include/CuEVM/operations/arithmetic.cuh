#pragma once
#include <CuEVM/core/stack.cuh>

/**
 * The arithmetic operations.
 * Contains the arithmetic operations 0s: Arithmetic Operations:
 * - ADD
 * - MUL
 * - SUB
 * - DIV
 * - SDIV
 * - MOD
 * - SMOD
 * - ADDMOD
 * - MULMOD
 * - EXP
 * - SIGNEXTEND
 * - AND
 * - OR
 * - XOR
 * - NOT
 * - BYTE
 * - SHL
 * - SHR
 * - SAR
 * - LT
 * - GT
 * - SLT
 * - SGT
 * - EQ
 * - ISZERO
 */
namespace CuEVM::operations {
/**
 * The ADD operation implementation.
 * Takes two values from the stack, adds them and
 * pushes the result back to the stack.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 * @return The error code. 0 if no error.
 */
__device__ int32_t ADD(const gas_t &gas_limit, gas_t &gas_used, evm_stack_t *stack);

/**
 * The MUL operation implementation.
 * Takes two values from the stack, multiplies them and
 * pushes the result back to the stack.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 * @return The error code. 0 if no error.
 */
__device__ int32_t MUL(const gas_t &gas_limit, gas_t &gas_used, evm_stack_t *stack);

/**
 * The SUB operation implementation.
 * Takes two values from the stack, subtracts the second value
 * from the first value and pushes the result back to the stack.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 * @return The error code. 0 if no error.
 */
__device__ int32_t SUB(const gas_t &gas_limit, gas_t &gas_used, evm_stack_t *stack);

/**
 * The DIV operation implementation.
 * Takes two values from the stack, divides the first value
 * by the second value and pushes the result back to the stack.
 * It consider the division by zero as zero.
 * Both values are considered unsigned.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 * @return The error code. 0 if no error.
 */
__device__ int32_t DIV(const gas_t &gas_limit, gas_t &gas_used, evm_stack_t *stack);

/**
 * The SDIV operation implementation.
 * Takes two values from the stack, divides the first value
 * by the second value and pushes the result back to the stack.
 * It consider the division by zero as zero.
 * The special case -2^254 / -1 = -2^254 is considered.
 * Both values are considered signed.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 * @return The error code. 0 if no error.
 */
__device__ int32_t SDIV(const gas_t &gas_limit, gas_t &gas_used, evm_stack_t *stack);
/**
 * The MOD operation implementation.
 * Takes two values from the stack, calculates the remainder
 * of the division of the first value by the second value and
 * pushes the result back to the stack.
 * It consider the remainder of division by zero as zero.
 * Both values are considered unsigned.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 * @return The error code. 0 if no error.
 */
__device__ int32_t MOD(const gas_t &gas_limit, gas_t &gas_used, evm_stack_t *stack);

/**
 * The SMOD operation implementation.
 * Takes two values from the stack, calculates the remainder
 * of the division of the first value by the second value and
 * pushes the result back to the stack.
 * It consider the remainder of division by zero as zero.
 * Both values are considered signed.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 * @return The error code. 0 if no error.
 */
__device__ int32_t SMOD(const gas_t &gas_limit, gas_t &gas_used, evm_stack_t *stack);

/**
 * The ADDMOD operation implementation.
 * Takes three values from the stack, adds the first two values,
 * calculates the remainder of the division of the result by the third value
 * and pushes the remainder back to the stack.
 * It consider the remainder of division by zero or one as zero.
 * All values are considered unsigned.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 * @return The error code. 0 if no error.
 */
__device__ int32_t ADDMOD(const gas_t &gas_limit, gas_t &gas_used, evm_stack_t *stack);

/**
 * The MULMOD operation implementation.
 * Takes three values from the stack, multiplies the first two values,
 * calculates the remainder of the division of the result by the third value
 * and pushes the remainder back to the stack.
 * It consider the remainder of division by zero or one as zero.
 * All values are considered unsigned.
 * The first two values goes though a modulo by the third value
 * before multiplication.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 * @return The error code. 0 if no error.
 */
__device__ int32_t MULMOD(const gas_t &gas_limit, gas_t &gas_used, evm_stack_t *stack);
/**
 * The EXP operation implementation.
 * Takes two values from the stack, calculates the first value
 * to the power of the second value and pushes the result back to the stack.
 * It consider the power of zero as one, even if the base is zero.
 * The dynamic gas cost is calculated based on the minimumu number of bytes
 * to store the exponent value.
 * @param[inout] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 * @return The error code. 0 if no error.
 */
__device__ int32_t EXP(const gas_t &gas_limit, gas_t &gas_used, evm_stack_t *stack);

/**
 * The SIGNEXTEND operation implementation.
 * Takes two values from the stack. It consider the second value
 * to have the size of the number of bytes given by the first value (b) + 1.
 * The operation sign extends the second value to the full size of the
 * aithmetic environment and pushes the result back to the stack.
 * In case the first value is out of range ((b+1) > BYTES) the operation
 * pushes the second value back to the stack.
 * If the second value has more bytes than the value (b+1),
 * the operation consider only the least significant (b+1) bytes
 * of the second value.
 * @param[inout] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 * @return The error code. 0 if no error.
 */
__device__ int32_t SIGNEXTEND(const gas_t &gas_limit, gas_t &gas_used, evm_stack_t *stack);
/**
 * The AND operation implementation.
 * Takes two values from the stack, performs a bitwise AND operation
 * and pushes the result back to the stack.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 */
__device__ int32_t AND(const gas_t &gas_limit, gas_t &gas_used, evm_stack_t *stack);

/**
 * The OR operation implementation.
 * Takes two values from the stack, performs a bitwise OR operation
 * and pushes the result back to the stack.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 */
__device__ int32_t OR(const gas_t &gas_limit, gas_t &gas_used, evm_stack_t *stack);

/**
 * The XOR operation implementation.
 * Takes two values from the stack, performs a bitwise XOR operation
 * and pushes the result back to the stack.
 * @param[in] arith The arithmetical environment.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 */
__device__ int32_t XOR(const gas_t &gas_limit, gas_t &gas_used, evm_stack_t *stack);

/**
 * The NOT operation implementation.
 * Takes a value from the stack, performs a bitwise NOT operation
 * and pushes the result back to the stack.
 * Similar operation with XOR with only ones.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 */
__device__ int32_t NOT(const gas_t &gas_limit, gas_t &gas_used, evm_stack_t *stack);

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
__device__ int32_t BYTE(const gas_t &gas_limit, gas_t &gas_used, evm_stack_t *stack);

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
__device__ int32_t SHL(const gas_t &gas_limit, gas_t &gas_used, evm_stack_t *stack);

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
__device__ int32_t SHR(const gas_t &gas_limit, gas_t &gas_used, evm_stack_t *stack);

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
__device__ int32_t SAR(const gas_t &gas_limit, gas_t &gas_used, evm_stack_t *stack);

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
__device__ int32_t LT(const gas_t &gas_limit, gas_t &gas_used, evm_stack_t *stack);

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
__device__ int32_t GT(const gas_t &gas_limit, gas_t &gas_used, evm_stack_t *stack);

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
__device__ int32_t SLT(const gas_t &gas_limit, gas_t &gas_used, evm_stack_t *stack);

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
__device__ int32_t SGT(const gas_t &gas_limit, gas_t &gas_used, evm_stack_t *stack);

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
__device__ int32_t EQ(const gas_t &gas_limit, gas_t &gas_used, evm_stack_t *stack);

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
__device__ int32_t ISZERO(const gas_t &gas_limit, gas_t &gas_used, evm_stack_t *stack);

}  // namespace CuEVM::operations
