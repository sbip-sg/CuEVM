// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Date: 2024-07-15
// SPDX-License-Identifier: MIT

#ifndef _CUEVM_ARITHMETIC_OP_H_
#define _CUEVM_ARITHMETIC_OP_H_

#include "../utils/arith.cuh"
#include "../core/stack.cuh"

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
 */
namespace cuEVM {
    namespace operations {
        /**
         * The ADD operation implementation.
         * Takes two values from the stack, adds them and
         * pushes the result back to the stack.
         * @param[in] arith The arithmetical environment.
         * @param[in] gas_limit The gas limit.
         * @param[inout] gas_used The gas used.
         * @param[inout] stack The stack.
         * @return The error code. 0 if no error.
        */
        __host__ __device__ int32_t ADD(
            ArithEnv &arith,
            const bn_t &gas_limit,
            bn_t &gas_used,
            cuEVM::stack::evm_stack_t &stack);

        /**
         * The MUL operation implementation.
         * Takes two values from the stack, multiplies them and
         * pushes the result back to the stack.
         * @param[in] arith The arithmetical environment.
         * @param[in] gas_limit The gas limit.
         * @param[inout] gas_used The gas used.
         * @param[inout] stack The stack.
         * @return The error code. 0 if no error.
        */
        __host__ __device__ int32_t MUL(
            ArithEnv &arith,
            const bn_t &gas_limit,
            bn_t &gas_used,
            cuEVM::stack::evm_stack_t &stack);

        /**
         * The SUB operation implementation.
         * Takes two values from the stack, subtracts the second value
         * from the first value and pushes the result back to the stack.
         * @param[in] arith The arithmetical environment.
         * @param[in] gas_limit The gas limit.
         * @param[inout] gas_used The gas used.
         * @param[inout] stack The stack.
         * @return The error code. 0 if no error.
        */
        __host__ __device__ int32_t SUB(
            ArithEnv &arith,
            const bn_t &gas_limit,
            bn_t &gas_used,
            cuEVM::stack::evm_stack_t &stack);

        /**
         * The DIV operation implementation.
         * Takes two values from the stack, divides the first value
         * by the second value and pushes the result back to the stack.
         * It consider the division by zero as zero.
         * Both values are considered unsigned.
         * @param[in] arith The arithmetical environment.
         * @param[in] gas_limit The gas limit.
         * @param[inout] gas_used The gas used.
         * @param[inout] stack The stack.
         * @return The error code. 0 if no error.
        */
        __host__ __device__ int32_t DIV(
            ArithEnv &arith,
            const bn_t &gas_limit,
            bn_t &gas_used,
            cuEVM::stack::evm_stack_t &stack);

        /**
         * The SDIV operation implementation.
         * Takes two values from the stack, divides the first value
         * by the second value and pushes the result back to the stack.
         * It consider the division by zero as zero.
         * The special case -2^254 / -1 = -2^254 is considered.
         * Both values are considered signed.
         * @param[in] arith The arithmetical environment.
         * @param[in] gas_limit The gas limit.
         * @param[inout] gas_used The gas used.
         * @param[inout] stack The stack.
         * @return The error code. 0 if no error.
        */
        __host__ __device__ int32_t SDIV(
            ArithEnv &arith,
            const bn_t &gas_limit,
            bn_t &gas_used,
            cuEVM::stack::evm_stack_t &stack);
        /**
         * The MOD operation implementation.
         * Takes two values from the stack, calculates the remainder
         * of the division of the first value by the second value and
         * pushes the result back to the stack.
         * It consider the remainder of division by zero as zero.
         * Both values are considered unsigned.
         * @param[in] arith The arithmetical environment.
         * @param[in] gas_limit The gas limit.
         * @param[inout] gas_used The gas used.
         * @param[inout] stack The stack.
         * @return The error code. 0 if no error.
        */
        __host__ __device__ int32_t MOD(
            ArithEnv &arith,
            const bn_t &gas_limit,
            bn_t &gas_used,
            cuEVM::stack::evm_stack_t &stack);

        /**
         * The SMOD operation implementation.
         * Takes two values from the stack, calculates the remainder
         * of the division of the first value by the second value and
         * pushes the result back to the stack.
         * It consider the remainder of division by zero as zero.
         * Both values are considered signed.
         * @param[in] arith The arithmetical environment.
         * @param[in] gas_limit The gas limit.
         * @param[inout] gas_used The gas used.
         * @param[inout] stack The stack.
         * @return The error code. 0 if no error.
        */
        __host__ __device__ int32_t SMOD(
            ArithEnv &arith,
            const bn_t &gas_limit,
            bn_t &gas_used,
            cuEVM::stack::evm_stack_t &stack);

        /**
         * The ADDMOD operation implementation.
         * Takes three values from the stack, adds the first two values,
         * calculates the remainder of the division of the result by the third value
         * and pushes the remainder back to the stack.
         * It consider the remainder of division by zero or one as zero.
         * All values are considered unsigned.
         * @param[in] arith The arithmetical environment.
         * @param[in] gas_limit The gas limit.
         * @param[inout] gas_used The gas used.
         * @param[inout] stack The stack.
         * @return The error code. 0 if no error.
        */
        __host__ __device__ int32_t ADDMOD(
            ArithEnv &arith,
            const bn_t &gas_limit,
            bn_t &gas_used,
            cuEVM::stack::evm_stack_t &stack);

        /**
         * The MULMOD operation implementation.
         * Takes three values from the stack, multiplies the first two values,
         * calculates the remainder of the division of the result by the third value
         * and pushes the remainder back to the stack.
         * It consider the remainder of division by zero or one as zero.
         * All values are considered unsigned.
         * The first two values goes though a modulo by the third value
         * before multiplication.
         * @param[in] arith The arithmetical environment.
         * @param[in] gas_limit The gas limit.
         * @param[inout] gas_used The gas used.
         * @param[inout] stack The stack.
         * @return The error code. 0 if no error.
        */
        __host__ __device__ int32_t MULMOD(
            ArithEnv &arith,
            const bn_t &gas_limit,
            bn_t &gas_used,
            cuEVM::stack::evm_stack_t &stack);
        /**
         * The EXP operation implementation.
         * Takes two values from the stack, calculates the first value
         * to the power of the second value and pushes the result back to the stack.
         * It consider the power of zero as one, even if the base is zero.
         * The dynamic gas cost is calculated based on the minimumu number of bytes
         * to store the exponent value.
         * @param[in] arith The arithmetical environment.
         * @param[inout] gas_limit The gas limit.
         * @param[inout] gas_used The gas used.
         * @param[inout] stack The stack.
         * @return The error code. 0 if no error.
        */
        __host__ __device__ int32_t EXP(
            ArithEnv &arith,
            const bn_t &gas_limit,
            bn_t &gas_used,
            cuEVM::stack::evm_stack_t &stack);

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
         * @param[in] arith The arithmetical environment.
         * @param[inout] gas_limit The gas limit.
         * @param[inout] gas_used The gas used.
         * @param[inout] stack The stack.
         * @return The error code. 0 if no error.
        */
        __host__ __device__ int32_t SIGNEXTEND(
            ArithEnv &arith,
            const bn_t &gas_limit,
            bn_t &gas_used,
            cuEVM::stack::evm_stack_t &stack);
    }
}

#endif