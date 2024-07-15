// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Date: 2023-07-15
// SPDX-License-Identifier: MIT

#ifndef _CUEVM_COMPARE_OP_H_
#define _CUEVM_COMPARE_OP_H_

#include "../utils/arith.cuh"
#include "../core/stack.cuh"

/**
 * The comparison operations class.
 * Contains the next operations 10s: Comparison Operations:
 * - LT
 * - GT
 * - SLT
 * - SGT
 * - EQ
 * - ISZERO
 */
namespace cuEVM::operations {
    /**
     * The LT operation implementation.
     * Takes two values from the stack, compares them and pushes the result
     * back to the stack.
     * The two values are considered unsigned.
     * The result is 1 if the first value is less than the second value,
     * 0 otherwise.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] pc The program counter.
     * @param[inout] stack The stack.
    */
    __host__ __device__ int32_t LT(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &pc,
        cuEVM::stack::evm_stack_t &stack);

    /**
     * The GT operation implementation.
     * Takes two values from the stack, compares them and pushes the result
     * back to the stack.
     * The two values are considered unsigned.
     * The result is 1 if the first value is greater than the second value,
     * 0 otherwise.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] pc The program counter.
     * @param[inout] stack The stack.
    */
    __host__ __device__ int32_t GT(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &pc,
        cuEVM::stack::evm_stack_t &stack);

    /**
     * The SLT operation implementation.
     * Takes two values from the stack, compares them and pushes the result
     * back to the stack.
     * The two values are considered signed.
     * The result is 1 if the first value is less than the second value,
     * 0 otherwise.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] pc The program counter.
     * @param[inout] stack The stack.
    */
    __host__ __device__ int32_t SLT(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &pc,
        cuEVM::stack::evm_stack_t &stack);

    /**
     * The SGT operation implementation.
     * Takes two values from the stack, compares them and pushes the result
     * back to the stack.
     * The two values are considered signed.
     * The result is 1 if the first value is greater than the second value,
     * 0 otherwise.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] pc The program counter.
     * @param[inout] stack The stack.
    */
    __host__ __device__ int32_t SGT(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &pc,
        cuEVM::stack::evm_stack_t &stack);

    /**
     * The EQ operation implementation.
     * Takes two values from the stack, compares them and pushes the result
     * back to the stack.
     * The result is 1 if the first value is equal to the second value,
     * 0 otherwise.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] pc The program counter.
     * @param[inout] stack The stack.
    */
    __host__ __device__ int32_t EQ(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &pc,
        cuEVM::stack::evm_stack_t &stack);

    /**
     * The ISZERO operation implementation.
     * Takes a value from the stack, compares it with zero and pushes the result
     * back to the stack.
     * The result is 1 if the value is equal to zero,
     * 0 otherwise.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] pc The program counter.
     * @param[inout] stack The stack.
    */
    __host__ __device__ int32_t ISZERO(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &pc,
        cuEVM::stack::evm_stack_t &stack);
}

#endif