// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#ifndef _CUEVM_FLOW_OP_H_
#define _CUEVM_FLOW_OP_H_

#include "../core/stack.cuh"
#include "../core/message.cuh"
#include "../core/jump_destinations.cuh"
#include "../utils/arith.cuh"


/**
 * 50s: Flow Operations:
 * - JUMP
 * - JUMPI
 * - PC
 * - GAS
 * - JUMPDEST
 */
namespace cuEVM::operations {
    /**
     * The JUMP operation implementation.
     * Takes the destination from the stack and sets the program counter
     * to the destination if it is a valid jump destination.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] pc The program counter.
     * @param[in] stack The stack.
     * @param[in] message The message.
     * @return 0 if the operation was successful, an error code otherwise.
     */
    __host__ __device__ int32_t JUMP(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &pc,
        cuEVM::stack::evm_stack_t &stack,
        const cuEVM::evm_message_call_t &message);
    /**
     * The JUMPI operation implementation.
     * Takes the destination and the condition from the stack and sets the program counter
     * to the destination if it is a valid jump destination and the condition is not 0.
     * If the condition is 0 the program counter is incremented by 1.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] pc The program counter.
     * @param[in] stack The stack.
     * @param[in] message The message.
     * @return 0 if the operation was successful, an error code otherwise.
     */
    __host__ __device__ int32_t JUMPI(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &pc,
        cuEVM::stack::evm_stack_t &stack,
        const cuEVM::evm_message_call_t &message);

    /**
     * The PC operation implementation.
     * Pushes the program counter to the stack.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[in] pc The program counter.
     * @param[out] stack The stack.
     * @return 0 if the operation was successful, an error code otherwise.
     */
    __host__ __device__ int32_t PC(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        const uint32_t &pc,
        cuEVM::stack::evm_stack_t &stack);

    /**
     * The GAS operation implementation.
     * Pushes the gas left to the stack after this operation.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] stack The stack.
     * @return 0 if the operation was successful, an error code otherwise.
     */
    __host__ __device__ int32_t GAS(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        cuEVM::stack::evm_stack_t &stack);

    /**
     * The JUMPDEST operation implementation.
     * It increments the program counter by 1.
     * It is used as a valid jump destination.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @return 0 if the operation was successful, an error code otherwise.
     */
    __host__ __device__ int32_t JUMPDEST(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used);
}

#endif