// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#ifndef _CUEVM_LOG_OP_H_
#define _CUEVM_LOG_OP_H_

#include "../core/stack.cuh"
#include "../core/message.cuh"
#include "../core/memory.cuh"
#include "../state/logs.cuh"
#include "../utils/arith.cuh"

/**
 * a0s: Logging Operations:
 * - LOGX
 */
namespace cuEVM::operations {
    /**
     * The LOGX operation implementation.
     * Takes the memory offset, the memory length and the topics from the stack and
     * stores the memory data togheter with the topics in the logs.
     * Adittional gas cost is added for the memory expansion and for the topics.
     * Every byte of memory costs additional GAS_LOG_DATA gas.
     * Every topic costs additional GAS_LOG_TOPIC gas.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[in] stack The stack.
     * @param[in] memory The memory.
     * @param[in] message The message that started the execution.
     * @param[out] log_state The logs state.
     * @param[in] opcode The opcode.
     * @return 0 if the operation was successful, an error code otherwise.
    */
    __host__ __device__ int32_t LOGX(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        cuEVM::evm_stack_t &stack,
        cuEVM::evm_memory_t &memory,
        const cuEVM::evm_message_call_t &message,
        cuEVM::state::log_state_data_t &log_state,
        const uint8_t &opcode);  
}

#endif