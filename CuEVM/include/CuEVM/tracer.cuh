// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#ifndef _CUEVM_TRACER_H_
#define _CUEVM_TRACER_H_

#include <CuEVM/utils/arith.cuh>
#include <CuEVM/core/message.cuh>
#include <CuEVM/core/stack.cuh>
#include <CuEVM/core/memory.cuh>
#include <CuEVM/core/return_data.cuh>
#include <CuEVM/state/touch_state.cuh>

#include <cjson/cJSON.h>

namespace CuEVM::utils {
    struct trace_data_t {
        uint32_t pc; /**< The program counter */
        uint8_t op; /**< The opcode */
        evm_word_t gas; /**< Gas left before executing this operation */
        evm_word_t gas_cost; /**< Gas cost of this operation */
        uint32_t mem_size; /**< The size of the memory before op*/
        evm_word_t *stack; /**< The stack before op*/
        uint32_t stack_size; /**< The size of the stack before op*/
        uint32_t depth; /**< The depth of the call stack */
        CuEVM::byte_array_t* return_data; /**< The return data */
        evm_word_t refund; /**< The gas refund */
        #ifdef EIP_3155_OPTIONAL
        uint32_t error_code; /**< The error code */
        uint8_t *memory; /**< The memory before op*/
        CuEVM::TouchState touch_state; /**< The touch state */
        #endif

        __host__ cJSON* to_json();

        __host__ void print_err(char *hex_string_ptr = nullptr);
    };

    struct tracer_t {
        trace_data_t *data; /**< The trace data */
        CuEVM::byte_array_t return_data; /**< The return data */
        CuEVM::evm_word_t gas_used; /**< The gas used */
        uint32_t status; /**< The status of the trace */
        uint32_t size; /**< The size of the trace */
        uint32_t capacity; /**< The capacity of the trace */

        __host__ __device__ tracer_t();

        __host__ __device__ ~tracer_t();

        __host__ __device__ void grow();

        __host__ __device__ uint32_t start_operation(
            ArithEnv &arith,
            const uint32_t pc,
            const uint8_t op,
            const CuEVM::evm_memory_t &memory,
            const CuEVM::evm_stack_t &stack,
            const uint32_t depth,
            const CuEVM::evm_return_data_t &return_data,
            const bn_t &gas_limit,
            const bn_t &gas_used
        );

        __host__ __device__ void finish_operation(
            ArithEnv &arith,
            const uint32_t idx,
            const bn_t &gas_used,
            const bn_t &gas_refund
            #ifdef EIP_3155_OPTIONAL
            , const uint32_t error_code,
            const CuEVM::TouchState &touch_state
            #endif
        );

        __host__ __device__ void finish_transaction(
            ArithEnv &arith,
            const CuEVM::byte_array_t &return_data,
            const bn_t &gas_used,
            uint32_t error_code
        );

        __host__ __device__ void print(ArithEnv &arith);

        __host__ void print_err();

        __host__ cJSON* to_json();
    };

}
// EIP-3155

#endif
