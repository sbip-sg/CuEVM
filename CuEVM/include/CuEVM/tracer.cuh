// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation
// Programme) Author: Stefan-Dan Ciocirlan Data: 2023-11-30
// SPDX-License-Identifier: MIT

#ifndef _CUEVM_TRACER_H_
#define _CUEVM_TRACER_H_

#include <cjson/cJSON.h>

#include <CuEVM/core/memory.cuh>
#include <CuEVM/core/message.cuh>
#include <CuEVM/core/return_data.cuh>
#include <CuEVM/core/stack.cuh>
#include <CuEVM/evm_call_state.cuh>
#include <CuEVM/state/touch_state.cuh>
#include <CuEVM/utils/arith.cuh>
namespace CuEVM::utils {
// PyObject* branches = PyList_New(0);
// PyObject* bugs = PyList_New(0);
// PyObject* calls = PyList_New(0);
// PyObject* storage_write = PyList_New(0);

// PyObject* tracer_json = PyList_New(0);
// PyObject* item = NULL;
// PyObject* stack_json = NULL;
// add sub mul div mod exp
// sstore
#define MAX_TRACE_EVENTS 1024
#define MAX_ADDRESSES_TRACING 32
#define MAX_CALLS_TRACING 32
#define MAX_BRANCHES_TRACING 32  // only track the latest 32 branches
struct simple_event_trace {
    // pc // op //  operand 1, operand 2, res
    uint32_t pc;
    uint8_t op;
    // uint8_t address_idx;
    // evm_word_t address; // temporarily disabled
    evm_word_t operand_1;
    evm_word_t operand_2;
    evm_word_t res;  // blank in some cases
};
struct call_trace {
    uint32_t pc;
    uint8_t op;
    // uint8_t address_idx;
    evm_word_t sender;
    evm_word_t receiver;
    evm_word_t value;
    uint8_t success;  // 0 or 1
    // todo add more depth + result etc
};
struct branch_trace {
    uint32_t pc_src;
    uint32_t pc_dst;
    evm_word_t distance;  // distance between pc_src and pc_dst

    // todo: use evm_word_t for distance
};

struct simplified_trace_data {
    simple_event_trace events[MAX_TRACE_EVENTS];
    // evm_word_t addresses[MAX_ADDRESSES_TRACING];
    call_trace calls[MAX_CALLS_TRACING];
    branch_trace branches[MAX_BRANCHES_TRACING];  // pc_src jump to pc_dest
    uint32_t no_addresses = 0;
    // uint32_t current_address_idx = 0;
    uint32_t no_events = 0;
    uint32_t no_calls = 0;
    uint32_t no_returns = 0;
    uint32_t no_branches = 0;
    evm_word_t last_distance;  // use to track branch distance by comparison opcodes

    __host__ __device__ void start_operation(const uint32_t pc, const uint8_t op, const CuEVM::evm_stack_t &stack_ptr);
    __host__ __device__ void finish_operation(const CuEVM::evm_stack_t &stack_ptr, uint32_t error_code);
    __host__ __device__ void start_call(uint32_t pc, evm_message_call_t *message_call_ptr);
    __host__ __device__ void finish_call(uint8_t success);
    __host__ __device__ void record_branch(uint32_t pc_src, uint32_t pc_dst);
    __host__ __device__ void record_distance(ArithEnv &arith, uint8_t op, const CuEVM::evm_stack_t &stack_ptr);
    __host__ __device__ void print();
};
struct trace_data_t {
    uint32_t pc;                      /**< The program counter */
    uint8_t op;                       /**< The opcode */
    evm_word_t gas;                   /**< Gas left before executing this operation */
    evm_word_t gas_cost;              /**< Gas cost of this operation */
    uint32_t mem_size;                /**< The size of the memory before op*/
    evm_word_t *stack;                /**< The stack before op*/
    uint32_t stack_size;              /**< The size of the stack before op*/
    uint32_t depth;                   /**< The depth of the call stack */
    CuEVM::byte_array_t *return_data; /**< The return data */
    evm_word_t refund;                /**< The gas refund */
#ifdef EIP_3155_OPTIONAL
    uint32_t error_code; /**< The error code */
    uint8_t *memory;     /**< The memory before op*/
// CuEVM::contract_storage_t storage; /**< The storage */
#endif

    __host__ cJSON *to_json();

    __host__ __device__ void print_err(char *hex_string_ptr = nullptr);
};

struct tracer_t {
    trace_data_t *data;              /**< The trace data */
    CuEVM::byte_array_t return_data; /**< The return data */
    CuEVM::evm_word_t gas_used;      /**< The gas used */
    uint32_t status;                 /**< The status of the trace */
    uint32_t size;                   /**< The size of the trace */
    uint32_t capacity;               /**< The capacity of the trace */

    __host__ __device__ tracer_t();

    __host__ __device__ ~tracer_t();

    __host__ __device__ void grow();

    __host__ __device__ uint32_t start_operation(ArithEnv &arith, const uint32_t pc, const uint8_t op,
                                                 const CuEVM::evm_memory_t &memory, const CuEVM::evm_stack_t &stack,
                                                 const uint32_t depth, const CuEVM::evm_return_data_t &return_data,
                                                 const bn_t &gas_limit, const bn_t &gas_used);

    __host__ __device__ void finish_operation(ArithEnv &arith, const uint32_t idx, const bn_t &gas_used,
                                              const bn_t &gas_refund
#ifdef EIP_3155_OPTIONAL
                                              ,
                                              const uint32_t error_code
// , const CuEVM::contract_storage_t &storage
#endif
    );

    __host__ __device__ void finish_transaction(ArithEnv &arith, const CuEVM::byte_array_t &return_data,
                                                const bn_t &gas_used, uint32_t error_code);

    __host__ __device__ void print(ArithEnv &arith);

    __host__ __device__ void print_err();

    __device__ void print_device_err();

    __host__ cJSON *to_json();
};
__device__ void print_device_data(tracer_t *device_tracer);

}  // namespace CuEVM::utils
// EIP-3155

#endif
