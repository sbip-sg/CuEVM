#pragma once
#include <cjson/cJSON.h>

#include <CuEVM/core/data_structures.cuh>
#include <CuEVM/core/evm_call_context.cuh>
#include <CuEVM/core/memory.cuh>
#include <CuEVM/core/stack.cuh>
#include <CuEVM/utils/error_codes.cuh>
#include <CuEVM/utils/opcodes.cuh>
#include <iostream>
#include <string>

namespace CuEVM::utils {
void print_tracer_data(char *h_buffer);
// PyObject* branches = PyList_New(0);
// PyObject* bugs = PyList_New(0);
// PyObject* calls = PyList_New(0);
// PyObject* storage_write = PyList_New(0);

// PyObject* tracer_json = PyList_New(0);
// PyObject* item = NULL;
// PyObject* stack_json = NULL;
// add sub mul div mod exp
// sstore

struct trace_data_t {
    uint32_t pc;               /**< The program counter */
    uint8_t op;                /**< The opcode */
    gas_t gas;                 /**< Gas left before executing this operation */
    gas_t gas_cost;            /**< Gas cost of this operation */
    uint32_t mem_size;         /**< The size of the memory before op*/
    evm_word_t *stack;         /**< The stack before op*/
    uint32_t stack_size;       /**< The size of the stack before op*/
    uint32_t depth;            /**< The depth of the call stack */
    byte_array_t *return_data; /**< The return data */
    gas_t refund;              /**< The gas refund */

    __host__ cJSON *to_json();

    __device__ void print_err();
    __device__ void append_to_buffer(char *buffer, size_t &offset);
    __device__ bool serialize(char *buf, uint32_t buf_size, uint32_t &offset);
};

struct tracer_t {
    trace_data_t *data;        /**< The trace data */
    byte_array_t *return_data; /**< The return data */
    gas_t gas_used;            /**< The gas used */
    uint32_t status;           /**< The status of the trace */
    uint32_t size;             /**< The size of the trace */
    uint32_t capacity;         /**< The capacity of the trace */

    __device__ tracer_t();

    __device__ ~tracer_t();

    __device__ void grow();

    __device__ void start_operation(const uint32_t pc, const uint8_t op, const CuEVM::evm_memory_t *memory,
                                    const CuEVM::evm_stack_t *stack, const uint32_t depth,
                                    const byte_array_t *return_data, const CuEVM::gas_t &gas_limit,
                                    const CuEVM::gas_t &gas_used);

    __device__ void finish_operation(const gas_t &gas_used, const gas_t &gas_refund);

    __device__ void finish_transaction(const byte_array_t *return_data, const gas_t &gas_used, uint32_t error_code);

    __device__ void print();

    __device__ void print_err();

    __device__ void print_device_err();
    __device__ bool serialize(char *buf, uint32_t buf_size, uint32_t &offset);
};
__device__ void print_device_data(tracer_t *device_tracer);

}  // namespace CuEVM::utils
// EIP-3155
