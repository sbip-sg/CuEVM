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
/**
 * @brief Print tracer data from the host buffer
 * @param[in] h_buffer The host buffer containing tracer data
 */
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

/**
 * @brief Structure to store tracing data for a single EVM operation
 */
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

    /**
     * @brief Convert trace data to JSON format
     * @return JSON representation of the trace data
     */
    __host__ cJSON *to_json();

    /**
     * @brief Print EIP3155 format trace
     */
    __device__ void print_err();

    /**
     * @brief Append trace data to a buffer
     * @param[in,out] buffer The buffer to append to
     * @param[in,out] offset The current offset in the buffer
     */
    __device__ void append_to_buffer(char *buffer, size_t &offset);

    /**
     * @brief Serialize trace data to a buffer
     * @param[in,out] buf The buffer to serialize to
     * @param[in] buf_size The size of the buffer
     * @param[in,out] offset The current offset in the buffer
     * @return true if serialization was successful, false otherwise
     */
    __device__ bool serialize(char *buf, uint32_t buf_size, uint32_t &offset);
};

/**
 * @brief Tracer for collecting execution trace during EVM execution
 */
struct tracer_t {
    trace_data_t *data;        /**< The trace data */
    byte_array_t *return_data; /**< The return data */
    gas_t gas_used;            /**< The gas used */
    uint32_t status;           /**< The status of the trace */
    uint32_t size;             /**< The size of the trace */
    uint32_t capacity;         /**< The capacity of the trace */

    /**
     * @brief Construct a new tracer object
     */
    __device__ tracer_t();

    /**
     * @brief Destroy the tracer object
     */
    __device__ ~tracer_t();

    /**
     * @brief Grow the capacity of the tracer
     */
    __device__ void grow();

    /**
     * @brief Record the start of an EVM operation
     * @param[in] pc The program counter
     * @param[in] op The opcode
     * @param[in] memory The EVM memory
     * @param[in] stack The EVM stack
     * @param[in] depth The call depth
     * @param[in] return_data The return data
     * @param[in] gas_limit The gas limit
     * @param[in] gas_used The gas used so far
     */
    __device__ void start_operation(const uint32_t pc, const uint8_t op, const CuEVM::evm_memory_t *memory,
                                    const CuEVM::evm_stack_t *stack, const uint32_t depth,
                                    const byte_array_t *return_data, const CuEVM::gas_t &gas_limit,
                                    const CuEVM::gas_t &gas_used);

    /**
     * @brief Record the finish of an EVM operation
     * @param[in] gas_used The gas used by the operation
     * @param[in] gas_refund The gas refund from the operation
     */
    __device__ void finish_operation(const gas_t &gas_used, const gas_t &gas_refund);

    /**
     * @brief Record the finish of a transaction
     * @param[in] return_data The return data
     * @param[in] gas_used The total gas used
     * @param[in] error_code The error code
     */
    __device__ void finish_transaction(const byte_array_t *return_data, const gas_t &gas_used, uint32_t error_code);

    /**
     * @brief Print the human readable trace
     *
     */
    __device__ void print();

    /**
     * @brief Print the EIP3155 format trace
     */
    __device__ void print_err();

    /**
     * @brief Serialize the trace to a buffer
     * @param[in,out] buf The buffer to serialize to
     * @param[in] buf_size The size of the buffer
     * @param[in,out] offset The starting offset in the buffer
     * @return true if serialization was successful, false otherwise
     */
    __device__ bool serialize(char *buf, uint32_t buf_size, uint32_t &offset);
};

}  // namespace CuEVM::utils
// EIP-3155
