#pragma once

#include <CuEVM/core/jump_destinations.cuh>
#include <CuEVM/state/logs.cuh>
#include <CuEVM/state/state_db.cuh>
#include <CuEVM/utils/evm_defines.cuh>

namespace CuEVM {

/**
 * @brief Cached version of the EVM call context for efficient access during execution
 *
 * This structure maintains a local cache of essential execution state information
 * such as program counter, gas tracking, stack pointer, and bytecode. It provides
 * methods for transferring data between the cache and the actual state, as well as
 * utility functions for managing the bytecode.
 */
struct cached_evm_call_context {
    uint32_t pc;                   /**< The program counter */
    gas_t gas_used;                /**< The gas used so far in execution */
    gas_t gas_limit;               /**< The maximum gas allowed for execution */
    CuEVM::evm_stack_t* stack_ptr; /**< Pointer to the execution stack */
    uint32_t byte_code_size;       /**< The size of the contract bytecode */
    uint8_t* byte_code_data;       /**< Pointer to the contract bytecode */

    /**
     * @brief Construct a new cached context from an existing state
     * @param[in] state Pointer to the original EVM call context to cache
     */
    __device__ cached_evm_call_context(evm_call_context_t* state);  // copy from state to cache

    /**
     * @brief Default constructor
     */
    __device__ cached_evm_call_context() {};

    /**
     * @brief Set the bytecode from a byte array
     * @param[in] byte_code Pointer to the byte array containing the bytecode
     */
    __device__ void set_byte_code(const byte_array_t* byte_code);

    /**
     * @brief Set the bytecode directly from raw data
     * @param[in] byte_code Pointer to the bytecode data
     * @param[in] size Size of the bytecode
     */
    __device__ void set_byte_code(uint8_t* byte_code, uint32_t size);

    /**
     * @brief Write the cached state back to the original context
     * @param[out] state Pointer to the EVM call context to update
     */
    __device__ void write_cache_to_state(evm_call_context_t* state);  // copy from cache to state

    /**
     * @brief Print the cached context information for debugging
     */
    __device__ void print() const;
};
}  // namespace CuEVM
