#pragma once

#include <CuEVM/core/jump_destinations.cuh>
#include <CuEVM/core/memory.cuh>
#include <CuEVM/core/stack.cuh>
#include <CuEVM/state/logs.cuh>
#include <CuEVM/state/state_db.cuh>
#include <CuEVM/utils/evm_defines.cuh>

namespace CuEVM {

// pc, gas_used, gas_limit, stack_ptr, bytecode should be in local or shared memory
struct cached_evm_call_context {
    uint32_t pc;                   /**< The program counter */
    gas_t gas_used;                /**< The gas */
    gas_t gas_limit;               /**< The gas limit */
    CuEVM::evm_stack_t* stack_ptr; /**< The stack */
    uint32_t byte_code_size;       /**< The size of the byte code */
    uint8_t* byte_code_data;       /**< The byte code */

    __device__ cached_evm_call_context(evm_call_context_t* state);  // copy from state to cache
    __device__ cached_evm_call_context() {};
    __device__ void set_byte_code(const byte_array_t* byte_code);
    __device__ void set_byte_code(uint8_t* byte_code, uint32_t size);
    __device__ void write_cache_to_state(evm_call_context_t* state);  // copy from cache to state
    __device__ void print() const;
};
}  // namespace CuEVM
