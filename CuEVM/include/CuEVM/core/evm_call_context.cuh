#pragma once

#include <CuEVM/core/jump_destinations.cuh>
#include <CuEVM/core/memory.cuh>
#include <CuEVM/core/return_data.cuh>
#include <CuEVM/core/stack.cuh>
#include <CuEVM/state/logs.cuh>
#include <CuEVM/state/state_db.cuh>
#include <CuEVM/utils/evm_defines.cuh>

namespace CuEVM {

struct evm_call_context_t {
    bool static_env;                 /**< The static flag (STATICCALL) YP: \f$w\f$ */
    uint32_t depth;                  /**< The depth of the state */
    uint32_t pc;                     /**< The program counter */
    gas_t gas_used;                  /**< The gas */
    gas_t gas_refund;                /**< The gas refund */
    gas_t gas_limit;                 /**< The gas limit */
    CuEVM::evm_stack_t* stack_ptr;   /**< The stack */
    CuEVM::evm_memory_t* memory_ptr; /**< The memory */

    evm_word_t from;
    evm_word_t to;
    evm_word_t storage_address;
    evm_word_t value;
    uint32_t call_type; /**< The call type internal has the opcode */
    uint8_t* call_data; /**< The data YP: \f$d\f$ */
    uint32_t call_data_size;
    uint8_t* byte_code; /**< The byte code YP: \f$b\f$ or \f$I_{b}\f$*/
    uint32_t byte_code_size;

    uint32_t return_data_size = 0;
    uint32_t return_data_offset = 0;

    CuEVM::jump_destinations_t* jump_destinations; /**< The jump destinations */

    evm_call_context_t* parent;
#ifdef EIP_3155
    uint32_t trace_idx; /**< The index in the trace */
#endif

    /**
     * The complete constructor of the evm_call_context_t
     */
    __device__ void initiate_values(uint32_t depth, gas_t gas_limit, CuEVM::evm_stack_t* stack_ptr,
                                    CuEVM::evm_memory_t* memory_ptr, evm_word_t from, evm_word_t to,
                                    evm_word_t storage_address, evm_word_t value, uint32_t call_type,
                                    uint8_t* call_data, uint32_t call_data_size, uint8_t* byte_code,
                                    uint32_t byte_code_size, evm_call_context_t* parent = nullptr,
                                    bool static_env = false, gas_t gas_refund = 0

    );

    /**
     * The constructor with the parent's state
     **/
    __device__ void initiate_values(evm_call_context_t* parent, gas_t gas_limit, evm_word_t from, evm_word_t to,
                                    evm_word_t storage_address, evm_word_t value, uint32_t call_type,
                                    uint8_t* call_data, uint32_t call_data_size, uint8_t* byte_code,
                                    uint32_t byte_code_size, bool static_env = false, gas_t gas_refund = 0

    );

    __device__ evm_call_context_t() {};

    /**
     * The destructor of the evm_call_state_t
     */
    __device__ ~evm_call_context_t();

    __device__ void print() const;
    __device__ int32_t revert();
};
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
