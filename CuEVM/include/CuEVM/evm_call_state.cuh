#ifndef _CUEVM_EVM_STATE_H_
#define _CUEVM_EVM_STATE_H_

#include <CuEVM/core/memory.cuh>
#include <CuEVM/core/message.cuh>
#include <CuEVM/core/return_data.cuh>
#include <CuEVM/core/stack.cuh>
#include <CuEVM/state/logs.cuh>
#include <CuEVM/state/touch_state.cuh>
#include <CuEVM/utils/arith.cuh>
#include <CuEVM/utils/evm_defines.cuh>

namespace CuEVM {
struct evm_call_state_t {
    CuEVM::evm_call_state_t* parent; /**< The parent state */
    uint32_t depth;                  /**< The depth of the state */
    uint32_t pc;                     /**< The program counter */
    bn_t gas_used;                   /**< The gas */
    bn_t gas_refund;                 /**< The gas refund */
    CuEVM::evm_message_call_t*
        message_ptr; /**< The message that started the execution */
    bn_t gas_limit;  /**< The gas limit */
    CuEVM::evm_stack_t* stack_ptr;                  /**< The stack */
    CuEVM::evm_memory_t* memory_ptr;                /**< The memory */
    CuEVM::log_state_data_t* log_state_ptr;         /**< The logs state */
    CuEVM::TouchState touch_state;                  /**< The touch state */
    CuEVM::evm_return_data_t* last_return_data_ptr; /**< The return data */
#ifdef EIP_3155
    uint32_t trace_idx; /**< The index in the trace */
#endif

    /**
     * The complete constructor of the evm_state_t
     */
    __host__ __device__ evm_call_state_t(
        ArithEnv& arith, CuEVM::evm_call_state_t* parent, uint32_t depth,
        uint32_t pc, bn_t gas_used, bn_t gas_refund,
        CuEVM::evm_message_call_t* message_ptr, CuEVM::evm_stack_t* stack_ptr,
        CuEVM::evm_memory_t* memory_ptr, CuEVM::log_state_data_t* log_state_ptr,
        CuEVM::TouchState touch_state,
        CuEVM::evm_return_data_t* last_return_data_ptr);

    /**
     * The constructor with the parent state and message call
     */
    __host__ __device__
    evm_call_state_t(ArithEnv& arith, CuEVM::evm_call_state_t* parent,
                     CuEVM::evm_message_call_t* message_ptr);

    /**
     * The constructor with no parent state and message call
     */
    __host__ __device__ evm_call_state_t(
        ArithEnv& arith, CuEVM::WorldState* word_state_ptr,
        CuEVM::evm_stack_t* stack_ptr, CuEVM::evm_memory_t* memory_ptr,
        CuEVM::log_state_data_t* log_state_ptr,
        CuEVM::state_access_t* state_access_ptr,
        CuEVM::evm_return_data_t* last_return_data_ptr);
    /**
     * The destructor of the evm_call_state_t
     */
    __host__ __device__ ~evm_call_state_t();

    __host__ __device__ int32_t update(ArithEnv& arith,
                                       evm_call_state_t& other);
};
}  // namespace CuEVM

#endif