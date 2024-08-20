#ifndef _CUEVM_EVM_STATE_H_
#define _CUEVM_EVM_STATE_H_

#include "utils/arith.cuh"
#include "core/message.cuh"
#include "core/stack.cuh"
#include "core/memory.cuh"
#include "state/logs.cuh"
#include "state/touch_state.cuh"
#include "core/return_data.cuh"

namespace cuEVM {
    struct evm_call_state_t {
        cuEVM::evm_call_state_t* parent; /**< The parent state */
        uint32_t depth; /**< The depth of the state */
        uint32_t pc; /**< The program counter */
        bn_t gas_used; /**< The gas */
        bn_t gas_refund; /**< The gas refund */
        cuEVM::evm_message_call_t* message_ptr; /**< The message that started the execution */
        bn_t gas_limit; /**< The gas limit */
        cuEVM::evm_stack_t* stack_ptr; /**< The stack */
        cuEVM::evm_memory_t* memory_ptr; /**< The memory */
        cuEVM::state::log_state_data_t* log_state_ptr; /**< The logs state */
        cuEVM::state::TouchState touch_state; /**< The touch state */
        cuEVM::evm_return_data_t* last_return_data_ptr; /**< The return data */
        #ifdef EIP_3155
        uint32_t trace_idx; /**< The index in the trace */
        #endif

        /**
         * The complete constructor of the evm_state_t
         */
        __host__ __device__ evm_call_state_t(
            ArithEnv &arith,
            cuEVM::evm_call_state_t* parent,
            uint32_t depth,
            uint32_t pc,
            bn_t gas_used,
            bn_t gas_refund,
            cuEVM::evm_message_call_t* message_ptr,
            cuEVM::evm_stack_t* stack_ptr,
            cuEVM::evm_memory_t* memory_ptr,
            cuEVM::state::log_state_data_t* log_state_ptr,
            cuEVM::state::TouchState touch_state,
            cuEVM::evm_return_data_t* last_return_data_ptr
        );

        /**
         * The constructor with the parent state and message call
         */
        __host__ __device__ evm_call_state_t(
            ArithEnv &arith,
            cuEVM::evm_call_state_t* parent,
            cuEVM::evm_message_call_t *message_ptr
        );

        /**
         * The constructor with no parent state and message call
         */
        __host__ __device__ evm_call_state_t(
            ArithEnv &arith,
            cuEVM::state::AccessState *access_state_ptr,
            cuEVM::evm_stack_t* stack_ptr,
            cuEVM::evm_memory_t* memory_ptr,
            cuEVM::state::log_state_data_t* log_state_ptr, 
            cuEVM::state::state_access_t* state_access_ptr,
            cuEVM::evm_return_data_t* last_return_data_ptr
            );
        /**
         * The destructor of the evm_call_state_t
         */
        __host__ __device__ ~evm_call_state_t();

        __host__ __device__ int32_t update(ArithEnv &arith, evm_call_state_t &other);

    };
}


#endif