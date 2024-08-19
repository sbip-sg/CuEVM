#ifndef _CUEVM_EVM_H_
#define _CUEVM_EVM_H_

#include "utils/arith.cuh"
#include "core/block_info.cu"
#include "core/message.cuh"
#include "core/transaction.cuh"
#include "state/state.cuh"
#include "state/world_state.cuh"
#include "state/access_state.cuh"
#include "evm_call_state.cuh"

namespace cuEVM {
    struct evm_t {
        cuEVM::state::WorldState world_state;
        cuEVM::state::AccessState access_state;
        const cuEVM::block_info_t* block_info_ptr;
        const cuEVM::evm_transaction_t* transaction_ptr;
        cuEVM::evm_call_state_t* call_state_ptr;
        bn_t gas_price;
        bn_t gas_priority_fee;
        uint32_t status;

        __host__ __device__  evm_t(
            ArithEnv &arith,
            cuEVM::state::state_t *world_state_data_ptr,
            cuEVM::block_info_t* block_info_ptr,
            cuEVM::evm_transaction_t* transaction_ptr,
            cuEVM::state::state_access_t *access_state_data_ptr,
            cuEVM::state::state_access_t *touch_state_data_ptr,
            cuEVM::state::log_state_data_t* log_state_ptr,
            cuEVM::evm_return_data_t* return_data_ptr
        );

        __host__ __device__ ~evm_t();


        __host__ __device__ int32_t start_CALL(ArithEnv &arith);

        __host__ __device__ int32_t evm_t::finish_CALL(ArithEnv &arith);

        __host__ __device__ int32_t evm_t::finish_CREATE(ArithEnv &arith);

        __host__ __device__ int32_t evm_t::finish_TRANSACTION(ArithEnv &arith, int32_t error_code);

        __host__ __device__ void run(ArithEnv &arith);

    };

    typedef int32_t (*evm_operation_f)(cuEVM::evm_call_state_t* call_state);

    
}


#endif