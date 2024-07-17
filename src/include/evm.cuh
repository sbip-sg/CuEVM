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

        __host__ __device__  evm_t(
            ArithEnv &arith,
            cuEVM::state::state_t *world_state_data_ptr,
            cuEVM::block_info_t* block_info_ptr,
            cuEVM::evm_transaction_t* transaction_ptr,
            cuEVM::state::state_access_t *access_state_data_ptr,
            cuEVM::state::state_access_t *touch_state_data_ptr,
            cuEVM::state::log_state_data_t* log_state_ptr,
            cuEVM::evm_return_data_t* return_data_ptr
        ) : world_state(world_state_data_ptr), block_info_ptr(block_info_ptr), transaction_ptr(transaction_ptr), access_state(access_state_data_ptr, &world_state) {
            call_state_ptr = new cuEVM::evm_call_state_t(
                arith,
                &access_state,
                nullptr,
                nullptr,
                log_state_ptr,
                touch_state_data_ptr,
                return_data_ptr
            );
            int32_t error_code = transaction_ptr->validate(
                arith,
                access_state,
                call_state_ptr->touch_state,
                *block_info_ptr,
                call_state_ptr->gas_used,
                gas_price,
                gas_priority_fee
            );
            if (error_code == ERROR_SUCCESS) {
                cuEVM::evm_message_call_t *transaction_call_message_ptr = nullptr;
                error_code = transaction_ptr->get_message_call(
                    arith,
                    access_state,
                    transaction_call_message_ptr
                );
                cuEVM::evm_call_state_t* child_call_state_ptr = new cuEVM::evm_call_state_t(
                    arith,
                    call_state_ptr,
                    transaction_call_message_ptr
                );
                call_state_ptr = child_call_state_ptr;
            }
        }

        __host__ __device__ ~evm_t() {
            if (call_state_ptr != nullptr) {
                delete call_state_ptr;
            }
            call_state_ptr = nullptr;
            block_info_ptr = nullptr;
            transaction_ptr = nullptr;
        }

        __host__ __device__ void run(ArithEnv &arith) {
            
        }

    };
}


#endif