#ifndef _CUEVM_EVM_H_
#define _CUEVM_EVM_H_

#include <CuEVM/utils/arith.cuh>
#include <CuEVM/core/block_info.cuh>
#include <CuEVM/core/message.cuh>
#include <CuEVM/core/transaction.cuh>
#include <CuEVM/state/state.cuh>
#include <CuEVM/state/world_state.cuh>
#include <CuEVM/state/access_state.cuh>
#include <CuEVM/evm_call_state.cuh>
#include <CuEVM/tracer.cuh>

namespace cuEVM {
    struct evm_instance_t {
        cuEVM::state::state_t *world_state_data_ptr;
        cuEVM::block_info_t* block_info_ptr;
        cuEVM::evm_transaction_t* transaction_ptr;
        cuEVM::state::state_access_t *access_state_data_ptr;
        cuEVM::state::state_access_t *touch_state_data_ptr;
        cuEVM::state::log_state_data_t* log_state_ptr;
        cuEVM::evm_return_data_t* return_data_ptr;
        #ifdef EIP_3155
        cuEVM::utils::tracer_t* tracer_ptr;
        #endif
    };
    struct evm_t {
        cuEVM::state::WorldState world_state;
        cuEVM::state::AccessState access_state;
        const cuEVM::block_info_t* block_info_ptr;
        const cuEVM::evm_transaction_t* transaction_ptr;
        cuEVM::evm_call_state_t* call_state_ptr;
        bn_t gas_price;
        bn_t gas_priority_fee;
        uint32_t status;
        #ifdef EIP_3155
        cuEVM::utils::tracer_t* tracer_ptr;
        #endif

        __host__ __device__  evm_t(
            ArithEnv &arith,
            cuEVM::state::state_t *world_state_data_ptr,
            cuEVM::block_info_t* block_info_ptr,
            cuEVM::evm_transaction_t* transaction_ptr,
            cuEVM::state::state_access_t *access_state_data_ptr,
            cuEVM::state::state_access_t *touch_state_data_ptr,
            cuEVM::state::log_state_data_t* log_state_ptr,
            cuEVM::evm_return_data_t* return_data_ptr
            #ifdef EIP_3155
            , cuEVM::utils::tracer_t* tracer_ptr
            #endif
        );

        __host__ __device__  evm_t(
            ArithEnv &arith,
            cuEVM::evm_instance_t &evm_instance
        );

        __host__ __device__ ~evm_t();


        __host__ __device__ int32_t start_CALL(ArithEnv &arith);

        __host__ __device__ int32_t finish_CALL(ArithEnv &arith, int32_t error_code);

        __host__ __device__ int32_t finish_CREATE(ArithEnv &arith);

        __host__ __device__ int32_t finish_TRANSACTION(ArithEnv &arith, int32_t error_code);

        __host__ __device__ void run(ArithEnv &arith);

    };

    typedef int32_t (*evm_operation_f)(cuEVM::evm_call_state_t* call_state);

    __host__ int32_t get_cpu_evm_instances(
        ArithEnv &arith,
        evm_instance_t* &evm_instances,
        const cJSON* test_json,
        uint32_t &num_instances
    );

    __host__ void free_cpu_evm_instances(
        evm_instance_t* &evm_instances,
        uint32_t num_instances
    );


    
}


#endif