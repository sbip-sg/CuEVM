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

namespace CuEVM {
    struct evm_instance_t {
        CuEVM::state::state_t *world_state_data_ptr;
        CuEVM::block_info_t* block_info_ptr;
        CuEVM::evm_transaction_t* transaction_ptr;
        CuEVM::state::state_access_t *access_state_data_ptr;
        CuEVM::state::state_access_t *touch_state_data_ptr;
        CuEVM::state::log_state_data_t* log_state_ptr;
        CuEVM::evm_return_data_t* return_data_ptr;
        #ifdef EIP_3155
        CuEVM::utils::tracer_t* tracer_ptr;
        #endif
    };
    struct evm_t {
        CuEVM::state::WorldState world_state;
        CuEVM::state::AccessState access_state;
        const CuEVM::block_info_t* block_info_ptr;
        const CuEVM::evm_transaction_t* transaction_ptr;
        CuEVM::evm_call_state_t* call_state_ptr;
        bn_t gas_price;
        bn_t gas_priority_fee;
        uint32_t status;
        #ifdef EIP_3155
        CuEVM::utils::tracer_t* tracer_ptr;
        #endif

        __host__ __device__  evm_t(
            ArithEnv &arith,
            CuEVM::state::state_t *world_state_data_ptr,
            CuEVM::block_info_t* block_info_ptr,
            CuEVM::evm_transaction_t* transaction_ptr,
            CuEVM::state::state_access_t *access_state_data_ptr,
            CuEVM::state::state_access_t *touch_state_data_ptr,
            CuEVM::state::log_state_data_t* log_state_ptr,
            CuEVM::evm_return_data_t* return_data_ptr
            #ifdef EIP_3155
            , CuEVM::utils::tracer_t* tracer_ptr
            #endif
        );

        __host__ __device__  evm_t(
            ArithEnv &arith,
            CuEVM::evm_instance_t &evm_instance
        );

        __host__ __device__ ~evm_t();


        __host__ __device__ int32_t start_CALL(ArithEnv &arith);

        __host__ __device__ int32_t finish_CALL(ArithEnv &arith, int32_t error_code);

        __host__ __device__ int32_t finish_CREATE(ArithEnv &arith);

        __host__ __device__ int32_t finish_TRANSACTION(ArithEnv &arith, int32_t error_code);

        __host__ __device__ void run(ArithEnv &arith);

    };

    typedef int32_t (*evm_operation_f)(CuEVM::evm_call_state_t* call_state);

    /**
     * @brief Get the CPU EVM instances object
     * Get the evm instances from the json file
     * @param[in] arith The arithmetic environment
     * @param[in] test_json The json object
     * @param[out] evm_instances The evm instances
     * @param[out] num_instances The number of instances
     * @param[in] managed Whether the memory is managed
     * @return int32_t The error code, 0 if successful
     */
    __host__ int32_t get_evm_instances(
        ArithEnv &arith,
        evm_instance_t* &evm_instances,
        const cJSON* test_json,
        uint32_t &num_instances,
        int32_t managed = 0
    );

    /**
     * @brief Free the EVM instances object
     * Free the evm instances
     * @param[in] evm_instances The evm instances
     * @param[in] num_instances The number of instances
     * @param[in] managed Whether the memory is managed
     */
    __host__ void free_evm_instances(
        evm_instance_t* &evm_instances,
        uint32_t num_instances,
        int32_t managed = 0
    );


    
}


#endif