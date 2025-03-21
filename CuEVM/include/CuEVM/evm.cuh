#pragma once
#include <CuEVM/core/block_info.cuh>
#include <CuEVM/core/data_structures.cuh>
#include <CuEVM/core/evm_call_context.cuh>
#include <CuEVM/core/memory_pool.cuh>
#include <CuEVM/core/transaction.cuh>
#include <CuEVM/gas_cost.cuh>
#include <CuEVM/operations/arithmetic.cuh>
#include <CuEVM/operations/block.cuh>
#include <CuEVM/operations/environmental.cuh>
#include <CuEVM/operations/flow.cuh>
#include <CuEVM/operations/log.cuh>
#include <CuEVM/operations/memory.cuh>
#include <CuEVM/operations/stack.cuh>
#include <CuEVM/operations/storage.cuh>
#include <CuEVM/operations/system.cuh>
#include <CuEVM/precompile.cuh>
#include <CuEVM/state/state_db.cuh>
#include <CuEVM/tracer.cuh>
#include <CuEVM/utils/error_codes.cuh>
#include <CuEVM/utils/evm_utils.cuh>
#include <CuEVM/utils/opcodes.cuh>
#ifdef BUILD_LIBRARY
#include <CuEVM/utils/library_utils.h>
#endif
#ifdef BUILD_PYTHON_LIBRARY
#include <CuEVM/utils/python_utils.h>
#endif
namespace CuEVM {
struct evm_instance_t {
    CuEVM::StateDb* state_db_ptr;                              /**< The world state pointer*/
    CuEVM::transaction::TransactionList* transaction_list_ptr; /**< The transaction pointer*/
    CuEVM::log_state_data_t* log_state_ptr;                    /**< The log state pointer*/

#ifdef EIP_3155
    CuEVM::utils::tracer_t* tracer_ptr; /**< The tracer pointer*/
#endif

#ifdef BUILD_LIBRARY
    CuEVM::serialized_worldstate_data* serialized_worldstate_data_ptr; /**< The serialized worldstate data */
    CuEVM::simplified_trace_data* simplified_trace_data_ptr;           /**< The simplified trace data pointer */
#endif
};

struct evm_t {
    const CuEVM::transaction::TransactionList* transaction_list_ptr; /**< The transaction pointer */
    CuEVM::evm_call_context_t* call_state_ptr;                       /**< The call state pointer store in global mem*/
    gas_t gas_price;                                                 /**< The gas price */
    gas_t gas_priority_fee;                                          /**< The gas priority fee */
    uint32_t status;                                                 /**< The status */
#ifdef EIP_3155
    CuEVM::utils::tracer_t* tracer_ptr; /**< The tracer pointer */
#endif
    // CuEVM::serialized_worldstate_data* serialized_worldstate_data_ptr; /**< The serialized worldstate data */
    // CuEVM::utils::simplified_trace_data* simplified_trace_data_ptr;    /**< The simplified trace data pointer */
    /**
     * @brief Construct a new evm_t object
     * Construct a new evm_t object
     * @param[in] arith The arithmetic environment
     * @param[in] world_state_data_ptr The world state pointer
     * @param[in] transaction_ptr The transaction pointer
     * @param[in] touch_state_data_ptr The touch state pointer
     * @param[in] log_state_ptr The log state pointer
     * @param[in] return_data_ptr The return data pointer
     * @param[in] tracer_ptr The tracer pointer
     */
    __device__ evm_t(CuEVM::transaction::TransactionList* transaction_list_ptr,
                     CuEVM::evm_call_context_t* call_context_ptr, CuEVM::evm_word_t* shared_stack_ptr
#ifdef EIP_3155
                     ,
                     CuEVM::utils::tracer_t* tracer_ptr
#endif

    );

    __device__ evm_t(CuEVM::transaction::TransactionList* transaction_list_ptr);

    /**
     * @brief Destroy the evm_t object
     * Destroy the evm_t object
     */
    __device__ ~evm_t();

    /**
     * @brief Start a new call operation
     * Start a new call operation the call state pointer must be set
     * to the the child call state before calling this function
     * @param[in] arith The arithmetic environment
     * @return int32_t The error code, or 0 if successful
     */
    __device__ int32_t start_CALL(cached_evm_call_context& cache_call_state);

    /**
     * @brief Finish a call operation
     * Finish a call operation, the call state pointer is set to the parent
     * inside this function. It frees the child call state. Updates the
     * gas used and gas refund.
     * @param[in] arith The arithmetic environment
     * @param[in] error_code The error code
     * @return int32_t The error code, or 0 if successful
     */
    __device__ int32_t finish_CALL(int32_t error_code);

    /**
     * @brief Finish a CREATEX operation.
     * Finish a CREATEX operation. Updates the parent state with the
     * new contract created and updates the gas used and gas refund.
     * @param[in] arith The arithmetic environment
     * @return int32_t The error code, or 0 if successful
     */
    __device__ int32_t finish_CREATE(cached_evm_call_context& cache_call_state);

    /**
     * @brief Finish a transaction operation.
     * Finish a transaction. Compute the gas left and updates the balances acording to
     * the it and gas refund. Free the depth 1 call state and set the call state to
     * the transaction call state after updating its touchs state if we had a successful
     * transaction.
     * @param[in] arith The arithmetic environment
     * @param[in] error_code The error code
     * @return int32_t The error code, or 0 if successful
     */
    __device__ int32_t finish_TRANSACTION(int32_t error_code, bool copy_state_data);

    /**
     * @brief run the EVM for the given transaction
     * Run the EVM for the given transaction
     * @param[in] arith The arithmetic environment
     */
    __device__ void run(cached_evm_call_context& cache_call_state, bool copy_state_data);
    // __device__ void run();
};

}  // namespace CuEVM
