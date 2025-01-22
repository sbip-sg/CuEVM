#ifndef _CUEVM_SYSTEMS_OP_H_
#define _CUEVM_SYSTEMS_OP_H_

#include <CuEVM/core/evm_call_context.cuh>
#include <CuEVM/core/memory.cuh>
#include <CuEVM/core/memory_pool.cuh>
#include <CuEVM/core/message.cuh>
#include <CuEVM/core/stack.cuh>
#include <CuEVM/state/state_db.cuh>

/**
 * The system operations class.
 * It contains the implementation of the system operations.
 * 00:
 * - STOP
 * f0s: System operations:
 * - CREATE
 * - CALL
 * - CALLCODE
 * - RETURN
 * - DELEGATECALL
 * - CREATE2
 * - STATICCALL
 * - REVERT
 * - INVALID
 * - SELFDESTRUCT
 */
namespace CuEVM::operations {
/**
 * The STOP operation.
 * @param[out] return_data The return data.
 * @return return error code.
 */
__device__ int32_t STOP(CuEVM::evm_call_context_t *call_state_ptr);

/**
 * The CREATE operation. gives the new evm call state
 * @param[in] current_state The current state.
 * @param[out] new_state_ptr The new state pointer.
 * @return 0 if the operation is successful, otherwise the error code.
 */
__device__ int32_t CREATE(CuEVM::evm_call_context_t *current_context, CuEVM::evm_call_context_t *&new_context_ptr,
                          CuEVM::cached_evm_call_context &cached_state);

/**
 * The CALL operation. gives the new evm call state
 * @param[in] current_state The current state.
 * @param[out] new_state_ptr The new state pointer.
 * @return 0 if the operation is successful, otherwise the error code.
 */
__device__ int32_t CALL(CuEVM::evm_call_context_t *current_context, CuEVM::evm_call_context_t *&new_context_ptr,
                        CuEVM::cached_evm_call_context &cached_state);

/**
 * The CALLCODE operation. gives the new evm call state
 * @param[in] current_state The current state.
 * @param[out] new_state_ptr The new state pointer.
 * @return 0 if the operation is successful, otherwise the error code.
 */
__device__ int32_t CALLCODE(CuEVM::evm_call_context_t *current_context, CuEVM::evm_call_context_t *&new_context_ptr,
                            CuEVM::cached_evm_call_context &cached_state);

/**
 * The RETURN operation.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[in] stack The stack.
 * @param[in] memory The memory.
 * @param[out] return_data The return data.
 * @return 0 if the operation is successful, otherwise the error code.
 */
__device__ int32_t RETURN(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack,
                          CuEVM::evm_call_context_t *call_context);
/**
 * The DELEGATECALL operation. gives the new evm call state
 * @param[in] current_state The current state.
 * @param[out] new_state_ptr The new state pointer.
 * @return 0 if the operation is successful, otherwise the error code.
 */
__device__ int32_t DELEGATECALL(CuEVM::evm_call_context_t *current_context, CuEVM::evm_call_context_t *&new_context_ptr,
                                CuEVM::cached_evm_call_context &cached_state);

/**
 * The CREATE2 operation. gives the new evm call state
 * @param[in] current_state The current state.
 * @param[out] new_state_ptr The new state pointer.
 * @return 0 if the operation is successful, otherwise the error code.
 */
__device__ int32_t CREATE2(CuEVM::evm_call_context_t *current_context, CuEVM::evm_call_context_t *&new_context_ptr,
                           CuEVM::cached_evm_call_context &cached_state);

/**
 * The STATICCALL operation. gives the new evm call state
 * @param[in] current_state The current state.
 * @param[out] new_state_ptr The new state pointer.
 * @return 0 if the operation is successful, otherwise the error code.
 */
__device__ int32_t STATICCALL(CuEVM::evm_call_context_t *current_context, CuEVM::evm_call_context_t *&new_context_ptr,
                              CuEVM::cached_evm_call_context &cached_state);

/**
 * The REVERT operation.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[in] stack The stack.
 * @param[in] memory The memory.
 * @param[out] return_data The return data.
 */
__device__ int32_t REVERT(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack,
                          CuEVM::evm_call_context_t *call_state_ptr);

/**
 * The INVALID operation.
 * @return The error code.
 */
__device__ int32_t INVALID();

/**
 * The SELFDESTRUCT operation.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 * @param[in] message The current context message call.
 * @param[inout] touch_state The touch state.
 * @param[out] return_data The return data.
 * @return 0 if the operation is successful, otherwise the error code.
 */
__device__ int32_t SELFDESTRUCT(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack,
                                CuEVM::evm_call_context_t *call_state_ptr);
}  // namespace CuEVM::operations

#endif