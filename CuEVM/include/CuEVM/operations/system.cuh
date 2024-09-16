#ifndef _CUEVM_SYSTEMS_OP_H_
#define _CUEVM_SYSTEMS_OP_H_


#include <CuEVM/utils/arith.cuh>
#include <CuEVM/core/stack.cuh>
#include <CuEVM/core/memory.cuh>
#include <CuEVM/core/message.cuh>
#include <CuEVM/state/touch_state.cuh>
#include <CuEVM/evm_call_state.cuh>

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
namespace CuEVM::operations
{
    /**
     * The STOP operation.
     * @param[out] return_data The return data.
     * @return return error code.
     */
    __host__ __device__ int32_t STOP(
        CuEVM::evm_return_data_t &return_data);

    /**
     * The CREATE operation. gives the new evm call state
     * @param[in] arith The arithmetical environment.
     * @param[in] access_state The access state.
     * @param[in] current_state The current state.
     * @param[out] new_state_ptr The new state pointer.
     * @return 0 if the operation is successful, otherwise the error code.
     */
    __host__ __device__ int32_t CREATE(
        ArithEnv &arith,
        CuEVM::AccessState &access_state,
        CuEVM::evm_call_state_t &current_state,
        CuEVM::evm_call_state_t* &new_state_ptr);

    /**
     * The CALL operation. gives the new evm call state
     * @param[in] arith The arithmetical environment.
     * @param[in] access_state The access state.
     * @param[in] current_state The current state.
     * @param[out] new_state_ptr The new state pointer.
     * @return 0 if the operation is successful, otherwise the error code.
     */
    __host__ __device__ int32_t CALL(
        ArithEnv &arith,
        CuEVM::AccessState &access_state,
        CuEVM::evm_call_state_t &current_state,
        CuEVM::evm_call_state_t* &new_state_ptr);

    /**
     * The CALLCODE operation. gives the new evm call state
     * @param[in] arith The arithmetical environment.
     * @param[in] access_state The access state.
     * @param[in] current_state The current state.
     * @param[out] new_state_ptr The new state pointer.
     * @return 0 if the operation is successful, otherwise the error code.
    */
    __host__ __device__ int32_t CALLCODE(
        ArithEnv &arith,
        CuEVM::AccessState &access_state,
        CuEVM::evm_call_state_t &current_state,
        CuEVM::evm_call_state_t* &new_state_ptr);

    /**
     * The RETURN operation.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[in] stack The stack.
     * @param[in] memory The memory.
     * @param[out] return_data The return data.
     * @return 0 if the operation is successful, otherwise the error code.
    */
    __host__ __device__ int32_t RETURN(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        CuEVM::evm_stack_t &stack,
        CuEVM::evm_memory_t &memory,
        CuEVM::evm_return_data_t &return_data);
    /**
     * The DELEGATECALL operation. gives the new evm call state
     * @param[in] arith The arithmetical environment.
     * @param[in] access_state The access state.
     * @param[in] current_state The current state.
     * @param[out] new_state_ptr The new state pointer.
     * @return 0 if the operation is successful, otherwise the error code.
    */
    __host__ __device__ int32_t DELEGATECALL(
        ArithEnv &arith,
        CuEVM::AccessState &access_state,
        CuEVM::evm_call_state_t &current_state,
        CuEVM::evm_call_state_t* &new_state_ptr);

    /**
     * The CREATE2 operation. gives the new evm call state
     * @param[in] arith The arithmetical environment.
     * @param[in] access_state The access state.
     * @param[in] current_state The current state.
     * @param[out] new_state_ptr The new state pointer.
     * @return 0 if the operation is successful, otherwise the error code.
     */
    __host__ __device__ int32_t CREATE2(
        ArithEnv &arith,
        CuEVM::AccessState &access_state,
        CuEVM::evm_call_state_t &current_state,
        CuEVM::evm_call_state_t* &new_state_ptr);

    /**
     * The STATICCALL operation. gives the new evm call state
     * @param[in] arith The arithmetical environment.
     * @param[in] access_state The access state.
     * @param[in] current_state The current state.
     * @param[out] new_state_ptr The new state pointer.
     * @return 0 if the operation is successful, otherwise the error code.
    */
    __host__ __device__ int32_t STATICCALL(
        ArithEnv &arith,
        CuEVM::AccessState &access_state,
        CuEVM::evm_call_state_t &current_state,
        CuEVM::evm_call_state_t* &new_state_ptr);

    /**
     * The REVERT operation.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[in] stack The stack.
     * @param[in] memory The memory.
     * @param[out] return_data The return data.
    */
    __host__ __device__ int32_t REVERT(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        CuEVM::evm_stack_t &stack,
        CuEVM::evm_memory_t &memory,
        CuEVM::evm_return_data_t &return_data);

    /**
     * The INVALID operation.
     * @return The error code.
    */
    __host__ __device__ int32_t INVALID();

    /**
     * The SELFDESTRUCT operation.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] stack The stack.
     * @param[in] message The current context message call.
     * @param[inout] touch_state The touch state.
     * @param[out] return_data The return data.
     * @return 0 if the operation is successful, otherwise the error code.
    */
    __host__ __device__ int32_t SELFDESTRUCT(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        CuEVM::evm_stack_t &stack,
        CuEVM::evm_message_call_t &message,
        CuEVM::TouchState &touch_state,
        CuEVM::evm_return_data_t &return_data);
} // namespace CuEVM::operation

#endif