
#ifndef _CUEVM_STORAGE_OP_H_
#define _CUEVM_STORAGE_OP_H_

#include <CuEVM/core/message.cuh>
#include <CuEVM/core/stack.cuh>
#include <CuEVM/state/touch_state.cuh>

/**
 * 50s: Storage Operations:
 * - SLOAD
 * - SSTORE
 */
namespace CuEVM::operations {
/**
 * The SLOAD operation implementation.
 * Takes the key from the stack and push the value from
 * the storage at the given key.
 * Adittional gas cost is added for the storage access, if
 * the key is warm or cold.
 * The storage is given by the message call.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 * @param[in] touch_state The touch state.
 * @param[in] message The message that started the execution.
 * @return 0 if the operation was successful, an error code otherwise.
 */
__host__ __device__ int32_t SLOAD(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack,
                                  CuEVM::TouchState &touch_state, const CuEVM::evm_message_call_t &message);

/**
 * The SSTORE operation implementation.
 * Takes the key and the value from the stack and stores the
 * value in the storage at the given key.
 * Adittional gas cost is added depending on the value and if
 * the key is warm or cold.
 * The storage is given by the message call.
 * Depending on the value the refund gas is updated.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] gas_refund The gas refund.
 * @param[in] stack The stack.
 * @param[out] touch_state The touch state.
 * @param[in] message The message that started the execution.
 * @return 0 if the operation was successful, an error code otherwise.
 */
__host__ __device__ int32_t SSTORE(const gas_t &gas_limit, gas_t &gas_used, gas_t &gas_refund,
                                   CuEVM::evm_stack_t &stack, CuEVM::TouchState &touch_state,
                                   const CuEVM::evm_message_call_t &message);

}  // namespace CuEVM::operations

#endif