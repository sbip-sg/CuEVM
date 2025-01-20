#ifndef _CUEVM_LOG_OP_H_
#define _CUEVM_LOG_OP_H_

#include <CuEVM/core/evm_call_context.cuh>
#include <CuEVM/core/memory.cuh>
#include <CuEVM/core/stack.cuh>
#include <CuEVM/gas_cost.cuh>
#include <CuEVM/state/logs.cuh>
#include <CuEVM/utils/error_codes.cuh>
/**
 * a0s: Logging Operations:
 * - LOGX
 */
namespace CuEVM::operations {
/**
 * The LOGX operation implementation.
 * Takes the memory offset, the memory length and the topics from the stack and
 * stores the memory data togheter with the topics in the logs.
 * Adittional gas cost is added for the memory expansion and for the topics.
 * Every byte of memory costs additional GAS_LOG_DATA gas.
 * Every topic costs additional GAS_LOG_TOPIC gas.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[in] stack The stack.
 * @param[in] memory The memory.
 * @param[in] message The message that started the execution.
 * @param[out] log_state The logs state.
 * @param[in] opcode The opcode.
 * @return 0 if the operation was successful, an error code otherwise.
 */
__device__ int32_t LOGX(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack,
                        const CuEVM::evm_call_context_t *call_context, const uint8_t &opcode);
}  // namespace CuEVM::operations

#endif