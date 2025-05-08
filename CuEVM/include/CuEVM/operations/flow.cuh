#pragma once
#include <CuEVM/core/data_structures.cuh>
#include <CuEVM/core/evm_call_context.cuh>
#include <CuEVM/gas_cost.cuh>
#ifdef BUILD_LIBRARY
#include <CuEVM/utils/library_utils.h>
#endif
#include <CuEVM/utils/error_codes.cuh>
#include <CuEVM/utils/opcodes.cuh>
/**
 * 50s: Flow Operations:
 * - JUMP
 * - JUMPI
 * - PC
 * - GAS
 * - JUMPDEST
 */
namespace CuEVM::operations {
/**
 * The JUMP operation implementation.
 * Takes the destination from the stack and sets the program counter
 * to the destination if it is a valid jump destination.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] pc The program counter.
 * @param[in] stack The stack.
 * @param[in] message The message.
 * @return 0 if the operation was successful, an error code otherwise.
 */
__device__ int32_t JUMP(const gas_t &gas_limit, gas_t &gas_used, uint32_t &pc, CuEVM::evm_stack_t &stack,
                        evm_call_context_t *call_context
#ifdef BUILD_LIBRARY
                        ,
                        simplified_trace_data *simplified_trace_data_ptr
#endif
);
/**
 * The JUMPI operation implementation.
 * Takes the destination and the condition from the stack and sets the program counter
 * to the destination if it is a valid jump destination and the condition is not 0.
 * If the condition is 0 the program counter is incremented by 1.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] pc The program counter.
 * @param[in] stack The stack.
 * @param[in] message The message.
 * @return 0 if the operation was successful, an error code otherwise.
 */
__device__ int32_t JUMPI(const gas_t &gas_limit, gas_t &gas_used, uint32_t &pc, CuEVM::evm_stack_t &stack,
                         evm_call_context_t *call_context
#ifdef BUILD_LIBRARY
                         ,
                         simplified_trace_data *simplified_trace_data_ptr
#endif
);

/**
 * The PC operation implementation.
 * Pushes the program counter to the stack.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[in] pc The program counter.
 * @param[out] stack The stack.
 * @return 0 if the operation was successful, an error code otherwise.
 */
__device__ int32_t PC(const gas_t &gas_limit, gas_t &gas_used, const uint32_t &pc, CuEVM::evm_stack_t &stack);

/**
 * The JUMPDEST operation implementation.
 * It increments the program counter by 1.
 * It is used as a valid jump destination.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @return 0 if the operation was successful, an error code otherwise.
 */
__device__ int32_t JUMPDEST(const gas_t &gas_limit, gas_t &gas_used);
}  // namespace CuEVM::operations
