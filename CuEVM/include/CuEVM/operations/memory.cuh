#ifndef _CUEVM_MEMORY_OP_H_
#define _CUEVM_MEMORY_OP_H_

#include <CuEVM/core/memory.cuh>
#include <CuEVM/core/stack.cuh>

/**
 * 50s: Memory Operations:
 * - MLOAD
 * - MSTORE
 * - MSTORE8
 * - MSIZE
 */
namespace CuEVM::operations {
/**
 * The MLOAD operation implementation.
 * Takes the memory offset from the stack and push the evm word from
 * the memory at the given memory offset.
 * Adittional gas cost is added for the memory expansion.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 * @param[in] memory The memory.
 * @return 0 if the operation was successful, an error code otherwise.
 */
__host__ __device__ int32_t MLOAD(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack,
                                  CuEVM::evm_memory_t &memory);

/**
 * The MSTORE operation implementation.
 * Takes the memory offset and the value from the stack and stores the
 * value in the memory at the given memory offset.
 * Adittional gas cost is added for the memory expansion.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[in] stack The stack.
 * @param[out] memory The memory.
 * @return 0 if the operation was successful, an error code otherwise.
 */
__host__ __device__ int32_t MSTORE(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack,
                                   CuEVM::evm_memory_t &memory);

/**
 * The MSTORE8 operation implementation.
 * Takes the memory offset and the value from the stack and stores the
 * least significant byte of the value in the memory at the given memory offset.
 * Adittional gas cost is added for the memory expansion.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[in] stack The stack.
 * @param[out] memory The memory.
 * @return 0 if the operation was successful, an error code otherwise.
 */
__host__ __device__ int32_t MSTORE8(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack,
                                    CuEVM::evm_memory_t &memory);

/**
 * The MSIZE operation implementation.
 * Pushes the memory size to the stack.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[out] stack The stack.
 * @param[in] memory The memory.
 * @return 0 if the operation was successful, an error code otherwise.
 */
__host__ __device__ int32_t MSIZE(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack,
                                  const CuEVM::evm_memory_t &memory);
}  // namespace CuEVM::operations

#endif