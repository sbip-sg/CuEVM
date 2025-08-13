#pragma once

#include <CuEVM/core/block_info.cuh>
#include <CuEVM/core/data_structures.cuh>

// 40s: Block Information

/**
 * The block operations.
 * Contains the block operations 40s: Block Information:
 * - BLOCKHASH
 * - COINBASE
 * - TIMESTAMP
 * - NUMBER
 * - DIFFICULTY
 * - GASLIMIT
 * - CHAINID
 * - BASEFEE
 *
 * SELFBALANCE is moved to environmental operations because it is
 * not related to the block.
 */
namespace CuEVM::operations {
/**
 * The BLOCKHASH operation implementation.
 * Takes the number from the stack and pushes the hash of the block
 * with that number.
 * The number can be at most 256 blocks behind the current block.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 * @return The error code. 0 if no error.
 */
__device__ int32_t BLOCKHASH(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack
#ifdef BUILD_GO_LIBRARY
                             ,
                             const transaction::TransactionList
                                 *transaction_list_ptr  // supplement the block info by adding delay per transaction
#endif
);

/**
 * The COINBASE operation implementation.
 * Pushes on the stack the coinbase address of the current block.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[out] stack The stack.
 * @return The error code. 0 if no error.
 */
__device__ int32_t COINBASE(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack);

/**
 * The TIMESTAMP operation implementation.
 * Pushes on the stack the timestamp of the current block.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[out] stack The stack.
 * @return The error code. 0 if no error.
 */
__device__ int32_t TIMESTAMP(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack
#ifdef BUILD_GO_LIBRARY
                             ,
                             const transaction::TransactionList
                                 *transaction_list_ptr  // supplement the block info by adding delay per transaction
#endif
);

/**
 * The NUMBER operation implementation.
 * Pushes on the stack the number of the current block.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[out] stack The stack.
 * @return The error code. 0 if no error.
 */
__device__ int32_t NUMBER(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack
#ifdef BUILD_GO_LIBRARY
                          ,
                          const transaction::TransactionList
                              *transaction_list_ptr  // supplement the block info by adding delay per transaction
#endif
);

/**
 * The DIFFICULTY/PREVRANDAO operation implementation.
 * Pushes on the stack the difficulty/prevandao of the current block.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[out] stack The stack.
 * @return The error code. 0 if no error.
 */
__device__ int32_t PREVRANDAO(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack);

/**
 * The GASLIMIT operation implementation.
 * Pushes on the stack the gas limit of the current block.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[out] stack The stack.
 * @return The error code. 0 if no error.
 */
__device__ int32_t GASLIMIT(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack);

/**
 * The CHAINID operation implementation.
 * Pushes on the stack the chain id of the current block.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[out] stack The stack.
 * @return The error code. 0 if no error.
 */
__device__ int32_t CHAINID(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack);

/**
 * The BASEFEE operation implementation.
 * Pushes on the stack the base fee of the current block.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[out] stack The stack.
 * @return The error code. 0 if no error.
 */
__device__ int32_t BASEFEE(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack);
}  // namespace CuEVM::operations
