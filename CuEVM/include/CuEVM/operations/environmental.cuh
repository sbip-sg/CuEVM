#pragma once
#include <CuEVM/core/block_info.cuh>
#include <CuEVM/core/data_structures.cuh>
#include <CuEVM/core/evm_call_context.cuh>
#include <CuEVM/core/transaction.cuh>
#include <CuEVM/gas_cost.cuh>
#include <CuEVM/state/state_db.cuh>
/**
 * The environmental operations class.
 * Contains the environmental operations
 * - 20s: KECCAK256:
 *      - SHA3
 * - 30s: Environmental Information:
 *      - ADDRESS
 *      - BALANCE
 *      - ORIGIN
 *      - CALLER
 *      - CALLVALUE
 *      - CALLDATALOAD
 *      - CALLDATASIZE
 *      - CALLDATACOPY
 *      - CODESIZE
 *      - CODECOPY
 *      - GASPRICE
 *      - EXTCODESIZE
 *      - EXTCODECOPY
 *      - RETURNDATASIZE
 *      - RETURNDATACOPY
 *      - EXTCODEHASH
 *  - 47: SELFBALANCE
 * SELFBALANCE is moved here from block operations.
 */
namespace CuEVM::operations {
/**
 * The SHA3 operation implementation.
 * Takes the offset and length from the stack and pushes the hash of the
 * data from the memory at the given offset for the given length.
 * The dynamic gas cost is computed as:
 * - word_size = (length + 31) / 32
 * - dynamic_gas_cost = word_size * GAS_KECCAK256_WORD
 * Adittional gas cost is added for the memory expansion.
 *
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 * @param[inout] memory The memory object.
 * @return The error code. 0 if no error.
 */
__device__ int32_t SHA3(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack,
                        CuEVM::evm_memory_t &memory);

/**
 * The ADDRESS operation implementation.
 * Pushes on the stack the address of currently executing account.
 * The executing account is consider the current context, so it can be
 * different than the owner of the code.
 *
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[in] call_context The call context containing stack and other execution state.
 * @return The error code. 0 if no error.
 */
__device__ int32_t ADDRESS(const gas_t &gas_limit, gas_t &gas_used, const CuEVM::evm_call_context_t *call_context);

/**
 * The BALANCE operation implementation.
 * Takes the address from the stack and pushes the balance of the
 * account with that address.
 * Gas is charged for accessing the account if it is warm
 * or cold access.
 *
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[in] call_context The call context containing stack and other execution state.
 * @return The error code. 0 if no error.
 */
__device__ int32_t BALANCE(const gas_t &gas_limit, gas_t &gas_used, const CuEVM::evm_call_context_t *call_context);

/**
 * The ORIGIN operation implementation.
 * Pushes on the stack the address of the sender of the transaction
 * that started the execution.
 *
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[out] stack The stack.
 * @param[in] transaction_list The transaction list.
 * @return The error code. 0 if no error.
 */
__device__ int32_t ORIGIN(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack,
                          const CuEVM::transaction::TransactionList *transaction_list);

/**
 * The CALLER operation implementation.
 * Pushes on the stack the address of the sender of the message
 * that started the execution.
 *
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[in] call_context The call context containing stack and other execution state.
 * @return The error code. 0 if no error.
 */
__device__ int32_t CALLER(const gas_t &gas_limit, gas_t &gas_used, const CuEVM::evm_call_context_t *call_context);

/**
 * The CALLVALUE operation implementation.
 * Pushes on the stack the value of the message that started the execution.
 *
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[in] call_context The call context containing stack and other execution state.
 * @return The error code. 0 if no error.
 */
__device__ int32_t CALLVALUE(const gas_t &gas_limit, gas_t &gas_used, const CuEVM::evm_call_context_t *call_context);

/**
 * The CALLDATALOAD operation implementation.
 * Takes the index from the stack and pushes the data
 * from the message call data at the given index.
 * The data pushed is a evm word.
 * If the call data has less bytes than neccessay to fill the evm word,
 * the remaining bytes are filled with zeros. (the least significant bytes)
 *
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[in] call_context The call context containing stack and other execution state.
 * @return The error code. 0 if no error.
 */
__device__ int32_t CALLDATALOAD(const gas_t &gas_limit, gas_t &gas_used, const CuEVM::evm_call_context_t *call_context);

/**
 * The CALLDATASIZE operation implementation.
 * Pushes on the stack the size of the message call data.
 *
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[in] call_context The call context containing stack and other execution state.
 * @return The error code. 0 if no error.
 */
__device__ int32_t CALLDATASIZE(const gas_t &gas_limit, gas_t &gas_used, const CuEVM::evm_call_context_t *call_context);

/**
 * The CALLDATACOPY operation implementation.
 * Takes the memory offset, data offset and length from the stack and
 * copies the data from the message call data at the given data offset for
 * the given length to the memory at the given memory offset.
 * If the call data has less bytes than neccessay to fill the memory,
 * the remaining bytes are filled with zeros.
 * The dynamic gas cost is computed as:
 * - word_size = (length + 31) / 32
 * - dynamic_gas_cost = word_size * GAS_MEMORY
 * Adittional gas cost is added for the memory expansion.
 *
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[in] call_context The call context containing stack, memory and other execution state.
 * @return The error code. 0 if no error.
 */
__device__ int32_t CALLDATACOPY(const gas_t &gas_limit, gas_t &gas_used, const CuEVM::evm_call_context_t *call_context);

/**
 * The CODESIZE operation implementation.
 * Pushes on the stack the size of code running in current environment.
 *
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[in] call_context The call context containing stack and other execution state.
 * @return The error code. 0 if no error.
 */
__device__ int32_t CODESIZE(const gas_t &gas_limit, gas_t &gas_used, const CuEVM::evm_call_context_t *call_context);

/**
 * The CODECOPY operation implementation.
 * Takes the memory offset, code offset and length from the stack and
 * copies code running in current environment at the given code offset for
 * the given length to the memory at the given memory offset.
 * If the code has less bytes than neccessay to fill the memory,
 * the remaining bytes are filled with zeros.
 * The dynamic gas cost is computed as:
 * - word_size = (length + 31) / 32
 * - dynamic_gas_cost = word_size * GAS_MEMORY
 * Adittional gas cost is added for the memory expansion.
 *
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[in] call_context The call context containing stack, memory and other execution state.
 * @return The error code. 0 if no error.
 */
__device__ int32_t CODECOPY(const gas_t &gas_limit, gas_t &gas_used, const CuEVM::evm_call_context_t *call_context);

/**
 * The GASPRICE operation implementation.
 * Pushes on the stack the gas price of the current transaction.
 * The gas price is the price per unit of gas in the transaction.
 *
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[out] stack The stack.
 * @param[in] block The block information.
 * @param[in] transaction_list The transaction list.
 * @return The error code. 0 if no error.
 */
__device__ int32_t GASPRICE(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack,
                            const CuEVM::block_info_t &block,
                            const CuEVM::transaction::TransactionList *transaction_list);

/**
 * The EXTCODESIZE operation implementation.
 * Takes the address from the stack and pushes the size of the code
 * of the account with that address.
 * Gas is charged for accessing the account if it is warm
 * or cold access.
 *
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[in] call_context The call context containing stack, state_db and other execution state.
 * @return The error code. 0 if no error.
 */
__device__ int32_t EXTCODESIZE(const gas_t &gas_limit, gas_t &gas_used, const CuEVM::evm_call_context_t *call_context);

/**
 * The EXTCODECOPY operation implementation.
 * Takes the address, memory offset, code offset and length from the stack and
 * copies the code from the account with the given address at the given code offset for
 * the given length to the memory at the given memory offset.
 * If the code has less bytes than neccessay to fill the memory,
 * the remaining bytes are filled with zeros.
 * The dynamic gas cost is computed as:
 * - word_size = (length + 31) / 32
 * - dynamic_gas_cost = word_size * GAS_MEMORY
 * Adittional gas cost is added for the memory expansion.
 * Gas is charged for accessing the account if it is warm
 * or cold access.
 *
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[in] call_context The call context containing stack, memory, state_db and other execution state.
 * @return The error code. 0 if no error.
 */
__device__ int32_t EXTCODECOPY(const gas_t &gas_limit, gas_t &gas_used, const CuEVM::evm_call_context_t *call_context);

/**
 * The RETURNDATASIZE operation implementation.
 * Pushes on the stack the size of the return data of the last call.
 *
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[in] call_context The call context containing stack and return data information.
 * @return The error code. 0 if no error.
 */
__device__ int32_t RETURNDATASIZE(const gas_t &gas_limit, gas_t &gas_used,
                                  const CuEVM::evm_call_context_t *call_context);

/**
 * The RETURNDATACOPY operation implementation.
 * Takes the memory offset, data offset and length from the stack and
 * copies the return data from the last call at the given data offset for
 * the given length to the memory at the given memory offset.
 * If the return data has less bytes than neccessay to fill the memory,
 * an ERROR is generated.
 * The dynamic gas cost is computed as:
 * - word_size = (length + 31) / 32
 * - dynamic_gas_cost = word_size * GAS_MEMORY
 * Adittional gas cost is added for the memory expansion.
 *
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] call_context The call context containing stack, memory and return data information.
 * @return The error code. 0 if no error.
 */
__device__ int32_t RETURNDATACOPY(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_call_context_t *call_context);

/**
 * The EXTCODEHASH operation implementation.
 * Takes the address from the stack and pushes the hash of the code
 * of the account with that address.
 * Gas is charged for accessing the account if it is warm
 * or cold access.
 * If the account does not exist or is empty or the account is
 * selfdestructed, the hash is zero.
 *
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[in] call_context The call context containing stack, state_db and other execution state.
 * @return The error code. 0 if no error.
 */
__device__ int32_t EXTCODEHASH(const gas_t &gas_limit, gas_t &gas_used, const CuEVM::evm_call_context_t *call_context);

/**
 * The GAS operation implementation.
 * Pushes the gas left to the stack after this operation.
 *
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[out] stack The stack.
 * @return 0 if the operation was successful, an error code otherwise.
 */
__device__ int32_t GAS(const gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_stack_t &stack);

/**
 * The SELFBALANCE operation implementation.
 * Pushes on the stack the balance of the current contract.
 * The current contract is consider the contract that owns the
 * execution code.
 *
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[in] call_context The call context containing stack, state_db and other execution state.
 * @return The error code. 0 if no error.
 */
__device__ int32_t SELFBALANCE(const gas_t &gas_limit, gas_t &gas_used, const CuEVM::evm_call_context_t *call_context);
}  // namespace CuEVM::operations
