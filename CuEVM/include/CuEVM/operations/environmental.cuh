// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#ifndef _CUEVM_ENV_OP_H_
#define _CUEVM_ENV_OP_H_

#include <CuEVM/utils/arith.cuh>
#include <CuEVM/core/block_info.cuh>
#include <CuEVM/core/stack.cuh>
#include <CuEVM/core/memory.cuh>
#include <CuEVM/core/message.cuh>
#include <CuEVM/core/transaction.cuh>
#include <CuEVM/state/touch_state.cuh>
#include <CuEVM/core/return_data.cuh>

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
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] stack The stack.
     * @param[inout] memory The memory object.
     * @return The error code. 0 if no error.
    */
    __host__ __device__ int32_t SHA3(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        CuEVM::evm_stack_t &stack,
        CuEVM::evm_memory_t &memory);

    /**
     * The ADDRESS operation implementation.
     * Pushes on the stack the address of currently executing account.
     * The executing account is consider the current context, so it can be
     * different than the owner of the code.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] stack The stack.
     * @param[in] message The message.
     * @return The error code. 0 if no error.
    */
    __host__ __device__ int32_t ADDRESS(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        CuEVM::evm_stack_t &stack,
        const CuEVM::evm_message_call_t &message);

    /**
     * The BALANCE operation implementation.
     * Takes the address from the stack and pushes the balance of the
     * account with that address.
     * Gas is charged for accessing the account if it is warm
     * or cold access.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] stack The stack.
     * @param[in] touch_state The touch state object.
     * @return The error code. 0 if no error.
    */
    __host__ __device__ int32_t BALANCE(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        CuEVM::evm_stack_t &stack,
        CuEVM::TouchState &touch_state);

    /**
     * The ORIGIN operation implementation.
     * Pushes on the stack the address of the sender of the transaction
     * that started the execution.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] stack The stack.
     * @param[in] transaction The transaction.
     * @return The error code. 0 if no error.
    */
    __host__ __device__ int32_t ORIGIN(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        CuEVM::evm_stack_t &stack,
        const CuEVM::evm_transaction_t &transaction);

    /**
     * The CALLER operation implementation.
     * Pushes on the stack the address of the sender of the message
     * that started the execution.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] stack The stack.
     * @param[in] message The message.
     * @return The error code. 0 if no error.
    */
    __host__ __device__ int32_t CALLER(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        CuEVM::evm_stack_t &stack,
        const CuEVM::evm_message_call_t &message);

    /**
     * The CALLVALUE operation implementation.
     * Pushes on the stack the value of the message that started the execution.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] stack The stack.
     * @param[in] message The message.
     * @return The error code. 0 if no error.
    */
    __host__ __device__ int32_t CALLVALUE(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        CuEVM::evm_stack_t &stack,
        const CuEVM::evm_message_call_t &message);

    /**
     * The CALLDATALOAD operation implementation.
     * Takes the index from the stack and pushes the data
     * from the message call data at the given index.
     * The data pushed is a evm word.
     * If the call data has less bytes than neccessay to fill the evm word,
     * the remaining bytes are filled with zeros. (the least significant bytes)
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] stack The stack.
     * @param[in] message The message.
     * @return The error code. 0 if no error.
    */
    __host__ __device__ int32_t CALLDATALOAD(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        CuEVM::evm_stack_t &stack,
        const CuEVM::evm_message_call_t &message);

    /**
     * The CALLDATASIZE operation implementation.
     * Pushes on the stack the size of the message call data.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] stack The stack.
     * @param[in] message The message.
     * @return The error code. 0 if no error.
    */
    __host__ __device__ int32_t CALLDATASIZE(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        CuEVM::evm_stack_t &stack,
        const CuEVM::evm_message_call_t &message);

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
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] stack The stack.
     * @param[in] message The message.
     * @param[out] memory The memory.
     * @return The error code. 0 if no error.
    */
    __host__ __device__ int32_t CALLDATACOPY(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        CuEVM::evm_stack_t &stack,
        const CuEVM::evm_message_call_t &message,
        CuEVM::evm_memory_t &memory);

    /**
     * The CODESIZE operation implementation.
     * Pushes on the stack the size of code running in current environment.
     *
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] stack The stack.
     * @param[in] message The message.
     * @return The error code. 0 if no error.
    */
    __host__ __device__ int32_t CODESIZE(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        CuEVM::evm_stack_t &stack,
        const CuEVM::evm_message_call_t &message);

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
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] stack The stack.
     * @param[in] message The message.
     * @param[in] touch_state The touch state object. The executing world state.
     * @param[out] memory The memory.
     * @return The error code. 0 if no error.
    */
    __host__ __device__ int32_t CODECOPY(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        CuEVM::evm_stack_t &stack,
        const CuEVM::evm_message_call_t &message,
        CuEVM::evm_memory_t &memory);

    /**
     * The GASPRICE operation implementation.
     * Pushes on the stack the gas price of the current transaction.
     * The gas price is the price per unit of gas in the transaction.
     *
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] stack The stack.
     * @param[in] block The block.
     * @param[in] transaction The transaction.
     * @return The error code. 0 if no error.
    */
    __host__ __device__ int32_t GASPRICE(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        CuEVM::evm_stack_t &stack,
        const CuEVM::block_info_t &block,
        const CuEVM::evm_transaction_t &transaction);

    /**
     * The EXTCODESIZE operation implementation.
     * Takes the address from the stack and pushes the size of the code
     * of the account with that address.
     * Gas is charged for accessing the account if it is warm
     * or cold access.
     *
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] stack The stack.
     * @param[in] touch_state The touch state object. The executing world state.
     * @return The error code. 0 if no error.
    */
    __host__ __device__ int32_t EXTCODESIZE(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        CuEVM::evm_stack_t &stack,
        CuEVM::TouchState &touch_state);

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
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[in] stack The stack.
     * @param[in] touch_state The touch state object. The executing world state.
     * @param[out] memory The memory.
     * @return The error code. 0 if no error.
    */
    __host__ __device__ int32_t EXTCODECOPY(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        CuEVM::evm_stack_t &stack,
        CuEVM::TouchState &touch_state,
        CuEVM::evm_memory_t &memory);

    /**
     * The RETURNDATASIZE operation implementation.
     * Pushes on the stack the size of the return data of the last call.
     *
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] stack The stack.
     * @param[in] return_data The return data.
     * @return The error code. 0 if no error.
    */
    __host__ __device__ int32_t RETURNDATASIZE(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        CuEVM::evm_stack_t &stack,
        const CuEVM::evm_return_data_t &return_data);

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
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[in] stack The stack.
     * @param[out] memory The memory.
     * @param[in] return_data The return data.
     * @return The error code. 0 if no error.
    */
    __host__ __device__ int32_t RETURNDATACOPY(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        CuEVM::evm_stack_t &stack,
        CuEVM::evm_memory_t &memory,
        const CuEVM::evm_return_data_t &return_data);

    /**
     * The EXTCODEHASH operation implementation.
     * Takes the address from the stack and pushes the hash of the code
     * of the account with that address.
     * Gas is charged for accessing the account if it is warm
     * or cold access.
     * If the account does not exist or is empty or the account is
     * selfdestructed, the hash is zero.
     *
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] stack The stack.
     * @param[in] touch_state The touch state object. The executing world state.
     * @return The error code. 0 if no error.
    */
    __host__ __device__ int32_t EXTCODEHASH(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        CuEVM::evm_stack_t &stack,
        CuEVM::TouchState &touch_state);

    /**
     * The SELFBALANCE operation implementation.
     * Pushes on the stack the balance of the current contract.
     * The current contract is consider the contract that owns the
     * execution code.
     *
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] stack The stack.
     * @param[inout] touch_state The touch state object. The executing world state.
     * @param[in] transaction The transaction.
     * @return The error code. 0 if no error.
    */
    __host__ __device__ int32_t SELFBALANCE(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        CuEVM::evm_stack_t &stack,
        CuEVM::TouchState &touch_state,
        const CuEVM::evm_message_call_t &message);
}

#endif
