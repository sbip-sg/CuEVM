// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#ifndef _ENVIRONMENTAL_OP_H_
#define _ENVIRONMENTAL_OP_H_

#include "utils.h"
#include "stack.cuh"
#include "block.cuh"
#include "state.cuh"
#include "message.cuh"

// 40s: Block Information

/**
 * The block operations class.
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
template <class params>
class block_operations
{
public:
    /**
     * The arithmetical environment used by the arbitrary length
     * integer library.
     */
    typedef arith_env_t<params> arith_t;
    /**
     * The arbitrary length integer type.
     */
    typedef typename arith_t::bn_t bn_t;
    /**
     * The stackk class.
     */
    typedef stack_t<params> stack_t;
    /**
     * The block class.
     */
    typedef block_t<params> block_t;

    /**
     * The BLOCKHASH operation implementation.
     * Takes the number from the stack and pushes the hash of the block
     * with that number.
     * The number can be at most 256 blocks behind the current block.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[inout] stack The stack.
     * @param[in] block The block.
     */
    __host__ __device__ __forceinline__ static void operation_BLOCKHASH(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        block_t &block)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BLOCKHASH);
        if (arith.has_gas(gas_limit, gas_used, error_code))
        {
            bn_t number;
            stack.pop(number, error_code);

            bn_t hash;
            block.get_previous_hash(hash, number, error_code);

            stack.push(hash, error_code);

            pc = pc + 1;
        }
    }

    /**
     * The COINBASE operation implementation.
     * Pushes on the stack the coinbase address of the current block.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     * @param[in] block The block.
    */
    __host__ __device__ __forceinline__ static void operation_COINBASE(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        block_t &block)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);
        if (arith.has_gas(gas_limit, gas_used, error_code))
        {
            bn_t coinbase;
            block.get_coin_base(coinbase);

            stack.push(coinbase, error_code);

            pc = pc + 1;
        }
    }

    /**
     * The TIMESTAMP operation implementation.
     * Pushes on the stack the timestamp of the current block.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     * @param[in] block The block.
    */
    __host__ __device__ __forceinline__ static void operation_TIMESTAMP(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        block_t &block)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);
        if (arith.has_gas(gas_limit, gas_used, error_code))
        {
            bn_t timestamp;
            block.get_time_stamp(timestamp);

            stack.push(timestamp, error_code);

            pc = pc + 1;
        }
    }

    /**
     * The NUMBER operation implementation.
     * Pushes on the stack the number of the current block.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     * @param[in] block The block.
    */
    __host__ __device__ __forceinline__ static void operation_NUMBER(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        block_t &block)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);
        if (arith.has_gas(gas_limit, gas_used, error_code))
        {
            bn_t number;
            block.get_number(number);

            stack.push(number, error_code);

            pc = pc + 1;
        }
    }

    /**
     * The DIFFICULTY/PREVRANDAO operation implementation.
     * Pushes on the stack the difficulty/prevandao of the current block.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     * @param[in] block The block.
    */
    __host__ __device__ __forceinline__ static void operation_PREVRANDAO(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        block_t &block)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);
        if (arith.has_gas(gas_limit, gas_used, error_code))
        {
            bn_t prev_randao;
            // TODO: to change depending on the evm version
            // block.get_prev_randao(prev_randao);
            block.get_difficulty(prev_randao);

            stack.push(prev_randao, error_code);

            pc = pc + 1;
        }
    }

    /**
     * The GASLIMIT operation implementation.
     * Pushes on the stack the gas limit of the current block.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     * @param[in] block The block.
    */
    __host__ __device__ __forceinline__ static void operation_GASLIMIT(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        block_t &block)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);
        if (arith.has_gas(gas_limit, gas_used, error_code))
        {
            bn_t gas_limit;
            block.get_gas_limit(gas_limit);

            stack.push(gas_limit, error_code);

            pc = pc + 1;
        }
    }

    /**
     * The CHAINID operation implementation.
     * Pushes on the stack the chain id of the current block.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     * @param[in] block The block.
    */
    __host__ __device__ __forceinline__ static void operation_CHAINID(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        block_t &block)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);
        if (arith.has_gas(gas_limit, gas_used, error_code))
        {
            bn_t chain_id;
            block.get_chain_id(chain_id);

            stack.push(chain_id, error_code);

            pc = pc + 1;
        }
    }

    /**
     * The BASEFEE operation implementation.
     * Pushes on the stack the base fee of the current block.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     * @param[in] block The block.
    */
    __host__ __device__ __forceinline__ static void operation_BASEFEE(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        block_t &block)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);
        if (arith.has_gas(gas_limit, gas_used, error_code))
        {
            bn_t base_fee;
            block.get_base_fee(base_fee);

            stack.push(base_fee, error_code);

            pc = pc + 1;
        }
    }
};

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
*/
template <class params>
class environmental_operations
{
public:
    /**
     * The arithmetical environment used by the arbitrary length
     * integer library.
     */
    typedef arith_env_t<params> arith_t;
    /**
     * The arbitrary length integer type.
     */
    typedef typename arith_t::bn_t bn_t;
    /**
     * The stackk class.
     */
    typedef stack_t<params> stack_t;
    /**
     * The block class.
     */
    typedef block_t<params> block_t;
    /**
     * The touch state class.
     */
    typedef touch_state_t<params> touch_state_t;
    /**
     * The memory class.
     */
    typedef memory_t<params> memory_t;
    /**
     * The transaction class.
     */
    typedef transaction_t<params> transaction_t;
    /**
     * The message class.
     */
    typedef message_t<params> message_t;
    /**
     * The keccak class.
     */
    typedef keccak::keccak_t keccak_t;
    /**
     * The numver of bytes in a hash.
     */
    static const uint32_t HASH_BYTES = 32;

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
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[inout] stack The stack.
     * @param[in] keccak The keccak object.
     * @param[inout] memory The memory object.
    */
    __host__ __device__ __forceinline__ static void operation_SHA3(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        keccak_t &keccak,
        memory_t &memory)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_KECCAK256);

        // Get the offset and length from the stack
        bn_t offset, length;
        stack.pop(offset, error_code);
        stack.pop(length, error_code);

        // compute the dynamic gas cost
        bn_t dynamic_gas_cost;
        // word_size = (length + 31) / 32
        cgbn_add_ui32(arith._env, dynamic_gas_cost, length, 31);
        cgbn_div_ui32(arith._env, dynamic_gas_cost, dynamic_gas_cost, 32);
        // dynamic_gas_cost = word_size * GAS_KECCAK256_WORD
        cgbn_mul_ui32(arith._env, dynamic_gas_cost, dynamic_gas_cost, GAS_KECCAK256_WORD);
        // gas_used += dynamic_gas_cost
        cgbn_add(arith._env, gas_used, gas_used, dynamic_gas_cost);

        // get the memory expansion gas cost
        memory.grow_cost(
            offset,
            length,
            gas_used,
            error_code);

        if (error_code == ERR_NONE)
        {
            if (arith.has_gas(gas_limit, gas_used, error_code))
            {
                uint8_t *data;
                data = memory.get(
                    offset,
                    length,
                    error_code);
                uint8_t hash[HASH_BYTES];
                size_t input_length;
                arith.size_t_from_cgbn(input_length, length);
                if (error_code == ERR_NONE)
                {
                    keccak.sha3(
                        data,
                        input_length,
                        &(hash[0]),
                        HASH_BYTES);
                    bn_t hash_bn;
                    arith.cgbn_from_memory(
                        hash_bn,
                        &(hash[0]));

                    stack.push(hash_bn, error_code);

                    pc = pc + 1;
                }
            }
        }
    }

    /**
     * The ADDRESS operation implementation.
     * Pushes on the stack the address of currently executing account.
     * The executing account is consider the current context, so it can be
     * different than the owner of the code.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     * @param[in] message The message.
    */
    __host__ __device__ __forceinline__ static void operation_ADDRESS(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        message_t &message)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);
        if (arith.has_gas(gas_limit, gas_used, error_code))
        {
            bn_t recipient_address;
            message.get_recipient(recipient_address);

            stack.push(recipient_address, error_code);

            pc = pc + 1;
        }
    }

    /**
     * The BALANCE operation implementation.
     * Takes the address from the stack and pushes the balance of the
     * account with that address.
     * Gas is charged for accessing the account if it is warm
     * or cold access.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[inout] stack The stack.
     * @param[in] touch_state The touch state object.
    */
    __host__ __device__ __forceinline__ static void operation_BALANCE(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        touch_state_t &touch_state)
    {
        // cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_ZERO);
        bn_t address;
        stack.pop(address, error_code);

        if (error_code == ERR_NONE)
        {
            touch_state.charge_gas_access_account(
                address,
                gas_used);
            if (arith.has_gas(gas_limit, gas_used, error_code))
            {

                bn_t balance;
                touch_state.get_account_balance(
                    address,
                    balance);

                stack.push(balance, error_code);

                pc = pc + 1;
            }
        }
    }

    /**
     * The ORIGIN operation implementation.
     * Pushes on the stack the address of the sender of the transaction
     * that started the execution.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     * @param[in] transaction The transaction.
    */
    __host__ __device__ __forceinline__ static void operation_ORIGIN(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        transaction_t &transaction)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);
        if (arith.has_gas(gas_limit, gas_used, error_code))
        {
            bn_t origin;
            transaction.get_sender(origin);

            stack.push(origin, error_code);

            pc = pc + 1;
        }
    }

    /**
     * The CALLER operation implementation.
     * Pushes on the stack the address of the sender of the message
     * that started the execution.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     * @param[in] message The message.
    */
    __host__ __device__ __forceinline__ static void operation_CALLER(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        message_t &message)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);
        if (arith.has_gas(gas_limit, gas_used, error_code))
        {
            bn_t caller;
            message.get_sender(caller);

            stack.push(caller, error_code);

            pc = pc + 1;
        }
    }

    /**
     * The CALLVALUE operation implementation.
     * Pushes on the stack the value of the message that started the execution.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     * @param[in] message The message.
    */
    __host__ __device__ __forceinline__ static void operation_CALLVALUE(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        message_t &message)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);
        if (arith.has_gas(gas_limit, gas_used, error_code))
        {
            bn_t call_value;
            message.get_value(call_value);

            stack.push(call_value, error_code);

            pc = pc + 1;
        }
    }

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
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[inout] stack The stack.
     * @param[in] message The message.
    */
    __host__ __device__ __forceinline__ static void operation_CALLDATALOAD(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        message_t &message)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_VERY_LOW);
        if (arith.has_gas(gas_limit, gas_used, error_code))
        {
            bn_t index;
            stack.pop(index, error_code);
            bn_t length;
            cgbn_set_ui32(arith._env, length, arith_t::BYTES);

            size_t available_data;
            uint8_t *data;
            data = message.get_data(
                index,
                length,
                available_data);

            stack.pushx(arith_t::BYTES, error_code, data, available_data);

            pc = pc + 1;
        }
    }

    /**
     * The CALLDATASIZE operation implementation.
     * Pushes on the stack the size of the message call data.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     * @param[in] message The message.
    */
    __host__ __device__ __forceinline__ static void operation_CALLDATASIZE(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        message_t &message)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);
        if (arith.has_gas(gas_limit, gas_used, error_code))
        {
            bn_t length;
            size_t length_s;
            length_s = message.get_data_size();
            arith.cgbn_from_size_t(length, length_s);

            stack.push(length, error_code);

            pc = pc + 1;
        }
    }

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
     * @param[out] error_code The error code.
     * @param[out] pc The program counter.
     * @param[inout] stack The stack.
     * @param[in] message The message.
     * @param[out] memory The memory.
    */
    __host__ __device__ __forceinline__ static void operation_CALLDATACOPY(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        message_t &message,
        memory_t &memory)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_VERY_LOW);

        bn_t memory_offset, data_offset, length;
        stack.pop(memory_offset, error_code);
        stack.pop(data_offset, error_code);
        stack.pop(length, error_code);

        // compute the dynamic gas cost
        bn_t dynamic_gas_cost;
        // word_size = (length + 31) / 32
        cgbn_add_ui32(arith._env, dynamic_gas_cost, length, 31);
        cgbn_div_ui32(arith._env, dynamic_gas_cost, dynamic_gas_cost, 32);
        // dynamic_gas_cost = word_size * 6
        cgbn_mul_ui32(arith._env, dynamic_gas_cost, dynamic_gas_cost, GAS_MEMORY);
        // gas_used += dynamic_gas_cost
        cgbn_add(arith._env, gas_used, gas_used, dynamic_gas_cost);

        // get the memory expansion gas cost
        memory.grow_cost(
            memory_offset,
            length,
            gas_used,
            error_code);

        if (error_code == ERR_NONE)
        {
            if (arith.has_gas(gas_limit, gas_used, error_code))
            {
                size_t available_data;
                uint8_t *data;
                data = message.get_data(
                    data_offset,
                    length,
                    available_data);

                memory.set(
                    data,
                    memory_offset,
                    length,
                    available_data,
                    error_code);

                pc = pc + 1;
            }
        }
    }

    /**
     * The CODESIZE operation implementation.
     * Pushes on the stack the size of code running in current environment.
     * 
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     * @param[in] message The message.
    */
    __host__ __device__ __forceinline__ static void operation_CODESIZE(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        message_t &message)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);
        if (arith.has_gas(gas_limit, gas_used, error_code))
        {
            size_t code_size;
            code_size = message.get_code_size();

            bn_t code_size_bn;
            arith.cgbn_from_size_t(code_size_bn, code_size);

            stack.push(code_size_bn, error_code);

            pc = pc + 1;
        }
    }

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
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[inout] stack The stack.
     * @param[in] message The message.
     * @param[in] touch_state The touch state object. The executing world state.
     * @param[out] memory The memory.
    */
    __host__ __device__ __forceinline__ static void operation_CODECOPY(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        message_t &message,
        memory_t &memory)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_VERY_LOW);

        bn_t memory_offset, code_offset, length;
        stack.pop(memory_offset, error_code);
        stack.pop(code_offset, error_code);
        stack.pop(length, error_code);

        if (error_code == ERR_NONE)
        {
            // compute the dynamic gas cost
            bn_t dynamic_gas_cost;
            // word_size = (length + 31) / 32
            cgbn_add_ui32(arith._env, dynamic_gas_cost, length, 31);
            cgbn_div_ui32(arith._env, dynamic_gas_cost, dynamic_gas_cost, 32);
            // dynamic_gas_cost = word_size * 6
            cgbn_mul_ui32(arith._env, dynamic_gas_cost, dynamic_gas_cost, GAS_MEMORY);
            // gas_used += dynamic_gas_cost
            cgbn_add(arith._env, gas_used, gas_used, dynamic_gas_cost);

            // get the memory expansion gas cost
            memory.grow_cost(
                memory_offset,
                length,
                gas_used,
                error_code);

            if (error_code == ERR_NONE)
            {
                if (arith.has_gas(gas_limit, gas_used, error_code))
                {

                    size_t available_data;
                    uint8_t *data;
                    data = message.get_byte_code_data(
                        code_offset,
                        length,
                        available_data);

                    memory.set(
                        data,
                        memory_offset,
                        length,
                        available_data,
                        error_code);

                    pc = pc + 1;
                }
            }
        }
    }

    /**
     * The GASPRICE operation implementation.
     * Pushes on the stack the gas price of the current transaction.
     * The gas price is the price per unit of gas in the transaction.
     * 
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     * @param[in] block The block.
     * @param[in] transaction The transaction.
    */
    __host__ __device__ __forceinline__ static void operation_GASPRICE(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        block_t &block,
        transaction_t &transaction)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);
        if (arith.has_gas(gas_limit, gas_used, error_code))
        {
            bn_t block_base_fee;
            block.get_base_fee(block_base_fee);

            bn_t gas_price;
            transaction.get_computed_gas_price(
                gas_price,
                block_base_fee,
                error_code);

            stack.push(gas_price, error_code);

            pc = pc + 1;
        }
    }

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
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[inout] stack The stack.
     * @param[in] touch_state The touch state object. The executing world state.
    */
    __host__ __device__ __forceinline__ static void operation_EXTCODESIZE(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        touch_state_t &touch_state)
    {
        // cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_ZERO);
        bn_t address;
        stack.pop(address, error_code);
        if (error_code == ERR_NONE)
        {
            touch_state.charge_gas_access_account(
                address,
                gas_used);
            if (arith.has_gas(gas_limit, gas_used, error_code))
            {
                size_t code_size = touch_state.get_account_code_size(
                    address);
                bn_t code_size_bn;
                arith.cgbn_from_size_t(code_size_bn, code_size);

                stack.push(code_size_bn, error_code);

                pc = pc + 1;
            }
        }
    }

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
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[in] stack The stack.
     * @param[in] touch_state The touch state object. The executing world state.
     * @param[out] memory The memory.
    */
    __host__ __device__ __forceinline__ static void operation_EXTCODECOPY(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        touch_state_t &touch_state,
        memory_t &memory)
    {
        // cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_ZERO);

        bn_t address, memory_offset, code_offset, length;
        stack.pop(address, error_code);
        stack.pop(memory_offset, error_code);
        stack.pop(code_offset, error_code);
        stack.pop(length, error_code);

        if (error_code == ERR_NONE)
        {
            // compute the dynamic gas cost
            bn_t dynamic_gas_cost;
            // word_size = (length + 31) / 32
            cgbn_add_ui32(arith._env, dynamic_gas_cost, length, 31);
            cgbn_div_ui32(arith._env, dynamic_gas_cost, dynamic_gas_cost, 32);
            // dynamic_gas_cost = word_size * 6
            cgbn_mul_ui32(arith._env, dynamic_gas_cost, dynamic_gas_cost, GAS_MEMORY);
            // gas_used += dynamic_gas_cost
            cgbn_add(arith._env, gas_used, gas_used, dynamic_gas_cost);

            // get the memory expansion gas cost
            memory.grow_cost(
                memory_offset,
                length,
                gas_used,
                error_code);

            touch_state.charge_gas_access_account(
                address,
                gas_used);

            if (error_code == ERR_NONE)
            {
                if (arith.has_gas(gas_limit, gas_used, error_code))
                {
                    size_t available_data;
                    uint8_t *data;
                    data = touch_state.get_account_code_data(
                        address,
                        code_offset,
                        length,
                        available_data);

                    memory.set(
                        data,
                        memory_offset,
                        length,
                        available_data,
                        error_code);

                    pc = pc + 1;
                }
            }
        }
    }

    /**
     * The RETURNDATASIZE operation implementation.
     * Pushes on the stack the size of the return data of the last call.
     * 
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     * @param[in] return_data The return data.
    */
    __host__ __device__ __forceinline__ static void operation_RETURNDATASIZE(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        return_data_t &return_data)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);
        if (arith.has_gas(gas_limit, gas_used, error_code))
        {
            bn_t length;
            size_t length_s;
            length_s = return_data.size();
            arith.cgbn_from_size_t(length, length_s);

            stack.push(length, error_code);

            pc = pc + 1;
        }
    }

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
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[in] stack The stack.
     * @param[out] memory The memory.
     * @param[in] return_data The return data.
    */
    __host__ __device__ __forceinline__ static void operation_RETURNDATACOPY(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        memory_t &memory,
        return_data_t &return_data)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_VERY_LOW);

        bn_t memory_offset, data_offset, length;
        stack.pop(memory_offset, error_code);
        stack.pop(data_offset, error_code);
        stack.pop(length, error_code);

        if (error_code == ERR_NONE)
        {
            // compute the dynamic gas cost
            bn_t dynamic_gas_cost;
            // word_size = (length + 31) / 32
            cgbn_add_ui32(arith._env, dynamic_gas_cost, length, 31);
            cgbn_div_ui32(arith._env, dynamic_gas_cost, dynamic_gas_cost, 32);
            // dynamic_gas_cost = word_size * 6
            cgbn_mul_ui32(arith._env, dynamic_gas_cost, dynamic_gas_cost, GAS_MEMORY);
            // gas_used += dynamic_gas_cost
            cgbn_add(arith._env, gas_used, gas_used, dynamic_gas_cost);

            // get the memory expansion gas cost
            memory.grow_cost(
                memory_offset,
                length,
                gas_used,
                error_code);

            if (error_code == ERR_NONE)
            {
                if (arith.has_gas(gas_limit, gas_used, error_code))
                {
                    uint8_t *data;
                    size_t data_offset_s, length_s;
                    int32_t overflow;
                    overflow = arith.size_t_from_cgbn(data_offset_s, data_offset);
                    overflow = overflow || arith.size_t_from_cgbn(length_s, length);
                    if (overflow)
                    {
                        error_code = ERROR_RETURN_DATA_OVERFLOW;
                    }
                    else
                    {
                        data = return_data.get(
                            data_offset_s,
                            length_s,
                            error_code);

                        memory.set(
                            data,
                            memory_offset,
                            length,
                            length_s,
                            error_code);

                        pc = pc + 1;
                    }
                }
            }
        }
    }

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
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[inout] stack The stack.
     * @param[in] touch_state The touch state object. The executing world state.
     * @param[in] keccak The keccak object.
    */
    __host__ __device__ __forceinline__ static void operation_EXTCODEHASH(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        touch_state_t &touch_state,
        keccak_t &keccak)
    {
        // cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_ZERO);
        bn_t address;
        stack.pop(address, error_code);
        if (error_code == ERR_NONE)
        {
            touch_state.charge_gas_access_account(
                address,
                gas_used);
            if (arith.has_gas(gas_limit, gas_used, error_code))
            {
                bn_t hash_bn;
                // TODO: look on the difference between destroyed and empty
                if (touch_state.is_empty_account(address))
                {
                    cgbn_set_ui32(arith._env, hash_bn, 0);
                }
                else if (touch_state.is_delete_account(address))
                {
                    cgbn_set_ui32(arith._env, hash_bn, 0);
                }
                else
                {
                    uint8_t *bytecode;
                    size_t code_size;
                    uint8_t hash[HASH_BYTES];

                    // TODO: make more efficient
                    bytecode = touch_state.get_account_code(
                        address);
                    code_size = touch_state.get_account_code_size(
                        address);

                    keccak.sha3(
                        bytecode,
                        code_size,
                        &(hash[0]),
                        HASH_BYTES);
                    arith.cgbn_from_memory(
                        hash_bn,
                        &(hash[0]));
                }

                stack.push(hash_bn, error_code);

                pc = pc + 1;
            }
        }
    }

    /**
     * The SELFBALANCE operation implementation.
     * Pushes on the stack the balance of the current contract.
     * The current contract is consider the contract that owns the
     * execution code.
     * 
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     * @param[inout] touch_state The touch state object. The executing world state.
     * @param[in] transaction The transaction.
    */
    __host__ __device__ __forceinline__ static void operation_SELFBALANCE(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        touch_state_t &touch_state,
        message_t &message)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_LOW);
        bn_t address;
        message.get_recipient(address);

        if (arith.has_gas(gas_limit, gas_used, error_code))
        {

            bn_t balance;
            touch_state.get_account_balance(
                address,
                balance);

            stack.push(balance, error_code);

            pc = pc + 1;
        }
    }
};

#endif