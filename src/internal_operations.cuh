// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#ifndef _INTERNAL_OP_H_
#define _INTERNAL_OP_H_

#include "uitls.h"
#include "stack.cuh"
#include "block.cuh"
#include "state.cuh"
#include "message.cuh"

/**
 * The internal operations class.
 * It contains the implementation of the internal operations.
 * 50s: Stack, Memory, Storage and Flow Operations:
 * - POP
 * - MLOAD
 * - MSTORE
 * - MSTORE8
 * - SLOAD
 * - SSTORE
 * - JUMP
 * - JUMPI
 * - PC
 * - MSIZE
 * - GAS
 * - JUMPDEST
 */
template <class params>
class internal_operations
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
     * The CGBN wide type with double the given number of bits in environment.
     */
    typedef typename env_t::cgbn_wide_t bn_wide_t;
    /**
     * The arbitrary length integer type used for the storage.
     * It is defined as the EVM word type.
     */
    typedef cgbn_mem_t<params::BITS> evm_word_t;
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
     * The return data class.
     */
    typedef return_data_t<params> return_data_t;
    /**
     * The numver of bytes in a hash.
     */
    static const uint32_t HASH_BYTES = 32;

    /**
     * The POP operation implementation.
     * It pops the top element from the stack.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     */
    __host__ __device__ __forceinline__ static void operation_POP(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);

        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
            bn_t y;

            stack.pop(y, error_code);

            pc = pc + 1;
        }
    }

    /**
     * The MLOAD operation implementation.
     * Takes the memory offset from the stack and push the evm word from
     * the memory at the given memory offset.
     * Adittional gas cost is added for the memory expansion.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[inout] stack The stack.
     * @param[in] memory The memory.
     */
    __host__ __device__ __forceinline__ static void operation_MLOAD(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        memory_t &memory)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_VERYLOW);

        bn_t memory_offset;
        stack.pop(memory_offset, error_code);
        bn_t length;
        cgbn_set_ui32(arith._env, length, arith::BYTES);

        memory.grow_cost(
            memory_offset,
            length,
            gas_used,
            error_code);

        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
            uint8_t *data;
            data = memory.get(
                memory_offset,
                length,
                error_code);

            bn_t value;
            arith.cgbn_from_memory(
                value,
                data);

            stack.push(value, error_code);

            pc = pc + 1;
        }
    }

    /**
     * The MSTORE operation implementation.
     * Takes the memory offset and the value from the stack and stores the
     * value in the memory at the given memory offset.
     * Adittional gas cost is added for the memory expansion.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[in] stack The stack.
     * @param[out] memory The memory.
     */
    __host__ __device__ __forceinline__ static void operation_MSTORE(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        memory_t &memory)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_VERY_LOW);

        bn_t memory_offset;
        stack.pop(memory_offset, error_code);
        bn_t value;
        stack.pop(value, error_code);
        bn_t length;
        cgbn_set_ui32(arith._env, length, arith::BYTES);

        if (error_code == ERR_NONE)
        {
            memory.grow_cost(
                memory_offset,
                length,
                gas_used,
                error_code);

            if (has_gas(arith, gas_limit, gas_used, error_code))
            {
                uint8_t data[arith::BYTES];
                arith.memory_from_cgbn(
                    &(data[0]),
                    value);

                memory.set(
                    memory_offset,
                    length,
                    &(data[0]),
                    error_code);

                pc = pc + 1;
            }
        }
    }

    /**
     * The MSTORE8 operation implementation.
     * Takes the memory offset and the value from the stack and stores the
     * least significant byte of the value in the memory at the given memory offset.
     * Adittional gas cost is added for the memory expansion.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[in] stack The stack.
     * @param[out] memory The memory.
     */
    __host__ __device__ __forceinline__ static void operation_MSTORE8(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        memory_t &memory)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_VERY_LOW);

        bn_t memory_offset;
        stack.pop(memory_offset, error_code);
        bn_t value;
        stack.pop(value, error_code);
        bn_t length;
        cgbn_set_ui32(arith._env, length, 1);

        if (error_code == ERR_NONE)
        {
            memory.grow_cost(
                memory_offset,
                length,
                gas_used,
                error_code);

            if (has_gas(arith, gas_limit, gas_used, error_code))
            {
                uint8_t data[arith::BYTES];
                arith.memory_from_cgbn(
                    &(data[0]),
                    value);

                memory.set(
                    memory_offset,
                    length,
                    &(data[arith::BYTES - 1]),
                    error_code);

                pc = pc + 1;
            }
        }
    }

    /**
     * The SLOAD operation implementation.
     * Takes the key from the stack and push the value from
     * the storage at the given key.
     * Adittional gas cost is added for the storage access, if
     * the key is warm or cold.
     * The storage is given by the message call.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[inout] stack The stack.
     * @param[in] touch_state The touch state.
     * @param[in] message The message that started the execution.
     */
    __host__ __device__ __forceinline__ static void operation_SLOAD(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        touch_state_t &touch_state,
        message_t &message, )
    {
        // cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_ZERO);

        bn_t key;
        stack.pop(key, error_code);

        if (error_code == ERR_NONE)
        {
            bn_t storage_address;
            message.get_storage_address(storage_address);

            touch_state.charge_gas_access_storage(
                storage_address,
                key,
                gas_used);

            if (has_gas(arith, gas_limit, gas_used, error_code))
            {
                bn_t value;
                touch_state.get_value(
                    storage_address,
                    key,
                    value);

                stack.push(value, error_code);

                pc = pc + 1;
            }
        }
    }

    /**
     * The SSTORE operation implementation.
     * Takes the key and the value from the stack and stores the
     * value in the storage at the given key.
     * Adittional gas cost is added depending on the value and if
     * the key is warm or cold.
     * The storage is given by the message call.
     * Depending on the value the refund gas is updated.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] gas_refund The gas refund.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[in] stack The stack.
     * @param[out] touch_state The touch state.
     * @param[in] message The message that started the execution.
     */
    __host__ __device__ __forceinline__ static void operation_SSTORE(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        bn_t &gas_refund,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        touch_state_t &touch_state,
        message_t &message, )
    {
        // only if is not a static call
        if (message.get_static_env() == 0)
        {
            // cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_ZERO);
            bn_t gas_left;
            cgbn_sub(arith._env, gas_left, gas_limit, gas_used);
            if (cgbn_compare_ui32(arith._env, gas_left, GAS_STIPEND) < 0)
            {
                error_code = ERR_OUT_OF_GAS;
            }
            else
            {
                bn_t key;
                stack.pop(key, error_code);
                bn_t value;
                stack.pop(value, error_code);

                if (error_code == ERR_NONE)
                {
                    bn_t storage_address;
                    message.get_storage_address(storage_address);

                    touch_state.charge_gas_set_storage(
                        storage_address,
                        key,
                        value,
                        gas_used,
                        gas_refund);

                    if (has_gas(arith, gas_limit, gas_used, error_code))
                    {
                        touch_state.set_value(
                            storage_address,
                            key,
                            value);

                        pc = pc + 1;
                    }
                }
            }
        }
        else
        {
            error_code = ERROR_STATIC_CALL_CONTEXT_SSTORE;
        }
    }

    /**
     * The JUMP operation implementation.
     * Takes the destination from the stack and sets the program counter
     * to the destination if it is a valid jump destination.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[in] stack The stack.
     * @param[in] jumpdest The jump destinations.
     */
    __host__ __device__ __forceinline__ static void operation_JUMP(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        jump_destinations_t &jumpdest)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_MID);

        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
            bn_t destination;
            stack.pop(destination, error_code);
            size_t destination_s;
            int32_t overflow;
            overflow = arith.size_t_from_cgbn(destination_s, destination);

            if (error_code == ERR_NONE)
            {
                // if is not a valid jump destination
                if (
                    (overflow == 1) ||
                    (jumpdest.has(destination_s) == 0))
                {
                    error_code = ERR_INVALID_JUMP_DESTINATION;
                }
                else
                {
                    pc = destination_s;
                }
            }
        }
    }

    /**
     * The JUMPI operation implementation.
     * Takes the destination and the condition from the stack and sets the program counter
     * to the destination if it is a valid jump destination and the condition is not 0.
     * If the condition is 0 the program counter is incremented by 1.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[in] stack The stack.
     * @param[in] jumpdest The jump destinations.
     */
    __host__ __device__ __forceinline__ static void operation_JUMPI(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        jump_destinations_t &jumpdest)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_HIGH);

        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
            bn_t destination;
            stack.pop(destination, error_code);
            bn_t condition;
            stack.pop(condition, error_code);

            if (error_code == ERR_NONE)
            {
                if (cgbn_compare_ui32(arith._env, condition, 0) != 0)
                {
                    size_t destination_s;
                    int32_t overflow;
                    overflow = arith.size_t_from_cgbn(destination_s, destination);
                    // if is not a valid jump destination
                    if (
                        (overflow == 1) ||
                        (jumpdest.has(destination_s) == 0))
                    {
                        error_code = ERR_INVALID_JUMP_DESTINATION;
                    }
                    else
                    {
                        pc = destination_s;
                    }
                }
                else
                {
                    pc = pc + 1;
                }
            }
        }
    }

    /**
     * The PC operation implementation.
     * Pushes the program counter to the stack.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     */
    __host__ __device__ __forceinline__ static void operation_PC(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);

        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
            bn_t pc_bn;
            cgbn_set_ui32(arith._env, pc_bn, pc);

            stack.push(pc_bn, error_code);

            pc = pc + 1;
        }
    }

    /**
     * The MSIZE operation implementation.
     * Pushes the memory size to the stack.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     * @param[in] memory The memory.
     */
    __host__ __device__ __forceinline__ static void operation_MSIZE(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        memory_t &memory)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);

        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
            bn_t size;
            size_t size_s;
            size_s = memory.get_size(size);

            arith.cgbn_from_size_t(
                size,
                size_s);

            stack.push(size, error_code);

            pc = pc + 1;
        }
    }

    /**
     * The GAS operation implementation.
     * Pushes the gas left to the stack after this operation.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     */
    __host__ __device__ __forceinline__ static void operation_GAS(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);

        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
            bn_t gas_left;
            cgbn_sub(arith._env, gas_left, gas_limit, gas_used);

            stack.push(gas_left, error_code);

            pc = pc + 1;
        }
    }

    /**
     * The JUMPDEST operation implementation.
     * It increments the program counter by 1.
     * It is used as a valid jump destination.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     */
    __host__ __device__ __forceinline__ static void operation_JUMPDEST(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_JUMP_DEST);

        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
            pc = pc + 1;
        }
    }
};

#endif