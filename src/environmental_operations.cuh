// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#ifndef _ENVIRONMENTAL_OP_H_
#define _ENVIRONMENTAL_OP_H_

#include "uitls.h"
#include "stack.cuh"
#include "block.cuh"
#include "state.cuh"
#include "message.cuh"


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
     * The keccak class.
    */
    typedef keccak::keccak_t keccak_t;
    /**
     * The return data class.
    */
    typedef return_data_t<params> return_data_t;
    /**
     * The numver of bytes in a hash.
    */
    static const uint32_t HASH_BYTES = 32;

    // 20s: KECCAK256
    __host__ __device__ __forceinline__ static void operation_SHA3(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        keccak_t &keccak,
        memory_t &memory
    )
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
        // dynamic_gas_cost = word_size * 6
        cgbn_mul_ui32(arith._env, dynamic_gas_cost, dynamic_gas_cost, 6);
        // gas_used += dynamic_gas_cost
        cgbn_add(arith._env, gas_used, gas_used, dynamic_gas_cost);

        // get the memory expansion gas cost
        memory.grow_cost(
            offset,
            length,
            gas_used,
            error_code
        );

        if (error_code == ERR_NONE)
        {
            if (has_gas(arith, gas_limit, gas_used, error_code))
            {
                uint8_t *data;
                data = memory.get(
                    offset,
                    length,
                    error_code
                );
                uint8_t hash[HASH_BYTES];
                size_t input_length;
                arith.size_t_from_cgbn(input_length, length);
                if (error_code == ERR_NONE)
                {
                    keccak.sha3(
                        data,
                        input_length,
                        &(hash[0]),
                        HASH_BYTES
                    );
                    bn_t hash_bn;
                    arith.cgbn_from_memory(
                        hash_bn,
                        &(hash[0])
                    );

                    stack.push(hash_bn, error_code);

                    pc = pc + 1;
                }
            }
        }
    }

    __host__ __device__ __forceinline__ static void operation_ADDRESS(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        message_t &message
    )
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);
        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
            bn_t contract_address;
            message.get_contract_address(contract_address);

            stack.push(contract_address, error_code);

            pc = pc + 1;
        }
    }

    __host__ __device__ __forceinline__ static void operation_BALANCE(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        touch_state_t &touch_state
    )
    {
        //cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_ZERO);
        bn_t address;
        stack.pop(address, error_code);

        if (error_code == ERR_NONE)
        {
            touch_state.charge_gas_access_account(
                address,
                gas_used);
            if (has_gas(arith, gas_limit, gas_used, error_code))
            {

                bn_t balance;
                touch_state.get_account_balance(
                    address,
                    balance
                );
                
                stack.push(balance);

                pc = pc + 1;
            }
        }
    }

    __host__ __device__ __forceinline__ static void operation_ORIGIN(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        transaction_t &transaction
    )
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);
        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
            bn_t origin;
            transaction.get_sender(origin);

            stack.push(origin, error_code);

            pc = pc + 1;
        }
    }
    

    __host__ __device__ __forceinline__ static void operation_CALLER(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        message_t &message
    )
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);
        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
            bn_t caller;
            message.get_sender(caller);

            stack.push(caller, error_code);

            pc = pc + 1;
        }
    }

    __host__ __device__ __forceinline__ static void operation_CALLVALUE(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        message_t &message
    )
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);
        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
            bn_t call_value;
            message.get_value(call_value);

            stack.push(call_value, error_code);

            pc = pc + 1;
        }
    }

    __host__ __device__ __forceinline__ static void operation_CALLDATALOAD(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc, 
        stack_t &stack,
        message_t &message
    )
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_VERY_LOW);
        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
            bn_t index;
            stack.pop(index, error_code);
            bn_t length;
            cgbn_set_ui32(arith._env, length, arith::BYTES);

            size_t available_data;
            uint8_t *data;
            data = message.get_data(
                index,
                length,
                available_data,
            );

            stack.pushx(arith::BYTES, error_code, data, available_data);

            pc = pc + 1;
        }
    }

    __host__ __device__ __forceinline__ static void operation_CALLDATASIZE(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code, 
        uint32_t &pc,
        stack_t &stack,
        message_t &message
    )
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);
        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
            bn_t length;
            size_t length_s;
            length_s = message.get_data_size();
            arith.cgbn_from_size_t(length, length_s);

            stack.push(length, error_code);

            pc = pc + 1;
        }
    }

    __host__ __device__ __forceinline__ static void operation_CALLDATACOPY(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code, 
        uint32_t &pc,
        stack_t &stack,
        message_t &message,
        memory_t &memory
    )
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
            error_code
        );

        if (error_code == ERR_NONE)
        {
            if (has_gas(arith, gas_limit, gas_used, error_code))
            {
                size_t available_data;
                uint8_t *data;
                data = message.get_data(
                    data_offset,
                    length,
                    available_data,
                );

                memory.set(
                    data,
                    memory_offset,
                    length,
                    available_data,
                    error_code
                );

                pc = pc + 1;
            }
        }
    }

    __host__ __device__ __forceinline__ static void operation_CODESIZE(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code, 
        uint32_t &pc,
        message_t &message,
        touch_state_t &touch_state
    )
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);
        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
            bn_t contract_address;
            message.get_contract_address(contract_address);

            size_t code_size;
            code_size = touch_state.get_account_code_size(
                contract_address
            );
            bn_t code_size_bn;
            arith.cgbn_from_size_t(code_size_bn, code_size);

            stack.push(code_size_bn, error_code);

            pc = pc + 1;
        }
    }

    __host__ __device__ __forceinline__ static void operation_CODECOPY(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code, 
        uint32_t &pc,
        stack_t &stack,
        message_t &message,
        touch_state_t &touch_state,
        memory_t &memory
    )
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
                error_code
            );

            
            if (error_code == ERR_NONE)
            {
                if (has_gas(arith, gas_limit, gas_used, error_code))
                {
                    bn_t contract_address;
                    message.get_contract_address(contract_address);

                    size_t available_data;
                    uint8_t *data;
                    data = touch_state.get_account_code_data(
                        contract_address,
                        code_offset,
                        length,
                        available_data
                    );

                    memory.set(
                        data,
                        memory_offset,
                        length,
                        available_data,
                        error_code
                    );

                    pc = pc + 1;
                }
            }
        }
    }

    __host__ __device__ __forceinline__ static void operation_GASPRICE(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code, 
        uint32_t &pc,
        stack_t &stack,
        block_t &block,
        transaction_t &transaction
    )
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);
        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
            bn_t block_base_fee;
            block.get_base_fee(block_base_fee);

            bn_t gas_price;
            transaction.get_computed_gas_price(
                gas_price,
                block_base_fee,
                error_code
            )

            stack.push(gas_price, error_code);

            pc = pc + 1;
        }
    }

    __host__ __device__ __forceinline__ static void operation_EXTCODESIZE(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        touch_state_t &touch_state
    )
    {
        //cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_ZERO);
        bn_t address;
        stack.pop(address, error_code);
        if (error_code == ERR_NONE)
        {
            touch.state.charge_gas_access_account(
                address,
                gas_used);
            if (has_gas(arith, gas_limit, gas_used, error_code))
            {
                code_size = touch_state.get_account_code_size(
                    address
                );
                bn_t code_size_bn;
                arith.cgbn_from_size_t(code_size_bn, code_size);

                stack.push(code_size_bn, error_code);

                pc = pc + 1;
            }

        }
    }

    __host__ __device__ __forceinline__ static void operation_EXTCODECOPY(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code, 
        uint32_t &pc,
        stack_t &stack,
        touch_state_t &touch_state,
        memory_t &memory
    )
    {
        //cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_ZERO);
        
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
                error_code
            );

            touch.state.charge_gas_access_account(
                address,
                gas_used);

            if (error_code == ERR_NONE)
            {
                if (has_gas(arith, gas_limit, gas_used, error_code))
                {
                    size_t available_data;
                    uint8_t *data;
                    data = touch_state.get_account_code_data(
                        address,
                        code_offset,
                        length,
                        available_data
                    );

                    memory.set(
                        data,
                        memory_offset,
                        length,
                        available_data,
                        error_code
                    );

                    pc = pc + 1;
                }
            }
        }
    }

    __host__ __device__ __forceinline__ static void operation_RETURNDATASIZE(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code, 
        uint32_t &pc,
        stack_t &stack,
        return_data_t &return_data
    )
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);
        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
            bn_t length;
            size_t length_s;
            length_s = return_data.size();
            arith.cgbn_from_size_t(length, length_s);

            stack.push(length, error_code);

            pc = pc + 1;
        }
    }

    __host__ __device__ __forceinline__ static void operation_RETURNDATACOPY(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code, 
        uint32_t &pc,
        stack_t &stack,
        memory_t &memory,
        return_data_t &return_data
    )
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
                error_code
            );

            if (error_code == ERR_NONE)
            {
                if (has_gas(arith, gas_limit, gas_used, error_code))
                {
                    size_t available_data;
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
                            error_code
                        );

                        memory.set(
                            data,
                            memory_offset,
                            length,
                            length_s,
                            error_code
                        );

                        pc = pc + 1;
                    }
                }
            }
        }
    }

    __host__ __device__ __forceinline__ static void operation_EXTCODEHASH(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        touch_state_t &touch_state,
        keccak_t &keccak
    )
    {
        //cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_ZERO);
        bn_t address;
        stack.pop(address, error_code);
        if (error_code == ERR_NONE)
        {
            touch.state.charge_gas_access_account(
                address,
                gas_used);
            if (has_gas(arith, gas_limit, gas_used, error_code))
            {
                // TODO: find if the contract is empty or destroyed
                uint8_t *bytecode;
                size_t code_size;
                uint8_t hash[HASH_BYTES];
                
                // TODO: make more efficient
                bytecode = touch_state.get_account_code(
                    address
                );
                code_size = touch_state.get_account_code_size(
                    address
                );

                keccak.sha3(
                    bytecode,
                    code_size,
                    &(hash[0]),
                    HASH_BYTES
                );
                bn_t hash_bn;
                arith.cgbn_from_memory(
                    hash_bn,
                    &(hash[0])
                );

                stack.push(hash_bn, error_code);

                pc = pc + 1;
            }

        }
    }

    __host__ __device__ __forceinline__ static void operation_SELFBALANCE(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        touch_state_t &touch_state,
        transaction_t &transaction
    )
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_LOW);
        bn_t address;
        transaction.get_contract_address(address);

        if (has_gas(arith, gas_limit, gas_used, error_code))
        {

            bn_t balance;
            touch_state.get_account_balance(
                address,
                balance
            );
            
            stack.push(balance);

            pc = pc + 1;
        }
    }
};


#endif