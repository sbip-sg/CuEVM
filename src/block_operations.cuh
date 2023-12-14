// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#ifndef _BLOCK_OP_H_
#define _BLOCK_OP_H_

#include "uitls.h"
#include "stack.cuh"
#include "block.cuh"
#include "state.cuh"
#include "message.cuh"


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
    

    // 40s: Block Information
    __host__ __device__ __forceinline__ static void operation_BLOCKHASH(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        block_t &block
    )
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BLOCKHASH);
        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
            bn_t number;
            stack.pop(number, error_code);

            bn_t hash;
            block.get_previous_hash(hash, number, error_code);

            stack.push(hash, error_code);

            pc = pc + 1;
        }

    }

    __host__ __device__ __forceinline__ static void operation_COINBASE(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        block_t &block
    )
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);
        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
            bn_t coinbase;
            block.get_coin_base(coinbase);

            stack.push(coinbase, error_code);

            pc = pc + 1;
        }
    }

    __host__ __device__ __forceinline__ static void operation_TIMESTAMP(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        block_t &block
    )
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);
        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
            bn_t timestamp;
            block.get_time_stamp(timestamp);

            stack.push(timestamp, error_code);

            pc = pc + 1;
        }
    }

    __host__ __device__ __forceinline__ static void operation_NUMBER(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        block_t &block
    )
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);
        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
            bn_t number;
            block.get_number(number);

            stack.push(number, error_code);

            pc = pc + 1;
        }
    }

    __host__ __device__ __forceinline__ static void operation_PREVRANDAO(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        block_t &block
    )
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);
        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
            bn_t prev_randao;
            // TODO: to change depending on the evm version
            //block.get_prev_randao(prev_randao);
            block.get_difficulty(prev_randao);

            stack.push(prev_randao, error_code);

            pc = pc + 1;
        }
    }

    __host__ __device__ __forceinline__ static void operation_GASLIMIT(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        block_t &block
    )
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);
        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
            bn_t gas_limit;
            block.get_gas_limit(gas_limit);

            stack.push(gas_limit, error_code);

            pc = pc + 1;
        }
    }

    __host__ __device__ __forceinline__ static void operation_CHAINID(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        block_t &block
    )
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);
        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
            bn_t chain_id;
            block.get_chain_id(chain_id);

            stack.push(chain_id, error_code);

            pc = pc + 1;
        }
    }

    // operation_SELFBALANCE in environmental_operations

    __host__ __device__ __forceinline__ static void operation_BASEFEE(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        block_t &block
    )
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);
        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
            bn_t base_fee;
            block.get_base_fee(base_fee);

            stack.push(base_fee, error_code);

            pc = pc + 1;
        }
    }

};


#endif