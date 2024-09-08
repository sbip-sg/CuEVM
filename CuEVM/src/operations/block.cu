// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#include <CuEVM/operations/block.cuh>
#include <CuEVM/gas_cost.cuh>
#include <CuEVM/utils/error_codes.cuh>


namespace cuEVM::operations {
    __host__ __device__ int32_t BLOCKHASH(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        cuEVM::evm_stack_t &stack,
        const cuEVM::block_info_t &block)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_BLOCKHASH);
        int32_t error_code = cuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        if (error_code == ERROR_SUCCESS)
        {
            bn_t number;
            error_code |= stack.pop(arith, number);

            bn_t hash;
            // even if error of invalid number/index, the hash is set to zero
            uint32_t tmp_error_code;
            error_code |= block.get_previous_hash(arith, hash, number);

            error_code |= stack.push(arith, hash);
        }
        return error_code;
    }

    __host__ __device__ int32_t COINBASE(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        cuEVM::evm_stack_t &stack,
        const cuEVM::block_info_t &block)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_BASE);
        int32_t error_code = cuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        if (error_code == ERROR_SUCCESS)
        {
            bn_t coin_base;
            block.get_coin_base(arith, coin_base);

            error_code |= stack.push(arith, coin_base);
        }
        return error_code;
    }

    __host__ __device__ int32_t TIMESTAMP(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        cuEVM::evm_stack_t &stack,
        const cuEVM::block_info_t &block)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_BASE);
        int32_t error_code = cuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        if (error_code == ERROR_SUCCESS)
        {
            bn_t time_stamp;
            block.get_time_stamp(arith, time_stamp);

            error_code |= stack.push(arith, time_stamp);
        }
        return error_code;
    }

    __host__ __device__ int32_t NUMBER(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        cuEVM::evm_stack_t &stack,
        const cuEVM::block_info_t &block)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_BASE);
        int32_t error_code = cuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        if (error_code == ERROR_SUCCESS)
        {
            bn_t number;
            block.get_number(arith, number);

            error_code |= stack.push(arith, number);
        }
        return error_code;
    }

    __host__ __device__ int32_t PREVRANDAO(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        cuEVM::evm_stack_t &stack,
        const cuEVM::block_info_t &block)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_BASE);
        int32_t error_code = cuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        if (error_code == ERROR_SUCCESS)
        {
            bn_t prev_randao;
            // TODO: to change depending on the evm version
            block.get_prevrandao(arith, prev_randao); // Assuming after merge fork
            // block.get_difficulty(prev_randao);

            error_code |= stack.push(arith, prev_randao);
        }
        return error_code;
    }

    __host__ __device__ int32_t GASLIMIT(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        cuEVM::evm_stack_t &stack,
        const cuEVM::block_info_t &block)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_BASE);
        int32_t error_code = cuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        if (error_code == ERROR_SUCCESS)
        {
            bn_t gas_limit;
            block.get_gas_limit(arith, gas_limit);

            error_code |= stack.push(arith, gas_limit);
        }
        return error_code;
    }

    __host__ __device__ int32_t CHAINID(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        cuEVM::evm_stack_t &stack,
        const cuEVM::block_info_t &block)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_BASE);
        int32_t error_code = cuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        if (error_code == ERROR_SUCCESS)
        {
            bn_t chain_id;
            block.get_chain_id(arith, chain_id);

            error_code |= stack.push(arith, chain_id);
        }
        return error_code;
    }

    __host__ __device__ int32_t BASEFEE(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        cuEVM::evm_stack_t &stack,
        const cuEVM::block_info_t &block)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_BASE);
        int32_t error_code = cuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        if (error_code == ERROR_SUCCESS)
        {
            bn_t base_fee;
            block.get_base_fee(arith, base_fee);

            error_code |= stack.push(arith, base_fee);
        }
        return error_code;
    }
}