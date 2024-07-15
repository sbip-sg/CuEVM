// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#include "../include/operations/block.cuh"
#include "../include/gas_cost.cuh"
#include "../include/utils/error_codes.cuh"

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
namespace cuEVM::operations {
    /**
     * The BLOCKHASH operation implementation.
     * Takes the number from the stack and pushes the hash of the block
     * with that number.
     * The number can be at most 256 blocks behind the current block.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] pc The program counter.
     * @param[inout] stack The stack.
     * @param[in] block The block.
     * @return The error code. 0 if no error.
     */
    __host__ __device__ int32_t BLOCKHASH(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &pc,
        cuEVM::stack::evm_stack_t &stack,
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

            pc = pc + 1;
        }
        return error_code;
    }

    /**
     * The COINBASE operation implementation.
     * Pushes on the stack the coinbase address of the current block.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     * @param[in] block The block.
     * @return The error code. 0 if no error.
    */
    __host__ __device__ int32_t COINBASE(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &pc,
        cuEVM::stack::evm_stack_t &stack,
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

            pc = pc + 1;
        }
        return error_code;
    }

    /**
     * The TIMESTAMP operation implementation.
     * Pushes on the stack the timestamp of the current block.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     * @param[in] block The block.
     * @return The error code. 0 if no error.
    */
    __host__ __device__ int32_t TIMESTAMP(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &pc,
        cuEVM::stack::evm_stack_t &stack,
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

            pc = pc + 1;
        }
        return error_code;
    }

    /**
     * The NUMBER operation implementation.
     * Pushes on the stack the number of the current block.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     * @param[in] block The block.
     * @return The error code. 0 if no error.
    */
    __host__ __device__ int32_t NUMBER(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &pc,
        cuEVM::stack::evm_stack_t &stack,
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

            pc = pc + 1;
        }
        return error_code;
    }

    /**
     * The DIFFICULTY/PREVRANDAO operation implementation.
     * Pushes on the stack the difficulty/prevandao of the current block.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     * @param[in] block The block.
     * @return The error code. 0 if no error.
    */
    __host__ __device__ int32_t PREVRANDAO(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &pc,
        cuEVM::stack::evm_stack_t &stack,
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

            pc = pc + 1;
        }
        return error_code;
    }

    /**
     * The GASLIMIT operation implementation.
     * Pushes on the stack the gas limit of the current block.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     * @param[in] block The block.
     * @return The error code. 0 if no error.
    */
    __host__ __device__ int32_t GASLIMIT(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &pc,
        cuEVM::stack::evm_stack_t &stack,
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

            pc = pc + 1;
        }
        return error_code;
    }

    /**
     * The CHAINID operation implementation.
     * Pushes on the stack the chain id of the current block.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     * @param[in] block The block.
     * @return The error code. 0 if no error.
    */
    __host__ __device__ int32_t CHAINID(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &pc,
        cuEVM::stack::evm_stack_t &stack,
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

            pc = pc + 1;
        }
        return error_code;
    }

    /**
     * The BASEFEE operation implementation.
     * Pushes on the stack the base fee of the current block.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     * @param[in] block The block.
     * @return The error code. 0 if no error.
    */
    __host__ __device__ int32_t BASEFEE(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &pc,
        cuEVM::stack::evm_stack_t &stack,
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

            pc = pc + 1;
        }
        return error_code;
    }
}