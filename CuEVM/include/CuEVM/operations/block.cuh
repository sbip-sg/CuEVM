// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#ifndef _CUEVM_BLOCK_OP_H_
#define _CUEVM_BLOCK_OP_H_

#include <CuEVM/utils/arith.cuh>
#include <CuEVM/core/block_info.cuh>
#include <CuEVM/core/stack.cuh>

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
namespace cuEVM::operations {
    /**
     * The BLOCKHASH operation implementation.
     * Takes the number from the stack and pushes the hash of the block
     * with that number.
     * The number can be at most 256 blocks behind the current block.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] stack The stack.
     * @param[in] block The block.
     * @return The error code. 0 if no error.
     */
    __host__ __device__ int32_t BLOCKHASH(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        cuEVM::evm_stack_t &stack,
        const cuEVM::block_info_t &block);

    /**
     * The COINBASE operation implementation.
     * Pushes on the stack the coinbase address of the current block.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] stack The stack.
     * @param[in] block The block.
     * @return The error code. 0 if no error.
    */
    __host__ __device__ int32_t COINBASE(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        cuEVM::evm_stack_t &stack,
        const cuEVM::block_info_t &block);

    /**
     * The TIMESTAMP operation implementation.
     * Pushes on the stack the timestamp of the current block.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] stack The stack.
     * @param[in] block The block.
     * @return The error code. 0 if no error.
    */
    __host__ __device__ int32_t TIMESTAMP(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        cuEVM::evm_stack_t &stack,
        const cuEVM::block_info_t &block);

    /**
     * The NUMBER operation implementation.
     * Pushes on the stack the number of the current block.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] stack The stack.
     * @param[in] block The block.
     * @return The error code. 0 if no error.
    */
    __host__ __device__ int32_t NUMBER(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        cuEVM::evm_stack_t &stack,
        const cuEVM::block_info_t &block);

    /**
     * The DIFFICULTY/PREVRANDAO operation implementation.
     * Pushes on the stack the difficulty/prevandao of the current block.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] stack The stack.
     * @param[in] block The block.
     * @return The error code. 0 if no error.
    */
    __host__ __device__ int32_t PREVRANDAO(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        cuEVM::evm_stack_t &stack,
        const cuEVM::block_info_t &block);

    /**
     * The GASLIMIT operation implementation.
     * Pushes on the stack the gas limit of the current block.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] stack The stack.
     * @param[in] block The block.
     * @return The error code. 0 if no error.
    */
    __host__ __device__ int32_t GASLIMIT(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        cuEVM::evm_stack_t &stack,
        const cuEVM::block_info_t &block);

    /**
     * The CHAINID operation implementation.
     * Pushes on the stack the chain id of the current block.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] stack The stack.
     * @param[in] block The block.
     * @return The error code. 0 if no error.
    */
    __host__ __device__ int32_t CHAINID(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        cuEVM::evm_stack_t &stack,
        const cuEVM::block_info_t &block);

    /**
     * The BASEFEE operation implementation.
     * Pushes on the stack the base fee of the current block.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] stack The stack.
     * @param[in] block The block.
     * @return The error code. 0 if no error.
    */
    __host__ __device__ int32_t BASEFEE(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        cuEVM::evm_stack_t &stack,
        const cuEVM::block_info_t &block);
}

#endif