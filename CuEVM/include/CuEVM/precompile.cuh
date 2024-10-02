// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation
// Programme) Author: Stefan-Dan Ciocirlan Data: 2024-03-13
// SPDX-License-Identifier: MIT

#pragma once

#include <CuCrypto/blake2.cuh>
#include <CuCrypto/keccak.cuh>
#include <CuCrypto/ripemd160.cuh>
#include <CuEVM/core/memory.cuh>
#include <CuEVM/core/message.cuh>
#include <CuEVM/core/return_data.cuh>
#include <CuEVM/core/transaction.cuh>
#include <CuEVM/gas_cost.cuh>
#include <CuEVM/state/touch_state.cuh>
#include <CuEVM/utils/arith.cuh>
#include <CuEVM/utils/evm_defines.cuh>
#include <CuBigInt/bigint.cuh>

namespace CuEVM {
/**
 * The precompile contracts
 * 0x00 Invalid
 * 0x01 ecRecover
 * 0x02 SHA-256
 * 0x03 RIPEMD-160
 * 0x04 Identity
 * 0x05 Modexp
 * 0x06 ecAdd
 * 0x07 ecMul
 * 0x08 ecPairing
 * 0x09 Blake2
 */
namespace precompile_operations {
/**
 * The Identity precompile contract
 * MEMCPY through the message data and return data
 */
__host__ __device__ int32_t operation_IDENTITY(
    ArithEnv &arith, bn_t &gas_limit, bn_t &gas_used,
    CuEVM::evm_return_data_t *return_data, CuEVM::evm_message_call_t *message);

/**
 * The SHA2-256 precompile contract
 * SHA2 through the message data and return data
 * @param[in] arith The arithmetic environment
 * @param[in] gas_limit The gas limit
 * @param[out] gas_used The gas used
 * @param[out] error_code The error code
 * @param[out] return_data The return data
 * @param[in] message The message
 */
__host__ __device__ int32_t operation_SHA256(
    ArithEnv &arith, bn_t &gas_limit, bn_t &gas_used,
    CuEVM::evm_return_data_t *return_data, CuEVM::evm_message_call_t *message);

__host__ __device__ int32_t operation_RIPEMD160(
    ArithEnv &arith, bn_t &gas_limit, bn_t &gas_used,
    CuEVM::evm_return_data_t *return_data, CuEVM::evm_message_call_t *message);

__host__ __device__ int32_t operation_MODEXP(
    ArithEnv &arith, bn_t &gas_limit, bn_t &gas_used,
    CuEVM::evm_return_data_t *return_data, CuEVM::evm_message_call_t *message);

__host__ __device__ int32_t operation_BLAKE2(
    ArithEnv &arith, bn_t &gas_limit, bn_t &gas_used,
    CuEVM::evm_return_data_t *return_data, CuEVM::evm_message_call_t *message);

__host__ __device__ int32_t operation_ecRecover(
    ArithEnv &arith, bn_t &gas_limit, bn_t &gas_used,
    CuEVM::evm_return_data_t *return_data, CuEVM::evm_message_call_t *message);

__host__ __device__ int32_t operation_ecAdd(
    ArithEnv &arith, bn_t &gas_limit, bn_t &gas_used,
    CuEVM::evm_return_data_t *return_data, CuEVM::evm_message_call_t *message);

__host__ __device__ int32_t operation_ecMul(
    ArithEnv &arith, bn_t &gas_limit, bn_t &gas_used,
    CuEVM::evm_return_data_t *return_data, CuEVM::evm_message_call_t *message);

__host__ __device__ int32_t operation_ecPairing(
    ArithEnv &arith, bn_t &gas_limit, bn_t &gas_used,
    CuEVM::evm_return_data_t *return_data, CuEVM::evm_message_call_t *message);

}  // namespace precompile_operations
}  // namespace CuEVM
