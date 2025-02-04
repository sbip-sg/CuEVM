
#pragma once

#include <CuBigInt/bigint.cuh>
#include <CuCrypto/blake2.cuh>
#include <CuCrypto/keccak.cuh>
#include <CuCrypto/ripemd160.cuh>
#include <CuCrypto/sha256.cuh>
#include <CuEVM/core/evm_call_context.cuh>
#include <CuEVM/core/memory.cuh>
#include <CuEVM/core/return_data.cuh>
#include <CuEVM/core/transaction.cuh>
#include <CuEVM/gas_cost.cuh>
#include <CuEVM/utils/ecc.cuh>
#include <CuEVM/utils/evm_defines.cuh>

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
__device__ int32_t operation_IDENTITY(gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_call_context_t *call_context);

/**
 * The SHA2-256 precompile contract
 * SHA2 through the message data and return data
 * @param[in] gas_limit The gas limit
 * @param[out] gas_used The gas used
 * @param[out] error_code The error code
 * @param[out] return_data The return data
 * @param[in] message The message
 */
__device__ int32_t operation_SHA256(gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_call_context_t *call_context);

__device__ int32_t operation_RIPEMD160(gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_call_context_t *call_context);

__device__ int32_t operation_MODEXP(gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_call_context_t *call_context);

__device__ int32_t operation_BLAKE2(gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_call_context_t *call_context);

__device__ int32_t operation_ecRecover(CuEVM::EccConstants *constants, gas_t &gas_limit, gas_t &gas_used,
                                       CuEVM::evm_call_context_t *call_context);

__device__ int32_t operation_ecAdd(CuEVM::EccConstants *constants, gas_t &gas_limit, gas_t &gas_used,
                                   CuEVM::evm_call_context_t *call_context);

__device__ int32_t operation_ecMul(CuEVM::EccConstants *constants, gas_t &gas_limit, gas_t &gas_used,
                                   CuEVM::evm_call_context_t *call_context);

__device__ int32_t operation_ecPairing(CuEVM::EccConstants *constants, gas_t &gas_limit, gas_t &gas_used,
                                       CuEVM::evm_call_context_t *call_context);

}  // namespace precompile_operations
}  // namespace CuEVM
