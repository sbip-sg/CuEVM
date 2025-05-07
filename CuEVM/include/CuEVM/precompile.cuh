#pragma once

#include <CuBigInt/bigint.cuh>
#include <CuCrypto/blake2.cuh>
#include <CuCrypto/keccak.cuh>
#include <CuCrypto/ripemd160.cuh>
#include <CuCrypto/sha256.cuh>
#include <CuEVM/core/evm_call_context.cuh>
#include <CuEVM/core/memory.cuh>
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
/**
 * @namespace precompile_operations
 * @brief Contains implementations of Ethereum's precompiled contracts
 *
 * This namespace contains functions that implement all Ethereum precompiled contracts.
 * Each function handles gas computation and the contract-specific logic.
 */
namespace precompile_operations {
/**
 * The Identity precompile contract
 * MEMCPY through the message data and return data
 *
 * @param[in] gas_limit The gas limit for this operation
 * @param[out] gas_used The gas consumed by this operation
 * @param[in,out] call_context The EVM call context
 * @return Error code: 0 for success, non-zero for failure
 */
__device__ int32_t operation_IDENTITY(gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_call_context_t *call_context);

/**
 * The SHA2-256 precompile contract
 * SHA2 through the message data and return data
 *
 * @param[in] gas_limit The gas limit for this operation
 * @param[out] gas_used The gas consumed by this operation
 * @param[in,out] call_context The EVM call context
 * @return Error code: 0 for success, non-zero for failure
 */
__device__ int32_t operation_SHA256(gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_call_context_t *call_context);

/**
 * The RIPEMD-160 precompile contract
 * Computes the RIPEMD-160 hash of the input data
 *
 * @param[in] gas_limit The gas limit for this operation
 * @param[out] gas_used The gas consumed by this operation
 * @param[in,out] call_context The EVM call context
 * @return Error code: 0 for success, non-zero for failure
 */
__device__ int32_t operation_RIPEMD160(gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_call_context_t *call_context);

/**
 * The Modular Exponentiation precompile contract
 * Computes (base^exp) % mod for arbitrary sized integers
 *
 * @param[in] gas_limit The gas limit for this operation
 * @param[out] gas_used The gas consumed by this operation
 * @param[in,out] call_context The EVM call context
 * @return Error code: 0 for success, non-zero for failure
 */
__device__ int32_t operation_MODEXP(gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_call_context_t *call_context);

/**
 * The Blake2 compression function F precompile contract
 * Implements Blake2 compression function F
 *
 * @param[in] gas_limit The gas limit for this operation
 * @param[out] gas_used The gas consumed by this operation
 * @param[in,out] call_context The EVM call context
 * @return Error code: 0 for success, non-zero for failure
 */
__device__ int32_t operation_BLAKE2(gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_call_context_t *call_context);

/**
 * The ECDSA public key recovery precompile contract
 * Recovers the public key from a signed message
 *
 * @param[in] constants ECC constants for the Curve
 * @param[in] gas_limit The gas limit for this operation
 * @param[out] gas_used The gas consumed by this operation
 * @param[in,out] call_context The EVM call context
 * @return Error code: 0 for success, non-zero for failure
 */
__device__ int32_t operation_ecRecover(CuEVM::EccConstants *constants, gas_t &gas_limit, gas_t &gas_used,
                                       CuEVM::evm_call_context_t *call_context);

/**
 * The Elliptic Curve Addition precompile contract
 * Adds two points on the alt_bn128 curve
 *
 * @param[in] constants ECC constants for the Curve
 * @param[in] gas_limit The gas limit for this operation
 * @param[out] gas_used The gas consumed by this operation
 * @param[in,out] call_context The EVM call context
 * @return Error code: 0 for success, non-zero for failure
 */
__device__ int32_t operation_ecAdd(CuEVM::EccConstants *constants, gas_t &gas_limit, gas_t &gas_used,
                                   CuEVM::evm_call_context_t *call_context);

/**
 * The Elliptic Curve Scalar Multiplication precompile contract
 * Multiplies a point on the alt_bn128 curve by a scalar
 *
 * @param[in] constants ECC constants for the Curve
 * @param[in] gas_limit The gas limit for this operation
 * @param[out] gas_used The gas consumed by this operation
 * @param[in,out] call_context The EVM call context
 * @return Error code: 0 for success, non-zero for failure
 */
__device__ int32_t operation_ecMul(CuEVM::EccConstants *constants, gas_t &gas_limit, gas_t &gas_used,
                                   CuEVM::evm_call_context_t *call_context);

/**
 * The Elliptic Curve Pairing Check precompile contract
 * Performs pairing checks on the alt_bn128 curve for zero-knowledge proofs
 *
 * @param[in] constants ECC constants for the Curve
 * @param[in] gas_limit The gas limit for this operation
 * @param[out] gas_used The gas consumed by this operation
 * @param[in,out] call_context The EVM call context
 * @return Error code: 0 for success, non-zero for failure
 */
__device__ int32_t operation_ecPairing(CuEVM::EccConstants *constants, gas_t &gas_limit, gas_t &gas_used,
                                       CuEVM::evm_call_context_t *call_context);

}  // namespace precompile_operations
}  // namespace CuEVM
