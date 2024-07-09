#ifndef __GAS_COST_H__
#define __GAS_COST_H__

#include "evm_defines.h"
#include "arith.cuh"
#include "access_state.cuh"
#include "touch_state.cuh"
#include "transaction.cuh"

#define GAS_ZERO 0
#define GAS_JUMP_DEST 1
#define GAS_BASE 2
#define GAS_VERY_LOW 3
#define GAS_LOW 5
#define GAS_MID 8
#define GAS_HIGH 10
#define GAS_WARM_ACCESS 100
#define GAS_WARM_SLOAD GAS_WARM_ACCESS
#define GAS_SLOAD GAS_WARM_SLOAD
#define GAS_ACCESS_LIST_ADDRESS 2400
#define GAS_ACCESS_LIST_STORAGE 1900
#define GAS_COLD_ACCOUNT_ACCESS 2600
#define GAS_COLD_SLOAD 2100
#define GAS_STORAGE_SET 20000
#define GAS_STORAGE_RESET 2900
#define GAS_STORAGE_CLEAR_REFUND 4800 //can be defined as GAS_SRESET + GAS_ACCESS_LIST_STORAGE
#define GAS_SSTORE_RESET 5000 - GAS_COLD_SLOAD // change eip-2929 from eip-2020
#define GAS_SSTORE_CLEARS_SCHEDULE 4800 // EIP-3529 SSTORE_RESET - COLD_SLOAD_COST + ACCESS_LIST_STORAGE_KEY = 5000 - 2100 + 1900 = 4800
#define GAS_WARM_SSOTRE_RESET 1900 // SSTORE_RESET - COLD_SLOAD_COST
#define GAS_SELFDESTRUCT 5000
#define GAS_CREATE 32000
#define GAS_CODE_DEPOSIT 200
#define GAS_CALL_VALUE 9000
#define GAS_CALL_STIPEND 2300
#define GAS_NEW_ACCOUNT 25000
#define GAS_EXP 10
#define GAS_EXP_BYTE 50
#define GAS_MEMORY 3
#define GAS_TX_CREATE 32000
#define GAS_TX_DATA_ZERO 4
#define GAS_TX_DATA_NONZERO 16
#define GAS_TRANSACTION 21000
#define GAS_LOG 375
#define GAS_LOG_DATA 8
#define GAS_LOG_TOPIC 375
#define GAS_KECCAK256 30
#define GAS_KECCAK256_WORD 6
#define GAS_COPY 3
#define GAS_BLOCKHASH 20
#define GAS_STIPEND 2300

#ifdef EIP_3860
    #define GAS_INITCODE_WORD_COST 2
#else
    #define GAS_INITCODE_WORD_COST 0
#endif

#define GAS_PRECOMPILE_ECRECOVER 3000
#define GAS_PRECOMPILE_SHA256 60
#define GAS_PRECOMPILE_SHA256_WORD 12
#define GAS_PRECOMPILE_RIPEMD160 600
#define GAS_PRECOMPILE_RIPEMD160_WORD 120
#define GAS_PRECOMPILE_IDENTITY 15
#define GAS_PRECOMPILE_IDENTITY_WORD 3
#define GAS_PRECOMPILE_MODEXP_MAX 200
#define GAS_PRECOMPILE_ECADD 150
#define GAS_PRECOMPILE_ECMUL 6000
#define GAS_PRECOMPILE_ECPAIRING 45000
#define GAS_PRECOMPILE_ECPAIRING_PAIR 34000
#define GAS_PRECOMPILE_BLAKE2_ROUND 1

namespace cuEVM {
    namespace gas_cost {
        /**
         * Verify if is enough gas for the operation.
         * @param[in] arith The arithmetic environment
         * @param[in] gas_limit The gas limit
         * @param[in] gas_used The gas used
         * @param[inout] error_code The error code
         * @return 1 for enough gas and no previous errors, 0 otherwise
         */
        __host__ __device__ int32_t has_gas(
            ArithEnv &arith,
            bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &error_code);
        /**
         * Compute the max gas call.
         * @param[in] arith The arithmetic environment
         * @param[out] gas_capped The gas capped
         * @param[in] gas_limit The gas limit
         * @param[in] gas_used The gas used
         */
        __host__ __device__ void max_gas_call(
            ArithEnv &arith,
            bn_t &gas_capped,
            bn_t &gas_limit,
            bn_t &gas_used);
        /**
         * Add the gas cost for the given length of bytes, but considering
         * evm words.
         * @param[in] arith The arithmetic environment
         * @param[inout] gas_used The gas used
         * @param[in] length The length of the bytes
         * @param[in] gas_per_word The gas per evm word
         */
        __host__ __device__ void evm_words_gas_cost
        (
            ArithEnv &arith,
            bn_t &gas_used,
            bn_t &length,
            uint32_t gas_per_word);
        /**
         * Add the cost for initiliasation code.
         * EIP-3860: https://eips.ethereum.org/EIPS/eip-3860
         * @param[in] arith The arithmetic environment
         * @param[inout] gas_used The gas used
         * @param[in] initcode_length The length of the initcode
         */
        __host__ __device__ void initcode_cost(
            ArithEnv &arith,
            bn_t &gas_used,
            bn_t &initcode_length);
        /**
         * Add the cost for keccak hashing.
         * @param[in] arith The arithmetic environment
         * @param[inout] gas_used The gas used
         * @param[in] length The length of the data in bytes
         */
        __host__ __device__ void keccak_cost(
            ArithEnv &arith,
            bn_t &gas_used,
            bn_t &length);
        /**
         * Add the cost for memory operation on call data/return data.
         * @param[in] arith The arithmetic environment
         * @param[inout] gas_used The gas used
         * @param[in] length The length of the data in bytes
         */
        __host__ __device__ void memory_cost(
            ArithEnv &arith,
            bn_t &gas_used,
            bn_t &length);
        /**
         * Add the cost for sha256 hashing.
         * @param[in] arith The arithmetic environment
         * @param[inout] gas_used The gas used
         * @param[in] length The length of the data in bytes
         */
        __host__ __device__ void sha256_cost(
            ArithEnv &arith,
            bn_t &gas_used,
            bn_t &length);
        /**
         * Add the dynamic cost for ripemd160 hashing.
         * @param[in] arith The arithmetic environment
         * @param[inout] gas_used The gas used
         * @param[in] length The length of the data in bytes
         */
        __host__ __device__ void ripemd160_cost(
            ArithEnv &arith,
            bn_t &gas_used,
            bn_t &length);

        /**
         * Add the dynamics cost for blake2 hashing.
         * @param[in] arith The arithmetic environment
         * @param[inout] gas_used The gas used
         * @param[in] rounds Number of rounds (big-endian unsigned integer)
         */
        __host__ __device__ void blake2_cost(
            ArithEnv &arith,
            bn_t &gas_used,
            uint32_t rounds);

        /**
         * Add the pairing cost to the gas used.
         * @param[in] arith The arithmetic environment
         * @param[inout] gas_used The gas used
         * @param[in] data_size The size of the data in bytes
         */
        __host__ __device__ void ecpairing_cost(
            ArithEnv &arith,
            bn_t &gas_used,
            size_t data_size
        );

        /**
         * Add the cost for the SLOAD operation.
         * @param[in] arith The arithmetic environment
         * @param[inout] gas_used The gas used
         * @param[in] touch_state The touch state
         * @param[in] address The address of the account
         * @param[in] key The key of the storage
         * @return 1 for success, 0 for failure
         */
        __host__ __device__ int32_t sload_cost(
            ArithEnv &arith,
            bn_t &gas_used,
            cuEVM::state::AccessState &access_state,
            const bn_t &address,
            const bn_t &key);
        
        /**
         * Add the cost and refund for the SSTORE operation.
         * @param[in] arith The arithmetic environment
         * @param[inout] gas_used The gas used
         * @param[inout] gas_refund The refund
         * @param[in] touch_state The touch state
         * @param[in] address The address of the account
         * @param[in] key The key of the storage
         * @param[in] value The value of the storage
         * @return 1 for success, 0 for failure
         */
        __host__ __device__ int32_t sstore_cost(
            ArithEnv &arith,
            bn_t &gas_used,
            bn_t &gas_refund,
            cuEVM::state::TouchState &touch_state,
            cuEVM::state::AccessState &access_state,
            const bn_t &address,
            const bn_t &key,
            const bn_t &value);
        
        /**
         * Get the transaction intrinsic gas.
         * @param[in] arith The arithmetic environment
         * @param[in] transaction The transaction
         * @param[out] gas_intrinsic The intrinsic gas
         * @return 1 for success, 0 for failure
         */
        __host__ __device__ int32_t transaction_intrinsic_gas(
            ArithEnv &arith,
            const cuEVM::transaction::transaction_t &transaction,
            bn_t &gas_intrinsic);
    } // namespace gas_cost
} // namespace cuEVM


#endif  // __GAS_COST_H__
