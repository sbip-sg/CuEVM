#include "include/gas_cost.cuh"
#include "include/error_codes.h"


namespace cuEVM {
    namespace gas_cost {
        __host__ __device__ int32_t has_gas(
            ArithEnv &arith,
            bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &error_code) {
            int32_t gas_sign = cgbn_compare(arith.env, gas_limit, gas_used);
            error_code = (gas_sign < 0) ? ERROR_GAS_LIMIT_EXCEEDED : error_code;
            return (gas_sign >= 0) && (error_code == ERR_NONE);
        }
        
        __host__ __device__ void max_gas_call(
            ArithEnv &arith,
            bn_t &gas_capped,
            bn_t &gas_limit,
            bn_t &gas_used) {
            // compute the remaining gas
            bn_t gas_left;
            cgbn_sub(arith.env, gas_left, gas_limit, gas_used);

            // gas capped = (63/64) * gas_left
            cgbn_div_ui32(arith.env, gas_capped, gas_left, 64);
            cgbn_sub(arith.env, gas_capped, gas_left, gas_capped);
        }
        
        __host__ __device__ void evm_words_gas_cost
        (
            ArithEnv &arith,
            bn_t &gas_used,
            bn_t &length,
            uint32_t gas_per_word) {
            // gas_used += gas_per_word * emv word count of length
            // length = (length + 31) / 32
            bn_t evm_words_gas;
            cgbn_add_ui32(arith.env, evm_words_gas, length, EVM_WORD_SIZE -1);
            cgbn_div_ui32(arith.env, evm_words_gas, evm_words_gas, EVM_WORD_SIZE);
            cgbn_mul_ui32(arith.env, evm_words_gas, evm_words_gas, gas_per_word);
            cgbn_add(arith.env, gas_used, gas_used, evm_words_gas);
        }
        
        __host__ __device__ void initcode_cost(
            ArithEnv &arith,
            bn_t &gas_used,
            bn_t &initcode_length
        ) {
            // gas_used += GAS_INITCODE_WORD_COST * emv word count of initcode
            // length = (initcode_length + 31) / 32
            evm_words_gas_cost(arith, gas_used, initcode_length, GAS_INITCODE_WORD_COST);
        }
        
        __host__ __device__ void keccak_cost(
            ArithEnv &arith,
            bn_t &gas_used,
            bn_t &length) {
            evm_words_gas_cost(arith, gas_used, length, GAS_KECCAK256_WORD);
        }
        
        __host__ __device__ void memory_cost(
            ArithEnv &arith,
            bn_t &gas_used,
            bn_t &length) {
            evm_words_gas_cost(arith, gas_used, length, GAS_MEMORY);
        }
        
        __host__ __device__ void sha256_cost(
            ArithEnv &arith,
            bn_t &gas_used,
            bn_t &length) {
            evm_words_gas_cost(arith, gas_used, length, GAS_PRECOMPILE_SHA256_WORD);
        }
        
        __host__ __device__ void ripemd160_cost(
            ArithEnv &arith,
            bn_t &gas_used,
            bn_t &length) {
            evm_words_gas_cost(arith, gas_used, length, GAS_PRECOMPILE_RIPEMD160_WORD);
        }
        
        __host__ __device__ void blake2_cost(
            ArithEnv &arith,
            bn_t &gas_used,
            uint32_t rounds) {
            // gas_used += GAS_PRECOMPILE_BLAKE2_ROUND * rounds
            bn_t temp;
            cgbn_set_ui32(arith.env, temp, rounds);
            cgbn_mul_ui32(arith.env, temp, temp, GAS_PRECOMPILE_BLAKE2_ROUND);
            cgbn_add(arith.env, gas_used, gas_used, temp);
        }
        
        __host__ __device__ void ecpairing_cost(
            ArithEnv &arith,
            bn_t &gas_used,
            size_t data_size) {
            // gas_used += GAS_PRECOMPILE_ECPAIRING + data_size/192 * GAS_PRECOMPILE_ECPAIRING_PAIR
            cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_PRECOMPILE_ECPAIRING);
            bn_t temp;
            arith.cgbn_from_size_t(temp, data_size);
            cgbn_div_ui32(arith.env, temp, temp, 192);
            cgbn_mul_ui32(arith.env, temp, temp, GAS_PRECOMPILE_ECPAIRING_PAIR);
            cgbn_add(arith.env, gas_used, gas_used, temp);
        }
    } // namespace gas_cost
} // namespace cuEVM