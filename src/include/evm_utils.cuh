#ifndef _CUEVM_EVM_UTILS_H_
#define _CUEVM_EVM_UTILS_H_

#include <stdint.h>
#include <cuda.h>
#include "arith.cuh"
#include "byte_array.cuh"

namespace cuEVM {
    namespace utils {

    /**
     * Get the contract address from the sender address and the sender nonce.
     * For the simple CREATE operation.
     * @param[in] arith The arithmetic environment
     * @param[out] contract_address The contract address
     * @param[in] sender_address The sender address
     * @param[in] sender_nonce The sender nonce
     */
    __host__ __device__ int32_t get_contract_address_create(
        ArithEnv &arith,
        bn_t &contract_address,
        const bn_t &sender_address,
        const bn_t &sender_nonce);

    /**
     * Get the contract address from the sender address, the salt, and the init code.
     * For the CREATE2 operation.
     * @param[in] arith The arithmetic environment
     * @param[out] contract_address The contract address
     * @param[in] sender_address The sender address
     * @param[in] salt The salt
     * @param[in] init_code The init code
     */
    __host__ __device__ int32_t get_contract_address_create2(
      ArithEnv &arith,
      bn_t &contract_address,
      const bn_t &sender_address,
      const bn_t &salt,
      const cuEVM::byte_array_t &init_code);
    }
}

#endif // _CUEVM_EVM_UTILS_H_