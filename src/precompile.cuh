// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2024-03-13
// SPDX-License-Identifier: MIT

#ifndef _PRECOMPILE_H_
#define _PRECOMPILE_H_

#include "include/utils.h"
#include "memory.cuh"
#include "message.cuh"
#include "returndata.cuh"
#include "sha256.cuh"
#include "stack.cuh"

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
 * The sha256 class.
 */
using sha256::sha256_t;

/**
 * The Identity precompile contract
 * MEMCPY through the message data and return data
 * @param[in] arith The arithmetic environment
 * @param[in] gas_limit The gas limit
 * @param[out] gas_used The gas used
 * @param[out] error_code The error code
 * @param[out] return_data The return data
 * @param[in] message The message
 */
__host__ __device__ static void operation_IDENTITY(arith_t &arith, bn_t &gas_limit, bn_t &gas_used,
                                                   uint32_t &error_code, return_data_t &return_data,
                                                   message_t &message) {
    // Identity function

    // static gas
    cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_PRECOMPILE_IDENTITY);

    // dynamic gas
    // compute the dynamic gas cost
    bn_t length;
    size_t length_size = message.get_data_size();
    arith.cgbn_from_size_t(length, length_size);
    arith.memory_cost(gas_used, length);

    if (arith.has_gas(gas_limit, gas_used, error_code)) {
        bn_t index;
        cgbn_set_ui32(arith._env, index, 0);
        return_data.set(message.get_data(index, length, length_size), message.get_data_size());
        error_code = ERR_RETURN;
    }
}

/**
 * The SHA2-256 precompile contract
 * SHA2 through the message data and return data
 * @param[in] arith The arithmetic environment
 * @param[in] gas_limit The gas limit
 * @param[out] gas_used The gas used
 * @param[out] error_code The error code
 * @param[out] return_data The return data
 * @param[in] message The message
 * @param[in] sha The sha256 class
 */
__host__ __device__ static void operation_SHA256(arith_t &arith, bn_t &gas_limit, bn_t &gas_used, uint32_t &error_code,
                                                 return_data_t &return_data, message_t &message, sha256_t &sha) {
    // static gas
    cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_PRECOMPILE_SHA256);

    // dynamic gas
    // compute the dynamic gas cost
    bn_t length;
    size_t length_size = message.get_data_size();
    arith.cgbn_from_size_t(length, length_size);
    arith.sha256_cost(gas_used, length);

    if (arith.has_gas(gas_limit, gas_used, error_code)) {
        bn_t index;
        cgbn_set_ui32(arith._env, index, 0);
        uint8_t hash[32];
        sha.sha(message.get_data(index, length, length_size), length_size, &(hash[0]));
        return_data.set(&(hash[0]), 32);
        error_code = ERR_RETURN;
    }
}

__host__ __device__ static void operation_BLAKE2(arith_t &arith, bn_t &gas_limit, bn_t &gas_used, uint32_t &error_code,
                                                 return_data_t &return_data, message_t &message) {
    uint32_t rounds;
    size_t actual_size_read;
    bn_t offset, size;

    arith.cgbn_from_size_t(offset, 0);
    arith.cgbn_from_size_t(size, 4);



    rounds = *((uint32_t *)message.get_data(offset, size, actual_size_read));

    if (actual_size_read < 4) { // maybe we don't need this
        error_code = ERR_RETURN;
        return;
    }

    arith.blake2_cost(gas_used, rounds);

    // todo if we check the input size == 212 now, we wouldn't need to do other size checks in message.get_data?
    if (message.get_data_size() != 212) {
        error_code = ERR_RETURN;
        return;
    }

    if (arith.has_gas(gas_limit, gas_used, error_code)) {
        uint8_t h[64];
        uint8_t m[128];
        uint8_t t[16];
        uint8_t f[1];


        ONE_THREAD_PER_INSTANCE(memcpy(h, message._content->data.data + 4, 64));
        ONE_THREAD_PER_INSTANCE(memcpy(m, message._content->data.data + 4 + 64, 128));
        ONE_THREAD_PER_INSTANCE(memcpy(t, message._content->data.data + 4 + 64 + 128, 16));
        ONE_THREAD_PER_INSTANCE(memcpy(f, message._content->data.data + 4 + 64 + 128 + 16, 1));

        // blake2b(h, m, t, f, rounds); // todo_cl

    }
}
}  // namespace precompile_operations

#endif