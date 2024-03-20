// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2024-03-13
// SPDX-License-Identifier: MIT

#ifndef _PRECOMPILE_H_
#define _PRECOMPILE_H_

#include "include/utils.h"
#include "stack.cuh"
#include "memory.cuh"
#include "message.cuh"
#include "returndata.cuh"
#include "sha256.cuh"
#include "ripemd160.cuh"
#include "blake2/blake2f.cuh"

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
    __host__ __device__ static void operation_IDENTITY(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        return_data_t &return_data,
        message_t &message) {
        // Identity function

        // static gas
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_PRECOMPILE_IDENTITY);

        // dynamic gas
        // compute the dynamic gas cost
        bn_t length;
        size_t length_size = message.get_data_size();
        arith.cgbn_from_size_t(length, length_size);
        arith.memory_cost(
            gas_used,
            length
        );


        if (arith.has_gas(gas_limit, gas_used, error_code)) {
            bn_t index;
            cgbn_set_ui32(arith._env, index, 0);
            return_data.set(
                message.get_data(index, length, length_size),
                message.get_data_size()
            );
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
    __host__ __device__ static void operation_SHA256(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        return_data_t &return_data,
        message_t &message,
        sha256_t &sha) {

        // static gas
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_PRECOMPILE_SHA256);

        // dynamic gas
        // compute the dynamic gas cost
        bn_t length;
        size_t length_size = message.get_data_size();
        arith.cgbn_from_size_t(length, length_size);
        arith.sha256_cost(
            gas_used,
            length
        );

        if (arith.has_gas(gas_limit, gas_used, error_code)) {
            bn_t index;
            cgbn_set_ui32(arith._env, index, 0);
            uint8_t hash[32];
            sha.sha(
                message.get_data(index, length, length_size),
                length_size,
                &(hash[0]));
            return_data.set(
                &(hash[0]),
                32
            );
            error_code = ERR_RETURN;
        }
    }

    __host__ __device__ __forceinline__ static void operation_RIPEMD160(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        return_data_t &return_data,
        message_t &message) {

        // static gas
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_PRECOMPILE_RIPEMD160);

        size_t size;
        size = message.get_data_size();
        
        bn_t length;
        arith.cgbn_from_size_t(length, size);
        
        arith.ripemd160_cost(gas_used, length);

        if (arith.has_gas(gas_limit, gas_used, error_code)) {
            // get the input
            bn_t index;
            cgbn_set_ui32(arith._env, index, 0);
            SHARED_MEMORY uint8_t *input;
            input = message.get_data(index, length, size);
            
            // output allocation
            SHARED_MEMORY uint8_t output[32];
            ONE_THREAD_PER_INSTANCE(
                memset(&(output[0]), 0, 32);
            )
            SHARED_MEMORY uint8_t *hash;
            hash = output+12;
            ripemd160(input, size, hash);
            ONE_THREAD_PER_INSTANCE(
                memcpy(output + 12, hash, 20);)
            return_data.set(output, 32);
            error_code = ERR_RETURN;
        }
    }


    __host__ __device__ static void operation_BLAKE2(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        return_data_t &return_data,
        message_t &message) {
        
        // expecting 213 bytes inputs
        size_t size = message.get_data_size();
        if (size != 213) {
            error_code = ERROR_PRECOMPILE_UNEXPECTED_INPUT_LENGTH;
            return;
        }

        bn_t index;
        cgbn_set_ui32(arith._env, index, 0);
        bn_t length;
        arith.cgbn_from_size_t(length, size);

        SHARED_MEMORY uint8_t *input;
        input = message.get_data(index, length, size);
        uint8_t f = input[212];
        
        // final byte must be 1 or 0
        if ((f>>1) != 0) {  
            error_code = ERROR_PRECOMPILE_UNEXPECTED_INPUT;
            return;
        }

        uint32_t rounds;
        rounds = (
            ((uint32_t)input[0] << 24) |
            ((uint32_t)input[1] << 16) |
            ((uint32_t)input[2] << 8) |
            ((uint32_t)input[3])
        );

        arith.blake2_cost(gas_used, rounds);

        if (arith.has_gas(gas_limit, gas_used, error_code)) {
            uint64_t h[8];
            uint64_t m[16];
            uint64_t t[2];

            ONE_THREAD_PER_INSTANCE(memcpy(h, &(input[4]), 64);)
            ONE_THREAD_PER_INSTANCE(memcpy(m, &(input[68]), 128);)
            ONE_THREAD_PER_INSTANCE(memcpy(t, &(input[196]), 16);)

            blake2f(rounds, h, m, t, f);

            return_data.set((uint8_t *)h, 64);
            error_code = ERR_RETURN;
        }
    }

}

#endif
