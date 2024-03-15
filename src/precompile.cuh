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
#include "ecc.cuh"
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

  __host__ __device__ __forceinline__ static void operation_RIPEMD160(arith_t &arith,
                                                                      bn_t &gas_limit,
                                                                      bn_t &gas_used,
                                                                      uint32_t &error_code,
                                                                      return_data_t &return_data,
                                                                      message_t &message
                                                                      )
  {
    cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_PRECOMPILE_RIPEMD160);

    size_t size;
    uint8_t *input;
    size = message._content->data.size;
    input = message._content->data.data;

    uint8_t output[32] = {0};
    uint8_t *hash;
    hash = output+12;
    bn_t length;
    arith.cgbn_from_size_t(length, size);
    arith.ripemd160_cost(gas_used, length);

    if (arith.has_gas(gas_limit, gas_used, error_code)) {
      ripemd160(input, size, hash);
      ONE_THREAD_PER_INSTANCE(memcpy(output + 12, hash, 20);)
      return_data.set(output, 32);
      error_code = ERR_RETURN;
    }
  }


   __host__ __device__  static void operation_ecRecover(arith_t &arith,
                                                            keccak::keccak_t* _keccak,
                                                            bn_t &gas_limit,
                                                            bn_t &gas_used,
                                                            uint32_t &error_code,
                                                            return_data_t &return_data,
                                                            message_t &message
                                                            )
  {

    size_t size;
    uint8_t *input;
    size = message._content->data.size;
    input = message._content->data.data;

    cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_PRECOMPILE_ECRECOVER);
    ecc::signature_t signature;
    bn_t msg_hash, v, r, s, signer;
    evm_word_t scratch_pad;

    arith.cgbn_from_memory(msg_hash, input);
    arith.cgbn_from_memory(v, input + 32);
    arith.cgbn_from_memory(r, input + 64);
    arith.cgbn_from_memory(s, input + 96);
    signature.v = cgbn_get_ui32(arith._env, v);
    cgbn_store(arith._env, &signature.msg_hash, msg_hash);
    cgbn_store(arith._env, &signature.r, r);
    cgbn_store(arith._env, &signature.s, s);
    if (arith.has_gas(gas_limit, gas_used, error_code)) {
        uint8_t output[32];
        ecc::ec_recover(arith, _keccak, signature, signer);
        cgbn_store(arith._env, &scratch_pad, signer);
        arith.byte_array_from_cgbn_memory(output, size, scratch_pad);
        return_data.set(output, 32);

      error_code = ERR_RETURN;
    }
  }

   __host__ __device__  static void operation_ecAdd(arith_t &arith,
                                                            bn_t &gas_limit,
                                                            bn_t &gas_used,
                                                            uint32_t &error_code,
                                                            return_data_t &return_data,
                                                            message_t &message
                                                            )
  {

    size_t size;
    uint8_t *input;
    size = message._content->data.size;
    input = message._content->data.data;
    ecc::Curve curve = ecc::get_curve(arith,128);
    cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_PRECOMPILE_ECADD);
    bn_t x1, y1, x2, y2;
    evm_word_t scratch_pad;

    arith.cgbn_from_memory(x1, input);
    arith.cgbn_from_memory(y1, input + 32);
    arith.cgbn_from_memory(x2, input + 64);
    arith.cgbn_from_memory(y2, input + 96);

    if (arith.has_gas(gas_limit, gas_used, error_code)) {
        uint8_t output[64];
        int res = ecc::ec_add(arith, curve, x1, y1, x1, y1, x2, y2);
        if (res==0){
            cgbn_store(arith._env, &scratch_pad, x1);
            arith.byte_array_from_cgbn_memory(output, size, scratch_pad);
            cgbn_store(arith._env, &scratch_pad, y1);
            arith.byte_array_from_cgbn_memory(output + 32, size, scratch_pad);
            return_data.set(output, 64);
            error_code = ERR_RETURN;
        } else{
            // consumes all gas
            // TODO
        }
    }
  }

   __host__ __device__  static void operation_ecMul(arith_t &arith,
                                                            bn_t &gas_limit,
                                                            bn_t &gas_used,
                                                            uint32_t &error_code,
                                                            return_data_t &return_data,
                                                            message_t &message
                                                            )
  {

    size_t size;
    uint8_t *input;
    size = message._content->data.size;
    input = message._content->data.data;
    ecc::Curve curve = ecc::get_curve(arith,128);
    cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_PRECOMPILE_ECMUL);
    bn_t x, y, k;
    evm_word_t scratch_pad;

    arith.cgbn_from_memory(x, input);
    arith.cgbn_from_memory(y, input + 32);
    arith.cgbn_from_memory(k, input + 64);

    if (arith.has_gas(gas_limit, gas_used, error_code)) {
        uint8_t output[64];
        int res = ecc::ec_mul(arith, curve, x, y, x, y, k);
        if (res==0){
            cgbn_store(arith._env, &scratch_pad, x);
            arith.byte_array_from_cgbn_memory(output, size, scratch_pad);
            cgbn_store(arith._env, &scratch_pad, y);
            arith.byte_array_from_cgbn_memory(output + 32, size, scratch_pad);
            return_data.set(output, 64);
            error_code = ERR_RETURN;
        } else{
            // consumes all gas
            // TODO
        }
    }
  }

}
#endif
