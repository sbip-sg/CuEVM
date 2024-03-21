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
            /*
            ONE_THREAD_PER_INSTANCE(
                memcpy(output + 12, hash, 20);
            )
            */
            return_data.set(output, 32);
            error_code = ERR_RETURN;
        }
    }

    __host__ __device__ static void operation_MODEXP(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        return_data_t &return_data,
        message_t &message) {
        
        printf("MODEXP\n");
        // get the size of the base, exponent and modulus
        // from the call data
        bn_t base_size, exponent_size, modulus_size;
        bn_t index, length;
        cgbn_set_ui32(arith._env, index, 0);
        cgbn_set_ui32(arith._env, length, 32);
        uint8_t *size_bytearray;
        size_t size_bytearray_len;
        size_bytearray = message.get_data(index, length, size_bytearray_len);
        arith.cgbn_from_fixed_memory(
            base_size,
            size_bytearray,
            size_bytearray_len);
        cgbn_add(arith._env, index, index, length);
        size_bytearray = message.get_data(index, length, size_bytearray_len);
        arith.cgbn_from_fixed_memory(
            exponent_size,
            size_bytearray,
            size_bytearray_len);
        cgbn_add(arith._env, index, index, length);
        size_bytearray = message.get_data(index, length, size_bytearray_len);
        arith.cgbn_from_fixed_memory(
            modulus_size,
            size_bytearray,
            size_bytearray_len);
        cgbn_add(arith._env, index, index, length);
        bn_t data_index;
        cgbn_set(arith._env, data_index, index);


        // first compute the gas cost follow by special casses
        bn_t max_length;
        cgbn_set(arith._env, max_length, base_size);
        if (cgbn_compare(arith._env, max_length, modulus_size) < 0) {
            cgbn_set(arith._env, max_length, modulus_size);
        }
        printf("max_length: %u\n", cgbn_get_ui32(arith._env, max_length));
        // words = (max_length + 7) / 8
        // add 7
        cgbn_add_ui32(arith._env, max_length, max_length, 7);
        // divide by 8
        cgbn_shift_right(arith._env, max_length, max_length, 3);
        bn_t multiplication_complexity;
        cgbn_mul(arith._env, multiplication_complexity, max_length, max_length);
        
        printf("multiplication_complexity: %u\n", cgbn_get_ui32(arith._env, multiplication_complexity));

        size_t tmp_size;
        uint8_t *tmp;
        cgbn_set(arith._env, index, data_index);
        cgbn_add(arith._env, index, index, base_size);
        tmp = message.get_data(index, exponent_size, tmp_size);

        size_t exponent_bit_length;
        if (tmp_size > 0) {
            size_t exponent_byte_length;
            exponent_byte_length = tmp_size - 1;
            while (
                (exponent_byte_length > 0) && 
                (tmp[tmp_size - exponent_byte_length - 1] == 0)
            ) {
                exponent_byte_length--;
            }
            uint8_t exponent_byte;
            exponent_byte = tmp[tmp_size - exponent_byte_length - 1];
            if (exponent_byte == 0)
                exponent_bit_length = 0;
            else
                exponent_bit_length = (exponent_byte_length + 1) * 8;
            if (exponent_bit_length != 0) {
                while (
                    (exponent_byte & 0x80) == 0
                ) {
                    exponent_bit_length--;
                    exponent_byte <<= 1;
                }
            }
        } else {
            exponent_bit_length = 0;
        }
        bn_t exponent_bit_length_bn;
        bn_t tmp_size_bn;
        arith.cgbn_from_size_t(tmp_size_bn, tmp_size);
        if (exponent_bit_length > 0) {
            arith.cgbn_from_size_t(exponent_bit_length_bn, exponent_bit_length);
            cgbn_sub(arith._env, tmp_size_bn, exponent_size, tmp_size_bn);
            cgbn_mul_ui32(arith._env, tmp_size_bn, tmp_size_bn, 8);
            cgbn_add(arith._env, exponent_bit_length_bn, exponent_bit_length_bn, tmp_size_bn);
        } else {
            cgbn_set_ui32(arith._env, exponent_bit_length_bn, 0);
        }
        printf("exponent_bit_length: %u\n", cgbn_get_ui32(arith._env, exponent_bit_length_bn));
        bn_t iteration_count;
        cgbn_set_ui32(arith._env, iteration_count, 0);
        if (cgbn_compare_ui32(arith._env, exponent_size, 32) <= 0) {
            if (exponent_bit_length != 0) {
                // exponent.bit_length() - 1
                cgbn_sub_ui32(arith._env, iteration_count, exponent_bit_length_bn, 1);
            }
        } else {
            // elif Esize > 32: iteration_count = (8 * (Esize - 32)) + ((exponent & (2**256 - 1)).bit_length() - 1)
            cgbn_sub_ui32(arith._env, iteration_count, exponent_size, 32);
            cgbn_mul_ui32(arith._env, iteration_count, iteration_count, 8);
            cgbn_add(arith._env, iteration_count, iteration_count, exponent_bit_length_bn);
            cgbn_sub_ui32(arith._env, iteration_count, iteration_count, 1);
        }
        printf("iteration_count: %u\n", cgbn_get_ui32(arith._env, iteration_count));
        // iteration_count = max(iteration_count, 1)
        if (cgbn_compare_ui32(arith._env, iteration_count, 1) < 0) {
            cgbn_set_ui32(arith._env, iteration_count, 1);
        }
        printf("iteration_count: %u\n", cgbn_get_ui32(arith._env, iteration_count));

        bn_t dynamic_gas;
        // dynamic_gas = max(200, multiplication_complexity * iteration_count / 3)
        cgbn_mul(arith._env, dynamic_gas, iteration_count, multiplication_complexity);
        cgbn_div_ui32(arith._env, dynamic_gas, dynamic_gas, 3);
        if (cgbn_compare_ui32(arith._env, dynamic_gas, 200) < 0) {
            cgbn_set_ui32(arith._env, dynamic_gas, 200);
        }
        printf("dynamic_gas: %u\n", cgbn_get_ui32(arith._env, dynamic_gas));
        cgbn_add(arith._env, gas_used, gas_used, dynamic_gas);


        if (arith.has_gas(gas_limit, gas_used, error_code)) {
            // special cases
            if (cgbn_compare_ui32(arith._env, modulus_size, 0) == 0) {
                return_data.set(NULL, 0);
                error_code = ERR_RETURN;
                return;
            }

            // prepare result
            size_t modulus_len;
            if (arith.size_t_from_cgbn(modulus_len, modulus_size)) {
                error_code = ERROR_PRECOMPILE_MODEXP_OVERFLOW;
                return;
            }
            SHARED_MEMORY uint8_t *result, *modulus;
            ONE_THREAD_PER_INSTANCE(
                result = new uint8_t[modulus_len];
                memset(result, 0, modulus_len);
                modulus = new uint8_t[modulus_len];
                memset(modulus, 0, modulus_len);
            )
            cgbn_set(arith._env, index, data_index);
            cgbn_add(arith._env, index, index, base_size);
            cgbn_add(arith._env, index, index, exponent_size);
            tmp = message.get_data(index, modulus_size, tmp_size);
            for (size_t i = 0; i < tmp_size; i++) {
                modulus[i] = tmp[i];
            }
            bigint modulus_bigint[1];
            bigint_init(modulus_bigint);
            bigint_from_bytes(modulus_bigint, modulus, modulus_len);
            // 0 modulus
            if (bigint_cmp_abs_word(modulus_bigint, 0) == 0) {
                return_data.set(result, modulus_len);
                error_code = ERR_RETURN;
                ONE_THREAD_PER_INSTANCE(
                    delete[] result;
                    delete[] modulus;
                )
                bigint_free(modulus_bigint);
                return;
            }
            // 0 exponent
            bigint result_bigint[1];
            bigint_init(result_bigint);
            bigint tmp_bigint[1];
            bigint_init(result_bigint);
            if (cgbn_compare_ui32(arith._env, exponent_bit_length_bn, 0) == 0) {
            
                bigint_from_word(tmp_bigint, 1);
                bigint_mod(result_bigint, tmp_bigint, modulus_bigint);
                bigint_to_bytes(result, result_bigint, modulus_len);
                return_data.set(result, modulus_len);
                error_code = ERR_RETURN;
                ONE_THREAD_PER_INSTANCE(
                    delete[] result;
                    delete[] modulus;
                )
                bigint_free(modulus_bigint);
                bigint_free(result_bigint);
                bigint_free(tmp_bigint);
                return;
            }
            // 0 base and not zero exponent
            
            // get the base, exponent and modulus
            SHARED_MEMORY uint8_t *base, *exponent;
            size_t base_len, exponent_len, modulus_len;

            overflow = arith.size_t_from_cgbn(base_len, base_size);
            overflow = (
                overflow |
                arith.size_t_from_cgbn(exponent_len, exponent_size)
            );
            overflow = (
                overflow |
                arith.size_t_from_cgbn(modulus_len, modulus_size)
            );
            if (overflow) {
                error_code = ERROR_PRECOMPILE_MODEXP_OVERFLOW;
                return;
            }

        }




        printf("base_len: %lu\n", base_len);
        printf("exponent_len: %lu\n", exponent_len);
        printf("modulus_len: %lu\n", modulus_len);


        if (arith.has_gas(gas_limit, partial_gas, error_code)) {

            ONE_THREAD_PER_INSTANCE(
                exponent = new uint8_t[exponent_len];
                memset(exponent, 0, exponent_len);
            )

            size_t tmp_size;
            uint8_t *tmp;
            cgbn_set(arith._env, index, data_index);
            cgbn_add(arith._env, index, index, base_size);
            tmp = message.get_data(index, exponent_size, tmp_size);
            for (size_t i = 0; i < tmp_size; i++) {
                exponent[i] = tmp[i];
            }



            
            if (arith.has_gas(gas_limit, gas_used, error_code)) {
                ONE_THREAD_PER_INSTANCE(
                    base = new uint8_t[base_len];
                    memset(base, 0, base_len);
                )

                size_t tmp_size;
                uint8_t *tmp;
                cgbn_set(arith._env, index, data_index);
                tmp = message.get_data(index, base_size, tmp_size);
                for (size_t i = 0; i < tmp_size; i++) {
                    base[i] = tmp[i];
                }
                cgbn_add(arith._env, index, index, base_size);
                cgbn_add(arith._env, index, index, exponent_size);
                tmp = message.get_data(index, modulus_size, tmp_size);
                for (size_t i = 0; i < tmp_size; i++) {
                    modulus[i] = tmp[i];
                }

                // print the base, exponent and modulus
                printf("base: ");
                for (size_t i = 0; i < base_len; i++) {
                    printf("%02x", base[i]);
                }
                printf("\n");
                printf("exponent: ");
                for (size_t i = 0; i < exponent_len; i++) {
                    printf("%02x", exponent[i]);
                }
                printf("\n");
                printf("modulus: ");
                for (size_t i = 0; i < modulus_len; i++) {
                    printf("%02x", modulus[i]);
                }
                printf("\n");
                // perform the modular exponentiation
                SHARED_MEMORY uint8_t *result;
                size_t result_len;
                result_len = modulus_len;
                ONE_THREAD_PER_INSTANCE(
                    result = new uint8_t[modulus_len];
                    memset(result, 0, modulus_len);
                )
                char buf[65536];
                bigint a[1], b[1], c[1], d[1];
                bigint_init(a);
                bigint_init(b);
                bigint_init(c);
                bigint_init(d);
                bigint_from_bytes(a, base, base_len);
                bigint_from_bytes(b, exponent, exponent_len);
                bigint_from_bytes(c, modulus, modulus_len);
                printf("a: %s\n", bigint_write(buf, sizeof(buf), a));
                printf("b: %s\n", bigint_write(buf, sizeof(buf), b));
                printf("c: %s\n", bigint_write(buf, sizeof(buf), c));
                if (bigint_cmp_abs_word(c, 0) == 0) {
                    bigint_from_word(d, 0);
                } else if (
                    // (bigint_cmp_abs_word(a, 0) == 0) &&
                    (bigint_cmp_abs_word(b, 0) == 0)
                ) {
                    bigint_from_word(a, 1);
                    bigint_from_word(b, 1);
                    bigint_pow_mod(d, a, b, c);
                } else {
                    bigint_pow_mod(d, a, b, c);
                }
                printf("d: %s\n", bigint_write(buf, sizeof(buf), d));
                //bigint_mul(c, a, b);
                bigint_to_bytes(result, d, result_len);
                bigint_free(a);
                bigint_free(b);
                bigint_free(c);
                bigint_free(d);

                return_data.set(result, result_len);
                error_code = ERR_RETURN;
                ONE_THREAD_PER_INSTANCE(
                    delete[] result;
                    delete[] base;
                    delete[] modulus;
                )
            }
            ONE_THREAD_PER_INSTANCE(
                delete[] exponent;
            )

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
