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
     * The keccak class.
     */
    using keccak::keccak_t;

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

        // get the size of the base, exponent and modulus
        // from the call data each of them have 32 byts
        // the missing byutes are consider 0 value bytes
        bn_t base_size, exponent_size, modulus_size;
        bn_t index, length;
        uint32_t dynamic_gas_overflow;
        dynamic_gas_overflow = 0;
        cgbn_set_ui32(arith._env, index, 0);
        cgbn_set_ui32(arith._env, length, 32);
        uint8_t *size_bytearray;
        size_t size_bytearray_len;
        size_bytearray = message.get_data(
            index,
            length,
            size_bytearray_len);
        arith.cgbn_from_fixed_memory(
            base_size,
            size_bytearray,
            size_bytearray_len);
        cgbn_add(arith._env, index, index, length);
        size_bytearray = message.get_data(
            index,
            length,
            size_bytearray_len);
        arith.cgbn_from_fixed_memory(
            exponent_size,
            size_bytearray,
            size_bytearray_len);
        cgbn_add(arith._env, index, index, length);
        size_bytearray = message.get_data(
            index,
            length,
            size_bytearray_len);
        arith.cgbn_from_fixed_memory(
            modulus_size,
            size_bytearray,
            size_bytearray_len);
        cgbn_add(arith._env, index, index, length);
        // set the index in call data from where the data
        // byte array for the three elements base, exponent
        // and modulus starts
        bn_t data_index;
        cgbn_set(arith._env, data_index, index);

        // compute the gas cost
        bn_t max_length;
        cgbn_set(arith._env, max_length, base_size);
        if (cgbn_compare(arith._env, max_length, modulus_size) < 0) {
            cgbn_set(arith._env, max_length, modulus_size);
        }
        // words = (max_length + 7) / 8
        // add 7
        cgbn_add_ui32(arith._env, max_length, max_length, 7);
        // divide by 8
        cgbn_shift_right(arith._env, max_length, max_length, 3);
        // multiplication_complexity = words ^ 2
        bn_t multiplication_complexity;

        cgbn_mul_high(
            arith._env,
            multiplication_complexity,
            max_length,
            max_length
        );
        dynamic_gas_overflow = dynamic_gas_overflow | (
            cgbn_compare_ui32(
                arith._env,
                multiplication_complexity,
                0
            ) != 0
        );
        cgbn_mul(
            arith._env,
            multiplication_complexity,
            max_length,
            max_length
        );
        //cgbn_mul(arith._env, multiplication_complexity, max_length, max_length);

        // find the most signigicant position of non-zero bit
        // in the least significant 256 bits of the expoennt
        size_t call_exponent_size;
        uint8_t *call_exponent_data;
        cgbn_set(arith._env, index, data_index);
        cgbn_add(arith._env, index, index, base_size);
        // get a pointer to available bytes (call_exponent_size) of exponen
        // send through call data the remaining bytes are consider
        // 0 value bytes. The bytes of the call data are the most
        // significant bytes of the exponent
        call_exponent_data = message.get_data(
            index,
            exponent_size,
            call_exponent_size);
        bn_t call_exponent_size_bn;
        arith.cgbn_from_size_t(call_exponent_size_bn, call_exponent_size);
        bn_t exponent_bit_length_bn;
        uint8_t *exponent_MSB_32_bytes = NULL;
        exponent_MSB_32_bytes = arith.padded_malloc_byte_array(call_exponent_data, call_exponent_size, 32);
        bigint tmp_exponent_bigint[1];
        bigint_init(tmp_exponent_bigint);
        bigint_from_bytes(tmp_exponent_bigint, exponent_MSB_32_bytes, 32);
        int bitlength = bigint_bitlength(tmp_exponent_bigint);
        cgbn_set_ui32(arith._env, exponent_bit_length_bn, (uint32_t) bitlength);
        bigint_free(tmp_exponent_bigint);
        ONE_THREAD_PER_INSTANCE(
            delete[] exponent_MSB_32_bytes;
        )

        /* OLD WAY
        // how many bytes are not part of the call data
        bn_t remainig_exponent_size;
        cgbn_sub(
            arith._env,
            remainig_exponent_size,
            exponent_size,
            call_exponent_size_bn);
        // if they are more the 32 bytes than we know for sure
        // that the last 256 bits are all zero
        if (cgbn_compare_ui32(arith._env, remainig_exponent_size, 32) >= 0) {
            // more than 256 bits
            cgbn_set_ui32(arith._env, exponent_bit_length_bn, 0);
        } else {
            // otherwise we take the least significant bytes
            // of the call data exponent that are part of the
            // least significant 256 bits of the exponent
            size_t available_bytes;
            bn_t available_bytes_bn;
            // we are interest only in the least siginificant
            // 32 bytes (256 bits)
            cgbn_set_ui32(arith._env, available_bytes_bn, 32);
            // we substract the 0 value bytes outside of the
            // call data
            cgbn_sub(
                arith._env,
                available_bytes_bn,
                available_bytes_bn,
                remainig_exponent_size);
            arith.size_t_from_cgbn(available_bytes, available_bytes_bn);
            // if there are more bytes available for the least significant
            // 32 bytes than the are sent through call data we only consider
            // the one sent, the other are considered 0 value bytes
            if (available_bytes > call_exponent_size) {
                available_bytes = call_exponent_size;
            }
            // if we have any bytes available in the least siginifcant
            // 32 bytes of the exponent
            size_t exponent_bit_length;
            if (available_bytes > 0) {
                size_t exponent_byte_length;
                // we start from the most significant byte
                // of the least sigiinificant 32 bytes of the
                // the exponent that are part of the call data
                // and we find the most siginificant non-zero byte
                // Note the bytes of the exponent are in MSB order
                // so the byte on the 0 position is the most significant
                // We have too look on the last bytes
                exponent_byte_length = available_bytes - 1;
                while (
                    (exponent_byte_length > 0) &&
                    (
                        call_exponent_data[
                            call_exponent_size  - exponent_byte_length - 1
                        ] == 0
                    )
                ) {
                    exponent_byte_length--;
                }
                uint8_t exponent_byte;
                // take the found byte
                exponent_byte = call_exponent_data[
                    call_exponent_size - exponent_byte_length - 1
                ];
                // if is zero it means that all the bytes were 0 value bytes
                if (exponent_byte == 0) {
                    exponent_bit_length = 0;
                } else {
                    // transform in bits
                    exponent_bit_length = (exponent_byte_length + 1) * 8;
                }
                // go thtough the byte and find the most significant bit
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

            // if there is a non-zero bit in the exponent call data
            // we compute its position relative to the size of the exponent
            // and the zero value bytes not sent through the call data
            if (exponent_bit_length > 0) {
                // transform the remaining size from bytes to bits
                cgbn_mul_ui32(
                    arith._env,
                    remainig_exponent_size,
                    remainig_exponent_size,
                    8);
                // convert to cgbn the current bit length
                arith.cgbn_from_size_t(
                    exponent_bit_length_bn,
                    exponent_bit_length);
                // add the remaing size in bits to the
                // current bit length
                cgbn_add(
                    arith._env,
                    exponent_bit_length_bn,
                    exponent_bit_length_bn,
                    remainig_exponent_size);
            } else {
                cgbn_set_ui32(arith._env, exponent_bit_length_bn, 0);
            }
        }
        */

        // compute the iteration count depending on the size
        // of the exponent and its most significant non-zero
        // bit of the least siginifcant 256 bits
        bn_t iteration_count;
        cgbn_set_ui32(arith._env, iteration_count, 0);
        uint32_t iteration_count_overflow;
        iteration_count_overflow = 0;
        // if the size is less than 32 bytes (256 bits) we
        // just take the position of the most significant non-zero bit
        // and substract 1
        if (cgbn_compare_ui32(arith._env, exponent_size, 32) <= 0) {
            if (cgbn_get_ui32(arith._env, exponent_bit_length_bn) != 0) {
                // bitlength = bitlength - (32 - exponet_size) * 8
                bn_t tmp_value;
                cgbn_set_ui32(arith._env, tmp_value, 32);
                cgbn_sub(arith._env, tmp_value, tmp_value, exponent_size);
                cgbn_mul_ui32(arith._env, tmp_value, tmp_value, 8);
                cgbn_sub(
                    arith._env,
                    exponent_bit_length_bn,
                    exponent_bit_length_bn,
                    tmp_value);
                // exponent.bit_length() - 1
                cgbn_sub_ui32(
                    arith._env,
                    iteration_count,
                    exponent_bit_length_bn,
                    1);
            }
        } else {
            // elif Esize > 32: iteration_count = (8 * (Esize - 32)) + exponent.bit_length() - 1
            cgbn_sub_ui32(
                arith._env,
                iteration_count,
                exponent_size,
                32);
            // sometimes the iteration count can overflow
            // for high values of the exponent size
            iteration_count_overflow = cgbn_mul_ui32(
                arith._env,
                iteration_count,
                iteration_count,
                8);
            if (cgbn_compare_ui32(arith._env, exponent_bit_length_bn, 1) > 0) {
                iteration_count_overflow = iteration_count_overflow | cgbn_add(
                    arith._env,
                    iteration_count,
                    iteration_count,
                    exponent_bit_length_bn);
                cgbn_sub_ui32(
                    arith._env,
                    iteration_count,
                    iteration_count,
                    1);

            }
        }
        // iteration_count = max(iteration_count, 1)
        if (cgbn_compare_ui32(arith._env, iteration_count, 1) < 0) {
            cgbn_set_ui32(arith._env, iteration_count, 1);
        }

        bn_t dynamic_gas;
        // dynamic_gas = max(200, multiplication_complexity * iteration_count / 3)
        // The dynamic gas value can overflow from the overflow
        // of iteration count when the multiplication complexity
        // is non-zero or from the simple multiplication of
        // the iteration count and multiplication complexity
        // in both case the value is way over the gas limit
        // and we just throw an error which will consume the
        // entire gas given for the call
        cgbn_mul_high(
            arith._env,
            dynamic_gas,
            iteration_count,
            multiplication_complexity
        );
        dynamic_gas_overflow = dynamic_gas_overflow | (
            cgbn_compare_ui32(
                arith._env,
                dynamic_gas,
                0
            ) != 0
        );
        cgbn_mul(
            arith._env,
            dynamic_gas,
            iteration_count,
            multiplication_complexity
        );
        dynamic_gas_overflow = dynamic_gas_overflow | (
            iteration_count_overflow && (
                cgbn_compare_ui32(
                    arith._env,
                    multiplication_complexity,
                    0) != 0
            )
        );
        if (dynamic_gas_overflow) {
            error_code = ERROR_PRECOMPILE_MODEXP_OVERFLOW;
            return;
        }
        cgbn_div_ui32(
            arith._env,
            dynamic_gas,
            dynamic_gas,
            3);
        if (cgbn_compare_ui32(arith._env, dynamic_gas, 200) < 0) {
            cgbn_set_ui32(arith._env, dynamic_gas, 200);
        }
        cgbn_add(arith._env, gas_used, gas_used, dynamic_gas);

        // if we have enough gas for the operation we start doing it
        // also by considering some special cases before allocating
        // the memory.
        if (arith.has_gas(gas_limit, gas_used, error_code)) {
            // If modulus is zero than the result is 0
            // if the modules size is zero than the returning size
            // must also be zero
            if (cgbn_compare_ui32(arith._env, modulus_size, 0) == 0) {
                return_data.set(NULL, 0);
                error_code = ERR_RETURN;
                return;
            }

            // There are some overflow casses for the memory allocation
            // This way we are avoiding bad allocations
            size_t modulus_len;
            if (arith.size_t_from_cgbn(modulus_len, modulus_size)) {
                error_code = ERROR_PRECOMPILE_MODEXP_OVERFLOW;
                return;
            }
            // the result must have the same size as the
            // modulus. we alocate the memory and init it
            // with zero value bytes
            SHARED_MEMORY uint8_t *result, *modulus;
            ONE_THREAD_PER_INSTANCE(
                result = new uint8_t[modulus_len];
                memset(result, 0, modulus_len);
                modulus = new uint8_t[modulus_len];
                memset(modulus, 0, modulus_len);
            )
            // set the bytes of the modulus with
            // the bytes from call data
            uint8_t *call_modulus_data;
            size_t call_modulus_size;
            cgbn_set(arith._env, index, data_index);
            cgbn_add(arith._env, index, index, base_size);
            cgbn_add(arith._env, index, index, exponent_size);
            call_modulus_data = message.get_data(
                index,
                modulus_size,
                call_modulus_size);
            for (size_t i = 0; i < call_modulus_size; i++) {
                modulus[i] = call_modulus_data[i];
            }
            // set the bigint modulus value
            bigint modulus_bigint[1];
            bigint_init(modulus_bigint);
            bigint_from_bytes(modulus_bigint, modulus, modulus_len);
            // Special case when the value of the modulus is zero
            // than the result is zero (yellow paper)
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

            // bigint for result and temporary value
            bigint result_bigint[1];
            bigint_init(result_bigint);
            bigint tmp_bigint[1];
            bigint_init(tmp_bigint);
            uint32_t is_zero_exponent;
            is_zero_exponent = 1;
            for (size_t idx = 0; idx < call_exponent_size; idx++) {
                if (call_exponent_data[idx] != 0)
                    is_zero_exponent = 0;
            }
            // if the exponent is zero than the exponention
            // result is 1 even for 0 ^ 0 (yellow paper)
            if (is_zero_exponent) {
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
            // 0 base size and not zero exponent
            // 0^x, where x is different than 0 is 0
            // and the result will be 0
            if (cgbn_compare_ui32(arith._env, base_size, 0) == 0) {
                bigint_from_word(result_bigint, 0);
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

            // The geeneral case we get the values for the
            // vase and exponent but also verify for the
            // allocation possible overflows
            SHARED_MEMORY uint8_t *base, *exponent;
            size_t base_len, exponent_len;
            uint32_t overflow;
            overflow = arith.size_t_from_cgbn(base_len, base_size);
            overflow = (
                overflow |
                arith.size_t_from_cgbn(exponent_len, exponent_size)
            );

            if (overflow) {
                error_code = ERROR_PRECOMPILE_MODEXP_OVERFLOW;
                ONE_THREAD_PER_INSTANCE(
                    delete[] result;
                    delete[] modulus;
                )
                bigint_free(modulus_bigint);
                bigint_free(result_bigint);
                bigint_free(tmp_bigint);
                return;
            }
            // no need for the temporary bigint value anymore
            bigint_free(tmp_bigint);

            // alocate the memory
            ONE_THREAD_PER_INSTANCE(
                base = new uint8_t[base_len];
                memset(base, 0, base_len);
                exponent = new uint8_t[exponent_len];
                memset(exponent, 0, exponent_len);
            )

            uint8_t *call_base_data;
            size_t call_base_size;
            cgbn_set(arith._env, index, data_index);
            call_base_data = message.get_data(
                index,
                base_size,
                call_base_size);
            for (size_t i = 0; i < call_base_size; i++) {
                base[i] = call_base_data[i];
            }
            for (size_t i = 0; i < call_exponent_size; i++) {
                exponent[i] = call_exponent_data[i];
            }

            // convert to bigint values
            bigint base_bigint[1], exponent_bigint[1];
            bigint_init(base_bigint);
            bigint_init(exponent_bigint);
            bigint_from_bytes(base_bigint, base, base_len);
            bigint_from_bytes(exponent_bigint, exponent, exponent_len);
            // make the pow mod operation
            bigint_pow_mod(
                result_bigint,
                base_bigint,
                exponent_bigint,
                modulus_bigint);
            bigint_to_bytes(result, result_bigint, modulus_len);
            return_data.set(result, modulus_len);
            error_code = ERR_RETURN;
            ONE_THREAD_PER_INSTANCE(
                delete[] result;
                delete[] base;
                delete[] exponent;
                delete[] modulus;
            )
            bigint_free(modulus_bigint);
            bigint_free(result_bigint);
            bigint_free(base_bigint);
            bigint_free(exponent_bigint);
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


   __host__ __device__  static void operation_ecRecover(
        arith_t &arith,
        keccak_t &keccak,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        return_data_t &return_data,
        message_t &message
    ) {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_PRECOMPILE_ECRECOVER);
        if ( arith.has_gas(gas_limit, gas_used, error_code) ) {
            size_t size;
            SHARED_MEMORY uint8_t *input, *tmp_input;
            size = message.get_data_size();
            bn_t length;
            arith.cgbn_from_size_t(length, size);
            bn_t index;
            cgbn_set_ui32(arith._env, index, 0);
            tmp_input = message.get_data(index, length, size);
            // complete with zeroes the remaing bytes
            input = arith.padded_malloc_byte_array(tmp_input, size, 128);

            ecc::signature_t signature;
            bn_t msg_hash, v, r, s, signer;

            arith.cgbn_from_memory(msg_hash, input);
            arith.cgbn_from_memory(v, input + 32);
            arith.cgbn_from_memory(r, input + 64);
            arith.cgbn_from_memory(s, input + 96);
            signature.v = cgbn_get_ui32(arith._env, v);
            cgbn_store(arith._env, &signature.msg_hash, msg_hash);
            cgbn_store(arith._env, &signature.r, r);
            cgbn_store(arith._env, &signature.s, s);
            //printf("\n v %d\n", signature.v);
            //printf("r : %s\n", ecc::bnt_to_string(arith._env, r));
            //printf("s : %s\n", ecc::bnt_to_string(arith._env, s));
            //printf("msgh: %s\n", ecc::bnt_to_string(arith._env, msg_hash));

            // TODO: is not 27 and 28, only?
            if (cgbn_compare_ui32(arith._env, v, 28) <= 0) {
                SHARED_MEMORY uint8_t output[32];
                size_t res = ecc::ec_recover(arith, keccak, signature, signer);
                if (res==0){
                    arith.memory_from_cgbn(
                        output,
                        signer
                    );
                    return_data.set(output, 32);
                    error_code = ERR_RETURN;
                } else {
                    // TODO: do we consume all gas?
                    // it happens by default because of the error code
                    // error_code = ERROR_PRECOMPILE_UNEXPECTED_INPUT;
                    return_data.set(NULL, 0);
                    error_code = ERR_RETURN;
                }
            } else {
                return_data.set(NULL, 0);
                error_code = ERR_RETURN;
            }
            ONE_THREAD_PER_INSTANCE(
                delete[] input;
            )
        }
    }

    __host__ __device__  static void operation_ecAdd(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        return_data_t &return_data,
        message_t &message
    ) {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_PRECOMPILE_ECADD);
        if (arith.has_gas(gas_limit, gas_used, error_code)) {
            size_t size;
            SHARED_MEMORY uint8_t *input, *tmp_input;
            size = message.get_data_size();
            if (size > 128) size = 128;
            bn_t length;
            arith.cgbn_from_size_t(length, size);
            bn_t index;
            cgbn_set_ui32(arith._env, index, 0);
            tmp_input = message.get_data(index, length, size);
            // complete with zeroes the remaing bytes
            input = arith.padded_malloc_byte_array(tmp_input, size, 128);


            ecc::Curve curve = ecc::get_curve(arith,128);

            bn_t x1, y1, x2, y2;
            arith.cgbn_from_memory(x1, input);
            arith.cgbn_from_memory(y1, input + 32);
            arith.cgbn_from_memory(x2, input + 64);
            arith.cgbn_from_memory(y2, input + 96);
            // print
            //printf("x1: %s\n", ecc::bnt_to_string(arith._env, x1));
            //printf("y1: %s\n", ecc::bnt_to_string(arith._env, y1));
            //printf("x2: %s\n", ecc::bnt_to_string(arith._env, x2));
            //printf("y2: %s\n", ecc::bnt_to_string(arith._env, y2));
            SHARED_MEMORY uint8_t output[64];
            int res = ecc::ec_add(arith, curve, x1, y1, x1, y1, x2, y2);
            if (res==0){
                arith.memory_from_cgbn(
                    output,
                    x1
                );
                arith.memory_from_cgbn(
                    output + 32,
                    y1
                );
                return_data.set(output, 64);
                error_code = ERR_RETURN;
            } else{
                // consume all gas because it is an error
                error_code = ERROR_PRECOMPILE_UNEXPECTED_INPUT;
            }
            ONE_THREAD_PER_INSTANCE(
                delete[] input;
            )
            //print res
            //printf("xres: %s\n", ecc::bnt_to_string(arith._env, x1));
            //printf("yres: %s\n", ecc::bnt_to_string(arith._env, y1));
        }
  }

    __host__ __device__ static void operation_ecMul(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        return_data_t &return_data,
        message_t &message
    ) {

        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_PRECOMPILE_ECMUL);
        if (arith.has_gas(gas_limit, gas_used, error_code)) {
            size_t size;
            SHARED_MEMORY uint8_t *input, *tmp_input;
            size = message.get_data_size();
            if (size > 96) size = 96;
            bn_t length;
            arith.cgbn_from_size_t(length, size);
            bn_t index;
            cgbn_set_ui32(arith._env, index, 0);
            tmp_input = message.get_data(index, length, size);
            // complete with zeroes the remaing bytes
            input = arith.padded_malloc_byte_array(tmp_input, size, 96);
            ecc::Curve curve = ecc::get_curve(arith,128);

            bn_t x, y, k;
            arith.cgbn_from_memory(x, input);
            arith.cgbn_from_memory(y, input + 32);
            arith.cgbn_from_memory(k, input + 64);
            // print
            //printf("mul x: %s\n", ecc::bnt_to_string(arith._env, x));
            //printf("mul y: %s\n", ecc::bnt_to_string(arith._env, y));
            //printf("k: %s\n", ecc::bnt_to_string(arith._env, k));

            SHARED_MEMORY uint8_t output[64];
            int res = ecc::ec_mul(arith, curve, x, y, x, y, k);
            // print result
            //printf("xres: %s\n", ecc::bnt_to_string(arith._env, x));
            //printf("yres: %s\n", ecc::bnt_to_string(arith._env, y));
            if (res==0) {
                arith.memory_from_cgbn(
                    output,
                    x
                );
                arith.memory_from_cgbn(
                    output + 32,
                    y
                );
                return_data.set(output, 64);
                // print_data_content_t(*return_data._content);
                error_code = ERR_RETURN;
            } else {
                // consume all the gas because it is an error
                error_code = ERROR_PRECOMPILE_UNEXPECTED_INPUT;
            }
            ONE_THREAD_PER_INSTANCE(
                delete[] input;
            )
        }
    }

    __host__ __device__  static void operation_ecPairing(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        return_data_t &return_data,
        message_t &message
    ) {

        size_t size;
        SHARED_MEMORY uint8_t *input;
        size = message.get_data_size();
        bn_t length;
        arith.cgbn_from_size_t(length, size);
        bn_t index;
        cgbn_set_ui32(arith._env, index, 0);
        input = message.get_data(index, length, size);
        arith.ecpairing_cost(gas_used, size);
        arith.print_byte_array_as_hex(input,size);
        if (arith.has_gas(gas_limit, gas_used, error_code)) {
            if (size % 192 != 0) {
                error_code = ERROR_PRECOMPILE_UNEXPECTED_INPUT;
            } else {
                int res = ecc::pairing_multiple(arith, input, size);

                if (res== -1) {
                    error_code = ERROR_PRECOMPILE_UNEXPECTED_INPUT;
                } else {
                    SHARED_MEMORY uint8_t output[32];
                    ONE_THREAD_PER_INSTANCE(
                        memset(output, 0, 32);
                    )
                    output[31] = (res == 1);
                    return_data.set(output, 32);
                    error_code = ERR_RETURN;

                }
            }

        } else {
            error_code = ERR_OUT_OF_GAS;
        }
    }

}

#endif
