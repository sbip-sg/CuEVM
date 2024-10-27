// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation
// Programme) Author: Stefan-Dan Ciocirlan Data: 2024-03-13
// SPDX-License-Identifier: MIT
#include <CuEVM/gas_cost.cuh>
#include <CuEVM/precompile.cuh>
#include <CuEVM/utils/arith.cuh>
#include <CuEVM/utils/error_codes.cuh>
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
 * @param[in] arith The arithmetic environment
 * @param[in] gas_limit The gas limit
 * @param[out] gas_used The gas used
 * @param[out] error_code The error code
 * @param[out] return_data The return data
 * @param[in] message The message
 */
__host__ __device__ int32_t operation_IDENTITY(ArithEnv &arith, bn_t &gas_limit, bn_t &gas_used,
                                               CuEVM::evm_return_data_t *return_data,
                                               CuEVM::evm_message_call_t *message) {
    // static gas
    cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_PRECOMPILE_IDENTITY);

    // dynamic gas
    // compute the dynamic gas cost
    bn_t length;
    uint32_t length_size = message->data.size;
    cgbn_set_ui32(arith.env, length, length_size);
    CuEVM::gas_cost::memory_cost(arith, gas_used, length);

    int32_t error = CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);

    if (error) {
        return error;
    }

    *return_data = byte_array_t(message->data.data, length_size);

    return ERROR_RETURN;
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
 */
__host__ __device__ int32_t operation_SHA256(ArithEnv &arith, bn_t &gas_limit, bn_t &gas_used,
                                             CuEVM::evm_return_data_t *return_data,
                                             CuEVM::evm_message_call_t *message) {
    // static gas
    cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_PRECOMPILE_SHA256);

    // dynamic gas
    // compute the dynamic gas cost
    bn_t length;
    uint32_t length_size = message->data.size;
    cgbn_set_ui32(arith.env, length, length_size);
    CuEVM::gas_cost::sha256_cost(arith, gas_used, length);

    int32_t error = CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);

    if (error) {
        return error;
    }

    uint8_t hash[32] = {0};
    CuCrypto::sha256::sha(message->data.data, length_size, &(hash[0]));
    *return_data = byte_array_t(hash, 32);
    return ERROR_RETURN;
}

__host__ __device__ int32_t operation_RIPEMD160(ArithEnv &arith, bn_t &gas_limit, bn_t &gas_used,
                                                CuEVM::evm_return_data_t *return_data,
                                                CuEVM::evm_message_call_t *message) {
    // static gas
    cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_PRECOMPILE_RIPEMD160);

    uint32_t data_size = message->data.size;

    bn_t length;
    cgbn_set_ui32(arith.env, length, data_size);

    CuEVM::gas_cost::ripemd160_cost(arith, gas_used, length);

    int32_t error = CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);

    if (error) {
        return error;
    }

    // output allocation
    uint8_t output[32] = {0};
    uint8_t *hash;
    hash = output + 12;
    CuCrypto::ripemd160::ripemd160(message->data.data, data_size, hash);
    *return_data = byte_array_t(output, 32);

    return ERROR_RETURN;
}

__host__ __device__ int32_t operation_MODEXP(ArithEnv &arith, bn_t &gas_limit, bn_t &gas_used,
                                             CuEVM::evm_return_data_t *return_data,
                                             CuEVM::evm_message_call_t *message) {
    bn_t base_size, exponent_size, modulus_size;

    CuEVM::byte_array_t input_data(message->get_data(), 0, 96);
    byte_array_t bsize_array = byte_array_t(input_data.data, 32);
    byte_array_t esize_array = byte_array_t(input_data.data + 32, 32);
    byte_array_t msize_array = byte_array_t(input_data.data + 64, 32);

    int32_t error = cgbn_set_byte_array_t(arith.env, base_size, bsize_array);
    error |= cgbn_set_byte_array_t(arith.env, exponent_size, esize_array);
    error |= cgbn_set_byte_array_t(arith.env, modulus_size, msize_array);

#ifdef EIP_3155
    printf("base size\n");
    print_bnt(arith, base_size);
    printf("exponent size\n");
    print_bnt(arith, exponent_size);
    printf("modulus size\n");
    print_bnt(arith, modulus_size);
#endif

    if (error) {
        return error;
    }

    uint32_t base_len, exp_len, mod_len, data_len;
    data_len = message->data.size;
    error = cgbn_get_uint32_t(arith.env, base_len, base_size);
    error |= cgbn_get_uint32_t(arith.env, mod_len, modulus_size);
    error |= cgbn_get_uint32_t(arith.env, exp_len, exponent_size);

    // Handle a special case when both the base and mod length are zero.
    if (cgbn_compare_ui32(arith.env, base_size, 0) == 0 && cgbn_compare_ui32(arith.env, modulus_size, 0) == 0) {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_PRECOMPILE_MODEXP_MAX);
        error = CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);
        return error;
    }
    if (error) {
        // printf("cgbn_get_uint32_t error %d\n", error);
        return error;
    }

    bn_t max_length;
    cgbn_set(arith.env, max_length, base_size);
    if (cgbn_compare(arith.env, max_length, modulus_size) < 0) {
        cgbn_set(arith.env, max_length, modulus_size);
    }
    // words = (max_length + 7) / 8
    // add 7
    cgbn_add_ui32(arith.env, max_length, max_length, 7);
    // divide by 8
    cgbn_shift_right(arith.env, max_length, max_length, 3);
    // multiplication_complexity = words ^ 2
    bn_t multiplication_complexity;
    cgbn_mul(arith.env, multiplication_complexity, max_length, max_length);

    bn_t exponent_bit_length_bn;

    bool exp_is_zero = true;
#ifdef __CUDA_ARCH__
    printf("data len %d\n", data_len);
    message->data.print();
#endif
    // get a pointer to available bytes (call_exponent_size) of exponen
    // send through call data the remaining bytes are consider
    // 0 value bytes. The bytes of the call data are the most
    // significant bytes of the exponent
    // uint8_t *e_data = new uint8_t[exp_len];
    byte_array_t e_data = byte_array_t(exp_len);

    for (uint32_t i = 0; i < exp_len; i++) {
        auto idx = 96 + base_len + i;
        __ONE_GPU_THREAD_WOSYNC_BEGIN__

        if (idx < data_len) {
            e_data.data[i] = message->data.data[idx];
        } else {
            e_data.data[i] = 0;
        }
        printf("idx %d e_data %d\n", idx, e_data.data[i]);
        __ONE_GPU_THREAD_END__
        if (e_data.data[i] != 0) {
            exp_is_zero = false;
        }
    }

    uint8_t adjusted_exp_data[32] = {0};
    uint32_t iteration_length = min(32, e_data.size);
    for (uint32_t i = 0; i < iteration_length; i++) {
        adjusted_exp_data[32 - iteration_length + i] = e_data.data[i];
    }

    int bit_size = 0;
    int found_non_zero = 0;

    for (int i = 0; i < 32; i++) {
        if (adjusted_exp_data[i] != 0) {
            found_non_zero = 1;
            // Count significant bits in the most significant byte
            for (int j = 7; j >= 0; j--) {
                if (adjusted_exp_data[i] & (1 << j)) {
                    bit_size = (32 - i) * 8 - (7 - j);
                    break;
                }
            }
            break;
        }
    }

    if (!found_non_zero) {
        bit_size = 0;  // If all bytes are zero
    }
    cgbn_set_ui32(arith.env, exponent_bit_length_bn, bit_size);

    error |=
        CuEVM::gas_cost::modexp_cost(arith, gas_used, exponent_size, exponent_bit_length_bn, multiplication_complexity);

    error |= CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);
    // #ifdef __CUDA_ARCH__
    //     printf("modexp cost %d\n", error);
    //     print_bnt(arith, gas_used);
    //     printf("exponent bit length\n");
    //     printf("Thread idx %d error %d\n", threadIdx.x, error);
    // #endif
    if (error) {
        return error;
    }

    bn_t base;
    bool base_is_zero = true;
    // uint8_t base_data[32] = {0};
    byte_array_t base_data = byte_array_t(base_len);
    for (uint32_t i = 0; i < base_len; i++) {
        auto idx = 96 + i;
        if (idx < data_len) {
            __ONE_GPU_THREAD_WOSYNC_BEGIN__
            base_data.data[i] = message->data.data[idx];
            __ONE_GPU_THREAD_END__
        } else {
            break;
        }

        if (base_data.data[i] != 0) {
            base_is_zero = false;
        }
    }

    if (error) {
        return error;
    }

    // uint8_t *mod_data = new uint8_t[mod_len];
    byte_array_t mod_data = byte_array_t(mod_len);
    // loop and check for zero, if zero, then return 0
    bool mod_is_one = true;
    bool mod_is_zero = true;
    for (uint32_t i = 0; i < mod_len; i++) {
        auto idx = 96 + base_len + exp_len + i;
        __ONE_GPU_THREAD_WOSYNC_BEGIN__
        mod_data.data[i] = (idx < data_len) ? message->data.data[idx] : 0;
        __ONE_GPU_THREAD_END__
        if (mod_data.data[i] != 0) {
            mod_is_zero = false;
            if (mod_data.data[i] != 1 || i != mod_len - 1) {
                mod_is_one = false;
            }
        } else if (i == mod_len - 1) {
            mod_is_one = false;
        }
    }
    // #ifdef __CUDA_ARCH__
    //     __ONE_GPU_THREAD_WOSYNC_BEGIN__
    //     printf("mod data\n");
    //     for (int i = 0; i < mod_len; i++) {
    //         printf("%d ", mod_data.data[i]);
    //     }
    //     printf("\n");
    //     printf("base data\n");
    //     for (int i = 0; i < base_len; i++) {
    //         printf("%d ", base_data.data[i]);
    //     }
    //     printf("\n");
    //     printf("exp data\n");
    //     for (int i = 0; i < exp_len; i++) {
    //         printf("%d ", e_data.data[i]);
    //     }
    //     printf("\n");
    //     __ONE_GPU_THREAD_WOSYNC_END__
    //     printf("thread idx %d mod_is_zero %d, base_is_zero %d, exp_is_zero %d, mod_is_one %d\n", threadIdx.x,
    //     mod_is_zero,
    //            base_is_zero, exp_is_zero, mod_is_one);
    // #endif

    // early return special cases
    if (mod_is_zero) {
        *return_data = byte_array_t(mod_len);  // return 0
        return ERROR_RETURN;
    }

    if (exp_is_zero) {
        *return_data = byte_array_t(mod_len);

        if (!mod_is_one) return_data->data[mod_len - 1] = 1;  // return 1
        return ERROR_RETURN;
    }

    // convert to bigint values
    bigint base_bigint = {}, exponent_bigint = {}, result_bigint = {}, modulus_bigint = {};
    uint8_t result[32] = {0};

    bigint_from_bytes(&base_bigint, base_data.data, base_len);
    bigint_from_bytes(&exponent_bigint, e_data.data, exp_len);
    bigint_from_bytes(&modulus_bigint, mod_data.data, mod_len);

    // make the pow mod operation
    bigint_pow_mod(&result_bigint, &base_bigint, &exponent_bigint, &modulus_bigint);
    bigint_to_bytes(result, &result_bigint, mod_len);
    *return_data = byte_array_t(result, mod_len);

    return ERROR_RETURN;
}

__host__ __device__ int32_t operation_BLAKE2(ArithEnv &arith, bn_t &gas_limit, bn_t &gas_used,
                                             CuEVM::evm_return_data_t *return_data,
                                             CuEVM::evm_message_call_t *message) {
    // expecting 213 bytes inputs
    uint32_t length_size = message->data.size;

    if (length_size != 213) {
        return ERROR_PRECOMPILE_UNEXPECTED_INPUT_LENGTH;
    }

    uint8_t *input;
    input = message->data.data;
    uint8_t f = input[212];

    // final byte must be 1 or 0
    if ((f >> 1) != 0) {
        return ERROR_PRECOMPILE_UNEXPECTED_INPUT;
    }

    uint32_t rounds;
    rounds =
        (((uint32_t)input[0] << 24) | ((uint32_t)input[1] << 16) | ((uint32_t)input[2] << 8) | ((uint32_t)input[3]));

    CuEVM::gas_cost::blake2_cost(arith, gas_used, rounds);

    int32_t error = CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);

    if (error) {
        return error;
    }

    uint64_t h[8];
    uint64_t m[16];
    uint64_t t[2];

    memcpy(h, &(input[4]), 64);
    memcpy(m, &(input[68]), 128);
    memcpy(t, &(input[196]), 16);

    CuCrypto::blake2::blake2f(rounds, h, m, t, f);

    *return_data = byte_array_t((uint8_t *)h, 64);

    return ERROR_RETURN;
}

__host__ __device__ int32_t operation_ecRecover(ArithEnv &arith, CuEVM::EccConstants *constants, bn_t &gas_limit,
                                                bn_t &gas_used, CuEVM::evm_return_data_t *return_data,
                                                CuEVM::evm_message_call_t *message) {
    cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_PRECOMPILE_ECRECOVER);
    int32_t error_code = ERROR_SUCCESS;
    error_code |= CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);
    // printf("has gas %d\n", error_code);
    // printf("gas limit \n");
    // print_bnt(arith, gas_limit);
    // printf("data size %d\n", message->data.size);
    message->data.print();
    if (error_code == ERROR_SUCCESS) {
        bn_t length;
        cgbn_set_ui32(arith.env, length, message->data.size);
        bn_t index;
        cgbn_set_ui32(arith.env, index, 0);
        // complete with zeroes the remaing bytes
        // input = arith.padded_malloc_byte_array(tmp_input, size, 128);
        CuEVM::byte_array_t input(message->get_data(), 0, 128);
        __SHARED_MEMORY__ ecc::signature_t signature;
        bn_t msg_hash, v, r, s, signer;
        cgbn_set_memory(arith.env, msg_hash, input.data, 32);
        cgbn_set_memory(arith.env, v, input.data + 32, 32);
        cgbn_set_memory(arith.env, r, input.data + 64, 32);
        cgbn_set_memory(arith.env, s, input.data + 96, 32);

        cgbn_store(arith.env, &signature.msg_hash, msg_hash);
        cgbn_store(arith.env, &signature.r, r);
        cgbn_store(arith.env, &signature.s, s);
        signature.v = cgbn_get_ui32(arith.env, v);
#ifdef EIP_3155
        printf("\n v %d\n", signature.v);
        printf("r : \n");
        print_bnt(arith, r);
        printf("s : \n");
        print_bnt(arith, s);
        printf("msgh: \n");
        print_bnt(arith, msg_hash);
#endif
        // TODO: is not 27 and 28, only?
        if (cgbn_compare_ui32(arith.env, v, 28) <= 0) {
            __SHARED_MEMORY__ uint8_t output[32];
            size_t res = ecc::ec_recover(arith, constants, signature, signer);
#ifdef EIP_3155
            printf("ec recover %d\n", res);
            print_bnt(arith, signer);
#endif
            if (res == ERROR_SUCCESS) {
                memory_from_cgbn(arith, output, signer);
                *return_data = byte_array_t(output, 32);
                error_code = ERROR_RETURN;
            } else {
                // TODO: do we consume all gas?
                // it happens by default because of the error code
                error_code = ERROR_PRECOMPILE_UNEXPECTED_INPUT;
            }
        }
        return ERROR_RETURN;
    }
    return error_code;
}

__host__ __device__ int32_t operation_ecAdd(ArithEnv &arith, CuEVM::EccConstants *constants, bn_t &gas_limit,
                                            bn_t &gas_used, CuEVM::evm_return_data_t *return_data,
                                            CuEVM::evm_message_call_t *message) {
    printf("ecAdd\n");
    int32_t error_code = ERROR_SUCCESS;
    cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_PRECOMPILE_ECADD);
    error_code |= CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        CuEVM::byte_array_t input(message->get_data(), 0, 128);

        // ecc::Curve curve = ecc::get_curve(arith, 128);

        bn_t x1, y1, x2, y2;
        cgbn_set_memory(arith.env, x1, input.data);
        cgbn_set_memory(arith.env, y1, input.data + 32);
        cgbn_set_memory(arith.env, x2, input.data + 64);
        cgbn_set_memory(arith.env, y2, input.data + 96);
        // print
        // printf("x1: %s\n", ecc::bnt_to_string(arith._env, x1));
        // printf("y1: %s\n", ecc::bnt_to_string(arith._env, y1));
        // printf("x2: %s\n", ecc::bnt_to_string(arith._env, x2));
        // printf("y2: %s\n", ecc::bnt_to_string(arith._env, y2));
        __SHARED_MEMORY__ uint8_t output[64];
        int res = ecc::ec_add(arith, constants->alt_BN128, x1, y1, x1, y1, x2, y2);
        if (res == 0) {
            memory_from_cgbn(arith, output, x1);
            memory_from_cgbn(arith, output + 32, y1);
            // return_data.set(output, 64);
            *return_data = byte_array_t(output, 64);
            error_code = ERROR_RETURN;
        } else {
            // consume all gas because it is an error
            error_code = ERROR_PRECOMPILE_UNEXPECTED_INPUT;
        }
    }
    return error_code;
}

__host__ __device__ int32_t operation_ecMul(ArithEnv &arith, CuEVM::EccConstants *constants, bn_t &gas_limit,
                                            bn_t &gas_used, CuEVM::evm_return_data_t *return_data,
                                            CuEVM::evm_message_call_t *message) {
    cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_PRECOMPILE_ECMUL);
    int32_t error_code = ERROR_SUCCESS;
    error_code |= CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        CuEVM::byte_array_t input(message->get_data(), 0, 128);

        bn_t x, y, k;
        cgbn_set_memory(arith.env, x, input.data);
        cgbn_set_memory(arith.env, y, input.data + 32);
        cgbn_set_memory(arith.env, k, input.data + 64);
        // print
        // printf("mul x: %s\n", ecc::bnt_to_string(arith._env, x));
        // printf("mul y: %s\n", ecc::bnt_to_string(arith._env, y));
        // printf("k: %s\n", ecc::bnt_to_string(arith._env, k));

        __SHARED_MEMORY__ uint8_t output[64];
        int res = ecc::ec_mul(arith, constants->alt_BN128, x, y, x, y, k);
        // print result
        // printf("xres: %s\n", ecc::bnt_to_string(arith._env, x));
        // printf("yres: %s\n", ecc::bnt_to_string(arith._env, y));
        if (res == 0) {
            memory_from_cgbn(arith, output, x);
            memory_from_cgbn(arith, output + 32, y);
            // return_data.set(output, 64);
            *return_data = byte_array_t(output, 64);
            error_code = ERROR_RETURN;
        } else {
            // consume all gas because it is an error
            error_code = ERROR_PRECOMPILE_UNEXPECTED_INPUT;
        }
    }
    return error_code;
}

__host__ __device__ int32_t operation_ecPairing(ArithEnv &arith, CuEVM::EccConstants *constants, bn_t &gas_limit,
                                                bn_t &gas_used, CuEVM::evm_return_data_t *return_data,
                                                CuEVM::evm_message_call_t *message) {
    __ONE_THREAD_PER_INSTANCE(printf("ecPairing\n"); printf("input size %d\n", message->data.size););
    // input = message.get_data(index, length, size);
    CuEVM::byte_array_t input(message->get_data(), 0, message->data.size);
    CuEVM::gas_cost::ecpairing_cost(arith, gas_used, message->data.size);
    int32_t error_code = ERROR_SUCCESS;
    error_code |= CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        if (message->data.size % 192 != 0) {
            error_code = ERROR_PRECOMPILE_UNEXPECTED_INPUT;
        } else {
            // 0 inputs is valid and returns 1.
            int res =
                message->data.size == 0 ? 1 : ecc::pairing_multiple(arith, constants, input.data, message->data.size);
#ifdef __CUDA_ARCH__
            printf("res: %d, idx %d \n", res, threadIdx.x);
#endif
            if (res == -1) {
                error_code = ERROR_PRECOMPILE_UNEXPECTED_INPUT;
            } else {
                __SHARED_MEMORY__ uint8_t output[32];
                __ONE_GPU_THREAD_BEGIN__
                memset(output, 0, 32);
                output[31] = (res == 1);
                __ONE_GPU_THREAD_END__
                *return_data = byte_array_t(output, 32);
                error_code = ERROR_RETURN;
            }
        }
    }
    return error_code;
}

}  // namespace precompile_operations

}  // namespace CuEVM
