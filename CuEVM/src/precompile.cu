#include <CuEVM/gas_cost.cuh>
#include <CuEVM/precompile.cuh>
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
 * @param[in] gas_limit The gas limit
 * @param[out] gas_used The gas used
 * @param[out] error_code The error code
 * @param[out] return_data The return data
 * @param[in] message The message
 */
__device__ int32_t operation_IDENTITY(gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_call_context_t *call_context) {
    // static gas
    gas_used += GAS_PRECOMPILE_IDENTITY;

    // dynamic gas
    // compute the dynamic gas cost

    CuEVM::gas_cost::memory_cost(gas_used, call_context->call_data_size);

    int32_t error = CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    if (error) {
        return error;
    }

    call_context->set_parent_return_data(call_context->call_data, call_context->call_data_size);

    return ERROR_RETURN;
}

/**
 * The SHA2-256 precompile contract
 * SHA2 through the message data and return data
 * @param[in] gas_limit The gas limit
 * @param[out] gas_used The gas used
 * @param[out] error_code The error code
 * @param[out] return_data The return data
 * @param[in] message The message
 */
__device__ int32_t operation_SHA256(gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_call_context_t *call_context) {
    // static gas
    gas_used += GAS_PRECOMPILE_SHA256;

    // dynamic gas
    // compute the dynamic gas cost
    CuEVM::gas_cost::sha256_cost(gas_used, call_context->call_data_size);

    int32_t error = CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    if (error) {
        return error;
    }

    uint8_t hash[32] = {0};
    CuCrypto::sha256::sha(call_context->call_data, call_context->call_data_size, &(hash[0]));
    call_context->set_parent_return_data(hash, 32);

    return ERROR_RETURN;
}

__device__ int32_t operation_RIPEMD160(gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_call_context_t *call_context) {
    // static gas
    gas_used += GAS_PRECOMPILE_RIPEMD160;

    CuEVM::gas_cost::ripemd160_cost(gas_used, call_context->call_data_size);

    int32_t error = CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    if (error) {
        return error;
    }

    // output allocation
    uint8_t output[32] = {0};
    uint8_t *hash;
    hash = output + 12;
    CuCrypto::ripemd160::ripemd160(call_context->call_data, call_context->call_data_size, hash);
    call_context->set_parent_return_data(hash, 32);

    return ERROR_RETURN;
}

__device__ int32_t operation_MODEXP(gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_call_context_t *call_context) {
    evm_word_t base_size, exponent_size, modulus_size;
    // TODO: fix this
    // CuEVM::byte_array_t input_data(message->get_data(), 0, 96);
    byte_array_t bsize_array = byte_array_t(call_context->call_data, 32);
    byte_array_t esize_array = byte_array_t(call_context->call_data + 32, 32);
    byte_array_t msize_array = byte_array_t(call_context->call_data + 64, 32);

    uint256_from_bytes(&base_size, bsize_array.data, bsize_array.size);
    uint256_from_bytes(&exponent_size, esize_array.data, esize_array.size);
    uint256_from_bytes(&modulus_size, msize_array.data, msize_array.size);

    // if (error) {
    //     return error;
    // }
    int32_t error = ERROR_SUCCESS;

    uint32_t base_len, exp_len, mod_len, data_len;
    data_len = call_context->call_data_size;

    // Handle a special case when both the base and mod length are zero.
    if (uint256_is_zero(&base_size) && uint256_is_zero(&modulus_size)) {
        gas_used += GAS_PRECOMPILE_MODEXP_MAX;
        error = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
        return error;
    }

    evm_word_t max_length = base_size;
    if (uint256_cmp(&max_length, &modulus_size) < 0) {
        max_length = modulus_size;
    }
    // words = (max_length + 7) / 8
    // add 7
    uint256_add_word(&max_length, &max_length, 7);
    // divide by 8
    uint256_shift_right(&max_length, &max_length, 3);
    // multiplication_complexity = words ^ 2
    evm_word_t multiplication_complexity;
    uint256_mul(&multiplication_complexity, &max_length, &max_length);

    evm_word_t exponent_bit_length_bn;

    bool exp_is_zero = true;

    printf("data len %d\n", data_len);

    // get a pointer to available bytes (call_exponent_size) of exponen
    // send through call data the remaining bytes are consider
    // 0 value bytes. The bytes of the call data are the most
    // significant bytes of the exponent
    // uint8_t *e_data = new uint8_t[exp_len];
    byte_array_t e_data = byte_array_t(exp_len);

    for (uint32_t i = 0; i < exp_len; i++) {
        auto idx = 96 + base_len + i;
        if (idx < data_len) {
            e_data.data[i] = call_context->call_data[idx];
        } else {
            e_data.data[i] = 0;
        }

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
    exponent_bit_length_bn = bit_size;

    error |= CuEVM::gas_cost::modexp_cost(gas_used, exponent_size, exponent_bit_length_bn, multiplication_complexity);

    error |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    if (error) {
        return error;
    }

    bool base_is_zero = true;
    // uint8_t base_data[32] = {0};
    byte_array_t base_data = byte_array_t(base_len);
    for (uint32_t i = 0; i < base_len; i++) {
        auto idx = 96 + i;
        if (idx < data_len) {
            base_data.data[i] = call_context->call_data[idx];
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
        mod_data.data[i] = (idx < data_len) ? call_context->call_data[idx] : 0;

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

    //     printf("thread idx %d mod_is_zero %d, base_is_zero %d, exp_is_zero %d, mod_is_one %d\n", threadIdx.x,
    //     mod_is_zero,
    //            base_is_zero, exp_is_zero, mod_is_one);
    // #endif
    // uint8_t result[32] = {0};

    // early return special cases
    if (mod_is_zero) {
        if (mod_len != 0) {
            call_context->set_parent_return_data(mod_data.data, mod_len);
        } else if (call_context->parent != nullptr) {
            call_context->parent->dynamic_ret_size = 0;
        }
        return ERROR_RETURN;
    }

    if (exp_is_zero) {
        for (uint32_t i = 0; i < mod_len; i++) {
            mod_data.data[i] = 0;
        }
        if (!mod_is_one) {
            mod_data.data[mod_len - 1] = 1;  // return 1
        }
        call_context->set_parent_return_data(mod_data.data, mod_len);

        return ERROR_RETURN;
    }

    // convert to bigint values
    bigint base_bigint = {}, exponent_bigint = {}, result_bigint = {}, modulus_bigint = {};

    bigint_from_bytes(&base_bigint, base_data.data, base_len);
    bigint_from_bytes(&exponent_bigint, e_data.data, exp_len);
    bigint_from_bytes(&modulus_bigint, mod_data.data, mod_len);

    // make the pow mod operation
    bigint_pow_mod(&result_bigint, &base_bigint, &exponent_bigint, &modulus_bigint);
    bigint_to_bytes(mod_data.data, &result_bigint, mod_len);
    call_context->set_parent_return_data(mod_data.data, mod_len);
    return ERROR_RETURN;
}

__device__ int32_t operation_BLAKE2(gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_call_context_t *call_context) {
    // expecting 213 bytes inputs
    uint32_t length_size = call_context->call_data_size;

    if (length_size != 213) {
        return ERROR_PRECOMPILE_UNEXPECTED_INPUT_LENGTH;
    }

    uint8_t *input = call_context->call_data;
    uint8_t f = input[212];

    // final byte must be 1 or 0
    if ((f >> 1) != 0) {
        return ERROR_PRECOMPILE_UNEXPECTED_INPUT;
    }

    uint32_t rounds;
    rounds =
        (((uint32_t)input[0] << 24) | ((uint32_t)input[1] << 16) | ((uint32_t)input[2] << 8) | ((uint32_t)input[3]));

    CuEVM::gas_cost::blake2_cost(gas_used, rounds);

    int32_t error = CuEVM::gas_cost::has_gas(gas_limit, gas_used);

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

    // *return_data = byte_array_t((uint8_t *)h, 64);
    call_context->set_parent_return_data((uint8_t *)h, 64);
    return ERROR_RETURN;
}

__device__ int32_t operation_ecRecover(CuEVM::EccConstants *constants, CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used,
                                       CuEVM::evm_call_context_t *call_context) {
    gas_used += GAS_PRECOMPILE_ECRECOVER;
    int32_t error_code = ERROR_SUCCESS;
    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    printf("has gas %d\n", error_code);
    printf("gas limit %ld\n", gas_limit);
    printf("gas used %ld\n", gas_used);
    printf("data size %d\n", call_context->call_data_size);

    if (error_code == ERROR_SUCCESS) {
        // complete with zeroes the remaing bytes
        // input = arith.padded_malloc_byte_array(tmp_input, size, 128);
        CuEVM::byte_array_t input(call_context->call_data, 128);
        ecc::signature_t *signature = new ecc::signature_t();
        evm_word_t msg_hash, v, r, s, signer;
        uint256_from_bytes(&msg_hash, input.data, 32);
        uint256_from_bytes(&v, input.data + 32, 32);
        uint256_from_bytes(&r, input.data + 64, 32);
        uint256_from_bytes(&s, input.data + 96, 32);

        signature->msg_hash = msg_hash;
        signature->r = r;
        signature->s = s;
        signature->v = uint256_get_uint32_t(&v);
        printf("Sig.s \n");
        signature->s.print();
        printf("Sig.r \n");
        signature->r.print();
        printf("v %d\n", signature->v);
        printf("msg_hash \n");
        signature->msg_hash.print();

        // TODO: is not 27 and 28, only?
        if (signature->v <= 28) {
            uint8_t *output = new uint8_t[32];
            size_t res = ecc::ec_recover(constants, signature, &signer);

            if (res == ERROR_SUCCESS) {
                uint256_to_bytes(output, &signer, 32);
                printf("signer \n");
                signer.print();
                error_code = ERROR_RETURN;
                call_context->set_parent_return_data(output, 32);
            } else {
                // TODO: do we consume all gas?
                // it happens by default because of the error code
                error_code = ERROR_PRECOMPILE_UNEXPECTED_INPUT;
            }

            delete[] output;
        }
        delete signature;

        return error_code;
    }
    return error_code;
}

__device__ int32_t operation_ecAdd(CuEVM::EccConstants *constants, CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used,
                                   CuEVM::evm_call_context_t *call_context) {
    printf("ecAdd\n");
    int32_t error_code = ERROR_SUCCESS;
    gas_used += GAS_PRECOMPILE_ECADD;
    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        CuEVM::byte_array_t input(call_context->call_data, 128);

        evm_word_t x1, y1, x2, y2;
        uint256_from_bytes(&x1, input.data, 32);
        uint256_from_bytes(&y1, input.data + 32, 32);
        uint256_from_bytes(&x2, input.data + 64, 32);
        uint256_from_bytes(&y2, input.data + 96, 32);
        // print
        // printf("x1: %s\n", ecc::bnt_to_string(arith._env, x1));
        // printf("y1: %s\n", ecc::bnt_to_string(arith._env, y1));
        // printf("x2: %s\n", ecc::bnt_to_string(arith._env, x2));
        // printf("y2: %s\n", ecc::bnt_to_string(arith._env, y2));
        uint8_t *output = new uint8_t[64];
        int res = ecc::ec_add(constants->alt_BN128, &x1, &y1, &x1, &y1, &x2, &y2);
        if (res == 0) {
            uint256_to_bytes(output, &x1, 32);
            uint256_to_bytes(output + 32, &y1, 32);
            // return_data.set(output, 64);
            // *return_data = byte_array_t(output, 64);
            error_code = ERROR_RETURN;
            call_context->set_parent_return_data(output, 64);
        } else {
            // consume all gas because it is an error
            error_code = ERROR_PRECOMPILE_UNEXPECTED_INPUT;
        }
        delete[] output;
    }
    return error_code;
}

__device__ int32_t operation_ecMul(CuEVM::EccConstants *constants, CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used,
                                   CuEVM::evm_call_context_t *call_context) {
    gas_used += GAS_PRECOMPILE_ECMUL;
    int32_t error_code = ERROR_SUCCESS;
    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        CuEVM::byte_array_t input(call_context->call_data, 128);

        evm_word_t x, y, k;
        uint256_from_bytes(&x, input.data, 32);
        uint256_from_bytes(&y, input.data + 32, 32);
        uint256_from_bytes(&k, input.data + 64, 32);
        // print
        // printf("mul x: %s\n", ecc::bnt_to_string(arith._env, x));
        // printf("mul y: %s\n", ecc::bnt_to_string(arith._env, y));
        // printf("k: %s\n", ecc::bnt_to_string(arith._env, k));

        uint8_t *output = new uint8_t[64];
        int res = ecc::ec_mul(constants->alt_BN128, &x, &y, &x, &y, &k);
        // print result
        // printf("xres: %s\n", ecc::bnt_to_string(arith._env, x));
        // printf("yres: %s\n", ecc::bnt_to_string(arith._env, y));
        if (res == 0) {
            uint256_to_bytes(output, &x, 32);
            uint256_to_bytes(output + 32, &y, 32);
            // return_data.set(output, 64);
            // *return_data = byte_array_t(output, 64);
            error_code = ERROR_RETURN;
            call_context->set_parent_return_data(output, 64);
        } else {
            // consume all gas because it is an error
            error_code = ERROR_PRECOMPILE_UNEXPECTED_INPUT;
        }
        delete[] output;
    }
    return error_code;
}

__device__ int32_t operation_ecPairing(CuEVM::EccConstants *constants, CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used,
                                       CuEVM::evm_call_context_t *call_context) {
    printf("ecPairing\n");
    printf("input size %d\n", call_context->call_data_size);
    // input = message.get_data(index, length, size);
    CuEVM::byte_array_t input(call_context->call_data, call_context->call_data_size);
    CuEVM::gas_cost::ecpairing_cost(gas_used, call_context->call_data_size);
    int32_t error_code = ERROR_SUCCESS;
    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        if (call_context->call_data_size % 192 != 0) {
            error_code = ERROR_PRECOMPILE_UNEXPECTED_INPUT;
        } else {
            // 0 inputs is valid and returns 1.
            int res =
                0;  // message->data->size == 0 ? 1 : ecc::pairing_multiple(constants, input.data, message->data->size);

            printf("res: %d, idx %d \n", res, threadIdx.x);

            if (res == -1) {
                error_code = ERROR_PRECOMPILE_UNEXPECTED_INPUT;
            } else {
                uint8_t output[32];
                memset(output, 0, 32);
                output[31] = (res == 1);
                // *return_data = byte_array_t(output, 32);
                error_code = ERROR_RETURN;
            }
        }
    }
    return error_code;
}

}  // namespace precompile_operations

}  // namespace CuEVM
