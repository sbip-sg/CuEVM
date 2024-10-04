// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation
// Programme) Author: Stefan-Dan Ciocirlan Data: 2024-03-13
// SPDX-License-Identifier: MIT
#include <CuEVM/precompile.cuh>
#include <CuEVM/utils/error_codes.cuh>
#include <CuEVM/utils/arith.cuh>
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
__host__ __device__ int32_t operation_IDENTITY(
    ArithEnv &arith, bn_t &gas_limit, bn_t &gas_used,
    CuEVM::evm_return_data_t *return_data, CuEVM::evm_message_call_t *message) {
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
__host__ __device__ int32_t operation_SHA256(
    ArithEnv &arith, bn_t &gas_limit, bn_t &gas_used,
    CuEVM::evm_return_data_t *return_data, CuEVM::evm_message_call_t *message) {
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
    CuCrypto::keccak::sha3_256(message->data.data, length_size, &(hash[0]));
    *return_data = byte_array_t(hash, 32);
    return ERROR_RETURN;
}

__host__ __device__ int32_t operation_RIPEMD160(
    ArithEnv &arith, bn_t &gas_limit, bn_t &gas_used,
    CuEVM::evm_return_data_t *return_data, CuEVM::evm_message_call_t *message) {
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


__host__ __device__ int32_t operation_MODEXP(
    ArithEnv &arith, bn_t &gas_limit, bn_t &gas_used,
    CuEVM::evm_return_data_t *return_data, CuEVM::evm_message_call_t *message) {
    bn_t base_size, exponent_size, modulus_size;

    byte_array_t bsize_array = byte_array_t(message->data.data, 32);
    byte_array_t esize_array = byte_array_t(message->data.data + 32, 32);
    byte_array_t msize_array = byte_array_t(message->data.data + 64, 32);

    int32_t error = cgbn_set_byte_array_t(arith.env, base_size, bsize_array);
    error |= cgbn_set_byte_array_t(arith.env, exponent_size, esize_array);
    error |= cgbn_set_byte_array_t(arith.env, modulus_size, msize_array);

    if (error) {
        return error;
    }

    uint32_t base_len, exp_len, mod_len, data_len;

    error = cgbn_get_uint32_t(arith.env, base_len, base_size);
    error |= cgbn_get_uint32_t(arith.env, mod_len, modulus_size);
    data_len = message->data.size;

    if (error) {
        return error;
    }

    // Handle a special case when both the base and mod length are zero.
    if (base_len == 0 && mod_len == 0) {
        return ERROR_SUCCESS;
    }

    error = cgbn_get_uint32_t(arith.env, exp_len, exponent_size);

    if (error) {
        return error;
    }

    exp_len = min(exp_len, 32);

    // get a pointer to available bytes (call_exponent_size) of exponen
    // send through call data the remaining bytes are consider
    // 0 value bytes. The bytes of the call data are the most
    // significant bytes of the exponent
    uint8_t *e_data= new uint8_t[exp_len];
    for (uint32_t i = 0; i < exp_len; i++) {
        auto idx = 96 + base_len + i;
        if (idx < data_len) {
            e_data[i] = message->data.data[idx];
        } else {
            e_data[i] = 0;
        }
    }

    uint8_t adjusted_exp_data[32] = {0};

    for (uint32_t i = 0; i < exp_len; i++) {
        adjusted_exp_data[32 - exp_len + i] = e_data[i];
    }

    // todo calculate gas required
    // int32_t error = CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);

    // if (error) {
    //     return error;
    // }

    bn_t base;
    uint8_t base_data[32] = {0};
    for (uint32_t i = 0; i < base_len; i++) {
        auto idx = 96 + i;
        if (idx < data_len) {
            base_data[i] = message->data.data[idx];
        } else {
            break;
        }
    }

    auto temp_byte_array = byte_array_t(base_data, 32);
    error = cgbn_set_byte_array_t(arith.env, base, temp_byte_array);
    if (error) {
        return error;
    }

    uint8_t *mod_data = new uint8_t[mod_len];
    for (uint32_t i = 0; i < mod_len; i++) {
        auto idx = 96 + base_len + exp_len + i;
        if (idx < data_len) {
            mod_data[i] = message->data.data[idx];
        } else {
            mod_data[i] = 0;
        }
    }

    uint8_t *output = new uint8_t[mod_len];

    // convert to bigint values
    bigint base_bigint = {}, exponent_bigint = {}, result_bigint = {},
           modulus_bigint = {};
    uint8_t result[32] = {0};

    bigint_from_bytes(&base_bigint, base_data, 32);
    bigint_from_bytes(&exponent_bigint, adjusted_exp_data, 32);
    bigint_from_bytes(&modulus_bigint, mod_data, 32);
    // make the pow mod operation
    bigint_pow_mod(&result_bigint, &base_bigint, &exponent_bigint,
                   &modulus_bigint);
    bigint_to_bytes(result, &result_bigint, 32);
    *return_data = byte_array_t(result, mod_len);

    return ERROR_RETURN;
}

__host__ __device__ int32_t operation_BLAKE2(
    ArithEnv &arith, bn_t &gas_limit, bn_t &gas_used,
    CuEVM::evm_return_data_t *return_data, CuEVM::evm_message_call_t *message) {
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
    rounds = (((uint32_t)input[0] << 24) | ((uint32_t)input[1] << 16) |
              ((uint32_t)input[2] << 8) | ((uint32_t)input[3]));

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

__host__ __device__ int32_t operation_ecRecover(
    ArithEnv &arith, bn_t &gas_limit, bn_t &gas_used,
    CuEVM::evm_return_data_t *return_data, CuEVM::evm_message_call_t *message) {
    cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_PRECOMPILE_ECRECOVER);
    int32_t error_code = ERROR_SUCCESS;
    error_code |= CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);
    printf("has gas %d\n", error_code);
    printf("gas limit \n");
    print_bnt(arith, gas_limit);

    if (error_code == ERROR_SUCCESS) {

        bn_t length;
        cgbn_set_ui32(arith.env, length, message->data.size);
        bn_t index;
        cgbn_set_ui32(arith.env, index, 0);
        // complete with zeroes the remaing bytes
        // input = arith.padded_malloc_byte_array(tmp_input, size, 128);
        CuEVM::byte_array_t input(message->get_data(), 0, 128);

        // (msg_hash, input.data);
        // (v, input.data + 32);
        // (r, input.data + 64);
        // (s, input.data + 96);

        //printf("\n v %d\n", signature.v);
        //printf("r : %s\n", ecc::bnt_to_string(arith._env, r));
        //printf("s : %s\n", ecc::bnt_to_string(arith._env, s));
        //printf("msgh: %s\n", ecc::bnt_to_string(arith._env, msg_hash));
        error_code = ERROR_PRECOMPILE_UNEXPECTED_INPUT;
        // TODO: is not 27 and 28, only?
        // if (cgbn_compare_ui32(arith.env, v, 28) <= 0) {
        //     SHARED_MEMORY uint8_t output[32];
        //     size_t res = ecc::ec_recover(arith, keccak, signature, signer);
        //     if (res==0){
        //         arith.memory_from_cgbn(
        //             output,
        //             signer
        //         );
        //         return_data.set(output, 32);
        //         error_code = ERR_RETURN;
        //     } else {
        //         // TODO: do we consume all gas?
        //         // it happens by default because of the error code
        //         error_code = ERROR_PRECOMPILE_UNEXPECTED_INPUT;
        //     }

        // }
    }
    return error_code;
}

__host__ __device__ int32_t operation_ecAdd(
    ArithEnv &arith, bn_t &gas_limit, bn_t &gas_used,
    CuEVM::evm_return_data_t *return_data, CuEVM::evm_message_call_t *message) {
    return ERROR_REVERT;
}

__host__ __device__ int32_t operation_ecMul(
    ArithEnv &arith, bn_t &gas_limit, bn_t &gas_used,
    CuEVM::evm_return_data_t *return_data, CuEVM::evm_message_call_t *message) {
    return ERROR_REVERT;
}

__host__ __device__ int32_t operation_ecPairing(
    ArithEnv &arith, bn_t &gas_limit, bn_t &gas_used,
    CuEVM::evm_return_data_t *return_data, CuEVM::evm_message_call_t *message) {
    return ERROR_SUCCESS;
}

}  // namespace precompile_operations

}  // namespace CuEVM
