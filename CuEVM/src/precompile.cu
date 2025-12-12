#include <CuEVM/gas_cost.cuh>
#include <CuEVM/precompile.cuh>
#include <CuEVM/utils/error_codes.cuh>
#ifdef BUILD_LIBRARY
#include <CuEVM/utils/library_utils.h>
#endif
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
    call_context->set_parent_return_data(output, 32);

    return ERROR_RETURN;
}

__device__ int32_t operation_MODEXP(gas_t &gas_limit, gas_t &gas_used, CuEVM::evm_call_context_t *call_context) {
    evm_word_t base_size, exponent_size, modulus_size;
    uint8_t all_input_data[96];
    for (int i = 0; i < 96; i++) {
        if (i < call_context->call_data_size) {
            all_input_data[i] = call_context->call_data[i];
        } else {
            all_input_data[i] = 0;
        }
    }

    uint256_from_bytes(&base_size, all_input_data, 32);
    uint256_from_bytes(&exponent_size, all_input_data + 32, 32);
    uint256_from_bytes(&modulus_size, all_input_data + 64, 32);

#ifdef BUILD_GO_LIBRARY
    // bypass fuzzing mode
    // TODO: make it configurable or allow max exp len
    return ERROR_RETURN;
#endif

    int32_t error = ERROR_SUCCESS;

    uint32_t base_len, exp_len, mod_len, data_len;
    data_len = call_context->call_data_size;
    base_len = uint256_get_uint32_t(&base_size);
    exp_len = uint256_get_uint32_t(&exponent_size);
    mod_len = uint256_get_uint32_t(&modulus_size);

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

    // Safe allocation: ensure at least 1 byte to avoid malloc(0)
    uint32_t safe_exp_len = (exp_len == 0) ? 1 : exp_len;
    if (safe_exp_len > data_len) {
        safe_exp_len = data_len;
    }
    uint8_t *e_data = new uint8_t[safe_exp_len];

    // Initialize with zeros if exp_len was 0
    if (exp_len == 0) {
        e_data[0] = 0;
    }

    for (uint32_t i = 0; i < exp_len; i++) {
        uint32_t idx = 96 + base_len + i;
        if (idx < data_len) {
            e_data[i] = call_context->call_data[idx];
        } else {
            if (i < safe_exp_len)
                e_data[i] = 0;
            else
                break;
        }

        if (e_data[i] != 0) {
            exp_is_zero = false;
        }
    }

    // take head 32 bytes of the exponent
    uint8_t adjusted_exp_data[32] = {0};
    uint32_t iteration_length = min(32, exp_len);
    for (uint32_t i = 0; i < iteration_length; i++) {
        adjusted_exp_data[32 - iteration_length + i] = e_data[i];
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

    // Safe allocation for base_data
    uint32_t safe_base_len = (base_len == 0) ? 1 : base_len;
    // This is a safe condition to avoid large memory allocation
    // To be checked with max gas limit and the max len supported.
    // TODO: adjust by checking correctness with eth-tests
    if (safe_base_len > data_len) {
        safe_base_len = data_len;
    }
    uint8_t *base_data = new uint8_t[safe_base_len];

    // Initialize with zeros if base_len was 0
    if (base_len == 0) {
        base_data[0] = 0;
    }

    for (uint32_t i = 0; i < base_len; i++) {
        auto idx = 96 + i;
        if (idx < data_len) {
            base_data[i] = call_context->call_data[idx];
        } else {
            if (i < safe_base_len)
                base_data[i] = 0;
            else
                break;
        }

        if (base_data[i] != 0) {
            base_is_zero = false;
        }
    }

    // Safe allocation for mod_data
    uint32_t safe_mod_len = (mod_len == 0) ? 1 : mod_len;
    if (safe_mod_len > data_len) {
        safe_mod_len = data_len;
    }
    uint8_t *mod_data = new uint8_t[safe_mod_len];

    // Initialize with zeros if mod_len was 0
    if (mod_len == 0) {
        mod_data[0] = 0;
    }

    // loop and check for zero, if zero, then return 0
    bool mod_is_one = true;
    bool mod_is_zero = true;
    for (uint32_t i = 0; i < mod_len; i++) {
        auto idx = 96 + base_len + exp_len + i;
        // mod_data[i] = (idx < data_len) ? call_context->call_data[idx] : 0;
        if (idx < data_len) {
            mod_data[i] = call_context->call_data[idx];
        } else {
            if (i < safe_mod_len)
                mod_data[i] = 0;
            else
                break;
        }

        if (mod_data[i] != 0) {
            mod_is_zero = false;
            if (mod_data[i] != 1 || i != mod_len - 1) {
                mod_is_one = false;
            }
        } else if (i == mod_len - 1) {
            mod_is_one = false;
        }
    }

    // early return special cases
    if (mod_is_zero) {
        if (mod_len != 0) {
            call_context->set_parent_return_data(mod_data, mod_len);
        } else if (call_context->parent != nullptr) {
            call_context->parent->dynamic_ret_size = 0;
        }
        return ERROR_RETURN;
    }

    if (exp_is_zero) {
        for (uint32_t i = 0; i < mod_len; i++) {
            mod_data[i] = 0;
        }
        if (!mod_is_one) {
            mod_data[mod_len - 1] = 1;  // return 1
        }
        call_context->set_parent_return_data(mod_data, mod_len);
        delete[] e_data;
        return ERROR_RETURN;
    }

    // convert to bigint values
    bigint base_bigint = {}, exponent_bigint = {}, result_bigint = {}, modulus_bigint = {};

    bigint_from_bytes(&base_bigint, base_data, base_len);
    bigint_from_bytes(&exponent_bigint, e_data, exp_len);
    bigint_from_bytes(&modulus_bigint, mod_data, mod_len);
    printf("call pow mod\n");
    // make the pow mod operation
    bigint_pow_mod(&result_bigint, &base_bigint, &exponent_bigint, &modulus_bigint);
    bigint_to_bytes(mod_data, &result_bigint, mod_len);
    call_context->set_parent_return_data(mod_data, mod_len);
    delete[] e_data;
    delete[] base_data;
    delete[] mod_data;
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
#ifdef DEBUG
    printf("has gas %d\n", error_code);
    printf("gas limit %ld\n", gas_limit);
    printf("gas used %ld\n", gas_used);
    printf("data size %d\n", call_context->call_data_size);
#endif

    if (error_code == ERROR_SUCCESS) {
        // complete with zeroes the remaing bytes
        // input = arith.padded_malloc_byte_array(tmp_input, size, 128);
        uint8_t input[128];
        for (uint32_t i = 0; i < 128; i++) {
            input[i] = i < call_context->call_data_size ? call_context->call_data[i] : 0;
        }
        ecc::signature_t *signature = new ecc::signature_t();
        evm_word_t msg_hash, v, r, s, signer;
        uint256_from_bytes(&msg_hash, input, 32);
        uint256_from_bytes(&v, input + 32, 32);
        uint256_from_bytes(&r, input + 64, 32);
        uint256_from_bytes(&s, input + 96, 32);

        signature->msg_hash = msg_hash;
        signature->r = r;
        signature->s = s;
        signature->v = uint256_get_uint32_t(&v);
#ifdef DEBUG
        printf("Sig.s \n");
        signature->s.print();
        printf("Sig.r \n");
        signature->r.print();
        printf("v %d\n", signature->v);
        printf("msg_hash \n");
        signature->msg_hash.print();
#endif
#ifdef BUILD_GO_LIBRARY
        // bypass fuzzing mode
        // TODO: make it configurable
        if (signature->v == 28 || signature->v == 27) {
            uint8_t *output = new uint8_t[32];

            size_t res = ERROR_SUCCESS;
            signer = 0;
#else
        // TODO: is not 27 and 28, only?
        if (signature->v == 28 || signature->v == 27) {
            uint8_t *output = new uint8_t[32];
            size_t res = ecc::ec_recover(constants, signature, &signer);
#endif

            if (res == ERROR_SUCCESS) {
                uint256_to_bytes(output, &signer, 32);
#ifdef DEBUG
                if (threadIdx.x == 0) {
                    printf(" THREAD %d signer \n", threadIdx.x);
                    signer.print();
                }
#endif
                error_code = ERROR_RETURN;
                call_context->set_parent_return_data(output, 32);
            } else {
                // TODO: do we consume all gas?
                // it happens by default because of the error code
                error_code = ERROR_PRECOMPILE_UNEXPECTED_INPUT;
            }

            delete[] output;
        } else {
            error_code = ERROR_RETURN;
        }
        delete signature;

        return error_code;
    }
    return error_code;
}

__device__ int32_t operation_ecAdd(CuEVM::EccConstants *constants, CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used,
                                   CuEVM::evm_call_context_t *call_context) {
    int32_t error_code = ERROR_SUCCESS;
    gas_used += GAS_PRECOMPILE_ECADD;
    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        uint8_t input[128];
        for (uint32_t i = 0; i < 128; i++) {
            input[i] = i < call_context->call_data_size ? call_context->call_data[i] : 0;
        }
        evm_word_t x1, y1, x2, y2;
        uint256_from_bytes(&x1, input, 32);
        uint256_from_bytes(&y1, input + 32, 32);
        uint256_from_bytes(&x2, input + 64, 32);
        uint256_from_bytes(&y2, input + 96, 32);

        uint8_t *output = new uint8_t[64];
#ifdef BUILD_GO_LIBRARY
        // bypass fuzzing mode
        // TODO: make it configurable
        int res = 0;
#else
        int res = ecc::ec_add(constants->alt_BN128, &x1, &y1, &x1, &y1, &x2, &y2);
#endif
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
        uint8_t input[96];
        for (uint32_t i = 0; i < 96; i++) {
            input[i] = i < call_context->call_data_size ? call_context->call_data[i] : 0;
        }
        evm_word_t x, y, k;
        uint256_from_bytes(&x, input, 32);
        uint256_from_bytes(&y, input + 32, 32);
        uint256_from_bytes(&k, input + 64, 32);

        uint8_t *output = new uint8_t[64];
#ifdef BUILD_GO_LIBRARY
        // bypass fuzzing mode
        // TODO: make it configurable
        int res = 0;
#else
        int res = ecc::ec_mul(constants->alt_BN128, &x, &y, &x, &y, &k);
#endif

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
    // printf("ecPairing\n");
    // printf("input size %d\n", call_context->call_data_size);
    // input = message.get_data(index, length, size);

    CuEVM::gas_cost::ecpairing_cost(gas_used, call_context->call_data_size);
    int32_t error_code = ERROR_SUCCESS;
    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        if (call_context->call_data_size % 192 != 0) {
            error_code = ERROR_PRECOMPILE_UNEXPECTED_INPUT;
        } else {
            // 0 inputs is valid and returns 1.
#ifdef BUILD_GO_LIBRARY
            // bypass fuzzing mode
            // TODO: make it configurable
            int res = 1;
#else
            // TODO: fix this
            int res = 1;  // ecc::pairing_multiple(constants, input.data, call_context->call_data_size);
#endif

            // printf("res: %d, idx %d \n", res, threadIdx.x);

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

#ifdef BUILD_LIBRARY
__device__ int32_t operation_TransparentAttacker(CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used,
                                                 CuEVM::evm_call_context_t *call_context) {
    // todo when optimizing, reentrancy attacker becomes a precompile
    // printf("Thread %d: TransparentAttacker\n", INSTANCE_GLOBAL_IDX);
    uint8_t *output = g_fuzzing_constants->return_buffer;
    // printf("Thread %d TransparentAttacker return buffer: %p\n", INSTANCE_GLOBAL_IDX, output);
    call_context->set_parent_return_data(output, RETURN_BUFFER_SIZE);
    return ERROR_RETURN;
}
// todo when optimizing, reentrancy attacker becomes a precompile
__device__ int32_t operation_TransparentAttackerEnhanced(CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used,
                                                         CuEVM::evm_call_context_t *call_context,
                                                         const transaction::TransactionList *transaction_list_ptr) {
    // todo when optimizing, attacker becomes a precompile
    // uint8_t *output = g_fuzzing_constants->return_buffer;
    // // printf("Thread %d TransparentAttacker return buffer: %p\n", INSTANCE_GLOBAL_IDX, output);
    // call_context->set_parent_return_data(output, RETURN_BUFFER_SIZE);

    if (call_context->parent == nullptr) return ERROR_RETURN;
    uint8_t *buf =
        CuEVM::memory_pool::preallocated_return_data_base + INSTANCE_GLOBAL_IDX * memory_pool_return_data_preallocate;
    memset(buf, 0, memory_pool_return_data_preallocate);  // Zero entire 128 bytes upfront for simplicity.
    uint8_t mode = transaction_list_ptr->block_number[INSTANCE_GLOBAL_IDX] % 3;
    uint8_t *start_ptr = nullptr;  // to copy 32 bytes to return data.
    uint64_t word = 0;             // for randomly generated value
    uint32_t seed = 0;
    if (mode == 0) {
        // start_ptr = g_fuzzing_constants->return_buffer;
        buf[31] = 1;
    } else if (mode == 1) {
        // address: cast to uint160 for right-alignment

        uint32_t indx = transaction_list_ptr->time_stamp[INSTANCE_GLOBAL_IDX] % 8;
        // printf("Thread %d  address mode indx %d\n", INSTANCE_GLOBAL_IDX, indx);
        uint8_t *src = (indx < g_fuzzing_constants->address_constants_count)
                           ? g_fuzzing_constants->address_constants + indx * 32
                           : g_fuzzing_constants->return_buffer;  // Fallback to return_buffer (assume zeros)
        memcpy(buf, src, 32);
    } else {
        // number: direct uint256 value
        seed = uint32_t(transaction_list_ptr->time_stamp[INSTANCE_GLOBAL_IDX]);
        if (seed % 2 == 0) {
            // Generate up to 2 uint32 words (high to low significance)
            uint32_t word1 = 0, word2 = 0;
            int count = 0;
            uint32_t curr = seed * 1664525u + 1013904223u;
            word1 = curr;
            curr = curr * 1664525u + 1013904223u;
            if (curr % 2 == 0) {
                word2 = curr;
            }
            // Place big-endian bytes right-aligned
            int byte_offset = 32 - count * 4;

            buf[byte_offset++] = (word1 >> 24) & 0xFF;
            buf[byte_offset++] = (word1 >> 16) & 0xFF;
            buf[byte_offset++] = (word1 >> 8) & 0xFF;
            buf[byte_offset++] = word1 & 0xFF;
            buf[byte_offset++] = (word2 >> 24) & 0xFF;
            buf[byte_offset++] = (word2 >> 16) & 0xFF;
            buf[byte_offset++] = (word2 >> 8) & 0xFF;
            buf[byte_offset++] = word2 & 0xFF;

        } else {
            seed = (seed * 1664525 + 1013904223) % 16;
            uint8_t *src = (seed < g_fuzzing_constants->integer_constants_count)
                               ? g_fuzzing_constants->integer_constants + seed * 32
                               : g_fuzzing_constants->return_buffer;  // Fallback to return_buffer (assume zeros)
            memcpy(buf, src, 32);
        }
    }

    // printf("set_parent_return_data %u %u\n", data, size);
    call_context->parent->dynamic_ret_size = memory_pool_return_data_preallocate;
    call_context->dynamic_ret_size = memory_pool_return_data_preallocate;

    return ERROR_RETURN;
}
#endif

}  // namespace precompile_operations

}  // namespace CuEVM
