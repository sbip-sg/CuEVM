#include <CuEVM/gas_cost.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM {
namespace gas_cost {
__host__ __device__ int32_t has_gas(ArithEnv &arith, const bn_t &gas_limit, const bn_t &gas_used) {
    return (cgbn_compare(arith.env, gas_limit, gas_used) < 0) ? ERROR_GAS_LIMIT_EXCEEDED : ERROR_SUCCESS;
}

__host__ __device__ void max_gas_call(ArithEnv &arith, bn_t &gas_capped, const bn_t &gas_limit, const bn_t &gas_used) {
    // compute the remaining gas
    bn_t gas_left;
    cgbn_sub(arith.env, gas_left, gas_limit, gas_used);
    // cap to uint64_t in case overflow following go-ethereum
    cgbn_bitwise_mask_and(arith.env, gas_left, gas_left, 64);
    // gas capped = (63/64) * gas_left
    cgbn_div_ui32(arith.env, gas_capped, gas_left, 64);
    cgbn_sub(arith.env, gas_capped, gas_left, gas_capped);
}

__host__ __device__ void evm_words_gas_cost(ArithEnv &arith, bn_t &gas_used, const bn_t &length,
                                            const uint32_t gas_per_word) {
    // gas_used += gas_per_word * emv word count of length
    // length = (length + 31) / 32
    bn_t evm_words_gas;
    cgbn_add_ui32(arith.env, evm_words_gas, length, CuEVM::word_size - 1);
    cgbn_div_ui32(arith.env, evm_words_gas, evm_words_gas, CuEVM::word_size);
    cgbn_mul_ui32(arith.env, evm_words_gas, evm_words_gas, gas_per_word);
    cgbn_add(arith.env, gas_used, gas_used, evm_words_gas);
}

__host__ __device__ void evm_bytes_gas_cost(ArithEnv &arith, bn_t &gas_used, const bn_t &length,
                                            const uint32_t gas_per_byte) {
    // gas_used += gas_per_byte * bytes count of length
    bn_t evm_bytes_gas;
    cgbn_mul_ui32(arith.env, evm_bytes_gas, length, gas_per_byte);
    cgbn_add(arith.env, gas_used, gas_used, evm_bytes_gas);
}

__host__ __device__ int32_t exp_bytes_gas_cost(ArithEnv &arith, bn_t &gas_used, const bn_t &exponent) {
    // dynamic gas calculation (G_expbyte * bytes_in_exponent)
    int32_t last_bit;
    last_bit = CuEVM::word_bits - 1 - cgbn_clz(arith.env, exponent);
    uint32_t exponent_byte_size = (last_bit == -1) ? 0 : (last_bit) / 8 + 1;
    bn_t dynamic_gas;
    cgbn_set_ui32(arith.env, dynamic_gas, exponent_byte_size);
    cgbn_mul_ui32(arith.env, dynamic_gas, dynamic_gas, GAS_EXP_BYTE);
    cgbn_add(arith.env, gas_used, gas_used, dynamic_gas);
    return last_bit;
}

__host__ __device__ void initcode_cost(ArithEnv &arith, bn_t &gas_used, const bn_t &initcode_length) {
    // gas_used += GAS_INITCODE_WORD_COST * emv word count of initcode
    // length = (initcode_length + 31) / 32
    evm_words_gas_cost(arith, gas_used, initcode_length, GAS_INITCODE_WORD_COST);
}

__host__ __device__ void code_cost(ArithEnv &arith, bn_t &gas_used, const bn_t &code_length) {
    // gas_used += GAS_CODE_DEPOSIT * length
    evm_bytes_gas_cost(arith, gas_used, code_length, GAS_CODE_DEPOSIT);
}

__host__ __device__ void keccak_cost(ArithEnv &arith, bn_t &gas_used, const bn_t &length) {
    evm_words_gas_cost(arith, gas_used, length, GAS_KECCAK256_WORD);
}

__host__ __device__ void memory_cost(ArithEnv &arith, bn_t &gas_used, const bn_t &length) {
    evm_words_gas_cost(arith, gas_used, length, GAS_MEMORY);
}

__host__ __device__ void log_record_cost(ArithEnv &arith, bn_t &gas_used, const bn_t &length) {
    evm_bytes_gas_cost(arith, gas_used, length, GAS_LOG_DATA);
}

__host__ __device__ void log_topics_cost(ArithEnv &arith, bn_t &gas_used, const uint32_t &no_topics) {
    bn_t topic_gas;
    cgbn_set_ui32(arith.env, topic_gas, GAS_LOG_TOPIC);
    cgbn_mul_ui32(arith.env, topic_gas, topic_gas, no_topics);
    cgbn_add(arith.env, gas_used, gas_used, topic_gas);
}

__host__ __device__ void sha256_cost(ArithEnv &arith, bn_t &gas_used, const bn_t &length) {
    evm_words_gas_cost(arith, gas_used, length, GAS_PRECOMPILE_SHA256_WORD);
}

__host__ __device__ void ripemd160_cost(ArithEnv &arith, bn_t &gas_used, const bn_t &length) {
    evm_words_gas_cost(arith, gas_used, length, GAS_PRECOMPILE_RIPEMD160_WORD);
}

__host__ __device__ void blake2_cost(ArithEnv &arith, bn_t &gas_used, const uint32_t rounds) {
    // gas_used += GAS_PRECOMPILE_BLAKE2_ROUND * rounds
    bn_t temp;
    cgbn_set_ui32(arith.env, temp, rounds);
    cgbn_mul_ui32(arith.env, temp, temp, GAS_PRECOMPILE_BLAKE2_ROUND);
    cgbn_add(arith.env, gas_used, gas_used, temp);
}

__host__ __device__ int32_t modexp_cost(ArithEnv &arith, bn_t &gas_used, const bn_t &exponent_size,
                                        const bn_t &exponent_bit_length_bn, const bn_t &multiplication_complexity) {
    // compute the iteration count depending on the size
    // of the exponent and its most significant non-zero
    // bit of the least siginifcant 256 bits
    bn_t iteration_count, adjusted_exponent_bit_length;
    // cgbn_set_ui32(arith.env, iteration_count, 0);
    cgbn_set_ui32(arith.env, adjusted_exponent_bit_length, 0);
    uint32_t iteration_count_overflow;
    iteration_count_overflow = 0;
    // if the size is less than 32 bytes (256 bits) we
    // just take the position of the most significant non-zero bit
    // and substract 1
    if (cgbn_get_ui32(arith.env, exponent_bit_length_bn) != 0) {
        // exponent.bit_length() - 1
        cgbn_sub_ui32(arith.env, adjusted_exponent_bit_length, exponent_bit_length_bn, 1);
    }
    cgbn_set(arith.env, iteration_count, adjusted_exponent_bit_length);
    if (cgbn_compare_ui32(arith.env, exponent_size, 32) > 0) {
        // } else {
        // elif Esize > 32: iteration_count = (8 * (Esize - 32)) + ((exponent &
        // (2**256 - 1)).bit_length() - 1)
        cgbn_sub_ui32(arith.env, iteration_count, exponent_size, 32);
        // sometimes the iteration count can overflow
        // for high values of the exponent size
        iteration_count_overflow = cgbn_mul_ui32(arith.env, iteration_count, iteration_count, 8);
        iteration_count_overflow = iteration_count_overflow |
                                   cgbn_add(arith.env, iteration_count, iteration_count, adjusted_exponent_bit_length);
        // cgbn_sub_ui32(arith.env, iteration_count, iteration_count, 1);
    }
    // iteration_count = max(iteration_count, 1)
    if (cgbn_compare_ui32(arith.env, iteration_count, 1) < 0) {
        cgbn_set_ui32(arith.env, iteration_count, 1);
    }
#ifdef __CUDA_ARCH__

#endif
    bn_t dynamic_gas;
    uint32_t dynamic_gas_overflow;
    dynamic_gas_overflow = 0;
    // dynamic_gas = max(200, multiplication_complexity * iteration_count / 3)
    // The dynamic gas value can overflow from the overflow
    // of iteration count when the multiplication complexity
    // is non-zero or from the simple multiplication of
    // the iteration count and multiplication complexity
    // in both case the value is way over the gas limit
    // and we just throw an error which will consume the
    // entire gas given for the call
    cgbn_mul_high(arith.env, dynamic_gas, iteration_count, multiplication_complexity);
    dynamic_gas_overflow = (cgbn_compare_ui32(arith.env, dynamic_gas, 0) != 0);

    // #ifdef __CUDA_ARCH__
    //     print_bnt(arith, iteration_count);
    //     print_bnt(arith, multiplication_complexity);
    //     printf("dynamic_gas_overflow: %d\n", dynamic_gas_overflow);
    //     printf("dynamic_gas: %d\n", cgbn_get_ui32(arith.env, dynamic_gas));

    //     printf("iteration_count: %d\n", cgbn_get_ui32(arith.env, iteration_count));
    //     printf("iteration_count_overflow: %d\n", iteration_count_overflow);
    //     printf("multiplication complexity: %d\n", cgbn_get_ui32(arith.env, multiplication_complexity));
    // #endif

    cgbn_mul(arith.env, dynamic_gas, iteration_count, multiplication_complexity);
    dynamic_gas_overflow = dynamic_gas_overflow || (iteration_count_overflow &&
                                                    (cgbn_compare_ui32(arith.env, multiplication_complexity, 0) != 0));

    if (dynamic_gas_overflow) return ERROR_PRECOMPILE_MODEXP_OVERFLOW;
    cgbn_div_ui32(arith.env, dynamic_gas, dynamic_gas, 3);
    if (cgbn_compare_ui32(arith.env, dynamic_gas, 200) < 0) {
        cgbn_set_ui32(arith.env, dynamic_gas, 200);
    }
    // #ifdef __CUDA_ARCH__
    //     printf("dynamic_gas: %d\n", cgbn_get_ui32(arith.env, dynamic_gas));
    // #endif
    cgbn_add(arith.env, gas_used, gas_used, dynamic_gas);
    return ERROR_SUCCESS;
}
__host__ __device__ void ecpairing_cost(ArithEnv &arith, bn_t &gas_used, uint32_t data_size) {
    // gas_used += GAS_PRECOMPILE_ECPAIRING + data_size/192 *
    // GAS_PRECOMPILE_ECPAIRING_PAIR
    cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_PRECOMPILE_ECPAIRING);
    bn_t temp;
    cgbn_set_ui32(arith.env, temp, data_size);
    cgbn_div_ui32(arith.env, temp, temp, 192);
    cgbn_mul_ui32(arith.env, temp, temp, GAS_PRECOMPILE_ECPAIRING_PAIR);
    cgbn_add(arith.env, gas_used, gas_used, temp);
}

__host__ __device__ int32_t access_account_cost(ArithEnv &arith, bn_t &gas_used, CuEVM::TouchState &touch_state,
                                                const evm_word_t *address) {
    if (touch_state.is_warm_account(arith, address)) {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_WARM_ACCESS);
    } else {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_COLD_ACCOUNT_ACCESS);
        // set the account warm in case it's cold
        // assuming this function is called only when the account is accessed
        // TODO: remove redundant logic
        touch_state.set_warm_account(arith, address);
    }
    return ERROR_SUCCESS;
}

__host__ __device__ int32_t sload_cost(ArithEnv &arith, bn_t &gas_used, const CuEVM::TouchState &touch_state,
                                       const evm_word_t *address, const bn_t &key) {
    // get the key warm
    if (touch_state.is_warm_key(arith, address, key)) {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_WARM_ACCESS);
    } else {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_COLD_SLOAD);
    }

    return ERROR_SUCCESS;
}
__host__ __device__ int32_t sstore_cost(ArithEnv &arith, bn_t &gas_used, bn_t &gas_refund,
                                        const CuEVM::TouchState &touch_state, const evm_word_t *address,
                                        const bn_t &key, const bn_t &new_value) {
    // get the key warm
    if (touch_state.is_warm_key(arith, address, key) == false) {
        // #ifdef __CUDA_ARCH__
        //         printf("SSTORE cold %d\n", threadIdx.x);
        // #endif
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_COLD_SLOAD);
    }
    bn_t original_value, current_value;
    touch_state.poke_original_value(arith, address, key, original_value);
    touch_state.poke_value(arith, address, key, current_value);
    // #ifdef __CUDA_ARCH__
    //     printf("SSTORE COST %d\n", threadIdx.x);
    //     print_bnt(arith, original_value);
    //     print_bnt(arith, current_value);
    //     print_bnt(arith, new_value);
    // #endif
    // EIP-2200
    if (cgbn_compare(arith.env, new_value, current_value) == 0) {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_SLOAD);
    } else {
        if (cgbn_compare(arith.env, current_value, original_value) == 0) {
            if (cgbn_compare_ui32(arith.env, original_value, 0) == 0) {
                cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_STORAGE_SET);
            } else {
                cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_SSTORE_RESET);
                if (cgbn_compare_ui32(arith.env, new_value, 0) == 0) {
                    cgbn_add_ui32(arith.env, gas_refund, gas_refund, GAS_SSTORE_CLEARS_SCHEDULE);
                }
            }
        } else {
            cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_SLOAD);
            if (cgbn_compare_ui32(arith.env, original_value, 0) != 0) {
                if (cgbn_compare_ui32(arith.env, current_value, 0) == 0) {
                    cgbn_sub_ui32(arith.env, gas_refund, gas_refund, GAS_STORAGE_CLEAR_REFUND);
                } else if (cgbn_compare_ui32(arith.env, new_value, 0) == 0) {
                    cgbn_add_ui32(arith.env, gas_refund, gas_refund, GAS_STORAGE_CLEAR_REFUND);
                }
            }
            if (cgbn_compare(arith.env, original_value, new_value) == 0) {
                if (cgbn_compare_ui32(arith.env, original_value, 0) == 0) {
                    cgbn_add_ui32(arith.env, gas_refund, gas_refund, GAS_STORAGE_SET - GAS_SLOAD);
                } else {
                    cgbn_add_ui32(arith.env, gas_refund, gas_refund, GAS_STORAGE_RESET - GAS_SLOAD);
                }
            }
        }
    }
    return ERROR_SUCCESS;
}

__host__ __device__ int32_t transaction_intrinsic_gas(ArithEnv &arith, const CuEVM::evm_transaction_t &transaction,
                                                      bn_t &gas_intrinsic) {
    // gas_intrinsic = GAS_TRANSACTION
    cgbn_set_ui32(arith.env, gas_intrinsic, GAS_TRANSACTION);

    // gas_intrinsic += GAS_TRANSACTION_CREATE if transaction.create
    if (transaction.is_create) {
        cgbn_add_ui32(arith.env, gas_intrinsic, gas_intrinsic, GAS_TX_CREATE);
    }

    // gas_intrinsic += GAS_TX_DATA_ZERO/GAS_TX_DATA_NONZERO for each byte in
    // transaction.data
    for (uint32_t idx = 0; idx < transaction.data_init.size; idx++) {
        if (transaction.data_init.data[idx] == 0) {
            cgbn_add_ui32(arith.env, gas_intrinsic, gas_intrinsic, GAS_TX_DATA_ZERO);
        } else {
            cgbn_add_ui32(arith.env, gas_intrinsic, gas_intrinsic, GAS_TX_DATA_NONZERO);
        }
    }

    // if transaction type is 1 it might have access list
    if (transaction.type == 1) {
        // gas_intrinsic += GAS_ACCESS_LIST_ADDRESS/GAS_ACCESS_LIST_STORAGE for
        // each address in transaction.access_list
        for (uint32_t idx = 0; idx < transaction.access_list.accounts_count; idx++) {
            cgbn_add_ui32(arith.env, gas_intrinsic, gas_intrinsic, GAS_ACCESS_LIST_ADDRESS);
            cgbn_add_ui32(arith.env, gas_intrinsic, gas_intrinsic,
                          GAS_ACCESS_LIST_STORAGE * transaction.access_list.accounts[idx].storage_keys_count);
        }
    }

#ifdef EIP_3860
    // gas_intrinsic += GAS_INITCODE_COST if create transaction
    if (transaction.is_create) {
        if (transaction.data_init.size > max_initcode_size > 0) return ERROR_CREATE_INIT_CODE_SIZE_EXCEEDED;
        bn_t initcode_length;
        cgbn_set_ui32(arith.env, initcode_length, transaction.data_init.size);
        initcode_cost(arith, gas_intrinsic, initcode_length);
    }
#endif
    return ERROR_SUCCESS;
}

__host__ __device__ int32_t memory_grow_cost(ArithEnv &arith, const CuEVM::evm_memory_t &memory, const bn_t &index,
                                             const bn_t &length, bn_t &memory_expansion_cost, bn_t &gas_used) {
    // reset to 0;
    cgbn_set_ui32(arith.env, memory_expansion_cost, 0);
    do {
        if (cgbn_compare_ui32(arith.env, length, 0) <= 0) {
            return ERROR_SUCCESS;
        }
        bn_t offset;
        uint32_t offset_ui32;
        if (cgbn_add(arith.env, offset, index, length) != 0) {
            break;
        }
        if (cgbn_get_uint32_t(arith.env, offset_ui32, offset) == ERROR_VALUE_OVERFLOW) {
            break;
        }
        bn_t old_memory_cost;
        memory.get_memory_cost(arith, old_memory_cost);
        // memort_size_word = (offset + 31) / 32
        bn_t memory_size_word;
        if (cgbn_add_ui32(arith.env, memory_size_word, offset, 31) != 0) {
            break;
        }
        cgbn_div_ui32(arith.env, memory_size_word, memory_size_word, 32);
        // memory_cost = (memory_size_word * memory_size_word) / 512 + 3 *
        // memory_size_word
        bn_t memory_cost;
        bn_wide_t memory_size_word_wide;
        cgbn_mul_wide(arith.env, memory_size_word_wide, memory_size_word, memory_size_word);
        if (cgbn_compare_ui32(arith.env, memory_size_word_wide._high, 0) != 0) {
            break;
        }
        cgbn_set(arith.env, memory_cost, memory_size_word_wide._low);
        cgbn_div_ui32(arith.env, memory_cost, memory_cost, 512);
        bn_t tmp;
        // TODO: verify overflow in another way
        // LOOK ok from CGBN documentation
        if (cgbn_mul_ui32(arith.env, tmp, memory_size_word, GAS_MEMORY) != 0) {
            break;
        }
        if (cgbn_add(arith.env, memory_cost, memory_cost, tmp) != 0) {
            break;
        }
        //  gas_used = gas_used + memory_cost - old_memory_cost
        if (cgbn_sub(arith.env, memory_expansion_cost, memory_cost, old_memory_cost) != 0) {
            cgbn_set_ui32(arith.env, memory_expansion_cost, 0);
        }
        if (cgbn_add(arith.env, gas_used, gas_used, memory_expansion_cost) != 0) {
            break;
        }
        // size is always a multiple of 32
        if (cgbn_mul_ui32(arith.env, offset, memory_size_word, 32) != 0) {
            break;
        }
        return ERROR_SUCCESS;
    } while (0);
    return ERR_MEMORY_INVALID_OFFSET;
}
}  // namespace gas_cost
}  // namespace CuEVM