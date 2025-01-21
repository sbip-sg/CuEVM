#include <CuEVM/gas_cost.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM {
namespace gas_cost {
__device__ int32_t has_gas(const gas_t &gas_limit, const gas_t &gas_used) {
    return (gas_limit < gas_used) ? ERROR_GAS_LIMIT_EXCEEDED : ERROR_SUCCESS;
}

__device__ gas_t max_gas_call(const gas_t &gas_limit, const gas_t &gas_used) {
    // compute the remaining gas
    // cap to uint64_t in case overflow following go-ethereum
    // gas_left = gas_left & 0xFFFFFFFFFFFFFFFF;
    // gas capped = (63/64) * gas_left
    gas_t available_gas = gas_limit - gas_used;
    return available_gas - available_gas / 64;
}

__device__ void evm_words_gas_cost(gas_t &gas_used, const gas_t &length, const uint32_t gas_per_word) {
    // gas_used += gas_per_word * emv word count of length
    // length = (length + 31) / 32
    gas_t evm_words_gas;
    evm_words_gas = (length + 31) / 32;
    evm_words_gas = evm_words_gas * gas_per_word;
    gas_used += evm_words_gas;
}

__device__ void evm_bytes_gas_cost(gas_t &gas_used, const gas_t &length, const uint32_t gas_per_byte) {
    gas_used += length * gas_per_byte;
}

__device__ int32_t exp_bytes_gas_cost(gas_t &gas_used, const evm_word_t &exponent) {
    // dynamic gas calculation (G_expbyte * bytes_in_exponent)
    // int32_t last_bit;
    // last_bit = CuEVM::word_bits - 1 - cgbn_clz(exponent);
    // uint32_t exponent_byte_size = (last_bit == -1) ? 0 : (last_bit) / 8 + 1;
    uint32_t exponent_byte_size = 0;  // todo fix this
    gas_t dynamic_gas;
    dynamic_gas = exponent_byte_size * GAS_EXP_BYTE;
    gas_used += dynamic_gas;
    return ERROR_SUCCESS;
}

__device__ void initcode_cost(gas_t &gas_used, const gas_t &initcode_length) {
    // gas_used += GAS_INITCODE_WORD_COST * emv word count of initcode
    // length = (initcode_length + 31) / 32
    evm_words_gas_cost(gas_used, initcode_length, GAS_INITCODE_WORD_COST);
}

__device__ void code_cost(gas_t &gas_used, const gas_t &code_length) {
    // gas_used += GAS_CODE_DEPOSIT * length
    evm_bytes_gas_cost(gas_used, code_length, GAS_CODE_DEPOSIT);
}

__device__ void keccak_cost(gas_t &gas_used, const gas_t &length) {
    evm_words_gas_cost(gas_used, length, GAS_KECCAK256_WORD);
}

__device__ void memory_cost(gas_t &gas_used, const gas_t &length) { evm_words_gas_cost(gas_used, length, GAS_MEMORY); }

__device__ void log_record_cost(gas_t &gas_used, const gas_t &length) {
    evm_bytes_gas_cost(gas_used, length, GAS_LOG_DATA);
}

__device__ void log_topics_cost(gas_t &gas_used, const uint32_t &no_topics) { gas_used += GAS_LOG_TOPIC * no_topics; }

__device__ void sha256_cost(gas_t &gas_used, const gas_t &length) {
    evm_words_gas_cost(gas_used, length, GAS_PRECOMPILE_SHA256_WORD);
}

__device__ void ripemd160_cost(gas_t &gas_used, const gas_t &length) {
    evm_words_gas_cost(gas_used, length, GAS_PRECOMPILE_RIPEMD160_WORD);
}

__device__ void blake2_cost(gas_t &gas_used, const gas_t &rounds) {
    // gas_used += GAS_PRECOMPILE_BLAKE2_ROUND * rounds
    gas_used += GAS_PRECOMPILE_BLAKE2_ROUND * rounds;
}

__device__ int32_t modexp_cost(gas_t &gas_used, const uint32_t &exponent_size, const uint32_t &exponent_bit_length_bn,
                               const uint32_t &multiplication_complexity) {
    // compute the iteration count depending on the size
    // of the exponent and its most significant non-zero
    // bit of the least siginifcant 256 bits
    /*
    gas_t iteration_count, adjusted_exponent_bit_length;
    // cgbn_set_ui32(arith.env, iteration_count, 0);
    cgbn_set_ui32(adjusted_exponent_bit_length, 0);
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
    */ // TODO: reimplement this
    return ERROR_SUCCESS;
}
__device__ void ecpairing_cost(gas_t &gas_used, const gas_t &data_size) {
    // gas_used += GAS_PRECOMPILE_ECPAIRING + data_size/192 *
    // GAS_PRECOMPILE_ECPAIRING_PAIR
    gas_used += GAS_PRECOMPILE_ECPAIRING + data_size / 192 * GAS_PRECOMPILE_ECPAIRING_PAIR;
}

__device__ int32_t access_account_cost(gas_t &gas_used, CuEVM::StateDb *state_db, const evm_word_t *address) {
    if (state_db->is_warm_account(address)) {
        gas_used += GAS_WARM_ACCESS;
    } else {
        gas_used += GAS_COLD_ACCOUNT_ACCESS;
        // set the account warm in case it's cold
        // assuming this function is called only when the account is accessed
        // TODO: remove redundant logic
        state_db->set_warm_account(address);
    }
    return ERROR_SUCCESS;
}

__device__ int32_t sload_cost(gas_t &gas_used, CuEVM::StateDb *state_db, const evm_word_t *address,
                              const evm_word_t *key) {
    // get the key warm
    if (state_db->is_warm_key(address, key)) {
        gas_used += GAS_WARM_ACCESS;
    } else {
        gas_used += GAS_COLD_SLOAD;
    }

    return ERROR_SUCCESS;
}
__device__ int32_t sstore_cost(gas_t &gas_used, gas_t &gas_refund, CuEVM::StateDb *state_db, const evm_word_t *address,
                               const evm_word_t *key, const evm_word_t *new_value) {
    // get the key warm
    if (state_db->is_warm_key(address, key) == false) {
        gas_used += GAS_COLD_SLOAD;
    }

    evm_word_t *original_value = nullptr, *current_value = nullptr;
    ValueStatus *value_status = state_db->get_value_status(address, key);
    bool blank_storage = false;
    if (value_status == nullptr)
        blank_storage = true;
    else {
        original_value = &value_status->original_value;
        current_value = &value_status->value;
    }
    // printf("original value %p\n", original_value);
    // printf("current value %p\n", current_value);
    // printf("new value %p\n", new_value);
    // new_value->print();
    // EIP-2200
    if (uint256_cmp(new_value, current_value) == 0) {
        gas_used += GAS_SLOAD;
    } else {
        if (uint256_cmp(current_value, original_value) == 0) {
            // printf("current value is equal to original value\n");
            if (uint256_is_zero(current_value)) {
                // printf("original value is zero\n");
                gas_used += GAS_STORAGE_SET;
            } else {
                // printf("original value is not zero\n");
                gas_used += GAS_SSTORE_RESET;
                if (uint256_is_zero(new_value)) {
                    gas_refund += GAS_SSTORE_CLEARS_SCHEDULE;
                }
            }
        } else {
            gas_used += GAS_SLOAD;
            if (uint256_is_zero(original_value)) {
                if (uint256_is_zero(current_value)) {
                    gas_refund -= GAS_STORAGE_CLEAR_REFUND;
                } else if (uint256_is_zero(new_value)) {
                    gas_refund += GAS_STORAGE_CLEAR_REFUND;
                }
            }
            if (uint256_cmp(original_value, new_value) == 0 ||
                (blank_storage && uint256_cmp(new_value, current_value) == 0)) {
                if (uint256_is_zero(original_value)) {
                    gas_refund += GAS_STORAGE_SET - GAS_SLOAD;
                } else {
                    gas_refund += GAS_STORAGE_RESET - GAS_SLOAD;
                }
            }
        }
    }
    return ERROR_SUCCESS;
}

__device__ int32_t transaction_intrinsic_gas(const CuEVM::transaction::TransactionList *transaction_list,
                                             gas_t &gas_intrinsic) {
    // gas_intrinsic = GAS_TRANSACTION
    gas_intrinsic = GAS_TRANSACTION;

    // gas_intrinsic += GAS_TRANSACTION_CREATE if transaction.create
    // if (transaction_list->is_create) {
    //     gas_intrinsic += GAS_TX_CREATE;
    // }

    // gas_intrinsic += GAS_TX_DATA_ZERO/GAS_TX_DATA_NONZERO for each byte in
    // transaction.data
    for (uint32_t idx = 0; idx < transaction_list->call_data_size[INSTANCE_GLOBAL_IDX]; idx++) {
        if (transaction_list->call_data[transaction_list->call_data_offset[INSTANCE_GLOBAL_IDX] + idx] == 0) {
            gas_intrinsic += GAS_TX_DATA_ZERO;
        } else {
            gas_intrinsic += GAS_TX_DATA_NONZERO;
        }
    }

    // gas_intrinsic += GAS_ACCESS_LIST_ADDRESS/GAS_ACCESS_LIST_STORAGE for
    // each address in transaction.access_list

    // for (uint32_t idx = 0; idx < transaction.access_list.accounts_count; idx++) {
    //     gas_intrinsic += GAS_ACCESS_LIST_ADDRESS;
    //     gas_intrinsic += GAS_ACCESS_LIST_STORAGE * transaction.access_list.accounts[idx].storage_keys_count;
    // }

#ifdef EIP_3860
    // gas_intrinsic += GAS_INITCODE_COST if create transaction
    // if (transaction.is_create) {
    //     if (transaction.data_init.size > max_initcode_size > 0) return ERROR_CREATE_INIT_CODE_SIZE_EXCEEDED;
    //     initcode_cost(gas_intrinsic, transaction.data_init.size);
    // }
#endif
    return ERROR_SUCCESS;
}

__device__ int32_t memory_grow_cost(const CuEVM::evm_memory_t *memory, const uint32_t index, const uint32_t length,
                                    gas_t &memory_expansion_cost, gas_t &gas_used) {
    // reset to 0;
    memory_expansion_cost = 0;
    if (length == 0) return ERROR_SUCCESS;
    gas_t new_size = index + length;
    gas_t new_size_words = (new_size + 31) / 32;
    // gas_cost = (new_mem_size_words ^ 2 // 512) + (3 * new_mem_size_words) - Cmem(old_state
    gas_t new_cost = (new_size_words * new_size_words / 512) + (3 * new_size_words);
    if (new_cost > memory->memory_cost) {
        memory_expansion_cost = new_cost - memory->memory_cost;
        gas_used += memory_expansion_cost;
    }
    return ERROR_SUCCESS;
}
}  // namespace gas_cost
}  // namespace CuEVM