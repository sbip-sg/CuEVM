#include <CuEVM/gas_cost.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM {
namespace gas_cost {
__device__ int32_t has_gas(const gas_t &gas_limit, const gas_t &gas_used) {
    return (gas_limit < gas_used) ? ERROR_GAS_LIMIT_EXCEEDED : ERROR_SUCCESS;
}

__device__ gas_t max_gas_call(const gas_t &gas_limit, const gas_t &gas_used) {
    // compute the remaining gas
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
    // Start of Selection
}
__device__ int32_t modexp_cost(gas_t &gas_used, const evm_word_t &exponent_size,
                               const evm_word_t &exponent_bit_length_bn, const evm_word_t &multiplication_complexity) {
    // Compute the effective iteration count for dynamic gas cost as defined in EIP-198.
    // For exponent sizes greater than 32:
    //      extra_exponent = (exponent_size - 32) * 8
    // For the lower 256-bit portion of the exponent:
    //      msb = (exponent_bit_length_bn > 0) ? (exponent_bit_length_bn - 1) : 0
    // The adjusted exponent (i.e. iteration count) is then:
    //      adjusted_exponent = max(extra_exponent + msb, 1)

    printf("exponent_size ");
    print_uint256(&exponent_size);
    printf("exponent_bit_length_bn ");
    print_uint256(&exponent_bit_length_bn);
    printf("multiplication_complexity ");
    print_uint256(&multiplication_complexity);

    evm_word_t extra_exponent = 0;
    const evm_word_t multiplier_for_extra = 8;
    if (uint256_cmp_word(&exponent_size, 32) > 0) {
        // extra_exponent = (exponent_size - 32) * 8
        uint256_sub_word(&extra_exponent, &exponent_size, 32);
        uint256_mul(&extra_exponent, &extra_exponent, &multiplier_for_extra);
    }

    // Calculate msb = (exponent_bit_length_bn > 0) ? (exponent_bit_length_bn - 1) : 0
    evm_word_t msb = 0;
    if (!uint256_is_zero(&exponent_bit_length_bn)) {
        uint256_sub_word(&msb, &exponent_bit_length_bn, 1);
    }

    // Compute adjusted_exponent = extra_exponent + msb, ensuring a minimum value of 1.
    evm_word_t adjusted_exponent;
    uint256_add(&adjusted_exponent, &extra_exponent, &msb);
    if (uint256_is_zero(&adjusted_exponent)) {
        adjusted_exponent = 1;
    }

    printf("adjusted_exponent ");
    print_uint256(&adjusted_exponent);

    // Compute dynamic gas cost using the EIP-198 formula:
    //      dynamic_gas = floor((multiplication_complexity * adjusted_exponent) / 3)
    // with a minimum cost of 200.
    evm_word_t dynamic_gas;
    const evm_word_t divisor = 3;
    uint256_mul(&dynamic_gas, &multiplication_complexity, &adjusted_exponent);
    uint256_div(&dynamic_gas, &dynamic_gas, &divisor);

    if (uint256_cmp_word(&dynamic_gas, 200) < 0) {
        dynamic_gas = 200;
    }

    printf("dynamic_gas ");
    print_uint256(&dynamic_gas);

    gas_used += uint256_get_uint64_t(&dynamic_gas);
    return ERROR_SUCCESS;
}
// End of Selectio
__device__ void ecpairing_cost(gas_t &gas_used, const gas_t &data_size) {
    // gas_used += GAS_PRECOMPILE_ECPAIRING + data_size/192 *
    // GAS_PRECOMPILE_ECPAIRING_PAIR
    gas_used += GAS_PRECOMPILE_ECPAIRING + data_size / 192 * GAS_PRECOMPILE_ECPAIRING_PAIR;
}

__device__ int32_t access_account_cost(gas_t &gas_used, CuEVM::StateDb *state_db, const evm_word_t *address,
                                       SnapshotState *snapshot_state, bool set_warm) {
    if (state_db->is_warm_account(address, snapshot_state, set_warm)) {
        // printf("warm account\n");
        gas_used += GAS_WARM_ACCESS;
    } else {
        // printf("cold account\n");
        gas_used += GAS_COLD_ACCOUNT_ACCESS;
        // set the account warm in case it's cold
        // assuming this function is called only when the account is accessed
        // TODO: remove redundant logic
        // state_db->set_warm_account(address);
    }
    return ERROR_SUCCESS;
}

__device__ int32_t sstore_cost(gas_t &gas_used, gas_t &gas_refund, CuEVM::StateDb *state_db, const evm_word_t *address,
                               const evm_word_t *key, const evm_word_t *new_value, int32_t &address_index,
                               ValueStatus *&found_value, SnapshotState *snapshot_state) {
    // printf("gas used before %lu thread %d\n", gas_used, THREADIDX);
    // get the key warm
    if (state_db->is_warm_key_with_offset(address, key, address_index, found_value, snapshot_state, true) == false) {
        // printf("cold sstore\n");
        gas_used += GAS_COLD_SLOAD;
    }

    evm_word_t *original_value = nullptr, *current_value = nullptr;
    // ValueStatus *value_status = state_db->get_value_status(address, key);
    bool blank_storage = false;
    if (found_value == nullptr)
        blank_storage = true;
    else {
        original_value = &found_value->original_value;
        current_value = &found_value->value;
    }
    // if (THREADIDX == 0) {
    //     printf("original value %p\n", original_value);
    //     if (original_value != nullptr) {
    //         original_value->print();
    //     }
    //     printf("current value %p\n", current_value);
    //     if (current_value != nullptr) {
    //         current_value->print();
    //     }

    //     new_value->print();
    // }
    // __syncthreads();
    // if (THREADIDX == 1) {
    //     printf("original value %p\n", original_value);
    //     if (original_value != nullptr) {
    //         original_value->print();
    //     }
    //     printf("current value %p\n", current_value);
    //     if (current_value != nullptr) {
    //         current_value->print();
    //     }

    //     new_value->print();
    // }
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
    // printf("gas used after %lu thread %d\n", gas_used, THREADIDX);
    return ERROR_SUCCESS;
}

__device__ int32_t transaction_intrinsic_gas(const CuEVM::transaction::TransactionList *transaction_list,
                                             gas_t &gas_intrinsic) {
    // gas_intrinsic = GAS_TRANSACTION
    gas_intrinsic = GAS_TRANSACTION;

    // gas_intrinsic += GAS_TRANSACTION_CREATE if transaction.create

    // gas_intrinsic += GAS_TX_DATA_ZERO/GAS_TX_DATA_NONZERO for each byte in
    // transaction.data
    uint32_t call_data_size = transaction_list->call_data_size[INSTANCE_GLOBAL_IDX];
    uint32_t call_data_offset = transaction_list->call_data_offset[INSTANCE_GLOBAL_IDX];
    for (uint32_t idx = 0; idx < call_data_size; idx++) {
        if (transaction_list->call_data[call_data_offset + idx] == 0) {
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
    if (transaction_list->type == SPECIAL_CREATE_TRANSACTION_TYPE) {
        gas_intrinsic += GAS_TX_CREATE;
        if (call_data_size > max_initcode_size > 0) return ERROR_CREATE_INIT_CODE_SIZE_EXCEEDED;
        initcode_cost(gas_intrinsic, call_data_size);
    }
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