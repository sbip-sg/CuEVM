#include "include/gas_cost.cuh"
#include "include/utils/error_codes.cuh"


namespace cuEVM {
    namespace gas_cost {
        __host__ __device__ int32_t has_gas(
            ArithEnv &arith,
            const bn_t &gas_limit,
            const bn_t &gas_used) {
            return (cgbn_compare(arith.env, gas_limit, gas_used) < 0) ? ERROR_GAS_LIMIT_EXCEEDED : ERROR_SUCCESS;
        }
        
        __host__ __device__ void max_gas_call(
            ArithEnv &arith,
            bn_t &gas_capped,
            const bn_t &gas_limit,
            const bn_t &gas_used) {
            // compute the remaining gas
            bn_t gas_left;
            cgbn_sub(arith.env, gas_left, gas_limit, gas_used);

            // gas capped = (63/64) * gas_left
            cgbn_div_ui32(arith.env, gas_capped, gas_left, 64);
            cgbn_sub(arith.env, gas_capped, gas_left, gas_capped);
        }
        
        __host__ __device__ void evm_words_gas_cost
        (
            ArithEnv &arith,
            bn_t &gas_used,
            const bn_t &length,
            const uint32_t gas_per_word) {
            // gas_used += gas_per_word * emv word count of length
            // length = (length + 31) / 32
            bn_t evm_words_gas;
            cgbn_add_ui32(arith.env, evm_words_gas, length, cuEVM::word_size -1);
            cgbn_div_ui32(arith.env, evm_words_gas, evm_words_gas, cuEVM::word_size);
            cgbn_mul_ui32(arith.env, evm_words_gas, evm_words_gas, gas_per_word);
            cgbn_add(arith.env, gas_used, gas_used, evm_words_gas);
        }
        
        __host__ __device__ void initcode_cost(
            ArithEnv &arith,
            bn_t &gas_used,
            const bn_t &initcode_length
        ) {
            // gas_used += GAS_INITCODE_WORD_COST * emv word count of initcode
            // length = (initcode_length + 31) / 32
            evm_words_gas_cost(arith, gas_used, initcode_length, GAS_INITCODE_WORD_COST);
        }
        
        __host__ __device__ void keccak_cost(
            ArithEnv &arith,
            bn_t &gas_used,
            const bn_t &length) {
            evm_words_gas_cost(arith, gas_used, length, GAS_KECCAK256_WORD);
        }
        
        __host__ __device__ void memory_cost(
            ArithEnv &arith,
            bn_t &gas_used,
            const bn_t &length) {
            evm_words_gas_cost(arith, gas_used, length, GAS_MEMORY);
        }
        
        __host__ __device__ void sha256_cost(
            ArithEnv &arith,
            bn_t &gas_used,
            const bn_t &length) {
            evm_words_gas_cost(arith, gas_used, length, GAS_PRECOMPILE_SHA256_WORD);
        }
        
        __host__ __device__ void ripemd160_cost(
            ArithEnv &arith,
            bn_t &gas_used,
            const bn_t &length) {
            evm_words_gas_cost(arith, gas_used, length, GAS_PRECOMPILE_RIPEMD160_WORD);
        }
        
        __host__ __device__ void blake2_cost(
            ArithEnv &arith,
            bn_t &gas_used,
            const uint32_t rounds) {
            // gas_used += GAS_PRECOMPILE_BLAKE2_ROUND * rounds
            bn_t temp;
            cgbn_set_ui32(arith.env, temp, rounds);
            cgbn_mul_ui32(arith.env, temp, temp, GAS_PRECOMPILE_BLAKE2_ROUND);
            cgbn_add(arith.env, gas_used, gas_used, temp);
        }
        
        __host__ __device__ void ecpairing_cost(
            ArithEnv &arith,
            bn_t &gas_used,
            size_t data_size) {
            // gas_used += GAS_PRECOMPILE_ECPAIRING + data_size/192 * GAS_PRECOMPILE_ECPAIRING_PAIR
            cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_PRECOMPILE_ECPAIRING);
            bn_t temp;
            arith.cgbn_from_size_t(temp, data_size);
            cgbn_div_ui32(arith.env, temp, temp, 192);
            cgbn_mul_ui32(arith.env, temp, temp, GAS_PRECOMPILE_ECPAIRING_PAIR);
            cgbn_add(arith.env, gas_used, gas_used, temp);
        }

        __host__ __device__ int32_t access_account_cost(
            ArithEnv &arith,
            bn_t &gas_used,
            const cuEVM::state::AccessState &access_state,
            const bn_t &address) {
            if (access_state.is_warm_account(arith, address)) {
                cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_WARM_ACCESS);
            } else {
                cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_COLD_ACCOUNT_ACCESS);
            }
            return ERROR_SUCCESS;
        }

        __host__ __device__ int32_t sload_cost(
            ArithEnv &arith,
            bn_t &gas_used,
            const cuEVM::state::AccessState &access_state,
            const bn_t &address,
            const bn_t &key) {
            // get the key warm
            if (access_state.is_warm_key(arith, address, key)) {
                cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_WARM_ACCESS);
            } else {
                cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_COLD_SLOAD);
            }

            return ERROR_SUCCESS;

        }
        __host__ __device__ int32_t sstore_cost(
            ArithEnv &arith,
            bn_t &gas_used,
            bn_t &gas_refund,
            const cuEVM::state::TouchState &touch_state,
            const cuEVM::state::AccessState &access_state,
            const bn_t &address,
            const bn_t &key,
            const bn_t &new_value) {
            // get the key warm
            if (access_state.is_warm_key(arith, address, key) == 0) {
                cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_COLD_SLOAD);
            }
            bn_t original_value, current_value;
            access_state.poke_value(arith, address, key, original_value);
            touch_state.poke_value(arith, address, key, current_value);

            // EIP-2200
            if (cgbn_compare(arith.env, new_value, current_value) == 0)
            {
                cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_SLOAD);
            }
            else
            {
                if (cgbn_compare(arith.env, current_value, original_value) == 0)
                {
                    if (cgbn_compare_ui32(arith.env, original_value, 0) == 0)
                    {
                        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_STORAGE_SET);
                    }
                    else
                    {
                        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_SSTORE_RESET);
                        if (cgbn_compare_ui32(arith.env, new_value, 0)==0){
                            cgbn_add_ui32(arith.env, gas_refund, gas_refund, GAS_SSTORE_CLEARS_SCHEDULE);
                        }
                    }
                }
                else
                {
                    cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_SLOAD);
                    if (cgbn_compare_ui32(arith.env, original_value, 0) != 0)
                    {
                        if (cgbn_compare_ui32(arith.env, current_value, 0) == 0)
                        {
                            cgbn_sub_ui32(arith.env, gas_refund, gas_refund, GAS_STORAGE_CLEAR_REFUND);
                        }else if (cgbn_compare_ui32(arith.env, new_value, 0) == 0)
                        {
                            cgbn_add_ui32(arith.env, gas_refund, gas_refund, GAS_STORAGE_CLEAR_REFUND);
                        }
                    }
                    if (cgbn_compare(arith.env, original_value, new_value) == 0)
                    {
                        if (cgbn_compare_ui32(arith.env, original_value, 0) == 0)
                        {
                            cgbn_add_ui32(arith.env, gas_refund, gas_refund, GAS_STORAGE_SET - GAS_SLOAD);
                        }
                        else
                        {
                            cgbn_add_ui32(arith.env, gas_refund, gas_refund, GAS_STORAGE_RESET - GAS_SLOAD);
                        }
                    }
                }
            }
            return ERROR_SUCCESS;
        }

        __host__ __device__ int32_t transaction_intrinsic_gas(
            ArithEnv &arith,
            const cuEVM::transaction::transaction_t &transaction,
            bn_t &gas_intrinsic) {
            
            // gas_intrinsic = GAS_TRANSACTION
            cgbn_set_ui32(arith.env, gas_intrinsic, GAS_TRANSACTION);

            // gas_intrinsic += GAS_TRANSACTION_CREATE if transaction.create
            if (transaction.is_contract_creation(arith)) {
                cgbn_add_ui32(arith.env, gas_intrinsic, gas_intrinsic, GAS_TX_CREATE);
            }

            // gas_intrinsic += GAS_TX_DATA_ZERO/GAS_TX_DATA_NONZERO for each byte in transaction.data
            for (uint32_t idx = 0; idx < transaction.data_init.size; idx++) {
                if (transaction.data_init.data[idx] == 0) {
                    cgbn_add_ui32(arith.env, gas_intrinsic, gas_intrinsic, GAS_TX_DATA_ZERO);
                } else {
                    cgbn_add_ui32(arith.env, gas_intrinsic, gas_intrinsic, GAS_TX_DATA_NONZERO);
                }
            }

            // if transaction type is 1 it might have access list
            if (transaction.type == 1) {
                // gas_intrinsic += GAS_ACCESS_LIST_ADDRESS/GAS_ACCESS_LIST_STORAGE for each address in transaction.access_list
                for (uint32_t idx = 0; idx < transaction.access_list.accounts_count; idx++) {
                    cgbn_add_ui32(arith.env, gas_intrinsic, gas_intrinsic, GAS_ACCESS_LIST_ADDRESS);
                    cgbn_add_ui32(arith.env, gas_intrinsic, gas_intrinsic, GAS_ACCESS_LIST_STORAGE * transaction.access_list.accounts[idx].storage_keys_count);
                }
            }

            #ifdef EIP_3860
            // gas_intrinsic += GAS_INITCODE_COST if create transaction
            if (transaction.is_contract_creation(arith)) {
                bn_t initcode_length;
                cgbn_set_ui32(arith.env, initcode_length, transaction.data_init.size);
                initcode_cost(arith, gas_intrinsic, initcode_length);
            }
            #endif
            return ERROR_SUCCESS;
        }

        __host__ __device__ int32_t memory_grow_cost(
            ArithEnv &arith,
            const cuEVM::memory::evm_memory_t &memory,
            const bn_t &index,
            const bn_t &length,
            bn_t &memory_expansion_cost,
            bn_t &gas_used
        ) {
            do {
                if (cgbn_compare_ui32(arith.env, length, 0) <= 0)
                {
                    return ERROR_SUCCESS;
                }
                bn_t offset;
                uint32_t offset_ui32;
                if (cgbn_add(arith.env, offset, index, length) != 0) {
                    break;
                }
                if (arith.uint32_t_from_cgbn(offset_ui32, offset) != 0) {
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
                // memory_cost = (memory_size_word * memory_size_word) / 512 + 3 * memory_size_word
                bn_t memory_cost;
                if (cgbn_mul(arith.env, memory_cost, memory_size_word, memory_size_word) != 0) {
                    break;
                }
                cgbn_div_ui32(arith.env, memory_cost, memory_cost, 512);
                bn_t tmp;
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
    } // namespace gas_cost
} // namespace cuEVM