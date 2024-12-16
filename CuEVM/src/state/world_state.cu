// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation
// Programme) Author: Stefan-Dan Ciocirlan Data: 2024-06-20
// SPDX-License-Identifier: MIT

#include <CuCrypto/keccak.cuh>
#include <CuEVM/state/world_state.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM {
__host__ __device__ int32_t WorldState::get_account(ArithEnv &arith, const evm_word_t *address,
                                                    CuEVM::account_t *&account_ptr) {
    return _state->get_account(arith, address, account_ptr);
}

__host__ __device__ int32_t WorldState::get_value(ArithEnv &arith, const evm_word_t *address, const bn_t &key,
                                                  bn_t &value) {
    account_t *account_ptr = nullptr;
    cgbn_set_ui32(arith.env, value, 0);
    return (_state->get_account(arith, address, account_ptr) || account_ptr->get_storage_value(arith, key, value));
}

__host__ __device__ int32_t WorldState::update(ArithEnv &arith, const CuEVM::state_access_t *other) {
    return _state->update(arith, other->accounts, other->flags, other->no_accounts);
}

__host__ __device__ void WorldState::serialize_data(ArithEnv &arith, serialized_worldstate_data *data) {
    data->no_accounts = _state->no_accounts;
    char *code_hash_hex_string_ptr = nullptr;
    char *code_hex_string_ptr = nullptr;
    CuEVM::byte_array_t *hash;

    for (uint32_t idx = 0; idx < _state->no_accounts; idx++) {
        account_t *account_ptr = &_state->accounts[idx];
        account_ptr->address.address_to_hex(data->addresses[idx]);
        account_ptr->balance.to_hex(data->balance[idx]);
        data->nonce[idx] = account_ptr->nonce._limbs[0];  // check if limbs 0

        hash = new CuEVM::byte_array_t(CuEVM::hash_size);
        CuCrypto::keccak::sha3(account_ptr->byte_code.data, account_ptr->byte_code.size, hash->data, hash->size);
        code_hash_hex_string_ptr = hash->to_hex();

        for (auto i=0; i<67; i++){
          data->code_hash[idx][i] = code_hash_hex_string_ptr[i];
        }

        if (account_ptr->storage.size > 0) {
            for (uint32_t idx_storage = 0; idx_storage < account_ptr->storage.size; idx_storage++) {
                account_ptr->storage.storage[idx_storage].key.to_hex(
                    data->storage_keys[data->no_storage_elements + idx_storage]);
                account_ptr->storage.storage[idx_storage].value.to_hex(
                    data->storage_values[data->no_storage_elements + idx_storage]);
                data->storage_indexes[data->no_storage_elements + idx_storage] = idx;
            }
        }

        data->no_storage_elements += account_ptr->storage.size;
    }

    delete hash;
    delete code_hash_hex_string_ptr;
    delete code_hex_string_ptr;
}

  __host__ void serialized_worldstate_data::print_json() {
    printf("{\"accounts\":[");
    for (uint32_t idx = 0; idx < no_accounts; idx++) {
        printf("{\"address\": \"%s\", ", addresses[idx]);
        printf("\"balance\": \"%s\", ", balance[idx]);
        printf("\"nonce\": %d, ", nonce[idx]);
        printf("\"codeHash\": \"%s\", ", code_hash[idx]);
        printf("\"storage\":{");
        for (auto j = 0; j < no_storage_elements; j++){
          auto account_idx = storage_indexes[j];
          if (account_idx > idx) break;
          if (account_idx == idx){
            printf("\"%s\": \"%s\" ", storage_keys[j], storage_values[j]);
            if (j + 1 < no_storage_elements && storage_indexes[j+1]==idx){
              printf(",");
            }
          }
        }
        printf("}}");
        if (idx + 1 < no_accounts) {
            printf(",");
        }
    }
    printf("]}\n");

}


__host__ void serialized_worldstate_data::print() {
    printf("\nPrinting serialized worldstate data\n");
    printf("no_accounts: %d\n", no_accounts);
    printf("no_storage_elements: %d\n", no_storage_elements);
    for (uint32_t idx = 0; idx < no_accounts; idx++) {
        printf("address: %s\n", addresses[idx]);
        printf("balance: %s\n", balance[idx]);
        printf("nonce: %d\n", nonce[idx]);
    }
    for (uint32_t idx = 0; idx < no_storage_elements; idx++) {
        printf("storage_key: %s\n", storage_keys[idx]);
        printf("storage_value: %s\n", storage_values[idx]);
        printf("storage_index: %d\n", storage_indexes[idx]);
    }
}
}  // namespace CuEVM
