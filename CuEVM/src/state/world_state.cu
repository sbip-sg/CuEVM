
#include <CuEVM/state/world_state.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM {
__host__ __device__ int32_t WorldState::get_account(const evm_word_t *address, CuEVM::account_t *&account_ptr) {
    return _state->get_account(address, account_ptr);
}

__host__ __device__ int32_t WorldState::get_value(const evm_word_t *address, const evm_word_t &key, evm_word_t &value) {
    account_t *account_ptr = nullptr;
    uint256_from_uint32(&value, 0);
    return (_state->get_account(address, account_ptr) || account_ptr->get_storage_value(key, value));
}

__host__ __device__ int32_t WorldState::update(const CuEVM::state_access_t *other) {
    return _state->update(other->accounts, other->flags, other->no_accounts);
}

__host__ __device__ void WorldState::serialize_data(serialized_worldstate_data *data) {
    // TODO: reenable
    // data->no_accounts = _state->no_accounts;
    // for (uint32_t idx = 0; idx < _state->no_accounts; idx++) {
    //     account_t *account_ptr = &_state->accounts[idx];
    //     account_ptr->address.address_to_hex(data->addresses[idx]);
    //     account_ptr->balance.to_hex(data->balance[idx]);
    //     data->nonce[idx] = account_ptr->nonce._limbs[0];  // check if limbs 0
    //     if (account_ptr->storage.size > 0) {
    //         for (uint32_t idx_storage = 0; idx_storage < account_ptr->storage.size; idx_storage++) {
    //             account_ptr->storage.storage[idx_storage].key.to_hex(
    //                 data->storage_keys[data->no_storage_elements + idx_storage]);
    //             account_ptr->storage.storage[idx_storage].value.to_hex(
    //                 data->storage_values[data->no_storage_elements + idx_storage]);
    //             data->storage_indexes[data->no_storage_elements + idx_storage] = idx;
    //         }
    //     }
    //     data->no_storage_elements += account_ptr->storage.size;
    // }
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
