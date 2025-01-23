#include <CuEVM/state/state_db.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM {

__device__ int32_t SnapshotAccount::find_storage_key(const evm_word_t *key) const {
    for (uint32_t i = 0; i < storage_size; i++) {
        if (storage_keys[i] == *key) {
            return i;
        }
    }
    return -1;
}
__device__ void SnapshotAccount::grow_storage(const evm_word_t *key) {
    // printf("before grow storage thread %d account %p, storage size %d storage key %p storage values %p\n",
    //        INSTANCE_GLOBAL_IDX, this, storage_size, storage_keys, storage_values);
    evm_word_t *tmp_storage_keys = new evm_word_t[storage_size + 1];
    SnapshotValue *tmp_storage_values = new SnapshotValue[storage_size + 1];
    if (storage_size > 0) {
        memcpy(tmp_storage_keys, storage_keys, storage_size * sizeof(evm_word_t));
        memcpy(tmp_storage_values, storage_values, storage_size * sizeof(SnapshotValue));
    }
    // printf("after grow storage thread %d account %p, storage size %d storage key %p storage values %p\n",
    //        INSTANCE_GLOBAL_IDX, this, storage_size, storage_keys, storage_values);
    if (storage_keys != nullptr) {
        delete[] storage_keys;
    }
    if (storage_values != nullptr) {
        delete[] storage_values;
    }
    storage_keys = tmp_storage_keys;
    storage_values = tmp_storage_values;
    storage_keys[storage_size] = *key;
    storage_size++;
}

__host__ __device__ Snapshot::~Snapshot() {
    if (address_index_list != nullptr) {
        delete[] address_index_list;
    }
    if (accounts != nullptr) {
        delete[] accounts;
    }
}
__device__ int32_t Snapshot::get_snapshot_index(const uint32_t address_index) const {
    for (uint32_t i = 0; i < num_accounts; i++) {
        if (address_index_list[i] == address_index) {
            return i;
        }
    }
    return -1;
}

__device__ void Snapshot::grow_account(const uint32_t address_index, const evm_word_t *balance, const uint32_t nonce) {
    // printf("grow account thread %d, num_accounts %d\n", INSTANCE_GLOBAL_IDX, num_accounts);
    uint32_t *tmp_address_index_list = new uint32_t[num_accounts + 1];
    SnapshotAccount *tmp_accounts = new SnapshotAccount[num_accounts + 1];
    if (num_accounts > 0) {
        memcpy(tmp_address_index_list, address_index_list, num_accounts * sizeof(uint32_t));
        memcpy(tmp_accounts, accounts, num_accounts * sizeof(SnapshotAccount));
    }
    // printf("grow account thread %d, num_accounts %d, tmp address %p address %p\n", INSTANCE_GLOBAL_IDX, num_accounts,
    //        &tmp_address_index_list[num_accounts], address_index_list);
    // tmp_address_list[num_accounts] = *address;
    tmp_address_index_list[num_accounts] = address_index;
    if (balance != nullptr) tmp_accounts[num_accounts].balance = *balance;
    tmp_accounts[num_accounts].nonce = nonce;
    num_accounts++;
    if (address_index_list != nullptr) {
        delete[] address_index_list;
    }
    if (accounts != nullptr) {
        delete[] accounts;
    }
    address_index_list = tmp_address_index_list;
    accounts = tmp_accounts;
}

__device__ void Snapshot::set_account(const uint32_t address_index, const evm_word_t *balance, const uint32_t nonce) {
    // only snapshot first time
    if (get_snapshot_index(address_index) == -1) {
        grow_account(address_index, balance, nonce);
    }
}

__device__ void Snapshot::set_code(const uint32_t address_index, const uint32_t code_size, const uint8_t *code) {
    // TODO: implement
    // int32_t address_index = get_address_index(address);
    // if (address_index == -1) {
    //     grow_account(address, nullptr, 0);
    //     address_index = num_accounts - 1;
    // }
    // accounts[address_index].code_size = code_size;
    // accounts[address_index].code = new uint8_t[code_size];
    // memcpy(accounts[address_index].code, code, code_size);
    // accounts[address_index].code_modified = true;
}

__device__ void Snapshot::set_storage(const uint32_t address_index, const evm_word_t *key, const ValueStatus *value) {
    int32_t snapshot_index = get_snapshot_index(address_index);
    if (snapshot_index == -1) {
        grow_account(address_index, nullptr, 0);
        snapshot_index = num_accounts - 1;
    }

    // printf("find storage key thread %d, key %p accounts %p address_index %d\n", INSTANCE_GLOBAL_IDX, key, accounts,
    //        address_index);
    int32_t key_offset = accounts[snapshot_index].find_storage_key(key);
    if (key_offset == -1) {
        // printf("key not found, grow storage thread %d account %p\n", INSTANCE_GLOBAL_IDX, &accounts[snapshot_index]);
        accounts[snapshot_index].grow_storage(key);
        key_offset = accounts[snapshot_index].storage_size - 1;
        accounts[snapshot_index].storage_values[key_offset].value = value->value;
        accounts[snapshot_index].storage_values[key_offset].is_warm = value->is_warm;
        // printf("key offset %d storage size %d\n", key_offset, accounts[address_index].storage_size);
    }
}

__device__ SnapshotValue *Snapshot::get_storage(const uint32_t address_index, const evm_word_t *key) const {
    int32_t snapshot_index = get_snapshot_index(address_index);
    if (snapshot_index == -1) {
        return nullptr;
    }
    int32_t key_offset = accounts[snapshot_index].find_storage_key(key);
    if (key_offset == -1) {
        return nullptr;
    }
    return &accounts[snapshot_index].storage_values[key_offset];
}

__host__ __device__ StateDb::StateDb(uint32_t num_states) {
    num_states = num_states;
    num_accounts = 0;
    num_storage_elements = 0;
    storage_capacity = 0;
    all_account_codes = nullptr;
    address_list = nullptr;
    account_balances = nullptr;
    account_nonces = nullptr;
    account_storage_size = nullptr;
    account_codes_size = nullptr;
    account_codes_offset = nullptr;
    all_keys = nullptr;
    keys_list_offset = nullptr;
    prealloc_values_pool = nullptr;  // offset works for this
    dynamic_keys_pool = nullptr;
    dynamic_values_pool = nullptr;
    dynamic_pool_capacity = nullptr;
}
__device__ StateDb::StateDb(uint32_t num_states, uint32_t num_accounts, uint32_t num_storage_elements,
                            uint32_t storage_capacity, uint8_t *all_account_codes, evm_word_t *address_list,
                            evm_word_t *account_balances, uint32_t *account_nonces, uint32_t *account_storage_size,
                            uint32_t *account_codes_size, uint32_t *account_codes_offset, KeyOffset *all_keys,
                            uint32_t *keys_list_offset, uint32_t *keys_list_size, ValueStatus *prealloc_values_pool,
                            evm_word_t **dynamic_keys_pool, ValueStatus **dynamic_values_pool,
                            uint32_t *dynamic_pool_size, Snapshot *original_snapshot)
    : num_states(num_states),
      num_accounts(num_accounts),
      num_storage_elements(num_storage_elements),
      storage_capacity(storage_capacity),
      all_account_codes(all_account_codes),
      address_list(address_list),
      account_balances(account_balances),
      account_nonces(account_nonces),
      account_storage_size(account_storage_size),
      account_codes_size(account_codes_size),
      account_codes_offset(account_codes_offset),
      all_keys(all_keys),
      keys_list_offset(keys_list_offset),
      prealloc_values_pool(prealloc_values_pool),
      dynamic_keys_pool(dynamic_keys_pool),
      dynamic_values_pool(dynamic_values_pool),
      dynamic_pool_capacity(dynamic_pool_capacity),
      original_snapshot(original_snapshot) {}

__device__ int32_t StateDb::get_address_index(const evm_word_t *address) const {
    // printf("get address index thread %d, num_accounts %d\n", INSTANCE_GLOBAL_IDX, num_accounts);
    // Todo: global list -> parallel search
    for (uint32_t i = 0; i < num_accounts; i++) {
        if (address_list[i] == *address) {
            return i;
        }
    }
    return -1;
}

__device__ int32_t StateDb::get_value_offset(int32_t address_index, const evm_word_t *key) const {
    if (address_index == -1) {
        return -1;
    }
    // printf("get value offset thread %d, address_index %d\n", INSTANCE_GLOBAL_IDX, address_index);
    uint32_t key_list_idx = keys_list_offset[address_index];
    // printf("search for key\n");
    // key->print();
    // printf("account storage size %d \n", account_storage_size[address_index]);
    // printf("account prealloc keys size %d \n", account_prealloc_keys_size);
    uint32_t loop_iter =
        min(account_storage_size[address_index * num_states + INSTANCE_GLOBAL_IDX], account_prealloc_keys_size);
    for (uint32_t i = 0; i < loop_iter; i++) {
        if (all_keys[key_list_idx + i].key == *key) {
            return all_keys[key_list_idx + i].offset;
        }
    }
    return -1;
}

__device__ void StateDb::new_account(const evm_word_t *address, const evm_word_t *balance, const uint32_t nonce) {
    address_list[num_accounts] = *address;
    account_balances[num_accounts * num_states + INSTANCE_GLOBAL_IDX] = *balance;
    account_nonces[num_accounts * num_states + INSTANCE_GLOBAL_IDX] = nonce;
    num_accounts++;
}

__device__ void StateDb::update_account(const evm_word_t *address, const evm_word_t *balance, const uint32_t nonce) {
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        return;
    }
    uint32_t instance_idx = address_index * num_states + INSTANCE_GLOBAL_IDX;
    account_balances[instance_idx] = *balance;
    account_nonces[instance_idx] = nonce;
}

__device__ void StateDb::update_balance(const evm_word_t *address, const evm_word_t *balance) {
    // printf("update balance thread %d, address %p\n", INSTANCE_GLOBAL_IDX, address);
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        return;
    }
    account_balances[address_index * num_states + INSTANCE_GLOBAL_IDX] = *balance;
}

__device__ void StateDb::update_nonce(const evm_word_t *address, const uint32_t nonce) {
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        return;
    }
    account_nonces[address_index * num_states + INSTANCE_GLOBAL_IDX] = nonce;
}

__device__ void StateDb::update_code(const evm_word_t *address, const byte_array_t *code) {
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        return;
    }
    // todo : implement
}

__device__ int32_t StateDb::transfer(const evm_word_t *sender, const evm_word_t *recipient, const evm_word_t *value) {
    // todo : implement
    return ERROR_SUCCESS;
}

__device__ int32_t StateDb::get_dynamic_value_offset(int32_t instance_idx, const evm_word_t *key) const {
    // printf("get dynamic value offset thread %d, instance_idx %d\n", INSTANCE_GLOBAL_IDX, instance_idx);
    if (dynamic_keys_pool[instance_idx] == nullptr) {
        // printf("dynamic keys pool is nullptr\n");
        return -1;
    }
    uint32_t loop_iter = account_storage_size[instance_idx] - account_prealloc_keys_size;
    evm_word_t *keys_pool = dynamic_keys_pool[instance_idx];
    for (uint32_t i = 0; i < loop_iter; i++) {
        if (keys_pool[i] == *key) {
            return i;
        }
    }
    return -1;
}
__device__ void StateDb::write_storage_with_known_index(const evm_word_t *address, const evm_word_t *key,
                                                        const evm_word_t *value, uint32_t address_index,
                                                        ValueStatus *found_value, Snapshot *snapshot, bool is_warm) {
    if (address_index == -1) {
        // printf("address not found\n");
        return;  // should never write directly to storage before creating account
    }
    uint32_t instance_idx = address_index * num_states + INSTANCE_GLOBAL_IDX;

    if (found_value == nullptr) {
        if (account_storage_size[instance_idx] < account_prealloc_keys_size) {
            uint32_t key_list_start = keys_list_offset[address_index];

            uint32_t key_list_size = account_storage_size[instance_idx];
            KeyOffset *new_key_offset = &all_keys[key_list_start + key_list_size];
            uint32_t key_offset = num_storage_elements * num_states;
            new_key_offset->key = *key;
            new_key_offset->offset = key_offset;
            prealloc_values_pool[key_offset + INSTANCE_GLOBAL_IDX].set_value(value, is_warm);

            account_storage_size[instance_idx]++;
            // num_storage_elements++;
        } else {
            uint32_t key_offset = get_dynamic_value_offset(instance_idx, key);
            if (key_offset == -1) {
                uint32_t storage_size = account_storage_size[instance_idx];
                grow_storage(instance_idx);
                dynamic_keys_pool[instance_idx][storage_size - account_prealloc_keys_size] = *key;
                dynamic_values_pool[instance_idx][storage_size - account_prealloc_keys_size].set_value(value, is_warm);
                account_storage_size[instance_idx]++;
                // num_storage_elements++;
            }
        }

    } else {
        // write to exsiting slot
        if (snapshot != nullptr) {
            snapshot->set_storage(address_index, key, found_value);
        }
        found_value->set_value(value, is_warm);
    }
    account_is_warm[instance_idx] = true;
}

__device__ void StateDb::write_storage(const evm_word_t *address, const evm_word_t *key, const evm_word_t *value,
                                       Snapshot *snapshot, bool is_warm) {
    // printf("Write storage \n");
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        // printf("address not found\n");
        return;  // should never write directly to storage before creating account
    }
    uint32_t instance_idx = address_index * num_states + INSTANCE_GLOBAL_IDX;
    // find in prealloc_values_pool
    int32_t key_offset = get_value_offset(address_index, key);
    if (key_offset == -1) {
        // printf("key not found\n");
        if (account_storage_size[instance_idx] < account_prealloc_keys_size) {
            // printf(" not found in prealloc_values_pool, account storage size: %d\n",
            //        account_storage_size[address_index]);
            // create new key val pair on prealloc_values_pool
            uint32_t key_list_start = keys_list_offset[address_index];

            uint32_t key_list_size = account_storage_size[instance_idx];
            KeyOffset *new_key_offset = &all_keys[key_list_start + key_list_size];
            key_offset = num_storage_elements * num_states;
            new_key_offset->key = *key;
            new_key_offset->offset = key_offset;
            prealloc_values_pool[key_offset + INSTANCE_GLOBAL_IDX].set_value(value, is_warm);

            account_storage_size[instance_idx]++;
            // num_storage_elements++;
        } else {
            // printf("THREAD %d not found in prealloc_values_pool, create dynamic storage , account storage size:
            // %d\n",
            //        INSTANCE_GLOBAL_IDX, account_storage_size[instance_idx]);
            // find in dynamic_keys_pool
            key_offset = get_dynamic_value_offset(instance_idx, key);
            if (key_offset == -1) {
                uint32_t storage_size = account_storage_size[instance_idx];
                grow_storage(instance_idx);
                dynamic_keys_pool[instance_idx][storage_size - account_prealloc_keys_size] = *key;
                dynamic_values_pool[instance_idx][storage_size - account_prealloc_keys_size].set_value(value, is_warm);

                account_storage_size[instance_idx]++;
                // num_storage_elements++;
            } else {
                // write to exsiting slot
                if (snapshot != nullptr) {
                    snapshot->set_storage(address_index, key, &dynamic_values_pool[instance_idx][key_offset]);
                }

                dynamic_values_pool[instance_idx][key_offset].set_value(value, is_warm);
            }
        }

    } else {
        // printf("found in prealloc_values_pool\n");
        if (snapshot != nullptr) {
            // printf("set snapshot\n");
            snapshot->set_storage(address_index, key, &prealloc_values_pool[key_offset + INSTANCE_GLOBAL_IDX]);
        }
        prealloc_values_pool[key_offset + INSTANCE_GLOBAL_IDX].set_value(value, is_warm);
    }

    account_is_warm[instance_idx] = true;

    // todo : write snapshot for potential future revert
}
__device__ void StateDb::snapshot_account(const uint32_t address_index, const evm_word_t *balance, const uint32_t nonce,
                                          Snapshot *snapshot) {
    if (snapshot == nullptr) snapshot = original_snapshot;
    snapshot->set_account(address_index, balance, nonce);
}
__device__ void StateDb::snapshot_storage(const uint32_t address_index, evm_word_t *key, ValueStatus *value,
                                          Snapshot *snapshot) {
    if (snapshot == nullptr) snapshot = original_snapshot;
    snapshot->set_storage(address_index, key, value);
    // todo : implement
}
__device__ void StateDb::snapshot_code(const uint32_t address_index, Snapshot *snapshot) {
    // todo : implement
}
__device__ void StateDb::revert_to_snapshot(Snapshot *snapshot) {
    if (snapshot == nullptr) snapshot = original_snapshot;
    for (uint32_t i = 0; i < snapshot->num_accounts; i++) {
        int32_t address_index = snapshot->address_index_list[i];
        if (address_index == -1) {
            continue;
        }
        uint32_t instance_idx = address_index * num_states + INSTANCE_GLOBAL_IDX;
        account_balances[instance_idx] = snapshot->accounts[i].balance;
        account_nonces[instance_idx] = snapshot->accounts[i].nonce;
        for (uint32_t j = 0; j < snapshot->accounts[i].storage_size; j++) {
            write_storage(&address_list[address_index], &snapshot->accounts[i].storage_keys[j],
                          &snapshot->accounts[i].storage_values[j].value, nullptr,
                          snapshot->accounts[i].storage_values[j].is_warm);
        }
    }
}
__device__ evm_word_t *StateDb::get_storage_with_known_index(const evm_word_t *address, const evm_word_t *key,
                                                             uint32_t address_index, ValueStatus *found_value,
                                                             bool set_warm) {
    // printf("get_storage_with_known_index thread %d, address %p\n", INSTANCE_GLOBAL_IDX, address);

    uint32_t instance_idx = address_index * num_states + INSTANCE_GLOBAL_IDX;
    if (found_value == nullptr) {
        evm_word_t zero = 0;
        uint32_t storage_size = account_storage_size[instance_idx];
        if (set_warm) {
            if (storage_size < account_prealloc_keys_size) {
                uint32_t key_list_start = keys_list_offset[address_index];

                KeyOffset *new_key_offset = &all_keys[key_list_start + storage_size];
                uint32_t key_offset = num_storage_elements * num_states;
                new_key_offset->key = *key;
                new_key_offset->offset = key_offset;
                prealloc_values_pool[key_offset + INSTANCE_GLOBAL_IDX].set_value(&zero, true);

            } else {
                grow_storage(instance_idx);
                dynamic_keys_pool[instance_idx][storage_size - account_prealloc_keys_size] = *key;
                dynamic_values_pool[instance_idx][storage_size - account_prealloc_keys_size].set_value(&zero, true);
            }
            account_storage_size[instance_idx]++;
        }
        return nullptr;
    } else {
        // printf("found in dynamic storage\n");
        found_value->is_warm = set_warm;
        account_is_warm[instance_idx] |= set_warm;
        return &found_value->value;
    }
}

__device__ evm_word_t *StateDb::get_storage(const evm_word_t *address, const evm_word_t *key, bool set_warm) {
    // printf("get storage thread %d, address %p\n", INSTANCE_GLOBAL_IDX, address);
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        // todo grow account
        return nullptr;
    }
    uint32_t instance_idx = address_index * num_states + INSTANCE_GLOBAL_IDX;
    int32_t key_offset = get_value_offset(address_index, key);
    if (key_offset == -1) {
        key_offset = get_dynamic_value_offset(instance_idx, key);
        if (key_offset == -1) {
            evm_word_t zero = 0;
            uint32_t storage_size = account_storage_size[instance_idx];
            if (set_warm) {
                if (storage_size < account_prealloc_keys_size) {
                    uint32_t key_list_start = keys_list_offset[address_index];

                    KeyOffset *new_key_offset = &all_keys[key_list_start + storage_size];
                    key_offset = num_storage_elements * num_states;
                    new_key_offset->key = *key;
                    new_key_offset->offset = key_offset;
                    prealloc_values_pool[key_offset + INSTANCE_GLOBAL_IDX].set_value(&zero, true);

                } else {
                    grow_storage(instance_idx);
                    dynamic_keys_pool[instance_idx][storage_size - account_prealloc_keys_size] = *key;
                    dynamic_values_pool[instance_idx][storage_size - account_prealloc_keys_size].set_value(&zero, true);
                }
                account_storage_size[instance_idx]++;
            }
            return nullptr;
        } else {
            // printf("found in dynamic storage\n");
            dynamic_values_pool[instance_idx][key_offset].is_warm = set_warm;
            account_is_warm[instance_idx] |= set_warm;
            return &dynamic_values_pool[instance_idx][key_offset].value;
        }

    } else {
        // printf("found in prealloc_values_pool\n");
        prealloc_values_pool[key_offset + INSTANCE_GLOBAL_IDX].is_warm = set_warm;
        account_is_warm[instance_idx] |= set_warm;
        return &prealloc_values_pool[key_offset + INSTANCE_GLOBAL_IDX].value;
    }
}

__device__ ValueStatus *StateDb::get_value_status(const evm_word_t *address, const evm_word_t *key) const {
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        return nullptr;
    }
    uint32_t instance_idx = address_index * num_states + INSTANCE_GLOBAL_IDX;
    int32_t key_offset = get_value_offset(address_index, key);
    if (key_offset == -1) {
        // printf(" not found in prealloc_values_pool, search dynamic storage\n");
        // find in dynamic_keys_pool
        key_offset = get_dynamic_value_offset(instance_idx, key);
        if (key_offset == -1) {
            // printf("not found at all\n");
            return nullptr;
        } else {
            // printf("found in dynamic storage\n");
            return &dynamic_values_pool[instance_idx][key_offset];
        }
    } else {
        // printf("found in prealloc_values_pool\n");
        return &prealloc_values_pool[key_offset + INSTANCE_GLOBAL_IDX];
    }
}
__device__ ValueStatus *StateDb::get_value_status(const uint32_t address_index, const evm_word_t *key) const {
    uint32_t instance_idx = address_index * num_states + INSTANCE_GLOBAL_IDX;
    int32_t key_offset = get_value_offset(address_index, key);
    if (key_offset == -1) {
        // printf(" not found in prealloc_values_pool, search dynamic storage\n");
        // find in dynamic_keys_pool
        key_offset = get_dynamic_value_offset(instance_idx, key);
        if (key_offset == -1) {
            // printf("not found at all\n");
            return nullptr;
        } else {
            // printf("found in dynamic storage\n");
            return &dynamic_values_pool[instance_idx][key_offset];
        }
    } else {
        // printf("found in prealloc_values_pool\n");
        return &prealloc_values_pool[key_offset + INSTANCE_GLOBAL_IDX];
    }
}

__device__ void StateDb::grow_storage(int32_t instance_idx) {
    // printf("grow storage thread %d, instance_idx %d\n", INSTANCE_GLOBAL_IDX, instance_idx);
    // grow individual thread storage.
    uint32_t pool_capacity = dynamic_pool_capacity[instance_idx];
    uint32_t storage_size = account_storage_size[instance_idx];
    // printf("storage size: %d\n", storage_size);
    // printf("pool capacity: %d\n", pool_capacity);
    if (storage_size >= account_prealloc_keys_size) {
        // printf("grow storage\n");
        if (pool_capacity == 0) {
            dynamic_pool_capacity[instance_idx] = dynamic_pool_base_size;
            dynamic_keys_pool[instance_idx] = new evm_word_t[dynamic_pool_base_size];
            dynamic_values_pool[instance_idx] = new ValueStatus[dynamic_pool_base_size];
            new ValueStatus[dynamic_pool_base_size];
            // printf("new dynamic pool capacity: %d\n", dynamic_pool_capacity[address_index]);
            // printf("new dynamic keys pool: %p\n", dynamic_keys_pool[address_index]);
            // printf("new dynamic values pool: %p\n", dynamic_values_pool[address_index]);
        } else if (storage_size == pool_capacity - account_prealloc_keys_size) {
            uint32_t new_pool_capacity = pool_capacity * 2;
            evm_word_t *new_keys_pool = new evm_word_t[new_pool_capacity];
            ValueStatus *new_values_pool = new ValueStatus[new_pool_capacity];

            // Copy existing data to the new pools
            memcpy(new_keys_pool, dynamic_keys_pool[instance_idx], pool_capacity * sizeof(evm_word_t));
            memcpy(new_values_pool, dynamic_values_pool[instance_idx], pool_capacity * sizeof(ValueStatus));

            // Free old pools
            delete[] dynamic_keys_pool[instance_idx];
            delete[] dynamic_values_pool[instance_idx];

            // Update pointers and size
            dynamic_keys_pool[instance_idx] = new_keys_pool;
            dynamic_values_pool[instance_idx] = new_values_pool;
            dynamic_pool_capacity[instance_idx] = new_pool_capacity;
        }
    }
}

__device__ evm_word_t *StateDb::get_balance(const evm_word_t *address) {
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        return nullptr;
    }

    return &account_balances[address_index * num_states + INSTANCE_GLOBAL_IDX];
}

__device__ uint32_t StateDb::get_nonce(const evm_word_t *address) {
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        return 0;
    }
    return account_nonces[address_index * num_states + INSTANCE_GLOBAL_IDX];
}

__device__ uint8_t *StateDb::get_code(uint32_t &code_size, const evm_word_t *address) {
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        return nullptr;
    }

    code_size = account_codes_size[address_index];
    return &all_account_codes[account_codes_offset[address_index]];
}

__device__ void StateDb::set_warm_account(const evm_word_t *address) {
    // todo :implement
}

__device__ void StateDb::set_warm_key(const evm_word_t *address, const evm_word_t *key) {
    // todo :implement
}

__device__ bool StateDb::is_warm_account(const evm_word_t *address) const {
    // todo :implement
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        return false;
    }
    return account_is_warm[address_index * num_states + INSTANCE_GLOBAL_IDX];
}

__device__ bool StateDb::is_warm_key(const evm_word_t *address, const evm_word_t *key) const {
    if (!is_warm_account(address)) {
        return false;
    }
    ValueStatus *value_status = get_value_status(address, key);
    if (value_status == nullptr) {
        return false;
    }
    return value_status->is_warm;
}

__device__ bool StateDb::is_warm_key_with_offset(const evm_word_t *address, const evm_word_t *key,
                                                 uint32_t &address_index, ValueStatus *&found_value) {
    address_index = get_address_index(address);
    if (address_index == -1) {
        found_value = nullptr;
        return false;
    }
    found_value = get_value_status(address_index, key);
    if (found_value == nullptr) {
        return false;
    }
    return found_value->is_warm;
}

__device__ bool StateDb::is_empty_account(const evm_word_t *address) const {
    // todo :implement
    return false;
}

__device__ bool StateDb::is_deleted_account(const evm_word_t *address) const {
    // todo :implement
    return false;
}

__device__ bool StateDb::is_empty_create(const evm_word_t *address) const {
    // todo :implement
    return false;
}
__device__ bool StateDb::is_contract(const evm_word_t *address) const {
    // todo :implement
    return true;
}
__host__ void StateDb::GPUfromJson(StateDb *&state_db, const cJSON *state_json, uint32_t num_states) {
    StateDb *state_db_cpu = new StateDb(num_states);
    StateDb::CPUfromJson(state_db_cpu, state_json, num_states);
    StateDb *tmp_state_db = new StateDb(num_states);
    tmp_state_db->num_accounts = state_db_cpu->num_accounts;
    tmp_state_db->num_states = state_db_cpu->num_states;
    tmp_state_db->num_storage_elements = state_db_cpu->num_storage_elements;
    tmp_state_db->storage_capacity = state_db_cpu->storage_capacity;

    uint32_t num_accounts = state_db_cpu->num_accounts;
    uint32_t code_size =
        state_db_cpu->account_codes_size[num_accounts - 1] + state_db_cpu->account_codes_offset[num_accounts - 1];

    // Grouped memory allocation
    CUDA_CHECK(cudaMalloc(&tmp_state_db->address_list, num_accounts * sizeof(evm_word_t)));

    CUDA_CHECK(cudaMalloc(&tmp_state_db->account_balances, num_states * num_accounts * sizeof(evm_word_t)));
    CUDA_CHECK(cudaMalloc(&tmp_state_db->account_nonces, num_states * num_accounts * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&tmp_state_db->account_storage_size, num_states * num_accounts * sizeof(uint32_t)));

    CUDA_CHECK(cudaMalloc(&tmp_state_db->account_codes_size, num_accounts * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&tmp_state_db->account_codes_offset, num_accounts * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&tmp_state_db->all_keys, account_prealloc_keys_size * num_accounts * sizeof(KeyOffset)));
    CUDA_CHECK(cudaMalloc(&tmp_state_db->keys_list_offset, num_accounts * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&tmp_state_db->prealloc_values_pool,
                          state_db_cpu->storage_capacity * num_states * sizeof(evm_word_t)));
    CUDA_CHECK(cudaMalloc(&tmp_state_db->all_account_codes, code_size * sizeof(uint8_t)));

    CUDA_CHECK(cudaMalloc(&tmp_state_db->dynamic_keys_pool, num_states * num_accounts * sizeof(evm_word_t *)));
    CUDA_CHECK(cudaMalloc(&tmp_state_db->dynamic_values_pool, num_states * num_accounts * sizeof(ValueStatus *)));
    CUDA_CHECK(cudaMalloc(&tmp_state_db->dynamic_pool_capacity, num_states * num_accounts * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&tmp_state_db->account_is_warm, num_states * num_accounts * sizeof(bool)));

    CUDA_CHECK(cudaMalloc(&tmp_state_db->original_snapshot, num_states * sizeof(Snapshot)));
    // Grouped memory copy
    CUDA_CHECK(cudaMemcpy(tmp_state_db->address_list, state_db_cpu->address_list, num_accounts * sizeof(evm_word_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_state_db->account_balances, state_db_cpu->account_balances,
                          num_states * num_accounts * sizeof(evm_word_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_state_db->account_nonces, state_db_cpu->account_nonces,
                          num_states * num_accounts * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_state_db->account_storage_size, state_db_cpu->account_storage_size,
                          num_states * num_accounts * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_state_db->account_codes_size, state_db_cpu->account_codes_size,
                          num_accounts * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_state_db->account_codes_offset, state_db_cpu->account_codes_offset,
                          num_accounts * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_state_db->all_keys, state_db_cpu->all_keys,
                          account_prealloc_keys_size * num_accounts * sizeof(KeyOffset), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_state_db->keys_list_offset, state_db_cpu->keys_list_offset,
                          num_accounts * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_state_db->prealloc_values_pool, state_db_cpu->prealloc_values_pool,
                          state_db_cpu->storage_capacity * num_states * sizeof(evm_word_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_state_db->all_account_codes, state_db_cpu->all_account_codes, code_size * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_state_db->original_snapshot, state_db_cpu->original_snapshot,
                          num_states * sizeof(Snapshot), cudaMemcpyHostToDevice));
    // dynamic storage
    CUDA_CHECK(cudaMemcpy(tmp_state_db->dynamic_pool_capacity, state_db_cpu->dynamic_pool_capacity,
                          num_states * num_accounts * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_state_db->account_is_warm, state_db_cpu->account_is_warm,
                          num_states * num_accounts * sizeof(bool), cudaMemcpyHostToDevice));

    // printf("state db cpu\n");
    // state_db_cpu->print();

    StateDb *state_db_gpu;
    CUDA_CHECK(cudaMalloc(&state_db_gpu, sizeof(StateDb)));
    CUDA_CHECK(cudaMemcpy(state_db_gpu, tmp_state_db, sizeof(StateDb), cudaMemcpyHostToDevice));
    cudaMemcpyToSymbol(global_state_db_ptr, &state_db_gpu, sizeof(StateDb *));
    delete state_db_cpu;
    delete tmp_state_db;
    state_db = state_db_gpu;
}
// Return a pointer to the StateDb object on device memory
__host__ void StateDb::CPUfromJson(StateDb *&state_db, const cJSON *state_json, uint32_t num_states) {
    // if (!cJSON_IsArray(state_json)) return 0;
    uint32_t num_accounts = cJSON_GetArraySize(state_json);
    // if (num_accounts == 0)
    //     ;
    printf("num_accounts: %d\n", num_accounts);

    // StateDb *state_db = new StateDb(num_states);
    state_db->num_accounts = num_accounts;
    state_db->num_states = num_states;
    state_db->num_storage_elements = 0;
    state_db->storage_capacity = account_prealloc_keys_size * num_accounts;
    state_db->all_account_codes = nullptr;
    state_db->address_list = new evm_word_t[num_accounts];

    state_db->account_nonces = new uint32_t[num_accounts * num_states];
    state_db->account_storage_size = new uint32_t[num_accounts * num_states];
    state_db->account_balances = new evm_word_t[num_accounts * num_states];

    state_db->account_codes_size = new uint32_t[num_accounts];
    state_db->account_codes_offset = new uint32_t[num_accounts];
    state_db->all_keys = new KeyOffset[account_prealloc_keys_size * num_accounts];
    state_db->keys_list_offset = new uint32_t[num_accounts];
    state_db->prealloc_values_pool = new ValueStatus[state_db->storage_capacity * num_states];
    state_db->account_is_warm = new bool[num_states * num_accounts];
    // dynamic storage to grow later
    state_db->dynamic_keys_pool = new evm_word_t *[num_states * num_accounts];
    state_db->dynamic_values_pool = new ValueStatus *[num_states * num_accounts];
    state_db->dynamic_pool_capacity = new uint32_t[num_states * num_accounts];
    state_db->original_snapshot = new Snapshot[num_states];
    uint32_t idx = 0;
    cJSON *account_json;
    uint32_t bytecode_offset = 0;
    uint32_t key_offset_wo_states = 0;  // offset without multiplying by num_states
    uint32_t current_code_size = 0;
    memset(state_db->account_is_warm, 0, num_states * num_accounts * sizeof(bool));
    memset(state_db->dynamic_pool_capacity, 0, num_states * num_accounts * sizeof(uint32_t));
    memset(state_db->dynamic_keys_pool, 0, num_states * num_accounts * sizeof(evm_word_t *));
    memset(state_db->dynamic_values_pool, 0, num_states * num_accounts * sizeof(ValueStatus *));

    cJSON_ArrayForEach(account_json, state_json) {
        cJSON *balance_json, *nonce_json;

        state_db->address_list[idx].from_hex(account_json->string);

        // set the balance
        balance_json = cJSON_GetObjectItemCaseSensitive(account_json, "balance");
        state_db->account_balances[idx].from_hex(balance_json->valuestring);

        // set the nonce
        nonce_json = cJSON_GetObjectItemCaseSensitive(account_json, "nonce");
        evm_word_t nonce;
        nonce.from_hex(nonce_json->valuestring);
        state_db->account_nonces[idx] = uint256_get_uint32_t(&nonce);
        byte_array_t byte_code;
        byte_code.from_hex(cJSON_GetObjectItemCaseSensitive(account_json, "code")->valuestring, LITTLE_ENDIAN,
                           NO_PADDING);
        state_db->account_codes_size[idx] = byte_code.size;
        state_db->account_codes_offset[idx] = bytecode_offset;
        uint8_t *tmp = state_db->all_account_codes;
        state_db->all_account_codes = new uint8_t[bytecode_offset + byte_code.size];
        memcpy(state_db->all_account_codes, tmp, bytecode_offset * sizeof(uint8_t));
        delete[] tmp;
        memcpy(&state_db->all_account_codes[bytecode_offset], byte_code.data, byte_code.size);
        bytecode_offset += byte_code.size;
        // allocate dynamic storage

        // preallocate fix sized storage
        cJSON *storage_json = cJSON_GetObjectItemCaseSensitive(account_json, "storage");
        if (storage_json != nullptr) {
            uint32_t storage_size = cJSON_GetArraySize(storage_json);
            for (uint32_t i = 0; i < num_states; i++) {
                state_db->account_storage_size[idx * num_states + i] = storage_size;
            }
            state_db->num_storage_elements += state_db->account_storage_size[idx * num_states];
            // allocate new page per account
            if (state_db->num_storage_elements >= state_db->storage_capacity) {
                ValueStatus *tmp = state_db->prealloc_values_pool;
                state_db->prealloc_values_pool = new ValueStatus[state_db->num_storage_elements * num_states];
                if (tmp != nullptr) {
                    memcpy(state_db->prealloc_values_pool, tmp,
                           state_db->num_storage_elements * num_states * sizeof(ValueStatus));
                    delete[] tmp;
                }
            }
            // todo: handle storage size > account_prealloc_keys_size
        } else
            memset(&state_db->account_storage_size[idx * num_states], 0, num_states * sizeof(uint32_t));

        uint32_t pre_alloc_keys_idx = idx * account_prealloc_keys_size;
        for (uint32_t i = 0; i < state_db->account_storage_size[idx * num_states]; i++) {
            cJSON *storage_element_json = cJSON_GetArrayItem(storage_json, i);

            state_db->all_keys[pre_alloc_keys_idx + i].key.from_hex(storage_element_json->string);
            state_db->all_keys[pre_alloc_keys_idx + i].offset = (key_offset_wo_states + i) * num_states;
            state_db->prealloc_values_pool[state_db->all_keys[pre_alloc_keys_idx + i].offset].from_hex(
                storage_element_json->valuestring);
            // duplicate the value for all states
            for (uint32_t j = 1; j < num_states; j++) {
                state_db->prealloc_values_pool[state_db->all_keys[pre_alloc_keys_idx + i].offset + j] =
                    state_db->prealloc_values_pool[state_db->all_keys[pre_alloc_keys_idx + i].offset];
            }
            // printf("key: %d \n", pre_alloc_keys_idx + i);
            // state_db->all_keys[pre_alloc_keys_idx + i].key.print();
            // printf("offset: %d\n", state_db->all_keys[pre_alloc_keys_idx + i].offset);
        }
        state_db->keys_list_offset[idx] = pre_alloc_keys_idx;
        key_offset_wo_states += state_db->account_storage_size[idx * num_states];
        idx++;
    }
    state_db->print();
}
__host__ StateDb *StateDb::GPUFromCPU(StateDb *&state_db) {
    StateDb *state_db_gpu = (StateDb *)malloc(sizeof(StateDb));
    memcpy(state_db_gpu, state_db, sizeof(StateDb));
    return state_db_gpu;
}
__host__ StateDb *StateDb::CPUFromGPU(StateDb *&state_db) {
    StateDb *state_db_cpu = (StateDb *)malloc(sizeof(StateDb));
    memcpy(state_db_cpu, state_db, sizeof(StateDb));
    // copy other inner data
    uint32_t num_accounts = state_db_cpu->num_accounts;
    uint32_t num_storage_elements = state_db_cpu->num_storage_elements;
    uint32_t num_states = state_db_cpu->num_states;
    uint32_t storage_capacity = state_db_cpu->storage_capacity;
    // memcopy other inner data
    memcpy(state_db_cpu->address_list, state_db->address_list, sizeof(evm_word_t) * num_accounts);
    memcpy(state_db_cpu->account_balances, state_db->account_balances, sizeof(evm_word_t) * num_accounts);
    memcpy(state_db_cpu->account_nonces, state_db->account_nonces, sizeof(uint32_t) * num_accounts);
    // double check
    memcpy(state_db_cpu->all_keys, state_db->all_keys, sizeof(KeyOffset) * num_storage_elements * num_states);
    memcpy(state_db_cpu->prealloc_values_pool, state_db->prealloc_values_pool,
           sizeof(evm_word_t) * num_storage_elements * num_states);
    return state_db_cpu;
}

__device__ void StateDb::serialize_data(serialized_worldstate_data *data) {
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

__host__ __device__ void StateDb::print() {
    printf("num_accounts: %d\n", num_accounts);
    printf("num_storage_elements: %d\n", num_storage_elements);
    for (uint32_t i = 0; i < num_accounts; i++) {
        printf("\n\n address: \n");
        address_list[i].print();
        printf("balance: \n");
        account_balances[i].print();
        printf("nonce: %d\n", account_nonces[i]);
        if (account_storage_size[i] > 0) {
            printf("keys size %d\n", account_storage_size[i]);
            for (uint32_t j = 0; j < account_storage_size[i]; j++) {
                printf("\n key: \n");
                all_keys[keys_list_offset[i] + j].key.print();
                printf("offset: %d\n", all_keys[keys_list_offset[i] + j].offset);
                printf("value: \n");
                prealloc_values_pool[all_keys[keys_list_offset[i] + j].offset + INSTANCE_GLOBAL_IDX].print();
            }
        }
        if (account_codes_size[i] > 0) {
            printf("code size %d\n", account_codes_size[i]);
            printf("code: \n");
            for (uint32_t j = 0; j < account_codes_size[i]; j++) {
                printf("%02x", all_account_codes[account_codes_offset[i] + j]);
            }
            printf("\n");
        }
    }
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
__device__ StateDb *global_state_db_ptr;
}  // namespace CuEVM
