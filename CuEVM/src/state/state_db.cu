#include <CuEVM/state/state_db.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM {

// return snapshot storage and the offset in the page
__device__ int32_t SnapshotAccount::find_dynamic_offset(ValueStatus *value) {
    // first page:
    uint32_t iter = (storage_size - memory_pool_snapshot_preallocate) % snapshot_page_size;
    SnapshotStoragePage *current_page = storage_page;
    if (current_page == nullptr) return -1;
    for (uint32_t i = 0; i < iter; i++) {
        if (current_page->restore_ptr[i] == value) {
            return i;
        }
    }
    current_page = current_page->next_page;
    while (current_page != nullptr) {
        for (uint32_t i = 0; i < snapshot_page_size; i++) {
            if (current_page->restore_ptr[i] == value) {
                return i;
            }
        }
        current_page = current_page->next_page;
    }
    return -1;
}
__device__ void SnapshotAccount::revert_to_depth(const uint16_t depth) {
    // TODO: implement
    SnapshotAccount *snapshot_account = this;
    while (snapshot_account != nullptr && snapshot_account->depth > depth) {
        snapshot_account = snapshot_account->next_account;
    }
    if (snapshot_account == nullptr) return;
    for (uint32_t i = 0; i < snapshot_account->storage_size; i++) {
        if (i < memory_pool_snapshot_preallocate) {
            CuEVM::memory_pool::preallocated_snapshot_restore_ptr[i]->value =
                CuEVM::memory_pool::preallocated_snapshot_values[i].value;
            CuEVM::memory_pool::preallocated_snapshot_restore_ptr[i]->is_warm =
                CuEVM::memory_pool::preallocated_snapshot_values[i].is_warm;
        } else {
            SnapshotStoragePage *current_page = snapshot_account->storage_page;
            while (current_page != nullptr) {
                for (uint32_t j = 0; j < snapshot_page_size; j++) {
                    if (current_page->restore_ptr[j] != nullptr) {
                        current_page->restore_ptr[j]->value = current_page->values[j].value;
                        current_page->restore_ptr[j]->is_warm = current_page->values[j].is_warm;
                    }
                }
                current_page = current_page->next_page;
            }
        }
    }
}
__device__ void SnapshotAccount::set_storage(const uint16_t depth, const uint32_t contract_index, const evm_word_t *key,
                                             ValueStatus *src_value) {
    // N contract x mempool__snapshot_preallocate x num_states
    uint32_t start_offset =
        (memory_pool_snapshot_preallocate * contract_index) * global_state_db_ptr->num_states + INSTANCE_GLOBAL_IDX;
    for (uint32_t i = 0; i < min(storage_size, memory_pool_snapshot_preallocate); i++) {
        uint32_t offset = start_offset + i * global_state_db_ptr->num_states + INSTANCE_GLOBAL_IDX;
        if (CuEVM::memory_pool::preallocated_snapshot_restore_ptr[offset] == src_value) return;
    }
    if (storage_size > memory_pool_snapshot_preallocate) {
        uint32_t dynamic_offset = find_dynamic_offset(src_value);
        if (dynamic_offset == -1) {
            dynamic_offset = (storage_size - memory_pool_snapshot_preallocate) % snapshot_page_size;
            if (dynamic_offset == 0) {
                SnapshotStoragePage *new_page = new SnapshotStoragePage();
                new_page->next_page = storage_page;
                storage_page = new_page;
                dynamic_offset = 0;
            }
            storage_page->restore_ptr[dynamic_offset] = src_value;
            storage_page->values[dynamic_offset].value = src_value->value;
            storage_page->values[dynamic_offset].is_warm = src_value->is_warm;
        }
    }
}
// DynamicSnapshotAccount
__device__ DynamicSnapshotAccount::~DynamicSnapshotAccount() {}
__device__ void DynamicSnapshotAccount::set_account(const uint16_t depth, const evm_word_t *balance,
                                                    const uint32_t nonce) {
    this->depth = depth;
    this->balance = *balance;
    this->nonce = nonce;
}
// __device__ void revert_to_depth(const uint16_t depth);
// __device__ void set_account(const uint16_t depth, const evm_word_t *balance, const uint32_t nonce);
// __device__ void set_storage(const uint16_t depth, const uint32_t contract_index, const evm_word_t *key,
//                             ValueStatus *value);
__device__ void DynamicSnapshotAccount::set_storage(const uint16_t depth, const evm_word_t *key, ValueStatus *value) {
    int32_t key_offset = -1;
    uint32_t iter = (storage_size) % snapshot_page_size;
    SnapshotStoragePage *current_page = storage_page;
    if (current_page != nullptr) {
        for (uint32_t i = 0; i < iter; i++) {
            if (current_page->restore_ptr[i] == value) {
                key_offset = i;
            }
        }
        current_page = current_page->next_page;
        while (current_page != nullptr) {
            for (uint32_t i = 0; i < snapshot_page_size; i++) {
                if (current_page->restore_ptr[i] == value) {
                    key_offset = i;
                }
            }
            current_page = current_page->next_page;
        }
    }
    if (key_offset == -1) {
        key_offset = (storage_size) % snapshot_page_size;
        if (key_offset == 0) {
            SnapshotStoragePage *new_page = new SnapshotStoragePage();
            new_page->next_page = storage_page;
            storage_page = new_page;
            key_offset = 0;
        }
    }
    this->storage_page->restore_ptr[key_offset] = value;
    this->storage_page->values[key_offset].value = value->value;
    this->storage_page->values[key_offset].is_warm = value->is_warm;
}

// DynamicAccount
__device__ DynamicAccount::DynamicAccount(const evm_word_t *address, const evm_word_t *balance, const uint32_t nonce,
                                          const uint32_t storage_size, const uint32_t code_size, const uint8_t *code) {
    this->address = *address;
    if (balance != nullptr)
        this->balance = *balance;
    else
        this->balance = 0;
    this->nonce = nonce;
    this->storage_size = storage_size;
    if (code_size > 0 && code != nullptr) {
        if (this->code != nullptr) {
            delete[] this->code;
        }
        this->code_size = code_size;
        this->code = new uint8_t[code_size];
        memcpy(this->code, code, code_size);
    }
    this->snapshot_account = new DynamicSnapshotAccount();
}

__device__ ValueStatus *DynamicAccount::get_value_status(const evm_word_t *key) {
    // first page iter
    uint32_t iter = (storage_size) % value_page_size;
    StateDbStoragePage *current_page = storage_page;
    if (current_page == nullptr) return nullptr;
    for (uint32_t i = 0; i < iter; i++) {
        if (current_page->keys[i] == *key) {
            return &current_page->values[i];
        }
    }
    current_page = current_page->next_page;
    while (current_page != nullptr) {
        for (uint32_t i = 0; i < value_page_size; i++) {
            if (current_page->keys[i] == *key) {
                return &current_page->values[i];
            }
        }
        current_page = current_page->next_page;
    }
    return nullptr;
}
__device__ void DynamicAccount::set_storage(const uint16_t depth, const evm_word_t *key, const evm_word_t *value,
                                            bool is_warm) {
    ValueStatus *found_value = get_value_status(key);
    if (found_value == nullptr) {
        printf("not found in dynamic account\n");
        uint32_t offset = (storage_size) % value_page_size;
        if (offset == 0) {
            StateDbStoragePage *new_page = new StateDbStoragePage();
            new_page->next_page = storage_page;
            storage_page = new_page;
        }
        storage_page->keys[offset] = *key;
        storage_page->values[offset].value = *value;
        storage_page->values[offset].is_warm = is_warm;
        storage_size++;
        printf("set storage thread %d, address %p, key %p, value %p, is_warm %d\n", INSTANCE_GLOBAL_IDX, address, key,
               value, is_warm);
        return;
    } else {
        found_value->value = *value;
        found_value->original_value = 0;
        found_value->is_warm = is_warm;
    }
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
    prealloc_keys_pool = nullptr;
    prealloc_values_pool = nullptr;  // offset works for this
    dynamic_storage_pages = nullptr;
    dynamic_pool_capacity = nullptr;
}
__device__ StateDb::StateDb(uint32_t num_states, uint32_t num_accounts, uint32_t num_storage_elements,
                            uint32_t storage_capacity, uint8_t *all_account_codes, evm_word_t *address_list,
                            evm_word_t *account_balances, uint32_t *account_nonces, uint32_t *account_storage_size,
                            uint32_t *account_codes_size, uint32_t *account_codes_offset,
                            evm_word_t *prealloc_keys_pool, ValueStatus *prealloc_values_pool,
                            StateDbStoragePage **dynamic_storage_pages, uint32_t *dynamic_pool_capacity,
                            SnapshotAccount *snapshot_accounts)
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
      prealloc_keys_pool(prealloc_keys_pool),
      prealloc_values_pool(prealloc_values_pool),
      dynamic_storage_pages(dynamic_storage_pages),
      dynamic_pool_capacity(dynamic_pool_capacity),
      snapshot_accounts(snapshot_accounts) {}

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

__device__ int32_t StateDb::get_value_offset(uint32_t storage_size, uint32_t contract_idx,
                                             const evm_word_t *key) const {
    uint32_t loop_iter = min(storage_size, account_prealloc_keys_size);
    // ACC 1 Slot 1 .... Account prealloc_keys_size(xnum_states)  => ACC 2 Slot 1 .... Account
    // prealloc_keys_size(num_states)
    uint32_t base_offset = (contract_idx * account_prealloc_keys_size * num_states);
    for (uint32_t i = 0; i < loop_iter; i++) {
        uint32_t current_offset = base_offset + i * num_states + INSTANCE_GLOBAL_IDX;
        if (prealloc_keys_pool[current_offset] == *key) {
            return current_offset;
        }
    }
    return -1;
}

__device__ DynamicAccount *StateDb::get_dynamic_account(const evm_word_t *address) const {
    // TODO: implement
    DynamicAccount *current_account = dynamic_accounts[INSTANCE_GLOBAL_IDX];
    printf("get dynamic account thread %d, address %p current_account %p\n", INSTANCE_GLOBAL_IDX, address,
           current_account);
    while (current_account != nullptr) {
        printf("current_account address %p\n", current_account->address);
        current_account->address.print();
        if (current_account->address == *address) {
            return current_account;
        }
        current_account = current_account->next_account;
    }
    printf("End of get dynamic account, failed to find \n");

    return nullptr;
}

__device__ DynamicAccount *StateDb::new_account(const uint16_t depth, const evm_word_t *address,
                                                const evm_word_t *balance, const uint32_t nonce,
                                                const uint32_t code_size, uint8_t *code) {
    printf("Create new dynamic accoun \n");
    DynamicAccount *new_acc = new DynamicAccount(address, balance, nonce, 0, code_size, code);
    new_acc->next_account = dynamic_accounts[INSTANCE_GLOBAL_IDX];
    dynamic_accounts[INSTANCE_GLOBAL_IDX] = new_acc;
    return new_acc;
}
__device__ int32_t StateDb::create_contract(const uint16_t depth, const evm_word_t *address, const uint32_t code_size,
                                            uint8_t *code) {
    // not used
    int32_t address_index = get_address_index(address);
    if (address_index != -1) {
        // clear existing prealloc account
        if (account_storage_size[address_index] != 0) {
            return ERROR_ACCOUNT_NOT_EMPTY;
        } else {
            address_list[address_index] = 1;  // non-collision address, precompiled
        }
    }

    DynamicAccount *dynamic_account = get_dynamic_account(address);
    if (dynamic_account == nullptr) {
        dynamic_account = StateDb::new_account(depth, address, 0, 0, code_size, code);
    }
    dynamic_account->code_size = code_size;
    dynamic_account->code = code;
    return ERROR_SUCCESS;
}
__device__ void StateDb::update_account(const uint16_t depth, const evm_word_t *address, const evm_word_t *balance,
                                        const uint32_t nonce) {
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        DynamicAccount *dynamic_account = get_dynamic_account(address);
        if (dynamic_account == nullptr) {
            dynamic_account = StateDb::new_account(depth, address, balance, nonce, 0, nullptr);
        }
        dynamic_account->balance = *balance;
        dynamic_account->nonce = nonce;
        return;
    }
    uint32_t instance_idx = address_index * num_states + INSTANCE_GLOBAL_IDX;
    account_balances[instance_idx] = *balance;
    account_nonces[instance_idx] = nonce;
}
__device__ void StateDb::update_code(const uint16_t depth, const evm_word_t *address, const uint32_t code_size,
                                     uint8_t *code) {
    int32_t address_index = get_address_index(address);
    if (address_index != -1) {
        address_list[address_index] = 1;  // non-collision address, precompiled
        contract_index[address_index] = -1;
    }
    DynamicAccount *dynamic_account = get_dynamic_account(address);
    if (dynamic_account == nullptr) {
        dynamic_account = StateDb::new_account(depth, address, 0, 0, code_size, code);
    }
    dynamic_account->code_size = code_size;
    dynamic_account->code = code;
}
__device__ void StateDb::update_balance(const uint16_t depth, const evm_word_t *address, const evm_word_t *balance,
                                        bool is_warm) {
    // printf("update balance thread %d, address %p\n", INSTANCE_GLOBAL_IDX, address);
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        DynamicAccount *dynamic_account = get_dynamic_account(address);
        if (dynamic_account == nullptr) {
            dynamic_account = StateDb::new_account(depth, address, balance, 0, 0, nullptr);
        }
        dynamic_account->balance = *balance;
        dynamic_account->is_warm = is_warm;
        return;
    }
    account_balances[address_index * num_states + INSTANCE_GLOBAL_IDX] = *balance;
    account_is_warm[address_index * num_states + INSTANCE_GLOBAL_IDX] = is_warm;
}
// shortcut for deducting balance
__device__ int32_t StateDb::deduct_balance(const uint16_t depth, const evm_word_t *address, const evm_word_t *amount) {
    printf("deduct balance thread %d\n", INSTANCE_GLOBAL_IDX);
    address->print();

    int32_t address_index = get_address_index(address);
    printf("address index %d\n", address_index);
    evm_word_t *current_balance;
    if (address_index == -1) {
        DynamicAccount *dynamic_account = get_dynamic_account(address);
        if (dynamic_account == nullptr) {
            return ERROR_INSUFFICIENT_FUNDS;
        }
        current_balance = &dynamic_account->balance;
    } else {
        current_balance = &account_balances[address_index * num_states + INSTANCE_GLOBAL_IDX];
    }
    printf("current balance\n");
    current_balance->print();
    printf("amount\n");
    amount->print();
    if (uint256_cmp(current_balance, amount) < 0) {
        return ERROR_INSUFFICIENT_FUNDS;
    }
    uint256_sub(current_balance, current_balance, amount);
    return ERROR_SUCCESS;
}

__device__ void StateDb::update_nonce(const uint16_t depth, const evm_word_t *address, const uint32_t nonce) {
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        DynamicAccount *dynamic_account = get_dynamic_account(address);
        if (dynamic_account == nullptr) {
            dynamic_account = StateDb::new_account(depth, address, 0, nonce, 0, nullptr);
        }
        dynamic_account->nonce = nonce;
        return;
    }
    account_nonces[address_index * num_states + INSTANCE_GLOBAL_IDX] = nonce;
}

__device__ int32_t StateDb::transfer(const uint16_t depth, const evm_word_t *sender, const evm_word_t *recipient,
                                     const evm_word_t *value) {
    int32_t error_code = deduct_balance(depth, sender, value);
    printf("transfer error code %d\n", error_code);
    if (error_code != ERROR_SUCCESS) return error_code;
    update_balance(depth, recipient, value);
    return ERROR_SUCCESS;
}

__device__ ValueStatus *StateDb::get_dynamic_value_location(uint32_t storage_size, uint32_t instance_idx,
                                                            const evm_word_t *key) const {
    // first page :
    uint32_t iter = (storage_size - account_prealloc_keys_size) % value_page_size;
    StateDbStoragePage *current_page = dynamic_storage_pages[instance_idx];
    if (current_page == nullptr) return nullptr;

    for (uint32_t i = 0; i < iter; i++) {
        if (current_page->keys[i] == *key) {
            return &current_page->values[i];
        }
    }
    current_page = current_page->next_page;
    while (current_page != nullptr) {
        for (uint32_t i = 0; i < value_page_size; i++) {
            if (current_page->keys[i] == *key) {
                return &current_page->values[i];
            }
        }
        current_page = current_page->next_page;
    }
    return nullptr;
}

__device__ void StateDb::write_storage_with_known_index(const uint16_t depth, const evm_word_t *address,
                                                        const evm_word_t *key, const evm_word_t *value,
                                                        int32_t address_index, ValueStatus *found_value, bool is_warm) {
    printf("write storage with known index thread %d, address %p\n", INSTANCE_GLOBAL_IDX, address);
    if (address_index == -1) {
        DynamicAccount *dynamic_account = get_dynamic_account(address);
        printf("dynamic_account %p\n", dynamic_account);
        if (dynamic_account == nullptr) {
            return;  // should never write directly to storage before creating account
        }
        dynamic_account->set_storage(depth, key, value, is_warm);
        return;
    }
    uint32_t instance_idx = address_index * num_states + INSTANCE_GLOBAL_IDX;
    uint32_t contract_idx = contract_index[address_index];
    // printf("storage before write\n");
    // for (uint32_t i = 0; i < account_prealloc_keys_size; i++) {
    //     uint32_t current_offset = (contract_idx * account_prealloc_keys_size + i) * num_states +
    //     INSTANCE_GLOBAL_IDX; printf("key: \n"); prealloc_keys_pool[current_offset].print(); printf("value: \n");
    //     prealloc_values_pool[current_offset].print();
    // }
    // printf("write storage  with know index found_value %p\n", found_value);
    // printf("account_storage_size %d\n", account_storage_size[instance_idx]);

    uint32_t storage_size = account_storage_size[instance_idx];
    if (found_value == nullptr) {
        if (storage_size < account_prealloc_keys_size) {
            uint32_t new_offset =
                (contract_idx * account_prealloc_keys_size + storage_size) * num_states + INSTANCE_GLOBAL_IDX;
            printf("storage size %d, new offset %d\n", storage_size, new_offset);
            prealloc_keys_pool[new_offset] = *key;
            prealloc_values_pool[new_offset].set_value(value, is_warm);

            account_storage_size[instance_idx]++;
            // num_storage_elements++;

        } else {
            ValueStatus *dynamic_value = get_dynamic_value_location(storage_size, instance_idx, key);
            if (dynamic_value == nullptr) {
                dynamic_value = grow_storage_and_set_key(storage_size, instance_idx, key);
                dynamic_value->set_value(value, is_warm);
                account_storage_size[instance_idx]++;
                // num_storage_elements++;
            }
        }

    } else {
        // write to exsiting slot
        snapshot_accounts[instance_idx].set_storage(depth, contract_idx, key, found_value);
        found_value->set_value(value, is_warm);
    }
    account_is_warm[instance_idx] = true;
    // printf("storage after write\n");
    // for (uint32_t i = 0; i < account_prealloc_keys_size; i++) {
    //     uint32_t current_offset = (contract_idx * account_prealloc_keys_size + i) * num_states +
    //     INSTANCE_GLOBAL_IDX; printf("key: \n"); prealloc_keys_pool[current_offset].print(); printf("value: \n");
    //     prealloc_values_pool[current_offset].print();
    // }
}

__device__ void StateDb::snapshot_account(const uint16_t depth, const uint32_t address_index, const evm_word_t *balance,
                                          const uint32_t nonce) {
    uint32_t instance_idx = address_index * num_states + INSTANCE_GLOBAL_IDX;
    snapshot_accounts[instance_idx].set_account(depth, balance, nonce);
}
__device__ void StateDb::snapshot_storage(const uint16_t depth, const uint32_t address_index, evm_word_t *key,
                                          ValueStatus *value) {
    uint32_t instance_idx = address_index * num_states + INSTANCE_GLOBAL_IDX;
    snapshot_accounts[instance_idx].set_storage(depth, contract_index[address_index], key, value);
}
__device__ void StateDb::snapshot_code(const uint16_t depth, const uint32_t address_index) {
    // snapshot_accounts[address_index].set_code(depth, code_size);
}

__device__ void StateDb::revert_to_snapshot(const uint16_t depth) {
    // TODO: implement
    for (uint32_t i = 0; i < num_accounts; i++) {
        uint32_t instance_idx = i * num_states + INSTANCE_GLOBAL_IDX;
        snapshot_accounts[instance_idx].revert_to_depth(depth);
    }
}
__device__ evm_word_t *StateDb::get_storage_with_known_index(const uint16_t depth, const evm_word_t *address,
                                                             const evm_word_t *key, int32_t address_index,
                                                             ValueStatus *found_value, bool set_warm) {
    uint32_t instance_idx = address_index * num_states + INSTANCE_GLOBAL_IDX;
    // printf("get_storage_with_known_index found_value %p\n", found_value);
    // printf("account_storage_size %d\n", account_storage_size[instance_idx]);

    if (found_value == nullptr) {
        evm_word_t zero = 0;
        uint32_t storage_size = account_storage_size[instance_idx];
        if (set_warm) {
            if (storage_size < account_prealloc_keys_size) {
                uint32_t new_offset =
                    (contract_index[address_index] * account_prealloc_keys_size + storage_size) * num_states +
                    INSTANCE_GLOBAL_IDX;
                prealloc_keys_pool[new_offset] = *key;
                prealloc_values_pool[new_offset].set_value(&zero, true);
            } else {
                ValueStatus *value_status = grow_storage_and_set_key(storage_size, instance_idx, key);
                value_status->set_value(&zero, true);
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

/*
__device__ evm_word_t *StateDb::get_storage(const uint16_t depth, const evm_word_t *address, const evm_word_t *key,
                                            bool set_warm) {
    // printf("get storage thread %d, address %p\n", INSTANCE_GLOBAL_IDX, address);
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        // todo grow account
        return nullptr;
    }
    uint32_t instance_idx = address_index * num_states + INSTANCE_GLOBAL_IDX;
    uint32_t contract_idx = contract_index[address_index];
    uint32_t storage_size = account_storage_size[instance_idx];
    int32_t key_offset = get_value_offset(storage_size, contract_idx, key);
    if (key_offset == -1) {
        ValueStatus *value_status = get_dynamic_value_location(storage_size, instance_idx, key);
        if (value_status == nullptr) {
            evm_word_t zero = 0;
            if (set_warm) {
                if (storage_size < account_prealloc_keys_size) {
                    uint32_t new_offset =
                        (contract_idx * account_prealloc_keys_size + storage_size) * num_states + INSTANCE_GLOBAL_IDX;
                    prealloc_values_pool[new_offset].set_value(&zero, true);
                } else {
                    ValueStatus *value_status = grow_storage_and_set_key(storage_size, instance_idx, key);
                    value_status->set_value(&zero, true);
                }
                account_storage_size[instance_idx]++;
            }
            return nullptr;
        } else {
            // printf("found in dynamic storage\n");
            value_status->is_warm = set_warm;
            account_is_warm[instance_idx] |= set_warm;
            return &value_status->value;
        }

    } else {
        // printf("found in prealloc_values_pool\n");
        prealloc_values_pool[key_offset].is_warm = set_warm;
        account_is_warm[instance_idx] |= set_warm;
        return &prealloc_values_pool[key_offset].value;
    }
}
*/

// __device__ ValueStatus *StateDb::get_value_status(const evm_word_t *address, const evm_word_t *key) const {
//     int32_t address_index = get_address_index(address);
//     if (address_index == -1) {
//         return nullptr;
//     }
//     uint32_t instance_idx = address_index * num_states + INSTANCE_GLOBAL_IDX;
//     uint32_t contract_idx = contract_index[address_index];
//     uint32_t storage_size = account_storage_size[instance_idx];
//     int32_t key_offset = get_value_offset(storage_size, contract_idx, key);
//     if (key_offset == -1) {
//         // printf(" not found in prealloc_values_pool, search dynamic storage\n");
//         // find in dynamic_keys_pool
//         ValueStatus *value_status = get_dynamic_value_location(storage_size, instance_idx, key);
//         if (value_status == nullptr) {
//             // printf("not found at all\n");
//             return nullptr;
//         } else {
//             // printf("found in dynamic storage\n");
//             return value_status;
//         }
//     } else {
//         // printf("found in prealloc_values_pool\n");
//         return &prealloc_values_pool[key_offset];
//     }
// }
__device__ ValueStatus *StateDb::get_value_status(const int32_t address_index, const evm_word_t *key) const {
    uint32_t instance_idx = address_index * num_states + INSTANCE_GLOBAL_IDX;
    uint32_t contract_idx = contract_index[address_index];
    uint32_t storage_size = account_storage_size[instance_idx];
    printf("get_value_status address_index %d, storage_size %d\n", address_index, storage_size);
    int32_t key_offset = get_value_offset(storage_size, contract_idx, key);
    if (key_offset == -1) {
        // printf(" not found in prealloc_values_pool, search dynamic storage\n");
        // find in dynamic_keys_pool
        ValueStatus *value_status = get_dynamic_value_location(storage_size, instance_idx, key);
        if (value_status == nullptr) {
            // printf("not found at all\n");
            return nullptr;
        } else {
            // printf("found in dynamic storage\n");
            return value_status;
        }
    } else {
        // printf("found in prealloc_values_pool\n");
        return &prealloc_values_pool[key_offset];
    }
}

__device__ ValueStatus *StateDb::grow_storage_and_set_key(uint32_t storage_size, int32_t instance_idx,
                                                          const evm_word_t *key) {
    uint32_t pool_capacity = dynamic_pool_capacity[instance_idx];
    // printf("grow_storage_and_set_key\n");
    // printf("pool capacity %d storage size %d\n", pool_capacity, storage_size);
    if (storage_size >= account_prealloc_keys_size) {
        uint32_t new_idx = (storage_size - account_prealloc_keys_size) % value_page_size;
        if (pool_capacity == 0) {
            dynamic_pool_capacity[instance_idx] = value_page_size;
            dynamic_storage_pages[instance_idx] = new StateDbStoragePage();

        } else {
            if (new_idx == 0) {
                dynamic_pool_capacity[instance_idx] += value_page_size;
                StateDbStoragePage *new_page = new StateDbStoragePage();
                new_page->next_page = dynamic_storage_pages[instance_idx];
                dynamic_storage_pages[instance_idx] = new_page;
            }
        }
        dynamic_storage_pages[instance_idx]->keys[new_idx] = *key;
        return &dynamic_storage_pages[instance_idx]->values[new_idx];
    }
    return nullptr;
}

__device__ evm_word_t *StateDb::get_balance(const evm_word_t *address) {
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        DynamicAccount *dynamic_account = get_dynamic_account(address);
        if (dynamic_account == nullptr) {
            return nullptr;
        }
        return &dynamic_account->balance;
    }

    return &account_balances[address_index * num_states + INSTANCE_GLOBAL_IDX];
}

__device__ uint32_t StateDb::get_nonce(const evm_word_t *address) {
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        DynamicAccount *dynamic_account = get_dynamic_account(address);
        if (dynamic_account == nullptr) {
            return 0;
        }
        return dynamic_account->nonce;
    }
    return account_nonces[address_index * num_states + INSTANCE_GLOBAL_IDX];
}

__device__ uint8_t *StateDb::get_code(uint32_t &code_size, const evm_word_t *address) {
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        DynamicAccount *dynamic_account = get_dynamic_account(address);
        if (dynamic_account == nullptr) {
            return nullptr;
        }
        return dynamic_account->code;
    }

    code_size = account_codes_size[address_index];
    return &all_account_codes[account_codes_offset[address_index]];
}

__device__ void StateDb::set_warm_account(const evm_word_t *address) {
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        DynamicAccount *dynamic_account = get_dynamic_account(address);
        if (dynamic_account == nullptr) {
            DynamicAccount *new_acc = StateDb::new_account(0, address, 0, 0, 0, nullptr);

            dynamic_account = new_acc;
        }
        dynamic_account->is_warm = true;
    } else
        account_is_warm[address_index * num_states + INSTANCE_GLOBAL_IDX] = true;
}

__device__ void StateDb::set_warm_key(const evm_word_t *address, const evm_word_t *key) {
    // todo :implement
}

__device__ bool StateDb::is_warm_account(const evm_word_t *address) const {
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        DynamicAccount *dynamic_account = get_dynamic_account(address);
        if (dynamic_account == nullptr) {
            return false;
        } else {
            return dynamic_account->is_warm;
        }
    }
    uint32_t instance_idx = address_index * num_states + INSTANCE_GLOBAL_IDX;
    return account_is_warm[instance_idx];
}

// __device__ bool StateDb::is_warm_key(const evm_word_t *address, const evm_word_t *key) const {
//     if (!is_warm_account(address)) {
//         return false;
//     }
//     ValueStatus *value_status = get_value_status(address, key);
//     if (value_status == nullptr) {
//         return false;
//     }
//     return value_status->is_warm;
// }

__device__ bool StateDb::is_warm_key_with_offset(const evm_word_t *address, const evm_word_t *key,
                                                 int32_t &address_index, ValueStatus *&found_value) {
    address_index = get_address_index(address);
    if (address_index == -1) {
        found_value = nullptr;
        DynamicAccount *dynamic_account = get_dynamic_account(address);
        if (dynamic_account != nullptr) {
            found_value = dynamic_account->get_value_status(key);
            if (found_value != nullptr) {
                return found_value->is_warm;
            }
        }
        return false;
    }
    found_value = get_value_status(address_index, key);
    if (found_value == nullptr) {
        return false;
    }
    return found_value->is_warm;
}

__device__ bool StateDb::is_empty_account(const evm_word_t *address) const {
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        DynamicAccount *dynamic_account = get_dynamic_account(address);
        if (dynamic_account == nullptr) {
            return true;
        } else {
            return dynamic_account->code_size == 0 && dynamic_account->nonce == 0 &&
                   uint256_is_zero(&dynamic_account->balance);
        }
    }
    uint32_t instance_idx = address_index * num_states + INSTANCE_GLOBAL_IDX;
    return (account_nonces[instance_idx] == 0 && account_storage_size[instance_idx] == 0 &&
            uint256_is_zero(&account_balances[instance_idx]));
}

__device__ bool StateDb::is_deleted_account(const evm_word_t *address) const {
    // todo :implement
    return false;
}

__device__ bool StateDb::is_empty_create(const evm_word_t *address) {
    // Todo : This definition changes overtime, need to be refactored
    // Either check storage size == 0 or not
    int32_t address_index = get_address_index(address);

    if (address_index == -1) {
        DynamicAccount *dynamic_account = get_dynamic_account(address);
        if (dynamic_account == nullptr) {
            printf("dynamic account is null, create new account\n");
            DynamicAccount *new_acc = new_account(0, address, 0, 0, 0, nullptr);
            return true;
        } else {
            if (dynamic_account->code_size == 0 && dynamic_account->nonce == 0) {
                // reset storage size to 0, TODO: free
                dynamic_account->storage_size = 0;
                return true;
            }
            return false;
        }
    }
    uint32_t instance_idx = address_index * num_states + INSTANCE_GLOBAL_IDX;

    if (account_nonces[instance_idx] == 0 && account_codes_size[address_index] == 0) {
        // reset storage size to 0, TODO: free
        account_storage_size[instance_idx] = 0;
        address_list[address_index] = 1;  // non-collision address, precompiled
        // create a new dynamic account
        // TODO: pass the depth and revert
        DynamicAccount *new_acc = new_account(0, address, &account_balances[instance_idx], 0, 0, nullptr);

        return true;
    }
    return false;
}
__device__ bool StateDb::is_contract(const evm_word_t *address) const {
    // todo :implement
    return true;
}
__host__ void StateDb::GPUfromJson(StateDb *&state_db, const cJSON *state_json, uint32_t num_states,
                                   uint32_t &num_accounts) {
    StateDb *state_db_cpu = new StateDb(num_states);
    StateDb::CPUfromJson(state_db_cpu, state_json, num_states);
    StateDb *tmp_state_db = new StateDb(num_states);
    tmp_state_db->num_accounts = state_db_cpu->num_accounts;
    tmp_state_db->num_states = state_db_cpu->num_states;
    tmp_state_db->num_storage_elements = state_db_cpu->num_storage_elements;
    tmp_state_db->storage_capacity = state_db_cpu->storage_capacity;
    tmp_state_db->num_contracts = state_db_cpu->num_contracts;

    num_accounts = state_db_cpu->num_accounts;
    uint32_t code_size =
        state_db_cpu->account_codes_size[num_accounts - 1] + state_db_cpu->account_codes_offset[num_accounts - 1];

    // Grouped memory allocation
    CUDA_CHECK(cudaMalloc(&tmp_state_db->address_list, num_accounts * sizeof(evm_word_t)));
    CUDA_CHECK(cudaMalloc(&tmp_state_db->contract_index, num_accounts * sizeof(int16_t)));

    CUDA_CHECK(cudaMalloc(&tmp_state_db->account_balances, num_states * num_accounts * sizeof(evm_word_t)));
    CUDA_CHECK(cudaMalloc(&tmp_state_db->account_nonces, num_states * num_accounts * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&tmp_state_db->account_storage_size, num_states * num_accounts * sizeof(uint32_t)));

    CUDA_CHECK(cudaMalloc(&tmp_state_db->account_codes_size, num_accounts * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&tmp_state_db->account_codes_offset, num_accounts * sizeof(uint32_t)));

    // prealloc storage , only num_contracts are allocated
    CUDA_CHECK(cudaMalloc(&tmp_state_db->prealloc_keys_pool,
                          account_prealloc_keys_size * state_db_cpu->num_contracts * num_states * sizeof(evm_word_t)));
    CUDA_CHECK(cudaMalloc(&tmp_state_db->prealloc_values_pool,
                          account_prealloc_keys_size * state_db_cpu->num_contracts * num_states * sizeof(ValueStatus)));
    CUDA_CHECK(cudaMalloc(&tmp_state_db->all_account_codes, code_size * sizeof(uint8_t)));
    CUDA_CHECK(
        cudaMalloc(&tmp_state_db->dynamic_storage_pages, num_states * num_accounts * sizeof(StateDbStoragePage *)));
    CUDA_CHECK(cudaMalloc(&tmp_state_db->dynamic_pool_capacity, num_states * num_accounts * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&tmp_state_db->account_is_warm, num_states * num_accounts * sizeof(bool)));

    CUDA_CHECK(cudaMalloc(&tmp_state_db->dynamic_accounts, num_states * sizeof(DynamicAccount *)));
    CUDA_CHECK(cudaMalloc(&tmp_state_db->snapshot_accounts, num_states * num_accounts * sizeof(SnapshotAccount)));
    // CUDA_CHECK(cudaMalloc(&tmp_state_db->snapshot_total_storage_size, num_states * num_accounts *
    // sizeof(uint32_t))); Grouped memory copy
    CUDA_CHECK(cudaMemcpy(tmp_state_db->address_list, state_db_cpu->address_list, num_accounts * sizeof(evm_word_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_state_db->contract_index, state_db_cpu->contract_index, num_accounts * sizeof(int16_t),
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

    // prealloc storage , only num_contracts are allocated
    CUDA_CHECK(cudaMemcpy(tmp_state_db->prealloc_keys_pool, state_db_cpu->prealloc_keys_pool,
                          account_prealloc_keys_size * state_db_cpu->num_contracts * num_states * sizeof(evm_word_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_state_db->prealloc_values_pool, state_db_cpu->prealloc_values_pool,
                          account_prealloc_keys_size * state_db_cpu->num_contracts * num_states * sizeof(ValueStatus),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_state_db->all_account_codes, state_db_cpu->all_account_codes, code_size * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(tmp_state_db->dynamic_accounts, state_db_cpu->dynamic_accounts,
                          num_states * sizeof(DynamicAccount *), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(tmp_state_db->snapshot_accounts, state_db_cpu->snapshot_accounts,
                          num_states * num_accounts * sizeof(SnapshotAccount), cudaMemcpyHostToDevice));
    // dynamic storage
    CUDA_CHECK(cudaMemcpy(tmp_state_db->dynamic_pool_capacity, state_db_cpu->dynamic_pool_capacity,
                          num_states * num_accounts * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_state_db->account_is_warm, state_db_cpu->account_is_warm,
                          num_states * num_accounts * sizeof(bool), cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy(tmp_state_db->snapshot_total_storage_size, state_db_cpu->snapshot_total_storage_size,
    //                       num_states * num_accounts * sizeof(uint32_t), cudaMemcpyHostToDevice));

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
    state_db->num_contracts = 0;
    state_db->num_states = num_states;
    state_db->num_storage_elements = 0;
    state_db->storage_capacity = account_prealloc_keys_size * num_accounts;
    state_db->all_account_codes = nullptr;
    state_db->address_list = new evm_word_t[num_accounts];
    state_db->contract_index = new int16_t[num_accounts];

    state_db->account_nonces = new uint32_t[num_accounts * num_states];
    state_db->account_storage_size = new uint32_t[num_accounts * num_states];
    state_db->account_balances = new evm_word_t[num_accounts * num_states];

    state_db->account_codes_size = new uint32_t[num_accounts];
    state_db->account_codes_offset = new uint32_t[num_accounts];

    state_db->dynamic_accounts = new DynamicAccount *[num_states];

    // cpu side prealloc with num_accounts because we dont know the number of contracts yet.
    // GPU side will only use num_contracts
    state_db->prealloc_keys_pool = new evm_word_t[account_prealloc_keys_size * num_accounts * num_states];
    state_db->prealloc_values_pool = new ValueStatus[account_prealloc_keys_size * num_accounts * num_states];
    state_db->account_is_warm = new bool[num_states * num_accounts];
    // dynamic storage to grow later
    state_db->dynamic_storage_pages = new StateDbStoragePage *[num_states * num_accounts];
    state_db->dynamic_pool_capacity = new uint32_t[num_states * num_accounts];
    state_db->snapshot_accounts = new SnapshotAccount[num_states * num_accounts];
    // state_db->snapshot_total_storage_size = new uint32_t[num_states * num_accounts];
    uint32_t idx = 0;
    cJSON *account_json;
    uint32_t bytecode_offset = 0;

    memset(state_db->account_is_warm, 0, num_states * num_accounts * sizeof(bool));
    memset(state_db->dynamic_pool_capacity, 0, num_states * num_accounts * sizeof(uint32_t));

    memset(state_db->prealloc_keys_pool, 0,
           account_prealloc_keys_size * num_accounts * num_states * sizeof(evm_word_t));
    memset(state_db->prealloc_values_pool, 0,
           account_prealloc_keys_size * num_accounts * num_states * sizeof(ValueStatus));
    memset(state_db->dynamic_accounts, 0, num_states * sizeof(DynamicAccount *));
    // memset(state_db->snapshot_total_storage_size, 0, num_states * num_accounts * sizeof(uint32_t));

    cJSON_ArrayForEach(account_json, state_json) {
        cJSON *balance_json, *nonce_json;

        state_db->address_list[idx].from_hex(account_json->string);
        state_db->contract_index[idx] = -1;
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
            if ((byte_code.size > 0) || (storage_size > 0)) {
                state_db->contract_index[idx] = state_db->num_contracts;
                state_db->num_contracts++;
            }
            for (uint32_t i = 0; i < num_states; i++) {
                state_db->account_storage_size[idx * num_states + i] = storage_size;
            }
            state_db->num_storage_elements += state_db->account_storage_size[idx * num_states];
            // allocate new page per account
            if (storage_size > account_prealloc_keys_size) {
                // todo: handle storage size > account_prealloc_keys_size
                // clone num_states times of storage page
                printf("PreState storage size: %d greater than supported\n", storage_size);
                break;
            }
        } else
            memset(&state_db->account_storage_size[idx * num_states], 0, num_states * sizeof(uint32_t));

        for (uint32_t i = 0; i < state_db->account_storage_size[idx * num_states]; i++) {
            cJSON *storage_element_json = cJSON_GetArrayItem(storage_json, i);
            uint32_t contract_idx = state_db->num_contracts - 1;
            uint32_t pre_alloc_keys_idx = (account_prealloc_keys_size * contract_idx + i) * num_states;
            printf(" contract_idx: %d, pre_alloc_keys_idx: %d\n", contract_idx, pre_alloc_keys_idx);
            state_db->prealloc_keys_pool[pre_alloc_keys_idx].from_hex(storage_element_json->string);
            state_db->prealloc_values_pool[pre_alloc_keys_idx].from_hex(storage_element_json->valuestring);

            // duplicate the value for all states
            for (uint32_t j = 1; j < num_states; j++) {
                state_db->prealloc_keys_pool[pre_alloc_keys_idx + j] = state_db->prealloc_keys_pool[pre_alloc_keys_idx];
                state_db->prealloc_values_pool[pre_alloc_keys_idx + j] =
                    state_db->prealloc_values_pool[pre_alloc_keys_idx];
            }
        }

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
    /*
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
        memcpy(state_db_cpu->prealloc_keys_pool, state_db->prealloc_keys_pool,
               sizeof(evm_word_t) * account_prealloc_keys_size * num_accounts * num_states);
        memcpy(state_db_cpu->prealloc_values_pool, state_db->prealloc_values_pool,
               sizeof(ValueStatus) * account_prealloc_keys_size * num_accounts * num_states);
               */
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
            uint32_t contract_idx = contract_index[i];
            for (uint32_t j = 0; j < account_storage_size[i]; j++) {
                printf("\n key: \n");
                prealloc_keys_pool[(account_prealloc_keys_size * contract_idx + j) * num_states].print();
                printf("value: \n");
                prealloc_values_pool[(account_prealloc_keys_size * contract_idx + j) * num_states].print();
            }
            if (num_states > 1) {
                printf("\n state 2 \n");
                printf("keys size %d\n", account_storage_size[i]);
                for (uint32_t j = 0; j < account_storage_size[i]; j++) {
                    printf("\n key: \n");
                    prealloc_keys_pool[(account_prealloc_keys_size * contract_idx + j) * num_states + 1].print();
                    printf("value: \n");
                    prealloc_values_pool[(account_prealloc_keys_size * contract_idx + j) * num_states + 1].print();
                }
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
