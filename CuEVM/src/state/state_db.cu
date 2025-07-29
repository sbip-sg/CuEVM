#include <CuEVM/core/block_info.cuh>
#include <CuEVM/core/jump_table.cuh>
#include <CuEVM/state/state_db.cuh>
#include <CuEVM/utils/error_codes.cuh>
namespace CuEVM {

// return snapshot storage and the offset in the page
__device__ int32_t SnapshotState::find_dynamic_offset(ValueStatus *value) {
    // first page:
    uint32_t iter = (storage_size - memory_pool_snapshot_preallocate_slots) % snapshot_page_size;
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
__device__ void SnapshotState::print() const {
    printf("SnapshotState: %p\n", this);

    printf("  Storage Page Ptr: %p\n", storage_page);
    printf("  Next State Ptr: %p\n", next_state);
}

__device__ void SnapshotState::clear() {
    storage_size = 0;

    storage_page = nullptr;
    next_state = nullptr;
}

__device__ SnapshotState *SnapshotState::revert() {
#ifdef DEBUG_PERF
    printf("\n\nrevert snapshot_state %p next_state %p, storage_size %d touched_account_counts %d\n", this, next_state,
           storage_size, touched_account_counts);
    printf("revert touched account to cold num touched %d\n", touched_account_counts);
#endif
    if (touched_account_counts > 0) {
        for (uint32_t i = 0; i < touched_account_counts; i++) {
            int32_t address_index = preallocated_touched_accounts[i];
            if (address_index >= 0 && address_index < global_state_db_ptr->num_accounts) {
                // printf("set cold revert thread %d, address_index %d\n", INSTANCE_GLOBAL_IDX, address_index);
                uint32_t instance_idx = address_index * global_state_db_ptr->num_states + INSTANCE_GLOBAL_IDX;
                global_state_db_ptr->account_is_warm[instance_idx] = false;
            } else {
                // printf("todo negative address index set cold revert\n");

                // find the dynamic account
                DynamicAccount *dynamic_account = global_state_db_ptr->dynamic_accounts[INSTANCE_GLOBAL_IDX];
                while (dynamic_account != nullptr) {
                    if (dynamic_account->dynamic_account_index == address_index) {
                        dynamic_account->is_warm = false;
                        break;
                    }
                    dynamic_account = dynamic_account->next_account;
                }
            }
        }
    }
    if (diff_account_counts > 0) {
#ifdef DEBUG_PERF
        printf("revert account diff_account_counts %d\n", diff_account_counts);
#endif
        SnapshotAccount *current_account = accounts;
        while (current_account != nullptr) {
#ifdef DEBUG_PERF
            printf("revert account address_index %d\n", current_account->address_index);
#endif
            if (current_account->address_index >= 0 &&
                current_account->address_index < global_state_db_ptr->num_accounts) {
                uint32_t instance_idx =
                    current_account->address_index * global_state_db_ptr->num_states + INSTANCE_GLOBAL_IDX;
                global_state_db_ptr->account_balances[instance_idx] = current_account->balance;
            } else {
                DynamicAccount *dynamic_account = global_state_db_ptr->dynamic_accounts[INSTANCE_GLOBAL_IDX];
                while (dynamic_account != nullptr) {
                    if (dynamic_account->dynamic_account_index == current_account->address_index) {
                        dynamic_account->balance = current_account->balance;
                        break;
                    }
                    dynamic_account = dynamic_account->next_account;
                }
            }
            current_account = current_account->next_account;
        }
    }
    // N contract x mempool__snapshot_preallocate x num_states
    uint32_t start_offset = preallocated_offset;
#ifdef DEBUG_PERF
    printf("thread %d, state %p, start_offset %d\n", INSTANCE_GLOBAL_IDX, this, start_offset);
#endif
    uint32_t end_offset = min(start_offset + storage_size, memory_pool_snapshot_preallocate_slots);
    for (uint32_t i = start_offset; i < end_offset; i++) {
        uint32_t coalesced_offset = i * global_state_db_ptr->num_states + INSTANCE_GLOBAL_IDX;
        if (i < memory_pool_snapshot_preallocate_slots) {
#ifdef DEBUG_PERF
            if (INSTANCE_GLOBAL_IDX == 1) {
                printf("thread %d, state %p, i %d, coalesced_offset %d, preallocated snapshot current value \n",
                       INSTANCE_GLOBAL_IDX, this, i, coalesced_offset);
                CuEVM::memory_pool::preallocated_snapshot_values[coalesced_offset].value.print();
            }
#endif
            // Todo: check logic of this case
            if (CuEVM::memory_pool::preallocated_snapshot_restore_ptr[coalesced_offset] == nullptr) continue;
#ifdef DEBUG_PERF
            if (INSTANCE_GLOBAL_IDX == 1) {
                printf("thread %d, preallocated snapshot restore ptr %p\n", INSTANCE_GLOBAL_IDX,
                       CuEVM::memory_pool::preallocated_snapshot_restore_ptr[coalesced_offset]);
                printf("thread %d, preallocated snapshot restore value \n", INSTANCE_GLOBAL_IDX);
                CuEVM::memory_pool::preallocated_snapshot_restore_ptr[coalesced_offset]->value.print();
            }
#endif
            CuEVM::memory_pool::preallocated_snapshot_restore_ptr[coalesced_offset]->value =
                CuEVM::memory_pool::preallocated_snapshot_values[coalesced_offset].value;
            CuEVM::memory_pool::preallocated_snapshot_restore_ptr[coalesced_offset]->is_warm =
                CuEVM::memory_pool::preallocated_snapshot_values[coalesced_offset].is_warm;
        } else {
            SnapshotStoragePage *current_page = storage_page;
            while (current_page != nullptr) {
                for (uint32_t j = 0; j < snapshot_page_size; j++) {
                    if (current_page->restore_ptr[j] != nullptr) {
#ifdef DEBUG_PERF
                        printf("current page restore ptr %p\n", current_page->restore_ptr[j]);
                        printf("current page value \n");
                        current_page->restore_ptr[j]->value.print();
                        printf("current page restore value \n");
                        current_page->values[j].value.print();
                        printf("current page restore value is_warm %d\n", current_page->values[j].is_warm);
#endif
                        current_page->restore_ptr[j]->value = current_page->values[j].value;
                        current_page->restore_ptr[j]->is_warm = current_page->values[j].is_warm;
                    }
                }
                current_page = current_page->next_page;
            }
        }
    }
    // restore blank key
    for (uint32_t i = 0; i < touched_storage_counts; i++) {
        if (i < preallocated_touched_storage_keys_size) {
            global_state_db_ptr->reset_key(&address, &preallocated_touched_storage_keys[i]);
        } else {
            global_state_db_ptr->reset_key(&address,
                                           &dynamic_touched_storage_keys[i - preallocated_touched_storage_keys_size]);
        }
    }

    return next_state;
}

__device__ void SnapshotState::set_touched_account(const int32_t address_index) {
    // printf("snapshot %p set touched account thread %d, address_index %d, touched_account_counts %d\n", this,
    //        INSTANCE_GLOBAL_IDX, address_index, touched_account_counts);
    if (touched_account_counts < preallocated_touched_accounts_size) {
        preallocated_touched_accounts[touched_account_counts] = address_index;
    } else {
        // allocate dynamic touched accounts
        // todo: optimize
#ifdef DEBUG_PERF
        printf("allocate dynamic touched accounts\n");
#endif
        int32_t *new_dynamic_touched_accounts =
            new int32_t[touched_account_counts + 1 - preallocated_touched_accounts_size];
        for (uint32_t i = 0; i < touched_account_counts - preallocated_touched_accounts_size; i++) {
            new_dynamic_touched_accounts[i] = dynamic_touched_accounts[i];
        }
        dynamic_touched_accounts = new_dynamic_touched_accounts;
        dynamic_touched_accounts[touched_account_counts - preallocated_touched_accounts_size] = address_index;
    }
    touched_account_counts++;
}
__device__ void SnapshotState::set_blank_key(const evm_word_t *key) {
    int32_t offset = -1;
    for (uint32_t i = 0; i < touched_storage_counts; i++) {
        if (preallocated_touched_storage_keys[i] == *key) {
            offset = i;
            break;
        }
    }
    if (offset != -1) {
        return;
    }

    if (touched_storage_counts < preallocated_touched_storage_keys_size) {
        preallocated_touched_storage_keys[touched_storage_counts] = *key;
    } else {
        for (uint32_t i = 0; i < touched_storage_counts - preallocated_touched_storage_keys_size; i++) {
            if (dynamic_touched_storage_keys[i] == *key) {
                offset = i;
                return;
            }
        }

        evm_word_t *new_dynamic_touched_storage_keys =
            new evm_word_t[touched_storage_counts - preallocated_touched_storage_keys_size + 1];
        for (uint32_t i = 0; i < touched_storage_counts - preallocated_touched_storage_keys_size; i++) {
            new_dynamic_touched_storage_keys[i] = dynamic_touched_storage_keys[i];
        }
        dynamic_touched_storage_keys = new_dynamic_touched_storage_keys;
        new_dynamic_touched_storage_keys[touched_storage_counts - preallocated_touched_storage_keys_size] = *key;
    }
    // printf("set blank key thread %d, touched_storage_counts %d\n", INSTANCE_GLOBAL_IDX, touched_storage_counts);
    // key->print();
    touched_storage_counts++;
}

__device__ void SnapshotState::set_storage(ValueStatus *src_value) {
    if (this == nullptr) {
        return;
    }
    // uint32_t start_offset = this->preallocated_offset;
    // uint32_t end_offset = min(start_offset + storage_size, memory_pool_snapshot_preallocate_slots);
    // for (uint32_t i = start_offset; i < end_offset; i++) {
    //     uint32_t coalesced_offset = i * global_state_db_ptr->num_states + INSTANCE_GLOBAL_IDX;
    //     if (CuEVM::memory_pool::preallocated_snapshot_restore_ptr[coalesced_offset] == src_value) return;
    // }
    uint32_t next_offset;
    if (preallocated_offset < memory_pool_snapshot_preallocate_slots) {
        next_offset = CuEVM::memory_pool::get_next_snapshot_offset();
    } else {
        next_offset = preallocated_offset + storage_size;
    }
    // printf(
    //     "set storage thread %d, snapshot_state %p, next_offset %d, preallocated_offset %d, storage_size %d src_value
    //     "
    //     "%p\n",
    //     INSTANCE_GLOBAL_IDX, this, next_offset, preallocated_offset, storage_size, src_value);
    if (next_offset >= memory_pool_snapshot_preallocate_slots) {
        uint32_t dynamic_offset = find_dynamic_offset(src_value);
        if (dynamic_offset == -1) {
            dynamic_offset = (next_offset - memory_pool_snapshot_preallocate_slots) % snapshot_page_size;
            // printf("next_offset %d memory_pool_snapshot_preallocate_slots %d snapshot_page_size %d dynamic_offset
            // %d\n",
            //        next_offset, memory_pool_snapshot_preallocate_slots, snapshot_page_size, dynamic_offset);
            if (dynamic_offset == 0) {
#ifdef DEBUG_PERF
                printf("allocate new page %p\n", storage_page);
#endif
                SnapshotStoragePage *new_page = new SnapshotStoragePage();
                new_page->next_page = storage_page;
                storage_page = new_page;
            }
            storage_page->restore_ptr[dynamic_offset] = src_value;
            storage_page->values[dynamic_offset].value = src_value->value;
            storage_page->values[dynamic_offset].is_warm = src_value->is_warm;
#ifdef DEBUG_PERF
            printf("to restore value %p, is_warm %d\n", storage_page->restore_ptr[dynamic_offset],
                   storage_page->values[dynamic_offset].is_warm);
#endif
        }
    } else {
        uint32_t coalesced_offset = next_offset * global_state_db_ptr->num_states + INSTANCE_GLOBAL_IDX;
        // printf("set storage thread %d, coalesced_offset %d to write src %p\n", INSTANCE_GLOBAL_IDX, coalesced_offset,
        //        src_value);
        CuEVM::memory_pool::preallocated_snapshot_restore_ptr[coalesced_offset] = src_value;
        CuEVM::memory_pool::preallocated_snapshot_values[coalesced_offset].value = src_value->value;
        CuEVM::memory_pool::preallocated_snapshot_values[coalesced_offset].is_warm = src_value->is_warm;
        // printf("SnapshotState set storage thread %d , offset %d,  restore ptr value %p\n", INSTANCE_GLOBAL_IDX,
        //        offset, CuEVM::memory_pool::preallocated_snapshot_restore_ptr[offset]);
        // printf("current value \n");
        // CuEVM::memory_pool::preallocated_snapshot_values[offset].value.print();
        // printf("To be restored value \n");
        // src_value->value.print();
    }
    storage_size++;
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
    // this->snapshot_account = new DynamicSnapshotState();
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
__device__ void DynamicAccount::set_storage(const evm_word_t *key, const evm_word_t *value, bool is_warm) {
    ValueStatus *found_value = get_value_status(key);
    if (found_value == nullptr) {
        // printf("not found in dynamic account\n");
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
        // printf("set storage thread %d, address %p, key %p, value %p, is_warm %d\n", INSTANCE_GLOBAL_IDX, address,
        // key,
        //        value, is_warm);
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
    global_jump_table = nullptr;
}

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
__device__ DynamicAccount *StateDb::get_dynamic_account_and_set_warm(const evm_word_t *address) const {
    // TODO: implement
    DynamicAccount *current_account = dynamic_accounts[INSTANCE_GLOBAL_IDX];
    // printf("get dynamic account thread %d, address %p current_account %p\n", INSTANCE_GLOBAL_IDX, address,
    //        current_account);
    while (current_account != nullptr) {
        // printf("current_account address %p\n", current_account->address);
        // current_account->address.print();
        if (current_account->address == *address) {
            current_account->is_warm = true;
            return current_account;
        }
        current_account = current_account->next_account;
    }
    // printf("End of get dynamic account, failed to find \n");
    // printf("create new dynamic account\n");
    DynamicAccount *new_acc = new DynamicAccount(address, 0, 0, 0, 0, nullptr);
    new_acc->is_warm = true;
    new_acc->next_account = dynamic_accounts[INSTANCE_GLOBAL_IDX];
    dynamic_accounts[INSTANCE_GLOBAL_IDX] = new_acc;
    return new_acc;
}

__device__ DynamicAccount *StateDb::get_dynamic_account(const evm_word_t *address) const {
    // TODO: implement
    DynamicAccount *current_account = dynamic_accounts[INSTANCE_GLOBAL_IDX];
    // printf("get dynamic account thread %d, address %p current_account %p\n", INSTANCE_GLOBAL_IDX, address,
    //        current_account);
    while (current_account != nullptr) {
        // printf("current_account address %p\n", current_account->address);
        // current_account->address.print();
        if (current_account->address == *address) {
            return current_account;
        }
        current_account = current_account->next_account;
    }
    // printf("End of get dynamic account, failed to find \n");

    return nullptr;
}

__device__ DynamicAccount *StateDb::new_account(const evm_word_t *address, const evm_word_t *balance,
                                                const uint32_t nonce, const uint32_t code_size, uint8_t *code) {
    // printf("Create new dynamic account \n");
    DynamicAccount *new_acc = new DynamicAccount(address, balance, nonce, 0, code_size, code);

    new_acc->next_account = dynamic_accounts[INSTANCE_GLOBAL_IDX];

    // dynamic account list has index from -1 to -N
    if (dynamic_accounts[INSTANCE_GLOBAL_IDX] == nullptr) {
        new_acc->dynamic_account_index = -1;
    } else {
        new_acc->dynamic_account_index = dynamic_accounts[INSTANCE_GLOBAL_IDX]->dynamic_account_index - 1;
    }
    dynamic_accounts[INSTANCE_GLOBAL_IDX] = new_acc;
    return new_acc;
}
__device__ int32_t StateDb::create_contract(const evm_word_t *address, const uint32_t code_size, uint8_t *code) {
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
        dynamic_account = StateDb::new_account(address, 0, 0, code_size, code);
    }
    dynamic_account->code_size = code_size;
    dynamic_account->code = code;
    return ERROR_SUCCESS;
}
__device__ void StateDb::update_account(const evm_word_t *address, const evm_word_t *balance, const uint32_t nonce) {
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        DynamicAccount *dynamic_account = get_dynamic_account(address);
        if (dynamic_account == nullptr) {
            dynamic_account = StateDb::new_account(address, balance, nonce, 0, nullptr);
        }
        dynamic_account->balance = *balance;
        dynamic_account->nonce = nonce;
        return;
    }
    uint32_t instance_idx = address_index * num_states + INSTANCE_GLOBAL_IDX;
    account_balances[instance_idx] = *balance;
    account_nonces[instance_idx] = nonce;
}
__device__ void StateDb::update_code(const evm_word_t *address, const uint32_t code_size, uint8_t *code) {
    int32_t address_index = get_address_index(address);
    if (address_index != -1) {
        address_list[address_index] = 1;  // non-collision address, precompiled
        contract_index[address_index] = -1;
    }
    DynamicAccount *dynamic_account = get_dynamic_account(address);
    if (dynamic_account == nullptr) {
        dynamic_account = StateDb::new_account(address, 0, 0, code_size, code);
    }
    dynamic_account->code_size = code_size;
    // clone code to dynamic account
    dynamic_account->is_warm = true;
    dynamic_account->code = new uint8_t[code_size];
    memcpy(dynamic_account->code, code, code_size);
}
__device__ void StateDb::set_balance(const evm_word_t *address, const evm_word_t *balance, bool is_warm) {
    // printf("update balance thread %d, address %p\n", INSTANCE_GLOBAL_IDX, address);
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        DynamicAccount *dynamic_account = get_dynamic_account(address);
        if (dynamic_account == nullptr) {
            dynamic_account = StateDb::new_account(address, balance, 0, 0, nullptr);
        }
        dynamic_account->balance = *balance;
        dynamic_account->is_warm = is_warm;
        return;
    }
    account_balances[address_index * num_states + INSTANCE_GLOBAL_IDX] = *balance;
    account_is_warm[address_index * num_states + INSTANCE_GLOBAL_IDX] = is_warm;
}

__device__ void StateDb::increase_balance(const evm_word_t *address, const evm_word_t *balance,
                                          SnapshotState *snapshot_state, bool is_warm) {
    // printf("update balance thread %d, address %p\n", INSTANCE_GLOBAL_IDX, address);

    int32_t address_index = get_address_index(address);
    evm_word_t *current_balance;
    if (address_index == -1) {
        DynamicAccount *dynamic_account = get_dynamic_account(address);
        if (dynamic_account == nullptr) {
            evm_word_t zero = 0;
            dynamic_account = StateDb::new_account(address, &zero, 0, 0, nullptr);
        }
        address_index = dynamic_account->dynamic_account_index;
        current_balance = &dynamic_account->balance;
        dynamic_account->is_warm = is_warm;
    } else {
        current_balance = &account_balances[address_index * num_states + INSTANCE_GLOBAL_IDX];
        account_is_warm[address_index * num_states + INSTANCE_GLOBAL_IDX] = is_warm;
    }
    if (snapshot_state != nullptr) {
        // set snapshot of the balance
        SnapshotAccount *snapshot_account = new SnapshotAccount();
        snapshot_account->address_index = address_index;
        snapshot_account->balance = *current_balance;
        snapshot_account->next_account = snapshot_state->accounts;
        snapshot_state->accounts = snapshot_account;
        snapshot_state->diff_account_counts++;
    }
    // if (INSTANCE_GLOBAL_IDX == 0) {
    //     printf("increase balance \n");
    //     address->print();
    //     current_balance->print();
    //     balance->print();
    // }
    uint256_add(current_balance, current_balance, balance);
}
// shortcut for deducting balance
__device__ int32_t StateDb::deduct_balance(const evm_word_t *address, const evm_word_t *amount,
                                           SnapshotState *snapshot_state, bool set_warm) {
    // printf("deduct balance thread %d\n", INSTANCE_GLOBAL_IDX);
    // address->print();

    int32_t address_index = get_address_index(address);
    // printf("address index %d instance %d\n", address_index, INSTANCE_GLOBAL_IDX);
    evm_word_t *current_balance;
    if (address_index == -1) {
        DynamicAccount *dynamic_account = get_dynamic_account(address);
        if (dynamic_account == nullptr) {
            return ERROR_INSUFFICIENT_FUNDS;
        }
        address_index = dynamic_account->dynamic_account_index;
        current_balance = &dynamic_account->balance;
    } else {
        if (set_warm) {
            account_is_warm[address_index * num_states + INSTANCE_GLOBAL_IDX] = true;
        }
        current_balance = &account_balances[address_index * num_states + INSTANCE_GLOBAL_IDX];
    }

    if (uint256_cmp(current_balance, amount) < 0) {
        // printf("deduct balance insufficient funds instance %d\n", INSTANCE_GLOBAL_IDX);
        return ERROR_INSUFFICIENT_FUNDS;
    }

    // set snapshot of the balance
    if (snapshot_state != nullptr) {
        SnapshotAccount *snapshot_account = new SnapshotAccount();
        snapshot_account->address_index = address_index;
        snapshot_account->balance = *current_balance;
        snapshot_account->next_account = snapshot_state->accounts;
        snapshot_state->accounts = snapshot_account;
        snapshot_state->diff_account_counts++;
    }
    // if (INSTANCE_GLOBAL_IDX == 0) {
    //     printf("deduct balance \n");
    //     address->print();
    //     current_balance->print();
    //     amount->print();
    // }

    uint256_sub(current_balance, current_balance, amount);

    return ERROR_SUCCESS;
}

__device__ int32_t StateDb::deduct_balance_sender(const evm_word_t *address, const evm_word_t *amount) {
    // sender always must have balance and in static statedb
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        return ERROR_INSUFFICIENT_FUNDS;
    }
    uint32_t instance_idx = address_index * num_states + INSTANCE_GLOBAL_IDX;
    // printf("address index %d instance %d\n", address_index, INSTANCE_GLOBAL_IDX);
    evm_word_t *current_balance;

    account_is_warm[instance_idx] = true;
    account_nonces[instance_idx]++;
    current_balance = &account_balances[instance_idx];
    if (uint256_cmp(current_balance, amount) < 0) {
        // printf("deduct balance insufficient funds instance %d\n", INSTANCE_GLOBAL_IDX);
        return ERROR_INSUFFICIENT_FUNDS;
    }
    uint256_sub(current_balance, current_balance, amount);
    return ERROR_SUCCESS;
}

__device__ void StateDb::update_nonce(const evm_word_t *address, const uint32_t nonce) {
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        DynamicAccount *dynamic_account = get_dynamic_account(address);
        if (dynamic_account == nullptr) {
            dynamic_account = StateDb::new_account(address, 0, nonce, 0, nullptr);
        }
        dynamic_account->nonce = nonce;
        return;
    }
    account_nonces[address_index * num_states + INSTANCE_GLOBAL_IDX] = nonce;
}

__device__ int32_t StateDb::transfer(const evm_word_t *sender, const evm_word_t *recipient, const evm_word_t *value,
                                     SnapshotState *snapshot_state) {
    int32_t error_code = deduct_balance(sender, value, snapshot_state);
    // printf("transfer error code %d\n", error_code);
    if (error_code != ERROR_SUCCESS) return error_code;
    increase_balance(recipient, value, snapshot_state);
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

__device__ void StateDb::reset_key(const evm_word_t *address, const evm_word_t *key) {
#ifdef DEBUG_PERF
    printf("state db reset key thread %d, address %p, key %p\n", INSTANCE_GLOBAL_IDX, address, key);
    if (INSTANCE_GLOBAL_IDX == 0) {
        printf("address\n");
        address->print();
        printf("key\n");
        key->print();
    }
#endif

    int32_t address_index = global_state_db_ptr->get_address_index(address);
    // for revert logic, the account must be found , and the key must exist
    ValueStatus *found_value = nullptr;

    if (address_index == -1) {
        found_value = nullptr;
        DynamicAccount *dynamic_account = global_state_db_ptr->get_dynamic_account(address);
        if (dynamic_account != nullptr) {
            found_value = dynamic_account->get_value_status(key);
            if (found_value != nullptr) {
                found_value->is_warm = false;
                found_value->value = 0;
            }
        }
        return;
    }

    found_value = global_state_db_ptr->get_value_status(address_index, key);
    if (found_value != nullptr) {
        found_value->is_warm = false;
        found_value->value = 0;
    }
}

__device__ void StateDb::write_storage_with_known_index(const evm_word_t *address, const evm_word_t *key,
                                                        const evm_word_t *value, int32_t address_index,
                                                        ValueStatus *found_value, bool is_warm) {
    // printf("write storage with known index thread %d, address %p\n", INSTANCE_GLOBAL_IDX, address);
    if (address_index == -1) {
        DynamicAccount *dynamic_account = get_dynamic_account(address);
        // printf("dynamic_account %p\n", dynamic_account);
        if (dynamic_account == nullptr) {
            return;  // should never write directly to storage before creating account
        }
        dynamic_account->set_storage(key, value, is_warm);
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
            // printf("storage size %d, new offset %d\n", storage_size, new_offset);
            prealloc_keys_pool[new_offset] = *key;
            prealloc_values_pool[new_offset].set_value(value, is_warm);

            account_storage_size[instance_idx]++;
            // num_storage_elements++;

        } else {
#ifdef DEBUG_PERF
            printf("Dynamic storage grow\n");
#endif
            ValueStatus *dynamic_value = get_dynamic_value_location(storage_size, instance_idx, key);
            if (dynamic_value == nullptr) {
                dynamic_value = grow_storage_and_set_key(storage_size, instance_idx, key);
                dynamic_value->set_value(value, is_warm);
                account_storage_size[instance_idx]++;
                // num_storage_elements++;
            }
        }

    } else {
        // write to exsiting slot // already written when calling is_warm_key_with_offset
        // if (snapshot_state != nullptr) snapshot_state->set_storage(found_value);

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

__device__ void StateDb::init_snapshot(evm_call_context_t *call_context, const uint16_t depth,
                                       const evm_word_t *address) {
    uint32_t address_index = get_address_index(address);
    if (address_index == -1) {
#ifdef DEBUG_PERF
        printf("init_snapshot address not found in state db\n");
        address->print();
#endif
        SnapshotState *tmp = CuEVM::memory_pool::get_snapshot_state();
        tmp->address = *address;
        tmp->storage_size = 0;
        tmp->touched_account_counts = 0;
        tmp->preallocated_offset = memory_pool_snapshot_preallocate_slots;
        call_context->snapshot_state = tmp;
        tmp->next_state = nullptr;
        return;
    }
    uint32_t instance_idx = address_index * num_states + INSTANCE_GLOBAL_IDX;
    //

    SnapshotState *tmp = CuEVM::memory_pool::get_snapshot_state();
    // printf("tmp %p\n", tmp);
    //
    call_context->snapshot_state = tmp;
    tmp->address = *address;
    tmp->storage_size = 0;
    tmp->touched_account_counts = 0;
    tmp->diff_account_counts = 0;
    tmp->preallocated_offset = min(CuEVM::memory_pool::global_memory_pool->snapshot_slot_counts[INSTANCE_GLOBAL_IDX],
                                   memory_pool_snapshot_preallocate_slots);
    // printf("thread %d, state %p, preallocated_offset %d\n", INSTANCE_GLOBAL_IDX, this, tmp->preallocated_offset);
    tmp->next_state = nullptr;
}

__device__ evm_word_t *StateDb::get_storage_with_known_index(const evm_word_t *address, const evm_word_t *key,
                                                             int32_t address_index, ValueStatus *found_value,
                                                             bool set_warm) {
    uint32_t instance_idx = address_index * num_states + INSTANCE_GLOBAL_IDX;
    if (address_index == -1) {
        DynamicAccount *dynamic_account = get_dynamic_account(address);
        if (dynamic_account == nullptr) {
            return nullptr;
        }
        found_value = dynamic_account->get_value_status(key);
        if (found_value == nullptr) {
            if (set_warm) {
                // insert new value
                evm_word_t zero = 0;
                dynamic_account->set_storage(key, &zero, true);
            }
            return nullptr;
        }

        found_value->is_warm = set_warm;
        return &found_value->value;
    }

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
#ifdef DEBUG_PERF
                printf("Dynamic storage grow in get_storage_with_known_index\n");
#endif
                ValueStatus *value_status = grow_storage_and_set_key(storage_size, instance_idx, key);
                value_status->set_value(&zero, true);
            }
            account_storage_size[instance_idx]++;
        }
        return nullptr;
    } else {
        found_value->is_warm = set_warm;
        account_is_warm[instance_idx] |= set_warm;
        return &found_value->value;
    }
}

__device__ ValueStatus *StateDb::get_value_status(const int32_t address_index, const evm_word_t *key) const {
    uint32_t instance_idx = address_index * num_states + INSTANCE_GLOBAL_IDX;
    // printf("get_value_status address_index %d, instance_idx %d instance %d\n", address_index, instance_idx,
    //        INSTANCE_GLOBAL_IDX);
    uint32_t contract_idx = contract_index[address_index];
    uint32_t storage_size = account_storage_size[instance_idx];
    // printf("get_value_status address_index %d, storage_size %d instance %d\n", address_index, storage_size,
    //        INSTANCE_GLOBAL_IDX);
    int32_t key_offset = get_value_offset(storage_size, contract_idx, key);
    // printf("get_value_status key_offset %d instance %d\n", key_offset, INSTANCE_GLOBAL_IDX);

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
        // printf("found thread %d key offset %d key %p\n", THREADIDX, key_offset,
        // &prealloc_values_pool[key_offset]);
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

__device__ evm_word_t *StateDb::get_balance(const evm_word_t *address, bool set_warm) {
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        DynamicAccount *dynamic_account =
            set_warm ? get_dynamic_account_and_set_warm(address) : get_dynamic_account(address);
        if (dynamic_account == nullptr) {
            return nullptr;
        }
        return &dynamic_account->balance;
    }
    if (set_warm) {
        account_is_warm[address_index * num_states + INSTANCE_GLOBAL_IDX] = true;
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

__device__ uint8_t *StateDb::get_code(uint32_t &code_size, const evm_word_t *address, bool set_warm) {
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        DynamicAccount *dynamic_account = get_dynamic_account(address);
        if (dynamic_account == nullptr) {
            return nullptr;
        }
        // todo implement revert warm state for dynamic account
        if (set_warm) {
            dynamic_account->is_warm = true;
        }
        code_size = dynamic_account->code_size;
        // printf("get_code dynamic, code_size %d, code %p\n", code_size, dynamic_account->code);
        return dynamic_account->code;
    }

    code_size = account_codes_size[address_index];
    // printf("get_code address_index %d, code_size %d\n", address_index, code_size);
    if (set_warm) {
        account_is_warm[address_index * num_states + INSTANCE_GLOBAL_IDX] = true;
    }
    return &all_account_codes[account_codes_offset[address_index]];
}

__device__ void StateDb::set_warm_account(const evm_word_t *address) {
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        DynamicAccount *dynamic_account = get_dynamic_account(address);
        if (dynamic_account == nullptr) {
            DynamicAccount *new_acc = StateDb::new_account(address, 0, 0, 0, nullptr);

            dynamic_account = new_acc;
        }
        dynamic_account->is_warm = true;
    } else
        account_is_warm[address_index * num_states + INSTANCE_GLOBAL_IDX] = true;
}

__device__ void StateDb::set_warm_key(const evm_word_t *address, const evm_word_t *key) {
    // todo :implement
}

__device__ bool StateDb::is_warm_account(const evm_word_t *address, SnapshotState *snapshot_state, bool set_warm) {
    if (address->is_precompile()) return true;  // precompile contracts are warm
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        // if coinbase is in "pre state" it is set warm before executing kernel
        if (uint256_cmp(address, &CuEVM::global_block_info->coin_base) == 0) return true;
        DynamicAccount *dynamic_account = get_dynamic_account(address);

        if (dynamic_account == nullptr) {
            if (set_warm) {
                dynamic_account = new_account(address, 0, 0, 0, nullptr);
                dynamic_account->is_warm = true;

                if (snapshot_state != nullptr) {
                    // set the snapshot state
                    snapshot_state->set_touched_account(dynamic_account->dynamic_account_index);
                }
            }
            return false;
        } else {
            if (set_warm) {
                bool temp = dynamic_account->is_warm;
                dynamic_account->is_warm = set_warm;
                return temp;
            }
            return dynamic_account->is_warm;
        }
    }
    uint32_t instance_idx = address_index * num_states + INSTANCE_GLOBAL_IDX;
    if (set_warm && account_is_warm[instance_idx] == false) {
        if (snapshot_state != nullptr) {
            snapshot_state->set_touched_account(address_index);
        }
        account_is_warm[instance_idx] = true;
        return false;
    }
    return account_is_warm[instance_idx];
}

__device__ bool StateDb::is_warm_key_with_offset(const evm_word_t *address, const evm_word_t *key,
                                                 int32_t &address_index, ValueStatus *&found_value,
                                                 SnapshotState *snapshot_state, bool write_snapshot) {
    address_index = get_address_index(address);
    if (address_index == -1) {
        found_value = nullptr;
#ifdef DEBUG_PERF
        printf("Dynamic storage grow\n");
#endif
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
        // todo: // implement value not found in both pools
        // printf("is_warm_key_with_offset value not found in both pools\n");
        if (snapshot_state != nullptr) {
            // printf("set snapshot_state %p to restore null_ptr\n", snapshot_state);
            snapshot_state->set_blank_key(key);
        }
        return false;
    }
    if (snapshot_state != nullptr && (found_value->is_warm == false || write_snapshot)) {
        // only take snapshot if the value is cold -> warm
        snapshot_state->set_storage(found_value);
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
    return (account_nonces[instance_idx] == 0 && uint256_is_zero(&account_balances[instance_idx]) &&
            account_codes_size[address_index] == 0);
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
#ifdef DEBUG_PERF
            printf("dynamic account is null, create new account\n");
#endif
            DynamicAccount *new_acc = new_account(address, 0, 0, 0, nullptr);
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
        DynamicAccount *new_acc = new_account(address, &account_balances[instance_idx], 0, 0, nullptr);

        return true;
    }
    return false;
}
__device__ bool StateDb::is_contract(const evm_word_t *address) const {
    // todo :implement
    return true;
}

__host__ void StateDb::GPUfromJson(StateDb *&state_db, const cJSON *state_json, uint32_t num_states,
                                   uint32_t &num_accounts
#ifdef BUILD_GO_LIBRARY
                                   ,
                                   StateDb *&snapshot_state_db
#endif
) {
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
    CUDA_CHECK(cudaMalloc(&tmp_state_db->global_jump_table, sizeof(GlobalJumpTable)));
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

    // CUDA_CHECK(cudaMalloc(&tmp_state_db->snapshot_total_storage_size, num_states * num_accounts *
    // sizeof(uint32_t))); Grouped memory copy
    CUDA_CHECK(cudaMemcpy(tmp_state_db->global_jump_table, state_db_cpu->global_jump_table, sizeof(GlobalJumpTable),
                          cudaMemcpyHostToDevice));
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

    // dynamic storage
    CUDA_CHECK(cudaMemcpy(tmp_state_db->dynamic_pool_capacity, state_db_cpu->dynamic_pool_capacity,
                          num_states * num_accounts * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_state_db->account_is_warm, state_db_cpu->account_is_warm,
                          num_states * num_accounts * sizeof(bool), cudaMemcpyHostToDevice));

#ifdef BUILD_GO_LIBRARY
    StateDb *tmp_snapshot_state_db = new StateDb(num_states);
    tmp_snapshot_state_db->num_accounts = state_db_cpu->num_accounts;
    tmp_snapshot_state_db->num_states = state_db_cpu->num_states;
    tmp_snapshot_state_db->num_storage_elements = state_db_cpu->num_storage_elements;
    tmp_snapshot_state_db->storage_capacity = state_db_cpu->storage_capacity;
    tmp_snapshot_state_db->num_contracts = state_db_cpu->num_contracts;

    // Grouped memory allocation
    CUDA_CHECK(cudaMalloc(&tmp_snapshot_state_db->global_jump_table, sizeof(GlobalJumpTable)));
    CUDA_CHECK(cudaMalloc(&tmp_snapshot_state_db->address_list, num_accounts * sizeof(evm_word_t)));
    CUDA_CHECK(cudaMalloc(&tmp_snapshot_state_db->contract_index, num_accounts * sizeof(int16_t)));

    CUDA_CHECK(cudaMalloc(&tmp_snapshot_state_db->account_balances, num_states * num_accounts * sizeof(evm_word_t)));
    CUDA_CHECK(cudaMalloc(&tmp_snapshot_state_db->account_nonces, num_states * num_accounts * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&tmp_snapshot_state_db->account_storage_size, num_states * num_accounts * sizeof(uint32_t)));

    CUDA_CHECK(cudaMalloc(&tmp_snapshot_state_db->account_codes_size, num_accounts * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&tmp_snapshot_state_db->account_codes_offset, num_accounts * sizeof(uint32_t)));

    // prealloc storage , only num_contracts are allocated
    CUDA_CHECK(cudaMalloc(&tmp_snapshot_state_db->prealloc_keys_pool,
                          account_prealloc_keys_size * state_db_cpu->num_contracts * num_states * sizeof(evm_word_t)));
    CUDA_CHECK(cudaMalloc(&tmp_snapshot_state_db->prealloc_values_pool,
                          account_prealloc_keys_size * state_db_cpu->num_contracts * num_states * sizeof(ValueStatus)));
    CUDA_CHECK(cudaMalloc(&tmp_snapshot_state_db->all_account_codes, code_size * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&tmp_snapshot_state_db->dynamic_storage_pages,
                          num_states * num_accounts * sizeof(StateDbStoragePage *)));
    CUDA_CHECK(cudaMalloc(&tmp_snapshot_state_db->dynamic_pool_capacity, num_states * num_accounts * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&tmp_snapshot_state_db->account_is_warm, num_states * num_accounts * sizeof(bool)));

    CUDA_CHECK(cudaMalloc(&tmp_snapshot_state_db->dynamic_accounts, num_states * sizeof(DynamicAccount *)));

    // CUDA_CHECK(cudaMalloc(&tmp_state_db->snapshot_total_storage_size, num_states * num_accounts *
    // sizeof(uint32_t))); Grouped memory copy
    CUDA_CHECK(cudaMemcpy(tmp_snapshot_state_db->global_jump_table, state_db_cpu->global_jump_table,
                          sizeof(GlobalJumpTable), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_snapshot_state_db->address_list, state_db_cpu->address_list,
                          num_accounts * sizeof(evm_word_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_snapshot_state_db->contract_index, state_db_cpu->contract_index,
                          num_accounts * sizeof(int16_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(tmp_snapshot_state_db->account_balances, state_db_cpu->account_balances,
                          num_states * num_accounts * sizeof(evm_word_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_snapshot_state_db->account_nonces, state_db_cpu->account_nonces,
                          num_states * num_accounts * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_snapshot_state_db->account_storage_size, state_db_cpu->account_storage_size,
                          num_states * num_accounts * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_snapshot_state_db->account_codes_size, state_db_cpu->account_codes_size,
                          num_accounts * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_snapshot_state_db->account_codes_offset, state_db_cpu->account_codes_offset,
                          num_accounts * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // prealloc storage , only num_contracts are allocated
    CUDA_CHECK(cudaMemcpy(tmp_snapshot_state_db->prealloc_keys_pool, state_db_cpu->prealloc_keys_pool,
                          account_prealloc_keys_size * state_db_cpu->num_contracts * num_states * sizeof(evm_word_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_snapshot_state_db->prealloc_values_pool, state_db_cpu->prealloc_values_pool,
                          account_prealloc_keys_size * state_db_cpu->num_contracts * num_states * sizeof(ValueStatus),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_snapshot_state_db->all_account_codes, state_db_cpu->all_account_codes,
                          code_size * sizeof(uint8_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(tmp_snapshot_state_db->dynamic_accounts, state_db_cpu->dynamic_accounts,
                          num_states * sizeof(DynamicAccount *), cudaMemcpyHostToDevice));

    // dynamic storage
    CUDA_CHECK(cudaMemcpy(tmp_snapshot_state_db->dynamic_pool_capacity, state_db_cpu->dynamic_pool_capacity,
                          num_states * num_accounts * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_snapshot_state_db->account_is_warm, state_db_cpu->account_is_warm,
                          num_states * num_accounts * sizeof(bool), cudaMemcpyHostToDevice));

#endif
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
#ifdef BUILD_GO_LIBRARY
    StateDb *snapshot_state_db_gpu;
    CUDA_CHECK(cudaMalloc(&snapshot_state_db_gpu, sizeof(StateDb)));
    CUDA_CHECK(cudaMemcpy(snapshot_state_db_gpu, tmp_snapshot_state_db, sizeof(StateDb), cudaMemcpyHostToDevice));
    delete tmp_snapshot_state_db;
    snapshot_state_db = snapshot_state_db_gpu;
#endif
}

__host__ void StateDb::GPUfromJsonMultiGPU(std::vector<StateDb *> &state_db, const cJSON *state_json,
                                           uint32_t num_states, uint32_t &num_accounts,
                                           std::vector<StateDb *> &snapshot_state_db) {
    StateDb *state_db_cpu = new StateDb(num_states);
    StateDb::CPUfromJson(state_db_cpu, state_json, num_states);

    for (int i = 0; i < state_db.size(); i++) {
        CUDA_CHECK(cudaSetDevice(i));
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
        CUDA_CHECK(cudaMalloc(&tmp_state_db->global_jump_table, sizeof(GlobalJumpTable)));
        CUDA_CHECK(cudaMalloc(&tmp_state_db->address_list, num_accounts * sizeof(evm_word_t)));
        CUDA_CHECK(cudaMalloc(&tmp_state_db->contract_index, num_accounts * sizeof(int16_t)));

        CUDA_CHECK(cudaMalloc(&tmp_state_db->account_balances, num_states * num_accounts * sizeof(evm_word_t)));
        CUDA_CHECK(cudaMalloc(&tmp_state_db->account_nonces, num_states * num_accounts * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&tmp_state_db->account_storage_size, num_states * num_accounts * sizeof(uint32_t)));

        CUDA_CHECK(cudaMalloc(&tmp_state_db->account_codes_size, num_accounts * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&tmp_state_db->account_codes_offset, num_accounts * sizeof(uint32_t)));

        // prealloc storage , only num_contracts are allocated
        CUDA_CHECK(
            cudaMalloc(&tmp_state_db->prealloc_keys_pool,
                       account_prealloc_keys_size * state_db_cpu->num_contracts * num_states * sizeof(evm_word_t)));
        CUDA_CHECK(
            cudaMalloc(&tmp_state_db->prealloc_values_pool,
                       account_prealloc_keys_size * state_db_cpu->num_contracts * num_states * sizeof(ValueStatus)));
        CUDA_CHECK(cudaMalloc(&tmp_state_db->all_account_codes, code_size * sizeof(uint8_t)));
        CUDA_CHECK(
            cudaMalloc(&tmp_state_db->dynamic_storage_pages, num_states * num_accounts * sizeof(StateDbStoragePage *)));
        CUDA_CHECK(cudaMalloc(&tmp_state_db->dynamic_pool_capacity, num_states * num_accounts * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&tmp_state_db->account_is_warm, num_states * num_accounts * sizeof(bool)));

        CUDA_CHECK(cudaMalloc(&tmp_state_db->dynamic_accounts, num_states * sizeof(DynamicAccount *)));

        // CUDA_CHECK(cudaMalloc(&tmp_state_db->snapshot_total_storage_size, num_states * num_accounts *
        // sizeof(uint32_t))); Grouped memory copy
        CUDA_CHECK(cudaMemcpy(tmp_state_db->global_jump_table, state_db_cpu->global_jump_table, sizeof(GlobalJumpTable),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(tmp_state_db->address_list, state_db_cpu->address_list, num_accounts * sizeof(evm_word_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(tmp_state_db->contract_index, state_db_cpu->contract_index,
                              num_accounts * sizeof(int16_t), cudaMemcpyHostToDevice));

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
        CUDA_CHECK(
            cudaMemcpy(tmp_state_db->prealloc_keys_pool, state_db_cpu->prealloc_keys_pool,
                       account_prealloc_keys_size * state_db_cpu->num_contracts * num_states * sizeof(evm_word_t),
                       cudaMemcpyHostToDevice));
        CUDA_CHECK(
            cudaMemcpy(tmp_state_db->prealloc_values_pool, state_db_cpu->prealloc_values_pool,
                       account_prealloc_keys_size * state_db_cpu->num_contracts * num_states * sizeof(ValueStatus),
                       cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(tmp_state_db->all_account_codes, state_db_cpu->all_account_codes,
                              code_size * sizeof(uint8_t), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(tmp_state_db->dynamic_accounts, state_db_cpu->dynamic_accounts,
                              num_states * sizeof(DynamicAccount *), cudaMemcpyHostToDevice));

        // dynamic storage
        CUDA_CHECK(cudaMemcpy(tmp_state_db->dynamic_pool_capacity, state_db_cpu->dynamic_pool_capacity,
                              num_states * num_accounts * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(tmp_state_db->account_is_warm, state_db_cpu->account_is_warm,
                              num_states * num_accounts * sizeof(bool), cudaMemcpyHostToDevice));

#ifdef BUILD_GO_LIBRARY
        StateDb *tmp_snapshot_state_db = new StateDb(num_states);
        tmp_snapshot_state_db->num_accounts = state_db_cpu->num_accounts;
        tmp_snapshot_state_db->num_states = state_db_cpu->num_states;
        tmp_snapshot_state_db->num_storage_elements = state_db_cpu->num_storage_elements;
        tmp_snapshot_state_db->storage_capacity = state_db_cpu->storage_capacity;
        tmp_snapshot_state_db->num_contracts = state_db_cpu->num_contracts;

        // Grouped memory allocation
        CUDA_CHECK(cudaMalloc(&tmp_snapshot_state_db->global_jump_table, sizeof(GlobalJumpTable)));
        CUDA_CHECK(cudaMalloc(&tmp_snapshot_state_db->address_list, num_accounts * sizeof(evm_word_t)));
        CUDA_CHECK(cudaMalloc(&tmp_snapshot_state_db->contract_index, num_accounts * sizeof(int16_t)));

        CUDA_CHECK(
            cudaMalloc(&tmp_snapshot_state_db->account_balances, num_states * num_accounts * sizeof(evm_word_t)));
        CUDA_CHECK(cudaMalloc(&tmp_snapshot_state_db->account_nonces, num_states * num_accounts * sizeof(uint32_t)));
        CUDA_CHECK(
            cudaMalloc(&tmp_snapshot_state_db->account_storage_size, num_states * num_accounts * sizeof(uint32_t)));

        CUDA_CHECK(cudaMalloc(&tmp_snapshot_state_db->account_codes_size, num_accounts * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&tmp_snapshot_state_db->account_codes_offset, num_accounts * sizeof(uint32_t)));

        // prealloc storage , only num_contracts are allocated
        CUDA_CHECK(
            cudaMalloc(&tmp_snapshot_state_db->prealloc_keys_pool,
                       account_prealloc_keys_size * state_db_cpu->num_contracts * num_states * sizeof(evm_word_t)));
        CUDA_CHECK(
            cudaMalloc(&tmp_snapshot_state_db->prealloc_values_pool,
                       account_prealloc_keys_size * state_db_cpu->num_contracts * num_states * sizeof(ValueStatus)));
        CUDA_CHECK(cudaMalloc(&tmp_snapshot_state_db->all_account_codes, code_size * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&tmp_snapshot_state_db->dynamic_storage_pages,
                              num_states * num_accounts * sizeof(StateDbStoragePage *)));
        CUDA_CHECK(
            cudaMalloc(&tmp_snapshot_state_db->dynamic_pool_capacity, num_states * num_accounts * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&tmp_snapshot_state_db->account_is_warm, num_states * num_accounts * sizeof(bool)));

        CUDA_CHECK(cudaMalloc(&tmp_snapshot_state_db->dynamic_accounts, num_states * sizeof(DynamicAccount *)));

        // CUDA_CHECK(cudaMalloc(&tmp_state_db->snapshot_total_storage_size, num_states * num_accounts *
        // sizeof(uint32_t))); Grouped memory copy
        CUDA_CHECK(cudaMemcpy(tmp_snapshot_state_db->global_jump_table, state_db_cpu->global_jump_table,
                              sizeof(GlobalJumpTable), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(tmp_snapshot_state_db->address_list, state_db_cpu->address_list,
                              num_accounts * sizeof(evm_word_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(tmp_snapshot_state_db->contract_index, state_db_cpu->contract_index,
                              num_accounts * sizeof(int16_t), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(tmp_snapshot_state_db->account_balances, state_db_cpu->account_balances,
                              num_states * num_accounts * sizeof(evm_word_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(tmp_snapshot_state_db->account_nonces, state_db_cpu->account_nonces,
                              num_states * num_accounts * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(tmp_snapshot_state_db->account_storage_size, state_db_cpu->account_storage_size,
                              num_states * num_accounts * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(tmp_snapshot_state_db->account_codes_size, state_db_cpu->account_codes_size,
                              num_accounts * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(tmp_snapshot_state_db->account_codes_offset, state_db_cpu->account_codes_offset,
                              num_accounts * sizeof(uint32_t), cudaMemcpyHostToDevice));

        // prealloc storage , only num_contracts are allocated
        CUDA_CHECK(
            cudaMemcpy(tmp_snapshot_state_db->prealloc_keys_pool, state_db_cpu->prealloc_keys_pool,
                       account_prealloc_keys_size * state_db_cpu->num_contracts * num_states * sizeof(evm_word_t),
                       cudaMemcpyHostToDevice));
        CUDA_CHECK(
            cudaMemcpy(tmp_snapshot_state_db->prealloc_values_pool, state_db_cpu->prealloc_values_pool,
                       account_prealloc_keys_size * state_db_cpu->num_contracts * num_states * sizeof(ValueStatus),
                       cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(tmp_snapshot_state_db->all_account_codes, state_db_cpu->all_account_codes,
                              code_size * sizeof(uint8_t), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(tmp_snapshot_state_db->dynamic_accounts, state_db_cpu->dynamic_accounts,
                              num_states * sizeof(DynamicAccount *), cudaMemcpyHostToDevice));

        // dynamic storage
        CUDA_CHECK(cudaMemcpy(tmp_snapshot_state_db->dynamic_pool_capacity, state_db_cpu->dynamic_pool_capacity,
                              num_states * num_accounts * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(tmp_snapshot_state_db->account_is_warm, state_db_cpu->account_is_warm,
                              num_states * num_accounts * sizeof(bool), cudaMemcpyHostToDevice));

#endif
        // CUDA_CHECK(cudaMemcpy(tmp_state_db->snapshot_total_storage_size, state_db_cpu->snapshot_total_storage_size,
        //                       num_states * num_accounts * sizeof(uint32_t), cudaMemcpyHostToDevice));

        // printf("state db cpu\n");
        // state_db_cpu->print();

        StateDb *state_db_gpu;
        CUDA_CHECK(cudaMalloc(&state_db_gpu, sizeof(StateDb)));
        CUDA_CHECK(cudaMemcpy(state_db_gpu, tmp_state_db, sizeof(StateDb), cudaMemcpyHostToDevice));
        cudaMemcpyToSymbol(global_state_db_ptr, &state_db_gpu, sizeof(StateDb *));

        delete tmp_state_db;
        state_db[i] = state_db_gpu;
#ifdef BUILD_GO_LIBRARY
        StateDb *snapshot_state_db_gpu;
        CUDA_CHECK(cudaMalloc(&snapshot_state_db_gpu, sizeof(StateDb)));
        CUDA_CHECK(cudaMemcpy(snapshot_state_db_gpu, tmp_snapshot_state_db, sizeof(StateDb), cudaMemcpyHostToDevice));
        delete tmp_snapshot_state_db;
        snapshot_state_db[i] = snapshot_state_db_gpu;
#endif
    }

    delete state_db_cpu;
}

// Return a pointer to the StateDb object on device memory
__host__ void StateDb::CPUfromJson(StateDb *&state_db, const cJSON *state_json, uint32_t num_states) {
    // if (!cJSON_IsArray(state_json)) return 0;
    uint32_t num_accounts = cJSON_GetArraySize(state_json);
    // if (num_accounts == 0)
    //     ;
    printf("num_accounts: %d\n", num_accounts);

    // StateDb *state_db = new StateDb(num_states);
    state_db->global_jump_table = new GlobalJumpTable();
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

    // state_db->snapshot_total_storage_size = new uint32_t[num_states * num_accounts];
    uint32_t idx = 0;
    cJSON *account_json;
    uint32_t bytecode_offset = 0;

    memset(state_db->global_jump_table, 0, sizeof(GlobalJumpTable));
    memset(state_db->account_is_warm, 0, num_states * num_accounts * sizeof(bool));
    memset(state_db->dynamic_pool_capacity, 0, num_states * num_accounts * sizeof(uint32_t));

    memset(state_db->prealloc_keys_pool, 0,
           account_prealloc_keys_size * num_accounts * num_states * sizeof(evm_word_t));
    memset(state_db->prealloc_values_pool, 0,
           account_prealloc_keys_size * num_accounts * num_states * sizeof(ValueStatus));
    memset(state_db->dynamic_accounts, 0, num_states * sizeof(DynamicAccount *));

    cJSON_ArrayForEach(account_json, state_json) {
        cJSON *balance_json, *nonce_json;

        state_db->address_list[idx].from_hex(account_json->string);
        state_db->contract_index[idx] = -1;
        // set the balance
        balance_json = cJSON_GetObjectItemCaseSensitive(account_json, "balance");

        // set the nonce
        nonce_json = cJSON_GetObjectItemCaseSensitive(account_json, "nonce");
        evm_word_t nonce;
        nonce.from_hex(nonce_json->valuestring);

        state_db->account_nonces[idx * num_states] = uint256_get_uint32_t(&nonce);
        state_db->account_balances[idx * num_states].from_hex(balance_json->valuestring);

        // TODO: justify if it is appropriate to perform here
        // if (state_db->address_list[idx] == sender) {
        //     uint256_sub(&state_db->account_balances[idx * num_states], &state_db->account_balances[idx *
        //     num_states],
        //                 &upfront_cost);
        // }

        byte_array_t byte_code;
        byte_code.from_hex(cJSON_GetObjectItemCaseSensitive(account_json, "code")->valuestring, LITTLE_ENDIAN,
                           NO_PADDING);
        if (byte_code.size) {
            auto err = state_db->global_jump_table->analyze(byte_code.data, bytecode_offset, byte_code.size);
            if (err) {
                printf("Error %d: Invalid jumptable, things can be broken!\n", err);
            }
        }
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
            if (storage_size > account_prealloc_keys_size) {
                // todo: handle storage size > account_prealloc_keys_size
                // clone num_states times of storage page
                printf("PreState storage size: %d greater than supported, set to maximum %d\n", storage_size,
                       account_prealloc_keys_size);
                storage_size = account_prealloc_keys_size;
            }
            state_db->account_storage_size[idx * num_states] = storage_size;
            // for (uint32_t i = 0; i < num_states; i++) {
            //     state_db->account_storage_size[idx * num_states + i] = storage_size;
            // }
            state_db->num_storage_elements += state_db->account_storage_size[idx * num_states];
            // allocate new page per account

        } else
            state_db->account_storage_size[idx * num_states] = 0;
        // memset(&state_db->account_storage_size[idx * num_states], 0, num_states * sizeof(uint32_t));
        // clones to num_states
        for (uint32_t i = 0; i < num_states; i++) {
            state_db->account_storage_size[idx * num_states + i] = state_db->account_storage_size[idx * num_states];
            state_db->account_balances[idx * num_states + i] = state_db->account_balances[idx * num_states];
            state_db->account_nonces[idx * num_states + i] = state_db->account_nonces[idx * num_states];
        }

        for (uint32_t i = 0; i < state_db->account_storage_size[idx * num_states]; i++) {
            cJSON *storage_element_json = cJSON_GetArrayItem(storage_json, i);
            uint32_t contract_idx = state_db->num_contracts - 1;
            uint32_t pre_alloc_keys_idx = (account_prealloc_keys_size * contract_idx + i) * num_states;
            // printf(" contract_idx: %d, pre_alloc_keys_idx: %d\n", contract_idx, pre_alloc_keys_idx);
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
    // CuEVM debug printing
    // state_db->print();
}
__host__ StateDb *StateDb::GPUFromCPU(StateDb *&state_db) {
    StateDb *state_db_gpu = (StateDb *)malloc(sizeof(StateDb));
    memcpy(state_db_gpu, state_db, sizeof(StateDb));
    return state_db_gpu;
}

__host__ __device__ void StateDb::print() {
    printf("num_accounts: %d\n", num_accounts);
    printf("num_storage_elements: %d\n", num_storage_elements);
    for (uint32_t i = 0; i < num_accounts; i++) {
        printf("\n\n address: \n");
        address_list[i].print();
        printf("balance: \n");
        account_balances[i * num_states].print();
        printf("nonce: %d\n", account_nonces[i * num_states]);
        uint32_t account_storage_size_i = account_storage_size[i * num_states];
        if (account_storage_size_i > 0) {
            uint32_t contract_idx = contract_index[i];
            printf("keys size %d contract_idx: %d\n", account_storage_size_i, contract_idx);
            for (uint32_t j = 0; j < account_storage_size_i; j++) {
                printf("\n key: \n");
                prealloc_keys_pool[(account_prealloc_keys_size * contract_idx + j) * num_states].print();
                printf("value: \n");
                prealloc_values_pool[(account_prealloc_keys_size * contract_idx + j) * num_states].print();
            }
            if (num_states > 1) {
                printf("\n state 2 \n");
                printf("keys size %d\n", account_storage_size_i);
                for (uint32_t j = 0; j < account_storage_size_i; j++) {
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

__device__ int32_t find_global_bytecode_offset(const evm_word_t *address) {
    int32_t address_index = (address == nullptr ? -1 : global_state_db_ptr->get_address_index(address));
    if (address_index == -1) {
        return -1;
    }
    return global_state_db_ptr->account_codes_offset[address_index];
}

__device__ StateDb *global_state_db_ptr;
}  // namespace CuEVM
