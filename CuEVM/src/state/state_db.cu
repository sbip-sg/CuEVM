#include <CuEVM/state/state_db.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM {
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
    values_pool = nullptr;
}
__device__ StateDb::StateDb(uint32_t num_states, uint32_t num_accounts, uint32_t num_storage_elements,
                            uint32_t storage_capacity, uint8_t *all_account_codes, evm_word_t *address_list,
                            evm_word_t *account_balances, uint32_t *account_nonces, uint32_t *account_storage_size,
                            uint32_t *account_codes_size, uint32_t *account_codes_offset, KeyOffset *all_keys,
                            uint32_t *keys_list_offset, uint32_t *keys_list_size, evm_word_t *values_pool)
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
      values_pool(values_pool) {}

__device__ int32_t StateDb::get_address_index(const evm_word_t *address) const {
    // Todo: global list -> parallel search
    for (uint32_t i = 0; i < num_accounts; i++) {
        if (address_list[i] == *address) {
            return i;
        }
    }
    return -1;
}

__device__ int32_t StateDb::get_value_offset(const evm_word_t *address, const evm_word_t *key) const {
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        return -1;
    }
    uint32_t key_list_idx = keys_list_offset[address_index];

    for (uint32_t i = 0; i < account_storage_size[address_index]; i++) {
        if (all_keys[key_list_idx + i].key == *key) {
            return all_keys[key_list_idx + i].offset;
        }
    }
    return -1;
}

__device__ void StateDb::new_account(const evm_word_t *address, const evm_word_t *balance, const uint32_t nonce) {
    address_list[num_accounts] = *address;
    account_balances[num_accounts] = *balance;
    account_nonces[num_accounts] = nonce;
    num_accounts++;
}

__device__ void StateDb::update_account(const evm_word_t *address, const evm_word_t *balance, const uint32_t nonce) {
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        return;
    }
    account_balances[address_index] = *balance;
    account_nonces[address_index] = nonce;
}

__device__ void StateDb::update_balance(const evm_word_t *address, const evm_word_t *balance) {
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        return;
    }
    account_balances[address_index] = *balance;
}

__device__ void StateDb::update_nonce(const evm_word_t *address, const uint32_t nonce) {
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        return;
    }
    account_nonces[address_index] = nonce;
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

__device__ void StateDb::write_storage(const evm_word_t *address, const evm_word_t *key, const evm_word_t *value,
                                       uint16_t call_depth) {
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        return;  // should never write directly to storage before creating account
    }
    int32_t key_offset = get_value_offset(address, key);
    if (key_offset == -1) {
        grow_storage();
        uint32_t key_list_start = keys_list_offset[address_index];
        uint32_t key_list_size = account_storage_size[address_index];
        KeyOffset *new_key_offset = &all_keys[key_list_start + key_list_size];
        key_offset = num_storage_elements * num_states;
        new_key_offset->key = *key;
        new_key_offset->offset = key_offset;
        account_storage_size[address_index]++;
        num_storage_elements++;
    }
    values_pool[key_offset + INSTANCE_GLOBAL_IDX] = *value;
    // todo : write snapshot for potential future revert
}

__device__ evm_word_t *StateDb::get_storage(const evm_word_t *address, const evm_word_t *key) const {
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        return nullptr;
    }
    int32_t key_offset = get_value_offset(address, key);
    if (key_offset == -1) {
        return nullptr;
    }
    return &values_pool[key_offset + INSTANCE_GLOBAL_IDX];
}

__device__ void StateDb::grow_storage() {
    if (num_storage_elements + 1 > storage_capacity) {
        storage_capacity += value_page_size;
        // allocate new page
        evm_word_t *tmp = values_pool;
        values_pool = new evm_word_t[storage_capacity * num_states];
        if (tmp != nullptr) {
            memcpy(values_pool, tmp, num_storage_elements * num_states * sizeof(evm_word_t));
            delete[] tmp;
        }
    }
}

__device__ evm_word_t *StateDb::get_balance(const evm_word_t *address) {
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        return nullptr;
    }
    return &account_balances[address_index];
}

__device__ uint32_t StateDb::get_nonce(const evm_word_t *address) {
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        return 0;
    }
    return account_nonces[address_index];
}

__device__ uint8_t *StateDb::get_code(uint32_t &code_size, const evm_word_t *address) {
    int32_t address_index = get_address_index(address);
    if (address_index == -1) {
        return nullptr;
    }
    code_size = account_codes_size[address_index];
    return &all_account_codes[account_codes_offset[address_index]];
}

// __device__ bool is_warm_account(const evm_word_t *address);
// __device__ bool is_warm_key(const evm_word_t *address, const evm_word_t *key);
// __device__ void set_warm_account(const evm_word_t *address);
// __device__ void set_warm_key(const evm_word_t *address, const evm_word_t *key);
// __device__ evm_word_t *get_original_value(const evm_word_t *address, const evm_word_t *key);
// __device__ evm_word_t *get_value(const evm_word_t *address, const evm_word_t *key);
__device__ evm_word_t *StateDb::get_original_value(const evm_word_t *address, const evm_word_t *key) const {
    // todo :implement
    return get_value(address, key);
}
__device__ evm_word_t *StateDb::get_value(const evm_word_t *address, const evm_word_t *key) const {
    return get_storage(address, key);
}

__device__ void StateDb::set_warm_account(const evm_word_t *address) {
    // todo :implement
}

__device__ void StateDb::set_warm_key(const evm_word_t *address, const evm_word_t *key) {
    // todo :implement
}

__device__ bool StateDb::is_warm_account(const evm_word_t *address) const {
    // todo :implement
    return false;
}

__device__ bool StateDb::is_warm_key(const evm_word_t *address, const evm_word_t *key) const {
    // todo :implement
    return false;
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
    CUDA_CHECK(cudaMalloc(&tmp_state_db->account_balances, num_accounts * sizeof(evm_word_t)));
    CUDA_CHECK(cudaMalloc(&tmp_state_db->account_nonces, num_accounts * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&tmp_state_db->account_storage_size, num_accounts * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&tmp_state_db->account_codes_size, num_accounts * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&tmp_state_db->account_codes_offset, num_accounts * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&tmp_state_db->all_keys, account_prealloc_keys_size * num_accounts * sizeof(KeyOffset)));
    CUDA_CHECK(cudaMalloc(&tmp_state_db->keys_list_offset, num_accounts * sizeof(uint32_t)));
    CUDA_CHECK(
        cudaMalloc(&tmp_state_db->values_pool, state_db_cpu->storage_capacity * num_states * sizeof(evm_word_t)));
    CUDA_CHECK(cudaMalloc(&tmp_state_db->all_account_codes, code_size * sizeof(uint8_t)));

    // Grouped memory copy
    CUDA_CHECK(cudaMemcpy(tmp_state_db->address_list, state_db_cpu->address_list, num_accounts * sizeof(evm_word_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_state_db->account_balances, state_db_cpu->account_balances,
                          num_accounts * sizeof(evm_word_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_state_db->account_nonces, state_db_cpu->account_nonces, num_accounts * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_state_db->account_storage_size, state_db_cpu->account_storage_size,
                          num_accounts * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_state_db->account_codes_size, state_db_cpu->account_codes_size,
                          num_accounts * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_state_db->account_codes_offset, state_db_cpu->account_codes_offset,
                          num_accounts * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_state_db->all_keys, state_db_cpu->all_keys,
                          account_prealloc_keys_size * num_accounts * sizeof(KeyOffset), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_state_db->keys_list_offset, state_db_cpu->keys_list_offset,
                          num_accounts * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_state_db->values_pool, state_db_cpu->values_pool,
                          state_db_cpu->storage_capacity * num_states * sizeof(evm_word_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_state_db->all_account_codes, state_db_cpu->all_account_codes, code_size * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));

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
    state_db->storage_capacity = value_page_size;
    state_db->all_account_codes = nullptr;
    state_db->address_list = new evm_word_t[num_accounts];
    state_db->account_balances = new evm_word_t[num_accounts];
    state_db->account_nonces = new uint32_t[num_accounts];
    state_db->account_storage_size = new uint32_t[num_accounts];
    state_db->account_codes_size = new uint32_t[num_accounts];
    state_db->account_codes_offset = new uint32_t[num_accounts];
    state_db->all_keys = new KeyOffset[account_prealloc_keys_size * num_accounts];
    state_db->keys_list_offset = new uint32_t[num_accounts];
    state_db->values_pool = new evm_word_t[state_db->storage_capacity * num_states];

    uint32_t idx = 0;
    cJSON *account_json;
    uint32_t bytecode_offset = 0;
    uint32_t key_offset_wo_states = 0;  // offset without multiplying by num_states
    uint32_t current_code_size = 0;
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

        cJSON *storage_json = cJSON_GetObjectItemCaseSensitive(account_json, "storage");
        if (storage_json != nullptr) {
            state_db->account_storage_size[idx] = cJSON_GetArraySize(storage_json);

            state_db->num_storage_elements += state_db->account_storage_size[idx];
            // allocate new page per account
            if (state_db->num_storage_elements >= state_db->storage_capacity) {
                evm_word_t *tmp = state_db->values_pool;
                state_db->values_pool = new evm_word_t[state_db->num_storage_elements * num_states];
                if (tmp != nullptr) {
                    memcpy(state_db->values_pool, tmp,
                           state_db->num_storage_elements * num_states * sizeof(evm_word_t));
                    delete[] tmp;
                }
            }
            // todo: handle storage size > account_prealloc_keys_size
        } else
            state_db->account_storage_size[idx] = 0;

        uint32_t pre_alloc_keys_idx = idx * account_prealloc_keys_size;
        for (uint32_t i = 0; i < state_db->account_storage_size[idx]; i++) {
            cJSON *storage_element_json = cJSON_GetArrayItem(storage_json, i);

            state_db->all_keys[pre_alloc_keys_idx + i].key.from_hex(storage_element_json->string);
            state_db->all_keys[pre_alloc_keys_idx + i].offset = (key_offset_wo_states + i) * num_states;
            state_db->values_pool[state_db->all_keys[pre_alloc_keys_idx + i].offset].from_hex(
                storage_element_json->valuestring);
            // duplicate the value for all states
            for (uint32_t j = 1; j < num_states; j++) {
                state_db->values_pool[state_db->all_keys[pre_alloc_keys_idx + i].offset + j] =
                    state_db->values_pool[state_db->all_keys[pre_alloc_keys_idx + i].offset];
            }
            printf("key: %d \n", pre_alloc_keys_idx + i);
            state_db->all_keys[pre_alloc_keys_idx + i].key.print();
            printf("offset: %d\n", state_db->all_keys[pre_alloc_keys_idx + i].offset);
        }
        state_db->keys_list_offset[idx] = pre_alloc_keys_idx;
        key_offset_wo_states += state_db->account_storage_size[idx];
        idx++;
    }
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
    memcpy(state_db_cpu->values_pool, state_db->values_pool, sizeof(evm_word_t) * num_storage_elements * num_states);
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
                values_pool[all_keys[keys_list_offset[i] + j].offset + INSTANCE_GLOBAL_IDX].print();
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
