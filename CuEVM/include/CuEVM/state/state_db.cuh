#pragma once
#include <CuEVM/core/data_structures.cuh>
#include <CuEVM/utils/cuda_utils.cuh>
#include <CuEVM/utils/evm_defines.cuh>
namespace CuEVM {
// for convenient data transfer between host and device. set a fixed maximum size for the number of addresses to be
// transferred
// todo : optimize this later

struct serialized_worldstate_data {
    uint32_t no_accounts;
    uint32_t no_storage_elements;
    char addresses[worldstate_addresses_size][43];  // 0x + ... + \0
    char balance[worldstate_addresses_size][67];    // 0x + ... + \0
    uint32_t nonce[worldstate_addresses_size];
    uint16_t storage_indexes[worldstate_storage_values_size];
    char storage_keys[worldstate_storage_values_size][67];    // 0x + ... + \0
    char storage_values[worldstate_storage_values_size][67];  // 0x + ... + \0
    // currently dont support copy back the bytecode hex string
    // TODO: use 1 large preallocated buffer for bytecode
    void print();
};

struct SnapshotAccount {
    // store only the modified fields
    uint16_t depth = 0;
    evm_word_t balance;
    uint32_t nonce = 0;
    uint32_t storage_size = 0;
    uint32_t code_size = 0;
    uint8_t *code = nullptr;
    SnapshotAccount *next_account = nullptr;
    bool is_warm = false;
    __host__ __device__ SnapshotAccount() : nonce(0), storage_size(0), code_size(0), code(nullptr), is_warm(false) {}
    // __device__ int32_t find_storage_key(const evm_word_t *key) const;
    __device__ void grow_storage(const evm_word_t *key);
    __host__ __device__ ~SnapshotAccount();
    __device__ void revert_to_depth(const uint16_t depth);
    __device__ void set_account(const uint16_t depth, const evm_word_t *balance, const uint32_t nonce);
    __device__ void set_code(const uint16_t depth, const uint32_t code_size, const uint8_t *code);
    // __device__ void set_storage(const uint16_t depth, const evm_word_t *key, const ValueStatus *value);
    __device__ void set_storage(const uint16_t depth, const uint32_t contract_index,
                                const uint32_t account_total_storage_size, const evm_word_t *key, ValueStatus *value);
};
/*
struct Snapshot {
    SnapshotAccount *accounts = nullptr;
    __host__ __device__ Snapshot() : accounts(nullptr) {}
    __host__ __device__ ~Snapshot();
    __device__ void grow_account(const uint16_t depth, const uint32_t address_index, const evm_word_t *balance,
                                 const uint32_t nonce);
    __device__ void set_account(const uint16_t depth, const uint32_t address_index, const evm_word_t *balance,
                                const uint32_t nonce);
    __device__ void set_code(const uint16_t depth, const uint32_t address_index, const uint32_t code_size,
                             const uint8_t *code);
    __device__ void set_storage(const uint16_t depth, const uint32_t address_index, const evm_word_t *key,
                                const ValueStatus *value);
    // __device__ SnapshotValue *get_storage(const uint32_t address_index, const evm_word_t *key) const;
    __device__ void revert_to_depth(const uint16_t depth);
};
*/

class StateDb {
   public:
    uint32_t num_states;
    uint32_t num_accounts;
    uint32_t num_contracts;
    uint32_t num_storage_elements;
    uint32_t storage_capacity;
    // lookup address to index in the account array
    evm_word_t *address_list;  // assume this list is short and not added so often
    int16_t *contract_index;   // -1 for non-contracts, 0...N for contracts in order of appearance
    // [balance A1S1, A1S2,...,A1SN, A2S1, A2S2,...., A2SN]; N = num_states
    evm_word_t *account_balances;
    uint32_t *account_nonces;
    uint32_t *account_storage_size;
    uint32_t *account_codes_offset;
    uint32_t *account_codes_size;

    SnapshotAccount *snapshot_accounts;
    uint32_t *snapshot_total_storage_size;  // total accumulated size used by all snapshot value
    bool *account_is_warm;
    // constraints: code are the same accross instances; keep 1 version
    // pointers [A1's code, A2's code,...]
    uint8_t *all_account_codes;

    // storage
    // mapping key uint to index if found
    // [A1's list of keys, A2's list of keys, ...]
    // for each key, it contains the key and the offset in the values_pool
    // Current assumption: each account preallocates account_page_size keys
    KeyOffset *all_keys;
    uint32_t *keys_list_offset;  // offset in the keys_list of ith account

    // uint32_t *keys_list_offset;
    // flattened value pool, prealloc fixed size;
    // [A1K1S1'V , A1K1S2'V , ... , A1K1SN'V ]
    ValueStatus *prealloc_values_pool;
    // dynamic values pool, allocate new page when full. no pattern
    evm_word_t **dynamic_keys_pool;     // size equals number of accounts
    ValueStatus **dynamic_values_pool;  // size equals number of accounts
    uint32_t *dynamic_pool_capacity;    // size equals number of accounts
    /**
     * The default constructor
     */
    __host__ __device__ StateDb(uint32_t num_states);
    __device__ StateDb(uint32_t num_states, uint32_t num_accounts, uint32_t num_storage_elements,
                       uint32_t storage_capacity, uint8_t *all_account_codes, evm_word_t *address_list,
                       evm_word_t *account_balances, uint32_t *account_nonces, uint32_t *account_storage_size,
                       uint32_t *account_codes_size, uint32_t *account_codes_offset, KeyOffset *all_keys,
                       uint32_t *keys_list_offset, uint32_t *keys_list_size, ValueStatus *prealloc_values_pool,
                       evm_word_t **dynamic_keys_pool, ValueStatus **dynamic_values_pool,
                       uint32_t *dynamic_pool_capacity, SnapshotAccount *snapshot_accounts);
    /**
     * Get the index of the address in the address list
     * @param[in] address The address to get the index of
     * @return The index of the address, -1 if not found
     */
    __device__ int32_t get_address_index(const evm_word_t *address) const;
    __device__ int32_t get_value_offset(int32_t address_index, const evm_word_t *key) const;
    __device__ int32_t get_dynamic_value_offset(int32_t address_index, const evm_word_t *key) const;
    __device__ void new_account(const uint16_t depth, const evm_word_t *address, const evm_word_t *balance,
                                const uint32_t nonce);
    __device__ void update_account(const uint16_t depth, const evm_word_t *address, const evm_word_t *balance,
                                   const uint32_t nonce);
    __device__ void update_balance(const uint16_t depth, const evm_word_t *address, const evm_word_t *balance);
    __device__ void update_nonce(const uint16_t depth, const evm_word_t *address, const uint32_t nonce);
    __device__ void update_code(const uint16_t depth, const evm_word_t *address, const byte_array_t *code);
    __device__ int32_t transfer(const uint16_t depth, const evm_word_t *sender, const evm_word_t *recipient,
                                const evm_word_t *value);

    __device__ evm_word_t *get_balance(const evm_word_t *address);
    __device__ uint8_t *get_code(uint32_t &code_size, const evm_word_t *address);
    __device__ uint32_t get_nonce(const evm_word_t *address);

    __device__ void grow_storage(int32_t instance_idx);
    __device__ void write_storage(const uint16_t depth, const evm_word_t *address, const evm_word_t *key,
                                  const evm_word_t *value, bool is_warm = true);
    __device__ void write_storage_with_known_index(const uint16_t depth, const evm_word_t *address,
                                                   const evm_word_t *key, const evm_word_t *value,
                                                   uint32_t address_index, ValueStatus *found_value,
                                                   bool is_warm = true);
    // return the pointer to the storage value
    __device__ evm_word_t *get_storage(const uint16_t depth, const evm_word_t *address, const evm_word_t *key,
                                       bool set_warm);
    __device__ evm_word_t *get_storage_with_known_index(const uint16_t depth, const evm_word_t *address,
                                                        const evm_word_t *key, uint32_t address_index,
                                                        ValueStatus *found_value, bool set_warm);
    __device__ ValueStatus *get_value_status(const evm_word_t *address, const evm_word_t *key) const;
    __device__ ValueStatus *get_value_status(const uint32_t address_index, const evm_word_t *key) const;
    __device__ bool is_warm_account(const evm_word_t *address) const;
    __device__ bool is_warm_key(const evm_word_t *address, const evm_word_t *key) const;
    __device__ bool is_warm_key_with_offset(const evm_word_t *address, const evm_word_t *key, uint32_t &address_index,
                                            ValueStatus *&found_value);
    __device__ void set_warm_account(const evm_word_t *address);
    __device__ void set_warm_key(const evm_word_t *address, const evm_word_t *key);

    // __device__ evm_word_t *get_original_value(const evm_word_t *address, const evm_word_t *key);
    // __device__ evm_word_t *get_value(const evm_word_t *address, const evm_word_t *key, bool set_warm = false);

    __device__ bool is_empty_account(const evm_word_t *address) const;
    __device__ bool is_deleted_account(const evm_word_t *address) const;
    __device__ bool is_contract(const evm_word_t *address) const;
    __device__ bool is_empty_create(const evm_word_t *address) const;

    // __device__ void clear_account(const evm_word_t *address);
    __device__ void snapshot_account(const uint16_t depth, const uint32_t address_index, const evm_word_t *balance,
                                     const uint32_t nonce);
    __device__ void snapshot_storage(const uint16_t depth, const uint32_t address_index, evm_word_t *key,
                                     ValueStatus *value);
    __device__ void snapshot_code(const uint16_t depth, const uint32_t address_index);
    __device__ void revert_to_snapshot(const uint16_t depth);

    __device__ void serialize_data(serialized_worldstate_data *data);

    __host__ __device__ void print();

    __host__ static void GPUfromJson(StateDb *&state_db, const cJSON *state_json, uint32_t num_states,
                                     uint32_t &num_accounts);
    __host__ static void CPUfromJson(StateDb *&state_db, const cJSON *state_json, uint32_t num_states);

    __host__ static StateDb *CPUFromGPU(StateDb *&state_db);
    __host__ static StateDb *GPUFromCPU(StateDb *&state_db);
};
extern __device__ StateDb *global_state_db_ptr;
}  // namespace CuEVM
