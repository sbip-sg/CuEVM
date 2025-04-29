#pragma once

#include <CuEVM/core/jump_table.cuh>
#include <CuEVM/core/memory_pool.cuh>
#include <CuEVM/utils/cuda_utils.cuh>
#include <CuEVM/utils/evm_defines.cuh>

namespace CuEVM {

struct DynamicAccount {
    evm_word_t address;
    evm_word_t balance;
    uint32_t nonce = 0;
    uint32_t storage_size = 0;
    uint32_t dynamic_storage_capacity = 0;
    uint32_t code_size = 0;
    // keep track of the index of the account in the dynamic account list
    int32_t dynamic_account_index = 0;
    uint8_t *code = nullptr;
    StateDbStoragePage *storage_page = nullptr;
    DynamicAccount *next_account = nullptr;

    bool is_warm = false;
    __device__ void set_code(const uint32_t code_size, const uint8_t *code);
    __device__ DynamicAccount(const evm_word_t *address, const evm_word_t *balance, const uint32_t nonce,
                              const uint32_t storage_size, const uint32_t code_size, const uint8_t *code);
    __device__ ValueStatus *get_value_status(const evm_word_t *key);
    __device__ void set_storage(const evm_word_t *key, const evm_word_t *value, bool is_warm = true);
};

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

    // for acccounts created during execution
    DynamicAccount **dynamic_accounts = nullptr;

    bool *account_is_warm;
    // constraints: code are the same accross instances; keep 1 version
    // pointers [A1's code, A2's code,...]
    uint8_t *all_account_codes;

    // storage
    // mapping key uint to index if found
    // [A1's list of keys, A2's list of keys, ...]
    // for each key, it contains the key and the offset in the values_pool
    // Current assumption: each account preallocates account_prealloc_keys_size keys
    // flattened k-v pool, prealloc fixed size;
    // [A1K1S1'V , A1K1S2'V , ... , A1K1SN'V ]
    evm_word_t *prealloc_keys_pool;
    ValueStatus *prealloc_values_pool;
    // dynamic values pool, allocate new page when full. no pattern
    // evm_word_t **dynamic_keys_pool;     // size equals number of accounts
    // ValueStatus **dynamic_values_pool;  // size equals number of accounts
    StateDbStoragePage **dynamic_storage_pages;

    GlobalJumpTable *global_jump_table;

    uint32_t *dynamic_pool_capacity;  // size equals number of accounts
    /**
     * The default constructor
     */
    __host__ __device__ StateDb(uint32_t num_states);

    /**
     * Get the index of the address in the address list
     * @param[in] address The address to get the index of
     * @return The index of the address, -1 if not found
     */
    __device__ int32_t get_address_index(const evm_word_t *address) const;
    __device__ int32_t get_value_offset(uint32_t storage_size, uint32_t contract_idx, const evm_word_t *key) const;
    __device__ ValueStatus *get_dynamic_value_location(uint32_t storage_size, uint32_t instance_idx,
                                                       const evm_word_t *key) const;
    __device__ DynamicAccount *new_account(const evm_word_t *address, const evm_word_t *balance, const uint32_t nonce,
                                           const uint32_t code_size, uint8_t *code);
    __device__ void update_account(const evm_word_t *address, const evm_word_t *balance, const uint32_t nonce);
    // shortcuts to avoid search for balance location multiple times
    __device__ int32_t deduct_balance(const evm_word_t *address, const evm_word_t *amount,
                                      SnapshotState *snapshot_state, bool set_warm = false);

    __device__ void increase_balance(const evm_word_t *address, const evm_word_t *balance,
                                     SnapshotState *snapshot_state, bool is_warm = true);
    // shortcut to deduct balance from sender and update nonce
    __device__ int32_t deduct_balance_sender(const evm_word_t *address, const evm_word_t *amount);
    __device__ void set_balance(const evm_word_t *address, const evm_word_t *balance, bool is_warm = true);
    __device__ void update_nonce(const evm_word_t *address, const uint32_t nonce);
    __device__ void update_code(const evm_word_t *address, const uint32_t code_size, uint8_t *code);
    __device__ int32_t create_contract(const evm_word_t *address, const uint32_t code_size, uint8_t *code);
    __device__ int32_t transfer(const evm_word_t *sender, const evm_word_t *recipient, const evm_word_t *value,
                                SnapshotState *snapshot_state);

    __device__ DynamicAccount *get_dynamic_account(const evm_word_t *address) const;
    __device__ DynamicAccount *get_dynamic_account_and_set_warm(const evm_word_t *address) const;
    __device__ evm_word_t *get_balance(const evm_word_t *address, bool set_warm = false);
    __device__ uint8_t *get_code(uint32_t &code_size, const evm_word_t *address, bool set_warm = true);
    __device__ uint32_t get_nonce(const evm_word_t *address);

    // Grow storage to store a key, return the pointer to the storage value for dynamic storage
    __device__ ValueStatus *grow_storage_and_set_key(uint32_t storage_size, int32_t instance_idx,
                                                     const evm_word_t *key);
    // __device__ void write_storage(const uint16_t depth, const evm_word_t *address, const evm_word_t *key,
    //                               const evm_word_t *value, bool is_warm = true);
    __device__ void write_storage_with_known_index(const evm_word_t *address, const evm_word_t *key,
                                                   const evm_word_t *value, int32_t address_index,
                                                   ValueStatus *found_value, bool is_warm = true);
    // return the pointer to the storage value
    // __device__ evm_word_t *get_storage(const uint16_t depth, const evm_word_t *address, const evm_word_t *key,
    //                                    bool set_warm);
    __device__ evm_word_t *get_storage_with_known_index(const evm_word_t *address, const evm_word_t *key,
                                                        int32_t address_index, ValueStatus *found_value,
                                                        bool set_warm = true);
    __device__ ValueStatus *get_value_status(const evm_word_t *address, const evm_word_t *key) const;
    __device__ ValueStatus *get_value_status(const int32_t address_index, const evm_word_t *key) const;
    __device__ bool is_warm_account(const evm_word_t *address, SnapshotState *snapshot_state, bool set_warm = false);
    __device__ bool is_warm_key(const evm_word_t *address, const evm_word_t *key) const;
    __device__ bool is_warm_key_with_offset(const evm_word_t *address, const evm_word_t *key, int32_t &address_index,
                                            ValueStatus *&found_value, SnapshotState *snapshot_state,
                                            bool write_snapshot = false);
    __device__ void set_warm_account(const evm_word_t *address);
    __device__ void set_warm_key(const evm_word_t *address, const evm_word_t *key);

    // for revert to non-existent key
    __device__ void reset_key(const evm_word_t *address, const evm_word_t *key);
    // __device__ evm_word_t *get_original_value(const evm_word_t *address, const evm_word_t *key);
    // __device__ evm_word_t *get_value(const evm_word_t *address, const evm_word_t *key, bool set_warm = false);

    __device__ bool is_empty_account(const evm_word_t *address) const;
    __device__ bool is_deleted_account(const evm_word_t *address) const;
    __device__ bool is_contract(const evm_word_t *address) const;
    __device__ bool is_empty_create(const evm_word_t *address);

    // __device__ void clear_account(const evm_word_t *address);

    __device__ void init_snapshot(evm_call_context_t *call_context, const uint16_t depth, const evm_word_t *address);

    __device__ void revert_to_snapshot(const uint16_t depth);

    __host__ __device__ void print();

    __host__ static void GPUfromJson(StateDb *&state_db, const cJSON *state_json, uint32_t num_states,
                                     uint32_t &num_accounts
#ifdef BUILD_GO_LIBRARY
                                     ,
                                     StateDb *&snapshot_state_db
#endif
    );
    __host__ static void CPUfromJson(StateDb *&state_db, const cJSON *state_json, uint32_t num_states);

    __host__ static StateDb *CPUFromGPU(StateDb *&state_db);
    __host__ static StateDb *GPUFromCPU(StateDb *&state_db);
};

// Get the global bytecode offset from a prestate address
__device__ int32_t find_global_bytecode_offset(const evm_word_t *address);
extern __device__ StateDb *global_state_db_ptr;
}  // namespace CuEVM
