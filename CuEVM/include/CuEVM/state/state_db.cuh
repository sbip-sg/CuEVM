#pragma once

#include <CuEVM/core/jump_table.cuh>
#include <CuEVM/core/memory_pool.cuh>
#include <CuEVM/utils/cuda_utils.cuh>
#include <CuEVM/utils/evm_defines.cuh>

/**
 * @file state_db.cuh
 * @brief State database for the EVM implementation.
 *
 * This file contains the implementation of the state database for the EVM.
 * It provides functionality for managing accounts, storage, snapshots, and
 * state modifications with the ability to revert changes.
 */

namespace CuEVM {

/**
 * @brief Dynamic account structure for runtime-created accounts.
 *
 * Represents accounts that are created during execution, containing
 * all necessary account data including balance, nonce, code, and storage.
 */
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

    /**
     * @brief Set the code for this account.
     * @param[in] code_size The size of the code.
     * @param[in] code The code bytes.
     */
    __device__ void set_code(const uint32_t code_size, const uint8_t *code);

    /**
     * @brief Constructor for a dynamic account.
     * @param[in] address The account address.
     * @param[in] balance The initial balance.
     * @param[in] nonce The initial nonce.
     * @param[in] storage_size The initial storage size.
     * @param[in] code_size The size of the account code.
     * @param[in] code The account code bytes.
     */
    __device__ DynamicAccount(const evm_word_t *address, const evm_word_t *balance, const uint32_t nonce,
                              const uint32_t storage_size, const uint32_t code_size, const uint8_t *code);

    /**
     * @brief Get the value status for a specific storage key.
     * @param[in] key The storage key.
     * @return Pointer to the value status or nullptr if not found.
     */
    __device__ ValueStatus *get_value_status(const evm_word_t *key);

    /**
     * @brief Set a storage value for a specific key.
     * @param[in] key The storage key.
     * @param[in] value The storage value.
     * @param[in] is_warm Flag indicating if the storage key is warm.
     */
    __device__ void set_storage(const evm_word_t *key, const evm_word_t *value, bool is_warm = true);
};

/**
 * @brief State database for EVM execution.
 *
 * Manages the world state during EVM execution, including accounts, storage,
 * balance transfers, and state changes. Provides functionality for state
 * snapshots and reverting changes.
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
     * @brief Constructor for the StateDb.
     * @param[in] num_states The number of states to manage.
     */
    __host__ __device__ StateDb(uint32_t num_states);

    /**
     * @brief Get the index of the address in the address list.
     * @param[in] address The address to get the index of.
     * @return The index of the address, -1 if not found.
     */
    __device__ int32_t get_address_index(const evm_word_t *address) const;

    /**
     * @brief Get the offset of a value in the storage.
     * @param[in] storage_size The storage size.
     * @param[in] contract_idx The contract index.
     * @param[in] key The storage key.
     * @return The offset of the value in storage.
     */
    __device__ int32_t get_value_offset(uint32_t storage_size, uint32_t contract_idx, const evm_word_t *key) const;

    /**
     * @brief Get the location of a dynamic value in storage.
     * @param[in] storage_size The storage size.
     * @param[in] instance_idx The instance index.
     * @param[in] key The storage key.
     * @return Pointer to the value status or nullptr if not found.
     */
    __device__ ValueStatus *get_dynamic_value_location(uint32_t storage_size, uint32_t instance_idx,
                                                       const evm_word_t *key) const;

    /**
     * @brief Create a new account.
     * @param[in] address The address of the new account.
     * @param[in] balance The initial balance.
     * @param[in] nonce The initial nonce.
     * @param[in] code_size The size of the account code.
     * @param[in] code The account code bytes.
     * @return Pointer to the newly created dynamic account.
     */
    __device__ DynamicAccount *new_account(const evm_word_t *address, const evm_word_t *balance, const uint32_t nonce,
                                           const uint32_t code_size, uint8_t *code);

    /**
     * @brief Update an existing account.
     * @param[in] address The address of the account to update.
     * @param[in] balance The new balance.
     * @param[in] nonce The new nonce.
     */
    __device__ void update_account(const evm_word_t *address, const evm_word_t *balance, const uint32_t nonce);

    /**
     * @brief Deduct balance from an account.
     * @param[in] address The address of the account.
     * @param[in] amount The amount to deduct.
     * @param[in] snapshot_state The snapshot state to record the change.
     * @param[in] set_warm Flag to set the account as warm.
     * @return 0 for success, error code otherwise.
     */
    __device__ int32_t deduct_balance(const evm_word_t *address, const evm_word_t *amount,
                                      SnapshotState *snapshot_state, bool set_warm = false);

    /**
     * @brief Increase the balance of an account.
     * @param[in] address The address of the account.
     * @param[in] balance The amount to add.
     * @param[in] snapshot_state The snapshot state to record the change.
     * @param[in] is_warm Flag indicating if the account is warm.
     */
    __device__ void increase_balance(const evm_word_t *address, const evm_word_t *balance,
                                     SnapshotState *snapshot_state, bool is_warm = true);

    /**
     * @brief Deduct balance from sender and update nonce.
     * @param[in] address The address of the sender.
     * @param[in] amount The amount to deduct.
     * @return 0 for success, error code otherwise.
     */
    __device__ int32_t deduct_balance_sender(const evm_word_t *address, const evm_word_t *amount);

    /**
     * @brief Set the balance of an account.
     * @param[in] address The address of the account.
     * @param[in] balance The new balance.
     * @param[in] is_warm Flag indicating if the account is warm.
     */
    __device__ void set_balance(const evm_word_t *address, const evm_word_t *balance, bool is_warm = true);

    /**
     * @brief Update the nonce of an account.
     * @param[in] address The address of the account.
     * @param[in] nonce The new nonce.
     */
    __device__ void update_nonce(const evm_word_t *address, const uint32_t nonce);

    /**
     * @brief Update the code of an account.
     * @param[in] address The address of the account.
     * @param[in] code_size The size of the new code.
     * @param[in] code The new code bytes.
     */
    __device__ void update_code(const evm_word_t *address, const uint32_t code_size, uint8_t *code);

    /**
     * @brief Create a contract account.
     * @param[in] address The address of the new contract.
     * @param[in] code_size The size of the contract code.
     * @param[in] code The contract code bytes.
     * @return The index of the created contract, or error code.
     */
    __device__ int32_t create_contract(const evm_word_t *address, const uint32_t code_size, uint8_t *code);

    /**
     * @brief Transfer value between accounts.
     * @param[in] sender The address of the sender.
     * @param[in] recipient The address of the recipient.
     * @param[in] value The amount to transfer.
     * @param[in] snapshot_state The snapshot state to record the change.
     * @return 0 for success, error code otherwise.
     */
    __device__ int32_t transfer(const evm_word_t *sender, const evm_word_t *recipient, const evm_word_t *value,
                                SnapshotState *snapshot_state);

    /**
     * @brief Get a dynamic account by address.
     * @param[in] address The address of the account.
     * @return Pointer to the dynamic account or nullptr if not found.
     */
    __device__ DynamicAccount *get_dynamic_account(const evm_word_t *address) const;

    /**
     * @brief Get a dynamic account and mark it as warm.
     * @param[in] address The address of the account.
     * @return Pointer to the dynamic account or nullptr if not found.
     */
    __device__ DynamicAccount *get_dynamic_account_and_set_warm(const evm_word_t *address) const;

    /**
     * @brief Get the balance of an account.
     * @param[in] address The address of the account.
     * @param[in] set_warm Flag to set the account as warm.
     * @return Pointer to the balance.
     */
    __device__ evm_word_t *get_balance(const evm_word_t *address, bool set_warm = false);

    /**
     * @brief Get the code of an account.
     * @param[out] code_size The size of the code.
     * @param[in] address The address of the account.
     * @param[in] set_warm Flag to set the account as warm.
     * @return Pointer to the code.
     */
    __device__ uint8_t *get_code(uint32_t &code_size, const evm_word_t *address, bool set_warm = true);

    /**
     * @brief Get the nonce of an account.
     * @param[in] address The address of the account.
     * @return The nonce of the account.
     */
    __device__ uint32_t get_nonce(const evm_word_t *address);

    /**
     * @brief Grow storage to accommodate a new key.
     * @param[in] storage_size The current storage size.
     * @param[in] instance_idx The instance index.
     * @param[in] key The storage key to add.
     * @return Pointer to the value status.
     */
    __device__ ValueStatus *grow_storage_and_set_key(uint32_t storage_size, int32_t instance_idx,
                                                     const evm_word_t *key);

    /**
     * @brief Write to storage with a known account index.
     * @param[in] address The address of the account.
     * @param[in] key The storage key.
     * @param[in] value The storage value.
     * @param[in] address_index The index of the account.
     * @param[in] found_value Pointer to the found value status.
     * @param[in] is_warm Flag indicating if the storage key is warm.
     */
    __device__ void write_storage_with_known_index(const evm_word_t *address, const evm_word_t *key,
                                                   const evm_word_t *value, int32_t address_index,
                                                   ValueStatus *found_value, bool is_warm = true);

    /**
     * @brief Get storage value with a known account index.
     * @param[in] address The address of the account.
     * @param[in] key The storage key.
     * @param[in] address_index The index of the account.
     * @param[in] found_value Pointer to the found value status.
     * @param[in] set_warm Flag to set the storage key as warm.
     * @return Pointer to the storage value.
     */
    __device__ evm_word_t *get_storage_with_known_index(const evm_word_t *address, const evm_word_t *key,
                                                        int32_t address_index, ValueStatus *found_value,
                                                        bool set_warm = true);

    /**
     * @brief Get the value status for a storage key.
     * @param[in] address The address of the account.
     * @param[in] key The storage key.
     * @return Pointer to the value status or nullptr if not found.
     */
    __device__ ValueStatus *get_value_status(const evm_word_t *address, const evm_word_t *key) const;

    /**
     * @brief Get the value status for a storage key using account index.
     * @param[in] address_index The index of the account.
     * @param[in] key The storage key.
     * @return Pointer to the value status or nullptr if not found.
     */
    __device__ ValueStatus *get_value_status(const int32_t address_index, const evm_word_t *key) const;

    /**
     * @brief Check if an account is warm.
     * @param[in] address The address of the account.
     * @param[in] snapshot_state The snapshot state.
     * @param[in] set_warm Flag to set the account as warm.
     * @return True if the account is warm, false otherwise.
     */
    __device__ bool is_warm_account(const evm_word_t *address, SnapshotState *snapshot_state, bool set_warm = false);

    /**
     * @brief Check if a storage key is warm.
     * @param[in] address The address of the account.
     * @param[in] key The storage key.
     * @return True if the key is warm, false otherwise.
     */
    __device__ bool is_warm_key(const evm_word_t *address, const evm_word_t *key) const;

    /**
     * @brief Check if a storage key is warm with a given found value status
     * @param[in] address The address of the account.
     * @param[in] key The storage key.
     * @param[out] address_index The address index.
     * @param[out] found_value The found value status.
     * @param[in] snapshot_state The snapshot state.
     * @param[in] write_snapshot Flag to write to snapshot.
     * @return True if the key is warm, false otherwise.
     */
    __device__ bool is_warm_key_with_offset(const evm_word_t *address, const evm_word_t *key, int32_t &address_index,
                                            ValueStatus *&found_value, SnapshotState *snapshot_state,
                                            bool write_snapshot = false);

    /**
     * @brief Mark an account as warm.
     * @param[in] address The address of the account.
     */
    __device__ void set_warm_account(const evm_word_t *address);

    /**
     * @brief Mark a storage key as warm.
     * @param[in] address The address of the account.
     * @param[in] key The storage key.
     */
    __device__ void set_warm_key(const evm_word_t *address, const evm_word_t *key);

    /**
     * @brief Reset a storage key (for reverting).
     * @param[in] address The address of the account.
     * @param[in] key The storage key to reset.
     */
    __device__ void reset_key(const evm_word_t *address, const evm_word_t *key);

    /**
     * @brief Check if an account is empty.
     * @param[in] address The address of the account.
     * @return True if the account is empty, false otherwise.
     */
    __device__ bool is_empty_account(const evm_word_t *address) const;

    /**
     * @brief Check if an account is deleted.
     * @param[in] address The address of the account.
     * @return True if the account is deleted, false otherwise.
     */
    __device__ bool is_deleted_account(const evm_word_t *address) const;

    /**
     * @brief Check if an address is a contract.
     * @param[in] address The address to check.
     * @return True if the address is a contract, false otherwise.
     */
    __device__ bool is_contract(const evm_word_t *address) const;

    /**
     * @brief Check if a CREATE operation results in an empty account.
     * @param[in] address The address to check.
     * @return True if the CREATE operation results in an empty account, false otherwise.
     */
    __device__ bool is_empty_create(const evm_word_t *address);

    /**
     * @brief Initialize a snapshot for state modifications.
     * @param[in] call_context The call context.
     * @param[in] depth The call depth.
     * @param[in] address The address of the account.
     */
    __device__ void init_snapshot(evm_call_context_t *call_context, const uint16_t depth, const evm_word_t *address);

    /**
     * @brief Revert to a previous snapshot.
     * @param[in] depth The depth of the snapshot to revert to.
     */
    __device__ void revert_to_snapshot(const uint16_t depth);

    /**
     * @brief Print the state database contents.
     */
    __host__ __device__ void print();

    /**
     * @brief Create a GPU state database from JSON.
     * @param[out] state_db The output state database.
     * @param[in] state_json The JSON state data.
     * @param[in] num_states The number of states.
     * @param[out] num_accounts The number of accounts.
     * @param[out] snapshot_state_db The output snapshot state database to restore to original state in fuzzing use case
     */
    __host__ static void GPUfromJson(StateDb *&state_db, const cJSON *state_json, uint32_t num_states,
                                     uint32_t &num_accounts
#ifdef BUILD_GO_LIBRARY
                                     ,
                                     StateDb *&snapshot_state_db
#endif
    );

    /**
     * @brief Create multiple GPU state databases from JSON for multi-GPU scenario.
     * @param[out] state_db The output state databases.
     * @param[in] state_json The JSON state data.
     * @param[in] num_states The number of states.
     * @param[out] num_accounts The number of accounts.
     * @param[out] snapshot_state_db The output snapshot state databases.
     */
    __host__ static void GPUfromJsonMultiGPU(std::vector<StateDb *> &state_db, const cJSON *state_json,
                                             uint32_t num_states, uint32_t &num_accounts,
                                             std::vector<StateDb *> &snapshot_state_db

    );

    /**
     * @brief Create a CPU state database from JSON.
     * @param[out] state_db The output state database.
     * @param[in] state_json The JSON state data.
     * @param[in] num_states The number of states.
     */
    __host__ static void CPUfromJson(StateDb *&state_db, const cJSON *state_json, uint32_t num_states);

    /**
     * @brief Create a GPU state database from a CPU state database.
     * @param[in] state_db The CPU state database.
     * @return The GPU state database.
     */
    __host__ static StateDb *GPUFromCPU(StateDb *&state_db);
};

/**
 * @brief Find global bytecode offset from a prestate address.
 * @param[in] address The address to find the bytecode offset for.
 * @return The global bytecode offset or -1 if not found.
 */
__device__ int32_t find_global_bytecode_offset(const evm_word_t *address);

/**
 * @brief Global state database pointer.
 */
extern __device__ StateDb *global_state_db_ptr;
}  // namespace CuEVM
