#include <CuEVM/libcuevm_go.h>


// Initialize static counter
int GoStateData::call_counter = 0;

// Initialize static counter
int GoTransactionData::tx_counter = 0;

// Helper function declarations
std::string uint256_to_hex(const go_uint256_t& value);
std::string bytes_to_hex(const unsigned char* data, int len);
go_uint256_t bytes_to_uint256(const unsigned char* data, int data_len);

// Helper function implementations - MOVE THESE OUTSIDE extern "C" BLOCK
// Helper function to convert uint256 to hex string for printing
std::string uint256_to_hex(const go_uint256_t& value) {
    std::stringstream ss;
    ss << "0x";
    for (int i = UINT256_WORDS - 1; i >= 0; i--) {
        ss << std::hex << std::setfill('0') << std::setw(8) << value.words[i];
    }
    return ss.str();
}

// Helper function to convert bytes to hex string
std::string bytes_to_hex(const unsigned char* data, int len) {
    std::stringstream ss;
    ss << "0x";
    for (int i = 0; i < len; i++) {
        ss << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(data[i]);
    }
    return ss.str();
}

// Convert byte array to uint256 (big-endian to little-endian)
go_uint256_t bytes_to_uint256(const unsigned char* data, int data_len) {
    go_uint256_t result;
    memset(&result, 0, sizeof(go_uint256_t));
    
    int copyLen = data_len < UINT256_BYTES ? data_len : UINT256_BYTES;
    if (copyLen > 0) {
        for (int i = 0; i < copyLen; i++) {
            int pos = i % 4;
            int word = i / 4;
            result.words[word] |= (static_cast<uint32_t>(data[data_len - 1 - i]) << (pos * 8));
        }
    }
    
    return result;
}

// Ensure C linkage for Go to call these functions
#ifdef __cplusplus
extern "C" {
#endif

// Create a new state data container
GoStateData* create_state_data() {
    printf("Go interface: Creating new state data container\n");
    return new GoStateData();
}

// Set state root from byte array
void set_state_root(GoStateData* state, const unsigned char* root, int root_len) {
    printf("Go interface: Setting state root, length: %d\n", root_len);
    state->root = bytes_to_uint256(root, root_len);
}

// Add an account to state data with byte array inputs
void add_account(GoStateData* state, 
                 const unsigned char* addr, int addr_len,
                 const unsigned char* balance, int balance_len,
                 uint64_t nonce,
                 const unsigned char* root, int root_len,
                 const unsigned char* code_hash, int code_hash_len,
                 const unsigned char* code, int code_len,
                 bool has_code) {
    
    printf("Go interface: Adding account, address length: %d, code length: %d\n", addr_len, code_len);
    
    // Create a new account
    GoAccount account;
    
    // Convert address and balance bytes to uint256
    account.address = bytes_to_uint256(addr, addr_len);
    account.balance = bytes_to_uint256(balance, balance_len);
    
    // Set the account nonce
    account.nonce = nonce;
    
    // Set the root if provided
    if (root != nullptr && root_len > 0) {
        account.root.resize(root_len);
        memcpy(account.root.data(), root, root_len);
    }
    
    // Set the code hash if provided
    if (code_hash != nullptr && code_hash_len > 0) {
        account.codeHash.resize(code_hash_len);
        memcpy(account.codeHash.data(), code_hash, code_hash_len);
    }
    
    // Set the code if provided
    if (code != nullptr && code_len > 0) {
        account.code.resize(code_len);
        memcpy(account.code.data(), code, code_len);
    }
    
    // Set whether the account has code
    account.hasCode = has_code;
    
    // Add the account to our state data
    state->accounts.push_back(account);
}

// Add a storage entry to the last added account
void add_storage_entry(GoStateData* state, 
                       const unsigned char* key, int key_len, 
                       const unsigned char* value, int value_len) {
    
    // Ensure we have at least one account
    if (state->accounts.empty()) {
        printf("Go interface: Cannot add storage entry: No accounts added yet\n");
        return;
    }
    
    printf("Go interface: Adding storage entry, key length: %d, value length: %d\n", key_len, value_len);
    
    // Convert key and value bytes to uint256
    go_uint256_t key_uint256 = bytes_to_uint256(key, key_len);
    go_uint256_t value_uint256 = bytes_to_uint256(value, value_len);
    
    // Add to the last account's storage
    state->accounts.back().storageKeys.push_back(key_uint256);
    state->accounts.back().storageVals.push_back(value_uint256);
}
using namespace CuEVM;
// Process state data on GPU - initialize StateDB and print data
int process_state_data_gpu(GoStateData* state) {
    printf("CuEVM Go interface: Processing state data in GPU C++ function...\n");
    printf("State root: %s\n", uint256_to_hex(state->root).c_str());
    printf("Number of accounts: %zu\n", state->accounts.size());
    
    try {

        // NEW: Initialize StateDB for GPU processing
        uint32_t num_states = 2; // Start with a single state
        uint32_t num_accounts = state->accounts.size();

        // Create CPU-side StateDB
        StateDb* state_db_cpu = new StateDb(num_states);
        state_db_cpu->num_accounts = num_accounts;
        state_db_cpu->num_states = num_states;
        
        // Allocate memory for accounts
        state_db_cpu->address_list = new evm_word_t[num_accounts];
        state_db_cpu->contract_index = new int16_t[num_accounts];
        state_db_cpu->account_balances = new evm_word_t[num_states * num_accounts];
        state_db_cpu->account_nonces = new uint32_t[num_states * num_accounts];
        state_db_cpu->account_storage_size = new uint32_t[num_states * num_accounts];
        state_db_cpu->account_codes_size = new uint32_t[num_accounts];
        state_db_cpu->account_codes_offset = new uint32_t[num_accounts];
        
        // Initialize dynamic memory structures
        uint32_t bytecode_offset = 0;
        
        // Count contracts and determine storage size
        uint32_t num_contracts = 0;
        uint32_t total_storage_size = 0;
        
        for (size_t i = 0; i < state->accounts.size(); i++) {
            const auto& account = state->accounts[i];
            bool is_contract = !account.code.empty() || !account.storageKeys.empty();
            if (is_contract) {
                num_contracts++;
            }
            total_storage_size += account.storageKeys.size();
        }
        printf("num contract %d\n", num_contracts);
        
        state_db_cpu->num_contracts = num_contracts;
        state_db_cpu->num_storage_elements = total_storage_size;
        
        // Allocate preallocated storage pool
        state_db_cpu->prealloc_keys_pool = new evm_word_t[account_prealloc_keys_size * num_contracts * num_states];
        state_db_cpu->prealloc_values_pool = new ValueStatus[account_prealloc_keys_size * num_contracts * num_states];
        
        // Initialize other necessary structures
        state_db_cpu->dynamic_pool_capacity = new uint32_t[num_states * num_accounts];
        state_db_cpu->account_is_warm = new bool[num_states * num_accounts];
        state_db_cpu->dynamic_accounts = new DynamicAccount*[num_states];
        
        memset(state_db_cpu->prealloc_keys_pool, 0, 
               account_prealloc_keys_size * num_contracts * num_states * sizeof(evm_word_t));
        memset(state_db_cpu->prealloc_values_pool, 0,
               account_prealloc_keys_size * num_contracts * num_states * sizeof(ValueStatus));
        memset(state_db_cpu->dynamic_accounts, 0, num_states * sizeof(DynamicAccount*));
        
        // Transfer account data from Go structure to StateDB
        uint32_t contract_idx = 0;
        for (uint32_t i = 0; i < num_accounts; i++) {
            const auto& account = state->accounts[i];
            uint32_t base_idx = i * num_states;
            
            // Convert Go uint256 to evm_word_t - first for state 0
            for (int w = 0; w < UINT256_WORDS; w++) {
                state_db_cpu->address_list[i].words[w] = account.address.words[w];
                state_db_cpu->account_balances[base_idx].words[w] = account.balance.words[w];
            }
            
            // Set nonce for state 0
            state_db_cpu->account_nonces[base_idx] = static_cast<uint32_t>(account.nonce);
            
            // Handle code
            state_db_cpu->account_codes_size[i] = account.code.size();
            state_db_cpu->account_codes_offset[i] = bytecode_offset;
            
            bool is_contract = !account.code.empty() || !account.storageKeys.empty();
            
            if (is_contract) {
                state_db_cpu->contract_index[i] = contract_idx;
                
                // Handle storage for state 0
                state_db_cpu->account_storage_size[base_idx] = account.storageKeys.size();
                
                // Copy storage entries for state 0
                for (size_t s = 0; s < account.storageKeys.size() && s < account_prealloc_keys_size; s++) {
                    uint32_t pre_alloc_keys_idx = 
                        (account_prealloc_keys_size * contract_idx + s) * num_states;
                    printf("contract_idx %d s %d pre_alloc_keys_idx %d\n", contract_idx, s, pre_alloc_keys_idx);
                    
                    // Copy key and value for state 0
                    for (int w = 0; w < UINT256_WORDS; w++) {
                        state_db_cpu->prealloc_keys_pool[pre_alloc_keys_idx].words[w] = 
                            account.storageKeys[s].words[w];
                        state_db_cpu->prealloc_values_pool[pre_alloc_keys_idx].value.words[w] = 
                            account.storageVals[s].words[w];
                    }
                    state_db_cpu->prealloc_values_pool[pre_alloc_keys_idx].original_value = 
                        state_db_cpu->prealloc_values_pool[pre_alloc_keys_idx].value;
                    state_db_cpu->prealloc_values_pool[pre_alloc_keys_idx].is_warm = false;
                    
                    // Clone to all other states
                    for (uint32_t state_idx = 1; state_idx < num_states; state_idx++) {
                        uint32_t dst_storage_idx = pre_alloc_keys_idx + state_idx;
                        // Copy key and value
                        state_db_cpu->prealloc_keys_pool[dst_storage_idx] = 
                            state_db_cpu->prealloc_keys_pool[pre_alloc_keys_idx];
                        state_db_cpu->prealloc_values_pool[dst_storage_idx] = 
                            state_db_cpu->prealloc_values_pool[pre_alloc_keys_idx];
                    }
                }
                
                contract_idx++;
            } else {
                state_db_cpu->contract_index[i] = -1;
                state_db_cpu->account_storage_size[base_idx] = 0;
            }
            
            // Add code if it exists
            if (!account.code.empty()) {
                uint8_t* tmp = state_db_cpu->all_account_codes;
                state_db_cpu->all_account_codes = new uint8_t[bytecode_offset + account.code.size()];
                if (tmp != nullptr) {
                    memcpy(state_db_cpu->all_account_codes, tmp, bytecode_offset * sizeof(uint8_t));
                    delete[] tmp;
                }
                memcpy(&state_db_cpu->all_account_codes[bytecode_offset], account.code.data(), account.code.size());
                bytecode_offset += account.code.size();
            }
            
            // Clone account data to all other states right after processing this account
            for (uint32_t state_idx = 1; state_idx < num_states; state_idx++) {
                uint32_t dst_idx = base_idx + state_idx;
                
                // Copy balance
                state_db_cpu->account_balances[dst_idx] = state_db_cpu->account_balances[base_idx];
                
                // Copy nonce and storage size
                state_db_cpu->account_nonces[dst_idx] = state_db_cpu->account_nonces[base_idx];
                state_db_cpu->account_storage_size[dst_idx] = state_db_cpu->account_storage_size[base_idx];
                
                // Initialize other per-state properties
                state_db_cpu->account_is_warm[dst_idx] = state_db_cpu->account_is_warm[base_idx];
                state_db_cpu->dynamic_pool_capacity[dst_idx] = state_db_cpu->dynamic_pool_capacity[base_idx];
            }
        }
        
        // Print the StateDB contents to verify
        printf("\nInitialized StateDB from Go data:\n");
        printf("Num accounts: %d\n", state_db_cpu->num_accounts);
        printf("Num contracts: %d\n", state_db_cpu->num_contracts);
        printf("Num storage elements: %d\n", state_db_cpu->num_storage_elements);
        
        // Print some account details
        state_db_cpu->print();
        // Cleanup - in a real implementation we would transfer to GPU before this
        delete state_db_cpu;
        
        // Increment call counter and return success
        state->call_counter++;
        printf("\nGo interface call number: %d\n", state->call_counter);
        
        return 0; // Success
    } catch (const std::exception& e) {
        printf("Error in process_state_data_gpu: %s\n", e.what());
        return 1; // Error code
    } catch (...) {
        printf("Unknown error in process_state_data_gpu\n");
        return 2; // Different error code
    }
}

// Free memory
void free_state_data(GoStateData* state) {
    printf("Go interface: Freeing state data\n");
    delete state;
}

// Function to get the call counter
int get_call_count() {
    return GoStateData::call_counter;
}

// In the future, we would implement the run_interpreter_go function similar to run_interpreter_pyobject
// For now, just a placeholder that increments the counter and returns success
int run_interpreter_go(const char* json_input, uint32_t skip_trace_parsing, 
                     uint32_t copy_state_data, uint32_t reuse_state_data) {
    printf("Go interface: Running interpreter with JSON input\n");
    printf("Run configuration skip_trace_parsing: %d, copy_state_data: %d, reuse_state_data: %d\n", 
           skip_trace_parsing, copy_state_data, reuse_state_data);
    
    // Increment call counter
    GoStateData::call_counter++;
    printf("Total call count: %d\n", GoStateData::call_counter);
    
    // This is where we would process the JSON input and run the CUDA kernel
    // For now, just return success
    return 0;
}

// Create a new transaction data container
GoTransactionData* create_transaction_data() {
    printf("Go interface: Creating new transaction data container\n");
    return new GoTransactionData();
}

// Add a transaction to the data container
void add_transaction(GoTransactionData* txData,
                     const unsigned char* from, int from_len,
                     const unsigned char* to, int to_len,
                     const unsigned char* value, int value_len,
                     uint64_t gas,
                     const unsigned char* gasPrice, int gasPrice_len,
                     uint64_t nonce,
                     const unsigned char* data, int data_len) {
    
    GoTransactionData* typedTxData = static_cast<GoTransactionData*>(txData);
    printf("Go interface: Adding transaction, from length: %d, to length: %d, data length: %d\n", 
           from_len, to_len, data_len);
    
    // Create a new transaction
    GoTransaction tx;
    
    // Convert addresses and values to uint256
    tx.from = bytes_to_uint256(from, from_len);
    tx.to = bytes_to_uint256(to, to_len);
    tx.value = bytes_to_uint256(value, value_len);
    tx.gasPrice = bytes_to_uint256(gasPrice, gasPrice_len);
    
    // Set gas and nonce
    tx.gas = gas;
    tx.nonce = nonce;
    
    // Set transaction data if provided
    if (data != nullptr && data_len > 0) {
        tx.data.resize(data_len);
        memcpy(tx.data.data(), data, data_len);
    }
    
    // Add the transaction to our data container
    typedTxData->transactions.push_back(tx);
}

// Process transaction data on GPU
int process_transaction_data_gpu(GoTransactionData* txData) {
    printf("CuEVM Go interface: Processing transaction data in GPU C++ function...\n");
    printf("Number of transactions: %zu\n", static_cast<GoTransactionData*>(txData)->transactions.size());
    
    try {
        // Print transaction details (for debugging)
        int count = 0;
        for (const auto& tx : static_cast<GoTransactionData*>(txData)->transactions) {
            if (count >= 10) {
                printf("\n... and %zu more transactions\n", static_cast<GoTransactionData*>(txData)->transactions.size() - 10);
                break;
            }
            
            printf("\n=== Transaction %d ===\n", count);
            printf("  From:     %s\n", uint256_to_hex(tx.from).c_str());
            if (tx.to.words[0] != 0 || tx.to.words[1] != 0 || 
                tx.to.words[2] != 0 || tx.to.words[3] != 0 || 
                tx.to.words[4] != 0 || tx.to.words[5] != 0 || 
                tx.to.words[6] != 0 || tx.to.words[7] != 0) {
                printf("  To:       %s\n", uint256_to_hex(tx.to).c_str());
            } else {
                printf("  To:       <contract creation>\n");
            }
            printf("  Value:    %s\n", uint256_to_hex(tx.value).c_str());
            printf("  Gas:      %lu\n", tx.gas);
            printf("  GasPrice: %s\n", uint256_to_hex(tx.gasPrice).c_str());
            printf("  Nonce:    %lu\n", tx.nonce);
            
            // Print data information
            if (!tx.data.empty()) {
                int dataPreviewSize = tx.data.size() < 64 ? tx.data.size() : 64;
                printf("  Data:     %s...\n", bytes_to_hex(tx.data.data(), dataPreviewSize).c_str());
                printf("  Data Length: %zu bytes\n", tx.data.size());
            } else {
                printf("  Data:     <empty>\n");
            }
            
            count++;
        }
        
        // TODO: Future implementation will integrate with interpreter
        // This would involve transferring transactions to GPU memory and executing them
        
        // Increment call counter and return success
        static_cast<GoTransactionData*>(txData)->tx_counter++;
        printf("\nTransaction processing call number: %d\n", static_cast<GoTransactionData*>(txData)->tx_counter);
        
        return 0; // Success
    } catch (const std::exception& e) {
        printf("Error in process_transaction_data_gpu: %s\n", e.what());
        return 1; // Error code
    } catch (...) {
        printf("Unknown error in process_transaction_data_gpu\n");
        return 2; // Different error code
    }
}

// Free transaction data memory
void free_transaction_data(GoTransactionData* txData) {
    printf("Go interface: Freeing transaction data\n");
    delete txData;
}

// Function to get the transaction call counter
int get_tx_call_count() {
    return GoTransactionData::tx_counter;
}

#ifdef __cplusplus
}
#endif