#include <CuEVM/utils/evm_utils.cuh>
#include <stdio.h>
#include <stdint.h>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

// Move struct definitions OUTSIDE the extern "C" block
// Define the uint256 structure (must match the one in Go)
#define UINT256_WORDS 8
#define UINT256_BITS 256
#define UINT256_BYTES 32

typedef struct {
    uint32_t words[UINT256_WORDS];
} go_uint256_t;

// Account data structure
struct GoAccount {
    go_uint256_t address;
    go_uint256_t balance;
    uint64_t nonce;
    std::vector<unsigned char> root;
    std::vector<unsigned char> codeHash;
    std::vector<unsigned char> code;
    bool hasCode;
    std::vector<go_uint256_t> storageKeys;
    std::vector<go_uint256_t> storageVals;
};

// State data container
struct GoStateData {
    go_uint256_t root;
    std::vector<GoAccount> accounts;
    static int call_counter;
};

// Initialize static counter
int GoStateData::call_counter = 0;

// Helper function declarations
std::string uint256_to_hex(const go_uint256_t& value);
std::string bytes_to_hex(const unsigned char* data, int len);

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
    
    // Initialize to zero
    memset(&state->root, 0, sizeof(go_uint256_t));
    
    // Convert bytes to uint256 (big-endian to little-endian)
    int copyLen = root_len < UINT256_BYTES ? root_len : UINT256_BYTES;
    if (copyLen > 0) {
        for (int i = 0; i < copyLen; i++) {
            int pos = i % 4;
            int word = i / 4;
            state->root.words[word] |= (static_cast<uint32_t>(root[root_len - 1 - i]) << (pos * 8));
        }
    }
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
    
    // Convert address bytes to uint256
    memset(&account.address, 0, sizeof(go_uint256_t));
    int addrCopyLen = addr_len < UINT256_BYTES ? addr_len : UINT256_BYTES;
    for (int i = 0; i < addrCopyLen; i++) {
        int pos = i % 4;
        int word = i / 4;
        account.address.words[word] |= (static_cast<uint32_t>(addr[addr_len - 1 - i]) << (pos * 8));
    }
    
    // Convert balance bytes to uint256
    memset(&account.balance, 0, sizeof(go_uint256_t));
    int balanceCopyLen = balance_len < UINT256_BYTES ? balance_len : UINT256_BYTES;
    for (int i = 0; i < balanceCopyLen; i++) {
        int pos = i % 4;
        int word = i / 4;
        account.balance.words[word] |= (static_cast<uint32_t>(balance[balance_len - 1 - i]) << (pos * 8));
    }
    
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
    
    // Convert key bytes to uint256
    go_uint256_t key_uint256;
    memset(&key_uint256, 0, sizeof(go_uint256_t));
    int keyCopyLen = key_len < UINT256_BYTES ? key_len : UINT256_BYTES;
    for (int i = 0; i < keyCopyLen; i++) {
        int pos = i % 4;
        int word = i / 4;
        key_uint256.words[word] |= (static_cast<uint32_t>(key[key_len - 1 - i]) << (pos * 8));
    }
    
    // Convert value bytes to uint256
    go_uint256_t value_uint256;
    memset(&value_uint256, 0, sizeof(go_uint256_t));
    int valueCopyLen = value_len < UINT256_BYTES ? value_len : UINT256_BYTES;
    for (int i = 0; i < valueCopyLen; i++) {
        int pos = i % 4;
        int word = i / 4;
        value_uint256.words[word] |= (static_cast<uint32_t>(value[value_len - 1 - i]) << (pos * 8));
    }
    
    // Add to the last account's storage
    state->accounts.back().storageKeys.push_back(key_uint256);
    state->accounts.back().storageVals.push_back(value_uint256);
}

// Process state data on GPU - for now just print the data
int process_state_data_gpu(GoStateData* state) {
    printf("CuEVM Go interface: Processing state data in GPU C++ function...\n");
    printf("State root: %s\n", uint256_to_hex(state->root).c_str());
    printf("Number of accounts: %zu\n", state->accounts.size());
    
    try {
        int count = 0;
        for (const auto& account : state->accounts) {
            if (count >= 10) {
                printf("\n... and %zu more accounts\n", state->accounts.size() - 10);
                break;
            }
            
            printf("\n=== Account %s ===\n", uint256_to_hex(account.address).c_str());
            printf("  Balance:  %s\n", uint256_to_hex(account.balance).c_str());
            printf("  Nonce:    %lu\n", account.nonce);
            printf("  Has Code: %s\n", account.hasCode ? "true" : "false");
            
            // Print root
            if (!account.root.empty()) {
                printf("  Root:     %s\n", bytes_to_hex(account.root.data(), account.root.size()).c_str());
            }
            
            // Print code hash
            if (!account.codeHash.empty()) {
                printf("  CodeHash: %s\n", bytes_to_hex(account.codeHash.data(), account.codeHash.size()).c_str());
            }
            
            // Print code information
            if (!account.code.empty()) {
                int codePreviewSize = account.code.size() < 64 ? account.code.size() : 64;
                printf("  Code:     %s...\n", bytes_to_hex(account.code.data(), codePreviewSize).c_str());
                printf("  Code Length: %zu bytes\n", account.code.size());
            } else {
                printf("  Code:     <empty>\n");
            }
            
            // Print storage
            if (!account.storageKeys.empty()) {
                printf("  Storage:\n");
                size_t storageDisplayLimit = 5;
                for (size_t i = 0; i < (account.storageKeys.size() < storageDisplayLimit ? account.storageKeys.size() : storageDisplayLimit); i++) {
                    printf("    %s: %s\n", 
                           uint256_to_hex(account.storageKeys[i]).c_str(), 
                           uint256_to_hex(account.storageVals[i]).c_str());
                }
                if (account.storageKeys.size() > storageDisplayLimit) {
                    printf("    ... and %zu more entries\n", account.storageKeys.size() - storageDisplayLimit);
                }
            } else {
                printf("  Storage:  <empty>\n");
            }
            
            count++;
        }
        
        // This is where we'd launch a CUDA kernel in the future
        // For now, just print that we're incrementing the call counter
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

#ifdef __cplusplus
}
#endif