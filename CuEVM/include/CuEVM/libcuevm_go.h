#pragma once
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <CuEVM/utils/evm_utils.cuh>
#include <CuEVM/state/state_db.cuh>
#include <CuEVM/core/data_structures.cuh>
#include <CuEVM/core/evm_word.cuh>
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


// Transaction data structure
struct GoTransaction {
    go_uint256_t from;
    go_uint256_t to;
    go_uint256_t value;
    uint64_t gas;
    go_uint256_t gasPrice;
    uint64_t nonce;
    std::vector<unsigned char> data;
};

// Transaction batch container
struct GoTransactionData {
    std::vector<GoTransaction> transactions;
    static int tx_counter;
};

// Ensure C linkage for Go to call these functions
#ifdef __cplusplus
extern "C" {
#endif
// CuEVM Go interface functions
GoStateData* create_state_data();
void set_state_root(GoStateData* state, const unsigned char* root, int root_len);
void add_account(GoStateData* state,
                 const unsigned char* addr, int addr_len,
                 const unsigned char* balance, int balance_len,
                 unsigned long nonce,
                 const unsigned char* root, int root_len,
                 const unsigned char* code_hash, int code_hash_len,
                 const unsigned char* code, int code_len,
                 bool has_code);
void add_storage_entry(GoStateData* state,
                       const unsigned char* key, int key_len,
                       const unsigned char* value, int value_len);
int process_state_data_gpu(GoStateData* state);
void free_state_data(GoStateData* state);

// Transaction C API functions
GoTransactionData* create_transaction_data();
void add_transaction(GoTransactionData* txData,
                     const unsigned char* from, int from_len,
                     const unsigned char* to, int to_len,
                     const unsigned char* value, int value_len,
                     uint64_t gas,
                     const unsigned char* gasPrice, int gasPrice_len,
                     uint64_t nonce,
                     const unsigned char* data, int data_len);
int process_transaction_data_gpu(GoTransactionData* txData);
void free_transaction_data(GoTransactionData* txData);

int run_interpreter_go(const char* json_input, unsigned int skip_trace_parsing,
                       unsigned int copy_state_data, unsigned int reuse_state_data);
int get_call_count();
#ifdef __cplusplus
}
#endif
