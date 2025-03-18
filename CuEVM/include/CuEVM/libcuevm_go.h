#pragma once
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <CuEVM/core/block_info.cuh>
#include <CuEVM/core/data_structures.cuh>
#include <CuEVM/core/evm_word.cuh>
#include <CuEVM/state/state_db.cuh>
#include <CuEVM/utils/evm_utils.cuh>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
// Move struct definitions OUTSIDE the extern "C" block
// Define the uint256 structure (must match the one in Go)
#define UINT256_WORDS 8
#define UINT256_BITS 256
#define UINT256_BYTES 32

// Ensure C linkage for Go to call these functions
#ifdef __cplusplus
extern "C" {
#endif
// CuEVM Go interface functions

int run_interpreter_go(const char* json_input, unsigned int skip_trace_parsing, unsigned int copy_state_data,
                       unsigned int reuse_state_data);

// Add this to the extern "C" block
int process_json_state_gpu(const char* json_state, uint32_t num_instances);

// Batch transaction processing with single from/to address
int process_batch_transactions(const unsigned char* fromAddr, const unsigned char* toAddr, const unsigned char* values,
                               const unsigned char* callData, int callDataLen, const uint32_t* dataOffsets,
                               int dataOffsetsLen, const uint32_t* dataSizes, int dataSizesLen, int txCount);

#ifdef __cplusplus
}
#endif
