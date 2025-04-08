#pragma once
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <CuEVM/utils/evm_utils.cuh>
#include <CuEVM/state/state_db.cuh>
#include <CuEVM/core/data_structures.cuh>
#include <CuEVM/core/evm_word.cuh>
#include <CuEVM/core/block_info.cuh>
#include <CuEVM/utils/library_utils.h>
#include <CuEVM/tracer.cuh>
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




// Ensure C linkage for Go to call these functions
#ifdef __cplusplus
extern "C" {
#endif
// CuEVM Go interface functions

// // Define in the C++ side (gpu_execution.h or similar)
// struct GPUExecutionResult {
//     // Return data for each instance
//     std::vector<std::vector<uint8_t>> returnData;
    
//     // Coverage data for each instance
//     struct CoverageData {
//         std::vector<std::string> addresses;  // Contract addresses as hex strings
//         std::vector<std::vector<uint8_t>> pcCoverage;  // PC coverage for each address
//     };
//     std::vector<CoverageData> coverage;
// };

// Define C-compatible structures that can be shared with Go
typedef struct {
    uint8_t* data;  // Pointer to the return data
    uint32_t length;  // Length of the return data
} ReturnDataEntry;

typedef struct {
    char** addresses;  // Array of contract addresses as strings
    uint32_t num_addresses;  // Number of addresses
    
    uint64_t** branch_coverage;  // Array of branch coverage arrays
    uint32_t* branch_coverage_lengths;  // Length of each branch coverage array
} CoverageDataEntry;

typedef struct {
    ReturnDataEntry* return_data;  // Array of return data entries
    uint32_t num_return_data;  // Number of7 return data entries
    CoverageDataEntry* coverage;  // Array of coverage data entries
    uint32_t num_coverage;  // Number of coverage entries
    uint8_t* success_status; 
} GPUExecutionResultC;


int run_interpreter_go(const char* json_input, unsigned int skip_trace_parsing,
                       unsigned int copy_state_data, unsigned int reuse_state_data);

// Updated function declaration with reuse_state_data parameter
GPUExecutionResultC* process_batch_transactions(const unsigned char* fromAddr, const unsigned char* toAddr, const unsigned char* values,
                               const unsigned char* callData, int callDataLen, const uint32_t* dataOffsets,
                               int dataOffsetsLen, const uint32_t* dataSizes, int dataSizesLen, int txCount);

// Updated function declaration with reset_state parameter
int process_json_state_gpu(const char* json_state, uint32_t num_instances);


// Function to get GPU execution results
GPUExecutionResultC* get_gpu_execution_results();

// // Function to free GPU execution results
// void free_gpu_execution_results(GPUExecutionResultC* result);


#ifdef __cplusplus
}
#endif
