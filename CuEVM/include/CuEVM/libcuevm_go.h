#pragma once
#include <CuEVM/utils/library_utils.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <CuEVM/core/block_info.cuh>
#include <CuEVM/core/data_structures.cuh>
#include <CuEVM/core/evm_word.cuh>
#include <CuEVM/state/state_db.cuh>
#include <CuEVM/tracer.cuh>
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
/**
 * @file libcuevm_go.h
 * @brief CuEVM Go interface functions and structures.
 *
 * This file defines the C API exposed to Go applications for interacting with the CuEVM library.
 * It provides C-compatible data structures and functions that can be called from Go code
 * through CGO, enabling GPU-accelerated Ethereum Virtual Machine execution from Go applications.
 */

/**
 * @brief C-compatible structure for representing return data from EVM execution.
 *
 * Contains the data returned by a contract execution and its length.
 */
typedef struct {
    uint8_t* data;   /**< Pointer to the return data */
    uint32_t length; /**< Length of the return data */
} ReturnDataEntry;

/**
 * @brief C-compatible structure for representing code coverage data.
 *
 * Contains information about contract addresses and branch coverage metrics for
 * code coverage analysis.
 */
typedef struct {
    char** addresses;       /**< Array of contract addresses as strings */
    uint32_t num_addresses; /**< Number of addresses */

    uint64_t** branch_coverage;        /**< Array of branch coverage arrays */
    uint32_t* branch_coverage_lengths; /**< Length of each branch coverage array */
} CoverageDataEntry;

/**
 * @brief C-compatible structure for representing GPU execution results.
 *
 * This structure contains all the data returned from a batch execution on the GPU,
 * including return data, coverage information, and error codes.
 */
typedef struct {
    ReturnDataEntry* return_data; /**< Array of return data entries */
    uint32_t num_return_data;     /**< Number of return data entries */
    CoverageDataEntry* coverage;  /**< Array of coverage data entries */
    uint32_t num_coverage;        /**< Number of coverage entries */
    uint8_t* error_codes;         /**< Array of error codes for each instance */
    uint8_t allocations_valid;    /**< Track if allocations are valid (1) or freed (0) */
} GPUExecutionResultC;

typedef struct {
    uint32_t* new_coverage_idx;         /**< Array of new coverage indices */
    uint32_t num_new_coverage;          /**< Number of new coverage entries for each batch, size is tx_sequence size*/
    uint32_t* new_bug_idx;              /**< Array of new bug indices */
    uint32_t* new_bug_pc;               /**< Array of new bug PCs */
    uint32_t num_new_bugs;              /**< Number of new bug entries for each batch, size is tx_sequence size*/
    uint8_t* branch_call_data;          /**< Array of branch call data */
    uint32_t* branch_call_data_sizes;   /**< Size of the branch call data */
    uint32_t* branch_call_data_offsets; /**< Offset of the branch call data */
    uint8_t* bug_call_data;             /**< Array of bug call data */
    uint32_t* bug_call_data_sizes;      /**< Size of the bug call data */
    uint32_t* bug_call_data_offsets;    /**< Offset of the bug call data */
} SimplifiedGPUResultSingleBatchC;

typedef struct {
    SimplifiedGPUResultSingleBatchC* results;
    uint32_t num_results;
} SimplifiedGPUResultC;

/**
 * @brief Process a batch of transactions on the GPU.
 *
 * Executes multiple Ethereum transactions in parallel on the GPU.
 *
 * @param[in] blockNumber Array of block numbers
 * @param[in] timeStamp Array of timestamps
 * @param[in] fromAddr Array of sender addresses in byte format
 * @param[in] toAddr Array of recipient addresses in byte format
 * @param[in] values Array of transaction values
 * @param[in] callData Combined call data for all transactions
 * @param[in] callDataLen Length of the combined call data
 * @param[in] dataOffsets Array of offsets into the call data for each transaction
 * @param[in] dataSizes Array of sizes for each transaction's call data
 * @param[in] txBatchCount Number of transactions in each batch
 * @param[in] sequenceLength Length of each sequence of transactions
 * @return SimplifiedGPUResultC* Pointer to the execution results structure
 */
#ifdef BUILD_GO_LIBRARY
SimplifiedGPUResultC* process_batch_transactions(
    const uint64_t* blockNumber, const uint64_t* timeStamp, const unsigned char* fromAddr, const unsigned char* toAddr,
    const unsigned char* values, const unsigned char* callData, uint32_t callDataLen, const uint32_t* dataOffsets,
    const uint32_t* dataSizes, const uint32_t* markerOffsets, const uint32_t* markerCounts, const uint32_t* markerData,
    uint32_t markerDataLen, uint32_t txBatchCount, uint32_t sequenceLength, uint32_t start_seed);
#endif

/**
 * @brief Process JSON state data on the GPU.
 *
 * Loads blockchain state from JSON format into the GPU for execution.
 *
 * @param[in] json_state JSON string containing the state information
 * @param[in] num_instances Number of EVM instances to create
 * @param[in] reset_state Flag to reset the state before processing (true to reset, false to keep existing state)
 * @return int Error code (0 for success, non-zero for error)
 */
int process_json_state_gpu(const char* json_state, uint32_t num_instances, bool reset_state = false,
                           uint32_t skipTxSize = 1, const char* fuzzing_constants = nullptr);

/**
 * @brief Reset the state database to its initial state.
 *
 * Clears all state data and returns the EVM to its initial configuration.
 */
void reset_state_db();

/**
 * @brief Get the results of the most recent GPU execution.
 *
 * Retrieves execution results, including return data and code coverage information.
 *
 * @return GPUExecutionResultC* Pointer to the execution results structure
 */
GPUExecutionResultC* get_gpu_execution_results();

/**
 * @brief Setup fuzzing constants.
 *
 * Sets up fuzzing constants for the GPU.
 *
 * @param[in] fuzzing_constants JSON string containing the fuzzing constants
 */
void setup_fuzzing_constants(const char* fuzzing_constants);

/**
 * @brief Get the number of instances per device.
 *
 * Retrieves the number of instances per device.
 *
 * @return uint32_t Number of instances per device
 */
uint32_t get_num_instances_per_device();

/**
 * @brief Get the results of the most recent GPU execution.
 *
 * Retrieves execution results, including return data and code coverage information.
 *
 * @return SimplifiedGPUExecutionResultC* Pointer to the execution results structure
 */
void get_gpu_execution_results_optimized(SimplifiedGPUResultSingleBatchC* result, const uint8_t* callData,
                                         const uint32_t* dataOffsets, const uint32_t* dataSizes);
// // Function to free GPU execution results
// void free_gpu_execution_results(GPUExecutionResultC* result);

/**
 * @brief Free the memory allocated for a SimplifiedGPUResultC structure.
 *
 * This function frees the memory allocated for a SimplifiedGPUResultC structure,
 * including the arrays and the structure itself.
 *
 * @param[in] result Pointer to the SimplifiedGPUResultC structure to be freed
 */
void free_simplified_gpu_result(SimplifiedGPUResultC* result);

#ifdef __cplusplus
}
#endif
