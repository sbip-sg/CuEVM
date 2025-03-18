#include <CuEVM/libcuevm_go.h>

// Global variables to track state between function calls
static uint32_t g_num_accounts = 0;
static uint32_t g_num_instances = 0;
static CuEVM::StateDb* g_state_db_ptr = nullptr;

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

using namespace CuEVM;
// Process state data on GPU - initialize StateDB and print data

// In the future, we would implement the run_interpreter_go function similar to run_interpreter_pyobject
// For now, just a placeholder that increments the counter and returns success
int run_interpreter_go(const char* json_input, uint32_t skip_trace_parsing, uint32_t copy_state_data,
                       uint32_t reuse_state_data) {
    printf("Go interface: Running interpreter with JSON input\n");
    printf("Run configuration skip_trace_parsing: %d, copy_state_data: %d, reuse_state_data: %d\n", skip_trace_parsing,
           copy_state_data, reuse_state_data);

    // This is where we would process the JSON input and run the CUDA kernel
    // For now, just return success
    return 0;
}

int process_json_state_gpu(const char* json_state, uint32_t num_instances) {
    cJSON* stateJson = cJSON_Parse(json_state);
    if (stateJson == NULL) {
        printf("Error parsing JSON state\n");
        return -1;
    }

    printf("Process json state GPU , num_instances: %u\n", num_instances);
    cJSON* world_state_json = cJSON_GetObjectItemCaseSensitive(stateJson, "pre");

    // Store the num_instances value globally
    g_num_instances = num_instances;
    uint32_t num_transactions = num_instances;

    // Free previous state DB if it exists
    if (g_state_db_ptr != nullptr) {
        delete g_state_db_ptr;
        g_state_db_ptr = nullptr;
    }

    // Initialize and store the state DB and account count globally
    CuEVM::StateDb::GPUfromJson(g_state_db_ptr, world_state_json, num_transactions, g_num_accounts);
    CuEVM::get_block_info(stateJson);
    printf("Process json state GPU done, found %u accounts\n", g_num_accounts);

    // Free JSON object
    cJSON_Delete(stateJson);
    return 0;
}

// Simplified batch transaction processing with single from/to address
int process_batch_transactions(const unsigned char* fromAddr, const unsigned char* toAddr, const unsigned char* values,
                               const unsigned char* callData, int callDataLen, const uint32_t* dataOffsets,
                               int dataOffsetsLen, const uint32_t* dataSizes, int dataSizesLen, int txCount) {
    printf("CuEVM Go interface: Processing batch of %d transactions\n", txCount);

    try {
        // Update global num_instances if provided
        if (txCount > 0) {
            g_num_instances = txCount;
        }

        // Set up CUDA environment
        CUDA_CHECK(cudaSetDevice(0));

        // Set larger heap size for GPU memory
        size_t heap_size = (size_t(1) << 32);  // 4GB
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_size));
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 4 * 1024));
        CUDA_CHECK(cudaDeviceSynchronize());

        // Create TransactionList on host
        CuEVM::transaction::TransactionList* host_transaction_list = new CuEVM::transaction::TransactionList();
        host_transaction_list->size = txCount;

        // Allocate memory for transaction data on host
        host_transaction_list->value = new evm_word_t[txCount];
        host_transaction_list->call_data_offset = new uint32_t[txCount];
        host_transaction_list->call_data_size = new uint32_t[txCount];

        // Use fixed gas limit for all transactions
        host_transaction_list->gas_limit = new uint64_t[txCount];
        for (int i = 0; i < txCount; i++) {
            host_transaction_list->gas_limit[i] = 30000000;  // Fixed gas limit
        }

        // Copy data offsets and sizes
        memcpy(host_transaction_list->call_data_offset, dataOffsets, txCount * sizeof(uint32_t));
        memcpy(host_transaction_list->call_data_size, dataSizes, txCount * sizeof(uint32_t));

        // Handle sender (common for all transactions)
        uint256_from_bytes(&host_transaction_list->sender, fromAddr, 32);

        // Handle recipient (common for all transactions)
        uint256_from_bytes(&host_transaction_list->to, toAddr, 32);

        // Set fixed gas price and nonce
        host_transaction_list->gas_price = 1;  // 1 wei fixed gas price
        host_transaction_list->nonce = 0;      // Fixed nonce

        // Handle values for each transaction
        for (int i = 0; i < txCount; i++) {
            uint256_from_bytes(&host_transaction_list->value[i], &values[i * 32], 32);
        }

        // Handle call data (if any)
        uint8_t* host_call_data = nullptr;
        if (callDataLen > 0) {
            host_call_data = new uint8_t[callDataLen];
            memcpy(host_call_data, callData, callDataLen);
            host_transaction_list->call_data = host_call_data;
        } else {
            host_transaction_list->call_data = nullptr;
        }

        // Set transaction type (default to 0)
        host_transaction_list->type = 0;

        printf("Transaction batch prepared for GPU\n");
        host_transaction_list->print();

        // Now we need to allocate GPU memory and transfer data
        CuEVM::transaction::TransactionList* d_transaction_list_ptr;
        CuEVM::transaction::TransactionList* temp_transaction_list = new CuEVM::transaction::TransactionList();
        memcpy(temp_transaction_list, host_transaction_list, sizeof(CuEVM::transaction::TransactionList));

        // Allocate GPU memory for transaction data
        CUDA_CHECK(cudaMalloc(&temp_transaction_list->value, txCount * sizeof(evm_word_t)));
        CUDA_CHECK(cudaMalloc(&temp_transaction_list->gas_limit, txCount * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&temp_transaction_list->call_data_offset, txCount * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&temp_transaction_list->call_data_size, txCount * sizeof(uint32_t)));

        // Allocate GPU memory for call data if needed
        if (callDataLen > 0) {
            CUDA_CHECK(cudaMalloc(&temp_transaction_list->call_data, callDataLen * sizeof(uint8_t)));
            CUDA_CHECK(cudaMemcpy(temp_transaction_list->call_data, host_call_data, callDataLen * sizeof(uint8_t),
                                  cudaMemcpyHostToDevice));
        }

        // Copy data from host to GPU
        CUDA_CHECK(cudaMemcpy(temp_transaction_list->value, host_transaction_list->value, txCount * sizeof(evm_word_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(temp_transaction_list->gas_limit, host_transaction_list->gas_limit,
                              txCount * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(temp_transaction_list->call_data_offset, host_transaction_list->call_data_offset,
                              txCount * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(temp_transaction_list->call_data_size, host_transaction_list->call_data_size,
                              txCount * sizeof(uint32_t), cudaMemcpyHostToDevice));

        // Allocate memory for the transaction list on GPU and copy the structure
        CUDA_CHECK(cudaMalloc(&d_transaction_list_ptr, sizeof(CuEVM::transaction::TransactionList)));
        CUDA_CHECK(cudaMemcpy(d_transaction_list_ptr, temp_transaction_list,
                              sizeof(CuEVM::transaction::TransactionList), cudaMemcpyHostToDevice));

        // Initialize memory pool using the globally stored account count
        printf("Using %u accounts and %u instances for memory pool\n", g_num_accounts, g_num_instances);
        CuEVM::memory_pool::create_memory_pool(g_num_instances, g_num_accounts);

        // Configure kernel launch parameters
        uint32_t num_blocks = (g_num_instances + INSTANCES_PER_BLOCK - 1) / INSTANCES_PER_BLOCK;
        printf("Running %d instances on GPU, blocks: %d, threads per block: %d\n", g_num_instances, num_blocks,
               INSTANCES_PER_BLOCK);

        // Create CUDA timing events
        cudaEvent_t start, stop;
        float milliseconds = 0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        // Execute the GPU kernel with the device pointer, not the host pointer
        CuEVM::kernel_evm_multiple_instances<<<num_blocks, INSTANCES_PER_BLOCK>>>(d_transaction_list_ptr,
                                                                                  g_num_instances);

        // Record timing and synchronize
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("GPU kernel execution time: %f milliseconds\n", milliseconds);

        // Check for errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            return -1;
        }

        printf("GPU execution completed successfully\n");

        // Clean up device memory
        cudaFree(temp_transaction_list->value);
        cudaFree(temp_transaction_list->gas_limit);
        cudaFree(temp_transaction_list->call_data_offset);
        cudaFree(temp_transaction_list->call_data_size);
        if (callDataLen > 0) {
            cudaFree(temp_transaction_list->call_data);
        }
        cudaFree(d_transaction_list_ptr);
        delete temp_transaction_list;

        // Clean up host memory
        delete[] host_transaction_list->gas_limit;
        delete[] host_transaction_list->value;
        delete[] host_transaction_list->call_data_offset;
        delete[] host_transaction_list->call_data_size;
        if (host_call_data != nullptr) {
            delete[] host_call_data;
        }
        delete host_transaction_list;

        // Clean up CUDA events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        return 0;  // Success
    } catch (const std::exception& e) {
        printf("Error in process_batch_transactions: %s\n", e.what());
        return 1;  // Error
    } catch (...) {
        printf("Unknown error in process_batch_transactions\n");
        return 2;  // Unknown error
    }
}

#ifdef __cplusplus
}
#endif