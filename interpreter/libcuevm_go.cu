#include <CuEVM/libcuevm_go.h>

// Global variables to track state between function calls
static uint32_t g_num_accounts = 0;
static uint32_t g_num_instances = 0;
static CuEVM::StateDb* g_state_db_ptr = nullptr;
static int call_counter = 0;
// Global variable to hold persistent jump table
CuEVM::ContractPCsMap contract_pcs_map;

// Global variable to hold the last execution result
static char* g_last_result = nullptr;

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
    // Reset counter when new state is processed and reset_state is true

    // Set larger heap size for GPU memory if not reusing state data

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaDeviceReset());
    printf("Initializing GPU device\n");
    size_t heap_size = (size_t(1) << 32);  // 4GB
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_size));
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 4 * 1024));
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Resetting call counter\n");

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
    // Modify GPUfromJson to use the persistent jump table
    CuEVM::StateDb::GPUfromJson(g_state_db_ptr, world_state_json, num_transactions, g_num_accounts);
    CuEVM::get_block_info(stateJson);
    printf("Process json state GPU done, found %u accounts\n", g_num_accounts);

    call_counter = 0;
    // Initialize the global jump table for coverage tracking

    // Print the contract_pcs_map after processing the state
    printf("Coverage data after processing state:\n");
    for (const auto& entry : contract_pcs_map) {
        // Print contract address - using proper hex conversion
        printf("Contract ");
        char addr_buf[70]; // Buffer large enough for 0x + 64 hex chars + null terminator
        entry.first.to_hex(addr_buf); // Use the proper to_hex method on evm_word_t
        printf("%s:\n", addr_buf);
        
        // Print coverage information
        printf("  Bytecode size: %zu bytes\n", entry.second.size());
        
        // Print first few PCs (up to 20) for verification
        printf("  First PCs with coverage: ");
        int count = 0;
        for (uint32_t i = 0; i < entry.second.size() && count < 20; i++) {
            if (entry.second[i] == 1) {
                printf("%u ", i);
                count++;
            }
        }
        printf("\n");
    }
    printf("Total contracts with coverage data: %zu\n", contract_pcs_map.size());
  

    // Free JSON object
    cJSON_Delete(stateJson);
    return 0;
}
// ... existing code ...

void print_evm_instances_results(uint32_t num_instances, bool copy_state_data) {
    // Copy trace data from device memory
    CuEVM::simplified_trace_data* trace_data = new CuEVM::simplified_trace_data[num_instances];
    CuEVM::simplified_trace_data* d_trace_data;
    
    // Retrieve the device pointers stored in the global symbols
    CUDA_CHECK(cudaMemcpyFromSymbol(&d_trace_data, global_simplified_trace, sizeof(d_trace_data)));
    
    // Copy the data arrays from device to host memory
    CUDA_CHECK(cudaMemcpy(trace_data, d_trace_data, sizeof(CuEVM::simplified_trace_data) * num_instances,
                          cudaMemcpyDeviceToHost));

    // Copy world state data if requested
    CuEVM::serialized_worldstate_data* world_data = nullptr;
    if (copy_state_data) {
        CuEVM::serialized_worldstate_data* d_world_data;
        world_data = new CuEVM::serialized_worldstate_data[num_instances];
        
        CUDA_CHECK(cudaMemcpyFromSymbol(&d_world_data, global_serialized_worldstate, sizeof(d_world_data)));
        CUDA_CHECK(cudaMemcpy(world_data, d_world_data, sizeof(CuEVM::serialized_worldstate_data) * num_instances,
                              cudaMemcpyDeviceToHost));
    }

    printf("===== Printing results for %u EVM instances =====\n", num_instances);
    
    // Print information for each instance
    for (uint32_t idx = 0; idx < num_instances; idx++) {
        printf("\n----- Instance %u -----\n", idx);
        
        // Print trace data
        printf("Trace Data:\n");
        printf("  Events: %u\n", trace_data[idx].no_events);
        printf("  Calls: %u\n", trace_data[idx].no_calls);
        printf("  Branches: %u\n", trace_data[idx].no_branches);
        
        // Print ALL events
        if (trace_data[idx].no_events > 0) {
            printf("  All Events:\n");
            for (uint32_t e = 0; e < trace_data[idx].no_events; e++) {
                printf("    Event[%u]: PC=%u, OP=%u, Operand1=", 
                       e, 
                       trace_data[idx].events[e].pc, 
                       trace_data[idx].events[e].op);
                trace_data[idx].events[e].operand_1.print();
                printf(", Operand2=");
                trace_data[idx].events[e].operand_2.print();
                printf(", Result=");
                trace_data[idx].events[e].res.print();
                printf("\n");
            }
        }
        
        // Print ALL calls
        if (trace_data[idx].no_calls > 0) {
            printf("  All Calls:\n");
            for (uint32_t c = 0; c < trace_data[idx].no_calls; c++) {
                printf("    Call[%u]: PC=%u, OP=%u, Sender=", 
                       c, 
                       trace_data[idx].calls[c].pc, 
                       trace_data[idx].calls[c].op);
                trace_data[idx].calls[c].sender.print();
                printf(", Receiver=");
                trace_data[idx].calls[c].receiver.print();
                printf(", Value=");
                trace_data[idx].calls[c].value.print();
                printf(", Success=%s\n", 
                       trace_data[idx].calls[c].success == 1 ? "true" : 
                       (trace_data[idx].calls[c].success == 0 ? "false" : "unknown"));
            }
        }
        
        // Print ALL branches
        if (trace_data[idx].no_branches > 0) {
            printf("  All Branches:\n");
            for (uint32_t b = 0; b < trace_data[idx].no_branches; b++) {
                printf("    Branch[%u]: PC_src=%u, PC_dst=%u, PC_missed=%u, Distance=",
                       b,
                       trace_data[idx].branches[b].pc_src,
                       trace_data[idx].branches[b].pc_dst,
                       trace_data[idx].branches[b].pc_missed);
                trace_data[idx].branches[b].distance.print();
                printf("\n");
            }
        }
        
        // Print ALL world state data if available
        if (copy_state_data && world_data != nullptr) {
            printf("\n  World State:\n");
            printf("    Accounts: %u\n", world_data[idx].no_accounts);
            printf("    Storage Elements: %u\n", world_data[idx].no_storage_elements);
            
            // Print ALL account info
            if (world_data[idx].no_accounts > 0) {
                printf("    All Accounts:\n");
                for (uint32_t a = 0; a < world_data[idx].no_accounts; a++) {
                    char addr_buf[70];
                    world_data[idx].addresses[a].to_hex(addr_buf);
                    char balance_buf[70];
                    world_data[idx].balance[a].to_hex(balance_buf);
                    
                    printf("      Account[%u]: Address=%s, Balance=%s, Nonce=%u\n", 
                           a, 
                           addr_buf, 
                           balance_buf,
                           world_data[idx].nonce[a]);
                }
            }
            
            // Print ALL storage elements
            if (world_data[idx].no_storage_elements > 0) {
                printf("    All Storage Elements:\n");
                for (uint32_t s = 0; s < world_data[idx].no_storage_elements; s++) {
                    char key_buf[70];
                    world_data[idx].storage_keys[s].to_hex(key_buf);
                    char value_buf[70];
                    world_data[idx].storage_values[s].to_hex(value_buf);
                    
                    printf("      Storage[%u]: Account_Index=%u, Key=%s, Value=%s\n",
                           s,
                           world_data[idx].storage_indexes[s],
                           key_buf,
                           value_buf);
                }
            }
        }
    }
    
    printf("\n===== End of results =====\n");
    
    // Clean up memory
    delete[] trace_data;
    if (copy_state_data && world_data != nullptr) {
        delete[] world_data;
    }
}
// Creates transaction list on both host and device

CuEVM::transaction::TransactionList* 
create_transaction_list(
    const unsigned char* fromAddr, 
    const unsigned char* toAddr, 
    const unsigned char* values,
    const unsigned char* callData, 
    int callDataLen, 
    const uint32_t* dataOffsets,
    int dataOffsetsLen, 
    const uint32_t* dataSizes, 
    int dataSizesLen, 
    int txCount,
    bool copy_state_data=false
) {
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
    
    printf("Transaction batch prepared for GPU\n");
    host_transaction_list->print();

    // trace and serialized state data
   // Simplified trace data
    CuEVM::simplified_trace_data* d_trace_data;

    CUDA_CHECK(cudaMalloc(&d_trace_data, txCount * sizeof(CuEVM::simplified_trace_data)));
    cudaMemset(d_trace_data, 0, txCount * sizeof(CuEVM::simplified_trace_data));
    cudaMemcpyToSymbol(global_simplified_trace, &d_trace_data, sizeof(CuEVM::simplified_trace_data*));

    if (copy_state_data) {
        CuEVM::serialized_worldstate_data* d_serialized_worldstate_data;
        CUDA_CHECK(
            cudaMalloc(&d_serialized_worldstate_data, txCount * sizeof(CuEVM::serialized_worldstate_data)));
        cudaMemset(d_serialized_worldstate_data, 0, txCount * sizeof(CuEVM::serialized_worldstate_data));
        cudaMemcpyToSymbol(global_serialized_worldstate, &d_serialized_worldstate_data,
                           sizeof(CuEVM::serialized_worldstate_data*));
    }
    // Free host memory
    delete[] host_transaction_list->gas_limit;
    delete[] host_transaction_list->value;
    delete[] host_transaction_list->call_data_offset;
    delete[] host_transaction_list->call_data_size;
    if (host_transaction_list->call_data != nullptr) {
        delete[] host_transaction_list->call_data;
    }
    delete host_transaction_list;
    return  d_transaction_list_ptr;
}


// Simplified batch transaction processing with single from/to address
GPUExecutionResultC* process_batch_transactions(const unsigned char* fromAddr, const unsigned char* toAddr, const unsigned char* values,
                               const unsigned char* callData, int callDataLen, const uint32_t* dataOffsets,
                               int dataOffsetsLen, const uint32_t* dataSizes, int dataSizesLen, int txCount) {
    printf("CuEVM Go interface: Processing batch of %d transactions, call number: %d\n", txCount, call_counter);

    try {
        // Update global num_instances if provided
        if (txCount > 0) {
            g_num_instances = txCount;
        }
        bool copy_state_data = false;
        // Create and transfer transaction list to GPU
        CuEVM::transaction::TransactionList* d_transaction_list_ptr = create_transaction_list(
            fromAddr, toAddr, values, callData, callDataLen, 
            dataOffsets, dataOffsetsLen, dataSizes, dataSizesLen, txCount, copy_state_data
        );

        // Initialize memory pool using the globally stored account count
        // Only create memory pool if not reusing state or first call
        if (call_counter == 0) {
            printf("Creating memory pool with %u accounts and %u instances\n", g_num_accounts, g_num_instances);
            CuEVM::memory_pool::create_memory_pool(g_num_instances, g_num_accounts);
        } else {
            printf("Reusing existing memory pool\n");
            printf("Clearing memory pool\n");
            printf("g_num_instances: %u\n", g_num_instances);
            printf("g_num_accounts: %u\n", g_num_accounts);

            CuEVM::memory_pool::clear_memory_pool();

        }
// tracer
#ifdef EIP_3155
    const size_t BUFFER_SIZE = 100 * 1024 * 1024;  // 100 MB
    char *d_buffer;
    cudaMalloc(&d_buffer, BUFFER_SIZE);
#endif
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
                                                                                  g_num_instances
#ifdef EIP_3155
                                                                                  ,
                                                                                  d_buffer, BUFFER_SIZE
#endif
                                                                                  ,copy_state_data
        );

        // Record timing and synchronize
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("GPU kernel execution time: %f milliseconds\n", milliseconds);

        // Check for errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            return nullptr;
        }

        printf("GPU execution completed successfully\n");
        print_evm_instances_results(g_num_instances, copy_state_data);
        
#ifdef EIP_3155
    // After kernel execution, copy the buffer back to the host
    char *h_buffer = new char[BUFFER_SIZE];
    cudaMemcpy(h_buffer, d_buffer, BUFFER_SIZE, cudaMemcpyDeviceToHost);
    CuEVM::utils::print_tracer_data(h_buffer);

    // Clean up
    delete[] h_buffer;
    cudaFree(d_buffer);
#endif

        // Clean up transaction lists
        // cleanup_transaction_list(d_transaction_list_ptr, callDataLen);
        GPUExecutionResultC* result = get_gpu_execution_results();
        CuEVM::freeTransactionList(d_transaction_list_ptr);
        CuEVM::freeTraceData(copy_state_data);
        // Clean up CUDA events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // Increment call counter
        call_counter++;
        printf("Call number: %d\n", call_counter);
        return result;  // Success
    } catch (const std::exception& e) {
        printf("Error in process_batch_transactions: %s\n", e.what());
        return nullptr;  // Error
    } catch (...) {
        printf("Unknown error in process_batch_transactions\n");
        return nullptr;  // Unknown error
    }
}


GPUExecutionResultC* get_gpu_execution_results() {
    // Allocate result structure
    GPUExecutionResultC* result = (GPUExecutionResultC*)malloc(sizeof(GPUExecutionResultC));
    if (result == nullptr) {
        printf("Failed to allocate memory for GPUExecutionResultC\n");
        return nullptr;
    }
    
    // Retrieve trace data from device memory - similar to print_evm_instances_results
    CuEVM::simplified_trace_data* trace_data = new CuEVM::simplified_trace_data[g_num_instances];
    CuEVM::simplified_trace_data* d_trace_data;
    
    // Retrieve the device pointers stored in the global symbols
    CUDA_CHECK(cudaMemcpyFromSymbol(&d_trace_data, global_simplified_trace, sizeof(d_trace_data)));
    
    // Copy the data arrays from device to host memory
    CUDA_CHECK(cudaMemcpy(trace_data, d_trace_data, sizeof(CuEVM::simplified_trace_data) * g_num_instances,
                          cudaMemcpyDeviceToHost));
    
    // Initialize return data
    result->num_return_data = g_num_instances;
    result->return_data = (ReturnDataEntry*)malloc(sizeof(ReturnDataEntry) * result->num_return_data);
    if (result->return_data == nullptr) {
        printf("Failed to allocate memory for return_data\n");
        delete[] trace_data;
        free(result);
        return nullptr;
    }
    
    // Initialize all return data entries as empty
    for (uint32_t i = 0; i < result->num_return_data; i++) {
        result->return_data[i].length = 0;
        result->return_data[i].data = nullptr;
    }
    
    // Set up coverage data for all instances
    result->num_coverage = g_num_instances;
    result->coverage = (CoverageDataEntry*)malloc(sizeof(CoverageDataEntry) * result->num_coverage);
    if (result->coverage == nullptr) {
        printf("Failed to allocate memory for coverage\n");
        delete[] trace_data;
        free(result->return_data);
        free(result);
        return nullptr;
    }


    result->success_status = (uint8_t*)malloc(sizeof(uint8_t) * g_num_instances);
    memset(result->success_status, 0, sizeof(uint8_t) * g_num_instances);
    
    printf("===== Initializing Coverage Data for %u Instances =====\n", g_num_instances);
    
    // Process each instance's trace data
    for (uint32_t idx = 0; idx < g_num_instances; idx++) {
        // Set up coverage data structure
        result->coverage[idx].num_addresses = 1;  // One contract per instance for simplicity
        result->coverage[idx].addresses = (char**)malloc(sizeof(char*) * result->coverage[idx].num_addresses);
        result->coverage[idx].pc_coverage = (uint8_t**)malloc(sizeof(uint8_t*) * result->coverage[idx].num_addresses);
        result->coverage[idx].pc_coverage_lengths = (uint32_t*)malloc(sizeof(uint32_t) * result->coverage[idx].num_addresses);
        
        // Use contract address from receiver of first call if available, otherwise use placeholder
        if (trace_data[idx].no_calls > 0) {
            char addr_buf[70];
            trace_data[idx].calls[0].receiver.to_hex(addr_buf);
            result->coverage[idx].addresses[0] = strdup(addr_buf);
            result->success_status[idx] = trace_data[idx].calls[trace_data[idx].no_calls - 1].success;
        } else {
            // Use placeholder address
            result->coverage[idx].addresses[0] = strdup("0x0000000000000000000000000000000000000000");
        }
        
        // Get valid PCs vector from the hash map
        uint32_t coverage_size = 0;
        std::vector<uint8_t> valid_pcs;
        
        if (trace_data[idx].no_calls > 0) {
            auto it = contract_pcs_map.find(trace_data[idx].calls[0].receiver);
            if (it != contract_pcs_map.end()) {
                valid_pcs = it->second;
                coverage_size = valid_pcs.size();
            } else {
                printf("Instance %u: Contract not found in PC map, skipping\n", idx);
                free(result->coverage[idx].addresses[0]);
                free(result->coverage[idx].addresses);
                free(result->coverage[idx].pc_coverage);
                free(result->coverage[idx].pc_coverage_lengths);
                
                result->coverage[idx].num_addresses = 0;
                result->coverage[idx].addresses = nullptr;
                result->coverage[idx].pc_coverage = nullptr;
                result->coverage[idx].pc_coverage_lengths = nullptr;
                continue;
            }
        } else {
            printf("Instance %u: No calls available, skipping\n", idx);
            free(result->coverage[idx].addresses[0]);
            free(result->coverage[idx].addresses);
            free(result->coverage[idx].pc_coverage);
            free(result->coverage[idx].pc_coverage_lengths);
            
            result->coverage[idx].num_addresses = 0;
            result->coverage[idx].addresses = nullptr;
            result->coverage[idx].pc_coverage = nullptr;
            result->coverage[idx].pc_coverage_lengths = nullptr;
            continue;
        }
        
        // Store the coverage length
        result->coverage[idx].pc_coverage_lengths[0] = coverage_size;
        result->coverage[idx].pc_coverage[0] = (uint8_t*)malloc(coverage_size);
        memset(result->coverage[idx].pc_coverage[0], 0, coverage_size);
        
        // Get the last PC from the last call (for execution termination)
        uint32_t last_pc = 0;
        if (trace_data[idx].no_calls > 0) {
            last_pc = trace_data[idx].calls[trace_data[idx].no_calls - 1].last_pc;
            printf("Instance %u: Last PC from last call: %u, success status: %u\n", idx, last_pc, result->success_status[idx]);
        }
        
        // Simple linear PC execution with jumps
        uint32_t pc = 0;
        uint32_t jump_idx = 0;  // Current jump index
        
        // Process all branches/jumps
        while (jump_idx < trace_data[idx].no_branches) {
            // Mark current PC as covered if it's a valid PC and within coverage size
            if (pc < coverage_size && valid_pcs[pc] == 1) {
                result->coverage[idx].pc_coverage[0][pc] = 1;
            }
            
            // Check if we need to jump
            if (trace_data[idx].branches[jump_idx].pc_src == pc) {
                // Jump to destination
                pc = trace_data[idx].branches[jump_idx].pc_dst;
                
                // Mark the destination PC as covered immediately
                if (pc < coverage_size && valid_pcs[pc] == 1) {
                    result->coverage[idx].pc_coverage[0][pc] = 1;
                }
                
                // Move to next jump
                jump_idx++;
            } else {
                // Move to next PC
                pc++;
            }
        }
        
        // After processing all jumps, continue execution from current PC to the last PC
        while (pc <= last_pc && pc < coverage_size) {
            if (valid_pcs[pc] == 1) {
                result->coverage[idx].pc_coverage[0][pc] = 1;
            }
            pc++;
        }
        
        // Debug print coverage info
        printf("Instance %u coverage:\n", idx);
        printf("  Contract address: %s\n", result->coverage[idx].addresses[0]);
        printf("  Bytecode size: %u bytes\n", coverage_size);
        
        // Count covered PCs
        uint32_t covered_count = 0;
        for (uint32_t i = 0; i < coverage_size; i++) {
            if (result->coverage[idx].pc_coverage[0][i] == 1) {
                covered_count++;
                printf("%u ", i);
            }
        }
        printf("\n");
        printf("  Total covered PCs: %u\n", covered_count);
    }
    
    // Clean up trace data
    delete[] trace_data;
    
    return result;
}

void free_gpu_execution_results(GPUExecutionResultC* result) {
    if (result == nullptr) return;
    
    // Free return data
    for (uint32_t i = 0; i < result->num_return_data; i++) {
        free(result->return_data[i].data);
    }
    free(result->return_data);
    
    // Free coverage data
    for (uint32_t i = 0; i < result->num_coverage; i++) {
        // Free addresses
        for (uint32_t j = 0; j < result->coverage[i].num_addresses; j++) {
            free(result->coverage[i].addresses[j]);
        }
        free(result->coverage[i].addresses);
        
        // Free PC coverage
        for (uint32_t j = 0; j < result->coverage[i].num_addresses; j++) {
            free(result->coverage[i].pc_coverage[j]);
        }
        free(result->coverage[i].pc_coverage);
        free(result->coverage[i].pc_coverage_lengths);
    }
    free(result->coverage);
    // Free success status data
    if (result->success_status != nullptr) {
        free(result->success_status); // Freed because malloc was used
    }
    // Free the result itself
    free(result);
}

#ifdef __cplusplus
}
#endif