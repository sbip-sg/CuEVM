#include <CuEVM/libcuevm_go.h>
#include <omp.h>

#include <cassert>
#include <sstream>
#include <unordered_map>
#include <vector>

// Global variables to track state between function calls
static uint32_t g_num_accounts = 0;
static uint32_t g_num_instances_per_device = 0;
static uint32_t g_skipTxSize = 1;
// static uint32_t g_num_instances;
static std::vector<CuEVM::StateDb*> g_state_db_ptr;
static std::vector<CuEVM::StateDb*> g_snapshot_state_db_ptr;  // store original snapshot state db, not free/modified

static std::vector<BranchInfoEntry*> d_branch_infos;    // vector size = num gpus
static std::vector<BugInfoEntry*> d_bug_infos;          // vector size = num gpus
static std::vector<StorageInfoEntry*> d_storage_infos;  // vector size = num gpus
static std::vector<uint32_t*>
    d_new_coverage_bitmaps;  // vector size = num gpus for tracking each thread if coverage hit
static std::vector<GPUFeedbackCount*> d_gpu_feedback_counts;  // vector size = num gpus

static std::vector<uint64_t*> device_block_numbers;  // vector size = num gpus for persistent block numbers
static std::vector<uint64_t*> device_time_stamps;    // vector size = num gpus for persistent time stamps

static int call_counter = 0;

static int g_num_gpus = 1;

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

uint32_t get_num_instances_per_device() { return g_num_instances_per_device; }

// Process state data on GPU - initialize StateDB and print data

int process_json_state_gpu(const char* json_state, uint32_t num_instances, bool reset_state, uint32_t skipTxSize,
                           const char* fuzzing_constants, uint32_t* markerData, uint32_t markerDataLen) {
    // Reset counter when new state is processed and reset_state is true
    if (reset_state) {
        reset_state_db();
        return 0;
    }
    g_skipTxSize = skipTxSize;
    // Set larger heap size for GPU memory if not reusing state data
    cudaError_t err = cudaGetDeviceCount(&g_num_gpus);
    if (err != cudaSuccess || g_num_gpus <= 0) {
        printf("Error getting GPU count or no GPUs found: %s\n", cudaGetErrorString(err));
        g_num_gpus = 0;
        return -1;
    }
    // debug printing marker data
    printf("CuEVM Debug: markerDataLen: %d\n", markerDataLen);
    for (int i = 0; i < markerDataLen; i++) {
        printf("CuEVM Debug: markerData[%d]: %d\n", i, markerData[i]);
    }

    printf("Found %d GPUs.\n", g_num_gpus);
    printf("Resetting GPU devices\n");

    for (int i = 0; i < g_num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaDeviceReset());
        printf("Initializing GPU device\n");
        size_t heap_size = (size_t(1) << 32);  // 4GB
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_size));
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 4 * 1024));
        CUDA_CHECK(cudaDeviceSynchronize());

        // create and set the coverage bitmap once for each fuzzing campaign
        uint32_t* d_coverage_bitmap;

        CUDA_CHECK(cudaMalloc(&d_coverage_bitmap, BITMAP_SIZE * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_coverage_bitmap, 0, BITMAP_SIZE * sizeof(uint32_t)));

        CUDA_CHECK(cudaMemcpyToSymbol(CuEVM::g_events_bitmap, &d_coverage_bitmap, sizeof(uint32_t*)));

        // total bug table
        uint32_t* d_total_bug_table;
        CUDA_CHECK(cudaMalloc(&d_total_bug_table, MAX_NEW_BUGS * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpyToSymbol(g_total_bug_table, &d_total_bug_table, sizeof(uint32_t*)));

        // total bug count

        uint32_t* d_total_bug_count;
        CUDA_CHECK(cudaMalloc(&d_total_bug_count, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpyToSymbol(g_total_bug_count, &d_total_bug_count, sizeof(uint32_t*)));
        // reset bug counts:
        CUDA_CHECK(cudaMemset(d_total_bug_count, 0, sizeof(uint32_t)));

        setup_fuzzing_constants(fuzzing_constants, markerData, markerDataLen);

        // setup and push back to the vector
        // branch tracker
        BranchInfoEntry* d_new_branch_info;
        CUDA_CHECK(cudaMalloc(&d_new_branch_info, MAX_NEW_BRANCHES * sizeof(BranchInfoEntry)));
        CUDA_CHECK(cudaMemcpyToSymbol(g_new_branch_info, &d_new_branch_info, sizeof(BranchInfoEntry*)));
        // Storage tracker
        StorageInfoEntry* d_new_storage_info;
        CUDA_CHECK(cudaMalloc(&d_new_storage_info, MAX_NEW_STORAGE * sizeof(StorageInfoEntry)));
        CUDA_CHECK(cudaMemcpyToSymbol(g_new_storage_info, &d_new_storage_info, sizeof(StorageInfoEntry*)));
        // Bug tracker
        BugInfoEntry* d_new_bug_info;
        CUDA_CHECK(cudaMalloc(&d_new_bug_info, MAX_NEW_BUGS * sizeof(BugInfoEntry)));
        CUDA_CHECK(cudaMemcpyToSymbol(g_new_bug_info, &d_new_bug_info, sizeof(BugInfoEntry*)));

        // GPU feedback count
        GPUFeedbackCount* d_gpu_feedback_count;
        CUDA_CHECK(cudaMalloc(&d_gpu_feedback_count, sizeof(GPUFeedbackCount)));
        CUDA_CHECK(cudaMemcpyToSymbol(g_gpu_feedback_count, &d_gpu_feedback_count, sizeof(GPUFeedbackCount*)));

        uint32_t num_integer_elements_bitmap = (num_instances + 31) / 32;
        printf("num_integer_elements_bitmap: %u\n", num_integer_elements_bitmap);

        d_branch_infos.push_back(d_new_branch_info);
        d_storage_infos.push_back(d_new_storage_info);
        d_bug_infos.push_back(d_new_bug_info);

        d_gpu_feedback_counts.push_back(d_gpu_feedback_count);

        // block number and timestamp
        uint64_t* d_block_number;
        CUDA_CHECK(cudaMalloc(&d_block_number, num_instances * sizeof(uint64_t)));
        device_block_numbers.push_back(d_block_number);
        uint64_t* d_time_stamp;
        CUDA_CHECK(cudaMalloc(&d_time_stamp, num_instances * sizeof(uint64_t)));
        device_time_stamps.push_back(d_time_stamp);
    }
    // Include <cassert> header at the top of the file for this to work.
    // Standard assert takes only one argument (the condition).
    if (num_instances % g_num_gpus != 0) {
        printf("Error: Number of instances (%u) must be a multiple of the number of GPUs (%d).\n", num_instances,
               g_num_gpus);
        printf("Please adjust the number of instances/workers or number of available GPUs in your configuration.\n");
        // Optionally, return an error code or exit if this is a fatal condition
        return -1;  // Indicate an error
    }
    printf("Resetting call counter\n");

    cJSON* stateJson = cJSON_Parse(json_state);
    if (stateJson == NULL) {
        printf("Error parsing JSON state\n");
        return -1;
    }

    printf("Process json state GPU , num_instances: %u\n", num_instances);
    // cJSON* world_state_json = cJSON_GetObjectItemCaseSensitive(stateJson, "pre");

    // Store the num_instances value globally
    g_num_instances_per_device = num_instances / g_num_gpus;
    // g_num_instances = num_instances;
    // Free previous state DB if it exists
    // CuEVM debug, device reset over library calls, dangling pointer
    // TODO: more efficient state db reset mechanism
    for (int i = 0; i < g_num_gpus; i++) {
        if (g_state_db_ptr.size() <= i) {
            g_state_db_ptr.push_back(nullptr);
        } else {
            // cudaDeviceReset invalidates pointers, so just set to nullptr
            g_state_db_ptr[i] = nullptr;
        }

        if (g_snapshot_state_db_ptr.size() <= i) {
            g_snapshot_state_db_ptr.push_back(nullptr);
        } else {
            // cudaDeviceReset invalidates pointers, so just set to nullptr
            g_snapshot_state_db_ptr[i] = nullptr;
        }
    }

    // Initialize and store the state DB and account count globally
    // Modify GPUfromJson to use the persistent jump table

    CuEVM::StateDb::GPUfromJsonMultiGPU(g_state_db_ptr, stateJson, g_num_instances_per_device, g_num_accounts,
                                        g_snapshot_state_db_ptr);

    for (int i = 0; i < g_num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CuEVM::get_block_info(stateJson);
    }

    CuEVM::memory_pool::create_memory_pool(g_num_instances_per_device, g_num_accounts, g_num_gpus);

    printf("Process json state GPU done, found %u accounts\n", g_num_accounts);

    call_counter = 0;
    // Initialize the global jump table for coverage tracking
    // Free JSON object
    cJSON_Delete(stateJson);
    return 0;
}

void setup_fuzzing_constants(const char* fuzzing_constants, uint32_t* markerData, uint32_t markerDataLen) {
    printf("Setting up fuzzing constants\n");
    if (fuzzing_constants == nullptr) {
        printf("No fuzzing constants provided\n");
        return;
    }

    cJSON* constantsJson = cJSON_Parse(fuzzing_constants);
    if (constantsJson == NULL) {
        printf("Error parsing fuzzing constants\n");
        return;
    }

    cJSON* address_constants = cJSON_GetObjectItemCaseSensitive(constantsJson, "address");
    cJSON* integer_constants = cJSON_GetObjectItemCaseSensitive(constantsJson, "integer");
    cJSON* sender_constants = cJSON_GetObjectItemCaseSensitive(constantsJson, "sender");

    if (!cJSON_IsArray(address_constants) || !cJSON_IsArray(integer_constants)) {
        printf("Error: address or integer is not an array\n");
        cJSON_Delete(constantsJson);
        return;
    }
    int address_count = cJSON_GetArraySize(address_constants);
    int integer_count = cJSON_GetArraySize(integer_constants);
    int sender_count = cJSON_GetArraySize(sender_constants);

    printf("Address count: %d\n", address_count);
    printf("Integer count: %d\n", integer_count);
    printf("Sender count: %d\n", sender_count);

    // last position in the address contants is for special sender
    address_count += 2;
    // Allocate host arrays
    uint8_t* host_address_constants = new uint8_t[address_count * 32];
    uint8_t* host_integer_constants = new uint8_t[integer_count * 32];
    // evm_word_t* host_address_list = new evm_word_t[address_count];
    evm_word_t* host_sender_list = new evm_word_t[sender_count];
    CuEVM::fuzzing_constants* host_fuzzing_constants = new CuEVM::fuzzing_constants();
    host_fuzzing_constants->address_constants_count =
        address_count - 2;  // normal address count, last 2 are special attackers
    host_fuzzing_constants->integer_constants_count = integer_count;
    host_fuzzing_constants->sender_counts = sender_count;
    for (int i = 0; i < MAX_ARBITRARY_CALL_CHECK; ++i) {
        host_fuzzing_constants->arbitrary_call_check[i] = 0;
    }
    evm_word_t temp_word;

    for (int i = 0; i < sender_count; ++i) {
        cJSON* item = cJSON_GetArrayItem(sender_constants, i);
        if (cJSON_IsString(item) && item->valuestring) {
            temp_word.from_hex(item->valuestring);
            host_sender_list[i] = temp_word;
        }
    }

    // Parse address constants
    for (int i = 0; i < address_count; ++i) {
        cJSON* item = cJSON_GetArrayItem(address_constants, i);
        if (cJSON_IsString(item) && item->valuestring) {
            temp_word.from_hex(item->valuestring);
            uint256_to_bytes(host_address_constants + i * 32, &temp_word, 32);
            // host_address_list[i] = temp_word;
        }
    }
    // add special sender addresses to address constants
    temp_word = host_sender_list[sender_count - 1];
    uint256_to_bytes(host_address_constants + (address_count - 1) * 32, &temp_word, 32);
    temp_word = host_sender_list[sender_count - 2];
    uint256_to_bytes(host_address_constants + (address_count - 2) * 32, &temp_word, 32);

    // Parse integer constants
    for (int i = 0; i < integer_count; ++i) {
        cJSON* item = cJSON_GetArrayItem(integer_constants, i);
        if (cJSON_IsString(item) && item->valuestring) {
            temp_word.from_hex(item->valuestring);
            uint256_to_bytes(host_integer_constants + i * 32, &temp_word, 32);
        }
    }
    // print all in hex
    printf("Sender constants size: %d\n", host_fuzzing_constants->sender_counts);
    for (int i = 0; i < sender_count; ++i) {
        // Assuming evm_word_t has a to_hex() or similar, otherwise print bytes
        char hexstr[65] = {0};
        host_sender_list[i].to_hex(hexstr);  // You may need to implement this if not present
        printf("  [%d]: %s\n", i, hexstr);
    }

    // Print all address constants in hex
    printf("Address constants size: %d\n", host_fuzzing_constants->address_constants_count);
    for (int i = 0; i < address_count; ++i) {
        printf("  [%d]: 0x", i);
        for (int j = 0; j < 32; ++j) {
            printf("%02x", host_address_constants[i * 32 + j]);
        }
        printf("\n");
    }

    // Print all integer constants in hex
    printf("Integer constants size: %d\n", host_fuzzing_constants->integer_constants_count);
    for (int i = 0; i < integer_count; ++i) {
        printf("  [%d]: 0x", i);
        for (int j = 0; j < 32; ++j) {
            printf("%02x", host_integer_constants[i * 32 + j]);
        }
        printf("\n");
    }

    uint8_t* host_return_data = new uint8_t[RETURN_BUFFER_SIZE];
    memset(host_return_data, 0, RETURN_BUFFER_SIZE);
    host_return_data[31] = 0x01;

    uint8_t* d_address_constants;
    uint8_t* d_integer_constants;
    uint8_t* d_return_data_buffer;
    evm_word_t* d_sender_list;
    // evm_word_t* d_address_list;
    CUDA_CHECK(cudaMalloc(&d_address_constants, address_count * sizeof(evm_word_t)));
    CUDA_CHECK(cudaMalloc(&d_integer_constants, integer_count * sizeof(evm_word_t)));
    CUDA_CHECK(cudaMalloc(&d_return_data_buffer, RETURN_BUFFER_SIZE * sizeof(uint8_t)));
    // CUDA_CHECK(cudaMalloc(&d_address_list, address_count * sizeof(evm_word_t)));
    CUDA_CHECK(cudaMalloc(&d_sender_list, sender_count * sizeof(evm_word_t)));
    printf("Copying address constants to device address array size: %lu %lu\n", address_count * sizeof(evm_word_t),
           address_count * 32 * sizeof(uint8_t));
    printf("Copying integer constants to device integer array size: %lu %lu\n", integer_count * sizeof(evm_word_t),
           integer_count * 32 * sizeof(uint8_t));
    CUDA_CHECK(cudaMemcpy(d_address_constants, host_address_constants, address_count * sizeof(evm_word_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_integer_constants, host_integer_constants, integer_count * sizeof(evm_word_t),
                          cudaMemcpyHostToDevice));
    // CUDA_CHECK(
    //     cudaMemcpy(d_address_list, host_address_list, address_count * sizeof(evm_word_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sender_list, host_sender_list, sender_count * sizeof(evm_word_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_return_data_buffer, host_return_data, RETURN_BUFFER_SIZE * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));
    host_fuzzing_constants->address_constants = d_address_constants;
    host_fuzzing_constants->integer_constants = d_integer_constants;
    // host_fuzzing_constants->address_list = d_address_list;
    host_fuzzing_constants->sender_list = d_sender_list;
    host_fuzzing_constants->return_buffer = d_return_data_buffer;
    CuEVM::fuzzing_constants* d_fuzzing_constants;
    CUDA_CHECK(cudaMalloc(&d_fuzzing_constants, sizeof(CuEVM::fuzzing_constants)));
    CUDA_CHECK(cudaMemcpy(d_fuzzing_constants, host_fuzzing_constants, sizeof(CuEVM::fuzzing_constants),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(CuEVM::g_fuzzing_constants, &d_fuzzing_constants, sizeof(CuEVM::fuzzing_constants*)));

    // copy marker data to device
    uint32_t* d_marker_data;
    CUDA_CHECK(cudaMalloc(&d_marker_data, markerDataLen * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_marker_data, markerData, markerDataLen * sizeof(uint32_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpyToSymbol(CuEVM::g_static_marker_data, &d_marker_data, sizeof(uint32_t*)));

    cJSON_Delete(constantsJson);
    delete[] host_address_constants;
    delete[] host_integer_constants;
    // delete[] host_address_list;
    delete[] host_sender_list;
    delete[] host_return_data;
}

void reset_state_db() {
    printf("Resetting state DB\n");
    for (int i = 0; i < g_num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        if (g_state_db_ptr[i] != nullptr && g_snapshot_state_db_ptr[i] != nullptr) {
            // Create temporary host copies to read the device pointers
            // Allocate host memory for the structs themselves
            CuEVM::StateDb* host_state_db =
                new CuEVM::StateDb(1);  // num_states=1 is placeholder, not used for allocation size here
            CuEVM::StateDb* host_snapshot_state_db = new CuEVM::StateDb(1);

            // Copy the StateDb structs (containing device pointers) from device to host
            CUDA_CHECK(cudaMemcpy(host_state_db, g_state_db_ptr[i], sizeof(CuEVM::StateDb), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(host_snapshot_state_db, g_snapshot_state_db_ptr[i], sizeof(CuEVM::StateDb),
                                  cudaMemcpyDeviceToHost));

            // Now host_state_db and host_snapshot_state_db contain the *device* addresses
            // Read necessary metadata from the host copies
            uint32_t num_accounts = host_snapshot_state_db->num_accounts;  // Use snapshot's count
            uint32_t num_states = host_snapshot_state_db->num_states;      // Use snapshot's count
            uint32_t num_contracts_snapshot = host_snapshot_state_db->num_contracts;

            // --- Direct Device-to-Device Copies using pointers read from host structs ---

            // Copy account balances, nonces, storage sizes
            size_t balances_size = (size_t)num_states * num_accounts * sizeof(evm_word_t);
            CUDA_CHECK(cudaMemcpy(host_state_db->account_balances, host_snapshot_state_db->account_balances,
                                  balances_size, cudaMemcpyDeviceToDevice));

            size_t nonces_size = (size_t)num_states * num_accounts * sizeof(uint32_t);
            CUDA_CHECK(cudaMemcpy(host_state_db->account_nonces, host_snapshot_state_db->account_nonces, nonces_size,
                                  cudaMemcpyDeviceToDevice));

            size_t storage_size_size = (size_t)num_states * num_accounts * sizeof(uint32_t);
            CUDA_CHECK(cudaMemcpy(host_state_db->account_storage_size, host_snapshot_state_db->account_storage_size,
                                  storage_size_size, cudaMemcpyDeviceToDevice));

            // Copy prealloc storage pools (using snapshot's num_contracts)
            // Ensure account_prealloc_keys_size is accessible/correct here.
            size_t prealloc_keys_bytes =
                (size_t)account_prealloc_keys_size * num_contracts_snapshot * num_states * sizeof(evm_word_t);
            if (prealloc_keys_bytes > 0) {
                CUDA_CHECK(cudaMemcpy(host_state_db->prealloc_keys_pool, host_snapshot_state_db->prealloc_keys_pool,
                                      prealloc_keys_bytes, cudaMemcpyDeviceToDevice));
            }

            size_t prealloc_values_bytes =
                (size_t)account_prealloc_keys_size * num_contracts_snapshot * num_states * sizeof(CuEVM::ValueStatus);
            if (prealloc_values_bytes > 0) {
                CUDA_CHECK(cudaMemcpy(host_state_db->prealloc_values_pool, host_snapshot_state_db->prealloc_values_pool,
                                      prealloc_values_bytes, cudaMemcpyDeviceToDevice));
            }

            // Copy warm account flags
            size_t warm_flags_size = (size_t)num_states * num_accounts * sizeof(bool);
            CUDA_CHECK(cudaMemcpy(host_state_db->account_is_warm, host_snapshot_state_db->account_is_warm,
                                  warm_flags_size, cudaMemcpyDeviceToDevice));
            // Alternative: Reset warm flags instead of copying snapshot state:
            // CUDA_CHECK(cudaMemset(host_state_db->account_is_warm, 0, warm_flags_size));

            // --- Resetting Dynamic Storage Pointers/Capacities ---
            // IMPORTANT: See previous explanation about external memory management for dynamic pages.
            size_t dynamic_accounts_size = (size_t)num_states * sizeof(CuEVM::DynamicAccount*);
            CUDA_CHECK(cudaMemcpy(host_state_db->dynamic_accounts, host_snapshot_state_db->dynamic_accounts,
                                  dynamic_accounts_size, cudaMemcpyDeviceToDevice));

            size_t dynamic_pages_size = (size_t)num_states * num_accounts * sizeof(CuEVM::StateDbStoragePage*);
            CUDA_CHECK(cudaMemcpy(host_state_db->dynamic_storage_pages, host_snapshot_state_db->dynamic_storage_pages,
                                  dynamic_pages_size, cudaMemcpyDeviceToDevice));

            size_t dynamic_capacity_size = (size_t)num_states * num_accounts * sizeof(uint32_t);
            CUDA_CHECK(cudaMemcpy(host_state_db->dynamic_pool_capacity, host_snapshot_state_db->dynamic_pool_capacity,
                                  dynamic_capacity_size, cudaMemcpyDeviceToDevice));

            // Reset other fields if needed
            // No need to copy host_state_db back to g_state_db_ptr, as we modified the data *pointed to* by
            // g_state_db_ptr.

            // Update the global symbol pointer in device memory (good practice for consistency).
            CUDA_CHECK(cudaMemcpyToSymbol(CuEVM::global_state_db_ptr, &g_state_db_ptr[i], sizeof(CuEVM::StateDb*)));

            // Clean up host allocations
            delete host_state_db;
            delete host_snapshot_state_db;

            // printf("State DB reset complete\n");
        } else {
            printf("State DB reset error: pointers not initialized.\n");
        }
    }
    CuEVM::memory_pool::clear_memory_pool(g_num_gpus);
}

// Creates transaction list on both host and device

std::vector<CuEVM::transaction::TransactionList*> create_transaction_list(
    const uint64_t* blockNumber, const uint64_t* timeStamp, const unsigned char* fromAddr, const unsigned char* toAddr,
    const unsigned char* values, const unsigned char* callData, int callDataLen, const uint32_t* dataOffsets,
    int dataOffsetsLen, const uint32_t* dataSizes, int txCount, const int32_t* markerOffsets,
    const uint32_t* markerData, int markerDataLen, uint32_t start_seed = 0, uint32_t sequence_idx = 0) {
    printf("create transaction list with start seed %u\n", start_seed);
    // Create TransactionList on host

    std::vector<CuEVM::transaction::TransactionList*> d_transaction_list_ptrs;
    uint32_t transaction_per_gpu = txCount / g_num_gpus;
    for (int i = 0; i < g_num_gpus; i++) {
        printf("\n CuEVM: allocation %d txs on GPU %d \n", transaction_per_gpu, i);
        CUDA_CHECK(cudaSetDevice(i));

        // Now we need to allocate GPU memory and transfer data
        CuEVM::transaction::TransactionList* d_transaction_list_ptr;
        CuEVM::transaction::TransactionList* temp_transaction_list = new CuEVM::transaction::TransactionList();
        // memcpy(temp_transaction_list, host_transaction_list, sizeof(CuEVM::transaction::TransactionList));
#ifdef BUILD_GO_LIBRARY
        temp_transaction_list->start_seed = start_seed + i;
        temp_transaction_list->nonce = 0;
        temp_transaction_list->gas_price = 1;
        temp_transaction_list->gas_limit = 1000000;
        temp_transaction_list->type = 0;
        // temp_transaction_list->sequence_id = min(sequence_idx + 1, 15);  // 15 4-bit is the max sequence id
        temp_transaction_list->size = transaction_per_gpu;
        uint256_from_bytes(&temp_transaction_list->to, toAddr, 32);
        printf("host_transaction_list->start_seed: %u\n", temp_transaction_list->start_seed);
#endif
        // Allocate GPU memory for transaction data
        CUDA_CHECK(cudaMalloc(&temp_transaction_list->value, transaction_per_gpu * sizeof(evm_word_t)));
        CUDA_CHECK(cudaMalloc(&temp_transaction_list->call_data_offset, transaction_per_gpu * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&temp_transaction_list->call_data_size, transaction_per_gpu * sizeof(uint32_t)));

#ifdef BUILD_GO_LIBRARY
        // Allocate GPU memory for sender array
        CUDA_CHECK(cudaMalloc(&temp_transaction_list->sender, transaction_per_gpu * sizeof(uint8_t)));

        temp_transaction_list->block_number = device_block_numbers[i];
        temp_transaction_list->time_stamp = device_time_stamps[i];

        // initialize marker data
        CUDA_CHECK(
            cudaMalloc(&temp_transaction_list->marker_offset, transaction_per_gpu / g_skipTxSize * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&temp_transaction_list->marker_data, markerDataLen * sizeof(uint32_t)));
#endif

        // Allocate GPU memory for call data if needed
        if (callDataLen > 0) {
            CUDA_CHECK(cudaMalloc(&temp_transaction_list->call_data, callDataLen * sizeof(uint8_t)));
            CUDA_CHECK(cudaMemcpy(temp_transaction_list->call_data, callData, callDataLen * sizeof(uint8_t),
                                  cudaMemcpyHostToDevice));
        }

        // Copy data from host to GPU

        // CuEVM debug June 27, disable value for now
        // memset zero for blocknumber
        CUDA_CHECK(cudaMemset(temp_transaction_list->value, 0, transaction_per_gpu * sizeof(evm_word_t)));

        CUDA_CHECK(cudaMemcpy(temp_transaction_list->call_data_offset, dataOffsets + i * transaction_per_gpu,
                              transaction_per_gpu * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(temp_transaction_list->call_data_size, dataSizes + i * transaction_per_gpu,
                              transaction_per_gpu * sizeof(uint32_t), cudaMemcpyHostToDevice));

#ifdef BUILD_GO_LIBRARY
        // Copy sender array from host to GPU
        CUDA_CHECK(cudaMemcpy(temp_transaction_list->sender, fromAddr + i * transaction_per_gpu,
                              transaction_per_gpu * sizeof(uint8_t), cudaMemcpyHostToDevice));

        // Block number and timestamp persistent after the first tx
        if (sequence_idx == 0) {
            CUDA_CHECK(cudaMemcpy(temp_transaction_list->block_number, blockNumber + i * transaction_per_gpu,
                                  transaction_per_gpu * sizeof(uint64_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(temp_transaction_list->time_stamp, timeStamp + i * transaction_per_gpu,
                                  transaction_per_gpu * sizeof(uint64_t), cudaMemcpyHostToDevice));
        }
        // Copy marker data from host to GPU
        CUDA_CHECK(cudaMemcpy(temp_transaction_list->marker_offset,
                              markerOffsets + i * transaction_per_gpu / g_skipTxSize,
                              transaction_per_gpu / g_skipTxSize * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(temp_transaction_list->marker_data, markerData, markerDataLen * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));

#endif

        // Allocate memory for the transaction list on GPU and copy the structure
        CUDA_CHECK(cudaMalloc(&d_transaction_list_ptr, sizeof(CuEVM::transaction::TransactionList)));
        CUDA_CHECK(cudaMemcpy(d_transaction_list_ptr, temp_transaction_list,
                              sizeof(CuEVM::transaction::TransactionList), cudaMemcpyHostToDevice));

        // Clean up temporary transaction list
        delete temp_transaction_list;

        // trace and serialized state data
        // Simplified trace data

        CuEVM::simplified_trace_data* d_trace_data;

        CUDA_CHECK(cudaMalloc(&d_trace_data, transaction_per_gpu * sizeof(CuEVM::simplified_trace_data)));
        cudaMemset(d_trace_data, 0, transaction_per_gpu * sizeof(CuEVM::simplified_trace_data));
        cudaMemcpyToSymbol(global_simplified_trace, &d_trace_data, sizeof(CuEVM::simplified_trace_data*));

        d_transaction_list_ptrs.push_back(d_transaction_list_ptr);
    }
    // Free host memory

    return d_transaction_list_ptrs;
}

// Simplified batch transaction processing with single from/to address
SimplifiedGPUResultC* process_batch_transactions(const uint64_t* blockNumber, const uint64_t* timeStamp,
                                                 const unsigned char* fromAddr, const unsigned char* toAddr,
                                                 const unsigned char* values, const unsigned char* callData,
                                                 uint32_t callDataLen, const uint32_t* dataOffsets,
                                                 const uint32_t* dataSizes,
                                                 // June: added marker data
                                                 const int32_t* markerOffsets, const uint32_t* markerData,
                                                 uint32_t markerDataLen, uint32_t txBatchCount, uint32_t sequenceLength,
                                                 uint32_t start_seed) {
    printf("CuEVM Go interface: Processing batch of %d transactions, num tx per gpu %d, sequence length: %d\n",
           txBatchCount, g_num_instances_per_device, sequenceLength);
    std::vector<cudaStream_t> streams;  // TODO move streams to global
    try {
        // Update global num_instances if provided
        if (txBatchCount > 0) {
            if (txBatchCount != g_num_instances_per_device * g_num_gpus) {
                printf("txBatchCount: %u, g_num_instances_per_device: %u\n", txBatchCount, g_num_instances_per_device);
                assert(txBatchCount == g_num_instances_per_device * g_num_gpus);
            }
        }

        uint32_t current_calldata_offset = 0;

        SimplifiedGPUResultC* final_result = new SimplifiedGPUResultC();
        final_result->results = new SimplifiedGPUResultSingleBatchC[sequenceLength];
        final_result->num_results = sequenceLength;
        for (uint32_t sequenceIdx = 0; sequenceIdx < sequenceLength; sequenceIdx++) {
            uint32_t current_idx = sequenceIdx * txBatchCount;

            if (current_idx != 0) {
                current_calldata_offset += dataOffsets[current_idx - 1] + dataSizes[current_idx - 1];
            }
            callDataLen = dataOffsets[current_idx + txBatchCount - 1] + dataSizes[current_idx + txBatchCount - 1];
            // June debug, check this

            printf("current_idx: %u, current_calldata_offset: %u, callDataLen: %u\n", current_idx,
                   current_calldata_offset, callDataLen);

            // Create and transfer transaction list to GPU

            // from addr is uint8 array
            const uint8_t* newFromAddr = fromAddr + current_idx;

            const unsigned char* newValues = values + 32 * current_idx;
            auto d_transaction_list_ptrs = create_transaction_list(
                blockNumber + current_idx, timeStamp + current_idx, newFromAddr, toAddr, newValues,
                callData + current_calldata_offset, callDataLen, dataOffsets + current_idx, txBatchCount,
                dataSizes + current_idx, txBatchCount, markerOffsets + current_idx / g_skipTxSize, markerData,
                markerDataLen, start_seed + sequenceIdx * txBatchCount, sequenceIdx);

            // Initialize memory pool using the globally stored account count
            // Only create memory pool if not reusing state or first call

            // create new coverage tracking variables
            for (int i = 0; i < g_num_gpus; i++) {
                // Set device
                CUDA_CHECK(cudaSetDevice(i));

                // reset trackers
                CUDA_CHECK(cudaMemset(d_branch_infos[i], 0, MAX_NEW_BRANCHES * sizeof(BranchInfoEntry)));
                CUDA_CHECK(cudaMemset(d_storage_infos[i], 0, MAX_NEW_STORAGE * sizeof(StorageInfoEntry)));
                CUDA_CHECK(cudaMemset(d_bug_infos[i], 0, MAX_NEW_BUGS * sizeof(BugInfoEntry)));

                // reset counters
                CUDA_CHECK(cudaMemset(d_gpu_feedback_counts[i], 0, sizeof(GPUFeedbackCount)));
            }

#ifdef EIP_3155
            const size_t BUFFER_SIZE = 100 * 1024 * 1024;  // 100 MB
            std::vector<char*> d_buffers;
            for (int i = 0; i < g_num_gpus; i++) {
                char* d_buffer;
                CUDA_CHECK(cudaSetDevice(i));
                CUDA_CHECK(cudaMalloc(&d_buffer, BUFFER_SIZE));
                d_buffers.push_back(d_buffer);
            }
#endif
            // Configure kernel launch parameters
            uint32_t num_blocks = (g_num_instances_per_device + INSTANCES_PER_BLOCK - 1) / INSTANCES_PER_BLOCK;
            printf("Running %d instances on GPU, blocks: %d, threads per block: %d\n", g_num_instances_per_device,
                   num_blocks, INSTANCES_PER_BLOCK);

            // Create timing variables and arrays for events
            float total_milliseconds = 0;
            cudaEvent_t* start_events = new cudaEvent_t[g_num_gpus];
            cudaEvent_t* stop_events = new cudaEvent_t[g_num_gpus];
            std::vector<cudaStream_t> streams;

            // First: create events and launch all kernels (non-blocking)
            for (int i = 0; i < g_num_gpus; i++) {
                // Set device
                CUDA_CHECK(cudaSetDevice(i));
                // Create CUDA timing events for this device
                cudaEventCreate(&start_events[i]);
                cudaEventCreate(&stop_events[i]);

                // Create CUDA streams for truly asynchronous execution
                cudaStream_t stream;
                CUDA_CHECK(cudaStreamCreate(&stream));

                cudaEventRecord(start_events[i], stream);

                // Execute the GPU kernel with the device pointer for this GPU in its own stream
                CuEVM::kernel_evm_multiple_instances<<<num_blocks, INSTANCES_PER_BLOCK, 0, stream>>>(
                    d_transaction_list_ptrs[i], g_num_instances_per_device
#ifdef EIP_3155
                    ,
                    d_buffers[i], BUFFER_SIZE
#endif
                    ,
                    false);

                // Record the stop event but don't synchronize yet (non-blocking)
                cudaEventRecord(stop_events[i], stream);

                // Store stream for later cleanup
                streams.push_back(stream);
            }

            // Second: wait for all GPUs to finish and collect timings
            for (int i = 0; i < g_num_gpus; i++) {
                // Set device
                CUDA_CHECK(cudaSetDevice(i));

                // Wait for this GPU to finish
                cudaEventSynchronize(stop_events[i]);
                cudaStreamSynchronize(streams[i]);

                // Check for errors on this GPU
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) printf("CUDA error on GPU %d: %s\n", i, cudaGetErrorString(err));

                // Get timing for this GPU
                float milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start_events[i], stop_events[i]);
                total_milliseconds = std::max(total_milliseconds, milliseconds);

                // Clean up events for this GPU
                cudaEventDestroy(start_events[i]);
                cudaEventDestroy(stop_events[i]);
                cudaStreamDestroy(streams[i]);
            }

#ifdef EIP_3155
            // After kernel execution, copy the buffer back to the host
            char* h_buffer = new char[BUFFER_SIZE];
            for (int i = 0; i < g_num_gpus; i++) {
                CUDA_CHECK(cudaSetDevice(i));
                cudaMemcpy(h_buffer, d_buffers[i], BUFFER_SIZE, cudaMemcpyDeviceToHost);
                CuEVM::utils::print_tracer_data(h_buffer);
                cudaFree(d_buffers[i]);
            }

            // Clean up
            delete[] h_buffer;

#endif
            printf("\n\nGPU kernel execution time: %f milliseconds (max across all GPUs)\n", total_milliseconds);

            // Clean up event arrays
            delete[] start_events;
            delete[] stop_events;

            // ... rest of the function ...

            // Clean up transaction lists
            // cleanup_transaction_list(d_transaction_list_ptr, callDataLen);
            get_gpu_execution_results_optimized(&final_result->results[sequenceIdx], callData + current_calldata_offset,
                                                dataOffsets + current_idx, dataSizes + current_idx);
            for (int i = 0; i < g_num_gpus; i++) {
                CUDA_CHECK(cudaSetDevice(i));
                CuEVM::freeTransactionList(d_transaction_list_ptrs[i]);
                CuEVM::freeTraceData(false);
            }
            // Increment call counter
            call_counter++;
            printf("Call number: %d\n", call_counter);
        }
        // Debug
        return final_result;  // Success
    } catch (const std::exception& e) {
        printf("Error in process_batch_transactions: %s\n", e.what());
        return nullptr;  // Error
    } catch (...) {
        printf("Unknown error in process_batch_transactions\n");
        return nullptr;  // Unknown error
    }
}
#define DEBUG
// Minimalized version of get_gpu_execution_results
void get_gpu_execution_results_optimized(SimplifiedGPUResultSingleBatchC* result, const uint8_t* callData,
                                         const uint32_t* dataOffsets, const uint32_t* dataSizes) {
    // Arrays to store counts from each GPU
    std::vector<uint32_t> branch_counts(g_num_gpus);
    std::vector<uint32_t> bug_counts(g_num_gpus);
    std::vector<uint32_t> storage_counts(g_num_gpus);

    uint32_t total_num_new_branch = 0;
    uint32_t total_num_new_bug = 0;
    uint32_t total_num_new_storage = 0;
    // First pass: count total new coverage and bugs across all GPUs
    GPUFeedbackCount host_counter;
    for (int i = 0; i < g_num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));

        // Get the count of new branches
        // uint32_t* host_counter_ptr = nullptr;
        GPUFeedbackCount* host_counter_ptr = nullptr;
        CUDA_CHECK(cudaMemcpyFromSymbol(&host_counter_ptr, g_gpu_feedback_count, sizeof(GPUFeedbackCount*)));

        CUDA_CHECK(cudaMemcpy(&host_counter, host_counter_ptr, sizeof(GPUFeedbackCount), cudaMemcpyDeviceToHost));
        // using min in corner case where atomic add over the max value

        branch_counts[i] = std::min(host_counter.new_branch_count, MAX_NEW_BRANCHES);
        bug_counts[i] = std::min(host_counter.new_bug_count, MAX_NEW_BUGS);
        storage_counts[i] = std::min(host_counter.new_storage_count, MAX_NEW_STORAGE);
        total_num_new_branch += branch_counts[i];
        total_num_new_bug += bug_counts[i];
        total_num_new_storage += storage_counts[i];
    }

    printf("Total new coverage branches: %u, total new bugs: %u, total new storage: %u\n", total_num_new_branch,
           total_num_new_bug, total_num_new_storage);

    // Allocate memory for result arrays once we know the total sizes
    result->num_new_branch = total_num_new_branch;
    result->num_new_bug = total_num_new_bug;
    result->num_new_storage = total_num_new_storage;

    if (total_num_new_branch > 0) {
        result->new_branch_info = new BranchInfoEntry[total_num_new_branch];
    } else {
        result->new_branch_info = nullptr;
    }

    if (total_num_new_bug > 0) {
        result->new_bug_info = new BugInfoEntry[total_num_new_bug];
    } else {
        result->new_bug_info = nullptr;
    }

    if (total_num_new_storage > 0) {
        result->new_storage_info = new StorageInfoEntry[total_num_new_storage];
    } else {
        result->new_storage_info = nullptr;
    }

    // Second pass: copy the actual data
    uint32_t coverage_offset = 0;
    uint32_t bug_offset = 0;
    uint32_t storage_offset = 0;

    for (int i = 0; i < g_num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));

        // Copy new branches data - use stored count from first pass
        if (branch_counts[i] > 0) {
            BranchInfoEntry* d_new_branch_info = nullptr;
            CUDA_CHECK(cudaMemcpyFromSymbol(&d_new_branch_info, g_new_branch_info, sizeof(BranchInfoEntry*)));
            CUDA_CHECK(cudaMemcpy(&result->new_branch_info[coverage_offset], d_new_branch_info,
                                  branch_counts[i] * sizeof(BranchInfoEntry), cudaMemcpyDeviceToHost));

            // adjust the idx from the offset
            if (coverage_offset != 0) {
                for (uint32_t j = 0; j < branch_counts[i]; j++) {
                    result->new_branch_info[coverage_offset + j].branch_thread_idx += i * g_num_instances_per_device;
                }
            }
#ifdef DEBUG
            // Debug output
            for (uint32_t j = 0; j < branch_counts[i]; j++) {
                printf("GPU %d: new_branches[%d]: %d\n", i, j,
                       result->new_branch_info[coverage_offset + j].branch_thread_idx);
            }
#endif
            coverage_offset += branch_counts[i];
        }

        // Copy new bugs data - use stored count from first pass
        if (bug_counts[i] > 0) {
            BugInfoEntry* d_new_bug_info = nullptr;
            CUDA_CHECK(cudaMemcpyFromSymbol(&d_new_bug_info, g_new_bug_info, sizeof(BugInfoEntry*)));

            CUDA_CHECK(cudaMemcpy(&result->new_bug_info[bug_offset], d_new_bug_info,
                                  bug_counts[i] * sizeof(BugInfoEntry), cudaMemcpyDeviceToHost));

            // adjust the idx from the offset
            if (bug_offset != 0) {
                for (uint32_t j = 0; j < bug_counts[i]; j++) {
                    result->new_bug_info[bug_offset + j].bug_thread_idx += i * g_num_instances_per_device;
                }
            }

// Debug output
#ifdef DEBUG
            for (uint32_t j = 0; j < bug_counts[i]; j++) {
                printf("GPU %d: new_bugs[%d]: idx=%d, id=%d\n", i, j,
                       result->new_bug_info[bug_offset + j].bug_thread_idx,
                       result->new_bug_info[bug_offset + j].bug_id);
            }
#endif
            bug_offset += bug_counts[i];
        }

        // Copy new storage data - use stored count from first pass
        if (storage_counts[i] > 0) {
            StorageInfoEntry* d_new_storage_info = nullptr;
            CUDA_CHECK(cudaMemcpyFromSymbol(&d_new_storage_info, g_new_storage_info, sizeof(StorageInfoEntry*)));

            CUDA_CHECK(cudaMemcpy(&result->new_storage_info[storage_offset], d_new_storage_info,
                                  storage_counts[i] * sizeof(StorageInfoEntry), cudaMemcpyDeviceToHost));

            // adjust the idx from the offset
            if (storage_offset != 0) {
                for (uint32_t j = 0; j < storage_counts[i]; j++) {
                    result->new_storage_info[storage_offset + j].storage_thread_idx += i * g_num_instances_per_device;
                }
            }

// Debug output
#ifdef DEBUG
            for (uint32_t j = 0; j < storage_counts[i]; j++) {
                printf("GPU %d: new_storage[%d]: idx=%d, id=%d\n", i, j,
                       result->new_storage_info[storage_offset + j].storage_thread_idx,
                       result->new_storage_info[storage_offset + j].storage_id);
            }
#endif
            storage_offset += storage_counts[i];
        }
    }
}

// For debugging, tracking unique marker patterns
std::unordered_map<std::string, std::vector<uint32_t>> unique_marker_patterns;

void free_simplified_gpu_result(SimplifiedGPUResultC* result) {
    // TODO:
}
void free_gpu_execution_results(GPUExecutionResultC* result) {
    if (result == nullptr || result->allocations_valid == 0) return;

    // Free return data
    if (result->return_data != nullptr) {
        for (uint32_t i = 0; i < result->num_return_data; i++) {
            if (result->return_data[i].data != nullptr) {
                free(result->return_data[i].data);
            }
        }
        free(result->return_data);
    }

    // Free coverage data
    if (result->coverage != nullptr) {
        for (uint32_t i = 0; i < result->num_coverage; i++) {
            if (result->coverage[i].addresses != nullptr) {
                for (uint32_t j = 0; j < result->coverage[i].num_addresses; j++) {
                    if (result->coverage[i].addresses[j] != nullptr) {
                        free(result->coverage[i].addresses[j]);
                    }
                }
                free(result->coverage[i].addresses);
            }

            if (result->coverage[i].branch_coverage != nullptr) {
                for (uint32_t j = 0; j < result->coverage[i].num_addresses; j++) {
                    if (result->coverage[i].branch_coverage[j] != nullptr) {
                        free(result->coverage[i].branch_coverage[j]);
                    }
                }
                free(result->coverage[i].branch_coverage);
            }
            free(result->coverage[i].branch_coverage_lengths);
        }
        free(result->coverage);
    }

    // Free error codes data
    if (result->error_codes != nullptr) {
        free(result->error_codes);
    }

    // Mark as freed and free the result itself
    result->allocations_valid = 0;
    free(result);
}

#ifdef __cplusplus
}
#endif
