#include <CuEVM/utils/library_utils.h>

#include <CuEVM/utils/error_codes.cuh>
#include <cassert>
#include <sstream>

#define CHECK_AND_RETURN_ON_ERROR(expr)                                                                            \
    do {                                                                                                           \
        auto status = (expr);                                                                                      \
        if (status != 0) {                                                                                         \
            std::ostringstream oss;                                                                                \
            oss << "Error in expression: " << #expr << " (status: 0x" << std::hex << status << std::dec << ") at " \
                << __FILE__ << ":" << __LINE__;                                                                    \
            throw std::runtime_error(oss.str());                                                                   \
        }                                                                                                          \
    } while (0)

namespace CuEVM {

__device__ uint32_t* g_coverage_bitmap = nullptr;
__device__ uint32_t* g_new_coverage_bitmap = nullptr;
__device__ uint32_t* g_new_coverage_count = nullptr;
__device__ uint32_t* g_new_coverage_idx = nullptr;
__device__ uint32_t* g_new_bug_pc = nullptr;
__device__ uint32_t* g_new_bug_count = nullptr;
__device__ uint32_t* g_new_bug_idx = nullptr;

__device__ fuzzing_constants* g_fuzzing_constants = nullptr;

__host__ __device__ void fuzzing_constants::print() {
    printf("address_constants_count: %d\n", address_constants_count);
    printf("integer_constants_count: %d\n", integer_constants_count);
    for (uint32_t i = 0; i < address_constants_count; i++) {
        for (uint32_t j = 0; j < 32; j++) {
            printf("%x", address_constants[i * 32 + j]);
        }
        printf("\n");
    }
    for (uint32_t i = 0; i < integer_constants_count; i++) {
        for (uint32_t j = 0; j < 32; j++) {
            printf("%x", integer_constants[i * 32 + j]);
        }
        printf("\n");
    }
}
__host__ void serialized_worldstate_data::print() {
    printf("\nPrinting serialized worldstate data\n");
    printf("no_accounts: %d\n", no_accounts);
    printf("no_storage_elements: %d\n", no_storage_elements);
    for (uint32_t idx = 0; idx < no_accounts; idx++) {
        printf("address: %s\n", addresses[idx]);
        printf("balance: %s\n", balance[idx]);
        printf("nonce: %d\n", nonce[idx]);
    }
    for (uint32_t idx = 0; idx < no_storage_elements; idx++) {
        printf("storage_key: %s\n", storage_keys[idx]);
        printf("storage_value: %s\n", storage_values[idx]);
        printf("storage_index: %d\n", storage_indexes[idx]);
    }
}

__device__ void simplified_trace_data::update_coverage_bitmap(uint32_t pc_src, uint32_t pc_dst, bool is_bug) {
    // printf("thread %d update_coverage_bitmap pc_src %u pc_dst %u\n", threadIdx.x, pc_src, pc_dst);
    uint32_t branch_id = (static_cast<uint32_t>(pc_src) << 16) | pc_dst;
    // printf("thread %d branch_id %x\n", threadIdx.x, branch_id);
    uint32_t h1 = 2166136261U;
    h1 = (h1 ^ (branch_id & 0xFF)) * 16777619U;
    h1 = (h1 ^ ((branch_id >> 8) & 0xFF)) * 16777619U;
    h1 = (h1 ^ ((branch_id >> 16) & 0xFF)) * 16777619U;
    h1 = (h1 ^ (branch_id >> 24)) * 16777619U;

    uint32_t h2 = (branch_id * 2654435761U) & 0xFFFFFFFF;
    bool is_new = false;

    // Double hashing for k=2
    // TODO: double check
    for (int i = 0; i < 2; ++i) {
        uint32_t hash_i = (h1 + i * h2) & 0x3FFFF;  // 2^18 - 1
        uint32_t index = hash_i >> 5;               // Divide by 32
        uint32_t bit_pos = hash_i & 31;             // Modulo 32
        unsigned int mask = 1U << bit_pos;

        // Set bit and check if it was newly set
        unsigned int old = atomicOr(&g_coverage_bitmap[index], mask);
        if (!(old & mask)) {
            is_new = true;
        }
    }
    // Record new branch
    if (is_new) {
        int bitmap_idx = INSTANCE_GLOBAL_IDX / 32;
        int bit_pos = INSTANCE_GLOBAL_IDX % 32;
        atomicOr(&g_new_coverage_bitmap[bitmap_idx], 1 << bit_pos);
        // printf("thread %d bitmap_idx %d bit_pos %d  new branch or bug \n", INSTANCE_GLOBAL_IDX, bitmap_idx, bit_pos);
        if (is_bug) {
            // printf("thread %d is bug, add to bug list \n", threadIdx.x);
            int idx = atomicAdd(g_new_bug_count, 1);
            if (idx < CuEVM::MAX_NEW_BUGS) {
                g_new_bug_idx[idx] = INSTANCE_GLOBAL_IDX;
                g_new_bug_pc[idx] = pc_src;
            }
        }
    }
}
__device__ void finalize_coverage_bitmap() {
    int bitmap_idx = INSTANCE_GLOBAL_IDX / 32;
    int bit_pos = INSTANCE_GLOBAL_IDX % 32;
    uint32_t mask = 1U << bit_pos;

    // Check if this thread's bit is set in the new coverage bitmap
    if (g_new_coverage_bitmap[bitmap_idx] & mask) {
        // Atomically increment the counter and get the previous value
        int idx = atomicAdd(g_new_coverage_count, 1);

        // If we haven't exceeded the maximum number of new branches to track
        if (idx < CuEVM::MAX_NEW_BRANCHES) {
            // Record this thread's global index in the coverage index array
            g_new_coverage_idx[idx] = INSTANCE_GLOBAL_IDX;
        }

        // printf("g_new_coverage_bitmap[idx] %d\n", g_new_coverage_bitmap[idx]);
    }

    // printf("g_new_coverage_count %d\n", g_new_coverage_count[0]);
}
__device__ void simplified_trace_data::start_operation(const uint32_t pc, const uint8_t op,
                                                       const CuEVM::evm_stack_t& stack_ptr) {
    if (no_events >= MAX_TRACE_EVENTS) return;
    events[no_events].pc = pc;
    events[no_events].op = op;
    if (op != OP_INVALID && op != OP_SELFDESTRUCT) {
        // printf("add new operation, src data %d \n", THREADIDX);
        // printf("stack size %d\n", stack_ptr.size());

        events[no_events].operand_1 = *stack_ptr.get_address_at_index(1);
        events[no_events].operand_2 = *stack_ptr.get_address_at_index(2);
    }
}

__device__ void simplified_trace_data::record_branch(uint32_t pc_src, uint32_t pc_dst, uint32_t pc_missed) {
    if (no_branches >= MAX_BRANCHES_TRACING) no_branches = 0;
    branches[no_branches].pc_src = pc_src;
    branches[no_branches].pc_dst = pc_dst;
    branches[no_branches].pc_missed = pc_missed;
    branches[no_branches].distance = last_distance;
    // printf("record branch pc_src %u pc_dst %u distance %s\n", pc_src, pc_dst,
    // branches[no_branches].distance.to_hex());
#ifdef BUILD_GO_LIBRARY
    update_coverage_bitmap(pc_src, pc_dst);
#endif
    no_branches++;
}

__device__ void simplified_trace_data::record_distance(uint8_t op, const CuEVM::evm_stack_t& stack_ptr) {
    evm_word_t distance, op1, op2;
    uint32_t stack_size = stack_ptr.size();

    op1 = *stack_ptr.get_address_at_index(1);
    op2 = *stack_ptr.get_address_at_index(2);

    if (uint256_cmp(&op1, &op2) >= 1)
        uint256_sub(&distance, &op1, &op2);
    else
        uint256_sub(&distance, &op2, &op1);

    if (op != OP_EQ) uint256_add_word(&distance, &distance, 1);

    last_distance = distance;
}

__device__ void simplified_trace_data::record_operation(const uint32_t pc, const uint8_t op) {
    // for simple trace, no need stack content
    if (no_events >= MAX_TRACE_EVENTS) return;
    events[no_events].pc = pc;
    events[no_events].op = op;
    no_events++;
}

__device__ void simplified_trace_data::finish_operation(const CuEVM::evm_stack_t& stack_ptr, uint32_t error_code) {
    if (no_events >= MAX_TRACE_EVENTS) return;
    if (events[no_events].op < OP_REVERT && events[no_events].op != OP_SSTORE)
        events[no_events].res = *stack_ptr.get_address_at_index(1);
    no_events++;
}
__device__ int simplified_trace_data::start_call(uint32_t pc, evm_call_context_t* call_context_ptr) {
    assert(call_context_ptr != nullptr);
    if (no_calls >= MAX_CALLS_TRACING) return;
    // add address and increment current_address_idx
    // addresses[current_address_idx] = cached_call_state->addresses[cached_call_state->current_address_idx];
    // printf("start call simplified trace data pc %d no calls %d\n", pc, no_calls);
    calls[no_calls].sender = call_context_ptr->from;
    calls[no_calls].receiver = call_context_ptr->to;
    calls[no_calls].pc = pc;
    calls[no_calls].op = call_context_ptr->call_type;
    calls[no_calls].value = call_context_ptr->value;
    calls[no_calls].error_code = RESERVED_ERROR_CODE;
    calls[no_calls].last_pc = 0;
    no_calls++;
    if (no_calls > MAX_RECURSION) {
        // printf("\nThread %d: No calls %d\n", INSTANCE_GLOBAL_IDX, no_calls);
        uint8_t loop_found = 0;
        // loop back 10 calls and check if we found loops
        for (int i = no_calls - 2; i >= no_calls - MAX_RECURSION; i--) {
            if (calls[i].receiver == calls[no_calls - 1].receiver) {
                loop_found++;
            }
        }
        // printf("Thread %d: Loop found %d\n", INSTANCE_GLOBAL_IDX, loop_found);

        if (loop_found >= MAX_RECURSION / 2) {
            // printf("\nThread %d: Reentrancy detected, loop_found %d\n", INSTANCE_GLOBAL_IDX, loop_found);
            return ERROR_REENTRANCY;
        }
    }
#ifdef BUILD_GO_LIBRARY

    if (no_branches >= MAX_BRANCHES_TRACING) no_branches = 0;
    // add extra markers for entering call
    branches[no_branches].pc_src = START_CALL_BRANCH_MARKER;
    branches[no_branches].pc_dst = 0;
    branches[no_branches].pc_missed = 0;
    branches[no_branches].distance = 0;
    no_branches++;
#endif
    return ERROR_SUCCESS;
}
__device__ void simplified_trace_data::finish_call(uint8_t error_code, uint32_t last_pc) {
    if (no_calls > MAX_CALLS_TRACING) return;
    int i;
    for (i = no_calls - 1; i >= 0; i--) {
        // Check if this call is marked as unfinished (using the sentinel value)
        if (calls[i].error_code == RESERVED_ERROR_CODE) {
            // Found the correct call frame, update its results
            calls[i].last_pc = last_pc;
            if (error_code == ERROR_RETURN || error_code == ERROR_SUCCESS)
                calls[i].error_code = ERROR_SUCCESS;
            else
                calls[i].error_code = error_code;
            // Stop searching, we've updated the corresponding call
            break;
        }
    }
#ifdef BUILD_GO_LIBRARY
    if (no_branches >= MAX_BRANCHES_TRACING) no_branches = 0;
    // add extra markers for exiting call
    branches[no_branches].pc_src = END_CALL_BRANCH_MARKER;
    branches[no_branches].pc_dst = 0;
    branches[no_branches].pc_missed = 0;
    branches[no_branches].distance = 0;
    no_branches++;
#endif

#ifdef BUILD_GO_LIBRARY
    if (error_code == ERROR_INVALID_OPCODE) {
        update_coverage_bitmap(last_pc, 0, true);
    }
#endif
}
__host__ __device__ void simplified_trace_data::print() {
    printf("no_events %u\n", no_events);
    printf("no_calls %u\n", no_calls);
    printf("events\n");
    for (uint32_t i = 0; i < no_events; i++) {
        printf("pc %u op %u operand_1 %s operand_2 %s res %s\n", events[i].pc, events[i].op,
               events[i].operand_1.to_hex(), events[i].operand_2.to_hex(), events[i].res.to_hex());
    }
    printf("calls\n");
    for (uint32_t i = 0; i < no_calls; i++) {
        printf("pc %u op %u sender %s receiver %s value %s error_code %u\n", calls[i].pc, calls[i].op,
               calls[i].sender.to_hex(), calls[i].receiver.to_hex(), calls[i].value.to_hex(), calls[i].error_code);
    }
    printf("branches\n");
    for (uint32_t i = 0; i < no_branches; i++) {
        printf("pc_src %u pc_dst %u distance %s\n", branches[i].pc_src, branches[i].pc_dst,
               branches[i].distance.to_hex());
    }
}
__device__ serialized_worldstate_data* global_serialized_worldstate;
__device__ simplified_trace_data* global_simplified_trace;

void freeTransactionList(TransactionList* d_transaction_list_ptr) {
    if (d_transaction_list_ptr == nullptr) {
        return;
    }

    // Create a temporary TransactionList to store device pointers
    TransactionList temp_list;
    CUDA_CHECK(cudaMemcpy(&temp_list, d_transaction_list_ptr, sizeof(TransactionList), cudaMemcpyDeviceToHost));

    // Free all device memory allocations
    if (temp_list.value != nullptr) {
        CUDA_CHECK(cudaFree(temp_list.value));
    }
    if (temp_list.gas_limit != nullptr) {
        CUDA_CHECK(cudaFree(temp_list.gas_limit));
    }
    if (temp_list.call_data != nullptr) {
        CUDA_CHECK(cudaFree(temp_list.call_data));
    }
    if (temp_list.call_data_offset != nullptr) {
        CUDA_CHECK(cudaFree(temp_list.call_data_offset));
    }
    if (temp_list.call_data_size != nullptr) {
        CUDA_CHECK(cudaFree(temp_list.call_data_size));
    }
#ifdef BUILD_GO_LIBRARY
    // Free sender array when using GO library
    if (temp_list.sender != nullptr) {
        CUDA_CHECK(cudaFree(temp_list.sender));
    }
#endif

    // Finally free the TransactionList itself
    CUDA_CHECK(cudaFree(d_transaction_list_ptr));
}

void freeTraceData(bool copy_state_data) {
    // Free simplified trace data
    CuEVM::simplified_trace_data* d_trace_data;
    CUDA_CHECK(cudaMemcpyFromSymbol(&d_trace_data, global_simplified_trace, sizeof(CuEVM::simplified_trace_data*)));
    if (d_trace_data != nullptr) {
        CUDA_CHECK(cudaFree(d_trace_data));
    }

    // Free serialized worldstate data if it was allocated
    if (copy_state_data) {
        CuEVM::serialized_worldstate_data* d_serialized_worldstate_data;
        CUDA_CHECK(cudaMemcpyFromSymbol(&d_serialized_worldstate_data, global_serialized_worldstate,
                                        sizeof(CuEVM::serialized_worldstate_data*)));
        if (d_serialized_worldstate_data != nullptr) {
            CUDA_CHECK(cudaFree(d_serialized_worldstate_data));
        }
    }
}

#ifdef BUILD_GO_LIBRARY
// LCG parameters (commonly used for a 32-bit generator)

// seed = (a * seed + c) % m;
// chances out of 100

#define CHANCE_TO_TAKE_INTEGER_FROM_CONSTANTS 50  // percent
#define CHANCE_TO_CREATE_NEW_INTEGER 2            // one in 2
#define CHANCE_TO_CREATE_NEW_ADDRESS 5            // percent
#define A_LCG 1664525
#define C_LCG 1013904223
#define M_LCG 0xFFFFFFFF  // 2^32 - 1

__device__ unsigned int mutate_byte_array(uint8_t* data, uint32_t element_length, uint32_t byte_length,
                                          unsigned int seed, bool create_new) {
    seed = (A_LCG * seed + C_LCG) % M_LCG;
    uint8_t mutated_byte = seed % (byte_length + 1);
    if (create_new) {
        seed = (A_LCG * seed + C_LCG) % M_LCG;
        uint32_t random_chance = seed % 100;
        if (random_chance <= CHANCE_TO_TAKE_INTEGER_FROM_CONSTANTS) {
            seed = (A_LCG * seed + C_LCG) % M_LCG;
            // take from constants
            uint32_t random_index = seed % g_fuzzing_constants->integer_constants_count;
            for (int i = element_length - byte_length; i < element_length; i++) {
                data[i] = g_fuzzing_constants->integer_constants[random_index * 32 + i];
            }
        } else {
            for (int i = 0; i < element_length - mutated_byte; i++) {
                data[i] = 0;
            }
        }
    }
    uint8_t* start_offset = data + element_length - mutated_byte;
    for (int mutate_byte_index = 0; mutate_byte_index < mutated_byte; mutate_byte_index++) {
        seed = (A_LCG * seed + C_LCG) % M_LCG;
        uint8_t random_byte = seed & 0xFF;
        start_offset[mutate_byte_index] = random_byte;
    }
    return seed;
}

__device__ void mutate_transaction_data(CuEVM::transaction::TransactionList* transaction_list_ptr) {
    unsigned int seed = INSTANCE_GLOBAL_IDX + transaction_list_ptr->start_seed;

    uint32_t marker_idx = INSTANCE_GLOBAL_IDX / CUEVM_MUTATE_GROUP_SIZE;
    uint32_t marker_size = transaction_list_ptr->marker_size[marker_idx];
    if (marker_size == 0) return;
    uint32_t current_marker_offset = transaction_list_ptr->marker_offset[marker_idx];
    uint8_t* call_data = &transaction_list_ptr->call_data[transaction_list_ptr->call_data_offset[INSTANCE_GLOBAL_IDX]];
    for (int j = 0; j < marker_size; j++) {
        uint32_t element_offset = transaction_list_ptr->marker_data[current_marker_offset + j * 3];
        uint32_t element_type = transaction_list_ptr->marker_data[current_marker_offset + j * 3 + 1];
        uint32_t element_length = transaction_list_ptr->marker_data[current_marker_offset + j * 3 + 2];

        if (element_type != 0 && element_type != 1) {
            uint32_t byte_length = element_type / 8;
            // randomize the marker data
            seed = (A_LCG * seed + C_LCG) % M_LCG;
            bool create_new = (seed % CHANCE_TO_CREATE_NEW_INTEGER) == 0;
            seed = mutate_byte_array(call_data + element_offset, element_length, byte_length, seed, create_new);
        } else if (element_type == 1) {  // address
            // randomize the marker data
            seed = (A_LCG * seed + C_LCG) % M_LCG;
            uint32_t random_chance = seed % 100;

            if (random_chance <= CHANCE_TO_CREATE_NEW_ADDRESS) {
                // printf("thread %d create new address\n", INSTANCE_GLOBAL_IDX);
                seed = mutate_byte_array(call_data + element_offset, 32, 20, seed, true);
            } else {
                seed = (A_LCG * seed + C_LCG) % M_LCG;
                // select from the constants
                uint32_t address_constants_count = g_fuzzing_constants->address_constants_count;
                uint32_t random_index = seed % address_constants_count;
                // printf("thread %d seed %d address_constants_count %d random_address_index %d\n", INSTANCE_GLOBAL_IDX,
                //        seed, address_constants_count, random_index);
                for (int i = 0; i < 20; i++) {
                    call_data[element_offset + 12 + i] =
                        g_fuzzing_constants->address_constants[random_index * 32 + 12 + i];
                }
            }
        }
    }

    /*
        uint32_t instance = INSTANCE_GLOBAL_IDX;
        if (instance == 0) {
            printf("marker data instance %d marker_idx %d marker_size %d \n", instance, instance/8,
       transaction_list_ptr->marker_size[instance/8]);

            uint32_t current_marker_offset = transaction_list_ptr->marker_offset[instance/8];
            for (int j = 0; j < transaction_list_ptr->marker_size[instance/8]; j++) {
                printf("marker data %d: offset %d type %d length %d \n", j ,
                       transaction_list_ptr->marker_data[current_marker_offset + j * 3],
                       transaction_list_ptr->marker_data[current_marker_offset + j * 3 + 1],
                       transaction_list_ptr->marker_data[current_marker_offset + j * 3 + 2]);
            }
        }
        __syncthreads();
        if (instance == 1) {
            printf("marker data instance %d marker_idx %d marker_size %d \n", instance, instance/8,
       transaction_list_ptr->marker_size[instance/8]);

            uint32_t current_marker_offset = transaction_list_ptr->marker_offset[instance/8];
            for (int j = 0; j < transaction_list_ptr->marker_size[instance/8]; j++) {
                printf("marker data %d: offset %d type %d length %d \n", j,
                       transaction_list_ptr->marker_data[current_marker_offset + j * 3],
                       transaction_list_ptr->marker_data[current_marker_offset + j * 3 + 1],
                       transaction_list_ptr->marker_data[current_marker_offset + j * 3 + 2]);
            }
        }
        __syncthreads();
        if (instance == 10) {
            printf("marker data instance %d marker_idx %d marker_size %d \n", instance, instance/8,
       transaction_list_ptr->marker_size[instance/8]);

            uint32_t current_marker_offset = transaction_list_ptr->marker_offset[instance/8];
            for (int j = 0; j < transaction_list_ptr->marker_size[instance/8]; j++) {
                printf("marker data %d: offset %d type %d length %d \n", j,
                       transaction_list_ptr->marker_data[current_marker_offset + j * 3],
                       transaction_list_ptr->marker_data[current_marker_offset + j * 3 + 1],
                       transaction_list_ptr->marker_data[current_marker_offset + j * 3 + 2]);
            }
        }

        __syncthreads();
        if (instance == 16) {
            printf("marker data instance %d marker_idx %d marker_size %d \n", instance, instance/8,
       transaction_list_ptr->marker_size[instance/8]);

            uint32_t current_marker_offset = transaction_list_ptr->marker_offset[instance/8];
            for (int j = 0; j < transaction_list_ptr->marker_size[instance/8]; j++) {
                printf("marker data %d: offset %d type %d length %d \n", j,
                       transaction_list_ptr->marker_data[current_marker_offset + j * 3],
                       transaction_list_ptr->marker_data[current_marker_offset + j * 3 + 1],
                       transaction_list_ptr->marker_data[current_marker_offset + j * 3 + 2]);
            }
        }

           __syncthreads();
        if (instance == 31) {
            printf("marker data instance %d marker_idx %d marker_size %d \n", instance, instance/8,
       transaction_list_ptr->marker_size[instance/8]);

            uint32_t current_marker_offset = transaction_list_ptr->marker_offset[instance/8];
            for (int j = 0; j < transaction_list_ptr->marker_size[instance/8]; j++) {
                printf("marker data %d: offset %d type %d length %d \n", j,
                       transaction_list_ptr->marker_data[current_marker_offset + j * 3],
                       transaction_list_ptr->marker_data[current_marker_offset + j * 3 + 1],
                       transaction_list_ptr->marker_data[current_marker_offset + j * 3 + 2]);
            }
        }
    */
}
#endif

__device__ void serialize_state_data(CuEVM::serialized_worldstate_data* data) {
    // Use the global state database pointer to access the account data
    StateDb* state = global_state_db_ptr;
    if (state == nullptr) {
        // Ideally, handle the error appropriately (or abort) if the state is missing.
        return;
    }

    // Set the number of accounts in the serialized state.
    data->no_accounts = state->num_accounts;
    // Start with no storage elements serialized.
    data->no_storage_elements = 0;

    // uint32_t new_offset =
    // (contract_index[address_index] * account_prealloc_keys_size + storage_size) * num_states + INSTANCE_GLOBAL_IDX;
    // uint32_t instance_idx = address_index * num_states + INSTANCE_GLOBAL_IDX;
    // // printf("get_value_status address_index %d, instance_idx %d instance %d\n", address_index, instance_idx,
    // //        INSTANCE_GLOBAL_IDX);
    // uint32_t contract_idx = contract_index[address_index];
    // uint32_t storage_size = account_storage_size[instance_idx];

    // Iterate through each account.
    // We assume that the account data (address, balance, nonce) is stored in parallel arrays,
    // and for simplicity we pick the first state (index 0) as the canonical view.
    for (uint32_t acct = 0; acct < state->num_accounts; acct++) {
        uint32_t instance_idx = acct * state->num_states + INSTANCE_GLOBAL_IDX;
        int16_t cidx = state->contract_index[acct];
        uint32_t storage_size = state->account_storage_size[instance_idx];

        // Convert the account's address to a hex string.
        data->addresses[acct] = state->address_list[acct];
        // Convert the account's balance (using the snapshot at state index 0) to hex.
        data->balance[acct] = state->account_balances[instance_idx];
        // Copy the account's nonce (again using state index 0).
        data->nonce[acct] = state->account_nonces[instance_idx];

        // if (INSTANCE_GLOBAL_IDX == 0) {
        //     printf("address: \n");
        //     state->address_list[acct].print();
        //     printf("balance: \n");
        //     state->account_balances[instance_idx].print();

        //     // Get the storage size for this account (again, from the first snapshot).
        //     printf("account %d instance %d storage size: %d\n", acct, instance_idx, storage_size);
        // }

        if (storage_size > 0) {
            // The contract index tells us which section of the preallocated storage pool to use.
            for (uint32_t s = 0; s < storage_size; s++) {
                if (s > account_prealloc_keys_size) break;
                // Compute the index into the preallocated storage arrays.
                // (account_prealloc_keys_size * contract_index + storage_element)
                // is multiplied by num_states because storage is stored for every state.
                uint32_t prealloc_idx =
                    (account_prealloc_keys_size * cidx + s) * state->num_states + INSTANCE_GLOBAL_IDX;

                // Convert the storage key and value into hex strings.
                data->storage_keys[data->no_storage_elements + s] = state->prealloc_keys_pool[prealloc_idx];
                data->storage_values[data->no_storage_elements + s] = state->prealloc_values_pool[prealloc_idx].value;
                // Record which account this storage element belongs to.
                data->storage_indexes[data->no_storage_elements + s] = acct;
            }
        }
        // Increment the total count of storage elements serialized.
        data->no_storage_elements += storage_size;
    }
}
}  // namespace CuEVM