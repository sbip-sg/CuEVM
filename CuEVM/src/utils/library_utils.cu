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

__device__ uint32_t* g_events_bitmap = nullptr;
__device__ uint32_t* g_total_bug_table = nullptr;
__device__ uint32_t* g_total_bug_count = nullptr;

// event trackers, reset every kernel launch

__device__ BranchInfoEntry* g_new_branch_info = nullptr;
__device__ StorageInfoEntry* g_new_storage_info = nullptr;
__device__ BugInfoEntry* g_new_bug_info = nullptr;
__device__ GPUFeedbackCount* g_gpu_feedback_count =
    nullptr;  // counter for interesting events, reset every kernel launch

__device__ fuzzing_constants* g_fuzzing_constants = nullptr;
__device__ uint32_t* g_static_marker_data = nullptr;

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

// Simplified coverage: Pure AFL simple hash (shift + XOR) for max performance, single atomicMin
__device__ void simplified_trace_data::update_coverage_bitmap_with_distance(uint32_t pc_src, uint32_t pc_dst,
                                                                            uint32_t pc_missed, uint8_t distance_bits) {
    // AFL simple hash: Extremely fast (shift + XOR), citable from AFL (widely used in fuzzing papers)

    uint32_t afl_hash_covered = (pc_src << 1) ^ pc_dst ^ current_account_id;
    uint32_t afl_hash_missed = (pc_src << 1) ^ pc_missed ^ current_account_id;

    uint32_t bitmap_idx_covered = afl_hash_covered % BITMAP_SIZE;
    uint32_t bitmap_idx_missed = afl_hash_missed % BITMAP_SIZE;

    unsigned int prev_dist = atomicAdd(&g_events_bitmap[bitmap_idx_covered], 1);

    // Fast check: if 24 bit lower bits of prev_dist is 0, then this is a new coverage
    if ((prev_dist & 0xFFFFFF) == 0) {
        last_covered_branch_id = bitmap_idx_covered + 1;  // +1 to avoid 0
    } else {
        // only track the last one
        if (last_covered_branch_id != 0) {
            last_covered_branch_id = bitmap_idx_covered + 1;  // +1 to avoid 0
        }
    }
    uint32_t dist_compare = (255 - distance_bits) << 24;  // first 8 bits are distance bit of the branch
    // update missed branch
    prev_dist = atomicMax(&g_events_bitmap[bitmap_idx_missed], dist_compare);

    if (dist_compare > prev_dist) {
        last_missed_branch_id = bitmap_idx_missed + 1;  // +1 to avoid 0
        last_distance_bits = distance_bits;
    } else {
        // only track the last one
        if (last_missed_branch_id != 0) {
            last_missed_branch_id = bitmap_idx_missed + 1;  // +1 to avoid 0
            last_distance_bits = distance_bits;
        }
    }
}
__device__ __forceinline__ uint32_t fnv1a(uint32_t value) {
    uint32_t hash = 2166136261U;                 // FNV offset basis
    hash = (hash ^ (value & 0xFF)) * 16777619U;  // FNV prime
    hash = (hash ^ ((value >> 8) & 0xFF)) * 16777619U;
    hash = (hash ^ ((value >> 16) & 0xFF)) * 16777619U;
    hash = (hash ^ (value >> 24)) * 16777619U;
    return hash;
}
__device__ void simplified_trace_data::update_storage_coverage(uint32_t pc, uint16_t storage_slot, uint8_t is_write) {
    if (current_account_id == REENTRANCY_ATTACKER_ADDRESS) return;

    uint32_t storage_hash_id =
        is_write ? pc ^ (storage_slot + BITMAP_SIZE / 2) ^ current_account_id : pc ^ storage_slot ^ current_account_id;

    // uint32_t bitmap_idx = fnv1a(storage_id) % BITMAP_SIZE;
    uint32_t bitmap_idx = storage_hash_id % BITMAP_SIZE;
    unsigned int prev_dist = atomicMax(&g_events_bitmap[bitmap_idx], 1);

    if (prev_dist == 0) {
        int idx = atomicAdd(&g_gpu_feedback_count->new_storage_count, 1);
        if (idx < CuEVM::MAX_NEW_STORAGE) {
            g_new_storage_info[idx].storage_thread_idx = INSTANCE_GLOBAL_IDX;
            // Construct 32-bit storage_id:
            // [31:30] 2 bits: is_write (last 2 bits)
            // [29:16] 14 bits: account_idx (last 14 bits)
            // [15:0]  16 bits: storage_slot
            g_new_storage_info[idx].storage_id = ((uint32_t)(is_write & 0x3) << 30) |
                                                 ((uint32_t)(current_account_id & 0x3FFF) << 16) |
                                                 ((uint32_t)storage_slot & 0xFFFF);
        }
    }
}
__device__ void simplified_trace_data::add_bugs_for_later(uint32_t pc, uint8_t bug_type) {
    uint32_t bug_id = pc << 16 | bug_type << 8 | (current_account_id & 0xFF);
    // printf("thread %d add_bugs_for_later pc %u bug_type %u bug_id 0x%08x to_addr 0x%08x\n", INSTANCE_GLOBAL_IDX, pc,
    //        bug_type, bug_id, current_account_id);

    bugs[no_bugs] = bug_id;
    if (no_bugs >= MAX_BUGS_TRACING) {
        // printf("thread %d no_bugs >= MAX_BUGS_TRACING, %d\n", INSTANCE_GLOBAL_IDX, no_bugs);
        no_bugs = 0;
        return;
    }
    no_bugs++;
}
// Completed bugs: FNV-1a hash (citable, e.g., from Fowler–Noll–Vo papers) + quadratic probing (optimized, bounded)
__device__ void simplified_trace_data::update_bugs(uint32_t bug_id) {
    // FNV-1a:
    uint32_t hash = fnv1a(bug_id);

    bool inserted = false;
    for (uint32_t probe = 0; probe < MAX_NEW_BUGS; ++probe) {
        uint32_t idx = (hash + (probe * probe)) % MAX_NEW_BUGS;
        // printf("thread %d update_bugs probe %u idx %u\n", INSTANCE_GLOBAL_IDX, probe, idx);

        unsigned int cas_result = atomicCAS(&g_total_bug_table[idx], 0, bug_id);
        if (cas_result == 0) {
            // Successful insert into empty slot
            inserted = true;
            break;
        } else if (cas_result == bug_id) {
            // Another thread already inserted this exact bug
            return;
        }
        // Slot contains different bug_id, continue probing
        // printf("thread %d update_bugs slot occupied, continue probing probe %u\n", INSTANCE_GLOBAL_IDX, probe);
    }

    if (inserted) {
        int idx = atomicAdd(&g_gpu_feedback_count->new_bug_count, 1);
        if (idx < CuEVM::MAX_NEW_BUGS) {
            g_new_bug_info[idx].bug_thread_idx = INSTANCE_GLOBAL_IDX;
            g_new_bug_info[idx].bug_id = bug_id;
        }
    }
}

__device__ void simplified_trace_data::finalize_coverage_bitmap(int32_t error_code) {
    // printf("thread %d finalize_coverage_bitmap last_covered_branch_id %u last_missed_branch_id %u\n",
    //        INSTANCE_GLOBAL_IDX, last_covered_branch_id, last_missed_branch_id);
    if (last_covered_branch_id != 0) {
        int idx = atomicAdd(&g_gpu_feedback_count->new_branch_count, 1);
        // printf("Found thread %d last_covered_branch_id %u\n", INSTANCE_GLOBAL_IDX, last_covered_branch_id);
        if (idx < CuEVM::MAX_NEW_BRANCHES) {
            g_new_branch_info[idx].branch_thread_idx = INSTANCE_GLOBAL_IDX;
            g_new_branch_info[idx].branch_id = last_covered_branch_id;
        }
        return;
    }
    if (last_missed_branch_id != 0) {
        uint32_t global_distance_bits = 255 - (g_events_bitmap[last_missed_branch_id - 1] >> 24);
        // printf("thread %d global_distance_bits %u last_distance_bits %u\n", INSTANCE_GLOBAL_IDX,
        // global_distance_bits,
        //        last_distance_bits);
        if (global_distance_bits == last_distance_bits) {
            // printf("Found thread %d last_missed_branch_id %u last_distance_bits %u global_distance_bits %u\n",
            //        INSTANCE_GLOBAL_IDX, last_missed_branch_id, last_distance_bits, global_distance_bits);
            int idx = atomicAdd(&g_gpu_feedback_count->new_branch_count, 1);
            if (idx < CuEVM::MAX_NEW_BRANCHES) {
                g_new_branch_info[idx].branch_thread_idx = INSTANCE_GLOBAL_IDX;
                g_new_branch_info[idx].branch_id = last_missed_branch_id;
            }
        }
    }

    // add bugs that require sucess tx
    if (no_bugs > 0 && (error_code == ERROR_SUCCESS || error_code == ERROR_RETURN)) {
        // printf("thread %d finalize_coverage_bitmap state_written %d\n", INSTANCE_GLOBAL_IDX, state_written);
        for (uint32_t i = 0; i < no_bugs; i++) {
            // printf("thread %d add bug %x state_written %d\n", INSTANCE_GLOBAL_IDX, bugs[i], state_written);
            uint8_t bug_type = static_cast<uint8_t>((bugs[i] >> 8) & 0xFF);

            if (bug_type == BUG_INTEGER_BUG) {
                if (state_written) update_bugs(bugs[i]);
            } else {
                update_bugs(bugs[i]);
            }
        }
    }
}

__device__ bool simplified_trace_data::increase_branch_count() {
    if (no_branches >= MAX_BRANCHES_TRACING) {
        no_branches = MAX_BRANCHES_TRACING;
        return true;
    }
    no_branches++;
    return false;
}
__device__ bool simplified_trace_data::record_branch(uint32_t pc_src, uint32_t pc_dst, uint32_t pc_missed) {
    // printf("thread %d branch count %d\n", INSTANCE_GLOBAL_IDX, no_branches);
    if (no_branches >= MAX_BRANCHES_TRACING) {
        no_branches = MAX_BRANCHES_TRACING;
        // printf("Thread %d: no_branches >= MAX_BRANCHES_TRACING, %d\n", INSTANCE_GLOBAL_IDX, no_branches);
        return true;
    }

    // printf("record branch pc_src %u pc_dst %u distance %s\n", pc_src, pc_dst,
    // branches[no_branches].distance.to_hex());
#ifdef BUILD_GO_LIBRARY
    // calculate distance bits
    uint32_t distance_bits = uint256_bitlength(&last_distance);
    if (distance_bits > 255) {
        distance_bits = 255;
    }

    // go library: branch info is recorded on CPU side
    // TODO Aug: remove this check after we bypass logic of two attackers
    if (current_account_id != REENTRANCY_ATTACKER_ADDRESS && current_account_id != RANDOM_ATTACKER_ADDRESS &&
        no_branches > 0)  // skip first branch
        update_coverage_bitmap_with_distance(pc_src, pc_dst, pc_missed, distance_bits);
#endif

    no_branches++;
    return false;
}

__device__ void simplified_trace_data::record_distance(uint8_t op, const CuEVM::evm_stack_t& stack_ptr) {
    evm_word_t distance;
    evm_word_t* op1 = stack_ptr.get_address_at_index(1);
    evm_word_t* op2 = stack_ptr.get_address_at_index(2);
    uint32_t stack_size = stack_ptr.size();
    if (stack_size < 2) return;
    if (uint256_cmp(op1, op2) >= 1)
        uint256_sub(&distance, op1, op2);
    else
        uint256_sub(&distance, op2, op1);

    if (op != OP_EQ) uint256_add_word(&distance, &distance, 1);
    // printf("thread %d record_distance op %u distance %s\n", INSTANCE_GLOBAL_IDX, op, distance.to_hex());
    last_distance = distance;
}

__device__ void simplified_trace_data::start_call(uint32_t pc, evm_call_context_t* call_context_ptr) {
    assert(call_context_ptr != nullptr);
    // if (current_account_id != 0)
    //     printf("thread %d start_call pc %u current_account_id %x\n", INSTANCE_GLOBAL_IDX, pc, current_account_id);
#ifdef BUILD_GO_LIBRARY

    no_branches += 10;  // call saturates the branch limit faster than normal jumps
                        // state_written = true;
#endif
    if (no_calls >= MAX_CALLS_TRACING) return;
    // add address and increment current_address_idx
    // addresses[current_address_idx] = cached_call_state->addresses[cached_call_state->current_address_idx];
    // printf("start call simplified trace data pc %d no calls %d\n", pc, no_calls);
    calls[no_calls].sender_id = call_context_ptr->from.words[0];  // & 0xffff;
    current_account_id = call_context_ptr->to.words[0];           // & 0xffff;
    calls[no_calls].receiver_id = current_account_id;
    // printf("thread %d start_call sender_id %x receiver_id %x\n", INSTANCE_GLOBAL_IDX, calls[no_calls].sender_id,
    //        calls[no_calls].receiver_id);
    calls[no_calls].pc = pc;
    calls[no_calls].op = call_context_ptr->call_type;
    if (call_context_ptr->parent != nullptr) {
        calls[no_calls].value_leaking = uint256_cmp(&call_context_ptr->value, &call_context_ptr->parent->value) > 0;
    } else {
        calls[no_calls].value_leaking = false;
    }
    calls[no_calls].call_data_size = call_context_ptr->call_data_size;
    if (calls[no_calls].call_data_size > 0) {
        calls[no_calls].first_byte_call_data = call_context_ptr->call_data[0];
    }

    if (call_context_ptr->value.words[0] != 0) {
        state_written = true;  // transfer = true
    }
    calls[no_calls].error_code = RESERVED_ERROR_CODE;
    calls[no_calls].last_pc = 0;
    no_calls++;

    // return ERROR_SUCCESS;
}
__device__ void simplified_trace_data::start_create() {
    // printf("thread %d finish_create no_branches %d\n", INSTANCE_GLOBAL_IDX, no_branches);
    no_branches += MAX_BRANCHES_TRACING / 2;
    state_written = true;
}
__device__ void simplified_trace_data::selfdestruct_oracle(uint32_t pc) {
    // printf("thread %d selfdestruct_oracle\n", INSTANCE_GLOBAL_IDX);
    if (calls[0].sender_id == RANDOM_ATTACKER_ADDRESS)
        update_bugs(pc << 16 | BUG_SELF_DESTRUCT << 8 | (current_account_id & 0xFF));
}

__device__ void simplified_trace_data::reentrancy_oracle(uint32_t pc) {
    // printf("thread %d reentrancy_oracle pc %u\n", INSTANCE_GLOBAL_IDX, pc);
    update_bugs(pc << 16 | BUG_REENTRANCY << 8 | (current_account_id & 0xFF));
}
__device__ void simplified_trace_data::leaking_ether_oracle(uint32_t pc) {
    // printf("thread %d leaking_ether_oracle pc %u receiver_id %x\n", INSTANCE_GLOBAL_IDX, pc, receiver_id);

    // printf("thread %d leaking_ether_oracle\n", INSTANCE_GLOBAL_IDX);
    // update_bugs(pc << 16 | BUG_LEAKING_ETHER << 8 | (current_account_id & 0xFF));
    add_bugs_for_later(pc, BUG_LEAKING_ETHER);
}

__device__ void simplified_trace_data::arbitrary_call_oracle(uint32_t pc, uint8_t first_byte_call_data) {
    // Pack into single 32-bit value to avoid pc/data visibility races: value = (pc << 16) | first_byte_call_data
    // Assumes pc fits in 16 bits.
    uint32_t packed = (pc << 16) | (uint32_t)first_byte_call_data;
    bool bug = false;
    for (int i = 0; i < MAX_ARBITRARY_CALL_CHECK; i++) {
        uint32_t stored = g_fuzzing_constants->arbitrary_call_check[i];

        if (stored == packed) return;  // exact match already present

        if (stored == 0) {
            uint32_t prev = atomicCAS(&g_fuzzing_constants->arbitrary_call_check[i], 0, packed);
            if (prev == 0 || prev == packed) return;  // claimed or identical inserted concurrently
        } else {
            // Same pc but different first byte? Compare high 16 bits.
            if ((stored >> 16) == pc && (stored & 0xFFFFu) != (uint32_t)first_byte_call_data) {
                bug = true;
                break;
            }
        }
    }

    // same PC but different first byte
    if (bug) add_bugs_for_later(pc, BUG_ARBITRARY_CALL);
}

__device__ void simplified_trace_data::invalid_opcode_oracle(uint32_t pc) {
    update_bugs(pc << 16 | BUG_INVALID_OPCODE << 8 | (calls[0].receiver_id & 0xFF));
}

__device__ void simplified_trace_data::finish_call(uint8_t error_code, uint32_t last_pc, uint32_t _current_account_id) {
    // printf("thread %d finish_call error_code %u last_pc %u, no_calls %u\n", INSTANCE_GLOBAL_IDX, error_code, last_pc,
    //        no_calls);

    if (no_calls > MAX_CALLS_TRACING) {
        // printf("THREAD %d no_calls > MAX_CALLS_TRACING, no_calls %d\n", INSTANCE_GLOBAL_IDX, no_calls);

        return;
    }
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
    if (i < 0) i = 0;

#ifdef BUILD_GO_LIBRARY
    if (error_code == ERROR_INVALID_OPCODE) {
        invalid_opcode_oracle(last_pc);
    }

    if (no_calls > 1 && calls[i].receiver_id == RANDOM_ATTACKER_ADDRESS) {
        // current value is greater than the first call value
        if (calls[i].value_leaking) {
            leaking_ether_oracle(last_pc);
        }

        if (calls[i].call_data_size > 1) {
            arbitrary_call_oracle(last_pc, calls[i].first_byte_call_data);
        }
    }

    current_account_id = _current_account_id;

#endif
}
__host__ __device__ void simplified_trace_data::print() {
    // printf("no_events %u\n", no_events);
    printf("no_calls %u\n", no_calls);
    printf("no_bugs %u\n", no_bugs);
    printf("bugs\n");
    for (uint32_t i = 0; i < no_bugs; i++) {
        printf("bug %u\n", bugs[i]);
    }

    printf("calls\n");
    for (uint32_t i = 0; i < no_calls; i++) {
        printf("pc %u op %u sender_id %u receiver_id %u value_leaking %u first_byte_call_data %x error_code %u\n",
               calls[i].pc, calls[i].op, calls[i].sender_id, calls[i].receiver_id, calls[i].value_leaking,
               calls[i].first_byte_call_data, calls[i].error_code);
    }
    printf("branches\n");
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
#ifndef BUILD_GO_LIBRARY
    if (temp_list.gas_limit != nullptr) {
        CUDA_CHECK(cudaFree(temp_list.gas_limit));
    }
#endif
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
// #define LCG_A 1664525
// #define LCG_C 1013904223
// #define LCG_M 0xFFFFFFFF  // 2^32 - 1
#define GLIBC_LCG_A 1103515245
#define GLIBC_LCG_C 12345

// AFL-style mutation configuration
#define CHANCE_TO_CREATE_NEW_ADDRESS 0  // 1 percent
#define CHANCE_TO_SKIP_MUTATE 50        // percent we skip a marker
#define CHANCE_TO_SKIP_MUTATE_VALUE 25  // percent we skip a marker
// #define CHANCE_TO_SKIP_MUTATE_BLOCK 25  // percent we skip block mutation
#define CHANCE_HAVOC_MUTATION 6  // percent chance for havoc (stacked mutations)
#define MAX_HAVOC_STACK 3        // maximum number of stacked mutations in havoc
#define VALUE_MUTATE_INT32 7     // 2**(7*4*8) = 2**224

// Mutation context to reduce parameter passing
struct MutationContext {
    uint8_t* data;
    uint32_t length;
    uint32_t seed;
};

// Helper: Generate next random number
__device__ __forceinline__ uint32_t next_rand(uint32_t& seed) {
    // printf("Thread %d next_rand original seed %u\n", INSTANCE_GLOBAL_IDX, seed);
    // seed = (LCG_A * seed + LCG_C) % LCG_M;
    uint64_t temp = (uint64_t)GLIBC_LCG_A * seed + GLIBC_LCG_C;
    seed = temp & 0x7FFFFFFF;  // Modulo 2^31
    return seed;               // 31-bit output
}

// Helper: Get random in range [0, max)
__device__ __forceinline__ uint32_t rand_range(uint32_t& seed, uint32_t max) {
    return max == 0 ? 0 : next_rand(seed) % max;
}

// AFL mutation types
enum MutationType {
    MUTATE_BIT_FLIP = 0,
    MUTATE_BYTE_FLIP,
    MUTATE_ARITHMETIC,
    MUTATE_KNOWN_INTEGER,
    MUTATE_RANDOM_BYTES,
    MUTATE_HAVOC,
    MUTATE_TYPE_COUNT
};

__device__ void mutate_bit_flip(MutationContext& ctx) {
    if (ctx.length == 0) return;

    uint32_t flip_size = 1 << rand_range(ctx.seed, 3);  // 1, 2, or 4 bits
    uint32_t max_bit_pos = ctx.length * 8 - flip_size + 1;
    uint32_t bit_pos = rand_range(ctx.seed, max_bit_pos);

    uint32_t byte_idx = bit_pos / 8;
    uint32_t bit_idx = bit_pos % 8;

    uint32_t mask = ((1 << flip_size) - 1) << bit_idx;
    if (bit_idx + flip_size <= 8) {
        ctx.data[byte_idx] ^= mask;
    } else {
        // Handle cross-byte boundary
        ctx.data[byte_idx] ^= mask & 0xFF;
        if (byte_idx + 1 < ctx.length) {
            ctx.data[byte_idx + 1] ^= (mask >> 8) & 0xFF;
        }
    }
}

__device__ void mutate_byte_flip(MutationContext& ctx) {
    if (ctx.length == 0) return;

    uint32_t flip_size = 1 << rand_range(ctx.seed, 3);  // 1, 2, or 4 bytes
    if (flip_size > ctx.length) flip_size = ctx.length;

    uint32_t start_idx = rand_range(ctx.seed, ctx.length - flip_size + 1);

    for (uint32_t i = 0; i < flip_size; i++) {
        ctx.data[start_idx + i] ^= 0xFF;
    }
}

__device__ void mutate_arithmetic(MutationContext& ctx) {
    if (ctx.length == 0) return;

    uint32_t delta = 1 + rand_range(ctx.seed, 35);  // Small arithmetic delta (1-35)
    bool is_add = rand_range(ctx.seed, 2) == 0;

    if (ctx.length >= 8) {
        // 64-bit arithmetic with big-endian
        // for efficiency, we dont do arithmetic on higher bytes
        uint32_t start_idx = ctx.length - 8;
        uint64_t value = ((uint64_t)ctx.data[start_idx] << 56) | ((uint64_t)ctx.data[start_idx + 1] << 48) |
                         ((uint64_t)ctx.data[start_idx + 2] << 40) | ((uint64_t)ctx.data[start_idx + 3] << 32) |
                         ((uint64_t)ctx.data[start_idx + 4] << 24) | ((uint64_t)ctx.data[start_idx + 5] << 16) |
                         ((uint64_t)ctx.data[start_idx + 6] << 8) | ((uint64_t)ctx.data[start_idx + 7]);
        value = is_add ? (value + delta) : (value - delta);
        ctx.data[start_idx] = (value >> 56) & 0xFF;
        ctx.data[start_idx + 1] = (value >> 48) & 0xFF;
        ctx.data[start_idx + 2] = (value >> 40) & 0xFF;
        ctx.data[start_idx + 3] = (value >> 32) & 0xFF;
        ctx.data[start_idx + 4] = (value >> 24) & 0xFF;
        ctx.data[start_idx + 5] = (value >> 16) & 0xFF;
        ctx.data[start_idx + 6] = (value >> 8) & 0xFF;
        ctx.data[start_idx + 7] = value & 0xFF;
    } else if (ctx.length >= 4) {
        // 32-bit arithmetic with big-endian
        uint32_t start_idx = ctx.length - 4;
        uint32_t value = (ctx.data[start_idx] << 24) | (ctx.data[start_idx + 1] << 16) |
                         (ctx.data[start_idx + 2] << 8) | ctx.data[start_idx + 3];
        value = is_add ? (value + delta) : (value - delta);
        ctx.data[start_idx] = (value >> 24) & 0xFF;
        ctx.data[start_idx + 1] = (value >> 16) & 0xFF;
        ctx.data[start_idx + 2] = (value >> 8) & 0xFF;
        ctx.data[start_idx + 3] = value & 0xFF;
    } else {
        // 8-bit arithmetic
        uint32_t byte_idx = ctx.length - 1;
        ctx.data[byte_idx] = is_add ? ((ctx.data[byte_idx] + delta) & 0xFF) : ((ctx.data[byte_idx] - delta) & 0xFF);
    }
}

__device__ void mutate_known_integer(MutationContext& ctx) {
    if (ctx.length == 0 || g_fuzzing_constants->integer_constants_count == 0) return;

    uint32_t random_index = rand_range(ctx.seed, g_fuzzing_constants->integer_constants_count);
    uint32_t copy_length = ctx.length > 32 ? 32 : ctx.length;

    for (uint32_t i = 0; i < copy_length; i++) {
        ctx.data[i] = g_fuzzing_constants->integer_constants[random_index * 32 + i];
    }
}

__device__ void mutate_random_bytes(MutationContext& ctx) {
    if (ctx.length == 0) return;

    uint32_t start_pos = rand_range(ctx.seed, ctx.length);
    uint32_t mutated_bytes = 1 + rand_range(ctx.seed, ctx.length - start_pos);

    for (uint32_t i = 0; i < mutated_bytes; i++) {
        ctx.data[start_pos + i] = next_rand(ctx.seed) & 0xFF;
    }
}

__device__ void mutate_havoc(MutationContext& ctx) {
    uint32_t stack_count = 1 + rand_range(ctx.seed, MAX_HAVOC_STACK);

    for (uint32_t i = 0; i < stack_count; i++) {
        uint32_t mutation_type = rand_range(ctx.seed, MUTATE_TYPE_COUNT - 1);  // Exclude MUTATE_HAVOC

        switch (mutation_type) {
            case MUTATE_BIT_FLIP:
                mutate_bit_flip(ctx);
                break;
            case MUTATE_BYTE_FLIP:
                mutate_byte_flip(ctx);
                break;
            case MUTATE_ARITHMETIC:
                mutate_arithmetic(ctx);
                break;
            case MUTATE_KNOWN_INTEGER:
                mutate_known_integer(ctx);
                break;
            case MUTATE_RANDOM_BYTES:
                mutate_random_bytes(ctx);
                break;
        }
    }
}

__device__ uint32_t afl_mutate_byte_array(uint8_t* data, uint32_t data_length, uint32_t element_bits, uint32_t seed) {
    if (data == nullptr || data_length == 0) return seed;

    uint32_t element_bytes = element_bits / 8;
    if (element_bytes > data_length) element_bytes = data_length;

    MutationContext ctx = {.data = data + data_length - element_bytes, .length = element_bytes, .seed = seed};

    // Check for havoc mutation first
    if (rand_range(ctx.seed, 100) < CHANCE_HAVOC_MUTATION) {
        mutate_havoc(ctx);
        return ctx.seed;
    }

    // Choose regular mutation type
    uint32_t mutation_type = rand_range(ctx.seed, MUTATE_TYPE_COUNT - 1);  // Exclude MUTATE_HAVOC

    switch (mutation_type) {
        case MUTATE_BIT_FLIP:
            mutate_bit_flip(ctx);
            break;
        case MUTATE_BYTE_FLIP:
            mutate_byte_flip(ctx);
            break;
        case MUTATE_ARITHMETIC:
            mutate_arithmetic(ctx);
            break;
        case MUTATE_KNOWN_INTEGER:
            mutate_known_integer(ctx);
            break;
        case MUTATE_RANDOM_BYTES:
        default:
            // clear the data
            for (uint32_t i = 0; i < data_length; i++) {
                data[i] = 0;
            }
            // mutate the data
            mutate_random_bytes(ctx);
            break;
    }

    return ctx.seed;
}

__device__ uint32_t mutate_block_values_senders(uint32_t seed, uint64_t* block_numbers, uint64_t* block_timestamps,
                                                uint8_t* senders) {
    // printf("Thread %d seed %d block_numbers orig %lu block_timestamps orig %lu\n", INSTANCE_GLOBAL_IDX, seed,
    //        block_numbers[INSTANCE_GLOBAL_IDX], block_timestamps[INSTANCE_GLOBAL_IDX]);
    // block number first
    if (rand_range(seed, 100) <= CHANCE_TO_SKIP_MUTATE) {
        // printf("Thread %d skip block mutation %d\n", INSTANCE_GLOBAL_IDX);
        block_numbers[INSTANCE_GLOBAL_IDX] += 1;
        block_timestamps[INSTANCE_GLOBAL_IDX] += 1;
        return seed;
    }

    uint32_t block_number = rand_range(seed, g_fuzzing_constants->block_number_delay_max);
    uint32_t block_timestamp = rand_range(seed, g_fuzzing_constants->block_timestamp_delay_max);
    // printf("Thread %d mutated block_number %d block_timestamp %d seed %d\n", INSTANCE_GLOBAL_IDX, block_number,
    //        block_timestamp, seed);
    if (block_timestamp == 0)
        block_number = 0;
    else
        block_number = block_number % block_timestamp;
    // printf("Thread %d mutated block_number %d block_timestamp %d\n", INSTANCE_GLOBAL_IDX, block_number,
    //        block_timestamp);
    block_numbers[INSTANCE_GLOBAL_IDX] += block_number;
    block_timestamps[INSTANCE_GLOBAL_IDX] += block_timestamp;
    // printf("Thread %d mutated block_numbers %lu block_timestamps %lu\n", INSTANCE_GLOBAL_IDX,
    //        block_numbers[INSTANCE_GLOBAL_IDX], block_timestamps[INSTANCE_GLOBAL_IDX]);
    senders[INSTANCE_GLOBAL_IDX] = rand_range(seed, g_fuzzing_constants->sender_counts);
    // printf("Thread %d sender_counts %d sender %d seed %d\n", INSTANCE_GLOBAL_IDX, g_fuzzing_constants->sender_counts,
    //        senders[INSTANCE_GLOBAL_IDX], seed);
    return seed;
}

__device__ uint32_t mutate_value(uint32_t seed, evm_word_t* value) {
    if (rand_range(seed, 100) <= CHANCE_TO_SKIP_MUTATE_VALUE) {
        return seed;
    }
    // always mutate at least one value
    uint32_t value_int_type = rand_range(seed, VALUE_MUTATE_INT32 - 1);
    for (int i = 0; i < value_int_type + 1; i++) {
        value->words[i] = next_rand(seed);
    }

    return seed;
}
__device__ void mutate_transaction_data(CuEVM::transaction::TransactionList* transaction_list_ptr) {
    uint32_t seed = INSTANCE_GLOBAL_IDX + transaction_list_ptr->start_seed;

    uint32_t marker_idx = INSTANCE_GLOBAL_IDX / CUEVM_MUTATE_GROUP_SIZE;
    // printf("thread %d marker_idx %d , marker offset %d seed %u \n", INSTANCE_GLOBAL_IDX, marker_idx,
    //        transaction_list_ptr->marker_offset[marker_idx], seed);
    int32_t marker_offset = transaction_list_ptr->marker_offset[marker_idx];
    uint32_t* marker_data;

    if (marker_offset < 0) {
        marker_data = g_static_marker_data + (-marker_offset - 1);
    } else {
        marker_data = transaction_list_ptr->marker_data + marker_offset;
    }
    // The first element is the marker size, then marker_data points to the next element
    uint32_t marker_size = *marker_data++;

    // mutate block values
    seed = mutate_block_values_senders(seed, transaction_list_ptr->block_number, transaction_list_ptr->time_stamp,
                                       transaction_list_ptr->sender);

    // printf("Thread %d sender %d, marker_size %d, seed %d\n", INSTANCE_GLOBAL_IDX,
    //        transaction_list_ptr->sender[INSTANCE_GLOBAL_IDX], marker_size, seed);

    if (marker_size == 0) return;

    uint8_t* call_data = &transaction_list_ptr->call_data[transaction_list_ptr->call_data_offset[INSTANCE_GLOBAL_IDX]];
    uint32_t distance_from_count =
        g_fuzzing_constants->sender_counts - transaction_list_ptr->sender[INSTANCE_GLOBAL_IDX];

    for (int j = 0; j < marker_size; j++) {
        uint32_t element_offset = *marker_data++;
        uint32_t element_type = *marker_data++;
        uint32_t element_length = *marker_data++;

        if (element_type == ELEMENT_VALUE_TYPE && distance_from_count != 1) {  // random attacker does not mutate value
            // printf("thread %d value mutation\n", INSTANCE_GLOBAL_IDX);
            seed = mutate_value(seed, &transaction_list_ptr->value[INSTANCE_GLOBAL_IDX]);

            continue;
        }

        if (element_type == ELEMENT_BOOL_TYPE) {  // always mutate bool
            // printf("thread %d bool mutation\n", INSTANCE_GLOBAL_IDX);
            call_data[element_offset + 31] = next_rand(seed) % 2;  // set last byte to 0 or 1
            continue;
        }

        if (rand_range(seed, 100) <= CHANCE_TO_SKIP_MUTATE && element_type != ELEMENT_ADDRESS_TYPE) {
            continue;
        }

        if (element_type > 7) {
            seed = afl_mutate_byte_array(call_data + element_offset, element_length, element_type, seed);

        } else if (element_type == ELEMENT_ADDRESS_TYPE) {  // address
            // if (rand_range(seed, 100) < CHANCE_TO_CREATE_NEW_ADDRESS) {
            //     // printf("thread %d create new address\n", INSTANCE_GLOBAL_IDX);
            //     MutationContext ctx = {.data = call_data + element_offset + 12, .length = 20, .seed = seed};
            //     mutate_random_bytes(ctx);
            //     seed = ctx.seed;
            // } else {
            // select from the constants
            uint32_t address_constants_count = g_fuzzing_constants->address_constants_count;
            if (distance_from_count <= 2) address_constants_count += 3 - distance_from_count;
            // distance_from_count == 1 // last address -> random attacker => use all address constants (+2)
            // distance_from_count == 2 // last 2 address -> reentrancy attacker => use all address constants except
            // random attacker (+1)

            uint32_t random_index = rand_range(seed, address_constants_count);
            for (int i = 0; i < 20; i++) {
                call_data[element_offset + 12 + i] = g_fuzzing_constants->address_constants[random_index * 32 + 12 + i];
            }
            // }
        }
    }
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