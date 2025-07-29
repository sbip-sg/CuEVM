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

// __device__ uint32_t* g_coverage_bitmap = nullptr;
// __device__ uint32_t* g_new_coverage_bitmap = nullptr;
// __device__ uint32_t* g_new_coverage_count = nullptr;
// __device__ uint32_t* g_new_coverage_idx = nullptr;
// __device__ uint32_t* g_new_bug_pc = nullptr;
// __device__ uint32_t* g_new_bug_count = nullptr;
// __device__ uint32_t* g_new_bug_idx = nullptr;

__device__ uint32_t* g_events_bitmap = nullptr;
__device__ uint32_t* g_total_bug_table = nullptr;
__device__ uint32_t* g_total_bug_count = nullptr;

// event trackers, reset every kernel launch
// __device__ uint32_t* g_new_coverage_bitmap =
//     nullptr;  // for each thread to set a flag if they encounter a new branch or a
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

/*
Deprecated: use update_coverage_bitmap_with_distance and update_bugs instead
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
*/
/* // template code, to remove
__device__ void simplified_trace_data::update_coverage_bitmap_with_distance(uint32_t pc_src, uint32_t pc_dst,
                                                                            uint8_t distance_bits) {
    uint32_t bitmap_idx = ((pc_src >> 1) ^ pc_dst) % BITMAP_SIZE;
    printf("thread %d update_coverage_bitmap_with_distance pc_src %u pc_dst %u distance_bits %u, bitmap_idx %u\n",
           INSTANCE_GLOBAL_IDX, pc_src, pc_dst, distance_bits, bitmap_idx);
    unsigned int old_dist = g_coverage_bitmap[bitmap_idx];
    unsigned int prev_dist = atomicMin(&g_coverage_bitmap[bitmap_idx], distance_bits);
    if (distance_bits < old_dist) {
    }
}

__device__ void simplified_trace_data::update_bugs(uint32_t pc, uint8_t bug_type) {
    // use global array to record bugs. Search through exsiting bug list and add new bug if not found
    uint32_t bug_id = pc << 16 | bug_type;
    for (int i = 0; i < g_total_bug_count; i++) {
        if (g_bug_list[i] == bug_id) {
            return;
        }
    }
    // add new bug, atomic add to g_total_bug_count, persist across kernels
    int idx = atomicInc(g_total_bug_count, CuEVM::MAX_NEW_BUGS);
    if (idx < CuEVM::MAX_NEW_BUGS) {
        g_total_bug_list[idx] = bug_id;
        g_total_bug_idx[idx] = INSTANCE_GLOBAL_IDX;  // for reconstructing on CPU side
    }
}
*/

// Simplified coverage: Pure AFL simple hash (shift + XOR) for max performance, single atomicMin
__device__ void simplified_trace_data::update_coverage_bitmap_with_distance(uint32_t pc_src, uint32_t pc_dst,
                                                                            uint32_t pc_missed, uint8_t distance_bits) {
    // AFL simple hash: Extremely fast (shift + XOR), citable from AFL (widely used in fuzzing papers)
    // printf(
    //     "thread %d update_coverage_bitmap_with_distance pc_src %u pc_dst %u pc_missed %u distance_bits %u "
    //     "current_account_id %x\n",
    //     INSTANCE_GLOBAL_IDX, pc_src, pc_dst, pc_missed, distance_bits, current_account_id);
    uint32_t afl_hash_covered = (pc_src << 1) ^ pc_dst ^ current_account_id;
    uint32_t afl_hash_missed = (pc_src << 1) ^ pc_missed ^ current_account_id;

    uint32_t bitmap_idx_covered = afl_hash_covered % BITMAP_SIZE;
    uint32_t bitmap_idx_missed = afl_hash_missed % BITMAP_SIZE;

    // printf(
    //     "thread %d update_coverage_bitmap_with_distance pc_src %u pc_dst %u pc_missed %u distance_bits %u, "
    //     "bitmap_idx_covered %u, bitmap_idx_missed %u\n",
    //     INSTANCE_GLOBAL_IDX, pc_src, pc_dst, pc_missed, distance_bits, bitmap_idx_covered, bitmap_idx_missed);

    unsigned int prev_dist = atomicAdd(&g_events_bitmap[bitmap_idx_covered], 1);
    // printf("thread %d atomic add prev_dist %x\n", INSTANCE_GLOBAL_IDX, prev_dist);
    // Fast check: if 24 bit lower bits of prev_dist is 0, then this is a new coverage
    if ((prev_dist & 0xFFFFFF) == 0) {
        // printf(
        //     "thread %d update_coverage_bitmap_with_distance covered branch pc_src %u pc_dst %u pc_missed %u, "
        //     "prev_dist %u\n",
        //     INSTANCE_GLOBAL_IDX, pc_src, pc_dst, pc_missed, prev_dist);
        last_covered_branch_id = afl_hash_covered + 1;  // +1 to avoid 0
    }
    uint32_t dist_compare = (255 - distance_bits) << 24;  // first 8 bits are distance bit of the branch
    // update missed branch
    prev_dist = atomicMax(&g_events_bitmap[bitmap_idx_missed], dist_compare);

    // printf("thread %d atomic max prev_dist %x %x\n", INSTANCE_GLOBAL_IDX, prev_dist, dist_compare);
    if (dist_compare > prev_dist) {
        // printf(
        //     "thread %d update_coverage_bitmap_with_distance missed branch pc_src %u pc_dst %u pc_missed %u, "
        //     "distance_bits %u\n",
        //     INSTANCE_GLOBAL_IDX, pc_src, pc_dst, pc_missed, distance_bits);
        last_missed_branch_id = bitmap_idx_missed + 1;  // +1 to avoid 0
        last_distance_bits = distance_bits;
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
    // printf("thread %d update_storage_coverage pc %u storage_slot %u is_write %u current_account_id %u\n",
    //        INSTANCE_GLOBAL_IDX, pc, storage_slot, is_write, current_account_id);
    uint32_t storage_hash_id =
        is_write ? pc ^ (storage_slot + BITMAP_SIZE / 2) ^ current_account_id : pc ^ storage_slot ^ current_account_id;

    // uint32_t bitmap_idx = fnv1a(storage_id) % BITMAP_SIZE;
    uint32_t bitmap_idx = storage_hash_id % BITMAP_SIZE;
    unsigned int prev_dist = atomicMax(&g_events_bitmap[bitmap_idx], 1);

    // printf(
    //     "thread %d update_storage_coverage pc %u storage_slot %u account_idx %u is_write %u storage_id 0x%08x, "
    //     "prev_dist %u bitmap_idx %u\n",
    //     INSTANCE_GLOBAL_IDX, pc, storage_slot, account_idx, is_write, storage_id, prev_dist, bitmap_idx);

    if (prev_dist == 0) {
        // printf("thread %d new_storage_coverage pc %u storage_slot %u account_idx %u is_write %u storage_id 0x%08x\n",
        //        INSTANCE_GLOBAL_IDX, pc, storage_slot, account_idx, is_write, storage_id);

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
    // uint32_t bug_id = (pc << 16) | bug_type;  // Unique ID
    // printf("thread %d update_bugs pc %u bug_type %u bug_id 0x%08x\n", INSTANCE_GLOBAL_IDX, pc, bug_type, bug_id);
    // FNV-1a:
    uint32_t hash = fnv1a(bug_id);

    bool inserted = false;
    for (uint32_t probe = 0; probe < MAX_NEW_BUGS; ++probe) {
        uint32_t idx = (hash + (probe * probe)) % MAX_NEW_BUGS;
        // printf("thread %d update_bugs probe %u idx %u\n", INSTANCE_GLOBAL_IDX, probe, idx);

        unsigned int cas_result = atomicCAS(&g_total_bug_table[idx], 0, bug_id);
        if (cas_result == 0) {
            // Successful insert into empty slot
            // printf("thread %d update_bugs inserted\n", INSTANCE_GLOBAL_IDX);
            inserted = true;
            break;
        } else if (cas_result == bug_id) {
            // Another thread already inserted this exact bug
            // printf("thread %d update_bugs duplicate found\n", INSTANCE_GLOBAL_IDX);
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

__device__ void simplified_trace_data::finalize_coverage_bitmap() {
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

    if (no_bugs > 0) {
        // printf("thread %d finalize_coverage_bitmap no_bugs %d\n", INSTANCE_GLOBAL_IDX, no_bugs);
        // for (uint32_t i = 0; i < no_bugs; i++) {
        //     printf("bug %x state_written %d\n", bugs[i], state_written);
        // }
        if (state_written) {
            // printf("thread %d finalize_coverage_bitmap state_written %d\n", INSTANCE_GLOBAL_IDX, state_written);
            for (uint32_t i = 0; i < no_bugs; i++) {
                // printf("thread %d add bug %x state_written %d\n", INSTANCE_GLOBAL_IDX, bugs[i], state_written);
                uint8_t bug_type = static_cast<uint8_t>((bugs[i] >> 8) & 0xFF);

                if (bug_type == BUG_INTEGER_OVERFLOW || bug_type == BUG_INTEGER_UNDERFLOW) {
                    if (state_written) update_bugs(bugs[i]);
                }
            }
        }
    }

    // int bitmap_idx = INSTANCE_GLOBAL_IDX / 32;
    // int bit_pos = INSTANCE_GLOBAL_IDX % 32;
    // uint32_t mask = 1U << bit_pos;

    // // Check if this thread's bit is set in the new coverage bitmap
    // if (g_new_coverage_bitmap[bitmap_idx] & mask) {
    //     // Atomically increment the counter and get the previous value
    //     int idx = atomicAdd(&g_gpu_feedback_count->new_branch_count, 1);

    //     // If we haven't exceeded the maximum number of new branches to track
    //     if (idx < CuEVM::MAX_NEW_BRANCHES) {
    //         // Record this thread's global index in the coverage index array
    //         g_new_branch_info[idx].branch_thread_idx = INSTANCE_GLOBAL_IDX;
    //         g_new_branch_info[idx].branch_id = last_branch_id;
    //     }

    //     // printf("g_new_coverage_bitmap[idx] %d\n", g_new_coverage_bitmap[idx]);
    // }

    // printf("g_new_coverage_count %d\n", g_new_coverage_count[0]);
}

/*
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
*/

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
    // printf("record branch thread %d pc_src %u pc_dst %u pc_missed %u, distance_bits %d\n", INSTANCE_GLOBAL_IDX,
    // pc_src,
    //        pc_dst, pc_missed, distance_bits);
    // go library: branch info is recorded on CPU side
    update_coverage_bitmap_with_distance(pc_src, pc_dst, pc_missed, distance_bits);
#else
    branches[no_branches].pc_src = pc_src;
    branches[no_branches].pc_dst = pc_dst;
    branches[no_branches].pc_missed = pc_missed;
    branches[no_branches].distance = last_distance;
#endif

    no_branches++;
    return false;
}

__device__ void simplified_trace_data::record_distance(uint8_t op, const CuEVM::evm_stack_t& stack_ptr) {
    evm_word_t distance;
    evm_word_t* op1 = stack_ptr.get_address_at_index(1);
    evm_word_t* op2 = stack_ptr.get_address_at_index(2);
    uint32_t stack_size = stack_ptr.size();

    if (uint256_cmp(op1, op2) >= 1)
        uint256_sub(&distance, op1, op2);
    else
        uint256_sub(&distance, op2, op1);

    if (op != OP_EQ) uint256_add_word(&distance, &distance, 1);
    // printf("thread %d record_distance op %u distance %s\n", INSTANCE_GLOBAL_IDX, op, distance.to_hex());
    last_distance = distance;
}

/*
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
*/
__device__ int simplified_trace_data::start_call(uint32_t pc, evm_call_context_t* call_context_ptr) {
    assert(call_context_ptr != nullptr);
    // printf("thread %d start_call pc %u current_account_id %x\n", INSTANCE_GLOBAL_IDX, pc, current_account_id);
#ifdef BUILD_GO_LIBRARY

    no_branches += 20;  // call saturates the branch limit faster than normal jumps
    // state_written = true;
#endif
    if (no_calls >= MAX_CALLS_TRACING) return;
    // add address and increment current_address_idx
    // addresses[current_address_idx] = cached_call_state->addresses[cached_call_state->current_address_idx];
    // printf("start call simplified trace data pc %d no calls %d\n", pc, no_calls);
    calls[no_calls].sender_id = call_context_ptr->from.words[0] & 0xffff;
    current_account_id = call_context_ptr->to.words[0] & 0xffff;
    calls[no_calls].receiver_id = current_account_id;
    calls[no_calls].pc = pc;
    calls[no_calls].op = call_context_ptr->call_type;
    calls[no_calls].value = call_context_ptr->value;
    if (calls[no_calls].value.words[0] != 0) {
        state_written = true;  // transfer = true
    }
    calls[no_calls].error_code = RESERVED_ERROR_CODE;
    calls[no_calls].last_pc = 0;
    no_calls++;
    if (no_calls > MAX_RECURSION) {
        // printf("\nThread %d: No calls %d\n", INSTANCE_GLOBAL_IDX, no_calls);
        uint8_t loop_found = 0;
        // loop back 10 calls and check if we found loops
        for (int i = no_calls - 2; i >= no_calls - MAX_RECURSION; i--) {
            if (calls[i].receiver_id == calls[no_calls - 1].receiver_id) {
                loop_found++;
            }
        }
        // printf("Thread %d: Loop found %d\n", INSTANCE_GLOBAL_IDX, loop_found);

        if (loop_found >= MAX_RECURSION / 2) {
            // printf("\nThread %d: Reentrancy detected, loop_found %d\n", INSTANCE_GLOBAL_IDX, loop_found);
            return ERROR_REENTRANCY;
        }
    }

    return ERROR_SUCCESS;
}
__device__ void simplified_trace_data::start_create() {
    // printf("thread %d finish_create no_branches %d\n", INSTANCE_GLOBAL_IDX, no_branches);
    no_branches += MAX_BRANCHES_TRACING / 2;
    state_written = true;
}
__device__ void simplified_trace_data::finish_call(uint8_t error_code, uint32_t last_pc) {
    // printf("thread %d finish_call error_code %u last_pc %u depth %u\n", INSTANCE_GLOBAL_IDX, error_code, last_pc,
    //        depth);
#ifdef BUILD_GO_LIBRARY
    no_branches += 2;
#endif
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
    if (i > 0) {
        current_account_id = calls[i - 1].receiver_id;
    }

#ifdef BUILD_GO_LIBRARY

    if (error_code == ERROR_INVALID_OPCODE) {
        // update_coverage_bitmap(last_pc, 0, true);
        // printf("thread %d add invalid bug %u\n", INSTANCE_GLOBAL_IDX, last_pc << 16 | BUG_INVALID_OPCODE);
        update_bugs(last_pc << 16 | BUG_INVALID_OPCODE << 8 | (calls[0].receiver_id & 0xFF));
    }

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
    // printf("events\n");
    // for (uint32_t i = 0; i < no_events; i++) {
    //     printf("pc %u op %u operand_1 %s operand_2 %s res %s\n", events[i].pc, events[i].op,
    //            events[i].operand_1.to_hex(), events[i].operand_2.to_hex(), events[i].res.to_hex());
    // }
    printf("calls\n");
    for (uint32_t i = 0; i < no_calls; i++) {
        printf("pc %u op %u sender_id %u receiver_id %u value %s error_code %u\n", calls[i].pc, calls[i].op,
               calls[i].sender_id, calls[i].receiver_id, calls[i].value.to_hex(), calls[i].error_code);
    }
    printf("branches\n");
#ifndef BUILD_GO_LIBRARY
    for (uint32_t i = 0; i < no_branches; i++) {
        printf("pc_src %u pc_dst %u distance %s\n", branches[i].pc_src, branches[i].pc_dst,
               branches[i].distance.to_hex());
    }
#endif
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

// seed = (a * seed + c) % m;
// chances out of 100

#define CHANCE_TO_TAKE_INTEGER_FROM_CONSTANTS 10  // percent
#define CHANCE_TO_CREATE_NEW_INTEGER 2            // one in 2
#define CHANCE_TO_CREATE_NEW_ADDRESS 0            // 1 percent
#define CHANCE_TO_SKIP_MUTATE 50                  // percent we skip a marker
#define CHANCE_TO_SMALL_DELTA 5                   // percent we do small delta mutation
#define VALUE_MUTATE_INT32 7                      // 2**(7*4*8) = 2**224
// #define VALUE_CHANCE_TO_STOP_INT_32 30            // 30 percent.

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

__device__ unsigned int mutate_block_values_senders(unsigned int seed, uint64_t* block_numbers,
                                                    uint64_t* block_timestamps, uint8_t* senders) {
    // block number first
    seed = (A_LCG * seed + C_LCG) % M_LCG;
    uint32_t random_chance = seed % 100;
    if (random_chance <= CHANCE_TO_SKIP_MUTATE) {
        return seed;
    }
    seed = (A_LCG * seed + C_LCG) % M_LCG;
    uint32_t block_number = seed % g_fuzzing_constants->block_number_delay_max;

    seed = (A_LCG * seed + C_LCG) % M_LCG;
    uint32_t block_timestamp = seed % g_fuzzing_constants->block_timestamp_delay_max;

    if (block_timestamp == 0)
        block_number = 0;
    else
        block_number = block_number % block_timestamp;
    block_numbers[INSTANCE_GLOBAL_IDX] = block_number;
    block_timestamps[INSTANCE_GLOBAL_IDX] = block_timestamp;
    // mutate sender
    // need to skip again ?
    // seed = (A_LCG * seed + C_LCG) % M_LCG;
    // random_chance = seed % 100;
    // if (random_chance <= CHANCE_TO_SKIP_MUTATE) {
    //     return seed;
    // }
    seed = (A_LCG * seed + C_LCG) % M_LCG;
    senders[INSTANCE_GLOBAL_IDX] = seed % g_fuzzing_constants->sender_counts;
    // printf("thread %d block_number %u block_timestamp %u sender %u seed %u\n", INSTANCE_GLOBAL_IDX, block_number,
    //        block_timestamp, senders[INSTANCE_GLOBAL_IDX], seed);
    return seed;
    // blockNumberDelay %= blockTimestampDelay
}
__device__ unsigned int mutate_value(unsigned int seed, evm_word_t* value) {
    seed = (A_LCG * seed + C_LCG) % M_LCG;
    uint32_t random_chance = seed % 100;
    if (random_chance <= CHANCE_TO_SKIP_MUTATE) {
        return seed;
    }
    for (int i = 0; i < VALUE_MUTATE_INT32; i++) {
        seed = (A_LCG * seed + C_LCG) % M_LCG;
        value->words[i] = seed;
        // printf("thread %d value %s seed %u\n", INSTANCE_GLOBAL_IDX, value->to_hex(), seed);
        seed = (A_LCG * seed + C_LCG) % M_LCG;
        random_chance = seed % 100;
        // if (random_chance <= VALUE_CHANCE_TO_STOP_INT_32) {
        //     // printf("thread %d stopping at seed %u\n", INSTANCE_GLOBAL_IDX, seed);
        //     return seed;
        // }
    }
    return seed;
}
__device__ void mutate_transaction_data(CuEVM::transaction::TransactionList* transaction_list_ptr) {
    uint32_t seed = INSTANCE_GLOBAL_IDX + transaction_list_ptr->start_seed;

    uint32_t marker_idx = INSTANCE_GLOBAL_IDX / CUEVM_MUTATE_GROUP_SIZE;
    // printf("thread %d marker_idx %d , marker offset %d seed %u\n", INSTANCE_GLOBAL_IDX, marker_idx,
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

    if (marker_size == 0) return;
    // if (INSTANCE_GLOBAL_IDX % CUEVM_MUTATE_GROUP_SIZE != 0) return;  // skip the first sequence in each group

    uint8_t* call_data = &transaction_list_ptr->call_data[transaction_list_ptr->call_data_offset[INSTANCE_GLOBAL_IDX]];

    for (int j = 0; j < marker_size; j++) {
        uint32_t element_offset = *marker_data++;
        uint32_t element_type = *marker_data++;
        uint32_t element_length = *marker_data++;

        if (element_type == ELEMENT_VALUE_TYPE) {  // always mutate value
            // printf("thread %d value mutation\n", INSTANCE_GLOBAL_IDX);
            seed = mutate_value(seed, &transaction_list_ptr->value[INSTANCE_GLOBAL_IDX]);

            continue;
        }

        if (element_type == ELEMENT_BOOL_TYPE) {  // always mutate bool
            // printf("thread %d bool mutation\n", INSTANCE_GLOBAL_IDX);
            seed = (A_LCG * seed + C_LCG) % M_LCG;
            call_data[element_offset + 31] = seed % 2;  // set last byte to 0 or 1
            continue;
        }

        seed = (A_LCG * seed + C_LCG) % M_LCG;
        uint32_t random_chance = seed % 100;
        if (random_chance <= CHANCE_TO_SKIP_MUTATE) {
            if (element_type > 7) {
                // mutate the marker data
                seed = (A_LCG * seed + C_LCG) % M_LCG;
                random_chance = seed % 100;
                if (random_chance <= CHANCE_TO_SMALL_DELTA) {
                    // do small delta mutation
                    seed = mutate_byte_array(call_data + element_offset, element_length, 1, seed, false);
                }
            }
            continue;
        }
        // printf("thread %d marker_idx %d marker_offset %d element_offset %d element_type %d element_length
        // %d\n",
        //        INSTANCE_GLOBAL_IDX, marker_idx, marker_offset, element_offset, element_type, element_length);
        if (element_type > 7) {
            uint32_t byte_length = element_type / 8;
            // randomize the marker data
            seed = (A_LCG * seed + C_LCG) % M_LCG;
            bool create_new = (seed % CHANCE_TO_CREATE_NEW_INTEGER) == 0;
            seed = mutate_byte_array(call_data + element_offset, element_length, byte_length, seed, create_new);
        } else if (element_type == ELEMENT_ADDRESS_TYPE) {  // address
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
                // printf("thread %d seed %d address_constants_count %d random_address_index %d\n",
                // INSTANCE_GLOBAL_IDX,
                //        seed, address_constants_count, random_index);
                for (int i = 0; i < 20; i++) {
                    call_data[element_offset + 12 + i] =
                        g_fuzzing_constants->address_constants[random_index * 32 + 12 + i];
                }
            }
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

    // uint32_t new_offset =
    // (contract_index[address_index] * account_prealloc_keys_size + storage_size) * num_states +
    // INSTANCE_GLOBAL_IDX; uint32_t instance_idx = address_index * num_states + INSTANCE_GLOBAL_IDX;
    // // printf("get_value_status address_index %d, instance_idx %d instance %d\n", address_index,
    // instance_idx,
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