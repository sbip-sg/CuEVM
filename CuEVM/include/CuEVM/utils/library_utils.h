#pragma once

#include <CuEVM/core/block_info.cuh>
#include <CuEVM/core/data_structures.cuh>
#include <CuEVM/core/evm_word.cuh>
#include <CuEVM/core/memory_pool.cuh>
#include <CuEVM/utils/error_codes.cuh>
#include <CuEVM/utils/evm_defines.cuh>
#include <CuEVM/utils/opcodes.cuh>
#include <unordered_set>

#define CUEVM_MUTATE_GROUP_SIZE 32  // same as fuzzer SkipSequenceSize
/**
 * @file library_utils.h
 * @brief Utility functions and data structures for library integration.
 *
 * This file contains structures and utilities for data transfer between host and device,
 * execution tracing, and state serialization for the CuEVM library.
 */
typedef struct {
    uint32_t bug_thread_idx;  // Thread index
    uint32_t bug_id;          // id of the bug (pc << 16 | bug_type) // to be decoded
} BugInfoEntry;

typedef struct {
    uint32_t branch_thread_idx;  // Thread index
    uint32_t branch_id;          // Branch ID (pc_src << 16 | pc_dst) // to be decoded
} BranchInfoEntry;

typedef struct {
    uint32_t storage_thread_idx;  // Thread index
    uint32_t
        storage_id;  // Storage ID (account_idx (8bit) | storate_type (8bit) | storage_slot (16bit)) // to be decoded
    // storage_type: 1 for write, 2 for read, 3 for balance
} StorageInfoEntry;

typedef struct {
    uint32_t new_branch_count;
    uint32_t new_bug_count;
    uint32_t new_storage_count;
} GPUFeedbackCount;

// for convenient data transfer between host and device. set a fixed maximum size for the number of addresses to be
// transferred
// todo : optimize this later
namespace CuEVM {
using CuEVM::transaction::TransactionList;

// #ifdef BUILD_GO_LIBRARY
// Constants
// constexpr CONSTANT uint32_t BITMAP_SIZE_IN_BITS = 262144;                    // 2^18 bits, ~32 KB
// constexpr CONSTANT uint32_t BITMAP_SIZE_IN_INTS = BITMAP_SIZE_IN_BITS / 32;  // 8192 unsigned ints
constexpr CONSTANT uint32_t BITMAP_SIZE = 65536;     // AFL size 64KB
constexpr CONSTANT uint32_t MAX_NEW_BRANCHES = 512;  // per kenel launch
constexpr CONSTANT uint32_t MAX_NEW_STORAGE = 64;    // per kenel launch
constexpr CONSTANT uint32_t MAX_NEW_BUGS = 128;
constexpr CONSTANT uint32_t MAX_NEW_MEMORY = 32768;       // per instance
constexpr CONSTANT uint32_t MAX_RETURN_DATA_SIZE = 4096;  // per call

// persistent state across kernel launches
extern __device__ uint32_t* g_events_bitmap;
extern __device__ uint32_t* g_total_bug_table;
extern __device__ uint32_t* g_total_bug_count;

// event trackers, reset every kernel launch
// extern __device__ uint32_t* g_new_coverage_bitmap;  // for each thread to set a flag if they encounter a new branch
// or a
extern __device__ BranchInfoEntry* g_new_branch_info;
extern __device__ StorageInfoEntry* g_new_storage_info;
extern __device__ BugInfoEntry* g_new_bug_info;
extern __device__ GPUFeedbackCount* g_gpu_feedback_count;  // counter for interesting events, reset every kernel launch

// // bug (interesting)
// extern __device__ uint32_t* g_new_coverage_idx;
// extern __device__ uint32_t* g_new_coverage_count;
// // for bug detection tracking
// extern __device__ uint32_t* g_new_bug_idx;
// extern __device__ uint32_t* g_new_bug_pc;
// extern __device__ uint32_t* g_new_bug_count;

#define ELEMENT_ADDRESS_TYPE 1
#define ELEMENT_VALUE_TYPE 2
#define ELEMENT_BOOL_TYPE 3

// bug types
#define BUG_INTEGER_BUG 0x01
#define BUG_SELF_DESTRUCT 0x02
#define BUG_LEAKING_ETHER 0x03
#define BUG_ARBITRARY_CALL 0x04
#define BUG_REENTRANCY 0x05
#define BUG_INVALID_OPCODE 0xFF

// special attacker address for oracles (last 32 bit)
#define REENTRANCY_ATTACKER_ADDRESS 0xCAFECAFE
#define RANDOM_ATTACKER_ADDRESS 0xC0DE0001
// #define RANDOM_ATTACKER_ADDRESS 0xC0DE0002  // never appears in address dict

#define RETURN_BUFFER_SIZE 128  // return buffer of reentrancy attacker.
struct fuzzing_constants {
    uint8_t* address_constants;
    uint32_t address_constants_count;  // number of address constants
    // evm_word_t* address_list;          // mirroring address constant but with evm_word_t type
    uint8_t* integer_constants;
    uint32_t integer_constants_count;  // number of uint256 constants
    uint32_t block_number_delay_max = 60480 * 2;
    uint32_t block_timestamp_delay_max = 604800 * 4;  // 1 month
    evm_word_t* sender_list;                          // sender list for fuzzing
    uint32_t sender_counts = 3;
    uint8_t* return_buffer;  // for return data RETURN_BUFFER_SIZE
    __host__ __device__ void print();
};
// for fuzzing utilities
extern __device__ fuzzing_constants* g_fuzzing_constants;
extern __device__ uint32_t* g_static_marker_data;
// Max new branches to record per execution
// #endif
/**
 * @brief Structure for serialized world state data transfer between host and device.
 *
 * Contains account data including addresses, balances, nonces, and storage elements
 * with fixed maximum sizes for efficient transfer.
 */
struct serialized_worldstate_data {
    uint32_t no_accounts;
    uint32_t no_storage_elements;
    evm_word_t addresses[serialized_worldstate_addresses_size];  // 0x + ... + \0
    evm_word_t balance[serialized_worldstate_addresses_size];    // 0x + ... + \0
    uint32_t nonce[serialized_worldstate_addresses_size];
    uint16_t storage_indexes[serialized_worldstate_storage_slots];
    evm_word_t storage_keys[serialized_worldstate_storage_slots];
    evm_word_t storage_values[serialized_worldstate_storage_slots];
    // currently dont support copy back the bytecode hex string
    // TODO: use 1 large preallocated buffer for bytecode

    /**
     * @brief Print the serialized world state data.
     */
    void print();
};

#define START_CALL_BRANCH_MARKER 0xFF000000
#define END_CALL_BRANCH_MARKER 0xFE000000

#define MAX_TRACE_EVENTS 512
#define MAX_ADDRESSES_TRACING 16
#define MAX_CALLS_TRACING 32
#define MAX_BRANCHES_TRACING 128  // only track this number of branches in one trace
#define MAX_BUGS_TRACING 32       // only track this number of bugs in one tx
// In fuzzing mode if gas exceed this value, considered DOS / out of gas flag raised
#define MAX_GAS_FUZZING 1000000
#define MAX_FUZZING_LOOP_LIMIT 200

/**
 * @brief Structure for tracing simple EVM events.
 *
 * Records program counter, operation, operands, and result for each traced event.
 */
struct simple_event_trace {
    // pc // op //  operand 1, operand 2, res
    uint32_t pc;
    uint8_t op;
    // uint8_t address_idx;
    // evm_word_t address; // temporarily disabled
    evm_word_t operand_1;
    evm_word_t operand_2;
    evm_word_t res;  // blank in some cases
};

/**
 * @brief Structure for tracing call operations.
 *
 * Records details of call operations including sender, receiver, value, and result.
 */
struct call_trace {
    uint32_t pc;
    uint8_t op;
    uint32_t sender_id;    // unique identifer, last 8bit of address
    uint32_t receiver_id;  // unique identifer, last 8bit of address
    evm_word_t value;
    uint32_t call_data_size;
    uint8_t error_code = RESERVED_ERROR_CODE;  // 0 or 1
    uint32_t last_pc;                          // the last pc of the call before returning
    // todo add more depth + result etc
};

/**
 * @brief Structure for tracing branching operations.
 *
 * Records source and destination program counters, missed branches, and distance metrics.
 */
struct branch_trace {
#ifndef BUILD_GO_LIBRARY
    uint32_t pc_src;
    uint32_t pc_dst;
    uint32_t pc_missed;
    evm_word_t distance;  // distance between pc_src and pc_dst
#else
    // Jul 1 save memory for go library
    // todo: use evm_word_t for distance
#endif
};

// struct return_data {
//     uint8_t data_length;
//     uint8_t data[32];  // dont support return data copy yet.
// };

/**
 * @brief Structure for simplified execution tracing.
 *
 * Contains arrays of events, calls, and branches for execution tracing,
 * with methods to record various execution events.
 */
struct simplified_trace_data {
    // simple_event_trace events[MAX_TRACE_EVENTS];
    // evm_word_t addresses[MAX_ADDRESSES_TRACING];

    call_trace calls[MAX_CALLS_TRACING];

    uint32_t no_calls = 0;
    uint32_t no_branches = 0;
    evm_word_t last_distance;         // use to track branch distance by comparison opcodes
    uint32_t last_covered_branch_id;  // use to track last branch id that has improved distance
    uint32_t last_missed_branch_id;   // use to track last branch id that has improved distance
    uint8_t last_distance_bits;       // use to track last distance bits
    uint8_t state_written = false;
    uint32_t current_account_id = 0;
    uint32_t no_bugs = 0;
    uint8_t reentrancy_count = 0;
    uint32_t bugs[MAX_BUGS_TRACING];

    /**
     * @brief Check if coverage exists.
     * @return True if coverage exists, false otherwise.
     */
    // __device__ void update_coverage_bitmap(uint32_t pc_src, uint32_t pc_dst, bool is_bug = false);

    /**
     * @brief Update the coverage bitmap with the distance between pc_src and pc_dst.
     * @param[in] pc_src The source program counter.
     * @param[in] pc_dst The destination program counter.
     * @param[in] distance_bits The distance in number of bits before satisfying the jump.
     */
    __device__ void update_coverage_bitmap_with_distance(uint32_t pc_src, uint32_t pc_dst, uint32_t pc_missed,
                                                         uint8_t distance_bits);

    /**
     * @brief Update the storage coverage bitmap.
     * @param[in] pc The program counter.
     * @param[in] storage_slot The storage slot.
     * @param[in] account_idx The account index.
     * @param[in] is_write The write flag.
     */
    __device__ void update_storage_coverage(uint32_t pc, uint16_t storage_slot, uint8_t is_write);

    /**
     * @brief Update the bug table with the bug type.
     * @param[in] pc The program counter.
     * @param[in] bug_type The bug type
     */
    __device__ void update_bugs(uint32_t bug_id);

    /**
     * @brief Add bugs for later.
     * @param[in] pc The program counter.
     * @param[in] bug_type The bug type
     */
    __device__ void add_bugs_for_later(uint32_t pc, uint8_t bug_type);

    /**
     * @brief Add selfdestruct oracle.
     * @param[in] pc The program counter.
     */
    __device__ void selfdestruct_oracle(uint32_t pc);

    /**
     * @brief Add reentrancy oracle.
     * @param[in] pc The program counter.
     */
    __device__ void reentrancy_oracle(uint32_t pc);

    /**
     * @brief Add invalid opcode oracle.
     * @param[in] pc The program counter.
     */
    __device__ void invalid_opcode_oracle(uint32_t pc);

    /**
     * @brief Add leaking ether oracle.
     * @param[in] pc The program counter.
     */
    __device__ void leaking_ether_oracle(uint32_t pc);

    /**
     * @brief Add arbitrary call oracle.
     * @param[in] pc The program counter.
     */
    __device__ void arbitrary_call_oracle(uint32_t pc);

    /**
     * @brief Begin recording an operation in the trace.
     * @param[in] pc The program counter.
     * @param[in] op The operation code.
     * @param[in] stack_ptr The stack pointer.
     */
    __device__ void start_operation(const uint32_t pc, const uint8_t op, const CuEVM::evm_stack_t& stack_ptr);

    /**
     * @brief Complete recording an operation in the trace.
     * @param[in] stack_ptr The stack pointer.
     * @param[in] error_code The error code.
     */
    __device__ void finish_operation(const CuEVM::evm_stack_t& stack_ptr, uint32_t error_code);

    /**
     * @brief Record a simple operation in the trace (no need to record stack content).
     * @param[in] pc The program counter.
     * @param[in] op The operation code.
     */
    __device__ void record_operation(const uint32_t pc, const uint8_t op);

    /**
     * @brief Start recording a call operation.
     * @param[in] pc The program counter.
     * @param[in] call_context_ptr The call context pointer.
     * @return The error code.
     */
    __device__ void start_call(uint32_t pc, evm_call_context_t* call_context_ptr);

    __device__ void start_create();

    /**
     * @brief Complete recording a call operation.
     * @param[in] success The success flag.
     * @param[in] last_pc The last program counter.
     */
    __device__ void finish_call(uint8_t success, uint32_t last_pc, uint32_t _current_account_id);

    /**
     * @brief Record a branch operation.
     * @param[in] pc_src The source program counter.
     * @param[in] pc_dst The destination program counter.
     * @param[in] pc_missed The missed program counter.
     */
    __device__ bool record_branch(uint32_t pc_src, uint32_t pc_dst, uint32_t pc_missed);

    /**
     * @brief Increase the branch count.
     */
    __device__ bool increase_branch_count();

    /**
     * @brief Record the distance metric for a branch operation.
     * @param[in] op The operation code.
     * @param[in] stack_ptr The stack pointer.
     */
    __device__ void record_distance(uint8_t op, const CuEVM::evm_stack_t& stack_ptr);

    /**
     * @brief Print the simplified trace data.
     */
    __device__ void print();

    /**
     * @brief Finalize the coverage bitmap.
     */
    __device__ void finalize_coverage_bitmap(int32_t error_code);
};

/**
 * @brief Global serialized world state data.
 */
extern __device__ serialized_worldstate_data* global_serialized_worldstate;

/**
 * @brief Global simplified trace data.
 */
extern __device__ simplified_trace_data* global_simplified_trace;
/**
 * @brief Serialize state data from world state to the given data structure.
 * @param[out] data The data structure to serialize into.
 */
__device__ void serialize_state_data(CuEVM::serialized_worldstate_data* data);

#ifdef BUILD_GO_LIBRARY
__device__ void mutate_transaction_data(CuEVM::transaction::TransactionList* transaction_list_ptr);
#endif
/**
 * @brief Free transaction list resources.
 * @param[in] d_transaction_list_ptr Pointer to the transaction list.
 */
void freeTransactionList(CuEVM::transaction::TransactionList* d_transaction_list_ptr);

/**
 * @brief Free trace data resources.
 * @param[in] copy_state_data Flag to indicate whether to copy state data.
 */
void freeTraceData(bool copy_state_data);
}  // namespace CuEVM

/**
 * @brief Macro for getting string from python dictionary with default value.
 * @param dict The dictionary.
 * @param key The key to look up.
 * @param default_value The default value if key is not found.
 */
#define GET_STR_FROM_DICT_WITH_DEFAULT(dict, key, default_value) \
    (PyDict_GetItemString(dict, key) ? PyUnicode_AsUTF8(PyDict_GetItemString(dict, key)) : default_value)

/**
 * @brief Namespace containing default block values.
 *
 * Provides default values for block parameters used when specific values
 * are not provided by the caller.
 */
namespace DefaultBlock {
constexpr char BaseFee[] = "0x0a";
constexpr char CoinBase[] = "0x2adc25665018aa1fe0e6bc666dac8fc2697ff9ba";
constexpr char Difficulty[] = "0x020000";
constexpr char BlockNumber[] = "0x01";
constexpr char GasLimit[] = "0x05f5e100";
constexpr char TimeStamp[] = "0x03e8";
constexpr char PreviousHash[] = "0x5e20a0453cecd065ea59c37ac63e079ee08998b6045136a8ce6635c7912ec0b6";
}  // namespace DefaultBlock
