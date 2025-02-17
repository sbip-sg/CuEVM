#pragma once

#include <Python.h>

#include <CuEVM/core/block_info.cuh>
#include <CuEVM/core/data_structures.cuh>
#include <CuEVM/core/evm_word.cuh>
#include <CuEVM/core/memory_pool.cuh>
#include <CuEVM/utils/evm_defines.cuh>
#include <CuEVM/utils/opcodes.cuh>
#include <unordered_set>

// for convenient data transfer between host and device. set a fixed maximum size for the number of addresses to be
// transferred
// todo : optimize this later
namespace CuEVM {

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
    void print();
};

#define MAX_TRACE_EVENTS 1024
#define MAX_ADDRESSES_TRACING 32
#define MAX_CALLS_TRACING 32
#define MAX_BRANCHES_TRACING 32  // only track the latest 32 branches
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
struct call_trace {
    uint32_t pc;
    uint8_t op;
    // uint8_t address_idx;
    evm_word_t sender;
    evm_word_t receiver;
    evm_word_t value;
    uint8_t success = UINT8_MAX;  // 0 or 1
    // todo add more depth + result etc
};
struct branch_trace {
    uint32_t pc_src;
    uint32_t pc_dst;
    uint32_t pc_missed;
    evm_word_t distance;  // distance between pc_src and pc_dst

    // todo: use evm_word_t for distance
};

struct simplified_trace_data {
    simple_event_trace events[MAX_TRACE_EVENTS];
    // evm_word_t addresses[MAX_ADDRESSES_TRACING];
    call_trace calls[MAX_CALLS_TRACING];
    branch_trace branches[MAX_BRANCHES_TRACING];  // pc_src jump to pc_dest
    uint32_t no_addresses = 0;
    // uint32_t current_address_idx = 0;
    uint32_t no_events = 0;
    uint32_t no_calls = 0;
    uint32_t no_branches = 0;
    evm_word_t last_distance;  // use to track branch distance by comparison opcodes

    __device__ void start_operation(const uint32_t pc, const uint8_t op, const CuEVM::evm_stack_t& stack_ptr);
    __device__ void finish_operation(const CuEVM::evm_stack_t& stack_ptr, uint32_t error_code);
    __device__ void start_call(uint32_t pc, evm_call_context_t* call_context_ptr);
    __device__ void finish_call(uint8_t success);
    __device__ void record_branch(uint32_t pc_src, uint32_t pc_dst, uint32_t pc_missed);
    __device__ void record_distance(uint8_t op, const CuEVM::evm_stack_t& stack_ptr);
    __device__ void print();
};

extern __device__ serialized_worldstate_data* global_serialized_worldstate;
extern __device__ simplified_trace_data* global_simplified_trace;
}  // namespace CuEVM

#define GET_STR_FROM_DICT_WITH_DEFAULT(dict, key, default_value) \
    (PyDict_GetItemString(dict, key) ? PyUnicode_AsUTF8(PyDict_GetItemString(dict, key)) : default_value)
namespace DefaultBlock {
constexpr char BaseFee[] = "0x0a";
constexpr char CoinBase[] = "0x2adc25665018aa1fe0e6bc666dac8fc2697ff9ba";
constexpr char Difficulty[] = "0x020000";
constexpr char BlockNumber[] = "0x01";
constexpr char GasLimit[] = "0x05f5e100";
constexpr char TimeStamp[] = "0x03e8";
constexpr char PreviousHash[] = "0x5e20a0453cecd065ea59c37ac63e079ee08998b6045136a8ce6635c7912ec0b6";
}  // namespace DefaultBlock

void copy_dict_recursive(PyObject* read_root, PyObject* write_root);
static PyObject* print_dict(PyObject* self, PyObject* args);

namespace python_utils {

__device__ void serialize_state_data(CuEVM::serialized_worldstate_data* data);

void get_block_info_from_PyObject(PyObject* data);

void print_dict_recursive(PyObject* dict, int indent_level = 2);

CuEVM::transaction::TransactionList* getTransactionDataFromListofPyObject(PyObject* read_roots);

void getStateDataFromPyObject(PyObject* data, uint32_t num_states);

void getPreStateDataFromTransactionList(PyObject* readroot, uint32_t num_states);

CuEVM::transaction::TransactionList* get_evm_instances_from_PyObject(PyObject* read_roots, uint32_t& num_instances);

std::unordered_set<int> const bug_opcodes = {OP_ADD, OP_MUL, OP_SUB, OP_MOD, OP_EXP, OP_SELFDESTRUCT, OP_ORIGIN};
std::unordered_set<int> const call_opcodes = {OP_CALL, OP_CALLCODE, OP_DELEGATECALL};  // ignore static call for now
std::unordered_set<int> const comparison_opcodes = {OP_LT, OP_GT, OP_SLT, OP_SGT, OP_EQ};
std::unordered_set<int> const revert_opcodes = {OP_REVERT, OP_INVALID};
// OP_SSTORE
// OP_JUMPI
// OP_SELFDESTRUCT

PyObject* pyobject_from_serialized_state(CuEVM::serialized_worldstate_data* serialized_worldstate_instance);

/**
 * Get the pyobject from the evm instances after the transaction execution.
 * @param[in] instances evm instances
 * @return pyobject
 */
PyObject* pyobject_from_evm_instances(uint32_t num_instances);
}  // namespace python_utils
