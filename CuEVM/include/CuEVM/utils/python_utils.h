#pragma once
#include <CuEVM/utils/library_utils.h>
#include <Python.h>


std::unordered_set<int> const bug_opcodes = {OP_ADD, OP_MUL, OP_SUB, OP_MOD, OP_EXP, OP_SELFDESTRUCT, OP_ORIGIN};
std::unordered_set<int> const call_opcodes = {OP_CALL, OP_CALLCODE, OP_DELEGATECALL};  // ignore static call for now
std::unordered_set<int> const comparison_opcodes = {OP_LT, OP_GT, OP_SLT, OP_SGT, OP_EQ};
std::unordered_set<int> const revert_opcodes = {OP_REVERT, OP_INVALID};

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

void get_block_info_from_PyObject(PyObject* data);

void print_dict_recursive(PyObject* dict, int indent_level = 2);

CuEVM::transaction::TransactionList* getTransactionDataFromListofPyObject(PyObject* read_roots);

void getStateDataFromPyObject(PyObject* data, uint32_t num_states);

void getPreStateDataFromTransactionList(PyObject* readroot, uint32_t num_states);

CuEVM::transaction::TransactionList* get_evm_instances_from_PyObject(PyObject* read_roots, uint32_t& num_instances,
                                                                     bool reuse_state_data = false,
                                                                     bool copy_state_data = true,
                                                                     uint32_t call_counter = 0);

// OP_SSTORE
// OP_JUMPI
// OP_SELFDESTRUCT

PyObject* pyobject_from_serialized_state(CuEVM::serialized_worldstate_data* serialized_worldstate_instance);

/**
 * Get the pyobject from the evm instances after the transaction execution.
 * @param[in] instances evm instances
 * @return pyobject
 */
PyObject* pyobject_from_evm_instances(uint32_t num_instances, bool copy_state_data = false);

/*
 * Convert a Python int (assumed to be non-negative and fitting in 256 bits)
 * into a native uint256 value.
 */
int py_long_to_uint256(PyObject* py_num, uint256* dst);

/*
 * Convert a native uint256 value to a Python int.
 */
PyObject* uint256_to_py_long(const uint256* src);

}  // namespace python_utils
