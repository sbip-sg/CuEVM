#pragma once

#include <Python.h>

#include <CuEVM/evm.cuh>
#include <CuEVM/utils/opcodes.cuh>
#include <unordered_set>

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

CuEVM::block_info_t* getBlockDataFromPyObject(PyObject* data);

void print_dict_recursive(PyObject* dict, int indent_level);

CuEVM::evm_transaction_t* getTransactionDataFromListofPyObject(PyObject* read_roots);

CuEVM::state_t* getStateDataFromPyObject(PyObject* data);
void get_evm_instances_from_PyObject(CuEVM::evm_instance_t*& evm_instances, PyObject* read_roots,
                                     uint32_t& num_instances);

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
PyObject* pyobject_from_evm_instances(CuEVM::evm_instance_t* instances, uint32_t num_instances);
}  // namespace python_utils
