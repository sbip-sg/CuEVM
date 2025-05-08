#pragma once
#include <CuEVM/utils/library_utils.h>
#include <Python.h>

/**
 * @file python_utils.h
 * @brief Utility functions for Python integration with CuEVM.
 *
 * This file provides functions and utilities for interfacing between Python and CuEVM,
 * including conversion between Python objects and CuEVM data structures, handling
 * transaction data, and state serialization.
 */

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

/**
 * @brief Recursively copy a Python dictionary.
 * @param[in] read_root Source Python dictionary to copy from.
 * @param[out] write_root Destination Python dictionary to copy to.
 */
void copy_dict_recursive(PyObject* read_root, PyObject* write_root);

/**
 * @brief Print a Python dictionary.
 * @param[in] self The module self reference.
 * @param[in] args Arguments containing the dictionary to print.
 * @return PyObject* A new Python reference.
 */
static PyObject* print_dict(PyObject* self, PyObject* args);

/**
 * @namespace python_utils
 * @brief Namespace containing utilities for Python integration with CuEVM.
 *
 * This namespace provides functions for converting between Python objects and
 * CuEVM data structures, manipulating transaction data, and handling state serialization.
 */
namespace python_utils {

/**
 * @brief Extract block information from a Python object.
 * @param[in] data Python object containing block information.
 */
void get_block_info_from_PyObject(PyObject* data);

/**
 * @brief Recursively print a Python dictionary with indentation.
 * @param[in] dict The Python dictionary to print.
 * @param[in] indent_level The indentation level for nested structures.
 */
void print_dict_recursive(PyObject* dict, int indent_level = 2);

/**
 * @brief Convert a list of Python objects to a CuEVM transaction list.
 * @param[in] read_roots List of Python objects representing transactions.
 * @return CuEVM::transaction::TransactionList* Pointer to the created transaction list.
 */
CuEVM::transaction::TransactionList* getTransactionDataFromListofPyObject(PyObject* read_roots);

/**
 * @brief Extract state data from a Python object.
 * @param[in] data Python object containing state data.
 * @param[in] num_states Number of states to extract.
 */
void getStateDataFromPyObject(PyObject* data, uint32_t num_states);

/**
 * @brief Extract pre-state data from a transaction list Python object.
 * @param[in] readroot Python object containing transaction list with pre-state data.
 * @param[in] num_states Number of states to extract.
 */
void getPreStateDataFromTransactionList(PyObject* readroot, uint32_t num_states);

/**
 * @brief Create EVM instances from a Python object.
 * @param[in] read_roots Python object containing EVM instance data.
 * @param[out] num_instances Number of EVM instances created.
 * @param[in] reuse_state_data Flag to reuse existing state data.
 * @param[in] copy_state_data Flag to copy state data.
 * @param[in] call_counter Counter for call operations.
 * @return CuEVM::transaction::TransactionList* Pointer to the created transaction list.
 */
CuEVM::transaction::TransactionList* get_evm_instances_from_PyObject(PyObject* read_roots, uint32_t& num_instances,
                                                                     bool reuse_state_data = false,
                                                                     bool copy_state_data = true,
                                                                     uint32_t call_counter = 0);

// OP_SSTORE
// OP_JUMPI
// OP_SELFDESTRUCT

/**
 * @brief Create a Python object from serialized state data.
 * @param[in] serialized_worldstate_instance Pointer to serialized world state data.
 * @return PyObject* Python dictionary representing the serialized state.
 */
PyObject* pyobject_from_serialized_state(CuEVM::serialized_worldstate_data* serialized_worldstate_instance);

/**
 * @brief Create a Python object from EVM instances after transaction execution.
 * @param[in] num_instances Number of EVM instances.
 * @param[in] copy_state_data Flag to copy state data.
 * @return PyObject* Python object representing the EVM instances after execution.
 */
PyObject* pyobject_from_evm_instances(uint32_t num_instances, bool copy_state_data = false);

/**
 * @brief Convert a Python integer to a uint256 value.
 * @param[in] py_num Python integer object to convert.
 * @param[out] dst Destination uint256 value.
 * @return int Error code (0 for success, non-zero for failure).
 */
int py_long_to_uint256(PyObject* py_num, uint256* dst);

/**
 * @brief Convert a uint256 value to a Python integer.
 * @param[in] src Source uint256 value.
 * @return PyObject* Python integer object representing the uint256 value.
 */
PyObject* uint256_to_py_long(const uint256* src);

}  // namespace python_utils
