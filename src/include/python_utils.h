#pragma once
#ifndef IGNORELIB
#include <Python.h>
#include <unordered_set>

#include "../evm.cuh"

#define GET_STR_FROM_DICT_WITH_DEFAULT(dict, key, default_value) (PyDict_GetItemString(dict, key) ? PyUnicode_AsUTF8(PyDict_GetItemString(dict, key)) : default_value)
namespace DefaultBlock {
    constexpr char BaseFee[] = "0x0a";
    constexpr char CoinBase[] = "0x2adc25665018aa1fe0e6bc666dac8fc2697ff9ba";
    constexpr char Difficulty[] = "0x020000";
    constexpr char BlockNumber[] = "0x01";
    constexpr char GasLimit[] = "0x05f5e100";
    constexpr char TimeStamp[] = "0x03e8";
    constexpr char PreviousHash[] = "0x5e20a0453cecd065ea59c37ac63e079ee08998b6045136a8ce6635c7912ec0b6";
}

void copy_dict_recursive(PyObject *read_root, PyObject *write_root);
static PyObject* print_dict(PyObject* self, PyObject* args);
using block_data_t = block_t::block_data_t;
using state_data_t = world_state_t::state_data_t;
using transaction_data_t = transaction_t::transaction_data_t;
using account_t = world_state_t::account_t;
using tracer_data_t = tracer_t::tracer_data_t;
using evm_instances_t = evm_t::evm_instances_t;
using contract_storage_t = world_state_t::contract_storage_t;
using touch_state_data_t = touch_state_t::touch_state_data_t ;


namespace python_utils{

    block_data_t* getBlockDataFromPyObject(arith_t &arith, PyObject* data){
        // construct blockt_t

        block_data_t* block_data;
        #ifndef ONLY_CPU
        CUDA_CHECK(cudaMallocManaged(
            (void **)&(block_data),
            sizeof(block_data_t)
        ));
        #else
        block_data = new block_data_t;
        #endif

        const char* base_fee = GET_STR_FROM_DICT_WITH_DEFAULT(data, "currentBaseFee", DefaultBlock::BaseFee);
        const char* coin_base = GET_STR_FROM_DICT_WITH_DEFAULT(data, "currentCoinbase", DefaultBlock::CoinBase);
        const char* difficulty = GET_STR_FROM_DICT_WITH_DEFAULT(data, "currentDifficulty", DefaultBlock::Difficulty);
        const char* block_number = GET_STR_FROM_DICT_WITH_DEFAULT(data, "currentNumber", DefaultBlock::BlockNumber);
        const char* gas_limit = GET_STR_FROM_DICT_WITH_DEFAULT(data, "currentGasLimit", DefaultBlock::GasLimit);
        const char* time_stamp = GET_STR_FROM_DICT_WITH_DEFAULT(data, "currentTimestamp", DefaultBlock::TimeStamp);
        const char* previous_hash = GET_STR_FROM_DICT_WITH_DEFAULT(data, "previousHash", DefaultBlock::PreviousHash);
        arith.cgbn_memory_from_hex_string(block_data->coin_base, coin_base);
        arith.cgbn_memory_from_hex_string(block_data->difficulty, difficulty);
        arith.cgbn_memory_from_hex_string(block_data->number, block_number);
        arith.cgbn_memory_from_hex_string(block_data->gas_limit, gas_limit);
        arith.cgbn_memory_from_hex_string(block_data->time_stamp, time_stamp);
        arith.cgbn_memory_from_hex_string(block_data->base_fee, base_fee);
        arith.cgbn_memory_from_hex_string(block_data->chain_id, "0x01");
        arith.cgbn_memory_from_hex_string(block_data->previous_blocks[0].hash, previous_hash);

        return block_data;
    }


    void print_dict_recursive(PyObject* dict, int indent_level) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;

        // Iterate over dictionary items
        while (PyDict_Next(dict, &pos, &key, &value)) {
            // Print indent
            for (int i = 0; i < indent_level; ++i) {
                printf("    "); // 4 spaces for each indent level
            }

            // Convert key to string and print
            PyObject* keyStrObj = PyObject_Str(key);
            const char* keyStr = PyUnicode_AsUTF8(keyStrObj);
            printf("%s: ", keyStr);
            Py_DECREF(keyStrObj);

            // Check if value is a dictionary
            if (PyDict_Check(value)) {
                printf("\n");
                print_dict_recursive(value, indent_level + 1);  // Recursively print nested dictionary
            } else if (PyList_Check(value)) {
                // Handle list of items
                printf("[\n");
                for (Py_ssize_t i = 0; i < PyList_Size(value); ++i) {
                    PyObject* item = PyList_GetItem(value, i);  // Borrowed reference, no need to DECREF
                    // Print list item with additional indent
                    for (int j = 0; j <= indent_level; ++j) {
                        printf("    ");
                    }
                    PyObject* itemStrObj = PyObject_Str(item);
                    const char* itemStr = PyUnicode_AsUTF8(itemStrObj);
                    printf("%s\n", itemStr);
                    Py_DECREF(itemStrObj);
                }
                // Print closing bracket with indent
                for (int i = 0; i < indent_level; ++i) {
                    printf("    ");
                }
                printf("]\n");
            } else {
                // For other types, convert to string and print
                PyObject* valueStrObj = PyObject_Str(value);
                const char* valueStr = PyUnicode_AsUTF8(valueStrObj);
                printf("%s\n", valueStr);
                Py_DECREF(valueStrObj);
            }
        }
    }

    transaction_data_t* getTransactionDataFromListofPyObject(arith_t &arith, PyObject* read_roots){
        Py_ssize_t count = PyList_Size(read_roots);
        transaction_data_t *transactions;
        #ifndef ONLY_CPU
            CUDA_CHECK(cudaMallocManaged((void **)&(transactions), count * sizeof(transaction_data_t)));
        #else
            transactions = new transaction_data_t[count];
        #endif

        for (Py_ssize_t idx = 0; idx < count; idx++) {

            PyObject *read_root = PyList_GetItem(read_roots, idx);
            if (!PyDict_Check(read_root)) {
                PyErr_SetString(PyExc_TypeError, "Each item in the list must be a dictionary.");
                return NULL;
            }

            PyObject* data = PyDict_GetItemString(read_root, "transaction");
            // get data size
            PyObject* tx_data =  PyDict_GetItemString(data, "data");
            PyObject* tx_gas_limit = PyDict_GetItemString(data, "gasLimit");
            PyObject* tx_value = PyDict_GetItemString(data, "value");
            if (tx_data == NULL || !PyList_Check(tx_data)){
                printf("Invalid transaction data\n");
                return NULL;
            }

            printf("Transaction count: %d\n", count);

            transaction_data_t *template_transaction = new transaction_data_t;
            memset(template_transaction, 0, sizeof(transaction_data_t));

            uint8_t type;

            type = 0;
            arith.cgbn_memory_from_hex_string(template_transaction->nonce, PyUnicode_AsUTF8(PyDict_GetItemString(data, "nonce")));
            arith.cgbn_memory_from_hex_string(template_transaction->to, PyUnicode_AsUTF8(PyDict_GetItemString(data, "to")));
            arith.cgbn_memory_from_hex_string(template_transaction->sender, PyUnicode_AsUTF8(PyDict_GetItemString(data, "sender")));

            type = 0;
            arith.cgbn_memory_from_size_t(template_transaction->max_fee_per_gas, 0);
            arith.cgbn_memory_from_size_t(template_transaction->max_priority_fee_per_gas, 0);
            arith.cgbn_memory_from_hex_string(template_transaction->gas_price,"0x0a");

            template_transaction->type = type;

            // char *bytes_string = NULL;
            memcpy(&(transactions[idx]), template_transaction, sizeof(transaction_data_t));
            arith.cgbn_memory_from_hex_string(transactions[idx].gas_limit, PyUnicode_AsUTF8(PyList_GetItem(tx_gas_limit, 0)));
            arith.cgbn_memory_from_hex_string(transactions[idx].value, PyUnicode_AsUTF8(PyList_GetItem(tx_value, 0)));
            const char* bytes_string = PyUnicode_AsUTF8(PyList_GetItem(tx_data, 0));

            bytes_string = adjust_hex_string(bytes_string);
            transactions[idx].data_init.size = strlen(bytes_string)/2;
            if (transactions[idx].data_init.size > 0)
            {
        #ifndef ONLY_CPU
                CUDA_CHECK(cudaMallocManaged(
                    (void **)&(transactions[idx].data_init.data),
                    transactions[idx].data_init.size * sizeof(uint8_t)));
        #else
                transactions[idx].data_init.data = new uint8_t[transactions[idx].data_init.size];
        #endif
                hex_to_bytes(
                    bytes_string,
                    transactions[idx].data_init.data,
                    2 * transactions[idx].data_init.size);
            }
            else
            {
                transactions[idx].data_init.data = NULL;
            }

            delete template_transaction;
        }

        return transactions;
    }

    transaction_data_t* getTransactionDataFromPyObject(arith_t &arith, PyObject* data, size_t &instances_count){
        // get data size
        PyObject* tx_data =  PyDict_GetItemString(data, "data");
        PyObject* tx_gas_limit = PyDict_GetItemString(data, "gasLimit");
        PyObject* tx_value = PyDict_GetItemString(data, "value");
        if (tx_data == NULL || !PyList_Check(tx_data)){
            printf("Invalid transaction data\n");
            return NULL;
        }
        size_t tx_data_count = PyList_Size(tx_data);
        size_t gas_limit_count = PyList_Size(tx_gas_limit);
        size_t value_count = PyList_Size(tx_value);
        size_t count = max(max(tx_data_count, gas_limit_count), value_count);
        instances_count = count;
        printf("Transaction count: %d\n", count);
        transaction_data_t *transactions;
    #ifndef ONLY_CPU
        CUDA_CHECK(cudaMallocManaged((void **)&(transactions), count * sizeof(transaction_data_t)));
    #else
        transactions = new transaction_data_t[count];
    #endif
        transaction_data_t *template_transaction = new transaction_data_t;
        memset(template_transaction, 0, sizeof(transaction_data_t));

        uint8_t type;
        size_t idx = 0;

        type = 0;

        arith.cgbn_memory_from_hex_string(template_transaction->nonce, PyUnicode_AsUTF8(PyDict_GetItemString(data, "nonce")));

        arith.cgbn_memory_from_hex_string(template_transaction->to, PyUnicode_AsUTF8(PyDict_GetItemString(data, "to")));

        arith.cgbn_memory_from_hex_string(template_transaction->sender, PyUnicode_AsUTF8(PyDict_GetItemString(data, "sender")));


        type = 0;
        arith.cgbn_memory_from_size_t(template_transaction->max_fee_per_gas, 0);
        arith.cgbn_memory_from_size_t(template_transaction->max_priority_fee_per_gas, 0);
        arith.cgbn_memory_from_hex_string(template_transaction->gas_price,"0x0a");

        template_transaction->type = type;

        size_t index;
        // char *bytes_string = NULL;
        for (idx = 0; idx < count; idx++)
        {

            memcpy(&(transactions[idx]), template_transaction, sizeof(transaction_data_t));
            arith.cgbn_memory_from_hex_string(transactions[idx].gas_limit, PyUnicode_AsUTF8(PyList_GetItem(tx_gas_limit, idx < gas_limit_count ? idx : 0)));
            arith.cgbn_memory_from_hex_string(transactions[idx].value, PyUnicode_AsUTF8(PyList_GetItem(tx_value, idx < value_count ? idx : 0)));
            const char* bytes_string = PyUnicode_AsUTF8(PyList_GetItem(tx_data, idx < tx_data_count ? idx : 0));

            bytes_string = adjust_hex_string(bytes_string);
            transactions[idx].data_init.size = strlen(bytes_string)/2;
            if (transactions[idx].data_init.size > 0)
            {
        #ifndef ONLY_CPU
                CUDA_CHECK(cudaMallocManaged(
                    (void **)&(transactions[idx].data_init.data),
                    transactions[idx].data_init.size * sizeof(uint8_t)));
        #else
                transactions[idx].data_init.data = new uint8_t[transactions[idx].data_init.size];
        #endif
                hex_to_bytes(
                    bytes_string,
                    transactions[idx].data_init.data,
                    2 * transactions[idx].data_init.size);
            }
            else
            {
                transactions[idx].data_init.data = NULL;
            }

        }
        delete template_transaction;
        return transactions;
    }

    state_data_t* getStateDataFromPyObject(arith_t &arith, PyObject* data) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        state_data_t* state_data;
        #ifndef ONLY_CPU
        CUDA_CHECK(cudaMallocManaged(
            (void **)&(state_data),
            sizeof(state_data_t)
        ));
        #else
        state_data = new state_data_t;
        #endif

        state_data->no_accounts = PyDict_Size(data);

        #ifndef ONLY_CPU
        CUDA_CHECK(cudaMallocManaged((void **)&(state_data->accounts), state_data->no_accounts * sizeof(account_t)));
        #else
        state_data->accounts = new account_t[state_data->no_accounts];
        #endif

        while (PyDict_Next(data, &pos, &key, &value)) {
            int account_index = pos - 1;  // Adjust index since PyDict_Next increments pos
            const char* address_str = PyUnicode_AsUTF8(key);

            // Extract balance, nonce, and code
            const char* balance = GET_STR_FROM_DICT_WITH_DEFAULT(value, "balance", "0x0");
            const char* nonce = GET_STR_FROM_DICT_WITH_DEFAULT(value, "nonce", "0x0");
            const char* code = GET_STR_FROM_DICT_WITH_DEFAULT(value, "code", "");

            // Initialize account details
            arith.cgbn_memory_from_hex_string(state_data->accounts[account_index].address, address_str);
            arith.cgbn_memory_from_hex_string(state_data->accounts[account_index].balance, balance);
            arith.cgbn_memory_from_hex_string(state_data->accounts[account_index].nonce, nonce);
            // printf("Account %d: %s\n", account_index, address_str);
            // Bytecode handling
            code = adjust_hex_string(code);
            state_data->accounts[account_index].code_size = strlen(code) / 2;  // Assuming each byte is represented by 2 hex characters, prefix 0x
            if (state_data->accounts[account_index].code_size > 0) {
                #ifndef ONLY_CPU
                CUDA_CHECK(cudaMallocManaged((void **)&(state_data->accounts[account_index].bytecode), state_data->accounts[account_index].code_size));
                #else
                state_data->accounts[account_index].bytecode = new uint8_t[state_data->accounts[account_index].code_size];
                #endif
                hex_to_bytes(code, state_data->accounts[account_index].bytecode, strlen(code));
            } else {
                state_data->accounts[account_index].bytecode = NULL;
            }

            // Storage handling (assuming a function to handle storage initialization)
            PyObject* storage_dict = PyDict_GetItemString(value, "storage");
            state_data->accounts[account_index].storage_size = PyDict_Size(storage_dict);
            if (storage_dict && state_data->accounts[account_index].storage_size > 0) {

                // Assuming you have a function to handle storage initialization
                // initialize_storage(&state_data->accounts[account_index], storage_dict);
                PyObject *key_storage, *value_storage;

                Py_ssize_t pos_storage = 0;
                Py_ssize_t idx_storage = 0;
                // allocate the storage
                #ifndef ONLY_CPU
                CUDA_CHECK(cudaMallocManaged(
                    (void **)&(state_data->accounts[account_index].storage),
                    state_data->accounts[account_index].storage_size * sizeof(contract_storage_t)
                ));
                #else
                state_data->accounts[account_index].storage = new contract_storage_t[state_data->accounts[account_index].storage_size];
                #endif

                size_t idx = 0;
                // Iterate through the dictionary
                while (PyDict_Next(storage_dict, &pos_storage, &key_storage, &value_storage)) {
                    // Increase reference count for key and value before storing them
                    Py_INCREF(key_storage);
                    Py_INCREF(value_storage);

                    PyObject* keyStrObj = PyObject_Str(key_storage);
                    const char* keyStr = PyUnicode_AsUTF8(keyStrObj);
                    arith.cgbn_memory_from_hex_string(state_data->accounts[account_index].storage[idx].key, keyStr);
                    Py_DECREF(keyStrObj);

                    PyObject* valueStrObj = PyObject_Str(value_storage);
                    const char* valueStr = PyUnicode_AsUTF8(valueStrObj);
                    arith.cgbn_memory_from_hex_string(state_data->accounts[account_index].storage[idx].value, valueStr);
                    Py_DECREF(valueStrObj);

                    idx++;
                }


            } else {
                state_data->accounts[account_index].storage = NULL;
                state_data->accounts[account_index].storage_size = 0;
            }

        }

        return state_data;
    }


    std::unordered_set<int> interesting_opcodes = {OP_ADD, OP_MUL, OP_SUB, OP_MOD, OP_EXP, \
                                                OP_CALL, OP_CALLCODE, OP_DELEGATECALL, \
                                                OP_LT, OP_GT, OP_SLT, OP_SGT, OP_EQ, \
                                                OP_REVERT, OP_INVALID, OP_SSTORE, OP_JUMPI, OP_SELFDESTRUCT};
    std::unordered_set<int> bug_opcodes = {OP_ADD, OP_MUL, OP_SUB, OP_MOD, OP_EXP, OP_SELFDESTRUCT, OP_ORIGIN};
    std::unordered_set<int> call_opcodes = {OP_CALL, OP_CALLCODE, OP_DELEGATECALL}; // ignore static call for now
    std::unordered_set<int> comparison_opcodes = {OP_LT, OP_GT, OP_SLT, OP_SGT, OP_EQ};
    std::unordered_set<int> revert_opcodes = {OP_REVERT, OP_INVALID};
    // OP_SSTORE
    // OP_JUMPI
    // OP_SELFDESTRUCT

    /**
     * Get the python dict object from the tracer data structure.
     * @param[in] arith The arithmetical environment.
     * @param[in] tracer_data The trace data structure.
     * @return The pythonobject.
    */
    __host__ static PyObject* pyobject_from_tracer_data_t(arith_t &arith, tracer_data_t tracer_data) {
        char* hex_string_ptr = new char[arith_t::BYTES * 2 + 3];
        PyObject* tracer_root = PyDict_New();

        PyObject* branches = PyList_New(0);
        PyObject* bugs = PyList_New(0);
        PyObject* calls = PyList_New(0);
        PyObject* storage_write = PyList_New(0);

        PyObject* tracer_json = PyList_New(0);
        PyObject* item = NULL;
        PyObject* stack_json = NULL;

        size_t previous_distance;

        for (size_t idx = 0; idx < tracer_data.size; idx++) {
            uint8_t current_opcode = tracer_data.opcodes[idx];
            uint32_t current_pc = tracer_data.pcs[idx];

            if (interesting_opcodes.find(current_opcode) == interesting_opcodes.end()) {
                continue;
            }

            item = PyDict_New();
            arith.hex_string_from_cgbn_memory(hex_string_ptr, tracer_data.addresses[idx], 5);
            PyDict_SetItemString(item, "address", PyUnicode_FromString(hex_string_ptr));

            PyDict_SetItemString(item, "pc", PyLong_FromSize_t(current_pc));
            PyDict_SetItemString(item, "opcode", PyLong_FromSize_t(current_opcode));

            stack_json = stack_t::pyobject_from_stack_data_t(arith, tracer_data.stacks[idx]);  // Assuming toPyObject() is implemented
            PyDict_SetItemString(item, "stack", stack_json);
            Py_DECREF(stack_json);

            PyList_Append(tracer_json, item);
            Py_DECREF(item);  // Decrement reference count since PyList_Append increases it

            if (bug_opcodes.find(current_opcode) != bug_opcodes.end()) {
                // Simple OP:
                if(current_opcode == OP_SELFDESTRUCT || current_opcode == OP_ORIGIN){
                    PyObject* bug_item = PyDict_New();
                    PyDict_SetItemString(bug_item, "address", PyUnicode_FromString(hex_string_ptr));
                    PyDict_SetItemString(bug_item, "pc", PyLong_FromSize_t(current_pc));
                    PyDict_SetItemString(bug_item, "opcode", PyLong_FromSize_t(current_opcode));
                    PyList_Append(bugs, bug_item);
                    Py_DECREF(bug_item);
                } else {
                    PyObject* bug_item = PyDict_New();
                    PyDict_SetItemString(bug_item, "address", PyUnicode_FromString(hex_string_ptr));
                    PyDict_SetItemString(bug_item, "pc", PyLong_FromSize_t(current_pc));
                    PyDict_SetItemString(bug_item, "opcode", PyLong_FromSize_t(current_opcode));

                    uint32_t stack_size = tracer_data.stacks[idx].stack_offset;

                    evm_word_t * operand_1 = tracer_data.stacks[idx].stack_base + (stack_size - 1);
                    arith.hex_string_from_cgbn_memory(hex_string_ptr, *operand_1);
                    PyDict_SetItemString(bug_item, "operand_1", PyUnicode_FromString(hex_string_ptr));

                    evm_word_t * operand_2 = tracer_data.stacks[idx].stack_base + (stack_size - 2);
                    arith.hex_string_from_cgbn_memory(hex_string_ptr, *operand_2);
                    PyDict_SetItemString(bug_item, "operand_2", PyUnicode_FromString(hex_string_ptr));
                    PyList_Append(bugs, bug_item);
                    Py_DECREF(bug_item);

                }
            }

            if (current_opcode == OP_JUMPI){
                // process jumpi
                PyObject* branch_item = PyDict_New();
                PyDict_SetItemString(branch_item, "address", PyUnicode_FromString(hex_string_ptr));
                PyDict_SetItemString(branch_item, "pc", PyLong_FromSize_t(current_pc));

                bn_t jump_cond, destination;
                uint32_t stack_size = tracer_data.stacks[idx].stack_offset;
                cgbn_load(arith._env, destination, tracer_data.stacks[idx].stack_base + stack_size -1);
                cgbn_load(arith._env, jump_cond, tracer_data.stacks[idx].stack_base + stack_size - 2);

                // printf("\n\n JumpI %d %d\n\n", cgbn_get_ui32(arith._env, jump_cond), cgbn_get_ui32(arith._env, destination));
                PyDict_SetItemString(branch_item, "distance", PyLong_FromSize_t(previous_distance));
                if (cgbn_get_ui32(arith._env, jump_cond) == 0){
                    PyDict_SetItemString(branch_item, "destination", PyLong_FromSize_t(current_pc + 1));
                    PyDict_SetItemString(branch_item, "missed_destination", PyLong_FromSize_t(cgbn_get_ui32(arith._env, destination)));
                } else {
                    PyDict_SetItemString(branch_item, "destination", PyLong_FromSize_t(cgbn_get_ui32(arith._env, destination)));
                    PyDict_SetItemString(branch_item, "missed_destination", PyLong_FromSize_t(current_pc + 1));
                }
                PyList_Append(branches, branch_item);
            }

            // process comparison
            if (comparison_opcodes.find(current_opcode) != comparison_opcodes.end()) {
                // process comparison
                // stack_size = computation._stack.__len__()
                uint32_t stack_size = tracer_data.stacks[idx].stack_offset;
                bn_t distance, op1, op2;

                cgbn_load(arith._env, op1, tracer_data.stacks[idx].stack_base + stack_size - 1);
                if (stack_size > 1)
                    cgbn_load(arith._env, op2, tracer_data.stacks[idx].stack_base + stack_size - 2);

                if (cgbn_compare(arith._env, op1, op2) >= 1)
                    cgbn_sub(arith._env, distance, op1, op2);
                else
                    cgbn_sub(arith._env, distance, op2, op1);

                if (current_opcode != OP_EQ)
                    cgbn_add_ui32(arith._env, distance, distance, 1);

                arith.size_t_from_cgbn(previous_distance, distance);

            }

            //process calls
            if (call_opcodes.find(current_opcode) != call_opcodes.end()){
                bn_t value;
                PyObject* call_item = PyDict_New();

                if (current_opcode != OP_DELEGATECALL){
                    evm_word_t * value = tracer_data.stacks[idx].stack_base + tracer_data.stacks[idx].stack_offset - 3;
                    arith.hex_string_from_cgbn_memory(hex_string_ptr, *value);
                    PyDict_SetItemString(call_item, "value", PyUnicode_FromString(hex_string_ptr));
                }
                else{
                    PyDict_SetItemString(call_item, "value", PyUnicode_FromString("0x00"));
                }

                evm_word_t * address = tracer_data.stacks[idx].stack_base + tracer_data.stacks[idx].stack_offset - 2;

                arith.hex_string_from_cgbn_memory(hex_string_ptr, *address, 5);

                PyDict_SetItemString(call_item, "address", PyUnicode_FromString(hex_string_ptr));
                PyDict_SetItemString(call_item, "pc", PyLong_FromSize_t(current_pc));

                PyList_Append(calls, call_item);
            }
            // process revert
            if (current_opcode == OP_REVERT || current_opcode == OP_INVALID){
                PyObject* revert_item = PyDict_New();
                PyDict_SetItemString(revert_item, "address", PyUnicode_FromString("0x00"));
                PyDict_SetItemString(revert_item, "pc", PyLong_FromSize_t(0));
                PyDict_SetItemString(revert_item, "value", PyUnicode_FromString("0x00"));

                PyList_Append(calls, revert_item);
                Py_DECREF(revert_item);
            }

            if (current_opcode == OP_SSTORE){
                PyObject* storage_item = PyDict_New();
                PyDict_SetItemString(storage_item, "address", PyUnicode_FromString(hex_string_ptr));
                PyDict_SetItemString(storage_item, "pc", PyLong_FromSize_t(current_pc));

                uint32_t stack_size = tracer_data.stacks[idx].stack_offset;
                evm_word_t * value = tracer_data.stacks[idx].stack_base + stack_size - 2;
                evm_word_t * key = tracer_data.stacks[idx].stack_base + stack_size - 1;

                arith.hex_string_from_cgbn_memory(hex_string_ptr, *key);
                PyDict_SetItemString(storage_item, "key", PyUnicode_FromString(hex_string_ptr));

                arith.hex_string_from_cgbn_memory(hex_string_ptr, *value);
                PyDict_SetItemString(storage_item, "value", PyUnicode_FromString(hex_string_ptr));

                PyList_Append(storage_write, storage_item);
                Py_DECREF(storage_item);
            }
        }

        PyDict_SetItemString(tracer_root, "traces", tracer_json);
        PyDict_SetItemString(tracer_root, "branches", branches);
        PyDict_SetItemString(tracer_root, "bugs", bugs);
        PyDict_SetItemString(tracer_root, "calls", calls);
        PyDict_SetItemString(tracer_root, "storage_write", storage_write);

        delete[] hex_string_ptr;
        return tracer_root;
    }

    /*
        * Get PyObject of the account
        * @param[in] arith The arithmetical environment
        * @param[in] account The account
        * @return The PyObject of the account
    */
    __host__ static PyObject* pyobject_from_account_t(arith_t &arith, account_t account) {
        PyObject* account_json = PyDict_New();
        PyObject* storage_json = PyDict_New();
        char* hex_string_ptr = new char[arith_t::BYTES * 2 + 3];
        char* value_hex_string_ptr = new char[arith_t::BYTES * 2 + 3];
        size_t jdx = 0;

        // Set the address
        arith.hex_string_from_cgbn_memory(hex_string_ptr, account.address, 5);
        PyDict_SetItemString(account_json, "address", PyUnicode_FromString(hex_string_ptr));

        // Set the balance
        arith.pretty_hex_string_from_cgbn_memory(hex_string_ptr, account.balance);
        PyDict_SetItemString(account_json, "balance", PyUnicode_FromString(hex_string_ptr));

        // Set the nonce
        arith.pretty_hex_string_from_cgbn_memory(hex_string_ptr, account.nonce);
        PyDict_SetItemString(account_json, "nonce", PyUnicode_FromString(hex_string_ptr));

        // Set the code
        if (account.code_size > 0) {
            char* bytes_string = hex_from_bytes(account.bytecode, account.code_size);  // Assuming hex_from_bytes is defined elsewhere
            PyDict_SetItemString(account_json, "code", PyUnicode_FromString(bytes_string));
            delete[] bytes_string;
        } else {
            PyDict_SetItemString(account_json, "code", PyUnicode_FromString("0x"));
        }

        // Set the storage
        PyDict_SetItemString(account_json, "storage", storage_json);
        if (account.storage_size > 0) {
            for (jdx = 0; jdx < account.storage_size; jdx++) {
                arith.pretty_hex_string_from_cgbn_memory(hex_string_ptr, account.storage[jdx].key);
                arith.pretty_hex_string_from_cgbn_memory(value_hex_string_ptr, account.storage[jdx].value);
                PyDict_SetItemString(storage_json, hex_string_ptr, PyUnicode_FromString(value_hex_string_ptr));
            }
        }

        // Clean up
        delete[] hex_string_ptr;
        delete[] value_hex_string_ptr;

        // Decrement the reference count of storage_json since it's now owned by account_json
        Py_DECREF(storage_json);

        return account_json;
    }

    /**
     * Get pyobject of the state
    */
    __host__ __forceinline__ static PyObject* pyobject_from_state_data_t(arith_t &arith, state_data_t* state_data) {

        PyObject* state_json = PyDict_New();
        PyObject* account_json = NULL;
        char* hex_string_ptr = new char[arith_t::BYTES * 2 + 3];
        size_t idx = 0;

        for (idx = 0; idx < state_data->no_accounts; idx++) {
            // Assuming an equivalent function or method exists that returns a PyObject* for an account
            account_json = pyobject_from_account_t(arith, state_data->accounts[idx]);  // Replace with actual function/method

            // Convert account address to hex string
            arith.hex_string_from_cgbn_memory(hex_string_ptr, state_data->accounts[idx].address, 5);

            // Add account PyObject to the state dictionary using the address as the key
            PyDict_SetItemString(state_json, hex_string_ptr, account_json);

            // Decrement the reference count of account_json since PyDict_SetItemString increases it
            Py_DECREF(account_json);
        }

        delete[] hex_string_ptr;
        return state_json;
    }
    /*
        * Get PyObject of the transaction
        * @param[in] arith The arithmetical environment
        * @param[in] transaction The transaction
        * @return The PyObject of the transaction
    */
    PyObject* pyobject_from_transaction_content(arith_t &_arith, transaction_data_t* _content) {
        PyObject* transaction_json = PyDict_New();
        char* hex_string_ptr = new char[arith_t::BYTES * 2 + 3];
        char* bytes_string = NULL;

        // Set the type
        PyDict_SetItemString(transaction_json, "type", PyLong_FromLong(_content->type));

        // Set the nonce
        _arith.hex_string_from_cgbn_memory(hex_string_ptr, _content->nonce);
        PyDict_SetItemString(transaction_json, "nonce", PyUnicode_FromString(hex_string_ptr));

        // Set the gas limit
        _arith.hex_string_from_cgbn_memory(hex_string_ptr, _content->gas_limit);
        PyDict_SetItemString(transaction_json, "gasLimit", PyUnicode_FromString(hex_string_ptr));

        // Set the to address
        _arith.hex_string_from_cgbn_memory(hex_string_ptr, _content->to, 5);
        PyDict_SetItemString(transaction_json, "to", PyUnicode_FromString(hex_string_ptr));

        // Set the value
        _arith.hex_string_from_cgbn_memory(hex_string_ptr, _content->value);
        PyDict_SetItemString(transaction_json, "value", PyUnicode_FromString(hex_string_ptr));

        // Set the sender
        _arith.hex_string_from_cgbn_memory(hex_string_ptr, _content->sender, 5);
        PyDict_SetItemString(transaction_json, "sender", PyUnicode_FromString(hex_string_ptr));
        // TODO: delete this from revmi comparator
        PyDict_SetItemString(transaction_json, "origin", PyUnicode_FromString(hex_string_ptr));

        // Set the access list
        if (_content->type >= 1) {
            PyObject* access_list_json = PyList_New(0);
            PyDict_SetItemString(transaction_json, "accessList", access_list_json);
            Py_DECREF(access_list_json);  // Decrement because PyDict_SetItemString increases the ref count

            for (size_t idx = 0; idx < _content->access_list.no_accounts; idx++) {
                PyObject* account_json = PyDict_New();
                PyList_Append(access_list_json, account_json);  // Append steals the reference, so no need to DECREF

                _arith.hex_string_from_cgbn_memory(hex_string_ptr, _content->access_list.accounts[idx].address, 5);
                PyDict_SetItemString(account_json, "address", PyUnicode_FromString(hex_string_ptr));

                PyObject* storage_keys_json = PyList_New(0);
                PyDict_SetItemString(account_json, "storageKeys", storage_keys_json);
                Py_DECREF(storage_keys_json);  // Decrement because PyDict_SetItemString increases the ref count

                for (size_t jdx = 0; jdx < _content->access_list.accounts[idx].no_storage_keys; jdx++) {
                    _arith.hex_string_from_cgbn_memory(hex_string_ptr, _content->access_list.accounts[idx].storage_keys[jdx]);
                    PyList_Append(storage_keys_json, PyUnicode_FromString(hex_string_ptr));
                }
            }
        }

        // Set the gas price
        if (_content->type == 2) {
            _arith.hex_string_from_cgbn_memory(hex_string_ptr, _content->max_fee_per_gas);
            PyDict_SetItemString(transaction_json, "maxFeePerGas", PyUnicode_FromString(hex_string_ptr));

            // Set the max priority fee per gas
            _arith.hex_string_from_cgbn_memory(hex_string_ptr, _content->max_priority_fee_per_gas);
            PyDict_SetItemString(transaction_json, "maxPriorityFeePerGas", PyUnicode_FromString(hex_string_ptr));
        } else {
            // Set the gas price
            _arith.hex_string_from_cgbn_memory(hex_string_ptr, _content->gas_price);
            PyDict_SetItemString(transaction_json, "gasPrice", PyUnicode_FromString(hex_string_ptr));
        }

        // Set the data
        if (_content->data_init.size > 0) {
            bytes_string = hex_from_data_content(_content->data_init);
            PyDict_SetItemString(transaction_json, "data", PyUnicode_FromString(bytes_string));
            delete[] bytes_string;
        } else {
            PyDict_SetItemString(transaction_json, "data", PyUnicode_FromString("0x"));
        }

        delete[] hex_string_ptr;
        return transaction_json;
    }


    /**
     * Get the pyobject from the evm instances after the transaction execution.
     * @param[in] arith arithmetic environment
     * @param[in] instances evm instances
     * @return pyobject
    */
    __host__ __forceinline__ static PyObject* pyobject_from_evm_instances_t(arith_t &arith, evm_instances_t instances) {
        PyObject* root = PyDict_New();
        // PyObject* world_state_json = pyobject_from_state_data_t(arith, instances.world_state_data);
        // PyDict_SetItemString(root, "pre", world_state_json);
        // Py_DECREF(world_state_json);
        PyObject* instances_json = PyList_New(0);
        PyDict_SetItemString(root, "post", instances_json);
        Py_DECREF(instances_json);  // Decrement here because PyDict_SetItemString increases the ref count

        for (uint32_t idx = 0; idx < instances.count; idx++) {

            state_data_t* world_state_instance = instances.world_state_data[idx]; // update on this
            touch_state_data_t prev_state, updated_state;
            touch_state_data_t ref_touch_state = instances.touch_states_data[idx];
            world_state_t world_state(arith, world_state_instance);
            accessed_state_t accessed_state_1(&world_state);
            accessed_state_t accessed_state_2(&world_state);
            updated_state.touch = new uint8_t[ref_touch_state.touch_accounts.no_accounts];

            for (size_t j = 0; j < ref_touch_state.touch_accounts.no_accounts; j++)
            {
                updated_state.touch[j] = ref_touch_state.touch[j];
            }

            updated_state.touch_accounts.no_accounts = ref_touch_state.touch_accounts.no_accounts;
            updated_state.touch_accounts.accounts = ref_touch_state.touch_accounts.accounts;

            prev_state.touch_accounts.no_accounts = world_state_instance->no_accounts;
            prev_state.touch_accounts.accounts = world_state_instance->accounts;
            prev_state.touch = new uint8_t[world_state_instance->no_accounts];

            touch_state_t tx_result_state(&updated_state, &accessed_state_1, arith);
            touch_state_t final_state(&prev_state, &accessed_state_2, arith);

            final_state.update_with_child_state(tx_result_state);

            char* temp = new char[arith_t::BYTES * 2 + 3];

            delete[] prev_state.touch;
            delete[] updated_state.touch;
            prev_state.touch = nullptr;
            updated_state.touch = nullptr;

            PyObject* instance_json = PyDict_New();

            // TODO: Print resultant state. to check with state_root branch
            PyObject* state_json = pyobject_from_state_data_t(arith, &prev_state.touch_accounts);//PyList_New(0);
            // print_dict_recursive(state_json, 0);
            PyObject* transaction_json = pyobject_from_transaction_content(arith, &instances.transactions_data[idx]);
            PyDict_SetItemString(instance_json, "msg", transaction_json);

            PyDict_SetItemString(instance_json, "state", state_json);

            #ifdef TRACER
            PyObject* tracer_json = pyobject_from_tracer_data_t(arith, instances.tracers_data[idx]);
            PyDict_SetItemString(instance_json, "traces", tracer_json);
            Py_DECREF(tracer_json);
            #endif

            PyDict_SetItemString(instance_json, "error", PyLong_FromLong(instances.errors[idx]));
            PyDict_SetItemString(instance_json, "success", PyBool_FromLong((instances.errors[idx] == ERR_NONE) || (instances.errors[idx] == ERR_RETURN) || (instances.errors[idx] == ERR_SUCCESS)));
            PyList_Append(instances_json, instance_json);  // Appends and steals the reference, so no need to DECREF
        }

        return root;
    }

}

#endif