#pragma once
#ifndef IGNORELIB
#include <Python.h>

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

namespace python_utils{

    block_data_t* getBlockDataFromPyObject(arith_t arith, PyObject* data){
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

    transaction_data_t* getTransactionDataFromPyObject(arith_t arith, PyObject* data){
        size_t count = 1;
        transaction_data_t *transactions;
    #ifndef ONLY_CPU
        CUDA_CHECK(cudaMallocManaged(
            (void **)&(transactions),
            count * sizeof(transaction_data_t)));
    #else
        transactions = new transaction_data_t[count];
    #endif
        transaction_data_t *template_transaction = new transaction_data_t;
        memset(template_transaction, 0, sizeof(transaction_data_t));

        uint8_t type;
        size_t data_index, gas_limit_index, value_index;
        size_t idx = 0, jdx = 0;

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
        char *bytes_string = NULL;
        for (idx = 0; idx < count; idx++)
        {

        data_index = index;
        gas_limit_index = index;
        value_index = index;
        memcpy(&(transactions[idx]), template_transaction, sizeof(transaction_data_t));
        arith.cgbn_memory_from_hex_string(
            transactions[idx].gas_limit,"0x04c4b400");
        arith.cgbn_memory_from_hex_string(
            transactions[idx].value, "0x01");
        bytes_string = "0x00";
        transactions[idx].data_init.size = adjusted_length(&bytes_string);
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

    state_data_t* getStateDataFromPyObject(arith_t arith, PyObject* data) {
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
            if (strlen(code) >= 2 && (code[0] == '0' && (code[1] == 'x' || code[1] == 'X')))
                code += 2;  // Skip the "0x" prefix
            if (strlen(code) % 2 != 0) {
                printf("Invalid bytecode length\n");
                return NULL;
            }
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
            if (storage_dict && PyDict_Size(storage_dict) > 0) {
                // Assuming you have a function to handle storage initialization
                // initialize_storage(&state_data->accounts[account_index], storage_dict);
                printf("Storage handling not implemented\n");
            } else {
                state_data->accounts[account_index].storage = NULL;
                state_data->accounts[account_index].storage_size = 0;
            }

        }

        return state_data;
    }

    /**
     * Get the python dict object from the tracer data structure.
     * @param[in] arith The arithmetical environment.
     * @param[in] tracer_data The trace data structure.
     * @return The pythonobject.
    */
    __host__ static PyObject* pyobject_from_tracer_data_t(arith_t &arith, tracer_data_t tracer_data) {
        char* hex_string_ptr = new char[arith_t::BYTES * 2 + 3];
        PyObject* tracer_json = PyList_New(0);
        PyObject* item = NULL;
        PyObject* stack_json = NULL;
        for (size_t idx = 0; idx < tracer_data.size; idx++) {
            item = PyDict_New();
            arith.hex_string_from_cgbn_memory(hex_string_ptr, tracer_data.addresses[idx], 5);
            PyDict_SetItemString(item, "address", PyUnicode_FromString(hex_string_ptr));

            PyDict_SetItemString(item, "pc", PyLong_FromSize_t(tracer_data.pcs[idx]));
            PyDict_SetItemString(item, "opcode", PyLong_FromSize_t(tracer_data.opcodes[idx]));

            stack_json = stack_t::pyobject_from_stack_data_t(arith, tracer_data.stacks[idx]);  // Assuming toPyObject() is implemented
            PyDict_SetItemString(item, "stack", stack_json);
            Py_DECREF(stack_json);

            PyList_Append(tracer_json, item);
            Py_DECREF(item);  // Decrement reference count since PyList_Append increases it
        }

        delete[] hex_string_ptr;
        return tracer_json;
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
        arith.hex_string_from_cgbn_memory(hex_string_ptr, account.balance);
        PyDict_SetItemString(account_json, "balance", PyUnicode_FromString(hex_string_ptr));

        // Set the nonce
        arith.hex_string_from_cgbn_memory(hex_string_ptr, account.nonce);
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
                arith.hex_string_from_cgbn_memory(hex_string_ptr, account.storage[jdx].key);
                arith.hex_string_from_cgbn_memory(value_hex_string_ptr, account.storage[jdx].value);
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

    /**
     * Get the pyobject from the evm instances after the transaction execution.
     * @param[in] arith arithmetic environment
     * @param[in] instances evm instances
     * @return pyobject
    */
    __host__ __forceinline__ static PyObject* pyobject_from_evm_instances_t(arith_t &arith, evm_instances_t instances) {
        PyObject* root = PyDict_New();
        PyObject* world_state_json = pyobject_from_state_data_t(arith, instances.world_state_data);
        PyDict_SetItemString(root, "pre", world_state_json);
        Py_DECREF(world_state_json);

        PyObject* instances_json = PyList_New(0);
        PyDict_SetItemString(root, "post", instances_json);
        Py_DECREF(instances_json);  // Decrement here because PyDict_SetItemString increases the ref count

        for (uint32_t idx = 0; idx < instances.count; idx++) {
            PyObject* instance_json = PyDict_New();
            PyList_Append(instances_json, instance_json);  // Appends and steals the reference, so no need to DECREF

            #ifdef TRACER
            PyObject* tracer_json = pyobject_from_tracer_data_t(arith, instances.tracers_data[idx]);
            PyDict_SetItemString(instance_json, "traces", tracer_json);
            Py_DECREF(tracer_json);
            #endif

            PyDict_SetItemString(instance_json, "error", PyLong_FromLong(instances.errors[idx]));
            PyDict_SetItemString(instance_json, "success", PyBool_FromLong((instances.errors[idx] == ERR_NONE) || (instances.errors[idx] == ERR_RETURN) || (instances.errors[idx] == ERR_SUCCESS)));
        }

        return root;
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

}

#endif