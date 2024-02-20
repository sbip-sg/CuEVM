#include <getopt.h>
#include <fstream>

#include <Python.h>

#include "utils.cu"
#include "evm.cuh"

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

using block_data_t = block_t::block_data_t;
using state_data_t = world_state_t::state_data_t;
using transaction_data_t = transaction_t::transaction_data_t;
using account_t = world_state_t::account_t;
void copy_dict_recursive(PyObject *read_root, PyObject *write_root);
void run_interpreter(cJSON *read_root, cJSON *write_root) {}
static PyObject* print_dict(PyObject* self, PyObject* args);

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
    
    arith.cgbn_memory_from_hex_string(template_transaction->nonce, "0x00");

    arith.cgbn_memory_from_hex_string(template_transaction->to, "0xcccccccccccccccccccccccccccccccccccccccc");
    
    arith.cgbn_memory_from_hex_string(template_transaction->sender, "0xa94f5374fce5edbc8e2a8697c15331677e6ebf0b");


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
        state_data->accounts[account_index].code_size = (strlen(code) - 2) / 2;  // Assuming each byte is represented by 2 hex characters, prefix 0x
        if (state_data->accounts[account_index].code_size > 0) {
            #ifndef ONLY_CPU
            CUDA_CHECK(cudaMallocManaged((void **)&(state_data->accounts[account_index].bytecode), state_data->accounts[account_index].code_size));
            #else
            state_data->accounts[account_index].bytecode = new uint8_t[state_data->accounts[account_index].code_size];
            #endif
            hex_to_bytes(code, state_data->accounts[account_index].bytecode, state_data->accounts[account_index].code_size);
        } else {
            state_data->accounts[account_index].bytecode = NULL;
        }

        // Storage handling (assuming a function to handle storage initialization)
        PyObject* storage_dict = PyDict_GetItemString(value, "storage");
        if (storage_dict && PyDict_Size(storage_dict) > 0) {
            // Assuming you have a function to handle storage initialization
            // initialize_storage(&state_data->accounts[account_index], storage_dict);
        } else {
            state_data->accounts[account_index].storage = NULL;
            state_data->accounts[account_index].storage_size = 0;
        }

    }

    return state_data;
}

void copy_dict_recursive(PyObject *read_root, PyObject *write_root) {
    PyObject *key, *value;
    Py_ssize_t pos = 0;

    // Iterate over dictionary items in read_root
    while (PyDict_Next(read_root, &pos, &key, &value)) {
        // If value is a dictionary, perform recursive copying
        if (PyDict_Check(value)) {
            PyObject *sub_write_root = PyDict_New();  // Create a new dictionary for the nested structure
            if (!sub_write_root) return;  // Error handling in case PyDict_New fails

            copy_dict_recursive(value, sub_write_root);  // Recursive call to copy nested dictionary

            // Set the nested dictionary in write_root
            if (PyDict_SetItem(write_root, key, sub_write_root) < 0) {
                Py_DECREF(sub_write_root);
                return;  // Error handling in case PyDict_SetItem fails
            }

            Py_DECREF(sub_write_root);  // Decrease reference count for the new dictionary
        } else {
            // For non-dictionary values, directly copy the value from read_root to write_root
            if (PyDict_SetItem(write_root, key, value) < 0) {
                return;  // Error handling in case PyDict_SetItem fails
            }
        }
    }
}

PyObject* run_interpreter_pyobject(PyObject *read_root) {

    typedef typename evm_t::evm_instances_t evm_instances_t;
    typedef arith_env_t<evm_params> arith_t;

    evm_instances_t         cpu_instances;
    #ifndef ONLY_CPU
    evm_instances_t tmp_gpu_instances, *gpu_instances;
    cgbn_error_report_t     *report;
    CUDA_CHECK(cudaDeviceReset());
    #endif

    arith_t arith(cgbn_report_monitor, 0);
  


    if (!PyDict_Check(read_root)) {
        PyErr_SetString(PyExc_TypeError, "Both arguments must be dictionaries.");
        return NULL;
    }

    block_data_t* block_data = getBlockDataFromPyObject(arith, PyDict_GetItemString(read_root, "env"));

    state_data_t* state_data = getStateDataFromPyObject(arith, PyDict_GetItemString(read_root, "pre"));
    
    transaction_data_t* transactions = getTransactionDataFromPyObject(arith, PyDict_GetItemString(read_root, "transaction"));

    size_t count = 1;

    // get instaces to run
    printf("Generating instances\n");
    evm_t::get_cpu_instances_plain_data(cpu_instances, state_data, block_data, transactions, count);
    printf("%d instances generated\n", cpu_instances.count);
    printf("\n print state data after get_cpu_instances_plain_data\n");
    for (size_t i = 0; i < state_data->no_accounts; i++) {
        world_state_t::print_account_t(arith, state_data->accounts[i]);
    }
    #ifndef ONLY_CPU
        evm_t::get_gpu_instances(tmp_gpu_instances, cpu_instances);
        CUDA_CHECK(cudaMalloc(&gpu_instances, sizeof(evm_instances_t)));
        CUDA_CHECK(cudaMemcpy(gpu_instances, &tmp_gpu_instances, sizeof(evm_instances_t), cudaMemcpyHostToDevice));
        size_t heap_size, stack_size;
        CUDA_CHECK(cgbn_error_report_alloc(&report));
        cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);
        heap_size = (size_t(2)<<30); // 2GB
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_size));
        // CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 256*1024));
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 64*1024));
        printf("Heap size: %zu\n", heap_size);
        cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);
        printf("Heap size: %zu\n", heap_size);
        printf("Running GPU kernel ...\n");
        CUDA_CHECK(cudaDeviceSynchronize());
        kernel_evm<evm_params><<<cpu_instances.count, evm_params::TPI>>>(report, gpu_instances);
        //CUDA_CHECK(cudaPeekAtLastError());
        // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("GPU kernel finished\n");
        CGBN_CHECK(report);

        // copy the results back to the CPU
        printf("Copying results back to CPU\n");
        CUDA_CHECK(cudaMemcpy(&tmp_gpu_instances, gpu_instances, sizeof(evm_instances_t), cudaMemcpyDeviceToHost));
        evm_t::get_cpu_instances_from_gpu_instances(cpu_instances, tmp_gpu_instances);
        printf("Results copied\n");
    #else
        printf("Running CPU EVM\n");
        // run the evm
        evm_t *evm = NULL;
        uint32_t tmp_error;
        for(uint32_t instance = 0; instance < cpu_instances.count; instance++) {
        printf("Running instance %d\n", instance);
        // print some test accounts 
        size_t no_accounts = cpu_instances.world_state_data->no_accounts > 2 ? 2 : cpu_instances.world_state_data->no_accounts;
        printf(" print directly from world_state_data\n");
        state_data_t *state_data_1 = cpu_instances.world_state_data;
        for (size_t i = 0; i < state_data_1->no_accounts; i++) {
            world_state_t::print_account_t(arith, state_data_1->accounts[i]);
        }
        evm = new evm_t(
            arith,
            cpu_instances.world_state_data,
            cpu_instances.block_data,
            cpu_instances.sha3_parameters,
            &(cpu_instances.transactions_data[instance]),
            &(cpu_instances.accessed_states_data[instance]),
            &(cpu_instances.touch_states_data[instance]),
            &(cpu_instances.logs_data[instance]),
            #ifdef TRACER
            &(cpu_instances.tracers_data[instance]),
            #endif
            instance,
            &(cpu_instances.errors[instance]));
        evm->run(tmp_error);
        delete evm;
        evm = NULL;
        }
        printf("CPU EVM finished\n");
    #endif


    // free the memory
    // printf("Freeing the memory ...\n");
    PyObject* write_root = evm_t::pyobject_from_evm_instances_t(arith, cpu_instances);
    evm_t::free_instances(cpu_instances);
    #ifndef ONLY_CPU
    CUDA_CHECK(cudaFree(gpu_instances));
    CUDA_CHECK(cgbn_error_report_free(report));
    CUDA_CHECK(cudaDeviceReset());
    #endif
    return write_root;
}

static PyObject* run_dict(PyObject* self, PyObject* args) {

    PyObject* read_root;

    // Parse the input PyObject* to get the Python object (dictionary)
    if (!PyArg_ParseTuple(args, "O", &read_root)) {
        return NULL; // If parsing fails, return NULL
    }

    if (!PyDict_Check(read_root)) {
        PyErr_SetString(PyExc_TypeError, "Parameter must be a dictionary.");
        return NULL;
    }

    PyObject* write_root = run_interpreter_pyobject(read_root);
    // Return the resulting PyObject* (no need for manual memory management on Python side)
    return write_root;
}

static PyObject* copy_dict(PyObject* self, PyObject* args) {

    PyObject* read_root;

    // Parse the input PyObject* to get the Python object (dictionary)
    if (!PyArg_ParseTuple(args, "O", &read_root)) {
        return NULL; // If parsing fails, return NULL
    }

    if (!PyDict_Check(read_root)) {
        PyErr_SetString(PyExc_TypeError, "Parameter must be a dictionary.");
        return NULL;
    }

    PyObject* write_root = PyDict_New();

    copy_dict_recursive(read_root, write_root);
    // Return the resulting PyObject* (no need for manual memory management on Python side)
    return write_root;
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

static PyObject* print_dict(PyObject* self, PyObject* args) {
    PyObject* dict;

    // Parse the Python argument (a dictionary)
    if (!PyArg_ParseTuple(args, "O", &dict)) return nullptr;

    // Ensure the object is a dictionary
    if (!PyDict_Check(dict)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a dictionary.");
        return nullptr;
    }
    // Start recursive printing with no indent
    print_dict_recursive(dict, 0);

    Py_RETURN_NONE;
}

// Method definition
static PyMethodDef ExampleMethods[] = {
    {"print_dict", print_dict, METH_VARARGS, "Print dictionary keys and values."},
    {"run_dict", run_dict, METH_VARARGS, "Run the interpreter with a JSON object."},
    {"copy_dict", copy_dict, METH_VARARGS, "Copy a dictionary."},
    {nullptr, nullptr, 0, nullptr}
};

// Module definition
static PyModuleDef examplemodule = {
    PyModuleDef_HEAD_INIT,
    "libcuevm",   // Module name
    nullptr,     // Module documentation
    -1,          // Size of per-interpreter state of the module
    ExampleMethods
};

// Initialization function
PyMODINIT_FUNC PyInit_libcuevm(void) {
    return PyModule_Create(&examplemodule);
}


extern "C" char* run_json_string(const char* read_json_string) {
    cJSON *read_root = cJSON_Parse(read_json_string);
    if (read_root == NULL) {
        // Handle parsing error (optional)
        return NULL;
    }

    cJSON *write_root = cJSON_CreateObject();

    // Assume run_interpreter modifies write_root based on read_root
    run_interpreter(read_root, write_root);
    cJSON_Delete(read_root);
    char *json_str = cJSON_Print(write_root);
    cJSON_Delete(write_root);

    return json_str; // Caller needs to free this memory
}

extern "C" void free_json_string(char* json_str) {
    // temporarily not working (invalid pointer) => potential memory leak
    // in the future, let python manage it with PyObject
    if (json_str) {
      cJSON_free(json_str); // Use the appropriate deallocation function
    }
}



