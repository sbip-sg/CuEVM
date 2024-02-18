#include <getopt.h>
#include <fstream>

#include <Python.h>

#include "utils.cu"
#include "evm.cuh"

using block_data_t = block_t::block_data_t;
using state_data_t = world_state_t::state_data_t;
using transaction_data_t = transaction_t::transaction_data_t;
using account_t = world_state_t::account_t;
void copy_dict_recursive(PyObject *read_root, PyObject *write_root);
void run_interpreter(cJSON *read_root, cJSON *write_root) {}
static PyObject* print_dict(PyObject* self, PyObject* args);

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
/*
"env" : {
            "currentBaseFee" : "0x0a",
            "currentBeaconRoot" : "0x0000000000000000000000000000000000000000000000000000000000000000",
            "currentCoinbase" : "0x2adc25665018aa1fe0e6bc666dac8fc2697ff9ba",
            "currentDifficulty" : "0x020000",
            "currentGasLimit" : "0x05f5e100",
            "currentNumber" : "0x01",
            "currentRandom" : "0x0000000000000000000000000000000000000000000000000000000000020000",
            "currentTimestamp" : "0x03e8",
            "currentWithdrawalsRoot" : "0x56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421",
            "previousHash" : "0x5e20a0453cecd065ea59c37ac63e079ee08998b6045136a8ce6635c7912ec0b6"
        },
        "pre" : {
            "0xa94f5374fce5edbc8e2a8697c15331677e6ebf0b" : {
                "balance" : "0x0ba1a9ce0ba1a9ce",
                "code" : "0x",
                "nonce" : "0x00",
                "storage" : {
                }
            },
            "0xcccccccccccccccccccccccccccccccccccccccc" : {
                "balance" : "0x0ba1a9ce0ba1a9ce",
                "code" : "0x600160019001600702600501600290046004906021900560170160030260059007600303600960110a60005560086000f3",
                "nonce" : "0x00",
                "storage" : {
                }
            }
        },
        "transaction" : {
            "data" : [
                "0x00"
            ],
            "gasLimit" : [
                "0x04c4b400"
            ],
            "gasPrice" : "0x0a",
            "nonce" : "0x00",
            "secretKey" : "0x45a915e4d060149eb4365960e6a7a45f334393093061116b197e3240065ff2d8",
            "sender" : "0xa94f5374fce5edbc8e2a8697c15331677e6ebf0b",
            "to" : "0xcccccccccccccccccccccccccccccccccccccccc",
            "value" : [
                "0x01"
            ]
        }
*/

void run_interpreter_pyobject(PyObject *read_root, PyObject *write_root) {

    typedef typename evm_t::evm_instances_t evm_instances_t;
    typedef arith_env_t<evm_params> arith_t;

    evm_instances_t         cpu_instances;
    #ifndef ONLY_CPU
    evm_instances_t tmp_gpu_instances, *gpu_instances;
    cgbn_error_report_t     *report;
    CUDA_CHECK(cudaDeviceReset());
    #endif

    arith_t arith(cgbn_report_monitor, 0);
  


    if (!PyDict_Check(read_root) || !PyDict_Check(write_root)) {
        PyErr_SetString(PyExc_TypeError, "Both arguments must be dictionaries.");
        return;
    }

    // read env from read_root
    PyObject *env = PyDict_GetItemString(read_root, "env");
    if (!env || !PyDict_Check(env)) {
        PyErr_SetString(PyExc_KeyError, "env key not found or not a dictionary.");
        return;
    }
    // print out env
    print_dict(nullptr, PyTuple_Pack(1, env));

 
    // construct blockt_t
    block_data_t* block_data = new block_data_t;


    // block_data.number = 0x01;
    // block_data.gas_limit = 0x05f5e100;
    // block_data.time_stamp = 0x03e8;
    // block_data.base_fee = 0x0a;
    // block_data.chain_id = 0x01;
    // block_data.previous_blocks[0] = 0x5e20a0453cecd065ea59c37ac63e079ee08998b6045136a8ce6635c7912ec0b6;
    // block_t* block_env = new block_t(arith, block_data);
    const char* block_data_const [8] = {
        "0x2adc25665018aa1fe0e6bc666dac8fc2697ff9ba",
        "0x020000",
        "0x01",
        "0x05f5e100",
        "0x03e8",
        "0x0a",
        "0x01",
        "0x5e20a0453cecd065ea59c37ac63e079ee08998b6045136a8ce6635c7912ec0b6"
    };
    
    arith.cgbn_memory_from_hex_string(block_data->coin_base,  block_data_const[0]);
    arith.cgbn_memory_from_hex_string(block_data->difficulty, block_data_const[1]);
    arith.cgbn_memory_from_hex_string(block_data->number, block_data_const[2]);
    arith.cgbn_memory_from_hex_string(block_data->gas_limit, block_data_const[3]);
    arith.cgbn_memory_from_hex_string(block_data->time_stamp, block_data_const[4]);
    arith.cgbn_memory_from_hex_string(block_data->base_fee, block_data_const[5]);
    arith.cgbn_memory_from_hex_string(block_data->chain_id, block_data_const[6]);

    //     /**
    //  * The account type.
    // */
    // typedef struct alignas(32)
    // {
    //     evm_word_t address; /**< The address of the account (YP: \f$a\f$) */
    //     evm_word_t balance; /**< The balance of the account (YP: \f$\sigma[a]_{b}\f$) */
    //     evm_word_t nonce; /**< The nonce of the account (YP: \f$\sigma[a]_{n}\f$) */
    //     size_t code_size; /**< The size of the bytecode (YP: \f$|b|\f$) */
    //     size_t storage_size; /**< The number of storage entries (YP: \f$|\sigma[a]_{s}|\f$) */
    //     uint8_t *bytecode; /**< The bytecode of the account (YP: \f$b\f$) */
    //     contract_storage_t *storage; /**< The storage of the account (YP: \f$\sigma[a]_{s}\f$) */
    // } account_t;

    // /**
    //  * The state data type.
    // */
    // typedef struct
    // {
    //     account_t *accounts; /**< The accounts in the state (YP: \f$\sigma\f$)*/
    //     size_t no_accounts; /**< The number of accounts in the state (YP: \f$|\sigma|\f$)*/
    // } state_data_t;
    //    "0xa94f5374fce5edbc8e2a8697c15331677e6ebf0b" : {
    //             "balance" : "0x0ba1a9ce0ba1a9ce",
    //             "code" : "0x",
    //             "nonce" : "0x00",
    //             "storage" : {
    //             }
    //         },
    //         "0xcccccccccccccccccccccccccccccccccccccccc" : {
    //             "balance" : "0x0ba1a9ce0ba1a9ce",
    //             "code" : "0x600160019001600702600501600290046004906021900560170160030260059007600303600960110a60005560086000f3",
    //             "nonce" : "0x00",
    //             "storage" : {
    //             }
    //         }
    // state_data_t *_content; /**< The content of the state */
        // allocate the content
    state_data_t* state_data;
    #ifndef ONLY_CPU
    CUDA_CHECK(cudaMallocManaged(
        (void **)&(state_data),
        sizeof(state_data_t)
    ));
    #else
    state_data = new state_data_t;
    #endif

    state_data->no_accounts = 2;
   // allocate the accounts
    #ifndef ONLY_CPU
    CUDA_CHECK(cudaMallocManaged(
        (void **)&(state_data->accounts),
        state_data->no_accounts * sizeof(account_t)
    ));
    #else
    state_data->accounts = new account_t[state_data->no_accounts];
    #endif

    arith.cgbn_memory_from_hex_string(state_data->accounts[0].address, "0xa94f5374fce5edbc8e2a8697c15331677e6ebf0b");
    arith.cgbn_memory_from_hex_string(state_data->accounts[0].balance, "0x0ba1a9ce0ba1a9ce");
    arith.cgbn_memory_from_hex_string(state_data->accounts[0].nonce, "0x00");
    state_data->accounts[0].code_size = 0;
    state_data->accounts[0].storage_size = 0;
    state_data->accounts[0].bytecode = NULL;

    arith.cgbn_memory_from_hex_string(state_data->accounts[1].address, "0xcccccccccccccccccccccccccccccccccccccccc");
    arith.cgbn_memory_from_hex_string(state_data->accounts[1].balance, "0x0ba1a9ce0ba1a9ce");
    arith.cgbn_memory_from_hex_string(state_data->accounts[1].nonce, "0x00");


    state_data->accounts[1].code_size = 0x31;
    state_data->accounts[1].storage_size = 0;
    if (state_data->accounts[1].code_size > 0)
    {
        #ifndef ONLY_CPU
        CUDA_CHECK(cudaMallocManaged(
            (void **)&(state_data->accounts[1].bytecode),
            state_data->accounts[1].code_size * sizeof(uint8_t)
        ));
        #else
        state_data->accounts[1].bytecode = new uint8_t[state_data->accounts[1].code_size];
        #endif
        hex_to_bytes(
            "0x600160019001600702600501600290046004906021900560170160030260059007600303600960110a60005560086000f3",
            state_data->accounts[1].bytecode,
            2 * state_data->accounts[1].code_size
        );
    }

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
    
    // read pre from read_root
    PyObject *pre = PyDict_GetItemString(read_root, "pre");
    if (!pre || !PyDict_Check(pre)) {
        PyErr_SetString(PyExc_KeyError, "pre key not found or not a dictionary.");
        return;
    }
    // print out pre
    print_dict(nullptr, PyTuple_Pack(1, pre));
    
    // read transaction from read_root
    PyObject *transaction = PyDict_GetItemString(read_root, "transaction");
    if (!transaction || !PyDict_Check(transaction)) {
        PyErr_SetString(PyExc_KeyError, "transaction key not found or not a dictionary.");
        return;
    }
    // print out transaction
    print_dict(nullptr, PyTuple_Pack(1, transaction));

  
    // get instaces to run
    printf("Generating instances\n");
    evm_t::get_cpu_instances_plain_data(cpu_instances, state_data, block_data, transactions, count);
    printf("%d instances generated\n", cpu_instances.count);

    #ifndef ONLY_CPU
    evm_t::get_gpu_instances(tmp_gpu_instances, cpu_instances);
    CUDA_CHECK(cudaMalloc(&gpu_instances, sizeof(evm_instances_t)));
    CUDA_CHECK(cudaMemcpy(gpu_instances, &tmp_gpu_instances, sizeof(evm_instances_t), cudaMemcpyHostToDevice));
    #endif

    // create a cgbn_error_report for CGBN to report back errors
    #ifndef ONLY_CPU
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


    printf("Json string printed\n");

    // free the memory
    printf("Freeing the memory ...\n");
    evm_t::free_instances(cpu_instances);
    #ifndef ONLY_CPU
    CUDA_CHECK(cudaFree(gpu_instances));
    CUDA_CHECK(cgbn_error_report_free(report));
    CUDA_CHECK(cudaDeviceReset());
    #endif

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

    PyObject* write_root = PyDict_New();

    run_interpreter_pyobject(read_root, write_root);
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



