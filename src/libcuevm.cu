#include <getopt.h>
#include <fstream>
#include <chrono>
#include <Python.h>
#include "include/python_utils.h"
#include "utils.cu"
#include "evm.cuh"

using namespace python_utils;

PyObject* run_interpreter_pyobject(PyObject *read_roots) {

    typedef typename evm_t::evm_instances_t evm_instances_t;
    typedef arith_env_t<evm_params> arith_t;

    evm_instances_t         cpu_instances;
    #ifndef ONLY_CPU
    evm_instances_t tmp_gpu_instances, *gpu_instances;
    cgbn_error_report_t     *report;
    CUDA_CHECK(cudaDeviceReset());
    #endif

    arith_t arith(cgbn_report_monitor, 0);

    if (!PyList_Check(read_roots)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a list of dictionaries.");
        return NULL;
    }

    Py_ssize_t count = PyList_Size(read_roots);
    block_data_t **all_block_data ;
    state_data_t **all_state_data ;
    #ifndef ONLY_CPU
        CUDA_CHECK(cudaMallocManaged(
            (void **)&(all_block_data),
            sizeof(block_data_t*)*count
        ));

        CUDA_CHECK(cudaMallocManaged(
            (void **)&(all_state_data),
            sizeof(state_data_t*)*count
        ));
    #else
        all_block_data = (block_data_t**)malloc(sizeof(block_data_t*)*count);
        all_state_data = (state_data_t**)malloc(sizeof(state_data_t*)*count);
    #endif
    // transaction_data_t* transactions = getTransactionDataFromPyObject(arith, PyDict_GetItemString(read_root, "transaction"));
    transaction_data_t* all_transactions = getTransactionDataFromListofPyObject(arith, read_roots);

    for (Py_ssize_t idx = 0; idx < count; idx++) {

        PyObject *read_root = PyList_GetItem(read_roots, idx);
        if (!PyDict_Check(read_root)) {
            PyErr_SetString(PyExc_TypeError, "Each item in the list must be a dictionary.");
            return NULL;
        }

        // block_data_t* block_data = getBlockDataFromPyObject(arith, PyDict_GetItemString(read_root, "env"));
        all_block_data[idx] = getBlockDataFromPyObject(arith, PyDict_GetItemString(read_root, "env"));

        // state_data_t* state_data = getStateDataFromPyObject(arith, PyDict_GetItemString(read_root, "pre"));
        all_state_data[idx] = getStateDataFromPyObject(arith, PyDict_GetItemString(read_root, "pre"));

    }

    // get instances to run
    // printf("Generating instances\n");
    evm_t::get_cpu_instances_plain_data(cpu_instances, all_state_data, all_block_data, all_transactions, count);
    // printf("%d instances generated\n", cpu_instances.count);
    //   printf("\n print state data after get_cpu_instances_plain_data\n");

    // for (Py_ssize_t idx = 0; idx < count; idx++) {
    //     printf("State data %d\n", idx);
    //     for (size_t i = 0; i < all_state_data[idx]->no_accounts; i++) {
    //         world_state_t::print_account_t(arith, all_state_data[idx]->accounts[i]);
    //     }
    // }

    // printf("\n print transaction data after get_cpu_instances_plain_data\n");
    // for (Py_ssize_t idx = 0; idx < count; idx++) {
    //     printf("Transaction data %d\n", idx);
    //     transaction_t::print_transaction_data_t(arith, all_transactions[idx]);

    // }


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
        cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);
        // printf("Heap size: %zu\n", heap_size);
        // printf("Running GPU kernel ...\n");
        // CUDA_CHECK(cudaDeviceSynchronize());
        kernel_evm<evm_params><<<cpu_instances.count, evm_params::TPI>>>(report, gpu_instances);
        // CUDA_CHECK(cudaPeekAtLastError());
        // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("GPU kernel finished\n");
        CGBN_CHECK(report);

        // copy the results back to the CPU
        // printf("Copying results back to CPU\n");
        CUDA_CHECK(cudaMemcpy(&tmp_gpu_instances, gpu_instances, sizeof(evm_instances_t), cudaMemcpyDeviceToHost));
        evm_t::get_cpu_instances_from_gpu_instances(cpu_instances, tmp_gpu_instances);

        // printf("Results copied\n");
    #else
        // printf("Running CPU EVM\n");
        // run the evm
        evm_t *evm = NULL;
        uint32_t tmp_error;
        auto cpu_start = std::chrono::high_resolution_clock::now();
        for(uint32_t instance = 0; instance < cpu_instances.count; instance++) {
        // printf("Running instance %d\n", instance);
        evm = new evm_t(
            arith,
            cpu_instances.world_state_data[instance],
            cpu_instances.block_data[instance],
            cpu_instances.sha3_parameters,
            &(cpu_instances.transactions_data[instance]),
            &(cpu_instances.accessed_states_data[instance]),
            &(cpu_instances.touch_states_data[instance]),
            &(cpu_instances.logs_data[instance]),
            &(cpu_instances.return_data[instance]),
            #ifdef TRACER
            &(cpu_instances.tracers_data[instance]),
            #endif
            instance,
            &(cpu_instances.errors[instance]));
        evm->run(tmp_error);
        delete evm;
        evm = NULL;

        }
        // printf("CPU EVM finished\n");
        // auto cpu_end = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
        // printf("CPU EVM execution took %f ms\n", cpu_duration.count());
    #endif

    // free the memory
    // printf("Freeing the memory ...\n");
    // printf("Printing the results ...\n");
    // evm_t::print_evm_instances_t(arith, cpu_instances, true );

    PyObject* write_root = python_utils::pyobject_from_evm_instances_t(arith, cpu_instances);
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


    PyObject* write_root = run_interpreter_pyobject(read_root);
    // Return the resulting PyObject* (no need for manual memory management on Python side)
    return write_root;
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

// deprecated strings interfaces
/*
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
*/

