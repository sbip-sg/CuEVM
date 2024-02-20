#include <getopt.h>
#include <fstream>

#include <Python.h>
#include "include/python_utils.h"
#include "utils.cu"
#include "evm.cuh"

using namespace python_utils;

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
        // size_t no_accounts = cpu_instances.world_state_data->no_accounts > 2 ? 2 : cpu_instances.world_state_data->no_accounts;
        // printf(" print directly from world_state_data\n");
        // state_data_t *state_data_1 = cpu_instances.world_state_data;
        // for (size_t i = 0; i < state_data_1->no_accounts; i++) {
        //     world_state_t::print_account_t(arith, state_data_1->accounts[i]);
        // }
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

    if (!PyDict_Check(read_root)) {
        PyErr_SetString(PyExc_TypeError, "Parameter must be a dictionary.");
        return NULL;
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

