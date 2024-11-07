#include <CuEVM/utils/python_utils.h>
#include <Python.h>
#include <getopt.h>

#include <CuEVM/utils/evm_utils.cuh>
#include <chrono>
#include <fstream>

using namespace python_utils;
// define the kernel function
__global__ void kernel_evm_multiple_instances(cgbn_error_report_t* report, CuEVM::evm_instance_t* instances,
                                              uint32_t count) {
    int32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / CuEVM::cgbn_tpi;
    if (instance >= count) return;
    CuEVM::ArithEnv arith(cgbn_no_checks, report, instance);
    // CuEVM::bn_t test;
    // printf("print simplified trace data device\n");
    // instances[instance].simplified_trace_data_ptr->print();
// printf("new instance %d\n", instance);
#ifdef EIP_3155
    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    printf("instance %d\n", instance);
    printf("world state\n");
    instances[instance].world_state_data_ptr->print();
    printf("touch state\n");
    instances[instance].touch_state_data_ptr->print();
    printf("instance %d\n", instance);
    printf("transaction\n");
    instances[instance].transaction_ptr->print();
    __ONE_GPU_THREAD_WOSYNC_END__
#endif
    __SHARED_MEMORY__ CuEVM::evm_message_call_t shared_message_call;
    __SHARED_MEMORY__ CuEVM::evm_word_t shared_stack[CuEVM::shared_stack_size];
    CuEVM::evm_t* evm = new CuEVM::evm_t(arith, instances[instance], &shared_message_call, shared_stack);
    CuEVM::cached_evm_call_state cached_state(arith, evm->call_state_ptr);
    // printf("\nevm->run(arith) instance %d\n", instance);
    // printf("print simplified trace data device inside evm\n");
    // evm->simplified_trace_data_ptr->print();
    __SYNC_THREADS__
    evm->run(arith, cached_state);

#ifdef EIP_3155
    if (instance == 0) {
        __ONE_GPU_THREAD_BEGIN__
        // instances[0].tracer_ptr->print(arith);
        instances[0].tracer_ptr->print_err();
        __ONE_GPU_THREAD_WOSYNC_END__
    }
#endif
    // print the final world state
    // __ONE_GPU_THREAD_WOSYNC_BEGIN__
    // instances[instance].world_state_data_ptr->print();
    // printf("simplified trace data\n");
    // instances[instance].simplified_trace_data_ptr->print();
    // __ONE_GPU_THREAD_WOSYNC_END__
    // delete evm;
    // evm = nullptr;
}

PyObject* run_interpreter_pyobject(PyObject* read_roots) {
    CuEVM::evm_instance_t* instances_data;
    CuEVM::ArithEnv arith(cgbn_no_checks, 0);
    printf("Running the interpreter\n");
#ifndef GPU
    printf("CPU libcuevm is not supported at the moment\n");
    return NULL;
#endif
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaDeviceReset());
    printf("Running on GPU\n");
    cgbn_error_report_t* report;
    CUDA_CHECK(cgbn_error_report_alloc(&report));
    cudaEvent_t start, stop;
    float milliseconds = 0;

    size_t size_value;
    cudaDeviceGetLimit(&size_value, cudaLimitStackSize);
    printf("current stack size %zu\n", size_value);
    cudaDeviceGetLimit(&size_value, cudaLimitStackSize);
    printf("current heap size %zu\n", size_value);
    size_t heap_size = (size_t(500) << 20);  // 500MB
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_size));
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 2 * 1024));
    cudaDeviceGetLimit(&size_value, cudaLimitStackSize);
    printf("current stack size %zu\n", size_value);
    CUDA_CHECK(cudaDeviceSynchronize());
    // CUDA_CHECK(cudaEventCreate(&start));
    // CUDA_CHECK(cudaEventCreate(&stop));

    // read the json file with the global state

    uint32_t num_instances = 0;
    uint32_t managed = 1;

    if (!PyList_Check(read_roots)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a list of dictionaries.");
        return NULL;
    }

    Py_ssize_t count = PyList_Size(read_roots);
    // transaction_data_t* transactions = getTransactionDataFromPyObject(arith, PyDict_GetItemString(read_root,
    // "transaction"));
    // void get_evm_instances_from_PyObject(CuEVM::evm_instance_t * &evm_instances, const cJSON* test_json,
    //                                      uint32_t& num_instances);

    python_utils::get_evm_instances_from_PyObject(instances_data, read_roots, num_instances);
    // printf("print simplified trace data host\n");
    // instances_data[0].simplified_trace_data_ptr->print();
    printf("Running on GPU %d %d\n", num_instances, CuEVM::cgbn_tpi);
    // run the evm
    kernel_evm_multiple_instances<<<num_instances, CuEVM::cgbn_tpi>>>(report, instances_data, num_instances);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    printf("GPU kernel finished\n");
    CGBN_CHECK(report);

    // printf("\n\ntesting world state printing on host\n\n");
    // instances_data[0].serialized_worldstate_data_ptr->print();
    // printf("print simplified trace data host\n");
    // for (uint32_t i = 0; i < num_instances; i++) {
    //     printf("\n\ninstance %d\n", i);
    //     instances_data[i].simplified_trace_data_ptr->print();
    // }
    PyObject* write_root = python_utils::pyobject_from_evm_instances(instances_data, num_instances);

    CuEVM::free_evm_instances(instances_data, num_instances, managed);

    CUDA_CHECK(cgbn_error_report_free(report));
    CUDA_CHECK(cudaDeviceReset());
    return write_root;
    // Py_RETURN_NONE;
}

static PyObject* run_dict(PyObject* self, PyObject* args) {
    PyObject* read_root;

    // Parse the input PyObject* to get the Python object (dictionary)
    if (!PyArg_ParseTuple(args, "O", &read_root)) {
        return NULL;  // If parsing fails, return NULL
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
static PyMethodDef ExampleMethods[] = {{"print_dict", print_dict, METH_VARARGS, "Print dictionary keys and values."},
                                       {"run_dict", run_dict, METH_VARARGS, "Run the interpreter with a JSON object."},
                                       {nullptr, nullptr, 0, nullptr}};

// Module definition
static PyModuleDef examplemodule = {PyModuleDef_HEAD_INIT,
                                    "libcuevm",  // Module name
                                    nullptr,     // Module documentation
                                    -1,          // Size of per-interpreter state of the module
                                    ExampleMethods};

// Initialization function
PyMODINIT_FUNC PyInit_libcuevm(void) { return PyModule_Create(&examplemodule); }

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
