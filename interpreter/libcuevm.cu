#include <CuEVM/utils/python_utils.h>
#include <Python.h>
#include <getopt.h>

#include <CuEVM/utils/evm_utils.cuh>
#include <chrono>
#include <fstream>

using namespace python_utils;

PyObject* run_interpreter_pyobject(PyObject* read_roots, uint32_t skip_trace_parsing) {
    CuEVM::evm_instance_t* instances_data;
    CuEVM::ArithEnv arith(cgbn_no_checks, 0);
    // printf("Running the interpreter\n");
#ifndef GPU
    printf("CPU libcuevm is not supported at the moment\n");
    return NULL;
#endif
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaDeviceReset());
    // printf("Running on GPU\n");
    cgbn_error_report_t* report = nullptr;
    // CUDA_CHECK(cgbn_error_report_alloc(&report));
    cudaEvent_t start, stop;
    float milliseconds = 0;

    size_t size_value;
    cudaDeviceGetLimit(&size_value, cudaLimitStackSize);
    // printf("current stack size %zu\n", size_value);
    cudaDeviceGetLimit(&size_value, cudaLimitStackSize);
    // printf("current heap size %zu\n", size_value);
    size_t heap_size = (size_t(500) << 20);  // 500MB
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_size));
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 2 * 1024));
    cudaDeviceGetLimit(&size_value, cudaLimitStackSize);
    // printf("current stack size %zu\n", size_value);
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

    python_utils::get_evm_instances_from_PyObject(instances_data, read_roots, num_instances);
    // printf("print simplified trace data host\n");
    // instances_data[0].simplified_trace_data_ptr->print();

    uint32_t num_blocks = (num_instances + CGBN_IBP - 1) / (CGBN_IBP);
    // printf("Running %d instances on GPU, num blocks %d, threads per block %d\n", num_instances, num_blocks,
    //        CGBN_TPI * CGBN_IBP);
    // run the evm
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    CuEVM::kernel_evm_multiple_instances<<<num_blocks, CGBN_TPI * CGBN_IBP>>>(report, instances_data, num_instances);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f milliseconds\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaGetLastError());
    // printf("GPU kernel finished\n");
    // CGBN_CHECK(report);

    // printf("\n\ntesting world state printing on host\n\n");
    // instances_data[0].serialized_worldstate_data_ptr->print();
    // printf("print simplified trace data host\n");
    // for (uint32_t i = 0; i < num_instances; i++) {
    //     printf("\n\ninstance %d\n", i);
    //     instances_data[i].simplified_trace_data_ptr->print();
    // }
    PyObject* write_root;
    if (!skip_trace_parsing) {
        write_root = python_utils::pyobject_from_evm_instances(instances_data, num_instances);
    } else {
        write_root = PyDict_New();
    }

    CuEVM::free_evm_instances(instances_data, num_instances, managed);

    CUDA_CHECK(cgbn_error_report_free(report));
    CUDA_CHECK(cudaDeviceReset());
    return write_root;
}

static PyObject* run_dict(PyObject* self, PyObject* args) {
    PyObject* read_root;
    uint32_t skip_trace_parsing = 0;
    // Parse the input PyObject* to get the Python object (dictionary)
    // if (!PyArg_ParseTuple(args, "O", &read_root)) {
    //     return NULL;  // If parsing fails, return NULL
    // }
    // Parse the input PyObject* to get the Python object (dictionary) and optionally the boolean flag
    if (!PyArg_ParseTuple(args, "O|i", &read_root, &skip_trace_parsing)) {
        return NULL;  // If parsing fails, return NULL
    }

    PyObject* write_root = run_interpreter_pyobject(read_root, skip_trace_parsing);
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
