#include <CuEVM/utils/python_utils.h>
#include <Python.h>
#include <getopt.h>

#include <CuEVM/utils/evm_utils.cuh>
#include <chrono>
#include <fstream>

using namespace python_utils;

PyObject* run_interpreter_pyobject(PyObject* read_roots, uint32_t skip_trace_parsing) {
    CuEVM::evm_instance_t* instances_data;

#ifndef GPU
    printf("CPU libcuevm is not supported at the moment\n");
    return NULL;
#endif

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaDeviceReset());

    cudaEvent_t start, stop;
    float milliseconds = 0;

    size_t heap_size = (size_t(500) << 20);  // 500MB
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_size));
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 2 * 1024));
    CUDA_CHECK(cudaDeviceSynchronize());

    // read the json file with the global state
    uint32_t num_instances = 0;
    uint32_t managed = 1;

    // Some basic data validity checking
    // todo we accept a list of Json objects, each representing a transaction
    // while here we assume that each transaction contains the same `pre`-state
    if (!PyList_Check(read_roots)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a list of dictionaries.");
        return NULL;
    }

    PyObject* first_item = PyList_GetItem(read_roots, 0);

    if (!PyDict_Check(first_item)) {
      PyErr_SetString(PyExc_TypeError, "First item in the list must be a dictionary.");
      return NULL;
    }

    PyObject* data = PyDict_GetItemString(first_item, "pre");

    if (!PyDict_Check(data)) {
      PyErr_SetString(PyExc_TypeError, "pre must be a dictionary.");
      return NULL;
    }

    auto num_accounts = PyDict_Size(data);

    python_utils::get_evm_instances_from_PyObject(instances_data, read_roots, num_instances);
    CuEVM::memory_pool::create_memory_pool(num_instances, num_accounts);

    uint32_t num_blocks = (num_instances + INSTANCES_PER_BLOCK - 1) / (INSTANCES_PER_BLOCK);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    CUDA_CHECK(cudaDeviceSynchronize());

    CuEVM::kernel_evm_multiple_instances<<<num_blocks, INSTANCES_PER_BLOCK>>>(instances_data->state_db_ptr, instances_data->transaction_list_ptr,
                                                                              instances_data->simplified_trace_data_ptr,
                                                                              num_instances);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f milliseconds\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CUDA_CHECK(cudaGetLastError());

    PyObject* write_root;
    if (!skip_trace_parsing) {
        write_root = python_utils::pyobject_from_evm_instances(instances_data, num_instances);
    } else {
        write_root = PyDict_New();
    }

    CuEVM::free_evm_instances(instances_data, num_instances);

    CUDA_CHECK(cudaDeviceReset());
    return write_root;
}

static PyObject* run_dict(PyObject* self, PyObject* args) {
    PyObject* read_root;
    uint32_t skip_trace_parsing = 0;

    if (!PyArg_ParseTuple(args, "O|i", &read_root, &skip_trace_parsing)) {
        return NULL;  // If parsing fails, return NULL
    }

    PyObject* write_root = run_interpreter_pyobject(read_root, skip_trace_parsing);
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
