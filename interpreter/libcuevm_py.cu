#include <CuEVM/utils/python_utils.h>
#include <getopt.h>

#include <CuEVM/utils/evm_utils.cuh>
#include <chrono>
#include <fstream>

using namespace python_utils;

// Add this near the top of the file, after the includes
static int call_counter = 0;

PyObject* run_interpreter_pyobject(PyObject* read_roots, uint32_t skip_trace_parsing, uint32_t copy_state_data,
                                   uint32_t reuse_state_data) {
    // CuEVM::evm_instance_t* instances_data;
    printf("run configuration skip_trace_parsing: %d, copy_state_data: %d, reuse_state_data: %d\n", skip_trace_parsing,
           copy_state_data, reuse_state_data);
    CUDA_CHECK(cudaSetDevice(0));
    if (!reuse_state_data) {
        CUDA_CHECK(cudaDeviceReset());

        size_t heap_size = (size_t(1) << 32);  // 4GB
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_size));
        CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 4 * 1024));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    cudaEvent_t start, stop;
    float milliseconds = 0;

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
    printf("num_accounts: %d\n", num_accounts);

    CuEVM::transaction::TransactionList* all_transactions = python_utils::get_evm_instances_from_PyObject(
        read_roots, num_instances, reuse_state_data, copy_state_data, call_counter);
    if (!reuse_state_data || call_counter == 0) {
        printf("create memory pool \n");
        CuEVM::memory_pool::create_memory_pool(num_instances, num_accounts);
    }

    uint32_t num_blocks = (num_instances + INSTANCES_PER_BLOCK - 1) / (INSTANCES_PER_BLOCK);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    CUDA_CHECK(cudaDeviceSynchronize());

    CuEVM::kernel_evm_multiple_instances<<<num_blocks, INSTANCES_PER_BLOCK>>>(all_transactions, num_instances,
                                                                              copy_state_data);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f milliseconds\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CUDA_CHECK(cudaGetLastError());
    auto start_time = std::chrono::high_resolution_clock::now();

    PyObject* write_root;
    if (!skip_trace_parsing) {
        write_root = python_utils::pyobject_from_evm_instances(num_instances, copy_state_data);
    } else {
        write_root = PyDict_New();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    printf("Time taken to create write_root: %lld milliseconds\n", duration.count());

    // CuEVM::free_evm_instances(instances_data, num_instances);
    if (!reuse_state_data) {
        printf("reset device\n");
        CUDA_CHECK(cudaDeviceReset());
    } else {
        // free other memory than the state data
        CuEVM::memory_pool::clear_memory_pool();
        python_utils::freeTransactionList(all_transactions);
        python_utils::freeTraceData(copy_state_data);
    }
    // Increment counter at the start of each call
    call_counter++;
    printf("Call number: %d\n", call_counter);

    return write_root;
}

static PyObject* run_dict(PyObject* self, PyObject* args) {
    PyObject* read_root;
    uint32_t skip_trace_parsing = 0;
    uint32_t copy_state_data = 0;
    uint32_t reuse_state_data = 0;

    if (!PyArg_ParseTuple(args, "O|iii", &read_root, &skip_trace_parsing, &copy_state_data, &reuse_state_data)) {
        printf("parse tuple failed\n");
        return NULL;  // If parsing fails, return NULL
    }

    PyObject* write_root = run_interpreter_pyobject(read_root, skip_trace_parsing, copy_state_data, reuse_state_data);

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

// Add a new method to get the counter value
static PyObject* get_call_count(PyObject* self, PyObject* args) { return PyLong_FromLong(call_counter); }

// Method definition
static PyMethodDef ExampleMethods[] = {
    {"print_dict", print_dict, METH_VARARGS, "Print dictionary keys and values."},
    {"run_dict", run_dict, METH_VARARGS, "Run the interpreter with a JSON object."},
    {"get_call_count", get_call_count, METH_VARARGS, "Get the number of times run_interpreter_pyobject was called."},
    {nullptr, nullptr, 0, nullptr}};

// Module definition
static PyModuleDef examplemodule = {PyModuleDef_HEAD_INIT,
                                    "libcuevm",  // Module name
                                    nullptr,     // Module documentation
                                    -1,          // Size of per-interpreter state of the module
                                    ExampleMethods};

// Initialization function
PyMODINIT_FUNC PyInit_libcuevm(void) { return PyModule_Create(&examplemodule); }
