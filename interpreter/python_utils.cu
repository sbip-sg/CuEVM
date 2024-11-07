
#include <CuEVM/utils/python_utils.h>

void copy_dict_recursive(PyObject* read_root, PyObject* write_root);
static PyObject* print_dict(PyObject* self, PyObject* args);

namespace python_utils {
using namespace CuEVM;
CuEVM::block_info_t* getBlockDataFromPyObject(PyObject* data) {
    // construct blockt_t

    block_info_t* block_data;

    CUDA_CHECK(cudaMallocManaged((void**)&(block_data), sizeof(block_info_t)));

    const char* base_fee = GET_STR_FROM_DICT_WITH_DEFAULT(data, "currentBaseFee", DefaultBlock::BaseFee);
    const char* coin_base = GET_STR_FROM_DICT_WITH_DEFAULT(data, "currentCoinbase", DefaultBlock::CoinBase);
    const char* difficulty = GET_STR_FROM_DICT_WITH_DEFAULT(data, "currentDifficulty", DefaultBlock::Difficulty);
    const char* block_number = GET_STR_FROM_DICT_WITH_DEFAULT(data, "currentNumber", DefaultBlock::BlockNumber);
    const char* gas_limit = GET_STR_FROM_DICT_WITH_DEFAULT(data, "currentGasLimit", DefaultBlock::GasLimit);
    const char* time_stamp = GET_STR_FROM_DICT_WITH_DEFAULT(data, "currentTimestamp", DefaultBlock::TimeStamp);
    const char* previous_hash = GET_STR_FROM_DICT_WITH_DEFAULT(data, "previousHash", DefaultBlock::PreviousHash);
    block_data->coin_base.from_hex(coin_base);
    block_data->difficulty.from_hex(difficulty);
    block_data->number.from_hex(block_number);
    block_data->gas_limit.from_hex(gas_limit);
    block_data->time_stamp.from_hex(time_stamp);
    block_data->base_fee.from_hex(base_fee);
    block_data->chain_id.from_hex("0x01");
    block_data->previous_blocks[0].hash.from_hex(previous_hash);
    return block_data;
}

void print_dict_recursive(PyObject* dict, int indent_level) {
    PyObject *key, *value;
    Py_ssize_t pos = 0;

    // Iterate over dictionary items
    while (PyDict_Next(dict, &pos, &key, &value)) {
        // Print indent
        for (int i = 0; i < indent_level; ++i) {
            printf("    ");  // 4 spaces for each indent level
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
const char* adjust_hex_string(const char* hex_string) {
    if (strlen(hex_string) >= 2 && (hex_string[0] == '0' && (hex_string[1] == 'x' || hex_string[1] == 'X')))
        hex_string += 2;  // Skip the "0x" prefix
    if (strlen(hex_string) % 2 != 0) {
        printf("Invalid hex_string length\n");
        return NULL;
    }
    return hex_string;
}
void hex_to_bytes(const char* hex_string, uint8_t* byte_array, size_t length) {
    for (size_t idx = 0; idx < length; idx += 2) {
        sscanf(&hex_string[idx], "%2hhx", &byte_array[idx / 2]);
    }
}
CuEVM::evm_transaction_t* getTransactionDataFromListofPyObject(PyObject* read_roots) {
    Py_ssize_t count = PyList_Size(read_roots);
    CuEVM::evm_transaction_t* transactions;

    CUDA_CHECK(cudaMallocManaged((void**)&(transactions), count * sizeof(CuEVM::evm_transaction_t)));

    for (Py_ssize_t idx = 0; idx < count; idx++) {
        PyObject* read_root = PyList_GetItem(read_roots, idx);
        if (!PyDict_Check(read_root)) {
            PyErr_SetString(PyExc_TypeError, "Each item in the list must be a dictionary.");
            return NULL;
        }

        PyObject* data = PyDict_GetItemString(read_root, "transaction");
        // get data size
        PyObject* tx_data = PyDict_GetItemString(data, "data");
        PyObject* tx_gas_limit = PyDict_GetItemString(data, "gasLimit");
        PyObject* tx_value = PyDict_GetItemString(data, "value");
        if (tx_data == NULL || !PyList_Check(tx_data)) {
            printf("Invalid transaction data\n");
            return NULL;
        }

        // printf("Transaction count: %d\n", count);

        CuEVM::evm_transaction_t* template_transaction = new evm_transaction_t();
        memset(template_transaction, 0, sizeof(CuEVM::evm_transaction_t));

        uint8_t type;

        type = 0;
        template_transaction->nonce.from_hex(PyUnicode_AsUTF8(PyDict_GetItemString(data, "nonce")));
        template_transaction->to.from_hex(PyUnicode_AsUTF8(PyDict_GetItemString(data, "to")));
        template_transaction->sender.from_hex(PyUnicode_AsUTF8(PyDict_GetItemString(data, "sender")));
        template_transaction->max_fee_per_gas = 0;
        template_transaction->max_priority_fee_per_gas = 0;
        template_transaction->gas_price.from_hex("0x0a");

        template_transaction->type = type;

        // char *bytes_string = NULL;
        memcpy(&(transactions[idx]), template_transaction, sizeof(CuEVM::evm_transaction_t));
        transactions[idx].gas_limit.from_hex(PyUnicode_AsUTF8(PyList_GetItem(tx_gas_limit, 0)));
        transactions[idx].value.from_hex(PyUnicode_AsUTF8(PyList_GetItem(tx_value, 0)));

        const char* bytes_string = PyUnicode_AsUTF8(PyList_GetItem(tx_data, 0));

        bytes_string = adjust_hex_string(bytes_string);
        transactions[idx].data_init.size = strlen(bytes_string) / 2;
        if (transactions[idx].data_init.size > 0) {
            CUDA_CHECK(cudaMallocManaged((void**)&(transactions[idx].data_init.data),
                                         transactions[idx].data_init.size * sizeof(uint8_t)));
            hex_to_bytes(bytes_string, transactions[idx].data_init.data, 2 * transactions[idx].data_init.size);
        } else {
            transactions[idx].data_init.data = NULL;
        }

        delete template_transaction;
    }

    return transactions;
}

CuEVM::evm_transaction_t* getTransactionDataFromPyObject(PyObject* data, size_t& instances_count) {
    // get data size
    PyObject* tx_data = PyDict_GetItemString(data, "data");
    PyObject* tx_gas_limit = PyDict_GetItemString(data, "gasLimit");
    PyObject* tx_value = PyDict_GetItemString(data, "value");
    if (tx_data == NULL || !PyList_Check(tx_data)) {
        printf("Invalid transaction data\n");
        return NULL;
    }
    size_t tx_data_count = PyList_Size(tx_data);
    size_t gas_limit_count = PyList_Size(tx_gas_limit);
    size_t value_count = PyList_Size(tx_value);
    size_t count = max(max(tx_data_count, gas_limit_count), value_count);
    instances_count = count;
    // printf("Transaction count: %d\n", count);
    CuEVM::evm_transaction_t* transactions;

    CUDA_CHECK(cudaMallocManaged((void**)&(transactions), count * sizeof(CuEVM::evm_transaction_t)));

    evm_transaction_t* template_transaction = new evm_transaction_t();
    memset(template_transaction, 0, sizeof(evm_transaction_t));

    uint8_t type;
    size_t idx = 0;

    type = 0;

    template_transaction->nonce.from_hex(PyUnicode_AsUTF8(PyDict_GetItemString(data, "nonce")));
    template_transaction->to.from_hex(PyUnicode_AsUTF8(PyDict_GetItemString(data, "to")));
    template_transaction->sender.from_hex(PyUnicode_AsUTF8(PyDict_GetItemString(data, "sender")));

    template_transaction->max_fee_per_gas = 0;
    template_transaction->max_priority_fee_per_gas = 0;
    template_transaction->gas_price.from_hex("0x0a");

    template_transaction->type = type;

    size_t index;
    // char *bytes_string = NULL;
    for (idx = 0; idx < count; idx++) {
        memcpy(&(transactions[idx]), template_transaction, sizeof(evm_transaction_t));

        transactions[idx].gas_limit.from_hex(
            PyUnicode_AsUTF8(PyList_GetItem(tx_gas_limit, idx < gas_limit_count ? idx : 0)));
        transactions[idx].value.from_hex(PyUnicode_AsUTF8(PyList_GetItem(tx_value, idx < value_count ? idx : 0)));

        const char* bytes_string = PyUnicode_AsUTF8(PyList_GetItem(tx_data, idx < tx_data_count ? idx : 0));

        bytes_string = adjust_hex_string(bytes_string);
        transactions[idx].data_init.size = strlen(bytes_string) / 2;
        if (transactions[idx].data_init.size > 0) {
            CUDA_CHECK(cudaMallocManaged((void**)&(transactions[idx].data_init.data),
                                         transactions[idx].data_init.size * sizeof(uint8_t)));

            hex_to_bytes(bytes_string, transactions[idx].data_init.data, 2 * transactions[idx].data_init.size);
        } else {
            transactions[idx].data_init.data = NULL;
        }
    }
    delete template_transaction;
    return transactions;
}

CuEVM::state_t* getStateDataFromPyObject(PyObject* data) {
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    CuEVM::state_t* state_data;

    CUDA_CHECK(cudaMallocManaged((void**)&(state_data), sizeof(CuEVM::state_t)));

    state_data->no_accounts = PyDict_Size(data);

    CUDA_CHECK(cudaMallocManaged((void**)&(state_data->accounts), state_data->no_accounts * sizeof(CuEVM::account_t)));

    while (PyDict_Next(data, &pos, &key, &value)) {
        int account_index = pos - 1;  // Adjust index since PyDict_Next increments pos
        const char* address_str = PyUnicode_AsUTF8(key);

        // Extract balance, nonce, and code
        const char* balance = GET_STR_FROM_DICT_WITH_DEFAULT(value, "balance", "0x0");
        const char* nonce = GET_STR_FROM_DICT_WITH_DEFAULT(value, "nonce", "0x0");
        const char* code = GET_STR_FROM_DICT_WITH_DEFAULT(value, "code", "");

        // Initialize account details
        state_data->accounts[account_index].address.from_hex(address_str);
        state_data->accounts[account_index].balance.from_hex(balance);
        state_data->accounts[account_index].nonce.from_hex(nonce);
        // arith.cgbn_memory_from_hex_string(state_data->accounts[account_index].balance, balance);
        // arith.cgbn_memory_from_hex_string(state_data->accounts[account_index].nonce, nonce);
        // printf("Account %d: %s\n", account_index, address_str);
        // Bytecode handling
        // code = adjust_hex_string(code);
        state_data->accounts[account_index].byte_code.from_hex(code, LITTLE_ENDIAN, NO_PADDING, 1);

        // Storage handling (assuming a function to handle storage initialization)
        PyObject* storage_dict = PyDict_GetItemString(value, "storage");
        uint32_t storage_size = PyDict_Size(storage_dict);
        if (storage_dict && storage_size > 0) {
            // Assuming you have a function to handle storage initialization
            // initialize_storage(&state_data->accounts[account_index], storage_dict);

            uint32_t capacity = CuEVM::initial_storage_capacity / 2;
            do {
                capacity *= 2;
            } while (capacity < storage_size);

            CUDA_CHECK(cudaMallocManaged(&state_data->accounts[account_index].storage.storage,
                                         capacity * sizeof(storage_element_t)));
            state_data->accounts[account_index].storage.capacity = capacity;
            state_data->accounts[account_index].storage.size = storage_size;
            PyObject *key_storage, *value_storage;

            Py_ssize_t pos_storage = 0;
            Py_ssize_t idx_storage = 0;
            // allocate the storage
            size_t idx = 0;
            // Iterate through the dictionary
            while (PyDict_Next(storage_dict, &pos_storage, &key_storage, &value_storage)) {
                // Increase reference count for key and value before storing them
                Py_INCREF(key_storage);
                Py_INCREF(value_storage);

                PyObject* keyStrObj = PyObject_Str(key_storage);
                const char* keyStr = PyUnicode_AsUTF8(keyStrObj);
                state_data->accounts[account_index].storage.storage[idx].key.from_hex(keyStr);
                Py_DECREF(keyStrObj);

                PyObject* valueStrObj = PyObject_Str(value_storage);
                const char* valueStr = PyUnicode_AsUTF8(valueStrObj);
                state_data->accounts[account_index].storage.storage[idx].value.from_hex(valueStr);
                Py_DECREF(valueStrObj);
                idx++;
            }

        } else {
            state_data->accounts[account_index].storage.capacity = 0;
            state_data->accounts[account_index].storage.size = 0;
            state_data->accounts[account_index].storage.storage = nullptr;
        }
    }

    return state_data;
}

void get_evm_instances_from_PyObject(CuEVM::evm_instance_t*& evm_instances, PyObject* read_roots,
                                     uint32_t& num_instances) {
    uint32_t num_transactions = PyList_Size(read_roots);
    // CuEVM::transaction::get_transactions(arith, transactions_ptr, test_json, num_transactions, managed,
    //                                      world_state_data_ptr);

    evm_transaction_t* all_transactions = getTransactionDataFromListofPyObject(read_roots);

    // generate the evm instances

    CUDA_CHECK(cudaMallocManaged(&evm_instances, num_transactions * sizeof(evm_instance_t)));
    for (Py_ssize_t index = 0; index < num_transactions; index++) {
        PyObject* read_root = PyList_GetItem(read_roots, index);
        if (!PyDict_Check(read_root)) {
            PyErr_SetString(PyExc_TypeError, "Each item in the list must be a dictionary.");
            return;
        }

        evm_instances[index].world_state_data_ptr = getStateDataFromPyObject(PyDict_GetItemString(read_root, "pre"));
        evm_instances[index].block_info_ptr = getBlockDataFromPyObject(PyDict_GetItemString(read_root, "env"));
        evm_instances[index].transaction_ptr = &all_transactions[index];

        CuEVM::state_access_t* access_state = new CuEVM::state_access_t();
        CUDA_CHECK(cudaMallocManaged(&evm_instances[index].touch_state_data_ptr, sizeof(CuEVM::state_access_t)));
        memcpy(evm_instances[index].touch_state_data_ptr, access_state, sizeof(CuEVM::state_access_t));
        delete access_state;
        CuEVM::log_state_data_t* log_state = new CuEVM::log_state_data_t();
        CUDA_CHECK(cudaMallocManaged(&evm_instances[index].log_state_ptr, sizeof(CuEVM::log_state_data_t)));
        memcpy(evm_instances[index].log_state_ptr, log_state, sizeof(CuEVM::log_state_data_t));
        delete log_state;
        CuEVM::evm_return_data_t* return_data = new CuEVM::evm_return_data_t();
        CUDA_CHECK(cudaMallocManaged(&evm_instances[index].return_data_ptr, sizeof(CuEVM::evm_return_data_t)));
        memcpy(evm_instances[index].return_data_ptr, return_data, sizeof(CuEVM::evm_return_data_t));
        delete return_data;

        CuEVM::EccConstants* ecc_constants_ptr = new CuEVM::EccConstants();
        CUDA_CHECK(cudaMallocManaged(&evm_instances[index].ecc_constants_ptr, sizeof(CuEVM::EccConstants)));
        memcpy(evm_instances[index].ecc_constants_ptr, ecc_constants_ptr, sizeof(CuEVM::EccConstants));
        delete ecc_constants_ptr;
#ifdef EIP_3155
        CuEVM::utils::tracer_t* tracer = new CuEVM::utils::tracer_t();
        CUDA_CHECK(cudaMallocManaged(&evm_instances[index].tracer_ptr, sizeof(CuEVM::utils::tracer_t)));
        memcpy(evm_instances[index].tracer_ptr, tracer, sizeof(CuEVM::utils::tracer_t));
        delete tracer;
#endif

        CuEVM::serialized_worldstate_data* serialized_worldstate_data = new CuEVM::serialized_worldstate_data();
        CUDA_CHECK(cudaMallocManaged(&evm_instances[index].serialized_worldstate_data_ptr,
                                     sizeof(CuEVM::serialized_worldstate_data)));
        memcpy(evm_instances[index].serialized_worldstate_data_ptr, serialized_worldstate_data,
               sizeof(CuEVM::serialized_worldstate_data));
        delete serialized_worldstate_data;

        CuEVM::utils::simplified_trace_data* simplified_trace_data = new CuEVM::utils::simplified_trace_data();
        CUDA_CHECK(cudaMallocManaged(&evm_instances[index].simplified_trace_data_ptr,
                                     sizeof(CuEVM::utils::simplified_trace_data)));
        memcpy(evm_instances[index].simplified_trace_data_ptr, simplified_trace_data,
               sizeof(CuEVM::utils::simplified_trace_data));
        delete simplified_trace_data;
        printf("size of serialized worldstate data %u\n", sizeof(CuEVM::serialized_worldstate_data));
        printf("size of simplified trace data %u\n", sizeof(CuEVM::utils::simplified_trace_data));
    }

    num_instances = num_transactions;
}

static PyObject* pyobject_from_simplified_trace(CuEVM::utils::simplified_trace_data trace_data) {
    PyObject* tracer_root = PyDict_New();

    PyObject* branches = PyList_New(0);
    PyObject* events = PyList_New(0);
    PyObject* calls = PyList_New(0);

    // process call
    for (size_t idx = 0; idx < trace_data.no_calls; idx++) {
        PyObject* call_item = PyDict_New();
        PyDict_SetItemString(call_item, "sender", PyUnicode_FromString(trace_data.calls[idx].sender.to_hex()));
        PyDict_SetItemString(call_item, "receiver", PyUnicode_FromString(trace_data.calls[idx].receiver.to_hex()));

        PyDict_SetItemString(call_item, "pc", PyLong_FromSize_t(trace_data.calls[idx].pc));
        PyDict_SetItemString(call_item, "op", PyLong_FromSize_t(trace_data.calls[idx].op));
        PyDict_SetItemString(call_item, "value", PyUnicode_FromString(trace_data.calls[idx].value.to_hex()));
        PyDict_SetItemString(call_item, "success", PyLong_FromSize_t(trace_data.calls[idx].success));
        PyList_Append(calls, call_item);
        Py_DECREF(call_item);
    }

    for (size_t idx = 0; idx < trace_data.no_events; idx++) {
        PyObject* event_item = PyDict_New();
        PyDict_SetItemString(event_item, "pc", PyLong_FromSize_t(trace_data.events[idx].pc));
        PyDict_SetItemString(event_item, "op", PyLong_FromSize_t(trace_data.events[idx].op));
        PyDict_SetItemString(event_item, "operand_1", PyUnicode_FromString(trace_data.events[idx].operand_1.to_hex()));
        PyDict_SetItemString(event_item, "operand_2", PyUnicode_FromString(trace_data.events[idx].operand_2.to_hex()));
        PyDict_SetItemString(event_item, "res", PyUnicode_FromString(trace_data.events[idx].res.to_hex()));
        PyList_Append(events, event_item);
        Py_DECREF(event_item);
    }

    for (size_t idx = 0; idx < trace_data.no_branches; idx++) {
        PyObject* branch_item = PyDict_New();
        PyDict_SetItemString(branch_item, "pc_src", PyLong_FromSize_t(trace_data.branches[idx].pc_src));
        PyDict_SetItemString(branch_item, "pc_dst", PyLong_FromSize_t(trace_data.branches[idx].pc_dst));
        PyDict_SetItemString(branch_item, "pc_missed", PyLong_FromSize_t(trace_data.branches[idx].pc_missed));
        PyDict_SetItemString(branch_item, "distance", PyUnicode_FromString(trace_data.branches[idx].distance.to_hex()));
        PyList_Append(branches, branch_item);
        Py_DECREF(branch_item);
    }

    PyDict_SetItemString(tracer_root, "events", events);
    PyDict_SetItemString(tracer_root, "branches", branches);
    PyDict_SetItemString(tracer_root, "calls", calls);

    return tracer_root;
}

PyObject* pyobject_from_serialized_state(CuEVM::serialized_worldstate_data* serialized_worldstate_instance) {
    PyObject* state_dict = PyDict_New();

    // Add accounts and storage elements
    // PyObject* accounts_list = PyList_New(0);
    uint32_t account_idx = 0;
    uint32_t storage_idx = 0;
    for (uint32_t i = 0; i < serialized_worldstate_instance->no_accounts; i++) {
        PyObject* account_dict = PyDict_New();

        PyDict_SetItemString(account_dict, "balance", PyUnicode_FromString(serialized_worldstate_instance->balance[i]));
        PyDict_SetItemString(account_dict, "nonce", PyLong_FromUnsignedLong(serialized_worldstate_instance->nonce[i]));

        // Add storage elements for the account if they exist
        PyObject* storage_dict = PyDict_New();
        while (storage_idx < serialized_worldstate_instance->no_storage_elements &&
               serialized_worldstate_instance->storage_indexes[storage_idx] == i) {
            PyObject* storage_key_value = PyDict_New();

            PyDict_SetItem(storage_dict,
                           PyUnicode_FromString(serialized_worldstate_instance->storage_keys[storage_idx]),
                           PyUnicode_FromString(serialized_worldstate_instance->storage_values[storage_idx]));
            Py_DECREF(storage_key_value);
            storage_idx++;
        }

        PyDict_SetItemString(account_dict, "storage", storage_dict);
        // PyList_Append(accounts_list, account_dict);
        PyDict_SetItemString(state_dict, serialized_worldstate_instance->addresses[i], account_dict);
        Py_DECREF(account_dict);
    }
    return state_dict;
}

PyObject* pyobject_from_evm_instances(CuEVM::evm_instance_t* instances, uint32_t num_instances) {
    PyObject* root = PyDict_New();
    // PyObject* world_state_json = pyobject_from_state_data_t(arith, instances.world_state_data);
    // PyDict_SetItemString(root, "pre", world_state_json);
    // Py_DECREF(world_state_json);
    PyObject* instances_json = PyList_New(0);
    PyDict_SetItemString(root, "post", instances_json);
    Py_DECREF(instances_json);  // Decrement here because PyDict_SetItemString increases the ref count

    for (uint32_t idx = 0; idx < num_instances; idx++) {
        CuEVM::state_t* world_state_instance = instances[idx].world_state_data_ptr;  // update on this
        CuEVM::serialized_worldstate_data* serialized_worldstate = instances[idx].serialized_worldstate_data_ptr;

        PyObject* instance_json = PyDict_New();

        // TODO: Print resultant state. to check with state_root branch
        PyObject* state_json = pyobject_from_serialized_state(serialized_worldstate);  // PyList_New(0);

        PyDict_SetItemString(instance_json, "state", state_json);
        PyObject* tracer_json = pyobject_from_simplified_trace(*instances[idx].simplified_trace_data_ptr);
        PyDict_SetItemString(instance_json, "trace", tracer_json);
        // printf("state json result\n");
        // print_dict_recursive(state_json, 0);
        // printf("tracer json result\n");
        // print_dict_recursive(tracer_json, 0);
        PyList_Append(instances_json, instance_json);  // Appends and steals the reference, so no need to DECREF
    }

    return root;
}

}  // namespace python_utils
