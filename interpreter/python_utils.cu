#include <CuEVM/utils/python_utils.h>

#include <sstream>

#define CHECK_AND_RETURN_ON_ERROR(expr)                                                                            \
    do {                                                                                                           \
        auto status = (expr);                                                                                      \
        if (status != 0) {                                                                                         \
            std::ostringstream oss;                                                                                \
            oss << "Error in expression: " << #expr << " (status: 0x" << std::hex << status << std::dec << ") at " \
                << __FILE__ << ":" << __LINE__;                                                                    \
            throw std::runtime_error(oss.str());                                                                   \
        }                                                                                                          \
    } while (0)

void copy_dict_recursive(PyObject* read_root, PyObject* write_root);
static PyObject* print_dict(PyObject* self, PyObject* args);
namespace CuEVM {
__host__ void serialized_worldstate_data::print() {
    printf("\nPrinting serialized worldstate data\n");
    printf("no_accounts: %d\n", no_accounts);
    printf("no_storage_elements: %d\n", no_storage_elements);
    for (uint32_t idx = 0; idx < no_accounts; idx++) {
        printf("address: %s\n", addresses[idx]);
        printf("balance: %s\n", balance[idx]);
        printf("nonce: %d\n", nonce[idx]);
    }
    for (uint32_t idx = 0; idx < no_storage_elements; idx++) {
        printf("storage_key: %s\n", storage_keys[idx]);
        printf("storage_value: %s\n", storage_values[idx]);
        printf("storage_index: %d\n", storage_indexes[idx]);
    }
}

__device__ void simplified_trace_data::start_operation(const uint32_t pc, const uint8_t op,
                                                       const CuEVM::evm_stack_t& stack_ptr) {
    if (no_events >= MAX_TRACE_EVENTS) return;
    events[no_events].pc = pc;
    events[no_events].op = op;
    if (op != OP_INVALID && op != OP_SELFDESTRUCT) {
        // printf("add new operation, src data %d \n", THREADIDX);
        // printf("stack size %d\n", stack_ptr.size());

        events[no_events].operand_1 = *stack_ptr.get_address_at_index(1);
        events[no_events].operand_2 = *stack_ptr.get_address_at_index(2);
    }
}

__device__ void simplified_trace_data::record_branch(uint32_t pc_src, uint32_t pc_dst, uint32_t pc_missed) {
    if (no_branches >= MAX_BRANCHES_TRACING) no_branches = 0;
    branches[no_branches].pc_src = pc_src;
    branches[no_branches].pc_dst = pc_dst;
    branches[no_branches].pc_missed = pc_missed;
    branches[no_branches].distance = last_distance;
    // printf("record branch pc_src %u pc_dst %u distance %s\n", pc_src, pc_dst,
    // branches[no_branches].distance.to_hex());
    no_branches++;
}

__device__ void simplified_trace_data::record_distance(uint8_t op, const CuEVM::evm_stack_t& stack_ptr) {
    evm_word_t distance, op1, op2;
    uint32_t stack_size = stack_ptr.size();

    op1 = *stack_ptr.get_address_at_index(1);
    op2 = *stack_ptr.get_address_at_index(2);

    if (uint256_cmp(&op1, &op2) >= 1)
        uint256_sub(&distance, &op1, &op2);
    else
        uint256_sub(&distance, &op2, &op1);

    if (op != OP_EQ) uint256_add_word(&distance, &distance, 1);

    last_distance = distance;
}

__device__ void simplified_trace_data::finish_operation(const CuEVM::evm_stack_t& stack_ptr, uint32_t error_code) {
    if (no_events >= MAX_TRACE_EVENTS) return;
    if (events[no_events].op < OP_REVERT && events[no_events].op != OP_SSTORE)
        events[no_events].res = *stack_ptr.get_address_at_index(1);
    no_events++;
}
__device__ void simplified_trace_data::start_call(uint32_t pc, evm_call_context_t* call_context_ptr) {
    assert(call_context_ptr != nullptr);
    if (no_calls >= MAX_CALLS_TRACING) return;
    // add address and increment current_address_idx
    // addresses[current_address_idx] = cached_call_state->addresses[cached_call_state->current_address_idx];
    // printf("start call simplified trace data pc %d op %d\n", pc, message_call_ptr->call_type);
    calls[no_calls].sender = call_context_ptr->from;
    calls[no_calls].receiver = call_context_ptr->to;
    calls[no_calls].pc = pc;
    calls[no_calls].op = call_context_ptr->call_type;
    calls[no_calls].value = call_context_ptr->value;

    no_calls++;
}
__device__ void simplified_trace_data::finish_call(uint8_t success) {
    if (no_calls > MAX_CALLS_TRACING) return;

    // printf("no_calls %u \n", no_calls);
    for (int i = no_calls - 1; i >= 0; i--) {
        if (calls[i].success == UINT8_MAX) {
            calls[i].success = success;
            break;
        }
    }
}
__host__ __device__ void simplified_trace_data::print() {
    printf("no_events %u\n", no_events);
    printf("no_calls %u\n", no_calls);
    printf("events\n");
    for (uint32_t i = 0; i < no_events; i++) {
        printf("pc %u op %u operand_1 %s operand_2 %s res %s\n", events[i].pc, events[i].op,
               events[i].operand_1.to_hex(), events[i].operand_2.to_hex(), events[i].res.to_hex());
    }
    printf("calls\n");
    for (uint32_t i = 0; i < no_calls; i++) {
        printf("pc %u op %u sender %s receiver %s value %s success %u\n", calls[i].pc, calls[i].op,
               calls[i].sender.to_hex(), calls[i].receiver.to_hex(), calls[i].value.to_hex(), calls[i].success);
    }
    printf("branches\n");
    for (uint32_t i = 0; i < no_branches; i++) {
        printf("pc_src %u pc_dst %u distance %s\n", branches[i].pc_src, branches[i].pc_dst,
               branches[i].distance.to_hex());
    }
}
__device__ serialized_worldstate_data* global_serialized_worldstate;
__device__ simplified_trace_data* global_simplified_trace;
}  // namespace CuEVM
namespace python_utils {

using namespace CuEVM;
using CuEVM::simplified_trace_data;
using CuEVM::transaction::TransactionList;

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

void get_c_byte_array_from_pyobject(PyObject* pyobject, uint8_t*& c_byte_array, uint32_t& size) {
    if (PyBytes_Check(pyobject)) {
        // Get the pointer to the underlying buffer (it's a char*, so cast it to uint8_t*)
        c_byte_array = reinterpret_cast<uint8_t*>(PyBytes_AsString(pyobject));
        // Retrieve the size of the byte string
        size = PyBytes_Size(pyobject);
        // Now you have both the C uint8_t array (c_byte_array) and its size (size)
    } else if (PyByteArray_Check(pyobject)) {
        // If it's a bytearray, use the corresponding functions
        c_byte_array = reinterpret_cast<uint8_t*>(PyByteArray_AsString(pyobject));
        size = PyByteArray_Size(pyobject);
        // Now you have the uint8_t array and its size
    } else {
        printf("pyobject to byte array failed: %p\n", pyobject);
        PyErr_SetString(PyExc_TypeError, "Expected a bytes or bytearray object.");
        exit(0);
        // return;
    }
}
// similar to CuEVM/src/core/transaction.cu#get_transactions
TransactionList* getTransactionDataFromListofPyObject(PyObject* read_roots) {
    Py_ssize_t count = PyList_Size(read_roots);
    TransactionList* transactions;

    transactions = new TransactionList();
    printf("getTransactionDataFromListofPyObject count: %d\n", count);
    PyObject* first_transaction = PyList_GetItem(read_roots, 0);
    first_transaction = PyDict_GetItemString(first_transaction, "transaction");
    // evm_word_t nonce;
    // evm_word_t sender;
    // evm_word_t to;
    // evm_word_t max_fee_per_gas;
    // evm_word_t max_priority_fee_per_gas;
    // evm_word_t gas_price;
    // uint16_t type;
    // transactions->nonce.from_hex(GET_STR_FROM_DICT_WITH_DEFAULT(first_transaction, "nonce", "0x00"));
    // transactions->sender.from_hex(GET_STR_FROM_DICT_WITH_DEFAULT(first_transaction, "sender", "0x00"));
    // transactions->to.from_hex(GET_STR_FROM_DICT_WITH_DEFAULT(first_transaction, "to", "0x00"));
    py_long_to_uint256(PyDict_GetItemString(first_transaction, "to"), &transactions->to);
    // printf("transactions->to: %s\n", transactions->to.to_hex());
    py_long_to_uint256(PyDict_GetItemString(first_transaction, "sender"), &transactions->sender);
    // printf("transactions->sender: %s\n", transactions->sender.to_hex());
    py_long_to_uint256(PyDict_GetItemString(first_transaction, "nonce"), &transactions->nonce);
    // printf("transactions->nonce: %s\n", transactions->nonce.to_hex());

    transactions->max_fee_per_gas.from_hex("0x00");
    transactions->max_priority_fee_per_gas.from_hex("0x00");
    transactions->gas_price.from_hex("0x01");
    transactions->size = count;
    transactions->type = 0;
    transactions->value = new evm_word_t[count];
    transactions->gas_limit = new uint64_t[count];
    transactions->call_data_offset = new uint32_t[count];
    transactions->call_data_size = new uint32_t[count];

    // printf("first transaction \n");
    // print_dict_recursive(first_transaction, 1);

    uint32_t curr_call_data_offset = 0;

    // CuEVM::byte_array_t data_init;

    for (Py_ssize_t idx = 0; idx < count; idx++) {
        PyObject* current_instance = PyList_GetItem(read_roots, idx);
        if (!PyDict_Check(current_instance)) {
            printf("current_instance: %p\n", current_instance);
            PyErr_SetString(PyExc_TypeError, "Each item in the list must be a dictionary.");
            return NULL;
        }

        PyObject* data = PyDict_GetItemString(current_instance, "transaction");

        // printf("transaction: %p\n", current_instance);
        // print_dict_recursive(current_instance, 1);
        // printf("data: %p\n", data);
        // print_dict_recursive(data, 1);

        PyObject* first_tx_data = PyList_GetItem(PyDict_GetItemString(data, "data"), 0);

        uint8_t* tx_data;
        uint32_t tx_data_size;
        get_c_byte_array_from_pyobject(first_tx_data, tx_data, tx_data_size);

        PyObject* tx_gas_limit = PyList_GetItem(PyDict_GetItemString(data, "gasLimit"), 0);
        PyObject* tx_value = PyList_GetItem(PyDict_GetItemString(data, "value"), 0);

        evm_word_t tmp;

        // CHECK_AND_RETURN_ON_ERROR(tmp.from_hex(PyUnicode_AsUTF8(tx_gas_limit)));

        transactions->gas_limit[idx] = UINT32_MAX;  // use max gas limit for fuzzing uint256_get_uint64_t(&tmp);
        // CHECK_AND_RETURN_ON_ERROR(transactions->value[idx].from_hex(PyUnicode_AsUTF8(tx_value)));
        py_long_to_uint256(tx_value, &transactions->value[idx]);

        transactions->call_data_offset[idx] = curr_call_data_offset;
        transactions->call_data_size[idx] = tx_data_size;
        if (tx_data_size > 0) {
            if (transactions->call_data == nullptr) {
                transactions->call_data = new uint8_t[tx_data_size];
            } else {
                uint8_t* tmp = new uint8_t[curr_call_data_offset + tx_data_size];
                memcpy(tmp, transactions->call_data, curr_call_data_offset);
                delete[] transactions->call_data;
                transactions->call_data = tmp;
            }
            memcpy(&transactions->call_data[curr_call_data_offset], tx_data, tx_data_size);
        }

        curr_call_data_offset += tx_data_size;
    }

    uint32_t call_data_size = curr_call_data_offset;

    // Copy the transaction data to device memory,
    TransactionList* d_transaction_list_ptr;
    TransactionList* temp_transaction_list_ptr = new TransactionList();

    memcpy(temp_transaction_list_ptr, transactions, sizeof(TransactionList));
    CUDA_CHECK(cudaMalloc(&temp_transaction_list_ptr->value, count * sizeof(evm_word_t)));
    CUDA_CHECK(cudaMalloc(&temp_transaction_list_ptr->gas_limit, count * sizeof(gas_t)));
    CUDA_CHECK(cudaMalloc(&temp_transaction_list_ptr->call_data, call_data_size * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&temp_transaction_list_ptr->call_data_offset, count * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&temp_transaction_list_ptr->call_data_size, count * sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpy(temp_transaction_list_ptr->value, transactions->value, count * sizeof(evm_word_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(temp_transaction_list_ptr->gas_limit, transactions->gas_limit, count * sizeof(gas_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(temp_transaction_list_ptr->call_data, transactions->call_data,
                          call_data_size * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(temp_transaction_list_ptr->call_data_offset, transactions->call_data_offset,
                          count * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(temp_transaction_list_ptr->call_data_size, transactions->call_data_size,
                          count * sizeof(uint32_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_transaction_list_ptr, sizeof(TransactionList)));
    CUDA_CHECK(
        cudaMemcpy(d_transaction_list_ptr, temp_transaction_list_ptr, sizeof(TransactionList), cudaMemcpyHostToDevice));

    // todo_cl some data not freed?
    delete temp_transaction_list_ptr;

    return d_transaction_list_ptr;
}

// Similar to CuEVM/src/state/state_db.cu#StateDb::GPUfromJson
void getPreStateDataFromListofPyObject(PyObject* readroot, uint32_t num_states) {
    PyObject *key, *value;

    StateDb* state_db_cpu;

    PyObject* first_item = PyList_GetItem(readroot, 0);
    PyObject* data = PyDict_GetItemString(first_item, "pre");

    auto num_accounts = PyDict_Size(data);

    // create on host
    state_db_cpu = new StateDb(num_states);

    state_db_cpu->num_accounts = num_accounts;
    state_db_cpu->num_contracts = 0;
    state_db_cpu->num_states = num_states;
    state_db_cpu->num_storage_elements = 0;
    state_db_cpu->storage_capacity = account_prealloc_keys_size * num_accounts;
    state_db_cpu->all_account_codes = nullptr;
    state_db_cpu->address_list = new evm_word_t[num_accounts];
    state_db_cpu->contract_index = new int16_t[num_accounts];

    state_db_cpu->account_nonces = new uint32_t[num_accounts * num_states];
    state_db_cpu->account_storage_size = new uint32_t[num_accounts * num_states];
    state_db_cpu->account_balances = new evm_word_t[num_accounts * num_states];

    state_db_cpu->account_codes_size = new uint32_t[num_accounts];
    state_db_cpu->account_codes_offset = new uint32_t[num_accounts];

    state_db_cpu->dynamic_accounts = new DynamicAccount*[num_states];

    // cpu side prealloc with num_accounts because we dont know the number of contracts yet.
    // GPU side will only use num_contracts
    state_db_cpu->prealloc_keys_pool = new evm_word_t[account_prealloc_keys_size * num_accounts * num_states];
    state_db_cpu->prealloc_values_pool = new ValueStatus[account_prealloc_keys_size * num_accounts * num_states];
    state_db_cpu->account_is_warm = new bool[num_states * num_accounts];
    // dynamic storage to grow later
    state_db_cpu->dynamic_storage_pages = new StateDbStoragePage*[num_states * num_accounts];
    state_db_cpu->dynamic_pool_capacity = new uint32_t[num_states * num_accounts];

    // state_db->snapshot_total_storage_size = new uint32_t[num_states * num_accounts];
    cJSON* account_json;
    uint32_t bytecode_offset = 0;

    memset(state_db_cpu->account_is_warm, 0, num_states * num_accounts * sizeof(bool));
    memset(state_db_cpu->dynamic_pool_capacity, 0, num_states * num_accounts * sizeof(uint32_t));

    memset(state_db_cpu->prealloc_keys_pool, 0,
           account_prealloc_keys_size * num_accounts * num_states * sizeof(evm_word_t));
    memset(state_db_cpu->prealloc_values_pool, 0,
           account_prealloc_keys_size * num_accounts * num_states * sizeof(ValueStatus));
    memset(state_db_cpu->dynamic_accounts, 0, num_states * sizeof(DynamicAccount*));

    evm_word_t t_word;

    for (uint32_t state_idx = 0; state_idx < num_states; state_idx++) {
        // printf("state_idx: %d\n", state_idx);
        PyObject* current_state = PyList_GetItem(readroot, state_idx);
        data = PyDict_GetItemString(current_state, "pre");
        Py_ssize_t pos = 0;

        while (PyDict_Next(data, &pos, &key, &value)) {
            int address_index = pos - 1;  // Adjust index since PyDict_Next increments pos
            // printf("address_index: %d\n", address_index);
            // const char* address_str = PyUnicode_AsUTF8(key);
            uint32_t instance_idx = address_index * num_states + state_idx;
            // Extract balance, nonce, and code
            // const char* balance = GET_STR_FROM_DICT_WITH_DEFAULT(value, "balance", "0x0");
            // const char* nonce = GET_STR_FROM_DICT_WITH_DEFAULT(value, "nonce", "0x0");
            // const char* code = GET_STR_FROM_DICT_WITH_DEFAULT(value, "code", "");
            PyObject* code = PyDict_GetItemString(value, "code");
            PyObject* storage_dict = PyDict_GetItemString(value, "storage");
            uint32_t storage_size = PyDict_Size(storage_dict);

            // printf("storage_size: %d\n", storage_size);
            // Initialize account details
            if (state_idx == 0) {
                // state_db_cpu->address_list[address_index].from_hex(address_str);
                py_long_to_uint256(key, &state_db_cpu->address_list[address_index]);
                // Code
                // byte_array_t byte_code;
                uint8_t* byte_code;
                uint32_t byte_code_size;
                get_c_byte_array_from_pyobject(code, byte_code, byte_code_size);
                // byte_code.from_hex(code, LITTLE_ENDIAN, NO_PADDING);
                state_db_cpu->account_codes_size[address_index] = byte_code_size;
                state_db_cpu->account_codes_offset[address_index] = bytecode_offset;
                if (byte_code_size > 0) {
                    uint8_t* tmp = state_db_cpu->all_account_codes;
                    state_db_cpu->all_account_codes = new uint8_t[bytecode_offset + byte_code_size];
                    memcpy(state_db_cpu->all_account_codes, tmp, bytecode_offset * sizeof(uint8_t));
                    delete[] tmp;
                    memcpy(&state_db_cpu->all_account_codes[bytecode_offset], byte_code, byte_code_size);
                    bytecode_offset += byte_code_size;
                }
                if (byte_code_size > 0 || storage_size > 0) {
                    state_db_cpu->contract_index[address_index] = state_db_cpu->num_contracts;
                    state_db_cpu->num_contracts++;
                } else {
                    state_db_cpu->contract_index[address_index] = -1;
                }
            }

            // state_db_cpu->account_balances[instance_idx].from_hex(balance);
            // t_word.from_hex(nonce);
            // state_db_cpu->account_nonces[instance_idx] = uint256_get_uint32_t(&t_word);
            py_long_to_uint256(PyDict_GetItemString(value, "balance"), &state_db_cpu->account_balances[instance_idx]);
            py_long_to_uint256(PyDict_GetItemString(value, "nonce"), &t_word);
            state_db_cpu->account_nonces[instance_idx] = uint256_get_uint32_t(&t_word);
            // printf("account_nonces[%d] = %d\n", instance_idx, state_db_cpu->account_nonces[instance_idx]);
            // printf("account_balances[%d] = %s\n", instance_idx,
            // state_db_cpu->account_balances[instance_idx].to_hex()); printf("address_list[%d] = %s\n", address_index,
            // state_db_cpu->address_list[address_index].to_hex());

            state_db_cpu->account_storage_size[instance_idx] = storage_size;
            // Storage

            // state_db_cpu->num_storage_elements += state_db_cpu->account_storage_size[idx * num_states];
            if (storage_size > account_prealloc_keys_size) {
                printf("PreState storage size: %d greater than supported\n", storage_size);
                return;
            }

            // printf("state_idx: %d instance_idx: %d address_index: %d contract_idx: %d\n", state_idx, instance_idx,
            //        address_index, state_db_cpu->contract_index[address_index]);
            // Iterate through the dictionary
            PyObject *key_storage, *value_storage;
            Py_ssize_t pos_1 = 0;

            while (PyDict_Next(storage_dict, &pos_1, &key_storage, &value_storage)) {
                int storage_idx = pos_1 - 1;  // Adjust index since PyDict_Next increments pos
                Py_INCREF(key_storage);
                Py_INCREF(value_storage);

                uint32_t contract_idx = state_db_cpu->contract_index[address_index];
                // printf("contract_idx: %d\n", contract_idx);
                uint32_t pre_alloc_keys_idx =
                    (account_prealloc_keys_size * contract_idx + storage_idx) * num_states + state_idx;
                // printf("pre_alloc_keys_idx: %d\n", pre_alloc_keys_idx);
                // state_db_cpu->prealloc_keys_pool[pre_alloc_keys_idx].from_hex(PyUnicode_AsUTF8(key_storage));
                // state_db_cpu->prealloc_values_pool[pre_alloc_keys_idx].from_hex(PyUnicode_AsUTF8(value_storage));
                py_long_to_uint256(key_storage, &state_db_cpu->prealloc_keys_pool[pre_alloc_keys_idx]);
                py_long_to_uint256(value_storage, &state_db_cpu->prealloc_values_pool[pre_alloc_keys_idx].value);
                state_db_cpu->prealloc_values_pool[pre_alloc_keys_idx].original_value =
                    state_db_cpu->prealloc_values_pool[pre_alloc_keys_idx].value;
                state_db_cpu->prealloc_values_pool[pre_alloc_keys_idx].is_warm = false;
            }
        }
    }

    // printf("state_db_cpu->num_accounts: %d\n", state_db_cpu->num_accounts);
    // printf("statedb cpu\n");
    // state_db_cpu->print();
    // printf("end of statedb cpu\n");
    // copy to device. c.f., StateDb::GPUfromJson
    StateDb* tmp_state_db = new StateDb(num_states);
    tmp_state_db->num_accounts = state_db_cpu->num_accounts;
    tmp_state_db->num_states = state_db_cpu->num_states;
    tmp_state_db->num_storage_elements = state_db_cpu->num_storage_elements;
    tmp_state_db->storage_capacity = state_db_cpu->storage_capacity;
    tmp_state_db->num_contracts = state_db_cpu->num_contracts;

    num_accounts = state_db_cpu->num_accounts;
    uint32_t code_size =
        state_db_cpu->account_codes_size[num_accounts - 1] + state_db_cpu->account_codes_offset[num_accounts - 1];

    // Grouped memory allocation
    CUDA_CHECK(cudaMalloc(&tmp_state_db->address_list, num_accounts * sizeof(evm_word_t)));
    CUDA_CHECK(cudaMalloc(&tmp_state_db->contract_index, num_accounts * sizeof(int16_t)));

    CUDA_CHECK(cudaMalloc(&tmp_state_db->account_balances, num_states * num_accounts * sizeof(evm_word_t)));
    CUDA_CHECK(cudaMalloc(&tmp_state_db->account_nonces, num_states * num_accounts * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&tmp_state_db->account_storage_size, num_states * num_accounts * sizeof(uint32_t)));

    CUDA_CHECK(cudaMalloc(&tmp_state_db->account_codes_size, num_accounts * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&tmp_state_db->account_codes_offset, num_accounts * sizeof(uint32_t)));

    // prealloc storage , only num_contracts are allocated
    CUDA_CHECK(cudaMalloc(&tmp_state_db->prealloc_keys_pool,
                          account_prealloc_keys_size * state_db_cpu->num_contracts * num_states * sizeof(evm_word_t)));
    CUDA_CHECK(cudaMalloc(&tmp_state_db->prealloc_values_pool,
                          account_prealloc_keys_size * state_db_cpu->num_contracts * num_states * sizeof(ValueStatus)));
    CUDA_CHECK(cudaMalloc(&tmp_state_db->all_account_codes, code_size * sizeof(uint8_t)));
    CUDA_CHECK(
        cudaMalloc(&tmp_state_db->dynamic_storage_pages, num_states * num_accounts * sizeof(StateDbStoragePage*)));
    CUDA_CHECK(cudaMalloc(&tmp_state_db->dynamic_pool_capacity, num_states * num_accounts * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&tmp_state_db->account_is_warm, num_states * num_accounts * sizeof(bool)));

    CUDA_CHECK(cudaMalloc(&tmp_state_db->dynamic_accounts, num_states * sizeof(DynamicAccount*)));

    // CUDA_CHECK(cudaMalloc(&tmp_state_db->snapshot_total_storage_size, num_states * num_accounts *
    // sizeof(uint32_t))); Grouped memory copy
    CUDA_CHECK(cudaMemcpy(tmp_state_db->address_list, state_db_cpu->address_list, num_accounts * sizeof(evm_word_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_state_db->contract_index, state_db_cpu->contract_index, num_accounts * sizeof(int16_t),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(tmp_state_db->account_balances, state_db_cpu->account_balances,
                          num_states * num_accounts * sizeof(evm_word_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_state_db->account_nonces, state_db_cpu->account_nonces,
                          num_states * num_accounts * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_state_db->account_storage_size, state_db_cpu->account_storage_size,
                          num_states * num_accounts * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_state_db->account_codes_size, state_db_cpu->account_codes_size,
                          num_accounts * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_state_db->account_codes_offset, state_db_cpu->account_codes_offset,
                          num_accounts * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // prealloc storage , only num_contracts are allocated
    CUDA_CHECK(cudaMemcpy(tmp_state_db->prealloc_keys_pool, state_db_cpu->prealloc_keys_pool,
                          account_prealloc_keys_size * state_db_cpu->num_contracts * num_states * sizeof(evm_word_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_state_db->prealloc_values_pool, state_db_cpu->prealloc_values_pool,
                          account_prealloc_keys_size * state_db_cpu->num_contracts * num_states * sizeof(ValueStatus),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_state_db->all_account_codes, state_db_cpu->all_account_codes, code_size * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(tmp_state_db->dynamic_accounts, state_db_cpu->dynamic_accounts,
                          num_states * sizeof(DynamicAccount*), cudaMemcpyHostToDevice));

    // dynamic storage
    CUDA_CHECK(cudaMemcpy(tmp_state_db->dynamic_pool_capacity, state_db_cpu->dynamic_pool_capacity,
                          num_states * num_accounts * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_state_db->account_is_warm, state_db_cpu->account_is_warm,
                          num_states * num_accounts * sizeof(bool), cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy(tmp_state_db->snapshot_total_storage_size, state_db_cpu->snapshot_total_storage_size,
    //                       num_states * num_accounts * sizeof(uint32_t), cudaMemcpyHostToDevice));

    StateDb* state_db_gpu;
    CUDA_CHECK(cudaMalloc(&state_db_gpu, sizeof(StateDb)));
    CUDA_CHECK(cudaMemcpy(state_db_gpu, tmp_state_db, sizeof(StateDb), cudaMemcpyHostToDevice));
    cudaMemcpyToSymbol(global_state_db_ptr, &state_db_gpu, sizeof(StateDb*));
    delete state_db_cpu;
    delete tmp_state_db;
}

__device__ void serialize_state_data(CuEVM::serialized_worldstate_data* data) {
    // Use the global state database pointer to access the account data
    StateDb* state = global_state_db_ptr;
    if (state == nullptr) {
        // Ideally, handle the error appropriately (or abort) if the state is missing.
        return;
    }

    // Set the number of accounts in the serialized state.
    data->no_accounts = state->num_accounts;
    // Start with no storage elements serialized.
    data->no_storage_elements = 0;

    // uint32_t new_offset =
    // (contract_index[address_index] * account_prealloc_keys_size + storage_size) * num_states + INSTANCE_GLOBAL_IDX;
    // uint32_t instance_idx = address_index * num_states + INSTANCE_GLOBAL_IDX;
    // // printf("get_value_status address_index %d, instance_idx %d instance %d\n", address_index, instance_idx,
    // //        INSTANCE_GLOBAL_IDX);
    // uint32_t contract_idx = contract_index[address_index];
    // uint32_t storage_size = account_storage_size[instance_idx];

    // Iterate through each account.
    // We assume that the account data (address, balance, nonce) is stored in parallel arrays,
    // and for simplicity we pick the first state (index 0) as the canonical view.
    for (uint32_t acct = 0; acct < state->num_accounts; acct++) {
        uint32_t instance_idx = acct * state->num_states + INSTANCE_GLOBAL_IDX;
        int16_t cidx = state->contract_index[acct];
        uint32_t storage_size = state->account_storage_size[instance_idx];

        // Convert the account's address to a hex string.
        data->addresses[acct] = state->address_list[acct];
        // Convert the account's balance (using the snapshot at state index 0) to hex.
        data->balance[acct] = state->account_balances[instance_idx];
        // Copy the account's nonce (again using state index 0).
        data->nonce[acct] = state->account_nonces[instance_idx];

        // if (INSTANCE_GLOBAL_IDX == 0) {
        //     printf("address: \n");
        //     state->address_list[acct].print();
        //     printf("balance: \n");
        //     state->account_balances[instance_idx].print();

        //     // Get the storage size for this account (again, from the first snapshot).
        //     printf("account %d instance %d storage size: %d\n", acct, instance_idx, storage_size);
        // }

        if (storage_size > 0) {
            // The contract index tells us which section of the preallocated storage pool to use.
            for (uint32_t s = 0; s < storage_size; s++) {
                if (s > account_prealloc_keys_size) break;
                // Compute the index into the preallocated storage arrays.
                // (account_prealloc_keys_size * contract_index + storage_element)
                // is multiplied by num_states because storage is stored for every state.
                uint32_t prealloc_idx =
                    (account_prealloc_keys_size * cidx + s) * state->num_states + INSTANCE_GLOBAL_IDX;

                // Convert the storage key and value into hex strings.
                data->storage_keys[data->no_storage_elements + s] = state->prealloc_keys_pool[prealloc_idx];
                data->storage_values[data->no_storage_elements + s] = state->prealloc_values_pool[prealloc_idx].value;
                // Record which account this storage element belongs to.
                data->storage_indexes[data->no_storage_elements + s] = acct;
            }
        }
        // Increment the total count of storage elements serialized.
        data->no_storage_elements += storage_size;
    }
}

TransactionList* get_evm_instances_from_PyObject(PyObject* read_roots, uint32_t& num_instances, bool reuse_state_data,
                                                 bool copy_state_data, uint32_t call_counter) {
    uint32_t num_transactions = PyList_Size(read_roots);

    num_instances = num_transactions;
    // evm_instances = new evm_instance_t[num_instances];
    TransactionList* all_transactions;
    if (!reuse_state_data || call_counter == 0) {
        printf("create state data and block info\n");
        get_block_info_from_PyObject(read_roots);

        getPreStateDataFromListofPyObject(read_roots, num_transactions);
    }
    all_transactions = getTransactionDataFromListofPyObject(read_roots);

#ifdef BUILD_LIBRARY
    // Simplified trace data
    CuEVM::simplified_trace_data* d_trace_data;

    CUDA_CHECK(cudaMalloc(&d_trace_data, num_transactions * sizeof(CuEVM::simplified_trace_data)));
    cudaMemset(d_trace_data, 0, num_transactions * sizeof(CuEVM::simplified_trace_data));
    cudaMemcpyToSymbol(global_simplified_trace, &d_trace_data, sizeof(CuEVM::simplified_trace_data*));

    if (copy_state_data) {
        CuEVM::serialized_worldstate_data* d_serialized_worldstate_data;
        CUDA_CHECK(
            cudaMalloc(&d_serialized_worldstate_data, num_transactions * sizeof(CuEVM::serialized_worldstate_data)));
        cudaMemset(d_serialized_worldstate_data, 0, num_transactions * sizeof(CuEVM::serialized_worldstate_data));
        cudaMemcpyToSymbol(global_serialized_worldstate, &d_serialized_worldstate_data,
                           sizeof(CuEVM::serialized_worldstate_data*));
    }
#endif

    return all_transactions;
}

__host__ void get_block_info_from_PyObject(PyObject* data) {
    // construct blockt_t

    block_info_t* block_data = new block_info_t();

    PyObject* base_fee_obj = PyDict_GetItemString(data, "currentBaseFee");
    if (base_fee_obj) {
        py_long_to_uint256(base_fee_obj, &block_data->base_fee);
    } else {
        block_data->base_fee.from_hex(DefaultBlock::BaseFee);
    }

    PyObject* coin_base_obj = PyDict_GetItemString(data, "currentCoinbase");
    if (coin_base_obj) {
        py_long_to_uint256(coin_base_obj, &block_data->coin_base);
    } else {
        block_data->coin_base.from_hex(DefaultBlock::CoinBase);
    }

    PyObject* difficulty_obj = PyDict_GetItemString(data, "currentDifficulty");
    if (difficulty_obj) {
        py_long_to_uint256(difficulty_obj, &block_data->difficulty);
    } else {
        block_data->difficulty.from_hex(DefaultBlock::Difficulty);
    }

    PyObject* block_number_obj = PyDict_GetItemString(data, "currentNumber");
    if (block_number_obj) {
        py_long_to_uint256(block_number_obj, &block_data->number);
    } else {
        block_data->number.from_hex(DefaultBlock::BlockNumber);
    }

    PyObject* gas_limit_obj = PyDict_GetItemString(data, "currentGasLimit");
    if (gas_limit_obj) {
        py_long_to_uint256(gas_limit_obj, &block_data->gas_limit);
    } else {
        block_data->gas_limit.from_hex(DefaultBlock::GasLimit);
    }

    PyObject* time_stamp_obj = PyDict_GetItemString(data, "currentTimestamp");
    if (time_stamp_obj) {
        py_long_to_uint256(time_stamp_obj, &block_data->time_stamp);
    } else {
        block_data->time_stamp.from_hex(DefaultBlock::TimeStamp);
    }

    PyObject* previous_hash_obj = PyDict_GetItemString(data, "previousHash");
    if (previous_hash_obj) {
        py_long_to_uint256(previous_hash_obj, &block_data->previous_blocks[0].hash);
    } else {
        block_data->previous_blocks[0].hash.from_hex(DefaultBlock::PreviousHash);
    }

    block_data->chain_id = 1;

    // Allocate device memory for block info
    block_info_t* d_block_info;
    cudaMalloc(&d_block_info, sizeof(block_info_t));

    // Copy block info to device
    cudaMemcpy(d_block_info, block_data, sizeof(block_info_t), cudaMemcpyHostToDevice);

    // Copy pointer to symbol
    cudaMemcpyToSymbol(global_block_info, &d_block_info, sizeof(block_info_t*));
}
// static PyObject* get_utils_class(const char* class_name) {
//     // Get the main module's dict
//     PyObject* main_module = PyImport_AddModule("__main__");
//     PyObject* main_dict = PyModule_GetDict(main_module);

//     // Get the utils module (assuming it's imported as 'utils')
//     PyObject* utils = PyDict_GetItemString(main_dict, "utils");
//     if (!utils) {
//         PyErr_SetString(PyExc_ImportError, "Cannot find utils module");
//         return nullptr;
//     }

//     // Get the class from utils module
//     PyObject* class_obj = PyObject_GetAttrString(utils, class_name);
//     if (!class_obj) {
//         PyErr_SetString(PyExc_AttributeError, "Cannot find class in utils module");
//         return nullptr;
//     }

//     return class_obj;
// }
static PyObject* get_utils_class(const char* class_name) {
    // Get the main module's dict
    PyObject* main_module = PyImport_AddModule("__main__");
    if (!main_module) {
        printf("Failed to get __main__ module\n");
        return nullptr;
    }

    PyObject* main_dict = PyModule_GetDict(main_module);
    if (!main_dict) {
        printf("Failed to get main module dict\n");
        return nullptr;
    }

    // // Debug: Print all keys in main_dict
    // PyObject *key, *value;
    // Py_ssize_t pos = 0;
    // printf("Available modules in __main__:\n");
    // while (PyDict_Next(main_dict, &pos, &key, &value)) {
    //     const char* key_str = PyUnicode_AsUTF8(key);
    //     printf("  - %s\n", key_str);
    // }

    // Try different ways to get the utils module
    PyObject* utils = nullptr;

    // Try direct access to 'utils'
    utils = PyDict_GetItemString(main_dict, "utils");
    if (!utils) {
        // Try accessing through sys.modules
        PyObject* sys_modules = PyImport_GetModuleDict();
        utils = PyDict_GetItemString(sys_modules, "utils");
    }
    if (!utils) {
        printf("Cannot find utils module\n");
        return nullptr;
    }

    // // Debug: Print all attributes of utils module
    // printf("Available classes in utils module:\n");
    // PyObject* dir = PyObject_Dir(utils);
    // if (dir != nullptr) {
    //     Py_ssize_t size = PyList_Size(dir);
    //     for (Py_ssize_t i = 0; i < size; i++) {
    //         PyObject* attr = PyList_GetItem(dir, i);
    //         const char* attr_str = PyUnicode_AsUTF8(attr);
    //         printf("  - %s\n", attr_str);
    //     }
    //     Py_DECREF(dir);
    // }

    // Get the class from utils module
    PyObject* class_obj = PyObject_GetAttrString(utils, class_name);
    if (!class_obj) {
        printf("Cannot find class %s in utils module\n", class_name);
        return nullptr;
    }

    return class_obj;
}

// Add these helper functions to create Python dataclass instances directly
static PyObject* create_evm_call(const CuEVM::call_trace& call) {
    static PyObject* EVMCall = nullptr;

    // Cache the EVMCall class object (do this once)

    EVMCall = get_utils_class("EVMCall");
    if (!EVMCall) {
        printf("EVMCall class not found\n");
        return nullptr;
    }

    PyObject* args = Py_BuildValue("(iiNNNi)", call.pc, call.op, uint256_to_py_long(&call.sender),
                                   uint256_to_py_long(&call.receiver), uint256_to_py_long(&call.value), call.success);

    PyObject* call_instance = PyObject_CallObject(EVMCall, args);
    Py_DECREF(args);
    return call_instance;
}

static PyObject* create_trace_event(const CuEVM::simple_event_trace& event) {
    static PyObject* TraceEvent = nullptr;

    TraceEvent = get_utils_class("TraceEvent");
    if (!TraceEvent) {
        printf("TraceEvent class not found\n");
        return nullptr;
    }

    PyObject* args = Py_BuildValue("(iiNNN)", event.pc, event.op, uint256_to_py_long(&event.operand_1),
                                   uint256_to_py_long(&event.operand_2), uint256_to_py_long(&event.res));

    PyObject* event_instance = PyObject_CallObject(TraceEvent, args);
    Py_DECREF(args);
    return event_instance;
}

static PyObject* create_evm_branch(const CuEVM::branch_trace& branch) {
    static PyObject* EVMBranch = nullptr;

    EVMBranch = get_utils_class("EVMBranch");
    if (!EVMBranch) {
        printf("EVMBranch class not found\n");
        return nullptr;
    }

    PyObject* args =
        Py_BuildValue("(iiiN)", branch.pc_src, branch.pc_dst, branch.pc_missed, uint256_to_py_long(&branch.distance));

    PyObject* branch_instance = PyObject_CallObject(EVMBranch, args);
    Py_DECREF(args);
    return branch_instance;
}
// Add these new helper functions
static PyObject* create_evm_storage_write(uint32_t pc, const evm_word_t& key, const evm_word_t& value) {
    static PyObject* EVMStorageWrite = nullptr;

    if (EVMStorageWrite == nullptr) {
        EVMStorageWrite = get_utils_class("EVMStorageWrite");
        if (!EVMStorageWrite) {
            printf("EVMStorageWrite class not found\n");
            return nullptr;
        }
    }

    PyObject* args = Py_BuildValue("(iNN)", pc, uint256_to_py_long(&key), uint256_to_py_long(&value));

    PyObject* storage_write_instance = PyObject_CallObject(EVMStorageWrite, args);
    Py_DECREF(args);
    return storage_write_instance;
}

static PyObject* create_evm_bug(uint32_t pc, uint8_t opcode, const char* bug_type) {
    static PyObject* EVMBug = nullptr;

    if (EVMBug == nullptr) {
        EVMBug = get_utils_class("EVMBug");
        if (!EVMBug) {
            printf("EVMBug class not found\n");
            return nullptr;
        }
    }

    PyObject* args = Py_BuildValue("(iis)", pc, opcode, bug_type);
    PyObject* bug_instance = PyObject_CallObject(EVMBug, args);
    Py_DECREF(args);
    return bug_instance;
}

static PyObject* pyobject_from_simplified_trace(CuEVM::simplified_trace_data* trace_data) {
    PyObject* tracer_root = PyDict_New();
    PyObject* branches = PyList_New(0);
    PyObject* events = PyList_New(0);
    PyObject* calls = PyList_New(0);
    PyObject* storage_writes = PyList_New(0);
    // PyObject* bugs = PyList_New(0);

    // printf("trace data before conversion\n");
    // trace_data->print();
    // Process calls
    for (size_t idx = 0; idx < trace_data->no_calls; idx++) {
        PyObject* call_item = create_evm_call(trace_data->calls[idx]);
        PyList_Append(calls, call_item);
        Py_DECREF(call_item);
        // Detect ether leaking
        // if (detect_bug && trace_data->calls[idx].pc != 0) {
        //     evm_word_t zero;

        //     if (!uint256_is_zero(&trace_data->calls[idx].value)) {
        //         PyList_Append(bugs,
        //                       create_evm_bug(trace_data->calls[idx].pc, trace_data->calls[idx].op, "Leaking Ether"));
        //     }
        // }
    }

    // Process events
    for (size_t idx = 0; idx < trace_data->no_events; idx++) {
        PyObject* event_item = create_trace_event(trace_data->events[idx]);
        PyList_Append(events, event_item);
        Py_DECREF(event_item);

        // // Handle storage writes
        // if (trace_data->events[idx].opcode == OP_SSTORE) {
        //     PyObject* storage_write = create_evm_storage_write(
        //         trace_data->events[idx].pc, trace_data->events[idx].operand_1, trace_data->events[idx].operand_2);
        //     PyList_Append(storage_writes, storage_write);
        //     Py_DECREF(storage_write);
        // }
    }

    // Process branches
    for (size_t idx = 0; idx < trace_data->no_branches; idx++) {
        PyObject* branch_item = create_evm_branch(trace_data->branches[idx]);
        PyList_Append(branches, branch_item);
        Py_DECREF(branch_item);
    }

    PyDict_SetItemString(tracer_root, "events", events);
    PyDict_SetItemString(tracer_root, "branches", branches);
    PyDict_SetItemString(tracer_root, "calls", calls);
    // PyDict_SetItemString(tracer_root, "storage_write", storage_writes);
    // PyDict_SetItemString(tracer_root, "bugs", bugs);

    Py_DECREF(events);
    Py_DECREF(branches);
    Py_DECREF(calls);
    Py_DECREF(storage_writes);
    // Py_DECREF(bugs);

    return tracer_root;
}

PyObject* pyobject_from_serialized_state(CuEVM::serialized_worldstate_data* serialized_worldstate_instance) {
    PyObject* state_dict = PyDict_New();

    // Add accounts and storage elements
    // PyObject* accounts_list = PyList_New(0);
    uint32_t account_idx = 0;
    uint32_t storage_idx = 0;
    if (serialized_worldstate_instance == nullptr) {
        return state_dict;
    }
    // printf("cpu side\n");

    for (uint32_t i = 0; i < serialized_worldstate_instance->no_accounts; i++) {
        PyObject* account_dict = PyDict_New();
        // uint256_to_hex(hex_string_1, &serialized_worldstate_instance->balance[i]);
        // serialized_worldstate_instance->balance[i].to_hex(hex_string_1, 1);
        // printf("balance: %s\n", serialized_worldstate_instance->balance[i].to_hex());
        PyObject *py_long_1, *py_long_2;
        py_long_1 = uint256_to_py_long(&serialized_worldstate_instance->balance[i]);

        PyDict_SetItemString(account_dict, "balance", py_long_1);
        PyDict_SetItemString(account_dict, "nonce", PyLong_FromUnsignedLong(serialized_worldstate_instance->nonce[i]));

        // Add storage elements for the account if they exist
        PyObject* storage_dict = PyDict_New();
        while (storage_idx < serialized_worldstate_instance->no_storage_elements &&
               serialized_worldstate_instance->storage_indexes[storage_idx] == i) {
            PyObject* storage_key_value = PyDict_New();

            // serialized_worldstate_instance->storage_keys[storage_idx].to_hex(hex_string_1, 1);
            // printf("storage_key: %s\n", hex_string_1);
            // serialized_worldstate_instance->storage_values[storage_idx].to_hex(hex_string_2, 1);
            // printf("storage_value: %s\n", hex_string_2);
            py_long_1 = uint256_to_py_long(&serialized_worldstate_instance->storage_keys[storage_idx]);
            py_long_2 = uint256_to_py_long(&serialized_worldstate_instance->storage_values[storage_idx]);
            PyDict_SetItem(storage_dict, py_long_1, py_long_2);
            Py_DECREF(storage_key_value);
            storage_idx++;
        }

        PyDict_SetItemString(account_dict, "storage", storage_dict);
        // PyList_Append(accounts_list, account_dict);
        // uint256_to_hex(hex_string_1, &serialized_worldstate_instance->addresses[i]);
        PyObject* py_long_3 = uint256_to_py_long(&serialized_worldstate_instance->addresses[i]);

        PyDict_SetItem(state_dict, py_long_3, account_dict);
        Py_DECREF(account_dict);
    }
    // printf("state dict \n");
    // print_dict_recursive(state_dict, 1);
    return state_dict;
}

PyObject* pyobject_from_evm_instances(uint32_t num_instances, bool copy_state_data) {
    PyObject* root = PyDict_New();

    // todo_cl Need to copy from device memory before using
    CuEVM::simplified_trace_data* trace_data = new CuEVM::simplified_trace_data[num_instances];
    CuEVM::simplified_trace_data* d_trace_data;
    // First, retrieve the device pointers stored in the global symbols.
    CUDA_CHECK(cudaMemcpyFromSymbol(&d_trace_data, global_simplified_trace, sizeof(d_trace_data)));
    // Now copy the data arrays from the device to host memory.
    CUDA_CHECK(cudaMemcpy(trace_data, d_trace_data, sizeof(CuEVM::simplified_trace_data) * num_instances,
                          cudaMemcpyDeviceToHost));

    CuEVM::serialized_worldstate_data* world_data;
    CuEVM::serialized_worldstate_data* d_world_data;
    if (copy_state_data) {
        world_data = new CuEVM::serialized_worldstate_data[num_instances];
        CUDA_CHECK(cudaMemcpyFromSymbol(&d_world_data, global_serialized_worldstate, sizeof(d_world_data)));
        CUDA_CHECK(cudaMemcpy(world_data, d_world_data, sizeof(CuEVM::serialized_worldstate_data) * num_instances,
                              cudaMemcpyDeviceToHost));
    }
    // PyObject* world_state_json = pyobject_from_state_data_t(arith, instances.world_state_data);
    // PyDict_SetItemString(root, "pre", world_state_json);
    // Py_DECREF(world_state_json);
    PyObject* state_list = PyList_New(0);
    PyObject* trace_list = PyList_New(0);
    // PyDict_SetItemString(root, "post", instances_json);
    // Py_DECREF(instances_json);  // Decrement here because PyDict_SetItemString increases the ref count
    printf("num_instances: %d\n", num_instances);
    for (uint32_t idx = 0; idx < num_instances; idx++) {
        // printf("idx: %d\n", idx);

        PyObject* state_json;
        if (copy_state_data) {
            CuEVM::serialized_worldstate_data* serialized_worldstate = &world_data[idx];
            state_json = pyobject_from_serialized_state(serialized_worldstate);  // PyList_New(0);
        } else
            state_json = PyDict_New();

        PyObject* tracer_json = pyobject_from_simplified_trace(&trace_data[idx]);

        PyList_Append(state_list, state_json);   // Appends and steals the reference, so no need to DECREF
        PyList_Append(trace_list, tracer_json);  // Appends and steals the reference, so no need to DECREF
        Py_DECREF(state_json);
        Py_DECREF(tracer_json);
        // printf("done processing instance %d\n", idx);
    }
    PyDict_SetItemString(root, "states", state_list);
    PyDict_SetItemString(root, "traces", trace_list);

    Py_DECREF(state_list);
    Py_DECREF(trace_list);

    delete[] trace_data;
    if (copy_state_data) {
        delete[] world_data;
    }

    return root;
}
/*
 * Convert a Python int (assumed to be non-negative and fitting in 256 bits)
 * into a native uint256 value.
 */
int py_long_to_uint256(PyObject* py_num, uint256* dst) {
    if (!PyLong_Check(py_num)) {
        printf("pylong to uint256 error\n");
        PyErr_SetString(PyExc_TypeError, "Expected an int.");
        exit(0);
        // return 0;
    }

    // Zero out the destination buffer.
    // memset(dst->words, 0, sizeof(dst->words));

    // Py_ssize_t PyLong_AsNativeBytes(PyObject *pylong, void *buffer, Py_ssize_t n_bytes, int flags)¶
    return PyLong_AsNativeBytes(py_num, (unsigned char*)dst->words, sizeof(dst->words), -1);
}
/**
 * Convert a native uint256 structure to a Python long (PyLong) object.

 * Returns:
 *   A new reference to a PyLongObject on success, or NULL on failure.
 */
__host__ PyObject* uint256_to_py_long(const uint256* src) {
    if (src == nullptr) {
        printf("uint256_to_py_long error\n");
        PyErr_SetString(PyExc_ValueError, "NULL pointer provided for uint256_to_py_long conversion.");
        exit(0);
        // return NULL;
    }
    // PyObject *PyLong_FromUnsignedNativeBytes(const void *buffer, size_t n_bytes, int flags)¶
    return PyLong_FromUnsignedNativeBytes((const unsigned char*)src->words, sizeof(src->words), -1);
}

void freeTransactionList(TransactionList* d_transaction_list_ptr) {
    if (d_transaction_list_ptr == nullptr) {
        return;
    }

    // Create a temporary TransactionList to store device pointers
    TransactionList temp_list;
    CUDA_CHECK(cudaMemcpy(&temp_list, d_transaction_list_ptr, sizeof(TransactionList), cudaMemcpyDeviceToHost));

    // Free all device memory allocations
    if (temp_list.value != nullptr) {
        CUDA_CHECK(cudaFree(temp_list.value));
    }
    if (temp_list.gas_limit != nullptr) {
        CUDA_CHECK(cudaFree(temp_list.gas_limit));
    }
    if (temp_list.call_data != nullptr) {
        CUDA_CHECK(cudaFree(temp_list.call_data));
    }
    if (temp_list.call_data_offset != nullptr) {
        CUDA_CHECK(cudaFree(temp_list.call_data_offset));
    }
    if (temp_list.call_data_size != nullptr) {
        CUDA_CHECK(cudaFree(temp_list.call_data_size));
    }

    // Finally free the TransactionList itself
    CUDA_CHECK(cudaFree(d_transaction_list_ptr));
}

void freeTraceData(bool copy_state_data) {
    // Free simplified trace data
    CuEVM::simplified_trace_data* d_trace_data;
    CUDA_CHECK(cudaMemcpyFromSymbol(&d_trace_data, global_simplified_trace, sizeof(CuEVM::simplified_trace_data*)));
    if (d_trace_data != nullptr) {
        CUDA_CHECK(cudaFree(d_trace_data));
    }

    // Free serialized worldstate data if it was allocated
    if (copy_state_data) {
        CuEVM::serialized_worldstate_data* d_serialized_worldstate_data;
        CUDA_CHECK(cudaMemcpyFromSymbol(&d_serialized_worldstate_data, global_serialized_worldstate,
                                        sizeof(CuEVM::serialized_worldstate_data*)));
        if (d_serialized_worldstate_data != nullptr) {
            CUDA_CHECK(cudaFree(d_serialized_worldstate_data));
        }
    }
}

}  // namespace python_utils
