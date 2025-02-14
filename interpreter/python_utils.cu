#include <sstream>
#include <CuEVM/utils/python_utils.h>

#define CHECK_AND_RETURN_ON_ERROR(expr) do {                    \
    auto status = (expr);                                       \
    if (status != 0) {                                          \
      std::ostringstream oss;                                   \
      oss << "Error in expression: " << #expr                   \
          << " (status: 0x" << std::hex << status << std::dec   \
          << ") at " << __FILE__ << ":" << __LINE__;            \
      throw std::runtime_error(oss.str());                      \
    }                                                           \
  } while (0)

void copy_dict_recursive(PyObject* read_root, PyObject* write_root);
static PyObject* print_dict(PyObject* self, PyObject* args);

namespace python_utils {
  using namespace CuEVM;
  using CuEVM::transaction::TransactionList;
  using CuEVM::utils::simplified_trace_data;

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

  // similar to CuEVM/src/core/transaction.cu#get_transactions
TransactionList* getTransactionDataFromListofPyObject(PyObject* read_roots) {
    Py_ssize_t count = PyList_Size(read_roots);
    TransactionList* transactions;

    transactions = new TransactionList();


    transactions->call_data_offset = new uint32_t[count]();
    transactions->call_data_size = new uint32_t[count]();
    transactions->gas_limit = new uint64_t[count]();
    transactions->value = new evm_word_t[count];
    uint32_t curr_call_data_offset = 0;

    CuEVM::byte_array_t data_init;

    for (Py_ssize_t idx = 0; idx < count; idx++) {
        PyObject* read_root = PyList_GetItem(read_roots, idx);
        if (!PyDict_Check(read_root)) {
            PyErr_SetString(PyExc_TypeError, "Each item in the list must be a dictionary.");
            return NULL;
        }

        PyObject* data = PyDict_GetItemString(read_root, "transaction");
        // todo_cl data structure might differ from what's provided by Python
        PyObject* tx_data = PyList_GetItem(PyDict_GetItemString(data, "data"), 0); // data is a list with only one item ["0x..."]
        printf("tx_data: %s\n", PyUnicode_AsUTF8(tx_data));

        CHECK_AND_RETURN_ON_ERROR(data_init.from_hex(PyUnicode_AsUTF8(tx_data), LITTLE_ENDIAN, CuEVM::PaddingDirection::NO_PADDING));


        PyObject* tx_gas_limit = PyList_GetItem(PyDict_GetItemString(data, "gasLimit"), 0);
        PyObject* tx_value = PyList_GetItem(PyDict_GetItemString(data, "value"), 0);

        uint8_t type;
        type = 0;
        CHECK_AND_RETURN_ON_ERROR(transactions->nonce.from_hex(PyUnicode_AsUTF8(PyDict_GetItemString(data, "nonce"))));
        CHECK_AND_RETURN_ON_ERROR(data_init.from_hex(PyUnicode_AsUTF8(tx_data), LITTLE_ENDIAN, CuEVM::PaddingDirection::NO_PADDING));
        CHECK_AND_RETURN_ON_ERROR(transactions->sender.from_hex(PyUnicode_AsUTF8(PyDict_GetItemString(data, "sender"))));
        transactions->max_fee_per_gas = 0;
        transactions->max_priority_fee_per_gas = 0;
        CHECK_AND_RETURN_ON_ERROR(transactions->gas_price.from_hex("0x0a"));

        transactions->type = type;

        evm_word_t tmp;

        CHECK_AND_RETURN_ON_ERROR(tmp.from_hex(PyUnicode_AsUTF8(tx_gas_limit)));
        transactions->gas_limit[idx] = uint256_get_uint64_t(&tmp);
        CHECK_AND_RETURN_ON_ERROR(transactions->value[idx].from_hex(PyUnicode_AsUTF8(tx_value)));

        transactions->call_data_offset[idx] = curr_call_data_offset;
        transactions->call_data_size[idx] = data_init.size;

        if (data_init.size > 0) {
            if (transactions->call_data == nullptr) {
                transactions->call_data = new uint8_t[data_init.size];
            } else {
                uint8_t *tmp = new uint8_t[curr_call_data_offset];
                memcpy(tmp, transactions->call_data, curr_call_data_offset);
                delete[] transactions->call_data;
                transactions->call_data = tmp;
            }
            memcpy(&transactions->call_data[curr_call_data_offset], data_init.data, data_init.size);
        }

        curr_call_data_offset += data_init.size;
    }

    auto call_data_size = curr_call_data_offset;

    // Copy the transaction data to device memory,
    TransactionList *d_transaction_list_ptr;
    TransactionList *temp_transaction_list_ptr = new TransactionList();


    memcpy(temp_transaction_list_ptr, transactions, sizeof(TransactionList));
    CUDA_CHECK(cudaMalloc(&temp_transaction_list_ptr->value, count * sizeof(evm_word_t)));
    CUDA_CHECK(cudaMalloc(&temp_transaction_list_ptr->gas_limit, count * sizeof(gas_t)));
    CUDA_CHECK(cudaMalloc(&temp_transaction_list_ptr->call_data, call_data_size * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&temp_transaction_list_ptr->call_data_offset, count * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&temp_transaction_list_ptr->call_data_size, count * sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpy(temp_transaction_list_ptr->value, transactions->value,
                          count * sizeof(evm_word_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(temp_transaction_list_ptr->gas_limit, transactions->gas_limit,
                          count * sizeof(gas_t), cudaMemcpyHostToDevice));
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
  StateDb* getPreStateDataFromFristTransaction(PyObject* readroot, uint32_t num_states) {
    PyObject *key, *value;
    Py_ssize_t pos = 0;
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

    state_db_cpu->dynamic_accounts = new DynamicAccount *[num_states];

    // cpu side prealloc with num_accounts because we dont know the number of contracts yet.
    // GPU side will only use num_contracts
    state_db_cpu->prealloc_keys_pool = new evm_word_t[account_prealloc_keys_size * num_accounts * num_states];
    state_db_cpu->prealloc_values_pool = new ValueStatus[account_prealloc_keys_size * num_accounts * num_states];
    state_db_cpu->account_is_warm = new bool[num_states * num_accounts];
    // dynamic storage to grow later
    state_db_cpu->dynamic_storage_pages = new StateDbStoragePage *[num_states * num_accounts];
    state_db_cpu->dynamic_pool_capacity = new uint32_t[num_states * num_accounts];

    // state_db->snapshot_total_storage_size = new uint32_t[num_states * num_accounts];
    cJSON *account_json;
    uint32_t bytecode_offset = 0;

    memset(state_db_cpu->account_is_warm, 0, num_states * num_accounts * sizeof(bool));
    memset(state_db_cpu->dynamic_pool_capacity, 0, num_states * num_accounts * sizeof(uint32_t));

    memset(state_db_cpu->prealloc_keys_pool, 0,
           account_prealloc_keys_size * num_accounts * num_states * sizeof(evm_word_t));
    memset(state_db_cpu->prealloc_values_pool, 0,
           account_prealloc_keys_size * num_accounts * num_states * sizeof(ValueStatus));
    memset(state_db_cpu->dynamic_accounts, 0, num_states * sizeof(DynamicAccount *));

    evm_word_t t_word;

    while (PyDict_Next(data, &pos, &key, &value)) {
        int idx = pos - 1;  // Adjust index since PyDict_Next increments pos
        const char* address_str = PyUnicode_AsUTF8(key);

        // Extract balance, nonce, and code
        const char* balance = GET_STR_FROM_DICT_WITH_DEFAULT(value, "balance", "0x0");
        const char* nonce = GET_STR_FROM_DICT_WITH_DEFAULT(value, "nonce", "0x0");
        const char* code = GET_STR_FROM_DICT_WITH_DEFAULT(value, "code", "");
        PyObject* storage_dict = PyDict_GetItemString(value, "storage");
        uint32_t storage_size = PyDict_Size(storage_dict);


        // Initialize account details
        state_db_cpu->address_list[idx].from_hex(address_str);
        state_db_cpu->contract_index[idx] = -1;
        state_db_cpu->account_balances[idx].from_hex(balance);
        t_word.from_hex(nonce);
        state_db_cpu->account_nonces[idx] = uint256_get_uint32_t(&t_word);

        // Code
        byte_array_t byte_code;
        byte_code.from_hex(code, LITTLE_ENDIAN, NO_PADDING);
        state_db_cpu->account_codes_size[idx] = byte_code.size;
        state_db_cpu->account_codes_offset[idx] = bytecode_offset;

        if (byte_code.size > 0){
          uint8_t *tmp = state_db_cpu->all_account_codes;
          state_db_cpu->all_account_codes = new uint8_t[bytecode_offset + byte_code.size];
          memcpy(state_db_cpu->all_account_codes, tmp, bytecode_offset * sizeof(uint8_t));
          delete[] tmp;
          memcpy(&state_db_cpu->all_account_codes[bytecode_offset], byte_code.data, byte_code.size);
          bytecode_offset += byte_code.size;
        }



        // Storage
        if (storage_size > 0){
          for (uint32_t i = 0; i < num_states; i++) {
            state_db_cpu->account_storage_size[idx * num_states + i] = storage_size;
          }
          state_db_cpu->num_storage_elements += state_db_cpu->account_storage_size[idx * num_states];
          if (storage_size > account_prealloc_keys_size) {
            printf("PreState storage size: %d greater than supported\n", storage_size);
            return nullptr;
          }
        }else {
          memset(&state_db_cpu->account_storage_size[idx * num_states], 0, num_states * sizeof(uint32_t));
        }


        if (byte_code.size > 0 || storage_size > 0) {
          state_db_cpu->contract_index[idx] = state_db_cpu->num_contracts;
          state_db_cpu->num_contracts++;
        }

        // Iterate through the dictionary
        PyObject *key_storage, *value_storage;
        while (PyDict_Next(storage_dict, &pos, &key_storage, &value_storage)) {
          int storage_idx = pos - 1;  // Adjust index since PyDict_Next increments pos
          Py_INCREF(key_storage);
          Py_INCREF(value_storage);

          uint32_t contract_idx = state_db_cpu->num_contracts - 1;
          uint32_t pre_alloc_keys_idx = (account_prealloc_keys_size * contract_idx + storage_idx) * num_states;
          state_db_cpu->prealloc_keys_pool[pre_alloc_keys_idx].from_hex(PyUnicode_AsUTF8(key_storage));
          state_db_cpu->prealloc_values_pool[pre_alloc_keys_idx].from_hex(PyUnicode_AsUTF8(value_storage));

          // duplicate the value for all states
          for (uint32_t j = 1; j < num_states; j++) {
            state_db_cpu->prealloc_keys_pool[pre_alloc_keys_idx + j] = state_db_cpu->prealloc_keys_pool[pre_alloc_keys_idx];
            state_db_cpu->prealloc_values_pool[pre_alloc_keys_idx + j] = state_db_cpu->prealloc_values_pool[pre_alloc_keys_idx];
          }
        }
    }

    // copy to device. c.f., StateDb::GPUfromJson
    StateDb *tmp_state_db = new StateDb(num_states);
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
        cudaMalloc(&tmp_state_db->dynamic_storage_pages, num_states * num_accounts * sizeof(StateDbStoragePage *)));
    CUDA_CHECK(cudaMalloc(&tmp_state_db->dynamic_pool_capacity, num_states * num_accounts * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&tmp_state_db->account_is_warm, num_states * num_accounts * sizeof(bool)));

    CUDA_CHECK(cudaMalloc(&tmp_state_db->dynamic_accounts, num_states * sizeof(DynamicAccount *)));

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
                          num_states * sizeof(DynamicAccount *), cudaMemcpyHostToDevice));

    // dynamic storage
    CUDA_CHECK(cudaMemcpy(tmp_state_db->dynamic_pool_capacity, state_db_cpu->dynamic_pool_capacity,
                          num_states * num_accounts * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tmp_state_db->account_is_warm, state_db_cpu->account_is_warm,
                          num_states * num_accounts * sizeof(bool), cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy(tmp_state_db->snapshot_total_storage_size, state_db_cpu->snapshot_total_storage_size,
    //                       num_states * num_accounts * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // printf("state db cpu\n");
    // state_db_cpu->print();

    StateDb *state_db_gpu;
    CUDA_CHECK(cudaMalloc(&state_db_gpu, sizeof(StateDb)));
    CUDA_CHECK(cudaMemcpy(state_db_gpu, tmp_state_db, sizeof(StateDb), cudaMemcpyHostToDevice));
    cudaMemcpyToSymbol(global_state_db_ptr, &state_db_gpu, sizeof(StateDb *));
    delete state_db_cpu;
    delete tmp_state_db;

    return state_db_gpu;
}

void get_evm_instances_from_PyObject(CuEVM::evm_instance_t*& evm_instances, PyObject* read_roots, uint32_t& num_instances) {

    uint32_t num_transactions = PyList_Size(read_roots);
    num_instances = num_transactions;
    evm_instances = new evm_instance_t[num_instances];

    TransactionList* all_transactions = getTransactionDataFromListofPyObject(read_roots);
    StateDb* state_db = getPreStateDataFromFristTransaction(read_roots, num_transactions);

    // share world state data
    StateDb* shared_world_state_data = nullptr;
    shared_world_state_data = new StateDb(num_transactions);

#ifdef BUILD_LIBRARY
    // Simplified trace data
    CuEVM::utils::simplified_trace_data *trace_data, *d_trace_data;
    trace_data = new CuEVM::utils::simplified_trace_data[num_transactions]();
    CUDA_CHECK(cudaMalloc(&d_trace_data, num_transactions * sizeof(CuEVM::utils::simplified_trace_data)));
    CUDA_CHECK(cudaMemcpy(d_trace_data, trace_data, num_transactions * sizeof(CuEVM::utils::simplified_trace_data), cudaMemcpyHostToDevice));
    // todo_cl: allocate serialized_worldstate_data_ptr here
#endif

    for (uint32_t index = 0; index < num_transactions; index++) {
        evm_instances[index].state_db_ptr = state_db;
        evm_instances[index].transaction_list_ptr = all_transactions;
#ifdef BUILD_LIBRARY
        evm_instances[index].simplified_trace_data_ptr = d_trace_data; // Depends on how we use it, might be wrong.
        delete[] trace_data;
#endif
    }
}

static PyObject* pyobject_from_simplified_trace(CuEVM::utils::simplified_trace_data &trace_data) {
    PyObject* tracer_root = PyDict_New();
    return tracer_root;

    PyObject* branches = PyList_New(0);
    PyObject* events = PyList_New(0);
    PyObject* calls = PyList_New(0);
    char hex_string[43];
    // process call
    for (size_t idx = 0; idx < trace_data.no_calls; idx++) {
        PyObject* call_item = PyDict_New();
        PyDict_SetItemString(call_item, "sender",
                             PyUnicode_FromString(trace_data.calls[idx].sender.address_to_hex(hex_string)));
        PyDict_SetItemString(call_item, "receiver",
                             PyUnicode_FromString(trace_data.calls[idx].receiver.address_to_hex(hex_string)));

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
    if (serialized_worldstate_instance == nullptr){
        return state_dict;
    }
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

    assert(instances[0].simplified_trace_data_ptr != nullptr);
    assert(instances[0].serialized_worldstate_data_ptr != nullptr);

    // todo_cl Need to copy from device memory before using
    CuEVM::utils::simplified_trace_data *trace_data = new CuEVM::utils::simplified_trace_data[num_instances]();
    CuEVM::serialized_worldstate_data *world_data = new CuEVM::serialized_worldstate_data[num_instances]();

    CUDA_CHECK(cudaMemcpy(trace_data, instances[0].simplified_trace_data_ptr, sizeof(trace_data), cudaMemcpyDefault));
    CUDA_CHECK(cudaMemcpy(world_data, instances[0].serialized_worldstate_data_ptr, sizeof(world_data), cudaMemcpyDefault));

    // PyObject* world_state_json = pyobject_from_state_data_t(arith, instances.world_state_data);
    // PyDict_SetItemString(root, "pre", world_state_json);
    // Py_DECREF(world_state_json);
    PyObject* instances_json = PyList_New(0);
    PyDict_SetItemString(root, "post", instances_json);
    Py_DECREF(instances_json);  // Decrement here because PyDict_SetItemString increases the ref count

    for (uint32_t idx = 0; idx < num_instances; idx++) {
      CuEVM::serialized_worldstate_data* serialized_worldstate = world_data + idx;

        PyObject* instance_json = PyDict_New();

        PyObject* state_json = pyobject_from_serialized_state(serialized_worldstate);  // PyList_New(0);

        PyDict_SetItemString(instance_json, "state", state_json);
        PyObject* tracer_json = pyobject_from_simplified_trace(trace_data[idx]);
        PyDict_SetItemString(instance_json, "trace", tracer_json);
        PyList_Append(instances_json, instance_json);  // Appends and steals the reference, so no need to DECREF
    }

    delete[] trace_data;
    delete[] world_data;

    return root;
}

}  // namespace python_utils
