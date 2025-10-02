#include <CuEVM/core/transaction.cuh>
#include <CuEVM/gas_cost.cuh>
#include <CuEVM/utils/cuda_utils.cuh>
#include <CuEVM/utils/error_codes.cuh>
#include <CuEVM/utils/evm_utils.cuh>
#include <CuEVM/utils/opcodes.cuh>

namespace CuEVM {

namespace transaction {

__host__ __device__ void TransactionList::print() {
    printf("TransactionList:\n");
    printf("nonce: %d\n", nonce.words[0]);
#ifndef BUILD_GO_LIBRARY
    printf("sender: ");
    sender.print();
    return;
#endif
    printf("to: ");
    to.print();
    printf("max_fee_per_gas: ");
    max_fee_per_gas.print();
    printf("max_priority_fee_per_gas: ");
    max_priority_fee_per_gas.print();
    printf("gas_price: ");
    gas_price.print();
    printf("type: %d\n", type);
    printf("size: %d\n", size);
#ifdef BUILD_GO_LIBRARY
    printf("gas_limit: %lu\n", gas_limit);
#endif
    for (uint32_t i = 0; i < size; i++) {
        printf("value[%d]: ", i);
        value[i].print();
#ifdef BUILD_GO_LIBRARY
        printf("sender[%d]: %d\n", i, sender[i]);
        printf("block_number[%d]: %lu\n", i, block_number[i]);
        printf("time_stamp[%d]: %lu\n", i, time_stamp[i]);
#endif
#ifndef BUILD_GO_LIBRARY
        printf("gas_limit[%d]: %lu\n", i, gas_limit[i]);
#endif
        printf("call_data_offset[%d]: %d\n", i, call_data_offset[i]);
        printf("call_data_size[%d]: %d\n", i, call_data_size[i]);
        printf("call_data[%d]: ", i);
        for (uint32_t j = 0; j < call_data_size[i]; j++) {
            printf("%02x", call_data[call_data_offset[i] + j]);
        }
        printf("\n");
    }
}

__host__ uint32_t no_transactions(const cJSON *json) {
    cJSON *transaction_json = cJSON_GetObjectItemCaseSensitive(json, "transaction");
    const cJSON *data_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "data");
    size_t data_counts = cJSON_GetArraySize(data_json);
    const cJSON *gas_limit_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "gasLimit");
    size_t gas_limit_counts = cJSON_GetArraySize(gas_limit_json);
    const cJSON *value_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "value");
    size_t value_counts = cJSON_GetArraySize(value_json);
    return data_counts * gas_limit_counts * value_counts;
}

__host__ int32_t get_transactions(std::vector<TransactionList *> &transaction_list_ptrs, const cJSON *json,
                                  uint32_t &transactions_count, uint32_t num_gpus, uint32_t clones) {
#ifdef BUILD_GO_LIBRARY
    return ERROR_SUCCESS;
#else
    cJSON *transaction_json = cJSON_GetObjectItemCaseSensitive(json, "transaction");
    uint32_t available_transactions = no_transactions(json);
    transaction_list_ptrs.resize(num_gpus);
    TransactionList *host_transaction_list_ptr = new TransactionList();

    uint32_t type = 0;

    const cJSON *nonce_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "nonce");
    host_transaction_list_ptr->nonce.from_hex(nonce_json->valuestring);

    const cJSON *gas_limit_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "gasLimit");
    uint32_t gas_limit_counts = cJSON_GetArraySize(gas_limit_json);

    const cJSON *sender_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "sender");
    host_transaction_list_ptr->sender.from_hex(sender_json->valuestring);

    const cJSON *to_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "to");
    host_transaction_list_ptr->to.from_hex(to_json->valuestring);

    const cJSON *max_fee_per_gas_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "maxFeePerGas");

    const cJSON *max_priority_fee_per_gas_json =
        cJSON_GetObjectItemCaseSensitive(transaction_json, "maxPriorityFeePerGas");

    const cJSON *gas_price_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "gasPrice");

    const cJSON *value_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "value");
    uint32_t value_counts = cJSON_GetArraySize(value_json);

    const cJSON *data_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "data");
    uint32_t data_counts = cJSON_GetArraySize(data_json);

    uint32_t original_count;
    original_count = data_counts;
    transactions_count = original_count * clones;

    if (transactions_count % num_gpus != 0) {
        printf("Error: transactions_count %d is not divisible by num_gpus %d\n", transactions_count, num_gpus);
        return -1;
    }

    host_transaction_list_ptr->size = transactions_count;
    uint32_t access_list_counts = 0;
    const cJSON *access_list_json = cJSON_GetObjectItem(transaction_json, "accessLists");
    if (access_list_json != nullptr) access_list_counts = cJSON_GetArraySize(access_list_json);

    host_transaction_list_ptr->access_list_gas_cost = 0;
    if (access_list_counts > 0) {
        // retrieve the first access list
        const cJSON *first_access_list_json = cJSON_GetArrayItem(access_list_json, 0);

        // Check if the access list is empty (array with 0 elements)
        if (first_access_list_json != nullptr && cJSON_IsArray(first_access_list_json)) {
            uint32_t access_list_size = cJSON_GetArraySize(first_access_list_json);

            // Iterate through each address entry in the access list
            for (uint32_t i = 0; i < access_list_size; i++) {
                const cJSON *address_entry = cJSON_GetArrayItem(first_access_list_json, i);
                if (address_entry != nullptr && cJSON_IsObject(address_entry)) {
                    // Add gas cost for the address
                    host_transaction_list_ptr->access_list_gas_cost += GAS_ACCESS_LIST_ADDRESS;

                    // Add gas cost for storage keys
                    const cJSON *storage_keys_json = cJSON_GetObjectItemCaseSensitive(address_entry, "storageKeys");
                    if (storage_keys_json != nullptr && cJSON_IsArray(storage_keys_json)) {
                        uint32_t storage_keys_count = cJSON_GetArraySize(storage_keys_json);
                        host_transaction_list_ptr->access_list_gas_cost += GAS_ACCESS_LIST_STORAGE * storage_keys_count;
                    }
                }
            }
        }
    }

    if ((max_fee_per_gas_json != nullptr) && (max_priority_fee_per_gas_json != nullptr) &&
        (gas_price_json == nullptr)) {
        type = 2;
        host_transaction_list_ptr->max_fee_per_gas.from_hex(max_fee_per_gas_json->valuestring);
        host_transaction_list_ptr->max_priority_fee_per_gas.from_hex(max_priority_fee_per_gas_json->valuestring);
        host_transaction_list_ptr->gas_price.from_uint32_t(0);
    } else if ((max_fee_per_gas_json == nullptr) && (max_priority_fee_per_gas_json == nullptr) &&
               (gas_price_json != nullptr)) {
        // if (access_list_json == nullptr) {
        //     type = 0;
        // } else {
        //     type = 1;
        // }
        type = 0;

        host_transaction_list_ptr->max_fee_per_gas.from_uint32_t(0);
        host_transaction_list_ptr->max_priority_fee_per_gas.from_uint32_t(0);
        host_transaction_list_ptr->gas_price.from_hex(gas_price_json->valuestring);
    } else {
        printf("ERROR_TRANSACTION_TYPE\n");
        return ERROR_TRANSACTION_TYPE;
    }
    // CREATE transaction
    if (strlen(to_json->valuestring) == 0) {
        type = SPECIAL_CREATE_TRANSACTION_TYPE;
    }

    host_transaction_list_ptr->type = type;

    host_transaction_list_ptr->call_data_offset = new uint32_t[transactions_count];
    host_transaction_list_ptr->call_data_size = new uint32_t[transactions_count];
    host_transaction_list_ptr->gas_limit = new uint64_t[transactions_count];
    host_transaction_list_ptr->value = new evm_word_t[transactions_count];
    memset(host_transaction_list_ptr->value, 0, transactions_count * sizeof(evm_word_t));

    uint32_t index, gas_limit_index, value_index, call_data_offset = 0;
    for (uint32_t idx = 0; idx < data_counts; idx++) {
        // simplified logic, the host is responsible for constructing the simple test tx list
        index = idx % data_counts;
        // if (access_list_counts > 0) {
        //     access_list_index = data_index % access_list_counts;  // TODO: check if this is correct
        //     template_transaction_ptr->access_list.from_json(cJSON_GetArrayItem(access_list_json, access_list_index),
        //     );
        // }
        gas_limit_index = index % gas_limit_counts;
        value_index = index % value_counts;

        CuEVM::byte_array_t data_init;
        data_init.from_hex(cJSON_GetArrayItem(data_json, index)->valuestring, LITTLE_ENDIAN,
                           CuEVM::PaddingDirection::NO_PADDING);

        if (data_init.size > 0) {
            if (host_transaction_list_ptr->call_data == nullptr) {
                host_transaction_list_ptr->call_data = new uint8_t[data_init.size];
            } else {
                uint8_t *tmp = new uint8_t[call_data_offset + data_init.size];
                memcpy(tmp, host_transaction_list_ptr->call_data, call_data_offset);
                delete[] host_transaction_list_ptr->call_data;
                host_transaction_list_ptr->call_data = tmp;
            }
            memcpy(&host_transaction_list_ptr->call_data[call_data_offset], data_init.data, data_init.size);
        }
        host_transaction_list_ptr->call_data_offset[idx] = call_data_offset;
        host_transaction_list_ptr->call_data_size[idx] = data_init.size;
        call_data_offset += data_init.size;
        evm_word_t tmp;
        tmp.from_hex(cJSON_GetArrayItem(gas_limit_json, gas_limit_index)->valuestring);
        host_transaction_list_ptr->gas_limit[idx] = uint256_get_uint64_t(&tmp);
        // printf("gas limit uint64_t %lu\n", transaction_list_ptr->gas_limit[idx]);
        // TODO check if it is appropriate to perform here
        // if (idx == 0) {
        //     sender = transaction_list_ptr->sender;
        //     uint256_mul(&upfront_cost, &tmp, &transaction_list_ptr->gas_price);
        // }

        host_transaction_list_ptr->value[idx].from_hex(cJSON_GetArrayItem(value_json, value_index)->valuestring);

        host_transaction_list_ptr->value[idx].print();
    }

    // multiply the data
    uint32_t multiplier = transactions_count / data_counts;
    for (uint32_t idx = 1; idx < multiplier; idx++) {
        memcpy(&host_transaction_list_ptr->call_data_offset[idx * data_counts],
               host_transaction_list_ptr->call_data_offset, data_counts * sizeof(uint32_t));
        memcpy(&host_transaction_list_ptr->call_data_size[idx * data_counts], host_transaction_list_ptr->call_data_size,
               data_counts * sizeof(uint32_t));
        memcpy(&host_transaction_list_ptr->gas_limit[idx * data_counts], host_transaction_list_ptr->gas_limit,
               data_counts * sizeof(uint64_t));
        memcpy(&host_transaction_list_ptr->value[idx * data_counts], host_transaction_list_ptr->value,
               data_counts * sizeof(evm_word_t));
    }

    uint32_t call_data_size = host_transaction_list_ptr->call_data_offset[transactions_count - 1] +
                              host_transaction_list_ptr->call_data_size[transactions_count - 1];
    // printf("call_data_size %d\n", call_data_size);
    // TransactionList *d_transaction_list_ptr;
    uint32_t transaction_per_gpu = transactions_count / num_gpus;
    for (uint32_t i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        TransactionList *temp_transaction_list_ptr = new TransactionList();
        memcpy(temp_transaction_list_ptr, host_transaction_list_ptr, sizeof(TransactionList));
        temp_transaction_list_ptr->size = transaction_per_gpu;
        CUDA_CHECK(cudaMalloc(&temp_transaction_list_ptr->value, transaction_per_gpu * sizeof(evm_word_t)));
        CUDA_CHECK(cudaMalloc(&temp_transaction_list_ptr->gas_limit, transaction_per_gpu * sizeof(gas_t)));
        CUDA_CHECK(cudaMalloc(&temp_transaction_list_ptr->call_data, call_data_size * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&temp_transaction_list_ptr->call_data_offset, transaction_per_gpu * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&temp_transaction_list_ptr->call_data_size, transaction_per_gpu * sizeof(uint32_t)));

        CUDA_CHECK(cudaMemcpy(temp_transaction_list_ptr->value,
                              host_transaction_list_ptr->value + i * transaction_per_gpu,
                              transaction_per_gpu * sizeof(evm_word_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(temp_transaction_list_ptr->gas_limit,
                              host_transaction_list_ptr->gas_limit + i * transaction_per_gpu,
                              transaction_per_gpu * sizeof(gas_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(temp_transaction_list_ptr->call_data, host_transaction_list_ptr->call_data,
                              call_data_size * sizeof(uint8_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(temp_transaction_list_ptr->call_data_offset,
                              host_transaction_list_ptr->call_data_offset + i * transaction_per_gpu,
                              transaction_per_gpu * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(temp_transaction_list_ptr->call_data_size,
                              host_transaction_list_ptr->call_data_size + i * transaction_per_gpu,
                              transaction_per_gpu * sizeof(uint32_t), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&transaction_list_ptrs[i], sizeof(TransactionList)));
        CUDA_CHECK(cudaMemcpy(transaction_list_ptrs[i], temp_transaction_list_ptr, sizeof(TransactionList),
                              cudaMemcpyHostToDevice));
        delete temp_transaction_list_ptr;
    }
    // printf("transaction on host\n");
    // transaction_list_ptr->print();

    // delete transaction_list_ptr;
    // transaction_list_ptr = d_transaction_list_ptr;
    return ERROR_SUCCESS;
#endif
}

}  // namespace transaction
}  // namespace CuEVM
