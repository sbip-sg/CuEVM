// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#include "../include/state/logs.cuh"
#include "../utils/error_codes.cuh"


namespace cuEVM::state {
    __host__ __device__ int32_t log_state_data_t::grow() {
        log_data_t *new_logs = new log_data_t[capacity + log_page_size];
        if (new_logs == nullptr) {
            return ERROR_MEMORY_ALLOCATION_FAILED;
        }
        if (logs != nullptr) {
            std::copy(logs, logs + no_logs, new_logs);
            delete[] logs;
        }
        logs = new_logs;
        capacity = capacity + log_page_size;
        return ERROR_SUCCESS;
    }

    __host__ __device__ int32_t log_state_data_t::push(
        ArithEnv &arith,
        const bn_t &address,
        const cuEVM::byte_array_t &record,
        const bn_t &topic_1,
        const bn_t &topic_2,
        const bn_t &topic_3,
        const bn_t &topic_4,
        const uint32_t &no_topics
    ) {
        int32_t error_code = ERROR_SUCCESS;
        if (no_logs == capacity) {
            error_code |= grow();
        }
        logs[no_logs].record = record;
        cgbn_store(arith.env, &(logs[no_logs].address), address);
        cgbn_store(arith.env, &(logs[no_logs].topics[0]), topic_1);
        cgbn_store(arith.env, &(logs[no_logs].topics[1]), topic_2);
        cgbn_store(arith.env, &(logs[no_logs].topics[2]), topic_3);
        cgbn_store(arith.env, &(logs[no_logs].topics[3]), topic_4);
        logs[no_logs].no_topics = no_topics;
        no_logs++;
        return error_code;
    }

    __host__ __device__ int32_t log_state_data_t::update(
        ArithEnv &arith,
        const log_state_data_t &other
    ) {
        int32_t error_code = ERROR_SUCCESS;
        bn_t address, topic_1, topic_2, topic_3, topic_4;
        for (uint32_t idx = 0; idx < other.no_logs; idx++) {
            cgbn_load(arith.env, address, &(other.logs[idx].address));
            cgbn_load(arith.env, topic_1, &(other.logs[idx].topics[0]));
            cgbn_load(arith.env, topic_2, &(other.logs[idx].topics[1]));
            cgbn_load(arith.env, topic_3, &(other.logs[idx].topics[2]));
            cgbn_load(arith.env, topic_4, &(other.logs[idx].topics[3]));
            error_code |= push(
                arith,
                address,
                other.logs[idx].record,
                topic_1,
                topic_2,
                topic_3,
                topic_4,
                other.logs[idx].no_topics
            );
        }
        return error_code;
    }

    __host__ __device__ void log_state_data_t::print() const {
        printf("no_logs: %u\n", no_logs);
        for (uint32_t idx = 0; idx < no_logs; idx++) {
            printf("logs[%u]:\n", idx);
            printf("address: ");
            logs[idx].address.print();
            printf("\n");
            printf("no_topics: %u\n", logs[idx].no_topics);
            for (uint32_t jdx = 0; jdx < logs[idx].no_topics; jdx++) {
                printf("topics[%u]: ", jdx);
                logs[idx].topics[jdx].print();
            }
            logs[idx].record.print();
        }
    }

    __host__ cJSON* log_state_data_t::to_json() const {
        cJSON *log_data_json = cJSON_CreateObject();
        cJSON *logs_json = cJSON_CreateArray();
        cJSON *log_json = NULL;
        cJSON *topics_json = NULL;
        char *hex_string_ptr = new char[cuEVM::word_size * 2 + 3];
        for (uint32_t idx = 0; idx < no_logs; idx++) {
            log_json = cJSON_CreateObject();
            logs[idx].address.to_hex(hex_string_ptr, 0, 5);
            cJSON_AddStringToObject(log_json, "address", hex_string_ptr);
            topics_json = cJSON_CreateArray();
            for (uint32_t jdx = 0; jdx < logs[idx].no_topics; jdx++) {
                logs[idx].topics[jdx].to_hex(hex_string_ptr);
                cJSON_AddItemToArray(topics_json, cJSON_CreateString(hex_string_ptr));
            }
            cJSON_AddItemToObject(log_json, "topics", topics_json);
            cJSON_AddItemToObject(log_json, "record", logs[idx].record.to_json());
            cJSON_AddItemToArray(logs_json, log_json);
        }
        cJSON_AddItemToObject(log_data_json, "logs", logs_json);
        delete[] hex_string_ptr;
        hex_string_ptr = NULL;
        return log_data_json;
    }
}