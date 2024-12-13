#include <CuEVM/state/logs.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM {
__device__ int32_t log_state_data_t::grow() {
    log_data_t *new_logs;
    new_logs = new log_data_t[capacity + log_page_size];
    // printf("allocate capacity  %d for logpointer %p new_logs %p\n", capacity + log_page_size, this, new_logs);

    if (new_logs == nullptr) {
        return ERROR_MEMORY_ALLOCATION_FAILED;
    }
    if (logs != nullptr && no_logs > 0) {
        memcpy(new_logs, logs, no_logs * sizeof(log_data_t));
        delete[] logs;
    }
    logs = new_logs;
    capacity = capacity + log_page_size;
    return ERROR_SUCCESS;
}

__device__ int32_t log_state_data_t::push(const evm_word_t &address, const byte_array_t &record,
                                          const evm_word_t &topic_1, const evm_word_t &topic_2,
                                          const evm_word_t &topic_3, const evm_word_t &topic_4,
                                          const uint32_t &no_topics) {
    int32_t error_code = ERROR_SUCCESS;

    if (no_logs == capacity) {
        error_code |= grow();
    }

    logs[no_logs].record = record;
    logs[no_logs].address = address;
    logs[no_logs].topics[0] = topic_1;
    logs[no_logs].topics[1] = topic_2;
    logs[no_logs].topics[2] = topic_3;
    logs[no_logs].topics[3] = topic_4;
    logs[no_logs].no_topics = no_topics;
    no_logs++;

    return error_code;
}

__device__ int32_t log_state_data_t::update(const log_state_data_t &other) {
    int32_t error_code = ERROR_SUCCESS;
    evm_word_t address, topic_1, topic_2, topic_3, topic_4;

    for (uint32_t idx = 0; idx < other.no_logs; idx++) {
        address = other.logs[idx].address;
        topic_1 = other.logs[idx].topics[0];
        topic_2 = other.logs[idx].topics[1];
        topic_3 = other.logs[idx].topics[2];
        topic_4 = other.logs[idx].topics[3];
        error_code |=
            push(address, other.logs[idx].record, topic_1, topic_2, topic_3, topic_4, other.logs[idx].no_topics);
    }
    return error_code;
}

__device__ void log_state_data_t::print() const {
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

__host__ cJSON *log_state_data_t::to_json() const {
    cJSON *log_data_json = cJSON_CreateObject();
    cJSON *logs_json = cJSON_CreateArray();
    cJSON *log_json = NULL;
    cJSON *topics_json = NULL;
    char *hex_string_ptr = new char[CuEVM::word_size * 2 + 3];
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
}  // namespace CuEVM