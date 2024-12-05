
#ifndef _CUEVM_LOGS_H_
#define _CUEVM_LOGS_H_

#include <cjson/cJSON.h>

#include <CuEVM/core/byte_array.cuh>
#include <CuEVM/core/evm_word.cuh>
namespace CuEVM {
constexpr CONSTANT uint32_t log_page_size = 20U; /**< The log page size*/
/**
 * The log data structure.
 * The log data structure is used to store the logs of the EVM.
 */
struct log_data_t {
    evm_word_t address;         /**< The address of the log*/
    evm_word_t topics[4];       /**< The topics of the log*/
    uint32_t no_topics;         /**< The number of topics of the log*/
    CuEVM::byte_array_t record; /**< The record of the log*/
};

/**
 * The log state data structure.
 */
struct log_state_data_t {
    uint32_t no_logs;  /**< The number of logs*/
    uint32_t capacity; /**< The capacity of the logs in allocated memory*/
    log_data_t *logs;  /**< The logs*/

    /**
     * The default constructor of the log state data structure.
     */
    __host__ __device__ log_state_data_t() : logs(nullptr), no_logs(0), capacity(0) {}

    /**
     * The destructor
     */
    __host__ __device__ ~log_state_data_t() {
        if (logs != nullptr && capacity > 0) {
            __ONE_GPU_THREAD_WOSYNC_BEGIN__
            delete[] logs;
            __ONE_GPU_THREAD_WOSYNC_END__
        }
        logs = nullptr;
        capacity = 0;
        no_logs = 0;
    }

    /**
     * Increase the capacity of log state
     * @return 0 if success, error otherwise
     */
    __host__ __device__ int32_t grow();

    /**
     * Push a log to the log state
     * @param[in] arith The arithmetic environment
     * @param[in] address The address of the log
     * @param[in] record The record of the log
     * @param[in] topic_1 The first topic of the log
     * @param[in] topic_2 The second topic of the log
     * @param[in] topic_3 The third topic of the log
     * @param[in] topic_4 The fourth topic of the log
     * @param[in] no_topics The number of topics of the log
     * @return 0 if success, error otherwise
     */
    __host__ __device__ int32_t push(const evm_word_t &address, const CuEVM::byte_array_t &record,
                                     const evm_word_t &topic_1, const evm_word_t &topic_2, const evm_word_t &topic_3,
                                     const evm_word_t &topic_4, const uint32_t &no_topics);

    /**
     * Update the log state with the logs from another log state
     * @param[in] arith The arithmetic environment
     * @param[in] other The other log state
     * @return 0 if success, error otherwise
     */
    __host__ __device__ int32_t update(const log_state_data_t &other);

    /**
     * Print the log state
     */
    __host__ __device__ void print() const;

    /**
     * Convert the log state to a JSON object
     * @return The JSON object
     */
    __host__ cJSON *to_json() const;
};
}  // namespace CuEVM

#endif