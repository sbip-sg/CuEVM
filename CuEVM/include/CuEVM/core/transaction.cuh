#pragma once
#include <cjson/cJSON.h>

#include <CuEVM/core/block_info.cuh>
#include <CuEVM/core/byte_array.cuh>
#include <CuEVM/state/state_db.cuh>
#include <vector>

namespace CuEVM::transaction {

/**
 * Get the number of transactions from json
 * @param[in] json the json object.
 * @return the number of transactions.
 */
__host__ uint32_t no_transactions(const cJSON *json);

/**
 * Get the transactions from json
 * @param[out] transaction_list_ptrs the transactions.
 * @param[in] json the json object.
 * @param[in] transactions_count the number of transactions.
 * @param[in] num_gpus the number of GPUs.
 * @param[in] clones the number of clones.
 * @return 0 for success, error code for failure.
 */
__host__ int32_t get_transactions(std::vector<TransactionList *> &transaction_list_ptrs, const cJSON *json,
                                  uint32_t &transactions_count, uint32_t num_gpus = 1, uint32_t clones = 1);

/**
 * free the transactions
 * @param[in] transactions_ptr the transactions.
 * @param[in] transactions_count the number of transactions.
 * @param[in] managed the managed flag.
 * @return 0 for success, error code for failure.
 */
// __host__ int32_t free_instaces(evm_transaction_t *transactions_ptr, uint32_t transactions_count, int32_t managed =
// 0);

}  // namespace CuEVM::transaction
// alias fro transaction
// using evm_transaction_t = CuEVM::transaction::evm_transaction_t;
// namespace CuEVM
