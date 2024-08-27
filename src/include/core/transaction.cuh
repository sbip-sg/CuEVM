// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2024 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2024-07-12
// SPDX-License-Identifier: MIT
#ifndef _CUEVM_TRANSACTION_H_
#define _CUEVM_TRANSACTION_H_

#include "../utils/arith.cuh"
#include "byte_array.cuh"
#include "block_info.cuh"
#include "../state/access_state.cuh"
#include "../state/touch_state.cuh"
#include "message.cuh"
#include <cjson/cJSON.h>

namespace cuEVM {
    namespace transaction {
        /**
         * The access list account.
         * YP: \f$E_{a}\f$
         */
        struct access_list_account_t
        {
            evm_word_t address; /**< The address of the account YP: \f$a\f$ */
            uint32_t storage_keys_count; /**< The number of the storage keys YP: \f$|E_{s}|\f$ */
            evm_word_t *storage_keys; /**< The storage keys YP: \f$E_{s}\f$ */

            /**
             * The default constructor.
             */
            __host__ __device__ access_list_account_t() : address(), storage_keys_count(0), storage_keys(nullptr) {}

            /**
             * free the storage keys
             * @param[in] managed the managed flag.
             * @return 0 for success, error code for failure.
             */
            __host__ __device__ int32_t free(
                int32_t managed = 0);

            /**
             * Get the access list storage keys from json
             * @param[in] json the json object.
             * @param[in] managed the managed flag.
             * @return 0 for success, error code for failure.
             */
            __host__ int32_t from_json(
                const cJSON* json,
                int32_t managed = 0);
        };
        /**
         * The access list.
         * YP: \f$T_{A}\f$
         */
        struct access_list_t
        {
            uint32_t accounts_count; /**< The number of the accounts YP: \f$|T_{A}|\f$ */
            access_list_account_t *accounts; /**< The accounts \f$T_{A}\f$ */

            /**
             * The default constructor.
             */
            __host__ __device__ access_list_t() : accounts_count(0), accounts(nullptr) {}

            /**
             * free the accounts
             * @param[in] managed the managed flag.
             * @return 0 for success, error code for failure.
             */
            __host__ __device__ int32_t free(
                int32_t managed = 0);

            /**
             * Get the access list from json
             * @param[in] json the json object.
             * @param[in] managed the managed flag.
             * @return 0 for success, error code for failure.
             */
            __host__ int32_t from_json(
                const cJSON* json,
                int32_t managed = 0);
        };
        /**
         * The transaction struct.
         * YP: \f$T\f$
         * It does not contains:
         * - chain id (YP: \f$T_{c}\f$)
         * - yParity (YP: \f$T_{y}\f$)
         * - w (YP: \f$T_{w}\f$)
         */
        struct evm_transaction_t
        {
            uint32_t type;                        /**< The transaction type EIP-2718 (YP: YP: \f$\T_{x}\f$) */
            evm_word_t nonce;                    /**< The nonce YP: \f$T_{n}\f$ */
            evm_word_t gas_limit;                /**< The gas limit YP: \f$T_{g}\f$ */
            evm_word_t to;                       /**< The recipient address YP: \f$T_{t}\f$ */
            evm_word_t value;                    /**< The value YP: \f$T_{v}\f$ */
            evm_word_t sender;                   /**< The sender address YP: \f$T_{s}\f$ or \f$T_{r}\f$*/
            evm_word_t max_fee_per_gas;          /**< The max fee per gas YP: \f$T_{m}\f$ */
            evm_word_t max_priority_fee_per_gas; /**< The max priority fee per gas YP: \f$T_{f}\f$ */
            evm_word_t gas_price;                /**< The gas proce YP: \f$T_{p}\f$ */
            cuEVM::byte_array_t data_init;            /**< The init or data YP:\f$T_{i}\f$ or \f$T_{d}\f$ */
            access_list_t access_list;           /**< The state YP: \f$T_{A}\f$ entry are \f$E=(E_{a}, E_{s})\f$*/

            // TODO: better constructors and destructors
            __host__ __device__ evm_transaction_t() : type(), nonce(), gas_limit(), to(), value(), sender(), max_fee_per_gas(), max_priority_fee_per_gas(), gas_price(), data_init(), access_list() {}

            /**
             * the destructor. TODO: improve it
             */
            __host__ __device__ ~evm_transaction_t();

            /**
             * get the nonce of the transaction
             * @param[in] arith the arithmetic environment.
             * @param[out] nonce the nonce of the transaction YP: \f$T_{n}\f$.
             */
            __host__ __device__ void get_nonce(
                ArithEnv &arith,
                bn_t &nonce) const;
            
            /**
             * get the gas limit of the transaction
             * @param[in] arith the arithmetic environment.
             * @param[out] gas_limit the gas limit of the transaction YP: \f$T_{g}\f$.
             */
            __host__ __device__ void get_gas_limit(
                ArithEnv &arith,
                bn_t &gas_limit) const;
            /**
             * get the to address of the transaction
             * @param[in] arith the arithmetic environment.
             * @param[out] to the to address of the transaction YP: \f$T_{t}\f$.
             */
            __host__ __device__ void get_to(
                ArithEnv &arith,
                bn_t &to) const;

            /**
             * get the value of the transaction
             * @param[in] arith the arithmetic environment.
             * @param[out] value the value of the transaction YP: \f$T_{v}\f$.
             */
            __host__ __device__ void get_value(
                ArithEnv &arith,
                bn_t &value) const;

            /**
             * get the sender address of the transaction
             * @param[in] arith the arithmetic environment.
             * @param[out] sender the sender address of the transaction YP: \f$T_{s}\f$ or \f$T_{r}\f$.
             */
            __host__ __device__ void get_sender(
                ArithEnv &arith,
                bn_t &sender) const;

            /**
             * get the max fee per gas of the transaction
             * @param[in] arith the arithmetic environment.
             * @param[out] max_fee_per_gas the max fee per gas of the transaction YP: \f$T_{m}\f$.
             */
            __host__ __device__ void get_max_fee_per_gas(
                ArithEnv &arith,
                bn_t &max_fee_per_gas) const;

            /**
             * get the max priority fee per gas of the transaction
             * @param[in] arith the arithmetic environment.
             * @param[out] max_priority_fee_per_gas the max priority fee per gas of the transaction YP: \f$T_{f}\f$.
             */
            __host__ __device__ void get_max_priority_fee_per_gas(
                ArithEnv &arith,
                bn_t &max_priority_fee_per_gas) const;

            /**
             * get the gas price of the transaction
             * @param[in] arith the arithmetic environment.
             * @param[out] gas_price the gas price of the transaction YP: \f$T_{p}\f$.
             * @return 0 for success, error code for failure.
             */
            __host__ __device__ int32_t get_gas_price(
                ArithEnv &arith,
                const cuEVM::block_info_t &block_info,
                bn_t &gas_price) const;

            /**
             * get the data of the transaction
             * @param[in] arith the arithmetic environment.
             * @param[out] data_init the data of the transaction YP: \f$T_{i}\f$ or \f$T_{d}\f$.
             */
            __host__ __device__ void get_data(
                ArithEnv &arith,
                byte_array_t &data_init) const;

            /**
             * Get if the is a contract creation transaction
             * @param[in] arith the arithmetic environment.
             * @return 1 if the transaction is a contract creation transaction, 0 otherwise.
             */
            __host__ __device__ int32_t is_contract_creation(
                ArithEnv &arith) const;

            /**
             * Get the transaction fees
             * @param[in] arith the arithmetic environment.
             * @param[in] block_info the block information.
             * @param[out] gas_value the gas value YP: \f$T_{g} \cdot p\f$.
             * @param[out] gas_limit the gas limit YP: \f$T_{g}\f$.
             * @param[out] gas_price the gas price YP: \f$T_{p}\f$.
             * @param[out] gas_priority_fee the gas priority fee YP: \f$f\f$.
             * @param[out] up_front_cost the up front cost YP: \f$v_{0}\f$.
             * @param[out] m the max fee per gas YP: \f$m\f$.
             * @return 0 for success, error code for failure.
             */
            __host__ __device__ int32_t get_transaction_fees(
                ArithEnv &arith,
                cuEVM::block_info_t &block_info,
                bn_t &gas_value,
                bn_t &gas_limit,
                bn_t &gas_price,
                bn_t &gas_priority_fee,
                bn_t &up_front_cost,
                bn_t &m) const;

            /**
             * warm up the access list
             * @param[in] arith the arithmetic environment.
             * @param[in] access_state the access state.
             * @return 0 for success, error code for failure.
             */
            __host__ __device__ int32_t access_list_warm_up(
                ArithEnv &arith,
                cuEVM::state::AccessState &access_state) const;

            /**
             * validate the transaction
             * @param[in] arith the arithmetic environment.
             * @param[in] access_state the access state.
             * @param[in] touch_state the touch state.
             * @param[in] block_info the block information.
             * @param[out] gas_used the gas used YP: \f$g_{0}\f$.
             * @param[out] gas_price the gas price YP: \f$p\f$.
             * @param[out] gas_priority_fee the gas priority fee YP: \f$f\f$.
             * @return 0 for success, error code for failure.
             */
            __host__ __device__ int32_t validate(
                ArithEnv &arith,
                cuEVM::state::AccessState &access_state,
                cuEVM::state::TouchState &touch_state,
                cuEVM::block_info_t &block_info,
                bn_t &gas_used,
                bn_t &gas_price,
                bn_t &gas_priority_fee) const;

            /**
             * get the message call from the transaction
             * @param[in] arith the arithmetic environment.
             * @param[in] access_state the access state.
             * @param[out] evm_message_call_ptr the message call.
             * @return 0 for success, error code for failure.
             */
            __host__ __device__ int32_t get_message_call(
                ArithEnv &arith,
                cuEVM::state::AccessState &access_state,
                cuEVM::evm_message_call_t* &evm_message_call_ptr) const;

            __host__ __device__ void print();

            __host__ cJSON* to_json();
        };

        /**
         * Get the number of transactions from json
         * @param[in] json the json object.
         * @return the number of transactions.
         */
        __host__ __device__ uint32_t no_transactions(
            const cJSON* json);

        /**
         * Get the transactions from json
         * @param[in] arith the arithmetic environment.
         * @param[out] transactions_ptr the transactions.
         * @param[in] json the json object.
         * @param[in] transactions_count the number of transactions.
         * @param[in] managed the managed flag.
         * @param[in] start_index the start index.
         * @param[in] clones the number of clones.
         * @return 0 for success, error code for failure.
         */
        __host__ int32_t get_transactios(
            ArithEnv &arith,
            evm_transaction_t* &transactions_ptr,
            const cJSON* json,
            uint32_t &transactions_count,
            int32_t managed = 0,
            uint32_t start_index = 0,
            uint32_t clones = 1);

        /**
         * free the transactions
         * @param[in] transactions_ptr the transactions.
         * @param[in] transactions_count the number of transactions.
         * @param[in] managed the managed flag.
         * @return 0 for success, error code for failure.
         */
        __host__ int32_t free_instaces(
            evm_transaction_t* transactions_ptr,
            uint32_t transactions_count,
            int32_t managed = 0);
    }
    // alias fro transaction
    using evm_transaction_t = transaction::evm_transaction_t;
}

#endif