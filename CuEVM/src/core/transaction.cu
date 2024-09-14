// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2024 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2024-07-12
// SPDX-License-Identifier: MIT

#include <CuEVM/core/transaction.cuh>
#include <CuEVM/gas_cost.cuh>
#include <CuEVM/utils/opcodes.cuh>
#include <CuEVM/utils/evm_utils.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM {
    namespace transaction {
            __host__ __device__ int32_t access_list_account_t::free(
                int32_t managed) {
                if (storage_keys != nullptr) {
                    if (managed) {
                        CUDA_CHECK(cudaFree(storage_keys));
                    } else {
                        delete[] storage_keys;
                    }
                    storage_keys = nullptr;
                    storage_keys_count = 0;
                }
                return ERROR_SUCCESS;
            }

            __host__ int32_t access_list_account_t::from_json(
                const cJSON* json,
                int32_t managed) {
                cJSON* address_json = cJSON_GetObjectItemCaseSensitive(json, "address");
                if (address_json == NULL) {
                    return ERROR_FAILED;
                }
                if (!address.from_hex(address_json->valuestring)) {
                    return ERROR_FAILED;
                }
                cJSON* storage_keys_json = cJSON_GetObjectItemCaseSensitive(json, "storageKeys");
                if (storage_keys_json == NULL) {
                    storage_keys_count = 0;
                    return ERROR_SUCCESS;
                }
                storage_keys_count = cJSON_GetArraySize(storage_keys_json);
                if (storage_keys_count == 0) {
                    return ERROR_SUCCESS;
                }
                if (managed) {
                        CUDA_CHECK(cudaMallocManaged(
                            (void **)&(storage_keys),
                            storage_keys_count * sizeof(evm_word_t)));
                } else {
                    storage_keys = new evm_word_t[storage_keys_count];
                }
                for (uint32_t idx = 0; idx < storage_keys_count; idx++) {
                    if (storage_keys[idx].from_hex(cJSON_GetArrayItem(storage_keys_json, idx)->valuestring)) {
                        return ERROR_FAILED;
                    }
                }
                return ERROR_SUCCESS;
            }

            __host__ __device__ int32_t access_list_t::free(
                int32_t managed) {
                if (accounts != nullptr) {
                    for (uint32_t i = 0; i < accounts_count; i++) {
                        accounts[i].free(managed);
                    }
                    if (managed) {
                        CUDA_CHECK(cudaFree(accounts));
                    } else {
                        delete[] accounts;
                    }
                    accounts = nullptr;
                    accounts_count = 0;
                }
                return ERROR_SUCCESS;
            }

            __host__ int32_t access_list_t::from_json(
                const cJSON* json,
                int32_t managed) {
                accounts_count = cJSON_GetArraySize(json);
                if (accounts_count == 0) {
                    return ERROR_SUCCESS;
                }
                if (managed) {
                    CUDA_CHECK(cudaMallocManaged(
                        (void **)&(accounts),
                        accounts_count * sizeof(access_list_account_t)));
                } else {
                    accounts = new access_list_account_t[accounts_count];
                }

                for (uint32_t idx = 0; idx < accounts_count; idx++) {
                    if (accounts[idx].from_json(cJSON_GetArrayItem(json, idx), managed)) {
                        return ERROR_FAILED;
                    }
                }
                return ERROR_SUCCESS;
            }
            /**
             * the destructor. TODO: improve it
             */
            __host__ __device__ evm_transaction_t::~evm_transaction_t() {
            }

            __host__ __device__ void evm_transaction_t::get_nonce(
                ArithEnv &arith,
                bn_t &nonce) const {
                    cgbn_load(arith.env, nonce, (cgbn_evm_word_t_ptr) &(this->nonce));
            }

            __host__ __device__ void evm_transaction_t::get_gas_limit(
                ArithEnv &arith,
                bn_t &gas_limit) const {
                    cgbn_load(arith.env, gas_limit, (cgbn_evm_word_t_ptr) &(this->gas_limit));
            }

            __host__ __device__ void evm_transaction_t::get_to(
                ArithEnv &arith,
                bn_t &to) const {
                    cgbn_load(arith.env, to, (cgbn_evm_word_t_ptr) &(this->to));
            }

            __host__ __device__ void evm_transaction_t::get_value(
                ArithEnv &arith,
                bn_t &value) const {
                    cgbn_load(arith.env, value, (cgbn_evm_word_t_ptr) &(this->value));
            }

            __host__ __device__ void evm_transaction_t::get_sender(
                ArithEnv &arith,
                bn_t &sender) const {
                    cgbn_load(arith.env, sender, (cgbn_evm_word_t_ptr) &(this->sender));
            }

            __host__ __device__ void evm_transaction_t::get_max_fee_per_gas(
                ArithEnv &arith,
                bn_t &max_fee_per_gas) const {
                    cgbn_load(arith.env, max_fee_per_gas, (cgbn_evm_word_t_ptr) &(this->max_fee_per_gas));
            }

            __host__ __device__ void evm_transaction_t::get_max_priority_fee_per_gas(
                ArithEnv &arith,
                bn_t &max_priority_fee_per_gas) const {
                    cgbn_load(arith.env, max_priority_fee_per_gas, (cgbn_evm_word_t_ptr) &(this->max_priority_fee_per_gas));
            }

            __host__ __device__ int32_t evm_transaction_t::get_gas_price(
                ArithEnv &arith,
                const CuEVM::block_info_t &block_info,
                bn_t &gas_price) const {
                if ((type == 0) || (type == 1)) {
                    // \f$p = T_{p}\f$
                    cgbn_load(arith.env, gas_price, (cgbn_evm_word_t_ptr) &(this->gas_price));
                } else if (type == 2) {
                    cgbn_set_ui32(arith.env, gas_price, 0);
                    bn_t max_priority_fee_per_gas; // YP: \f$T_{f}\f$
                    bn_t max_fee_per_gas; // YP: \f$T_{m}\f$
                    bn_t gas_priority_fee; // YP: \f$f\f$
                    get_max_priority_fee_per_gas(arith, max_priority_fee_per_gas);
                    get_max_fee_per_gas(arith, max_fee_per_gas);
                    bn_t block_base_fee; // YP: \f$H_{f}\f$
                    block_info.get_base_fee(arith, block_base_fee);
                    // \f$T_{m} - H_{f}\f$
                    cgbn_sub(
                        arith.env,
                        gas_priority_fee,
                        max_fee_per_gas,
                        block_base_fee);
                    // \f$f=min(T_{f}, T_{m} - H_{f})\f$
                    if (cgbn_compare(arith.env, gas_priority_fee, max_priority_fee_per_gas) > 0)
                    {
                        cgbn_set(arith.env, gas_priority_fee, max_priority_fee_per_gas);
                    }
                    // \f$p = f + H_{f}\f$
                    cgbn_add(arith.env, gas_price, gas_priority_fee, block_base_fee);
                } else {
                    return ERROR_TRANSACTION_TYPE;
                }
                return ERROR_SUCCESS;
            }

            __host__ __device__ void evm_transaction_t::get_data(
                ArithEnv &arith,
                byte_array_t &data_init) const {
                    data_init = this->data_init;
            }

            __host__ __device__ int32_t evm_transaction_t::is_contract_creation(
                ArithEnv &arith) const {
                bn_t to;
                get_to(arith, to);
                return cgbn_compare_ui32(arith.env, to, 0) == 0;
            }

            __host__ __device__ int32_t evm_transaction_t::get_transaction_fees(
                ArithEnv &arith,
                CuEVM::block_info_t &block_info,
                bn_t &gas_value,
                bn_t &gas_limit,
                bn_t &gas_price,
                bn_t &gas_priority_fee,
                bn_t &up_front_cost,
                bn_t &m) const {

                bn_t max_fee_per_gas; // YP: \f$T_{m}\f$
                bn_t value; // YP: \f$T_{v}\f$
                get_max_fee_per_gas(arith, max_fee_per_gas);
                get_value(arith, value);
                get_gas_limit(arith, gas_limit);
                get_gas_price(arith, block_info, gas_price);
                bn_t block_base_fee; // YP: \f$H_{f}\f$
                block_info.get_base_fee(arith, block_base_fee);

            // \f$f = T_{p} - H_{f}\f$ //type 0 and 1
            // \f$p = f + H_{f}\f$ // type 2
                cgbn_sub(arith.env, gas_priority_fee, gas_price, block_base_fee);

                if ((type==0) || (type==1)) {
        // \f$v_{0} = T_{p} * T_{g} + T_{v}\f$
                    cgbn_mul(arith.env, up_front_cost, gas_limit, gas_price);
                    cgbn_add(arith.env, up_front_cost, up_front_cost, value);
        // \f$m = T_{p}\f$
                    cgbn_set(arith.env, m, gas_price);
                } else if (type==2) {
        // \f$v_{0} = T_{m} * T_{g} + T_{v}\f$
                    cgbn_mul(arith.env, up_front_cost, gas_limit, max_fee_per_gas);
                    cgbn_add(arith.env, up_front_cost, up_front_cost, value);
        // \f$m = T_{m}\f$
                    cgbn_set(arith.env, m, max_fee_per_gas);
                } else {
                    return ERROR_TRANSACTION_TYPE;
                }

                // gas value \f$= T_{g} \dot p\f$
                cgbn_mul(arith.env, gas_value, gas_limit, gas_price);
                return ERROR_SUCCESS;
            }

            __host__ __device__ int32_t evm_transaction_t::access_list_warm_up(
                ArithEnv &arith,
                CuEVM::state::AccessState &access_state) const {
                for (uint32_t i = 0; i < access_list.accounts_count; i++) {
                    bn_t address;
                    cgbn_load(arith.env, address, (cgbn_evm_word_t_ptr) &(access_list.accounts[i].address));
                    CuEVM::account::account_t* account_ptr = nullptr;
                    access_state.get_account(arith, address, account_ptr, ACCOUNT_NONE_FLAG);
                    for (uint32_t j = 0; j < access_list.accounts[i].storage_keys_count; j++) {
                        bn_t key;
                        cgbn_load(arith.env, key, (cgbn_evm_word_t_ptr) &(access_list.accounts[i].storage_keys[j]));
                        bn_t value;
                        access_state.get_value(arith, address, key, value);
                    }
                }
                return ERROR_SUCCESS;
            }

            __host__ __device__ int32_t evm_transaction_t::validate(
                ArithEnv &arith,
                CuEVM::state::AccessState &access_state,
                CuEVM::state::TouchState &touch_state,
                CuEVM::block_info_t &block_info,
                bn_t &gas_used,
                bn_t &gas_price,
                bn_t &gas_priority_fee) const {

                bn_t gas_intrinsic;
                CuEVM::gas_cost::transaction_intrinsic_gas(arith, *this, gas_intrinsic);
                bn_t gas_limit;
                bn_t gas_value;
                bn_t up_front_cost;
                bn_t m;
                if (get_transaction_fees(arith, block_info, gas_value, gas_limit, gas_price, gas_priority_fee, up_front_cost, m)) {
                    return ERROR_TRANSACTION_FEES;
                }

                bn_t sender_address;
                get_sender(arith, sender_address);
                CuEVM::account::account_t* sender_account = nullptr;
                access_state.get_account(arith, sender_address, sender_account, ACCOUNT_BALANCE_FLAG | ACCOUNT_NONCE_FLAG | ACCOUNT_BYTE_CODE_FLAG);
                bn_t sender_balance;
                sender_account->get_balance(arith, sender_balance);
                bn_t sender_nonce;
                sender_account->get_nonce(arith, sender_nonce);
                bn_t transaction_nonce;
                get_nonce(arith, transaction_nonce);
                bn_t max_fee_per_gas;
                get_max_fee_per_gas(arith, max_fee_per_gas);
                bn_t max_priority_fee_per_gas;
                get_max_priority_fee_per_gas(arith, max_priority_fee_per_gas);

                // Next possible errors in the transaction context:
                // sender is an empty account YP: \f$\sigma(T_{s}) \neq \varnothing\f$
                // sender is a contract YP: \f$\sigma(T_{s})_{c} \eq KEC(())\f$
                if (
                    (sender_account == nullptr) ||
                    (sender_account->is_empty()) ||
                    (sender_account->is_contract())
                ) {
                    return ERROR_TRANSACTION_SENDER_EMPTY;
                }
                // nonce are different YP: \f$T_{n} \eq \sigma(T_{s})_{n}\f$
                if (cgbn_compare(arith.env, sender_nonce, transaction_nonce)) {
                    return ERROR_TRANSACTION_NONCE;
                }
                // sent gas is less than intrinisec gas YP: \f$T_{g} \geq g_{0}\f$
                if (cgbn_compare(arith.env, gas_limit, gas_intrinsic) < 0) {
                    return ERROR_TRANSACTION_GAS;
                }
                // balance is less than up front cost YP: \f$\sigma(T_{s})_{b} \geq v_{0}\f$
                if (cgbn_compare(arith.env, sender_balance, up_front_cost) < 0) {
                    return ERROR_TRANSACTION_SENDER_BALANCE;
                }
                // gas fee is less than than block base fee YP: \f$m \geq H_{f}\f$
                bn_t block_base_fee;
                block_info.get_base_fee(arith, block_base_fee);
                if (cgbn_compare(arith.env, m, block_base_fee) < 0) {
                    return ERROR_TRANSACTION_GAS_PRICE;
                }
                // Max priority fee per gas is higher than max fee per gas YP: \f$T_{m} \geq T_{f}\f$
                if (cgbn_compare(arith.env, max_fee_per_gas, max_priority_fee_per_gas) < 0) {
                    return ERROR_TRANSACTION_GAS_PRIORITY;
                }
                // the other verification is about the block gas limit
                // YP: \f$T_{g} \leq H_{l}\f$ ... different because it takes in account
                // previous transactions
                bn_t block_gas_limit;
                block_info.get_gas_limit(arith, block_gas_limit);
                if (cgbn_compare(arith.env, gas_limit, block_gas_limit) > 0) {
                    return ERROR_TRANSACTION_BLOCK_GAS_LIMIT;
                }

                // if transaction is valid update the touch state
                // \f$\simga(T_{s})_{b} = \simga(T_{s})_{b} - (p \dot T_{g})\f$
                cgbn_sub(arith.env, sender_balance, sender_balance, gas_value);
                touch_state.set_balance(arith, sender_address, sender_balance);
                // \f$\simga(T_{s})_{n} = T_{n} + 1\f$
                cgbn_add_ui32(arith.env, sender_nonce, sender_nonce, 1);
                touch_state.set_nonce(arith, sender_address, sender_nonce);
                // set the gas used to the intrisinc gas
                cgbn_set(arith.env, gas_used, gas_intrinsic);
                // TODO: maybe sent the priority fee to the miner
                // or this is a final thing to verify within a transaction
                // by asking the balacne of the coinbase accountq

                // warm up the access list
                access_list_warm_up(arith, access_state);

                // warmup coinbase and precompile contracts
                #ifdef EIP_3651
                bn_t coin_base_address;
                block_info.get_coin_base(arith, coin_base_address);
                access_state.get_account(
                    arith,
                    coin_base_address,
                    sender_account,
                    ACCOUNT_BALANCE_FLAG);
                #endif
                bn_t precompile_contract_address;
                #pragma unroll
                for (uint32_t idx = 1; idx < CuEVM::no_precompile_contracts; idx++) {
                    cgbn_set_ui32(arith.env, precompile_contract_address, idx);
                    access_state.get_account(
                        arith,
                        precompile_contract_address,
                        sender_account,
                        ACCOUNT_BYTE_CODE_FLAG);
                }

                return ERROR_SUCCESS;
            }

            /**
             * get the message call from the transaction
             * @param[in] arith the arithmetic environment.
             * @param[in] access_state the access state.
             * @param[out] evm_message_call_ptr the message call.
             * @return 1 for success, 0 for failure.
             */
            __host__ __device__ int32_t evm_transaction_t::get_message_call(
                ArithEnv &arith,
                CuEVM::state::AccessState &access_state,
                CuEVM::evm_message_call_t* &evm_message_call_ptr) const {
                bn_t sender_address, to_address, value, gas_limit;
                get_sender(arith, sender_address);
                get_to(arith, to_address);
                get_value(arith, value);
                get_gas_limit(arith, gas_limit);
                uint32_t depth = 0;
                uint32_t call_type = OP_CALL;
                CuEVM::byte_array_t byte_code;
                // if is a contract creation
                if (is_contract_creation(arith)) {
                    call_type = OP_CREATE;
                    byte_code = data_init;
                    // TODO: code size does not execede the maximul allowed
                    bn_t sender_nonce;
                    CuEVM::account::account_t* sender_account = nullptr;
                    access_state.get_account(arith, sender_address, sender_account, ACCOUNT_NONCE_FLAG);
                    // nonce is -1 in YP but here is before validating the transaction
                    // and increasing the nonce
                    sender_account->get_nonce(arith, sender_nonce);
                    if(CuEVM::utils::get_contract_address_create(
                        arith,
                        to_address,
                        sender_address,
                        sender_nonce) != ERROR_SUCCESS) {
                        return ERROR_FAILED;
                    }
                } else {
                    CuEVM::account::account_t* to_account = nullptr;
                    access_state.get_account(arith, to_address, to_account, ACCOUNT_BYTE_CODE_FLAG);
                    byte_code = to_account->byte_code;
                }
                uint32_t static_env = 0;
                bn_t return_data_offset;
                cgbn_set_ui32(arith.env, return_data_offset, 0);
                bn_t return_data_size;
                cgbn_set_ui32(arith.env, return_data_size, 0);
                evm_message_call_ptr = new CuEVM::evm_message_call_t(
                    arith,
                    sender_address,
                    to_address,
                    to_address,
                    gas_limit,
                    value,
                    depth,
                    call_type,
                    to_address,
                    data_init,
                    byte_code,
                    return_data_offset,
                    return_data_size,
                    static_env);
                return ERROR_SUCCESS;
            }

            __host__ __device__ void evm_transaction_t::print() {
                printf("Transaction:\n");
                printf("Type: %d\n", type);
                printf("Nonce: ");
                nonce.print();
                printf("Gas Limit: ");
                gas_limit.print();
                printf("To: ");
                to.print();
                printf("Value: ");
                value.print();
                printf("Sender: ");
                sender.print();
                printf("Max Fee Per Gas: ");
                max_fee_per_gas.print();
                printf("Max Priority Fee Per Gas: ");
                max_priority_fee_per_gas.print();
                printf("Gas Price: ");
                gas_price.print();
                printf("Data: ");
                data_init.print();
                printf("Access List:\n");
                for (uint32_t i = 0; i < access_list.accounts_count; i++) {
                    printf("Account %d:\n", i);
                    printf("Address: ");
                    access_list.accounts[i].address.print();
                    printf("Storage Keys Count: %d\n", access_list.accounts[i].storage_keys_count);
                    for (uint32_t j = 0; j < access_list.accounts[i].storage_keys_count; j++) {
                        printf("Storage Key %d: ", j);
                        access_list.accounts[i].storage_keys[j].print();
                    }
                }
            }

            __host__ cJSON* evm_transaction_t::to_json() {
                cJSON* json = cJSON_CreateObject();
                char *hex_string_ptr = new char[CuEVM::word_size * 2 + 3];
                char *bytes_string = nullptr;
                cJSON_AddNumberToObject(json, "type", type);
                nonce.to_hex(hex_string_ptr);
                cJSON_AddStringToObject(json, "nonce", hex_string_ptr);
                gas_limit.to_hex(hex_string_ptr);
                cJSON_AddStringToObject(json, "gas_limit", hex_string_ptr);
                to.to_hex(hex_string_ptr, 0, 5);
                cJSON_AddStringToObject(json, "to", hex_string_ptr);
                value.to_hex(hex_string_ptr);
                cJSON_AddStringToObject(json, "value", hex_string_ptr);
                sender.to_hex(hex_string_ptr, 0, 5);
                cJSON_AddStringToObject(json, "sender", hex_string_ptr);
                cJSON_AddStringToObject(json, "origin", hex_string_ptr);
                max_fee_per_gas.to_hex(hex_string_ptr);
                cJSON_AddStringToObject(json, "max_fee_per_gas", hex_string_ptr);
                max_priority_fee_per_gas.to_hex(hex_string_ptr);
                cJSON_AddStringToObject(json, "max_priority_fee_per_gas", hex_string_ptr);
                gas_price.to_hex(hex_string_ptr);
                cJSON_AddStringToObject(json, "gas_price", hex_string_ptr);
                bytes_string = data_init.to_hex();
                cJSON_AddStringToObject(json, "data", bytes_string);
                delete[] bytes_string;
                cJSON* access_list_json = cJSON_CreateArray();
                cJSON_AddItemToObject(json, "access_list", access_list_json);
                for (uint32_t i = 0; i < access_list.accounts_count; i++) {
                    cJSON* account_json = cJSON_CreateObject();
                    cJSON_AddItemToArray(access_list_json, account_json);
                    access_list.accounts[i].address.to_hex(hex_string_ptr, 0, 5);
                    cJSON_AddStringToObject(account_json, "address", hex_string_ptr);
                    cJSON* storage_keys_json = cJSON_CreateArray();
                    cJSON_AddItemToObject(account_json, "storage_keys", storage_keys_json);
                    for (uint32_t j = 0; j < access_list.accounts[i].storage_keys_count; j++) {
                        access_list.accounts[i].storage_keys[j].to_hex(hex_string_ptr);
                        cJSON_AddItemToArray(storage_keys_json, cJSON_CreateString(hex_string_ptr));
                    }
                }
                delete[] hex_string_ptr;
                return json;
            }

        __host__ __device__ uint32_t no_transactions(
            const cJSON* json) {
            cJSON* transaction_json = cJSON_GetObjectItemCaseSensitive(json, "transaction");
            const cJSON *data_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "data");
            size_t data_counts = cJSON_GetArraySize(data_json);
            const cJSON *gas_limit_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "gasLimit");
            size_t gas_limit_counts = cJSON_GetArraySize(gas_limit_json);
            const cJSON *value_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "value");
            size_t value_counts = cJSON_GetArraySize(value_json);
            return data_counts * gas_limit_counts * value_counts;
        }

        __host__ int32_t get_transactios(
            ArithEnv &arith,
            evm_transaction_t* &transactions_ptr,
            const cJSON* json,
            uint32_t &transactions_count,
            int32_t managed,
            uint32_t start_index,
            uint32_t clones) {
            cJSON* transaction_json = cJSON_GetObjectItemCaseSensitive(json, "transaction");
            uint32_t available_transactions = no_transactions(json);
            if (start_index >= available_transactions) {
                transactions_count = 0;
                transactions_ptr = nullptr;
                return ERROR_FAILED;
            }
            uint32_t original_count;
            original_count = available_transactions - start_index;
            transactions_count = (original_count > clones) ? original_count : original_count*(clones/original_count);
            transactions_count = (transactions_count > CuEVM::max_transactions_count) ? CuEVM::max_transactions_count : transactions_count;

            if (managed) {
                CUDA_CHECK(cudaMallocManaged(
                    (void **)&(transactions_ptr),
                    transactions_count * sizeof(evm_transaction_t)));
            } else {
                transactions_ptr = new evm_transaction_t[transactions_count];
            }

            evm_transaction_t* template_transaction_ptr = new evm_transaction_t();
            uint32_t data_idnex, gas_limit_index, value_index, idx, jdx;

            uint32_t type = 0;

            const cJSON* nonce_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "nonce");
            template_transaction_ptr->nonce.from_hex(nonce_json->valuestring);

            const cJSON* gas_limit_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "gasLimit");
            uint32_t gas_limit_counts = cJSON_GetArraySize(gas_limit_json);

            const cJSON* to_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "to");
            // verify what is happening from strlen 0
            template_transaction_ptr->to.from_hex(to_json->valuestring);

            const cJSON* value_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "value");
            uint32_t value_counts = cJSON_GetArraySize(value_json);

            const cJSON* sender_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "sender");
            template_transaction_ptr->sender.from_hex(sender_json->valuestring);

            const cJSON* access_list_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "accessList");
            if (access_list_json != nullptr) {
                template_transaction_ptr->access_list.from_json(access_list_json, managed);
            }

            const cJSON* max_fee_per_gas_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "maxFeePerGas");

            const cJSON* max_priority_fee_per_gas_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "maxPriorityFeePerGas");

            const cJSON* gas_price_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "gasPrice");

            if (
                (max_fee_per_gas_json != nullptr) &&
                (max_priority_fee_per_gas_json != nullptr) &&
                (gas_price_json == nullptr))
            {
                type = 2;
                template_transaction_ptr->max_fee_per_gas.from_hex(max_fee_per_gas_json->valuestring);
                template_transaction_ptr->max_priority_fee_per_gas.from_hex(max_priority_fee_per_gas_json->valuestring);
                template_transaction_ptr->gas_price.from_uint32_t(0);
            } else if (
                (max_fee_per_gas_json == nullptr) &&
                (max_priority_fee_per_gas_json == nullptr) &&
                (gas_price_json != nullptr)) {
                if (access_list_json == nullptr) {
                    type = 0;
                } else {
                    type = 1;
                }
                template_transaction_ptr->max_fee_per_gas.from_uint32_t(0);
                template_transaction_ptr->max_priority_fee_per_gas.from_uint32_t(0);
                template_transaction_ptr->gas_price.from_hex(gas_price_json->valuestring);
            } else {
                return ERROR_TRANSACTION_TYPE;
            }

            const cJSON* data_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "data");
            uint32_t data_counts = cJSON_GetArraySize(data_json);

            template_transaction_ptr->type = type;

            uint32_t index;
            char* bytes_string = nullptr;
            for (idx = 0; idx < transactions_count; idx++) {
                index = (start_index + idx) % original_count;
                data_idnex = index % data_counts;
                gas_limit_index = (index / data_counts) % gas_limit_counts;
                value_index = (index / (data_counts * gas_limit_counts)) % value_counts;
                std::copy(template_transaction_ptr, template_transaction_ptr + 1, transactions_ptr + idx);
                transactions_ptr[idx].data_init.from_hex(cJSON_GetArrayItem(data_json, data_idnex)->valuestring, LITTLE_ENDIAN, CuEVM::PaddingDirection::NO_PADDING, managed);
                transactions_ptr[idx].gas_limit.from_hex(cJSON_GetArrayItem(gas_limit_json, gas_limit_index)->valuestring);
                transactions_ptr[idx].value.from_hex(cJSON_GetArrayItem(value_json, value_index)->valuestring);
            }

            delete template_transaction_ptr;
            return ERROR_SUCCESS;
        }

        __host__ int32_t free_instaces(
            evm_transaction_t* transactions_ptr,
            uint32_t transactions_count,
            int32_t managed) {
            if (transactions_ptr != nullptr) {
                transactions_ptr[0].access_list.free(managed);
                for (uint32_t i = 0; i < transactions_count; i++) {
                    // TODO: to see how to delete managed memory
                    transactions_ptr[i].data_init.~byte_array_t();
                }
                if (managed) {
                    CUDA_CHECK(cudaFree(transactions_ptr));
                } else {
                    delete[] transactions_ptr;
                }
            }
            return ERROR_SUCCESS;
        }
    }
}
