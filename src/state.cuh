// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#ifndef _STATE_T_H_
#define _STATE_T_H_

#include "include/utils.h"
#include "keccak.cuh"
#include <iostream>

#define READ_NONE 0
#define READ_BALANCE 1
#define READ_NONCE 2
#define READ_CODE 4
#define READ_STORAGE 8
#define WRITE_NONE 0
#define WRITE_BALANCE 1
#define WRITE_NONCE 2
#define WRITE_CODE 4
#define WRITE_STORAGE 8
#define WRITE_DELETE 16

#define STORAGE_CHUNK 32 // allocate storage in chunks of 32

/**
 * Kernel to copy the accounts details and read operations
 * between two instances of the accessed state data.
 * @param[out] dst_instances The destination instances
 * @param[in] src_instances The source instances
 * @param[in] count The number of instances
*/
template <typename T>
__global__ void kernel_accessed_state_S1(
    T *dst_instances,
    T *src_instances,
    uint32_t count);


/**
 * Kernel to copy the bytecode and storage
 * between two instances of the accessed state data.
 * @param[out] dst_instances The destination instances
 * @param[in] src_instances The source instances
 * @param[in] count The number of instances
*/
template <typename T>
__global__ void kernel_accessed_state_S2(
    T *dst_instances,
    T *src_instances,
    uint32_t count);

/**
 * Kernel to copy the accounts details and write operations
 * between two instances of the touch state data.
 * @param[out] dst_instances The destination instances
 * @param[in] src_instances The source instances
 * @param[in] count The number of instances
*/
template <typename T>
__global__ void kernel_touch_state_S1(
    T *dst_instances,
    T *src_instances,
    uint32_t count);

/**
 * Kernel to copy the bytecode and storage
 * between two instances of the touch state data.
 * @param[out] dst_instances The destination instances
 * @param[in] src_instances The source instances
 * @param[in] count The number of instances
*/
template <typename T>
__global__ void kernel_touch_state_S2(
    T *dst_instances,
    T *src_instances,
    uint32_t count);

/**
 * The world state (YP: \f$\sigma\f$) class. It cotains all the active accounts and
 * their storage.
*/

class world_state_t
{
public:

    /**
     * The storage entry type.
    */
    typedef struct
    {
        evm_word_t key; /**< The key of the storage */
        evm_word_t value; /**< The value of the storage for the given key */
    } contract_storage_t;

    /**
     * The account type.
    */
    typedef struct alignas(32)
    {
        evm_word_t address; /**< The address of the account (YP: \f$a\f$) */
        evm_word_t balance; /**< The balance of the account (YP: \f$\sigma[a]_{b}\f$) */
        evm_word_t nonce; /**< The nonce of the account (YP: \f$\sigma[a]_{n}\f$) */
        size_t code_size; /**< The size of the bytecode (YP: \f$|b|\f$) */
        size_t storage_size; /**< The number of storage entries (YP: \f$|\sigma[a]_{s}|\f$) */
        uint8_t *bytecode; /**< The bytecode of the account (YP: \f$b\f$) */
        contract_storage_t *storage; /**< The storage of the account (YP: \f$\sigma[a]_{s}\f$) */
    } account_t;

    /**
     * The state data type.
    */
    typedef struct
    {
        account_t *accounts; /**< The accounts in the state (YP: \f$\sigma\f$)*/
        size_t no_accounts; /**< The number of accounts in the state (YP: \f$|\sigma|\f$)*/
    } state_data_t;

    state_data_t *_content; /**< The content of the state */
    arith_t _arith; /**< The arithmetical environment */

    /**
     * The constructor of the state given the content.
     * @param arith The arithmetical environment
     * @param content The content of the state
    */
    __host__ __device__ __forceinline__ world_state_t(
        arith_t arith,
        state_data_t *content
    ) : _arith(arith), _content(content)
    {
    }

    /**
     * The destructor of the state.
    */
    __host__ __device__ __forceinline__ ~world_state_t()
    {
        _content = NULL;
    }
    /**
     * The destructor of the state on the host.
     * It use cudaFree to free the memory, because the
     * state is allocated with cudaMallocManaged.
    */
    __host__ void free_content()
    {
        #ifndef ONLY_CPU
        if (_content != NULL)
        {
            if (_content->accounts != NULL)
            {
                for (size_t idx = 0; idx < _content->no_accounts; idx++)
                {
                    if (_content->accounts[idx].bytecode != NULL)
                    {
                        CUDA_CHECK(cudaFree(_content->accounts[idx].bytecode));
                        _content->accounts[idx].bytecode = NULL;
                    }
                    if (_content->accounts[idx].storage != NULL)
                    {
                        CUDA_CHECK(cudaFree(_content->accounts[idx].storage));
                        _content->accounts[idx].storage = NULL;
                    }
                }
                CUDA_CHECK(cudaFree(_content->accounts));
                _content->accounts = NULL;
            }
            CUDA_CHECK(cudaFree(_content));
            _content = NULL;
        }
        #else
        if (_content != NULL)
        {
            if (_content->accounts != NULL)
            {
                for (size_t idx = 0; idx < _content->no_accounts; idx++)
                {
                    if (_content->accounts[idx].bytecode != NULL)
                    {
                        delete[] _content->accounts[idx].bytecode;
                        _content->accounts[idx].bytecode = NULL;
                    }
                    if (_content->accounts[idx].storage != NULL)
                    {
                        delete[] _content->accounts[idx].storage;
                        _content->accounts[idx].storage = NULL;
                    }
                }

                delete[] _content->accounts;
                _content->accounts = NULL;
            }
            delete _content;
            _content = NULL;
        }
        #endif
    }


    /**
     * The constructor of the state on the host.
     *
     * It reads the json file and creates the state.
     * It pass through all the account and their
     * storage and creates the corresponding data.
     * It used unnified memory so the state can be
     * easily used on the device.
     *
     * @param arith The arithmetical environment
     * @param test the json for the current test
    */
    __host__ world_state_t(
        arith_t arith,
        const cJSON *test
    ) : _arith(arith)
    {
        const cJSON *world_state_json = NULL; // the json for the world state
        const cJSON *account_json = NULL;    // the json for the current account
        const cJSON *balance_json = NULL;   // the json for the balance
        const cJSON *code_json = NULL;    // the json for the code
        const cJSON *nonce_json = NULL;  // the json for the nonce
        const cJSON *storage_json = NULL; // the json for the storage
        const cJSON *key_value_json = NULL; // the json for the key-value pair
        char *hex_string = NULL; // the hex string for the code
        size_t idx, jdx; // the index for the accounts and storage

        _content = NULL; // initialize the content
        // allocate the content
        #ifndef ONLY_CPU
        CUDA_CHECK(cudaMallocManaged(
            (void **)&(_content),
            sizeof(state_data_t)
        ));
        #else
        _content = new state_data_t;
        #endif

        // get the world state json
        if (cJSON_IsObject(test))
            world_state_json = cJSON_GetObjectItemCaseSensitive(test, "pre");
        else if (cJSON_IsArray(test))
            world_state_json = test;
        else
            printf("[ERROR] world_state_t: invalid test json\n");

        // get the number of accounts
        _content->no_accounts = cJSON_GetArraySize(world_state_json);
        if (_content->no_accounts == 0)
        {
            _content->accounts = NULL;
            return;
        }
        // allocate the accounts
        #ifndef ONLY_CPU
        CUDA_CHECK(cudaMallocManaged(
            (void **)&(_content->accounts),
            _content->no_accounts * sizeof(account_t)
        ));
        #else
        _content->accounts = new account_t[_content->no_accounts];
        #endif

        // iterate through the accounts
        idx = 0;
        cJSON_ArrayForEach(account_json, world_state_json)
        {
            // set the address
            _arith.cgbn_memory_from_hex_string(_content->accounts[idx].address, account_json->string);

            // set the balance
            balance_json = cJSON_GetObjectItemCaseSensitive(account_json, "balance");
            _arith.cgbn_memory_from_hex_string(_content->accounts[idx].balance, balance_json->valuestring);

            // set the nonce
            nonce_json = cJSON_GetObjectItemCaseSensitive(account_json, "nonce");
            _arith.cgbn_memory_from_hex_string(_content->accounts[idx].nonce, nonce_json->valuestring);

            // set the code
            code_json = cJSON_GetObjectItemCaseSensitive(account_json, "code");
            hex_string = code_json->valuestring;
            _content->accounts[idx].code_size = adjusted_length(&hex_string);
            if (_content->accounts[idx].code_size > 0)
            {
                #ifndef ONLY_CPU
                CUDA_CHECK(cudaMallocManaged(
                    (void **)&(_content->accounts[idx].bytecode),
                    _content->accounts[idx].code_size * sizeof(uint8_t)
                ));
                #else
                _content->accounts[idx].bytecode = new uint8_t[_content->accounts[idx].code_size];
                #endif
                hex_to_bytes(
                    hex_string,
                    _content->accounts[idx].bytecode,
                    2 * _content->accounts[idx].code_size
                );
            }
            else
            {
                _content->accounts[idx].bytecode = NULL;
            }

            // set the storage
            storage_json = cJSON_GetObjectItemCaseSensitive(account_json, "storage");
            _content->accounts[idx].storage_size = cJSON_GetArraySize(storage_json);
            // round to the next multiple of STORAGE_CHUNK
            // _content->accounts[idx].storage_size = ((_content->accounts[idx].storage_size + STORAGE_CHUNK - 1) / STORAGE_CHUNK) * STORAGE_CHUNK;
            if (_content->accounts[idx].storage_size > 0)
            {
                // allocate the storage
                #ifndef ONLY_CPU
                CUDA_CHECK(cudaMallocManaged(
                    (void **)&(_content->accounts[idx].storage),
                    _content->accounts[idx].storage_size * sizeof(contract_storage_t)
                ));
                #else
                _content->accounts[idx].storage = new contract_storage_t[_content->accounts[idx].storage_size];
                #endif
                // iterate through the storage
                jdx = 0;
                cJSON_ArrayForEach(key_value_json, storage_json)
                {
                    // set the key
                    _arith.cgbn_memory_from_hex_string(_content->accounts[idx].storage[jdx].key, key_value_json->string);

                    // set the value
                    _arith.cgbn_memory_from_hex_string(_content->accounts[idx].storage[jdx].value, key_value_json->valuestring);

                    jdx++;
                }
            }
            else
            {
                _content->accounts[idx].storage = NULL;
            }
            idx++;
        }
    }

    /**
     * Print the account.
     * @param[in] arith The arithmetical environment
     * @param[in] account The account
    */
    __host__ __device__ __forceinline__ static void print_account_t(
        arith_t &arith,
        account_t &account
    )
    {
        printf("address: ");
        arith.print_cgbn_memory(account.address);
        printf("balance: ");
        arith.print_cgbn_memory(account.balance);
        printf("nonce: ");
        arith.print_cgbn_memory(account.nonce);
        printf("code_size: %lu\n", account.code_size);
        printf("code: ");
        print_bytes(account.bytecode, account.code_size);
        printf("\n");
        printf("storage_size: %lu\n", account.storage_size);
        for (size_t idx = 0; idx < account.storage_size; idx++)
        {
            printf("storage[%lu].key: ", idx);
            arith.print_cgbn_memory(account.storage[idx].key);
            printf("storage[%lu].value: ", idx);
            arith.print_cgbn_memory(account.storage[idx].value);
        }
    }

    /**
     * Get json of the account
     * @param[in] arith The arithmetical environment
     * @param[in] account The account
     * @return The json of the account
    */
    __host__ static cJSON *json_from_account_t(
        arith_t &arith,
        account_t &account
    )
    {
        cJSON *account_json = NULL;
        cJSON *storage_json = NULL;
        char *bytes_string = NULL;
        char *hex_string_ptr = new char[arith_t::BYTES * 2 + 3];
        char *value_hex_string_ptr = new char[arith_t::BYTES * 2 + 3];
        size_t jdx = 0;
        account_json = cJSON_CreateObject();
        // set the address
        arith.hex_string_from_cgbn_memory(hex_string_ptr, account.address, 5);
        cJSON_SetValuestring(account_json, hex_string_ptr);
        // set the balance
        arith.hex_string_from_cgbn_memory(hex_string_ptr, account.balance);
        cJSON_AddStringToObject(account_json, "balance", hex_string_ptr);
        // set the nonce
        arith.hex_string_from_cgbn_memory(hex_string_ptr, account.nonce);
        cJSON_AddStringToObject(account_json, "nonce", hex_string_ptr);
        // set the code
        if (account.code_size > 0)
        {
            bytes_string = hex_from_bytes(account.bytecode, account.code_size);
            cJSON_AddStringToObject(account_json, "code", bytes_string);
            delete[] bytes_string;
        }
        else
        {
            cJSON_AddStringToObject(account_json, "code", "0x");
        }
        // set the storage
        storage_json = cJSON_CreateObject();
        cJSON_AddItemToObject(account_json, "storage", storage_json);
        if (account.storage_size > 0)
        {
            for (jdx = 0; jdx < account.storage_size; jdx++)
            {
                arith.hex_string_from_cgbn_memory(hex_string_ptr, account.storage[jdx].key);
                arith.hex_string_from_cgbn_memory(value_hex_string_ptr, account.storage[jdx].value);
                cJSON_AddStringToObject(storage_json, hex_string_ptr, value_hex_string_ptr);
            }
        }
        delete[] hex_string_ptr;
        hex_string_ptr = NULL;
        delete[] value_hex_string_ptr;
        value_hex_string_ptr = NULL;
        return account_json;
    }

    /**
     * Get the index of the account with the given address.
     * @param[in] address The address of the account
     * @param[out] error_code The error code if the account does not exist
     * @return The index of the account
    */
    __host__ __device__ __forceinline__ size_t get_account_index(
        bn_t &address,
        uint32_t &error_code
    )
    {
        bn_t local_address;
        for (size_t idx = 0; idx < _content->no_accounts; idx++)
        {
            cgbn_load(_arith._env, local_address, &(_content->accounts[idx].address));
            if (cgbn_compare(_arith._env, local_address, address) == 0)
            {
                return idx;
            }
        }
        error_code = ERR_STATE_INVALID_ADDRESS;
        return 0;
    }

    /**
     * Get the account with the given address.
     * @param[in] address The address of the account
     * @param[out] error_code The error code if the account does not exist
     * @return The account
    */
    __host__ __device__ __forceinline__ account_t *get_account(
        bn_t &address,
        uint32_t &error_code
    )
    {
        size_t account_idx = get_account_index(address, error_code);
        return &(_content->accounts[account_idx]);
    }

    /**
     * Get the index of the storage inside the given account
     * with the given key.
     * @param[in] account The account
     * @param[in] key The key of the storage
     * @param[out] error_code The error code if the key is not found
     * @return The index of the storage in the account
    */
    __host__ __device__ __forceinline__ size_t get_storage_index(
        account_t *account,
        bn_t &key,
        uint32_t &error_code
    )
    {
        bn_t local_key;
        for (size_t idx = 0; idx < account->storage_size; idx++)
        {
            cgbn_load(_arith._env, local_key, &(account->storage[idx].key));
            if (cgbn_compare(_arith._env, local_key, key) == 0)
            {
                return idx;
            }
        }
        error_code = ERR_STATE_INVALID_KEY;
        return 0;
    }

    /**
     * Get the value of the storage with the given key
     * from the account with the given address.
     * @param[in] address The address of the account
     * @param[in] key The key of the storage
     * @param[out] value The value of the storage
    */
    __host__ __device__ __forceinline__ void get_value(
        bn_t &address,
        bn_t &key,
        bn_t &value
    )
    {
        uint32_t tmp_error_code = ERR_SUCCESS;
        // get the account
        account_t *account = get_account(address, tmp_error_code);
        // if no account, return 0
        if (tmp_error_code != ERR_SUCCESS)
        {
            cgbn_set_ui32(_arith._env, value, 0);
        }
        else
        {
            // get the storage index
            size_t storage_idx = get_storage_index(account, key, tmp_error_code);
            // if no storage, return 0
            if (tmp_error_code != ERR_SUCCESS)
            {
                cgbn_set_ui32(_arith._env, value, 0);
            }
            else
            {
                // get the value
                cgbn_load(_arith._env, value, &(account->storage[storage_idx].value));
            }
        }
    }

    /**
     * Print the state data structure.
     * @param[in] arith The arithmetical environment
     * @param[in] state_data The state data
    */
    __host__ __device__ __forceinline__ static void print_state_data_t(
        arith_t &arith,
        state_data_t &state_data
    )
    {
        printf("no_accounts: %lu\n", state_data.no_accounts);
        for (size_t idx = 0; idx < state_data.no_accounts; idx++)
        {
            printf("accounts[%lu]:\n", idx);
            print_account_t(arith, state_data.accounts[idx]);
        }
    }

    /**
     * Print the state.
    */
    __host__ __device__ __forceinline__ void print()
    {
        print_state_data_t(_arith, *_content);
    }

    /**
     * Get json from the state data structure.
    */
    __host__ __forceinline__ static cJSON *json_from_state_data_t(
        arith_t &arith,
        state_data_t &state_data
    )
    {
        cJSON *state_json = NULL;
        cJSON *account_json = NULL;
        char *hex_string_ptr = new char[arith_t::BYTES * 2 + 3];
        size_t idx = 0;
        state_json = cJSON_CreateObject();
        for (idx = 0; idx < state_data.no_accounts; idx++)
        {
            account_json = json_from_account_t(arith, state_data.accounts[idx]);
            arith.hex_string_from_cgbn_memory(hex_string_ptr, state_data.accounts[idx].address, 5);
            cJSON_AddItemToObject(state_json, hex_string_ptr, account_json);
        }
        delete[] hex_string_ptr;
        hex_string_ptr = NULL;
        return state_json;
    }

    /**
     * Get json of the state
     * @return The json of the state
    */
    __host__ __forceinline__ cJSON *json()
    {
        return json_from_state_data_t(_arith, *_content);
    }
};

/**
 * The accessed state data class type.
 * YP: accrued transaction substate
 *  \f$A_{a}\f$ for accessed accounts
 *  \f$A_{k}\f$ for accessed storage keys
*/
class accessed_state_t
{
public:
    /**
     * The storage entry type.
    */
    typedef world_state_t::contract_storage_t contract_storage_t;
    /**
     * The account type.
    */
    typedef world_state_t::account_t account_t;
    /**
     * The state data type. Contains the accounts with their storage.
    */
    typedef world_state_t::state_data_t state_data_t;

    /**
     * The accessed state data type.
     * Contains the accounts with their storage similar to world state.
     * It also contains the information about the read operations
     * performed on the accounts.
     * It can be 0 - no, 1 - bytecode, 2 - balance, 4 - nonce, 8 - storage
    */
    typedef struct
    {
        state_data_t accessed_accounts; /**< The accessed accounts */
        uint8_t *reads; /**< The read operations performed on the accounts */
    } accessed_state_data_t;

    accessed_state_data_t *_content; /**< The content of the accessed state */
    arith_t _arith; /**< The arithmetical environment */
    world_state_t *_world_state; /**< The world state */

    /**
     * The constructor of the accessed state given the content.
     * @param content The content of the accessed state
     * @param world_state The world state
    */
    __host__ __device__ __forceinline__ accessed_state_t(
        accessed_state_data_t *content,
        world_state_t *world_state
    ) : _arith(world_state->_arith), _content(content), _world_state(world_state)
    {
    }

    /**
     * The constructor of the accessed state given the world state.
     * @param world_state The world state
    */
    __host__ __device__ __forceinline__ accessed_state_t(
        world_state_t *world_state
    ) : _arith(world_state->_arith), _world_state(world_state)
    {
        // allocate the content and initial setup with no accounts
        SHARED_MEMORY accessed_state_data_t *tmp_content;
        ONE_THREAD_PER_INSTANCE(
            tmp_content = new accessed_state_data_t;
            tmp_content->accessed_accounts.no_accounts = 0;
            tmp_content->accessed_accounts.accounts = NULL;
            tmp_content->reads = NULL;
        )
        _content = tmp_content;
    }

    /**
     * The destructor of the accessed state.
     * It frees the memory allocated for the content.
    */
    __host__ __device__ __forceinline__ ~accessed_state_t()
    {
        ONE_THREAD_PER_INSTANCE(
            if (_content != NULL)
            {
                if (_content->accessed_accounts.accounts != NULL)
                {
                    for (size_t idx = 0; idx < _content->accessed_accounts.no_accounts; idx++)
                    {
                        if (_content->accessed_accounts.accounts[idx].bytecode != NULL)
                        {
                            delete[] _content->accessed_accounts.accounts[idx].bytecode;
                            _content->accessed_accounts.accounts[idx].bytecode = NULL;
                        }
                        if (_content->accessed_accounts.accounts[idx].storage != NULL)
                        {
                            delete[] _content->accessed_accounts.accounts[idx].storage;
                            _content->accessed_accounts.accounts[idx].storage = NULL;
                        }
                    }
                    delete[] _content->accessed_accounts.accounts;
                    _content->accessed_accounts.accounts = NULL;
                    _content->accessed_accounts.no_accounts = 0;
                }
                if (_content->reads != NULL)
                {
                    delete[] _content->reads;
                    _content->reads = NULL;
                }
                delete _content;
            }
        )
        _content = NULL;
    }

    /**
     * Get the index of the account with the given address.
     * @param[in] address The address of the account
     * @param[out] error_code The error code if the account does not exist
     * @return The index of the account
    */
    __host__ __device__ __forceinline__ size_t get_account_index(
        bn_t &address,
        uint32_t &error_code
    )
    {
        bn_t local_address;
        for (size_t idx = 0; idx < _content->accessed_accounts.no_accounts; idx++)
        {
            cgbn_load(_arith._env, local_address, &(_content->accessed_accounts.accounts[idx].address));
            if (cgbn_compare(_arith._env, local_address, address) == 0)
            {
                return idx;
            }
        }
        error_code = ERR_STATE_INVALID_ADDRESS;
        return 0;
    }

    /**
     * Get the gas cost for accessing the account with the given address.
     * @param[in] address The address of the account
     * @param[out] gas_cost The gas cost for accessing the account
    */
    __host__ __device__ __forceinline__ void get_access_account_gas_cost(
        bn_t &address,
        bn_t &gas_cost
    )
    {
        uint32_t tmp_error_code = ERR_SUCCESS;
        // get the account index
        size_t account_idx = get_account_index(address, tmp_error_code);
        // if account was accessed before, it is warm
        // otherwise it is cold
        if (tmp_error_code == ERR_SUCCESS)
        {
            cgbn_set_ui32(_arith._env, gas_cost, GAS_WARM_ACCESS);
        }
        else
        {
            cgbn_set_ui32(_arith._env, gas_cost, GAS_COLD_ACCOUNT_ACCESS);
        }
    }

    /**
     * Get the account with the given address.
     * If the account does not exist, it duplicates the
     * one from the world state or if the world state does
     * not contain the account, it creates an empty one.
     * @param[in] address The address of the account
     * @param[in] read_type The type of the read operation
     * @return The account
    */
    __host__ __device__ __forceinline__ account_t *get_account(
        bn_t &address,
        uint32_t read_type
    )
    {
        uint32_t tmp_error_code = ERR_SUCCESS;
        // get the account index
        size_t account_idx = get_account_index(address, tmp_error_code);
        // if account does not exist, duplicate it from the world state
        // or create an empty one
        if (tmp_error_code != ERR_SUCCESS)
        {
            // get the account from the world state
            tmp_error_code = ERR_SUCCESS;
            account_t *account = _world_state->get_account(address, tmp_error_code);
            // the duplicate account
            SHARED_MEMORY account_t *dup_account;
            ONE_THREAD_PER_INSTANCE(
                dup_account = new account_t;
            )
            // if the account does not exist in the world state
            // create an empty one
            if (tmp_error_code != ERR_SUCCESS)
            {
                // empty account
                bn_t zero;
                cgbn_set_ui32(_arith._env, zero, 0);
                cgbn_store(_arith._env, &(dup_account->address), address);
                cgbn_store(_arith._env, &(dup_account->balance), zero);
                cgbn_store(_arith._env, &(dup_account->nonce), zero);
                dup_account->code_size = 0;
                dup_account->bytecode = NULL;
                dup_account->storage_size = 0;
                dup_account->storage = NULL;
            }
            else
            {
                // duplicate account
                ONE_THREAD_PER_INSTANCE(
                    memcpy(
                        dup_account,
                        account,
                        sizeof(account_t)
                    );
                    // copy the bytecode if it exists
                    if ((account->code_size > 0) && (account->bytecode != NULL)) {
                        dup_account->bytecode = new uint8_t[account->code_size * sizeof(uint8_t)];
                        memcpy(
                            dup_account->bytecode,
                            account->bytecode,
                            account->code_size * sizeof(uint8_t)
                        );
                        dup_account->code_size = account->code_size;
                    } else {
                        dup_account->bytecode = NULL;
                        dup_account->code_size = 0;
                    }
                    // no storage copy
                    dup_account->storage_size = 0;
                    dup_account->storage = NULL;
                )
            }
            // add the new account to the accessed accounts
            account_idx = _content->accessed_accounts.no_accounts;
            ONE_THREAD_PER_INSTANCE(
                account_t *tmp_accounts = new account_t[_content->accessed_accounts.no_accounts + 1];
                uint8_t *tmp_reads = new uint8_t[_content->accessed_accounts.no_accounts + 1];
                if (_content->accessed_accounts.no_accounts > 0) {
                    memcpy(
                        tmp_accounts,
                        _content->accessed_accounts.accounts,
                        _content->accessed_accounts.no_accounts * sizeof(account_t)
                    );
                    memcpy(
                        tmp_reads,
                        _content->reads,
                        _content->accessed_accounts.no_accounts * sizeof(uint8_t)
                    );
                    delete[] _content->accessed_accounts.accounts;
                    delete[] _content->reads;
                }
                _content->accessed_accounts.accounts = tmp_accounts;
                _content->accessed_accounts.no_accounts++;
                memcpy(
                    &(_content->accessed_accounts.accounts[account_idx]),
                    dup_account,
                    sizeof(account_t)
                );
                _content->reads = tmp_reads;
                _content->reads[account_idx] = 0;
                delete dup_account;
            )
        }
        _content->reads[account_idx] |= read_type;
        return &(_content->accessed_accounts.accounts[account_idx]);
    }

    /**
     * Get the index of the storage inside the given account
     * with the given key.
     * @param[in] account The account
     * @param[in] key The key of the storage
     * @param[out] error_code The error code if the key is not found
     * @return The index of the storage in the account
    */
    __host__ __device__ __forceinline__ size_t get_storage_index(
        account_t *account,
        bn_t &key,
        uint32_t &error_code
    )
    {
        bn_t local_key;
        for (size_t idx = 0; idx < account->storage_size; idx++)
        {
            cgbn_load(_arith._env, local_key, &(account->storage[idx].key));
            if (cgbn_compare(_arith._env, local_key, key) == 0)
            {
                return idx;
            }
        }
        error_code = ERR_STATE_INVALID_KEY;
        return 0;
    }

    /**
     * Get the gas cost for accessing the storage with the given key
     * from the account with the given address.
     * @param[in] address The address of the account
     * @param[in] key The key of the storage
     * @param[out] gas_cost The gas cost for accessing the storage
    */
    __host__ __device__ __forceinline__ void get_access_storage_gas_cost(
        bn_t &address,
        bn_t &key,
        bn_t &gas_cost
    )
    {
        uint32_t tmp_error_code = ERR_SUCCESS;
        // get the account index
        size_t account_idx = get_account_index(address, tmp_error_code);
        if (tmp_error_code == ERR_SUCCESS)
        {
            // get the storage index
            account_t *account = &(_content->accessed_accounts.accounts[account_idx]);
            size_t storage_idx = get_storage_index(account, key, tmp_error_code);
            // if storage was accessed before, it is warm
            // otherwise it is cold
            if (tmp_error_code == ERR_SUCCESS)
            {
                cgbn_set_ui32(_arith._env, gas_cost, GAS_WARM_ACCESS);
            }
            else
            {
                cgbn_set_ui32(_arith._env, gas_cost, GAS_COLD_SLOAD);
            }
        }
        else
        {
            printf("[ERROR] get_access_storage_gas_cost: ERR_STATE_INVALID_ADDRESS NOT SUPPOSED TO HAPPEN\n");
        }
    }

    /**
     * Get the value of the storage with the given key
     * from the account with the given address.
     * If the storage does not exist, it duplicates the
     * one from the world state or if the world state does
     * not contain the storage, it creates an empty one.
     * @param[in] address The address of the account
     * @param[in] key The key of the storage
     * @param[out] value The value of the storage
    */
    __host__ __device__ __forceinline__ void get_value(
        bn_t &address,
        bn_t &key,
        bn_t &value
    )
    {
        uint32_t tmp_error_code = ERR_SUCCESS;
        // get the account (and duplicate it if needed)
        account_t *account = get_account(address, READ_STORAGE);
        // get the storage index
        size_t storage_idx = get_storage_index(account, key, tmp_error_code);
        // if storage does not exist, duplicate it from the world state
        if (tmp_error_code != ERR_SUCCESS)
        {
            // get the storage from the world state
            _world_state->get_value(address, key, value);
            // add the new pair key-value to storage
            storage_idx = account->storage_size;
            ONE_THREAD_PER_INSTANCE(
                contract_storage_t *tmp_storage = new contract_storage_t[account->storage_size + 1];
                if (account->storage_size > 0) {
                    memcpy(
                        tmp_storage,
                        account->storage,
                        account->storage_size * sizeof(contract_storage_t)
                    );
                    delete[] account->storage;
                } account->storage = tmp_storage;
                account->storage_size++;)
            // set the key and value
            cgbn_store(_arith._env, &(account->storage[storage_idx].key), key);
            cgbn_store(_arith._env, &(account->storage[storage_idx].value), value);
        }
        // get the value
        cgbn_load(_arith._env, value, &(account->storage[storage_idx].value));
    }

    __host__ __device__ __forceinline__ bool is_warm(bn_t &address, bn_t &key){
    uint32_t tmp_error_code = ERR_SUCCESS;
    size_t account_idx = get_account_index(address, tmp_error_code);
    if (tmp_error_code == ERR_SUCCESS) {
          account_t *account = &(_content->accessed_accounts.accounts[account_idx]);
          get_storage_index(account, key, tmp_error_code);
          return tmp_error_code == ERR_SUCCESS;
    }
    return false;
  }

    /**
     * Copy the content of the accessed state to the given
     * accessed state data.
     * @param[out] accessed_state_data The accessed state data
    */
    __host__ __device__ __forceinline__ void to_accessed_state_data_t(
        accessed_state_data_t &accessed_state_data
    )
    {
        ONE_THREAD_PER_INSTANCE(
        // free the memory of the destination if needed
        if (accessed_state_data.accessed_accounts.no_accounts > 0)
        {
            for (size_t idx = 0; idx < accessed_state_data.accessed_accounts.no_accounts; idx++)
            {
                if (accessed_state_data.accessed_accounts.accounts[idx].bytecode != NULL)
                {
                    delete[] accessed_state_data.accessed_accounts.accounts[idx].bytecode;
                    accessed_state_data.accessed_accounts.accounts[idx].bytecode = NULL;
                }
                if (accessed_state_data.accessed_accounts.accounts[idx].storage != NULL)
                {
                    delete[] accessed_state_data.accessed_accounts.accounts[idx].storage;
                    accessed_state_data.accessed_accounts.accounts[idx].storage = NULL;
                }
            }
            delete[] accessed_state_data.accessed_accounts.accounts;
            accessed_state_data.accessed_accounts.no_accounts = 0;
            accessed_state_data.accessed_accounts.accounts = NULL;
            delete[] accessed_state_data.reads;
            accessed_state_data.reads = NULL;
        }

        // copy the content and alocate the necessary memory
        accessed_state_data.accessed_accounts.no_accounts = _content->accessed_accounts.no_accounts;
        if (accessed_state_data.accessed_accounts.no_accounts > 0)
        {
            accessed_state_data.accessed_accounts.accounts = new account_t[accessed_state_data.accessed_accounts.no_accounts];
            accessed_state_data.reads = new uint8_t[accessed_state_data.accessed_accounts.no_accounts];
            memcpy(
                accessed_state_data.accessed_accounts.accounts,
                _content->accessed_accounts.accounts,
                accessed_state_data.accessed_accounts.no_accounts * sizeof(account_t)
            );
            memcpy(
                accessed_state_data.reads,
                _content->reads,
                accessed_state_data.accessed_accounts.no_accounts * sizeof(uint8_t)
            );
            for (size_t idx = 0; idx < accessed_state_data.accessed_accounts.no_accounts; idx++)
            {
                if (accessed_state_data.accessed_accounts.accounts[idx].code_size > 0)
                {
                    accessed_state_data.accessed_accounts.accounts[idx].bytecode = new uint8_t[accessed_state_data.accessed_accounts.accounts[idx].code_size * sizeof(uint8_t)];
                    memcpy(
                        accessed_state_data.accessed_accounts.accounts[idx].bytecode,
                        _content->accessed_accounts.accounts[idx].bytecode,
                        accessed_state_data.accessed_accounts.accounts[idx].code_size * sizeof(uint8_t)
                    );
                }
                else
                {
                    accessed_state_data.accessed_accounts.accounts[idx].bytecode = NULL;
                }

                if (accessed_state_data.accessed_accounts.accounts[idx].storage_size > 0)
                {
                    accessed_state_data.accessed_accounts.accounts[idx].storage = new contract_storage_t[accessed_state_data.accessed_accounts.accounts[idx].storage_size];
                    memcpy(
                        accessed_state_data.accessed_accounts.accounts[idx].storage,
                        _content->accessed_accounts.accounts[idx].storage,
                        accessed_state_data.accessed_accounts.accounts[idx].storage_size * sizeof(contract_storage_t)
                    );
                }
                else
                {
                    accessed_state_data.accessed_accounts.accounts[idx].storage = NULL;
                }
            }

        }
        )
    }

    /**
     * Generate the CPU instances of the accessed state data.
     * @param[in] count The number of instances
    */
    __host__ static accessed_state_data_t *get_cpu_instances(
        uint32_t count
    )
    {
        // allocate the instances and initialize them
        accessed_state_data_t *cpu_instances = new accessed_state_data_t[count];
        for (size_t idx = 0; idx < count; idx++)
        {
            cpu_instances[idx].accessed_accounts.no_accounts = 0;
            cpu_instances[idx].accessed_accounts.accounts = NULL;
            cpu_instances[idx].reads = NULL;
        }
        return cpu_instances;
    }

    /**
     * Free the CPU instances of the accessed state data.
     * @param[in] cpu_instances The CPU instances
     * @param[in] count The number of instances
    */
    __host__ static void free_cpu_instances(
        accessed_state_data_t *cpu_instances,
        uint32_t count
    )
    {
        for (size_t idx = 0; idx < count; idx++)
        {
            if (cpu_instances[idx].accessed_accounts.accounts != NULL)
            {
                for (size_t jdx = 0; jdx < cpu_instances[idx].accessed_accounts.no_accounts; jdx++)
                {
                    if (cpu_instances[idx].accessed_accounts.accounts[jdx].bytecode != NULL)
                    {
                        delete[] cpu_instances[idx].accessed_accounts.accounts[jdx].bytecode;
                        cpu_instances[idx].accessed_accounts.accounts[jdx].bytecode = NULL;
                    }
                    if (cpu_instances[idx].accessed_accounts.accounts[jdx].storage != NULL)
                    {
                        delete[] cpu_instances[idx].accessed_accounts.accounts[jdx].storage;
                        cpu_instances[idx].accessed_accounts.accounts[jdx].storage = NULL;
                    }
                }
                delete[] cpu_instances[idx].accessed_accounts.accounts;
                cpu_instances[idx].accessed_accounts.accounts = NULL;
            }
            if (cpu_instances[idx].reads != NULL)
            {
                delete[] cpu_instances[idx].reads;
                cpu_instances[idx].reads = NULL;
            }
        }
        delete[] cpu_instances;
    }

    /**
     * Generate the GPU instances of the accessed state data from
     * the CPU counterparts.
     * @param[in] cpu_instances The CPU instances
     * @param[in] count The number of instances
    */
    __host__ static accessed_state_data_t *get_gpu_instances_from_cpu_instances(
        accessed_state_data_t *cpu_instances,
        uint32_t count
    )
    {

        accessed_state_data_t *gpu_instances, *tmp_cpu_instances;
        // allocate the GPU memory for instances
        CUDA_CHECK(cudaMalloc(
            (void **)&(gpu_instances),
            count * sizeof(accessed_state_data_t)
        ));
        // use a temporary CPU memory to allocate the GPU memory for the accounts
        // and storage
        tmp_cpu_instances = new accessed_state_data_t[count];
        memcpy(
            tmp_cpu_instances,
            cpu_instances,
            count * sizeof(accessed_state_data_t)
        );
        for (size_t idx = 0; idx < count; idx++)
        {
            // if the instance has accounts, allocate the GPU memory for them
            if (
                (tmp_cpu_instances[idx].accessed_accounts.accounts != NULL) &&
                (tmp_cpu_instances[idx].accessed_accounts.no_accounts > 0)
            )
            {
                account_t *tmp_accounts = new account_t[tmp_cpu_instances[idx].accessed_accounts.no_accounts];
                memcpy(
                    tmp_accounts,
                    tmp_cpu_instances[idx].accessed_accounts.accounts,
                    tmp_cpu_instances[idx].accessed_accounts.no_accounts * sizeof(account_t)
                );
                for (size_t jdx = 0; jdx < tmp_cpu_instances[idx].accessed_accounts.no_accounts; jdx++)
                {
                    // alocate the bytecode and storage if needed
                    if (
                        (tmp_cpu_instances[idx].accessed_accounts.accounts[jdx].bytecode != NULL) &&
                        (tmp_cpu_instances[idx].accessed_accounts.accounts[jdx].code_size > 0)
                    )
                    {
                        CUDA_CHECK(cudaMalloc(
                            (void **)&(tmp_accounts[jdx].bytecode),
                            tmp_cpu_instances[idx].accessed_accounts.accounts[jdx].code_size * sizeof(uint8_t)
                        ));
                        CUDA_CHECK(cudaMemcpy(
                            tmp_accounts[jdx].bytecode,
                            tmp_accounts[jdx].bytecode,
                            tmp_cpu_instances[idx].accessed_accounts.accounts[jdx].code_size * sizeof(uint8_t),
                            cudaMemcpyHostToDevice
                        ));
                    }
                    if (
                        (tmp_cpu_instances[idx].accessed_accounts.accounts[jdx].storage != NULL) &&
                        (tmp_cpu_instances[idx].accessed_accounts.accounts[jdx].storage_size > 0)
                    )
                    {
                        CUDA_CHECK(cudaMalloc(
                            (void **)&(tmp_accounts[jdx].storage),
                            tmp_cpu_instances[idx].accessed_accounts.accounts[jdx].storage_size * sizeof(contract_storage_t)
                        ));
                        CUDA_CHECK(cudaMemcpy(
                            tmp_accounts[jdx].storage,
                            tmp_accounts[jdx].storage,
                            tmp_cpu_instances[idx].accessed_accounts.accounts[jdx].storage_size * sizeof(contract_storage_t),
                            cudaMemcpyHostToDevice
                        ));
                    }
                }
                // allocate the GPU memory for the accounts and reads
                CUDA_CHECK(cudaMalloc(
                    (void **)&(tmp_cpu_instances[idx].accessed_accounts.accounts),
                    tmp_cpu_instances[idx].accessed_accounts.no_accounts * sizeof(account_t)
                ));
                CUDA_CHECK(cudaMemcpy(
                    tmp_cpu_instances[idx].accessed_accounts.accounts,
                    tmp_accounts,
                    tmp_cpu_instances[idx].accessed_accounts.no_accounts * sizeof(account_t),
                    cudaMemcpyHostToDevice
                ));
                CUDA_CHECK(cudaMalloc(
                    (void **)&(tmp_cpu_instances[idx].reads),
                    tmp_cpu_instances[idx].accessed_accounts.no_accounts * sizeof(uint8_t)
                ));
                CUDA_CHECK(cudaMemcpy(
                    tmp_cpu_instances[idx].reads,
                    cpu_instances[idx].reads,
                    tmp_cpu_instances[idx].accessed_accounts.no_accounts * sizeof(uint8_t),
                    cudaMemcpyHostToDevice
                ));
                delete[] tmp_accounts;
            }
        }

        CUDA_CHECK(cudaMemcpy(
            gpu_instances,
            tmp_cpu_instances,
            count * sizeof(accessed_state_data_t),
            cudaMemcpyHostToDevice
        ));
        delete[] tmp_cpu_instances;
        return gpu_instances;
    }

    /**
     * Free the GPU instances of the accessed state data.
     * @param[in] gpu_instances The GPU instances
     * @param[in] count The number of instances
    */
    __host__ static void free_gpu_instances(
        accessed_state_data_t *gpu_instances,
        uint32_t count
    )
    {
        accessed_state_data_t *tmp_cpu_instances = new accessed_state_data_t[count];
        CUDA_CHECK(cudaMemcpy(
            tmp_cpu_instances,
            gpu_instances,
            count * sizeof(accessed_state_data_t),
            cudaMemcpyDeviceToHost
        ));
        for (size_t idx = 0; idx < count; idx++)
        {
            if (
                (tmp_cpu_instances[idx].accessed_accounts.accounts != NULL) &&
                (tmp_cpu_instances[idx].accessed_accounts.no_accounts > 0)
            )
            {
                account_t *tmp_accounts = new account_t[tmp_cpu_instances[idx].accessed_accounts.no_accounts];
                CUDA_CHECK(cudaMemcpy(
                    tmp_accounts,
                    tmp_cpu_instances[idx].accessed_accounts.accounts,
                    tmp_cpu_instances[idx].accessed_accounts.no_accounts * sizeof(account_t),
                    cudaMemcpyDeviceToHost
                ));
                for (size_t jdx = 0; jdx < tmp_cpu_instances[idx].accessed_accounts.no_accounts; jdx++)
                {
                    if (
                        (tmp_accounts[jdx].bytecode != NULL) &&
                        (tmp_accounts[jdx].code_size > 0)
                    )
                    {
                        CUDA_CHECK(cudaFree(tmp_accounts[jdx].bytecode));
                        tmp_accounts[jdx].bytecode = NULL;
                    }
                    if (
                        (tmp_accounts[jdx].storage != NULL) &&
                        (tmp_accounts[jdx].storage_size > 0)
                    )
                    {
                        CUDA_CHECK(cudaFree(tmp_accounts[jdx].storage));
                        tmp_accounts[jdx].storage = NULL;
                    }
                }
                CUDA_CHECK(cudaFree(tmp_cpu_instances[idx].accessed_accounts.accounts));
                tmp_cpu_instances[idx].accessed_accounts.accounts = NULL;
                CUDA_CHECK(cudaFree(tmp_cpu_instances[idx].reads));
                tmp_cpu_instances[idx].reads = NULL;
                delete[] tmp_accounts;
                tmp_accounts = NULL;
            }
        }
        delete[] tmp_cpu_instances;
        tmp_cpu_instances = NULL;
        CUDA_CHECK(cudaFree(gpu_instances));
    }

    /**
     * Get the CPU instances of the accessed state data from
     * the GPU counterparts.
     * @param[in] gpu_instances The GPU instances
     * @param[in] count The number of instances
    */
    __host__ static accessed_state_data_t *get_cpu_instances_from_gpu_instances(
        accessed_state_data_t *gpu_instances,
        uint32_t count
    )
    {
        // temporary instances
        accessed_state_data_t *cpu_instances, *tmp_gpu_instances, *tmp_cpu_instances;
        // allocate the CPU memory for instances
        // and copy the initial details of the accessed state
        // like the number of accounts and the pointer to the accounts
        // and their reads
        cpu_instances = new accessed_state_data_t[count];
        CUDA_CHECK(cudaMemcpy(
            cpu_instances,
            gpu_instances,
            count * sizeof(accessed_state_data_t),
            cudaMemcpyDeviceToHost
        ));
        // STEP 1: get the accounts details and read operations from GPU
        // use an axiliary emmory to alocate the necesarry memory on GPU which can be accessed from
        // the host to copy the accounts details and read operations done on the accounts.
        tmp_cpu_instances = new accessed_state_data_t[count];
        memcpy(
            tmp_cpu_instances,
            cpu_instances,
            count * sizeof(accessed_state_data_t)
        );

        for (uint32_t idx = 0; idx < count; idx++)
        {
            // if the instance has accounts, allocate the GPU memory for them
            if (
                (cpu_instances[idx].accessed_accounts.accounts != NULL) &&
                (cpu_instances[idx].accessed_accounts.no_accounts > 0) &&
                (cpu_instances[idx].reads != NULL)
            )
            {
                CUDA_CHECK(cudaMalloc(
                    (void **)&(tmp_cpu_instances[idx].accessed_accounts.accounts),
                    cpu_instances[idx].accessed_accounts.no_accounts * sizeof(account_t)
                ));
                CUDA_CHECK(cudaMalloc(
                    (void **)&(tmp_cpu_instances[idx].reads),
                    cpu_instances[idx].accessed_accounts.no_accounts * sizeof(uint8_t)
                ));
            } else {
                tmp_cpu_instances[idx].accessed_accounts.accounts = NULL;
                tmp_cpu_instances[idx].accessed_accounts.no_accounts = 0;
                tmp_cpu_instances[idx].reads = NULL;
            }
        }
        CUDA_CHECK(cudaMalloc(
            (void **)&(tmp_gpu_instances),
            count * sizeof(accessed_state_data_t)
        ));
        CUDA_CHECK(cudaMemcpy(
            tmp_gpu_instances,
            tmp_cpu_instances,
            count * sizeof(accessed_state_data_t),
            cudaMemcpyHostToDevice
        ));
        delete[] tmp_cpu_instances;

        // run the first kernel which copy the accoutns details and read operations
        kernel_accessed_state_S1<accessed_state_data_t><<<count, 1>>>(tmp_gpu_instances, gpu_instances, count);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaFree(gpu_instances);

        // STEP 2: get the accounts storage and bytecode from GPU
        gpu_instances = tmp_gpu_instances;

        CUDA_CHECK(cudaMemcpy(
            cpu_instances,
            gpu_instances,
            count * sizeof(accessed_state_data_t),
            cudaMemcpyDeviceToHost
        ));
        tmp_cpu_instances = new accessed_state_data_t[count];
        memcpy(
            tmp_cpu_instances,
            cpu_instances,
            count * sizeof(accessed_state_data_t)
        );

        for (uint32_t idx = 0; idx < count; idx++)
        {
            // if the instance has accounts
            if (
                (cpu_instances[idx].accessed_accounts.accounts != NULL) &&
                (cpu_instances[idx].accessed_accounts.no_accounts > 0) &&
                (cpu_instances[idx].reads != NULL)
            )
            {
                account_t *tmp_accounts = new account_t[cpu_instances[idx].accessed_accounts.no_accounts];
                account_t *tmp_cpu_accounts = new account_t[cpu_instances[idx].accessed_accounts.no_accounts];
                CUDA_CHECK(cudaMemcpy(
                    tmp_cpu_accounts,
                    cpu_instances[idx].accessed_accounts.accounts,
                    cpu_instances[idx].accessed_accounts.no_accounts * sizeof(account_t),
                    cudaMemcpyDeviceToHost
                ));
                CUDA_CHECK(cudaMemcpy(
                    tmp_accounts,
                    cpu_instances[idx].accessed_accounts.accounts,
                    cpu_instances[idx].accessed_accounts.no_accounts * sizeof(account_t),
                    cudaMemcpyDeviceToHost
                ));
                // go through the accounts and allocate the memory for bytecode and storage
                for (size_t jdx = 0; jdx < cpu_instances[idx].accessed_accounts.no_accounts; jdx++)
                {
                    if (
                        (tmp_cpu_accounts[jdx].bytecode != NULL) &&
                        (tmp_cpu_accounts[jdx].code_size > 0)
                    )
                    {
                        CUDA_CHECK(cudaMalloc(
                            (void **)&(tmp_accounts[jdx].bytecode),
                            tmp_cpu_accounts[jdx].code_size * sizeof(uint8_t)
                        ));
                    }
                    if (
                        (tmp_cpu_accounts[jdx].storage != NULL) &&
                        (tmp_cpu_accounts[jdx].storage_size > 0)
                    )
                    {
                        CUDA_CHECK(cudaMalloc(
                            (void **)&(tmp_accounts[jdx].storage),
                            tmp_cpu_accounts[jdx].storage_size * sizeof(contract_storage_t)
                        ));
                    }
                }
                CUDA_CHECK(cudaMalloc(
                    (void **)&(tmp_cpu_instances[idx].accessed_accounts.accounts),
                    cpu_instances[idx].accessed_accounts.no_accounts * sizeof(account_t)
                ));
                CUDA_CHECK(cudaMemcpy(
                    tmp_cpu_instances[idx].accessed_accounts.accounts,
                    tmp_accounts,
                    cpu_instances[idx].accessed_accounts.no_accounts * sizeof(account_t),
                    cudaMemcpyHostToDevice
                ));
                delete[] tmp_cpu_accounts;
                tmp_cpu_accounts = NULL;
                delete[] tmp_accounts;
                tmp_accounts = NULL;
            }
        }

        CUDA_CHECK(cudaMalloc(
            (void **)&(tmp_gpu_instances),
            count * sizeof(accessed_state_data_t)
        ));
        CUDA_CHECK(cudaMemcpy(
            tmp_gpu_instances,
            tmp_cpu_instances,
            count * sizeof(accessed_state_data_t),
            cudaMemcpyHostToDevice
        ));
        delete[] tmp_cpu_instances;

        // run the second kernel which copy the bytecode and storage
        kernel_accessed_state_S2<accessed_state_data_t><<<count, 1>>>(tmp_gpu_instances, gpu_instances, count);
        CUDA_CHECK(cudaDeviceSynchronize());

        // free the memory on GPU for the first kernel (accounts details)
        // the read operations are fixed and they do not have more depth
        // so it can be kept
        for (size_t idx = 0; idx < count; idx++)
        {
            if (
                (cpu_instances[idx].accessed_accounts.accounts != NULL) &&
                (cpu_instances[idx].accessed_accounts.no_accounts > 0) &&
                (cpu_instances[idx].reads != NULL)
            )
            {
                CUDA_CHECK(cudaFree(cpu_instances[idx].accessed_accounts.accounts));
                //CUDA_CHECK(cudaFree(cpu_instances[idx].reads));
            }
        }

        CUDA_CHECK(cudaFree(gpu_instances));
        gpu_instances = tmp_gpu_instances;

        // STEP 3: copy the the entire accessed state data from GPU to CPU
        CUDA_CHECK(cudaMemcpy(
            cpu_instances,
            gpu_instances,
            count * sizeof(accessed_state_data_t),
            cudaMemcpyDeviceToHost
        ));
        tmp_cpu_instances = new accessed_state_data_t[count];
        memcpy(
            tmp_cpu_instances,
            cpu_instances,
            count * sizeof(accessed_state_data_t)
        );

        for (size_t idx = 0; idx < count; idx++)
        {
            // if the instance has accounts
            if (
                (tmp_cpu_instances[idx].accessed_accounts.accounts != NULL) &&
                (tmp_cpu_instances[idx].accessed_accounts.no_accounts > 0) &&
                (tmp_cpu_instances[idx].reads != NULL)
            )
            {
                account_t *tmp_accounts, *aux_tmp_accounts;
                tmp_accounts = new account_t[tmp_cpu_instances[idx].accessed_accounts.no_accounts];
                aux_tmp_accounts = new account_t[tmp_cpu_instances[idx].accessed_accounts.no_accounts];
                CUDA_CHECK(cudaMemcpy(
                    tmp_accounts,
                    tmp_cpu_instances[idx].accessed_accounts.accounts,
                    cpu_instances[idx].accessed_accounts.no_accounts * sizeof(account_t),
                    cudaMemcpyDeviceToHost
                ));
                CUDA_CHECK(cudaMemcpy(
                    aux_tmp_accounts,
                    tmp_cpu_instances[idx].accessed_accounts.accounts,
                    cpu_instances[idx].accessed_accounts.no_accounts * sizeof(account_t),
                    cudaMemcpyDeviceToHost
                ));
                // go through the accounts and copy the bytecode and the storage
                for (size_t jdx = 0; jdx < tmp_cpu_instances[idx].accessed_accounts.no_accounts; jdx++)
                {
                    if (
                        (aux_tmp_accounts[jdx].bytecode != NULL) &&
                        (aux_tmp_accounts[jdx].code_size > 0)
                    )
                    {
                        tmp_accounts[jdx].bytecode = new uint8_t[aux_tmp_accounts[jdx].code_size * sizeof(uint8_t)];
                        CUDA_CHECK(cudaMemcpy(
                            tmp_accounts[jdx].bytecode,
                            aux_tmp_accounts[jdx].bytecode,
                            aux_tmp_accounts[jdx].code_size * sizeof(uint8_t),
                            cudaMemcpyDeviceToHost
                        ));
                    }
                    if (
                        (aux_tmp_accounts[jdx].storage != NULL) &&
                        (aux_tmp_accounts[jdx].storage_size > 0)
                    )
                    {
                        tmp_accounts[jdx].storage = new contract_storage_t[aux_tmp_accounts[jdx].storage_size];
                        CUDA_CHECK(cudaMemcpy(
                            tmp_accounts[jdx].storage,
                            aux_tmp_accounts[jdx].storage,
                            aux_tmp_accounts[jdx].storage_size * sizeof(contract_storage_t),
                            cudaMemcpyDeviceToHost));
                    }
                }
                delete[] aux_tmp_accounts;
                aux_tmp_accounts = NULL;
                tmp_cpu_instances[idx].accessed_accounts.accounts = tmp_accounts;
                uint8_t *tmp_reads = new uint8_t[tmp_cpu_instances[idx].accessed_accounts.no_accounts];
                CUDA_CHECK(cudaMemcpy(
                    tmp_reads,
                    tmp_cpu_instances[idx].reads,
                    tmp_cpu_instances[idx].accessed_accounts.no_accounts * sizeof(uint8_t),
                    cudaMemcpyDeviceToHost
                ));
                tmp_cpu_instances[idx].reads = tmp_reads;
            }
        }

        free_gpu_instances(gpu_instances, count);
        memcpy(
            cpu_instances,
            tmp_cpu_instances,
            count * sizeof(accessed_state_data_t)
        );
        delete[] tmp_cpu_instances;
        tmp_cpu_instances = NULL;
        return cpu_instances;
    }

    /**
     * Print the accessed state data structure.
     * @param[in] arith The arithemtic instance
     * @param[in] accessed_state_data The accessed state data
    */
    __host__ __device__ __forceinline__ static void print_accessed_state_data_t(
        arith_t &arith,
        accessed_state_data_t &accessed_state_data
    )
    {
        printf("no_accounts: %lu\n", accessed_state_data.accessed_accounts.no_accounts);
        for (size_t idx = 0; idx < accessed_state_data.accessed_accounts.no_accounts; idx++)
        {
            printf("accounts[%lu]:\n", idx);
            world_state_t::print_account_t(arith, accessed_state_data.accessed_accounts.accounts[idx]);
            printf("read: %hhu\n", accessed_state_data.reads[idx]);
        }
    }

    /**
     * Print the state.
    */
    __host__ __device__ __forceinline__ void print()
    {
        print_accessed_state_data_t(_arith, *_content);
    }

    /**
     * Get json from the accessed state data structure.
     * @param[in] arith The arithemtic instance
     * @param[in] accessed_state_data The accessed state data
     * @return The json of the accessed state data
    */
    __host__ static cJSON *json_from_accessed_state_data_t(
        arith_t &arith,
        accessed_state_data_t &accessed_state_data
    )
    {
        cJSON *accessed_state_json = NULL;
        cJSON *account_json = NULL;
        char *hex_string_ptr = new char[arith_t::BYTES * 2 + 3];
        size_t idx = 0;
        accessed_state_json = cJSON_CreateObject();
        for (idx = 0; idx < accessed_state_data.accessed_accounts.no_accounts; idx++)
        {
            account_json = world_state_t::json_from_account_t(
                arith,
                accessed_state_data.accessed_accounts.accounts[idx]);
            cJSON_AddItemToObject(account_json, "read", cJSON_CreateNumber(accessed_state_data.reads[idx]));
            arith.hex_string_from_cgbn_memory(
                hex_string_ptr,
                accessed_state_data.accessed_accounts.accounts[idx].address,
                5
            );
            cJSON_AddItemToObject(accessed_state_json, hex_string_ptr, account_json);
        }
        delete[] hex_string_ptr;
        hex_string_ptr = NULL;
        return accessed_state_json;
    }
    /**
     * Get json of the state
     * @return The json of the state
    */
    __host__ __forceinline__ cJSON *json()
    {
        return json_from_accessed_state_data_t(_arith, *_content);
    }
};


template <typename T>
__global__ void kernel_accessed_state_S1(
    T *dst_instances,
    T *src_instances,
    uint32_t count)
{
    uint32_t instance = blockIdx.x * blockDim.x + threadIdx.x;
    typedef typename world_state_t::account_t account_t;

    if (instance >= count)
        return;

    if (
        (src_instances[instance].accessed_accounts.accounts != NULL) &&
        (src_instances[instance].accessed_accounts.no_accounts > 0) &&
        (src_instances[instance].reads != NULL)
    )
    {
        memcpy(
            dst_instances[instance].accessed_accounts.accounts,
            src_instances[instance].accessed_accounts.accounts,
            src_instances[instance].accessed_accounts.no_accounts * sizeof(account_t)
        );
        delete[] src_instances[instance].accessed_accounts.accounts;
        src_instances[instance].accessed_accounts.accounts = NULL;
        memcpy(
            dst_instances[instance].reads,
            src_instances[instance].reads,
            src_instances[instance].accessed_accounts.no_accounts * sizeof(uint8_t)
        );
        delete[] src_instances[instance].reads;
        src_instances[instance].reads = NULL;
    }
}

template <typename T>
__global__ void kernel_accessed_state_S2(
    T *dst_instances,
    T *src_instances,
    uint32_t count)
{
    uint32_t instance = blockIdx.x * blockDim.x + threadIdx.x;
    typedef typename world_state_t::account_t account_t;
    typedef typename world_state_t::contract_storage_t contract_storage_t;

    if (instance >= count)
        return;

    if (
        (src_instances[instance].accessed_accounts.accounts != NULL) &&
        (src_instances[instance].accessed_accounts.no_accounts > 0)
    )
    {
        for (size_t idx = 0; idx < src_instances[instance].accessed_accounts.no_accounts; idx++)
        {
            if (
                (src_instances[instance].accessed_accounts.accounts[idx].bytecode != NULL) &&
                (src_instances[instance].accessed_accounts.accounts[idx].code_size > 0)
            )
            {
                memcpy(
                    dst_instances[instance].accessed_accounts.accounts[idx].bytecode,
                    src_instances[instance].accessed_accounts.accounts[idx].bytecode,
                    src_instances[instance].accessed_accounts.accounts[idx].code_size * sizeof(uint8_t)
                );
                delete[] src_instances[instance].accessed_accounts.accounts[idx].bytecode;
                src_instances[instance].accessed_accounts.accounts[idx].bytecode = NULL;
            }

            if (
                (src_instances[instance].accessed_accounts.accounts[idx].storage != NULL) &&
                (src_instances[instance].accessed_accounts.accounts[idx].storage_size > 0)
            )
            {
                memcpy(
                    dst_instances[instance].accessed_accounts.accounts[idx].storage,
                    src_instances[instance].accessed_accounts.accounts[idx].storage,
                    src_instances[instance].accessed_accounts.accounts[idx].storage_size * sizeof(contract_storage_t)
                );
                delete[] src_instances[instance].accessed_accounts.accounts[idx].storage;
                src_instances[instance].accessed_accounts.accounts[idx].storage = NULL;
            }
        }
    }
}


/**
 * Class to represent the touch state.
 * The touch state is the state which
 * contains the acounts modified
 * by the execution of the transaction.
 * YP: accrued transaction substate
 *  \f$A_{t}\f$ for touch accounts
*/
class touch_state_t
{
public:

    /**
     * The storage entry type.
    */
    typedef world_state_t::contract_storage_t contract_storage_t;
    /**
     * The account type.
    */
    typedef world_state_t::account_t account_t;
    /**
     * The state data type. Contains the accounts with their storage.
    */
    typedef world_state_t::state_data_t state_data_t;
    /**
     * The accessed state data type. Contains the accounts with their storage
     * and the read operations done on the accounts.
    */
    // typedef accessed_state_t<params> accessed_state_t;

    /**
     * The touch state data type. Contains the accounts with their storage
     * and the write operations done on the accounts.
     * The write operations can be:
     *  0 - no, 1 - bytecode, 2 - balance, 4 - nonce, 8 - storage
    */
    typedef struct
    {
        state_data_t touch_accounts; /**< The touch accounts */
        uint8_t *touch;             /**< The write operations */
    } touch_state_data_t;

    touch_state_data_t *_content; /**< The content of the touch state */
    arith_t _arith;              /**< The arithmetical environment */
    accessed_state_t *_accessed_state; /**< The accessed state */
    touch_state_t *_parent_state;  /**< The parent touch state */
    bool nodestruct = false;


    /**
     * Constructor with given content.
     * @param[in] content The content of the touch state
     * @param[in] access_state The accessed state
     * @param[in] parent_state The parent touch state
    */
    __host__ __device__ __forceinline__ touch_state_t(
        touch_state_data_t *content,
        accessed_state_t *access_state,
        touch_state_t *parent_state
    ) : _arith(access_state->_arith),
        _content(content),
        _accessed_state(access_state),
        _parent_state(parent_state)
    {
    }

    __host__ __device__ __forceinline__ touch_state_t(
        touch_state_data_t *content,
        accessed_state_t *access_state,
        arith_t &arith
    ) : _arith(arith),
        _content(content),
        _accessed_state(access_state),
        _parent_state(nullptr),
        nodestruct(true)
    {
    }

    /**
     * Constructor with given accessed state and parent touch state.
     * @param[in] access_state The accessed state
     * @param[in] parent_state The parent touch state
    */
    __host__ __device__ __forceinline__ touch_state_t(
        accessed_state_t *access_state,
        touch_state_t *parent_state
    ) : _arith(access_state->_arith),
        _accessed_state(access_state),
        _parent_state(parent_state)
    {
        // aloocate the memory for the touch state
        // and initialize it
        SHARED_MEMORY touch_state_data_t *tmp_content;
        ONE_THREAD_PER_INSTANCE(
            tmp_content = new touch_state_data_t;
            tmp_content->touch_accounts.no_accounts = 0;
            tmp_content->touch_accounts.accounts = NULL;
            tmp_content->touch = NULL;
        )
        _content = tmp_content;
    }

    /**
     * The destructor.
     * It frees the memory allocated for the touch state.
    */
    __host__ __device__ __forceinline__ ~touch_state_t()
    {
        if (nodestruct) { // skip freeing internal memory, assuming they're borrowed
            return;
        };
        ONE_THREAD_PER_INSTANCE(
            if (_content != NULL)
            {
                if (_content->touch_accounts.accounts != NULL)
                {
                    for (size_t idx = 0; idx < _content->touch_accounts.no_accounts; idx++)
                    {
                        if (_content->touch_accounts.accounts[idx].bytecode != NULL)
                        {
                            delete[] _content->touch_accounts.accounts[idx].bytecode;
                            _content->touch_accounts.accounts[idx].bytecode = NULL;
                        }
                        if (_content->touch_accounts.accounts[idx].storage != NULL)
                        {
                            delete[] _content->touch_accounts.accounts[idx].storage;
                            _content->touch_accounts.accounts[idx].storage = NULL;
                        }
                    }
                    delete[] _content->touch_accounts.accounts;
                    _content->touch_accounts.accounts = NULL;
                    _content->touch_accounts.no_accounts = 0;
                }
                if (_content->touch != NULL)
                {
                    delete[] _content->touch;
                    _content->touch = NULL;
                }
                delete _content;
            }
        )
        _content = NULL;
    }

    /**
     * Get the index of the account with the given address.
     * @param[in] address The address of the account
     * @param[out] error_code The error code if the account does not exist
     * @return The index of the account
    */
    __host__ __device__ __forceinline__ size_t get_account_index(
        bn_t &address,
        uint32_t &error_code
    )
    {
        bn_t local_address;
        for (size_t idx = 0; idx < _content->touch_accounts.no_accounts; idx++)
        {
            cgbn_load(_arith._env, local_address, &(_content->touch_accounts.accounts[idx].address));
            if (cgbn_compare(_arith._env, local_address, address) == 0)
            {
                return idx;
            }
        }
        error_code = ERR_STATE_INVALID_ADDRESS;
        return 0;
    }

    /**
     * Get the gas for accessing the account with the given address.
     * @param[in] address The address of the account
     * @param[inout] gas_used The gas used after the access
    */
    __host__ __device__ __forceinline__ void charge_gas_access_account(
        bn_t &address,
        bn_t &gas_used
    )
    {
        bn_t gas_cost;
        _accessed_state->get_access_account_gas_cost(address, gas_cost);
        cgbn_add(_arith._env, gas_used, gas_used, gas_cost);
    }

    /**
     * Get the gas for accessing the given storage key
     * of the account with the given address.
     * @param[in] address The address of the account
     * @param[in] key The key of the storage
     * @param[inout] gas_used The gas used after the access
    */
    __host__ __device__ __forceinline__ void charge_gas_access_storage(
        bn_t &address,
        bn_t &key,
        bn_t &gas_used
    )
    {
        bn_t gas_cost;
        _accessed_state->get_access_storage_gas_cost(address, key, gas_cost);
        cgbn_add(_arith._env, gas_used, gas_used, gas_cost);
    }

    /**
     * Get the account with the given address.
     * If the account does not exist in the touch state,
     * it is searched in the parent touch state and if it
     * does not exist there (also their respective parant),
     * it is searched in the accessed state. At the end,
     * it set the read operation on the account in the
     * accessed state.
     * @param[in] address The address of the account
     * @param[out] read_type The read type
     * @return The account
    */
    __host__ __device__ __forceinline__ account_t *get_account(
        bn_t &address,
        uint32_t read_type
    )
    {
        uint32_t tmp_error_code = ERR_SUCCESS;
        // get the index of the account
        size_t account_idx = get_account_index(address, tmp_error_code);
        account_t *account = NULL;
        // if the account does not exist in the touch state
        if (tmp_error_code != ERR_SUCCESS)
        {
            // search the account in the parent touch state
            // recursivilly until it is found or the root touch state is reached
            touch_state_t *tmp_parent_state = _parent_state;
            while (tmp_parent_state != NULL)
            {
                tmp_error_code = ERR_SUCCESS;
                account_idx = tmp_parent_state->get_account_index(address, tmp_error_code);
                if (tmp_error_code == ERR_SUCCESS)
                {
                    account = &(tmp_parent_state->_content->touch_accounts.accounts[account_idx]);
                    break;
                }
                tmp_parent_state = tmp_parent_state->_parent_state;
            }
            // if the account does not exist in the parent touch state
            // search it in the accessed state
            if (tmp_error_code != ERR_SUCCESS)
            {
                account = _accessed_state->get_account(address, read_type);
            }
        }
        else
        {
            account = &(_content->touch_accounts.accounts[account_idx]);
        }

        // read the account in access state
        _accessed_state->get_account(address, read_type);

        return account;
    }

    /**
     * Get the account nonce of the account with the given address.
     * @param[in] address The address of the account
     * @param[out] nonce The nonce of the account
    */
    __host__ __device__ __forceinline__ void get_account_nonce(
        bn_t &address,
        bn_t &nonce
    )
    {
        account_t *account = get_account(address, READ_NONCE);
        cgbn_load(_arith._env, nonce, &(account->nonce));
    }

    /**
     * Get the account balance of the account with the given address.
     * @param[in] address The address of the account
     * @param[out] balance The balance of the account
    */
    __host__ __device__ __forceinline__ void get_account_balance(
        bn_t &address,
        bn_t &balance
    )
    {
        account_t *account = get_account(address, READ_BALANCE);
        cgbn_load(_arith._env, balance, &(account->balance));
    }

    /**
     * Get the account code size of the account with the given address.
     * @param[in] address The address of the account
     * @return The code size of the account
    */
    __host__ __device__ __forceinline__ size_t get_account_code_size(
        bn_t &address
    )
    {
        account_t *account = get_account(address, READ_CODE);
        return account->code_size;
    }

    /**
     * Get the account code of the account with the given address.
     * @param[in] address The address of the account
     * @return The code of the account
    */
    __host__ __device__ __forceinline__ uint8_t *get_account_code(
        bn_t &address
    )
    {
        account_t *account = get_account(address, READ_CODE);
        return account->bytecode;
    }

    /**
     * Get the account code data at the given index for the given length
     * of the account with the given address.
     * If the index is greater than the code size, it returns NULL.
     * If the length is greater than the code size - index, it returns
     * the code data from index to the end of the code and sets the
     * available size to the code size - index. Otherwise, it returns
     * the code data from index to index + length and sets the available
     * size to length.
     * @param[in] address The address of the account
     * @param[in] index The index of the code data
     * @param[in] length The length of the code data
     * @param[out] available_size The available size of the code data
    */
    __host__ __device__ __forceinline__ uint8_t *get_account_code_data(
        bn_t &address,
        bn_t &index,
        bn_t &length,
        size_t &available_size
    )
    {
        account_t *account = get_account(address, READ_CODE);
        data_content_t code_data;
        code_data.data = account->bytecode;
        code_data.size = account->code_size;
        return _arith.get_data(
            code_data,
            index,
            length,
            available_size
        );
    }

    /**
     * Get the index of the account given by the address,
     * or if it does not exist, add it to the list of accounts,
     * and return the index of the new account.
     *
     * It setup the new account with the details from the most updated
     * version in the parents or the accessed state (global).
     *
     * @param[in] address The address of the account
    */
    __host__ __device__  size_t set_account(
        bn_t &address
    )
    {
        uint32_t tmp_error_code = ERR_SUCCESS;
        // get the index of the account
        size_t account_idx = get_account_index(address, tmp_error_code);
        // if the account does not exist in the current touch state
        if (tmp_error_code != ERR_SUCCESS)
        {
            account_t *account = NULL;
            uint8_t touch = 0;
            SHARED_MEMORY account_t *dup_account;
            ONE_THREAD_PER_INSTANCE(
                dup_account = new account_t;
            )
            // search the account in the parent touch state
            // recursivilly until it is found or the root touch state is reached
            touch_state_t *tmp_parent_state = _parent_state;
            while (tmp_parent_state != NULL)
            {
                tmp_error_code = ERR_SUCCESS;
                account_idx = tmp_parent_state->get_account_index(address, tmp_error_code);
                if (tmp_error_code == ERR_SUCCESS)
                {
                    account = &(tmp_parent_state->_content->touch_accounts.accounts[account_idx]);
                    touch = tmp_parent_state->_content->touch[account_idx];
                    break;
                }
                tmp_parent_state = tmp_parent_state->_parent_state;
            }
            // if the account does not exist in the tree of touch states
            // search it in the accessed state/global state
            if (tmp_error_code != ERR_SUCCESS)
            {
                account = _accessed_state->get_account(address, READ_NONE);
                // there is no previous touch, so the account is not touched
                touch = WRITE_NONE;
            }
            // add the new account to the list
            account_idx = _content->touch_accounts.no_accounts;
            ONE_THREAD_PER_INSTANCE(
                // dulicate the account
                memcpy(
                    dup_account,
                    account,
                    sizeof(account_t)
                );
                // duplicate the bytecode if needed
                if (
                    (account->code_size > 0) &&
                    (account->bytecode != NULL)
                )
                {
                    dup_account->bytecode = new uint8_t[account->code_size * sizeof(uint8_t)];
                    memcpy(
                        dup_account->bytecode,
                        account->bytecode,
                        account->code_size * sizeof(uint8_t)
                    );
                    dup_account->code_size = account->code_size;
                } else {
                    delete[] dup_account->bytecode;
                    dup_account->bytecode = NULL;
                    dup_account->code_size = 0;
                }
                // no storage copy
                dup_account->storage_size = 0;
                dup_account->storage = NULL;
                // alocate the necessary memory for the new account
                account_t *tmp_accounts = new account_t[_content->touch_accounts.no_accounts + 1];
                uint8_t *tmp_touch = new uint8_t[_content->touch_accounts.no_accounts + 1];
                if (_content->touch_accounts.no_accounts > 0) {
                    memcpy(
                        tmp_accounts,
                        _content->touch_accounts.accounts,
                        _content->touch_accounts.no_accounts * sizeof(account_t)
                    );
                    memcpy(
                        tmp_touch,
                        _content->touch,
                        _content->touch_accounts.no_accounts * sizeof(uint8_t)
                    );

                    if (!nodestruct){
                      delete[] _content->touch_accounts.accounts;
                      delete[] _content->touch;
                    }
                }
                _content->touch_accounts.accounts = tmp_accounts;
                _content->touch_accounts.no_accounts++;
                memcpy(
                    &(_content->touch_accounts.accounts[account_idx]),
                    dup_account,
                    sizeof(account_t)
                );
                _content->touch = tmp_touch;
                _content->touch[account_idx] = 0;
            )

            delete dup_account;
            dup_account = nullptr;

            // set the touch
            _content->touch[account_idx] = touch;
        }
        return account_idx;
    }

    /**
     * Set the account nonce of the account with the given address.
     * @param[in] address The address of the account
     * @param[in] nonce The nonce of the account
    */
    __host__ __device__ __forceinline__ void set_account_nonce(
        bn_t &address,
        bn_t &nonce
    )
    {
        size_t account_idx = set_account(address);
        cgbn_store(_arith._env, &(_content->touch_accounts.accounts[account_idx].nonce), nonce);
        _content->touch[account_idx] |= WRITE_NONCE;
    }

    /**
     * Set the account balance of the account with the given address.
     * @param[in] address The address of the account
     * @param[in] balance The balance of the account
    */
    __host__ __device__ __forceinline__ void set_account_balance(
        bn_t &address,
        bn_t &balance
    )
    {
        size_t account_idx = set_account(address);
        cgbn_store(_arith._env, &(_content->touch_accounts.accounts[account_idx].balance), balance);
        _content->touch[account_idx] |= WRITE_BALANCE;
    }

    /**
     * Set the account code of the account with the given address.
     * @param[in] address The address of the account
     * @param[in] code The code of the account
     * @param[in] code_size The size of the code of the account
    */
    __host__ __device__ __forceinline__ void set_account_code(
        bn_t &address,
        uint8_t *code,
        size_t code_size
    )
    {
        size_t account_idx = set_account(address);
        ONE_THREAD_PER_INSTANCE(
            if (_content->touch_accounts.accounts[account_idx].bytecode != NULL)
            {
                delete[] _content->touch_accounts.accounts[account_idx].bytecode;
            }
            _content->touch_accounts.accounts[account_idx].bytecode = new uint8_t[code_size * sizeof(uint8_t)];
            memcpy(
                _content->touch_accounts.accounts[account_idx].bytecode,
                code,
                code_size * sizeof(uint8_t)
            );
            _content->touch_accounts.accounts[account_idx].code_size = code_size;
        )
        _content->touch[account_idx] |= WRITE_CODE;
    }

    /**
     * Get the index of the storage inside the given account
     * with the given key.
     * @param[in] account The account
     * @param[in] key The key of the storage
     * @param[out] error_code The error code if the key is not found
     * @return The index of the storage in the account
    */
    __host__ __device__ __forceinline__ size_t get_storage_index(
        account_t *account,
        bn_t &key,
        uint32_t &error_code
    )
    {
        bn_t local_key;
        for (size_t idx = 0; idx < account->storage_size; idx++)
        {
            cgbn_load(_arith._env, local_key, &(account->storage[idx].key));
            if (cgbn_compare(_arith._env, local_key, key) == 0)
            {
                return idx;
            }
        }
        error_code = ERR_STATE_INVALID_KEY;
        return 0;
    }

    /**
     * Get the storage value of the account with the given address
     * and the given key.
     * @param[in] address The address of the account
     * @param[in] key The key of the storage
     * @param[out] value The value of the storage
    */
    __host__ __device__ __forceinline__ void get_value(
        bn_t &address,
        bn_t &key,
        bn_t &value
    )
    {
        uint32_t tmp_error_code = ERR_SUCCESS;
        size_t account_idx = 0;
        size_t storage_idx = 0;
        account_t *account = NULL;
        // get the index of the account
        account_idx = get_account_index(address, tmp_error_code);
        // if the account exist in the current touch state
        if (tmp_error_code == ERR_SUCCESS)
        {
            account = &(_content->touch_accounts.accounts[account_idx]);
            storage_idx = get_storage_index(account, key, tmp_error_code);
            // if the storage exist in the current touch state
            if (tmp_error_code == ERR_SUCCESS)
            {
                cgbn_load(_arith._env, value, &(account->storage[storage_idx].value));
            }
        }
        // if the storage does not exist in the current touch state
        if (tmp_error_code != ERR_SUCCESS)
        {
            // search the storage in the  tree of parent touch states
            // recursivilly until it is found or the root touch state is reached
            touch_state_t *tmp_parent_state = _parent_state;
            while (tmp_parent_state != NULL)
            {
                tmp_error_code = ERR_SUCCESS;
                account_idx = tmp_parent_state->get_account_index(address, tmp_error_code);
                if (tmp_error_code == ERR_SUCCESS)
                {
                    account = &(tmp_parent_state->_content->touch_accounts.accounts[account_idx]);
                    storage_idx = tmp_parent_state->get_storage_index(account, key, tmp_error_code);
                    if (tmp_error_code == ERR_SUCCESS)
                    {
                        cgbn_load(_arith._env, value, &(account->storage[storage_idx].value));
                        break;
                    }
                }
                tmp_parent_state = tmp_parent_state->_parent_state;
            }
            // if the storage does not exist in the tree of parent touch states
            // search it in the accessed state/global state
            if (tmp_error_code != ERR_SUCCESS)
            {
                _accessed_state->get_value(address, key, value);
            }
        }
    }

    /**
     * Get the gas cost and gas refund for the storage set operation.
     *
     * @param[in] address The address of the account
     * @param[in] key The key of the storage
     * @param[in] value The new value for the storage
     * @param[out] gas_cost The gas cost
     * @param[out] gas_refund The gas refund
    */
    __host__ __device__ __forceinline__ void get_storage_set_gas_cost_gas_refund(
        bn_t &address,
        bn_t &key,
        bn_t &value,
        bn_t &gas_cost,
        bn_t &gas_refund)
    {
        // find out if it is a warm or cold storage access
        _accessed_state->get_access_storage_gas_cost(address, key, gas_cost);
        if (cgbn_compare_ui32(_arith._env, gas_cost, GAS_WARM_ACCESS) == 0)
        {
            cgbn_set_ui32(_arith._env, gas_cost, 0); // 100 is not add here
        }
        // get the original value from accessed state/global state
        bn_t original_value;
        _accessed_state->get_value(address, key, original_value);
        // get the current value from the touch state
        bn_t current_value;
        get_value(address, key, current_value);

        // TODO: if we keep separate gas refund and remaining gas we can delete this
        cgbn_set_ui32(_arith._env, gas_refund, 0);

        // EIP-2200
        if (cgbn_compare(_arith._env, value, current_value) == 0)
        {
            cgbn_add_ui32(_arith._env, gas_cost, gas_cost, GAS_SLOAD);
        }
        else
        {
            if (cgbn_compare(_arith._env, current_value, original_value) == 0)
            {
                if (cgbn_compare_ui32(_arith._env, original_value, 0) == 0)
                {
                    cgbn_add_ui32(_arith._env, gas_cost, gas_cost, GAS_STORAGE_SET);
                }
                else
                {
                    cgbn_add_ui32(_arith._env, gas_cost, gas_cost, GAS_SSTORE_RESET);
                    if (cgbn_compare_ui32(_arith._env, value, 0)==0){
                        cgbn_add_ui32(_arith._env, gas_refund, gas_refund, 4800);
                    }
                }
            }
            else
            {
                cgbn_add_ui32(_arith._env, gas_cost, gas_cost, GAS_SLOAD);
                if (cgbn_compare_ui32(_arith._env, original_value, 0) != 0)
                {
                    if (cgbn_compare_ui32(_arith._env, current_value, 0) == 0)
                    {
                        cgbn_sub_ui32(_arith._env, gas_refund, gas_refund, GAS_STORAGE_CLEAR_REFUND);
                        //cgbn_add_ui32(_arith._env, gas_cost, gas_cost, GAS_STORAGE_CLEAR_REFUND);
                    }else if (cgbn_compare_ui32(_arith._env, value, 0) == 0)
                    {
                        cgbn_add_ui32(_arith._env, gas_refund, gas_refund, GAS_STORAGE_CLEAR_REFUND);
                    }
                }
                if (cgbn_compare(_arith._env, original_value, value) == 0)
                {
                    if (cgbn_compare_ui32(_arith._env, original_value, 0) == 0)
                    {
                        cgbn_add_ui32(_arith._env, gas_refund, gas_refund, GAS_STORAGE_SET - GAS_SLOAD);
                    }
                    else
                    {
                        if (_accessed_state->is_warm(address, key)){
                            cgbn_add_ui32(_arith._env, gas_refund, gas_refund, GAS_STORAGE_RESET - GAS_SLOAD);
                        }else{
                            cgbn_add_ui32(_arith._env, gas_refund, gas_refund, 4900);
                        }
                    }
                }
            }
        }
    }

    /**
     * Get the gas cost and gas refund for the storage set operation
     * with the given value at the given storage key in the storage
     * of the account with the given address.
     * @param[in] address The address of the account
     * @param[in] key The key of the storage
     * @param[in] value The new value for the storage
     * @param[inout] gas_used The gas cost
     * @param[inout] gas_refund The gas refund
    */
    __host__ __device__ __forceinline__ void charge_gas_set_storage(
        bn_t &address,
        bn_t &key,
        bn_t &value,
        bn_t &gas_used,
        bn_t &gas_refund
    )
    {
        bn_t gas_cost, set_gas_refund;

        // todo missing check for eip-1706?
        get_storage_set_gas_cost_gas_refund(address, key, value, gas_cost, set_gas_refund);
        cgbn_add(_arith._env, gas_used, gas_used, gas_cost);
        cgbn_add(_arith._env, gas_refund, gas_refund, set_gas_refund);
    }

    /**
     * Set the storage value for the given key in the storage
     * of the account with the given address
     * @param[in] address The address of the account
     * @param[in] key The key of the storage
     * @param[in] value The new value for the storage
    */
    __host__ __device__ __forceinline__ void set_value(
        bn_t &address,
        bn_t &key,
        bn_t &value
    )
    {
        uint32_t tmp_error_code = ERR_SUCCESS;
        size_t account_idx = 0;
        size_t storage_idx = 0;
        account_t *account = NULL;
        // get the index of the account
        account_idx = set_account(address);
        account = &(_content->touch_accounts.accounts[account_idx]);
        // get the index of the storage
        storage_idx = get_storage_index(account, key, tmp_error_code);
        // if the the key is not in the storage
        // add a new storage entry
        if (tmp_error_code != ERR_SUCCESS)
        {
            storage_idx = account->storage_size;
            ONE_THREAD_PER_INSTANCE(
                size_t new_storage_size = ++account->storage_size;
                if (new_storage_size % STORAGE_CHUNK == 1) {
                    // Round up to the next multiple of STORAGE_CHUNK
                    size_t new_capacity = ((new_storage_size + STORAGE_CHUNK - 1) / STORAGE_CHUNK) * STORAGE_CHUNK;
                    contract_storage_t *tmp_storage = new contract_storage_t[new_capacity];
                    if (account->storage_size > 0)
                    {
                        memcpy(
                            tmp_storage,
                            account->storage,
                            (new_storage_size-1) * sizeof(contract_storage_t)
                        );
                    }
                    delete[] account->storage;
                    account->storage = tmp_storage;
                }
            )
            // set the key
            cgbn_store(_arith._env, &(account->storage[storage_idx].key), key);
        }
        // set the value
        cgbn_store(_arith._env, &(account->storage[storage_idx].value), value);
        _content->touch[account_idx] |= WRITE_STORAGE;
    }

    /**
     * Register the account for delete.
     * @param[in] address The address of the account
    */
    __host__ __device__ __forceinline__ void delete_account(
        bn_t &address
    )
    {
        size_t account_idx = set_account(address);
        _content->touch[account_idx] |= WRITE_DELETE;
    }

    /**
     * Get if the account with the given address is empty.
     * An account is empty if it has zero balance, zero nonce
     * and zero code size.
     * @param[in] address The address of the account
     * @return 1 if the account is empty, 0 otherwise
    */
    __host__ __device__ __forceinline__ int32_t is_empty_account(
        bn_t &address
    )
    {
        account_t *account = get_account(address, READ_NONE);
        bn_t balance, nonce;
        cgbn_load(_arith._env, balance, &(account->balance));
        cgbn_load(_arith._env, nonce, &(account->nonce));
        return (
            (cgbn_compare_ui32(_arith._env, balance, 0) == 0) &&
            (cgbn_compare_ui32(_arith._env, nonce, 0) == 0) &&
            (account->code_size == 0)
        );

    }

    /**
     * Get if the account with the given address is register for delete.
     * @param[in] address The address of the account
     * @return 1 if the account is register for delete, 0 otherwise
    */
    __host__ __device__ __forceinline__ int32_t is_delete_account(
        bn_t &address
    )
    {
        uint32_t tmp_error_code = ERR_SUCCESS;
        size_t account_idx = 0;
        // get the index of the account
        account_idx = get_account_index(address, tmp_error_code);
        // if the account exist in the current touch state
        if (tmp_error_code == ERR_SUCCESS)
        {
            return (_content->touch[account_idx] & WRITE_DELETE);
        }
        else
        {
            // if the storage does not exist in the current touch state
            // search the storage in the  tree of parent touch states
            // recursivilly until it is found or the root touch state is reached
            touch_state_t *tmp_parent_state = _parent_state;
            while (tmp_parent_state != NULL)
            {
                tmp_error_code = ERR_SUCCESS;
                account_idx = tmp_parent_state->get_account_index(address, tmp_error_code);
                if (tmp_error_code == ERR_SUCCESS)
                {
                    return (tmp_parent_state->_content->touch[account_idx] & WRITE_DELETE);
                }
                tmp_parent_state = tmp_parent_state->_parent_state;
            }

            return 0;
        }
    }


    /**
     * Get if the account with the given address exists.
     * @param[in] address The address of the account
     * @return 1 if the account exists, 0 otherwise
    */
    __host__ __device__ __forceinline__ int32_t is_alive_account(
        bn_t &address
    )
    {
        uint32_t tmp_error_code = ERR_SUCCESS;
        // get the index of the account
        get_account_index(address, tmp_error_code);
        // if the account exist in the current touch state
        if (tmp_error_code == ERR_SUCCESS)
        {
            return 1;
        }
        else
        {
            // if the storage does not exist in the current touch state
            // search the storage in the  tree of parent touch states
            // recursivilly until it is found or the root touch state is reached
            touch_state_t *tmp_parent_state = _parent_state;
            while (tmp_parent_state != NULL)
            {
                tmp_error_code = ERR_SUCCESS;
                tmp_parent_state->get_account_index(address, tmp_error_code);
                if (tmp_error_code == ERR_SUCCESS)
                {
                    return 1;
                }
                tmp_parent_state = tmp_parent_state->_parent_state;
            }


            // if the account does not exist in the tree of touch states
            // search it in the accessed state/global state
            tmp_error_code = ERR_SUCCESS;
            _accessed_state->get_account_index(
                address,
                tmp_error_code);
            if (tmp_error_code == ERR_SUCCESS)
            {
                return 1;
            }
            tmp_error_code = ERR_SUCCESS;
            _accessed_state->_world_state->get_account_index(
                address,
                tmp_error_code);
            if (tmp_error_code == ERR_SUCCESS)
            {
                return 1;
            }

            return 0;
        }
    }

    /**
     * Get if the account with the given address is a contract.
     * @param[in] address The address of the account
     * @return 1 if the account is a contract, 0 otherwise
    */
    __host__ __device__ __forceinline__ int32_t is_contract(
        bn_t &address
    )
    {
        uint32_t tmp_error_code = ERR_SUCCESS;
        size_t account_idx = 0;
        // get the index of the account
        account_idx = get_account_index(address, tmp_error_code);
        // if the account exist in the current touch state
        if (tmp_error_code == ERR_SUCCESS)
        {
            return _content->touch_accounts.accounts[account_idx].code_size > 0;
        }
        else
        {
            // if the storage does not exist in the current touch state
            // search the storage in the  tree of parent touch states
            // recursivilly until it is found or the root touch state is reached
            touch_state_t *tmp_parent_state = _parent_state;
            while (tmp_parent_state != NULL)
            {
                tmp_error_code = ERR_SUCCESS;
                account_idx = tmp_parent_state->get_account_index(address, tmp_error_code);
                if (tmp_error_code == ERR_SUCCESS)
                {
                    return tmp_parent_state->_content->touch_accounts.accounts[account_idx].code_size > 0;
                }
                tmp_parent_state = tmp_parent_state->_parent_state;
            }


            // if the account does not exist in the tree of touch states
            // search it in the accessed state/global state
            tmp_error_code = ERR_SUCCESS;
            account_idx = _accessed_state->get_account_index(address, tmp_error_code);
            if (tmp_error_code == ERR_SUCCESS)
            {
                return _accessed_state->_content->accessed_accounts.accounts[account_idx].code_size > 0;
            }
            tmp_error_code = ERR_SUCCESS;
            account_idx = _accessed_state->_world_state->get_account_index(address, tmp_error_code);
            if (tmp_error_code == ERR_SUCCESS)
            {
                return _accessed_state->_world_state->_content->accounts[account_idx].code_size > 0;
            }

            return 0;
        }
    }

    /**
     * Update the current touch state with the touch state of a children
     * @param[in] child The touch state of the child
    */
    __host__ __device__ __forceinline__ void update_with_child_state(
        const touch_state_t &child
    )
    {
        size_t idx, jdx;
        account_t *account = NULL;
        bn_t address, key, value;
        bn_t balance, nonce;
        size_t account_idx = 0;

        // go through all the accounts of the children
        for (idx = 0; idx < child._content->touch_accounts.no_accounts; idx++)
        {
            // get the address of the account
            cgbn_load(_arith._env, address, &(child._content->touch_accounts.accounts[idx].address));
            // get the index of the account or set the account if it is a new one
            account_idx = set_account(address);
            account = &(_content->touch_accounts.accounts[account_idx]);

            // if the account balance has been modified in the child touch state
            if (child._content->touch[idx] & WRITE_BALANCE)
            {
                cgbn_load(_arith._env, balance, &(child._content->touch_accounts.accounts[idx].balance));
                cgbn_store(_arith._env, &(account->balance), balance);
                _content->touch[account_idx] |= WRITE_BALANCE;
            }

            // if the account nonce has been modified in the child touch state
            if (child._content->touch[idx] & WRITE_NONCE)
            {
                cgbn_load(_arith._env, nonce, &(child._content->touch_accounts.accounts[idx].nonce));
                cgbn_store(_arith._env, &(account->nonce), nonce);
                _content->touch[account_idx] |= WRITE_NONCE;
            }

            // if the account code has been modified in the child touch state
            if (child._content->touch[idx] & WRITE_CODE)
            {
                ONE_THREAD_PER_INSTANCE(
                    if (account->bytecode != NULL)
                    {
                        delete[] account->bytecode;
                        account->bytecode = nullptr;
                    }
                    account->bytecode = new uint8_t[child._content->touch_accounts.accounts[idx].code_size * sizeof(uint8_t)];
                    memcpy(
                        account->bytecode,
                        child._content->touch_accounts.accounts[idx].bytecode,
                        child._content->touch_accounts.accounts[idx].code_size * sizeof(uint8_t)
                    );
                    account->code_size = child._content->touch_accounts.accounts[idx].code_size;
                )
                _content->touch[account_idx] |= WRITE_CODE;
            }

            // go through all the storage entries of the child account
            for (jdx = 0; jdx < child._content->touch_accounts.accounts[idx].storage_size; jdx++)
            {
                cgbn_load(_arith._env, key, &(child._content->touch_accounts.accounts[idx].storage[jdx].key));
                cgbn_load(_arith._env, value, &(child._content->touch_accounts.accounts[idx].storage[jdx].value));
                set_value(address, key, value);
            }
            // if the account storage has been modified in the child touch state
            if (child._content->touch[idx] & WRITE_STORAGE)
            {
                _content->touch[account_idx] |= WRITE_STORAGE;
            }
            // if the account has rgister for delete
            if (child._content->touch[idx] & WRITE_DELETE)
            {
                _content->touch[account_idx] |= WRITE_DELETE;
            }
        }
    }

    /**
     * Copy the content of the touch state to the given touch state data.
     * @param[out] touch_state_data The touch state data
    */
    __host__ __device__ __forceinline__ void to_touch_state_data_t(
        touch_state_data_t &touch_state_data
    )
    {
        ONE_THREAD_PER_INSTANCE(
        // free the memory if it is already allocated
        if (touch_state_data.touch_accounts.no_accounts > 0)
        {
            for (size_t idx = 0; idx < touch_state_data.touch_accounts.no_accounts; idx++)
            {
                if (touch_state_data.touch_accounts.accounts[idx].bytecode != NULL)
                {
                    delete[] touch_state_data.touch_accounts.accounts[idx].bytecode;
                    touch_state_data.touch_accounts.accounts[idx].bytecode = NULL;
                }
                if (touch_state_data.touch_accounts.accounts[idx].storage != NULL)
                {
                    delete[] touch_state_data.touch_accounts.accounts[idx].storage;
                    touch_state_data.touch_accounts.accounts[idx].storage = NULL;
                }
            }
            delete[] touch_state_data.touch_accounts.accounts;
            touch_state_data.touch_accounts.no_accounts = 0;
            touch_state_data.touch_accounts.accounts = NULL;
            delete[] touch_state_data.touch;
            touch_state_data.touch = NULL;
        }

        // copy the content and alocate the necessary memory
        touch_state_data.touch_accounts.no_accounts = _content->touch_accounts.no_accounts;
        if (touch_state_data.touch_accounts.no_accounts > 0)
        {
            touch_state_data.touch_accounts.accounts = new account_t[touch_state_data.touch_accounts.no_accounts];
            touch_state_data.touch = new uint8_t[touch_state_data.touch_accounts.no_accounts];
            memcpy(
                touch_state_data.touch_accounts.accounts,
                _content->touch_accounts.accounts,
                touch_state_data.touch_accounts.no_accounts * sizeof(account_t)
            );
            memcpy(
                touch_state_data.touch,
                _content->touch,
                touch_state_data.touch_accounts.no_accounts * sizeof(uint8_t)
            );
            for (size_t idx = 0; idx < touch_state_data.touch_accounts.no_accounts; idx++)
            {
                if (touch_state_data.touch_accounts.accounts[idx].code_size > 0)
                {
                    touch_state_data.touch_accounts.accounts[idx].bytecode = new uint8_t[touch_state_data.touch_accounts.accounts[idx].code_size * sizeof(uint8_t)];
                    memcpy(
                        touch_state_data.touch_accounts.accounts[idx].bytecode,
                        _content->touch_accounts.accounts[idx].bytecode,
                        touch_state_data.touch_accounts.accounts[idx].code_size * sizeof(uint8_t)
                    );
                }
                else
                {
                    touch_state_data.touch_accounts.accounts[idx].bytecode = NULL;
                }

                if (touch_state_data.touch_accounts.accounts[idx].storage_size > 0)
                {
                    touch_state_data.touch_accounts.accounts[idx].storage = new contract_storage_t[touch_state_data.touch_accounts.accounts[idx].storage_size];
                    memcpy(
                        touch_state_data.touch_accounts.accounts[idx].storage,
                        _content->touch_accounts.accounts[idx].storage,
                        touch_state_data.touch_accounts.accounts[idx].storage_size * sizeof(contract_storage_t)
                    );
                }
                else
                {
                    touch_state_data.touch_accounts.accounts[idx].storage = NULL;
                }
            }

        }
        )
    }

    /**
     * Generate the CPU instances of the touch state data.
     * @param[in] count The number of instances
    */
    __host__ static touch_state_data_t *get_cpu_instances(
        uint32_t count
    )
    {
        // allocate the instances and initialize them
        touch_state_data_t *cpu_instances = new touch_state_data_t[count];
        for (size_t idx = 0; idx < count; idx++)
        {
            cpu_instances[idx].touch_accounts.no_accounts = 0;
            cpu_instances[idx].touch_accounts.accounts = NULL;
            cpu_instances[idx].touch = NULL;
        }
        return cpu_instances;
    }

    /**
     * Free the CPU instances of the touch state data.
     * @param[in] cpu_instances The CPU instances
     * @param[in] count The number of instances
    */
    __host__ static void free_cpu_instances(
        touch_state_data_t *cpu_instances,
        uint32_t count
    )
    {
        for (size_t idx = 0; idx < count; idx++)
        {
            if (cpu_instances[idx].touch_accounts.accounts != NULL)
            {
                for (size_t jdx = 0; jdx < cpu_instances[idx].touch_accounts.no_accounts; jdx++)
                {
                    if (cpu_instances[idx].touch_accounts.accounts[jdx].bytecode != NULL)
                    {
                        delete[] cpu_instances[idx].touch_accounts.accounts[jdx].bytecode;
                        cpu_instances[idx].touch_accounts.accounts[jdx].bytecode = NULL;
                    }
                    if (cpu_instances[idx].touch_accounts.accounts[jdx].storage != NULL)
                    {
                        delete[] cpu_instances[idx].touch_accounts.accounts[jdx].storage;
                        cpu_instances[idx].touch_accounts.accounts[jdx].storage = NULL;
                    }
                }
                delete[] cpu_instances[idx].touch_accounts.accounts;
                cpu_instances[idx].touch_accounts.accounts = NULL;
            }
            if (cpu_instances[idx].touch != NULL)
            {
                delete[] cpu_instances[idx].touch;
                cpu_instances[idx].touch = NULL;
            }
        }
        delete[] cpu_instances;
    }

    /**
     * Generate the GPU instances of the touch state data from
     * the CPU counterparts.
     * @param[in] cpu_instances The CPU instances
     * @param[in] count The number of instances
    */
    __host__ static touch_state_data_t *get_gpu_instances_from_cpu_instances(
        touch_state_data_t *cpu_instances,
        uint32_t count
    )
    {

        touch_state_data_t *gpu_instances, *tmp_cpu_instances;
        // allocate the GPU memory for instances
        CUDA_CHECK(cudaMalloc(
            (void **)&(gpu_instances),
            count * sizeof(touch_state_data_t)
        ));
        // use a temporary CPU memory to allocate the GPU memory for the accounts
        // and storage
        tmp_cpu_instances = new touch_state_data_t[count];
        memcpy(
            tmp_cpu_instances,
            cpu_instances,
            count * sizeof(touch_state_data_t)
        );
        for (size_t idx = 0; idx < count; idx++)
        {
            // if the instance has accounts, allocate the GPU memory for them
            if (
                (tmp_cpu_instances[idx].touch_accounts.accounts != NULL) &&
                (tmp_cpu_instances[idx].touch_accounts.no_accounts > 0)
            )
            {
                account_t *tmp_accounts = new account_t[tmp_cpu_instances[idx].touch_accounts.no_accounts];
                memcpy(
                    tmp_accounts,
                    tmp_cpu_instances[idx].touch_accounts.accounts,
                    tmp_cpu_instances[idx].touch_accounts.no_accounts * sizeof(account_t)
                );
                for (size_t jdx = 0; jdx < tmp_cpu_instances[idx].touch_accounts.no_accounts; jdx++)
                {
                    // alocate the bytecode and storage if needed
                    if (
                        (tmp_cpu_instances[idx].touch_accounts.accounts[jdx].bytecode != NULL) &&
                        (tmp_cpu_instances[idx].touch_accounts.accounts[jdx].code_size > 0)
                    )
                    {
                        CUDA_CHECK(cudaMalloc(
                            (void **)&(tmp_accounts[jdx].bytecode),
                            tmp_cpu_instances[idx].touch_accounts.accounts[jdx].code_size * sizeof(uint8_t)
                        ));
                        CUDA_CHECK(cudaMemcpy(
                            tmp_accounts[jdx].bytecode,
                            tmp_accounts[jdx].bytecode,
                            tmp_cpu_instances[idx].touch_accounts.accounts[jdx].code_size * sizeof(uint8_t),
                            cudaMemcpyHostToDevice
                        ));
                    }
                    if (
                        (tmp_cpu_instances[idx].touch_accounts.accounts[jdx].storage != NULL) &&
                        (tmp_cpu_instances[idx].touch_accounts.accounts[jdx].storage_size > 0)
                    )
                    {
                        CUDA_CHECK(cudaMalloc(
                            (void **)&(tmp_accounts[jdx].storage),
                            tmp_cpu_instances[idx].touch_accounts.accounts[jdx].storage_size * sizeof(contract_storage_t)
                        ));
                        CUDA_CHECK(cudaMemcpy(
                            tmp_accounts[jdx].storage,
                            tmp_accounts[jdx].storage,
                            tmp_cpu_instances[idx].touch_accounts.accounts[jdx].storage_size * sizeof(contract_storage_t),
                            cudaMemcpyHostToDevice
                        ));
                    }
                }
                // allocate the GPU memory for the accounts and touch
                CUDA_CHECK(cudaMalloc(
                    (void **)&(tmp_cpu_instances[idx].touch_accounts.accounts),
                    tmp_cpu_instances[idx].touch_accounts.no_accounts * sizeof(account_t)
                ));
                CUDA_CHECK(cudaMemcpy(
                    tmp_cpu_instances[idx].touch_accounts.accounts,
                    tmp_accounts,
                    tmp_cpu_instances[idx].touch_accounts.no_accounts * sizeof(account_t),
                    cudaMemcpyHostToDevice
                ));
                CUDA_CHECK(cudaMalloc(
                    (void **)&(tmp_cpu_instances[idx].touch),
                    tmp_cpu_instances[idx].touch_accounts.no_accounts * sizeof(uint8_t)
                ));
                CUDA_CHECK(cudaMemcpy(
                    tmp_cpu_instances[idx].touch,
                    cpu_instances[idx].touch,
                    tmp_cpu_instances[idx].touch_accounts.no_accounts * sizeof(uint8_t),
                    cudaMemcpyHostToDevice
                ));
                delete[] tmp_accounts;
            }
        }

        CUDA_CHECK(cudaMemcpy(
            gpu_instances,
            tmp_cpu_instances,
            count * sizeof(touch_state_data_t),
            cudaMemcpyHostToDevice
        ));
        delete[] tmp_cpu_instances;
        return gpu_instances;
    }

    /**
     * Free the GPU instances of the touch state data.
     * @param[in] gpu_instances The GPU instances
     * @param[in] count The number of instances
    */
    __host__ static void free_gpu_instances(
        touch_state_data_t *gpu_instances,
        uint32_t count
    )
    {
        touch_state_data_t *tmp_cpu_instances = new touch_state_data_t[count];
        CUDA_CHECK(cudaMemcpy(
            tmp_cpu_instances,
            gpu_instances,
            count * sizeof(touch_state_data_t),
            cudaMemcpyDeviceToHost
        ));
        for (size_t idx = 0; idx < count; idx++)
        {
            if (
                (tmp_cpu_instances[idx].touch_accounts.accounts != NULL) &&
                (tmp_cpu_instances[idx].touch_accounts.no_accounts > 0)
            )
            {
                account_t *tmp_accounts = new account_t[tmp_cpu_instances[idx].touch_accounts.no_accounts];
                CUDA_CHECK(cudaMemcpy(
                    tmp_accounts,
                    tmp_cpu_instances[idx].touch_accounts.accounts,
                    tmp_cpu_instances[idx].touch_accounts.no_accounts * sizeof(account_t),
                    cudaMemcpyDeviceToHost
                ));
                for (size_t jdx = 0; jdx < tmp_cpu_instances[idx].touch_accounts.no_accounts; jdx++)
                {
                    if (
                        (tmp_accounts[jdx].bytecode != NULL) &&
                        (tmp_accounts[jdx].code_size > 0)
                    )
                    {
                        CUDA_CHECK(cudaFree(tmp_accounts[jdx].bytecode));
                        tmp_accounts[jdx].bytecode = NULL;
                    }
                    if (
                        (tmp_accounts[jdx].storage != NULL) &&
                        (tmp_accounts[jdx].storage_size > 0)
                    )
                    {
                        CUDA_CHECK(cudaFree(tmp_accounts[jdx].storage));
                        tmp_accounts[jdx].storage = NULL;
                    }
                }
                CUDA_CHECK(cudaFree(tmp_cpu_instances[idx].touch_accounts.accounts));
                tmp_cpu_instances[idx].touch_accounts.accounts = NULL;
                CUDA_CHECK(cudaFree(tmp_cpu_instances[idx].touch));
                tmp_cpu_instances[idx].touch = NULL;
                delete[] tmp_accounts;
                tmp_accounts = NULL;
            }
        }
        delete[] tmp_cpu_instances;
        tmp_cpu_instances = NULL;
        CUDA_CHECK(cudaFree(gpu_instances));
    }

    /**
     * Get the CPU instances of the touch state data from
     * the GPU counterparts.
     * @param[in] gpu_instances The GPU instances
     * @param[in] count The number of instances
    */
    __host__ static touch_state_data_t *get_cpu_instances_from_gpu_instances(
        touch_state_data_t *gpu_instances,
        uint32_t count
    )
    {
        // temporary instances
        touch_state_data_t *cpu_instances, *tmp_gpu_instances, *tmp_cpu_instances;
        // allocate the CPU memory for instances
        // and copy the initial details of the touch state
        // like the number of accounts and the pointer to the accounts
        // and their touch
        cpu_instances = new touch_state_data_t[count];
        CUDA_CHECK(cudaMemcpy(
            cpu_instances,
            gpu_instances,
            count * sizeof(touch_state_data_t),
            cudaMemcpyDeviceToHost
        ));
        // STEP 1: get the accounts details and read operations from GPU
        // use an axiliary emmory to alocate the necesarry memory on GPU which can be touch from
        // the host to copy the accounts details and read operations done on the accounts.
        tmp_cpu_instances = new touch_state_data_t[count];
        memcpy(
            tmp_cpu_instances,
            cpu_instances,
            count * sizeof(touch_state_data_t)
        );
        for (uint32_t idx = 0; idx < count; idx++)
        {
            // if the instance has accounts, allocate the GPU memory for them
            if (
                (cpu_instances[idx].touch_accounts.accounts != NULL) &&
                (cpu_instances[idx].touch_accounts.no_accounts > 0) &&
                (cpu_instances[idx].touch != NULL)
            )
            {
                CUDA_CHECK(cudaMalloc(
                    (void **)&(tmp_cpu_instances[idx].touch_accounts.accounts),
                    cpu_instances[idx].touch_accounts.no_accounts * sizeof(account_t)
                ));
                CUDA_CHECK(cudaMalloc(
                    (void **)&(tmp_cpu_instances[idx].touch),
                    cpu_instances[idx].touch_accounts.no_accounts * sizeof(uint8_t)
                ));
            }
        }
        CUDA_CHECK(cudaMalloc(
            (void **)&(tmp_gpu_instances),
            count * sizeof(touch_state_data_t)
        ));
        CUDA_CHECK(cudaMemcpy(
            tmp_gpu_instances,
            tmp_cpu_instances,
            count * sizeof(touch_state_data_t),
            cudaMemcpyHostToDevice
        ));
        delete[] tmp_cpu_instances;

        // run the first kernel which copy the accoutns details and read operations
        kernel_touch_state_S1<touch_state_data_t><<<count, 1>>>(tmp_gpu_instances, gpu_instances, count);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(gpu_instances));


        // STEP 2: get the accounts storage and bytecode from GPU
        gpu_instances = tmp_gpu_instances;

        CUDA_CHECK(cudaMemcpy(
            cpu_instances,
            gpu_instances,
            count * sizeof(touch_state_data_t),
            cudaMemcpyDeviceToHost
        ));
        tmp_cpu_instances = new touch_state_data_t[count];
        memcpy(
            tmp_cpu_instances,
            cpu_instances,
            count * sizeof(touch_state_data_t)
        );

        for (uint32_t idx = 0; idx < count; idx++)
        {
            // if the instance has accounts
            if (
                (cpu_instances[idx].touch_accounts.accounts != NULL) &&
                (cpu_instances[idx].touch_accounts.no_accounts > 0) &&
                (cpu_instances[idx].touch != NULL)
            )
            {
                account_t *tmp_accounts = new account_t[cpu_instances[idx].touch_accounts.no_accounts];
                account_t *aux_tmp_accounts = new account_t[cpu_instances[idx].touch_accounts.no_accounts];
                CUDA_CHECK(cudaMemcpy(
                    aux_tmp_accounts,
                    cpu_instances[idx].touch_accounts.accounts,
                    cpu_instances[idx].touch_accounts.no_accounts * sizeof(account_t),
                    cudaMemcpyDeviceToHost
                ));
                CUDA_CHECK(cudaMemcpy(
                    tmp_accounts,
                    cpu_instances[idx].touch_accounts.accounts,
                    cpu_instances[idx].touch_accounts.no_accounts * sizeof(account_t),
                    cudaMemcpyDeviceToHost
                ));
                // go through the accounts and allocate the memory for bytecode and storage
                for (size_t jdx = 0; jdx < cpu_instances[idx].touch_accounts.no_accounts; jdx++)
                {
                    if (
                        (aux_tmp_accounts[jdx].bytecode != NULL) &&
                        (aux_tmp_accounts[jdx].code_size > 0)
                    )
                    {
                        CUDA_CHECK(cudaMalloc(
                            (void **)&(tmp_accounts[jdx].bytecode),
                            aux_tmp_accounts[jdx].code_size * sizeof(uint8_t)
                        ));
                    }
                    if (
                        (aux_tmp_accounts[jdx].storage != NULL) &&
                        (aux_tmp_accounts[jdx].storage_size > 0)
                    )
                    {
                        CUDA_CHECK(cudaMalloc(
                            (void **)&(tmp_accounts[jdx].storage),
                            aux_tmp_accounts[jdx].storage_size * sizeof(contract_storage_t)
                        ));
                    }
                }
                CUDA_CHECK(cudaMalloc(
                    (void **)&(tmp_cpu_instances[idx].touch_accounts.accounts),
                    cpu_instances[idx].touch_accounts.no_accounts * sizeof(account_t)
                ));
                CUDA_CHECK(cudaMemcpy(
                    tmp_cpu_instances[idx].touch_accounts.accounts,
                    tmp_accounts,
                    cpu_instances[idx].touch_accounts.no_accounts * sizeof(account_t),
                    cudaMemcpyHostToDevice
                ));
                delete[] tmp_accounts;
                tmp_accounts = NULL;
                delete[] aux_tmp_accounts;
                aux_tmp_accounts = NULL;
            }
        }

        CUDA_CHECK(cudaMalloc(
            (void **)&(tmp_gpu_instances),
            count * sizeof(touch_state_data_t)
        ));
        CUDA_CHECK(cudaMemcpy(
            tmp_gpu_instances,
            tmp_cpu_instances,
            count * sizeof(touch_state_data_t),
            cudaMemcpyHostToDevice
        ));
        delete[] tmp_cpu_instances;

        // run the second kernel which copy the bytecode and storage
        kernel_touch_state_S2<touch_state_data_t><<<count, 1>>>(tmp_gpu_instances, gpu_instances, count);
        CUDA_CHECK(cudaDeviceSynchronize());

        // free the memory on GPU for the first kernel (accounts details)
        // the write operations can be kept because they don not have
        // more depth
        for (size_t idx = 0; idx < count; idx++)
        {
            if (
                (cpu_instances[idx].touch_accounts.accounts != NULL) &&
                (cpu_instances[idx].touch_accounts.no_accounts > 0) &&
                (cpu_instances[idx].touch != NULL)
            )
            {
                CUDA_CHECK(cudaFree(cpu_instances[idx].touch_accounts.accounts));
                //CUDA_CHECK(cudaFree(cpu_instances[idx].touch));
            }
        }

        CUDA_CHECK(cudaFree(gpu_instances));
        gpu_instances = tmp_gpu_instances;

        // STEP 3: copy the the entire touch state data from GPU to CPU
        CUDA_CHECK(cudaMemcpy(
            cpu_instances,
            gpu_instances,
            count * sizeof(touch_state_data_t),
            cudaMemcpyDeviceToHost
        ));
        tmp_cpu_instances = new touch_state_data_t[count];
        memcpy(
            tmp_cpu_instances,
            cpu_instances,
            count * sizeof(touch_state_data_t)
        );

        for (size_t idx = 0; idx < count; idx++)
        {
            // if the instance has accounts
            if (
                (tmp_cpu_instances[idx].touch_accounts.accounts != NULL) &&
                (tmp_cpu_instances[idx].touch_accounts.no_accounts > 0) &&
                (tmp_cpu_instances[idx].touch != NULL)
            )
            {
                account_t *tmp_accounts, *aux_tmp_accounts;
                tmp_accounts = new account_t[tmp_cpu_instances[idx].touch_accounts.no_accounts];
                aux_tmp_accounts = new account_t[tmp_cpu_instances[idx].touch_accounts.no_accounts];
                CUDA_CHECK(cudaMemcpy(
                    tmp_accounts,
                    tmp_cpu_instances[idx].touch_accounts.accounts,
                    cpu_instances[idx].touch_accounts.no_accounts * sizeof(account_t),
                    cudaMemcpyDeviceToHost
                ));
                CUDA_CHECK(cudaMemcpy(
                    aux_tmp_accounts,
                    tmp_cpu_instances[idx].touch_accounts.accounts,
                    cpu_instances[idx].touch_accounts.no_accounts * sizeof(account_t),
                    cudaMemcpyDeviceToHost
                ));
                // go through the accounts and copy the bytecode and the storage
                for (size_t jdx = 0; jdx < tmp_cpu_instances[idx].touch_accounts.no_accounts; jdx++)
                {
                    if (
                        (aux_tmp_accounts[jdx].bytecode != NULL) &&
                        (aux_tmp_accounts[jdx].code_size > 0)
                    )
                    {
                        tmp_accounts[jdx].bytecode = new uint8_t[aux_tmp_accounts[jdx].code_size * sizeof(uint8_t)];
                        CUDA_CHECK(cudaMemcpy(
                            tmp_accounts[jdx].bytecode,
                            aux_tmp_accounts[jdx].bytecode,
                            aux_tmp_accounts[jdx].code_size * sizeof(uint8_t),
                            cudaMemcpyDeviceToHost
                        ));
                    }
                    if (
                        (aux_tmp_accounts[jdx].storage != NULL) &&
                        (aux_tmp_accounts[jdx].storage_size > 0)
                    )
                    {
                        tmp_accounts[jdx].storage = new contract_storage_t[aux_tmp_accounts[jdx].storage_size];
                        CUDA_CHECK(cudaMemcpy(
                            tmp_accounts[jdx].storage,
                            aux_tmp_accounts[jdx].storage,
                            aux_tmp_accounts[jdx].storage_size * sizeof(contract_storage_t),
                            cudaMemcpyDeviceToHost));
                    }
                }
                tmp_cpu_instances[idx].touch_accounts.accounts = tmp_accounts;
                uint8_t *tmp_touch = new uint8_t[tmp_cpu_instances[idx].touch_accounts.no_accounts];
                CUDA_CHECK(cudaMemcpy(
                    tmp_touch,
                    tmp_cpu_instances[idx].touch,
                    tmp_cpu_instances[idx].touch_accounts.no_accounts * sizeof(uint8_t),
                    cudaMemcpyDeviceToHost
                ));
                tmp_cpu_instances[idx].touch = tmp_touch;
                delete[] aux_tmp_accounts;
                aux_tmp_accounts = NULL;
            }
        }

        free_gpu_instances(gpu_instances, count);
        memcpy(
            cpu_instances,
            tmp_cpu_instances,
            count * sizeof(touch_state_data_t)
        );
        delete[] tmp_cpu_instances;
        return cpu_instances;
    }

    /**
     * Print the touch state data structure
     * @param[in] arith The arithemtic instance
     * @param[in] touch_state_data The touch state data
    */
    __host__ __device__ __forceinline__ static void print_touch_state_data_t(
        arith_t &arith,
        touch_state_data_t &touch_state_data
    )
    {
        printf("no_accounts: %lu\n", touch_state_data.touch_accounts.no_accounts);
        for (size_t idx = 0; idx < touch_state_data.touch_accounts.no_accounts; idx++)
        {
            printf("accounts[%lu]:\n", idx);
            world_state_t::print_account_t(arith, touch_state_data.touch_accounts.accounts[idx]);
            printf("touch: %hhu\n", touch_state_data.touch[idx]);
        }
    }

    /**
     * Print the state.
    */
    __host__ __device__ __forceinline__ void print()
    {
        print_touch_state_data_t(_arith, *_content);
    }

    /**
     * Get json of the touch state data structure.
     * @param[in] arith The arithemtic instance
     * @param[in] touch_state_data The touch state data
     * @return The json of the touch state data
    */
    __host__ static cJSON *json_from_touch_state_data_t(
        arith_t &arith,
        touch_state_data_t &touch_state_data
    )
    {
        cJSON *state_json = NULL;
        cJSON *account_json = NULL;
        char *hex_string_ptr = new char[arith_t::BYTES * 2 + 3];
        size_t idx = 0;
        state_json = cJSON_CreateObject();
        for (idx = 0; idx < touch_state_data.touch_accounts.no_accounts; idx++)
        {
            account_json = world_state_t::json_from_account_t(
                arith,
                touch_state_data.touch_accounts.accounts[idx]);
            cJSON_AddNumberToObject(account_json, "touch", touch_state_data.touch[idx]);
            arith.hex_string_from_cgbn_memory(
                hex_string_ptr,
                touch_state_data.touch_accounts.accounts[idx].address,
                5
            );
            cJSON_AddItemToObject(state_json, hex_string_ptr, account_json);
        }
        delete[] hex_string_ptr;
        hex_string_ptr = NULL;
        return state_json;
    }

    /**
     * Get json of the state
     * @return The json of the state
    */
    __host__ __forceinline__ cJSON *json()
    {
        return json_from_touch_state_data_t(_arith, *_content);
    }


};


template <typename T>
__global__ void kernel_touch_state_S1(
    T *dst_instances,
    T *src_instances,
    uint32_t count)
{
    uint32_t instance = blockIdx.x * blockDim.x + threadIdx.x;
    typedef typename world_state_t::account_t account_t;

    if (instance >= count)
        return;

    if (
        (src_instances[instance].touch_accounts.accounts != NULL) &&
        (src_instances[instance].touch_accounts.no_accounts > 0) &&
        (src_instances[instance].touch != NULL)
    )
    {
        memcpy(
            dst_instances[instance].touch_accounts.accounts,
            src_instances[instance].touch_accounts.accounts,
            src_instances[instance].touch_accounts.no_accounts * sizeof(account_t)
        );
        delete[] src_instances[instance].touch_accounts.accounts;
        src_instances[instance].touch_accounts.accounts = NULL;
        memcpy(
            dst_instances[instance].touch,
            src_instances[instance].touch,
            src_instances[instance].touch_accounts.no_accounts * sizeof(uint8_t)
        );
        delete[] src_instances[instance].touch;
        src_instances[instance].touch = NULL;
    }
}

/**
 * Kernel to copy the bytecode and storage
 * between two instances of the touch state data.
 * @param[out] dst_instances The destination instances
 * @param[in] src_instances The source instances
 * @param[in] count The number of instances
*/
template <typename T>
__global__ void kernel_touch_state_S2(
    T *dst_instances,
    T *src_instances,
    uint32_t count)
{
    uint32_t instance = blockIdx.x * blockDim.x + threadIdx.x;
    typedef typename world_state_t::account_t account_t;
    typedef typename world_state_t::contract_storage_t contract_storage_t;

    if (instance >= count)
        return;

    if (
        (src_instances[instance].touch_accounts.accounts != NULL) &&
        (src_instances[instance].touch_accounts.no_accounts > 0)
    )
    {
        for (size_t idx = 0; idx < src_instances[instance].touch_accounts.no_accounts; idx++)
        {
            if (
                (src_instances[instance].touch_accounts.accounts[idx].bytecode != NULL) &&
                (src_instances[instance].touch_accounts.accounts[idx].code_size > 0)
            )
            {
                memcpy(
                    dst_instances[instance].touch_accounts.accounts[idx].bytecode,
                    src_instances[instance].touch_accounts.accounts[idx].bytecode,
                    src_instances[instance].touch_accounts.accounts[idx].code_size * sizeof(uint8_t)
                );
                delete[] src_instances[instance].touch_accounts.accounts[idx].bytecode;
                src_instances[instance].touch_accounts.accounts[idx].bytecode = NULL;
            }

            if (
                (src_instances[instance].touch_accounts.accounts[idx].storage != NULL) &&
                (src_instances[instance].touch_accounts.accounts[idx].storage_size > 0)
            )
            {
                memcpy(
                    dst_instances[instance].touch_accounts.accounts[idx].storage,
                    src_instances[instance].touch_accounts.accounts[idx].storage,
                    src_instances[instance].touch_accounts.accounts[idx].storage_size * sizeof(contract_storage_t)
                );
                delete[] src_instances[instance].touch_accounts.accounts[idx].storage;
                src_instances[instance].touch_accounts.accounts[idx].storage = NULL;
            }
        }
    }
}

#endif
