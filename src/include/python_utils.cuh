#pragma once
#ifdef IGNORE
#include <Python.h>

#include "utils.h"

namespace python_utils{
    state_data_t* world_state_t(
        arith_t arith,
        const PyObject *test
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
        return _content;
    }
}

#endif