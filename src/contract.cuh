#ifndef _STATE_T_H_
#define _STATE_T_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <gmp.h>
#ifndef __CGBN_H__
#define __CGBN_H__
#include <cgbn/cgbn.h>
#endif
#include "arith.cuh"
#include "utils.h"

template<class params>
class state_t {
    public:
    typedef arith_env_t<params>                     arith_t;
    typedef typename arith_t::bn_t                  bn_t;
    typedef cgbn_mem_t<params::BITS>                evm_word_t;

    typedef struct {
        evm_word_t key;
        evm_word_t value;
    } contract_storage_t;

    typedef struct alignas(32) {
        evm_word_t address;
        evm_word_t balance;
        evm_word_t nonce;
        size_t code_size;
        size_t storage_size;
        size_t changes; // 0 - no, | 1 bytecode, | 2 balance, | 4 nonce, | 8 storage
        uint8_t *bytecode;
        contract_storage_t *storage;
    } contract_t;

    typedef struct {
        contract_t *contracts;
        size_t no_contracts;
    } state_data_t;

    state_data_t            *_content;
    arith_t     _arith;
  
    //constructor
    __host__ __device__ __forceinline__ state_t(arith_t arith, state_data_t *content) : _arith(arith), _content(content) {
    }

    __host__ static state_data_t *get_global_state(const cJSON *test) {
        const cJSON *state_json = NULL;
        const cJSON *contract_json = NULL;
        const cJSON *balance_json = NULL;
        const cJSON *code_json = NULL;
        const cJSON *nonce_json = NULL;
        const cJSON *storage_json = NULL;
        const cJSON *key_value_json = NULL;
        mpz_t address, balance, nonce, key, value;
        mpz_init(address);
        mpz_init(balance);
        mpz_init(nonce);
        mpz_init(key);
        mpz_init(value);
        char *hex_string=NULL;
        size_t idx=0, jdx=0;

        state_data_t *state=(state_data_t *)malloc(sizeof(state_data_t));

        state_json = cJSON_GetObjectItemCaseSensitive(test, "pre");

        state->no_contracts = cJSON_GetArraySize(state_json);
        if (state->no_contracts == 0) {
            state->contracts = NULL;
            return state;
        }
        state->contracts = (contract_t *)malloc(state->no_contracts*sizeof(contract_t));

        cJSON_ArrayForEach(contract_json, state_json)
        {
            // set the address
            hex_string = contract_json->string;
            adjusted_length(&hex_string);
            mpz_set_str(address, hex_string, 16);
            from_mpz(state->contracts[idx].address._limbs, params::BITS/32, address);

            // set the balance
            balance_json = cJSON_GetObjectItemCaseSensitive(contract_json, "balance");
            hex_string = balance_json->valuestring;
            adjusted_length(&hex_string);
            mpz_set_str(balance, hex_string, 16);
            from_mpz(state->contracts[idx].balance._limbs, params::BITS/32, balance);

            // set the nonce
            nonce_json = cJSON_GetObjectItemCaseSensitive(contract_json, "nonce");
            hex_string = nonce_json->valuestring;
            adjusted_length(&hex_string);
            mpz_set_str(nonce, hex_string, 16);
            from_mpz(state->contracts[idx].nonce._limbs, params::BITS/32, nonce);

            // set the code
            code_json = cJSON_GetObjectItemCaseSensitive(contract_json, "code");
            hex_string = code_json->valuestring;
            state->contracts[idx].code_size = adjusted_length(&hex_string);
            if (state->contracts[idx].code_size > 0) {
                state->contracts[idx].bytecode = (uint8_t *)malloc(state->contracts[idx].code_size*sizeof(uint8_t));
                hex_to_bytes(hex_string, state->contracts[idx].bytecode, 2 * state->contracts[idx].code_size);
            } else {
                state->contracts[idx].bytecode = NULL;
            }
            state->contracts[idx].changes = 0;

            // set the storage
            storage_json = cJSON_GetObjectItemCaseSensitive(contract_json, "storage");
            state->contracts[idx].storage_size = cJSON_GetArraySize(storage_json);
            if (state->contracts[idx].storage_size > 0) {
                state->contracts[idx].storage = (contract_storage_t *)malloc(state->contracts[idx].storage_size*sizeof(contract_storage_t));
                jdx=0;
                cJSON_ArrayForEach(key_value_json, storage_json)
                {
                    // set the key
                    hex_string = key_value_json->string;
                    adjusted_length(&hex_string);
                    mpz_set_str(key, hex_string, 16);
                    from_mpz(state->contracts[idx].storage[jdx].key._limbs, params::BITS/32, key);

                    // set the value
                    hex_string = key_value_json->valuestring;
                    adjusted_length(&hex_string);
                    mpz_set_str(value, hex_string, 16);
                    from_mpz(state->contracts[idx].storage[jdx].value._limbs, params::BITS/32, value);

                    jdx++;
                }
            } else {
                state->contracts[idx].storage = NULL;
            }
            idx++;
        }
        mpz_clear(address);
        mpz_clear(balance);
        mpz_clear(nonce);
        mpz_clear(key);
        mpz_clear(value);
        return state;
    }

    // host constructor from json with cpu memory
    __host__ state_t(arith_t arith, const cJSON *test) : _arith(arith) {
        _content=get_global_state(test);
    }

    // runable by only on thread
    __host__ __device__ __forceinline__ static contract_t *get_empty_account() {
        contract_t *account = NULL;
        account = (contract_t *) malloc(sizeof(contract_t));
        memset(account, 0, sizeof(contract_t));
        return account;
    }


    __host__ __device__ __forceinline__ size_t get_account_idx_basic(bn_t &address, uint32_t &error_code) {
        bn_t local_address;
        for (size_t idx=0; idx<_content->no_contracts; idx++) {
            cgbn_load(_arith._env, local_address, &(_content->contracts[idx].address));
            if (cgbn_compare(_arith._env, local_address, address) == 0) {
                return idx;
            }
        }
        error_code = ERR_STATE_INVALID_ADDRESS;
        return 0;
    }

    __host__ __device__ __forceinline__ contract_t *get_local_account(bn_t &address, uint32_t &error_code) {
        size_t account_idx = get_account_idx_basic(address, error_code);
        return &(_content->contracts[account_idx]);
    }
    

    __host__ __device__ __forceinline__ size_t get_storage_idx_basic(contract_t *account, bn_t &key, uint32_t &error_code) {
        bn_t local_key;
        for (size_t idx=0; idx<account->storage_size; idx++) {
            cgbn_load(_arith._env, local_key, &(account->storage[idx].key));
            if (cgbn_compare(_arith._env, local_key, key) == 0) {
                return idx;
            }
        }
        error_code = ERR_STATE_INVALID_KEY;
        return 0;
    }

    __host__ __device__ __forceinline__ void get_local_value(bn_t &address, bn_t &key, bn_t &value, uint32_t &error_code) {
        contract_t *account = get_local_account(address, error_code);
        if (error_code != ERR_SUCCESS) {
            cgbn_set_ui32(_arith._env, value, 0); // if account does not exist return 0
        } else {
            size_t storage_idx = get_storage_idx_basic(account, key, error_code);
            if (error_code != ERR_SUCCESS) {
                cgbn_set_ui32(_arith._env, value, 0); // if storage does not exist return 0
            } else {
                cgbn_load(_arith._env, value, &(account->storage[storage_idx].value));
            }
        }
    }


    // runable only by one thread
    __host__ __device__ __forceinline__ static contract_t *duplicate_contract(
        contract_t *account,
        uint32_t type=0 // 0 - all // 1 - without storage
    ) {
        printf("ajunge 3-1\n");
        contract_t *new_account = (contract_t *) malloc(sizeof(contract_t));
        printf("ajunge 3-1-1\n");
        printf("ajunge 3-1-1 %p\n", account);
        memcpy(new_account, account, sizeof(contract_t));
        printf("ajunge 3-1-2\n");
        if (account->code_size > 0) {
            new_account->bytecode = (uint8_t *) malloc(account->code_size*sizeof(uint8_t));
            memcpy(new_account->bytecode, account->bytecode, account->code_size*sizeof(uint8_t));
        }
        printf("ajunge 3-2\n");
        if (type == 1) {
            new_account->storage_size=0;
            new_account->storage=NULL;
        } else {
            if (account->storage_size > 0) {
                new_account->storage = (contract_storage_t *) malloc(account->storage_size*sizeof(contract_storage_t));
                memcpy(new_account->storage, account->storage, account->storage_size*sizeof(contract_storage_t));
            }
        }
        printf("ajunge 3-3\n");
        return new_account;
    }

    __host__ __device__ __forceinline__ void set_local_account(bn_t &address, contract_t *account, uint32_t type=0, uint32_t empty=0) {
        uint32_t tmp_error_code = ERR_SUCCESS;
        uint32_t account_idx = get_account_idx_basic(address, tmp_error_code);
        printf("ajunge 2-1\n");
        if (tmp_error_code == ERR_STATE_INVALID_ADDRESS) {
            account_idx = _content->no_contracts;
        }
        SHARED_MEMORY contract_t dup_account;

        ONE_THREAD_PER_INSTANCE(
            //contract_t *dup_account;
            if (empty == 1) {
                // we have to add an empty account
                //dup_account = get_empty_account();
                printf("ajunge 5-1\n");
            } else {
                //dup_account = duplicate_contract(account, type);
                //dup_account = (contract_t *) malloc(sizeof(contract_t));
                printf("ajunge 3-1-1\n");
                printf("ajunge 3-1-1 %p\n", account);
                memcpy(&dup_account, account, sizeof(contract_t));
                printf("ajunge 3-1-2\n");
                if (account->code_size > 0) {
                    dup_account.bytecode = (uint8_t *) malloc(account->code_size*sizeof(uint8_t));
                    memcpy(dup_account.bytecode, account->bytecode, account->code_size*sizeof(uint8_t));
                }
                printf("ajunge 3-2\n");
                if (type == 1) {
                    dup_account.storage_size=0;
                    dup_account.storage=NULL;
                } else {
                    if (account->storage_size > 0) {
                        dup_account.storage = (contract_storage_t *) malloc(account->storage_size*sizeof(contract_storage_t));
                        memcpy(dup_account.storage, account->storage, account->storage_size*sizeof(contract_storage_t));
                    }
                }
            }
            printf("ajunge 2-2\n");
            if (tmp_error_code == ERR_STATE_INVALID_ADDRESS) {
                // contract does not exist needs to be added
                //account_idx = _content->no_contracts;
                contract_t *tmp_contracts = (contract_t *) malloc((_content->no_contracts+1)*sizeof(contract_t));
                memcpy(tmp_contracts, _content->contracts, _content->no_contracts*sizeof(contract_t));
                if (_content->no_contracts > 0) {
                    free(_content->contracts);
                }
                _content->contracts = tmp_contracts;
                _content->no_contracts++;
            } else {
                if ( (_content->contracts[account_idx].code_size > 0) && _content->contracts[account_idx].bytecode != NULL) {
                    free(_content->contracts[account_idx].bytecode);
                }
                if ( (_content->contracts[account_idx].storage_size > 0) && _content->contracts[account_idx].storage != NULL) {
                    free(_content->contracts[account_idx].storage);
                }
            }
            memcpy(&(_content->contracts[account_idx]), &dup_account, sizeof(contract_t));
            //free(dup_account);
        )
        if (empty == 1) {
            printf("ajunge 2-3\n");
            cgbn_store(_arith._env, &(_content->contracts[account_idx].address), address);
        }
        printf("ajunge 2-3\n");
    }

    
    __host__ __device__ __forceinline__ void set_local_value(bn_t &address, bn_t &key, bn_t &value) {
        uint32_t tmp_error_code = ERR_SUCCESS;
        contract_t *account = get_local_account(address, tmp_error_code);
        printf("ajunge 1\n");
        if (tmp_error_code == ERR_STATE_INVALID_ADDRESS) {
            printf("ajunge 2\n");
            set_local_account(address, account, 1, 1); //without storage // empy account
            printf("ajunge 2\n");
            printf("ajunge 2\n");
        }
        printf("ajunge 1\n");
        account = get_local_account(address, tmp_error_code);
        size_t storage_idx = get_storage_idx_basic(account, key, tmp_error_code);
        if (tmp_error_code == ERR_STATE_INVALID_KEY) {
            // add the extra storage key
            storage_idx = account->storage_size;
            ONE_THREAD_PER_INSTANCE(
                contract_storage_t *tmp_storage = (contract_storage_t *) malloc((account->storage_size+1)*sizeof(contract_storage_t));
                if (account->storage_size > 0) {
                    memcpy(tmp_storage, account->storage, account->storage_size*sizeof(contract_storage_t));
                    free(account->storage);
                }
                account->storage = tmp_storage;
                account->storage_size++;
            )
            cgbn_store(_arith._env, &(account->storage[storage_idx].key), key);
            tmp_error_code = ERR_SUCCESS;
        }
        printf("ajunge 1\n");
        cgbn_store(_arith._env, &(account->storage[storage_idx].value), value);
    }

    __host__ __device__ __forceinline__ contract_t *get_account(
        bn_t &address,
        state_t &global,
        state_t &access_list,
        state_t &parents,
        bn_t &gas_cost,
        uint32_t call_type // 1 - balance // 2 - nonce // 4 - code
    ) {
        uint32_t tmp_error_code = ERR_SUCCESS;
        size_t  warm_address=1; // we consider that it is a warm address
        contract_t *account = get_local_account(address, tmp_error_code);

        if (tmp_error_code == ERR_STATE_INVALID_ADDRESS) {
            // account does not exist in the current environment
            // we have too look up
            tmp_error_code = ERR_SUCCESS;
            account = parents.get_local_account(address, tmp_error_code);
            if (tmp_error_code == ERR_STATE_INVALID_ADDRESS) {
                tmp_error_code = ERR_SUCCESS;
                account = access_list.get_local_account(address, tmp_error_code);
                if (tmp_error_code == ERR_STATE_INVALID_ADDRESS) { // we have to go to global state
                    warm_address=0;
                    tmp_error_code = ERR_SUCCESS;
                    account = global.get_local_account(address, tmp_error_code);
                    if (tmp_error_code == ERR_STATE_INVALID_ADDRESS) {
                        // account does not exist in the global state
                        // we have to create it
                        printf("ajunge 0\n");
                        access_list.set_local_account(address, account, 1, 1); //without storage empty account
                    } else {
                        // account exists in the global state
                        // we have to add it to the access list
                        printf ("il ia din global\n");
                        access_list.set_local_account(address, account, 1); //without storage
                    }
                }
            }
        }
        contract_t *access_account = access_list.get_local_account(address, tmp_error_code);
        
        if (tmp_error_code == ERR_STATE_INVALID_ADDRESS) {
            // account does not exist in the access list
            // we have to add it
            // REAL PROBLEM HERE no suposed to be here
            access_list.set_local_account(address, account, 1); //without storage
            printf("DAT de BELEA\n");
        }
        access_account->changes |= call_type;
        if (warm_address == 1) {
            cgbn_add_ui32(_arith._env, gas_cost, gas_cost, 100);
        } else {
            cgbn_add_ui32(_arith._env, gas_cost, gas_cost, 2600);
        }

        return account;
        
    }

    __host__ __device__ __forceinline__ void get_account_balance(
        bn_t &address,
        bn_t &value,
        state_t &global,
        state_t &access_list,
        state_t &parents,
        bn_t &gas_cost
    ) {
        contract_t *account = get_account(address, global, access_list, parents, gas_cost, 0);
        cgbn_load(_arith._env, value, &(account->balance));
    }

    __host__ __device__ __forceinline__ void get_account_nonce(
        bn_t &address,
        bn_t &value,
        state_t &global,
        state_t &access_list,
        state_t &parents,
        bn_t &gas_cost
    ) {
        contract_t *account = get_account(address, global, access_list, parents, gas_cost, 1);
        cgbn_load(_arith._env, value, &(account->nonce));
    }

    __host__ __device__ __forceinline__ size_t get_account_code_size(
        bn_t &address,
        state_t &global,
        state_t &access_list,
        state_t &parents,
        bn_t &gas_cost
    ) {
        contract_t *account = get_account(address, global, access_list, parents, gas_cost, 2);
        return account->code_size;
    }

    __host__ __device__ __forceinline__ uint8_t *get_account_code(
        bn_t &address,
        state_t &global,
        state_t &access_list,
        state_t &parents,
        bn_t &gas_cost
    ) {
        contract_t *account = get_account(address, global, access_list, parents, gas_cost, 2);
        return account->bytecode;
    }

    

    __host__ __device__ __forceinline__ void set_account_balance(
        bn_t &address,
        bn_t &value,
        state_t &global,
        state_t &access_list,
        state_t &parents,
        bn_t &gas_cost
    ) {
        uint32_t tmp_error_code = ERR_SUCCESS;
        contract_t *account = get_local_account(address, tmp_error_code);
        if (tmp_error_code == ERR_STATE_INVALID_ADDRESS) {
            account = get_account(address, global, access_list, parents, gas_cost, 1);
            set_local_account(address, account, 1); //without storage
            account = get_local_account(address, tmp_error_code);
        }
        cgbn_store(_arith._env, &(account->balance), value);
    }

    __host__ __device__ __forceinline__ void set_account_nonce(
        bn_t &address,
        bn_t &value,
        state_t &global,
        state_t &access_list,
        state_t &parents,
        bn_t &gas_cost
    ) {
        uint32_t tmp_error_code = ERR_SUCCESS;
        contract_t *account = get_local_account(address, tmp_error_code);
        if (tmp_error_code == ERR_STATE_INVALID_ADDRESS) {
            account = get_account(address, global, access_list, parents, gas_cost, 2);
            set_local_account(address, account, 1); //without storage
            account = get_local_account(address, tmp_error_code);
        }
        cgbn_store(_arith._env, &(account->nonce), value);
    }

    __host__ __device__ __forceinline__ void set_account_code(
        bn_t &address,
        uint8_t *bytecode,
        size_t code_size,
        state_t &global,
        state_t &access_list,
        state_t &parents,
        bn_t &gas_cost
    ) {
        uint32_t tmp_error_code = ERR_SUCCESS;
        contract_t *account = get_local_account(address, tmp_error_code);
        if (tmp_error_code == ERR_STATE_INVALID_ADDRESS) {
            account = get_account(address, global, access_list, parents, gas_cost, 4);
            set_local_account(address, account, 1); //without storage
            account = get_local_account(address, tmp_error_code);
        }
        ONE_THREAD_PER_INSTANCE(
            if (account->code_size > 0) {
                free(account->bytecode);
            }
            account->bytecode = (uint8_t *) malloc(code_size*sizeof(uint8_t));
            memcpy(account->bytecode, bytecode, code_size*sizeof(uint8_t));
            account->code_size = code_size;
        )
    }
    
    __host__ __device__ __forceinline__ void get_value(
        bn_t &address,
        bn_t &key,
        bn_t &value,
        state_t &global,
        state_t &access_list,
        state_t &parents,
        bn_t &gas_cost
    ) {
        uint32_t tmp_error_code = ERR_SUCCESS;
        size_t  warm_key=1; // we consider that it is a warm key
        get_local_value(address, key, value, tmp_error_code);

        if (tmp_error_code != ERR_SUCCESS) {
            // account does not exist in the current environment
            // we have too look up
            tmp_error_code = ERR_SUCCESS;
            parents.get_local_value(address, key, value, tmp_error_code);
            if (tmp_error_code != ERR_SUCCESS) {
                tmp_error_code = ERR_SUCCESS;
                access_list.get_local_value(address, key, value, tmp_error_code);
                if (tmp_error_code != ERR_SUCCESS) { // we have to go to global state
                    warm_key=0;
                    tmp_error_code = ERR_SUCCESS;
                    global.get_local_value(address, key, value, tmp_error_code);
                    // set in access list
                    access_list.set_local_value(address, key, value);
                }
            }
        }
        if (warm_key == 1) {
            cgbn_add_ui32(_arith._env, gas_cost, gas_cost, 100);
        } else {
            cgbn_add_ui32(_arith._env, gas_cost, gas_cost, 2100);
        }
    }

    __host__ __device__ __forceinline__ void set_value(
        bn_t &address,
        bn_t &key,
        bn_t &value,
        state_t &global,
        state_t &access_list,
        state_t &parents,
        bn_t &gas_cost,
        bn_t &gas_refund
    ) {
        uint32_t tmp_error_code = ERR_SUCCESS;
        size_t  warm_key=1; // we consider that it is a warm key
        bn_t original_value;
        bn_t current_value;
        bn_t dummy_gas_cost;
        access_list.get_local_value(address, key, original_value, tmp_error_code);
        get_value(address, key, current_value, global, access_list, parents, dummy_gas_cost);
        printf("ajunge 2\n");
        if (tmp_error_code != ERR_SUCCESS) {
            warm_key=0;
            access_list.get_local_value(address, key, original_value, tmp_error_code);
        }
        printf("ajunge 3\n");
        set_local_value(address, key, value);
        printf("ajunge 3\n");

        if (cgbn_compare(_arith._env, value, current_value) == 0) {
            cgbn_add_ui32(_arith._env, gas_cost, gas_cost, 100);
        } else {
            if(cgbn_compare(_arith._env, current_value, original_value) == 0) {
                if(cgbn_compare_ui32(_arith._env, original_value, 0) == 0) {
                    cgbn_add_ui32(_arith._env, gas_cost, gas_cost, 20000);
                } else {
                    cgbn_add_ui32(_arith._env, gas_cost, gas_cost, 2900);
                }
            } else {
                cgbn_add_ui32(_arith._env, gas_cost, gas_cost, 100);
            }
        }
        
        printf("ajunge 4\n");

        // gas refund
        if (cgbn_compare(_arith._env, value, current_value) != 0) {
            if (cgbn_compare(_arith._env, current_value, original_value) == 0) {
                if ( (cgbn_compare_ui32(_arith._env, original_value, 0) != 0) &&
                     (cgbn_compare_ui32(_arith._env, value, 0) == 0) ) {
                    cgbn_add_ui32(_arith._env, gas_refund, gas_refund, 4800);
                }
            } else {
                if (cgbn_compare(_arith._env, value, original_value) == 0) {
                    if (cgbn_compare_ui32(_arith._env, original_value, 0) == 0) {
                        cgbn_add_ui32(_arith._env, gas_refund, gas_refund, 19900);
                    } else {
                        if (warm_key == 1) {
                            cgbn_add_ui32(_arith._env, gas_refund, gas_refund, 2800);
                        } else {
                            cgbn_add_ui32(_arith._env, gas_refund, gas_refund, 4900);
                        }
                    }
                }

                if(cgbn_compare_ui32(_arith._env, original_value, 0) != 0) {
                    if (cgbn_compare_ui32(_arith._env, current_value, 0) == 0) {
                        //cgbn_sub_ui32(_arith._env, gas_refund, gas_refund, 4800);
                        // better to add to gas cost TODO: look later
                        cgbn_add_ui32(_arith._env, gas_cost, gas_cost, 4800);
                    }
                    if (cgbn_compare_ui32(_arith._env, value, 0) == 0) {
                        cgbn_add_ui32(_arith._env, gas_refund, gas_refund, 4800);
                    }
                }
            }
        }
        printf("ajunge 5\n");
        if (warm_key == 1) {
            cgbn_add_ui32(_arith._env, gas_cost, gas_cost, 0);
        } else {
            cgbn_add_ui32(_arith._env, gas_cost, gas_cost, 2100);
        }
        printf("ajunge 6\n");
    }

    __host__ __device__ __forceinline__ void copy_to_state_data_t(state_data_t *that) {
        that->no_contracts = _content->no_contracts;
        that->contracts = _content->contracts;
    }

    __host__ __device__ __forceinline__ void copy_from_state_t(const state_t &that) {
        size_t idx, jdx;
        uint32_t error_code;
        contract_t *account;
        bn_t address, key, value;
        bn_t balance, nonce;
        for (idx=0; idx<that._content->no_contracts; idx++) {
            cgbn_load(_arith._env, address, &(that._content->contracts[idx].address));
            account = get_local_account(address, error_code);
            if (error_code == ERR_STATE_INVALID_ADDRESS) {
                // contract does not exist needs to be added
                error_code = ERR_SUCCESS;
                printf("ajunge y\n");
                set_local_account(address, &(that._content->contracts[idx]), 0); //with storage
                printf("ajunge y-2\n");
            } else {
                for (jdx=0; jdx<that._content->contracts[idx].storage_size; jdx++) {
                    cgbn_load(_arith._env, key, &(that._content->contracts[idx].storage[jdx].key));
                    cgbn_load(_arith._env, value, &(that._content->contracts[idx].storage[jdx].value));
                    set_local_value(address, key, value);
                    error_code = ERR_SUCCESS;
                }
                // override the other variables
                cgbn_load(_arith._env, balance, &(that._content->contracts[idx].balance));
                cgbn_load(_arith._env, nonce, &(that._content->contracts[idx].nonce));
                cgbn_store(_arith._env, &(account->balance), balance);
                cgbn_store(_arith._env, &(account->nonce), nonce);
            }
        }
    }

    __host__ __device__ static void free_instance(state_data_t *instance) {
        ONE_THREAD_PER_INSTANCE(
            if (instance != NULL) {
                if (instance->no_contracts > 0) {
                    for (size_t idx=0; idx<instance->no_contracts; idx++) {
                        if (instance->contracts[idx].code_size > 0) {
                            free(instance->contracts[idx].bytecode);
                        }
                        if (instance->contracts[idx].storage_size > 0) {
                            free(instance->contracts[idx].storage);
                        }
                    }

                    free(instance->contracts);
                }
                free(instance);
            }
        )
    }

    __host__ __device__ __forceinline__ void free_memory() {
        free_instance(_content);
    }

    __host__ static state_data_t *from_cpu_to_gpu(state_data_t *instance) {
        state_data_t *gpu_instance, *tmp_cpu_instance;
        tmp_cpu_instance=(state_data_t *)malloc(sizeof(state_data_t));
        tmp_cpu_instance->no_contracts = instance->no_contracts;
        if (tmp_cpu_instance->no_contracts > 0) {
            contract_t *tmp_cpu_contracts;
            tmp_cpu_contracts = (contract_t *)malloc(instance->no_contracts*sizeof(contract_t));
            memcpy(tmp_cpu_contracts, instance->contracts, instance->no_contracts*sizeof(contract_t));
            for (size_t idx=0; idx<instance->no_contracts; idx++) {
                if (tmp_cpu_contracts[idx].bytecode != NULL) {
                    cudaMalloc((void **)&(tmp_cpu_contracts[idx].bytecode), tmp_cpu_contracts[idx].code_size*sizeof(uint8_t));
                    cudaMemcpy(tmp_cpu_contracts[idx].bytecode, instance->contracts[idx].bytecode, tmp_cpu_contracts[idx].code_size*sizeof(uint8_t), cudaMemcpyHostToDevice);
                }
                if (tmp_cpu_contracts[idx].storage != NULL) {
                    cudaMalloc((void **)&(tmp_cpu_contracts[idx].storage), tmp_cpu_contracts[idx].storage_size*sizeof(contract_storage_t));
                    cudaMemcpy(tmp_cpu_contracts[idx].storage, instance->contracts[idx].storage, tmp_cpu_contracts[idx].storage_size*sizeof(contract_storage_t), cudaMemcpyHostToDevice);
                }
            }
            cudaMalloc((void **)&tmp_cpu_instance->contracts, instance->no_contracts*sizeof(contract_t));
            cudaMemcpy(tmp_cpu_instance->contracts, tmp_cpu_contracts, instance->no_contracts*sizeof(contract_t), cudaMemcpyHostToDevice);
            free(tmp_cpu_contracts);
        } else {
            tmp_cpu_instance->contracts = NULL;
        }
        cudaMalloc((void **)&gpu_instance, sizeof(state_data_t));
        cudaMemcpy(gpu_instance, tmp_cpu_instance, sizeof(state_data_t), cudaMemcpyHostToDevice);
        free(tmp_cpu_instance);
        return gpu_instance;
    }

    __host__ state_data_t *to_gpu() {
        state_data_t *gpu_state=from_cpu_to_gpu(_content);
        return gpu_state;
    }

    __host__ static void free_gpu_memory(state_data_t *gpu_state) {
        state_data_t *tmp_cpu_state;
        tmp_cpu_state=(state_data_t *)malloc(sizeof(state_data_t));
        cudaMemcpy(tmp_cpu_state, gpu_state, sizeof(state_data_t), cudaMemcpyDeviceToHost);
        if (tmp_cpu_state->contracts != NULL) {
            contract_t *tmp_cpu_contracts;
            tmp_cpu_contracts = (contract_t *)malloc(tmp_cpu_state->no_contracts*sizeof(contract_t));
            cudaMemcpy(tmp_cpu_contracts, tmp_cpu_state->contracts, tmp_cpu_state->no_contracts*sizeof(contract_t), cudaMemcpyDeviceToHost);
            for (size_t idx=0; idx<tmp_cpu_state->no_contracts; idx++) {
                if (tmp_cpu_contracts[idx].bytecode != NULL)
                    cudaFree(tmp_cpu_contracts[idx].bytecode);
                if (tmp_cpu_contracts[idx].storage != NULL)
                    cudaFree(tmp_cpu_contracts[idx].storage);
            }
            free(tmp_cpu_contracts);
            cudaFree(tmp_cpu_state->contracts);
        }
        free(tmp_cpu_state);
        cudaFree(gpu_state);
    }

    __host__ __device__ void print() {
        printf("no_contracts: %lu\n", _content->no_contracts);
        for (size_t idx=0; idx<_content->no_contracts; idx++) {
            printf("contract %lu\n", idx);
            printf("address: ");
            print_bn<params>(_content->contracts[idx].address);
            printf("\n");
            printf("balance: ");
            print_bn<params>(_content->contracts[idx].balance);
            printf("\n");
            printf("nonce: ");
            print_bn<params>(_content->contracts[idx].nonce);
            printf("\n");
            printf("code_size: %lu\n", _content->contracts[idx].code_size);
            if (_content->contracts[idx].code_size > 0) {
                printf("code: ");
                print_bytes(_content->contracts[idx].bytecode, _content->contracts[idx].code_size);
                printf("\n");
            }
            printf("storage_size: %lu\n", _content->contracts[idx].storage_size);
            for (size_t jdx=0; jdx<_content->contracts[idx].storage_size; jdx++) {
                printf("storage[%lu].key: ", jdx);
                print_bn<params>(_content->contracts[idx].storage[jdx].key);
                printf("\n");
                printf("storage[%lu].value: ", jdx);
                print_bn<params>(_content->contracts[idx].storage[jdx].value);
                printf("\n");
            }
        }
    }

    __host__ static state_data_t *get_local_states(uint32_t count) {
        state_data_t *states=(state_data_t *)malloc(count*sizeof(state_data_t));
        for (size_t idx=0; idx<count; idx++) {
            states[idx].no_contracts = 0;
            states[idx].contracts = NULL;
        }
        return states;
    }

    __host__ static void free_local_states(state_data_t *states, uint32_t count) {
        for (size_t idx=0; idx<count; idx++) {
            if ( (states[idx].contracts != NULL) && (states[idx].no_contracts > 0) ) {
                for (size_t jdx=0; jdx<states[idx].no_contracts; jdx++) {
                    if( (states[idx].contracts[jdx].bytecode != NULL) && (states[idx].contracts[jdx].code_size > 0) )
                        free(states[idx].contracts[jdx].bytecode);
                    if( (states[idx].contracts[jdx].storage != NULL) && (states[idx].contracts[jdx].storage_size > 0) )
                        free(states[idx].contracts[jdx].storage);
                }
                free(states[idx].contracts);
            }
        }
        free(states);
    }

    __host__ static state_data_t *get_gpu_local_states(state_data_t *cpu_local_states, uint32_t count) {
        state_data_t *gpu_local_states, *tmp_cpu_local_states;
        tmp_cpu_local_states = (state_data_t *)malloc(count*sizeof(state_data_t));
        memcpy(tmp_cpu_local_states, cpu_local_states, count*sizeof(state_data_t));
        for (size_t idx=0; idx<count; idx++) {
            if (tmp_cpu_local_states[idx].contracts != NULL) {
                contract_t *tmp_cpu_contracts;
                tmp_cpu_contracts = (contract_t *)malloc(tmp_cpu_local_states[idx].no_contracts*sizeof(contract_t));
                memcpy(tmp_cpu_contracts, tmp_cpu_local_states[idx].contracts, tmp_cpu_local_states[idx].no_contracts*sizeof(contract_t));
                for (size_t jdx=0; jdx<tmp_cpu_local_states[idx].no_contracts; jdx++) {
                    if (tmp_cpu_contracts[jdx].bytecode != NULL) {
                        cudaMalloc((void **)&(tmp_cpu_contracts[jdx].bytecode), tmp_cpu_contracts[jdx].code_size*sizeof(uint8_t));
                        cudaMemcpy(tmp_cpu_contracts[jdx].bytecode, cpu_local_states[idx].contracts[jdx].bytecode, tmp_cpu_contracts[jdx].code_size*sizeof(uint8_t), cudaMemcpyHostToDevice);
                    }
                    if (tmp_cpu_contracts[jdx].storage != NULL) {
                        cudaMalloc((void **)&(tmp_cpu_contracts[jdx].storage), tmp_cpu_contracts[jdx].storage_size*sizeof(contract_storage_t));
                        cudaMemcpy(tmp_cpu_contracts[jdx].storage, cpu_local_states[idx].contracts[jdx].storage, tmp_cpu_contracts[jdx].storage_size*sizeof(contract_storage_t), cudaMemcpyHostToDevice);
                    }
                }
                cudaMalloc((void **)&(tmp_cpu_local_states[idx].contracts), tmp_cpu_local_states[idx].no_contracts*sizeof(contract_t));
                cudaMemcpy(tmp_cpu_local_states[idx].contracts, tmp_cpu_contracts, tmp_cpu_local_states[idx].no_contracts*sizeof(contract_t), cudaMemcpyHostToDevice);
                free(tmp_cpu_contracts);
            }
        }
        cudaMalloc((void **)&gpu_local_states, count*sizeof(state_data_t));
        cudaMemcpy(gpu_local_states, tmp_cpu_local_states, count*sizeof(state_data_t), cudaMemcpyHostToDevice);
        free(tmp_cpu_local_states);
        return gpu_local_states;
    }

    __host__ static void free_gpu_local_states(state_data_t *gpu_local_states, uint32_t count) {
        state_data_t *tmp_cpu_local_states;
        tmp_cpu_local_states = (state_data_t *)malloc(count*sizeof(state_data_t));
        cudaMemcpy(tmp_cpu_local_states, gpu_local_states, count*sizeof(state_data_t), cudaMemcpyDeviceToHost);
        for (size_t idx=0; idx<count; idx++) {
            if (tmp_cpu_local_states[idx].contracts != NULL) {
                contract_t *tmp_cpu_contracts;
                tmp_cpu_contracts = (contract_t *)malloc(tmp_cpu_local_states[idx].no_contracts*sizeof(contract_t));
                cudaMemcpy(tmp_cpu_contracts, tmp_cpu_local_states[idx].contracts, tmp_cpu_local_states[idx].no_contracts*sizeof(contract_t), cudaMemcpyDeviceToHost);
                for (size_t jdx=0; jdx<tmp_cpu_local_states[idx].no_contracts; jdx++) {
                    if (tmp_cpu_contracts[jdx].bytecode != NULL) {
                        cudaFree(tmp_cpu_contracts[jdx].bytecode);
                    }
                    if (tmp_cpu_contracts[jdx].storage != NULL) {
                        cudaFree(tmp_cpu_contracts[jdx].storage);
                    }
                }
                cudaFree(tmp_cpu_local_states[idx].contracts);
                free(tmp_cpu_contracts);
            }
        }
        free(tmp_cpu_local_states);
        cudaFree(gpu_local_states);
    }

    __host__ __device__ static void print_local_states(state_data_t *states, uint32_t count) {
        for (size_t idx=0; idx<count; idx++) {
            printf("local state %lu\n", idx);
            printf("no_contracts: %lu\n", states[idx].no_contracts);
            for (size_t jdx=0; jdx<states[idx].no_contracts; jdx++) {
                printf("contract %lu\n", jdx);
                printf("address: ");
                print_bn<params>(states[idx].contracts[jdx].address);
                printf("\n");
                printf("balance: ");
                print_bn<params>(states[idx].contracts[jdx].balance);
                printf("\n");
                printf("nonce: ");
                print_bn<params>(states[idx].contracts[jdx].nonce);
                printf("\n");
                printf("code_size: %lu\n", states[idx].contracts[jdx].code_size);
                printf("code: ");
                print_bytes(states[idx].contracts[jdx].bytecode, states[idx].contracts[jdx].code_size);
                printf("\n");
                printf("storage_size: %lu\n", states[idx].contracts[jdx].storage_size);
                for (size_t kdx=0; kdx<states[idx].contracts[jdx].storage_size; kdx++) {
                    printf("storage[%lu].key: ", kdx);
                    print_bn<params>(states[idx].contracts[jdx].storage[kdx].key);
                    printf("\n");
                    printf("storage[%lu].value: ", kdx);
                    print_bn<params>(states[idx].contracts[jdx].storage[kdx].value);
                    printf("\n");
                }
            }
        }
    }

    __host__ static state_data_t *get_local_states_from_gpu(state_data_t *gpu_local_states, uint32_t count) {
        // STATE 1.1 I can only see the contracts values and number of contracts
        state_data_t *cpu_local_states;
        cpu_local_states = (state_data_t *)malloc(count*sizeof(state_data_t));
        cudaMemcpy(cpu_local_states, gpu_local_states, count*sizeof(state_data_t), cudaMemcpyDeviceToHost);
        // STATE 1.2 I can alocate the contracts array
        state_data_t *new_gpu_local_states, *tmp_cpu_local_states;
        tmp_cpu_local_states = (state_data_t *)malloc(count*sizeof(state_data_t));
        memcpy(tmp_cpu_local_states, cpu_local_states, count*sizeof(state_data_t));
        for (size_t idx=0; idx<count; idx++) {
            if (tmp_cpu_local_states[idx].contracts != NULL) {
                cudaMalloc((void **)&(tmp_cpu_local_states[idx].contracts), tmp_cpu_local_states[idx].no_contracts*sizeof(contract_t));
            }
        }
        cudaMalloc((void **)&new_gpu_local_states, count*sizeof(state_data_t));
        cudaMemcpy(new_gpu_local_states, tmp_cpu_local_states, count*sizeof(state_data_t), cudaMemcpyHostToDevice);
        free(tmp_cpu_local_states);
        // STATE 1.3 call the kernel
        kernel_get_local_states_S1<params><<<1, count>>>(new_gpu_local_states, gpu_local_states, count);
        CUDA_CHECK(cudaDeviceSynchronize());
        // STATE 1.4 free unnecasry memory
        cudaFree(gpu_local_states);
        gpu_local_states = new_gpu_local_states;

        // STATE 2.1 copy the contracts array
        cudaMemcpy(cpu_local_states, gpu_local_states, count*sizeof(state_data_t), cudaMemcpyDeviceToHost);
        // STATE 2.2 allocate the contracts array
        tmp_cpu_local_states = (state_data_t *)malloc(count*sizeof(state_data_t));
        memcpy(tmp_cpu_local_states, cpu_local_states, count*sizeof(state_data_t));
        for (size_t idx=0; idx<count; idx++) {
            if ( (tmp_cpu_local_states[idx].contracts != NULL) && (tmp_cpu_local_states[idx].no_contracts > 0) ) {
                contract_t *tmp_cpu_contracts;
                tmp_cpu_contracts = (contract_t *)malloc(tmp_cpu_local_states[idx].no_contracts*sizeof(contract_t));
                cudaMemcpy(tmp_cpu_contracts, tmp_cpu_local_states[idx].contracts, tmp_cpu_local_states[idx].no_contracts*sizeof(contract_t), cudaMemcpyDeviceToHost);
                for (size_t jdx=0; jdx<tmp_cpu_local_states[idx].no_contracts; jdx++) {
                    cudaMalloc((void **)&(tmp_cpu_contracts[jdx].bytecode), tmp_cpu_contracts[jdx].code_size*sizeof(uint8_t));
                    cudaMalloc((void **)&(tmp_cpu_contracts[jdx].storage), tmp_cpu_contracts[jdx].storage_size*sizeof(contract_storage_t));
                }
                cudaMalloc((void **)&(tmp_cpu_local_states[idx].contracts), tmp_cpu_local_states[idx].no_contracts*sizeof(contract_t));
                cudaMemcpy(tmp_cpu_local_states[idx].contracts, tmp_cpu_contracts, tmp_cpu_local_states[idx].no_contracts*sizeof(contract_t), cudaMemcpyHostToDevice);
                free(tmp_cpu_contracts);
            }
        }
        cudaMalloc((void **)&new_gpu_local_states, count*sizeof(state_data_t));
        cudaMemcpy(new_gpu_local_states, tmp_cpu_local_states, count*sizeof(state_data_t), cudaMemcpyHostToDevice);
        free(tmp_cpu_local_states);
        // STATE 2.3 call the kernel
        kernel_get_local_states_S2<params><<<1, count>>>(new_gpu_local_states, gpu_local_states, count);
        CUDA_CHECK(cudaDeviceSynchronize());
        // STATE 2.4 free unnecasry memory
        for (size_t idx=0; idx<count; idx++) {
            if (cpu_local_states[idx].contracts != NULL) {
                cudaFree(cpu_local_states[idx].contracts);
            }
        }
        cudaFree(gpu_local_states);
        gpu_local_states = new_gpu_local_states;

        // STATE 3.1 copy the contracts array
        cudaMemcpy(cpu_local_states, gpu_local_states, count*sizeof(state_data_t), cudaMemcpyDeviceToHost);
        // STATE 3.2 allocate the contracts array
        tmp_cpu_local_states = (state_data_t *)malloc(count*sizeof(state_data_t));
        memcpy(tmp_cpu_local_states, cpu_local_states, count*sizeof(state_data_t));
        for (size_t idx=0; idx<count; idx++) {
            if (tmp_cpu_local_states[idx].contracts != NULL) {
                contract_t *tmp_cpu_contracts, *aux_tmp_cpu_contract;
                tmp_cpu_contracts = (contract_t *)malloc(tmp_cpu_local_states[idx].no_contracts*sizeof(contract_t));
                aux_tmp_cpu_contract = (contract_t *)malloc(tmp_cpu_local_states[idx].no_contracts*sizeof(contract_t));
                cudaMemcpy(tmp_cpu_contracts, tmp_cpu_local_states[idx].contracts, tmp_cpu_local_states[idx].no_contracts*sizeof(contract_t), cudaMemcpyDeviceToHost);
                cudaMemcpy(aux_tmp_cpu_contract, tmp_cpu_local_states[idx].contracts, tmp_cpu_local_states[idx].no_contracts*sizeof(contract_t), cudaMemcpyDeviceToHost);
                for (size_t jdx=0; jdx<tmp_cpu_local_states[idx].no_contracts; jdx++) {
                    tmp_cpu_contracts[jdx].bytecode = (uint8_t *)malloc(tmp_cpu_contracts[jdx].code_size*sizeof(uint8_t));
                    cudaMemcpy(tmp_cpu_contracts[jdx].bytecode, aux_tmp_cpu_contract[jdx].bytecode, tmp_cpu_contracts[jdx].code_size*sizeof(uint8_t), cudaMemcpyDeviceToHost);
                    tmp_cpu_contracts[jdx].storage = (contract_storage_t *)malloc(tmp_cpu_contracts[jdx].storage_size*sizeof(contract_storage_t));
                    cudaMemcpy(tmp_cpu_contracts[jdx].storage, aux_tmp_cpu_contract[jdx].storage, tmp_cpu_contracts[jdx].storage_size*sizeof(contract_storage_t), cudaMemcpyDeviceToHost);
                }
                free(aux_tmp_cpu_contract);
                tmp_cpu_local_states[idx].contracts = tmp_cpu_contracts;
            }
        }
        // STATE 3.3 free gpu local states
        free_gpu_local_states(gpu_local_states, count);
        // STATE 3.4 copy to cpu final
        memcpy(cpu_local_states, tmp_cpu_local_states, count*sizeof(state_data_t));
        free(tmp_cpu_local_states);
        return cpu_local_states;
    }

    

    __host__ cJSON *to_json() {
        cJSON *state_json = NULL;
        cJSON *contract_json = NULL;
        cJSON *storage_json = NULL;
        char *bytes_string=NULL;
        char *hex_string_ptr=(char *) malloc(sizeof(char) * ((params::BITS/32)*8+3));
        char *value_hex_string_ptr=(char *) malloc(sizeof(char) * ((params::BITS/32)*8+3));
        size_t idx=0, jdx=0;
        state_json = cJSON_CreateObject();
        for (idx=0; idx<_content->no_contracts; idx++) {
            contract_json = cJSON_CreateObject();
            // set the address
            _arith.from_cgbn_memory_to_hex(_content->contracts[idx].address, hex_string_ptr, 5);
            cJSON_AddItemToObject(state_json, hex_string_ptr, contract_json);
            // set the balance
            _arith.from_cgbn_memory_to_hex(_content->contracts[idx].balance, hex_string_ptr);
            cJSON_AddStringToObject(contract_json, "balance", hex_string_ptr);
            // set the nonce
            _arith.from_cgbn_memory_to_hex(_content->contracts[idx].nonce, hex_string_ptr);
            cJSON_AddStringToObject(contract_json, "nonce", hex_string_ptr);
            // set the code
            if (_content->contracts[idx].code_size > 0) {
                bytes_string = bytes_to_hex(_content->contracts[idx].bytecode, _content->contracts[idx].code_size);
                cJSON_AddStringToObject(contract_json, "code", bytes_string);
                free(bytes_string);
            } else {
                cJSON_AddStringToObject(contract_json, "code", "0x");
            }
            // set if the code was modified
            if (_content->contracts[idx].changes != 0) {
                cJSON_AddStringToObject(contract_json, "changes", "true");
            } else {
                cJSON_AddStringToObject(contract_json, "changes", "false");
            }
            // set the storage
            storage_json = cJSON_CreateObject();
            cJSON_AddItemToObject(contract_json, "storage", storage_json);
            if (_content->contracts[idx].storage_size > 0) {
                for (jdx=0; jdx<_content->contracts[idx].storage_size; jdx++) {
                    _arith.from_cgbn_memory_to_hex(_content->contracts[idx].storage[jdx].key, hex_string_ptr);
                    _arith.from_cgbn_memory_to_hex(_content->contracts[idx].storage[jdx].value, value_hex_string_ptr);
                    cJSON_AddStringToObject(storage_json, hex_string_ptr, value_hex_string_ptr);
                }
            }
        }
        free(hex_string_ptr);
        free(value_hex_string_ptr);
        return state_json;
    }
};

template<class params>
__global__ void kernel_get_local_states_S1(typename state_t<params>::state_data_t *dst_instances, typename state_t<params>::state_data_t *src_instances, uint32_t instance_count) {
    uint32_t instance=blockIdx.x*blockDim.x + threadIdx.x;
    typedef typename state_t<params>::contract_t contract_t;

    if(instance>=instance_count)
        return;

    if ( (src_instances[instance].contracts != NULL) && (src_instances[instance].no_contracts > 0) ) {
        memcpy(dst_instances[instance].contracts, src_instances[instance].contracts, src_instances[instance].no_contracts*sizeof(contract_t));
        free(src_instances[instance].contracts);
    }
}


template<class params>
__global__ void kernel_get_local_states_S2(typename state_t<params>::state_data_t *dst_instances, typename state_t<params>::state_data_t *src_instances, uint32_t instance_count) {
    uint32_t instance=blockIdx.x*blockDim.x + threadIdx.x;
    typedef typename state_t<params>::contract_t contract_t;
    typedef typename state_t<params>::contract_storage_t contract_storage_t;

    if(instance>=instance_count)
        return;

    if (src_instances[instance].contracts != NULL) {
        for(size_t idx=0; idx<src_instances[instance].no_contracts; idx++) {
            if ( (src_instances[instance].contracts[idx].bytecode != NULL) && (src_instances[instance].contracts[idx].code_size > 0) ) {
                memcpy(dst_instances[instance].contracts[idx].bytecode, src_instances[instance].contracts[idx].bytecode, src_instances[instance].contracts[idx].code_size*sizeof(uint8_t));
                free(src_instances[instance].contracts[idx].bytecode);
            }
            if ( (src_instances[instance].contracts[idx].storage != NULL) && (src_instances[instance].contracts[idx].storage_size > 0) ) {
                memcpy(dst_instances[instance].contracts[idx].storage, src_instances[instance].contracts[idx].storage, src_instances[instance].contracts[idx].storage_size*sizeof(contract_storage_t));
                free(src_instances[instance].contracts[idx].storage);
            }
        }
    }
}

#endif