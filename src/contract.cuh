#ifndef _GPU_GLOBAL_STORAGE_H_
#define _GPU_GLOBAL_STORAGE_H_

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
#include "error_codes.h"
#include "utils.h"


template<class params>
class gpu_fixed_global_storage_t {
    public:
    typedef typename arith_env_t<params>::bn_t      bn_t;
    typedef typename arith_env_t<params>::bn_wide_t bn_wide_t;

    typedef struct {
        cgbn_mem_t<params::BITS> key;
        cgbn_mem_t<params::BITS> value;
    } gpu_contract_storage_t;

    typedef struct {
        cgbn_mem_t<params::BITS> address;
        cgbn_mem_t<params::BITS> balance;
        size_t code_size;
        uint8_t bytecode[params::MAX_CODE_SIZE];
        size_t storage_size;
        gpu_contract_storage_t storage[params::MAX_STORAGE_SIZE];
    } gpu_contract_t;



    gpu_contract_t *_accounts;
    size_t  _no_accounts;
    arith_env_t<params>     _arith;
  
    //constructor
    __device__ __forceinline__ gpu_fixed_global_storage_t(arith_env_t<params> arith, gpu_contract_t *accounts, size_t no_accounts) : _arith(arith), _accounts(accounts), _no_accounts(no_accounts) {
    }

    __device__ __forceinline__ size_t get_account_idx_basic(bn_t &address, uint32_t &error_code) {
        bn_t local_address;
        for (size_t idx=0; idx<_no_accounts; idx++) {
            cgbn_load(_arith._env, local_address, &(_accounts[idx].address));
            if (cgbn_compare(_arith._env, local_address, address) == 0) {
                return idx;
            }
        }
        error_code = ERR_GLOBAL_STORAGE_INVALID_ADDRESS;
        return 0;
    }

    __device__ __forceinline__ size_t get_storage_idx_basic(size_t account_idx, bn_t &key, uint32_t &error_code) {
        bn_t local_key;
        for (size_t idx=0; idx<_accounts[account_idx].storage_size; idx++) {
            cgbn_load(_arith._env, local_key, &(_accounts[account_idx].storage[idx].key));
            if (cgbn_compare(_arith._env, local_key, key) == 0) {
                return idx;
            }
        }
        error_code = ERR_GLOBAL_STORAGE_INVALID_KEY;
        return 0;
    }

    __device__ __forceinline__ size_t get_storage_idx_basic(gpu_contract_t *account, bn_t &key, uint32_t &error_code) {
        bn_t local_key;
        for (size_t idx=0; idx<account->storage_size; idx++) {
            cgbn_load(_arith._env, local_key, &(account->storage[idx].key));
            if (cgbn_compare(_arith._env, local_key, key) == 0) {
                return idx;
            }
        }
        error_code = ERR_GLOBAL_STORAGE_INVALID_KEY;
        return 0;
    }

    __device__ __forceinline__ gpu_contract_t *get_account(bn_t &address, uint32_t &error_code) {
        size_t account_idx = get_account_idx_basic(address, error_code);
        return &(_accounts[account_idx]);
    }

    __device__ __forceinline__ void get_value(bn_t &address, bn_t &key, bn_t &value, uint32_t &error_code) {
        gpu_contract_t *account = get_account(address, error_code);
        size_t storage_idx = get_storage_idx_basic(account, key, error_code);
        cgbn_load(_arith._env, value, &(account->storage[storage_idx].value));
    }

    __host__ static gpu_contract_t *generate_global_storage(size_t count) {
        gpu_contract_t *accounts = (gpu_contract_t *)malloc(count*sizeof(gpu_contract_t));
        mpz_t address, balance, key, value;
        char size_t_string[9]; // 8 bytes + '\0'
        mpz_init(address);
        mpz_init(balance);
        mpz_init(key);
        mpz_init(value);
        for (size_t idx=0; idx<count; idx++) {
            //mpz_set_ui(address, idx);
            snprintf (size_t_string, 9, "%lx", idx);
            //mpz_set_str(address, "0x123456789abcdef", 16);
            mpz_set_str(address, size_t_string, 16);
            from_mpz(accounts[idx].address._limbs, params::BITS/32, address);
            mpz_set_ui(balance, idx*2);
            from_mpz(accounts[idx].balance._limbs, params::BITS/32, balance);
            accounts[idx].code_size = idx % params::MAX_CODE_SIZE;
            accounts[idx].storage_size = idx % params::MAX_STORAGE_SIZE;
            for (size_t jdx=0; jdx<accounts[idx].code_size; jdx++) {
                accounts[idx].bytecode[jdx] = idx % 256;
            }
            for (size_t jdx=0; jdx<accounts[idx].storage_size; jdx++) {
                mpz_set_ui(key, jdx);
                mpz_set_ui(value, jdx+2);
                from_mpz(accounts[idx].storage[jdx].key._limbs, params::BITS/32, key);
                from_mpz(accounts[idx].storage[jdx].value._limbs, params::BITS/32, value);
            }
        }
        mpz_clear(address);
        mpz_clear(balance);
        mpz_clear(key);
        mpz_clear(value);
        return accounts;
    }

    __host__ static void free_global_storage(gpu_contract_t *accounts) {
        free(accounts);
    }

    __host__ static gpu_contract_t *generate_gpu_global_storage(gpu_contract_t *cpu_accounts, size_t count) {
        gpu_contract_t *gpu_accounts;
        cudaMalloc((void **)&gpu_accounts, count*sizeof(gpu_contract_t));
        cudaMemcpy(gpu_accounts, cpu_accounts, count*sizeof(gpu_contract_t), cudaMemcpyHostToDevice);
        return gpu_accounts;
    }

    __host__ static void free_gpu_global_storage(gpu_contract_t *gpu_accounts) {
        cudaFree(gpu_accounts);
    }
};




template<class params>
class gpu_global_storage_t {
    public:
    typedef typename arith_env_t<params>::bn_t      bn_t;
    typedef typename arith_env_t<params>::bn_wide_t bn_wide_t;

    typedef struct {
        cgbn_mem_t<params::BITS> key;
        cgbn_mem_t<params::BITS> value;
    } gpu_contract_storage_t;

    typedef struct {
        cgbn_mem_t<params::BITS> address;
        cgbn_mem_t<params::BITS> balance;
        size_t code_size;
        size_t storage_size;
        uint8_t *bytecode;
        gpu_contract_storage_t *storage;
    } gpu_contract_t;



    gpu_contract_t *_accounts;
    size_t  _no_accounts;
    arith_env_t<params>     _arith;
  
    //constructor
    __device__ __forceinline__ gpu_global_storage_t(arith_env_t<params> arith, gpu_contract_t *accounts, size_t no_accounts) : _arith(arith), _accounts(accounts), _no_accounts(no_accounts) {
    }

    __device__ __forceinline__ size_t get_account_idx_basic(bn_t &address, uint32_t &error_code) {
        bn_t local_address;
        for (size_t idx=0; idx<_no_accounts; idx++) {
            cgbn_load(_arith._env, local_address, &(_accounts[idx].address));
            if (cgbn_compare(_arith._env, local_address, address) == 0) {
                return idx;
            }
        }
        error_code = ERR_GLOBAL_STORAGE_INVALID_ADDRESS;
        return 0;
    }

    __device__ __forceinline__ size_t get_storage_idx_basic(size_t account_idx, bn_t &key, uint32_t &error_code) {
        bn_t local_key;
        for (size_t idx=0; idx<_accounts[account_idx].storage_size; idx++) {
            cgbn_load(_arith._env, local_key, &(_accounts[account_idx].storage[idx].key));
            if (cgbn_compare(_arith._env, local_key, key) == 0) {
                return idx;
            }
        }
        error_code = ERR_GLOBAL_STORAGE_INVALID_KEY;
        return 0;
    }

    __device__ __forceinline__ size_t get_storage_idx_basic(gpu_contract_t *account, bn_t &key, uint32_t &error_code) {
        bn_t local_key;
        for (size_t idx=0; idx<account->storage_size; idx++) {
            cgbn_load(_arith._env, local_key, &(account->storage[idx].key));
            if (cgbn_compare(_arith._env, local_key, key) == 0) {
                return idx;
            }
        }
        error_code = ERR_GLOBAL_STORAGE_INVALID_KEY;
        return 0;
    }

    __device__ __forceinline__ gpu_contract_t *get_account(bn_t &address, uint32_t &error_code) {
        size_t account_idx = get_account_idx_basic(address, error_code);
        return &(_accounts[account_idx]);
    }

    __device__ __forceinline__ void get_value(bn_t &address, bn_t &key, bn_t &value, uint32_t &error_code) {
        gpu_contract_t *account = get_account(address, error_code);
        size_t storage_idx = get_storage_idx_basic(account, key, error_code);
        cgbn_load(_arith._env, value, &(account->storage[storage_idx].value));
    }

    __host__ static gpu_contract_t *generate_global_storage(size_t count) {
        gpu_contract_t *accounts = (gpu_contract_t *)malloc(count*sizeof(gpu_contract_t));
        mpz_t address, balance, key, value;
        char size_t_string[params::BITS/4+1]; // 8 bytes + '\0'
        mpz_init(address);
        mpz_init(balance);
        mpz_init(key);
        mpz_init(value);
        for (size_t idx=0; idx<count; idx++) {
            //mpz_set_ui(address, idx);
            snprintf (size_t_string, params::BITS/4+1, "%lx", idx);
            //mpz_set_str(address, "0x123456789abcdef", 16);
            mpz_set_str(address, size_t_string, 16);
            from_mpz(accounts[idx].address._limbs, params::BITS/32, address);
            mpz_set_ui(balance, idx*2);
            from_mpz(accounts[idx].balance._limbs, params::BITS/32, balance);
            accounts[idx].code_size = idx % params::MAX_CODE_SIZE;
            accounts[idx].storage_size = idx % params::MAX_STORAGE_SIZE;
            accounts[idx].bytecode = (uint8_t *)malloc(accounts[idx].code_size*sizeof(uint8_t));
            for (size_t jdx=0; jdx<accounts[idx].code_size; jdx++) {
                accounts[idx].bytecode[jdx] = idx % 256;
            }
            accounts[idx].storage = (gpu_contract_storage_t *)malloc(accounts[idx].storage_size*sizeof(gpu_contract_storage_t));
            for (size_t jdx=0; jdx<accounts[idx].storage_size; jdx++) {
                mpz_set_ui(key, jdx);
                mpz_set_ui(value, jdx+2);
                from_mpz(accounts[idx].storage[jdx].key._limbs, params::BITS/32, key);
                from_mpz(accounts[idx].storage[jdx].value._limbs, params::BITS/32, value);
            }
        }
        mpz_clear(address);
        mpz_clear(balance);
        mpz_clear(key);
        mpz_clear(value);
        return accounts;
    }

    __host__ static void free_global_storage(gpu_contract_t *accounts, size_t count) {
        for (size_t idx=0; idx<count; idx++) {
            free(accounts[idx].bytecode);
            free(accounts[idx].storage);
        }
        free(accounts);
    }

    __host__ static gpu_contract_t *generate_gpu_global_storage(gpu_contract_t *cpu_accounts, size_t count) {
        gpu_contract_t *gpu_accounts, *tmp_cpu_accounts;
        tmp_cpu_accounts = (gpu_contract_t *)malloc(count*sizeof(gpu_contract_t));
        memcpy(tmp_cpu_accounts, cpu_accounts, count*sizeof(gpu_contract_t));
        for (size_t idx=0; idx<count; idx++) {
            cudaMalloc((void **)&(tmp_cpu_accounts[idx].bytecode), tmp_cpu_accounts[idx].code_size*sizeof(uint8_t));
            cudaMemcpy(tmp_cpu_accounts[idx].bytecode, cpu_accounts[idx].bytecode, tmp_cpu_accounts[idx].code_size*sizeof(uint8_t), cudaMemcpyHostToDevice);
            cudaMalloc((void **)&(tmp_cpu_accounts[idx].storage), tmp_cpu_accounts[idx].storage_size*sizeof(gpu_contract_storage_t));
            cudaMemcpy(tmp_cpu_accounts[idx].storage, cpu_accounts[idx].storage, tmp_cpu_accounts[idx].storage_size*sizeof(gpu_contract_storage_t), cudaMemcpyHostToDevice);
        }
        cudaMalloc((void **)&gpu_accounts, count*sizeof(gpu_contract_t));
        cudaMemcpy(gpu_accounts, tmp_cpu_accounts, count*sizeof(gpu_contract_t), cudaMemcpyHostToDevice);
        free(tmp_cpu_accounts);
        return gpu_accounts;
    }

    __host__ static void free_gpu_global_storage(gpu_contract_t *gpu_accounts, size_t count) {
        gpu_contract_t *tmp_cpu_accounts;
        tmp_cpu_accounts = (gpu_contract_t *)malloc(count*sizeof(gpu_contract_t));
        cudaMemcpy(tmp_cpu_accounts, gpu_accounts, count*sizeof(gpu_contract_t), cudaMemcpyDeviceToHost);
        for (size_t idx=0; idx<count; idx++) {
            cudaFree(tmp_cpu_accounts[idx].bytecode);
            cudaFree(tmp_cpu_accounts[idx].storage);
        }
        free(tmp_cpu_accounts);
        cudaFree(gpu_accounts);
    }

    __host__ static gpu_contract_storage_t *copy_gpu_to_cpu(gpu_contract_storage_t *gpu_accounts, size_t count) {
        gpu_contract_storage_t *cpu_accounts;
        cpu_accounts = (gpu_contract_storage_t *)malloc(count*sizeof(gpu_contract_storage_t));
        cudaMemcpy(cpu_accounts, gpu_accounts, count*sizeof(gpu_contract_storage_t), cudaMemcpyDeviceToHost);
        uint8_t *tmp_bytecode;
        gpu_contract_storage_t *tmp_storage;
        for (size_t idx=0; idx<count; idx++) {
            tmp_bytecode = cpu_accounts[idx].bytecode;
            cpu_accounts[idx].bytecode = (uint8_t *)malloc(cpu_accounts[idx].code_size*sizeof(uint8_t));
            cudaMemcpy(cpu_accounts[idx].bytecode, tmp_bytecode, cpu_accounts[idx].code_size*sizeof(uint8_t), cudaMemcpyDeviceToHost);
            tmp_storage = cpu_accounts[idx].storage;
            cpu_accounts[idx].storage = (gpu_contract_storage_t *)malloc(cpu_accounts[idx].storage_size*sizeof(gpu_contract_storage_t));
            cudaMemcpy(cpu_accounts[idx].storage, tmp_storage, cpu_accounts[idx].storage_size*sizeof(gpu_contract_storage_t), cudaMemcpyDeviceToHost);
        }
        return cpu_accounts;
    }
};

#endif