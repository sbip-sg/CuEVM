// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2024-06-20
// SPDX-License-Identifier: MIT
#include "../include/state/access_state.cuh"

namespace cuEVM {
    namespace state {

            __host__ __device__ int32_t AccessState::add_account(
                ArithEnv &arith,
                const bn_t &address,
                cuEVM::account::account_t* &account_ptr,
                const cuEVM::account::account_flags_t flag
            ) {
                cuEVM::account::account_t* tmp_account_ptr = nullptr;
                if(_world_state->get_account(arith, address, tmp_account_ptr)) {
                    return _state->add_duplicate_account(
                        account_ptr,
                        tmp_account_ptr,
                        flag);
                } else {
                    return _state->add_new_account(
                        arith,
                        address,
                        account_ptr,
                        flag);
                }
            }
            __host__ __device__ int32_t AccessState::get_account(
                ArithEnv &arith,
                const bn_t &address,
                cuEVM::account::account_t* &account_ptr,
                const cuEVM::account::account_flags_t flag = ACCOUNT_NONE_FLAG
            ) {
                if(_state->get_account(arith, address, account_ptr, flag)) {
                    return 1;
                } else {
                    return add_account(arith, address, account_ptr, flag);
                }
            }

            __host__ __device__ int32_t AccessState::get_value(
                ArithEnv &arith,
                const bn_t &address,
                const bn_t &key,
                bn_t &value
            ) {
                account::account_t* account_ptr = nullptr;
                if (_state->get_account(arith, address, account_ptr, ACCOUNT_STORAGE_FLAG) == 0) {
                    add_account(arith, address, account_ptr, ACCOUNT_STORAGE_FLAG);
                }
                if (account_ptr->get_storage_value(arith, key, value)) {
                    return 1;
                } else {
                    if (_world_state->get_value(arith, address, key, value) == 0) {
                        cgbn_set_ui32(arith.env, value, 0);
                    }
                    account_ptr->set_storage_value(arith, key, value);
                }
                return 1;
            }

            __host__ __device__ int32_t AccessState::poke_value(
                ArithEnv &arith,
                const bn_t &address,
                const bn_t &key,
                bn_t &value) const {
                account::account_t* account_ptr = nullptr;
                if (_state->get_account(arith, address, account_ptr, ACCOUNT_STORAGE_FLAG)) {
                    if (account_ptr->get_storage_value(arith, key, value)) {
                        return 1;
                    }
                }
                if (_world_state->get_value(arith, address, key, value) == 0) {
                    cgbn_set_ui32(arith.env, value, 0);
                    return 0;
                }
                return 1;
            }


            __host__ __device__ int32_t AccessState::is_warm_account(
                ArithEnv &arith,
                const bn_t &address) const {
                cuEVM::account::account_t* account_ptr = nullptr;
                if (_state->get_account(arith, address, account_ptr, ACCOUNT_NONE_FLAG)) {
                    return 1;
                }
                return 0;
            }

            __host__ __device__ int32_t AccessState::is_warm_key(
                ArithEnv &arith,
                const bn_t &address,
                const bn_t &key) const {
                cuEVM::account::account_t* account_ptr = nullptr;
                if (_state->get_account(arith, address, account_ptr, ACCOUNT_STORAGE_FLAG)) {
                    bn_t value;
                    return account_ptr->get_storage_value(arith, key, value);
                }
                return 0;
            }
    }
}
