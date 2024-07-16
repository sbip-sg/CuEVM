// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2024-06-20
// SPDX-License-Identifier: MIT
#include "../include/state/access_state.cuh"
#include "../include/utils/error_codes.cuh"

namespace cuEVM {
    namespace state {
        __host__ __device__ int32_t AccessState::add_account(
            ArithEnv &arith,
            const bn_t &address,
            cuEVM::account::account_t* &account_ptr,
            const cuEVM::account::account_flags_t flag
        ) {
            cuEVM::account::account_t* tmp_account_ptr = nullptr;
            return (
                _world_state->get_account(arith, address, tmp_account_ptr) ?
                _state->add_new_account(
                    arith,
                    address,
                    account_ptr,
                    flag) :
                _state->add_duplicate_account(
                    account_ptr,
                    tmp_account_ptr,
                    flag)
            );
        }
            __host__ __device__ int32_t AccessState::get_account(
                ArithEnv &arith,
                const bn_t &address,
                cuEVM::account::account_t* &account_ptr,
                const cuEVM::account::account_flags_t flag = ACCOUNT_NONE_FLAG
            ) {
                return (
                    _state->get_account(arith, address, account_ptr, flag) ?
                    add_account(arith, address, account_ptr, flag) :
                    ERROR_SUCCESS
                );
            }

            __host__ __device__ int32_t AccessState::get_value(
                ArithEnv &arith,
                const bn_t &address,
                const bn_t &key,
                bn_t &value
            ) {
                account::account_t* account_ptr = nullptr;
                int32_t error_code = (
                    _state->get_account(arith, address, account_ptr, ACCOUNT_STORAGE_FLAG) ?
                    add_account(arith, address, account_ptr, ACCOUNT_STORAGE_FLAG) :
                    ERROR_SUCCESS
                );
                error_code = (error_code || 
                    (
                        account_ptr->get_storage_value(arith, key, value) ?
                        (
                            ([&]() -> int32_t {
                                _world_state->get_value(arith, address, key, value);
                                return account_ptr->set_storage_value(arith, key, value);
                            })()
                        ) : ERROR_SUCCESS
                    )
                );
                return error_code;
            }

            __host__ __device__ int32_t AccessState::poke_value(
                ArithEnv &arith,
                const bn_t &address,
                const bn_t &key,
                bn_t &value) const {
                account::account_t* account_ptr = nullptr;
                return (
                    (_state->get_account(arith, address, account_ptr, ACCOUNT_NONE_FLAG) || account_ptr->get_storage_value(arith, key, value)) ?
                    _world_state->get_value(arith, address, key, value) :
                    ERROR_SUCCESS
                );
            }


            __host__ __device__ int32_t AccessState::is_warm_account(
                ArithEnv &arith,
                const bn_t &address) const {
                cuEVM::account::account_t* account_ptr = nullptr;
                return _state->get_account(arith, address, account_ptr, ACCOUNT_NONE_FLAG) == 0;
            }

            __host__ __device__ int32_t AccessState::is_warm_key(
                ArithEnv &arith,
                const bn_t &address,
                const bn_t &key) const {
                cuEVM::account::account_t* account_ptr = nullptr;
                bn_t value;
                return !(
                    _state->get_account(arith, address, account_ptr, ACCOUNT_STORAGE_FLAG) ||
                    account_ptr->get_storage_value(arith, key, value)
                );
            }
            __host__ __device__ int32_t AccessState::is_deleted_account(
                ArithEnv &arith,
                const bn_t &address) const {
                cuEVM::account::account_t* account_ptr = nullptr;
                return _world_state->get_account(arith, address, account_ptr);
            }
    }
}
