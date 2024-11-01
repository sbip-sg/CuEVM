// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation
// Programme) Author: Stefan-Dan Ciocirlan Data: 2024-06-20
// SPDX-License-Identifier: MIT

#include <CuEVM/state/access_state.cuh>
#include <CuEVM/utils/error_codes.cuh>
/*
namespace CuEVM {
__host__ __device__ int32_t AccessState::add_account(ArithEnv &arith, const bn_t &address,
                                                     CuEVM::account_t *&account_ptr,
                                                     const CuEVM::account_flags_t flag) {
    CuEVM::account_t *tmp_account_ptr = nullptr;
    return (_world_state->get_account(arith, address, tmp_account_ptr)
                ? _state->add_new_account(arith, address, account_ptr, flag)
                : _state->add_duplicate_account(arith, account_ptr, tmp_account_ptr, flag));
}

__host__ __device__ int32_t AccessState::get_account(ArithEnv &arith, const bn_t &address,
                                                     CuEVM::account_t *&account_ptr,
                                                     const CuEVM::account_flags_t flag) {
    bool res = _state->get_account(arith, address, account_ptr, flag);
    if (account_ptr == nullptr) account_ptr = new CuEVM::account_t(arith, address);
    return (res && flag.flags != ACCOUNT_POKE_FLAG ? add_account(arith, address, account_ptr, flag) : ERROR_SUCCESS);
}

__host__ __device__ int32_t AccessState::get_value(ArithEnv &arith, const bn_t &address, const bn_t &key, bn_t &value) {
    account_t *account_ptr = nullptr;
    int32_t error_code = (_state->get_account(arith, address, account_ptr, ACCOUNT_STORAGE_FLAG)
                              ? add_account(arith, address, account_ptr, ACCOUNT_STORAGE_FLAG)
                              : ERROR_SUCCESS);
    error_code = (error_code || (account_ptr->get_storage_value(arith, key, value) ? (([&]() -> int32_t {
                      _world_state->get_value(arith, address, key, value);
                      return account_ptr->set_storage_value(arith, key, value);
                  })())
                                                                                   : ERROR_SUCCESS));
    return error_code;
}

__host__ __device__ int32_t AccessState::poke_value(ArithEnv &arith, const bn_t &address, const bn_t &key,
                                                    bn_t &value) const {
    account_t *account_ptr = nullptr;
    return ((_state->get_account(arith, address, account_ptr, ACCOUNT_NONE_FLAG) ||
             account_ptr->get_storage_value(arith, key, value))
                ? _world_state->get_value(arith, address, key, value)
                : ERROR_SUCCESS);
}

__host__ __device__ int32_t AccessState::poke_balance(ArithEnv &arith, const bn_t &address, bn_t &balance) const {
    account_t *account_ptr = nullptr;
    _state->get_account(arith, address, account_ptr, ACCOUNT_NONE_FLAG);
    if (account_ptr != nullptr) {
        account_ptr->get_balance(arith, balance);
        return ERROR_SUCCESS;
    }
    _world_state->get_account(arith, address, account_ptr);
    if (account_ptr != nullptr) {
        account_ptr->get_balance(arith, balance);
        return ERROR_SUCCESS;
    }
    // not found, simply return 0 and not error
    cgbn_set_ui32(arith.env, balance, 0);
    return ERROR_SUCCESS;
}

// __host__ __device__ int32_t
// AccessState::is_warm_account(ArithEnv &arith, const bn_t &address) const {
//     CuEVM::account_t *account_ptr = nullptr;
//     return _state->get_account(arith, address, account_ptr,
//                                ACCOUNT_NONE_FLAG) == 0;
// }

// __host__ __device__ int32_t AccessState::is_warm_key(ArithEnv &arith,
//                                                      const bn_t &address,
//                                                      const bn_t &key) const {
//     CuEVM::account_t *account_ptr = nullptr;
//     bn_t value;
//     return !(_state->get_account(arith, address, account_ptr,
//                                  ACCOUNT_STORAGE_FLAG) ||
//              account_ptr->get_storage_value(arith, key, value));
// }
__host__ __device__ int32_t AccessState::is_deleted_account(ArithEnv &arith, const bn_t &address) const {
    CuEVM::account_t *account_ptr = nullptr;
    return _world_state->get_account(arith, address, account_ptr);
}

// __host__ __device__ int32_t
// AccessState::is_empty_account(ArithEnv &arith, const bn_t &address) const {
//     int32_t error_code = _state->is_empty_account(arith, address);
//     CuEVM::account_t *account_ptr = nullptr;
//     return (error_code == ERROR_STATE_ADDRESS_NOT_FOUND
//                 ? (_world_state->get_account(arith, address, account_ptr) ==
//                            ERROR_STATE_ADDRESS_NOT_FOUND
//                        ? ERROR_SUCCESS
//                        : account_ptr->is_empty())
//                 : error_code);
// }
__host__ __device__ int32_t AccessState::get_storage(ArithEnv &arith, const bn_t &address,
                                                     CuEVM::contract_storage_t &storage) const {
    CuEVM::account_t *account_ptr = nullptr;
    if (_state->get_account(arith, address, account_ptr) == ERROR_SUCCESS) {
        storage.update(arith, account_ptr->storage);
    }
    if (_world_state->get_account(arith, address, account_ptr) == ERROR_SUCCESS) {
        storage.update(arith, account_ptr->storage);
    }
    return ERROR_SUCCESS;
}
}  // namespace CuEVM

*/