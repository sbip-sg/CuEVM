// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2024-06-20
// SPDX-License-Identifier: MIT

#include <CuEVM/state/touch_state.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM::state {
    __host__ __device__ int32_t TouchState::add_account(
        ArithEnv &arith,
        const bn_t &address,
        CuEVM::account::account_t* &account_ptr,
        const CuEVM::account::account_flags_t acces_state_flag
    ) {
        CuEVM::account::account_t* tmp_account_ptr = nullptr;
        CuEVM::account::account_t *tmp_access_account_ptr = nullptr;
        TouchState* tmp = parent;
        _access_state->get_account(
            arith,
            address,
            tmp_access_account_ptr,
            acces_state_flag);
        while(
            (tmp != nullptr) && 
            (tmp->_state->get_account(arith, address, tmp_account_ptr))
        ) tmp = tmp->parent;
        return _state->add_duplicate_account(
            account_ptr,
            (
                (tmp != nullptr) ?
                tmp_account_ptr :
                tmp_access_account_ptr
            ),
            ACCOUNT_NONE_FLAG);
    }
    
    __host__ __device__ int32_t TouchState::get_account(
        ArithEnv &arith,
        const bn_t &address,
        CuEVM::account::account_t* &account_ptr,
        const CuEVM::account::account_flags_t acces_state_flag
    ) {
        CuEVM::account::account_t *tmp_ptr;
        _access_state->get_account(arith, address, tmp_ptr, acces_state_flag);
        return (
            _state->get_account(arith, address, account_ptr) ?
            ([&]() -> int32_t {
                TouchState* tmp = parent;
                while(
                    (tmp != nullptr) && 
                    (tmp->_state->get_account(arith, address, account_ptr))
                ) tmp = tmp->parent;
                account_ptr = (tmp != nullptr) ? account_ptr : tmp_ptr;
                return ERROR_SUCCESS;
            })() : ERROR_SUCCESS
        );
    }

    __host__ __device__ int32_t  TouchState::get_balance(
            ArithEnv &arith,
            const bn_t &address,
            bn_t &balance) {
        account::account_t* account_ptr = nullptr;
        int32_t error_code = get_account(arith, address, account_ptr, ACCOUNT_BALANCE_FLAG);
        if (error_code == ERROR_SUCCESS) {
            account_ptr->get_balance(arith, balance);
        }
        return error_code;
    }

    __host__ __device__ int32_t TouchState::get_nonce(
        ArithEnv &arith,
        const bn_t &address,
        bn_t &nonce
    ) {
        account::account_t* account_ptr = nullptr;
        int32_t error_code = get_account(arith, address, account_ptr, ACCOUNT_NONCE_FLAG);
        if (error_code == ERROR_SUCCESS) {
            account_ptr->get_nonce(arith, nonce);
        }
        return error_code;
    }

    __host__ __device__ int32_t TouchState::get_code(
        ArithEnv &arith,
        const bn_t &address,
        byte_array_t &byte_code
    ) {
        account::account_t* account_ptr = nullptr;
        int32_t error_code = get_account(arith, address, account_ptr, ACCOUNT_BYTE_CODE_FLAG);
        if (error_code == ERROR_SUCCESS) {
            byte_code = account_ptr->get_byte_code();
        }
        return error_code;
    }

    __host__ __device__ int32_t TouchState::get_value(
        ArithEnv &arith,
        const bn_t &address,
        const bn_t &key,
        bn_t &value
    ) {
        bn_t tmp_value;
        poke_value(arith, address, key, value);
        return _access_state->get_value(arith, address, key, tmp_value);
    }

    __host__ __device__ int32_t TouchState::poke_value(
        ArithEnv &arith,
        const bn_t &address,
        const bn_t &key,
        bn_t &value) const {
        account::account_t* account_ptr = nullptr;
        if (
            _state->get_account(arith, address, account_ptr, ACCOUNT_NONE_FLAG) ||
            account_ptr->get_storage_value(arith, key, value)) {
            return ERROR_SUCCESS;
        }
        TouchState* tmp = parent;
        while (tmp != nullptr) {
            if (
                !(
                tmp->_state->get_account(
                    arith,
                    address,
                    account_ptr,
                    ACCOUNT_NONE_FLAG) || 
                account_ptr->get_storage_value(
                        arith,
                        key,
                        value)
                )
            ) {
                return ERROR_SUCCESS;
            }
            tmp = tmp->parent;
        }
        return _access_state->poke_value(
            arith,
            address,
            key,
            value);
    }

    __host__ __device__ int32_t TouchState::set_balance(
        ArithEnv &arith,
        const bn_t &address,
        const bn_t &balance
    ) {
        account::account_t* account_ptr = nullptr;
        _access_state->get_account(arith, address, account_ptr, ACCOUNT_BALANCE_FLAG);
        if (_state->get_account(arith, address, account_ptr, ACCOUNT_BALANCE_FLAG)) {
            add_account(
                arith,
                address,
                account_ptr,
                ACCOUNT_BALANCE_FLAG);
        }
        account_ptr->set_balance(arith, balance);
        return ERROR_SUCCESS;
    }

    __host__ __device__ int32_t TouchState::set_nonce(
        ArithEnv &arith,
        const bn_t &address,
        const bn_t &nonce
    ) {
        account::account_t* account_ptr = nullptr;
        _access_state->get_account(arith, address, account_ptr, ACCOUNT_NONCE_FLAG);
        if (_state->get_account(arith, address, account_ptr, ACCOUNT_NONCE_FLAG)) {
            add_account(
                arith,
                address,
                account_ptr,
                ACCOUNT_NONCE_FLAG);
        }
        account_ptr->set_nonce(arith, nonce);
        return ERROR_SUCCESS;
    }
    
    __host__ __device__ int32_t TouchState::set_code(
        ArithEnv &arith,
        const bn_t &address,
        const byte_array_t &byte_code
    ) {
        account::account_t* account_ptr = nullptr;
        _access_state->get_account(arith, address, account_ptr, ACCOUNT_BYTE_CODE_FLAG);
        if (_state->get_account(arith, address, account_ptr, ACCOUNT_BYTE_CODE_FLAG)) {
            add_account(
                arith,
                address,
                account_ptr,
                ACCOUNT_BYTE_CODE_FLAG);
        }
        account_ptr->set_byte_code(byte_code);
        return ERROR_SUCCESS;
    }

    __host__ __device__ int32_t TouchState::set_storage_value(
        ArithEnv &arith,
        const bn_t &address,
        const bn_t &key,
        const bn_t &value
    ) {
        account::account_t* account_ptr = nullptr;
        _access_state->get_account(arith, address, account_ptr, ACCOUNT_STORAGE_FLAG);
        if (_state->get_account(arith, address, account_ptr, ACCOUNT_STORAGE_FLAG)) {
            add_account(
                arith,
                address,
                account_ptr,
                ACCOUNT_STORAGE_FLAG);
        }
        account_ptr->set_storage_value(arith, key, value);
        return ERROR_SUCCESS;
    }

    __host__ __device__ int32_t TouchState::delete_account(
        ArithEnv &arith,
        const bn_t &address
    ) {
        account::account_t* account_ptr = nullptr;
        if (_state->get_account(arith, address, account_ptr, ACCOUNT_DELETED_FLAG)) {
            add_account(
                arith,
                address,
                account_ptr,
                ACCOUNT_DELETED_FLAG);
        }
        return ERROR_SUCCESS;
    }

    __host__ __device__ int32_t TouchState::update(
        ArithEnv &arith,
        TouchState* other
    ) {
        return _state->update(arith, *(other->_state));
    }

    __host__ __device__ int32_t TouchState::is_empty_account(
        ArithEnv &arith,
        const bn_t &address
    ) {
        account::account_t* account_ptr = nullptr;
        get_account(
            arith,
            address,
            account_ptr,
            ACCOUNT_NON_STORAGE_FLAG);
        return account_ptr->is_empty(arith);
    }

    __host__ __device__ int32_t TouchState::is_deleted_account(
        ArithEnv &arith,
        const bn_t &address
    ) {
        CuEVM::account::account_t* account_ptr = nullptr;
        uint32_t index;
        if (_state->get_account_index(arith, address, index) == 0) {
            return _state->flags[index].has_deleted();
        }

        TouchState* tmp = parent;
        while(
            (tmp != nullptr) && 
            (tmp->_state->get_account_index(arith, address, index))
        ) tmp = tmp->parent;
        return (tmp != nullptr) ? tmp->_state->flags[index].has_deleted() : _access_state->is_deleted_account(arith, address);

    }


    __host__ __device__ int32_t TouchState::transfer(
        ArithEnv &arith,
        const bn_t &from,
        const bn_t &to,
        const bn_t &value) {
        bn_t from_balance, to_balance;
        int32_t error_code = get_balance(arith, from, from_balance);
        error_code |= get_balance(arith, to, to_balance);
        error_code |= cgbn_compare(arith.env, from_balance, value) < 0 ? ERROR_INSUFFICIENT_FUNDS : ERROR_SUCCESS;
        cgbn_sub(arith.env, from_balance, from_balance, value);
        cgbn_add(arith.env, to_balance, to_balance, value);
        error_code |= set_balance(arith, from, from_balance);
        error_code |= set_balance(arith, to, to_balance);
        return error_code;
    }

    
        __host__ void TouchState::print() const {
            _state->print();
        }
}
