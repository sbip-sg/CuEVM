
#include "include/touch_state.cuh"

namespace cuEVM {
    namespace state {
            __host__ __device__ int32_t TouchState::add_account(
                ArithEnv &arith,
                const bn_t &address,
                cuEVM::account::account_t* &account_ptr,
                const cuEVM::account::account_flags_t flag
            ) {
                cuEVM::account::account_t* tmp_account_ptr = nullptr;
                TouchState* tmp = parent;
                while (tmp != nullptr) {
                    if (tmp->_state->get_account(arith, address, tmp_account_ptr)) {
                        cuEVM::account::account_t *tmp_ptr;
                        _access_state->get_account(arith, address, tmp_ptr, flag);
                        return _state->add_duplicate_account(
                            account_ptr,
                            tmp_account_ptr,
                            flag);
                    }
                    tmp = tmp->parent;
                }

                if(_access_state->get_account(arith, address, tmp_account_ptr, flag)) {
                    return _state->add_duplicate_account(
                        account_ptr,
                        tmp_account_ptr,
                        flag);
                }
                return 0;
            }
            
            __host__ __device__ int32_t TouchState::get_account(
                ArithEnv &arith,
                const bn_t &address,
                cuEVM::account::account_t* &account_ptr,
                const cuEVM::account::account_flags_t flag = ACCOUNT_NONE_FLAG
            ) {
                if(_state->get_account(arith, address, account_ptr, flag)) {
                    return 1;
                } else {
                    TouchState* tmp = parent;
                    while (tmp != nullptr) {
                        if (tmp->_state->get_account(arith, address, account_ptr)) {
                            cuEVM::account::account_t *tmp_ptr;
                            _access_state->get_account(arith, address, tmp_ptr, flag);
                            return 1;
                        }
                        tmp = tmp->parent;
                    }
                    return _access_state->get_account(arith, address, account_ptr, flag);
                }
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
                bn_t &value
            ) {
                account::account_t* account_ptr = nullptr;
                if (_state->get_account(arith, address, account_ptr, ACCOUNT_NONE_FLAG)) {
                    if (account_ptr->get_storage_value(arith, key, value)) {
                        return 1;
                    }
                }
                TouchState* tmp = parent;
                while (tmp != nullptr) {
                    if (
                        tmp->_state->get_account(
                            arith,
                            address,
                            account_ptr,
                            ACCOUNT_NONE_FLAG)) {
                        if (
                            account_ptr->get_storage_value(
                                arith,
                                key,
                                value)) {
                            return 1;
                        }
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
                if (_state->get_account(arith, address, account_ptr, ACCOUNT_BALANCE_FLAG) == 0) {
                    add_account(
                        arith,
                        address,
                        account_ptr,
                        ACCOUNT_BALANCE_FLAG);
                }
                account_ptr->set_balance(arith, balance);
                return 1;
            }

            __host__ __device__ int32_t TouchState::set_nonce(
                ArithEnv &arith,
                const bn_t &address,
                const bn_t &nonce
            ) {
                account::account_t* account_ptr = nullptr;
                if (_state->get_account(arith, address, account_ptr, ACCOUNT_NONCE_FLAG) == 0) {
                    add_account(
                        arith,
                        address,
                        account_ptr,
                        ACCOUNT_NONCE_FLAG);
                }
                account_ptr->set_nonce(arith, nonce);
                return 1;
            }
            
            __host__ __device__ int32_t TouchState::set_code(
                ArithEnv &arith,
                const bn_t &address,
                const byte_array_t &byte_code
            ) {
                account::account_t* account_ptr = nullptr;
                if (_state->get_account(arith, address, account_ptr, ACCOUNT_BYTE_CODE_FLAG) == 0) {
                    add_account(
                        arith,
                        address,
                        account_ptr,
                        ACCOUNT_BYTE_CODE_FLAG);
                }
                account_ptr->set_byte_code(byte_code);
                return 1;
            }

            __host__ __device__ int32_t TouchState::set_storage_value(
                ArithEnv &arith,
                const bn_t &address,
                const bn_t &key,
                const bn_t &value
            ) {
                account::account_t* account_ptr = nullptr;
                if (_state->get_account(arith, address, account_ptr, ACCOUNT_STORAGE_FLAG) == 0) {
                    add_account(
                        arith,
                        address,
                        account_ptr,
                        ACCOUNT_STORAGE_FLAG);
                }
                account_ptr->set_storage_value(arith, key, value);
                return 1;
            }

            __host__ __device__ int32_t TouchState::delete_account(
                ArithEnv &arith,
                const bn_t &address
            ) {
                account::account_t* account_ptr = nullptr;
                if (_state->get_account(arith, address, account_ptr, ACCOUNT_DELETED_FLAG) == 0) {
                    add_account(
                        arith,
                        address,
                        account_ptr,
                        ACCOUNT_DELETED_FLAG);
                }
                return 0;
            }

            __host__ __device__ int32_t TouchState::update(
                ArithEnv &arith,
                TouchState* other
            ) {
                return _state->update(arith, *(other->_state));
            }
    }
}
