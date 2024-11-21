// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation
// Programme) Author: Stefan-Dan Ciocirlan Data: 2024-06-20
// SPDX-License-Identifier: MIT

#include <CuEVM/state/touch_state.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM {
__host__ __device__ int32_t TouchState::add_account(ArithEnv &arith, const evm_word_t *address,
                                                    CuEVM::account_t *&account_ptr,
                                                    const CuEVM::account_flags_t acces_state_flag) {
    CuEVM::account_t *tmp_account_ptr = nullptr;
#ifdef EIP_3155
    __ONE_THREAD_PER_INSTANCE(printf("TouchState::add_account - account_ptr: %p idx %d\n", account_ptr, THREADIDX);)
    address->print();
    // print_bnt(arith, address);
#endif

    // this->print();
    // __ONE_THREAD_PER_INSTANCE(printf("TouchState::add_account - acces_state_flag: %d\n", acces_state_flag););
    int32_t error_code = get_account(arith, address, tmp_account_ptr, acces_state_flag);
    if (error_code == ERROR_SUCCESS) {
        // #ifdef __CUDA_ARCH__
        //         printf("TouchState::add_account after get_account %d tmp_account_ptr %p\n", threadIdx.x,
        //         tmp_account_ptr);
        // #endif
        // check if tmp_account_ptr is not in the current _state
        uint32_t index;
        if (_state->get_account_index(arith, address, index) == ERROR_SUCCESS) {
            // account found in the current state
            // #ifdef __CUDA_ARCH__
            //             printf("TouchState::add_account account found in the current state %d  acc index %d\n",
            //             threadIdx.x, index);
            // #endif
            // print_bnt(arith, address);
            // address->print();
            // _state->print();
            _state->flags[index].update(acces_state_flag);
        } else
            return _state->add_duplicate_account(arith, account_ptr, tmp_account_ptr, acces_state_flag);
    }
    account_ptr = tmp_account_ptr;
    return ERROR_SUCCESS;
}

__host__ __device__ int32_t TouchState::get_account(ArithEnv &arith, const evm_word_t *address,
                                                    CuEVM::account_t *&account_ptr,
                                                    const CuEVM::account_flags_t acces_state_flag,
                                                    bool add_to_current_state) {
    _world_state->get_account(arith, address, account_ptr);
    // #ifdef __CUDA_ARCH__
    //     printf("_world_state->get_account %d\n", threadIdx.x);
    // #endif
    const TouchState *tmp = this;
    while ((tmp != nullptr) && (tmp->_state->get_account(arith, address, account_ptr, acces_state_flag)))
        tmp = tmp->parent;
    // #ifdef __CUDA_ARCH__
    //     printf("loop ->get_account %d\n", threadIdx.x);
    // #endif
    if (account_ptr == nullptr) {
        // #ifdef __CUDA_ARCH__
        //         printf("get_account not found, add to current state %d\n", threadIdx.x);
        // #endif
        _state->add_new_account(arith, address, account_ptr, acces_state_flag);
        return ERROR_STATE_ADDRESS_NOT_FOUND;  //  error code !+ ERROR_SUCCESS => just added
    }

    return ERROR_SUCCESS;
}

__host__ __device__ int32_t TouchState::poke_account(ArithEnv &arith, const evm_word_t *address,
                                                     CuEVM::account_t *&account_ptr, bool include_world_state) const {
    if (include_world_state) _world_state->get_account(arith, address, account_ptr);
    const TouchState *tmp = this;
    while ((tmp != nullptr) && (tmp->_state->get_account(arith, address, account_ptr))) tmp = tmp->parent;
    return account_ptr != nullptr ? ERROR_SUCCESS : ERROR_STATE_ADDRESS_NOT_FOUND;
}

__host__ __device__ int32_t TouchState::get_account_index(ArithEnv &arith, const evm_word_t *address,
                                                          uint32_t &index) const {
    index = 0;
    return _state->get_account_index(arith, address, index) == ERROR_SUCCESS ? ERROR_SUCCESS
                                                                             : ERROR_STATE_ADDRESS_NOT_FOUND;
}

__host__ __device__ int32_t TouchState::get_balance(ArithEnv &arith, const evm_word_t *address, bn_t &balance) {
    account_t *account_ptr = nullptr;
    // set to 0 first
    cgbn_set_ui32(arith.env, balance, 0);
    // #ifdef __CUDA_ARCH__
    //     printf("TouchState::get_balance %d\n", threadIdx.x);
    //     print_bnt(arith, address);
    // #endif
    int32_t error_code = get_account(arith, address, account_ptr, ACCOUNT_BALANCE_FLAG);
    if (error_code == ERROR_SUCCESS) {
        // printf("account found %p\n", account_ptr);
        account_ptr->get_balance(arith, balance);
    }
    return error_code;
}

__host__ __device__ int32_t TouchState::get_nonce(ArithEnv &arith, const evm_word_t *address, bn_t &nonce) {
    account_t *account_ptr = nullptr;
    // set to 0 first
    cgbn_set_ui32(arith.env, nonce, 0);
    int32_t error_code = get_account(arith, address, account_ptr, ACCOUNT_NONCE_FLAG);
    if (error_code == ERROR_SUCCESS) {
        account_ptr->get_nonce(arith, nonce);
    }
    return error_code;
}

__host__ __device__ int32_t TouchState::get_code(ArithEnv &arith, const evm_word_t *address, byte_array_t &byte_code) {
    account_t *account_ptr = nullptr;
    int32_t error_code = get_account(arith, address, account_ptr, ACCOUNT_BYTE_CODE_FLAG);
    if (error_code == ERROR_SUCCESS) {
        byte_code = account_ptr->get_byte_code();
    }
    return error_code;
}

__host__ __device__ int32_t TouchState::get_value(ArithEnv &arith, const evm_word_t *address, const bn_t &key,
                                                  bn_t &value) {
    poke_value(arith, address, key, value);
    // #ifdef __CUDA_ARCH__
    //     printf("TouchState::get_value %d\n", threadIdx.x);
    //     print_bnt(arith, address);
    //     print_bnt(arith, key);
    //     print_bnt(arith, value);
    // #endif
    return this->set_warm_key(arith, address, key, value);
}

__host__ __device__ int32_t TouchState::poke_value(ArithEnv &arith, const evm_word_t *address, const bn_t &key,
                                                   bn_t &value) const {
    account_t *account_ptr = nullptr;
    const TouchState *tmp = this;
    while (tmp != nullptr) {
        if (!(tmp->_state->get_account(arith, address, account_ptr, ACCOUNT_NONE_FLAG) ||
              account_ptr->get_storage_value(arith, key, value))) {
            return ERROR_SUCCESS;
        }
        tmp = tmp->parent;
    }
    return _world_state->get_value(arith, address, key, value);
}

__host__ __device__ int32_t TouchState::poke_original_value(ArithEnv &arith, const evm_word_t *address, const bn_t &key,
                                                            bn_t &value) const {
    return _world_state->get_value(arith, address, key, value);
}

__host__ __device__ int32_t TouchState::poke_balance(ArithEnv &arith, const evm_word_t *address, bn_t &balance) const {
    account_t *account_ptr = nullptr;
    const TouchState *tmp = this;

    while (tmp != nullptr) {
        if (!(tmp->_state->get_account(arith, address, account_ptr, ACCOUNT_BALANCE_FLAG))) {
            account_ptr->get_balance(arith, balance);
            return ERROR_SUCCESS;
        }

        tmp = tmp->parent;
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

__host__ __device__ bool TouchState::is_warm_account(ArithEnv &arith, const evm_word_t *address) const {
    if (*address < EVM_PRECOMPILED_CONTRACTS + 1 && !(*address == 0)) return true;
    account_t *account_ptr = nullptr;
    return (poke_account(arith, address, account_ptr) == ERROR_SUCCESS);
}

__host__ __device__ bool TouchState::is_warm_key(ArithEnv &arith, const evm_word_t *address, const bn_t &key) const {
    account_t *account_ptr = nullptr;
    bn_t value;
    const TouchState *tmp = this;
    while (tmp != nullptr) {
        if (!(tmp->_state->get_account(arith, address, account_ptr, ACCOUNT_NONE_FLAG))) {
            if (account_ptr->get_storage_value(arith, key, value) == ERROR_SUCCESS) return true;
        }
        tmp = tmp->parent;
    }
    return false;
}

__host__ __device__ bool TouchState::set_warm_account(ArithEnv &arith, const evm_word_t *address) {
    account_t *account_ptr = nullptr;
    if (poke_account(arith, address, account_ptr)) {
        add_account(arith, address, account_ptr, ACCOUNT_BALANCE_FLAG);
    }
}
__host__ __device__ bool TouchState::set_warm_key(ArithEnv &arith, const evm_word_t *address, const bn_t &key,
                                                  const bn_t &value) {
    account_t *account_ptr = nullptr;
    if (_state->get_account(arith, address, account_ptr, ACCOUNT_STORAGE_FLAG) != ERROR_SUCCESS) {
        // #ifdef __CUDA_ARCH__
        //         printf("TouchState::set_warm_key not found account, adding %d\n", threadIdx.x);
        // #endif
        add_account(arith, address, account_ptr, ACCOUNT_STORAGE_FLAG);
    }
    // #ifdef __CUDA_ARCH__
    //     printf("TouchState::set_warm_key  %d found account %p, setting value\n", threadIdx.x, account_ptr);
    // #endif
    // get_account(arith, address, account_ptr, ACCOUNT_STORAGE_FLAG, true);
    account_ptr->set_storage_value(arith, key, value);
}
__host__ __device__ int32_t TouchState::set_balance(ArithEnv &arith, const evm_word_t *address, const bn_t &balance) {
    account_t *account_ptr = nullptr;
    if (_state->get_account(arith, address, account_ptr, ACCOUNT_BALANCE_FLAG) != ERROR_SUCCESS) {
        // printf("TouchState::set_balance - get_account - account_ptr: %p\n", account_ptr);
        add_account(arith, address, account_ptr, ACCOUNT_BALANCE_FLAG);
        // printf("after add_account - account_ptr: %p\n", account_ptr);
    }
    // #ifdef __CUDA_ARCH__
    //     printf("TouchState::set_balance after add account %d, account_ptr %p\n" ,threadIdx.x, account_ptr);
    // #endif
    // get_account(arith, address, account_ptr, ACCOUNT_BALANCE_FLAG, true);
    account_ptr->set_balance(arith, balance);
    // printf("after set balance\n");
    // #ifdef __CUDA_ARCH__
    //     printf("TouchState::set_balance after set balance %d\n", threadIdx.x);
    //     print_bnt(arith, address);
    //     account_ptr->balance.print();
    // #endif
    return ERROR_SUCCESS;
}

__host__ __device__ int32_t TouchState::set_nonce(ArithEnv &arith, const evm_word_t *address, const bn_t &nonce) {
    account_t *account_ptr = nullptr;
    if (_state->get_account(arith, address, account_ptr, ACCOUNT_NONCE_FLAG) != ERROR_SUCCESS) {
        // printf("touch state cannot find account\n");
        add_account(arith, address, account_ptr, ACCOUNT_NONCE_FLAG);
    }
    // #ifdef __CUDA_ARCH__
    //     printf("TouchState::set_balance after add account %d account_ptr %p\n" ,threadIdx.x, account_ptr);
    // #endif
    account_ptr->set_nonce(arith, nonce);
    return ERROR_SUCCESS;
}

__host__ __device__ int32_t TouchState::set_code(ArithEnv &arith, const evm_word_t *address,
                                                 const byte_array_t &byte_code) {
    account_t *account_ptr = nullptr;
    if (_state->get_account(arith, address, account_ptr, ACCOUNT_BYTE_CODE_FLAG) != ERROR_SUCCESS) {
        add_account(arith, address, account_ptr, ACCOUNT_BYTE_CODE_FLAG);
    }
    // get_account(arith, address, account_ptr, ACCOUNT_BYTE_CODE_FLAG, true);
    account_ptr->set_byte_code(byte_code);
    return ERROR_SUCCESS;
}

__host__ __device__ int32_t TouchState::set_storage_value(ArithEnv &arith, const evm_word_t *address, const bn_t &key,
                                                          const bn_t &value) {
    account_t *account_ptr = nullptr;
    _world_state->get_account(arith, address, account_ptr);
    if (_state->get_account(arith, address, account_ptr, ACCOUNT_STORAGE_FLAG)) {
        add_account(arith, address, account_ptr, ACCOUNT_STORAGE_FLAG);
    }
    __SYNC_THREADS__
    // __ONE_THREAD_PER_INSTANCE(printf("Set storage value %p %d\n", account_ptr, THREADIDX););
    account_ptr->set_storage_value(arith, key, value);
    return ERROR_SUCCESS;
}

// __host__ __device__ int32_t TouchState::delete_account(ArithEnv &arith,
//                                                        const bn_t
//                                                        &address) {
//     account_t *account_ptr = nullptr;
//     int32_t error_code =
//         _state->get_account(arith, address, account_ptr,
//         ACCOUNT_DELETED_FLAG);
//     // printf("TouchState::delete_account - error_code: %d\n",
//     error_code);
//     // printf("TouchState::delete_account - account_ptr: %p\n",
//     account_ptr); if (account_ptr == nullptr)
//         account_ptr = new account_t(arith, address);
//     else
//         account_ptr->empty();
//     // account_ptr->print();
//     if (error_code)
//         add_account(arith, address, account_ptr, ACCOUNT_DELETED_FLAG);

//     return ERROR_SUCCESS;
// }
// __host__ __device__ int32_t TouchState::delete_account(ArithEnv &arith,
//                                                        const bn_t &address) {
//     account_t *account_ptr = nullptr;
//     int32_t error_code =
//         get_account(arith, address, account_ptr, ACCOUNT_NONE_FLAG);
//     account_ptr->byte_code.free();
//     account_ptr->storage.free();
//     CuEVM::bn_t zero;
//     cgbn_set_ui32(arith.env, zero, 0U);
//     account_ptr->set_balance(arith, zero);
//     account_ptr->set_nonce(arith, zero);
//     // get the full storage from the access state and world state
//     _world_state->get_storage(arith, address, account_ptr->storage);
//     TouchState *tmp = parent;
//     CuEVM::account_t *tmp_account_ptr = nullptr;
//     while (tmp != nullptr) {
//         if (tmp->_state->get_account(arith, address, tmp_account_ptr) ==
//             ERROR_SUCCESS) {
//             account_ptr->storage.update(arith, tmp_account_ptr->storage);
//         }
//         tmp = tmp->parent;
//     }
//     // zero the value in the storage
//     for (uint32_t idx = 0; idx < account_ptr->storage.size; idx++) {
//         cgbn_store(
//             arith.env,
//             (cgbn_evm_word_t_ptr)&account_ptr->storage.storage[idx].value,
//             zero);
//     }
//     // set all flags and deleted flag
//     _state->get_account(arith, address, account_ptr,
//                         ACCOUNT_DELETED_FLAG | ACCOUNT_ALL_FLAG);
//     return ERROR_SUCCESS;
// }

__host__ __device__ int32_t TouchState::update(ArithEnv &arith, TouchState *other) {
    return _state->update(arith, *(other->_state));
}

__host__ __device__ bool TouchState::is_empty_account(ArithEnv &arith, const evm_word_t *address) {
    account_t *account_ptr = nullptr;
    poke_account(arith, address, account_ptr, true);
    uint32_t result = account_ptr != nullptr ? account_ptr->is_empty() : 1;
    return result;
}

__host__ __device__ bool TouchState::is_empty_account_create(ArithEnv &arith, const evm_word_t *address) {
    account_t *account_ptr = nullptr;
    poke_account(arith, address, account_ptr, true);
    uint32_t result = account_ptr != nullptr ? account_ptr->is_empty_create() : 1;
    return result;
}

__host__ __device__ int32_t TouchState::is_deleted_account(ArithEnv &arith, const evm_word_t *address) {
    return ERROR_SUCCESS;
}

__host__ __device__ CuEVM::contract_storage_t TouchState::get_entire_storage(ArithEnv &arith,
                                                                             const uint32_t account_index) const {
    return _state->accounts[account_index].storage;
}

__host__ __device__ int32_t TouchState::transfer(ArithEnv &arith, const evm_word_t *from, const evm_word_t *to,
                                                 const bn_t &value) {
    bn_t from_balance, to_balance;
    // #ifdef __CUDA_ARCH__
    //     printf("TouchState::transfer %d\n", threadIdx.x);
    // #endif
    int32_t error_code = poke_balance(arith, from, from_balance);
    // #ifdef __CUDA_ARCH__
    //     printf("TouchState::transfer after poke err %d %d\n", error_code ,threadIdx.x);
    // #endif
    if (error_code != ERROR_SUCCESS || cgbn_compare(arith.env, from_balance, value) < 0)
        return ERROR_INSUFFICIENT_FUNDS;
    error_code |= poke_balance(arith, to, to_balance);
    cgbn_sub(arith.env, from_balance, from_balance, value);
    cgbn_add(arith.env, to_balance, to_balance, value);
    // #ifdef __CUDA_ARCH__
    //     printf("TouchState::transfer after poke err %d %d\n", error_code ,threadIdx.x);
    // #endif
    error_code |= set_balance(arith, from, from_balance);
    // #ifdef __CUDA_ARCH__
    //     printf("TouchState::transfer after set balance from err %d %d\n", error_code ,threadIdx.x);
    // #endif
    error_code |= set_balance(arith, to, to_balance);
    // #ifdef __CUDA_ARCH__
    //     printf("TouchState::transfer after set balance to err %d %d\n", error_code ,threadIdx.x);
    // #endif
    return error_code;
}

__host__ __device__ void TouchState::print() const { _state->print(); }
}  // namespace CuEVM
