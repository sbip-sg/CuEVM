#include <CuEVM/state/touch_state.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM {
__host__ __device__ int32_t TouchState::add_account(const evm_word_t *address, CuEVM::account_t *&account_ptr,
                                                    const CuEVM::account_flags_t acces_state_flag) {
    CuEVM::account_t *tmp_account_ptr = nullptr;
#ifdef EIP_3155
    __ONE_THREAD_PER_INSTANCE(printf("TouchState::add_account - account_ptr: %p idx %d\n", account_ptr, THREADIDX);)
    address->print();
    // print_bnt(arith, address);
#endif

    // this->print();
    // __ONE_THREAD_PER_INSTANCE(printf("TouchState::add_account - acces_state_flag: %d\n", acces_state_flag););
    int32_t error_code = get_account(address, tmp_account_ptr, acces_state_flag);
    if (error_code == ERROR_SUCCESS) {
        // #ifdef __CUDA_ARCH__
        //         printf("TouchState::add_account after get_account %d tmp_account_ptr %p\n", threadIdx.x,
        //         tmp_account_ptr);
        // #endif
        // check if tmp_account_ptr is not in the current _state
        uint32_t index;
        if (_state->get_account_index(address, index) == ERROR_SUCCESS) {
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
            return _state->add_duplicate_account(account_ptr, tmp_account_ptr, acces_state_flag);
    }
    account_ptr = tmp_account_ptr;
    return ERROR_SUCCESS;
}

__host__ __device__ int32_t TouchState::get_account(const evm_word_t *address, CuEVM::account_t *&account_ptr,
                                                    const CuEVM::account_flags_t acces_state_flag,
                                                    bool add_to_current_state) {
    _world_state->get_account(address, account_ptr);

    // __ONE_THREAD_PER_INSTANCE(printf("\n\n_world_state->get_account %d\n", THREADIDX););
    // address->print();

    const TouchState *tmp = this;
    while ((tmp != nullptr) && (tmp->_state->get_account(address, account_ptr, acces_state_flag))) tmp = tmp->parent;

    if (account_ptr == nullptr) {
        // __ONE_THREAD_PER_INSTANCE(printf("get_account not found, add to current state %d\n", THREADIDX););

        _state->add_new_account(address, account_ptr, acces_state_flag);
        return ERROR_STATE_ADDRESS_NOT_FOUND;  //  error code !+ ERROR_SUCCESS => just added
    }

    return ERROR_SUCCESS;
}

__host__ __device__ int32_t TouchState::poke_account(const evm_word_t *address, CuEVM::account_t *&account_ptr,
                                                     bool include_world_state) const {
    if (include_world_state) _world_state->get_account(address, account_ptr);
    const TouchState *tmp = this;
    while ((tmp != nullptr) && (tmp->_state->get_account(address, account_ptr))) tmp = tmp->parent;
    return account_ptr != nullptr ? ERROR_SUCCESS : ERROR_STATE_ADDRESS_NOT_FOUND;
}

__host__ __device__ int32_t TouchState::get_account_index(const evm_word_t *address, uint32_t &index) const {
    index = 0;
    return _state->get_account_index(address, index) == ERROR_SUCCESS ? ERROR_SUCCESS : ERROR_STATE_ADDRESS_NOT_FOUND;
}

__host__ __device__ int32_t TouchState::get_balance(const evm_word_t *address, evm_word_t &balance) {
    account_t *account_ptr = nullptr;
    balance = 0;
    // #ifdef __CUDA_ARCH__
    //     printf("TouchState::get_balance %d\n", threadIdx.x);
    //     print_bnt(arith, address);
    // #endif
    int32_t error_code = get_account(address, account_ptr, ACCOUNT_BALANCE_FLAG);
    if (error_code == ERROR_SUCCESS) {
        // printf("account found %p\n", account_ptr);
        account_ptr->get_balance(balance);
    }
    return error_code;
}

__host__ __device__ int32_t TouchState::get_nonce(const evm_word_t *address, evm_word_t &nonce) {
    account_t *account_ptr = nullptr;
    // set to 0 first
    nonce = 0;
    int32_t error_code = get_account(address, account_ptr, ACCOUNT_NONCE_FLAG);
    if (error_code == ERROR_SUCCESS) {
        account_ptr->get_nonce(nonce);
    }
    return error_code;
}

__host__ __device__ int32_t TouchState::get_code(const evm_word_t *address, byte_array_t &byte_code) {
    account_t *account_ptr = nullptr;
    int32_t error_code = get_account(address, account_ptr, ACCOUNT_BYTE_CODE_FLAG);
    if (error_code == ERROR_SUCCESS) {
        byte_code = account_ptr->get_byte_code();
    }
    return error_code;
}

__host__ __device__ int32_t TouchState::get_value(const evm_word_t *address, const evm_word_t &key, evm_word_t &value) {
    poke_value(address, key, value);
    // #ifdef __CUDA_ARCH__
    //     printf("TouchState::get_value %d\n", threadIdx.x);
    //     print_bnt(arith, address);
    //     print_bnt(arith, key);
    //     print_bnt(arith, value);
    // #endif
    return this->set_warm_key(address, key, value);
}

__host__ __device__ int32_t TouchState::poke_value(const evm_word_t *address, const evm_word_t &key,
                                                   evm_word_t &value) const {
    account_t *account_ptr = nullptr;
    const TouchState *tmp = this;
    while (tmp != nullptr) {
        if (!(tmp->_state->get_account(address, account_ptr, ACCOUNT_NONE_FLAG) ||
              account_ptr->get_storage_value(key, value))) {
            return ERROR_SUCCESS;
        }
        tmp = tmp->parent;
    }
    return _world_state->get_value(address, key, value);
}

__host__ __device__ int32_t TouchState::poke_original_value(const evm_word_t *address, const evm_word_t &key,
                                                            evm_word_t &value) const {
    return _world_state->get_value(address, key, value);
}

__host__ __device__ int32_t TouchState::poke_balance(const evm_word_t *address, evm_word_t &balance) const {
    account_t *account_ptr = nullptr;
    const TouchState *tmp = this;

    while (tmp != nullptr) {
        if (!(tmp->_state->get_account(address, account_ptr, ACCOUNT_BALANCE_FLAG))) {
            account_ptr->get_balance(balance);
            return ERROR_SUCCESS;
        }

        tmp = tmp->parent;
    }

    _world_state->get_account(address, account_ptr);
    if (account_ptr != nullptr) {
        account_ptr->get_balance(balance);
        return ERROR_SUCCESS;
    }
    // not found, simply return 0 and not error
    balance = 0;
    return ERROR_SUCCESS;
}

__host__ __device__ bool TouchState::is_warm_account(const evm_word_t *address) const {
    if (uint256_get_uint32_t(address) < EVM_PRECOMPILED_CONTRACTS + 1 && !uint256_is_zero(address)) return true;
    account_t *account_ptr = nullptr;
    return (poke_account(address, account_ptr) == ERROR_SUCCESS);
}

__host__ __device__ bool TouchState::is_warm_key(const evm_word_t *address, const evm_word_t &key) const {
    account_t *account_ptr = nullptr;
    evm_word_t value;
    const TouchState *tmp = this;
    while (tmp != nullptr) {
        if (!(tmp->_state->get_account(address, account_ptr, ACCOUNT_NONE_FLAG))) {
            if (account_ptr->get_storage_value(key, value) == ERROR_SUCCESS) return true;
        }
        tmp = tmp->parent;
    }
    return false;
}

__host__ __device__ bool TouchState::set_warm_account(const evm_word_t *address) {
    account_t *account_ptr = nullptr;
    if (poke_account(address, account_ptr)) {
        add_account(address, account_ptr, ACCOUNT_BALANCE_FLAG);
    }
}
__host__ __device__ bool TouchState::set_warm_key(const evm_word_t *address, const evm_word_t &key,
                                                  const evm_word_t &value) {
    account_t *account_ptr = nullptr;
    if (_state->get_account(address, account_ptr, ACCOUNT_STORAGE_FLAG) != ERROR_SUCCESS) {
        add_account(address, account_ptr, ACCOUNT_STORAGE_FLAG);
    }

    account_ptr->set_storage_value(key, value);
}
__host__ __device__ int32_t TouchState::set_balance(const evm_word_t *address, const evm_word_t &balance) {
    account_t *account_ptr = nullptr;
    if (_state->get_account(address, account_ptr, ACCOUNT_BALANCE_FLAG) != ERROR_SUCCESS) {
        // printf("TouchState::set_balance - get_account - account_ptr: %p\n", account_ptr);
        add_account(address, account_ptr, ACCOUNT_BALANCE_FLAG);
        // printf("after add_account - account_ptr: %p\n", account_ptr);
    }
    // #ifdef __CUDA_ARCH__
    //     printf("TouchState::set_balance after add account %d, account_ptr %p\n" ,threadIdx.x, account_ptr);
    // #endif
    // get_account(arith, address, account_ptr, ACCOUNT_BALANCE_FLAG, true);
    account_ptr->set_balance(balance);
    // printf("after set balance\n");
    // #ifdef __CUDA_ARCH__
    //     printf("TouchState::set_balance after set balance %d\n", threadIdx.x);
    //     print_bnt(arith, address);
    //     account_ptr->balance.print();
    // #endif
    return ERROR_SUCCESS;
}

__host__ __device__ int32_t TouchState::set_nonce(const evm_word_t *address, const evm_word_t &nonce) {
    account_t *account_ptr = nullptr;
    if (_state->get_account(address, account_ptr, ACCOUNT_NONCE_FLAG) != ERROR_SUCCESS) {
        // printf("touch state cannot find account\n");
        add_account(address, account_ptr, ACCOUNT_NONCE_FLAG);
    }
    // #ifdef __CUDA_ARCH__
    //     printf("TouchState::set_balance after add account %d account_ptr %p\n" ,threadIdx.x, account_ptr);
    // #endif
    account_ptr->set_nonce(nonce);
    return ERROR_SUCCESS;
}

__host__ __device__ int32_t TouchState::set_code(const evm_word_t *address, const byte_array_t &byte_code) {
    account_t *account_ptr = nullptr;
    if (_state->get_account(address, account_ptr, ACCOUNT_BYTE_CODE_FLAG) != ERROR_SUCCESS) {
        add_account(address, account_ptr, ACCOUNT_BYTE_CODE_FLAG);
    }
    account_ptr->set_byte_code(byte_code);
    return ERROR_SUCCESS;
}

__host__ __device__ int32_t TouchState::set_storage_value(const evm_word_t *address, const evm_word_t &key,
                                                          const evm_word_t &value) {
    account_t *account_ptr = nullptr;
    _world_state->get_account(address, account_ptr);
    if (_state->get_account(address, account_ptr, ACCOUNT_STORAGE_FLAG)) {
        add_account(address, account_ptr, ACCOUNT_STORAGE_FLAG);
    }

    account_ptr->set_storage_value(key, value);
    return ERROR_SUCCESS;
}

__host__ __device__ int32_t TouchState::update(TouchState *other) { return _state->update(*(other->_state)); }

__host__ __device__ bool TouchState::is_empty_account(const evm_word_t *address) {
    account_t *account_ptr = nullptr;
    poke_account(address, account_ptr);
    uint32_t result = account_ptr != nullptr ? account_ptr->is_empty() : 1;
    return result;
}

__host__ __device__ bool TouchState::is_empty_account_create(const evm_word_t *address) {
    account_t *account_ptr = nullptr;
    poke_account(address, account_ptr);
    uint32_t result = account_ptr != nullptr ? account_ptr->is_empty_create() : 1;
    return result;
}

__host__ __device__ int32_t TouchState::is_deleted_account(const evm_word_t *address) { return ERROR_SUCCESS; }

__host__ __device__ CuEVM::contract_storage_t TouchState::get_entire_storage(const uint32_t account_index) const {
    return _state->accounts[account_index].storage;
}

__host__ __device__ int32_t TouchState::transfer(const evm_word_t *from, const evm_word_t *to,
                                                 const evm_word_t &value) {
    evm_word_t from_balance, to_balance;

    int32_t error_code = poke_balance(from, from_balance);

    if (error_code != ERROR_SUCCESS || uint256_cmp(&from_balance, &value) < 0) return ERROR_INSUFFICIENT_FUNDS;
    error_code |= poke_balance(to, to_balance);
    uint256_sub(&from_balance, &from_balance, &value);
    uint256_add(&to_balance, &to_balance, &value);
    error_code |= set_balance(from, from_balance);
    error_code |= set_balance(to, to_balance);

    return error_code;
}

__host__ __device__ void TouchState::print() const { _state->print(); }
}  // namespace CuEVM
