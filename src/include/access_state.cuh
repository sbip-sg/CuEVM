#ifndef _CUEVM_ACCESS_STATE_H_
#define _CUEVM_ACCESS_STATE_H_

#include "state.cuh"
#include "world_state.cuh"

namespace cuEVM {
    namespace state {
        class AccessState {
        private:
            state_access_t* _state;
            WorldState* _world_state;
            __host__ __device__ int32_t add_duplicate_account(
                cuEVM::account::account_t* &account_ptr,
                cuEVM::account::account_t* &src_account_ptr,
                const cuEVM::account::account_flags_t flag
            ) {
                cuEVM::account::account_flags_t no_storage_copy(ACCOUNT_NON_STORAGE_FLAG);
                account_ptr = new cuEVM::account::account_t(
                    *src_account_ptr,
                    no_storage_copy);
                return _state->add_account(*account_ptr, flag);
            }
            __host__ __device__ int32_t add_new_account(
                ArithEnv &arith,
                const bn_t &address,
                cuEVM::account::account_t* &account_ptr,
                const cuEVM::account::account_flags_t flag
            ) {
                account_ptr = new cuEVM::account::account_t(
                    arith,
                    address);
                return _state->add_account(*account_ptr, flag);
            }

            __host__ __device__ int32_t add_account(
                ArithEnv &arith,
                const bn_t &address,
                cuEVM::account::account_t* &account_ptr,
                const cuEVM::account::account_flags_t flag
            ) {
                cuEVM::account::account_t* tmp_account_ptr = nullptr;
                if(_world_state->get_account(arith, address, tmp_account_ptr)) {
                    return add_duplicate_account(
                        account_ptr,
                        tmp_account_ptr,
                        flag);
                } else {
                    return add_new_account(
                        arith,
                        address,
                        account_ptr,
                        flag);
                }
            }
        public:
            __host__ __device__ AccessState() : _state(nullptr), _world_state(nullptr) {}
            __host__ __device__ AccessState(state_access_t* state, WorldState* world_state) : _state(state), _world_state(world_state) {}
            __host__ __device__ ~AccessState() {
                _state = nullptr;
                _world_state = nullptr;
            }
            
            __host__ __device__ int32_t get_account(
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

            __host__ __device__ int32_t get_value(
                ArithEnv &arith,
                const bn_t &address,
                const bn_t &key,
                bn_t &value,
                int32_t with_transfer = 1
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
                    if (with_transfer) {
                        account_ptr->set_storage_value(arith, key, value);
                    }
                }
                return 1;
            }

            __host__ __device__ int32_t is_warm_account(
                ArithEnv &arith,
                const bn_t &address
            ) {
                cuEVM::account::account_t* account_ptr = nullptr;
                if (_state->get_account(arith, address, account_ptr, ACCOUNT_NONE_FLAG)) {
                    return 1;
                }
                return 0;
            }

            __host__ __device__ int32_t is_warm_key(
                ArithEnv &arith,
                const bn_t &address,
                const bn_t &key
            ) {
                cuEVM::account::account_t* account_ptr = nullptr;
                if (_state->get_account(arith, address, account_ptr, ACCOUNT_STORAGE_FLAG)) {
                    bn_t value;
                    return account_ptr->get_storage_value(arith, key, value);
                }
                return 0;
            }
        };
    }
}


#endif