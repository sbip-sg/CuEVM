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
                cuEVM::account::account_flags_t flag = ACCOUNT_NONE_FLAG
            ) {
                if(_state->get_account(arith, address, account_ptr, flag)) {
                    return 1;
                } else {
                    cuEVM::account::account_t* tmp_account_ptr = nullptr;
                    if(_world_state->get_account(arith, address, tmp_account_ptr)) {
                        cuEVM::account::account_flags_t copy_flag;
                        copy_flag.set_all();
                        copy_flag.unset_storage();
                        cuEVM::account::account_t *new_account = new cuEVM::account::account_t(
                            *tmp_account_ptr,
                            copy_flag
                            );
                    } else {
                        cuEVM::account::account_flags_t new_flag;
                        new_flag.set_all();
                        new_flag.unset_storage();
                        account_ptr = new cuEVM::account::account_t(
                            arith,
                            new_flag
                        );
                    
                    }
                }
            }

            __host__ __device__ int32_t get_value(
                ArithEnv &arith,
                const bn_t &address,
                const bn_t &key,
                bn_t &value
            ) {
                account::account_t* account_ptr = nullptr;
                if (_state->get_account(arith, address, account_ptr)) {
                    return account_ptr->get_storage_value(arith, key, value);
                }
                return 0;
            }

            __host__ __device__ int32_t set_value(
                ArithEnv &arith,
                const bn_t &address,
                const bn_t &key,
                const bn_t &value
            ) {
                account::account_t* account_ptr = nullptr;
                if (_state->get_account(arith, address, account_ptr)) {
                    return account_ptr->set_storage_value(arith, key, value);
                }
                return 0;
            }
        };
    }
}


#endif