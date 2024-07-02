#ifndef _CUEVM_WORLD_STATE_H_
#define _CUEVM_WORLD_STATE_H_

#include "state.cuh"

namespace cuEVM {
    namespace state {
        class WorldState {
        private:
            state_t* _state;
        public:
            __host__ __device__ WorldState() : _state(nullptr) {}
            __host__ __device__ WorldState(state_t* state) : _state(state) {}
            __host__ __device__ ~WorldState() {_state = nullptr;}
            
            __host__ __device__ int32_t get_account(
                ArithEnv &arith,
                const bn_t &address,
                cuEVM::account::account_t* &account_ptr
            ) {
                return _state->get_account(arith, address, account_ptr);
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
        };
    }
}


#endif