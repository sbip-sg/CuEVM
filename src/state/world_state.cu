// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2024-06-20
// SPDX-License-Identifier: MIT
#include "../include/state/world_state.cuh"

namespace cuEVM {
    namespace state {
        __host__ __device__ int32_t WorldState::get_account(
            ArithEnv &arith,
            const bn_t &address,
            cuEVM::account::account_t* &account_ptr
        ) {
            return _state->get_account(arith, address, account_ptr);
        }

        __host__ __device__ int32_t WorldState::get_value(
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
    }
}
