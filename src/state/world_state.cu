// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2024-06-20
// SPDX-License-Identifier: MIT
#include "../include/state/world_state.cuh"
#include "../include/utils/error_codes.cuh"

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
            cgbn_set_ui32(arith.env, value, 0);
            return (
                _state->get_account(arith, address, account_ptr) ||
                account_ptr->get_storage_value(arith, key, value)
            );
        }
    }
}
