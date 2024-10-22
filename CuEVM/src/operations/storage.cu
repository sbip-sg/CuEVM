// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation
// Programme) Author: Stefan-Dan Ciocirlan Data: 2023-11-30
// SPDX-License-Identifier: MIT

#include <CuEVM/gas_cost.cuh>
#include <CuEVM/operations/storage.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM::operations {
__host__ __device__ int32_t SLOAD(ArithEnv &arith, const bn_t &gas_limit, bn_t &gas_used, CuEVM::evm_stack_t &stack,
                                  CuEVM::TouchState &touch_state, const CuEVM::evm_message_call_t &message) {
    // cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_ZERO);
    bn_t key;
    int32_t error_code = stack.pop(arith, key);
    bn_t storage_address;
    message.get_storage_address(arith, storage_address);
    error_code |= CuEVM::gas_cost::sload_cost(arith, gas_used, touch_state, storage_address, key);
    error_code |= CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);
#ifdef __CUDA_ARCH__
    printf("SLOAD %d error_code: %d\n", threadIdx.x, error_code);
#endif
    if (error_code == ERROR_SUCCESS) {
        bn_t value;
        error_code |= touch_state.get_value(arith, storage_address, key, value);
        error_code |= stack.push(arith, value);
    }
    return error_code;
}

__host__ __device__ int32_t SSTORE(ArithEnv &arith, const bn_t &gas_limit, bn_t &gas_used, bn_t &gas_refund,
                                   CuEVM::evm_stack_t &stack, CuEVM::TouchState &touch_state,
                                   const CuEVM::evm_message_call_t &message) {
    // only if is not a static call
    int32_t error_code = (message.get_static_env() ? ERROR_STATIC_CALL_CONTEXT_SSTORE : ERROR_SUCCESS);
    // cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_ZERO);
    bn_t gas_left;
    cgbn_sub(arith.env, gas_left, gas_limit, gas_used);
    error_code |= (cgbn_compare_ui32(arith.env, gas_left, GAS_STIPEND) < 0 ? ERROR_OUT_OF_GAS : error_code);
    if (error_code != ERROR_SUCCESS) {
        return error_code;
    }
    bn_t key;
    error_code |= stack.pop(arith, key);
    bn_t value;
    error_code |= stack.pop(arith, value);
    bn_t storage_address;
    message.get_storage_address(arith, storage_address);
    error_code |= CuEVM::gas_cost::sstore_cost(arith, gas_used, gas_refund, touch_state, storage_address, key, value);
    error_code |= CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);
    return (error_code ? error_code : touch_state.set_storage_value(arith, storage_address, key, value));
}
}  // namespace CuEVM::operations