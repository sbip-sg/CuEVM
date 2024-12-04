// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Date: 2023-07-15
// SPDX-License-Identifier: MIT

#include <CuEVM/gas_cost.cuh>
#include <CuEVM/operations/bitwise.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM::operations {
__host__ __device__ int32_t AND(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t a, b;
        error_code |= stack.pop(a);
        error_code |= stack.pop(b);
        evm_word_t r;

        uint256_bitwise_and(&r, &a, &b);

        error_code |= stack.push(r);
    }
    return error_code;
}
__host__ __device__ int32_t OR(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t a, b;
        error_code |= stack.pop(a);
        error_code |= stack.pop(b);
        evm_word_t r;

        uint256_bitwise_or(&r, &a, &b);

        error_code |= stack.push(r);
    }
    return error_code;
}

__host__ __device__ int32_t XOR(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t a, b;
        error_code |= stack.pop(a);
        error_code |= stack.pop(b);
        evm_word_t r;

        uint256_bitwise_xor(&r, &a, &b);

        error_code |= stack.push(r);
    }
    return error_code;
}

__host__ __device__ int32_t NOT(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t a;
        error_code |= stack.pop(a);
        evm_word_t r;

        uint256_bitwise_not(&r, &a);

        error_code |= stack.push(r);
    }
    return error_code;
}

__host__ __device__ int32_t BYTE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t i, x;
        error_code |= stack.pop(i);
        error_code |= stack.pop(x);
        evm_word_t r;
        /*
            if (cgbn_compare_ui32(arith.env, i, (CuEVM::word_size - 1)) == 1) {
                cgbn_set_ui32(arith.env, r, 0);
            } else {
                uint32_t index = cgbn_get_ui32(arith.env, i);
                uint32_t byte = cgbn_extract_bits_ui32(arith.env, x, 8 * ((CuEVM::word_size - 1) - index), 8);
                cgbn_set_ui32(arith.env, r, byte);
            }
    */
        error_code |= stack.push(r);
    }
    return error_code;
}

__host__ __device__ int32_t SHL(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t shift, value;
        error_code |= stack.pop(shift);
        error_code |= stack.pop(value);
        evm_word_t r;

        if (uint256_cmp_word(&shift, CuEVM::word_bits - 1) == 1) {
            r = 0;
        } else {
            uint32_t shift_left = uint256_get_uint32_t(&shift);
            uint256_shift_left(&r, &value, shift_left);
        }

        error_code |= stack.push(r);
    }
    return error_code;
}

__host__ __device__ int32_t SHR(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t shift, value;
        error_code |= stack.pop(shift);
        error_code |= stack.pop(value);
        evm_word_t r;

        if (uint256_cmp_word(&shift, CuEVM::word_bits - 1) == 1) {
            r = 0;
        } else {
            uint32_t shift_right = uint256_get_uint32_t(&shift);
            uint256_shift_right(&r, &value, shift_right);
        }

        error_code |= stack.push(r);
    }
    return error_code;
}

__host__ __device__ int32_t SAR(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t shift, value;
        error_code |= stack.pop(shift);
        error_code |= stack.pop(value);
        evm_word_t r;
        /*
        uint32_t sign_b = cgbn_extract_bits_ui32(arith.env, value, CuEVM::word_bits - 1, 1);
        uint32_t shift_right = cgbn_get_ui32(arith.env, shift);

        if (cgbn_compare_ui32(arith.env, shift, CuEVM::word_bits - 1) == 1) shift_right = CuEVM::word_bits;

        cgbn_shift_right(arith.env, r, value, shift_right);
        if (sign_b == 1) {
            cgbn_bitwise_mask_ior(arith.env, r, r, -shift_right);
        }
        */
        error_code |= stack.push(r);
    }
    return error_code;
}
}  // namespace CuEVM::operations