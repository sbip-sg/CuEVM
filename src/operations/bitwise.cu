// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Date: 2023-07-15
// SPDX-License-Identifier: MIT

#include "../include/operations/bitwise.cuh"
#include "../include/gas_cost.cuh"
#include "../include/utils/error_codes.cuh"

namespace cuEVM::operations {
    __host__ __device__ int32_t AND(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        cuEVM::evm_stack_t &stack)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_VERY_LOW);
        int32_t error_code = cuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        if (error_code == ERROR_SUCCESS)
        {
            bn_t a, b;
            error_code |= stack.pop(arith, a);
            error_code |= stack.pop(arith, b);
            bn_t r;

            cgbn_bitwise_and(arith.env, r, a, b);

            error_code |= stack.push(arith, r);
        }
        return error_code;
    }
    __host__ __device__ int32_t OR(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        cuEVM::evm_stack_t &stack)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_VERY_LOW);
        int32_t error_code = cuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        if (error_code == ERROR_SUCCESS)
        {
            bn_t a, b;
            error_code |= stack.pop(arith, a);
            error_code |= stack.pop(arith, b);
            bn_t r;

            cgbn_bitwise_ior(arith.env, r, a, b);

            error_code |= stack.push(arith, r);
        }
        return error_code;
    }

    __host__ __device__ int32_t XOR(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        cuEVM::evm_stack_t &stack)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_VERY_LOW);
        int32_t error_code = cuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        if (error_code == ERROR_SUCCESS)
        {
            bn_t a, b;
            error_code |= stack.pop(arith, a);
            error_code |= stack.pop(arith, b);
            bn_t r;

            cgbn_bitwise_xor(arith.env, r, a, b);

            error_code |= stack.push(arith, r);
        }
        return error_code;
    }

    __host__ __device__ int32_t NOT(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        cuEVM::evm_stack_t &stack)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_VERY_LOW);
        int32_t error_code = cuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        if (error_code == ERROR_SUCCESS)
        {
            bn_t a;
            error_code |=  stack.push(arith, a);
            bn_t r;

            cgbn_bitwise_mask_xor(arith.env, r, a, cuEVM::word_bits);

            error_code |= stack.push(arith, r);
        }
        return error_code;
    }

    __host__ __device__ int32_t BYTE(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        cuEVM::evm_stack_t &stack)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_VERY_LOW);
        int32_t error_code = cuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        if (error_code == ERROR_SUCCESS)
        {
            bn_t i, x;
            error_code |= stack.pop(arith, i);
            error_code |= stack.pop(arith, x);
            bn_t r;

            if (cgbn_compare_ui32(arith.env, i, (cuEVM::word_size-1)) == 1)
            {
                cgbn_set_ui32(arith.env, r, 0);
            }
            else
            {
                uint32_t index = cgbn_get_ui32(arith.env, i);
                uint32_t byte = cgbn_extract_bits_ui32(arith.env, x, 8 * ((cuEVM::word_size - 1) - index), 8);
                cgbn_set_ui32(arith.env, r, byte);
            }

            error_code |= stack.push(arith, r);
        }
        return error_code;
    }

    __host__ __device__ int32_t SHL(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        cuEVM::evm_stack_t &stack)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_VERY_LOW);
        int32_t error_code = cuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        if (error_code == ERROR_SUCCESS)
        {
            bn_t shift, value;
            error_code |= stack.pop(arith, shift);
            error_code |= stack.pop(arith, value);
            bn_t r;

            if (cgbn_compare_ui32(arith.env, shift, cuEVM::word_bits - 1) == 1)
            {
                cgbn_set_ui32(arith.env, r, 0);
            }
            else
            {
                uint32_t shift_left = cgbn_get_ui32(arith.env, shift);
                cgbn_shift_left(arith.env, r, value, shift_left);
            }

            error_code |= stack.push(arith, r);
        }
        return error_code;
    }

    __host__ __device__ int32_t SHR(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        cuEVM::evm_stack_t &stack)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_VERY_LOW);
        int32_t error_code = cuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        if (error_code == ERROR_SUCCESS)
        {
            bn_t shift, value;
            error_code |= stack.pop(arith, shift);
            error_code |= stack.pop(arith, value);
            bn_t r;

            if (cgbn_compare_ui32(arith.env, shift, cuEVM::word_bits - 1) == 1)
            {
                cgbn_set_ui32(arith.env, r, 0);
            }
            else
            {
                uint32_t shift_right = cgbn_get_ui32(arith.env, shift);
                cgbn_shift_right(arith.env, r, value, shift_right);
            }

            error_code |= stack.push(arith, r);
        }
        return error_code;
    }

    __host__ __device__ int32_t SAR(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        cuEVM::evm_stack_t &stack)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_VERY_LOW);
        int32_t error_code = cuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        if (error_code == ERROR_SUCCESS)
        {
            bn_t shift, value;
            error_code |= stack.pop(arith, shift);
            error_code |= stack.pop(arith, value);
            bn_t r;

            uint32_t sign_b = cgbn_extract_bits_ui32(arith.env, value, cuEVM::word_bits - 1, 1);
            uint32_t shift_right = cgbn_get_ui32(arith.env, shift);

            if (cgbn_compare_ui32(arith.env, shift, cuEVM::word_bits - 1) == 1)
                shift_right = cuEVM::word_bits;

            cgbn_shift_right(arith.env, r, value, shift_right);
            if (sign_b == 1)
            {
                cgbn_bitwise_mask_ior(arith.env, r, r, -shift_right);
            }

            error_code |= stack.push(arith, r);
        }
        return error_code;
    }
}