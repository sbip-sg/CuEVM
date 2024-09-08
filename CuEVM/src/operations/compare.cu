// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Date: 2023-07-15
// SPDX-License-Identifier: MIT

#include <CuEVM/operations/compare.cuh>
#include <CuEVM/gas_cost.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM::operations {
    /**
     * Compare the top two values from the stack.
     * The two values are considered unsigned.
     * -1 if the first value is less than the second value,
     * 0 if the first value is equal to the second value,
     * 1 if the first value is greater than the second value.
     * @param[in] arith The arithmetical environment.
     * @param[inout] stack The stack.
     * @param[out] result The result of the comparison.
     * @return 0 if the operation was successful, an error code otherwise.
    */
    __host__ __device__ int32_t compare(
        ArithEnv &arith,
        CuEVM::evm_stack_t &stack,
        int32_t &result) {
        bn_t a, b;
        int32_t error_code = stack.pop(arith, a);
        error_code |= stack.pop(arith, b);
        result = cgbn_compare(arith.env, a, b);
        return error_code;
    }

    /**
     * Compare the top two values from the stack.
     * The two values are considered signed.
     * -1 if the first value is less than the second value,
     * 0 if the first value is equal to the second value,
     * 1 if the first value is greater than the second value.
     * @param[in] arith The arithmetical environment.
     * @param[inout] stack The stack.
     * @param[out] result The result of the comparison.
     * @return 0 if the operation was successful, an error code otherwise.
    */
    __host__ __device__ int32_t scompare(
        ArithEnv &arith,
        CuEVM::evm_stack_t &stack,
        int32_t &result) {
        bn_t a, b;
        int32_t error_code = stack.pop(arith, a);
        error_code |= stack.pop(arith, b);

        uint32_t sign_a = cgbn_extract_bits_ui32(arith.env, a, CuEVM::word_bits - 1, 1);
        uint32_t sign_b = cgbn_extract_bits_ui32(arith.env, b, CuEVM::word_bits - 1, 1);
        result = (sign_a == 0 && sign_b == 1) ? 1 : (sign_a == 1 && sign_b == 0) ? -1 : cgbn_compare(arith.env, a, b);
        return error_code;
    }

    __host__ __device__ int32_t LT(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        CuEVM::evm_stack_t &stack)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_VERY_LOW);
        int32_t error_code = CuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        if (error_code == ERROR_SUCCESS)
        {

            int32_t int_result;
            error_code |= compare(
                arith,
                stack,
                int_result);
            uint32_t result = (int_result < 0) ? 1 : 0;
            bn_t r;

            cgbn_set_ui32(arith.env, r, result);

            error_code |= stack.push(arith, r);
        }
        return error_code;
    }

    __host__ __device__ int32_t GT(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        CuEVM::evm_stack_t &stack)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_VERY_LOW);
        int32_t error_code = CuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        if (error_code == ERROR_SUCCESS)
        {

            int32_t int_result;
            error_code |= compare(
                arith,
                stack,
                int_result);
            uint32_t result = (int_result > 0) ? 1 : 0;
            bn_t r;

            cgbn_set_ui32(arith.env, r, result);

            error_code |= stack.push(arith, r);
        }
        return error_code;
   }

 
    __host__ __device__ int32_t SLT(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        CuEVM::evm_stack_t &stack)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_VERY_LOW);
        int32_t error_code = CuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        if (error_code == ERROR_SUCCESS)
        {

            int32_t int_result;
            error_code |= scompare(
                arith,
                stack,
                int_result);
            uint32_t result = (int_result < 0) ? 1 : 0;
            bn_t r;

            cgbn_set_ui32(arith.env, r, result);

            error_code |= stack.push(arith, r);
        }
        return error_code;
    }

    __host__ __device__ int32_t SGT(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        CuEVM::evm_stack_t &stack)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_VERY_LOW);
        int32_t error_code = CuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        if (error_code == ERROR_SUCCESS)
        {

            int32_t int_result;
            error_code |= scompare(
                arith,
                stack,
                int_result);
            uint32_t result = (int_result > 0) ? 1 : 0;
            bn_t r;

            cgbn_set_ui32(arith.env, r, result);

            error_code |= stack.push(arith, r);
        }
        return error_code;
    }

    __host__ __device__ int32_t EQ(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        CuEVM::evm_stack_t &stack)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_VERY_LOW);
        int32_t error_code = CuEVM::gas_cost::has_gas(
                arith,
                gas_limit,
                gas_used);
            if (error_code == ERROR_SUCCESS)
        {

            int32_t int_result;
            error_code |= compare(
                arith,
                stack,
                int_result);
            uint32_t result = (int_result == 0) ? 1 : 0;
            bn_t r;

            cgbn_set_ui32(arith.env, r, result);

            error_code |= stack.push(arith, r);
        }
        return error_code;
    }

    __host__ __device__ int32_t ISZERO(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        CuEVM::evm_stack_t &stack)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_VERY_LOW);
        int32_t error_code = CuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        if (error_code == ERROR_SUCCESS)
        {
            bn_t a;
            error_code |= stack.pop(arith, a);
            bn_t r;

            int32_t compare = cgbn_compare_ui32(arith.env, a, 0);
            if (compare == 0)
            {
                cgbn_set_ui32(arith.env, r, 1);
            }
            else
            {
                cgbn_set_ui32(arith.env, r, 0);
            }

            error_code |= stack.push(arith, r);
        }
        return error_code;
    }
} // namespace CuEVM::operations