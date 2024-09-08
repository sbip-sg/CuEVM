// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Date: 2024-07-15
// SPDX-License-Identifier: MIT

#include <CuEVM/operations/arithmetic.cuh>
#include <CuEVM/gas_cost.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace cuEVM {
    namespace operations {
        __host__ __device__ int32_t ADD(
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
                bn_t a, b, r;
                error_code |= stack.pop(arith, a);
                error_code |= stack.pop(arith, b);

                cgbn_add(arith.env, r, a, b);

                error_code |= stack.push(arith, r);
            }
            return error_code;
        }

        __host__ __device__ int32_t MUL(
            ArithEnv &arith,
            const bn_t &gas_limit,
            bn_t &gas_used,
            cuEVM::evm_stack_t &stack)
        {
            cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_LOW);
            int32_t error_code = cuEVM::gas_cost::has_gas(
                arith,
                gas_limit,
                gas_used);
            if (error_code == ERROR_SUCCESS)
            {
                bn_t a, b, r;
                error_code |= stack.pop(arith, a);
                error_code |= stack.pop(arith, b);

                cgbn_mul(arith.env, r, a, b);

                error_code |= stack.push(arith, r);
            }
            return error_code;
        }

        __host__ __device__ int32_t SUB(
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
                bn_t a, b, r;
                error_code |= stack.pop(arith, a);
                error_code |= stack.pop(arith, b);

                cgbn_sub(arith.env, r, a, b);

                error_code |= stack.push(arith, r);
            }
            return error_code;
        }

        __host__ __device__ int32_t DIV(
            ArithEnv &arith,
            const bn_t &gas_limit,
            bn_t &gas_used,
            cuEVM::evm_stack_t &stack)
        {
            cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_LOW);
            int32_t error_code = cuEVM::gas_cost::has_gas(
                arith,
                gas_limit,
                gas_used);
            if (error_code == ERROR_SUCCESS)
            {
                bn_t a, b, r;
                error_code |= stack.pop(arith, a);
                error_code |= stack.pop(arith, b);

                // division by zero no error
                if (cgbn_compare_ui32(arith.env, b, 0) == 0)
                    cgbn_set_ui32(arith.env, r, 0);
                else
                    cgbn_div(arith.env, r, a, b);

                error_code |= stack.push(arith, r);
            }
            return error_code;
        }

        __host__ __device__ int32_t SDIV(
            ArithEnv &arith,
            const bn_t &gas_limit,
            bn_t &gas_used,
            cuEVM::evm_stack_t &stack)
        {
            cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_LOW);
            int32_t error_code = cuEVM::gas_cost::has_gas(
                arith,
                gas_limit,
                gas_used);
            if (error_code == ERROR_SUCCESS)
            {
                bn_t a, b, r;
                error_code |= stack.pop(arith, a);
                error_code |= stack.pop(arith, b);

                bn_t d;
                bn_t e;
                // d = -1
                cgbn_set_ui32(arith.env, d, 0);
                cgbn_sub_ui32(arith.env, d, d, 1);
                // e = -2^254
                cgbn_set_ui32(arith.env, e, 1);
                cgbn_shift_left(arith.env, e, e, cuEVM::word_bits - 1);
                uint32_t sign_a = cgbn_extract_bits_ui32(arith.env, a, cuEVM::word_bits - 1, 1);
                uint32_t sign_b = cgbn_extract_bits_ui32(arith.env, b, cuEVM::word_bits - 1, 1);
                uint32_t sign = sign_a ^ sign_b;
                // division by zero no error
                if (cgbn_compare_ui32(arith.env, b, 0) == 0)
                    cgbn_set_ui32(arith.env, r, 0);
                else if ( // -2^254 / -1 = -2^254
                    (cgbn_compare(arith.env, b, d) == 0) &&
                    (cgbn_compare(arith.env, a, e) == 0))
                {
                    cgbn_set(arith.env, r, e);
                }
                else
                {
                    // div between absolute values
                    if (sign_a == 1)
                    {
                        cgbn_negate(arith.env, a, a);
                    }
                    if (sign_b == 1)
                    {
                        cgbn_negate(arith.env, b, b);
                    }
                    cgbn_div(arith.env, r, a, b);
                    if (sign)
                    {
                        cgbn_negate(arith.env, r, r);
                    }
                }

                error_code |= stack.push(arith, r);
            }
            return error_code;
        }

        __host__ __device__ int32_t MOD(
            ArithEnv &arith,
            const bn_t &gas_limit,
            bn_t &gas_used,
            cuEVM::evm_stack_t &stack)
        {
            cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_LOW);
            int32_t error_code = cuEVM::gas_cost::has_gas(
                arith,
                gas_limit,
                gas_used);
            if (error_code == ERROR_SUCCESS)
            {
                bn_t a, b, r;
                error_code |= stack.pop(arith, a);
                error_code |= stack.pop(arith, b);

                // // rem by zero no error
                if (cgbn_compare_ui32(arith.env, b, 0) == 0)
                    cgbn_set_ui32(arith.env, r, 0);
                else
                    cgbn_rem(arith.env, r, a, b);

                error_code |= stack.push(arith, r);
            }
            return error_code;
        }

        __host__ __device__ int32_t SMOD(
            ArithEnv &arith,
            const bn_t &gas_limit,
            bn_t &gas_used,
            cuEVM::evm_stack_t &stack)
        {
            cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_LOW);
            int32_t error_code = cuEVM::gas_cost::has_gas(
                arith,
                gas_limit,
                gas_used);
            if (error_code == ERROR_SUCCESS)
            {
                bn_t a, b, r;
                error_code |= stack.pop(arith, a);
                error_code |= stack.pop(arith, b);

                uint32_t sign_a = cgbn_extract_bits_ui32(arith.env, a, cuEVM::word_bits - 1, 1);
                uint32_t sign_b = cgbn_extract_bits_ui32(arith.env, b, cuEVM::word_bits - 1, 1);
                uint32_t sign = sign_a ^ sign_b;
                if (cgbn_compare_ui32(arith.env, b, 0) == 0)
                    cgbn_set_ui32(arith.env, r, 0);
                else
                {
                    // mod between absolute values
                    if (sign_a == 1)
                    {
                        cgbn_negate(arith.env, a, a);
                    }
                    if (sign_b == 1)
                    {
                        cgbn_negate(arith.env, b, b);
                    }
                    cgbn_rem(arith.env, r, a, b);

                    // twos-complement if first number is negative
                    if (sign_a)
                    {
                        cgbn_negate(arith.env, r, r);
                    }
                }

                error_code |= stack.push(arith, r);
            }
            return error_code;
        }

        __host__ __device__ int32_t ADDMOD(
            ArithEnv &arith,
            const bn_t &gas_limit,
            bn_t &gas_used,
            cuEVM::evm_stack_t &stack)
        {
            cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_MID);
            int32_t error_code = cuEVM::gas_cost::has_gas(
                arith,
                gas_limit,
                gas_used);
            if (error_code == ERROR_SUCCESS)
            {
                bn_t a, b, c, N, r;
                error_code |= stack.pop(arith, a);
                error_code |= stack.pop(arith, b);
                error_code |= stack.pop(arith, N);

                if (cgbn_compare_ui32(arith.env, N, 0) == 0)
                {
                    cgbn_set_ui32(arith.env, r, 0);
                }
                else if (cgbn_compare_ui32(arith.env, N, 1) == 0)
                {
                    cgbn_set_ui32(arith.env, r, 0);
                }
                else
                {
                    int32_t carry = cgbn_add(arith.env, c, a, b);
                    bn_wide_t d;
                    if (carry == 1)
                    {
                        cgbn_set_ui32(arith.env, d._high, 1);
                        cgbn_set(arith.env, d._low, c);
                        cgbn_rem_wide(arith.env, r, d, N);
                    }
                    else
                    {
                        cgbn_rem(arith.env, r, c, N);
                    }
                }

                error_code |= stack.push(arith, r);
            }
            return error_code;
        }

        __host__ __device__ int32_t MULMOD(
            ArithEnv &arith,
            const bn_t &gas_limit,
            bn_t &gas_used,
            cuEVM::evm_stack_t &stack)
        {
            cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_MID);
            int32_t error_code = cuEVM::gas_cost::has_gas(
                arith,
                gas_limit,
                gas_used);
            if (error_code == ERROR_SUCCESS)
            {
                bn_t a, b, N, r;
                error_code |= stack.pop(arith, a);
                error_code |= stack.pop(arith, b);
                error_code |= stack.pop(arith, N);

                if (cgbn_compare_ui32(arith.env, N, 0) == 0)
                {
                    cgbn_set_ui32(arith.env, r, 0);
                }
                else
                {
                    bn_wide_t d;
                    cgbn_rem(arith.env, a, a, N);
                    cgbn_rem(arith.env, b, b, N);
                    cgbn_mul_wide(arith.env, d, a, b);
                    cgbn_rem_wide(arith.env, r, d, N);
                }

                error_code |= stack.push(arith, r);
            }
            return error_code;
        }

        __host__ __device__ int32_t EXP(
            ArithEnv &arith,
            const bn_t &gas_limit,
            bn_t &gas_used,
            cuEVM::evm_stack_t &stack)
        {
            cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_EXP);
            int32_t error_code = cuEVM::gas_cost::has_gas(
                arith,
                gas_limit,
                gas_used);
            if (error_code == ERROR_SUCCESS) {
                bn_t a, exponent, r;
                error_code |= stack.pop(arith, a);
                error_code |= stack.pop(arith, exponent);

                if (error_code == ERROR_SUCCESS) {
                    int32_t last_bit = cuEVM::gas_cost::exp_bytes_gas_cost(
                        arith,
                        gas_used,
                        exponent);
                    error_code |= cuEVM::gas_cost::has_gas(
                        arith,
                        gas_limit,
                        gas_used);
                    if (error_code == ERROR_SUCCESS) {
                        //^0=1 even for 0^0
                        if (last_bit == -1)
                        {
                            cgbn_set_ui32(arith.env, r, 1);
                        }
                        else
                        {
                            bn_t current, square;
                            cgbn_set_ui32(arith.env, current, 1); // r=1
                            cgbn_set(arith.env, square, a);       // square=a
                            for (int32_t bit = 0; bit <= last_bit; bit++)
                            {
                                if (cgbn_extract_bits_ui32(arith.env, exponent, bit, 1) == 1)
                                {
                                    cgbn_mul(arith.env, current, current, square); // r=r*square
                                }
                                cgbn_mul(arith.env, square, square, square); // square=square*square
                            }
                            cgbn_set(arith.env, r, current);
                        }

                        error_code |= stack.push(arith, r);
                    }
                }
            }
            return error_code;
        }

        __host__ __device__ int32_t SIGNEXTEND(
            ArithEnv &arith,
            const bn_t &gas_limit,
            bn_t &gas_used,
            cuEVM::evm_stack_t &stack) {
            /*
            Even if x has more bytes than the value b, the operation consider only the first
            (b+1) bytes of x and the other are considered zero and they don't have any influence
            on the final result.
            Optimised: use cgbn_bitwise_mask_ior instead of cgbn_insert_bits_ui32
            */
            cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_LOW);
            int32_t error_code = cuEVM::gas_cost::has_gas(
                arith,
                gas_limit,
                gas_used);
            if (error_code == ERROR_SUCCESS) {
                bn_t b, x, r;
                error_code |= stack.pop(arith, b);
                error_code |= stack.pop(arith, x);

                if (cgbn_compare_ui32(arith.env, b, (cuEVM::word_size - 1) ) == 1) {
                    cgbn_set(arith.env, r, x);
                } else {
                    uint32_t c = cgbn_get_ui32(arith.env, b) + 1;
                    uint32_t sign = cgbn_extract_bits_ui32(arith.env, x, c * 8 - 1, 1);
                    int32_t numbits = int32_t(c);
                    if (sign == 1)
                    {
                        numbits = int32_t(cuEVM::word_bits) - 8 * numbits;
                        numbits = -numbits;
                        cgbn_bitwise_mask_ior(arith.env, r, x, numbits);
                    }
                    else
                    {
                        cgbn_bitwise_mask_and(arith.env, r, x, 8 * numbits);
                    }
                }

                error_code |= stack.push(arith, r);
            }
            return error_code;
        }
                
    }
}