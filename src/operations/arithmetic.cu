// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Date: 2024-07-15
// SPDX-License-Identifier: MIT

#include "../include/operations/arithmetic.cuh"
#include "../include/gas_cost.cuh"
#include "../include/utils/error_codes.cuh"

/**
 * The arithmetic operations.
 * Contains the arithmetic operations 0s: Arithmetic Operations:
 * - ADD
 * - MUL
 * - SUB
 * - DIV
 * - SDIV
 * - MOD
 * - SMOD
 * - ADDMOD
 * - MULMOD
 * - EXP
 * - SIGNEXTEND
 */
namespace cuEVM {
    namespace operations {
        /**
         * The ADD operation implementation.
         * Takes two values from the stack, adds them and
         * pushes the result back to the stack.
         * @param[in] arith The arithmetical environment.
         * @param[in] gas_limit The gas limit.
         * @param[inout] gas_used The gas used.
         * @param[inout] pc The program counter.
         * @param[inout] stack The stack.
         * @return The error code. 0 if no error.
        */
        __host__ __device__ int32_t ADD(
            ArithEnv &arith,
            const bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &pc,
            cuEVM::stack::evm_stack_t &stack)
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

                pc = pc + 1;
            }
            return error_code;
        }

        /**
         * The MUL operation implementation.
         * Takes two values from the stack, multiplies them and
         * pushes the result back to the stack.
         * @param[in] arith The arithmetical environment.
         * @param[in] gas_limit The gas limit.
         * @param[inout] gas_used The gas used.
         * @param[inout] pc The program counter.
         * @param[inout] stack The stack.
         * @return The error code. 0 if no error.
        */
        __host__ __device__ int32_t MUL(
            ArithEnv &arith,
            const bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &pc,
            cuEVM::stack::evm_stack_t &stack)
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

                pc = pc + 1;
            }
            return error_code;
        }

        /**
         * The SUB operation implementation.
         * Takes two values from the stack, subtracts the second value
         * from the first value and pushes the result back to the stack.
         * @param[in] arith The arithmetical environment.
         * @param[in] gas_limit The gas limit.
         * @param[inout] gas_used The gas used.
         * @param[inout] pc The program counter.
         * @param[inout] stack The stack.
         * @return The error code. 0 if no error.
        */
        __host__ __device__ int32_t SUB(
            ArithEnv &arith,
            const bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &pc,
            cuEVM::stack::evm_stack_t &stack)
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

                pc = pc + 1;
            }
            return error_code;
        }

        /**
         * The DIV operation implementation.
         * Takes two values from the stack, divides the first value
         * by the second value and pushes the result back to the stack.
         * It consider the division by zero as zero.
         * Both values are considered unsigned.
         * @param[in] arith The arithmetical environment.
         * @param[in] gas_limit The gas limit.
         * @param[inout] gas_used The gas used.
         * @param[inout] pc The program counter.
         * @param[inout] stack The stack.
         * @return The error code. 0 if no error.
        */
        __host__ __device__ int32_t DIV(
            ArithEnv &arith,
            const bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &pc,
            cuEVM::stack::evm_stack_t &stack)
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

                pc = pc + 1;
            }
            return error_code;
        }

        /**
         * The SDIV operation implementation.
         * Takes two values from the stack, divides the first value
         * by the second value and pushes the result back to the stack.
         * It consider the division by zero as zero.
         * The special case -2^254 / -1 = -2^254 is considered.
         * Both values are considered signed.
         * @param[in] arith The arithmetical environment.
         * @param[in] gas_limit The gas limit.
         * @param[inout] gas_used The gas used.
         * @param[inout] pc The program counter.
         * @param[inout] stack The stack.
         * @return The error code. 0 if no error.
        */
        __host__ __device__ int32_t SDIV(
            ArithEnv &arith,
            const bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &pc,
            cuEVM::stack::evm_stack_t &stack)
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

                pc = pc + 1;
            }
            return error_code;
        }

        /**
         * The MOD operation implementation.
         * Takes two values from the stack, calculates the remainder
         * of the division of the first value by the second value and
         * pushes the result back to the stack.
         * It consider the remainder of division by zero as zero.
         * Both values are considered unsigned.
         * @param[in] arith The arithmetical environment.
         * @param[in] gas_limit The gas limit.
         * @param[inout] gas_used The gas used.
         * @param[inout] pc The program counter.
         * @param[inout] stack The stack.
         * @return The error code. 0 if no error.
        */
        __host__ __device__ int32_t MOD(
            ArithEnv &arith,
            const bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &pc,
            cuEVM::stack::evm_stack_t &stack)
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

                pc = pc + 1;
            }
            return error_code;
        }

        /**
         * The SMOD operation implementation.
         * Takes two values from the stack, calculates the remainder
         * of the division of the first value by the second value and
         * pushes the result back to the stack.
         * It consider the remainder of division by zero as zero.
         * Both values are considered signed.
         * @param[in] arith The arithmetical environment.
         * @param[in] gas_limit The gas limit.
         * @param[inout] gas_used The gas used.
         * @param[inout] pc The program counter.
         * @param[inout] stack The stack.
         * @return The error code. 0 if no error.
        */
        __host__ __device__ int32_t SMOD(
            ArithEnv &arith,
            const bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &pc,
            cuEVM::stack::evm_stack_t &stack)
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

                pc = pc + 1;
            }
            return error_code;
        }

        /**
         * The ADDMOD operation implementation.
         * Takes three values from the stack, adds the first two values,
         * calculates the remainder of the division of the result by the third value
         * and pushes the remainder back to the stack.
         * It consider the remainder of division by zero or one as zero.
         * All values are considered unsigned.
         * @param[in] arith The arithmetical environment.
         * @param[in] gas_limit The gas limit.
         * @param[inout] gas_used The gas used.
         * @param[inout] pc The program counter.
         * @param[inout] stack The stack.
         * @return The error code. 0 if no error.
        */
        __host__ __device__ int32_t ADDMOD(
            ArithEnv &arith,
            const bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &pc,
            cuEVM::stack::evm_stack_t &stack)
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

                pc = pc + 1;
            }
            return error_code;
        }

        /**
         * The MULMOD operation implementation.
         * Takes three values from the stack, multiplies the first two values,
         * calculates the remainder of the division of the result by the third value
         * and pushes the remainder back to the stack.
         * It consider the remainder of division by zero or one as zero.
         * All values are considered unsigned.
         * The first two values goes though a modulo by the third value
         * before multiplication.
         * @param[in] arith The arithmetical environment.
         * @param[in] gas_limit The gas limit.
         * @param[inout] gas_used The gas used.
         * @param[inout] pc The program counter.
         * @param[inout] stack The stack.
         * @return The error code. 0 if no error.
        */
        __host__ __device__ int32_t MULMOD(
            ArithEnv &arith,
            const bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &pc,
            cuEVM::stack::evm_stack_t &stack)
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

                pc = pc + 1;
            }
            return error_code;
        }

        /**
         * The EXP operation implementation.
         * Takes two values from the stack, calculates the first value
         * to the power of the second value and pushes the result back to the stack.
         * It consider the power of zero as one, even if the base is zero.
         * The dynamic gas cost is calculated based on the minimumu number of bytes
         * to store the exponent value.
         * @param[in] arith The arithmetical environment.
         * @param[inout] gas_limit The gas limit.
         * @param[inout] gas_used The gas used.
         * @param[inout] pc The program counter.
         * @param[inout] stack The stack.
         * @return The error code. 0 if no error.
        */
        __host__ __device__ int32_t EXP(
            ArithEnv &arith,
            const bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &pc,
            cuEVM::stack::evm_stack_t &stack)
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
                    // dynamic gas calculation (G_expbyte * bytes_in_exponent)
                    int32_t last_bit;
                    last_bit = cuEVM::word_bits - 1 - cgbn_clz(arith.env, exponent);
                    uint32_t exponent_byte_size;
                    if (last_bit == -1)
                    {
                        exponent_byte_size = 0;
                    }
                    else
                    {
                        exponent_byte_size = (last_bit) / 8 + 1;
                    }
                    bn_t dynamic_gas;
                    cgbn_set_ui32(arith.env, dynamic_gas, exponent_byte_size);
                    cgbn_mul_ui32(arith.env, dynamic_gas, dynamic_gas, GAS_EXP_BYTE);
                    cgbn_add(arith.env, gas_used, gas_used, dynamic_gas);
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

                        pc = pc + 1;
                    }
                }
            }
            return error_code;
        }

        /**
         * The SIGNEXTEND operation implementation.
         * Takes two values from the stack. It consider the second value
         * to have the size of the number of bytes given by the first value (b) + 1.
         * The operation sign extends the second value to the full size of the
         * aithmetic environment and pushes the result back to the stack.
         * In case the first value is out of range ((b+1) > BYTES) the operation
         * pushes the second value back to the stack.
         * If the second value has more bytes than the value (b+1),
         * the operation consider only the least significant (b+1) bytes
         * of the second value.
         * @param[in] arith The arithmetical environment.
         * @param[inout] gas_limit The gas limit.
         * @param[inout] gas_used The gas used.
         * @param[inout] pc The program counter.
         * @param[inout] stack The stack.
         * @return The error code. 0 if no error.
        */
        __host__ __device__ int32_t SIGNEXTEND(
            ArithEnv &arith,
            const bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &pc,
            cuEVM::stack::evm_stack_t &stack) {
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

                pc = pc + 1;
            }
            return error_code;
        }
                
    }
}