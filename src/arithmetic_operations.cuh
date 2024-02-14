#pragma once
#include "alu_operations.cuh"
/**
 * The arithmetic operations class.
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
namespace arithmetic_operations {
    /**
     * The arithmetical environment used by the arbitrary length
     * integer library.
     */
    using arith_t = arith_env_t<evm_params>;
    /**
     * The arbitrary length integer type.
     */
    using bn_t = typename arith_t::bn_t;
    /**
     * The arbitrary length integer type with double the size of the
     * EVM word type.
     */
    using bn_wide_t = typename arith_t::bn_wide_t;
    /**
     * The stackk class.
     */
    using stack_t = ::stack_t<evm_params> ;

    /**
     * The ADD operation implementation.
     * Takes two values from the stack, adds them and
     * pushes the result back to the stack.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[inout] stack The stack.
    */
    __host__ __device__ __forceinline__ static void operation_ADD(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_VERY_LOW);
        if (arith.has_gas(gas_limit, gas_used, error_code))
        {
            bn_t a, b, r;
            stack.pop(a, error_code);
            stack.pop(b, error_code);

            cgbn_add(arith._env, r, a, b);

            stack.push(r, error_code);

            pc = pc + 1;
        }
    }

    /**
     * The MUL operation implementation.
     * Takes two values from the stack, multiplies them and
     * pushes the result back to the stack.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[inout] stack The stack.
    */
    __host__ __device__ __forceinline__ static void operation_MUL(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_LOW);
        if (arith.has_gas(gas_limit, gas_used, error_code))
        {
            bn_t a, b, r;
            stack.pop(a, error_code);
            stack.pop(b, error_code);

            cgbn_mul(arith._env, r, a, b);

            stack.push(r, error_code);

            pc = pc + 1;
        }
    }

    /**
     * The SUB operation implementation.
     * Takes two values from the stack, subtracts the second value
     * from the first value and pushes the result back to the stack.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[inout] stack The stack.
    */
    __host__ __device__ __forceinline__ static void operation_SUB(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_VERY_LOW);
        if (arith.has_gas(gas_limit, gas_used, error_code))
        {
            bn_t a, b, r;
            stack.pop(a, error_code);
            stack.pop(b, error_code);

            cgbn_sub(arith._env, r, a, b);

            stack.push(r, error_code);

            pc = pc + 1;
        }
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
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[inout] stack The stack.
    */
    __host__ __device__ __forceinline__ static void operation_DIV(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_LOW);
        if (arith.has_gas(gas_limit, gas_used, error_code))
        {
            bn_t a, b, r;
            stack.pop(a, error_code);
            stack.pop(b, error_code);

            // division by zero no error
            if (cgbn_compare_ui32(arith._env, b, 0) == 0)
                cgbn_set_ui32(arith._env, r, 0);
            else
                cgbn_div(arith._env, r, a, b);

            stack.push(r, error_code);

            pc = pc + 1;
        }
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
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[inout] stack The stack.
    */
    __host__ __device__ __forceinline__ static void operation_SDIV(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_LOW);
        if (arith.has_gas(gas_limit, gas_used, error_code))
        {
            bn_t a, b, r;
            stack.pop(a, error_code);
            stack.pop(b, error_code);

            bn_t d;
            bn_t e;
            // d = -1
            cgbn_set_ui32(arith._env, d, 0);
            cgbn_sub_ui32(arith._env, d, d, 1);
            // e = -2^254
            cgbn_set_ui32(arith._env, e, 1);
            cgbn_shift_left(arith._env, e, e, arith_t::BITS - 1);
            uint32_t sign_a = cgbn_extract_bits_ui32(arith._env, a, arith_t::BITS - 1, 1);
            uint32_t sign_b = cgbn_extract_bits_ui32(arith._env, b, arith_t::BITS - 1, 1);
            uint32_t sign = sign_a ^ sign_b;
            // division by zero no error
            if (cgbn_compare_ui32(arith._env, b, 0) == 0)
                cgbn_set_ui32(arith._env, r, 0);
            else if ( // -2^254 / -1 = -2^254
                (cgbn_compare(arith._env, b, d) == 0) &&
                (cgbn_compare(arith._env, a, e) == 0))
            {
                cgbn_set(arith._env, r, e);
            }
            else
            {
                // div between absolute values
                if (sign_a == 1)
                {
                    cgbn_negate(arith._env, a, a);
                }
                if (sign_b == 1)
                {
                    cgbn_negate(arith._env, b, b);
                }
                cgbn_div(arith._env, r, a, b);
                if (sign)
                {
                    cgbn_negate(arith._env, r, r);
                }
            }

            stack.push(r, error_code);

            pc = pc + 1;
        }
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
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[inout] stack The stack.
    */
    __host__ __device__ __forceinline__ static void operation_MOD(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_LOW);
        if (arith.has_gas(gas_limit, gas_used, error_code))
        {
            bn_t a, b, r;
            stack.pop(a, error_code);
            stack.pop(b, error_code);

            // // rem by zero no error
            if (cgbn_compare_ui32(arith._env, b, 0) == 0)
                cgbn_set_ui32(arith._env, r, 0);
            else
                cgbn_rem(arith._env, r, a, b);

            stack.push(r, error_code);

            pc = pc + 1;
        }
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
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[inout] stack The stack.
    */
    __host__ __device__ __forceinline__ static void operation_SMOD(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_LOW);
        if (arith.has_gas(gas_limit, gas_used, error_code))
        {
            bn_t a, b, r;
            stack.pop(a, error_code);
            stack.pop(b, error_code);

            uint32_t sign_a = cgbn_extract_bits_ui32(arith._env, a, arith_t::BITS - 1, 1);
            uint32_t sign_b = cgbn_extract_bits_ui32(arith._env, b, arith_t::BITS - 1, 1);
            uint32_t sign = sign_a ^ sign_b;
            if (cgbn_compare_ui32(arith._env, b, 0) == 0)
                cgbn_set_ui32(arith._env, r, 0);
            else
            {
                // mod between absolute values
                if (sign_a == 1)
                {
                    cgbn_negate(arith._env, a, a);
                }
                if (sign_b == 1)
                {
                    cgbn_negate(arith._env, b, b);
                }
                cgbn_rem(arith._env, r, a, b);
                if (sign)
                {
                    cgbn_negate(arith._env, r, r);
                }
            }

            stack.push(r, error_code);

            pc = pc + 1;
        }
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
     * @param[inout] error_code The error code.
     * @param[inout] pc The program counter.
    */
    __host__ __device__ __forceinline__ static void operation_ADDMOD(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_MID);
        if (arith.has_gas(gas_limit, gas_used, error_code))
        {
            bn_t a, b, c, N, r;
            stack.pop(a, error_code);
            stack.pop(b, error_code);
            stack.pop(N, error_code);

            if (cgbn_compare_ui32(arith._env, N, 0) == 0)
            {
                cgbn_set_ui32(arith._env, r, 0);
            }
            else if (cgbn_compare_ui32(arith._env, N, 1) == 0)
            {
                cgbn_set_ui32(arith._env, r, 0);
            }
            else
            {
                int32_t carry = cgbn_add(arith._env, c, a, b);
                bn_wide_t d;
                if (carry == 1)
                {
                    cgbn_set_ui32(arith._env, d._high, 1);
                    cgbn_set(arith._env, d._low, c);
                    cgbn_rem_wide(arith._env, r, d, N);
                }
                else
                {
                    cgbn_rem(arith._env, r, c, N);
                }
            }

            stack.push(r, error_code);

            pc = pc + 1;
        }
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
     * @param[inout] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[inout] stack The stack.
    */
    __host__ __device__ __forceinline__ static void operation_MULMOD(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_MID);
        if (arith.has_gas(gas_limit, gas_used, error_code))
        {
            bn_t a, b, c, N, r;
            stack.pop(a, error_code);
            stack.pop(b, error_code);
            stack.pop(N, error_code);

            if (cgbn_compare_ui32(arith._env, N, 0) == 0)
            {
                cgbn_set_ui32(arith._env, r, 0);
            }
            else
            {
                bn_wide_t d;
                cgbn_rem(arith._env, a, a, N);
                cgbn_rem(arith._env, b, b, N);
                cgbn_mul_wide(arith._env, d, a, b);
                cgbn_rem_wide(arith._env, r, d, N);
            }

            stack.push(r, error_code);

            pc = pc + 1;
        }
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
     * @param[inout] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[inout] stack The stack.
    */
    __host__ __device__ __forceinline__ static void operation_EXP(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_EXP);
        bn_t a, exponent, r;
        stack.pop(a, error_code);
        stack.pop(exponent, error_code);

        int32_t last_bit;
        last_bit = arith_t::BITS - 1 - cgbn_clz(arith._env, exponent);
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
        cgbn_set_ui32(arith._env, dynamic_gas, exponent_byte_size);
        cgbn_mul_ui32(arith._env, dynamic_gas, dynamic_gas, GAS_EXP_BYTE);
        cgbn_add(arith._env, gas_used, gas_used, dynamic_gas);
        if (error_code == ERR_NONE)
        {
            if (arith.has_gas(gas_limit, gas_used, error_code))
            {
                //^0=1 even for 0^0
                if (last_bit == -1)
                {
                    cgbn_set_ui32(arith._env, r, 1);
                }
                else
                {
                    bn_t current, square;
                    cgbn_set_ui32(arith._env, current, 1); // r=1
                    cgbn_set(arith._env, square, a);       // square=a
                    for (int32_t bit = 0; bit <= last_bit; bit++)
                    {
                        if (cgbn_extract_bits_ui32(arith._env, exponent, bit, 1) == 1)
                        {
                            cgbn_mul(arith._env, current, current, square); // r=r*square
                        }
                        cgbn_mul(arith._env, square, square, square); // square=square*square
                    }
                    cgbn_set(arith._env, r, current);
                }

                stack.push(r, error_code);

                pc = pc + 1;
            }
        }
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
     * @param[inout] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[inout] stack The stack.
    */
    __host__ __device__ __forceinline__ static void operation_SIGNEXTEND(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack)
    {
        /*
        Even if x has more bytes than the value b, the operation consider only the first
        (b+1) bytes of x and the other are considered zero and they don't have any influence
        on the final result.
        Optimised: use cgbn_bitwise_mask_ior instead of cgbn_insert_bits_ui32
        */
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_LOW);
        if (arith.has_gas(gas_limit, gas_used, error_code))
        {
            bn_t b, x, r;
            stack.pop(b, error_code);
            stack.pop(x, error_code);

            if (cgbn_compare_ui32(arith._env, b, (arith_t::BYTES - 1) ) == 1)
            {
                cgbn_set(arith._env, r, x);
            }
            else
            {
                uint32_t c = cgbn_get_ui32(arith._env, b) + 1;
                uint32_t sign = cgbn_extract_bits_ui32(arith._env, x, c * 8 - 1, 1);
                int32_t numbits = int32_t(c);
                if (sign == 1)
                {
                    numbits = int32_t(arith_t::BITS) - 8 * numbits;
                    numbits = -numbits;
                    cgbn_bitwise_mask_ior(arith._env, r, x, numbits);
                }
                else
                {
                    cgbn_bitwise_mask_and(arith._env, r, x, 8 * numbits);
                }
            }

            stack.push(r, error_code);

            pc = pc + 1;
        }
    }
}

/**
 * The stack operations class.
 * Contains all the operations that can be performed on the stack.
 * - POP 50: POP
 * - PUSH0 5F: PUSH0
 * - PUSHX 60s & 70s: Push Operations
 * - DUPX 80s: Duplication Operations
 * - SWAPX 90s: Exchange Operations
 */
template <class params>
class stack_operations
{
public:
    /**
     * The arithmetical environment used by the arbitrary length
     * integer library.
     */
    typedef arith_env_t<params> arith_t;
    /**
     * The arbitrary length integer type.
     */
    typedef typename arith_t::bn_t bn_t;
    /**
     * The stackk class.
     */
    typedef stack_t<params> stack_t;

    /**
     * The POP operation implementation.
     * It pops the top element from the stack.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     */
    __host__ __device__ __forceinline__ static void operation_POP(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);

        if (arith.has_gas(gas_limit, gas_used, error_code))
        {
            bn_t y;

            stack.pop(y, error_code);

            pc = pc + 1;
        }
    }

    /**
     * The PUSH0 operation implementation.
     * Pushes a zero value to the stack.
     * @param[in] arith The arithmetical environment.
     * @param[inout] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[inout] stack The stack.
    */
    __host__ __device__ __forceinline__ static void operation_PUSH0(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);
        if (arith.has_gas(gas_limit, gas_used, error_code))
        {
            bn_t r;
            cgbn_set_ui32(arith._env, r, 0);

            stack.push(r, error_code);

            pc = pc + 1;
        }
    }

    /**
     * The PUSHX operation implementation.
     * Pushes a value from the bytecode to the stack.
     * If the bytecode is not long enough to provide the value,
     * the operation completes with zero bytes for the least
     * significant bytes.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[inout] stack The stack.
     * @param[in] bytecode The bytecode.
     * @param[in] code_size The size of the bytecode.
     * @param[in] opcode The opcode. 
    */
    __host__ __device__ __forceinline__ static void operation_PUSHX(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        uint8_t *bytecode,
        uint32_t &code_size,
        uint8_t &opcode)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_VERY_LOW);
        if (arith.has_gas(gas_limit, gas_used, error_code))
        {
            uint8_t push_size = (opcode & 0x1F) + 1;
            uint8_t *byte_data = &(bytecode[pc + 1]);
            uint8_t available_size = push_size;

            // if pushx is outside code size
            if ((pc + push_size) >= code_size)
            {
                available_size = code_size - pc - 1;
            }
            stack.pushx(push_size, error_code, byte_data, available_size);

            pc = pc + push_size + 1;
        }
    }

    /**
     * The DUPX operation implementation.
     * Duplicates a value from the stack and pushes it back to the stack.
     * The value to be duplicated is given by the opcode.
     * The opcode is in the range 0x80 - 0x8F.
     * The index of the value to be duplicated is given
     * by the opcode - 0x80 + 1.
     * We consider the least 4 significant bits of the opcode + 1.
     * The index is from the top of the stack. 1 is the top of the stack.
     * 2 is the second value from the top of the stack and so on.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[inout] stack The stack.
     * @param[in] opcode The opcode.
    */
    __host__ __device__ __forceinline__ static void operation_DUPX(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        uint8_t &opcode)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_VERY_LOW);
        if (arith.has_gas(gas_limit, gas_used, error_code))
        {
            uint8_t dup_index = (opcode & 0x0F) + 1;

            stack.dupx(dup_index, error_code);

            pc = pc + 1;
        }
    }

    /**
     * The SWAPX operation implementation.
     * Swaps the top of the stack with a value from the stack.
     * The index of the value to be swapped is given
     * by the opcode - 0x90 + 1.
     * We consider the least 4 significant bits of the opcode + 1.
     * The index is starts from the first value under the top of the stack.
     * 1 is the second value from the top of the stack, 2 is the third value
     * from the top of the stack and so on.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] error_code The error code.
     * @param[inout] pc The program counter.
     * @param[inout] stack The stack.
     * @param[in] opcode The opcode.
    */
    __host__ __device__ __forceinline__ static void operation_SWAPX(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        uint8_t &opcode)
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_VERY_LOW);
        if (arith.has_gas(gas_limit, gas_used, error_code))
        {
            uint8_t swap_index = (opcode & 0x0F) + 1;

            stack.swapx(swap_index, error_code);

            pc = pc + 1;
        }
    }
};
