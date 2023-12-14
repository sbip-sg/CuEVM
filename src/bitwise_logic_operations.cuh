// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#ifndef _BITWISE_OP_H_
#define _BITWISE_OP_H_

#include "uitls.h"
#include "stack.cuh"


template <class params>
class bitwise_logic_operations
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
     * The CGBN wide type with double the given number of bits in environment.
     */
    typedef typename env_t::cgbn_wide_t bn_wide_t;
    /**
     * The arbitrary length integer type used for the storage.
     * It is defined as the EVM word type.
     */
    typedef cgbn_mem_t<params::BITS> evm_word_t;
    /**
     * The stackk class.
    */
    typedef stack_t<params> stack_t;
    
    __host__ __device__ __forceinline__ int32_t compare(
        arith_t &arith,
        uint32_t &error_code,
        stack_t &stack
    )
    {
        bn_t a, b;
        stack.pop(a, error_code);
        stack.pop(b, error_code);
        
        return cgbn_compare(arith._env, a, b);
    }

    __host__ __device__ __forceinline__ int32_t scompare(
        arith_t &arith,
        uint32_t &error_code,
        stack_t &stack
    )
    {
        bn_t a, b;
        stack.pop(a, error_code);
        stack.pop(b, error_code);
        
        uint32_t sign_a = cgbn_extract_bits_ui32(arith._env, a, arith_t::BITS - 1, 1);
        uint32_t sign_b = cgbn_extract_bits_ui32(arith._env, b, arith_t::BITS - 1, 1);
        if (sign_a == 0 && sign_b == 1)
        {
            return 1;
        }
        else if (sign_a == 1 && sign_b == 0)
        {
            return -1;
        }
        else
        {
            return cgbn_compare(arith._env, a, b);
        }
    }


    //10s: Comparison & Bitwise Logic Operations

    // Comparison Operations
    __host__ __device__ __forceinline__ static void operation_LT(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack
    )
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_VERY_LOW);
        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
           
            int32_t int_result = compare(error_code);
            uint32_t result = (int_result < 0) ? 1 : 0;
            bn_t r;

            cgbn_set_ui32(arith._env, r, result);
            
            stack.push(r, error_code);
            
            pc = pc + 1;
        }
    }

    __host__ __device__ __forceinline__ static void operation_GT(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack
    )
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_VERY_LOW);
        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
           
            int32_t int_result = compare(error_code);
            uint32_t result = (int_result > 0) ? 1 : 0;
            bn_t r;

            cgbn_set_ui32(arith._env, r, result);
            
            stack.push(r, error_code);
            
            pc = pc + 1;
        }
    }

    __host__ __device__ __forceinline__ static void operation_SLT(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack
    )
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_VERY_LOW);
        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
           
            int32_t int_result = scompare(error_code);
            uint32_t result = (int_result < 0) ? 1 : 0;
            bn_t r;

            cgbn_set_ui32(arith._env, r, result);
            
            stack.push(r, error_code);
            
            pc = pc + 1;
        }
    }

    __host__ __device__ __forceinline__ static void operation_SGT(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack
    )
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_VERY_LOW);
        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
           
            int32_t int_result = scompare(error_code);
            uint32_t result = (int_result > 0) ? 1 : 0;
            bn_t r;

            cgbn_set_ui32(arith._env, r, result);
            
            stack.push(r, error_code);
            
            pc = pc + 1;
        }
    }

    __host__ __device__ __forceinline__ static void operation_EQ(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack
    )
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_VERY_LOW);
        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
           
            int32_t int_result = compare(error_code);
            uint32_t result = (int_result == 0) ? 1 : 0;
            bn_t r;

            cgbn_set_ui32(arith._env, r, result);
            
            stack.push(r, error_code);
            
            pc = pc + 1;
        }
    }

    __host__ __device__ __forceinline__ static void operation_ISZERO(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack
    )
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);
        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
            bn_t a;
            stack.pop(a, error_code);
            bn_t r;

            int32_t compare = cgbn_compare_ui32(_arith._env, a, 0);
            if (compare == 0)
            {
                cgbn_set_ui32(arith._env, r, 1);
            }
            else
            {
                cgbn_set_ui32(arith._env, r, 0);
            }
            
            stack.push(r, error_code);
            
            pc = pc + 1;
        }
    }

    // Bitwise Logic Operations

    __host__ __device__ __forceinline__ static void operation_AND(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack
    ) 
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_VERY_LOW);
        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
            bn_t a, b;
            stack.pop(a, error_code);
            stack.pop(b, error_code);
            bn_t r;

            cgbn_bitwise_and(arith._env, r, a, b);
            
            stack.push(r, error_code);
            
            pc = pc + 1;
        }
    }

    __host__ __device__ __forceinline__ static void operation_OR(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack
    ) 
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_VERY_LOW);
        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
            bn_t a, b;
            stack.pop(a, error_code);
            stack.pop(b, error_code);
            bn_t r;

            cgbn_bitwise_ior(arith._env, r, a, b);
            
            stack.push(r, error_code);
            
            pc = pc + 1;
        }
    }

    __host__ __device__ __forceinline__ static void operation_XOR(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack
    ) 
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_VERY_LOW);
        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
            bn_t a, b;
            stack.pop(a, error_code);
            stack.pop(b, error_code);
            bn_t r;

            cgbn_bitwise_xor(arith._env, r, a, b);
            
            stack.push(r, error_code);
            
            pc = pc + 1;
        }
    }

    __host__ __device__ __forceinline__ static void operation_NOT(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack 
    ) 
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_BASE);
        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
            bn_t a;
            stack.pop(a, error_code);
            bn_t r;

            cgbn_bitwise_mask_xor(_arith._env, r, a, arith_t::BITS);
            
            stack.push(r, error_code);
            
            pc = pc + 1;
        }
    }

    __host__ __device__ __forceinline__ static void operation_BYTE(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint8_t &byte,
        uint32_t &pc,
        stack_t &stack 
    ) 
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_VERY_LOW);
        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
            bn_t i, x;
            stack.pop(i, error_code);
            stack.pop(x, error_code);
            bn_t r;

            if (cgbn_compare_ui32(_arith._env, i, 31) == 1)
            {
                cgbn_set_ui32(_arith._env, r, 0);
            }
            else
            {
                uint32_t index = cgbn_get_ui32(_arith._env, i);
                uint32_t byte = cgbn_extract_bits_ui32(_arith._env, x, 8 * ((arith::BYTES - 1) - index), 8);
                cgbn_set_ui32(_arith._env, r, byte);
            }
            
            stack.push(r, error_code);
            
            pc = pc + 1;
        }
    }

    __host__ __device__ __forceinline__ static void operation_SHL(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint8_t &byte,
        uint32_t &pc,
        stack_t &stack 
    ) 
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_LOW);
        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
            bn_t shift, value;
            stack.pop(shift, error_code);
            stack.pop(value, error_code);
            bn_t r;

            if (cgbn_compare_ui32(_arith._env, shift, arith_t::BITS - 1) == 1)
            {
                cgbn_set_ui32(_arith._env, r, 0);
            }
            else
            {
                uint32_t shift_left = cgbn_get_ui32(_arith._env, shift);
                cgbn_shift_left(_arith._env, r, value, shift_left);
            }
            
            stack.push(r, error_code);
            
            pc = pc + 1;
        }
    }

    __host__ __device__ __forceinline__ static void operation_SHR(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint8_t &byte,
        uint32_t &pc,
        stack_t &stack 
    ) 
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_LOW);
        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
            bn_t shift, value;
            stack.pop(shift, error_code);
            stack.pop(value, error_code);
            bn_t r;

            if (cgbn_compare_ui32(_arith._env, shift, arith_t::BITS - 1) == 1)
            {
                cgbn_set_ui32(_arith._env, r, 0);
            }
            else
            {
                uint32_t shift_right = cgbn_get_ui32(_arith._env, shift);
                cgbn_shift_right(_arith._env, r, value, shift_right);
            }
            
            stack.push(r, error_code);
            
            pc = pc + 1;
        }
    }

    __host__ __device__ __forceinline__ static void operation_SAR(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint8_t &byte,
        uint32_t &pc,
        stack_t &stack 
    ) 
    {
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_LOW);
        if (has_gas(arith, gas_limit, gas_used, error_code))
        {
            bn_t shift, value;
            stack.pop(shift, error_code);
            stack.pop(value, error_code);
            bn_t r;

            uint32_t sign_b = cgbn_extract_bits_ui32(_arith._env, value, arith_t::BITS - 1, 1);
            uint32_t shift_right = cgbn_get_ui32(_arith._env, shift);

            if (cgbn_compare_ui32(_arith._env, shift, arith_t::BITS - 1) == 1)
                shift_right = arith_t::BITS;

            cgbn_shift_right(_arith._env, r, value, shift_right);
            if (sign_b == 1)
            {
                cgbn_bitwise_mask_ior(_arith._env, r, r, -shift_right);
            }
            
            stack.push(r, error_code);
            
            pc = pc + 1;
        }
    }
    
};


#endif