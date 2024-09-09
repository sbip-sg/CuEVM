// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#include <CuEVM/operations/flow.cuh>
#include <CuEVM/gas_cost.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM::operations {
    __host__ __device__ int32_t JUMP(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &pc,
        CuEVM::evm_stack_t &stack,
        const CuEVM::evm_message_call_t &message)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_MID);
        int32_t error_code = CuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        if (error_code == ERROR_SUCCESS)
        {
            bn_t destination;
            error_code |= stack.pop(arith, destination);
            uint32_t destination_u32;
            int32_t overflow;
            error_code = arith.uint32_t_from_cgbn(destination_u32, destination) ? ERROR_INVALID_JUMP_DESTINATION : error_code;
            if (error_code == ERROR_SUCCESS)
            {
                pc = message.get_jump_destinations()->has(destination_u32) ? destination_u32 - 1 : ([&]() -> uint32_t {
                    error_code = ERROR_INVALID_JUMP_DESTINATION;
                    return pc;
                })();
            }
        }
    }

    __host__ __device__ int32_t JUMPI(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &pc,
        CuEVM::evm_stack_t &stack,
        const CuEVM::evm_message_call_t &message)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_HIGH);
        int32_t error_code = CuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        if (error_code == ERROR_SUCCESS)
        {
            bn_t destination;
            error_code |= stack.pop(arith, destination);
            bn_t condition;
            error_code |= stack.pop(arith, condition);

            if (
                (error_code == ERROR_SUCCESS) &&
                (cgbn_compare_ui32(arith.env, condition, 0) != 0)
            ) {
                uint32_t destination_u32;
                int32_t overflow;
                error_code = arith.uint32_t_from_cgbn(destination_u32, destination) ? ERROR_INVALID_JUMP_DESTINATION : error_code;
                if (error_code == ERROR_SUCCESS)
                {
                    pc = message.get_jump_destinations()->has(destination_u32) ? destination_u32 - 1 : ([&]() -> uint32_t {
                        error_code = ERROR_INVALID_JUMP_DESTINATION;
                        return pc;
                    })();
                }
            }
        }
    }

    __host__ __device__ int32_t PC(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        const uint32_t &pc,
        CuEVM::evm_stack_t &stack)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_BASE);
        int32_t error_code = CuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        bn_t pc_bn;
        cgbn_set_ui32(arith.env, pc_bn, pc);
        error_code |= stack.push(arith, pc_bn);
    }

    __host__ __device__ int32_t GAS(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        CuEVM::evm_stack_t &stack)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_BASE);
        int32_t error_code = CuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        bn_t gas_left;
        cgbn_sub(arith.env, gas_left, gas_limit, gas_used);
        error_code |= stack.push(arith, gas_left);
    }

    __host__ __device__ int32_t JUMPDEST(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_JUMP_DEST);
        int32_t error_code = CuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
    }
}