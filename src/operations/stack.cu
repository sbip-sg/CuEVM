// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Date: 2023-07-15
// SPDX-License-Identifier: MIT

#include "../include/operations/stack.cuh"
#include "../include/gas_cost.cuh"
#include "../include/utils/error_codes.cuh"

namespace cuEVM {
    namespace operations {
        __host__ __device__ int32_t POP(
            ArithEnv &arith,
            const bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &pc,
            cuEVM::evm_stack_t &stack)
        {
            cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_BASE);

            int32_t error_code = cuEVM::gas_cost::has_gas(
                    arith,
                    gas_limit,
                    gas_used);
            if (error_code == ERROR_SUCCESS)
            {
                bn_t y;

                error_code |= stack.pop(arith, y);
            }
            return error_code;
        }

        __host__ __device__ int32_t PUSH0(
            ArithEnv &arith,
            const bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &pc,
            cuEVM::evm_stack_t &stack)
        {
            cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_BASE);
            int32_t error_code = cuEVM::gas_cost::has_gas(
                arith,
                gas_limit,
                gas_used);
            if (error_code == ERROR_SUCCESS)
            {
                bn_t r;
                cgbn_set_ui32(arith.env, r, 0);

                error_code |= stack.push(arith, r);
            }
            return error_code;
        }

        __host__ __device__ int32_t PUSHX(
            ArithEnv &arith,
            const bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &pc,
            cuEVM::evm_stack_t &stack,
            const cuEVM::byte_array_t &byte_code,
            const uint8_t &opcode)
        {
            cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_VERY_LOW);
            int32_t error_code = cuEVM::gas_cost::has_gas(
                arith,
                gas_limit,
                gas_used);
            if (error_code == ERROR_SUCCESS)
            {
                uint8_t push_size = (opcode & 0x1F) + 1;
                uint8_t *byte_data = &(byte_code.data[pc + 1]);
                // if pushx is outside code size
                uint32_t available_size = (pc + push_size >= byte_code.size) ?
                    byte_code.size - pc - 1 : push_size;
                // TODO: maybe make it a byte array for better transmission
                error_code |= stack.pushx(
                    arith,
                    push_size,
                    byte_data,
                    available_size);

                pc = pc + push_size;
            }
            return error_code;
        }

        __host__ __device__ int32_t DUPX(
            ArithEnv &arith,
            const bn_t &gas_limit,
            bn_t &gas_used,
            cuEVM::evm_stack_t &stack,
            const uint8_t &opcode)
        {
            cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_VERY_LOW);
            int32_t error_code = cuEVM::gas_cost::has_gas(
                arith,
                gas_limit,
                gas_used);
            if (error_code == ERROR_SUCCESS)
            {
                uint8_t dup_index = (opcode & 0x0F) + 1;

                error_code |= stack.dupx(arith, dup_index);
            }
            return error_code;
        }

        __host__ __device__ int32_t SWAPX(
            ArithEnv &arith,
            const bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &pc,
            cuEVM::evm_stack_t &stack,
            const uint8_t &opcode)
        {
            cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_VERY_LOW);
            int32_t error_code = cuEVM::gas_cost::has_gas(
                arith,
                gas_limit,
                gas_used);
            if (error_code == ERROR_SUCCESS)
            {
                uint8_t swap_index = (opcode & 0x0F) + 1;

                error_code |= stack.swapx(arith, swap_index);
            }
            return error_code;
        }
    }
}