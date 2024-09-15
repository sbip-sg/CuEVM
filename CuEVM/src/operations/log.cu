// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#include <CuEVM/operations/log.cuh>
#include <CuEVM/gas_cost.cuh>
#include <CuEVM/utils/error_codes.cuh>


/**
 * a0s: Logging Operations:
 * - LOGX
 */
namespace CuEVM::operations {
    __host__ __device__ int32_t LOGX(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        CuEVM::evm_stack_t &stack,
        CuEVM::evm_memory_t &memory,
        const CuEVM::evm_message_call_t &message,
        CuEVM::log_state_data_t &log_state,
        const uint8_t &opcode)
    {
        int32_t error_code = (
            message.get_static_env() ?
            ERROR_STATIC_CALL_CONTEXT_SSTORE :
            ERROR_SUCCESS);
        
        uint32_t no_topics = opcode & 0x0F;
        
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_LOG);
        error_code |= CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);

        bn_t memory_offset;
        error_code |= stack.pop(arith, memory_offset);
        bn_t length;
        error_code |= stack.pop(arith, length);

        CuEVM::gas_cost::log_record_cost(
            arith,
            gas_used,
            length);
        
        CuEVM::gas_cost::log_topics_cost(
            arith,
            gas_used,
            no_topics);


        bn_t topics[4];
        for (uint32_t idx = 0; idx < no_topics; idx++)
        {
            error_code |= stack.pop(arith, topics[idx]);
        }
        for (uint32_t idx = no_topics; idx < 4; idx++)
        {
            cgbn_set_ui32(arith.env, topics[idx], 0);
        }

        error_code |= CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);

        if (error_code == ERROR_SUCCESS)
        {
            bn_t memory_expansion_cost;
            // Get the memory expansion gas cost
            error_code |= CuEVM::gas_cost::memory_grow_cost(
                arith,
                memory,
                memory_offset,
                length,
                memory_expansion_cost,
                gas_used);
            
            error_code |= CuEVM::gas_cost::has_gas(
                arith,
                gas_limit,
                gas_used);

            if (error_code == ERROR_SUCCESS) {
                memory.increase_memory_cost(arith, memory_expansion_cost);
                CuEVM::byte_array_t record;
                error_code |= memory.get(
                    arith,
                    memory_offset,
                    length,
                    record);

                bn_t address;
                message.get_contract_address(arith, address);

                log_state.push(
                    arith,
                    address,
                    record,
                    topics[0],
                    topics[1],
                    topics[2],
                    topics[3],
                    no_topics);
            }
        }
        return error_code;
    }
}