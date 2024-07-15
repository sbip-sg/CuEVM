// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#include <CuCrypto/keccak.cuh>

#include "../include/operations/environmental.cuh"
#include "../include/gas_cost.cuh"
#include "../include/utils/error_codes.cuh"
#include "../include/core/byte_array.cuh"

/**
 * The environmental operations class.
 * Contains the environmental operations
 * - 20s: KECCAK256:
 *      - SHA3
 * - 30s: Environmental Information:
 *      - ADDRESS
 *      - BALANCE
 *      - ORIGIN
 *      - CALLER
 *      - CALLVALUE
 *      - CALLDATALOAD
 *      - CALLDATASIZE
 *      - CALLDATACOPY
 *      - CODESIZE
 *      - CODECOPY
 *      - GASPRICE
 *      - EXTCODESIZE
 *      - EXTCODECOPY
 *      - RETURNDATASIZE
 *      - RETURNDATACOPY
 *      - EXTCODEHASH
 *  - 47: SELFBALANCE
 * SELFBALANCE is moved here from block operations.
*/
namespace cuEVM::operations {
    /**
     * The SHA3 operation implementation.
     * Takes the offset and length from the stack and pushes the hash of the
     * data from the memory at the given offset for the given length.
     * The dynamic gas cost is computed as:
     * - word_size = (length + 31) / 32
     * - dynamic_gas_cost = word_size * GAS_KECCAK256_WORD
     * Adittional gas cost is added for the memory expansion.
     *
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] pc The program counter.
     * @param[inout] stack The stack.
     * @param[inout] memory The memory object.
    */
    __host__ __device__ int32_t SHA3(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &pc,
        cuEVM::stack::evm_stack_t &stack,
        cuEVM::memory::evm_memory_t &memory)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_KECCAK256);
        int32_t error_code = cuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        if (error_code == ERROR_SUCCESS) {
            // Get the offset and length from the stack
            bn_t offset, length;
            error_code |= stack.pop(arith, offset);
            error_code |= stack.pop(arith, length);

            cuEVM::gas_cost::keccak_cost(
                arith,
                gas_used,
                length);
            
            bn_t memory_expansion_cost;
            // Get the memory expansion gas cost
            error_code |= cuEVM::gas_cost::memory_grow_cost(
                arith,
                memory,
                offset,
                length,
                memory_expansion_cost,
                gas_used);
            
            error_code |= cuEVM::gas_cost::has_gas(
                arith,
                gas_limit,
                gas_used);

            if (error_code == ERROR_SUCCESS)
            {
                memory.increase_memory_cost(arith, memory_expansion_cost);
                cuEVM::byte_array_t memory_input;
                error_code |= memory.get(
                    arith,
                    offset,
                    length,
                    memory_input);
                cuEVM::byte_array_t hash(cuEVM::hash_size);
                if (error_code == ERR_NONE)
                {
                    CuCrypto::keccak::sha3(
                        memory_input.data,
                        memory_input.size,
                        hash.data,
                        hash.size);
                    bn_t hash_bn;
                    hash.to_bn_t(
                        arith,
                        hash_bn);

                    error_code |= stack.push(arith, hash_bn);

                    pc = pc + 1;
                }
                return error_code;
            }
        }
        return error_code;
    }

    /**
     * The ADDRESS operation implementation.
     * Pushes on the stack the address of currently executing account.
     * The executing account is consider the current context, so it can be
     * different than the owner of the code.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     * @param[in] message The message.
    */
    __host__ __device__ int32_t ADDRESS(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &pc,
        cuEVM::stack::evm_stack_t &stack,
        const cuEVM::evm_message_call_t &message)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_BASE);
        int32_t error_code = cuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        if (error_code == ERROR_SUCCESS)
        {
            bn_t recipient_address;
            message.get_recipient(arith, recipient_address);

            error_code |= stack.push(arith, recipient_address);

            pc = pc + 1;
        }
        return error_code;
    }

    /**
     * The BALANCE operation implementation.
     * Takes the address from the stack and pushes the balance of the
     * account with that address.
     * Gas is charged for accessing the account if it is warm
     * or cold access.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] pc The program counter.
     * @param[inout] stack The stack.
     * @param[in] touch_state The touch state object.
    */
    __host__ __device__ int32_t BALANCE(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &pc,
        cuEVM::stack::evm_stack_t &stack,
        const cuEVM::state::AccessState &access_state,
        cuEVM::state::TouchState &touch_state)
    {
        // cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_ZERO);
        bn_t address;
        int32_t error_code = stack.pop(arith, address);
        cuEVM::evm_address_conversion(arith, address);
        error_code |= cuEVM::gas_cost::access_account_cost(
            arith,
            gas_used,
            access_state,
            address);
        error_code |= cuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        
        if (error_code == ERROR_SUCCESS) {
            bn_t balance;
            touch_state.get_balance(
                arith,
                address,
                balance);

            error_code |= stack.push(arith, balance);

            pc = pc + 1;
        }
        return error_code;
    }

    /**
     * The ORIGIN operation implementation.
     * Pushes on the stack the address of the sender of the transaction
     * that started the execution.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     * @param[in] transaction The transaction.
    */
    __host__ __device__ int32_t ORIGIN(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &pc,
        cuEVM::stack::evm_stack_t &stack,
        cuEVM::transaction::transaction_t &transaction)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_BASE);
        int32_t error_code = cuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        if (error_code == ERROR_SUCCESS)
        {
            bn_t origin;
            transaction.get_sender(origin);

            stack.push(origin, error_code);

            pc = pc + 1;
        }
    }

    /**
     * The CALLER operation implementation.
     * Pushes on the stack the address of the sender of the message
     * that started the execution.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     * @param[in] message The message.
    */
    __host__ __device__ int32_t CALLER(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &pc,
        cuEVM::stack::evm_stack_t &stack,
        cuEVM::evm_message_call_t &message)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_BASE);
        int32_t error_code = cuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        if (error_code == ERROR_SUCCESS)
        {
            bn_t caller;
            message.get_sender(caller);

            stack.push(caller, error_code);

            pc = pc + 1;
        }
    }

    /**
     * The CALLVALUE operation implementation.
     * Pushes on the stack the value of the message that started the execution.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     * @param[in] message The message.
    */
    __host__ __device__ int32_t CALLVALUE(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &pc,
        cuEVM::stack::evm_stack_t &stack,
        cuEVM::evm_message_call_t &message)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_BASE);
        int32_t error_code = cuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        if (error_code == ERROR_SUCCESS)
        {
            bn_t call_value;
            message.get_value(call_value);

            stack.push(call_value, error_code);

            pc = pc + 1;
        }
    }

    /**
     * The CALLDATALOAD operation implementation.
     * Takes the index from the stack and pushes the data
     * from the message call data at the given index.
     * The data pushed is a evm word.
     * If the call data has less bytes than neccessay to fill the evm word,
     * the remaining bytes are filled with zeros. (the least significant bytes)
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] pc The program counter.
     * @param[inout] stack The stack.
     * @param[in] message The message.
    */
    __host__ __device__ int32_t CALLDATALOAD(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &pc,
        cuEVM::stack::evm_stack_t &stack,
        cuEVM::evm_message_call_t &message)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_VERY_LOW);
        int32_t error_code = cuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        if (error_code == ERROR_SUCCESS)
        {
            bn_t index;
            stack.pop(index, error_code);
            bn_t length;
            cgbn_set_ui32(arith.env, length, EVM_WORD_SIZE);

            size_t available_data;
            uint8_t *data;
            data = message.get_data(
                index,
                length,
                available_data);

            stack.pushx(EVM_WORD_SIZE, error_code, data, available_data);

            pc = pc + 1;
        }
    }

    /**
     * The CALLDATASIZE operation implementation.
     * Pushes on the stack the size of the message call data.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     * @param[in] message The message.
    */
    __host__ __device__ int32_t CALLDATASIZE(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &pc,
        cuEVM::stack::evm_stack_t &stack,
        cuEVM::evm_message_call_t &message)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_BASE);
        int32_t error_code = cuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        if (error_code == ERROR_SUCCESS)
        {
            bn_t length;
            size_t length_s;
            length_s = message.get_data_size();
            arith.cgbn_from_size_t(length, length_s);

            stack.push(length, error_code);

            pc = pc + 1;
        }
    }

    /**
     * The CALLDATACOPY operation implementation.
     * Takes the memory offset, data offset and length from the stack and
     * copies the data from the message call data at the given data offset for
     * the given length to the memory at the given memory offset.
     * If the call data has less bytes than neccessay to fill the memory,
     * the remaining bytes are filled with zeros.
     * The dynamic gas cost is computed as:
     * - word_size = (length + 31) / 32
     * - dynamic_gas_cost = word_size * GAS_MEMORY
     * Adittional gas cost is added for the memory expansion.
     *
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[out] pc The program counter.
     * @param[inout] stack The stack.
     * @param[in] message The message.
     * @param[out] memory The memory.
    */
    __host__ __device__ int32_t CALLDATACOPY(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &pc,
        cuEVM::stack::evm_stack_t &stack,
        cuEVM::evm_message_call_t &message,
        cuEVM::memory::evm_memory_t &memory)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_VERY_LOW);

        bn_t memory_offset, data_offset, length;
        stack.pop(memory_offset, error_code);
        stack.pop(data_offset, error_code);
        stack.pop(length, error_code);

        // compute the dynamic gas cost
        arith.memory_cost(
            gas_used,
            length
        );

        // get the memory expansion gas cost
        memory.grow_cost(
            memory_offset,
            length,
            gas_used,
            error_code);

        if (error_code == ERR_NONE)
        {
            if (arith.has_gas(gas_limit, gas_used, error_code))
            {
                size_t available_data;
                uint8_t *data;
                data = message.get_data(
                    data_offset,
                    length,
                    available_data);

                memory.set(
                    data,
                    memory_offset,
                    length,
                    available_data,
                    error_code);

                pc = pc + 1;
            }
        }
    }

    /**
     * The CODESIZE operation implementation.
     * Pushes on the stack the size of code running in current environment.
     *
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     * @param[in] message The message.
    */
    __host__ __device__ int32_t CODESIZE(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &pc,
        cuEVM::stack::evm_stack_t &stack,
        cuEVM::evm_message_call_t &message)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_BASE);
        int32_t error_code = cuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        if (error_code == ERROR_SUCCESS)
        {
            size_t code_size;
            code_size = message.get_code_size();

            bn_t code_size_bn;
            arith.cgbn_from_size_t(code_size_bn, code_size);

            stack.push(code_size_bn, error_code);

            pc = pc + 1;
        }
    }

    /**
     * The CODECOPY operation implementation.
     * Takes the memory offset, code offset and length from the stack and
     * copies code running in current environment at the given code offset for
     * the given length to the memory at the given memory offset.
     * If the code has less bytes than neccessay to fill the memory,
     * the remaining bytes are filled with zeros.
     * The dynamic gas cost is computed as:
     * - word_size = (length + 31) / 32
     * - dynamic_gas_cost = word_size * GAS_MEMORY
     * Adittional gas cost is added for the memory expansion.
     *
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] pc The program counter.
     * @param[inout] stack The stack.
     * @param[in] message The message.
     * @param[in] touch_state The touch state object. The executing world state.
     * @param[out] memory The memory.
    */
    __host__ __device__ int32_t CODECOPY(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &pc,
        cuEVM::stack::evm_stack_t &stack,
        cuEVM::evm_message_call_t &message,
        cuEVM::memory::evm_memory_t &memory)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_VERY_LOW);

        bn_t memory_offset, code_offset, length;
        stack.pop(memory_offset, error_code);
        stack.pop(code_offset, error_code);
        stack.pop(length, error_code);

        if (error_code == ERR_NONE)
        {
            // compute the dynamic gas cost
            arith.memory_cost(
                gas_used,
                length
            );

            // get the memory expansion gas cost
            memory.grow_cost(
                memory_offset,
                length,
                gas_used,
                error_code);

            if (error_code == ERR_NONE)
            {
                if (arith.has_gas(gas_limit, gas_used, error_code))
                {

                    size_t available_data;
                    uint8_t *data;
                    data = message.get_byte_code_data(
                        code_offset,
                        length,
                        available_data);

                    memory.set(
                        data,
                        memory_offset,
                        length,
                        available_data,
                        error_code);

                    pc = pc + 1;
                }
            }
        }
    }

    /**
     * The GASPRICE operation implementation.
     * Pushes on the stack the gas price of the current transaction.
     * The gas price is the price per unit of gas in the transaction.
     *
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     * @param[in] block The block.
     * @param[in] transaction The transaction.
    */
    __host__ __device__ int32_t GASPRICE(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &pc,
        cuEVM::stack::evm_stack_t &stack,
        cuEVM::block_info_t &block,
        cuEVM::transaction::transaction_t &transaction)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_BASE);
        int32_t error_code = cuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        if (error_code == ERROR_SUCCESS)
        {
            bn_t block_base_fee;
            block.get_base_fee(block_base_fee);

            bn_t gas_price;
            transaction.get_computed_gas_price(
                gas_price,
                block_base_fee,
                error_code);

            stack.push(gas_price, error_code);

            pc = pc + 1;
        }
    }

    /**
     * The EXTCODESIZE operation implementation.
     * Takes the address from the stack and pushes the size of the code
     * of the account with that address.
     * Gas is charged for accessing the account if it is warm
     * or cold access.
     *
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] pc The program counter.
     * @param[inout] stack The stack.
     * @param[in] touch_state The touch state object. The executing world state.
    */
    __host__ __device__ int32_t EXTCODESIZE(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &pc,
        cuEVM::stack::evm_stack_t &stack,
        cuEVM::state::TouchState &touch_state)
    {
        // cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_ZERO);
        bn_t address;
        stack.pop(address, error_code);
        arith.address_conversion(address);
        if (error_code == ERR_NONE)
        {
            touch_state.charge_gas_access_account(
                address,
                gas_used);
            if (arith.has_gas(gas_limit, gas_used, error_code))
            {
                size_t code_size = touch_state.get_account_code_size(
                    address);
                bn_t code_size_bn;
                arith.cgbn_from_size_t(code_size_bn, code_size);

                stack.push(code_size_bn, error_code);

                pc = pc + 1;
            }
        }
    }

    /**
     * The EXTCODECOPY operation implementation.
     * Takes the address, memory offset, code offset and length from the stack and
     * copies the code from the account with the given address at the given code offset for
     * the given length to the memory at the given memory offset.
     * If the code has less bytes than neccessay to fill the memory,
     * the remaining bytes are filled with zeros.
     * The dynamic gas cost is computed as:
     * - word_size = (length + 31) / 32
     * - dynamic_gas_cost = word_size * GAS_MEMORY
     * Adittional gas cost is added for the memory expansion.
     * Gas is charged for accessing the account if it is warm
     * or cold access.
     *
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] pc The program counter.
     * @param[in] stack The stack.
     * @param[in] touch_state The touch state object. The executing world state.
     * @param[out] memory The memory.
    */
    __host__ __device__ int32_t EXTCODECOPY(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &pc,
        cuEVM::stack::evm_stack_t &stack,
        cuEVM::state::TouchState &touch_state,
        cuEVM::memory::evm_memory_t &memory)
    {
        // cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_ZERO);

        bn_t address, memory_offset, code_offset, length;
        stack.pop(address, error_code);
        arith.address_conversion(address);
        stack.pop(memory_offset, error_code);
        stack.pop(code_offset, error_code);
        stack.pop(length, error_code);

        if (error_code == ERR_NONE)
        {
            // compute the dynamic gas cost
            arith.memory_cost(
                gas_used,
                length
            );

            // get the memory expansion gas cost
            memory.grow_cost(
                memory_offset,
                length,
                gas_used,
                error_code);

            touch_state.charge_gas_access_account(
                address,
                gas_used);

            if (error_code == ERR_NONE)
            {
                if (arith.has_gas(gas_limit, gas_used, error_code))
                {
                    size_t available_data;
                    uint8_t *data;
                    data = touch_state.get_account_code_data(
                        address,
                        code_offset,
                        length,
                        available_data);

                    memory.set(
                        data,
                        memory_offset,
                        length,
                        available_data,
                        error_code);

                    pc = pc + 1;
                }
            }
        }
    }

    /**
     * The RETURNDATASIZE operation implementation.
     * Pushes on the stack the size of the return data of the last call.
     *
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     * @param[in] return_data The return data.
    */
    __host__ __device__ int32_t RETURNDATASIZE(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &pc,
        cuEVM::stack::evm_stack_t &stack,
        cuEVM::evm_return_data_t &return_data)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_BASE);
        int32_t error_code = cuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        if (error_code == ERROR_SUCCESS)
        {
            bn_t length;
            size_t length_s;
            length_s = return_data.size();
            arith.cgbn_from_size_t(length, length_s);

            stack.push(length, error_code);

            pc = pc + 1;
        }
    }

    /**
     * The RETURNDATACOPY operation implementation.
     * Takes the memory offset, data offset and length from the stack and
     * copies the return data from the last call at the given data offset for
     * the given length to the memory at the given memory offset.
     * If the return data has less bytes than neccessay to fill the memory,
     * an ERROR is generated.
     * The dynamic gas cost is computed as:
     * - word_size = (length + 31) / 32
     * - dynamic_gas_cost = word_size * GAS_MEMORY
     * Adittional gas cost is added for the memory expansion.
     *
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] pc The program counter.
     * @param[in] stack The stack.
     * @param[out] memory The memory.
     * @param[in] return_data The return data.
    */
    __host__ __device__ int32_t RETURNDATACOPY(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &pc,
        cuEVM::stack::evm_stack_t &stack,
        cuEVM::memory::evm_memory_t &memory,
        cuEVM::evm_return_data_t &return_data)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_VERY_LOW);

        bn_t memory_offset, data_offset, length;
        stack.pop(memory_offset, error_code);
        stack.pop(data_offset, error_code);
        stack.pop(length, error_code);


        if (error_code == ERR_NONE)
        {
            // compute the dynamic gas cost
            arith.memory_cost(
                gas_used,
                length
            );

            // get the memory expansion gas cost
            memory.grow_cost(
                memory_offset,
                length,
                gas_used,
                error_code);

            if (error_code == ERR_NONE)
            {
                if (arith.has_gas(gas_limit, gas_used, error_code))
                {
                    uint8_t *data;
                    size_t data_offset_s, length_s;
                    int32_t overflow;
                    overflow = arith.size_t_from_cgbn(data_offset_s, data_offset);
                    overflow = overflow || arith.size_t_from_cgbn(length_s, length);

                    if (overflow)
                    {
                        error_code = ERROR_RETURN_DATA_OVERFLOW;
                    }
                    else
                    {
                        data = return_data.get(
                            data_offset_s,
                            length_s,
                            error_code);

                        memory.set(
                            data,
                            memory_offset,
                            length,
                            length_s,
                            error_code);

                        pc = pc + 1;
                    }
                }
            }
        }
    }

    /**
     * The EXTCODEHASH operation implementation.
     * Takes the address from the stack and pushes the hash of the code
     * of the account with that address.
     * Gas is charged for accessing the account if it is warm
     * or cold access.
     * If the account does not exist or is empty or the account is
     * selfdestructed, the hash is zero.
     *
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] pc The program counter.
     * @param[inout] stack The stack.
     * @param[in] touch_state The touch state object. The executing world state.
    */
    __host__ __device__ int32_t EXTCODEHASH(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &pc,
        cuEVM::stack::evm_stack_t &stack,
        cuEVM::state::TouchState &touch_state)
    {
        // cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_ZERO);
        bn_t address;
        stack.pop(address, error_code);
        arith.address_conversion(address);
        if (error_code == ERR_NONE)
        {
            touch_state.charge_gas_access_account(
                address,
                gas_used);
            if (arith.has_gas(gas_limit, gas_used, error_code))
            {
                bn_t hash_bn;
                // TODO: look on the difference between destroyed and empty
                if (touch_state.is_empty_account(address))
                {
                    cgbn_set_ui32(arith.env, hash_bn, 0);
                }
                else if (touch_state.is_delete_account(address))
                {
                    cgbn_set_ui32(arith.env, hash_bn, 0);
                }
                else
                {
                    uint8_t *bytecode;
                    size_t code_size;
                    uint8_t hash[HASH_BYTES];

                    // TODO: make more efficient
                    bytecode = touch_state.get_account_code(
                        address);
                    code_size = touch_state.get_account_code_size(
                        address);

                    CuCrypto::keccak::sha3(
                        bytecode,
                        code_size,
                        &(hash[0]),
                        HASH_BYTES);
                    arith.cgbn_from_memory(
                        hash_bn,
                        &(hash[0]));
                }

                stack.push(hash_bn, error_code);

                pc = pc + 1;
            }
        }
    }

    /**
     * The SELFBALANCE operation implementation.
     * Pushes on the stack the balance of the current contract.
     * The current contract is consider the contract that owns the
     * execution code.
     *
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[inout] pc The program counter.
     * @param[out] stack The stack.
     * @param[inout] touch_state The touch state object. The executing world state.
     * @param[in] transaction The transaction.
    */
    __host__ __device__ int32_t SELFBALANCE(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &pc,
        cuEVM::stack::evm_stack_t &stack,
        cuEVM::state::TouchState &touch_state,
        cuEVM::evm_message_call_t &message)
    {
        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_LOW);
        bn_t address;
        message.get_recipient(address);

        int32_t error_code = cuEVM::gas_cost::has_gas(
            arith,
            gas_limit,
            gas_used);
        if (error_code == ERROR_SUCCESS)
        {

            bn_t balance;
            touch_state.get_account_balance(
                address,
                balance);

            stack.push(balance, error_code);

            pc = pc + 1;
        }
    }
}