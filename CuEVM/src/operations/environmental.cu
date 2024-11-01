// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation
// Programme) Author: Stefan-Dan Ciocirlan Data: 2023-11-30
// SPDX-License-Identifier: MIT

#include <CuCrypto/keccak.cuh>
#include <CuEVM/core/byte_array.cuh>
#include <CuEVM/gas_cost.cuh>
#include <CuEVM/operations/environmental.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM::operations {
__host__ __device__ int32_t SHA3(ArithEnv &arith, const bn_t &gas_limit, bn_t &gas_used, CuEVM::evm_stack_t &stack,
                                 CuEVM::evm_memory_t &memory) {
    cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_KECCAK256);
    int32_t error_code = CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        // Get the offset and length from the stack
        bn_t offset, length;
        error_code |= stack.pop(arith, offset);
        error_code |= stack.pop(arith, length);

        CuEVM::gas_cost::keccak_cost(arith, gas_used, length);

        bn_t memory_expansion_cost;
        // Get the memory expansion gas cost
        error_code |= CuEVM::gas_cost::memory_grow_cost(arith, memory, offset, length, memory_expansion_cost, gas_used);

        error_code |= CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);

        if (error_code == ERROR_SUCCESS) {
            memory.increase_memory_cost(arith, memory_expansion_cost);
            CuEVM::byte_array_t memory_input;
            error_code |= memory.get(arith, offset, length, memory_input);
            if (error_code == ERROR_SUCCESS) {
                CuEVM::byte_array_t *hash;
                hash = new CuEVM::byte_array_t(CuEVM::hash_size);
                CuCrypto::keccak::sha3(memory_input.data, memory_input.size, hash->data, hash->size);
                bn_t hash_bn;
                error_code |= cgbn_set_byte_array_t(arith.env, hash_bn, *hash);
                delete hash;
                error_code |= stack.push(arith, hash_bn);
            }
        }
    }
    return error_code;
}

__host__ __device__ int32_t ADDRESS(ArithEnv &arith, const bn_t &gas_limit, bn_t &gas_used, CuEVM::evm_stack_t &stack,
                                    const CuEVM::evm_message_call_t &message) {
    cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_BASE);
    int32_t error_code = CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        bn_t recipient_address;
        message.get_recipient(arith, recipient_address);

        error_code |= stack.push(arith, recipient_address);
    }
    return error_code;
}

__host__ __device__ int32_t BALANCE(ArithEnv &arith, const bn_t &gas_limit, bn_t &gas_used, CuEVM::evm_stack_t &stack,
                                    CuEVM::TouchState &touch_state) {
    // cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_ZERO);
    bn_t address;
    int32_t error_code = stack.pop(arith, address);
    CuEVM::evm_address_conversion(arith, address);
    __SHARED_MEMORY__ evm_word_t address_shared;
    cgbn_store(arith.env, &address_shared, address);
    error_code |= CuEVM::gas_cost::access_account_cost(arith, gas_used, touch_state, &address_shared);
    error_code |= CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);

    if (error_code == ERROR_SUCCESS) {
        bn_t balance;
        touch_state.get_balance(arith, &address_shared, balance);

        error_code |= stack.push(arith, balance);
    }
    return error_code;
}

__host__ __device__ int32_t ORIGIN(ArithEnv &arith, const bn_t &gas_limit, bn_t &gas_used, CuEVM::evm_stack_t &stack,
                                   const CuEVM::evm_transaction_t &transaction) {
    cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_BASE);
    int32_t error_code = CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        bn_t origin;
        transaction.get_sender(arith, origin);

        error_code |= stack.push(arith, origin);
    }
    return error_code;
}

__host__ __device__ int32_t CALLER(ArithEnv &arith, const bn_t &gas_limit, bn_t &gas_used, CuEVM::evm_stack_t &stack,
                                   const CuEVM::evm_message_call_t &message) {
    cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_BASE);
    int32_t error_code = CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        bn_t caller;
        message.get_sender(arith, caller);

        error_code |= stack.push(arith, caller);
    }
    return error_code;
}

__host__ __device__ int32_t CALLVALUE(ArithEnv &arith, const bn_t &gas_limit, bn_t &gas_used, CuEVM::evm_stack_t &stack,
                                      const CuEVM::evm_message_call_t &message) {
    cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_BASE);
    int32_t error_code = CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        bn_t call_value;
        message.get_value(arith, call_value);

        error_code |= stack.push(arith, call_value);
    }
    return error_code;
}

__host__ __device__ int32_t CALLDATALOAD(ArithEnv &arith, const bn_t &gas_limit, bn_t &gas_used,
                                         CuEVM::evm_stack_t &stack, const CuEVM::evm_message_call_t &message) {
    cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_VERY_LOW);
    int32_t error_code = CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        bn_t index;
        error_code |= stack.pop(arith, index);
        uint32_t data_offset_ui32, length_ui32;
        // get values saturated to uint32_max, in overflow case
        data_offset_ui32 = cgbn_get_ui32(arith.env, index);
        if (cgbn_compare_ui32(arith.env, index, data_offset_ui32) != 0) data_offset_ui32 = UINT32_MAX;
        length_ui32 = CuEVM::word_size;
        CuEVM::byte_array_t data = CuEVM::byte_array_t(message.get_data(), data_offset_ui32, length_ui32);

        error_code |= stack.pushx(arith, CuEVM::word_size, data.data, data.size);
    }
    return error_code;
}

__host__ __device__ int32_t CALLDATASIZE(ArithEnv &arith, const bn_t &gas_limit, bn_t &gas_used,
                                         CuEVM::evm_stack_t &stack, const CuEVM::evm_message_call_t &message) {
    cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_BASE);
    int32_t error_code = CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        bn_t length;
        cgbn_set_ui32(arith.env, length, message.get_data().size);

        error_code |= stack.push(arith, length);
    }
    return error_code;
}

__host__ __device__ int32_t CALLDATACOPY(ArithEnv &arith, const bn_t &gas_limit, bn_t &gas_used,
                                         CuEVM::evm_stack_t &stack, const CuEVM::evm_message_call_t &message,
                                         CuEVM::evm_memory_t &memory) {
    cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_VERY_LOW);
    int32_t error_code = CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);

    bn_t memory_offset, data_offset, length;
    error_code |= stack.pop(arith, memory_offset);
    error_code |= stack.pop(arith, data_offset);
    error_code |= stack.pop(arith, length);

    // compute the dynamic gas cost
    CuEVM::gas_cost::memory_cost(arith, gas_used, length);

    // get the memory expansion gas cost
    bn_t memory_expansion_cost;
    error_code |=
        CuEVM::gas_cost::memory_grow_cost(arith, memory, memory_offset, length, memory_expansion_cost, gas_used);

    error_code |= CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);

    if (error_code == ERROR_SUCCESS) {
        memory.increase_memory_cost(arith, memory_expansion_cost);
        uint32_t data_offset_ui32, length_ui32;
        // get values saturated to uint32_max, in overflow case
        data_offset_ui32 = cgbn_get_ui32(arith.env, data_offset);
        if (cgbn_compare_ui32(arith.env, data_offset, data_offset_ui32) != 0) data_offset_ui32 = UINT32_MAX;
        length_ui32 = cgbn_get_ui32(arith.env, length);
        if (cgbn_compare_ui32(arith.env, length, length_ui32) != 0) length_ui32 = UINT32_MAX;
        CuEVM::byte_array_t data = CuEVM::byte_array_t(message.get_data(), data_offset_ui32, length_ui32);

        error_code |= memory.set(arith, data, memory_offset, length);
    }
    return error_code;
}

__host__ __device__ int32_t CODESIZE(ArithEnv &arith, const bn_t &gas_limit, bn_t &gas_used, CuEVM::evm_stack_t &stack,
                                     const CuEVM::evm_message_call_t &message) {
    cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_BASE);
    int32_t error_code = CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        bn_t code_size;
        cgbn_set_ui32(arith.env, code_size, message.get_byte_code().size);

        error_code |= stack.push(arith, code_size);
    }
    return error_code;
}

__host__ __device__ int32_t CODECOPY(ArithEnv &arith, const bn_t &gas_limit, bn_t &gas_used, CuEVM::evm_stack_t &stack,
                                     const CuEVM::evm_message_call_t &message, CuEVM::evm_memory_t &memory) {
    cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_VERY_LOW);
    int32_t error_code = CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);

    bn_t memory_offset, code_offset, length;
    error_code |= stack.pop(arith, memory_offset);
    error_code |= stack.pop(arith, code_offset);
    error_code |= stack.pop(arith, length);

    // compute the dynamic gas cost
    CuEVM::gas_cost::memory_cost(arith, gas_used, length);

    // get the memory expansion gas cost
    bn_t memory_expansion_cost;
    error_code |=
        CuEVM::gas_cost::memory_grow_cost(arith, memory, memory_offset, length, memory_expansion_cost, gas_used);

    error_code |= CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);

    if (error_code == ERROR_SUCCESS) {
        memory.increase_memory_cost(arith, memory_expansion_cost);
        uint32_t data_offset_ui32, length_ui32;
        // get values saturated to uint32_max, in overflow case
        data_offset_ui32 = cgbn_get_ui32(arith.env, code_offset);
        if (cgbn_compare_ui32(arith.env, code_offset, data_offset_ui32) != 0) data_offset_ui32 = UINT32_MAX;
        length_ui32 = cgbn_get_ui32(arith.env, length);
        if (cgbn_compare_ui32(arith.env, length, length_ui32) != 0) length_ui32 = UINT32_MAX;
        CuEVM::byte_array_t data(message.get_byte_code(), data_offset_ui32, length_ui32);

        error_code |= memory.set(arith, data, memory_offset, length);
    }
    return error_code;
}

__host__ __device__ int32_t GASPRICE(ArithEnv &arith, const bn_t &gas_limit, bn_t &gas_used, CuEVM::evm_stack_t &stack,
                                     const CuEVM::block_info_t &block, const CuEVM::evm_transaction_t &transaction) {
    cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_BASE);
    int32_t error_code = CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);
    bn_t gas_price;
    error_code |= transaction.get_gas_price(arith, block, gas_price);
    error_code |= stack.push(arith, gas_price);
    return error_code;
}

__host__ __device__ int32_t EXTCODESIZE(ArithEnv &arith, const bn_t &gas_limit, bn_t &gas_used,
                                        CuEVM::evm_stack_t &stack, CuEVM::TouchState &touch_state) {
    // cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_ZERO);
    bn_t address;
    int32_t error_code = stack.pop(arith, address);
    CuEVM::evm_address_conversion(arith, address);
    __SHARED_MEMORY__ evm_word_t address_shared;
    cgbn_store(arith.env, &address_shared, address);
    CuEVM::gas_cost::access_account_cost(arith, gas_used, touch_state, &address_shared);
    error_code |= CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);
    CuEVM::byte_array_t byte_code;
    // error_code |=
    touch_state.get_code(arith, &address_shared, byte_code);
    bn_t code_size;
    cgbn_set_ui32(arith.env, code_size, byte_code.size);
    error_code |= stack.push(arith, code_size);
    return error_code;
}

__host__ __device__ int32_t EXTCODECOPY(ArithEnv &arith, const bn_t &gas_limit, bn_t &gas_used,
                                        CuEVM::evm_stack_t &stack, CuEVM::TouchState &touch_state,
                                        CuEVM::evm_memory_t &memory) {
    // cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_ZERO);

    bn_t address, memory_offset, code_offset, length;
    int32_t error_code = stack.pop(arith, address);
    // TODO implement stack.pop_address;
    CuEVM::evm_address_conversion(arith, address);
    __SHARED_MEMORY__ evm_word_t address_shared;
    cgbn_store(arith.env, &address_shared, address);
    error_code |= stack.pop(arith, memory_offset);
    error_code |= stack.pop(arith, code_offset);
    error_code |= stack.pop(arith, length);

    // compute the dynamic gas cost
    CuEVM::gas_cost::memory_cost(arith, gas_used, length);

    // get the memory expansion gas cost
    bn_t memory_expansion_cost;
    error_code |=
        CuEVM::gas_cost::memory_grow_cost(arith, memory, memory_offset, length, memory_expansion_cost, gas_used);
    CuEVM::gas_cost::access_account_cost(arith, gas_used, touch_state, &address_shared);

    error_code |= CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);

    if (error_code == ERROR_SUCCESS) {
        memory.increase_memory_cost(arith, memory_expansion_cost);
        CuEVM::byte_array_t byte_code;
        // error_code |=
        touch_state.get_code(arith, &address_shared, byte_code);

        uint32_t data_offset_ui32, length_ui32;
        // get values saturated to uint32_max, in overflow case
        data_offset_ui32 = cgbn_get_ui32(arith.env, code_offset);
        if (cgbn_compare_ui32(arith.env, code_offset, data_offset_ui32) != 0) data_offset_ui32 = UINT32_MAX;
        length_ui32 = cgbn_get_ui32(arith.env, length);
        if (cgbn_compare_ui32(arith.env, length, length_ui32) != 0) length_ui32 = UINT32_MAX;
        CuEVM::byte_array_t data(byte_code, data_offset_ui32, length_ui32);

        error_code |= memory.set(arith, data, memory_offset, length);
    }
    return error_code;
}

__host__ __device__ int32_t RETURNDATASIZE(ArithEnv &arith, const bn_t &gas_limit, bn_t &gas_used,
                                           CuEVM::evm_stack_t &stack, const CuEVM::evm_return_data_t &return_data) {
    cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_BASE);
    int32_t error_code = CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        bn_t length;
        cgbn_set_ui32(arith.env, length, return_data.size);

        error_code |= stack.push(arith, length);
    }
    return error_code;
}

__host__ __device__ int32_t RETURNDATACOPY(ArithEnv &arith, const bn_t &gas_limit, bn_t &gas_used,
                                           CuEVM::evm_stack_t &stack, CuEVM::evm_memory_t &memory,
                                           const CuEVM::evm_return_data_t &return_data) {
    cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_VERY_LOW);
    int32_t error_code = CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);

    bn_t memory_offset, data_offset, length;
    error_code |= stack.pop(arith, memory_offset);
    error_code |= stack.pop(arith, data_offset);
    error_code |= stack.pop(arith, length);

    // compute the dynamic gas cost
    CuEVM::gas_cost::memory_cost(arith, gas_used, length);

    // get the memory expansion gas cost
    bn_t memory_expansion_cost;
    error_code |=
        CuEVM::gas_cost::memory_grow_cost(arith, memory, memory_offset, length, memory_expansion_cost, gas_used);

    error_code |= CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);

    bn_t temp_length;
    int32_t over_flow = cgbn_add(arith.env, temp_length, data_offset, length);
    // #ifdef __CUDA_ARCH__
    //     printf("RETURNDATACOPY: error_code: %d data_length %d idx %d\n", error_code, return_data.size, threadIdx.x);
    //     print_bnt(arith, data_offset);
    //     print_bnt(arith, length);
    //     print_bnt(arith, temp_length);
    // #endif
    // TODO: Check EOF format
    if (over_flow || cgbn_compare_ui32(arith.env, temp_length, return_data.size) > 0) {
        return ERROR_RETURN_DATA_OVERFLOW;
    }
    if (error_code == ERROR_SUCCESS) {
        memory.increase_memory_cost(arith, memory_expansion_cost);

        uint32_t data_offset_ui32, length_ui32;
        // get values saturated to uint32_max, in overflow case
        data_offset_ui32 = cgbn_get_ui32(arith.env, data_offset);
        if (cgbn_compare_ui32(arith.env, data_offset, data_offset_ui32) != 0) data_offset_ui32 = UINT32_MAX;
        length_ui32 = cgbn_get_ui32(arith.env, length);
        if (cgbn_compare_ui32(arith.env, length, length_ui32) != 0) length_ui32 = UINT32_MAX;
        CuEVM::byte_array_t data(return_data, data_offset_ui32, length_ui32);

        error_code |= memory.set(arith, data, memory_offset, length);
    }
    return error_code;
}

__host__ __device__ int32_t EXTCODEHASH(ArithEnv &arith, const bn_t &gas_limit, bn_t &gas_used,
                                        CuEVM::evm_stack_t &stack, CuEVM::TouchState &touch_state) {
    // cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_ZERO);
    bn_t address;
    int32_t error_code = stack.pop(arith, address);
    CuEVM::evm_address_conversion(arith, address);
    __SHARED_MEMORY__ evm_word_t address_shared;
    cgbn_store(arith.env, &address_shared, address);

    CuEVM::gas_cost::access_account_cost(arith, gas_used, touch_state, &address_shared);
    error_code |= CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);
    bn_t hash_bn;
    if ((touch_state.is_empty_account(arith, &address_shared)) ||
        touch_state.is_deleted_account(arith, &address_shared)) {
        cgbn_set_ui32(arith.env, hash_bn, 0);
    } else {
        CuEVM::byte_array_t byte_code;
        error_code |= touch_state.get_code(arith, &address_shared, byte_code);
        CuEVM::byte_array_t hash(CuEVM::hash_size);
        CuCrypto::keccak::sha3(byte_code.data, byte_code.size, hash.data, hash.size);
        error_code |= cgbn_set_byte_array_t(arith.env, hash_bn, hash);
    }
    error_code |= stack.push(arith, hash_bn);
    return error_code;
}

__host__ __device__ int32_t SELFBALANCE(ArithEnv &arith, const bn_t &gas_limit, bn_t &gas_used,
                                        CuEVM::evm_stack_t &stack, CuEVM::TouchState &touch_state,
                                        const CuEVM::evm_message_call_t &message) {
    cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_LOW);
    // bn_t address;
    // message.get_recipient(arith, address);
    int32_t error_code = CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        bn_t balance;
        touch_state.get_balance(arith, &message.recipient, balance);

        error_code |= stack.push(arith, balance);
    }
    return error_code;
}
}  // namespace CuEVM::operations