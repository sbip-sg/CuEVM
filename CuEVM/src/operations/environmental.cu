
#include <CuCrypto/keccak.cuh>
#include <CuEVM/core/byte_array.cuh>
#include <CuEVM/gas_cost.cuh>
#include <CuEVM/operations/environmental.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM::operations {
__host__ __device__ int32_t SHA3(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                                 CuEVM::evm_memory_t &memory) {
    gas_used += GAS_KECCAK256;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        // Get the offset and length from the stack
        evm_word_t offset, length;
        error_code |= stack.pop(offset);
        error_code |= stack.pop(length);

        CuEVM::gas_cost::keccak_cost(gas_used, uint256_get_uint32_t(&length));

        CuEVM::gas_t memory_expansion_cost;
        // Get the memory expansion gas cost
        error_code |= CuEVM::gas_cost::memory_grow_cost(memory, offset, length, memory_expansion_cost, gas_used);

        error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);

        if (error_code == ERROR_SUCCESS) {
            memory.increase_memory_cost(memory_expansion_cost);
            CuEVM::byte_array_t memory_input;
            error_code |= memory.get(offset, length, memory_input);
            if (error_code == ERROR_SUCCESS) {
                // CuEVM::byte_array_t *hash;
                // hash = new CuEVM::byte_array_t(CuEVM::hash_size);
                // CuCrypto::keccak::sha3(memory_input.data, memory_input.size, hash->data, hash->size);
                // bn_t hash_bn;
                // error_code |= cgbn_set_byte_array_t(arith.env, hash_bn, *hash);
                // delete hash;
                // error_code |= stack.push(arith, hash_bn);
                uint8_t hash_data[CuEVM::hash_size];
                CuCrypto::keccak::sha3(memory_input.data, memory_input.size, hash_data, CuEVM::hash_size);
                error_code |= stack.pushx(CuEVM::word_size, hash_data, CuEVM::hash_size);
            }
        }
    }
    return error_code;
}

__host__ __device__ int32_t ADDRESS(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                                    const CuEVM::evm_message_call_t &message) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t recipient_address;
        message.get_recipient(recipient_address);

        error_code |= stack.push(recipient_address);
    }
    return error_code;
}

__host__ __device__ int32_t BALANCE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                                    CuEVM::TouchState &touch_state) {
    // cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_ZERO);
    evm_word_t address;
    int32_t error_code = stack.pop(address);
    evm_address_conversion(address);

    error_code |= CuEVM::gas_cost::access_account_cost(gas_used, touch_state, &address);
    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    if (error_code == ERROR_SUCCESS) {
        evm_word_t balance;
        touch_state.get_balance(&address, balance);

        error_code |= stack.push(balance);
    }
    return error_code;
}

__host__ __device__ int32_t ORIGIN(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                                   const CuEVM::evm_transaction_t &transaction) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t origin;
        transaction.get_sender(origin);

        error_code |= stack.push(origin);
    }
    return error_code;
}

__host__ __device__ int32_t CALLER(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                                   const CuEVM::evm_message_call_t &message) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t caller;
        message.get_sender(caller);

        error_code |= stack.push(caller);
    }
    return error_code;
}

__host__ __device__ int32_t CALLVALUE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                                      const CuEVM::evm_message_call_t &message) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t call_value;
        message.get_value(call_value);

        error_code |= stack.push(call_value);
    }
    return error_code;
}

__host__ __device__ int32_t CALLDATALOAD(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used,
                                         CuEVM::evm_stack_t &stack, const CuEVM::evm_message_call_t &message) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t index;
        error_code |= stack.pop(index);
        uint32_t data_offset_ui32, length_ui32;
        // get values saturated to uint32_max, in overflow case
        data_offset_ui32 = uint256_get_uint32_t(&index);
        // printf("CALLDATALOAD: error_code: %d data_length %d idx %d\n", error_code, message.get_data().size,
        //        data_offset_ui32);
        if (uint256_cmp_word(&index, data_offset_ui32) != 0) data_offset_ui32 = UINT32_MAX;
        length_ui32 = CuEVM::word_size;
        CuEVM::byte_array_t data = CuEVM::byte_array_t(message.get_data(), data_offset_ui32, length_ui32);

        error_code |= stack.pushx(CuEVM::word_size, data.data, data.size);
    }
    return error_code;
}

__host__ __device__ int32_t CALLDATASIZE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used,
                                         CuEVM::evm_stack_t &stack, const CuEVM::evm_message_call_t &message) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t length;
        uint256_from_word(&length, message.get_data().size);

        error_code |= stack.push(length);
    }
    return error_code;
}

__host__ __device__ int32_t CALLDATACOPY(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used,
                                         CuEVM::evm_stack_t &stack, const CuEVM::evm_message_call_t &message,
                                         CuEVM::evm_memory_t &memory) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    evm_word_t memory_offset, data_offset, length;
    error_code |= stack.pop(memory_offset);
    error_code |= stack.pop(data_offset);
    error_code |= stack.pop(length);

    // compute the dynamic gas cost
    CuEVM::gas_cost::memory_cost(gas_used, uint256_get_uint32_t(&length));

    // get the memory expansion gas cost
    gas_t memory_expansion_cost;
    error_code |= CuEVM::gas_cost::memory_grow_cost(memory, memory_offset, length, memory_expansion_cost, gas_used);

    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    if (error_code == ERROR_SUCCESS) {
        memory.increase_memory_cost(memory_expansion_cost);
        uint32_t data_offset_ui32, length_ui32;
        // get values saturated to uint32_max, in overflow case
        data_offset_ui32 = uint256_get_uint32_t(&data_offset);
        if (uint256_cmp_word(&data_offset, data_offset_ui32) != 0) data_offset_ui32 = UINT32_MAX;
        length_ui32 = uint256_get_uint32_t(&length);
        if (uint256_cmp_word(&length, length_ui32) != 0) length_ui32 = UINT32_MAX;
        CuEVM::byte_array_t data = CuEVM::byte_array_t(message.get_data(), data_offset_ui32, length_ui32);

        error_code |= memory.set(data, memory_offset, length);
    }
    return error_code;
}

__host__ __device__ int32_t CODESIZE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                                     const CuEVM::evm_message_call_t &message) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t code_size;
        uint256_from_word(&code_size, message.get_byte_code().size);

        error_code |= stack.push(code_size);
    }
    return error_code;
}

__host__ __device__ int32_t CODECOPY(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                                     const CuEVM::evm_message_call_t &message, CuEVM::evm_memory_t &memory) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    evm_word_t memory_offset, code_offset, length;
    error_code |= stack.pop(memory_offset);
    error_code |= stack.pop(code_offset);
    error_code |= stack.pop(length);

    // compute the dynamic gas cost
    CuEVM::gas_cost::memory_cost(gas_used, uint256_get_uint32_t(&length));

    // get the memory expansion gas cost
    gas_t memory_expansion_cost;
    error_code |= CuEVM::gas_cost::memory_grow_cost(memory, memory_offset, length, memory_expansion_cost, gas_used);

    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    if (error_code == ERROR_SUCCESS) {
        memory.increase_memory_cost(memory_expansion_cost);
        uint32_t data_offset_ui32, length_ui32;
        // get values saturated to uint32_max, in overflow case
        data_offset_ui32 = uint256_get_uint32_t(&code_offset);
        if (uint256_cmp_word(&code_offset, data_offset_ui32) != 0) data_offset_ui32 = UINT32_MAX;
        length_ui32 = uint256_get_uint32_t(&length);
        if (uint256_cmp_word(&length, length_ui32) != 0) length_ui32 = UINT32_MAX;
        CuEVM::byte_array_t data(message.get_byte_code(), data_offset_ui32, length_ui32);

        error_code |= memory.set(data, memory_offset, length);
    }
    return error_code;
}

__host__ __device__ int32_t GASPRICE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                                     const CuEVM::block_info_t &block, const CuEVM::evm_transaction_t &transaction) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    evm_word_t gas_price;
    error_code |= transaction.get_gas_price(block, gas_price);
    error_code |= stack.push(gas_price);
    return error_code;
}

__host__ __device__ int32_t EXTCODESIZE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used,
                                        CuEVM::evm_stack_t &stack, CuEVM::TouchState &touch_state) {
    // cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_ZERO);
    evm_word_t address;
    int32_t error_code = stack.pop(address);
    CuEVM::evm_address_conversion(address);

    CuEVM::gas_cost::access_account_cost(gas_used, touch_state, &address);
    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    CuEVM::byte_array_t byte_code;
    // error_code |=
    touch_state.get_code(&address, byte_code);
    evm_word_t code_size;
    uint256_from_word(&code_size, byte_code.size);
    error_code |= stack.push(code_size);
    return error_code;
}

__host__ __device__ int32_t EXTCODECOPY(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used,
                                        CuEVM::evm_stack_t &stack, CuEVM::TouchState &touch_state,
                                        CuEVM::evm_memory_t &memory) {
    // cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_ZERO);

    evm_word_t address, memory_offset, code_offset, length;
    int32_t error_code = stack.pop(address);
    // TODO implement stack.pop_address;
    CuEVM::evm_address_conversion(address);
    error_code |= stack.pop(memory_offset);
    error_code |= stack.pop(code_offset);
    error_code |= stack.pop(length);

    // compute the dynamic gas cost
    CuEVM::gas_cost::memory_cost(gas_used, uint256_get_uint32_t(&length));

    // get the memory expansion gas cost
    gas_t memory_expansion_cost;
    error_code |= CuEVM::gas_cost::memory_grow_cost(memory, memory_offset, length, memory_expansion_cost, gas_used);
    CuEVM::gas_cost::access_account_cost(gas_used, touch_state, &address);

    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    if (error_code == ERROR_SUCCESS) {
        memory.increase_memory_cost(memory_expansion_cost);
        CuEVM::byte_array_t byte_code;
        // error_code |=
        touch_state.get_code(&address, byte_code);

        uint32_t data_offset_ui32, length_ui32;
        // get values saturated to uint32_max, in overflow case
        data_offset_ui32 = uint256_get_uint32_t(&code_offset);
        if (uint256_cmp_word(&code_offset, data_offset_ui32) != 0) data_offset_ui32 = UINT32_MAX;
        length_ui32 = uint256_get_uint32_t(&length);
        if (uint256_cmp_word(&length, length_ui32) != 0) length_ui32 = UINT32_MAX;
        CuEVM::byte_array_t data(byte_code, data_offset_ui32, length_ui32);

        error_code |= memory.set(data, memory_offset, length);
    }
    return error_code;
}

__host__ __device__ int32_t RETURNDATASIZE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used,
                                           CuEVM::evm_stack_t &stack, const CuEVM::evm_return_data_t &return_data) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t length;
        uint256_from_word(&length, return_data.size);

        error_code |= stack.push(length);
    }
    return error_code;
}

__host__ __device__ int32_t RETURNDATACOPY(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used,
                                           CuEVM::evm_stack_t &stack, CuEVM::evm_memory_t &memory,
                                           const CuEVM::evm_return_data_t &return_data) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    evm_word_t memory_offset, data_offset, length;
    error_code |= stack.pop(memory_offset);
    error_code |= stack.pop(data_offset);
    error_code |= stack.pop(length);

    // compute the dynamic gas cost
    CuEVM::gas_cost::memory_cost(gas_used, uint256_get_uint32_t(&length));

    // get the memory expansion gas cost
    gas_t memory_expansion_cost;
    error_code |= CuEVM::gas_cost::memory_grow_cost(memory, memory_offset, length, memory_expansion_cost, gas_used);

    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    evm_word_t temp_length;
    // int32_t over_flow = uint256_add_word(&temp_length, &data_offset, length);
    // #ifdef __CUDA_ARCH__
    //     printf("RETURNDATACOPY: error_code: %d data_length %d idx %d\n", error_code, return_data.size, threadIdx.x);
    //     print_bnt(arith, data_offset);
    //     print_bnt(arith, length);
    //     print_bnt(arith, temp_length);
    // #endif
    // TODO: Check EOF format
    // if (over_flow || uint256_cmp_word(&temp_length, return_data.size) > 0) {
    //     return ERROR_RETURN_DATA_OVERFLOW;
    // }
    if (error_code == ERROR_SUCCESS) {
        memory.increase_memory_cost(memory_expansion_cost);

        uint32_t data_offset_ui32, length_ui32;
        // get values saturated to uint32_max, in overflow case
        data_offset_ui32 = uint256_get_uint32_t(&data_offset);
        if (uint256_cmp_word(&data_offset, data_offset_ui32) != 0) data_offset_ui32 = UINT32_MAX;
        length_ui32 = uint256_get_uint32_t(&length);
        if (uint256_cmp_word(&length, length_ui32) != 0) length_ui32 = UINT32_MAX;
        CuEVM::byte_array_t data(return_data, data_offset_ui32, length_ui32);

        error_code |= memory.set(data, memory_offset, length);
    }
    return error_code;
}

__host__ __device__ int32_t EXTCODEHASH(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used,
                                        CuEVM::evm_stack_t &stack, CuEVM::TouchState &touch_state) {
    // cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_ZERO);
    evm_word_t address;
    int32_t error_code = stack.pop(address);
    CuEVM::evm_address_conversion(address);

    CuEVM::gas_cost::access_account_cost(gas_used, touch_state, &address);
    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    // bn_t hash_bn;
    if ((touch_state.is_empty_account(&address)) || touch_state.is_deleted_account(&address)) {
        // cgbn_set_ui32(arith.env, hash_bn, 0);
        address.set_zero();
    } else {
        CuEVM::byte_array_t byte_code;
        error_code |= touch_state.get_code(&address, byte_code);
        CuEVM::byte_array_t hash(CuEVM::hash_size);
        CuCrypto::keccak::sha3(byte_code.data, byte_code.size, hash.data, hash.size);
        // error_code |= cgbn_set_byte_array_t(arith.env, hash_bn, hash);
        address.from_byte_array_t(hash, BIG_ENDIAN);
    }
    // result is in address_shared[INSTANCE_IDX_PER_BLOCK]
    error_code |= stack.push(address);
    return error_code;
}

__host__ __device__ int32_t SELFBALANCE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used,
                                        CuEVM::evm_stack_t &stack, CuEVM::TouchState &touch_state,
                                        const CuEVM::evm_message_call_t &message) {
    CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    // bn_t address;
    // message.get_recipient(arith, address);
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t balance;
        touch_state.get_balance(&message.recipient, balance);

        error_code |= stack.push(balance);
    }
    return error_code;
}
}  // namespace CuEVM::operations