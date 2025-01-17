
#include <CuCrypto/keccak.cuh>
#include <CuEVM/core/byte_array.cuh>
#include <CuEVM/operations/environmental.cuh>
#include <CuEVM/utils/error_codes.cuh>
#include <CuEVM/utils/evm_utils.cuh>
namespace CuEVM::operations {
__device__ int32_t SHA3(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                        CuEVM::evm_memory_t &memory) {
    gas_used += GAS_KECCAK256;
    // int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    // if (error_code == ERROR_SUCCESS) {
    // Get the offset and length from the stack
    evm_word_t offset, length;
    int32_t error_code = stack.pop(offset);
    error_code |= stack.pop(length);

    CuEVM::gas_cost::keccak_cost(gas_used, uint256_get_uint32_t(&length));

    CuEVM::gas_t memory_expansion_cost;
    // Get the memory expansion gas cost

    uint32_t offset_u32 = uint256_get_uint32_t(&offset);
    uint32_t length_u32 = uint256_get_uint32_t(&length);
    if (uint256_cmp_word(&length, length_u32) != 0 || uint256_cmp_word(&offset, offset_u32) != 0) {
        return ERR_MEMORY_INVALID_OFFSET;
    }
    error_code |= CuEVM::gas_cost::memory_grow_cost(&memory, offset_u32, length_u32, memory_expansion_cost, gas_used);

    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    if (error_code == ERROR_SUCCESS) {
        memory.increase_memory_cost(memory_expansion_cost);

        uint8_t *memory_input = new uint8_t[length_u32];
        uint32_t memory_input_size = length_u32;
        error_code |= memory.copy(offset_u32, length_u32, memory_input);
        if (error_code == ERROR_SUCCESS) {
            uint8_t hash_data[CuEVM::hash_size];
            CuCrypto::keccak::sha3(memory_input, memory_input_size, hash_data, CuEVM::hash_size);
            evm_word_t hash_word;
            uint256_from_bytes(&hash_word, hash_data, CuEVM::hash_size);
            error_code |= stack.push(hash_word);
        }
        delete[] memory_input;
    }
    // }
    return error_code;
}

__device__ int32_t ADDRESS(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                           const CuEVM::evm_call_context_t *call_context) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        error_code |= stack.push(call_context->to);
    }
    return error_code;
}

__device__ int32_t BALANCE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                           CuEVM::StateDb *state_db) {
    // cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_ZERO);
    evm_word_t address;
    int32_t error_code = stack.pop(address);
    CuEVM::utils::evm_address_conversion(address);

    error_code |= CuEVM::gas_cost::access_account_cost(gas_used, state_db, &address);
    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    if (error_code == ERROR_SUCCESS) {
        evm_word_t *balance;
        balance = state_db->get_balance(&address);

        error_code |= stack.push(*balance);
    }
    return error_code;
}

__device__ int32_t ORIGIN(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                          const CuEVM::transaction::TransactionList *transaction_list) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t origin = transaction_list->sender;

        error_code |= stack.push(origin);
    }
    return error_code;
}

__device__ int32_t CALLER(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                          const CuEVM::evm_call_context_t *call_context) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        error_code |= stack.push(call_context->from);
    }
    return error_code;
}

__device__ int32_t CALLVALUE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                             const CuEVM::evm_call_context_t *call_context) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        error_code |= stack.push(call_context->value);
    }
    return error_code;
}

__device__ int32_t CALLDATALOAD(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                                const CuEVM::evm_call_context_t *call_context) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t index;
        error_code |= stack.pop(index);
        uint32_t data_offset_ui32;
        // get values saturated to uint32_max, in overflow case
        data_offset_ui32 = uint256_get_uint32_t(&index);

        if (uint256_cmp_word(&index, data_offset_ui32) != 0) data_offset_ui32 = UINT32_MAX;

        if (data_offset_ui32 > call_context->call_data_size) {
            error_code |= stack.push(evm_word_t(0));
        } else {
            error_code |= stack.pushx(CuEVM::word_size, call_context->call_data + data_offset_ui32, CuEVM::word_size);
        }
    }
    return error_code;
}

__device__ int32_t CALLDATASIZE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                                const CuEVM::evm_call_context_t *call_context) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        error_code |= stack.push(call_context->call_data_size);
    }
    return error_code;
}

__device__ int32_t CALLDATACOPY(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                                const CuEVM::evm_call_context_t *call_context, CuEVM::evm_memory_t &memory) {
    gas_used += GAS_VERY_LOW;
    evm_word_t memory_offset, data_offset, length;
    int32_t error_code = stack.pop(memory_offset);
    error_code |= stack.pop(data_offset);
    error_code |= stack.pop(length);
    uint32_t memory_offset_u32 = uint256_get_uint32_t(&memory_offset);
    uint32_t length_u32 = uint256_get_uint32_t(&length);

    if (uint256_cmp_word(&memory_offset, memory_offset_u32) != 0 || uint256_cmp_word(&length, length_u32) != 0) {
        return ERR_MEMORY_INVALID_OFFSET;
    }
    // compute the dynamic gas cost
    CuEVM::gas_cost::memory_cost(gas_used, length_u32);

    // get the memory expansion gas cost
    gas_t memory_expansion_cost;
    error_code |=
        CuEVM::gas_cost::memory_grow_cost(&memory, memory_offset_u32, length_u32, memory_expansion_cost, gas_used);

    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    if (error_code == ERROR_SUCCESS) {
        memory.increase_memory_cost(memory_expansion_cost);
        uint32_t data_offset_u32 = uint256_get_uint32_t(&data_offset);
        if (uint256_cmp_word(&data_offset, data_offset_u32) != 0) data_offset_u32 = UINT32_MAX;
        uint64_t total_data_length = length_u32 + data_offset_u32;

        printf("calldatacopy data_length_remaining %d , > 0 %d\n", total_data_length, total_data_length > 0);
        printf("call_context->call_data_size %d\n", call_context->call_data_size);
        printf("length_u32 %d\n", length_u32);
        printf("data_offset_u32 %d\n", data_offset_u32);
        if (total_data_length <= call_context->call_data_size && length_u32 != 0) {
            printf("if condition data_length_remaining > 0 && length_u32 <= data_length_remaining\n");
            memory.set(call_context->call_data + data_offset_u32, length_u32, memory_offset_u32, length_u32);
        } else {
            memory.set_zero(memory_offset_u32, length_u32);
        }
    }
    return error_code;
}

__device__ int32_t CODESIZE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                            const CuEVM::evm_call_context_t *call_context) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        error_code |= stack.push(call_context->byte_code_size);
    }
    return error_code;
}

__device__ int32_t CODECOPY(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                            const CuEVM::evm_call_context_t *call_context) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    evm_word_t memory_offset, code_offset, length;
    error_code |= stack.pop(memory_offset);
    error_code |= stack.pop(code_offset);
    error_code |= stack.pop(length);

    // compute the dynamic gas cost
    CuEVM::gas_cost::memory_cost(gas_used, uint256_get_uint32_t(&length));

    // get the memory expansion gas cost
    uint32_t memory_offset_u32 = uint256_get_uint32_t(&memory_offset);
    uint32_t length_u32 = uint256_get_uint32_t(&length);
    if (uint256_cmp_word(&memory_offset, memory_offset_u32) != 0 || uint256_cmp_word(&length, length_u32) != 0) {
        return ERR_MEMORY_INVALID_OFFSET;
    }
    gas_t memory_expansion_cost;
    error_code |= CuEVM::gas_cost::memory_grow_cost(call_context->memory_ptr, memory_offset_u32, length_u32,
                                                    memory_expansion_cost, gas_used);

    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    if (error_code == ERROR_SUCCESS) {
        call_context->memory_ptr->increase_memory_cost(memory_expansion_cost);
        uint32_t data_offset_ui32, length_ui32;
        // get values saturated to uint32_max, in overflow case
        data_offset_ui32 = uint256_get_uint32_t(&code_offset);
        if (uint256_cmp_word(&code_offset, data_offset_ui32) != 0) data_offset_ui32 = UINT32_MAX;
        length_ui32 = uint256_get_uint32_t(&length);
        if (uint256_cmp_word(&length, length_ui32) != 0) length_ui32 = UINT32_MAX;
        // TODO: fix this
        // CuEVM::byte_array_t data(message.get_byte_code(), data_offset_ui32, length_ui32);
        printf("code_offset %d, data_offset_ui32 %d, length_ui32 %d, call_context->byte_code_size %d\n",
               uint256_get_uint32_t(&code_offset), data_offset_ui32, length_ui32, call_context->byte_code_size);
        int64_t data_length_remaining = call_context->byte_code_size - (length_ui32 + data_offset_ui32);
        printf("data_length_remaining %d\n", data_length_remaining);
        printf("length_ui32 %d\n", length_ui32);
        for (uint32_t i = 0; i < length_ui32; i++) {
            printf("%x ", call_context->byte_code[data_offset_ui32 + i]);
        }
        printf("\n");
        if (data_length_remaining > 0) {
            call_context->memory_ptr->set(call_context->byte_code + data_offset_ui32, data_length_remaining,
                                          memory_offset_u32, length_ui32);
        } else {
            call_context->memory_ptr->set_zero(memory_offset_u32, length_ui32);
        }
    }
    return error_code;
}

__device__ int32_t GASPRICE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                            const CuEVM::block_info_t &block,
                            const CuEVM::transaction::TransactionList *transaction_list) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    evm_word_t gas_price = transaction_list->gas_price;
    error_code |= stack.push(gas_price);
    return error_code;
}

__device__ int32_t EXTCODESIZE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                               CuEVM::StateDb *state_db) {
    // cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_ZERO);
    evm_word_t address;
    int32_t error_code = stack.pop(address);
    CuEVM::utils::evm_address_conversion(address);

    CuEVM::gas_cost::access_account_cost(gas_used, state_db, &address);
    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    uint32_t code_size = 0;
    uint8_t *code = state_db->get_code(code_size, &address);
    CuEVM::byte_array_t byte_code(code, code_size);
    evm_word_t code_size_word;
    uint256_from_word(&code_size_word, code_size);
    error_code |= stack.push(code_size_word);
    return error_code;
}

__device__ int32_t EXTCODECOPY(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                               CuEVM::StateDb *state_db, CuEVM::evm_memory_t &memory) {
    // cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_ZERO);

    evm_word_t address, memory_offset, code_offset, length;
    int32_t error_code = stack.pop(address);
    // TODO implement stack.pop_address;
    CuEVM::utils::evm_address_conversion(address);
    error_code |= stack.pop(memory_offset);
    error_code |= stack.pop(code_offset);
    error_code |= stack.pop(length);

    // compute the dynamic gas cost
    CuEVM::gas_cost::memory_cost(gas_used, uint256_get_uint32_t(&length));
    uint32_t memory_offset_ui32 = uint256_get_uint32_t(&memory_offset);
    uint32_t length_ui32 = uint256_get_uint32_t(&length);
    if (uint256_cmp_word(&memory_offset, memory_offset_ui32) != 0 || uint256_cmp_word(&length, length_ui32) != 0) {
        return ERR_MEMORY_INVALID_OFFSET;
    }
    // get the memory expansion gas cost
    gas_t memory_expansion_cost;

    error_code |=
        CuEVM::gas_cost::memory_grow_cost(&memory, memory_offset_ui32, length_ui32, memory_expansion_cost, gas_used);

    CuEVM::gas_cost::access_account_cost(gas_used, state_db, &address);

    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    if (error_code == ERROR_SUCCESS) {
        memory.increase_memory_cost(memory_expansion_cost);
        uint32_t code_size = 0;
        uint8_t *code = state_db->get_code(code_size, &address);
        CuEVM::byte_array_t byte_code(code, code_size);

        uint32_t data_offset_ui32, length_ui32;
        // get values saturated to uint32_max, in overflow case
        data_offset_ui32 = uint256_get_uint32_t(&code_offset);
        if (uint256_cmp_word(&code_offset, data_offset_ui32) != 0) data_offset_ui32 = UINT32_MAX;
        length_ui32 = uint256_get_uint32_t(&length);
        if (uint256_cmp_word(&length, length_ui32) != 0) length_ui32 = UINT32_MAX;
        int64_t data_length_remaining = code_size - (length_ui32 + data_offset_ui32);
        if (data_length_remaining > 0 && data_length_remaining < length_ui32) {
            memory.set(code + data_offset_ui32, data_length_remaining, memory_offset_ui32, length_ui32);
        } else {
            memory.set_zero(memory_offset_ui32, length_ui32);
        }
    }
    return error_code;
}

__device__ int32_t RETURNDATASIZE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                                  const CuEVM::evm_call_context_t *call_context) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        error_code |= stack.push_uint32(call_context->dynamic_ret_size);
    }
    return error_code;
}

__device__ int32_t RETURNDATACOPY(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                                  CuEVM::evm_call_context_t *call_context) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    evm_word_t memory_offset, data_offset, length;
    error_code |= stack.pop(memory_offset);
    error_code |= stack.pop(data_offset);
    error_code |= stack.pop(length);

    uint32_t memory_offset_ui32 = uint256_get_uint32_t(&memory_offset);
    uint32_t length_ui32 = uint256_get_uint32_t(&length);
    uint32_t data_offset_ui32 = uint256_get_uint32_t(&data_offset);
    // compute the dynamic gas cost
    CuEVM::gas_cost::memory_cost(gas_used, length_ui32);

    if (uint256_cmp_word(&memory_offset, memory_offset_ui32) != 0 || uint256_cmp_word(&length, length_ui32) != 0 ||
        uint256_cmp_word(&data_offset, data_offset_ui32) != 0) {
        return ERR_MEMORY_INVALID_OFFSET;
    }
    // get the memory expansion gas cost
    gas_t memory_expansion_cost;
    CuEVM::evm_memory_t *memory_ptr = call_context->memory_ptr;
    error_code |=
        CuEVM::gas_cost::memory_grow_cost(memory_ptr, memory_offset_ui32, length_ui32, memory_expansion_cost, gas_used);

    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    // TODO: Check EOF format
    if (length_ui32 > call_context->dynamic_ret_size) {
        return ERROR_RETURN_DATA_OVERFLOW;
    }
    if (error_code == ERROR_SUCCESS) {
        memory_ptr->increase_memory_cost(memory_expansion_cost);
        memory_ptr->grow(memory_offset_ui32 + length_ui32);
        call_context->copy_return_data(call_context->memory_ptr->data + memory_offset_ui32, data_offset_ui32,
                                       length_ui32);
        // error_code |= call_context->memory_ptr->set(data.data, memory_offset_ui32, length_ui32);
    }
    return error_code;
}

__device__ int32_t EXTCODEHASH(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                               CuEVM::StateDb *state_db) {
    // cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_ZERO);
    evm_word_t address;
    int32_t error_code = stack.pop(address);
    CuEVM::utils::evm_address_conversion(address);

    CuEVM::gas_cost::access_account_cost(gas_used, state_db, &address);
    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    // bn_t hash_bn;
    if (state_db->is_empty_account(&address) || state_db->is_deleted_account(&address)) {
        // cgbn_set_ui32(arith.env, hash_bn, 0);
        address.set_zero();
    } else {
        uint32_t code_size = 0;
        uint8_t *code = state_db->get_code(code_size, &address);
        CuEVM::byte_array_t byte_code(code, code_size);
        CuEVM::byte_array_t hash(CuEVM::hash_size);
        CuCrypto::keccak::sha3(byte_code.data, byte_code.size, hash.data, hash.size);
        // error_code |= cgbn_set_byte_array_t(arith.env, hash_bn, hash);
        address.from_byte_array_t(hash, BIG_ENDIAN);
    }
    // result is in address_shared[INSTANCE_IDX_PER_BLOCK]
    error_code |= stack.push(address);
    return error_code;
}

__device__ int32_t SELFBALANCE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                               CuEVM::StateDb *state_db, const CuEVM::evm_call_context_t *call_context) {
    CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    // bn_t address;
    // message.get_recipient(arith, address);
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t *balance;
        balance = state_db->get_balance(&call_context->to);

        error_code |= stack.push(*balance);
    }
    return error_code;
}
}  // namespace CuEVM::operations