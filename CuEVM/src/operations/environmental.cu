
#include <CuCrypto/keccak.cuh>
#include <CuEVM/core/byte_array.cuh>
#include <CuEVM/operations/environmental.cuh>
#include <CuEVM/utils/error_codes.cuh>
#include <CuEVM/utils/evm_utils.cuh>
#ifdef BUILD_GO_LIBRARY
#include <CuEVM/utils/library_utils.h>
#endif
namespace CuEVM::operations {
__device__ int32_t SHA3(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                        CuEVM::evm_memory_t &memory
#ifdef BUILD_LIBRARY
                        ,
                        void *simplified_trace_data_ptr
#endif
) {
    gas_used += GAS_KECCAK256;

    // Get the offset and length from the stack
    evm_word_t offset, length;
    int32_t error_code = stack.pop(offset);
    error_code |= stack.pop(length);

    CuEVM::gas_cost::keccak_cost(gas_used, uint256_get_uint32_t(&length));

    CuEVM::gas_t memory_expansion_cost;
    // Get the memory expansion gas cost

    uint32_t offset_u32 = uint256_get_uint32_t(&offset);
    uint32_t length_u32 = uint256_get_uint32_t(&length);
    if (uint256_cmp_word(&length, length_u32) != 0 ||
        (uint256_cmp_word(&offset, offset_u32) != 0) && !uint256_is_zero(&length)) {
        return ERR_MEMORY_INVALID_OFFSET;
    }
    error_code |= CuEVM::gas_cost::memory_grow_cost(&memory, offset_u32, length_u32, memory_expansion_cost, gas_used);
    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        memory.increase_memory_cost(memory_expansion_cost);
        if (length_u32 > 0) memory.grow(offset_u32 + length_u32);

        uint8_t hash_data[CuEVM::hash_size];
        int32_t remaining_input_length = memory.size - offset_u32;
        if (remaining_input_length > length_u32)
            remaining_input_length = length_u32;
        else
            remaining_input_length = max(0, remaining_input_length);
        uint8_t *memory_data = nullptr;
        memory.get(offset_u32, remaining_input_length, memory_data);

        CuCrypto::keccak::sha3(memory_data, remaining_input_length, hash_data, CuEVM::hash_size);

        error_code |= stack.pushx(CuEVM::hash_size, hash_data, CuEVM::hash_size);
    }

    return error_code;
}

__device__ int32_t ADDRESS(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used,
                           const CuEVM::evm_call_context_t *call_context) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        error_code |= call_context->stack_ptr->push(call_context->to);
    }
    return error_code;
}

__device__ int32_t BALANCE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used,
                           const CuEVM::evm_call_context_t *call_context) {
    evm_word_t address;
    int32_t error_code = call_context->stack_ptr->pop(address);
    CuEVM::utils::evm_address_conversion(address);

    error_code |=
        CuEVM::gas_cost::access_account_cost(gas_used, global_state_db_ptr, &address, call_context->snapshot_state);
    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    if (error_code == ERROR_SUCCESS) {
        evm_word_t *balance;
        balance = global_state_db_ptr->get_balance(&address, true);
        if (balance != nullptr) {
            error_code |= call_context->stack_ptr->push(*balance);
        } else {
            error_code |= call_context->stack_ptr->push_uint32(0);
        }
    }
    return error_code;
}

__device__ int32_t ORIGIN(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                          const CuEVM::transaction::TransactionList *transaction_list) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
#ifdef BUILD_GO_LIBRARY
        evm_word_t origin = g_fuzzing_constants->sender_list[transaction_list->sender[INSTANCE_GLOBAL_IDX]];
#else
        evm_word_t origin = transaction_list->sender;
#endif

        error_code |= stack.push(origin);
    }
    return error_code;
}

__device__ int32_t CALLER(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used,
                          const CuEVM::evm_call_context_t *call_context) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        error_code |= call_context->stack_ptr->push(call_context->from);
    }
    return error_code;
}

__device__ int32_t CALLVALUE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used,
                             const CuEVM::evm_call_context_t *call_context) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        error_code |= call_context->stack_ptr->push(call_context->value);
    }
    return error_code;
}

__device__ int32_t CALLDATALOAD(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used,
                                const CuEVM::evm_call_context_t *call_context) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t index;
        error_code |= call_context->stack_ptr->pop(index);
        uint32_t data_offset_ui32;
        // get values saturated to uint32_max, in overflow case
        data_offset_ui32 = uint256_get_uint32_t(&index);

        if (uint256_cmp_word(&index, data_offset_ui32) != 0) data_offset_ui32 = UINT32_MAX;

        if (data_offset_ui32 > call_context->call_data_size) {
            error_code |= call_context->stack_ptr->push(evm_word_t(0));
        } else {
            uint32_t remaining_call_data_size = call_context->call_data_size - data_offset_ui32;
            if (remaining_call_data_size >= CuEVM::word_size) {
                error_code |= call_context->stack_ptr->pushx(
                    CuEVM::word_size, call_context->call_data + data_offset_ui32, CuEVM::word_size);
            } else {
                // padd zero
                uint8_t zero_padding[CuEVM::word_size];
                memset(zero_padding, 0, CuEVM::word_size);
                memcpy(zero_padding, call_context->call_data + data_offset_ui32, remaining_call_data_size);
                error_code |= call_context->stack_ptr->pushx(remaining_call_data_size, zero_padding, CuEVM::word_size);
            }
        }
    }
    return error_code;
}

__device__ int32_t CALLDATASIZE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used,
                                const CuEVM::evm_call_context_t *call_context) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        error_code |= call_context->stack_ptr->push(call_context->call_data_size);
    }
    return error_code;
}

__device__ int32_t CALLDATACOPY(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used,
                                const CuEVM::evm_call_context_t *call_context) {
    gas_used += GAS_VERY_LOW;
    evm_word_t memory_offset, data_offset, length;
    int32_t error_code = call_context->stack_ptr->pop(memory_offset);
    error_code |= call_context->stack_ptr->pop(data_offset);
    error_code |= call_context->stack_ptr->pop(length);
    uint32_t memory_offset_u32 = uint256_get_uint32_t(&memory_offset);
    uint32_t length_u32 = uint256_get_uint32_t(&length);

    if (uint256_cmp_word(&memory_offset, memory_offset_u32) != 0 || uint256_cmp_word(&length, length_u32) != 0) {
        return ERR_MEMORY_INVALID_OFFSET;
    }
    // compute the dynamic gas cost
    CuEVM::gas_cost::memory_cost(gas_used, length_u32);

    // get the memory expansion gas cost
    gas_t memory_expansion_cost;
    error_code |= CuEVM::gas_cost::memory_grow_cost(call_context->memory_ptr, memory_offset_u32, length_u32,
                                                    memory_expansion_cost, gas_used);

    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    if (error_code == ERROR_SUCCESS) {
        call_context->memory_ptr->increase_memory_cost(memory_expansion_cost);
        uint32_t data_offset_u32 = uint256_get_uint32_t(&data_offset);
        uint64_t data_offset_u64 = data_offset_u32;
        if (uint256_cmp_word(&data_offset, data_offset_u32) != 0) data_offset_u64 = UINT32_MAX;
        call_context->memory_ptr->set_buffer_data(call_context->call_data, data_offset_u64,
                                                  call_context->call_data_size, memory_offset_u32, length_u32);
    }
    return error_code;
}

__device__ int32_t CODESIZE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used,
                            const CuEVM::evm_call_context_t *call_context) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        error_code |= call_context->stack_ptr->push(call_context->byte_code_size);
    }
    return error_code;
}

__device__ int32_t CODECOPY(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used,
                            const CuEVM::evm_call_context_t *call_context) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    evm_word_t memory_offset, code_offset, length;
    error_code |= call_context->stack_ptr->pop(memory_offset);
    error_code |= call_context->stack_ptr->pop(code_offset);
    error_code |= call_context->stack_ptr->pop(length);

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
        uint32_t data_offset_u32 = uint256_get_uint32_t(&code_offset);
        uint64_t data_offset_u64 = data_offset_u32;
        if (uint256_cmp_word(&code_offset, data_offset_u32) != 0) data_offset_u64 = UINT32_MAX;
        call_context->memory_ptr->set_buffer_data(call_context->byte_code, data_offset_u64,
                                                  call_context->byte_code_size, memory_offset_u32, length_u32);
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

__device__ int32_t EXTCODESIZE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used,
                               const CuEVM::evm_call_context_t *call_context) {
    // cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_ZERO);
    evm_word_t address;
    int32_t error_code = call_context->stack_ptr->pop(address);
    CuEVM::utils::evm_address_conversion(address);

    CuEVM::gas_cost::access_account_cost(gas_used, global_state_db_ptr, &address, call_context->snapshot_state);
    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    uint32_t code_size = 0;
    uint8_t *code = global_state_db_ptr->get_code(code_size, &address);
    evm_word_t code_size_word;
    uint256_from_word(&code_size_word, code_size);
    error_code |= call_context->stack_ptr->push(code_size_word);
    return error_code;
}

__device__ int32_t EXTCODECOPY(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used,
                               const CuEVM::evm_call_context_t *call_context) {
    evm_word_t address, memory_offset, code_offset, length;
    int32_t error_code = call_context->stack_ptr->pop(address);
    // TODO implement stack.pop_address;
    CuEVM::utils::evm_address_conversion(address);
    error_code |= call_context->stack_ptr->pop(memory_offset);
    error_code |= call_context->stack_ptr->pop(code_offset);
    error_code |= call_context->stack_ptr->pop(length);

    // compute the dynamic gas cost
    CuEVM::gas_cost::memory_cost(gas_used, uint256_get_uint32_t(&length));
    uint32_t memory_offset_u32 = uint256_get_uint32_t(&memory_offset);
    uint32_t length_u32 = uint256_get_uint32_t(&length);
    if ((uint256_cmp_word(&memory_offset, memory_offset_u32) != 0 && !uint256_is_zero(&length)) ||
        uint256_cmp_word(&length, length_u32) != 0) {
        return ERR_MEMORY_INVALID_OFFSET;
    }
    // get the memory expansion gas cost
    gas_t memory_expansion_cost;

    error_code |= CuEVM::gas_cost::memory_grow_cost(call_context->memory_ptr, memory_offset_u32, length_u32,
                                                    memory_expansion_cost, gas_used);

    CuEVM::gas_cost::access_account_cost(gas_used, global_state_db_ptr, &address, call_context->snapshot_state);

    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    if (error_code == ERROR_SUCCESS) {
        call_context->memory_ptr->increase_memory_cost(memory_expansion_cost);
        uint32_t code_size = 0;
        uint8_t *code = global_state_db_ptr->get_code(code_size, &address);

        uint32_t data_offset_ui32 = uint256_get_uint32_t(&code_offset);
        uint64_t data_offset_u64 = data_offset_ui32;
        if (uint256_cmp_word(&code_offset, data_offset_ui32) != 0) data_offset_u64 = UINT32_MAX;
        call_context->memory_ptr->set_buffer_data(code, data_offset_u64, code_size, memory_offset_u32, length_u32);
    }
    return error_code;
}

__device__ int32_t RETURNDATASIZE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used,
                                  const CuEVM::evm_call_context_t *call_context) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    if (error_code == ERROR_SUCCESS) {
        error_code |= call_context->stack_ptr->push_uint32(call_context->dynamic_ret_size);
    }
    return error_code;
}

__device__ int32_t RETURNDATACOPY(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used,
                                  CuEVM::evm_call_context_t *call_context) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    evm_word_t memory_offset, data_offset, length;
    error_code |= call_context->stack_ptr->pop(memory_offset);
    error_code |= call_context->stack_ptr->pop(data_offset);
    error_code |= call_context->stack_ptr->pop(length);

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

    if (data_offset_ui32 > call_context->dynamic_ret_size ||
        (data_offset_ui32 + length_ui32) > call_context->dynamic_ret_size) {
        return ERROR_RETURN_DATA_OVERFLOW;
    }
    if (error_code == ERROR_SUCCESS) {
        memory_ptr->increase_memory_cost(memory_expansion_cost);
        memory_ptr->grow(memory_offset_ui32 + length_ui32);

        call_context->copy_return_data_to_memory(memory_offset_ui32, data_offset_ui32, length_ui32);
        // error_code |= call_context->memory_ptr->set(data.data, memory_offset_ui32, length_ui32);
    }
    return error_code;
}

__device__ int32_t EXTCODEHASH(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used,
                               const CuEVM::evm_call_context_t *call_context) {
    evm_word_t address;
    int32_t error_code = call_context->stack_ptr->pop(address);
    CuEVM::utils::evm_address_conversion(address);

    // this function create new account if not exist
    CuEVM::gas_cost::access_account_cost(gas_used, global_state_db_ptr, &address, call_context->snapshot_state);
    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    uint32_t code_size = 0;
    uint8_t *code = nullptr;

    if (global_state_db_ptr->is_empty_account(&address) || (global_state_db_ptr->is_deleted_account(&address)))

    {
        error_code |= call_context->stack_ptr->push_uint32(0);
        return error_code;
    } else {
        code = global_state_db_ptr->get_code(code_size, &address);
    }

    uint8_t hash_data[CuEVM::hash_size];
    CuCrypto::keccak::sha3(code, code_size, hash_data, CuEVM::hash_size);
    // result is in address_shared[INSTANCE_IDX_PER_BLOCK]

    error_code |= call_context->stack_ptr->pushx(CuEVM::hash_size, hash_data, CuEVM::hash_size);
    return error_code;
}

__device__ int32_t GAS(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    error_code |= stack.push_uint64(gas_limit - gas_used);
    return error_code;
}

__device__ int32_t SELFBALANCE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used,
                               const CuEVM::evm_call_context_t *call_context) {
    gas_used += GAS_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t *balance;
        balance = global_state_db_ptr->get_balance(&call_context->to, true);

        error_code |= call_context->stack_ptr->push(*balance);
    }
    return error_code;
}
}  // namespace CuEVM::operations