#include <CuEVM/operations/storage.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM::operations {
__device__ int32_t SLOAD(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                         CuEVM::StateDb *state_db, evm_call_context_t *call_context) {
    // cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_ZERO);
    evm_word_t *key;
    if (stack.size() < 1) {
        return ERROR_STACK_UNDERFLOW;
    }
    key = stack.get_address_at_index(1);
    stack.reduce_size(1);
    // bn_t storage_address;
    // message.get_storage_address(arith, storage_address);
    // int error_code = CuEVM::gas_cost::sload_cost(gas_used, state_db, &call_context->storage_address, key);

    // get the key warm
    uint32_t address_index;
    ValueStatus *found_value;
    if (state_db->is_warm_key_with_offset(&call_context->storage_address, key, address_index, found_value))
        gas_used += GAS_WARM_ACCESS;
    else
        gas_used += GAS_COLD_SLOAD;

    int error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    if (error_code == ERROR_SUCCESS) {
        evm_word_t *value = state_db->get_storage_with_known_index(call_context->depth, &call_context->storage_address,
                                                                   key, address_index, found_value, true);
        // printf("finish get storage thread %d, value %p\n", INSTANCE_GLOBAL_IDX, value);
        if (value == nullptr)
            error_code |= stack.push_uint32(0);
        else
            error_code |= stack.push(*value);
    }
    return error_code;
}

__device__ int32_t SSTORE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::gas_t &gas_refund,
                          CuEVM::evm_stack_t &stack, CuEVM::StateDb *state_db, evm_call_context_t *call_context) {
    // only if is not a static call
    int32_t error_code = (call_context->static_env ? ERROR_STATIC_CALL_CONTEXT_SSTORE : ERROR_SUCCESS);
    // cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_ZERO);
    gas_t gas_left = gas_limit - gas_used;
    error_code |= (gas_left < GAS_STIPEND ? ERROR_OUT_OF_GAS : error_code);
    if (error_code != ERROR_SUCCESS) {
        return error_code;
    }

    if (stack.size() < 2) {
        return ERROR_STACK_UNDERFLOW;
    }
    // pop the pointers without mem copy
    evm_word_t *key = stack.get_address_at_index(1);
    evm_word_t *value = stack.get_address_at_index(2);
    stack.reduce_size(2);

    uint32_t address_index;
    ValueStatus *found_value;
    error_code |= CuEVM::gas_cost::sstore_cost(gas_used, gas_refund, state_db, &call_context->storage_address, key,
                                               value, address_index, found_value);
    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        state_db->write_storage_with_known_index(call_context->depth, &call_context->storage_address, key, value,
                                                 address_index, found_value, true);
    }

    return error_code;
}
}  // namespace CuEVM::operations
