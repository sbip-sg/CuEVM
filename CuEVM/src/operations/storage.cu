
#include <CuEVM/gas_cost.cuh>
#include <CuEVM/operations/storage.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM::operations {
__host__ __device__ int32_t SLOAD(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                                  CuEVM::TouchState &touch_state, const CuEVM::evm_message_call_t &message) {
    // cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_ZERO);
    evm_word_t key;
    int32_t error_code = stack.pop(key);
    // bn_t storage_address;
    // message.get_storage_address(arith, storage_address);
    error_code |= CuEVM::gas_cost::sload_cost(gas_used, touch_state, &message.storage_address, key);
    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    // #ifdef __CUDA_ARCH__
    //     printf("SLOAD %d error_code: %d\n", threadIdx.x, error_code);
    // #endif
    if (error_code == ERROR_SUCCESS) {
        evm_word_t value;
        error_code |= touch_state.get_value(&message.storage_address, key, value);
        error_code |= stack.push(value);
    }
    return error_code;
}

__host__ __device__ int32_t SSTORE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::gas_t &gas_refund,
                                   CuEVM::evm_stack_t &stack, CuEVM::TouchState &touch_state,
                                   const CuEVM::evm_message_call_t &message) {
    // only if is not a static call
    int32_t error_code = (message.get_static_env() ? ERROR_STATIC_CALL_CONTEXT_SSTORE : ERROR_SUCCESS);
    // cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_ZERO);
    gas_t gas_left;
    gas_left = gas_limit - gas_used;
    error_code |= (gas_left < GAS_STIPEND ? ERROR_OUT_OF_GAS : error_code);
    if (error_code != ERROR_SUCCESS) {
        return error_code;
    }
    evm_word_t key;
    error_code |= stack.pop(key);
    evm_word_t value;
    error_code |= stack.pop(value);
    // bn_t storage_address;
    // message.get_storage_address(arith, storage_address);
    error_code |= CuEVM::gas_cost::sstore_cost(gas_used, gas_refund, touch_state, &message.storage_address, key, value);
    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    return (error_code ? error_code : touch_state.set_storage_value(&message.storage_address, key, value));
}
}  // namespace CuEVM::operations