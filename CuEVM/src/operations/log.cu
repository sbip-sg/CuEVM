
#include <CuEVM/operations/log.cuh>

/**
 * a0s: Logging Operations:
 * - LOGX
 */
namespace CuEVM::operations {
__device__ int32_t LOGX(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                        CuEVM::evm_memory_t &memory, const CuEVM::evm_call_context_t *call_context,
                        CuEVM::log_state_data_t &log_state, const uint8_t &opcode) {
    int32_t error_code = (call_context->static_env ? ERROR_STATIC_CALL_CONTEXT_SSTORE : ERROR_SUCCESS);

    uint32_t no_topics = opcode & 0x0F;

    CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    evm_word_t memory_offset;
    error_code |= stack.pop(memory_offset);
    evm_word_t length;
    error_code |= stack.pop(length);

    CuEVM::gas_cost::log_record_cost(gas_used, uint256_get_uint32_t(&length));

    CuEVM::gas_cost::log_topics_cost(gas_used, no_topics);

    evm_word_t topics[4];
    for (uint32_t idx = 0; idx < no_topics; idx++) {
        error_code |= stack.pop(topics[idx]);
    }
    for (uint32_t idx = no_topics; idx < 4; idx++) {
        topics[idx].set_zero();
    }

    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    if (error_code == ERROR_SUCCESS) {
        gas_t memory_expansion_cost;
        // Get the memory expansion gas cost
        error_code |= CuEVM::gas_cost::memory_grow_cost(memory, memory_offset, length, memory_expansion_cost, gas_used);

        error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);

        if (error_code == ERROR_SUCCESS) {
            memory.increase_memory_cost(memory_expansion_cost);
            CuEVM::byte_array_t record;
            error_code |= memory.get(memory_offset, length, record);

#ifdef ENABLE_LOGS
            log_state.push(call_context->contract_address, record, topics[0], topics[1], topics[2], topics[3],
                           no_topics);
#endif
        }
    }
    return error_code;
}
}  // namespace CuEVM::operations