
#include <CuEVM/operations/log.cuh>

/**
 * a0s: Logging Operations:
 * - LOGX
 */
namespace CuEVM::operations {
__device__ int32_t LOGX(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                        const CuEVM::evm_call_context_t *call_context, const uint8_t &opcode) {
    int32_t error_code = (call_context->static_env ? ERROR_STATIC_CALL_CONTEXT_SSTORE : ERROR_SUCCESS);

    uint32_t no_topics = opcode & 0x0F;

    evm_word_t memory_offset;
    error_code |= stack.pop(memory_offset);
    evm_word_t length;
    error_code |= stack.pop(length);

    CuEVM::gas_cost::log_record_cost(gas_used, uint256_get_uint32_t(&length));
    CuEVM::gas_cost::log_topics_cost(gas_used, no_topics + 1);
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
        uint32_t memory_offset_ui32 = uint256_get_uint32_t(&memory_offset);
        uint32_t length_ui32 = uint256_get_uint32_t(&length);
        if (uint256_cmp_word(&memory_offset, memory_offset_ui32) != 0 || uint256_cmp_word(&length, length_ui32) != 0) {
            return ERR_MEMORY_INVALID_OFFSET;
        }
        error_code |= CuEVM::gas_cost::memory_grow_cost(call_context->memory_ptr, memory_offset_ui32, length_ui32,
                                                        memory_expansion_cost, gas_used);

        error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);

        if (error_code == ERROR_SUCCESS) {
            call_context->memory_ptr->increase_memory_cost(memory_expansion_cost);
#ifdef ENABLE_LOGS
            CuEVM::byte_array_t record(length_ui32);
            error_code |= call_context->memory_ptr->get(memory_offset_ui32, length_ui32, record.data);
            log_state.push(call_context->contract_address, record, topics[0], topics[1], topics[2], topics[3],
                           no_topics);
#endif
        }
    }
    return error_code;
}
}  // namespace CuEVM::operations