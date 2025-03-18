
#include <CuEVM/operations/flow.cuh>

namespace CuEVM::operations {
__device__ int32_t JUMP(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, uint32_t &pc, CuEVM::evm_stack_t &stack,
                        evm_call_context_t *call_context) {
    gas_used += GAS_MID;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    if (error_code == ERROR_SUCCESS) {
        if (stack.size() < 1) return ERROR_STACK_UNDERFLOW;
        evm_word_t *destination = stack.get_address_at_index(1);
        stack.reduce_size(1);
        uint32_t destination_u32 = uint256_get_uint32_t(destination);

        if (uint256_cmp_word(destination, destination_u32)) return ERROR_INVALID_JUMP_DESTINATION;

        int32_t  address_index = CuEVM::global_state_db_ptr->get_address_index(&call_context->to);

        if (address_index < 0) {
            // Dynamically created address, not analyzed at the moment
            if ((destination_u32 >= call_context->byte_code_size) ||
                (call_context->byte_code[destination_u32] != OP_JUMPDEST)) {
                return ERROR_INVALID_JUMP_DESTINATION;
            }
        }else {
            auto err = CuEVM::global_state_db_ptr->global_jump_table->validate_jumpdest(address_index, destination_u32);
            if (err) {
                return err;
            }
        }
        pc = destination_u32 - 1;
    }
    return error_code;
}

__device__ int32_t JUMPI(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, uint32_t &pc, CuEVM::evm_stack_t &stack,
                         evm_call_context_t *call_context
#ifdef BUILD_LIBRARY
                         ,
                         simplified_trace_data *simplified_trace_data_ptr
#endif
) {
    gas_used += GAS_HIGH;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        if (stack.size() < 2) return ERROR_STACK_UNDERFLOW;
        evm_word_t *destination = stack.get_address_at_index(1);
        evm_word_t *condition = stack.get_address_at_index(2);
        stack.reduce_size(2);
        uint32_t destination_u32 = uint256_get_uint32_t(destination);
        if (uint256_cmp_word(destination, destination_u32)) return ERROR_INVALID_JUMP_DESTINATION;

        if ((error_code == ERROR_SUCCESS) && (uint256_cmp_word(condition, 0) != 0)) {
#ifdef BUILD_LIBRARY
            simplified_trace_data_ptr->record_branch(pc, destination_u32, pc + 1);
#endif

            int32_t  address_index = CuEVM::global_state_db_ptr->get_address_index(&call_context->to);
            if (address_index < 0) {
                // Dynamically created address, not analyzed at the moment
                if ((destination_u32 >= call_context->byte_code_size) ||
                    (call_context->byte_code[destination_u32] != OP_JUMPDEST)) {
                    return ERROR_INVALID_JUMP_DESTINATION;
                }
            }else {
                auto err = CuEVM::global_state_db_ptr->global_jump_table->validate_jumpdest(address_index, destination_u32);
                if (err) {
                    return err;
                }
            }
            pc = destination_u32 - 1;
        }
#ifdef BUILD_LIBRARY
        else {
            simplified_trace_data_ptr->record_branch(pc, pc + 1, destination_u32);
        }
#endif
    }
    return error_code;
}

__device__ int32_t PC(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, const uint32_t &pc,
                      CuEVM::evm_stack_t &stack) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    evm_word_t pc_bn;
    uint256_from_word(&pc_bn, pc);
    error_code |= stack.push(pc_bn);
    return error_code;
}

__device__ int32_t JUMPDEST(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used) {
    gas_used += GAS_JUMP_DEST;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    return error_code;
}
}  // namespace CuEVM::operations
