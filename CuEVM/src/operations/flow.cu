#include <CuEVM/gas_cost.cuh>
#include <CuEVM/operations/flow.cuh>
#include <CuEVM/utils/error_codes.cuh>
#include <CuEVM/utils/opcodes.cuh>
namespace CuEVM::operations {
__device__ int32_t JUMP(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, uint32_t &pc, CuEVM::evm_stack_t &stack,
                        const CuEVM::evm_message_call_t &message) {
    CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t destination;
        error_code |= stack.pop(destination);
        uint32_t destination_u32;
        error_code =
            uint256_get_uint32_t(&destination) == ERROR_VALUE_OVERFLOW ? ERROR_INVALID_JUMP_DESTINATION : error_code;
        if ((destination_u32 >= message.byte_code->size) || (message.byte_code->data[destination_u32] != OP_JUMPDEST)) {
            return ERROR_INVALID_JUMP_DESTINATION;
        }

        if (error_code == ERROR_SUCCESS) {
            pc = message.jump_destinations->has(destination_u32) == ERROR_SUCCESS
                     ? destination_u32 - 1
                     : ([&]() -> uint32_t {
                           error_code = ERROR_INVALID_JUMP_DESTINATION;
                           return pc;
                       })();
        }
    }
    return error_code;
}

__device__ int32_t JUMPI(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, uint32_t &pc, CuEVM::evm_stack_t &stack,
                         const CuEVM::evm_message_call_t &message
#ifdef BUILD_LIBRARY
                         ,
                         CuEVM::utils::simplified_trace_data *simplified_trace_data_ptr
#endif
) {
    CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t destination;
        error_code |= stack.pop(destination);
        evm_word_t condition;
        error_code |= stack.pop(condition);
        uint32_t destination_u32;
        error_code =
            uint256_get_uint32_t(&destination) == ERROR_VALUE_OVERFLOW ? ERROR_INVALID_JUMP_DESTINATION : error_code;
        if ((error_code == ERROR_SUCCESS) && (uint256_cmp_word(&condition, 0) != 0)) {
#ifdef BUILD_LIBRARY
            simplified_trace_data_ptr->record_branch(pc, destination_u32, pc + 1);
#endif

            if ((destination_u32 >= message.byte_code->size) ||
                (message.byte_code->data[destination_u32] != OP_JUMPDEST)) {
                return ERROR_INVALID_JUMP_DESTINATION;
            }
            if (error_code == ERROR_SUCCESS) {
                pc = message.jump_destinations->has(destination_u32) == ERROR_SUCCESS
                         ? destination_u32 - 1
                         : ([&]() -> uint32_t {
                               error_code = ERROR_INVALID_JUMP_DESTINATION;
                               return pc;
                           })();
            }
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
    CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    evm_word_t pc_bn;
    uint256_from_word(&pc_bn, pc);
    error_code |= stack.push(pc_bn);
    return error_code;
}

__device__ int32_t GAS(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack) {
    CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    gas_t gas_left;
    gas_left = gas_limit - gas_used;
    error_code |= stack.push(gas_left);
    return error_code;
}

__device__ int32_t JUMPDEST(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used) {
    CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    return error_code;
}
}  // namespace CuEVM::operations