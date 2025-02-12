#include <CuEVM/gas_cost.cuh>
#include <CuEVM/operations/stack.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM::operations {
__device__ int32_t POP(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t y;

        error_code |= stack.pop(y);
    }
    return error_code;
}

__device__ int32_t PUSH0(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t r;
        r.set_zero();

        error_code |= stack.push(r);
    }
    return error_code;
}

__device__ int32_t PUSHX(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, uint32_t &pc, CuEVM::evm_stack_t *stack,
                         uint8_t *byte_code, uint32_t byte_code_size, const uint8_t &opcode) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    if (error_code == ERROR_SUCCESS) {
        uint8_t push_size = (opcode & 0x1F) + 1;
        uint8_t *byte_data = &(byte_code[pc + 1]);
        uint32_t available_size = (pc + push_size >= byte_code_size) ? byte_code_size - pc - 1 : push_size;

        error_code |= stack->pushx(push_size, byte_data, available_size);
        pc = pc + push_size;
    }
    return error_code;
}

__device__ int32_t DUPX(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                        const uint8_t &opcode) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        uint8_t dup_index = (opcode & 0x0F) + 1;

        error_code |= stack.dupx(dup_index);
    }
    return error_code;
}

__device__ int32_t SWAPX(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                         const uint8_t &opcode) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        uint8_t swap_index = (opcode & 0x0F) + 1;

        error_code |= stack.swapx(swap_index);
    }
    return error_code;
}
}  // namespace CuEVM::operations