#include <CuEVM/gas_cost.cuh>
#include <CuEVM/operations/block.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM::operations {
__device__ int32_t BLOCKHASH(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack) {
    gas_used += GAS_BLOCKHASH;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t number;
        error_code |= stack.pop(number);
        evm_word_t hash;
        // even if error of invalid number/index, the hash is set to zero
        uint32_t tmp_error_code;
        tmp_error_code = global_block_info->get_previous_hash(hash, number);
        if (tmp_error_code != ERROR_SUCCESS) {
            uint256_set_zero(&hash);
        }

        error_code |= stack.push(hash);
    }
    return error_code;
}

__device__ int32_t COINBASE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        error_code |= stack.push(global_block_info->coin_base);
    }
    return error_code;
}

__device__ int32_t TIMESTAMP(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        error_code |= stack.push(global_block_info->time_stamp);
    }
    return error_code;
}

__device__ int32_t NUMBER(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        error_code |= stack.push(global_block_info->number);
    }
    return error_code;
}

__device__ int32_t PREVRANDAO(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        error_code |= stack.push(global_block_info->prevrandao);
    }
    return error_code;
}

__device__ int32_t GASLIMIT(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        error_code |= stack.push(global_block_info->gas_limit);
    }
    return error_code;
}

__device__ int32_t CHAINID(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        error_code |= stack.push(global_block_info->chain_id);
    }
    return error_code;
}

__device__ int32_t BASEFEE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        error_code |= stack.push(global_block_info->base_fee);
    }
    return error_code;
}
}  // namespace CuEVM::operations