#include <CuEVM/gas_cost.cuh>
#include <CuEVM/operations/block.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM::operations {
__device__ int32_t BLOCKHASH(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                             const CuEVM::block_info_t &block) {
    gas_used += GAS_BLOCKHASH;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t number;
        error_code |= stack.pop(number);
        evm_word_t hash;
        // even if error of invalid number/index, the hash is set to zero
        uint32_t tmp_error_code;
        tmp_error_code = block.get_previous_hash(hash, number);
        if (tmp_error_code != ERROR_SUCCESS) {
            uint256_set_zero(&hash);
        }

        error_code |= stack.push(hash);
    }
    return error_code;
}

__device__ int32_t COINBASE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                            const CuEVM::block_info_t &block) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t coin_base;
        block.get_coin_base(coin_base);

        error_code |= stack.push(coin_base);
    }
    return error_code;
}

__device__ int32_t TIMESTAMP(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                             const CuEVM::block_info_t &block) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t time_stamp;
        block.get_time_stamp(time_stamp);

        error_code |= stack.push(time_stamp);
    }
    return error_code;
}

__device__ int32_t NUMBER(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                          const CuEVM::block_info_t &block) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t number;
        block.get_number(number);

        error_code |= stack.push(number);
    }
    return error_code;
}

__device__ int32_t PREVRANDAO(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                              const CuEVM::block_info_t &block) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t prev_randao;
        // TODO: to change depending on the evm version
        block.get_prevrandao(prev_randao);  // Assuming after merge fork
        // block.get_difficulty(prev_randao);

        error_code |= stack.push(prev_randao);
    }
    return error_code;
}

__device__ int32_t GASLIMIT(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                            const CuEVM::block_info_t &block) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t gas_limit;
        block.get_gas_limit(gas_limit);

        error_code |= stack.push(gas_limit);
    }
    return error_code;
}

__device__ int32_t CHAINID(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                           const CuEVM::block_info_t &block) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t chain_id;
        block.get_chain_id(chain_id);

        error_code |= stack.push(chain_id);
    }
    return error_code;
}

__device__ int32_t BASEFEE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                           const CuEVM::block_info_t &block) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t base_fee;
        block.get_base_fee(base_fee);

        error_code |= stack.push(base_fee);
    }
    return error_code;
}
}  // namespace CuEVM::operations