#include <CuEVM/gas_cost.cuh>
#include <CuEVM/operations/memory.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM::operations {
__device__ int32_t MLOAD(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                         CuEVM::evm_memory_t &memory) {
    CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    evm_word_t memory_offset, length;
    error_code |= stack.pop(memory_offset);

    uint32_t memory_offset_u32 = uint256_get_uint32_t(&memory_offset);
    if (uint256_cmp_word(&memory_offset, memory_offset_u32) != 0) {
        return ERR_MEMORY_INVALID_OFFSET;
    }
    // get the memory expansion gas cost
    gas_t memory_expansion_cost;
    error_code |=
        CuEVM::gas_cost::memory_grow_cost(&memory, memory_offset_u32, UINT256_BYTES, memory_expansion_cost, gas_used);

    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t value_word;
        memory.increase_memory_cost(memory_expansion_cost);
        uint8_t *data;
        error_code |= memory.get(memory_offset_u32, UINT256_BYTES, data);
        error_code |= stack.pushx(32, data, UINT256_BYTES);
    }
    return error_code;
}

__device__ int32_t MSTORE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                          CuEVM::evm_memory_t &memory) {
    CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    evm_word_t memory_offset;
    error_code |= stack.pop(memory_offset);
    evm_word_t value;
    error_code |= stack.pop(value);
    uint32_t memory_offset_u32 = uint256_get_uint32_t(&memory_offset);
    if (uint256_cmp_word(&memory_offset, memory_offset_u32) != 0) {
        return ERR_MEMORY_INVALID_OFFSET;
    }
    // get the memory expansion gas cost
    gas_t memory_expansion_cost;
    error_code |=
        CuEVM::gas_cost::memory_grow_cost(&memory, memory_offset_u32, UINT256_BYTES, memory_expansion_cost, gas_used);
    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    if (error_code == ERROR_SUCCESS) {
        memory.increase_memory_cost(memory_expansion_cost);
        uint8_t *data = new uint8_t[UINT256_BYTES];
        uint256_to_bytes(data, &value, UINT256_BYTES);
        error_code |= memory.set(data, memory_offset_u32, UINT256_BYTES);
        delete[] data;
    }
    return error_code;
}

__device__ int32_t MSTORE8(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                           CuEVM::evm_memory_t &memory) {
    CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    // #ifdef __CUDA_ARCH__
    //     printf("MSTORE8 %d\n", threadIdx.x);
    // #endif
    evm_word_t memory_offset;
    error_code |= stack.pop(memory_offset);
    evm_word_t value;
    error_code |= stack.pop(value);
    uint32_t memory_offset_u32 = uint256_get_uint32_t(&memory_offset);
    if (uint256_cmp_word(&memory_offset, memory_offset_u32) != 0) {
        return ERR_MEMORY_INVALID_OFFSET;
    }
    // get the memory expansion gas cost
    gas_t memory_expansion_cost;
    error_code |= CuEVM::gas_cost::memory_grow_cost(&memory, memory_offset_u32, 1, memory_expansion_cost, gas_used);

    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    if (error_code == ERROR_SUCCESS) {
        memory.increase_memory_cost(memory_expansion_cost);
        uint8_t *data = new uint8_t[1];
        data[0] = value.words[0] & 0xFF;
        error_code |= memory.set(data, memory_offset_u32, 1);
        delete[] data;
    }
    return error_code;
}

__device__ int32_t MSIZE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                         const CuEVM::evm_memory_t &memory) {
    CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t size;
        size = memory.size;

        error_code |= stack.push(size);
    }
    return error_code;
}
}  // namespace CuEVM::operations
