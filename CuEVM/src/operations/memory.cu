#include <CuEVM/gas_cost.cuh>
#include <CuEVM/operations/memory.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM::operations {
__device__ int32_t MLOAD(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                         CuEVM::evm_memory_t &memory) {
    int32_t error_code = ERROR_SUCCESS;

    evm_word_t memory_offset, length;
    error_code |= stack.pop(memory_offset);

    uint32_t memory_offset_u32 = uint256_get_uint32_t(&memory_offset);
    if (uint256_cmp_word(&memory_offset, memory_offset_u32) != 0) {
        return ERR_MEMORY_INVALID_OFFSET;
    }
    gas_used += GAS_MEMORY;
    // get the memory expansion gas cost
    gas_t memory_expansion_cost;
    error_code |=
        CuEVM::gas_cost::memory_grow_cost(&memory, memory_offset_u32, UINT256_BYTES, memory_expansion_cost, gas_used);

    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t value_word;
        memory.increase_memory_cost(memory_expansion_cost);

        uint8_t *data = nullptr;
        // if (INSTANCE_GLOBAL_IDX == 1) {
        //     printf("MLOAD thread %d, memory_offset %u, UINT256_BYTES %u\n", INSTANCE_GLOBAL_IDX, memory_offset_u32,
        //            UINT256_BYTES);
        //     memory.print();
        // }
        error_code |= memory.get(memory_offset_u32, UINT256_BYTES, data);
        // if (INSTANCE_GLOBAL_IDX == 1) {
        //     printf("MLOAD thread %d, data\n", INSTANCE_GLOBAL_IDX);
        //     for (uint32_t i = 0; i < UINT256_BYTES; i++) {
        //         printf("%x ", data[i]);
        //     }
        //     printf("\n");
        // }
        error_code |= stack.pushx(32, data, UINT256_BYTES);
    }
    return error_code;
}

__device__ int32_t MSTORE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                          CuEVM::evm_memory_t &memory) {
    evm_word_t memory_offset;
    int32_t error_code = ERROR_SUCCESS;
    error_code |= stack.pop(memory_offset);
    evm_word_t value;
    error_code |= stack.pop(value);
    uint32_t memory_offset_u32 = uint256_get_uint32_t(&memory_offset);
    if (uint256_cmp_word(&memory_offset, memory_offset_u32) != 0) {
        return ERR_MEMORY_INVALID_OFFSET;
    }

    gas_used += GAS_MEMORY;
    // get the memory expansion gas cost
    gas_t memory_expansion_cost = 0;
    error_code |=
        CuEVM::gas_cost::memory_grow_cost(&memory, memory_offset_u32, UINT256_BYTES, memory_expansion_cost, gas_used);
    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    if (error_code == ERROR_SUCCESS) {
        memory.increase_memory_cost(memory_expansion_cost);
        uint8_t data[UINT256_BYTES];
        uint256_to_bytes(data, &value, UINT256_BYTES);

        error_code |= memory.set(data, UINT256_BYTES, memory_offset_u32, UINT256_BYTES);
    }
    return error_code;
}

__device__ int32_t MSTORE8(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                           CuEVM::evm_memory_t &memory) {
    int32_t error_code = ERROR_SUCCESS;
    evm_word_t memory_offset;
    error_code |= stack.pop(memory_offset);
    evm_word_t value;
    error_code |= stack.pop(value);
    uint32_t memory_offset_u32 = uint256_get_uint32_t(&memory_offset);
    if (uint256_cmp_word(&memory_offset, memory_offset_u32) != 0) {
        return ERR_MEMORY_INVALID_OFFSET;
    }
    gas_used += GAS_MEMORY;
    // get the memory expansion gas cost
    gas_t memory_expansion_cost;
    error_code |= CuEVM::gas_cost::memory_grow_cost(&memory, memory_offset_u32, 1, memory_expansion_cost, gas_used);

    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    if (error_code == ERROR_SUCCESS) {
        memory.increase_memory_cost(memory_expansion_cost);
        uint8_t *data = new uint8_t[1];
        data[0] = value.words[0] & 0xFF;
        error_code |= memory.set(data, 1, memory_offset_u32, 1);
        delete[] data;
    }
    return error_code;
}

__device__ int32_t MSIZE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                         const CuEVM::evm_memory_t &memory) {
    gas_used += GAS_BASE;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) error_code |= stack.push_uint32(memory.size);
    return error_code;
}
}  // namespace CuEVM::operations
