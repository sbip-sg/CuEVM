#include <CuEVM/gas_cost.cuh>
#include <CuEVM/operations/memory.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM::operations {
__host__ __device__ int32_t MLOAD(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                                  CuEVM::evm_memory_t &memory) {
    CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    evm_word_t memory_offset, length;
    error_code |= stack.pop(memory_offset);
    length = CuEVM::word_size;

    // get the memory expansion gas cost
    gas_t memory_expansion_cost;
    error_code |= CuEVM::gas_cost::memory_grow_cost(memory, memory_offset, length, memory_expansion_cost, gas_used);

    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t value_word;
        memory.increase_memory_cost(memory_expansion_cost);
        CuEVM::byte_array_t data;
        error_code |= memory.get(memory_offset, length, data);
        // printf("MLOAD ERROR_SUCCESS %d\n", THREADIDX);
        // bn_t value;
        value_word.from_byte_array_t(data, BIG_ENDIAN);
        // __ONE_GPU_THREAD_WOSYNC_BEGIN__
        // if (INSTANCE_IDX_PER_BLOCK == 1) {
        //     printf("MLOAD before push %d\n", THREADIDX);
        //     value_word[INSTANCE_IDX_PER_BLOCK].print();
        //     data.print();
        // }
        // __ONE_GPU_THREAD_WOSYNC_END__
        // error_code |= cgbn_set_byte_array_t(arith.env, value, data);
        // printf("MLOAD before push %d\n", THREADIDX);
        error_code |= stack.push(value_word);
        // error_code |= stack.push(arith, value);
        // printf("MLOAD push %d\n", THREADIDX);
    }
    return error_code;
}

__host__ __device__ int32_t MSTORE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                                   CuEVM::evm_memory_t &memory) {
    CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    evm_word_t memory_offset;
    error_code |= stack.pop(memory_offset);
    evm_word_t value;
    error_code |= stack.pop(value);
    evm_word_t length;
    length = CuEVM::word_size;

    // get the memory expansion gas cost
    gas_t memory_expansion_cost;
    error_code |= CuEVM::gas_cost::memory_grow_cost(memory, memory_offset, length, memory_expansion_cost, gas_used);

    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    if (error_code == ERROR_SUCCESS) {
        memory.increase_memory_cost(memory_expansion_cost);
        CuEVM::byte_array_t value_bytes(CuEVM::word_size);
        // printf("MSTORE %d %d instanceid %d\n", THREADIDX, THREAD_IDX_PER_INSTANCE, INSTANCE_IDX_PER_BLOCK);
        // printf("byte array after construction size %d data %p threadidx %d\n", value_bytes.size, value_bytes.data,
        //        THREADIDX);
        evm_word_t value_word;
        value_word.from_byte_array_t(value_bytes, BIG_ENDIAN);

        error_code |= memory.set(value_bytes, memory_offset, length);
    }
    return error_code;
}

__host__ __device__ int32_t MSTORE8(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
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
    evm_word_t length;
    length = 1;
    // get the memory expansion gas cost
    gas_t memory_expansion_cost;
    error_code |= CuEVM::gas_cost::memory_grow_cost(memory, memory_offset, length, memory_expansion_cost, gas_used);

    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    if (error_code == ERROR_SUCCESS) {
        memory.increase_memory_cost(memory_expansion_cost);
        CuEVM::byte_array_t value_bytes(CuEVM::word_size);

        // TODO: bnt directly to byte array
        evm_word_t value_word;
        value_word.from_byte_array_t(value_bytes, BIG_ENDIAN);

        CuEVM::byte_array_t value_byte(value_bytes.data + CuEVM::word_size - 1, 1);

        error_code |= memory.set(value_byte, memory_offset, length);
    }
    return error_code;
}

__host__ __device__ int32_t MSIZE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                                  const CuEVM::evm_memory_t &memory) {
    CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t size;
        size = memory.get_size();

        error_code |= stack.push(size);
    }
    return error_code;
}
}  // namespace CuEVM::operations
