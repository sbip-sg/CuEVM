// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#include <CuEVM/gas_cost.cuh>
#include <CuEVM/operations/memory.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM::operations {
__host__ __device__ int32_t MLOAD(ArithEnv &arith, const bn_t &gas_limit, bn_t &gas_used, CuEVM::evm_stack_t &stack,
                                  CuEVM::evm_memory_t &memory) {
    cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_VERY_LOW);
    int32_t error_code = CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);

    bn_t memory_offset, length;
    error_code |= stack.pop(arith, memory_offset);
    cgbn_set_ui32(arith.env, length, CuEVM::word_size);

    // get the memory expansion gas cost
    bn_t memory_expansion_cost;
    error_code |=
        CuEVM::gas_cost::memory_grow_cost(arith, memory, memory_offset, length, memory_expansion_cost, gas_used);

    error_code |= CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        __SHARED_MEMORY__ evm_word_t value_word[CGBN_IBP];
        memory.increase_memory_cost(arith, memory_expansion_cost);
        CuEVM::byte_array_t data;
        error_code |= memory.get(arith, memory_offset, length, data);
        // printf("MLOAD ERROR_SUCCESS %d\n", THREADIDX);
        // bn_t value;
        value_word[INSTANCE_IDX_PER_BLOCK].from_byte_array_t(data, BIG_ENDIAN);
        // __ONE_GPU_THREAD_WOSYNC_BEGIN__
        // if (INSTANCE_IDX_PER_BLOCK == 1) {
        //     printf("MLOAD before push %d\n", THREADIDX);
        //     value_word[INSTANCE_IDX_PER_BLOCK].print();
        //     data.print();
        // }
        // __ONE_GPU_THREAD_WOSYNC_END__
        // error_code |= cgbn_set_byte_array_t(arith.env, value, data);
        // printf("MLOAD before push %d\n", THREADIDX);
        error_code |= stack.push_evm_word_t(arith, &value_word[INSTANCE_IDX_PER_BLOCK]);
        // error_code |= stack.push(arith, value);
        // printf("MLOAD push %d\n", THREADIDX);
    }
    return error_code;
}

__host__ __device__ int32_t MSTORE(ArithEnv &arith, const bn_t &gas_limit, bn_t &gas_used, CuEVM::evm_stack_t &stack,
                                   CuEVM::evm_memory_t &memory) {
    cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_VERY_LOW);
    int32_t error_code = CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);

    bn_t memory_offset;
    error_code |= stack.pop(arith, memory_offset);
    bn_t value;
    error_code |= stack.pop(arith, value);
    bn_t length;
    cgbn_set_ui32(arith.env, length, CuEVM::word_size);

    // get the memory expansion gas cost
    bn_t memory_expansion_cost;
    error_code |=
        CuEVM::gas_cost::memory_grow_cost(arith, memory, memory_offset, length, memory_expansion_cost, gas_used);

    error_code |= CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);

    if (error_code == ERROR_SUCCESS) {
        memory.increase_memory_cost(arith, memory_expansion_cost);
        CuEVM::byte_array_t value_bytes(CuEVM::word_size);
        // printf("MSTORE %d %d instanceid %d\n", THREADIDX, THREAD_IDX_PER_INSTANCE, INSTANCE_IDX_PER_BLOCK);
        // printf("byte array after construction size %d data %p threadidx %d\n", value_bytes.size, value_bytes.data,
        //        THREADIDX);
        __SHARED_MEMORY__ evm_word_t value_word[CGBN_IBP];
        cgbn_store(arith.env, (cgbn_evm_word_t_ptr)&value_word[INSTANCE_IDX_PER_BLOCK], value);

        value_word[INSTANCE_IDX_PER_BLOCK].to_byte_array_t(value_bytes, BIG_ENDIAN);

        error_code |= memory.set(arith, value_bytes, memory_offset, length);
    }
    return error_code;
}

__host__ __device__ int32_t MSTORE8(ArithEnv &arith, const bn_t &gas_limit, bn_t &gas_used, CuEVM::evm_stack_t &stack,
                                    CuEVM::evm_memory_t &memory) {
    cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_VERY_LOW);
    int32_t error_code = CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);
    // #ifdef __CUDA_ARCH__
    //     printf("MSTORE8 %d\n", threadIdx.x);
    // #endif
    bn_t memory_offset;
    error_code |= stack.pop(arith, memory_offset);
    bn_t value;
    error_code |= stack.pop(arith, value);
    bn_t length;
    cgbn_set_ui32(arith.env, length, 1);
    // get the memory expansion gas cost
    bn_t memory_expansion_cost;
    error_code |=
        CuEVM::gas_cost::memory_grow_cost(arith, memory, memory_offset, length, memory_expansion_cost, gas_used);

    error_code |= CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);

    if (error_code == ERROR_SUCCESS) {
        memory.increase_memory_cost(arith, memory_expansion_cost);
        CuEVM::byte_array_t value_bytes(CuEVM::word_size);

        // TODO: bnt directly to byte array
        __SHARED_MEMORY__ evm_word_t value_word[CGBN_IBP];

        cgbn_store(arith.env, (cgbn_evm_word_t_ptr)&value_word[INSTANCE_IDX_PER_BLOCK], value);
        value_word[INSTANCE_IDX_PER_BLOCK].to_byte_array_t(value_bytes, BIG_ENDIAN);

        CuEVM::byte_array_t value_byte(value_bytes.data + CuEVM::word_size - 1, 1);

        error_code |= memory.set(arith, value_byte, memory_offset, length);
    }
    return error_code;
}

__host__ __device__ int32_t MSIZE(ArithEnv &arith, const bn_t &gas_limit, bn_t &gas_used, CuEVM::evm_stack_t &stack,
                                  const CuEVM::evm_memory_t &memory) {
    cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_BASE);
    int32_t error_code = CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        bn_t size;
        cgbn_set_ui32(arith.env, size, memory.get_size());

        error_code |= stack.push(arith, size);
    }
    return error_code;
}
}  // namespace CuEVM::operations
