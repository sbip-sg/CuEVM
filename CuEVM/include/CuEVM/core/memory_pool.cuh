#pragma once
#include <CuEVM/core/data_structures.cuh>
#include <CuEVM/core/evm_call_context.cuh>
#include <CuEVM/core/evm_word.cuh>
namespace CuEVM::memory_pool {
struct memory_pool_t {
    evm_call_context_t* call_context;
    evm_word_t* stack_base;
    SnapshotValue* preallocated_snapshot_values;
    uint8_t* return_data_base;
    uint32_t num_instances;
    uint32_t count_words;
    uint32_t count_call_context;
    uint32_t current_stack_page_size;
    __host__ memory_pool_t() {};
};
extern __device__ memory_pool_t* global_memory_pool;
__host__ void create_memory_pool(uint32_t num_instances, uint32_t num_accounts);
__host__ evm_word_t* preallocate_stack(uint32_t num_instances);
__device__ void expand_call_context(uint32_t num_instances);

__device__ evm_call_context_t* get_call_context(uint16_t depth);

__device__ evm_memory_t* get_memory(uint16_t depth);

__device__ void expand_words(uint32_t num_instances);
}  // namespace CuEVM::memory_pool
