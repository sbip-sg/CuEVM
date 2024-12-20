#pragma once
#include <CuEVM/core/evm_call_context.cuh>
#include <CuEVM/core/evm_word.cuh>
#define memory_pool_word_preallocate 16         // times num_instances
#define memory_pool_call_context_preallocate 2  // times num_instances

namespace CuEVM::memory_pool {
struct memory_pool_t {
    evm_call_context_t* call_context;
    evm_word_t* words;
    uint32_t num_instances;
    uint32_t count_words;
    uint32_t count_call_context;
    __host__ memory_pool_t() {};
};
extern __device__ memory_pool_t* global_memory_pool;
__host__ void create_memory_pool(uint32_t num_instances);
__device__ void expand_call_context(uint32_t num_instances);

__device__ evm_call_context_t* get_call_context(uint16_t depth);

__device__ evm_stack_t* get_stack(uint16_t depth);
__device__ evm_word_t* get_stack_base(uint16_t depth = 0);
__device__ evm_memory_t* get_memory(uint16_t depth);

__device__ void expand_words(uint32_t num_instances);
}  // namespace CuEVM::memory_pool
