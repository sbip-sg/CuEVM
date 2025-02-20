#pragma once
#include <CuEVM/core/data_structures.cuh>
#include <CuEVM/core/evm_call_context.cuh>
#include <CuEVM/core/evm_word.cuh>

namespace CuEVM::memory_pool {

__host__ void create_memory_pool(uint32_t num_instances, uint32_t num_accounts);

__device__ evm_call_context_t* get_call_context(uint16_t depth);
__device__ evm_stack_t* get_stack(uint16_t depth);
__device__ evm_memory_t* get_memory(uint16_t depth);
__device__ CuEVM::SnapshotState* get_snapshot_state();

__device__ uint32_t get_next_snapshot_offset();
__device__ void reset_snapshot_slot_offset(uint32_t offset);
__device__ void reset_snapshot_account_offset(uint32_t offset);

__host__ void free_memory_pool();
__host__ void clear_memory_pool();

}  // namespace CuEVM::memory_pool
