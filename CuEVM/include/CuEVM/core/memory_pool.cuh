#pragma once
#include <CuEVM/core/data_structures.cuh>
#include <CuEVM/core/evm_call_context.cuh>
#include <CuEVM/core/evm_word.cuh>

namespace CuEVM::memory_pool {

/**
 * @brief Create a memory pool for EVM execution.
 *
 * Allocates memory resources needed for EVM execution including stacks,
 * call contexts, memory instances, and snapshot states.
 *
 * @param[in] num_instances Number of EVM instances to allocate resources for
 * @param[in] num_accounts Number of accounts to support
 * @param[in] num_devices Number of GPU devices to allocate on (default: 1)
 */
__host__ void create_memory_pool(uint32_t num_instances, uint32_t num_accounts, uint32_t num_devices = 1);

/**
 * @brief Get a call context for the specified call depth.
 *
 * Returns a call context from the preallocated pool or dynamically allocates
 * a new one if the depth exceeds the preallocated limit.
 *
 * @param[in] depth Call depth for the requested context
 * @return Pointer to the call context
 */
__device__ evm_call_context_t* get_call_context(uint16_t depth);

/**
 * @brief Get a stack for the specified call depth.
 *
 * Returns a stack from the preallocated pool or dynamically allocates
 * a new one if the capacity exceeds the preallocated limit.
 *
 * @param[in] depth Call depth for the requested stack
 * @return Pointer to the stack
 */
__device__ evm_stack_t* get_stack(uint16_t depth);

/**
 * @brief Get a memory instance for the specified call depth.
 *
 * Returns a memory instance from the preallocated pool or dynamically allocates
 * a new one if the capacity exceeds the preallocated limit.
 *
 * @param[in] depth Call depth for the requested memory
 * @return Pointer to the memory instance
 */
__device__ evm_memory_t* get_memory(uint16_t depth);

/**
 * @brief Get a snapshot state from the pool.
 *
 * Returns a snapshot state from the preallocated pool or dynamically allocates
 * a new one if the pool is exhausted.
 *
 * @return Pointer to the snapshot state
 */
__device__ CuEVM::SnapshotState* get_snapshot_state();

/**
 * @brief Get the next snapshot slot offset.
 *
 * Retrieves and increments the snapshot slot counter for the current instance.
 *
 * @return The next available snapshot slot offset
 */
__device__ uint32_t get_next_snapshot_offset();

/**
 * @brief Reset the snapshot slot offset to a specified value.
 *
 * @param[in] offset The offset value to reset to
 */
__device__ void reset_snapshot_slot_offset(uint32_t offset);

/**
 * @brief Reset the snapshot account offset to a specified value.
 *
 * @param[in] offset The offset value to reset to
 */
__device__ void reset_snapshot_account_offset(uint32_t offset);

/**
 * @brief Free the memory pool resources.
 *
 * Deallocates all resources allocated by create_memory_pool.
 *
 * @param[in] num_devices Number of GPU devices to free resources from (default: 1)
 */
__host__ void free_memory_pool(uint32_t num_devices = 1);

/**
 * @brief Clear the memory pool contents.
 *
 * Resets all memory in the pool to zero without deallocating.
 *
 * @param[in] num_devices Number of GPU devices to clear (default: 1)
 */
__host__ void clear_memory_pool(uint32_t num_devices = 1);

}  // namespace CuEVM::memory_pool
