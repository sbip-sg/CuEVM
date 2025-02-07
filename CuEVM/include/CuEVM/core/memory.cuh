#pragma once

#include <CuEVM/core/byte_array.cuh>
#include <CuEVM/core/data_structures.cuh>
#include <CuEVM/core/evm_word.cuh>
#include <CuEVM/utils/evm_defines.cuh>
namespace CuEVM {
namespace memory {
// to change for making more optimal memory allocation current 1KB
// constexpr CONSTANT uint32_t page_size = 1024U;

/**
 * The memory data structure.
 */
struct evm_memory_t {
    uint32_t preallocated_base_offset;  // the start offset of the preallocated memory in the memory pool
    uint32_t size;                      /**< The size of the memory acceesed by now (YP: \f$32 \dot \mu_{i}\f$)*/
    gas_t memory_cost;                  /**< The memory cost (YP: \f$M(\mu_{i})\f$)*/
    uint8_t *dynamic_data = nullptr;    // pointer to the dynamic memory

    /**
     * The default constructor.
     */
    __host__ __device__ evm_memory_t() : preallocated_base_offset(0), size(0) { memory_cost = 0; }

    __device__ void init(uint32_t preallocated_base_offset) {
        // printf("init memory preallocated_base_offset %d\n", preallocated_base_offset);
        // printf("memory pointer %p\n", this);
        this->preallocated_base_offset = min(preallocated_base_offset, memory_prealloc_size);
        this->size = 0;
        this->memory_cost = 0;
        this->dynamic_data = nullptr;
    }

    /**
     * the destructor
     */
    __host__ __device__ ~evm_memory_t() {
        memory_cost = 0;
        size = 0;
    }

    /**
     * Print the memory data structure.
     */
    __host__ __device__ void print() const;

    /**
     * Get the json object from the memory data structure.
     * @return The json object.
     */
    __host__ cJSON *to_json() const;

    /**
     * Increase the memory cost.
     * @param[in] memory_expansion_cost The memory expansion cost.
     */
    __device__ void increase_memory_cost(gas_t memory_expansion_cost);

    /**
     * Increase the memory for the given offset if needed.
     * @param[in] new_size The new size of the memory.
     * @return 0 if success, otherwise the error code.
     */
    __device__ int32_t grow(uint32_t new_size);

    /**
     * Get the a pointer to the given memory data.
     * @param[in] index The index of the memory access.
     * @param[in] length The length of the memory access.
     * @param[out] data The pointer to the memory data.
     * @return 0 if success, otherwise the error code.
     */
    __device__ int32_t get(const uint32_t index, const uint32_t length, uint8_t *&data_);

    /**
     * Copy the given memory data.
     * @param[in] index The index of the memory access.
     * @param[in] length The length of the memory access.
     * @param[out] data The pointer to the memory data.
     * @return 0 if success, otherwise the error code.
     */
    __device__ int32_t copy(const uint32_t index, const uint32_t length, uint8_t *&data_);

    /**
     * Set the given memory data. Outside available_size is 0.
     * @param[in] data The data to be set.
     * @param[in] index The index of the memory access.
     * @param[in] length The length of the memory access.
     * @return 0 if success, otherwise the error code.
     */
    __device__ int32_t set(uint8_t *data_, uint32_t data_size, const uint32_t index, const uint32_t length);

    __device__ int32_t set_buffer_data(uint8_t *data_, uint64_t data_offset, uint32_t data_size, const uint32_t index,
                                       const uint32_t length);

    /**
     * Set the given memory data to zero.
     * @param[in] index The index of the memory access.
     * @param[in] length The length of the memory access.
     * @return 0 if success, otherwise the error code.
     */
    __device__ int32_t set_zero(const uint32_t index, const uint32_t length);
};

}  // namespace memory
// alias for evm_mmeory_t
using evm_memory_t = memory::evm_memory_t;

}  // namespace CuEVM
