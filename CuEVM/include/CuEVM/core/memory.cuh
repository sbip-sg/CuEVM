#pragma once

#include <CuEVM/core/byte_array.cuh>
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
    uint8_t *data;     /**< The data of the memory*/
    uint32_t size;     /**< The size of the memory acceesed by now (YP: \f$32 \dot \mu_{i}\f$)*/
    gas_t memory_cost; /**< The memory cost (YP: \f$M(\mu_{i})\f$)*/

    /**
     * The default constructor.
     */
    __host__ __device__ evm_memory_t() : data(nullptr), size(0) { memory_cost = 0; }

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
    __host__ __device__ void increase_memory_cost(gas_t memory_expansion_cost);

    /**
     * Increase the memory for the given offset if needed.
     * @param[in] new_size The new size of the memory.
     * @return 0 if success, otherwise the error code.
     */
    __host__ __device__ int32_t grow(uint32_t new_size);

    /**
     * Get the a pointer to the given memory data.
     * @param[in] index The index of the memory access.
     * @param[in] length The length of the memory access.
     * @param[out] data The pointer to the memory data.
     * @return 0 if success, otherwise the error code.
     */
    __host__ __device__ int32_t get(const uint32_t index, const uint32_t length, uint8_t *&data);

    /**
     * Set the given memory data. Outside available_size is 0.
     * @param[in] data The data to be set.
     * @param[in] index The index of the memory access.
     * @param[in] length The length of the memory access.
     * @return 0 if success, otherwise the error code.
     */
    __host__ __device__ int32_t set(uint8_t *data, const uint32_t index, const uint32_t length);

    /**
     * Set the given memory data to zero.
     * @param[in] index The index of the memory access.
     * @param[in] length The length of the memory access.
     * @return 0 if success, otherwise the error code.
     */
    __host__ __device__ int32_t set_zero(const uint32_t index, const uint32_t length);
};

}  // namespace memory
// alias for evm_mmeory_t
using evm_memory_t = memory::evm_memory_t;

}  // namespace CuEVM
