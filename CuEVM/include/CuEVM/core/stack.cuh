#pragma once
#include <CuEVM/core/evm_word.cuh>
#include <CuEVM/utils/cuda_utils.cuh>
#include <CuEVM/utils/evm_defines.cuh>
namespace CuEVM {
namespace stack {
constexpr CONSTANT uint32_t max_size = CuEVM::max_stack_size; /**< The maximum stack size*/

struct evm_stack_t {
    evm_word_t *global_stack_base; /**< The stack YP: (YP: \f$\mu_{s}\f$)*/  // global memory store from X+1 element
    evm_word_t *shared_stack_base;  // shared memory for X elements from the top => becomes preallocated stackbase
    uint16_t stack_offset;          // offset of the current stack (its size) from it's base offset in shared memory
    uint32_t stack_base_offset;     // offset of the stack base in shared memory or global memory
    /**
     * The default constructor
     * Stack base offset of the child stack = parent stack offset  + 1
     */
    __host__ __device__ evm_stack_t::evm_stack_t(evm_word_t *shared_stack_base, uint32_t stack_base_offset = 0)
        : global_stack_base(nullptr),
          shared_stack_base(shared_stack_base),
          stack_base_offset(stack_base_offset),
          stack_offset(0) {
        // printf("stack constructor shared stack base %p, stack base offset %d, stack offset %d\n", shared_stack_base,
        //        stack_base_offset, stack_offset);
    }
    /**
     * The destructor
     */
    __host__ __device__ ~evm_stack_t();

    /**
     * The copy constructor
     * @param[in] other The other stack
     */
    // __host__ __device__ evm_stack_t(const evm_stack_t &other);

    /**
     * Free the memory
     */
    __host__ __device__ void free();

    /**
     * Clear the content
     */
    __host__ __device__ void clear();

    /**
     * Extract the stack data for tracing
     * @param[in] other The other stack
     */
    __host__ __device__ void extract_data(evm_word_t *other) const;

    /**
     * Get the size of the stack
     * @return The size of the stack
     */
    __host__ __device__ uint32_t size() const;

    /**
     * Reduce the size (remove items) of the stack without memory operations
     * @param[in] num_items The number of items to remove
     */
    __device__ void reduce_size(uint32_t num_items);

    /**
     * Get the top of the stack
     * @return The top of the stack pointer
     */
    __host__ __device__ evm_word_t *top();

    /**
     * Push a value to the stack
     * @param[in] arith The arithmetical environment
     * @param[in] value The value to be pushed
     * @return 0 if the value is pushed, error code otherwise
     */
    __host__ __device__ int32_t push(const evm_word_t &value);
    __host__ __device__ int32_t push_uint32(uint32_t value);
    __host__ __device__ int32_t push_evm_word_t(const evm_word_t *value);
    /**
     * Pop a value from the stack
     * @param[in] arith The arithmetical environment
     * @param[out] y The value popped from the stack
     * @return 0 if the value is popped, error code otherwise
     */
    __host__ __device__ int32_t pop(evm_word_t &y);
    __host__ __device__ int32_t pop_evm_word(evm_word_t *&y);
    /**
     * Push a value to the stack from a byte array
     * @param[in] arith The arithmetical environment
     * @param[in] x the number of bytes of the value
     * @param[in] src_byte_data The source byte data
     * @param[in] src_byte_size The size of the source byte data
     * @return 0 if the value is pushed, error code otherwise
     */
    __host__ __device__ int32_t pushx(uint8_t x, uint8_t *src_byte_data, uint8_t src_byte_size);

    /**
     * Get the value from the stack at the given index
     * @param[in] arith The arithmetical environment
     * @param[in] index The index of the value
     * @param[out] y The value at the given index
     * @return 0 if the value is popped, error code otherwise
     */
    __host__ __device__ int32_t get_index(uint32_t index, evm_word_t &y);

    __host__ __device__ evm_word_t *get_address_at_index(uint32_t index) const;
    /**
     * Duplicvate the value at the given index and push
     * it at the top of the stack.
     * @param[in] arith The arithmetical environment
     * @param[in] x The index of the value
     * @return 0 if the value is duplicated, error code otherwise
     */
    __host__ __device__ int32_t dupx(uint32_t x);

    /**
     * Swap the values at the given index with the top of the stack
     * @param[in] arith The arithmetical environment
     * @param[in] x The index of the value
     * @return 0 if the value is swapped, error code otherwise
     */
    __host__ __device__ int32_t swapx(uint32_t x);

    /**
     * Print the stack
     */
    __host__ __device__ void print();

    /**
     * Get the JSON object from the stack
     * @return The JSON object
     */
    __host__ cJSON *to_json();

    /**
     * Generate the stack gpu instances from the stack cpu instances
     * @param[in] cpu_instances The stack cpu instances
     * @param[in] count The number of instances
     * @return The stack gpu instances
     */
    __host__ static evm_stack_t *gpu_from_cpu(evm_stack_t *cpu_instances, uint32_t count);

    /**
     * Free the stack gpu instances
     * @param[in] gpu_instances The stack gpu instances
     * @param[in] count The number of instances
     */
    __host__ static void gpu_free(evm_stack_t *gpu_instances, uint32_t count);
};

}  // namespace stack
   // Type alias for accessing evm_stack_t directly under the CuEVM namespace
using evm_stack_t = stack::evm_stack_t;
}  // namespace CuEVM
