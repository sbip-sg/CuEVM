#include <cuda_runtime.h>

#include <CuEVM/core/byte_array.cuh>
#include <CuEVM/core/data_structures.cuh>
#include <CuEVM/core/evm_word.cuh>
#include <CuEVM/utils/error_codes.cuh>
#include <CuEVM/utils/evm_defines.cuh>
namespace CuEVM::memory {

// experimental, not used
__device__ void warp_cooperative_set(uint8_t *ptr1, const uint8_t *ptr2, uint32_t length) {
    // Get the lane ID within the warp (0 to 31)
    uint32_t lane_id = threadIdx.x % 32;
    unsigned active_mask = __activemask();              // Bitmask of active threads
    uint32_t num_active_threads = __popc(active_mask);  // Count active threads
    
    // printf("warp_cooperative_set thread %d, num_active_threads %u, length %u ptr1 %p ptr2 %p\n", INSTANCE_GLOBAL_IDX, num_active_threads, length, ptr1, ptr2);
    
    // TODO : interleaving inactive threads
    // Iterate over each thread in the warp
#pragma unroll
    for (int i = 0; i < num_active_threads; i++) {
        // Cast pointers to unsigned long long for shuffling
        unsigned long long ptr1_int = reinterpret_cast<unsigned long long>(ptr1);
        unsigned long long ptr2_int = reinterpret_cast<unsigned long long>(ptr2);

        // Broadcast values using __shfl_sync
        unsigned long long current_ptr1_int = __shfl_sync(active_mask, ptr1_int, i);
        unsigned long long current_ptr2_int = __shfl_sync(active_mask, ptr2_int, i);
        uint32_t current_length = __shfl_sync(active_mask, length, i);

        // Cast back to pointers
        uint8_t *current_ptr1 = reinterpret_cast<uint8_t *>(current_ptr1_int);
        const uint8_t *current_ptr2 = reinterpret_cast<const uint8_t *>(current_ptr2_int);

        // Only proceed if there's data to process
        if (current_length > 0) {
            // Each thread handles a portion of the memory operation
            for (uint32_t offset = lane_id; offset < current_length; offset += num_active_threads) {
                if (offset < current_length) {
                    // Set: Copy from ptr2 (source) to ptr1 (destination)
                    current_ptr1[offset] = current_ptr2[offset];
                }
            }
        }
    }
}

__device__ void warp_cooperative_setzero(uint8_t *ptr1, uint32_t length) {
    // Get the lane ID within the warp (0 to 31)
    uint32_t lane_id = threadIdx.x % 32;
    unsigned active_mask = __activemask();              // Bitmask of active threads
    uint32_t num_active_threads = __popc(active_mask);  // Count active threads

    // TODO : interleaving inactive threads
    // Iterate over each thread in the warp
#pragma unroll
    for (int i = 0; i < num_active_threads; i++) {
        // Cast pointer to unsigned long long for shuffling
        unsigned long long ptr1_int = reinterpret_cast<unsigned long long>(ptr1);

        // Broadcast values using __shfl_sync
        unsigned long long current_ptr1_int = __shfl_sync(active_mask, ptr1_int, i);
        uint32_t current_length = __shfl_sync(active_mask, length, i);

        // Cast back to pointer
        uint8_t *current_ptr1 = reinterpret_cast<uint8_t *>(current_ptr1_int);

        // Only proceed if there's data to process
        if (current_length > 0) {
            // Each thread handles a portion of the memory operation
            for (uint32_t offset = lane_id; offset < current_length; offset += num_active_threads) {
                if (offset < current_length) {
                    // Set Zero: Zero out ptr1
                    current_ptr1[offset] = 0;
                }
            }
        }
    }
}
__device__ void evm_memory_t::print() const {
    printf("Memory data: \n");
    printf("Size: %d\n", size);
    printf("Memory cost: %lu\n", memory_cost);
    printf("\n");
    for (uint32_t i = 0; i < size; i++) {
        // if (i == 228) printf("---debug---\n");
        if (preallocated_base_offset + i < memory_prealloc_size) {
            printf("%02x", memory_pool::preallocated_memory_base[preallocated_base_offset + i]);
        } else {
            printf("%02x", dynamic_data[preallocated_base_offset + i - memory_prealloc_size]);
        }
    }
    printf("\n");
}

__device__ void evm_memory_t::increase_memory_cost(gas_t memory_expansion_cost) {
    memory_cost += memory_expansion_cost;
}

__device__ int32_t evm_memory_t::grow(uint32_t new_size) {
    // printf("grow new_size %u size %u\n", new_size, size);
    if (new_size > size) {
        new_size = (new_size + 31) / 32 * 32;
        if (new_size + preallocated_base_offset <= memory_prealloc_size) {
            // no need to allocate new page
            // clear the grow memory when return from subcontext
        } else {
#ifdef DEBUG_PERF
            printf("instance %u dynamic memory allocation new size %u currentsize %u base_offset %u\n",
                   INSTANCE_GLOBAL_IDX, new_size, size, preallocated_base_offset);
#endif
            // allocate new page
            uint8_t *new_data = new uint8_t[new_size + preallocated_base_offset - memory_prealloc_size];
            memset(new_data, 0, new_size + preallocated_base_offset - memory_prealloc_size);
            if (dynamic_data != nullptr) {
                memcpy(new_data, dynamic_data, size + preallocated_base_offset - memory_prealloc_size);
                delete[] dynamic_data;
            }
            dynamic_data = new_data;
        }
        size = new_size;
    }
    return ERROR_SUCCESS;
}

__device__ int32_t evm_memory_t::get(uint32_t index, uint32_t length, uint8_t *&data_) {
    int32_t error_code = ERROR_SUCCESS;
    if (length == 0) {
        data_ = nullptr;
        return error_code;
    }
    // printf("memory get index %u length %eu\n", index, length);
    // Ensure the memory is grown to cover up to index+length.
    error_code |= grow(index + length);
    if (error_code != ERROR_SUCCESS) {
        return error_code;
    }

    // Compute the absolute offset into the memory pool.
    uint32_t total_offset = preallocated_base_offset + index;

    // If the entire requested block fits within the preallocated memory,
    // note the boundary check now uses <= to include the case where the block exactly fits.
    if (total_offset + length <= memory_prealloc_size) {
        data_ = &memory_pool::preallocated_memory_base[memory_prealloc_size * INSTANCE_GLOBAL_IDX + total_offset];
    } else {
#ifdef DEBUG_PERF
        printf("memory get total_offset + length %d > memory_prealloc_size %d\n", total_offset + length,
               memory_prealloc_size);
#endif
        // The requested block spans the preallocated area and the dynamic area (or is entirely in dynamic)
        // Allocate a new buffer to hold the returned data.

        // there is a membug if data_ is not allocated and not initiated outside and not nullptr;
        if (data_ == nullptr) data_ = new uint8_t[length];

        // Copy the portion from the preallocated memory, if any.
        uint32_t prealloc_bytes = 0;
        if (total_offset < memory_prealloc_size) {
            prealloc_bytes = memory_prealloc_size - total_offset;
            if (prealloc_bytes > length) {
                prealloc_bytes = length;
            }

            memcpy(data_,
                   &memory_pool::preallocated_memory_base[memory_prealloc_size * INSTANCE_GLOBAL_IDX + total_offset],
                   prealloc_bytes);
        }
        //
        // Copy the remaining portion from the dynamic memory.
        uint32_t dynamic_bytes = length - prealloc_bytes;
        if (dynamic_bytes > 0) {
            // The dynamic_data pointer holds bytes starting from offset memory_prealloc_size.
            // Compute the corresponding offset into dynamic_data.
            uint32_t dynamic_offset = (total_offset + prealloc_bytes) - memory_prealloc_size;
            // printf("dynamic_bytes %u dynamic_offset %u dynamic_data %p\n", dynamic_bytes, dynamic_offset,
            // dynamic_data);
            memcpy(data_ + prealloc_bytes, dynamic_data + dynamic_offset, dynamic_bytes);
        }
    }
    return error_code;
}

__device__ int32_t evm_memory_t::copy(uint32_t index, uint32_t length, uint8_t *&data_) {
    int32_t error_code = ERROR_SUCCESS;
    if (length == 0) {
        data_ = nullptr;
        return error_code;
    }
    if (data_ == nullptr) data_ = new uint8_t[length];

    // Ensure the memory is grown to cover up to index+length.
    error_code |= grow(index + length);
    if (error_code != ERROR_SUCCESS) {
        return error_code;
    }

    // Compute the absolute offset into the memory pool.
    uint32_t total_offset = preallocated_base_offset + index;

    // If the entire requested block fits within the preallocated memory,
    // note the boundary check now uses <= to include the case where the block exactly fits.
    if (total_offset + length <= memory_prealloc_size) {
        // data_ = &memory_pool::preallocated_memory_base[memory_prealloc_size * INSTANCE_GLOBAL_IDX +
        // total_offset];
        // memcpy(data_, &memory_pool::preallocated_memory_base[memory_prealloc_size * INSTANCE_GLOBAL_IDX +
        // total_offset],
        //        length);
        memory::warp_cooperative_set(
            data_, &memory_pool::preallocated_memory_base[memory_prealloc_size * INSTANCE_GLOBAL_IDX + total_offset],
            length);
    } else {
        // The requested block spans the preallocated area and the dynamic area (or is entirely in dynamic)
        // Allocate a new buffer to hold the returned data.

        // Copy the portion from the preallocated memory, if any.
        uint32_t prealloc_bytes = 0;
        if (total_offset < memory_prealloc_size) {
            prealloc_bytes = memory_prealloc_size - total_offset;
            if (prealloc_bytes > length) {
                prealloc_bytes = length;
            }
            memcpy(data_,
                   &memory_pool::preallocated_memory_base[memory_prealloc_size * INSTANCE_GLOBAL_IDX + total_offset],
                   prealloc_bytes);
        }

        // Copy the remaining portion from the dynamic memory.
        uint32_t dynamic_bytes = length - prealloc_bytes;
        if (dynamic_bytes > 0) {
            // The dynamic_data pointer holds bytes starting from offset memory_prealloc_size.
            // Compute the corresponding offset into dynamic_data.
            uint32_t dynamic_offset = (total_offset + prealloc_bytes) - memory_prealloc_size;
            memcpy(data_ + prealloc_bytes, dynamic_data + dynamic_offset, dynamic_bytes);
        }
    }
    return error_code;
}

// Helper function moved outside of the class methods.
// It copies as much as available from 'src' into 'dest' and zero‑pads the rest.
// If 'src' is nullptr, it simply zeroes the destination.
__device__ inline void copy_with_padding(uint8_t *dest, const uint8_t *src, uint32_t src_available, uint32_t bytes) {
    uint32_t to_copy = (src != nullptr) ? ((src_available < bytes) ? src_available : bytes) : 0;
    unsigned active_mask = __activemask();
    uint32_t num_active_threads = __popc(active_mask);
    // printf("copy_with_padding thread %d num_active_threads %d to_copy %d bytes %d\n", THREADIDX, num_active_threads,
    // to_copy, bytes);
    if (src != nullptr && to_copy > 0) {
        // memcpy(dest, src, to_copy);
        if (num_active_threads == 32)
            CuEVM::memory::warp_cooperative_set(dest, src, to_copy);
        else
            memcpy(dest, src, to_copy);
    }
    if (to_copy < bytes) {
        // memset(dest + to_copy, 0, bytes - to_copy);
        if (num_active_threads == 32)
            CuEVM::memory::warp_cooperative_setzero(dest + to_copy, bytes - to_copy);
        else
            memset(dest + to_copy, 0, bytes - to_copy);
    }
}

// Refactored implementation of evm_memory_t::set using copy_with_padding.
// If data_ is not provided (i.e. is nullptr) the function does nothing (as in the original).
__device__ int32_t evm_memory_t::set(uint8_t *data_, uint32_t data_size, const uint32_t index, const uint32_t length) {
    int32_t error_code = ERROR_SUCCESS;
    if (length == 0) {
        return error_code;
    }

    // Grow the memory as needed.
    error_code |= grow(index + length);
    if (error_code != ERROR_SUCCESS) {
        return error_code;
    }

    // Only perform the write if source data is provided.
    if (data_ == nullptr) {
        return error_code;
    }

    uint32_t total_offset = preallocated_base_offset + index;
    // Determine how many bytes fall into the preallocated region.
    uint32_t available_prealloc = (total_offset < memory_prealloc_size) ? (memory_prealloc_size - total_offset) : 0;
    uint32_t prealloc_bytes = (length < available_prealloc) ? length : available_prealloc;
    uint32_t dynamic_bytes = length - prealloc_bytes;

    // Write into the preallocated region.
    uint8_t *prealloc_dest =
        &memory_pool::preallocated_memory_base[memory_prealloc_size * INSTANCE_GLOBAL_IDX + total_offset];

    copy_with_padding(prealloc_dest, data_, data_size, prealloc_bytes);

    // Write into the dynamic region if needed.
    if (dynamic_bytes > 0) {
        uint32_t dynamic_offset = (total_offset + prealloc_bytes) - memory_prealloc_size;
        uint8_t *dynamic_dest = dynamic_data + dynamic_offset;
        uint32_t remaining_source = (data_size > prealloc_bytes) ? (data_size - prealloc_bytes) : 0;

        copy_with_padding(dynamic_dest, data_ + prealloc_bytes, remaining_source, dynamic_bytes);
    }
    return error_code;
}

__device__ int32_t evm_memory_t::set_zero(const uint32_t index, const uint32_t length) {
    int32_t error_code = ERROR_SUCCESS;
    if (length == 0) {
        return error_code;
    }

    // Ensure that memory is grown to cover [index, index+length)
    error_code |= grow(index + length);
    if (error_code != ERROR_SUCCESS) {
        return error_code;
    }

    uint32_t total_offset = preallocated_base_offset + index;
    if (total_offset + length <= memory_prealloc_size) {
        // Entire block is within the preallocated memory.
        // memset(&memory_pool::preallocated_memory_base[memory_prealloc_size * INSTANCE_GLOBAL_IDX + total_offset], 0,
        //        length);
        CuEVM::memory::warp_cooperative_setzero(
            &memory_pool::preallocated_memory_base[memory_prealloc_size * INSTANCE_GLOBAL_IDX + total_offset], length);
    } else {
        // The request spans both preallocated and dynamic memory.
        uint32_t prealloc_bytes = 0;
        if (total_offset < memory_prealloc_size) {
            prealloc_bytes = memory_prealloc_size - total_offset;
            if (prealloc_bytes > length) {
                prealloc_bytes = length;
            }
            memset(&memory_pool::preallocated_memory_base[memory_prealloc_size * INSTANCE_GLOBAL_IDX + total_offset], 0,
                   prealloc_bytes);
        }
        uint32_t dynamic_bytes = length - prealloc_bytes;
        if (dynamic_bytes > 0) {
            uint32_t dynamic_offset = (total_offset + prealloc_bytes) - memory_prealloc_size;
            memset(dynamic_data + dynamic_offset, 0, dynamic_bytes);
        }
    }
    return error_code;
}

// Refactored implementation of evm_memory_t::set_buffer_data using copy_with_padding.
// When source data is not provided (or data_offset is invalid) the target is zero filled.
__device__ int32_t evm_memory_t::set_buffer_data(uint8_t *data_, uint32_t data_offset, uint32_t data_size,
                                                 const uint32_t index, const uint32_t length) {
    // printf("set_buffer_data thread %d data_offset %d data_size %d index %d length %d\n", THREADIDX, data_offset,
    //        data_size, index, length);
    int32_t error_code = ERROR_SUCCESS;
    if (length == 0) return error_code;

    // If no valid source data is provided, fill the entire region with zeros.
    if (data_ == nullptr || data_offset > data_size) return set_zero(index, length);

    // Grow the memory as needed.
    error_code |= grow(index + length);
    // if (error_code != ERROR_SUCCESS) return error_code;

    uint32_t total_offset = preallocated_base_offset + index;

    // Otherwise, valid source data is available.
    uint32_t available_prealloc = (total_offset < memory_prealloc_size) ? (memory_prealloc_size - total_offset) : 0;
    uint32_t prealloc_bytes = (length < available_prealloc) ? length : available_prealloc;
    uint32_t dynamic_bytes = length - prealloc_bytes;
    // printf("set_buffer_data thread %d total_offset %d available_prealloc %d prealloc_bytes %d dynamic_bytes
    // %d\n",
    //        THREADIDX, total_offset, available_prealloc, prealloc_bytes, dynamic_bytes);
    // For the preallocated region, calculate the available bytes from the buffer.
    uint32_t available_source = (data_offset < data_size) ? (data_size - data_offset) : 0;
    uint8_t *prealloc_dest =
        &memory_pool::preallocated_memory_base[memory_prealloc_size * INSTANCE_GLOBAL_IDX + total_offset];
    // printf("available_source %d\n", available_source);
    copy_with_padding(prealloc_dest, data_ + data_offset, available_source, prealloc_bytes);
    // printf("after copy_with_padding\n");
    // if (INSTANCE_GLOBAL_IDX == 0) {
    //     printf("prealloc_dest\n");
    //     for (uint32_t i = 0; i < prealloc_bytes; i++) {
    //         printf("%x ", prealloc_dest[i]);
    //     }
    //     printf("\n");
    // }
    // For the dynamic region, adjust the source pointer and available bytes.
    if (dynamic_bytes > 0) {
        uint32_t dynamic_offset = (total_offset + prealloc_bytes) - memory_prealloc_size;
        uint8_t *dynamic_dest = dynamic_data + dynamic_offset;
        uint32_t remaining_source =
            ((data_offset + prealloc_bytes) < data_size) ? (data_size - (data_offset + prealloc_bytes)) : 0;
        copy_with_padding(dynamic_dest, data_ + data_offset + prealloc_bytes, remaining_source, dynamic_bytes);
    }
    return error_code;
}

}  // namespace CuEVM::memory
