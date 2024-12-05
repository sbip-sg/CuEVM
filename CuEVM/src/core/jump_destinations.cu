// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation
// Programme) Author: Stefan-Dan Ciocirlan Data: 2023-11-30
// SPDX-License-Identifier: MIT

#include <CuEVM/core/jump_destinations.cuh>
#include <CuEVM/utils/cuda_utils.cuh>
#include <CuEVM/utils/error_codes.cuh>
#include <CuEVM/utils/evm_defines.cuh>
#include <CuEVM/utils/opcodes.cuh>

namespace CuEVM {
__host__ __device__ jump_destinations_t::jump_destinations_t(CuEVM::byte_array_t &byte_code) {
    return;  // temporarily disabled
    size = 0;
    capacity = 0;
    uint8_t opcode;
    uint8_t push_size;
    uint16_t pc;
    uint16_t current_capacity = 16;
    grow_capacity(current_capacity);

    for (pc = 0; pc < byte_code.size; pc++) {
        opcode = byte_code.data[pc];
        // if a push x
        if (((opcode & 0xF0) == 0x60) || ((opcode & 0xF0) == 0x70)) {
            push_size = (opcode & 0x1F) + 1;
            pc = pc + push_size;
        }
        if (opcode == OP_JUMPDEST) {
            size++;
            if (size > current_capacity) {
                current_capacity = current_capacity + 16;
                grow_capacity(current_capacity);
            }
            destinations[size - 1] = pc;
        }
    }

    // #ifdef __CUDA_ARCH__
    //     printf("jump destination initialized size %d capapcity %d\n", size, capacity, threadIdx.x);
    // #endif
    // if (size > 0) {
    //     destinations.grow(size, 1);
    //     uint32_t index = 0;
    //     for (pc = 0; pc < byte_code.size; pc++) {
    //         opcode = byte_code.data[pc];
    //         // if a push x
    //         if (((opcode & 0xF0) == 0x60) || ((opcode & 0xF0) == 0x70)) {
    //             push_size = (opcode & 0x1F) + 1;
    //             pc = pc + push_size;
    //         }
    //         if (opcode == OP_JUMPDEST) {
    //             destinations[index] = pc;
    //             index++;
    //         }
    //     }
    // }
}
__host__ __device__ void jump_destinations_t::set_bytecode(CuEVM::byte_array_t &byte_code) {
    return;  // temporarily disabled
    size = 0;
    uint8_t opcode;
    uint8_t push_size;
    uint16_t pc;
    uint16_t current_capacity = capacity;
    if (current_capacity == 0) {
        current_capacity = 16;
        grow_capacity(current_capacity);
    }

    // grow_capacity(current_capacity);

    for (pc = 0; pc < byte_code.size; pc++) {
        opcode = byte_code.data[pc];
        // if a push x
        if (((opcode & 0xF0) == 0x60) || ((opcode & 0xF0) == 0x70)) {
            push_size = (opcode & 0x1F) + 1;
            pc = pc + push_size;
        }
        if (opcode == OP_JUMPDEST) {
            size++;
            if (size > current_capacity) {
                current_capacity = current_capacity + 16;
                grow_capacity(current_capacity);
            }
            destinations[size - 1] = pc;
        }
    }
    // #ifdef __CUDA_ARCH__
    //     printf("jump destination copied set_bytecode size %d capapcity %d\n", size, capacity, threadIdx.x);
    // #endif
}
__host__ __device__ jump_destinations_t::~jump_destinations_t() {
    if (capacity > 0) {
        __ONE_GPU_THREAD_WOSYNC_BEGIN__
        delete[] destinations;
        __ONE_GPU_THREAD_WOSYNC_END__
    }
}

__host__ __device__ uint32_t jump_destinations_t::has(uint32_t pc) {
    return ERROR_SUCCESS;  // Temporarily disabled
    // return destinations.has_value(pc) == ERROR_SUCCESS ? ERROR_SUCCESS : ERROR_INVALID_JUMP_DESTINATION;

    __SHARED_MEMORY__ uint32_t error_code[CGBN_IBP];
    uint32_t index;
    error_code[INSTANCE_IDX_PER_BLOCK] = ERROR_INVALID_JUMP_DESTINATION;
    __SYNC_THREADS__
#ifdef __CUDA_ARCH__
    uint32_t slot_size = (size + CuEVM::cgbn_tpi) / CuEVM::cgbn_tpi;
    for (index = 0; index < slot_size; index++) {
        if (slot_size * threadIdx.x + index >= size) {
            break;
        }
        if (destinations[slot_size * threadIdx.x + index] > pc) {
            break;
        }
        if (destinations[slot_size * threadIdx.x + index] == pc) {
            error_code[INSTANCE_IDX_PER_BLOCK] = ERROR_SUCCESS;
            break;
        }
    }
    __SYNC_THREADS__
#else
    for (index = 0; index < size; index++) {
        if (destinations[index] == pc) {
            error_code[INSTANCE_IDX_PER_BLOCK] = ERROR_SUCCESS;
            break;
        }
    }
#endif
    return error_code[INSTANCE_IDX_PER_BLOCK];
}

__host__ __device__ int32_t jump_destinations_t::grow_capacity(uint32_t new_capacity) {
    if (new_capacity == capacity) return ERROR_SUCCESS;
    __SHARED_MEMORY__ uint16_t *new_data[CGBN_IBP];
    __ONE_GPU_THREAD_BEGIN__
    new_data[INSTANCE_IDX_PER_BLOCK] = new uint16_t[new_capacity];
    if (capacity > 0) {
        // printf("Copying destinations\n");
        // printf("New capacity: %d\n", new_capacity);
        // printf("Old capacity: %d\n", capacity);
        // printf("destination ptr: %p\n", destinations);
        // if (new_size > size) {
        memcpy(new_data[INSTANCE_IDX_PER_BLOCK], destinations, min(new_capacity, capacity) * sizeof(uint16_t));

        delete[] destinations;
    }
    __ONE_GPU_THREAD_END__
    destinations = new_data[INSTANCE_IDX_PER_BLOCK];
    capacity = new_capacity;
    return ERROR_SUCCESS;
}

__host__ __device__ void jump_destinations_t::print() {
    // // Temporarily disabled
    // for (uint32_t i = 0; i < real_size; i++) {
    //     printf("Jump destination %d: %d\n", i, destinations[i]);
    // }
}

}  // namespace CuEVM