#pragma once

#include <cuda.h>
#include <stdint.h>
#include <stdio.h>

#include <CuEVM/core/byte_array.cuh>
#include <CuEVM/core/evm_word.cuh>


#define GLOBAL_JUMP_TABLE_SIZE (1024 * 1024 * 1)
#define GLOBAL_JUMP_TABLE_MAX_ADDRESSES 1024

namespace CuEVM {

struct GlobalJumpTable {
    uint8_t bitmap[GLOBAL_JUMP_TABLE_SIZE];
    uint8_t addressOffset[GLOBAL_JUMP_TABLE_MAX_ADDRESSES];
    uint8_t codeSize[GLOBAL_JUMP_TABLE_MAX_ADDRESSES];
    uint32_t bitmap_usage;

    // Analyze the bytecode and fill the jump table. Returns non-zero when failed
    __host__ __device__ int32_t analyze(uint8_t* bytecode, uint32_t address_index, uint32_t size);

    // Check if the address is a valid jump destination
    // Returns
    //  - 0 if the pc is a valid jumpdest
    //  - ERROR_INVALID_JUMP_DESTINATION if it's not valid jumpdest
    //  - other values if there is an error
    __host__ __device__ int32_t validate_jumpdest(uint32_t address_index, uint32_t pc);

};

}
