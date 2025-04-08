#pragma once

#include <cuda.h>
#include <stdint.h>
#include <stdio.h>

#include <CuEVM/core/byte_array.cuh>
#include <CuEVM/core/evm_word.cuh>
#include <CuEVM/core/jump_table.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM {


    // Analyze the bytecode and fill the jump table. Returns non-zero when failed
    __host__ int32_t GlobalJumpTable::analyze(uint8_t* bytecode, uint32_t bitmap_offset_start, uint32_t size
    ){
        if (!size) {
            return JUMPTABLE_INVALID_BYTECODE_SIZE;
        }

        if (size + bitmap_offset_start > 8 * GLOBAL_JUMP_TABLE_SIZE) {
            return JUMPTABLE_BITMAP_FULL;
        }

        for (auto pc = 0; pc < size; pc++) {
            uint8_t opcode = bytecode[pc];
            uint8_t push_offset = opcode - 0x60;
            auto bit_offset = bitmap_offset_start + pc;

            if (opcode == 0x5B) {
                bitmap[bit_offset / 8] |= 1 << (bit_offset % 8);
            } else if (push_offset < 32) { // PUSH1 - PUSH32
                pc += push_offset + 1;
            }
        }

        return ERROR_SUCCESS;
    }

    // Check if the pc is a valid jump destination, assuming the `bitmap_offset_start` contains valid value
    __host__ __device__ int32_t GlobalJumpTable::validate_jumpdest(uint32_t bitmap_offset_start, uint32_t pc){
        uint32_t bit_offset = bitmap_offset_start + pc;

        if (bit_offset > 8 * GLOBAL_JUMP_TABLE_SIZE) {
            return JUMPTABLE_PC_EXCEED_BITMAP_SIZE;
        }

        return (bitmap[bit_offset / 8] & (1 << (bit_offset % 8))) ? ERROR_SUCCESS : ERROR_INVALID_JUMP_DESTINATION;
    }


} // namespace CuEVM
