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
    __host__ __device__ int32_t GlobalJumpTable::analyze(uint8_t* bytecode, uint32_t address_index, uint32_t size){
        if (!size) {
            return JUMPTABLE_INVALID_BYTECODE_SIZE;
        }

        if (address_index >= GLOBAL_JUMP_TABLE_MAX_ADDRESSES) {
            return JUMPTABLE_ADDRESS_FULL;
        }

        if (codeSize[address_index]){
            if (size == codeSize[address_index]){ // Assuming already analyzed
                return ERROR_SUCCESS;
            }
            return JUMPTABLE_ADDRESS_OCCUPIED;
        }

        if (size + bitmap_usage > 8 * GLOBAL_JUMP_TABLE_SIZE) {
            return JUMPTABLE_BITMAP_FULL;
        }

        codeSize[address_index] = size;
        addressOffset[address_index] = bitmap_usage;
        for (auto pc = 0; pc < size; pc++) {
            uint8_t opcode = bytecode[pc];
            uint8_t push_offset = opcode - 0x60;
            auto bit_offset = bitmap_usage + pc;

            if (opcode == 0x5B) {
                bitmap[bit_offset / 8] |= 1 << (bit_offset % 8);
            } else if (push_offset < 32) { // PUSH1 - PUSH32
                pc += push_offset + 1;
            }
        }

        bitmap_usage += size;
        return ERROR_SUCCESS;
    }

    // Check if the address is a valid jump destination
    __host__ __device__ int32_t GlobalJumpTable::validate_jumpdest(uint32_t address_index, uint32_t pc){
        if (address_index >= GLOBAL_JUMP_TABLE_MAX_ADDRESSES) {
            return JUMPTABLE_ADDRESS_INVALID;
        }

        if (!codeSize[address_index]) {
            return JUMPTABLE_ADDRESS_NOT_ANALYZED;
        }

        uint32_t offset = addressOffset[address_index];
        uint32_t bit_offset = offset + pc;

        return (bitmap[bit_offset / 8] & (1 << (bit_offset % 8))) ? ERROR_SUCCESS : ERROR_INVALID_JUMP_DESTINATION;
    }


} // namespace CuEVM
