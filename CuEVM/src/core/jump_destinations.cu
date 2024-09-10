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
__host__ __device__
jump_destinations_t::jump_destinations_t(CuEVM::byte_array_t &byte_code)
    : destinations(0U) {
    uint32_t size = 0;
    uint8_t opcode;
    uint8_t push_size;
    uint32_t pc;
    for (pc = 0; pc < byte_code.size; pc++) {
        opcode = byte_code.data[pc];
        // if a push x
        if (((opcode & 0xF0) == 0x60) || ((opcode & 0xF0) == 0x70)) {
            push_size = (opcode & 0x1F) + 1;
            pc = pc + push_size;
        }
        if (opcode == OP_JUMPDEST) {
            size = size + 1;
        }
    }
    if (size > 0) {
        destinations.grow(size, 1);
        uint32_t index = 0;
        for (pc = 0; pc < byte_code.size; pc++) {
            opcode = byte_code.data[pc];
            // if a push x
            if (((opcode & 0xF0) == 0x60) || ((opcode & 0xF0) == 0x70)) {
                push_size = (opcode & 0x1F) + 1;
                pc = pc + push_size;
            }
            if (opcode == OP_JUMPDEST) {
                destinations[index] = pc;
                index++;
            }
        }
    }
}

__host__ __device__ jump_destinations_t::~jump_destinations_t() {}

__host__ __device__ uint32_t jump_destinations_t::has(uint32_t pc) {
    return destinations.has_value(pc) == ERROR_SUCCESS
               ? ERROR_SUCCESS
               : ERROR_INVALID_JUMP_DESTINATION;
}

__host__ __device__ void jump_destinations_t::print() { destinations.print(); }

}  // namespace CuEVM