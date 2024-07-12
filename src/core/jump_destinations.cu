// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#include "../include/core/jump_destinations.cuh"
#include "../include/utils/opcodes.cuh"


namespace cuEVM {
    __host__ __device__ jump_destinations_t::jump_destinations_t(
        cuEVM::byte_array_t &byte_code
    )
    {
        _size = 0;
        _destinations = nullptr;
        uint8_t opcode;
        uint8_t push_size;
        uint32_t pc;
        for (pc = 0; pc < byte_code.size; pc++)
        {
            opcode = byte_code.data[pc];
            // if a push x
            if (
                ((opcode & 0xF0) == 0x60) ||
                ((opcode & 0xF0) == 0x70)
            )
            {
                push_size = (opcode & 0x1F) + 1;
                pc = pc + push_size;
            }
            if (opcode == OP_JUMPDEST)
            {
                _size = _size + 1;
            }
        }
        uint32_t *tmp_destinations;
        if (_size > 0) {
            tmp_destinations = new uint32_t[_size];
            uint32_t index = 0;
            for (pc = 0; pc < byte_code.size; pc++)
            {
                opcode = byte_code.data[pc];
                // if a push x
                if (
                    ((opcode & 0xF0) == 0x60) ||
                    ((opcode & 0xF0) == 0x70)
                ) {
                    push_size = (opcode & 0x1F) + 1;
                    pc = pc + push_size;
                }
                if (opcode == OP_JUMPDEST) {
                    tmp_destinations[index] = pc;
                    index = index + 1;
                }
            }
        } else {
            tmp_destinations = nullptr;
        }
        _destinations = tmp_destinations;
    }

    __host__ __device__ jump_destinations_t::~jump_destinations_t()
    {
        if ((_destinations != nullptr) && (_size > 0)) {
            delete[] _destinations;
        }
        _destinations = nullptr;
        _size = 0;
    }

    /**
     * Find out if a given pc is a valid JUMPDEST
     * @param[in] pc The pc to check
    */
    __host__ __device__ uint32_t jump_destinations_t::has(
        uint32_t pc
    )
    {
        uint32_t index;
        for (index = 0; index < _size; index++)
        {
            if (_destinations[index] == pc)
            {
                return 1;
            }
        }
        return 0;
    }

    __host__ __device__ void jump_destinations_t::print(
    )
    {
        uint32_t index;
        for (index = 0; index < _size; index++)
        {
            printf("%u\n", _destinations[index]);
        }
    }

}