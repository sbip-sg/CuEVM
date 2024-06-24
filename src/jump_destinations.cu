// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#include "include/jump_destinations.cuh"
#include "include/opcodes.h"
#include "include/utils.h"

namespace cuEVM {
    __host__ __device__ EVMJumpDestinations::EVMJumpDestinations(
        uint8_t *byte_code,
        uint32_t code_size
    )
    {
        _size = 0;
        _destinations = NULL;
        uint8_t opcode;
        uint8_t push_size;
        uint32_t pc;
        for (pc = 0; pc < code_size; pc++)
        {
            opcode = byte_code[pc];
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
        SHARED_MEMORY uint32_t *tmp_destinations;
        ONE_THREAD_PER_INSTANCE(
            if (_size > 0) {
                tmp_destinations = new uint32_t[_size];
                uint32_t index = 0;
                for (pc = 0; pc < code_size; pc++)
                {
                    opcode = byte_code[pc];
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
                tmp_destinations = NULL;
            }
        )
        _destinations = tmp_destinations;
    }

    __host__ __device__ EVMJumpDestinations::~EVMJumpDestinations()
    {
        ONE_THREAD_PER_INSTANCE(
            if ((_destinations != NULL) && (_size > 0)) {
                delete[] _destinations;
            })
        _destinations = NULL;
        _size = 0;
    }

    /**
     * Find out if a given pc is a valid JUMPDEST
     * @param[in] pc The pc to check
    */
    __host__ __device__ uint32_t EVMJumpDestinations::has(
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

    __host__ __device__ void EVMJumpDestinations::print(
    )
    {
        uint32_t index;
        for (index = 0; index < _size; index++)
        {
            printf("%u\n", _destinations[index]);
        }
    }

}