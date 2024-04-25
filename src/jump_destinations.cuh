// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#ifndef _JUMP_DESTINATION_H_
#define _JUMP_DESTINATION_H_


/**
 * The class for keeping the valid jump destination of the code
 * (i.e. the locations of the JUMPDEST opcodes).
 * (YP: \f$D(c)=D_{j}(c,0)\f$)
*/
class jump_destinations_t
{
private:
    size_t _size; /**< The number of valid JUMPDEST \f$|D(c)|\f */
    size_t *_destinations; /**< The array of valid JUMPDESTs pc \f$D(c)\f$ */

public:
    /**
     * Cosntructor from the given code
     * @param[in] byte_code The code
     * @param[in] code_size The size of the code
    */
    __host__ __device__ __forceinline__ jump_destinations_t(
        uint8_t *byte_code,
        size_t code_size
    )
    {
        _size = 0;
        _destinations = NULL;
        uint8_t opcode;
        uint8_t push_size;
        size_t pc;
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
        SHARED_MEMORY size_t *tmp_destinations;
        ONE_THREAD_PER_INSTANCE(
            if (_size > 0) {
                tmp_destinations = new size_t[_size];
                size_t index = 0;
                for (pc = 0; pc < code_size; pc++)
                {
                    opcode = byte_code[pc];
                    // if a push x
                    if (((opcode & 0xF0) == 0x60) || ((opcode & 0xF0) == 0x70))
                    {
                        push_size = (opcode & 0x1F) + 1;
                        pc = pc + push_size;
                    }
                    if (opcode == OP_JUMPDEST)
                    {
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

    /**
     * Destructor of the class
     * free the alocated memory
    */
    __host__ __device__ ~jump_destinations_t()
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
    __host__ __device__ uint32_t has(
        size_t pc
    )
    {
        size_t index;
        for (index = 0; index < _size; index++)
        {
            if (_destinations[index] == pc)
            {
                return 1;
            }
        }
        return 0;
    }

    __host__ __device__ void print(
    )
    {
        size_t index;
        for (index = 0; index < _size; index++)
        {
            printf("%lu\n", _destinations[index]);
        }
    }
};

#endif