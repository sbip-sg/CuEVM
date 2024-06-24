// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#ifndef _JUMP_DESTINATION_H_
#define _JUMP_DESTINATION_H_

#include <stdint.h>
#include <stdio.h>
#include <cuda.h>

namespace cuEVM {
    /**
     * The class for keeping the valid jump destination of the code
     * (i.e. the locations of the JUMPDEST opcodes).
     * (YP: \f$D(c)=D_{j}(c,0)\f$)
    */
    class EVMJumpDestinations
    {
    private:
        uint32_t _size; /**< The number of valid JUMPDEST \f$|D(c)|\f */
        uint32_t *_destinations; /**< The array of valid JUMPDESTs pc \f$D(c)\f$ */

    public:
        /**
         * Cosntructor from the given code
         * @param[in] byte_code The code
         * @param[in] code_size The size of the code
        */
        __host__ __device__ EVMJumpDestinations(
            uint8_t *byte_code,
            uint32_t code_size
        );

        /**
         * Destructor of the class
         * free the alocated memory
        */
        __host__ __device__ ~EVMJumpDestinations();

        /**
         * Find out if a given pc is a valid JUMPDEST
         * @param[in] pc The pc to check
        */
        __host__ __device__ uint32_t has(
            uint32_t pc
        );

        __host__ __device__ void print(
        );
    };

}


#endif