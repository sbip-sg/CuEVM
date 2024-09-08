// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#ifndef _CUEVM_JUMP_DESTINATION_H_
#define _CUEVM_JUMP_DESTINATION_H_

#include <stdint.h>
#include <stdio.h>
#include <cuda.h>
#include <CuEVM/core/byte_array.cuh>

namespace cuEVM {
    /**
     * The class for keeping the valid jump destination of the code
     * (i.e. the locations of the JUMPDEST opcodes).
     * (YP: \f$D(c)=D_{j}(c,0)\f$)
    */
    class jump_destinations_t
    {
    private:
        uint32_t _size; /**< The number of valid JUMPDEST \f$|D(c)|\f */
        uint32_t *_destinations; /**< The array of valid JUMPDESTs pc \f$D(c)\f$ */

    public:
        /**
         * The default constructor of the class
        */
        __host__ __device__ jump_destinations_t() : _size(0), _destinations(nullptr) {};
        /**
         * Cosntructor from the given code
         * @param[in] byte_code The code
        */
        __host__ __device__ jump_destinations_t(
            cuEVM::byte_array_t &byte_code
        );

        /**
         * Destructor of the class
         * free the alocated memory
        */
        __host__ __device__ ~jump_destinations_t();

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