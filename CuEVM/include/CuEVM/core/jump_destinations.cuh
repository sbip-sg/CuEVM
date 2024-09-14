// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation
// Programme) Author: Stefan-Dan Ciocirlan Data: 2023-11-30
// SPDX-License-Identifier: MIT

#pragma once

#include <cuda.h>
#include <stdint.h>
#include <stdio.h>

#include <CuEVM/core/byte_array.cuh>

namespace CuEVM {
/**
 * The struct for keeping the valid jump destination of the code
 * (i.e. the locations of the JUMPDEST opcodes).
 * (YP: \f$D(c)=D_{j}(c,0)\f$)
 */
struct jump_destinations_t {
   private:
    byte_array_t
        destinations; /**< The array of valid JUMPDESTs pc \f$D(c)\f$ */

   public:
    /**
     * Cosntructor from the given code
     * @param[in] byte_code The code
     */
    __host__ __device__ jump_destinations_t(CuEVM::byte_array_t &byte_code);

    /**
     * Destructor of the class
     * free the alocated memory
     */
    __host__ __device__ ~jump_destinations_t();

    /**
     * Find out if a given pc is a valid JUMPDEST
     * @param[in] pc The pc to check
     * @return 0 if is a valid JUMPDEST, 1 otherwise
     */
    __host__ __device__ uint32_t has(uint32_t pc);

    /**
     * Print the destinations
     */
    __host__ __device__ void print();
};

}  // namespace CuEVM
