// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation
// Programme) Author: Stefan-Dan Ciocirlan Data: 2023-11-30
// SPDX-License-Identifier: MIT

#pragma once

#include <cuda.h>
#include <stdint.h>

#include <CuEVM/core/byte_array.cuh>
#include <CuEVM/core/evm_word.cuh>
#include <CuEVM/utils/evm_defines.cuh>

namespace CuEVM {
/**
 * The arithmetic environment class is a wrapper around the CGBN library.
 * It provides a context, environment, and instance for the CGBN library.
 * It also provides some utility functions for converting between CGBN and other
 * types.
 */
class ArithEnv {
    // deprecated
};
}  // namespace CuEVM
