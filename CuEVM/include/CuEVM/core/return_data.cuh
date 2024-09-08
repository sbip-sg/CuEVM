// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#pragma once

#include <CuEVM/core/byte_array.cuh>

namespace CuEVM {
    /**
     * The return data class. (YP: \f$H_{return}(\mu)=H(\mu, I)\f$)
    */
    typedef struct byte_array_t evm_return_data_t ;
}
