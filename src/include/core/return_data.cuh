// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#ifndef _CUEVM_RETURN_DATA_H_
#define _CUEVM_RETURN_DATA_H_

#include "byte_array.cuh"


namespace cuEVM {
    /**
     * The return data class. (YP: \f$H_{return}(\mu)=H(\mu, I)\f$)
    */
    typedef struct evm_return_data_t byte_array_t;
}

#endif