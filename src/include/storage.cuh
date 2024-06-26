#ifndef _CUEVM_STORAGE_H_
#define _CUEVM_STORAGE_H_

#include "arith.cuh"

namespace cuEVM
{
    namespace storage {
        typedef struct
        {
            evm_word_t key; /**< The key of the storage */
            evm_word_t value; /**< The value of the storage for the given key */
        } storage_element_t;

    }

}

#endif