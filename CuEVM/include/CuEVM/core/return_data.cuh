#pragma once

#include <CuEVM/core/byte_array.cuh>

namespace CuEVM {
/**
 * The return data class. (YP: \f$H_{return}(\mu)=H(\mu, I)\f$)
 */
typedef struct byte_array_t evm_return_data_t;
}  // namespace CuEVM
