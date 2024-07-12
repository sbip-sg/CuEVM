#ifndef _CUEVM_DEFINES_H_
#define _CUEVM_DEFINES_H_

#include "cuda_utils.cuh"

#define EIP_170
#define EIP_3860

namespace cuEVM {

constexpr CONSTANT uint32_t bits_per_byte = 8;

constexpr CONSTANT uint32_t word_size = 32;

constexpr CONSTANT uint32_t word_bits = word_size * bits_per_byte;

constexpr CONSTANT uint32_t max_stack_size = 1024;

constexpr CONSTANT uint32_t address_size = 20;

constexpr CONSTANT uint32_t address_bits = address_size * bits_per_byte;

constexpr CONSTANT uint32_t max_depth = 1024;

constexpr CONSTANT uint32_t hash_size = 32;

#ifdef EIP_170
constexpr CONSTANT uint32_t max_code_size = 24576;
#else
#error "EIP_170 is not defined"
constexpr CONSTANT uint32_t max_code_size = std::numeric_limits<uint32_t>::max();
#endif

#ifdef EIP_3860
constexpr CONSTANT uint32_t max_initcode_size = 2 * max_code_size;
#else
#error "EIP_3860 is not defined"
constexpr CONSTANT uint32_t max_initcode_size = std::numeric_limits<uint32_t>::max();
#endif

// CGBN parameters
constexpr CONSTANT uint32_t cgbn_tpi = 32;

// CUEVM parameters
constexpr CONSTANT uint32_t max_transactions_count = 10000;

} // namespace cuEVM

#endif