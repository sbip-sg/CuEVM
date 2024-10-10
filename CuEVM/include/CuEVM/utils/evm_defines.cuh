// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation
// Programme) Author: Stefan-Dan Ciocirlan Date: 2024-09-15
// SPDX-License-Identifier: MIT

#pragma once

#include <CuEVM/utils/cuda_utils.cuh>
// tracer activated
// #define EIP_3155
// trace optional
// #define EIP_3155_OPTIONAL
#ifndef EVM_VERSION
#define EVM_VERSION SHANGHAI
#endif

#define SHANGHAI_VERSION SHANGHAI
#define PARIS_VERSION PARIS
#define BERLIN_VERSION BERLIN
#define LONDON_VERSION LONDON
#define ISTANBUL_VERSION ISTANBUL
#define CONSTANTINOPLE_VERSION CONSTANTINOPLE
#define BYZANTIUM_VERSION BYZANTIUM
#define TANGARINE_VERSION TANGARINE
#define DRAGON_VERSION DRAGON
#define HOMESTEAD_VERSION HOMESTEAD
#define CANCUN_VERSION CANCUN

#ifdef EVM_VERSION
#if (EVM_VERSION == SHANGHAI_VERSION)
#define SHANGHAI
#elif (EVM_VERSION == PARIS_VERSION)
#define PARIS
#elif (EVM_VERSION == BERLIN_VERSION)
#define BERLIN
#elif (EVM_VERSION == LONDON_VERSION)
#define LONDON
#elif (EVM_VERSION == ISTANBUL_VERSION)
#define ISTANBUL
#elif (EVM_VERSION == CONSTANTINOPLE_VERSION)
#define CONSTANTINOPLE
#elif (EVM_VERSION == BYZANTIUM_VERSION)
#define BYZANTIUM
#elif (EVM_VERSION == TANGARINE_VERSION)
#define TANGARINE
#elif (EVM_VERSION == DRAGON_VERSION)
#define DRAGON
#elif (EVM_VERSION == HOMESTEAD_VERSION)
#define HOMESTEAD
#elif (EVM_VERSION == CANCUN_VERSION)
#define CANCUN
#else
#error "EVM_VERSION is badly defined"
#endif
#else
#error "EVM_VERSION is not defined"
#endif

#ifdef CANCUN
#define EIP_1153
#define EIP_4788
#define EIP_4844
#define EIP_5656
#define EIP_6780
#define EIP_7516
#define SHANGHAI
#endif

#ifdef SHANGHAI
#define EIP_3651
#define EIP_3855
#define EIP_3860
#define EIP_4895
#define EIP_6049
#define PARIS
#endif

#ifdef PARIS
#define EIP_3675
#define EIP_4399
#define EIP_5133
#define LONDON
#endif

#ifdef LONDON
#define EIP_3541
#define BERLIN
#endif

#ifdef GRAY_GLACIER
#define EIP_5133
#endif

#ifdef EIP_5133
#define EIP_4345
#endif

#ifdef ARROW_GLACIER
#define EIP_4345
#endif

#ifdef EIP_4345
#define EIP_2070
#endif

#ifdef BERLIN
#define EIP_2070
#endif

#ifdef EIP_2070
#define EIP_1679
#endif

#ifdef INSTANBUL
#define EIP_1679
#endif

#ifdef EIP_1679
#define EIP_152
#define EIP_1108
#define EIP_1344
#define EIP_1716
#define EIP_1884
#define EIP_2028
#define EIP_2200
#define EIP_1013
#endif

#ifdef CONSTATINOPLE
#define EIP_1013
#endif

#ifdef EIP_1013
#define EIP_145
#define EIP_1014
#define EIP_1052
#define EIP_1234
#define EIP_1283
#define EIP_609
#endif

#ifdef BYZANTIUM
#define EIP_609
#endif

#ifdef EIP_609
#define EIP_100
#define EIP_140
#define EIP_196
#define EIP_197
#define EIP_198
#define EIP_211
#define EIP_214
#define EIP_649
#define EIP_658
#define EIP_607
#endif

#ifdef DRAGON
#define EIP_607
#endif
#ifdef EIP_607
#define EIP_155
#define EIP_160
#define EIP_161
#define EIP_170
#define EIP_608
#endif

#ifdef TANGARINE
#define EIP_608
#endif
#ifdef EIP_608
#define EIP_150
#define EIP_779
#define EIP_606
#endif

#ifdef HOMESTEAD
#define EIP_606
#endif
#ifdef EIP_606
#define EIP_2
#define EIP_7
#define EIP_8
#endif

namespace CuEVM {

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
constexpr CONSTANT uint32_t max_code_size =
    std::numeric_limits<uint32_t>::max();
#endif

#ifdef EIP_3860
constexpr CONSTANT uint32_t max_initcode_size = 2 * max_code_size;
#else
#error "EIP_3860 is not defined"
constexpr CONSTANT uint32_t max_initcode_size =
    std::numeric_limits<uint32_t>::max();
#endif

constexpr CONSTANT uint32_t no_precompile_contracts = 10;

// CGBN parameters
constexpr CONSTANT uint32_t cgbn_tpi = 32;

// CUEVM parameters
constexpr CONSTANT uint32_t max_transactions_count = 10000;

constexpr CONSTANT uint32_t cgbn_limbs = ((CuEVM::word_bits + 31) / 32);

// specific implementation constants
constexpr CONSTANT uint32_t initial_storage_capacity = 4;


/**
 * The CGBN context type.  This is a template type that takes
 * the number of threads per instance and the
 * parameters class as template parameters.
 */
#if defined(__CUDA_ARCH__)
using context_t = cgbn_context_t<CuEVM::cgbn_tpi, cgbn_default_parameters_t>;
#else
using context_t =
    cgbn_host_context_t<CuEVM::cgbn_tpi, cgbn_default_parameters_t>;
#endif

/**
 * The CGBN environment type. This is a template type that takes the
 * context type as a template parameter. It provides the CGBN functions.
 */
using env_t = cgbn_env_t<context_t, CuEVM::word_bits>;

/**
 * The CGBN base type for the given number of bit in environment.
 */
using bn_t = env_t::cgbn_t;
/**
 * The CGBN wide type with double the given number of bits in environment.
 */
using bn_wide_t = env_t::cgbn_wide_t;
}  // namespace CuEVM
