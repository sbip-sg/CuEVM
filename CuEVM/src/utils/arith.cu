// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation
// Programme) Author: Stefan-Dan Ciocirlan Data: 2023-11-30
// SPDX-License-Identifier: MIT

#include <CuEVM/utils/arith.cuh>
#include <CuEVM/utils/error_codes.cuh>
#include <CuEVM/utils/evm_utils.cuh>

namespace CuEVM {
__device__ ArithEnv::ArithEnv(cgbn_monitor_t monitor,
                              cgbn_error_report_t *report, uint32_t instance)
    : _context(monitor, report, instance), env(_context), instance(instance) {}

__device__ ArithEnv::ArithEnv(cgbn_monitor_t monitor)
    : _context(monitor), env(_context), instance(0) {}

__host__ ArithEnv::ArithEnv(cgbn_monitor_t monitor, uint32_t instance)
    : _context(monitor), env(_context), instance(instance) {}

__host__ __device__ ArithEnv::ArithEnv(const ArithEnv &env)
    : _context(env._context), env(_context), instance(env.instance) {}

__host__ __device__ void cgbn_set_memory(env_t env, bn_t &dst,
                                         const uint8_t *src,
                                         const uint32_t size) {
    cgbn_set_ui32(env, dst, 0);
    uint32_t word_size = env_t::BITS / 8;
    for (uint8_t idx = (word_size - size); idx < word_size; idx++) {
        cgbn_insert_bits_ui32(env, dst, dst, idx * 8, 8,
                              src[word_size - 1 - idx]);
    }
}

__host__ __device__ void cgbn_set_size_t(env_t env, bn_t &dst,
                                         const size_t src) {
    cgbn_set_ui32(env, dst, 0);
    cgbn_insert_bits_ui32(env, dst, dst, 32, 32, (src >> 32));
    cgbn_insert_bits_ui32(env, dst, dst, 0, 32, src);
}

__host__ __device__ int32_t cgbn_get_size_t(env_t env, size_t &dst,
                                            const bn_t &src) {
    bn_t tmp;
    cgbn_bitwise_mask_and(env, tmp, src, sizeof(size_t) * 8);
    size_t result = 0;
    for (uint8_t idx = 0; idx < sizeof(size_t); idx++) {
        result |= ((size_t)cgbn_extract_bits_ui32(env, tmp, idx * 8, 8))
                  << (idx * 8);
    }
    dst = result;
    return cgbn_compare(env, src, tmp) == 0 ? ERROR_SUCCESS
                                            : ERROR_VALUE_OVERFLOW;
}

__host__ __device__ int32_t cgbn_get_uint64_t(env_t env, uint64_t &dst,
                                              const bn_t &src) {
    bn_t tmp;
    cgbn_bitwise_mask_and(env, tmp, src, sizeof(uint64_t) * 8);
    size_t result = 0;
    for (uint8_t idx = 0; idx < sizeof(uint64_t); idx++) {
        result |= ((uint64_t)cgbn_extract_bits_ui32(env, tmp, idx * 8, 8))
                  << (idx * 8);
    }
    dst = result;
    return cgbn_compare(env, src, tmp) == 0 ? ERROR_SUCCESS
                                            : ERROR_VALUE_OVERFLOW;
}

__host__ __device__ int32_t cgbn_get_uint32_t(env_t env, uint32_t &dst,
                                              const bn_t &src) {
    bn_t tmp;
    dst = cgbn_get_ui32(env, src);
    cgbn_set_ui32(env, tmp, dst);
    return cgbn_compare(env, tmp, src) == 0 ? ERROR_SUCCESS
                                            : ERROR_VALUE_OVERFLOW;
}

__host__ __device__ int32_t
cgbn_set_byte_array_t(env_t env, bn_t &out, const byte_array_t &byte_array) {
    uint32_t word_size = env_t::BITS / 8;
    if (byte_array.size != word_size) return ERROR_INVALID_WORD_SIZE;
    for (uint32_t idx = 0; idx < word_size; idx++) {
        cgbn_insert_bits_ui32(env, out, out, env_t::BITS - (idx + 1) * 8, 8,
                              byte_array.data[idx]);
    }
    return ERROR_SUCCESS;
}

__host__ __device__ int32_t get_sub_byte_array_t(ArithEnv &arith,
                                                 const byte_array_t &byte_array,
                                                 const bn_t &index,
                                                 const bn_t &length,
                                                 byte_array_t &out) {
    uint32_t index_value, length_value;
    index_value = cgbn_get_ui32(arith.env, index);
    length_value = cgbn_get_ui32(arith.env, length);
    if ((cgbn_compare_ui32(arith.env, index, index_value) != 0) &&
        (cgbn_compare_ui32(arith.env, length, length_value) != 0)) {
        out = byte_array_t();
        return ERROR_BYTE_ARRAY_OVERFLOW_VALUES;
    }
    if (index_value + length_value > byte_array.size) {
        out = byte_array_t();
        return ERROR_BYTE_ARRAY_INVALID_SIZE;
    }
    out = byte_array_t(byte_array.data + index_value, length_value);
    return ERROR_SUCCESS;
}

__host__ __device__ void evm_address_conversion(ArithEnv &arith,
                                                bn_t &address) {
    cgbn_bitwise_mask_and(arith.env, address, address, CuEVM::address_bits);
}
__host__ __device__ void print_bnt(ArithEnv &arith, const bn_t &bn){
    evm_word_t tmp_word;
    cgbn_store(arith.env, &tmp_word, bn);
    tmp_word.print();
}
}  // namespace CuEVM