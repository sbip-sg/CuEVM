#pragma once
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
// #include "bigint.cuh"
// subset and fixed size version of bigint
#define UINT256_WORDS 8
#define UINT256_BITS 256
#define UINT256_BYTES 32
#define UINT256_LIMBS_BYTES 4

typedef struct uint256 {
    uint32_t words[UINT256_WORDS];
} uint256;
// wide type for 256 bit operations
typedef struct uint512 {
    uint32_t words[UINT256_WORDS * 2];
} uint512;
typedef void (*bigint_rand_func)(uint8_t *dst, int n);

__host__ __device__ int uint256_cmp(const uint256 *a, const uint256 *b);
__host__ __device__ int uint256_signed_cmp(const uint256 *a, const uint256 *b);

__host__ __device__ int uint256_cmp_word(const uint256 *a, uint32_t b);
__host__ __device__ bool uint256_is_zero(const uint256 *a);
__host__ __device__ int uint256_set_zero(uint256 *a);
__host__ __device__ uint32_t uint256_get_uint32_t(const uint256 *a);
__host__ __device__ uint64_t uint256_get_uint64_t(const uint256 *a);
__host__ __device__ uint256 *uint256_cpy(uint256 *dst, const uint256 *src);

// __host__ __device__ uint256*     uint256_clr_bit(uint256 *dst, unsigned bit_index);
__host__ __device__ uint256 *uint256_set_bit(uint256 *dst, unsigned bit_index, uint8_t value);

__host__ __device__ uint256 *uint256_mul(uint256 *dst, const uint256 *a, const uint256 *b);
__host__ __device__ uint256 *uint256_mulmod(uint256 *dst, const uint256 *a, const uint256 *b, const uint256 *N);
__host__ __device__ uint256 *uint256_mulmod_largeN(uint256 *dst, const uint256 *a, const uint256 *b, const uint256 *N);
__host__ __device__ uint256 *uint256_from_hex(uint256 *dst, const char *src);
__host__ __device__ uint256 *uint256_from_uint32(uint256 *dst, uint32_t src);
__host__ __device__ uint256 *uint256_from_word(uint256 *dst, uint32_t a);
__host__ __device__ uint256 *uint256_from_bytes(uint256 *dst, const uint8_t *src, size_t len);

__host__ __device__ uint256 *uint256_add(uint256 *dst, const uint256 *a, const uint256 *b);
__host__ __device__ uint256 *uint256_addmod(uint256 *dst, const uint256 *a, const uint256 *b, const uint256 *N);
__host__ __device__ uint256 *uint256_sub(uint256 *dst, const uint256 *a, const uint256 *b);
__host__ __device__ uint256 *uint256_submod(uint256 *dst, const uint256 *a, const uint256 *b, const uint256 *N);
__host__ __device__ uint256 *uint256_add_word(uint256 *dst, const uint256 *src_a, uint32_t b);
__host__ __device__ uint256 *uint256_sub_word(uint256 *dst, const uint256 *src_a, uint32_t b);

__host__ __device__ char *uint256_to_hex(char *dst, const uint256 *a);

__host__ __device__ uint256 *uint256_shift_left(uint256 *dst, const uint256 *src, uint32_t shift);
__host__ __device__ uint256 *uint256_shift_right(uint256 *dst, const uint256 *src, uint32_t shift);
__host__ __device__ uint256 *uint256_shift_arithmetic_right(uint256 *dst, const uint256 *src, uint32_t shift);
__host__ __device__ uint256 *uint256_bitwise_and(uint256 *dst, const uint256 *a, const uint256 *b);
__host__ __device__ uint256 *uint256_bitwise_or(uint256 *dst, const uint256 *a, const uint256 *b);
__host__ __device__ uint256 *uint256_bitwise_xor(uint256 *dst, const uint256 *a, const uint256 *b);
__host__ __device__ uint256 *uint256_bitwise_not(uint256 *dst, const uint256 *a);

__host__ __device__ uint32_t uint256_bitlength(const uint256 *a);

__host__ __device__ uint256 *uint256_div_mod(uint256 *dst_quotient, uint256 *dst_remainder,
                                             const uint256 *src_biginterator, const uint256 *src_denominator);
__host__ __device__ uint256 *uint256_negate(uint256 *dst, const uint256 *src);
__host__ __device__ uint256 *uint256_div(uint256 *dst, const uint256 *numerator, const uint256 *denominator);
__host__ __device__ uint256 *uint256_exp(uint256 *dst, const uint256 *base, const uint256 *exponent);
__host__ __device__ uint256 *uint256_mod(uint256 *dst, const uint256 *numerator, const uint256 *denominator);
__host__ __device__ uint256 *uint256_signed_mod(uint256 *dst, const uint256 *numerator, const uint256 *denominator);
__host__ __device__ uint256 *uint256_signed_div(uint256 *dst, const uint256 *numerator, const uint256 *denominator);
__host__ __device__ uint256 *uint256_powmod(uint256 *dst, const uint256 *base, const uint256 *exponent,
                                            const uint256 *modulus);
__host__ __device__ uint256 *uint256_sign_extension(uint256 *dst, const uint256 *src, const uint32_t bit_length);
__host__ __device__ uint8_t *uint256_to_bytes(uint8_t *dst, const uint256 *src, size_t len);

__host__ __device__ uint256 *uint256_extract_byte(uint256 *dst, const uint256 *src, uint32_t byte_index);

__host__ __device__ uint512 *uint256_mul_wide(uint512 *dst, const uint256 *a, const uint256 *b);
__host__ __device__ uint256 *uint512_mod(uint256 *dst_remainder, const uint512 *src_biginterator,
                                         const uint256 *src_denominator);
__host__ __device__ void print_uint256(const uint256 *a);
__host__ __device__ void print_uint512(const uint512 *a);
// __host__ __device__ void print_bigint(const bigint *a);
// __host__ __device__ uint256 *uint256_from_bigint(uint256 *dst, const bigint *src);
// __host__ __device__ bigint *bigint_from_uint256(bigint *dst, const uint256 *src,
//                                                 const uint32_t word_size = UINT256_WORDS);

// Helper functions for uint512

__host__ __device__ int uint512_cmp(const uint512 *a, const uint512 *b);
__host__ __device__ int uint512_bitlength(const uint512 *num);
__host__ __device__ uint512 *uint512_set_bit(uint512 *num, int bit_index, uint8_t value);
__host__ __device__ uint512 *uint512_sub(uint512 *dst, const uint512 *a, const uint512 *b);
__host__ __device__ uint512 *uint512_shift_left(uint512 *dst, const uint512 *src, uint32_t shift);
__host__ __device__ uint512 *uint512_shift_right(uint512 *dst, const uint512 *src, uint32_t shift);
__host__ __device__ uint256 *uint512_div_mod(uint256 *dst_quotient, uint256 *dst_remainder, const uint512 *src_dividend,
                                             const uint256 *src_divisor);

__device__ bool uint256_fast_div(uint256 *dst, const uint256 *src_num, const uint256 *src_den);
__device__ bool uint256_fast_exp(uint256 *dst, const uint256 *base, const uint256 *exponent);
