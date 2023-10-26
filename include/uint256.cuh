// base_uint256.cuh

#ifndef BASE_UINT256_CUH
#define BASE_UINT256_CUH

#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <cuda_runtime.h>

#define BITS 256
#define WIDTH (BITS / 32)

typedef struct
{
    uint32_t pn[WIDTH];
} base_uint;

// utility functions
__host__ int hexToInt(const char *hex);

__host__ void intToHex(int num, char *hex);

__host__ bool hex_to_decimal(const char *hex_str, char *dec_str);

__host__ __device__  void print_base_uint(const base_uint *val);

// conversion operations
__host__ bool base_uint_set_hex(base_uint *val, const char *hex);

__host__ void base_uint_to_string(const base_uint *val, char *out_str);

__host__ bool int_to_base_uint(int int_val, base_uint *val);

__host__ __device__ void base_uint_get_hex(const base_uint *val, char *hex);


// comparison operations
__host__ __device__ bool is_zero(const base_uint *num);

// bitwise operations
__host__ __device__ base_uint bitwise_not(const base_uint *num);

__host__ __device__ void base_uint_set_bit(base_uint *value, uint32_t bitpos);

// arithmetic operations

__host__ __device__ void base_uint_add(const base_uint *a, const base_uint *b, base_uint *result);

__host__ __device__ bool base_uint_sub(const base_uint *a, const base_uint *b, base_uint *result);

__host__ __device__ void base_uint_mul(const base_uint *a, const base_uint *b, base_uint *result);

__host__ __device__ void base_uint_shift_left(base_uint *a, size_t bits);

__host__ __device__ void base_uint_div(const base_uint *a, const base_uint *b, base_uint *quotient, base_uint *remainder);

#endif // BASE_UINT256_CUH
