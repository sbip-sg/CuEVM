#pragma once

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>


/* any unsigned integer type */
typedef uint32_t bigint_word;

#define BIGINT_KARATSUBA_WORD_THRESHOLD 20

#define BIGINT_WORD_BITS ((sizeof(bigint_word) * CHAR_BIT))
#define BIGINT_WORD_MAX ((bigint_word)-1)
#define BIGINT_HALF_WORD_MAX (BIGINT_WORD_MAX >> BIGINT_WORD_BITS / 2)

#define BIGINT_WORD_LO(a) ((a) & BIGINT_HALF_WORD_MAX)
#define BIGINT_WORD_HI(a) ((a) >> sizeof(a) * CHAR_BIT / 2)

#define BIGINT_MIN(a, b) ((a) < (b) ? (a) : (b))
#define BIGINT_MAX(a, b) ((a) > (b) ? (a) : (b))
#define BIGINT_INT_ABS(a) ((a) < 0 ? -(unsigned int)(a) : (unsigned int)(a))

#define BIGINT_SWAP(type, a, b) do { type _tmp = a; a = b; b = _tmp; } while (0)

#define BIGINT_REVERSE(type, data, n) do {\
    int _i;\
    for (_i = 0; _i < (n)/2; _i++) BIGINT_SWAP(type, data[_i], data[n - 1 - _i]);\
} while (0)

typedef struct bigint {
    bigint_word *words;
    int neg, size, capacity;
} bigint;

typedef void (*bigint_rand_func)(uint8_t *dst, int n);

__host__ __device__ bigint_word bigint_word_mul_lo(bigint_word a, bigint_word b);
__host__ __device__ bigint_word bigint_word_mul_hi(bigint_word a, bigint_word b);

__host__ __device__ bigint_word bigint_word_add_get_carry(bigint_word *dst, bigint_word a, bigint_word b);
__host__ __device__ bigint_word bigint_word_sub_get_carry(bigint_word *dst, bigint_word a, bigint_word b);

__host__ __device__ bigint_word bigint_word_from_char(char c);

__host__ __device__ int bigint_word_bitlength(bigint_word a);
__host__ __device__ int bigint_word_count_trailing_zeros(bigint_word a);

__host__ __device__ bigint_word bigint_word_gcd(bigint_word a, bigint_word b);
__host__ __device__ unsigned bigint_uint_gcd(unsigned a, unsigned b);
__host__ __device__ int bigint_int_gcd(int a, int b);

__host__ __device__ bigint* bigint_init(bigint *a);
__host__ __device__ bigint* bigint_reserve(bigint *a, int capacity);
__host__ __device__ void bigint_free(bigint *a);

__host__ __device__ int bigint_cmp_abs(const bigint *a, const bigint *b);
__host__ __device__ int bigint_cmp(const bigint *a, const bigint *b);
__host__ __device__ int bigint_cmp_abs_word(const bigint *a, bigint_word b);

__host__ __device__ bigint* bigint_set_neg(bigint *dst, int neg);
__host__ __device__ bigint* bigint_negate(bigint *dst);

__host__ __device__ bigint* bigint_cpy(bigint *dst, const bigint *src);

__host__ __device__ bigint*     bigint_clr_bit(bigint *dst, unsigned bit_index);
__host__ __device__ bigint*     bigint_set_bit(bigint *dst, unsigned bit_index);
__host__ __device__ bigint_word bigint_get_bit(const bigint *src, unsigned bit_index);

__host__ __device__ bigint* bigint_mul(bigint *dst, const bigint *a, const bigint *b);

__host__ __device__ int bigint_count_digits(const char *src);
__host__ __device__ int bigint_digits_bound(int n_digits_src, double src_base, double dst_base);
__host__ __device__ int bigint_write_size(const bigint *a, double dst_base);
__host__ __device__ bigint* bigint_from_str_base(bigint *dst, const char *src, int src_base);
__host__ __device__ bigint* bigint_from_str(bigint *dst, const char *src);
__host__ __device__ bigint* bigint_from_int(bigint *dst, int src);
__host__ __device__ bigint* bigint_from_word(bigint *dst, bigint_word a);
__host__ __device__ bigint* bigint_from_bytes(bigint *dst, const uint8_t *src, size_t len);

__host__ __device__ bigint* bigint_add_signed(bigint *dst, const bigint *a, int a_neg, const bigint *b, int b_neg);
__host__ __device__ bigint* bigint_add(bigint *dst, const bigint *a, const bigint *b);
__host__ __device__ bigint* bigint_sub(bigint *dst, const bigint *a, const bigint *b);
__host__ __device__ bigint* bigint_add_word_signed(bigint *dst, const bigint *src_a, bigint_word b, int b_neg);
__host__ __device__ bigint* bigint_add_word(bigint *dst, const bigint *src_a, bigint_word b);
__host__ __device__ bigint* bigint_sub_word(bigint *dst, const bigint *src_a, bigint_word b);

__host__ __device__ char* bigint_write_base(
    char *dst,
    int *n_dst,
    const bigint *a,
    bigint_word base,
    int zero_terminate
);

/* convenience function defaults to base 10 and zero terminates */
__host__ __device__ char* bigint_write(char *dst, int n_dst, const bigint *a);

__host__ __device__ bigint* bigint_shift_left (bigint *dst, const bigint *src, unsigned shift);
__host__ __device__ bigint* bigint_shift_right(bigint *dst, const bigint *src, unsigned shift);

__host__ __device__ int bigint_bitlength(const bigint *a);
__host__ __device__ int bigint_count_trailing_zeros(const bigint *a);

__host__ __device__ bigint* bigint_div_mod(
    bigint *dst_quotient,
    bigint *dst_remainder,
    const bigint *src_biginterator,
    const bigint *src_denominator
);

__host__ __device__ bigint* bigint_div(
    bigint *dst,
    const bigint *numerator,
    const bigint *denominator
);

__host__ __device__ bigint* bigint_mod(
    bigint *dst,
    const bigint *numerator,
    const bigint *denominator
);

__host__ __device__ bigint* bigint_div_mod_half_word(
    bigint *dst,
    bigint_word *dst_remainder,
    bigint_word denominator
);

__host__ __device__ bigint* bigint_gcd(bigint *dst, const bigint *src_a, const bigint *src_b);
//__host__ __device__ bigint* bigint_sqrt(bigint *dst, const bigint *src);

__host__ __device__ bigint* bigint_rand_bits(bigint *dst, int n_bits, bigint_rand_func rand_func);
__host__ __device__ bigint* bigint_rand_inclusive(bigint *dst, const bigint *n, bigint_rand_func rand_func);
__host__ __device__ bigint* bigint_rand_exclusive(bigint *dst, const bigint *n, bigint_rand_func rand_func);

__host__ __device__ bigint* bigint_pow_mod(
    bigint *dst,
    const bigint *src_base,
    const bigint *src_exponent,
    const bigint *src_modulus
);

/* probability for wrong positives is approximately 1/4^n_tests */
__host__ __device__ int bigint_is_probable_prime(const bigint *n, int n_tests, bigint_rand_func rand_func);

__host__ __device__ bigint* bigint_pow_word(bigint *dst, const bigint *src, bigint_word exponent);

__host__ __device__ double bigint_double(const bigint *src);

__host__ __device__ uint8_t* bigint_to_bytes(uint8_t *dst, const bigint *src, size_t len);


struct BigInt {
    bigint data[1];

    BigInt(){
        bigint_init(data);
    }

    BigInt(int b){
        bigint_init(data);
        bigint_from_int(data, b);
    }

    BigInt(const char *s, int base = 10){
        bigint_init(data);
        bigint_from_str_base(data, s, base);
    }

    BigInt(const BigInt &b){
        bigint_init(data);
        bigint_cpy(data, b.data);
    }

    BigInt& operator = (const BigInt &b){
        bigint_cpy(data, b.data);
        return *this;
    }

    ~BigInt(){
        bigint_free(data);
    }

    static void div_mod(
        BigInt &quotient, BigInt &remainder,
        const BigInt &biginterator,
        const BigInt &denominator
    ){
        bigint_div_mod(quotient.data, remainder.data,
            biginterator.data, denominator.data);
    }

    void write(
        char *dst,
        int *n_dst,
        int dst_base = 10,
        int zero_terminate = 1
    ) const {
        bigint_write_base(dst, n_dst, data, dst_base, zero_terminate);
    }

    template <class STREAM>
    STREAM& write(STREAM &s, int dst_base = 10) const {
        int n = bigint_write_size(data, dst_base);
        char *buf = (char*)malloc(n);
        write(buf, &n, dst_base);
        s << buf;
        free(buf);
        return s;
    }

    BigInt& operator <<= (int shift){
        bigint_shift_left(data, data, shift);
        return *this;
    }

    BigInt& operator >>= (int shift){
        bigint_shift_right(data, data, shift);
        return *this;
    }

    BigInt& operator += (const BigInt &b){
        bigint_add(data, data, b.data);
        return *this;
    }

    BigInt& operator -= (const BigInt &b){
        bigint_sub(data, data, b.data);
        return *this;
    }

    BigInt& operator *= (const BigInt &b){
        bigint_mul(data, data, b.data);
        return *this;
    }

    BigInt& operator /= (const BigInt &b){
        bigint_div(data, data, b.data);
        return *this;
    }

    BigInt& operator %= (const BigInt &b){
        bigint_mod(data, data, b.data);
        return *this;
    }

    BigInt& operator ++ (){
        bigint_add_word(data, data, 1);
        return *this;
    }

    BigInt& operator -- (){
        bigint_sub_word(data, data, 1);
        return *this;
    }

    BigInt& set_bit(int bit_index){
        bigint_set_bit(data, bit_index);
        return *this;
    }

    BigInt& clr_bit(int bit_index){
        bigint_clr_bit(data, bit_index);
        return *this;
    }

    bigint_word get_bit(int bit_index) const {
        return bigint_get_bit(data, bit_index);
    }

    int bitlength() const {
        return bigint_bitlength(data);
    }

    int count_trailing_zeros() const {
        return bigint_count_trailing_zeros(data);
    }

    bool is_probable_prime(int n_tests, bigint_rand_func rand_func) const {
        return bigint_is_probable_prime(data, n_tests, rand_func);
    }

    BigInt sqrt() const {
        BigInt b;
        //bigint_sqrt(b.data, data);
        return b;
    }

    BigInt pow(bigint_word exponent){
        BigInt b;
        bigint_pow_word(b.data, data, exponent);
        return b;
    }

    static BigInt gcd(const BigInt &a, const BigInt &b){
        BigInt c;
        bigint_gcd(c.data, a.data, b.data);
        return c;
    }

    static BigInt rand_bits(int n_bits, bigint_rand_func rand_func){
        BigInt b;
        bigint_rand_bits(b.data, n_bits, rand_func);
        return b;
    }

    static BigInt rand_inclusive(const BigInt &n, bigint_rand_func rand_func){
        BigInt b;
        bigint_rand_inclusive(b.data, n.data, rand_func);
        return b;
    }

    static BigInt rand_exclusive(const BigInt &n, bigint_rand_func rand_func){
        BigInt b;
        bigint_rand_exclusive(b.data, n.data, rand_func);
        return b;
    }
};

inline BigInt operator -(const BigInt &a){
    BigInt b(a);
    b.data->neg = !b.data->neg;
    return b;
}

inline BigInt operator + (const BigInt &a, const BigInt &b){
    return BigInt(a) += b;
}

inline BigInt operator - (const BigInt &a, const BigInt &b){
    return BigInt(a) -= b;
}

inline BigInt operator * (const BigInt &a, const BigInt &b){
    return BigInt(a) *= b;
}

inline BigInt operator / (const BigInt &a, const BigInt &b){
    return BigInt(a) /= b;
}

inline BigInt operator % (const BigInt &a, const BigInt &b){
    return BigInt(a) %= b;
}

inline BigInt operator << (const BigInt &a, int shift){
    return BigInt(a) <<= shift;
}

inline BigInt operator >> (const BigInt &a, int shift){
    return BigInt(a) >>= shift;
}

inline bool operator == (const BigInt &a, const BigInt &b){ return bigint_cmp(a.data, b.data) == 0; }
inline bool operator != (const BigInt &a, const BigInt &b){ return bigint_cmp(a.data, b.data) != 0; }
inline bool operator <= (const BigInt &a, const BigInt &b){ return bigint_cmp(a.data, b.data) <= 0; }
inline bool operator >= (const BigInt &a, const BigInt &b){ return bigint_cmp(a.data, b.data) >= 0; }
inline bool operator <  (const BigInt &a, const BigInt &b){ return bigint_cmp(a.data, b.data) <  0; }
inline bool operator >  (const BigInt &a, const BigInt &b){ return bigint_cmp(a.data, b.data) >  0; }