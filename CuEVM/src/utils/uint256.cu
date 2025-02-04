#include <CuEVM/utils/uint256.cuh>

__host__ __device__ int uint256_cmp(const uint256 *a, const uint256 *b) {
    if (a == nullptr) return !uint256_is_zero(b);
    if (b == nullptr) return !uint256_is_zero(a);
#pragma unroll
    for (int i = UINT256_WORDS - 1; i >= 0; i--) {
        if (a->words[i] < b->words[i]) return -1;
        if (a->words[i] > b->words[i]) return 1;
    }
    return 0;
}

__host__ __device__ int uint512_cmp(const uint512 *a, const uint512 *b) {
#pragma unroll
    for (int i = UINT256_WORDS * 2 - 1; i >= 0; i--) {
        if (a->words[i] < b->words[i]) return -1;
        if (a->words[i] > b->words[i]) return 1;
    }
    return 0;
}

__host__ __device__ int uint256_signed_cmp(const uint256 *a, const uint256 *b) {
    bool asign = a->words[UINT256_WORDS - 1] & 0x80000000;
    bool bsign = b->words[UINT256_WORDS - 1] & 0x80000000;

    if (asign && !bsign) {
        return -1;
    }
    if (!asign && bsign) {
        return 1;
    }
    return uint256_cmp(a, b);
}

__host__ __device__ int uint256_cmp_word(const uint256 *a, uint32_t b) {
    for (int i = UINT256_WORDS - 1; i >= 0; i--) {
        if (i != 0 && a->words[i] != 0) return 1;
        if (i == 0) {
            if (a->words[i] < b) return -1;
            if (a->words[i] > b) return 1;
        }
    }
    return 0;
}

__host__ __device__ bool uint256_is_zero(const uint256 *a) {
    if (a == nullptr) return true;
    for (int i = 0; i < UINT256_WORDS; i++) {
        if (a->words[i] != 0) return false;
    }
    return true;
}

__host__ __device__ int uint256_set_zero(uint256 *a) {
    memset(a->words, 0, sizeof(a->words));
    return 0;
}

__host__ __device__ uint32_t uint256_get_uint32_t(const uint256 *a) { return a->words[0]; }

__host__ __device__ uint64_t uint256_get_uint64_t(const uint256 *a) {
    return (uint64_t)a->words[1] << 32 | a->words[0];
}

__host__ __device__ uint256 *uint256_cpy(uint256 *dst, const uint256 *src) {
    // memcpy(dst->words, src->words, sizeof(dst->words));
#pragma unroll
    for (int i = 0; i < UINT256_WORDS; i++) {
        dst->words[i] = src->words[i];
    }
    return dst;
}

__host__ __device__ uint32_t uint256_bitlength(const uint256 *num) {
    // printf("bit length\n");
    for (int i = UINT256_WORDS - 1; i >= 0; --i) {
        uint32_t word = num->words[i];
        // printf("word %d: %08x\n", i, word);
        if (word != 0) {
            // Found the first non-zero word
            int bits = 32 * (i);
            // printf("bits: %d\n", bits);
            // Now find the highest bit set in word
            for (int j = 31; j >= 0; --j) {
                if (word & (1U << j)) {
                    return bits + j + 1;
                }
            }
        }
    }
    return 0;
}

// Return the bit-length of a 512-bit number.
__host__ __device__ int uint512_bitlength(const uint512 *num) {
    for (int i = UINT256_WORDS * 2 - 1; i >= 0; --i) {
        uint32_t word = num->words[i];
        if (word != 0) {
            int bits = 32 * i;
            for (int j = 31; j >= 0; --j) {
                if (word & (1U << j)) {
                    return bits + j + 1;
                }
            }
        }
    }
    return 0;
}

__host__ __device__ void uint256_set_bit(uint256 *num, int bit_index, uint8_t value) {
    int word_index = bit_index / 32;
    int bit_in_word = (bit_index % 32);  // Big-endian bit indexing
    if (value) {
        num->words[word_index] |= (1U << bit_in_word);
    } else {
        num->words[word_index] &= ~(1U << bit_in_word);
    }
}

// Set a particular bit in a 512-bit number.
__host__ __device__ uint512 *uint512_set_bit(uint512 *num, int bit_index, uint8_t value) {
    int word_index = bit_index / 32;
    int bit_in_word = bit_index % 32;
    if (value)
        num->words[word_index] |= (1U << bit_in_word);
    else
        num->words[word_index] &= ~(1U << bit_in_word);
    return num;
}
// __host__ __device__ uint256* uint256_clr_bit(uint256 *dst, unsigned bit_index) {
//     dst->words[bit_index / 32] &= ~(1 << (bit_index % 32));
//     return dst;
// }

// __host__ __device__ uint256* uint256_set_bit(uint256 *dst, unsigned bit_index) {
//     dst->words[bit_index / 32] |= (1 << (bit_index % 32));
//     return dst;
// }

// TODO: check if this is correct
__host__ __device__ uint256 *uint256_mul(uint256 *dst, const uint256 *a, const uint256 *b) {
    uint256 tmp;
    memset(tmp.words, 0, sizeof(tmp.words));  // Initialize dst to zero
    for (int i = 0; i < UINT256_WORDS; i++) {
        uint32_t carry = 0;
        for (int j = 0; j < UINT256_WORDS; j++) {
            if (i + j < UINT256_WORDS) {
                uint64_t product = (uint64_t)a->words[i] * b->words[j] + tmp.words[i + j] + carry;
                tmp.words[i + j] = (uint32_t)product;  // Store the lower part
                carry = product >> 32;                 // Carry the upper part
            }
        }
        // If there's a carry left and space in the array, add it
        if (i + UINT256_WORDS < UINT256_WORDS) {
            tmp.words[i + UINT256_WORDS] += carry;
        }
    }
    uint256_cpy(dst, &tmp);
    return dst;
}

__host__ __device__ uint512 *uint256_mul_wide(uint512 *dst, const uint256 *a, const uint256 *b) {
    uint512 tmp;
    // Zero out the wide result buffer (assumed to have 2*UINT256_WORDS words)
    memset(tmp.words, 0, sizeof(tmp.words));

    // Multiply each word of 'a' with each word of 'b'
    for (int i = 0; i < UINT256_WORDS; i++) {
        uint32_t carry = 0;
        for (int j = 0; j < UINT256_WORDS; j++) {
            int index = i + j;
            // Multiply the words, add what's already in tmp and the previous carry.
            uint64_t product = (uint64_t)a->words[i] * b->words[j] + tmp.words[index] + carry;
            tmp.words[index] = (uint32_t)product;  // lower 32 bits
            carry = product >> 32;                 // upper 32 bits as carry
        }
        // After processing row i, add the final carry to the next word.
        tmp.words[i + UINT256_WORDS] += carry;
    }

    // Copy the wide result into the destination uint512.
    memcpy(dst->words, tmp.words, sizeof(tmp.words));
    return dst;
}
__host__ __device__ uint256 *uint256_from_hex(uint256 *dst, const char *src) {
    size_t len = strlen(src);
    memset(dst->words, 0, sizeof(dst->words));

    // Check for "0x" or "0X" prefix and adjust the starting point
    size_t start = 0;
    if (len > 1 && src[0] == '0' && (src[1] == 'x' || src[1] == 'X')) {
        start = 2;
        len -= 2;
    }

    // Process each character from the start to the end for big-endian
    for (size_t i = 0; i < len; i++) {
        char c = src[len - 1 - i + start];
        uint32_t value;
        if (c >= '0' && c <= '9') {
            value = c - '0';
        } else if (c >= 'a' && c <= 'f') {
            value = 10 + (c - 'a');
        } else if (c >= 'A' && c <= 'F') {
            value = 10 + (c - 'A');
        } else {
            continue;  // skip invalid characters
        }
        size_t word_index = (i / (2 * sizeof(uint32_t)));
        size_t bit_position = (4 * (i % (2 * sizeof(uint32_t))));
        dst->words[word_index] |= value << bit_position;
    }
    return dst;
}

__host__ __device__ void print_uint256(const uint256 *a) {
    for (int i = 0; i < UINT256_WORDS; i++) {
        printf("%08x ", a->words[UINT256_WORDS - 1 - i]);
    }
    printf("\n");
}

__host__ __device__ void print_uint512(const uint512 *a) {
    for (int i = 0; i < UINT256_WORDS * 2; i++) {
        printf("%08x ", a->words[UINT256_WORDS * 2 - 1 - i]);
    }
    printf("\n");
}

// __host__ __device__ void print_bigint(const bigint *a) {
//     for (int i = 0; i < a->size; i++) {
//         printf("%08x ", a->words[a->size - 1 - i]);
//     }
//     printf("\n");
// }

__host__ __device__ uint256 *uint256_from_uint32(uint256 *dst, uint32_t src) {
    memset(dst->words, 0, sizeof(dst->words));
    dst->words[0] = src;
    return dst;
}

__host__ __device__ uint256 *uint256_from_word(uint256 *dst, uint32_t a) {
    memset(dst->words, 0, sizeof(dst->words));
    dst->words[0] = a;
    return dst;
}

__host__ __device__ uint256 *uint256_add(uint256 *dst, const uint256 *a, const uint256 *b) {
    uint32_t carry = 0;
    for (int i = 0; i < UINT256_WORDS; i++) {
        uint32_t sum = a->words[i] + b->words[i] + carry;
        carry = (sum < a->words[i] || (carry && sum == a->words[i])) ? 1 : 0;
        dst->words[i] = sum;
    }
    return dst;
}

// Improved implementation: modular addition without using uint512 arithmetic
// This version assumes that a and b have been reduced (i.e. < N)
// and uses the fact that (a + b) < 2N.
// If there is an overflow (carry), then (a + b) represents a 257-bit number
// whose value is (a+b mod 2^256) + 2^256. Since N < 2^256 we have:
//     2^256 mod N = 2^256 - N
// and the final result is then given by:
//     result = (a+b mod 2^256) + (carry ? (2^256 - N) : 0)
// followed by a single conditional subtraction if the result is >= N.
__host__ __device__ uint256 *uint256_addmod(uint256 *dst, const uint256 *a, const uint256 *b, const uint256 *N) {
    uint256 tmp_a;
    uint256 tmp_b;
    uint256 tmp;
    uint256_div_mod(&tmp, &tmp_a, a, N);
    uint256_div_mod(&tmp, &tmp_b, b, N);
    uint256 t;
    // Compute t = N - b
    uint256_sub(&t, N, b);
    if (uint256_cmp(&tmp_a, &t) >= 0) {
        uint256_sub(dst, &tmp_a, &t);
        return dst;
    }
    return uint256_add(dst, &tmp_a, &tmp_b);
}

// func SubMod(a, b, m uint64) uint64 {
//     a %= m
//     b %= m
//     if a < b {
//         return a + (m - b)
//     }
//     return a - b
// }

__host__ __device__ uint256 *uint256_submod(uint256 *dst, const uint256 *a, const uint256 *b, const uint256 *N) {
    uint256 tmp_a;
    uint256 tmp_b;
    uint256 tmp;
    uint256_div_mod(&tmp, &tmp_a, a, N);
    uint256_div_mod(&tmp, &tmp_b, b, N);

    if (uint256_cmp(&tmp_a, &tmp_b) < 0) {
        uint256_sub(&tmp, N, &tmp_b);
        return uint256_add(dst, &tmp_a, &tmp);
    }
    return uint256_sub(dst, &tmp_a, &tmp_b);
}

__host__ __device__ uint256 *uint256_powmod(uint256 *dst, const uint256 *base, const uint256 *exponent,
                                            const uint256 *N) {
    uint256 result, base_copy;
    uint256_set_zero(&result);
    result.words[0] = 1;  // Initialize result to 1
    uint256_cpy(&base_copy, base);
    uint32_t exponent_bit_length = uint256_bitlength(exponent);
    // uint32_t exponent_byte_length = (exponent_bit_length + 7) / 8;
    uint32_t exponent_word_length = (exponent_bit_length + 31) / 32;
    // printf("exponent_bit_length: %d\n", exponent_bit_length);
    // printf("exponent_byte_length: %d\n", exponent_word_length);
    for (int i = 0; i < exponent_word_length; i++) {
        for (int j = 0; j < 32; j++) {
            if ((exponent->words[i] >> j) & 1) {
                uint256_mulmod(&result, &result, &base_copy, N);
                // printf("\nresult ");
                // print_uint256(&result);
            }

            uint256_mulmod(&base_copy, &base_copy, &base_copy, N);
            // printf("double base_copy\n");
            // print_uint256(&base_copy);
        }
    }
    // todo : optimize
    uint256_cpy(dst, &result);
    return dst;
}

__host__ __device__ uint256 *uint256_mulmod(uint256 *dst, const uint256 *a, const uint256 *b, const uint256 *N) {
    uint512 mul_res;
    uint256_mul_wide(&mul_res, a, b);
    uint256 tmp;
    uint512_div_mod(&tmp, dst, &mul_res, N);
    return dst;
    /*bigint mul_res, big_N;
    if (uint256_is_zero(N)) {
        uint256_set_zero(dst);
        return dst;
    }
    bigint_init(&mul_res);
    bigint_reserve(&mul_res, UINT256_WORDS * 2);
    mul_res.size = UINT256_WORDS * 2;
    bigint big_a, big_b;
    bigint_from_uint256(&big_a, a, UINT256_WORDS * 2);
    bigint_from_uint256(&big_b, b, UINT256_WORDS * 2);
    bigint_mul(&mul_res, &big_a, &big_b);
    mul_res.size = UINT256_WORDS * 2;

    bigint_from_uint256(&big_N, N, UINT256_WORDS * 2);
    bigint_mod(&mul_res, &mul_res, &big_N);
    uint256_from_bigint(dst, &mul_res);
    return dst;*/
}

__host__ __device__ uint256 *uint256_sub(uint256 *dst, const uint256 *a, const uint256 *b) {
    uint32_t borrow = 0;
    for (int i = 0; i < UINT256_WORDS; i++) {
        uint64_t diff = (uint64_t)a->words[i] - (uint64_t)b->words[i] - borrow;
        borrow = (diff >> 32) & 1;
        dst->words[i] = diff;
    }
    return dst;
}
// Subtract two 512-bit numbers (a - b). Assumes a >= b.
__host__ __device__ uint512 *uint512_sub(uint512 *dst, const uint512 *a, const uint512 *b) {
    uint32_t borrow = 0;
    for (int i = 0; i < UINT256_WORDS * 2; i++) {
        uint64_t diff = (uint64_t)a->words[i] - (uint64_t)b->words[i] - borrow;
        borrow = (diff >> 32) & 1;
        dst->words[i] = (uint32_t)diff;
    }
    return dst;
}

// __host__ __device__ uint256* uint256_mul(uint256 *dst, const uint256 *a, const uint256 *b) {
//     uint256 tmp;
//     memset(dst->words, 0, sizeof(dst->words));
//     for (int i = 0; i < UINT256_WORDS; i++) {
//         if (a->words[i] == 0) continue;
//         memset(tmp.words, 0, sizeof(tmp.words));
//         uint32_t carry = 0;
//         for (int j = 0; j < UINT256_WORDS - i; j++) {
//             uint32_t product = a->words[i] * b->words[j] + carry;
//             carry = product >> uint32_t_BITS;
//             tmp.words[i + j] = product;
//         }
//         uint256_add(dst, dst, &tmp);
//     }
//     return dst;
// }

/*
__host__ __device__ uint256* uint256_div_mod(uint256 *dst_quotient, uint256 *dst_remainder,
                                             const uint256 *src_numerator, const uint256 *src_denominator) {
    uint256 quotient;
    memset(quotient.words, 0, sizeof(quotient.words));
    uint256 remainder;
    uint256 denominator;
    uint256_cpy(&remainder, src_numerator);
    uint256_cpy(&denominator, src_denominator);

    // Check for division by zero
    if (uint256_is_zero(&denominator)) {
        memset(dst_quotient->words, 0, sizeof(dst_quotient->words));
        memset(dst_remainder->words, 0, sizeof(dst_remainder->words));
        return NULL;
    }

    // If numerator < denominator, quotient is 0, remainder is numerator
    if (uint256_cmp(&remainder, &denominator) < 0) {
        *dst_quotient = quotient;
        *dst_remainder = remainder;
        return dst_quotient;
    }
    printf("start division\n");
    printf("remainder: ");
    print_uint256(&remainder);
    printf("quotient: ");
    print_uint256(&quotient);
    printf("\n-----------------------\n");
    // Main division loop using repeated subtraction
    while (uint256_cmp(&remainder, &denominator) >= 0) {
        printf("remainder: ");
        print_uint256(&remainder);
        printf("denominator: ");
        print_uint256(&denominator);
        uint256_sub(&remainder, &remainder, &denominator);
        uint256_add_word(&quotient, &quotient, 1);
        printf("quotient: ");
        print_uint256(&quotient);
        printf("remainder: ");
        print_uint256(&remainder);
        printf("\n------------------\n");
    }

    *dst_quotient = quotient;
    *dst_remainder = remainder;
    return dst_quotient;
}
*/
__host__ __device__ uint256 *uint512_div_mod(uint256 *dst_quotient, uint256 *dst_remainder, const uint512 *src_dividend,
                                             const uint256 *src_divisor) {
    // Check for division by zero
    if (uint256_is_zero(src_divisor)) {
        memset(dst_quotient->words, 0, sizeof(dst_quotient->words));
        memset(dst_remainder->words, 0, sizeof(dst_remainder->words));

        return dst_quotient;  // Division by zero.
    }

    // Copy the 512-bit dividend locally.
    uint512 numerator;
    memcpy(numerator.words, src_dividend->words, sizeof(src_dividend->words));

    // Convert the uint256 divisor into a uint512 (its high half will be zero).
    uint512 divisor512;
    memset(divisor512.words, 0, sizeof(divisor512.words));
    memcpy(divisor512.words, src_divisor->words, sizeof(src_divisor->words));

    // If the numerator is less than the divisor, the quotient is zero.
    if (uint512_cmp(&numerator, &divisor512) < 0) {
        // printf("numerator < divisor\n");
        memcpy(dst_remainder->words, numerator.words, sizeof(dst_remainder->words));
        return dst_quotient;
    }

    // Determine how many bits to shift. (The bit-length functions work similarly
    // to the uint256 versions but over twice the number of words.)
    int nbits = uint512_bitlength(&numerator);
    int dbits = uint512_bitlength(&divisor512);  // Should match uint256_bitlength(src_divisor).
    int shift = nbits - dbits;
    // printf("nbits: %d, dbits: %d, shift: %d\n", nbits, dbits, shift);
    // Prepare the quotient (we use a full 512-bit temporary; the quotient should fit in 256 bits).
    uint512 quotient;
    memset(quotient.words, 0, sizeof(quotient.words));

    // Align the divisor with the numerator.
    uint512 shifted_divisor;
    memcpy(shifted_divisor.words, divisor512.words, sizeof(shifted_divisor.words));
    uint512_shift_left(&shifted_divisor, &shifted_divisor, shift);

    // Perform the long-division loop.
    for (; shift >= 0; shift--) {
        if (uint512_cmp(&numerator, &shifted_divisor) >= 0) {
            uint512 temp;
            uint512_sub(&temp, &numerator, &shifted_divisor);
            memcpy(numerator.words, temp.words, sizeof(numerator.words));
            uint512_set_bit(&quotient, shift, 1);
        }
        uint512_shift_right(&shifted_divisor, &shifted_divisor, 1);
    }

    // Copy the resulting quotient (which should fit in 256 bits) and remainder.
    memcpy(dst_quotient->words, quotient.words, sizeof(dst_quotient->words));
    memcpy(dst_remainder->words, numerator.words, sizeof(dst_remainder->words));

    return dst_quotient;
}

__host__ __device__ uint256 *uint256_div_mod(uint256 *dst_quotient, uint256 *dst_remainder,
                                             const uint256 *src_numerator, const uint256 *src_denominator) {
    uint256 quotient;
    memset(quotient.words, 0, sizeof(quotient.words));
    uint256 numerator;
    uint256 denominator;
    uint256_cpy(&numerator, src_numerator);
    uint256_cpy(&denominator, src_denominator);

    // Check for division by zero
    if (uint256_is_zero(&denominator)) {
        memset(dst_quotient->words, 0, sizeof(dst_quotient->words));
        memset(dst_remainder->words, 0, sizeof(dst_remainder->words));
        return NULL;  // Division by zero
    }

    // If numerator < denominator, quotient is 0, remainder is numerator
    if (uint256_cmp(&numerator, &denominator) < 0) {
        *dst_quotient = quotient;    // Quotient is zero
        *dst_remainder = numerator;  // Remainder is the numerator
        return dst_quotient;
    }

    // Determine the shift required to align the numerator and denominator
    uint32_t nbits_numerator = uint256_bitlength(&numerator);

    uint32_t nbits_denominator = uint256_bitlength(&denominator);

    int shift = nbits_numerator - nbits_denominator;
    // return NULL;
    uint256 shifted_denominator;
    memset(shifted_denominator.words, 0, sizeof(shifted_denominator.words));

    uint256_shift_left(&shifted_denominator, &denominator, shift);

    for (; shift >= 0; shift--) {
        if (uint256_cmp(&numerator, &shifted_denominator) >= 0) {
            uint256_sub(&numerator, &numerator, &shifted_denominator);
            uint256_set_bit(&quotient, shift, 1);
        }
        uint256_shift_right(&shifted_denominator, &shifted_denominator, 1);
    }

    uint256_cpy(dst_quotient, &quotient);
    uint256_cpy(dst_remainder, &numerator);
    return dst_quotient;
}

/**
 * Compute the remainder when a 512-bit number (src_biginterator)
 * is divided by a 256-bit number (src_denominator).
 *
 * The algorithm processes the 512-bit dividend as an array of
 * 2*UINT256_WORDS 32-bit words (most significant digit first)
 * according to the standard digit-by-digit reduction.
 *
 * Returns dst_remainder = src_biginterator mod src_denominator.
 * If division by zero is attempted, dst_remainder is zeroed and NULL is returned.
 */
__host__ __device__ uint256 *uint512_mod(uint256 *dst_remainder, const uint512 *src_biginterator,
                                         const uint256 *src_denominator) {
    // Check for division by zero.
    if (uint256_is_zero(src_denominator)) {
        memset(dst_remainder->words, 0, sizeof(dst_remainder->words));
        return NULL;
    }

    // Our working remainder (256-bit) starts at 0.
    uint256 remainder;
    uint256_set_zero(&remainder);

    // The total number of 32-bit digits in our 512-bit number.
    int total_digits = 2 * UINT256_WORDS;

    // Process each 32-bit “digit” from most-significant to least.
    // Note: The uint512 'words' array is little-endian,
    // so the most significant word is at index (total_digits-1).
    for (int i = total_digits - 1; i >= 0; i--) {
        // Multiply the current remainder by 2^32.
        uint256_shift_left(&remainder, &remainder, 32);

        // Add the current digit.
        // Since src_biginterator->words[i] is a 32-bit word,
        // we use uint256_add_word to add it.
        uint256_add_word(&remainder, &remainder, src_biginterator->words[i]);

        // Reduce the result modulo src_denominator.
        // In many cases the inner loop will run only once.
        // printf("remainder: ");
        // print_uint256(&remainder);
        // printf("src_denominator: ");
        // print_uint256(src_denominator);
        uint256 temp;
        if (uint256_cmp(&remainder, src_denominator) >= 0) {
            uint256_div(&temp, &remainder, src_denominator);
            uint256_mul(&temp, &temp, src_denominator);
            uint256_sub(&remainder, &remainder, &temp);
        }
    }

    // Copy the final remainder to dst_remainder.
    uint256_cpy(dst_remainder, &remainder);
    return dst_remainder;
}
// __host__ __device__ bigint *bigint_from_uint256(bigint *dst, const uint256 *src, const uint32_t word_size) {
//     bigint_init(dst);
//     bigint_reserve(dst, word_size);
//     dst->size = word_size;
//     memset(dst->words, 0, word_size * sizeof(uint32_t));
//     memcpy(dst->words, src->words, UINT256_WORDS * sizeof(uint32_t));
//     return dst;
// }

// __host__ __device__ uint256 *uint256_from_bigint(uint256 *dst, const bigint *src) {
//     memset(dst->words, 0, sizeof(dst->words));
//     memcpy(dst->words, src->words, UINT256_WORDS * sizeof(uint32_t));
//     return dst;
// }

__host__ __device__ uint256 *uint256_div(uint256 *dst, const uint256 *numerator, const uint256 *denominator) {
    uint256 remainder;
    return uint256_div_mod(dst, &remainder, numerator, denominator);
}

__host__ __device__ uint256 *uint256_mod(uint256 *dst, const uint256 *numerator, const uint256 *denominator) {
    uint256 quotient;
    return uint256_div_mod(&quotient, dst, numerator, denominator);
}

__host__ __device__ uint256 *uint256_negate(uint256 *dst, const uint256 *src) {
    uint256 zero;
    uint256_from_word(&zero, 0);
    uint256_sub(dst, &zero, src);
    return dst;
}

__host__ __device__ uint256 *uint256_signed_mod(uint256 *dst, const uint256 *numerator, const uint256 *denominator) {
    uint256 quotient;
    bool numerator_neg = numerator->words[UINT256_WORDS - 1] & 0x80000000;
    bool denominator_neg = denominator->words[UINT256_WORDS - 1] & 0x80000000;
    uint256 numerator_abs;
    uint256 denominator_abs;

    if (numerator_neg) {
        uint256_negate(&numerator_abs, numerator);
    } else {
        uint256_cpy(&numerator_abs, numerator);
    }
    if (denominator_neg) {
        uint256_negate(&denominator_abs, denominator);
    } else {
        uint256_cpy(&denominator_abs, denominator);
    }
    uint256_div_mod(&quotient, dst, &numerator_abs, &denominator_abs);
    if (numerator_neg) {
        uint256_negate(dst, dst);
    }
    return dst;
}

__host__ __device__ uint256 *uint256_signed_div(uint256 *dst, const uint256 *numerator, const uint256 *denominator) {
    uint256 remainder;
    bool numerator_neg = numerator->words[UINT256_WORDS - 1] & 0x80000000;
    bool denominator_neg = denominator->words[UINT256_WORDS - 1] & 0x80000000;
    uint256 numerator_abs;
    uint256 denominator_abs;

    if (numerator_neg) {
        uint256_negate(&numerator_abs, numerator);
    } else {
        uint256_cpy(&numerator_abs, numerator);
    }
    if (denominator_neg) {
        uint256_negate(&denominator_abs, denominator);
    } else {
        uint256_cpy(&denominator_abs, denominator);
    }
    uint256_div(dst, &numerator_abs, &denominator_abs);
    if (numerator_neg != denominator_neg) {
        uint256_negate(dst, dst);
    }
    return dst;
}

__host__ __device__ uint256 *uint256_sign_extension(uint256 *dst, const uint256 *src, const uint32_t byte_index) {
    if (byte_index >= UINT256_WORDS * sizeof(uint32_t)) {
        // If byte_index is out of bounds, just copy the source
        uint256_cpy(dst, src);
        return dst;
    }
    uint32_t word_index = byte_index / sizeof(uint32_t);
    uint32_t byte_in_word = byte_index % sizeof(uint32_t);
    uint32_t sign_bit = (src->words[word_index] >> (byte_in_word * 8 + 7)) & 1;
    printf("sign_bit: %d\n", sign_bit);
    if (sign_bit) {
        // If the sign bit is set, extend with 1s
        for (uint32_t i = word_index; i < UINT256_WORDS; i++) {
            if (i == word_index) {
                dst->words[i] = src->words[i] | (~0U << (byte_in_word * 8 + 8));
            } else {
                dst->words[i] = ~0U;
            }
        }
    } else {
        // If the sign bit is not set, copy the source (only from the byte_index to the end)
        for (uint32_t i = 0; i < word_index; i++) {
            dst->words[i] = 0;
        }
        dst->words[word_index] = src->words[word_index] & ((1U << (byte_in_word * 8 + 8)) - 1);
        for (uint32_t i = word_index + 1; i < UINT256_WORDS; i++) {
            dst->words[i] = src->words[i];
        }
    }
    return dst;
}
__host__ __device__ uint256 *uint256_exp(uint256 *dst, const uint256 *base, const uint256 *exponent) {
    uint256 result, base_copy;
    uint256_set_zero(&result);
    result.words[0] = 1;  // Initialize result to 1
    uint256_cpy(&base_copy, base);
    uint32_t exponent_bit_length = uint256_bitlength(exponent);
    uint32_t exponent_byte_length = (exponent_bit_length + 7) / 8;

    for (int i = 0; i < exponent_byte_length; i++) {
        for (int j = 0; j < 32; j++) {
            if ((exponent->words[i] >> j) & 1) {
                uint256_mul(&result, &result, &base_copy);
            }

            uint256_mul(&base_copy, &base_copy, &base_copy);
        }
    }
    // todo : optimize
    uint256_cpy(dst, &result);
    return dst;
}

__host__ __device__ uint256 *uint256_shift_left(uint256 *dst, const uint256 *src, uint32_t shift) {
    if (dst != src) uint256_cpy(dst, src);
    if (shift <= 0) return dst;

    while (shift >= 32) {
        for (int i = UINT256_WORDS - 1; i > 0; --i) dst->words[i] = dst->words[i - 1];
        dst->words[0] = 0;
        shift -= 32;
    }
    if (shift > 0) {
        uint32_t carry = 0;
        for (int i = 0; i < UINT256_WORDS; ++i) {
            uint32_t word = dst->words[i];
            dst->words[i] = (word << shift) | carry;
            carry = word >> (32 - shift);
        }
    }
    return dst;
}
// Shift left by a given number of bits.
__host__ __device__ uint512 *uint512_shift_left(uint512 *dst, const uint512 *src, uint32_t shift) {
    if (dst != src) memcpy(dst->words, src->words, sizeof(dst->words));
    if (shift == 0) return dst;

    // Shift by whole words first.
    while (shift >= 32) {
        for (int i = UINT256_WORDS * 2 - 1; i > 0; --i) dst->words[i] = dst->words[i - 1];
        dst->words[0] = 0;
        shift -= 32;
    }
    // Shift the remaining bits.
    if (shift > 0) {
        uint32_t carry = 0;
        for (int i = 0; i < UINT256_WORDS * 2; ++i) {
            uint32_t word = dst->words[i];
            dst->words[i] = (word << shift) | carry;
            carry = word >> (32 - shift);
        }
    }
    return dst;
}

__host__ __device__ uint256 *uint256_shift_right(uint256 *dst, const uint256 *src, uint32_t shift) {
    if (shift <= 0) return dst;
    uint8_t offset = shift / 32;
    for (int i = 0; i < UINT256_WORDS - offset; i++) {
        dst->words[i] = src->words[i + offset];
    }
    for (int i = UINT256_WORDS - offset; i < UINT256_WORDS; i++) {
        dst->words[i] = 0;
    }

    shift = shift % 32;
    if (shift > 0) {
        uint32_t carry = 0;
        for (int i = UINT256_WORDS - 1; i >= 0; --i) {
            uint32_t word = src->words[i];
            dst->words[i] = (word >> shift) | carry;
            carry = word << (32 - shift);
        }
    }
    return dst;
}
// Shift right by a given number of bits.
__host__ __device__ uint512 *uint512_shift_right(uint512 *dst, const uint512 *src, uint32_t shift) {
    if (dst != src) memcpy(dst->words, src->words, sizeof(dst->words));
    if (shift == 0) return dst;

    // Shift by whole words.
    uint8_t offset = shift / 32;
    for (int i = 0; i < UINT256_WORDS * 2 - offset; i++) {
        dst->words[i] = src->words[i + offset];
    }
    for (int i = UINT256_WORDS * 2 - offset; i < UINT256_WORDS * 2; i++) {
        dst->words[i] = 0;
    }
    // Shift the remaining bits.
    shift %= 32;
    if (shift > 0) {
        uint32_t carry = 0;
        for (int i = UINT256_WORDS * 2 - 1; i >= 0; --i) {
            uint32_t word = src->words[i];
            dst->words[i] = (word >> shift) | carry;
            carry = word << (32 - shift);
        }
    }
    return dst;
}

__host__ __device__ uint256 *uint256_shift_arithmetic_right(uint256 *dst, const uint256 *src, uint32_t shift) {
    if (shift <= 0) return dst;

    // Determine if the most significant bit is set
    bool msb_set = src->words[UINT256_WORDS - 1] & (1U << (32 - 1));

    while (shift >= 32) {
        for (int i = UINT256_WORDS - 1; i > 0; --i) {
            dst->words[i] = src->words[i - 1];
        }
        dst->words[0] = msb_set ? ~0U : 0;  // Fill with 1s if msb_set, else 0s
        shift -= 32;
    }

    if (shift > 0) {
        uint32_t carry = msb_set ? ~0U << (32 - shift) : 0;
        for (int i = UINT256_WORDS - 1; i >= 0; --i) {
            uint32_t word = src->words[i];
            dst->words[i] = (word >> shift) | carry;
            carry = word << (32 - shift);
        }
    }
    return dst;
}

__host__ __device__ uint256 *uint256_bitwise_and(uint256 *dst, const uint256 *a, const uint256 *b) {
    for (int i = 0; i < UINT256_WORDS; i++) {
        dst->words[i] = a->words[i] & b->words[i];
    }
    return dst;
}

__host__ __device__ uint256 *uint256_bitwise_or(uint256 *dst, const uint256 *a, const uint256 *b) {
    for (int i = 0; i < UINT256_WORDS; i++) {
        dst->words[i] = a->words[i] | b->words[i];
    }
    return dst;
}

__host__ __device__ uint256 *uint256_bitwise_xor(uint256 *dst, const uint256 *a, const uint256 *b) {
    for (int i = 0; i < UINT256_WORDS; i++) {
        dst->words[i] = a->words[i] ^ b->words[i];
    }
    return dst;
}

__host__ __device__ uint256 *uint256_bitwise_not(uint256 *dst, const uint256 *a) {
    for (int i = 0; i < UINT256_WORDS; i++) {
        dst->words[i] = ~a->words[i];
    }
    return dst;
}

__host__ __device__ uint8_t *uint256_to_bytes(uint8_t *dst, const uint256 *src, size_t len) {
    size_t total_bytes = sizeof(src->words);
    for (size_t i = 0; i < len && i < total_bytes; i++) {
        size_t word_index = (total_bytes - 1 - i) / sizeof(uint32_t);
        size_t byte_position = i % sizeof(uint32_t);
        dst[i] = (src->words[word_index] >> (8 * (sizeof(uint32_t) - 1 - byte_position))) & 0xFF;
    }
    return dst;
}

__host__ __device__ uint256 *uint256_from_bytes(uint256 *dst, const uint8_t *src, size_t len) {
    // Initialize the words array to zero
    // memset(dst->words, 0, sizeof(dst->words));

    memset(dst->words, 0, UINT256_BYTES);
    uint8_t offset = UINT256_BYTES - len;
    // Convert the byte array to the uint256 structure

    for (size_t i = 0; i < len && i < UINT256_BYTES; i++) {
        uint8_t word_index = UINT256_WORDS - 1 - (i + offset) / UINT256_LIMBS_BYTES;
        uint8_t byte_position = (i + offset) % UINT256_LIMBS_BYTES;
        dst->words[word_index] |= ((uint32_t)src[i]) << (8 * (UINT256_LIMBS_BYTES - 1 - byte_position));
    }
    // size_t offset = UINT256_BYTES - len;

    // for (size_t i = 0; i < UINT256_BYTES; i++) {
    //     if (i < len) {
    //         size_t word_index = UINT256_WORDS - 1 - (i + offset) / UINT256_LIMBS_BYTES;
    //         size_t byte_position = (i + offset) % UINT256_LIMBS_BYTES;
    //         dst->words[word_index] |= ((uint32_t)src[i]) << (8 * (UINT256_LIMBS_BYTES - 1 - byte_position));
    //     } else {
    //         dst->words[i] = 0;
    //     }
    // }

    return dst;
}
__host__ __device__ uint256 *uint256_extract_byte(uint256 *dst, const uint256 *src, uint32_t byte_index) {
    uint32_t word_index = byte_index / sizeof(uint32_t);
    uint32_t byte_position = byte_index % sizeof(uint32_t);
    printf("word_index: %d, byte_position: %d\n", word_index, byte_position);
    printf("src->words[word_index]: %u\n", src->words[word_index]);
    uint32_t mask = 0xff << (8 * byte_position);
    uint32_t extracted_byte = (src->words[word_index] & mask) >> (8 * byte_position);
    printf("extracted_byte: %u\n", extracted_byte);
    uint256_from_word(dst, extracted_byte);
    return dst;
}

// __host__ __device__ uint256* uint256_add_word(uint256 *dst, const uint256 *src_a, uint32_t b);
__host__ __device__ uint256 *uint256_add_word(uint256 *dst, const uint256 *src_a, uint32_t b) {
    uint256 tmp;
    uint256_from_word(&tmp, b);
    return uint256_add(dst, src_a, &tmp);
}

// __host__ __device__ uint256* uint256_sub_word(uint256 *dst, const uint256 *src_a, uint32_t b);
__host__ __device__ uint256 *uint256_sub_word(uint256 *dst, const uint256 *src_a, uint32_t b) {
    uint256 tmp;
    uint256_from_word(&tmp, b);
    return uint256_sub(dst, src_a, &tmp);
}

// __host__ __device__ char* uint256_write_base(
//     char *dst,
//     int *n_dst,
//     const uint256 *a,
//     uint32_t base,
//     int zero_terminate
// );
__host__ __device__ char *uint256_to_hex(char *dst, const uint256 *a) {
    int n = 0;
    static const char *table = "0123456789abcdef";

    for (int i = UINT256_WORDS - 1; i >= 0; i--) {
        for (int j = sizeof(uint32_t) * 2 - 1; j >= 0; j--) {
            uint32_t byte = (a->words[i] >> (j * 4)) & 0xF;
            dst[n++] = table[byte];
        }
    }

    dst[n] = '\0';
    return dst;
}
