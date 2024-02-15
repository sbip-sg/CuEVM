
#include "uint256.cuh"
// implementation
__host__ int hexToInt(const char *hex) {
    int result = 0;
    int len = strlen(hex);

    for (int i = 0; i < len; i++) {
        char c = tolower(hex[i]);
        if (c >= '0' && c <= '9') {
            result = result * 16 + (c - '0');
        } else if (c >= 'a' && c <= 'f') {
            result = result * 16 + (c - 'a' + 10);
        } else {
            // Invalid hexadecimal character
            return -1;
        }
    }
    return result;
}

__host__ void intToHex(int num, char *hex) {
    // Assuming hex has enough space
    char *ptr = hex;
    do {
        int remainder = num % 16;
        if (remainder < 10) {
            *ptr++ = '0' + remainder;
        } else {
            *ptr++ = 'a' + (remainder - 10);
        }
        num /= 16;
    } while (num != 0);

    *ptr-- = '\0';  // NULL-terminate the string and point to the last valid character

    // Reverse the string
    char *start = hex;
    while (start < ptr) {
        char t = *start;
        *start = *ptr;
        *ptr = t;
        start++;
        ptr--;
    }
}

__host__ bool hex_to_decimal(const char *hex_str, char *dec_str) {
    unsigned long long result = 0;
    unsigned long long place = 1;

    int len = strlen(hex_str);
    for (int i = len - 1; i >= 0; i--) {
        char c = tolower(hex_str[i]);
        int digit;
        if (c >= '0' && c <= '9') {
            digit = c - '0';
        } else if (c >= 'a' && c <= 'f') {
            digit = 10 + (c - 'a');
        } else {
            return false;
        }

        result += digit * place;
        place *= 16;
    }

    sprintf(dec_str, "%llu", result);
    return true;
}

__host__ bool base_uint_set_hex(base_uint *val, const char *hex) {
    memset(val->pn, 0, sizeof(val->pn));

    size_t len = strlen(hex);
    if (len == 0 || len > BITS / 4) return false;

    // Iterate through the string from end to start
    for (size_t i = 0; i < len; i++) {
        char c = tolower(hex[len - 1 - i]);
        uint32_t number = 0;

        if (c >= '0' && c <= '9') {
            number = c - '0';
        } else if (c >= 'a' && c <= 'f') {
            number = c - 'a' + 10;
        } else {
            return false;  // Invalid character
        }

        // Determine which uint32_t element and position the hex character should be placed
        val->pn[i / 8] |= (number << ((i % 8) * 4));
    }
    return true;
}

__host__ void base_uint_to_string(const base_uint *val, char *out_str) {
    char hex_str[BITS / 4 + 1] = {0};
    base_uint_get_hex(val, hex_str);

    if (!hex_to_decimal(hex_str, out_str)) {
        strcpy(out_str, "Error");
    }
}

__host__ bool int_to_base_uint(int int_val, base_uint *val) {
    char *p;
    sprintf(p, "%08x", int_val);
    printf("%s\n", p);
    return base_uint_set_hex(val, p);
}

__host__ __device__ void base_uint_get_hex(const base_uint *val, char *hex) {
    char *p = hex;

    for (int i = WIDTH - 1; i >= 0; i--) {
        // printf("%d ", val->pn[i]);
        sprintf(p, "%08x", val->pn[i]);
        p += 8;
    }
}

__host__ __device__ void print_base_uint(const base_uint *val) {
    for (int i = 0; i < WIDTH; i++) {
        printf("%d ", val->pn[i]);
    }
}

__host__ __device__ bool is_zero(const base_uint *num) {
    for (int i = 0; i < WIDTH; i++) {
        if (num->pn[i] != 0) {
            return false;
        }
    }
    return true;
}

__host__ __device__ base_uint bitwise_not(const base_uint *num) {
    base_uint ret;
    for (int i = 0; i < WIDTH; i++) {
        ret.pn[i] = ~num->pn[i];
    }
    return ret;
}

__host__ __device__ void base_uint_set_bit(base_uint *value, uint32_t bitpos) {
    value->pn[bitpos / 32] |= (1 << (bitpos % 32));
}

__host__ __device__ void base_uint_add(const base_uint *a, const base_uint *b, base_uint *result) {
    uint64_t carry = 0;

    for (size_t i = 0; i < WIDTH; i++) {
        uint64_t sum = (uint64_t)a->pn[i] + b->pn[i] + carry;
        printf("%d %d = %d %d\n", a->pn[i], b->pn[i], sum, carry);
        result->pn[i] = (uint32_t)sum;  // Store lower 32 bits
        carry = sum >> 32;              // Take upper 32 bits as the next carry
    }
}

__host__ __device__ bool base_uint_sub(const base_uint *a, const base_uint *b, base_uint *result) {
    uint64_t borrow = 0;

    for (size_t i = 0; i < WIDTH; i++) {
        uint64_t res = 0x100000000ULL + (uint64_t)a->pn[i] - b->pn[i] - borrow;
        result->pn[i] = (uint32_t)res;
        if (res >= 0x100000000ULL) {
            borrow = 0;
        } else {
            borrow = 1;
        }
    }

    // If borrow is still 1 after looping through all words, then a < b.
    // Return false to indicate underflow
    return borrow == 0;
}

/*
Warming:
1. Not tested yet.
2. Overflow wraparound is not correctly implemented yet.
*/
__host__ __device__ void base_uint_mul(const base_uint *a, const base_uint *b, base_uint *result) {
    base_uint temp_result = {0};
    for (size_t i = 0; i < WIDTH; i++) {
        uint64_t carry = 0;
        for (size_t j = 0; j < WIDTH; j++) {
            if (i + j < WIDTH) {
                uint64_t product = (uint64_t)a->pn[i] * b->pn[j] + temp_result.pn[i + j] + carry;
                temp_result.pn[i + j] = (uint32_t)product;
                carry = product >> 32;
            }
        }
    }

    for (size_t i = 0; i < WIDTH; i++) {
        result->pn[i] = temp_result.pn[i];
    }
}
