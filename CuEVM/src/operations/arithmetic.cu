#include <CuEVM/gas_cost.cuh>
#include <CuEVM/operations/arithmetic.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM::operations {
__device__ int32_t ADD(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t r;
        if (stack->size() < 2) {
            return ERROR_STACK_UNDERFLOW;
        }
        // r = a + b;
        uint256_add(&r, stack->get_address_at_index(1), stack->get_address_at_index(2));
        stack->reduce_size(2);
        error_code |= stack->push(r);
    }
    return error_code;
}

__device__ int32_t MUL(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t a, b, r;
        error_code |= stack->pop(a);
        error_code |= stack->pop(b);

        // r = a * b;
        uint256_mul(&r, &a, &b);

        error_code |= stack->push(r);
    }
    return error_code;
}

__device__ int32_t SUB(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t a, b, r;
        error_code |= stack->pop(a);
        error_code |= stack->pop(b);

        // r = a - b;
        uint256_sub(&r, &a, &b);

        error_code |= stack->push(r);
    }
    return error_code;
}

__device__ int32_t DIV(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t a, b, r;
        error_code |= stack->pop(a);
        error_code |= stack->pop(b);
        if (b == 0) {
            stack->push_uint32(0);
            return ERROR_SUCCESS;
        }
        uint256_div(&r, &a, &b);

        error_code |= stack->push(r);
    }
    return error_code;
}

__device__ int32_t SDIV(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t a, b, r;
        error_code |= stack->pop(a);
        error_code |= stack->pop(b);
        if (b == 0) {
            stack->push_uint32(0);
            return ERROR_SUCCESS;
        }
        uint256_signed_div(&r, &a, &b);
        error_code |= stack->push(r);
    }

    return error_code;
}

__device__ int32_t MOD(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t a, b, r;
        error_code |= stack->pop(a);
        error_code |= stack->pop(b);

        // // rem by zero no error
        if (b == 0)
            r = 0;
        else
            uint256_mod(&r, &a, &b);

        error_code |= stack->push(r);
    }
    return error_code;
}

__device__ int32_t SMOD(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t a, b, r;
        error_code |= stack->pop(a);
        error_code |= stack->pop(b);

        uint256_signed_mod(&r, &a, &b);

        error_code |= stack->push(r);
    }

    return error_code;
}

__device__ int32_t ADDMOD(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_MID;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t a, b, N, r;
        error_code |= stack->pop(a);
        error_code |= stack->pop(b);
        error_code |= stack->pop(N);
        uint256_addmod(&r, &a, &b, &N);
        error_code |= stack->push(r);
    }
    return error_code;
}

__device__ int32_t MULMOD(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_MID;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t a, b, N, r;
        error_code |= stack->pop(a);
        error_code |= stack->pop(b);
        error_code |= stack->pop(N);
        uint256_mulmod(&r, &a, &b, &N);
        error_code |= stack->push(r);
    }
    return error_code;
}

__device__ int32_t EXP(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_EXP;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t a, exponent, r;
        error_code |= stack->pop(a);
        error_code |= stack->pop(exponent);

        if (error_code == ERROR_SUCCESS) {
            uint32_t exponent_bit_length = uint256_bitlength(&exponent);
            if (exponent_bit_length == 0) {
                error_code |= stack->push_uint32(1);
            } else {
                gas_used += GAS_EXP_BYTE * ((exponent_bit_length + 7) / 8);
                uint256_exp(&r, &a, &exponent);
                error_code |= stack->push(r);
            }
        }
    }
    return error_code;
}

__device__ int32_t SIGNEXTEND(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t b, x, r;
        error_code |= stack->pop(b);
        error_code |= stack->pop(x);
        if (uint256_cmp_word(&b, 30) > 0) {
            uint256_cpy(&r, &x);
        } else {
            uint256_sign_extension(&r, &x, uint256_get_uint32_t(&b));
        }
        error_code |= stack->push(r);
    }
    return error_code;
}
__device__ int32_t AND(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t a, b;
        error_code |= stack->pop(a);
        error_code |= stack->pop(b);
        evm_word_t r;

        uint256_bitwise_and(&r, &a, &b);

        error_code |= stack->push(r);
    }
    return error_code;
}
__device__ int32_t OR(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t a, b;
        error_code |= stack->pop(a);
        error_code |= stack->pop(b);
        evm_word_t r;

        uint256_bitwise_or(&r, &a, &b);

        error_code |= stack->push(r);
    }
    return error_code;
}

__device__ int32_t XOR(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t a, b;
        error_code |= stack->pop(a);
        error_code |= stack->pop(b);
        evm_word_t r;

        uint256_bitwise_xor(&r, &a, &b);

        error_code |= stack->push(r);
    }
    return error_code;
}

__device__ int32_t NOT(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t a;
        error_code |= stack->pop(a);
        evm_word_t r;

        uint256_bitwise_not(&r, &a);

        error_code |= stack->push(r);
    }
    return error_code;
}

__device__ int32_t BYTE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t i, x;
        error_code |= stack->pop(i);
        error_code |= stack->pop(x);
        evm_word_t r;
        printf("i: %u\n", uint256_get_uint32_t(&i));
        if (uint256_cmp_word(&i, UINT256_BYTES - 1) == 1) {
            r = 0;
        } else {
            uint256_extract_byte(&r, &x, UINT256_BYTES - 1 - uint256_get_uint32_t(&i));
        }
        error_code |= stack->push(r);
    }
    return error_code;
}

__device__ int32_t SHL(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t shift, value;
        error_code |= stack->pop(shift);
        error_code |= stack->pop(value);
        evm_word_t r;

        if (uint256_cmp_word(&shift, CuEVM::word_bits - 1) == 1) {
            r = 0;
        } else {
            uint32_t shift_left = uint256_get_uint32_t(&shift);
            uint256_shift_left(&r, &value, shift_left);
        }

        error_code |= stack->push(r);
    }
    return error_code;
}

__device__ int32_t SHR(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t shift, value;
        error_code |= stack->pop(shift);
        error_code |= stack->pop(value);
        evm_word_t r;

        if (uint256_cmp_word(&shift, CuEVM::word_bits - 1) == 1) {
            r = 0;
        } else {
            uint256_shift_right(&r, &value, shift.words[0]);
        }

        error_code |= stack->push(r);
    }
    return error_code;
}

__device__ int32_t SAR(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t shift, value;
        error_code |= stack->pop(shift);
        error_code |= stack->pop(value);
        evm_word_t r;
        if (uint256_cmp_word(&shift, UINT256_BITS - 1) == 1) {
            shift = UINT256_BITS;
        }
        uint32_t shift_right = uint256_get_uint32_t(&shift);
        uint256_shift_arithmetic_right(&r, &value, shift_right);

        error_code |= stack->push(r);
    }
    return error_code;
}

__device__ int32_t LT(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t a, b;
        error_code |= stack->pop(a);
        error_code |= stack->pop(b);

        error_code |= stack->push_uint32(uint256_cmp(&a, &b) < 0);
    }
    return error_code;
}

__device__ int32_t GT(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t a, b;
        error_code |= stack->pop(a);
        error_code |= stack->pop(b);

        error_code |= stack->push_uint32(uint256_cmp(&a, &b) > 0);
    }
    return error_code;
}

__device__ int32_t SLT(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t a, b;
        error_code |= stack->pop(a);
        error_code |= stack->pop(b);
        error_code |= stack->push_uint32(uint256_signed_cmp(&a, &b) < 0);
    }
    return error_code;
}

__device__ int32_t SGT(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t a, b;
        error_code |= stack->pop(a);
        error_code |= stack->pop(b);
        error_code |= stack->push_uint32(uint256_signed_cmp(&a, &b) > 0);
    }
    return error_code;
}

__device__ int32_t EQ(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t a, b;
        error_code |= stack->pop(a);
        error_code |= stack->pop(b);
        error_code |= stack->push_uint32(uint256_cmp(&a, &b) == 0);
    }
    return error_code;
}

__device__ int32_t ISZERO(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t a;
        error_code |= stack->pop(a);
        error_code |= stack->push_uint32(uint256_is_zero(&a));
    }
    return error_code;
}
}  // namespace CuEVM::operations
