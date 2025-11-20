#include <CuEVM/gas_cost.cuh>
#include <CuEVM/operations/arithmetic.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM::operations {
__device__ int32_t ADD(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack
#ifdef BUILD_LIBRARY
                       ,
                       uint32_t pc, simplified_trace_data *simplified_trace_data_ptr
#endif
) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t r;
        if (stack->size() < 2) {
            return ERROR_STACK_UNDERFLOW;
        }
        // r = a + b;
#ifdef BUILD_LIBRARY
        bool overflow = uint256_add_overflow(&r, stack->get_address_at_index(1), stack->get_address_at_index(2));
        if (overflow) {
            simplified_trace_data_ptr->add_bugs_for_later(pc, BUG_INTEGER_ADD);
        }
#else
        uint256_add(&r, stack->get_address_at_index(1), stack->get_address_at_index(2));
#endif
        stack->reduce_size(2);
        error_code |= stack->push(r);
    }
    return error_code;
}

__device__ int32_t MUL(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack
#ifdef BUILD_LIBRARY
                       ,
                       uint32_t pc, simplified_trace_data *simplified_trace_data_ptr
#endif
) {
    gas_used += GAS_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t r;
        if (stack->size() < 2) {
            return ERROR_STACK_UNDERFLOW;
        }

        // r = a * b;

        uint256_mul(&r, stack->get_address_at_index(1), stack->get_address_at_index(2));
#ifdef BUILD_LIBRARY

        // r < a or r < b
        if (uint256_cmp(stack->get_address_at_index(1), &r) == 1 ||
            uint256_cmp(stack->get_address_at_index(2), &r) == 1) {
            if (!uint256_is_zero(&r)) simplified_trace_data_ptr->add_bugs_for_later(pc, BUG_INTEGER_MUL);
        }
#endif
        stack->reduce_size(2);
        error_code |= stack->push(r);
    }
    return error_code;
}

__device__ int32_t SUB(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack
#ifdef BUILD_LIBRARY
                       ,
                       uint32_t pc, simplified_trace_data *simplified_trace_data_ptr
#endif
) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t r;
        if (stack->size() < 2) {
            return ERROR_STACK_UNDERFLOW;
        }
        // error_code |= stack->pop(a);
        // error_code |= stack->pop(b);
#ifdef BUILD_LIBRARY
        bool underflow = uint256_sub_overflow(&r, stack->get_address_at_index(1), stack->get_address_at_index(2));
        if (underflow) {
            // simple reducing FP:the common case to calculate uint256 0xff... mask
            if (!(uint256_is_zero(stack->get_address_at_index(1)) && stack->get_address_at_index(2)->words[0] == 1)) {
                simplified_trace_data_ptr->add_bugs_for_later(pc, BUG_INTEGER_SUB);
            }
        }
#else
        // r = a - b;
        uint256_sub(&r, stack->get_address_at_index(1), stack->get_address_at_index(2));
#endif
        stack->reduce_size(2);
        error_code |= stack->push(r);
    }
    return error_code;
}

__device__ int32_t DIV(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t r;
        evm_word_t *a, *b;
        if (stack->size() < 2) {
            return ERROR_STACK_UNDERFLOW;
        }
        a = stack->get_address_at_index(1);
        b = stack->get_address_at_index(2);
        // printf("div a \n");
        // a.print();
        // printf("div b \n");
        // b.print();
        stack->reduce_size(2);
        if (b == 0) {
            stack->push_uint32(0);
            return ERROR_SUCCESS;
        }
        if (uint256_fast_div(&r, a, b)) {
            // printf("fast div\n");
        } else {
            // printf("slow div\n");
            uint256_div(&r, a, b);
        }
        // printf("div r \n");
        // r.print();

        error_code |= stack->push(r);
    }
    return error_code;
}

__device__ int32_t SDIV(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t r;
        evm_word_t *a, *b;
        if (stack->size() < 2) {
            return ERROR_STACK_UNDERFLOW;
        }
        a = stack->get_address_at_index(1);
        b = stack->get_address_at_index(2);
        stack->reduce_size(2);
        if (b == 0) {
            stack->push_uint32(0);
            return ERROR_SUCCESS;
        }
        uint256_signed_div(&r, a, b);
        error_code |= stack->push(r);
    }

    return error_code;
}

__device__ int32_t MOD(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t r;
        evm_word_t *a, *b;
        if (stack->size() < 2) {
            return ERROR_STACK_UNDERFLOW;
        }
        a = stack->get_address_at_index(1);
        b = stack->get_address_at_index(2);

        // // rem by zero no error
        if (b == 0)
            r = 0;
        else
            uint256_mod(&r, a, b);

        stack->reduce_size(2);
        error_code |= stack->push(r);
    }
    return error_code;
}

__device__ int32_t SMOD(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t r;
        evm_word_t *a, *b;
        if (stack->size() < 2) {
            return ERROR_STACK_UNDERFLOW;
        }
        a = stack->get_address_at_index(1);
        b = stack->get_address_at_index(2);

        uint256_signed_mod(&r, a, b);

        stack->reduce_size(2);
        error_code |= stack->push(r);
    }

    return error_code;
}

__device__ int32_t ADDMOD(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_MID;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        // evm_word_t a, b, N, r;
        // error_code |= stack->pop(a);
        // error_code |= stack->pop(b);
        // error_code |= stack->pop(N);
        evm_word_t r;

        if (stack->size() < 3) {
            return ERROR_STACK_UNDERFLOW;
        }

        uint256_addmod(&r, stack->get_address_at_index(1), stack->get_address_at_index(2),
                       stack->get_address_at_index(3));
        stack->reduce_size(3);
        error_code |= stack->push(r);
    }
    return error_code;
}

__device__ int32_t MULMOD(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_MID;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t r;
        evm_word_t *a, *b, *N;
        if (stack->size() < 3) {
            return ERROR_STACK_UNDERFLOW;
        }
        a = stack->get_address_at_index(1);
        b = stack->get_address_at_index(2);
        N = stack->get_address_at_index(3);
        uint256_mulmod(&r, a, b, N);
        stack->reduce_size(3);
        error_code |= stack->push(r);
    }
    return error_code;
}

__device__ int32_t EXP(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_EXP;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t r;
        evm_word_t *a, *exponent;
        if (stack->size() < 2) {
            return ERROR_STACK_UNDERFLOW;
        }
        a = stack->get_address_at_index(1);
        exponent = stack->get_address_at_index(2);
        // if (threadIdx.x == 1) {
        //     printf("exp a \n");
        //     a.print();
        //     printf("exp exponent \n");
        //     exponent.print();
        // }
        stack->reduce_size(2);
        if (error_code == ERROR_SUCCESS) {
            uint32_t exponent_bit_length = uint256_bitlength(exponent);
            if (exponent_bit_length == 0) {
                error_code |= stack->push_uint32(1);
            } else {
                gas_used += GAS_EXP_BYTE * ((exponent_bit_length + 7) / 8);
                if (uint256_fast_exp(&r, a, exponent)) {
                    error_code |= stack->push(r);
                    // printf("fast exp\n");
                } else {
                    uint256_exp(&r, a, exponent);
                    error_code |= stack->push(r);
                    // printf("slow exp\n");
                }
            }
        }
        // if (threadIdx.x == 1) {
        //     printf("exp r \n");
        //     r.print();
        // }
    }
    return error_code;
}

__device__ int32_t SIGNEXTEND(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t r;
        evm_word_t *b, *x;
        if (stack->size() < 2) {
            return ERROR_STACK_UNDERFLOW;
        }
        b = stack->get_address_at_index(1);
        x = stack->get_address_at_index(2);
        if (uint256_cmp_word(b, 30) > 0) {
            uint256_cpy(&r, x);
        } else {
            uint256_sign_extension(&r, x, uint256_get_uint32_t(b));
        }
        stack->reduce_size(2);
        error_code |= stack->push(r);
    }
    return error_code;
}
__device__ int32_t AND(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t r;

        if (stack->size() < 2) {
            return ERROR_STACK_UNDERFLOW;
        }

        uint256_bitwise_and(&r, stack->get_address_at_index(1), stack->get_address_at_index(2));
        stack->reduce_size(2);
        error_code |= stack->push(r);
    }
    return error_code;
}
__device__ int32_t OR(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t r;

        if (stack->size() < 2) {
            return ERROR_STACK_UNDERFLOW;
        }

        uint256_bitwise_or(&r, stack->get_address_at_index(1), stack->get_address_at_index(2));
        stack->reduce_size(2);
        error_code |= stack->push(r);
    }
    return error_code;
}

__device__ int32_t XOR(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t r;

        if (stack->size() < 2) {
            return ERROR_STACK_UNDERFLOW;
        }

        uint256_bitwise_xor(&r, stack->get_address_at_index(1), stack->get_address_at_index(2));
        stack->reduce_size(2);
        error_code |= stack->push(r);
    }
    return error_code;
}

__device__ int32_t NOT(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t r;
        if (stack->size() < 1) {
            return ERROR_STACK_UNDERFLOW;
        }
        uint256_bitwise_not(&r, stack->get_address_at_index(1));
        stack->reduce_size(1);
        error_code |= stack->push(r);
    }
    return error_code;
}

__device__ int32_t BYTE(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t r;
        evm_word_t *i, *x;
        if (stack->size() < 2) {
            return ERROR_STACK_UNDERFLOW;
        }
        i = stack->get_address_at_index(1);
        x = stack->get_address_at_index(2);
        // printf("i: %u\n", uint256_get_uint32_t(i));
        if (uint256_cmp_word(i, UINT256_BYTES - 1) == 1) {
            r = 0;
        } else {
            uint256_extract_byte(&r, x, UINT256_BYTES - 1 - uint256_get_uint32_t(i));
        }
        stack->reduce_size(2);
        error_code |= stack->push(r);
    }
    return error_code;
}

__device__ int32_t SHL(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t *shift, *value;
        if (stack->size() < 2) {
            return ERROR_STACK_UNDERFLOW;
        }
        shift = stack->get_address_at_index(1);
        value = stack->get_address_at_index(2);
        // error_code |= stack->pop(shift);
        // error_code |= stack->pop(value);
        evm_word_t r;

        if (uint256_cmp_word(shift, CuEVM::word_bits - 1) == 1) {
            r = 0;
        } else {
            uint32_t shift_left = uint256_get_uint32_t(shift);
            uint256_shift_left(&r, value, shift_left);
        }

        stack->reduce_size(2);
        error_code |= stack->push(r);
    }
    return error_code;
}

__device__ int32_t SHR(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t *shift, *value;
        if (stack->size() < 2) {
            return ERROR_STACK_UNDERFLOW;
        }
        shift = stack->get_address_at_index(1);
        value = stack->get_address_at_index(2);
        // error_code |= stack->pop(shift);
        // error_code |= stack->pop(value);
        evm_word_t r;

        if (uint256_cmp_word(shift, CuEVM::word_bits - 1) == 1) {
            r = 0;
        } else {
            uint256_shift_right(&r, value, shift->words[0]);
        }

        stack->reduce_size(2);
        error_code |= stack->push(r);
    }
    return error_code;
}

__device__ int32_t SAR(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t *shift, *value;
        if (stack->size() < 2) {
            return ERROR_STACK_UNDERFLOW;
        }
        shift = stack->get_address_at_index(1);
        value = stack->get_address_at_index(2);
        // error_code |= stack->pop(shift);
        // error_code |= stack->pop(value);
        evm_word_t r;
        uint32_t shift_uint32 = uint256_get_uint32_t(shift);
        if (uint256_cmp_word(shift, UINT256_BITS - 1) == 1) {
            shift_uint32 = UINT256_BITS;
        }

        uint256_shift_arithmetic_right(&r, value, shift_uint32);
        stack->reduce_size(2);
        error_code |= stack->push(r);
    }
    return error_code;
}

__device__ int32_t LT(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        if (stack->size() < 2) {
            return ERROR_STACK_UNDERFLOW;
        }
        evm_word_t *a, *b;
        a = stack->get_address_at_index(1);
        b = stack->get_address_at_index(2);
        // error_code |= stack->pop(a);
        // error_code |= stack->pop(a);
        // error_code |= stack->pop(b);

        stack->reduce_size(2);
        error_code |= stack->push_uint32(uint256_cmp(a, b) < 0);
    }
    return error_code;
}

__device__ int32_t GT(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t *a, *b;
        if (stack->size() < 2) {
            return ERROR_STACK_UNDERFLOW;
        }
        a = stack->get_address_at_index(1);
        b = stack->get_address_at_index(2);
        // error_code |= stack->pop(a);
        // error_code |= stack->pop(b);
        stack->reduce_size(2);
        error_code |= stack->push_uint32(uint256_cmp(a, b) > 0);
    }
    return error_code;
}

__device__ int32_t SLT(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t *a, *b;
        if (stack->size() < 2) {
            return ERROR_STACK_UNDERFLOW;
        }
        a = stack->get_address_at_index(1);
        b = stack->get_address_at_index(2);
        // error_code |= stack->pop(a);
        // error_code |= stack->pop(b);
        stack->reduce_size(2);
        error_code |= stack->push_uint32(uint256_signed_cmp(a, b) < 0);
    }
    return error_code;
}

__device__ int32_t SGT(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t *a, *b;
        if (stack->size() < 2) {
            return ERROR_STACK_UNDERFLOW;
        }
        a = stack->get_address_at_index(1);
        b = stack->get_address_at_index(2);
        // error_code |= stack->pop(a);
        // error_code |= stack->pop(b);
        stack->reduce_size(2);
        error_code |= stack->push_uint32(uint256_signed_cmp(a, b) > 0);
    }
    return error_code;
}

__device__ int32_t EQ(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t *a, *b;
        if (stack->size() < 2) {
            return ERROR_STACK_UNDERFLOW;
        }
        a = stack->get_address_at_index(1);
        b = stack->get_address_at_index(2);
        // error_code |= stack->pop(a);
        // error_code |= stack->pop(b);
        stack->reduce_size(2);
        error_code |= stack->push_uint32(uint256_cmp(a, b) == 0);
    }
    return error_code;
}

__device__ int32_t ISZERO(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t *stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t *a;
        if (stack->size() < 1) {
            return ERROR_STACK_UNDERFLOW;
        }
        a = stack->get_address_at_index(1);
        // error_code |= stack->pop(a);
        stack->reduce_size(1);
        error_code |= stack->push_uint32(uint256_is_zero(a));
    }
    return error_code;
}
}  // namespace CuEVM::operations
