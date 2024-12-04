#include <CuEVM/gas_cost.cuh>
#include <CuEVM/operations/compare.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM::operations {
/**
 * Compare the top two values from the stack.
 * The two values are considered unsigned.
 * -1 if the first value is less than the second value,
 * 0 if the first value is equal to the second value,
 * 1 if the first value is greater than the second value.
 * @param[in] arith The arithmetical environment.
 * @param[inout] stack The stack.
 * @param[out] result The result of the comparison.
 * @return 0 if the operation was successful, an error code otherwise.
 */
__host__ __device__ int32_t compare(CuEVM::evm_stack_t &stack, int32_t &result) {
    evm_word_t a, b;
    int32_t error_code = stack.pop(a);
    error_code |= stack.pop(b);
    result = uint256_cmp(&a, &b);
    return error_code;
}

/**
 * Compare the top two values from the stack.
 * The two values are considered signed.
 * -1 if the first value is less than the second value,
 * 0 if the first value is equal to the second value,
 * 1 if the first value is greater than the second value.
 * @param[in] arith The arithmetical environment.
 * @param[inout] stack The stack.
 * @param[out] result The result of the comparison.
 * @return 0 if the operation was successful, an error code otherwise.
 */
__host__ __device__ int32_t scompare(CuEVM::evm_stack_t &stack, int32_t &result) {
    evm_word_t a, b;
    int32_t error_code = stack.pop(a);
    error_code |= stack.pop(b);
    /*
        uint32_t sign_a = cgbn_extract_bits_ui32(arith.env, a, CuEVM::word_bits - 1, 1);
        uint32_t sign_b = cgbn_extract_bits_ui32(arith.env, b, CuEVM::word_bits - 1, 1);
        result = (sign_a == 0 && sign_b == 1) ? 1 : (sign_a == 1 && sign_b == 0) ? -1 : cgbn_compare(arith.env, a, b);
    */
    return error_code;
}

__host__ __device__ int32_t LT(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        int32_t int_result;
        error_code |= compare(stack, int_result);
        uint32_t result = (int_result < 0) ? 1 : 0;
        evm_word_t r;

        uint256_set_zero(&r);

        error_code |= stack.push(r);
    }
    return error_code;
}

__host__ __device__ int32_t GT(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        int32_t int_result;
        error_code |= compare(stack, int_result);
        uint32_t result = (int_result > 0) ? 1 : 0;
        evm_word_t r;

        uint256_set_zero(&r);

        error_code |= stack.push(r);
    }
    return error_code;
}

__host__ __device__ int32_t SLT(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        int32_t int_result;
        error_code |= scompare(stack, int_result);
        uint32_t result = (int_result < 0) ? 1 : 0;
        evm_word_t r;

        uint256_from_uint32(&r, result);

        error_code |= stack.push(r);
    }
    return error_code;
}

__host__ __device__ int32_t SGT(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        int32_t int_result;
        error_code |= scompare(stack, int_result);
        uint32_t result = (int_result > 0) ? 1 : 0;
        evm_word_t r;

        uint256_from_uint32(&r, result);

        error_code |= stack.push(r);
    }
    return error_code;
}

__host__ __device__ int32_t EQ(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        int32_t int_result;
        error_code |= compare(stack, int_result);
        uint32_t result = (int_result == 0) ? 1 : 0;
        evm_word_t r;

        uint256_from_uint32(&r, result);

        error_code |= stack.push(r);
    }
    return error_code;
}

__host__ __device__ int32_t ISZERO(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack) {
    gas_used += GAS_VERY_LOW;
    int32_t error_code = CuEVM::gas_cost::has_gas(gas_limit, gas_used);
    if (error_code == ERROR_SUCCESS) {
        evm_word_t a;
        error_code |= stack.pop(a);
        evm_word_t r;

        int32_t compare = uint256_cmp_word(&a, 0);
        if (compare == 0) {
            uint256_from_uint32(&r, 1);
        } else {
            uint256_set_zero(&r);
        }

        error_code |= stack.push(r);
    }
    return error_code;
}
}  // namespace CuEVM::operations