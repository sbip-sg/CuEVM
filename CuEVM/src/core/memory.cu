// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#include <CuEVM/core/memory.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM {
namespace memory {
__host__ __device__ void evm_memory_t::print() const {
    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    printf("Memory data: \n");
    printf("Size: %d\n", size);
    printf("Memory cost: %lu\n", memory_cost);
    printf("\n");
    data.print();
    __ONE_GPU_THREAD_WOSYNC_END__
}

__host__ cJSON *evm_memory_t::to_json() const {
    cJSON *json = cJSON_CreateObject();
    cJSON_AddItemToObject(json, "size", cJSON_CreateNumber(size));
    cJSON_AddItemToObject(json, "memory_cost", cJSON_CreateNumber(memory_cost));
    cJSON_AddItemToObject(json, "data", data.to_json());
    return json;
}

__host__ __device__ void evm_memory_t::get_memory_cost(gas_t &cost) const { cost = memory_cost; }

__host__ __device__ void evm_memory_t::increase_memory_cost(gas_t memory_expansion_cost) {
    memory_cost += memory_expansion_cost;
}

__host__ __device__ int32_t evm_memory_t::allocate_pages(uint32_t new_size) {
    if (new_size < data.size) {
        return ERROR_SUCCESS;
    }
    uint32_t new_page_count = (new_size / CuEVM::memory::page_size) + 1;
    return data.grow(new_page_count * CuEVM::memory::page_size, 1);
}

__host__ __device__ int32_t evm_memory_t::get_last_offset(const evm_word_t &index, const evm_word_t &length,
                                                          uint32_t &offset) const {
    int32_t overflow = 0;
    /*    bn_t offset_bn;
        overflow = cgbn_add(arith.env, offset_bn, index, length);
        overflow |= cgbn_get_uint32_t(arith.env, offset, offset_bn);
        bn_t memory_size;
        overflow |= cgbn_add_ui32(arith.env, memory_size, offset_bn, 31);
        cgbn_div_ui32(arith.env, memory_size, memory_size, 32);
        overflow |= cgbn_mul_ui32(arith.env, offset_bn, memory_size, 32);
        overflow |= cgbn_get_uint32_t(arith.env, offset, offset_bn);
    */
    uint256 tmp;
    uint256_add(&tmp, &index, &length);
    offset = tmp.words[UINT256_WORDS - 1];
    offset = ((offset + 31) / 32) * 32;
    // todo fix this
    return overflow;
}

__host__ __device__ int32_t evm_memory_t::grow(const evm_word_t &index, const evm_word_t &length) {
    uint32_t offset;
    if (get_last_offset(index, length, offset) != 0) {
        return ERR_MEMORY_INVALID_OFFSET;
    }
    if (offset > size) {
        if (allocate_pages(offset) != 0) {
            return ERR_MEMORY_INVALID_ALLOCATION;
        }
        size = offset;
    }
    return ERROR_SUCCESS;
}

__host__ __device__ int32_t evm_memory_t::get(const evm_word_t &index, const evm_word_t &length,
                                              CuEVM::byte_array_t &data) {
    int32_t error_code = ERROR_SUCCESS;
    if (uint256_is_zero(&length)) {
        data = CuEVM::byte_array_t();
        return error_code;
    }
    // error_code = (uint256_is_negative(&length)) ? ERR_MEMORY_INVALID_SIZE : error_code;
    error_code |= grow(index, length);
    if (error_code == ERROR_SUCCESS) {
        uint32_t index_u32 = uint256_get_uint32_t(&index);
        uint32_t length_u32 = uint256_get_uint32_t(&length);
        data = CuEVM::byte_array_t(this->data.data + index_u32, length_u32);
    } else {
        data = CuEVM::byte_array_t();
    }
    return error_code;
}

__host__ __device__ int32_t evm_memory_t::set(const CuEVM::byte_array_t &data, const evm_word_t &index,
                                              const evm_word_t &length) {
    int32_t error_code = ERROR_SUCCESS;
    // error_code = (uint256_is_negative(&length)) ? ERR_MEMORY_INVALID_SIZE : error_code;
    error_code |= grow(index, length);
    if (error_code == ERROR_SUCCESS) {
        uint32_t index_u32 = uint256_get_uint32_t(&index);
        uint32_t length_u32 = uint256_get_uint32_t(&length);
        if (data.size > 0) {
            memcpy(this->data.data + index_u32, data.data, min(length_u32, data.size));
        }
    }
    return error_code;
}

__host__ evm_memory_t *get_cpu(uint32_t count) { return new evm_memory_t[count]; }

__host__ void cpu_free(evm_memory_t *instances, uint32_t count) { delete[] instances; }

}  // namespace memory
}  // namespace CuEVM