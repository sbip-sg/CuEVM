// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation
// Programme) Author: Stefan-Dan Ciocirlan Date: 2023-11-30
// SPDX-License-Identifier: MIT

#include <CuEVM/core/stack.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM::stack {

__host__ __device__ evm_stack_t::evm_stack_t(evm_word_t *shared_stack_base, uint32_t stack_base_offset)
    : shared_stack_base(shared_stack_base),
      global_stack_base(nullptr),
      stack_base_offset(stack_base_offset),
      stack_offset(0) {}

__host__ __device__ evm_stack_t::~evm_stack_t() { free(); }

// __host__ __device__ evm_stack_t::evm_stack_t(const evm_stack_t &other) {
//     // free();
//     duplicate(other);
// }

__host__ __device__ void evm_stack_t::free() {
    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    if (global_stack_base != nullptr) {
        delete[] global_stack_base;
    }
    __ONE_GPU_THREAD_WOSYNC_END__
    clear();
}

__host__ __device__ void evm_stack_t::clear() {
    stack_offset = 0;
    // capacity = 0;
    global_stack_base = nullptr;
}

// __host__ __device__ evm_stack_t &evm_stack_t::operator=(const evm_stack_t &other) {
//     if (this != &other) {
//         free();
//         duplicate(other);
//     }
//     return *this;
// }

// __host__ __device__ void evm_stack_t::duplicate(const evm_stack_t &other) {
//     __SHARED_MEMORY__ evm_word_t *tmp_stack_base;
//     __ONE_GPU_THREAD_BEGIN__
//     tmp_stack_base = new evm_word_t[other.capacity];
//     if (tmp_stack_base != nullptr) {
//         memcpy(tmp_stack_base, other.global_stack_base, other.stack_offset * sizeof(evm_word_t));
//     }
//     __ONE_GPU_THREAD_END__
//     stack_offset = other.stack_offset;
//     capacity = other.stack_offset;
//     global_stack_base = tmp_stack_base;
// }

// TODO : reimplement
__host__ __device__ void evm_stack_t::extract_data(evm_word_t *other) const {
    // printf("Extract data stack offset %d\n", stack_offset);
    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    if (stack_offset + stack_base_offset < CuEVM::shared_stack_size) {
        memcpy(other, shared_stack_base, stack_offset * sizeof(evm_word_t));
    } else {
        int32_t left_over = CuEVM::shared_stack_size - stack_base_offset;
        if (left_over > 0) {
            memcpy(other, shared_stack_base, left_over * sizeof(evm_word_t));
            memcpy(other, global_stack_base, (stack_offset - left_over) * sizeof(evm_word_t));
        } else
            memcpy(other, global_stack_base, stack_offset * sizeof(evm_word_t));
    }
    // if (global_stack_base != nullptr) {
    //     memcpy(other, global_stack_base, stack_offset * sizeof(evm_word_t));
    // }
    __ONE_GPU_THREAD_WOSYNC_END__
}

// __host__ __device__ int32_t evm_stack_t::grow() {
//     // capacity = (capacity == 0) ? initial_capacity : capacity * 2;
//     // if (capacity > max_size) {
//     //     return ERROR_STACK_OVERFLOW;
//     // }
//     if (stack_base_offset + stack_offset < CuEVM::shared_stack_size) {
//         return ERROR_SUCCESS;
//     }
//     __SHARED_MEMORY__ evm_word_t *new_stack_base;
//     __ONE_GPU_THREAD_BEGIN__
//     new_stack_base = new evm_word_t[capacity];
//     if (global_stack_base != nullptr && new_stack_base != nullptr) {
//         memcpy(new_stack_base, global_stack_base, stack_offset * sizeof(evm_word_t));
//         delete[] global_stack_base;
//     }
//     __ONE_GPU_THREAD_END__
//     if (new_stack_base == nullptr) {
//         return ERROR_MEMORY_ALLOCATION_FAILED;
//     }
//     global_stack_base = new_stack_base;
//     return ERROR_SUCCESS;
// }

__host__ __device__ uint32_t evm_stack_t::size() const { return stack_offset; }

__host__ __device__ evm_word_t *evm_stack_t::top() {
    if (stack_base_offset + stack_offset < CuEVM::shared_stack_size) {
        // printf("shared stack base %p stack offset %d\n", shared_stack_base, stack_offset);
        return shared_stack_base + stack_offset;
    } else {
        if (global_stack_base == nullptr) {
            __SHARED_MEMORY__ evm_word_t *new_stack_base;
            // allocate one
            __ONE_GPU_THREAD_WOSYNC_BEGIN__
            printf("allocating new stack base on global memory \n");
            new_stack_base = new evm_word_t[max_stack_size];
            __ONE_GPU_THREAD_END__
            global_stack_base = new_stack_base;
        }
        return global_stack_base + stack_base_offset + stack_offset - CuEVM::shared_stack_size;
    }
}

// __host__ __device__ int32_t evm_stack_t::push(ArithEnv &arith, const bn_t &value) {
//     int32_t error_code = (size() >= CuEVM::shared_stack_size) ? grow() : ERROR_SUCCESS;
//     if (error_code == ERROR_SUCCESS) {
//         cgbn_store(arith.env, top(), value);
//         stack_offset++;
//     }

//     return error_code;
// }
__host__ __device__ int32_t evm_stack_t::push(ArithEnv &arith, const bn_t &value) {
    if (stack_offset < max_stack_size) {
        cgbn_store(arith.env, top(), value);
        stack_offset++;
        // printf("Stack offset %d pointer %p  top %p\n", stack_offset, shared_stack_base, top());

        return ERROR_SUCCESS;
    } else {
        return ERROR_STACK_OVERFLOW;
    }
}

__host__ __device__ int32_t evm_stack_t::pop(ArithEnv &arith, bn_t &y) {
    if (size() == 0) {
        // TODO: delete maybe?
        cgbn_set_ui32(arith.env, y, 0);
        return ERROR_STACK_UNDERFLOW;
    }
    cgbn_load(arith.env, y, get_address_at_index(1));
    stack_offset--;
    // printf("Stack pop offset %d idx %d\n", stack_offset, THREADIDX);
    // cgbn_load(arith.env, y, top());
    return ERROR_SUCCESS;
}

__host__ __device__ int32_t evm_stack_t::pop_evm_word(ArithEnv &arith, evm_word_t *&y) {
    if (stack_offset == 0) {
        return ERROR_STACK_UNDERFLOW;
    }
    y = get_address_at_index(1);
    stack_offset--;
    return ERROR_SUCCESS;
}

#ifdef __CUDA_ARCH__
__host__ __device__ int32_t evm_stack_t::pushx(ArithEnv &arith, uint8_t x, uint8_t *src_byte_data,
                                               uint8_t src_byte_size) {
    // if (x > 32) {
    //     return ERROR_STACK_INVALID_SIZE;
    // }
    // x valud must be validated by checking opcode ouside
    // __ONE_GPU_THREAD_WOSYNC_BEGIN__
    // printf("pushx %d data size %d data to insert: \n", x, src_byte_size);
    // for (uint8_t idx = 0; idx < src_byte_size; idx++) {
    //     printf("%02x", src_byte_data[idx]);
    // }
    // printf("\n");
    // __ONE_GPU_THREAD_WOSYNC_END__

    // printf("pushx data size %d\n", src_byte_size);
    // __ONE_GPU_THREAD_WOSYNC_BEGIN__
    // for (uint8_t idx = 0; idx < src_byte_size; idx++) {
    //     printf("%02x", src_byte_data[idx]);
    // }
    // __ONE_GPU_THREAD_WOSYNC_END__

    // int32_t error_code = (size() >= capacity) ? grow() : ERROR_SUCCESS;
    if (stack_offset < max_stack_size) {
        int last_idx_from_left = 31 - min(x, src_byte_size);
        int my_idx = threadIdx.x % CuEVM::cgbn_tpi;
        // printf("pushx data inserted %d myidx %d firstidxleft %d\n", threadIdx.x, my_idx, last_idx_from_left);
        // if (my_idx < first_idx / 4) {
        //     top_->_limbs[my_idx] = 0;
        // } else {
        // each thead will insert 4 bytes/ hardcoded big endian for now
        int byte_start = (my_idx + 1) * 4 - 1;
        uint32_t limb_value = 0;
        if (byte_start > last_idx_from_left) {
            limb_value |= src_byte_data[src_byte_size - 1 - (31 - byte_start)];  //<< 24;
        }
        byte_start--;
        if (byte_start > last_idx_from_left) {
            limb_value |= src_byte_data[src_byte_size - 1 - (31 - byte_start)] << 8;  // << 16;
        }
        byte_start--;
        if (byte_start > last_idx_from_left) {
            limb_value |= src_byte_data[src_byte_size - 1 - (31 - byte_start)] << 16;  // << 8;
        }
        byte_start--;
        if (byte_start > last_idx_from_left) {
            // printf("bytestart %d src data idx %d thread idx %d, srcbyte %02x \n", byte_start, 31 - byte_start,
            //        threadIdx.x, src_byte_data[31 - byte_start]);
            limb_value |= src_byte_data[src_byte_size - 1 - (31 - byte_start)] << 24;
        }

        top()->_limbs[CuEVM::cgbn_tpi - my_idx - 1] = limb_value;
        stack_offset++;

        // __SYNC_THREADS__  // do we need to sync here?
        //     __ONE_THREAD_PER_INSTANCE(printf("pushx data inserted %d stack offset %d \n", threadIdx.x,
        //     stack_offset););
        // this->print();
        return ERROR_SUCCESS;
    } else {
        printf("pushx overflow stack offset %d\n", stack_offset);
        return ERROR_STACK_OVERFLOW;
    }
}

#else
__host__ __device__ int32_t evm_stack_t::pushx(ArithEnv &arith, uint8_t x, uint8_t *src_byte_data,
                                               uint8_t src_byte_size) {
    // TODO:: for sure is something more efficient here
    if (x > 32) {
        return ERROR_STACK_INVALID_SIZE;
    }
    bn_t r;
    cgbn_set_ui32(arith.env, r, 0);
    for (uint8_t idx = (x - src_byte_size); idx < x; idx++) {
        cgbn_insert_bits_ui32(arith.env, r, r, idx * 8, 8, src_byte_data[x - 1 - idx]);
    }

    return push(arith, r);
}
#endif

#ifdef __CUDA_ARCH__
// The caller must check underflow
__host__ __device__ evm_word_t *evm_stack_t::get_address_at_index(uint32_t index) const {
    if (stack_base_offset + stack_offset - index < CuEVM::shared_stack_size)
        return shared_stack_base + stack_offset - index;
    else {
        return global_stack_base + stack_base_offset + stack_offset - index - CuEVM::shared_stack_size;
    }
}

__host__ __device__ int32_t evm_stack_t::dupx(ArithEnv &arith, uint32_t x) {
    // if ((x > 16) || (x < 1)) return ERROR_STACK_INVALID_SIZE;
    // for efficiency this must be checked from outside this func

    if ((stack_offset < max_stack_size) && (x <= stack_offset)) {
        // cgbn_store(arith.env, top(), value);
        __ONE_GPU_THREAD_WOSYNC_BEGIN__
        // TODO: optimize memcpy
        memcpy(top(), get_address_at_index(x), sizeof(evm_word_t));
        __ONE_GPU_THREAD_WOSYNC_END__
        stack_offset++;
        return ERROR_SUCCESS;
    } else {
        printf(" dupx overflow or underflow stack offset %d\n", stack_offset);
        return ERROR_STACK_OVERFLOW;  // represent underflow also
    };
}

__host__ __device__ int32_t evm_stack_t::swapx(ArithEnv &arith, uint32_t x) {
    // if ((x > 16) || (x < 1)) return ERROR_STACK_INVALID_SIZE;
    // for efficiency this must be checked from outside this func

    // cgbn_store(arith.env, global_stack_base + size() - x - 1, a);
    // cgbn_store(arith.env, global_stack_base + size() - 1, b);
    x++;
    if (x > stack_offset) {
        return ERROR_STACK_UNDERFLOW;
    }
    __SHARED_MEMORY__ evm_word_t tmp;
    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    memcpy(&tmp, get_address_at_index(x), sizeof(evm_word_t));
    memcpy(get_address_at_index(x), get_address_at_index(1), sizeof(evm_word_t));
    memcpy(get_address_at_index(1), &tmp, sizeof(evm_word_t));
    __ONE_GPU_THREAD_WOSYNC_END__
    return ERROR_SUCCESS;
}

#else
__host__ __device__ evm_word_t *evm_stack_t::get_address_at_index(uint32_t index) const {}
__host__ __device__ int32_t evm_stack_t::get_index(ArithEnv &arith, uint32_t index, bn_t &y) {
    if (index > size()) {
        return ERROR_STACK_INVALID_INDEX;
    }
    cgbn_load(arith.env, y, global_stack_base + size() - index);
    return ERROR_SUCCESS;
}

__host__ __device__ int32_t evm_stack_t::dupx(ArithEnv &arith, uint32_t x) {
    bn_t r;
    int32_t error_code = ((x > 16) || (x < 1)) ? ERROR_STACK_INVALID_SIZE : get_index(arith, x, r);
    return error_code | push(arith, r);
}

__host__ __device__ int32_t evm_stack_t::swapx(ArithEnv &arith, uint32_t x) {
    bn_t a, b;
    int32_t error_code =
        ((x > 16) || (x < 1)) ? ERROR_STACK_INVALID_SIZE : (get_index(arith, 1, a) | get_index(arith, x + 1, b));
    if (error_code == ERROR_SUCCESS) {
        cgbn_store(arith.env, global_stack_base + size() - x - 1, a);
        cgbn_store(arith.env, global_stack_base + size() - 1, b);
    }
    return error_code;
}
#endif

__host__ __device__ void evm_stack_t::print() {
    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    printf("Stack size: %d, data:\n", size());
    for (uint32_t idx = 0; idx < size(); idx++) {
        shared_stack_base[idx].print();
    }
    __ONE_GPU_THREAD_WOSYNC_END__
}

__host__ cJSON *evm_stack_t::to_json() {
    cJSON *json = cJSON_CreateObject();
    char *hex_string_ptr = new char[CuEVM::word_size * 2 + 3];
    cJSON *stack = cJSON_CreateArray();
    for (uint32_t idx = 0; idx < size(); idx++) {
        global_stack_base[idx].to_hex(hex_string_ptr);
        cJSON_AddItemToArray(stack, cJSON_CreateString(hex_string_ptr));
    }
    cJSON_AddItemToObject(json, "data", stack);
    delete[] hex_string_ptr;
    return json;
}
__host__ evm_stack_t *evm_stack_t::get_cpu(uint32_t count) {
    evm_stack_t *instances = new evm_stack_t[count];
    return instances;
}
__host__ void evm_stack_t::cpu_free(evm_stack_t *instances, uint32_t count) { delete[] instances; }

/*__host__ evm_stack_t *evm_stack_t::gpu_from_cpu(evm_stack_t *cpu_instances, uint32_t count) {
    evm_stack_t *gpu_instances, *tmp_gpu_instances;
    tmp_gpu_instances = new evm_stack_t[count];
    for (uint32_t idx = 0; idx < count; idx++) {
        if (cpu_instances[idx].global_stack_base != nullptr) {
            CUDA_CHECK(cudaMalloc(&tmp_gpu_instances[idx].global_stack_base,
                                  cpu_instances[idx].capacity * sizeof(evm_word_t)));
            CUDA_CHECK(cudaMemcpy(tmp_gpu_instances[idx].global_stack_base, cpu_instances[idx].global_stack_base,
                                  cpu_instances[idx].stack_offset * sizeof(evm_word_t), cudaMemcpyHostToDevice));
        } else {
            tmp_gpu_instances[idx].global_stack_base = nullptr;
        }
        tmp_gpu_instances[idx].stack_offset = cpu_instances[idx].stack_offset;
        tmp_gpu_instances[idx].capacity = cpu_instances[idx].capacity;
    }
    CUDA_CHECK(cudaMalloc(&gpu_instances, count * sizeof(evm_stack_t)));
    CUDA_CHECK(cudaMemcpy(gpu_instances, tmp_gpu_instances, count * sizeof(evm_stack_t), cudaMemcpyHostToDevice));
    for (uint32_t idx = 0; idx < count; idx++) {
        tmp_gpu_instances[idx].clear();
    }
    delete[] tmp_gpu_instances;
    return gpu_instances;
}*/
__host__ void evm_stack_t::gpu_free(evm_stack_t *gpu_instances, uint32_t count) {
    evm_stack_t *tmp_gpu_instances = new evm_stack_t[count];
    CUDA_CHECK(cudaMemcpy(tmp_gpu_instances, gpu_instances, count * sizeof(evm_stack_t), cudaMemcpyDeviceToHost));
    for (uint32_t idx = 0; idx < count; idx++) {
        if (tmp_gpu_instances[idx].global_stack_base != nullptr && tmp_gpu_instances[idx].stack_offset > 0) {
            CUDA_CHECK(cudaFree(tmp_gpu_instances[idx].global_stack_base));
        }
        tmp_gpu_instances[idx].clear();
    }
    delete[] tmp_gpu_instances;
    CUDA_CHECK(cudaFree(gpu_instances));
}
/*
__host__ evm_stack_t *evm_stack_t::cpu_from_gpu(evm_stack_t *gpu_instances, uint32_t count) {
    evm_stack_t *cpu_instances = new evm_stack_t[count];
    evm_stack_t *tmp_gpu_instances = new evm_stack_t[count];
    evm_stack_t *tmp_cpu_instances = new evm_stack_t[count];
    CUDA_CHECK(cudaMemcpy(cpu_instances, gpu_instances, count * sizeof(evm_stack_t), cudaMemcpyDeviceToHost));
    for (uint32_t idx = 0; idx < count; idx++) {
        if (cpu_instances[idx].stack_offset > 0) {
            CUDA_CHECK(cudaMalloc(&tmp_cpu_instances[idx].global_stack_base,
                                  cpu_instances[idx].stack_offset * sizeof(evm_word_t)));
        } else {
            tmp_cpu_instances[idx].global_stack_base = nullptr;
        }
        tmp_cpu_instances[idx].stack_offset = cpu_instances[idx].stack_offset;
        tmp_cpu_instances[idx].capacity = cpu_instances[idx].stack_offset;
    }
    CUDA_CHECK(cudaMalloc(&tmp_gpu_instances, count * sizeof(evm_stack_t)));
    CUDA_CHECK(cudaMemcpy(tmp_gpu_instances, tmp_cpu_instances, count * sizeof(evm_stack_t), cudaMemcpyHostToDevice));
    for (uint32_t idx = 0; idx < count; idx++) {
        tmp_cpu_instances[idx].clear();
        cpu_instances[idx].clear();
    }
    transfer_kernel_evm_stack_t<<<count, 1>>>(tmp_gpu_instances, gpu_instances, count);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(gpu_instances));

    CUDA_CHECK(cudaMemcpy(tmp_cpu_instances, tmp_gpu_instances, sizeof(evm_stack_t) * count, cudaMemcpyDeviceToHost));

    for (uint32_t idx = 0; idx < count; idx++) {
        if (tmp_cpu_instances[idx].stack_offset > 0) {
            cpu_instances[idx].global_stack_base = new evm_word_t[tmp_cpu_instances[idx].stack_offset];
            CUDA_CHECK(cudaMemcpy(cpu_instances[idx].global_stack_base, tmp_cpu_instances[idx].global_stack_base,
                                  sizeof(evm_word_t) * tmp_cpu_instances[idx].stack_offset, cudaMemcpyDeviceToHost));
        } else {
            cpu_instances[idx].global_stack_base = NULL;
        }
        cpu_instances[idx].stack_offset = tmp_cpu_instances[idx].stack_offset;
        cpu_instances[idx].capacity = tmp_cpu_instances[idx].stack_offset;
    }
    for (uint32_t idx = 0; idx < count; idx++) {
        tmp_cpu_instances[idx].clear();
    }
    delete[] tmp_cpu_instances;
    evm_stack_t::gpu_free(tmp_gpu_instances, count);
    return cpu_instances;
}

__global__ void transfer_kernel_evm_stack_t(evm_stack_t *dst, evm_stack_t *src, uint32_t count) {
    uint32_t instance = blockIdx.x * blockDim.x + threadIdx.x;
    if (instance >= count) {
        return;
    }
    dst[instance].stack_offset = src[instance].stack_offset;
    dst[instance].capacity = src[instance].stack_offset;
    memcpy(dst[instance].global_stack_base, src[instance].global_stack_base,
           src[instance].stack_offset * sizeof(evm_word_t));
    delete[] src[instance].global_stack_base;
    src[instance].clear();
}*/

}  // namespace CuEVM::stack