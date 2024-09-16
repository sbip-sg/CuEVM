// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation
// Programme) Author: Stefan-Dan Ciocirlan Data: 2024-06-20
// SPDX-License-Identifier: MIT

#include <CuEVM/core/byte_array.cuh>
#include <CuEVM/utils/error_codes.cuh>
#include <CuEVM/utils/evm_utils.cuh>

namespace CuEVM {
__host__ __device__ byte_array_t::byte_array_t(const uint32_t size)
    : size(size) {
    __SHARED_MEMORY__ uint8_t *tmp_data;
    __ONE_GPU_THREAD_BEGIN__
    if (size > 0) {
        tmp_data = new uint8_t[size];
        memset(tmp_data, 0, size * sizeof(uint8_t));
    } else
        tmp_data = nullptr;
    __ONE_GPU_THREAD_END__
    data = tmp_data;
}

__host__ __device__ byte_array_t::byte_array_t(uint8_t *data, uint32_t size)
    : size(size) {
    __SHARED_MEMORY__ uint8_t *tmp_data;
    __ONE_GPU_THREAD_BEGIN__
    if (size > 0) {
        tmp_data = new uint8_t[size];
        memcpy(tmp_data, data, size * sizeof(uint8_t));
    } else
        tmp_data = nullptr;
    __ONE_GPU_THREAD_END__
    this->data = tmp_data;
}

__host__ __device__ byte_array_t::byte_array_t(
    const byte_array_t &src_byte_array, uint32_t offset, uint32_t size)
    : size(size) {
    __SHARED_MEMORY__ uint8_t *tmp_data;
    __ONE_GPU_THREAD_BEGIN__
    if (size > 0) {
        tmp_data = new uint8_t[size];
        memset(tmp_data, 0, size * sizeof(uint8_t));
        if (offset < src_byte_array.size)
            memcpy(tmp_data, src_byte_array.data + offset,
                   min(size, src_byte_array.size - offset) * sizeof(uint8_t));
    } else
        tmp_data = nullptr;
    __ONE_GPU_THREAD_END__
    this->data = tmp_data;
}

__host__ byte_array_t::byte_array_t(const char *hex_string, int32_t endian,
                                    PaddingDirection padding)
    : size(0), data(nullptr) {
    from_hex(hex_string, endian, padding, 0);
}

__host__ byte_array_t::byte_array_t(const char *hex_string, uint32_t size,
                                    int32_t endian, PaddingDirection padding)
    : size(size), data(nullptr) {
    from_hex(hex_string, endian, padding, 0);
}

__host__ __device__ byte_array_t::~byte_array_t() { free(); }

__host__ __device__ void byte_array_t::free() {
    __ONE_GPU_THREAD_BEGIN__
    if ((size > 0) && (data != nullptr)) {
        delete[] data;
    }
    __ONE_GPU_THREAD_END__
    clear();
}
__host__ void byte_array_t::free_managed() {
    if ((size > 0) && (data != nullptr)) {
        CUDA_CHECK(cudaFree(data));
    }
    clear();
}
__host__ __device__ void byte_array_t::clear() {
    data = nullptr;
    size = 0;
}

__host__ __device__ byte_array_t::byte_array_t(const byte_array_t &other)
    : size(other.size) {
    __SHARED_MEMORY__ uint8_t *tmp_data;
    __ONE_GPU_THREAD_BEGIN__
    if (size > 0) {
        tmp_data = new uint8_t[size];
        memcpy(tmp_data, other.data, size * sizeof(uint8_t));
    } else
        tmp_data = nullptr;
    __ONE_GPU_THREAD_END__
    data = tmp_data;
}

__host__ __device__ byte_array_t &byte_array_t::operator=(
    const byte_array_t &other) {
    __SHARED_MEMORY__ uint8_t *tmp_data;
    if (this != &other) {
        __ONE_GPU_THREAD_BEGIN__
        if (size != other.size) {
            if (data!= nullptr)
                delete[] data;
            tmp_data = (other.size > 0) ? new uint8_t[other.size] : nullptr;
        }
        if (other.size > 0) {
            memcpy(tmp_data, other.data, other.size * sizeof(uint8_t));
        } else
            tmp_data = nullptr;
        __ONE_GPU_THREAD_END__
        data = tmp_data;
        size = other.size;
    }
    return *this;
}

__host__ __device__ int32_t byte_array_t::grow(uint32_t new_size,
                                               int32_t zero_padding) {
    if (new_size == size) return ERROR_SUCCESS;
    __SHARED_MEMORY__ uint8_t *new_data;
    __ONE_GPU_THREAD_BEGIN__
    new_data = new uint8_t[new_size];
    if (zero_padding) memset(new_data, 0, new_size * sizeof(uint8_t));
    if (size > 0) {
        if (new_size > size) {
            memcpy(new_data, data, size * sizeof(uint8_t));
            // if (zero_padding)
            //   memset(new_data + size, 0, new_size - size);
        } else {
            memcpy(new_data, data, new_size * sizeof(uint8_t));
        }
        delete[] data;
    }
    __ONE_GPU_THREAD_END__
    data = new_data;
    size = new_size;
    return ERROR_SUCCESS;
}

__host__ __device__ uint32_t byte_array_t::has_value(uint8_t value) const {
    __SHARED_MEMORY__ uint32_t error_code;
    uint32_t index;
    error_code = ERROR_VALUE_NOT_FOUND;
    __SYNC_THREADS__
#ifdef __CUDA_ARCH__
    uint32_t slot_size = size / CuEVM::cgbn_tpi;
    for (index = 0; index < slot_size; index++) {
        if (data[slot_size * threadIdx.x + index] == value) {
            error_code = ERROR_SUCCESS;
        }
    }
    for (index = slot_size * CuEVM::cgbn_tpi; index < size; index++) {
        if (data[index] == value) {
            error_code = ERROR_SUCCESS;
        }
    }
    __SYNC_THREADS__
#else
    for (index = 0; index < size; index++) {
        if (data[index] == value) {
            return ERROR_SUCCESS;
        }
    }
#endif
    return error_code;
}

__host__ __device__ void byte_array_t::print() const {
    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    printf("size: %u\n", size);
    printf("data: ");
    for (uint32_t index = 0; index < size; index++) printf("%02x", data[index]);
    printf("\n");
    __ONE_GPU_THREAD_WOSYNC_END__
}

__host__ char *byte_array_t::to_hex() const {
    char *hex_string = new char[size * 2 + 3];  // 3 - 0x and \0
    hex_string[0] = '0';
    hex_string[1] = 'x';
    char *tmp_hex_string = (char *)hex_string + 2;
    uint8_t *tmp_data = data;
    for (uint32_t idx = 0; idx < size; idx++) {
        CuEVM::utils::hex_from_byte(tmp_hex_string, *(tmp_data++));
        tmp_hex_string += 2;
    }
    hex_string[size * 2 + 2] = 0;
    return hex_string;
}

__host__ cJSON *byte_array_t::to_json() const {
    cJSON *data_json = cJSON_CreateObject();
    cJSON_AddNumberToObject(data_json, "size", size);
    if (size > 0) {
        char *hex_string = to_hex();
        cJSON_AddStringToObject(data_json, "data", hex_string);
        delete[] hex_string;
    } else {
        cJSON_AddStringToObject(data_json, "data", "0x");
    }
    return data_json;
}

__host__ int32_t byte_array_t::from_hex_set_le(const char *clean_hex_string,
                                               int32_t length) {
    if ((length < 0) || ((size * 2) < length)) {
        return 1;
    }
    if (length > 0) {
        char *current_char;
        current_char = (char *)clean_hex_string;
        int32_t index;
        uint8_t *dst_ptr;
        dst_ptr = data;
        for (index = 0; index < ((length + 1) / 2) - 1; index++) {
            *(dst_ptr++) = CuEVM::utils::byte_from_two_hex_char(
                *(current_char), *(current_char + 1));
            current_char += 2;
        }
        if (length % 2 == 1) {
            *(dst_ptr++) =
                CuEVM::utils::byte_from_two_hex_char(*current_char++, '0');
        } else {
            *(dst_ptr++) = CuEVM::utils::byte_from_two_hex_char(
                *(current_char), *(current_char + 1));
            current_char += 2;
        }
    }
    return 0;
}

__host__ int32_t byte_array_t::from_hex_set_be(const char *clean_hex_string,
                                               int32_t length,
                                               PaddingDirection padding) {
    if ((length < 0) || ((size * 2) < length)) {
        return 1;
    }
    if (length > 0) {
        char *current_char;
        current_char = (char *)clean_hex_string;
        uint8_t *dst_ptr;
        if (padding == PaddingDirection::RIGHT_PADDING) {  // right padding
            dst_ptr = data + size - 1;
        } else if (padding == PaddingDirection::LEFT_PADDING) {  // left padding
            dst_ptr = data + (length + 1) / 2 - 1;
        } else {
            return 1;
        }

        if (length % 2 == 1) {
            *dst_ptr-- =
                CuEVM::utils::byte_from_two_hex_char('0', *current_char++);
        } else {
            *dst_ptr-- = CuEVM::utils::byte_from_two_hex_char(
                *(current_char), *(current_char + 1));
            current_char += 2;
        }
        while (*current_char != '\0') {
            *dst_ptr-- = CuEVM::utils::byte_from_two_hex_char(
                *(current_char), *(current_char + 1));
            current_char += 2;
        }
    }
    return 0;
}

__host__ int32_t byte_array_t::from_hex(const char *hex_string, int32_t endian,
                                        PaddingDirection padding,
                                        int32_t managed) {
    char *tmp_hex_char;
    tmp_hex_char = (char *)hex_string;
    int32_t length = CuEVM::utils::clean_hex_string(&tmp_hex_char);
    if (length < 0) {
        data = nullptr;
        return ERROR_INVALID_HEX_STRING;
    }
    size = (size == 0) ? (length + 1) / 2 : size;
    if (size > 0) {
        if (managed) {
            CUDA_CHECK(
                cudaMallocManaged((void **)&data, sizeof(uint8_t) * size));
            memset(data, 0, size * sizeof(uint8_t));
        } else {
            // data = (uint8_t*) std::calloc(size, sizeof(uint8_t));
            data = new uint8_t[size];
            memset(data, 0, size * sizeof(uint8_t));
        }
    }
    int32_t error_code = ERROR_SUCCESS;
    if (endian == LITTLE_ENDIAN) {
        error_code = this->from_hex_set_le(tmp_hex_char, length);
    } else {
        error_code = this->from_hex_set_be(tmp_hex_char, length, padding);
    }
    if (error_code != ERROR_SUCCESS) {
        if (managed) {
            CUDA_CHECK(cudaFree(data));
        } else {
            delete[] data;
        }
        data = nullptr;
        size = 0;
    }
    return error_code;
}

__host__ __device__ int32_t
byte_array_t::padded_copy_BE(const byte_array_t src) {
    uint32_t copy_size;
    int32_t size_diff;
    if (src.size == size) {
        size_diff = 0;
        copy_size = src.size;
    } else if (src.size < size) {
        size_diff = 1;
        copy_size = src.size;
    } else {
        size_diff = -1;
        copy_size = size;
    }
    __ONE_GPU_THREAD_BEGIN__
    memcpy(data, src.data, copy_size);
    memset(data + copy_size, 0, size - src.size);
    __ONE_GPU_THREAD_END__
    return size_diff;
}

__host__ __device__ uint8_t &byte_array_t::operator[](uint32_t index) {
    return data[index];
}

// STATIC FUNCTIONS

__host__ byte_array_t *byte_array_t::get_cpu(uint32_t count) {
    byte_array_t *cpu_instances = new byte_array_t[count];
    return cpu_instances;
}

__host__ void byte_array_t::cpu_free(byte_array_t *cpu_instances,
                                     uint32_t count) {
    delete[] cpu_instances;
}

__host__ byte_array_t *byte_array_t::gpu_from_cpu(byte_array_t *cpu_instances,
                                                  uint32_t count) {
    byte_array_t *gpu_instances, *tmp_cpu_instances;
    tmp_cpu_instances = new byte_array_t[count];
    for (uint32_t idx = 0; idx < count; idx++) {
        if (cpu_instances[idx].size > 0) {
            CUDA_CHECK(cudaMalloc((void **)&tmp_cpu_instances[idx].data,
                                  sizeof(uint8_t) * cpu_instances[idx].size));
            CUDA_CHECK(cudaMemcpy(tmp_cpu_instances[idx].data,
                                  cpu_instances[idx].data,
                                  sizeof(uint8_t) * cpu_instances[idx].size,
                                  cudaMemcpyHostToDevice));
            tmp_cpu_instances[idx].size = cpu_instances[idx].size;
        }
    }
    CUDA_CHECK(
        cudaMalloc((void **)&gpu_instances, sizeof(byte_array_t) * count));
    CUDA_CHECK(cudaMemcpy(gpu_instances, tmp_cpu_instances,
                          sizeof(byte_array_t) * count,
                          cudaMemcpyHostToDevice));
    for (uint32_t idx = 0; idx < count; idx++) {
        tmp_cpu_instances[idx].size = 0;
        tmp_cpu_instances[idx].data = nullptr;
    }
    delete[] tmp_cpu_instances;
    return gpu_instances;
}

__host__ void byte_array_t::gpu_free(byte_array_t *gpu_instances,
                                     uint32_t count) {
    byte_array_t *cpu_instances = new byte_array_t[count];
    CUDA_CHECK(cudaMemcpy(cpu_instances, gpu_instances,
                          sizeof(byte_array_t) * count,
                          cudaMemcpyDeviceToHost));
    for (uint32_t idx = 0; idx < count; idx++) {
        if (cpu_instances[idx].size > 0) {
            CUDA_CHECK(cudaFree(cpu_instances[idx].data));
            cpu_instances[idx].data = nullptr;
            cpu_instances[idx].size = 0;
        }
    }
    delete[] cpu_instances;
    CUDA_CHECK(cudaFree(gpu_instances));
}

__host__ byte_array_t *byte_array_t::cpu_from_gpu(byte_array_t *gpu_instances,
                                                  uint32_t count) {
    byte_array_t *cpu_instances;
    cpu_instances = new byte_array_t[count];
    CUDA_CHECK(cudaMemcpy(cpu_instances, gpu_instances,
                          sizeof(byte_array_t) * count,
                          cudaMemcpyDeviceToHost));

    // 1. alocate the memory for gpu memory as memory which can be addressed by
    // the cpu
    byte_array_t *tmp_cpu_instances, *tmp_gpu_instances;
    tmp_cpu_instances = new byte_array_t[count];
    for (uint32_t idx = 0; idx < count; idx++) {
        if (cpu_instances[idx].size > 0) {
            CUDA_CHECK(cudaMalloc((void **)&tmp_cpu_instances[idx].data,
                                  sizeof(uint8_t) * cpu_instances[idx].size));
            tmp_cpu_instances[idx].size = cpu_instances[idx].size;
        }
    }
    CUDA_CHECK(
        cudaMalloc((void **)&tmp_gpu_instances, sizeof(byte_array_t) * count));
    CUDA_CHECK(cudaMemcpy(tmp_gpu_instances, tmp_cpu_instances,
                          sizeof(byte_array_t) * count,
                          cudaMemcpyHostToDevice));

    // 2. call the kernel to copy the memory between the gpu memories
    CUDA_CHECK(cudaDeviceSynchronize());
    CuEVM::byte_array_t_transfer_kernel<<<count, 1>>>(tmp_gpu_instances,
                                                      gpu_instances, count);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(gpu_instances);
    gpu_instances = tmp_gpu_instances;
    tmp_gpu_instances = nullptr;

    // 3. copy the gpu memories back in the cpu memories
    CUDA_CHECK(cudaMemcpy(cpu_instances, gpu_instances,
                          sizeof(byte_array_t) * count,
                          cudaMemcpyDeviceToHost));
    for (uint32_t idx = 0; idx < count; idx++) {
        if (cpu_instances[idx].size > 0) {
            tmp_cpu_instances[idx].data = new uint8_t[cpu_instances[idx].size];
            CUDA_CHECK(cudaMemcpy(tmp_cpu_instances[idx].data,
                                  cpu_instances[idx].data,
                                  sizeof(uint8_t) * cpu_instances[idx].size,
                                  cudaMemcpyDeviceToHost));
            tmp_cpu_instances[idx].size = cpu_instances[idx].size;
        } else {
            tmp_cpu_instances[idx].data = nullptr;
            tmp_cpu_instances[idx].size = 0;
        }
    }
    // 4. free the temporary allocated memory
    byte_array_t::gpu_free(gpu_instances, count);
    for (uint32_t idx = 0; idx < count; idx++) {
        cpu_instances[idx].data = tmp_cpu_instances[idx].data;
        tmp_cpu_instances[idx].data = nullptr;
        tmp_cpu_instances[idx].size = 0;
    }
    delete[] tmp_cpu_instances;
    tmp_cpu_instances = nullptr;
    return cpu_instances;
}

__host__ __device__ void byte_array_t::transfer_memory(byte_array_t &dst,
                                                       byte_array_t &src) {
    dst.size = src.size;
    if (src.size > 0) {
        memcpy(dst.data, src.data, src.size * sizeof(uint8_t));
    } else {
        dst.data = nullptr;
    }
    src.free();
}

// CPU-GPU
__global__ void byte_array_t_transfer_kernel(byte_array_t *dst_instances,
                                             byte_array_t *src_instances,
                                             uint32_t count) {
    uint32_t instance = blockIdx.x * blockDim.x + threadIdx.x;

    if (instance >= count) return;

    CuEVM::byte_array_t::transfer_memory(dst_instances[instance],
                                         src_instances[instance]);
}
}  // namespace CuEVM