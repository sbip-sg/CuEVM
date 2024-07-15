// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2024-06-20
// SPDX-License-Identifier: MIT

#include "../include/core/byte_array.cuh"
#include "../include/utils/evm_utils.cuh"
#include "../include/utils/error_codes.cuh"

namespace cuEVM {
  __host__ __device__ byte_array_t::byte_array_t(
    uint32_t size) : size(size) {
      if (size > 0)
        data = new uint8_t[size];
      else
        data = nullptr;
  }

  __host__ __device__ byte_array_t::byte_array_t(
    uint8_t *data,
    uint32_t size) : size(size) {
      if (size > 0)
      {
        this->data = new uint8_t[size];
        std::copy(data, data + size, this->data);
      }
      else
        this->data = nullptr;
  }

  __host__ __device__ byte_array_t::byte_array_t(
    const char *hex_string,
    int32_t endian,
    PaddingDirection padding) : size(0), data(nullptr) {
      from_hex(hex_string, endian, padding, 0);
  }

  __host__ __device__ byte_array_t::byte_array_t(
    const char *hex_string,
    uint32_t size,
    int32_t endian,
    PaddingDirection padding) : size(size), data(nullptr) {
      from_hex(hex_string, endian, padding, 0);
  }

  __host__ __device__ byte_array_t::~byte_array_t() {
    if (
      (size > 0) &&
      (data != nullptr)
    ) {
      delete[] data;
      data = nullptr;
    }
    size = 0;
  }

  __host__ __device__ byte_array_t::byte_array_t(
    const byte_array_t &other) : size(other.size) {
      if (size > 0)
      {
        data = new uint8_t[size];
        std::copy(other.data, other.data + size, data);
      }
      else
        data = nullptr;
  }

  __host__ __device__ byte_array_t& byte_array_t::operator=(
    const byte_array_t &other) {
      if (this == &other)
        return *this;
      if ((size > 0) && (size != other.size)) {
        delete[] data;
        data = nullptr;
        size = other.size;
        data = new uint8_t[size];
      }
      if (size > 0)
      {
        std::copy(other.data, other.data + size, data);
      }
      else
        data = nullptr;
      return *this;
  }


  __host__ __device__ int32_t byte_array_t::grow(
    uint32_t new_size,
    int32_t zero_padding) {
      if (new_size == size)
        return 0;
      uint8_t *new_data = new uint8_t[new_size];
      if (size > 0)
      {
        if (new_size > size)
        {
          std::copy(data, data + size, new_data);
          if (zero_padding)
            std::fill(new_data + size, new_data + new_size, 0);
        }
        else
          std::copy(data, data + new_size, new_data);
        delete[] data;
      }
      data = new_data;
      size = new_size;
      return 1;
  }

  __host__ __device__ void byte_array_t::print() const {
      printf("size: %u\n", size);
      printf("data: ");
      for(uint32_t index=0; index<size; index++)
        printf("%02x", data[index]);
      printf("\n");
  }

  __host__ __device__ char *byte_array_t::to_hex() const {
    char *hex_string = new char[size*2+3]; // 3 - 0x and \0
    hex_string[0]='0';
    hex_string[1]='x';
    char *tmp_hex_string = (char *)hex_string + 2;
    uint8_t *tmp_data = data;
    for(uint32_t idx=0; idx<size; idx++) {
      cuEVM::utils::hex_from_byte(tmp_hex_string, *(tmp_data++));
      tmp_hex_string += 2;
    }
    hex_string[size*2+2]=0;
    return hex_string;
  }

  __host__ __device__ cJSON* byte_array_t::to_json() const {
    cJSON *data_json = cJSON_CreateObject();
    cJSON_AddNumberToObject(data_json, "size", size);
    if (size > 0)
    {
      char *hex_string = to_hex();
      cJSON_AddStringToObject(data_json, "data", hex_string);
      delete[] hex_string;
    } else {
      cJSON_AddStringToObject(data_json, "data", "0x");
    }
    return data_json;
  }

  __host__ __device__ int32_t byte_array_t::from_hex_set_le(
    const char *clean_hex_string,
    int32_t length) {
    if ( (length < 0) || ( (size * 2) < length ) ) {
      return 1;
    }
    if (length > 0)
    {
      char *current_char;
      current_char = (char *)clean_hex_string;
      int32_t index;
      uint8_t *dst_ptr;
      dst_ptr = data;
      for (index = 0; index < ((length+1)/2) - 1; index++)
      {
        *(dst_ptr++) = cuEVM::utils::byte_from_two_hex_char(
          *(current_char),
          *(current_char+1)
        );
        current_char += 2;
      }
      if (length % 2 == 1)
      {
        *(dst_ptr++) = cuEVM::utils::byte_from_two_hex_char(
          *current_char++,
          '0'
        );
      } else {
        *(dst_ptr++) = cuEVM::utils::byte_from_two_hex_char(
          *(current_char),
          *(current_char+1)
        );
        current_char += 2;
      }
    }
    return 0;
  }

  __host__ __device__ int32_t byte_array_t::from_hex_set_be(
    const char *clean_hex_string,
    int32_t length,
    PaddingDirection padding) {
    if ( (length < 0) || ( (size * 2) < length ) ) {
      return 1;
    }
    if (length > 0)
    {
      char *current_char;
      current_char = (char *)clean_hex_string;
      int32_t index;
      uint8_t *dst_ptr;
      if (padding == PaddingDirection::RIGHT_PADDING) { // right padding
        dst_ptr = data + size - 1;
      } else if (padding == PaddingDirection::LEFT_PADDING) { // left padding
        dst_ptr = data + (length + 1) / 2 - 1;
      } else {
        return 1;
      }
        
      if (length % 2 == 1)
      {
        *dst_ptr-- = cuEVM::utils::byte_from_two_hex_char(
          '0',
          *current_char++
        );
      } else {
        *dst_ptr-- = cuEVM::utils::byte_from_two_hex_char(
          *(current_char),
          *(current_char+1)
        );
        current_char += 2;
      }
      while(*current_char != '\0') {
        *dst_ptr-- = cuEVM::utils::byte_from_two_hex_char(
          *(current_char),
          *(current_char+1)
        );
        current_char += 2;
      }
    }
    return 0;
  }

  __host__ __device__ int32_t byte_array_t::from_hex(
    const char *hex_string,
    int32_t endian,
    PaddingDirection padding,
    int32_t managed)
  {
    char *tmp_hex_char;
    tmp_hex_char = (char *)hex_string;
    int32_t length = cuEVM::utils::clean_hex_string(&tmp_hex_char);
    if (length <= 0)
    {
      data = nullptr;
      return;
    }
    size = (length + 1) / 2;
    if (managed) {
      CUDA_CHECK(cudaMallocManaged(
        (void **)&data,
        sizeof(uint8_t) * size));
      memset(data, 0, size);
    } else {
      data = (uint8_t*) std::calloc(size, sizeof(uint8_t));
    }
    int32_t error;
    if (endian == LITTLE_ENDIAN)
    {
      error = this->from_hex_set_le(tmp_hex_char, length);
    }
    else
    {
      error = this->from_hex_set_be(tmp_hex_char, length, padding);
    }
    if (error != 0)
    {
      if (managed) {
        CUDA_CHECK(cudaFree(data));
      } else {
        delete[] data;
      }
      data = nullptr;
      size = 0;
    }
  }

  __host__ __device__ int32_t byte_array_t::padded_copy_BE(
    const byte_array_t src
  ) {
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
    memcpy(data, src.data, copy_size);
    memset(data + copy_size, 0, size - src.size);
    return size_diff;
  }

  __host__ __device__ int32_t byte_array_t::to_bn_t(
    ArithEnv &arith,
    bn_t &out
  ) const {
    if (size != cuEVM::word_size)
      return ERROR_INVALID_WORD_SIZE;
    for (uint32_t idx = 0; idx < cuEVM::word_size; idx++)
    {
      cgbn_insert_bits_ui32(
        arith.env,
        out,
        out,
        cuEVM::word_bits - (idx + 1) * 8,
        8,
        data[idx]);
    }
    return ERROR_SUCCESS;
  }

  namespace byte_array {

    // CPU-GPU
    __global__ void transfer_kernel(
          byte_array_t *dst_instances,
          byte_array_t *src_instances,
          uint32_t count)
    {
      uint32_t instance = blockIdx.x * blockDim.x + threadIdx.x;

      if (instance >= count)
        return;

      dst_instances[instance].size = src_instances[instance].size;
      if (src_instances[instance].size > 0)
      {
        memcpy(
            dst_instances[instance].data,
            src_instances[instance].data,
            src_instances[instance].size * sizeof(uint8_t));
        delete[] src_instances[instance].data;
        src_instances[instance].data = nullptr;
      }
    }

    __host__ byte_array_t *get_cpu(
        uint32_t count)
    {
      byte_array_t *cpu_instances = new byte_array_t[count];
      for (size_t idx = 0; idx < count; idx++)
      {
        cpu_instances[idx].size = 0;
        cpu_instances[idx].data = nullptr;
      }
      return cpu_instances;
    }

    __host__ void cpu_free(
        byte_array_t *cpu_instances,
        uint32_t count)
    {
      for (size_t idx = 0; idx < count; idx++)
      {
        if (
            (cpu_instances[idx].size > 0) &&
            (cpu_instances[idx].data != nullptr))
        {
          delete[] cpu_instances[idx].data;
          cpu_instances[idx].size = 0;
          cpu_instances[idx].data = nullptr;
        }
      }
      delete[] cpu_instances;
    }

    __host__ byte_array_t *gpu_from_cpu(
        byte_array_t *cpu_instances,
        uint32_t count)
    {
      byte_array_t *gpu_instances, *tmp_cpu_instances;
      tmp_cpu_instances = new byte_array_t[count];
      memcpy(
          tmp_cpu_instances,
          cpu_instances,
          sizeof(byte_array_t) * count);
      for (size_t idx = 0; idx < count; idx++)
      {
        if (tmp_cpu_instances[idx].size > 0)
        {
          CUDA_CHECK(cudaMalloc(
              (void **)&tmp_cpu_instances[idx].data,
              sizeof(uint8_t) * tmp_cpu_instances[idx].size));
          CUDA_CHECK(cudaMemcpy(
              tmp_cpu_instances[idx].data,
              cpu_instances[idx].data,
              sizeof(uint8_t) * tmp_cpu_instances[idx].size,
              cudaMemcpyHostToDevice));
        }
      }
      CUDA_CHECK(cudaMalloc(
          (void **)&gpu_instances,
          sizeof(byte_array_t) * count));
      CUDA_CHECK(cudaMemcpy(
          gpu_instances,
          tmp_cpu_instances,
          sizeof(byte_array_t) * count,
          cudaMemcpyHostToDevice));
      delete[] tmp_cpu_instances;
      return gpu_instances;
    }

    __host__ void gpu_free(
        byte_array_t *gpu_instances,
        uint32_t count)
    {
      byte_array_t *cpu_instances = new byte_array_t[count];
      CUDA_CHECK(cudaMemcpy(
          cpu_instances,
          gpu_instances,
          sizeof(byte_array_t) * count,
          cudaMemcpyDeviceToHost));
      for (size_t idx = 0; idx < count; idx++)
      {
        if (cpu_instances[idx].size > 0)
        {
          CUDA_CHECK(cudaFree(cpu_instances[idx].data));
        }
      }
      delete[] cpu_instances;
      CUDA_CHECK(cudaFree(gpu_instances));
    }

    __host__ byte_array_t *cpu_from_gpu(
        byte_array_t *gpu_instances,
        uint32_t count)
    {
      byte_array_t *cpu_instances;
      cpu_instances = new byte_array_t[count];
      CUDA_CHECK(cudaMemcpy(
          cpu_instances,
          gpu_instances,
          sizeof(byte_array_t) * count,
          cudaMemcpyDeviceToHost));

      // 1. alocate the memory for gpu memory as memory which can be addressed by the cpu
      byte_array_t *tmp_cpu_instances, *tmp_gpu_instances;
      tmp_cpu_instances = new byte_array_t[count];
      memcpy(
          tmp_cpu_instances,
          cpu_instances,
          sizeof(byte_array_t) * count);
      for (uint32_t idx = 0; idx < count; idx++)
      {
        if (tmp_cpu_instances[idx].size > 0)
        {
          CUDA_CHECK(cudaMalloc(
              (void **)&tmp_cpu_instances[idx].data,
              sizeof(uint8_t) * tmp_cpu_instances[idx].size));
        }
        else
        {
          tmp_cpu_instances[idx].data = nullptr;
        }
      }
      CUDA_CHECK(cudaMalloc(
          (void **)&tmp_gpu_instances,
          sizeof(byte_array_t) * count));
      CUDA_CHECK(cudaMemcpy(
          tmp_gpu_instances,
          tmp_cpu_instances,
          sizeof(byte_array_t) * count,
          cudaMemcpyHostToDevice));
      delete[] tmp_cpu_instances;
      tmp_cpu_instances = nullptr;

      // 2. call the kernel to copy the memory between the gpu memories
      cuEVM::byte_array::transfer_kernel<<<count, 1>>>(tmp_gpu_instances, gpu_instances, count);
      CUDA_CHECK(cudaDeviceSynchronize());
      cudaFree(gpu_instances);
      gpu_instances = tmp_gpu_instances;
      tmp_gpu_instances = nullptr;

      // 3. copy the gpu memories back in the cpu memories
      CUDA_CHECK(cudaMemcpy(
        cpu_instances,
        gpu_instances,
        sizeof(byte_array_t)*count,
        cudaMemcpyDeviceToHost
      ));
      tmp_cpu_instances=new byte_array_t[count];
      memcpy(
        tmp_cpu_instances,
        cpu_instances,
        sizeof(byte_array_t)*count
      );
      for(size_t idx=0; idx<count; idx++) {
        if (tmp_cpu_instances[idx].size > 0)
        {
          tmp_cpu_instances[idx].data = new uint8_t[tmp_cpu_instances[idx].size];
          CUDA_CHECK(cudaMemcpy(
              tmp_cpu_instances[idx].data,
              cpu_instances[idx].data,
              sizeof(uint8_t) * tmp_cpu_instances[idx].size,
              cudaMemcpyDeviceToHost));
        }
        else
        {
          tmp_cpu_instances[idx].data = nullptr;
        }
      }

      // 4. free the temporary allocated memory
      cuEVM::byte_array::gpu_free(gpu_instances, count);
      delete[] cpu_instances;
      cpu_instances=tmp_cpu_instances;
      tmp_cpu_instances=nullptr;
      return cpu_instances;
    }
  } // namespace byte_array
} // namespace cuEVM