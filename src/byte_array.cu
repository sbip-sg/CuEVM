// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2024-06-20
// SPDX-License-Identifier: MIT

#include "include/byte_array.cuh"
#include "include/utils.h"

namespace cuEVM {
  namespace byte_array {
    __host__ __device__ char hex_from_nibble(const uint8_t nibble) {
      return nibble < 10 ? '0' + nibble : 'a' + nibble - 10;
    }
    
    __host__ __device__ uint8_t nibble_from_hex(const char hex) {
      return hex >= '0' && hex <= '9' ? hex - '0' : (
        hex >= 'a' && hex <= 'f' ? hex - 'a' + 10 : (
          hex >= 'A' && hex <= 'F' ? hex - 'A' + 10 : 0
        )
      );
    }

    __host__ __device__ int32_t is_hex(const char hex) {
      return hex >= '0' && hex <= '9' ? 1 : (
        hex >= 'a' && hex <= 'f' ? 1 : (
          hex >= 'A' && hex <= 'F' ? 1 : 0
        )
      );
    }
      
    __host__ __device__ uint8_t byte_from_nibbles(const uint8_t high, const uint8_t low) {
      return (high << 4) | low;
    }
    
    __host__ __device__ void hex_from_byte(char *dst, const uint8_t byte){
      if (dst == NULL)
        return;
      dst[0] = hex_from_nibble(byte >> 4);
      dst[1] = hex_from_nibble(byte & 0x0F);
    }


    __host__ __device__ uint8_t byte_from_two_hex_char(const char high, const char low) {
      return byte_from_nibbles(nibble_from_hex(high), nibble_from_hex(low));
    }

    __host__ char *hex_from_bytes(uint8_t *bytes, size_t count) {
      char *hex_string = new char[count*2+1];
      char *return_string = new char[count*2+1+2];
      for(size_t idx=0; idx<count; idx++)
        hex_from_byte(&hex_string[idx*2], bytes[idx]);
      hex_string[count*2]=0;
      memcpy(return_string + 2, hex_string, count*2+1);
      delete[] hex_string;
      hex_string = NULL;
      return_string[0]='0';
      return_string[1]='x';
      return return_string;
    }

    __host__ __device__ void print_bytes(uint8_t *bytes, size_t count) {
      printf("data: ");
      for(size_t idx=0; idx<count; idx++)
        printf("%02x", bytes[idx]);
      printf("\n");
    }

    __host__ __device__ void print_byte_array_t(byte_array_t &data_content) {
      printf("size: %lu\n", data_content.size);
      print_bytes(data_content.data, data_content.size);
    }

    __host__ __device__ char *hex_from_byte_array_t(byte_array_t &data_content) {
      return hex_from_bytes(data_content.data, data_content.size);
    }

    __host__ __device__ int32_t hex_string_length(
      const char *hex_string)
    {
      int32_t length;
      int32_t error = 0;
      char *current_char;
      current_char = (char *)hex_string;
      if (
        (hex_string[0] == '0') &&
        ((hex_string[1] == 'x') || (hex_string[1] == 'X'))
      ) {
        current_char += 2; // Skip the "0x" prefix
      }
      length = 0;
      do {
        length++;
        error = error | (nibble_from_hex(current_char[length]) == 0);
      } while(current_char[length] != '\0');
      return error ? -1 : length;
    }

    __host__ __device__ int32_t clean_hex_string(
      char **hex_string)
    {
      char *current_char;
      current_char = (char *)*hex_string;
      if (current_char == NULL || current_char[0] == '\0')
      {
        return 1;
      }
      if (
        (current_char[0] == '0') &&
        ((current_char[1] == 'x') || (current_char[1] == 'X'))
      ) {
        current_char += 2; // Skip the "0x" prefix
        *hex_string += 2;
      }
      int32_t length = 0;
      int32_t error = 0;
      do {
        error = error || (is_hex(current_char[length++]) == 0);
      } while(current_char[length] != '\0');
      return error ? -1 : length;
    }

    __host__ __device__ int32_t byte_array_t_from_hex_set_le(
      byte_array_t &dst,
      const char *clean_hex_string,
      int32_t length)
    {
      // clean the memory
      memset(dst.data, 0, dst.size * sizeof(uint8_t));
      if ( (length < 0) || ( (dst.size * 2) < length ) ) {
        return 1;
      }
      if (length > 0)
      {
        char *current_char;
        current_char = (char *)clean_hex_string;
        int32_t index;
        for (index = 0; index < ((length+1)/2) - 1; index++)
        {
          dst.data[index] = byte_from_two_hex_char(
            *(current_char),
            *(current_char+1)
          );
          current_char += 2;
        }
        if (length % 2 == 1)
        {
          dst.data[index] = byte_from_two_hex_char(
            *current_char++,
            '0'
          );
        } else {
          dst.data[index] = byte_from_two_hex_char(
            *(current_char),
            *(current_char+1)
          );
          current_char += 2;
        }
      }
      return 0;
    }

    __host__ __device__ int32_t byte_array_t_from_hex_set_be(
      byte_array_t &dst,
      const char *clean_hex_string,
      int32_t length,
      int32_t padded)
    {
      // clean the memory
      memset(dst.data, 0, dst.size * sizeof(uint8_t));
      if ( (length < 0) || ( (dst.size * 2) < length ) ) {
        return 1;
      }
      if (length > 0)
      {
        char *current_char;
        current_char = (char *)clean_hex_string;
        int32_t index;
        uint8_t *dst_ptr;
        if (padded == 1) { // right padding
          dst_ptr = dst.data + dst.size - 1;
        } else { // left padding
          dst_ptr = dst.data + (length + 1) / 2 - 1;
        }
         
        if (length % 2 == 1)
        {
          *dst_ptr-- = byte_from_two_hex_char(
            '0',
            *current_char++
          );
        } else {
          *dst_ptr-- = byte_from_two_hex_char(
            *(current_char),
            *(current_char+1)
          );
          current_char += 2;
        }
        while(*current_char != '\0') {
          *dst_ptr-- = byte_from_two_hex_char(
            *(current_char),
            *(current_char+1)
          );
          current_char += 2;
        }
      }
      return 0;
    }
  
  // TODO BE and LE
    __host__ __device__ int32_t byte_array_t_from_hex_le(
      byte_array_t &dst,
      const char *hex_string)
    {
      char *current_char;
      current_char = (char *)hex_string;
      int32_t length = clean_hex_string(&current_char);
      if (length < 0)
      {
        return 1;
      }
      // TODO: maybe if length is odd throw an error
      dst.size = (length + 1) / 2;
      if (dst.size > 0)
      {
        dst.data = new uint8_t[dst.size];
        return byte_array_t_from_hex_set_le(dst, current_char, length);
      }
      return 0;
    }

    __host__ __device__ int32_t byte_array_t_from_hex_be(
      byte_array_t &dst,
      const char *hex_string)
    {
      char *current_char;
      current_char = (char *)hex_string;
      int32_t length = clean_hex_string(&current_char);
      if (length < 0)
      {
        return 1;
      }
      // TODO: maybe if length is odd throw an error
      dst.size = (length + 1) / 2;
      if (dst.size > 0)
      {
        dst.data = new uint8_t[dst.size];
        return byte_array_t_from_hex_set_be(dst, current_char, length);
      }
      return 0;
    }

    __host__ __device__ void rm_leading_zero_hex_string(
      char *hex_string) {
      size_t length;
      char *current_char;
      current_char = (char *)hex_string;
      if (
        (hex_string[0] == '0') &&
        ((hex_string[1] == 'x') || (hex_string[1] == 'X'))
      ) {
        current_char += 2; // Skip the "0x" prefix
      }
      for (length = 0; current_char[length] != '\0'; length++)
        ;
      size_t idx;
      for (idx = 0; idx < length; idx++)
      {
        if (current_char[idx] != '0')
        {
          break;
        }
      }
      if (idx == length)
      {
        hex_string[2] = '0';
        hex_string[3] = '\0';
      }
      else
      {
        for (size_t i = 0; i < length - idx; i++)
        {
          current_char[i] = current_char[i + idx];
        }
        current_char[length - idx] = '\0';
      }
    }

    __host__ cJSON *json_from_byte_array_t(byte_array_t &data_content) {
      cJSON *data_json = cJSON_CreateObject();
      char *hex_string;
      //cJSON_AddNumberToObject(json, "size", data_content.size);
      if (data_content.size > 0)
      {
        hex_string = hex_from_byte_array_t(data_content);
        cJSON_AddStringToObject(data_json, "data", hex_string);
        delete[] hex_string;
      } else {
        cJSON_AddStringToObject(data_json, "data", "0x");
      }
      return data_json;
    }

    __host__ __device__ int32_t padded_copy_BE(
      const byte_array_t dst,
      const byte_array_t src
    ) {
      size_t copy_size;
      int32_t size_diff;
      if (src.size == dst.size) {
        size_diff = 0;
        copy_size = src.size;
      } else if (src.size < dst.size) {
        size_diff = 1;
        copy_size = src.size;
      } else {
        size_diff = -1;
        copy_size = dst.size;
      }
      memcpy(dst.data, src.data, copy_size);
      memset(dst.data + copy_size, 0, dst.size - src.size);
      return size_diff;
    }

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
        src_instances[instance].data = NULL;
      }
    }

    __host__ byte_array_t *get_cpu_instances(
        uint32_t count)
    {
      byte_array_t *cpu_instances = new byte_array_t[count];
      for (size_t idx = 0; idx < count; idx++)
      {
        cpu_instances[idx].size = 0;
        cpu_instances[idx].data = NULL;
      }
      return cpu_instances;
    }

    __host__ void free_cpu_instances(
        byte_array_t *cpu_instances,
        uint32_t count)
    {
      for (size_t idx = 0; idx < count; idx++)
      {
        if (
            (cpu_instances[idx].size > 0) &&
            (cpu_instances[idx].data != NULL))
        {
          delete[] cpu_instances[idx].data;
          cpu_instances[idx].size = 0;
          cpu_instances[idx].data = NULL;
        }
      }
      delete[] cpu_instances;
    }

    __host__ byte_array_t *get_gpu_instances_from_cpu_instances(
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

    __host__ void free_gpu_instances(
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

    __host__ byte_array_t *get_cpu_instances_from_gpu_instances(
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
          tmp_cpu_instances[idx].data = NULL;
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
      tmp_cpu_instances = NULL;

      // 2. call the kernel to copy the memory between the gpu memories
      transfer_kernel<<<count, 1>>>(tmp_gpu_instances, gpu_instances, count);
      CUDA_CHECK(cudaDeviceSynchronize());
      cudaFree(gpu_instances);
      gpu_instances = tmp_gpu_instances;
      tmp_gpu_instances = NULL;

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
          tmp_cpu_instances[idx].data = NULL;
        }
      }

      // 4. free the temporary allocated memory
      free_gpu_instances(gpu_instances, count);
      delete[] cpu_instances;
      cpu_instances=tmp_cpu_instances;
      tmp_cpu_instances=NULL;
      return cpu_instances;
    }
  } // namespace byte_array
} // namespace cuEVM