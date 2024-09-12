// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#include <CuEVM/utils/error_codes.cuh>
#include <CuEVM/core/memory.cuh>

namespace CuEVM {
  namespace memory {
    __host__ __device__ void evm_memory_t::print() const {
      printf("Memory data: \n");
      printf("Size: %d\n", size);
      printf("Memory cost: ");
      memory_cost.print();
      printf("\n");
      data.print();
    }

    __host__ cJSON* evm_memory_t::to_json() const {
      cJSON *json = cJSON_CreateObject();
      cJSON_AddItemToObject(json, "size", cJSON_CreateNumber(size));
      char *hex_string_ptr = memory_cost.to_hex();
      cJSON_AddStringToObject(json, "memory_cost", hex_string_ptr);
      delete[] hex_string_ptr;
      cJSON_AddItemToObject(json, "data", data.to_json());
      return json;
    }

    __host__ __device__ void evm_memory_t::get_memory_cost(
      ArithEnv &arith,
      bn_t &cost) const {
        cgbn_load(
          arith.env,
          cost,
          (CuEVM::cgbn_evm_word_t_ptr) &memory_cost
        );
    }

    __host__ __device__ void evm_memory_t::increase_memory_cost(
      ArithEnv &arith,
      const bn_t &memory_expansion_cost) {
        bn_t cuurent_cost;
        cgbn_load(
          arith.env,
          cuurent_cost,
          (CuEVM::cgbn_evm_word_t_ptr) &memory_cost
        );
        cgbn_add(
          arith.env,
          cuurent_cost,
          cuurent_cost,
          memory_expansion_cost
        );
        cgbn_store(
          arith.env,
          (CuEVM::cgbn_evm_word_t_ptr) &memory_cost,
          cuurent_cost
        );
      }

    __host__ __device__ int32_t evm_memory_t::allocate_pages(
      uint32_t new_size
    )  {
      if (new_size < data.size) {
        return ERROR_SUCCESS;
      }
      uint32_t new_page_count = (new_size / CuEVM::memory::page_size) + 1;
      return data.grow(new_page_count * CuEVM::memory::page_size, 1);
    }

    __host__ __device__ int32_t evm_memory_t::get_last_offset(
      ArithEnv &arith,
      const bn_t &index,
      const bn_t &length,
      uint32_t &offset
    ) const {
      int32_t overflow = 0;
      bn_t offset_bn;
      overflow = cgbn_add(arith.env, offset_bn, index, length);
      overflow |= arith.uint32_t_from_cgbn(offset, offset_bn);
      bn_t memory_size;
      overflow |= cgbn_add_ui32(arith.env, memory_size, offset_bn, 31);
      cgbn_div_ui32(arith.env, memory_size, memory_size, 32);
      overflow |= cgbn_mul_ui32(arith.env, offset_bn, memory_size, 32);
      overflow |= arith.uint32_t_from_cgbn(offset, offset_bn);

      return overflow;
    }

    __host__ __device__ int32_t evm_memory_t::grow(
      ArithEnv &arith,
      const bn_t &index,
      const bn_t &length) {
      uint32_t offset;
      if(get_last_offset(arith, index, length, offset) != 0) {
        return ERR_MEMORY_INVALID_OFFSET;
      }
      if (offset > size) {
        if(allocate_pages(offset) != 0) {
          return ERR_MEMORY_INVALID_ALLOCATION;
        }
        size = offset;
      }
      return ERROR_SUCCESS;
    }

    __host__ __device__ int32_t evm_memory_t::get(
      ArithEnv &arith,
      const bn_t &index,
      const bn_t &length,
      CuEVM::byte_array_t &data) {
      int32_t error_code = ERROR_SUCCESS;
      error_code = (cgbn_compare_ui32(arith.env, length, 0) < 0) ? ERR_MEMORY_INVALID_SIZE : error_code;
      error_code |= grow(arith, index, length);
      if (error_code == ERROR_SUCCESS) {
        uint32_t index_u32, length_u32;
        arith.uint32_t_from_cgbn(index_u32, index);
        arith.uint32_t_from_cgbn(length_u32, length);
        data = CuEVM::byte_array_t(this->data.data + index_u32, length_u32);
      } else {
        data = CuEVM::byte_array_t();
      }
      return error_code;
    }

  __host__ __device__ int32_t evm_memory_t::set(
      ArithEnv &arith,
      const CuEVM::byte_array_t &data,
      const bn_t &index,
      const bn_t &length) {

      int32_t error_code = ERROR_SUCCESS;
      error_code = (cgbn_compare_ui32(arith.env, length, 0) < 0) ? ERR_MEMORY_INVALID_SIZE : error_code;
      error_code |= grow(arith, index, length);
      if (error_code == ERROR_SUCCESS) {
        uint32_t index_u32, length_u32;
        arith.uint32_t_from_cgbn(index_u32, index);
        arith.uint32_t_from_cgbn(length_u32, length);

        if (length_u32 && data.size) {
          // memcpy(this->data.data + index_u32, data.data, min(length_u32,data.size));
          std::copy(data.data, data.data + min(length_u32,data.size), this->data.data + index_u32);
        }
      }
      return error_code;
    }

    __global__ void transfer_kernel(
      evm_memory_t *dst_instances,
      evm_memory_t *src_instances,
      uint32_t instance_count
    ) {
      uint32_t instance = blockIdx.x * blockDim.x + threadIdx.x;
      if (instance >= instance_count)
        return;

      if (src_instances[instance].data.size > 0)
      {
        memcpy(
          dst_instances[instance].data.data,
          src_instances[instance].data.data,
          src_instances[instance].data.size * sizeof(uint8_t)
        );
        delete[] src_instances[instance].data.data;
        src_instances[instance].data.data = nullptr;
      }
    }

    __host__ evm_memory_t *get_cpu(
        uint32_t count
    ) {
        return new evm_memory_t[count];
    }


    __host__ void cpu_free(
        evm_memory_t* instances,
        uint32_t count
    ) {
      delete[] instances;
    }

    __host__ evm_memory_t *get_gpu_from_cpu(
        evm_memory_t *cpu_instances,
        uint32_t count
    ) {
        evm_memory_t *gpu_instances, *tmp_cpu_instances;
        CUDA_CHECK(cudaMalloc((void **)&gpu_instances, count * sizeof(evm_memory_t)));
        tmp_cpu_instances = new evm_memory_t[count];
        std::copy(cpu_instances, cpu_instances + count, tmp_cpu_instances);
        for (uint32_t i = 0; i < count; i++) {
            if (cpu_instances[i].data.size > 0) {
                CUDA_CHECK(cudaMalloc((void **)&tmp_cpu_instances[i].data.data, cpu_instances[i].data.size));
                CUDA_CHECK(cudaMemcpy(tmp_cpu_instances[i].data.data, cpu_instances[i].data.data, cpu_instances[i].data.size, cudaMemcpyHostToDevice));
            }
        }
        CUDA_CHECK(cudaMemcpy(tmp_cpu_instances, tmp_cpu_instances, count * sizeof(evm_memory_t), cudaMemcpyHostToDevice));
        delete[] tmp_cpu_instances;
        return gpu_instances;
    }

    __host__ void gpu_free(
        evm_memory_t *gpu_instances,
        uint32_t count
    ) {
        evm_memory_t *tmp_cpu_instances = new evm_memory_t[count];
        CUDA_CHECK(cudaMemcpy(tmp_cpu_instances, gpu_instances, count * sizeof(evm_memory_t), cudaMemcpyDeviceToHost));
        for (uint32_t i = 0; i < count; i++) {
            if (tmp_cpu_instances[i].data.size > 0) {
                CUDA_CHECK(cudaFree(tmp_cpu_instances[i].data.data));
            }
        }
        delete[] tmp_cpu_instances;
        CUDA_CHECK(cudaFree(gpu_instances));
    }

    __host__ evm_memory_t *get_cpu_from_gpu(
      evm_memory_t *gpu_instances,
      uint32_t count
    ) {
      evm_memory_t *cpu_instances = new evm_memory_t[count];
      evm_memory_t *tmp_cpu_instances = new evm_memory_t[count];
      evm_memory_t* tmp_gpu_instances = nullptr;
      CUDA_CHECK(cudaMemcpy(cpu_instances, gpu_instances, count * sizeof(evm_memory_t), cudaMemcpyDeviceToHost));
      std::copy(cpu_instances, cpu_instances + count, tmp_cpu_instances);
      for (uint32_t idx = 0; idx < count; idx++) {
        if (tmp_cpu_instances[idx].size > 0)
        {
          CUDA_CHECK(cudaMalloc(
            (void **)&tmp_cpu_instances[idx].data.data,
            sizeof(uint8_t) * tmp_cpu_instances[idx].data.size
          ));
        }
        else
        {
          tmp_cpu_instances[idx].data.data = nullptr;
        }
        tmp_cpu_instances[idx].size = tmp_cpu_instances[idx].size;
      }
      CUDA_CHECK(cudaMalloc((void **)&tmp_gpu_instances, count * sizeof(evm_memory_t)));
      CUDA_CHECK(cudaMemcpy(tmp_gpu_instances, tmp_cpu_instances, count * sizeof(evm_memory_t), cudaMemcpyHostToDevice));
      delete[] tmp_cpu_instances;
      // 2. call the kernel to copy the memory between the gpu memories
      CuEVM::memory::transfer_kernel<<<count, 1>>>(tmp_gpu_instances, gpu_instances, count);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaFree(gpu_instances));
      gpu_instances = tmp_gpu_instances;

      // 3. copy the gpu memories back in the cpu memories
      CUDA_CHECK(cudaMemcpy(
        cpu_instances,
        gpu_instances,
        sizeof(evm_memory_t) * count,
        cudaMemcpyDeviceToHost
      ));
      tmp_cpu_instances = new evm_memory_t[count];
      std::copy(cpu_instances, cpu_instances + count, tmp_cpu_instances);
      for (size_t idx = 0; idx < count; idx++)
      {
        if (tmp_cpu_instances[idx].data.size > 0)
        {
          tmp_cpu_instances[idx].data.data = new uint8_t[tmp_cpu_instances[idx].data.size];
          CUDA_CHECK(cudaMemcpy(
            tmp_cpu_instances[idx].data.data,
            cpu_instances[idx].data.data,
            sizeof(uint8_t) * tmp_cpu_instances[idx].data.size,
            cudaMemcpyDeviceToHost
          ));
        }
        else
        {
          tmp_cpu_instances[idx].data.data = nullptr;
        }
      }
      gpu_free(gpu_instances, count);
      std::copy(tmp_cpu_instances, tmp_cpu_instances + count, cpu_instances);
      delete[] tmp_cpu_instances;
      tmp_cpu_instances = NULL;
      return cpu_instances;
    }
  }
}