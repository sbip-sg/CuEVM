// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#ifndef _RETURN_DATA_H_
#define _RETURN_DATA_H_

#include "include/data_content.cuh"

/**
 * Copy data content between two device memories
 * @param[out] dst_instances the destination memory
 * @param[in] src_instances the source memory
 * @param[in] count the number of instances to copy
*/
__global__ void kernel_get_returns(
    data_content_t *dst_instances,
    data_content_t *src_instances,
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

/**
 * The return data class. (YP: \f$H_{return}(\mu)=H(\mu, I)\f$)
*/
class return_data_t
{
public:
  data_content_t *_content; /**< The content of the return data*/

  /**
   * The constructor with the given content
   * @param[in] content the content of the return data
  */
  __host__ __device__ __forceinline__ return_data_t(
      data_content_t *content) : _content(content) {}

  /**
   * The cosntrctuor without the content
  */
  __host__ __device__ __forceinline__ return_data_t()
  {
    SHARED_MEMORY data_content_t *tmp_content;
    ONE_THREAD_PER_INSTANCE(
      tmp_content = new data_content_t;
      tmp_content->size = 0;
      tmp_content->data = NULL;)
    _content = tmp_content;
  }

  /**
   * The destructor
  */
  __host__ __device__ __forceinline__ ~return_data_t()
  {
    ONE_THREAD_PER_INSTANCE(
      if (
          (_content->size > 0) &&
          (_content->data != NULL)
      )
      {
        delete[] _content->data;
        _content->size = 0;
        _content->data = NULL;
      }
      delete _content;
    )
    _content = NULL;
  }

  /**
   * Get the size of the return data
   * @return the size of the return data
  */
  __host__ __device__ __forceinline__ size_t size()
  {
    return _content->size;
  }

  /**
   * Get the content of the return data
   * @param[in] index the index of in the return data
   * @param[in] size the size of the content
   * @param[out] error_code the error code
   * @return the pointer in the return data
  */
  __host__ __device__ __forceinline__ uint8_t *get(
      size_t index,
      size_t size,
      uint32_t &error_code)
  {
    size_t request_size = index + size;
    if ((request_size < index) || (request_size < size))
    {
      error_code = ERROR_RETURN_DATA_OVERFLOW;
      return _content->data;
    }
    else if (request_size > _content->size)
    {
      error_code = ERROR_RETURN_DATA_INVALID_SIZE;
      return _content->data;
    }
    else
    {
      return _content->data + index;
    }
  }

  /**
   * Get the content of the return data
   * @return the pointer in the return data
  */
  __host__ __device__ __forceinline__ data_content_t *get_data()
  {
    return _content;
  }

  /**
   * Set the content of the return data
   * @param[in] data the data to be set
   * @param[in] size the size of the data
  */
  __host__ __device__ __forceinline__ void set(
      uint8_t *data,
      size_t size)
  {
    ONE_THREAD_PER_INSTANCE(
        if (_content->size > 0) {
          delete[] _content->data;
        } if (size > 0) {
          _content->data = new uint8_t[size];
          memcpy(_content->data, data, size);
        })
    _content->size = size;
  }

  __host__ __device__ __forceinline__ void to_data_content_t(
      data_content_t &data_content)
  {
    ONE_THREAD_PER_INSTANCE(
        if (data_content.size > 0) {
          delete[] data_content.data;
          data_content.data = NULL;
          data_content.size = 0;
        }
        if (_content->size > 0) {
          data_content.data = new uint8_t[_content->size];
          memcpy(data_content.data, _content->data, _content->size);
        } else {
          data_content.data = NULL;
        })
    data_content.size = _content->size;
  }

  /**
   * Get the cpu instances for the return data
   * @param[in] count the number of instances
   * @return the cpu instances
  */
  __host__ static data_content_t *get_cpu_instances(
      uint32_t count)
  {
    data_content_t *cpu_instances = new data_content_t[count];
    for (size_t idx = 0; idx < count; idx++)
    {
      cpu_instances[idx].size = 0;
      cpu_instances[idx].data = NULL;
    }
    return cpu_instances;
  }

  /**
   * Free the cpu instances
   * @param[in] cpu_instances the cpu instances
   * @param[in] count the number of instances
  */
  __host__ static void free_cpu_instances(
      data_content_t *cpu_instances,
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

  /**
   * Get the gpu instances for the return data from the cpu instances
   * @param[in] cpu_instances the cpu instances
   * @param[in] count the number of instances
   * @return the gpu instances
  */
  __host__ static data_content_t *get_gpu_instances_from_cpu_instances(
      data_content_t *cpu_instances,
      uint32_t count)
  {
    data_content_t *gpu_instances, *tmp_cpu_instances;
    tmp_cpu_instances = new data_content_t[count];
    memcpy(
        tmp_cpu_instances,
        cpu_instances,
        sizeof(data_content_t) * count);
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
        sizeof(data_content_t) * count));
    CUDA_CHECK(cudaMemcpy(
        gpu_instances,
        tmp_cpu_instances,
        sizeof(data_content_t) * count,
        cudaMemcpyHostToDevice));
    delete[] tmp_cpu_instances;
    return gpu_instances;
  }

  /**
   * Free the gpu instances
   * @param[in] gpu_instances the gpu instances
   * @param[in] count the number of instances
  */
  __host__ static void free_gpu_instances(
      data_content_t *gpu_instances,
      uint32_t count)
  {
    data_content_t *cpu_instances = new data_content_t[count];
    CUDA_CHECK(cudaMemcpy(
        cpu_instances,
        gpu_instances,
        sizeof(data_content_t) * count,
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

  /**
   * Get the cpu instances from the gpu instances
   * @param[in] gpu_instances the gpu instances
   * @param[in] count the number of instances
   * @return the cpu instances
  */
  __host__ static data_content_t *get_cpu_instances_from_gpu_instances(
      data_content_t *gpu_instances,
      uint32_t count)
  {
    data_content_t *cpu_instances;
    cpu_instances = new data_content_t[count];
    CUDA_CHECK(cudaMemcpy(
        cpu_instances,
        gpu_instances,
        sizeof(data_content_t) * count,
        cudaMemcpyDeviceToHost));

    // 1. alocate the memory for gpu memory as memory which can be addressed by the cpu
    data_content_t *tmp_cpu_instances, *tmp_gpu_instances;
    tmp_cpu_instances = new data_content_t[count];
    memcpy(
        tmp_cpu_instances,
        cpu_instances,
        sizeof(data_content_t) * count);
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
        sizeof(data_content_t) * count));
    CUDA_CHECK(cudaMemcpy(
        tmp_gpu_instances,
        tmp_cpu_instances,
        sizeof(data_content_t) * count,
        cudaMemcpyHostToDevice));
    delete[] tmp_cpu_instances;
    tmp_cpu_instances = NULL;

    // 2. call the kernel to copy the memory between the gpu memories
    kernel_get_returns<<<count, 1>>>(tmp_gpu_instances, gpu_instances, count);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(gpu_instances);
    gpu_instances = tmp_gpu_instances;
    tmp_gpu_instances = NULL;

    // 3. copy the gpu memories back in the cpu memories
    CUDA_CHECK(cudaMemcpy(
      cpu_instances,
      gpu_instances,
      sizeof(data_content_t)*count,
      cudaMemcpyDeviceToHost
    ));
    tmp_cpu_instances=new data_content_t[count];
    memcpy(
      tmp_cpu_instances,
      cpu_instances,
      sizeof(data_content_t)*count
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

  /**
   * Print the return data
  */
  __host__ __device__ void print()
  {
    print_data_content_t(*_content);
  }

  /**
   * Get the json representation of the return data
   * @return the json representation of the return data
  */
  __host__ cJSON *json()
  {
    return json_from_data_content_t(*_content);
  }
};
#endif