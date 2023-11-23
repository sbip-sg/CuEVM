#ifndef _RETURNDATA_H_
#define _RETURNDATA_H_

#include "utils.h"


__global__ void kernel_get_returns(data_content_t *dst_instances, data_content_t *src_instances, uint32_t instance_count) {
  uint32_t instance=blockIdx.x*blockDim.x + threadIdx.x;
  
  if(instance>=instance_count)
    return;

  dst_instances[instance].size=src_instances[instance].size;
  if (src_instances[instance].size > 0) {
    memcpy(dst_instances[instance].data, src_instances[instance].data, src_instances[instance].size);
    free(src_instances[instance].data);
  }
}


class return_data_t {
  public:

  data_content_t *_content;

  __host__ __device__ __forceinline__ return_data_t(data_content_t *content) : _content(content) {}
  
  __host__ __device__ __forceinline__ size_t size() {
    return _content->size;
  }
  __host__ __device__ __forceinline__ uint8_t *get(size_t index, size_t size, uint32_t &error_code) {
    size_t request_size = index+size;
    if ( (request_size > index) || (request_size > size) ) {
      error_code=ERR_RETURN_DATA_OVERFLOW;
      return _content->data;
    } else if (request_size > _content->size) {
      error_code=ERR_RETURN_DATA_INVALID_SIZE;
      return _content->data;
    } else {
      return _content->data+index;
    }
  }

  __host__ __device__ __forceinline__ void set(uint8_t *data, size_t size) {    
    #ifdef __CUDA_ARCH__
    if (threadIdx.x == 0) {
    #endif
    if (_content->size > 0) {
      free(_content->data);
    }
    if (size > 0) {
        _content->data = (uint8_t *)malloc(size);
        memcpy(_content->data, data, size);
    }
    #ifdef __CUDA_ARCH__
    }
    __syncthreads();
    #endif
    _content->size = size;
  }

  __host__ static data_content_t *get_returns(uint32_t count) {
    data_content_t *cpu_instances=(data_content_t *)malloc(sizeof(data_content_t)*count);
    for(size_t idx=0; idx<count; idx++) {
      cpu_instances[idx].size = 0;
      cpu_instances[idx].data = NULL;
    }
    return cpu_instances;
  }


  __host__ static void free_host_returns(data_content_t *cpu_instances, uint32_t count) {
    for(size_t idx=0; idx<count; idx++) {
      if (cpu_instances[idx].size > 0) {
        free(cpu_instances[idx].data);
      }
    }
    free(cpu_instances);
  }

  __host__ static data_content_t *get_gpu_returns(data_content_t *cpu_instances, uint32_t count) {
    data_content_t *gpu_instances, *tmp_cpu_instances;
    tmp_cpu_instances=(data_content_t *)malloc(sizeof(data_content_t)*count);
    memcpy(tmp_cpu_instances, cpu_instances, sizeof(data_content_t)*count);
    for(size_t idx=0; idx<count; idx++) {
      if (tmp_cpu_instances[idx].size > 0) {
        cudaMalloc((void **)&tmp_cpu_instances[idx].data, sizeof(uint8_t) * tmp_cpu_instances[idx].size);
      }
    }
    cudaMalloc((void **)&gpu_instances, sizeof(data_content_t)*count);
    cudaMemcpy(gpu_instances, tmp_cpu_instances, sizeof(data_content_t)*count, cudaMemcpyHostToDevice);
    free(tmp_cpu_instances);
    return gpu_instances;
  }

  __host__ static void free_gpu_returns(data_content_t *gpu_instances, uint32_t count) {
    data_content_t *cpu_instances=(data_content_t *)malloc(sizeof(data_content_t)*count);
    cudaMemcpy(cpu_instances, gpu_instances, sizeof(data_content_t)*count, cudaMemcpyDeviceToHost);
    for(size_t idx=0; idx<count; idx++) {
      if (cpu_instances[idx].size > 0) {
        cudaFree(cpu_instances[idx].data);
      }
    }
    free(cpu_instances);
    cudaFree(gpu_instances);
  }

  __host__ static data_content_t *get_cpu_returns_from_gpu(data_content_t *gpu_instances, uint32_t count) {
    data_content_t *cpu_instances;
    cpu_instances=(data_content_t *)malloc(sizeof(data_content_t)*count);
    cudaMemcpy(cpu_instances, gpu_instances, sizeof(data_content_t)*count, cudaMemcpyDeviceToHost);

    // 1. alocate the memory for gpu memory as memory which can be addressed by the cpu
    data_content_t  *tmp_cpu_instances, *new_gpu_instances;
    new_gpu_instances=get_gpu_returns(cpu_instances, count);
    
    // 2. call the kernel to copy the memory between the gpu memories
    kernel_get_returns<<<1, count>>>(new_gpu_instances, gpu_instances, count);
    cudaFree(gpu_instances);
    gpu_instances=new_gpu_instances;

    // 3. copy the gpu memories back in the cpu memories
    cudaMemcpy(cpu_instances, gpu_instances, sizeof(data_content_t)*count, cudaMemcpyDeviceToHost);
    tmp_cpu_instances=(data_content_t *)malloc(sizeof(data_content_t) * count);
    memcpy(tmp_cpu_instances, cpu_instances, sizeof(data_content_t)*count);
    for(size_t idx=0; idx<count; idx++) {
      if (tmp_cpu_instances[idx].size > 0) {
        tmp_cpu_instances[idx].data=(uint8_t *)malloc(sizeof(uint8_t) * tmp_cpu_instances[idx].size);
        cudaMemcpy(tmp_cpu_instances[idx].data, cpu_instances[idx].data, sizeof(uint8_t) * tmp_cpu_instances[idx].size, cudaMemcpyDeviceToHost);
      } else {
        tmp_cpu_instances[idx].data = NULL;
      }
    }

    // 4. free the temporary allocated memory
    free_gpu_returns(gpu_instances, count);
    free(cpu_instances);
    cpu_instances=tmp_cpu_instances;
    return cpu_instances;
  }

  __host__ __device__ void print() {
    printf("size=%lx\n", _content->size);
    printf("data: ");
    if (_content->size > 0)
      print_bytes(_content->data, _content->size);
    printf("\n");
  }
  
  __host__ cJSON *to_json() {
    char *bytes_string=NULL;
    cJSON *data_json = cJSON_CreateObject();
    
    cJSON_AddNumberToObject(data_json, "size", _content->size);

    if (_content->size > 0) {
      bytes_string = bytes_to_hex(_content->data, _content->size);
      cJSON_AddStringToObject(data_json, "data", bytes_string);
      free(bytes_string);
    } else {
      cJSON_AddStringToObject(data_json, "data", "0x");
    }
    
    return data_json;
  }

};
#endif