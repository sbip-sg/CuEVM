#ifndef _GPU_MEMORY_H_
#define _GPU_MEMORY_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>

template<class params>
class gpu_fixed_memory_t {
  public:

  //storage data structure  
  typedef struct {
    uint8_t _data[params::MEMORY_SIZE];
    size_t _size;
  } fixed_memory_data_t;


  //copntent of the storage
  fixed_memory_data_t *_content;
  
  //constructor
  __device__ __forceinline__ gpu_fixed_memory_t(fixed_memory_data_t *content) : _content(content) {
  }

//get the size of the memory
  __device__ __forceinline__ size_t size() {
    return  _content->_size;
  }

  //get the all data of the memory
  __device__ __forceinline__ uint8_t *data() {
    return _content->_data;
  }

  //get the data of the memory at a specific index and length
  __device__ __forceinline__ uint8_t *get(size_t index, size_t length) {
    if (index + length >  _content->_size) {
      printf("Error: index out of range\n");
      return NULL;
    }
    return &(_content->_data[index]);
  }

  //set the data of the memory at a specific index and length
  __device__ __forceinline__ void set(size_t index, size_t length, uint8_t *data) {
    __shared__ uint32_t error;
    if(index + length > params::MEMORY_SIZE) {
      error=1;
      return;
    }
    if (index + length > _content->_size) {
       _content->_size=index+length;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      memcpy(&(_content->_data[index]), data, length);
    }
    __syncthreads();
  }


  // generate the memory content structure info on the host
  __host__ static fixed_memory_data_t *generate_memory_data(uint32_t count) {
    fixed_memory_data_t *cpu_instances = (fixed_memory_data_t *)malloc(sizeof(fixed_memory_data_t) * count);
    return cpu_instances;
  }

  __host__ static fixed_memory_data_t *generate_gpu_memory_data(fixed_memory_data_t *cpu_instances, uint32_t count) {
    fixed_memory_data_t *gpu_instances;
    cudaMalloc((void **)&gpu_instances, sizeof(fixed_memory_data_t)*count);
    cudaMemcpy(gpu_instances, cpu_instances, sizeof(fixed_memory_data_t)*count, cudaMemcpyHostToDevice);
    return gpu_instances;
  }

  
  __device__ __forceinline__ void copy_memory_data(fixed_memory_data_t *dest) {
    __syncthreads();
    if (threadIdx.x == 0) {
      memcpy(dest, _content, sizeof(fixed_memory_data_t));
    }
    __syncthreads();
  }


  // free the storage structure info on the host
  __host__ static void free_memory_data(fixed_memory_data_t *cpu_instances, uint32_t count) {
    free(cpu_instances);
  }
  // free the gpu storage structure info on the host
  __host__ static void free_gpu_memory_data(fixed_memory_data_t *gpu_instances, uint32_t count) {
    cudaFree(gpu_instances);
  }

  __host__ static void write_memory(FILE *fp, fixed_memory_data_t *cpu_instances, uint32_t count) {
    for(uint32_t idx=0; idx<count; idx++) {
      fprintf(fp, "INSTACE: %08x , DATA_SIZE: %lx , MEMORY_DATA: ", idx, cpu_instances[idx]._size);
      for(uint32_t jdx=0; jdx<cpu_instances[idx]._size; jdx++) {
        fprintf(fp, "%02x ", cpu_instances[idx]._data[jdx]);
      }
      fprintf(fp, "\n");
    }
  }

};

class gpu_memory_t {
  public:

  //memory data structure  
  typedef struct {
    size_t _size;
    uint8_t *_data;
  } memory_data_t;


  //copntent of the memory
  memory_data_t *_content;
  
  
  //constructor
  __device__ __forceinline__ gpu_memory_t(memory_data_t *content) : _content(content) {
  }

  //get the size of the memory
  __device__ __forceinline__ size_t size() {
    return _content->_size;
  }

  //get the all data of the memory
  __device__ __forceinline__ uint8_t *data() {
    return _content->_data;
  }

  //get the data of the memory at a specific index and length
  __device__ __forceinline__ uint8_t *get(size_t index, size_t length) {
    if (index + length > _content->_size) {
      printf("Error: index out of range\n");
      return NULL;
    }
    return _content->_data + index;
  }

  //set the data of the memory at a specific index and length
  __device__ __forceinline__ void set(size_t index, size_t length, uint8_t *data) {
    __shared__ uint32_t error;
    __syncthreads();
    if (index + length > _content->_size) {
      grow(index + length);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      memcpy(_content->_data + index, data, length);
    }
    __syncthreads();
  }

  __device__ __forceinline__ void grow(size_t new_size) {
    if (new_size <= _content->_size) {
      return;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      uint8_t *new_data = (uint8_t *)malloc(new_size);
      memcpy(new_data, _content->_data, _content->_size);
      free(_content->_data);
      _content->_data = new_data;
    }
    __syncthreads();
    _content->_size = new_size;
  }

  // copy the data information
  __device__ __forceinline__ void copy_content_info_to(memory_data_t *dest) {
    __syncthreads();
    if (threadIdx.x == 0) {
      dest->_size = _content->_size;
      dest->_data = _content->_data;
    }
    __syncthreads();
  }

  // copy content to another memory
  __device__ __forceinline__ void copy_content_to(memory_data_t *dest) {
    __syncthreads();
    if (threadIdx.x == 0) {
      dest->_size = _content->_size;
      memcpy(dest->_data, _content->_data, _content->_size);
    }
    __syncthreads();
  }

  __device__ __forceinline__ void free_memory() {
    __syncthreads();
    if (threadIdx.x == 0) {
      free(_content->_data);
    }
    __syncthreads();
  }

  // generate the memory content structure info on the host
  __host__ static memory_data_t *generate_memory_info_data(uint32_t count) {
    memory_data_t *cpu_instances = (memory_data_t *)malloc(sizeof(memory_data_t) * count);
    for (uint32_t idx = 0; idx < count; idx++) {
      cpu_instances[idx]._size = 0;
      cpu_instances[idx]._data = NULL;
    }
    return cpu_instances;
  }

  __host__ static memory_data_t *generate_gpu_memory_info_data(memory_data_t *cpu_instances, uint32_t count) {
    memory_data_t *gpu_instances;
    cudaMalloc((void **)&gpu_instances, sizeof(memory_data_t)*count);
    cudaMemcpy(gpu_instances, cpu_instances, sizeof(memory_data_t)*count, cudaMemcpyHostToDevice);
    return gpu_instances;
  }


  // free the memory content structure info on the host
  __host__ static void free_memory_info_data(memory_data_t *cpu_instances, uint32_t count) {
    free(cpu_instances);
  }

  //generate the memory content structure on the device from the info from gpu
  __host__ static memory_data_t *generate_memory_data(memory_data_t *cpu_gpu_instances, uint32_t count) {
    memory_data_t *cpu_instances = (memory_data_t *)malloc(sizeof(memory_data_t) * count);
    for (uint32_t idx = 0; idx < count; idx++) {
      cpu_instances[idx]._size = cpu_gpu_instances[idx]._size;
      cpu_instances[idx]._data = (uint8_t *)malloc(sizeof(uint8_t) * cpu_instances[idx]._size);
    }
    return cpu_instances;
  }

  //free the memory content structure on the device from the info from gpu
  __host__ static void free_memory_data(memory_data_t *cpu_instances, uint32_t count) {
    for (uint32_t idx = 0; idx < count; idx++) {
      free(cpu_instances[idx]._data);
    }
    free(cpu_instances);
  }

  //generate the memory content structure on the device from the info from cpu
  __host__ static memory_data_t *generate_gpu_memory_data(memory_data_t *cpu_instances, uint32_t count) {
    memory_data_t *gpu_instances;
    cudaMalloc((void **)&gpu_instances, sizeof(memory_data_t) * count);
    for (uint32_t idx = 0; idx < count; idx++) {
      cudaMalloc((void **)&cpu_instances[idx]._data, sizeof(uint8_t) * cpu_instances[idx]._size);
    }
    cudaMemcpy(gpu_instances, cpu_instances, sizeof(memory_data_t)*count, cudaMemcpyHostToDevice);
    return gpu_instances;
  }

  // copy the gpu values back in the cpu
  __host__ static void copy_gpu_memory_data_cpu(memory_data_t *cpu_instances, memory_data_t *gpu_instances, uint32_t count) {
    for (uint32_t idx = 0; idx < count; idx++) {
      cpu_instances[idx]._size = gpu_instances[idx]._size;
      cudaMemcpy(cpu_instances[idx]._data, gpu_instances[idx]._data, sizeof(uint8_t) * cpu_instances[idx]._size, cudaMemcpyDeviceToHost);
    }
  }

  //free the memory content structure on the device from the info from cpu
  __host__ static void free_gpu_memory_data(memory_data_t *gpu_instances, memory_data_t *cpu_instances, uint32_t count) {
    for (uint32_t idx = 0; idx < count; idx++) {
      cudaFree(cpu_instances[idx]._data);
    }
    cudaFree(gpu_instances);
  }
  
  __host__ static void write_messages(FILE *fp, memory_data_t *cpu_instances, uint32_t count) {
    for(uint32_t idx=0; idx<count; idx++) {
      fprintf(fp, "INSTACE: %08x , MEMORY_DATA_SIZE: %lx , MEMORY_DATA: ", idx, cpu_instances[idx]._size);
      for(uint32_t jdx=0; jdx<cpu_instances[idx]._size; jdx++) {
        fprintf(fp, "%02x ", cpu_instances[idx]._data[jdx]);
      }
      fprintf(fp, "\n");
    }
  }
  

};

typedef typename gpu_memory_t::memory_data_t memory_data_t;
  
__global__ void kernel_get_memory(memory_data_t *dst_instances, memory_data_t *src_instances, uint32_t instance_count) {
  uint32_t instance=blockIdx.x*blockDim.x + threadIdx.x;
  
  if(instance>=instance_count)
    return;

  if (threadIdx.x == 0) {
    printf("GET size=%08x\n", src_instances->_size);
    printf("GET data address=%08x\n", src_instances->_data);
    printf("GET lowestbit=%02x\n", src_instances->_data[31]);
  }
  gpu_memory_t  memory(&(src_instances[instance]));
  memory.copy_content_to(&(dst_instances[instance]));
  if (threadIdx.x == 0) {
    printf("GET data address=%08x\n", memory._content->_data);
    printf("GET lowestbit=%02x\n", memory._content->_data[31]);
  }
  memory.free_memory();
}


  __host__ void get_memory_from_gpu(memory_data_t  *cpu_instances, memory_data_t  *gpu_instances, uint32_t instance_count) {
    memory_data_t  *final_cpu_instaces, *final_gpu_instaces;
    
    // copy the instances back from gpuMemory
    printf("Copying results back to CPU for content info\n");
    cudaMemcpy(cpu_instances, gpu_instances, sizeof(memory_data_t)*instance_count, cudaMemcpyDeviceToHost);
    printf("RUN H D size=%lx\n", cpu_instances[0]._size);
    printf("RUN H D data address=%08x\n", cpu_instances[0]._data);

    printf("Generate the necesary CPU memory to take the values given the content info\n");
    final_cpu_instaces=gpu_memory_t::generate_memory_data(cpu_instances, instance_count);

    printf("Generate the necesary GPU memory to take the values given the content info\n");
    final_gpu_instaces=gpu_memory_t::generate_gpu_memory_data(cpu_instances, instance_count);

    printf("Running GPU GET kernel ...\n");
    kernel_get_memory<<<1, instance_count>>>(final_gpu_instaces, gpu_instances, instance_count);

    cudaDeviceSynchronize();
    printf("COPY the GPU final content info in the cpu contenet info\n");

    cudaMemcpy(cpu_instances, final_gpu_instaces, sizeof(memory_data_t)*instance_count, cudaMemcpyDeviceToHost);
    printf("Copy on CPU content the content on the GPU using the CPU content info\n");
    gpu_memory_t::copy_gpu_memory_data_cpu(final_cpu_instaces, cpu_instances, instance_count);
    printf("Write the memory data at stdout\n");
    gpu_memory_t::write_messages(stdout, final_cpu_instaces, instance_count);
    printf("Free the CPU content\n");
    gpu_memory_t::free_memory_data(final_cpu_instaces, instance_count);
    printf("Free the GPU content\n");
    gpu_memory_t::free_gpu_memory_data(final_gpu_instaces, cpu_instances, instance_count);
    // clean up
    printf("Cleaning up ...\n");
    printf("Free the CPU content info\n");
    free(cpu_instances);
    printf("Free the GPU content info\n");
    cudaFree(gpu_instances);
  }
#endif