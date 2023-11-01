#ifndef _GPU_FUNCTIONRETURN_H_
#define _GPU_FUNCTIONRETURN_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>

template<class params>
struct gpu_function_return {
  uint32_t size;
  uint8_t *data;
};

template<class params>
__host__ gpu_function_return<params> *generate_host_function_return(uint32_t count) {
  gpu_function_return<params> *cpu_instances=(gpu_function_return<params> *)malloc(sizeof(gpu_function_return<params>)*count);
  for(uint8_t idx=0; idx<count; idx++)
    cpu_instances[idx].size = 0;
  return cpu_instances;
}


template<class params>
__host__ void free_host_messages(gpu_function_return<params> *cpu_instances, uint32_t count) {
  free(cpu_instances);
}

template<class params>
__host__ gpu_function_return<params> *generate_gpu_messages(gpu_function_return<params> *cpu_instances, uint32_t count) {
  gpu_function_return<params> *gpu_instances;
  cudaMalloc((void **)&gpu_instances, sizeof(gpu_function_return<params>)*count);
  cudaMemcpy(gpu_instances, cpu_instances, sizeof(instance_t)*count, cudaMemcpyHostToDevice);
  return gpu_instances;
}


template<class params>
__host__ void free_gpu_messages(gpu_function_return<params> *gpu_instances, uint32_t count) {
  cudaFree(gpu_instances);
}

template<class params>
__host__ void write_messages(FILE *fp, gpu_function_return<params> *cpu_instances, uint32_t count) {
  for(uint32_t idx=0; idx<count; idx++) {
    fprintf(fp, "INSTACE: %08x , RETURN_DATA_SIZE: %08x , RETURN_DATA: ", idx, cpu_instances[idx].size);
    for(uint32_t jdx=0; jdx<cpu_instances[idx].size; jdx++) {
      fprintf(fp, "%02x ", cpu_instances[idx].data[jdx]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp)
}

#endif