#ifndef _GPU_FUNCTIONRETURN_H_
#define _GPU_FUNCTIONRETURN_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>

typedef struct {
  size_t offset;
  size_t length;
} return_data_t;

__host__ return_data_t *generate_host_returns(uint32_t count) {
  return_data_t *cpu_instances=(return_data_t *)malloc(sizeof(return_data_t)*count);
  for(size_t idx=0; idx<count; idx++) {
    cpu_instances[idx].offset = 0;
    cpu_instances[idx].length = 32;
  }
  return cpu_instances;
}


__host__ void free_host_returns(return_data_t *cpu_instances, uint32_t count) {
  free(cpu_instances);
}

__host__ return_data_t *generate_gpu_returns(return_data_t *cpu_instances, uint32_t count) {
  return_data_t *gpu_instances;
  cudaMalloc((void **)&gpu_instances, sizeof(return_data_t)*count);
  cudaMemcpy(gpu_instances, cpu_instances, sizeof(return_data_t)*count, cudaMemcpyHostToDevice);
  return gpu_instances;
}


__host__ void free_gpu_returns(return_data_t *gpu_instances, uint32_t count) {
  cudaFree(gpu_instances);
}

template<class params>
__host__ void write_returns(FILE *fp, return_data_t *cpu_instances, uint32_t count) {
  for(size_t idx=0; idx<count; idx++) {
    fprintf(fp, "INSTACE: %08x , OFFSET: %lx , LENGTH: %lx\n", idx, cpu_instances[idx].offset, cpu_instances[idx].length);
  }
}

#endif