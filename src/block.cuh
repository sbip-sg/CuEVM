#ifndef _GPU_BLOCK_H_
#define _GPU_BLOCK_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <gmp.h>
#ifndef __CGBN_H__
#define __CGBN_H__
#include <cgbn/cgbn.h>
#endif

template<class params>
struct gpu_block {
  cgbn_mem_t<params::BITS> coin_base;
  cgbn_mem_t<params::BITS> time_stamp;
  cgbn_mem_t<params::BITS> number;
  cgbn_mem_t<params::BITS> dificulty;
  cgbn_mem_t<params::BITS> gas_limit;
  cgbn_mem_t<params::BITS> chain_id;
  cgbn_mem_t<params::BITS> base_fee;
};

template<class params>
__host__ gpu_block<params> *generate_cpu_blocks(uint32_t count) {
  gpu_block<params> *cpu_instances=(gpu_block<params> *)malloc(sizeof(gpu_block<params>)*count);

  for(uint32_t idx=0; idx<count; idx++) {
    cpu_instances[idx].coin_base._limbs[0]=1;
    cpu_instances[idx].time_stamp._limbs[0]=2;
    cpu_instances[idx].number._limbs[0]=3;
    cpu_instances[idx].dificulty._limbs[0]=4;
    cpu_instances[idx].gas_limit._limbs[0]=5;
    cpu_instances[idx].chain_id._limbs[0]=6;
    cpu_instances[idx].base_fee._limbs[0]=7;

    for(uint32_t jdx=1; jdx<params::BITS/32; jdx++) {
      cpu_instances[idx].coin_base._limbs[jdx]=0;
      cpu_instances[idx].time_stamp._limbs[jdx]=0;
      cpu_instances[idx].number._limbs[jdx]=0;
      cpu_instances[idx].dificulty._limbs[jdx]=0;
      cpu_instances[idx].gas_limit._limbs[jdx]=0;
      cpu_instances[idx].chain_id._limbs[jdx]=0;
      cpu_instances[idx].base_fee._limbs[jdx]=0;
    }
  }
  return cpu_instances;
}

template<class params>
__host__ void free_host_blocks(gpu_block<params> *cpu_instances, uint32_t count) {
  free(cpu_instances);
}

template<class params>
__host__ gpu_block<params> *generate_gpu_blocks(gpu_block<params> *cpu_instances, uint32_t count) {
  gpu_block<params> *gpu_instances;
  cudaMalloc((void **)&gpu_instances, sizeof(gpu_block<params>)*count);
  cudaMemcpy(gpu_instances, cpu_instances, sizeof(instance_t)*count, cudaMemcpyHostToDevice);
  return gpu_instances;
}

template<class params>
__host__ void free_gpu_blocks(gpu_block<params> *gpu_instances, uint32_t count) {
  cudaFree(gpu_instances);
}

template<class params>
__host__ void write_block(FILE *fp, gpu_block<params> *cpu_instances, uint32_t count) {
  for(uint32_t idx=0; idx<count; idx++) {
    fprintf(fp, "INSTACE: %08x , COINBASE: ", idx)
    for(uint32_t jdx=0; jdx<params::BITS/32; jdx++) {
      fprintf(fp, "%08x ", cpu_instances[idx].coin_base._limbs[jdx]);
    }
    fprintf(fp, ", TIMESTAMP: ");
    for(uint32_t jdx=0; jdx<params::BITS/32; jdx++) {
      fprintf(fp, "%08x ", cpu_instances[idx].time_stamp._limbs[jdx]);
    }
    fprintf(fp, ", NUMBER: ");
    for(uint32_t jdx=0; jdx<params::BITS/32; jdx++) {
      fprintf(fp, "%08x ", cpu_instances[idx].number._limbs[jdx]);
    }
    fprintf(fp, ", DIFICULTY: ");
    for(uint32_t jdx=0; jdx<params::BITS/32; jdx++) {
      fprintf(fp, "%08x ", cpu_instances[idx].dificulty._limbs[jdx]);
    }
    fprintf(fp, ", GASLIMIT: ");
    for(uint32_t jdx=0; jdx<params::BITS/32; jdx++) {
      fprintf(fp, "%08x ", cpu_instances[idx].gas_limit._limbs[jdx]);
    }
    fprintf(fp, ", CHAINID: ");
    for(uint32_t jdx=0; jdx<params::BITS/32; jdx++) {
      fprintf(fp, "%08x ", cpu_instances[idx].chain_id._limbs[jdx]);
    }
    fprintf(fp, ", BASE_FEE: ");
    for(uint32_t jdx=0; jdx<params::BITS/32; jdx++) {
      fprintf(fp, "%08x ", cpu_instances[idx].base_fee._limbs[jdx]);
    }
    fprintf(fp, "\n");
  }
}

#endif