#ifndef _GPU_MESSAGE_H_
#define _GPU_MESSAGE_H_

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
struct gpu_tx {
  cgbn_mem_t<params::BITS> origin;
  cgbn_mem_t<params::BITS> gasprice;
};

template<class params>
struct gpu_contract {
  cgbn_mem_t<params::BITS> address;
  uint32_t size;
  uint8_t *bytecode;
};

template<class params>
struct gpu_message_data {
  uint32_t size;
  uint8_t *data;
};

template<class params>
struct gpu_message {
  cgbn_mem_t<params::BITS> caller;
  cgbn_mem_t<params::BITS> value;
  cgbn_mem_t<params::BITS> to;
  gpu_tx<params> tx;
  gpu_message_data<params> data;
  gpu_contract<params> contract;
};

template<class params>
__host__ gpu_message<params> *generate_host_messages(uint32_t count) {
  gpu_message<params> *cpu_instances=(gpu_message<params> *)malloc(sizeof(gpu_message<params>)*count);

  // TODO: maybe later change with reading from the files
  for(uint32_t idx=0; idx<count; idx++) {
    // message info
    cpu_instances[idx].caller._limbs[0]=1;
    cpu_instances[idx].value._limbs[0]=2;
    cpu_instances[idx].to._limbs[0]=3;
    // tx
    cpu_instances[idx].tx.origin._limbs[0]=4;
    cpu_instances[idx].tx.gasprice._limbs[0]=5;
    // data
    cpu_instances[idx].data.size=4;
    cpu_instances[idx].data.data=(uint8_t *) malloc (cpu_instances[instance].data.size);
    // contract
    cpu_instances[idx].contract.address._limbs[0]=5;
    cpu_instances[idx].contract.size=6;
    cpu_instances[idx].contract.bytecode=(uint8_t *) malloc (cpu_instances[instance].contract.size);
  
    for(uint32_t jdx=1; jdx<params::BITS/32; jdx++) {
      // message info
      cpu_instances[idx].caller._limbs[jdx]=0;
      cpu_instances[idx].value._limbs[jdx]=0;
      cpu_instances[idx].to._limbs[jdx]=0;
      // tx
      cpu_instances[idx].tx.origin._limbs[jdx]=0;
      cpu_instances[idx].tx.gasprice._limbs[jdx]=0;
      // contract
      cpu_instances[idx].contract.address._limbs[jdx]=0;
    }
    // data
    for(uint32_t jdx=0; jdx<cpu_instances[idx].data.size; jdx++) {
      cpu_instances[idx].data.data[jdx]=(uint8_t)jdx;
    }
    // contract
    for(uint32_t jdx=0; jdx<cpu_instances[idx].contract.size; jdx++) {
      cpu_instances[idx].contract.bytecode[jdx]=(uint8_t)jdx;
    }
  }
  return cpu_instances;
}


template<class params>
__host__ void free_host_messages(gpu_message<params> *cpu_instances, uint32_t count) {

  for(uint32_t idx=0; idx<count; idx++) {
    // data
    free(cpu_instances[idx].data.data);
    // contract
    free(cpu_instances[idx].contract.bytecode);
  }
  free(cpu_instances);
}

template<class params>
__host__ gpu_message<params> *generate_gpu_messages(gpu_message<params> *cpu_instances, uint32_t count) {
  gpu_message<params> *gpu_instances;
  cudaMalloc((void **)&gpu_instances, sizeof(gpu_message<params>)*count);
  cudaMemcpy(gpu_instances, cpu_instances, sizeof(instance_t)*count, cudaMemcpyHostToDevice);

  for(uint32_t idx=0; idx<count; idx++) {
    // data
    cudaMalloc((void **)&gpu_instances[idx].data.data, sizeof(uint8_t)*cpu_instances[idx].data.size);
    cudaMemcpy(gpu_instances[idx].data.data, cpu_instances[idx].data.data, sizeof(uint8_t)*cpu_instances[idx].data.size, cudaMemcpyHostToDevice);
    // contract
    cudaMalloc((void **)&gpu_instances[idx].contract.bytecode, sizeof(uint8_t)*cpu_instances[idx].contract.size);
    cudaMemcpy(gpu_instances[idx].contract.bytecode, cpu_instances[idx].contract.bytecode, sizeof(uint8_t)*cpu_instances[idx].contract.size, cudaMemcpyHostToDevice);
  }
  return gpu_instances;
}


template<class params>
__host__ void free_gpu_messages(gpu_message<params> *gpu_instances, uint32_t count) {

  for(uint32_t idx=0; idx<count; idx++) {
    // data
    cudaFree(gpu_instances[idx].data.data);
    // contract
    cudaFree(gpu_instances[idx].contract.bytecode);
  }
  cudaFree(gpu_instances);
}

template<class params>
__host__ void write_messages(FILE *fp, gpu_message<params> *cpu_instances, uint32_t count) {
  for(uint32_t idx=0; idx<count; idx++) {
    fprintf(fp, "INSTACE: %08x , CALLER: ", idx)
    for(uint32_t jdx=0; jdx<params::BITS/32; jdx++) {
      fprintf(fp, "%08x ", cpu_instances[idx].caller._limbs[jdx]);
    }
    fprintf(fp, ", VALUE: ");
    for(uint32_t jdx=0; jdx<params::BITS/32; jdx++) {
      fprintf(fp, "%08x ", cpu_instances[idx].value._limbs[jdx]);
    }
    fprintf(fp, ", TO: ");
    for(uint32_t jdx=0; jdx<params::BITS/32; jdx++) {
      fprintf(fp, "%08x ", cpu_instances[idx].to._limbs[jdx]);
    }
    fprintf(fp, ", TX_ORIGIN: ");
    for(uint32_t jdx=0; jdx<params::BITS/32; jdx++) {
      fprintf(fp, "%08x ", cpu_instances[idx].tx.origin._limbs[jdx]);
    }
    fprintf(fp, ", TX_GASPRICE: ");
    for(uint32_t jdx=0; jdx<params::BITS/32; jdx++) {
      fprintf(fp, "%08x ", cpu_instances[idx].tx.gasprice._limbs[jdx]);
    }
    fprintf(fp, ", DATA_SIZE: ");
    fprintf(fp, "%08x ", cpu_instances[idx].data.size);
    fprintf(fp, ", DATA: ");
    for(uint32_t jdx=0; jdx<cpu_instances[idx].data.size; jdx++) {
      fprintf(fp, "%02x ", cpu_instances[idx].data.data[jdx]);
    }
    fprintf(fp, ", CONTRACT_ADDRESS: ");
    for(uint32_t jdx=0; jdx<params::BITS/32; jdx++) {
      fprintf(fp, "%08x ", cpu_instances[idx].contract.address._limbs[jdx]);
    }
    fprintf(fp, ", CONTRACT_SIZE: ");
    fprintf(fp, "%08x ", cpu_instances[idx].contract.size);
    fprintf(fp, ", CONTRACT_BYTECODE: ");
    for(uint32_t jdx=0; jdx<cpu_instances[idx].contract.size; jdx++) {
      fprintf(fp, "%02x ", cpu_instances[idx].contract.bytecode[jdx]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp)
}

#endif