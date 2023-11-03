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
#include "contract.cuh"
#include "utils.h"

template<class params>
struct gpu_tx {
  cgbn_mem_t<params::BITS> origin;
  cgbn_mem_t<params::BITS> gasprice;
};

template<class params>
struct gpu_message_data {
  size_t size;
  uint8_t *data;
};

template<class params>
struct gpu_message {
  cgbn_mem_t<params::BITS> caller;
  cgbn_mem_t<params::BITS> value;
  cgbn_mem_t<params::BITS> to;
  gpu_tx<params> tx;
  cgbn_mem_t<params::BITS> gas;
  uint32_t depth;
  uint32_t call_type;
  gpu_message_data<params> data;
  typename gpu_global_storage_t<params>::gpu_contract_t *contract;
};

template<class params>
__host__ gpu_message<params> *generate_host_messages(uint32_t count) {
  gpu_message<params> *cpu_instances=(gpu_message<params> *)malloc(sizeof(gpu_message<params>)*count);
  mpz_t caller, value, to, tx_origin, tx_gasprice;
  char hex_string_value[params::BITS/4+1]; // +1 for null terminator
  mpz_init(caller);
  mpz_init(value);
  mpz_init(to);
  mpz_init(tx_origin);
  mpz_init(tx_gasprice);

  for(size_t idx=0; idx<count; idx++) {
    // message info
    snprintf(hex_string_value, params::BITS/4+1, "%lx", idx+257);
    mpz_set_str(caller, hex_string_value, 16);
    snprintf(hex_string_value, params::BITS/4+1, "%lx", idx+1);
    mpz_set_str(value, hex_string_value, 16);
    snprintf(hex_string_value, params::BITS/4+1, "%lx", idx+2);
    mpz_set_str(to, hex_string_value, 16);
    from_mpz(cpu_instances[idx].caller._limbs, params::BITS/32, caller);
    from_mpz(cpu_instances[idx].value._limbs, params::BITS/32, value);
    from_mpz(cpu_instances[idx].to._limbs, params::BITS/32, to);
    // tx
    snprintf(hex_string_value, params::BITS/4+1, "%lx", idx+4);
    mpz_set_str(tx_origin, hex_string_value, 16);
    snprintf(hex_string_value, params::BITS/4+1, "%lx", idx+5);
    mpz_set_str(tx_gasprice, hex_string_value, 16);
    from_mpz(cpu_instances[idx].tx.origin._limbs, params::BITS/32, tx_origin);
    from_mpz(cpu_instances[idx].tx.gasprice._limbs, params::BITS/32, tx_gasprice);
    // depth and call_type
    cpu_instances[idx].depth=0;
    cpu_instances[idx].call_type=0;
    // data
    cpu_instances[idx].data.size=idx % 30 + 1;
    cpu_instances[idx].data.data=(uint8_t *) malloc (cpu_instances[idx].data.size);
    for(uint32_t jdx=0; jdx<cpu_instances[idx].data.size; jdx++) {
      cpu_instances[idx].data.data[jdx]=(uint8_t)jdx;
    }
    // contract
    cpu_instances[idx].contract = NULL;
  }
  mpz_clear(caller);
  mpz_clear(value);
  mpz_clear(to);
  mpz_clear(tx_origin);
  mpz_clear(tx_gasprice);
  return cpu_instances;
}


template<class params>
__host__ void free_host_messages(gpu_message<params> *cpu_instances, uint32_t count) {

  for(uint32_t idx=0; idx<count; idx++) {
    // data
    free(cpu_instances[idx].data.data);
  }
  free(cpu_instances);
}

template<class params>
__host__ gpu_message<params> *generate_gpu_messages(gpu_message<params> *cpu_instances, uint32_t count) {
  gpu_message<params> *gpu_instances, *tmp_cpu_instaces;
  tmp_cpu_instaces = (gpu_message<params> *)malloc(count*sizeof(gpu_message<params>));
  memcpy(tmp_cpu_instaces, cpu_instances, count*sizeof(gpu_message<params>));

  //write_messages<params>(stdout, tmp_cpu_instaces, count);
  for(uint32_t idx=0; idx<count; idx++) {
    // data
    cudaMalloc((void **)&(tmp_cpu_instaces[idx].data.data), sizeof(uint8_t)*tmp_cpu_instaces[idx].data.size);
    cudaMemcpy(tmp_cpu_instaces[idx].data.data, cpu_instances[idx].data.data, sizeof(uint8_t)*tmp_cpu_instaces[idx].data.size, cudaMemcpyHostToDevice);
  }
  //write_messages<params>(stdout, tmp_cpu_instaces, count);
  cudaMalloc((void **)&gpu_instances, sizeof(gpu_message<params>)*count);
  cudaMemcpy(gpu_instances, tmp_cpu_instaces, sizeof(gpu_message<params>)*count, cudaMemcpyHostToDevice);
  free(tmp_cpu_instaces);
  return gpu_instances;
}


template<class params>
__host__ void free_gpu_messages(gpu_message<params> *gpu_instances, uint32_t count) {
  gpu_message<params> *tmp_cpu_instaces;
  tmp_cpu_instaces = (gpu_message<params> *)malloc(count*sizeof(gpu_message<params>));
  cudaMemcpy(tmp_cpu_instaces, gpu_instances, count*sizeof(gpu_message<params>), cudaMemcpyDeviceToHost);

  for(uint32_t idx=0; idx<count; idx++) {
    // data
    cudaFree(tmp_cpu_instaces[idx].data.data);
  }
  free(tmp_cpu_instaces);
  cudaFree(gpu_instances);
}

template<class params>
__host__ void write_messages(FILE *fp, gpu_message<params> *cpu_instances, uint32_t count) {
  for(uint32_t idx=0; idx<count; idx++) {
    fprintf(fp, "INSTACE: %08x , CALLER: ", idx);
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
    fprintf(fp, "%lx ", cpu_instances[idx].data.size);
    /*
    fprintf(fp, ", DATA: ");
    for(size_t jdx=0; jdx<cpu_instances[idx].data.size; jdx++) {
      fprintf(fp, "%02x ", cpu_instances[idx].data.data[jdx]);
    }
    fprintf(fp, ", CONTRACT_ADDRESS: ");
    for(uint32_t jdx=0; jdx<params::BITS/32; jdx++) {
      fprintf(fp, "%08x ", cpu_instances[idx].contract->address._limbs[jdx]);
    }
    fprintf(fp, ", CONTRACT_SIZE: ");
    fprintf(fp, "%lx ", cpu_instances[idx].contract->code_size);
    fprintf(fp, ", CONTRACT_BYTECODE: ");
    for(size_t jdx=0; jdx<cpu_instances[idx].contract->code_size; jdx++) {
      fprintf(fp, "%02x ", cpu_instances[idx].contract->bytecode[jdx]);
    }
    */
    fprintf(fp, "\n");
  }
}

#endif