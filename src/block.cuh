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
__host__ gpu_block<params> *generate_block(uint32_t count) {
  gpu_block<params> *instances=(gpu_block<params> *)malloc(sizeof(gpu_block<params>)*count);

  for(uint32_t index=0;index<count;index++) {
    instances[index].coin_base._limbs[0]=1;
    instances[index].time_stamp._limbs[0]=1;
    instances[index].number._limbs[0]=1;
    instances[index].dificulty._limbs[0]=1;
    instances[index].gas_limit._limbs[0]=1;
    instances[index].chain_id._limbs[0]=1;
    instances[index].base_fee._limbs[0]=1;

    for(uint32_t j=1;j<params::BITS/32;j++) {
      instances[index].coin_base._limbs[j]=0;
      instances[index].time_stamp._limbs[j]=0;
      instances[index].number._limbs[j]=0;
      instances[index].dificulty._limbs[j]=0;
      instances[index].gas_limit._limbs[j]=0;
      instances[index].chain_id._limbs[j]=0;
      instances[index].base_fee._limbs[j]=0;
    }
  }
  return instances;
}

template<class params>
__host__ void show_block(gpu_block<params> *instances, uint32_t count) {
  for(uint32_t index=0;index<count;index++) {
    printf("Coin base : ");
    for(uint32_t j=0;j<params::BITS/32;j++) {
      printf("%08x ", instances[index].coin_base._limbs[j]);
    }
    printf("\n");
    printf("Timestamp : ");
    for(uint32_t j=0;j<params::BITS/32;j++) {
      printf("%08x ", instances[index].time_stamp._limbs[j]);
    }
    printf("\n");
    printf("Number : ");
    for(uint32_t j=0;j<params::BITS/32;j++) {
      printf("%08x ", instances[index].number._limbs[j]);
    }
    printf("\n");
    printf("Dificulty : ");
    for(uint32_t j=0;j<params::BITS/32;j++) {
      printf("%08x ", instances[index].dificulty._limbs[j]);
    }
    printf("\n");
    printf("Gas limit : ");
    for(uint32_t j=0;j<params::BITS/32;j++) {
      printf("%08x ", instances[index].gas_limit._limbs[j]);
    }
    printf("\n");
    printf("Chain ID : ");
    for(uint32_t j=0;j<params::BITS/32;j++) {
      printf("%08x ", instances[index].chain_id._limbs[j]);
    }
    printf("\n");
    printf("Base fee : ");
    for(uint32_t j=0;j<params::BITS/32;j++) {
      printf("%08x ", instances[index].base_fee._limbs[j]);
    }
    printf("\n");
  }
}

#endif