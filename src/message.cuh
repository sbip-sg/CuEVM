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
struct gpu_message {
  gpu_tx<params> tx;
  cgbn_mem_t<params::BITS> contract_address;
  cgbn_mem_t<params::BITS> caller;
  cgbn_mem_t<params::BITS> value;
  cgbn_mem_t<params::BITS> data_size;
  uint8_t data[params::MESSAGE_DATA_SIZE];
};

template<class params>
__host__ gpu_message<params> *generate_message(uint32_t count) {
  gpu_message<params> *instances=(gpu_message<params> *)malloc(sizeof(gpu_message<params>)*count);

  for(uint32_t index=0;index<count;index++) {
    instances[index].tx.origin._limbs[0]=2;
    instances[index].tx.gasprice._limbs[0]=3;
    instances[index].contract_address._limbs[0]=4;
    instances[index].caller._limbs[0]=1;
    instances[index].value._limbs[0]=10000;
    instances[index].data_size._limbs[0]=10;
    for(uint32_t j=1;j<params::BITS/32;j++) {
      instances[index].tx.origin._limbs[j]=0;
      instances[index].tx.gasprice._limbs[j]=0;
      instances[index].contract_address._limbs[j]=0;
      instances[index].caller._limbs[j]=0;
      instances[index].value._limbs[j]=0;
      instances[index].data_size._limbs[j]=0;
    }
    for(uint32_t i=0;i<params::MESSAGE_DATA_SIZE;i++) {
      instances[index].data[i]=(uint8_t)i;
    }
  }
  return instances;
}

template<class params>
__host__ void show_message(gpu_message<params> *instances, uint32_t count) {
  for(uint32_t index=0;index<count;index++) {
    printf("TX Origin : ");
    for(uint32_t j=0;j<params::BITS/32;j++) {
      printf("%08x ", instances[index].tx.origin._limbs[j]);
    }
    printf("\n");
    printf("TX Gasprice : ");
    for(uint32_t j=0;j<params::BITS/32;j++) {
      printf("%08x ", instances[index].tx.gasprice._limbs[j]);
    }
    printf("\n");
    printf("Contract Address : ");
    for(uint32_t j=0;j<params::BITS/32;j++) {
      printf("%08x ", instances[index].contract_address._limbs[j]);
    }
    printf("\n");
    printf("Caller : ");
    for(uint32_t j=0;j<params::BITS/32;j++) {
      printf("%08x ", instances[index].caller._limbs[j]);
    }
    printf("\n");
    printf("Value : ");
    for(uint32_t j=0;j<params::BITS/32;j++) {
      printf("%08x ", instances[index].value._limbs[j]);
    }
    printf("\n");
    printf("Data size : ");
    for(uint32_t j=0;j<params::BITS/32;j++) {
      printf("%08x ", instances[index].data_size._limbs[j]);
    }
    printf("\n");
    printf("Data : ");
    for(uint32_t i=0;i<params::MESSAGE_DATA_SIZE;i++) {
      printf("%02x ", instances[index].data[i]);
    }
    printf("\n");
  }
}

#endif