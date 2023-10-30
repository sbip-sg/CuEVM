#ifndef _GPU_STORAGE_H_
#define _GPU_STORAGE_H_

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
struct gpu_storage {
  cgbn_mem_t<params::BITS> keys[params::STORAGE_SIZE];
  cgbn_mem_t<params::BITS> values[params::STORAGE_SIZE];
  size_t size;
};

template<class params>
__host__ gpu_storage<params> *generate_storage(uint32_t count) {
  gpu_storage<params> *instances=(gpu_storage<params> *)malloc(sizeof(gpu_storage<params>)*count);

  for(uint32_t index=0;index<count;index++) {
    instances[index].size=params::STORAGE_SIZE;
    for(uint32_t i=0;i<params::STORAGE_SIZE;i++) {
      instances[index].keys[i]._limbs[0]=i;
      instances[index].values[i]._limbs[0]=i;
      for(uint32_t j=1;j<params::BITS/32;j++) {
        instances[index].keys[i]._limbs[j]=i;
        instances[index].values[i]._limbs[j]=i;
      }
    }
  }
}

template<class params>
__host__ void show_storage(gpu_storage<params> *instances, uint32_t count) {
  for(uint32_t index=0;index<count;index++) {
    printf("Instance %d\n", index);
    for(uint32_t i=0;i<instances[index].size;i++) {
      printf("  %d: key:", i);
      for(uint32_t j=0;j<params::BITS/32;j++) {
        printf("%08x ", instances[index].keys[i]._limbs[j]);
      }
      printf("  value:");
      for(uint32_t j=0;j<params::BITS/32;j++) {
        printf("%08x ", instances[index].values[i]._limbs[j]);
      }
      printf("\n");
    }
  }
}

#endif