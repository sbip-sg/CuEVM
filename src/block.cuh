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
#include "utils.h"

template<class params>
struct gpu_block {
  cgbn_mem_t<params::BITS> coin_base;
  cgbn_mem_t<params::BITS> time_stamp;
  cgbn_mem_t<params::BITS> number;
  cgbn_mem_t<params::BITS> difficulty;
  cgbn_mem_t<params::BITS> gas_limit;
  cgbn_mem_t<params::BITS> chain_id;
  cgbn_mem_t<params::BITS> base_fee;
};


template<class params>
struct gpu_block_hash {
  cgbn_mem_t<params::BITS> number;
  cgbn_mem_t<params::BITS> hash;
};

template<class params>
__host__ gpu_block<params> *generate_cpu_blocks(uint32_t count) {
  gpu_block<params> *cpu_instances=(gpu_block<params> *)malloc(sizeof(gpu_block<params>)*count);
  mpz_t coin_base, time_stamp, number, difficulty, gas_limit, chain_id, base_fee;
  mpz_init(coin_base);
  mpz_init(time_stamp);
  mpz_init(number);
  mpz_init(difficulty);
  mpz_init(gas_limit);
  mpz_init(chain_id);
  mpz_init(base_fee);
  char hex_string_value[params::BITS/4+1]; // +1 for null terminator

  for(size_t idx=0; idx<count; idx++) {
    snprintf(hex_string_value, params::BITS/4+1, "%lx", idx+257);
    mpz_set_str(coin_base, hex_string_value, 16);
    snprintf(hex_string_value, params::BITS/4+1, "%lx", idx+1);
    mpz_set_str(time_stamp, hex_string_value, 16);
    snprintf(hex_string_value, params::BITS/4+1, "%lx", idx+2);
    mpz_set_str(number, hex_string_value, 16);
    snprintf(hex_string_value, params::BITS/4+1, "%lx", idx+3);
    mpz_set_str(difficulty, hex_string_value, 16);
    snprintf(hex_string_value, params::BITS/4+1, "%lx", idx+4);
    mpz_set_str(gas_limit, hex_string_value, 16);
    snprintf(hex_string_value, params::BITS/4+1, "%lx", idx+5);
    mpz_set_str(chain_id, hex_string_value, 16);
    snprintf(hex_string_value, params::BITS/4+1, "%lx", idx+6);
    mpz_set_str(base_fee, hex_string_value, 16);

    from_mpz(cpu_instances[idx].coin_base._limbs, params::BITS/32, coin_base);
    from_mpz(cpu_instances[idx].time_stamp._limbs, params::BITS/32, time_stamp);
    from_mpz(cpu_instances[idx].number._limbs, params::BITS/32, number);
    from_mpz(cpu_instances[idx].difficulty._limbs, params::BITS/32, difficulty);
    from_mpz(cpu_instances[idx].gas_limit._limbs, params::BITS/32, gas_limit);
    from_mpz(cpu_instances[idx].chain_id._limbs, params::BITS/32, chain_id);
    from_mpz(cpu_instances[idx].base_fee._limbs, params::BITS/32, base_fee);
  }

  mpz_clear(coin_base);
  mpz_clear(time_stamp);
  mpz_clear(number);
  mpz_clear(difficulty);
  mpz_clear(gas_limit);
  mpz_clear(chain_id);
  mpz_clear(base_fee);
  return cpu_instances;
}

template<class params>
__host__ void free_host_blocks(gpu_block<params> *cpu_instances, uint32_t count) {
  free(cpu_instances);
}

template<class params>
__host__ void write_blocks(FILE *fp, gpu_block<params> *cpu_instances, uint32_t count) {
  for(uint32_t idx=0; idx<count; idx++) {
    fprintf(fp, "INSTACE: %08x , COINBASE: ", idx);
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
      fprintf(fp, "%08x ", cpu_instances[idx].difficulty._limbs[jdx]);
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



template<class params>
__host__ gpu_block_hash<params> *generate_cpu_blocks_hash(uint32_t count) {
  gpu_block_hash<params> *cpu_instances=(gpu_block_hash<params> *)malloc(sizeof(gpu_block_hash<params>)*count);
  mpz_t number, hash;
  char hex_string_value[params::BITS/4+1]; // +1 for null terminator
  mpz_init(number);
  mpz_init(hash);

  for(size_t idx=0; idx<count; idx++) {
    snprintf(hex_string_value, params::BITS/4+1, "%lx", 255-idx);
    mpz_set_str(number, hex_string_value, 16);
    snprintf(hex_string_value, params::BITS/4+1, "%lx", idx+1);
    mpz_set_str(hash, hex_string_value, 16);

    from_mpz(cpu_instances[idx].number._limbs, params::BITS/32, number);
    from_mpz(cpu_instances[idx].hash._limbs, params::BITS/32, hash);
  }

  mpz_clear(number);
  mpz_clear(hash);
  return cpu_instances;
}


template<class params>
__host__ void free_host_blocks_hash(gpu_block_hash<params> *cpu_instances, uint32_t count) {
  free(cpu_instances);
}

template<class params>
__host__ void write_blocks_hash(FILE *fp, gpu_block_hash<params> *cpu_instances, uint32_t count) {
  for(uint32_t idx=0; idx<count; idx++) {
    fprintf(fp, "INSTACE: %08x , NUMBER: ", idx);
    for(uint32_t jdx=0; jdx<params::BITS/32; jdx++) {
      fprintf(fp, "%08x ", cpu_instances[idx].number._limbs[jdx]);
    }
    fprintf(fp, ", HASH: ");
    for(uint32_t jdx=0; jdx<params::BITS/32; jdx++) {
      fprintf(fp, "%08x ", cpu_instances[idx].hash._limbs[jdx]);
    }
    fprintf(fp, "\n");
  }
}
#endif