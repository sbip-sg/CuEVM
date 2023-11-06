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
struct block_hash_t {
  cgbn_mem_t<params::BITS> number;
  cgbn_mem_t<params::BITS> hash;
};

template<class params>
struct block_t {
  cgbn_mem_t<params::BITS> coin_base;
  cgbn_mem_t<params::BITS> time_stamp;
  cgbn_mem_t<params::BITS> number;
  cgbn_mem_t<params::BITS> difficulty;
  cgbn_mem_t<params::BITS> gas_limit;
  cgbn_mem_t<params::BITS> chain_id;
  cgbn_mem_t<params::BITS> base_fee;
  block_hash_t<params> previous_blocks[256];
};




template<class params>
__host__ block_t<params> *cpu_block_from_json(cJSON * test) {
  block_t<params> *cpu_block=(block_t<params> *)malloc(sizeof(block_t<params>));
  // block related info
  mpz_t coin_base, time_stamp, number, difficulty, gas_limit, chain_id, base_fee;
  // previous blocks info
  mpz_t number_prev, hash_prev;
  mpz_init(coin_base);
  mpz_init(time_stamp);
  mpz_init(number);
  mpz_init(difficulty);
  mpz_init(gas_limit);
  mpz_init(chain_id);
  mpz_init(base_fee);
  mpz_init(number_prev);
  mpz_init(hash_prev);
  char *hex_string=NULL;
  cJSON *block_json=NULL;
  cJSON *coin_base_json=NULL;
  cJSON *time_stamp_json=NULL;
  cJSON *number_json=NULL;
  cJSON *difficulty_json=NULL;
  cJSON *gas_limit_json=NULL;
  cJSON *chain_id_json=NULL;
  cJSON *base_fee_json=NULL;
  cJSON *previous_blocks_json=NULL;
  cJSON *previous_block_json=NULL;
  cJSON *number_prev_json=NULL;
  cJSON *hash_prev_json=NULL;
  size_t idx=0, jdx=0;

  block_json=cJSON_GetObjectItemCaseSensitive(test, "env");

  coin_base_json=cJSON_GetObjectItemCaseSensitive(block_json, "currentCoinbase");
  hex_string = coin_base_json->valuestring;
  adjusted_length(&hex_string);
  mpz_set_str(coin_base, hex_string, 16);
  from_mpz(cpu_block->coin_base._limbs, params::BITS/32, coin_base);

  time_stamp_json=cJSON_GetObjectItemCaseSensitive(block_json, "currentTimestamp");
  hex_string = time_stamp_json->valuestring;
  adjusted_length(&hex_string);
  mpz_set_str(time_stamp, hex_string, 16);
  from_mpz(cpu_block->time_stamp._limbs, params::BITS/32, time_stamp);

  number_json=cJSON_GetObjectItemCaseSensitive(block_json, "currentNumber");
  hex_string = number_json->valuestring;
  adjusted_length(&hex_string);
  mpz_set_str(number, hex_string, 16);
  from_mpz(cpu_block->number._limbs, params::BITS/32, number);

  difficulty_json=cJSON_GetObjectItemCaseSensitive(block_json, "currentDifficulty");
  hex_string = difficulty_json->valuestring;
  adjusted_length(&hex_string);
  mpz_set_str(difficulty, hex_string, 16);
  from_mpz(cpu_block->difficulty._limbs, params::BITS/32, difficulty);

  gas_limit_json=cJSON_GetObjectItemCaseSensitive(block_json, "currentGasLimit");
  hex_string = gas_limit_json->valuestring;
  adjusted_length(&hex_string);
  mpz_set_str(gas_limit, hex_string, 16);
  from_mpz(cpu_block->gas_limit._limbs, params::BITS/32, gas_limit);

  //chain_id_json=cJSON_GetObjectItemCaseSensitive(block_json, "currentChainId");
  //hex_string = chain_id_json->valuestring;
  //adjusted_length(&hex_string);
  //mpz_set_str(chain_id, hex_string, 16);
  //from_mpz(cpu_block->chain_id._limbs, params::BITS/32, chain_id);
  mpz_set_ui(chain_id, 1); //mainnet
  from_mpz(cpu_block->chain_id._limbs, params::BITS/32, chain_id);

  base_fee_json=cJSON_GetObjectItemCaseSensitive(block_json, "currentBaseFee");
  hex_string = base_fee_json->valuestring;
  adjusted_length(&hex_string);
  mpz_set_str(base_fee, hex_string, 16);
  from_mpz(cpu_block->base_fee._limbs, params::BITS/32, base_fee);

  
  previous_blocks_json=cJSON_GetObjectItemCaseSensitive(block_json, "previousHashes");
  if (previous_blocks_json != NULL and cJSON_IsArray(previous_blocks_json)) {
    jdx=0;
    cJSON_ArrayForEach(previous_block_json, previous_blocks_json) {
      number_prev_json=cJSON_GetObjectItemCaseSensitive(previous_block_json, "number");
      hex_string = number_prev_json->valuestring;
      adjusted_length(&hex_string);
      mpz_set_str(number_prev, hex_string, 16);
      from_mpz(cpu_block->previous_blocks[jdx].number._limbs, params::BITS/32, number_prev);

      hash_prev_json=cJSON_GetObjectItemCaseSensitive(previous_block_json, "hash");
      hex_string = hash_prev_json->valuestring;
      adjusted_length(&hex_string);
      mpz_set_str(hash_prev, hex_string, 16);
      from_mpz(cpu_block->previous_blocks[jdx].hash._limbs, params::BITS/32, hash_prev);
      jdx++;
    }
  } else {
    jdx=0;
    previous_block_json = cJSON_GetObjectItemCaseSensitive(block_json, "previousHash");
    mpz_sub_ui(number_prev, number, 1);
    from_mpz(cpu_block->previous_blocks[0].number._limbs, params::BITS/32, number_prev);

    hex_string = previous_block_json->valuestring;
    adjusted_length(&hex_string);
    mpz_set_str(hash_prev, hex_string, 16);
    from_mpz(cpu_block->previous_blocks[0].hash._limbs, params::BITS/32, hash_prev);
    jdx++;
  }

  for(idx=jdx; idx<256; idx++) {
    mpz_set_ui(number_prev, 0);
    from_mpz(cpu_block->previous_blocks[idx].number._limbs, params::BITS/32, number_prev);
    mpz_set_ui(hash_prev, 0);
    from_mpz(cpu_block->previous_blocks[idx].hash._limbs, params::BITS/32, hash_prev);
  }

  mpz_clear(coin_base);
  mpz_clear(time_stamp);
  mpz_clear(number);
  mpz_clear(difficulty);
  mpz_clear(gas_limit);
  mpz_clear(chain_id);
  mpz_clear(base_fee);
  mpz_clear(number_prev);
  mpz_clear(hash_prev);
  return cpu_block;
}

template<class params>
__host__ void free_host_block(block_t<params> *cpu_block) {
  free(cpu_block);
}

template<class params>
__host__ void print_block(FILE *fp, block_t<params> *cpu_block) {
  uint32_t idx=0, jdx=0;
    fprintf(fp, "BLOCK: \n");
    fprintf(fp, "COINBASE: ");
    for(jdx=0; jdx<params::BITS/32; jdx++) {
      fprintf(fp, "%08x ", cpu_block->coin_base._limbs[jdx]);
    }
    fprintf(fp, ", TIMESTAMP: ");
    for(jdx=0; jdx<params::BITS/32; jdx++) {
      fprintf(fp, "%08x ", cpu_block->time_stamp._limbs[jdx]);
    }
    fprintf(fp, ", NUMBER: ");
    for(jdx=0; jdx<params::BITS/32; jdx++) {
      fprintf(fp, "%08x ", cpu_block->number._limbs[jdx]);
    }
    fprintf(fp, ", DIFICULTY: ");
    for(jdx=0; jdx<params::BITS/32; jdx++) {
      fprintf(fp, "%08x ", cpu_block->difficulty._limbs[jdx]);
    }
    fprintf(fp, ", GASLIMIT: ");
    for(jdx=0; jdx<params::BITS/32; jdx++) {
      fprintf(fp, "%08x ", cpu_block->gas_limit._limbs[jdx]);
    }
    fprintf(fp, ", CHAINID: ");
    for(jdx=0; jdx<params::BITS/32; jdx++) {
      fprintf(fp, "%08x ", cpu_block->chain_id._limbs[jdx]);
    }
    fprintf(fp, ", BASE_FEE: ");
    for(jdx=0; jdx<params::BITS/32; jdx++) {
      fprintf(fp, "%08x ", cpu_block->base_fee._limbs[jdx]);
    }
    fprintf(fp, "PREVIOUS_BLOCKS: \n");
    for(idx=0; idx<256; idx++) {
      fprintf(fp, "NUMBER: ");
      for(jdx=0; jdx<params::BITS/32; jdx++) {
        fprintf(fp, "%08x ", cpu_block->previous_blocks[idx].number._limbs[jdx]);
      }
      fprintf(fp, ", HASH: ");
      for(jdx=0; jdx<params::BITS/32; jdx++) {
        fprintf(fp, "%08x ", cpu_block->previous_blocks[idx].hash._limbs[jdx]);
      }
      fprintf(fp, "\n");
    }
}

template<class params>
__host__ cJSON *block_to_json(block_t<params> *cpu_block) {
  uint32_t idx=0, jdx=0;
  char hex_string[67]="0x";
  mpz_t coin_base, time_stamp, number, difficulty, gas_limit, chain_id, base_fee;
  mpz_t number_prev, hash_prev;
  cJSON *block_json=NULL;
  cJSON *previous_blocks_json=NULL;
  cJSON *previous_block_json=NULL;
  mpz_init(coin_base);
  mpz_init(time_stamp);
  mpz_init(number);
  mpz_init(difficulty);
  mpz_init(gas_limit);
  mpz_init(chain_id);
  mpz_init(base_fee);
  mpz_init(number_prev);
  mpz_init(hash_prev);

  block_json=cJSON_CreateObject();

  to_mpz(cpu_block->coin_base._limbs, params::BITS/32, coin_base);
  strcpy(hex_string+2, mpz_get_str(NULL, 16, coin_base));
  cJSON_AddStringToObject(block_json, "currentCoinbase", hex_string);
  
  to_mpz(cpu_block->time_stamp._limbs, params::BITS/32, time_stamp);
  strcpy(hex_string+2, mpz_get_str(NULL, 16, time_stamp));
  cJSON_AddStringToObject(block_json, "currentTimestamp", hex_string);

  to_mpz(cpu_block->number._limbs, params::BITS/32, number);
  strcpy(hex_string+2, mpz_get_str(NULL, 16, number));
  cJSON_AddStringToObject(block_json, "currentNumber", hex_string);

  to_mpz(cpu_block->difficulty._limbs, params::BITS/32, difficulty);
  strcpy(hex_string+2, mpz_get_str(NULL, 16, difficulty));
  cJSON_AddStringToObject(block_json, "currentDifficulty", hex_string);

  to_mpz(cpu_block->gas_limit._limbs, params::BITS/32, gas_limit);
  strcpy(hex_string+2, mpz_get_str(NULL, 16, gas_limit));
  cJSON_AddStringToObject(block_json, "currentGasLimit", hex_string);

  to_mpz(cpu_block->chain_id._limbs, params::BITS/32, chain_id);
  strcpy(hex_string+2, mpz_get_str(NULL, 16, chain_id));
  cJSON_AddStringToObject(block_json, "currentChainId", hex_string);

  to_mpz(cpu_block->base_fee._limbs, params::BITS/32, base_fee);
  strcpy(hex_string+2, mpz_get_str(NULL, 16, base_fee));
  cJSON_AddStringToObject(block_json, "currentBaseFee", hex_string);

  previous_blocks_json=cJSON_CreateArray();
  for(idx=0; idx<256; idx++) {
    previous_block_json=cJSON_CreateObject();
    to_mpz(cpu_block->previous_blocks[idx].number._limbs, params::BITS/32, number_prev);
    strcpy(hex_string+2, mpz_get_str(NULL, 16, number_prev));
    cJSON_AddStringToObject(previous_block_json, "number", hex_string);

    to_mpz(cpu_block->previous_blocks[idx].hash._limbs, params::BITS/32, hash_prev);
    strcpy(hex_string+2, mpz_get_str(NULL, 16, hash_prev));
    cJSON_AddStringToObject(previous_block_json, "hash", hex_string);

    cJSON_AddItemToArray(previous_blocks_json, previous_block_json);
  }

  cJSON_AddItemToObject(block_json, "previousHashes", previous_blocks_json);

  mpz_clear(coin_base);
  mpz_clear(time_stamp);
  mpz_clear(number);
  mpz_clear(difficulty);
  mpz_clear(gas_limit);
  mpz_clear(chain_id);
  mpz_clear(base_fee);
  mpz_clear(number_prev);
  mpz_clear(hash_prev);
  return block_json;
}
#endif