#ifndef _BLOCK_H_
#define _BLOCK_H_

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
class block_t {
  public:
  typedef typename arith_env_t<params>::bn_t      bn_t;
  typedef cgbn_mem_t<params::BITS>                evm_word_t;
  typedef arith_env_t<params>                     arith_t;

  
  typedef struct {
    evm_word_t number;
    evm_word_t hash;
  } block_hash_t;

  typedef struct {
    evm_word_t    coin_base;
    evm_word_t    time_stamp;
    evm_word_t    number;
    evm_word_t    difficulty;
    evm_word_t    gas_limit;
    evm_word_t    chain_id;
    evm_word_t    base_fee;
    block_hash_t  previous_blocks[256];
  } block_data_t;

  block_data_t          *_content;
  arith_t   _arith;

  __device__ block_t(arith_t arith, block_data_t *content)  : _arith(arith), _content(content) {
  }

  __host__ block_t(arith_t arith, const cJSON * test)  : _arith(arith) {
    block_data_t *cpu_block=(block_data_t *)malloc(sizeof(block_data_t));
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
    _content=cpu_block;
  }

  __host__ void free_memory() {
    free(_content);
  }

  __host__ block_data_t *to_gpu() {
    block_data_t *gpu_block=NULL;
    cudaMalloc((void **)&gpu_block, sizeof(block_data_t));
    cudaMemcpy(gpu_block, _content, sizeof(block_data_t), cudaMemcpyHostToDevice);
    return gpu_block;
  }

  __host__ void from_gpu(block_data_t *gpu_block) {
    cudaMemcpy(_content, gpu_block, sizeof(block_data_t), cudaMemcpyDeviceToHost);
  }

  __host__ static void free_gpu(block_data_t *gpu_block) {
    cudaFree(gpu_block);
  }

  __host__ __device__ __forceinline__ void get_coin_base(bn_t &coin_base) {
    cgbn_load(_arith._env, coin_base, &(_content->coin_base));
  }

  __host__ __device__ __forceinline__ void get_time_stamp(bn_t &time_stamp) {
    cgbn_load(_arith._env, time_stamp, &(_content->time_stamp));
  }

  __host__ __device__ __forceinline__ void get_number(bn_t &number) {
    cgbn_load(_arith._env, number, &(_content->number));
  }

  __host__ __device__ __forceinline__ void get_difficulty(bn_t &difficulty) {
    cgbn_load(_arith._env, difficulty, &(_content->difficulty));
  }

  __host__ __device__ __forceinline__ void get_gas_limit(bn_t &gas_limit) {
    cgbn_load(_arith._env, gas_limit, &(_content->gas_limit));
  }

  __host__ __device__ __forceinline__ void get_chain_id(bn_t &chain_id) {
    cgbn_load(_arith._env, chain_id, &(_content->chain_id));
  }

  __host__ __device__ __forceinline__ void get_base_fee(bn_t &base_fee) {
    cgbn_load(_arith._env, base_fee, &(_content->base_fee));
  }

  __host__ __device__ __forceinline__ void get_previous_hash(bn_t &previous_number, bn_t &previous_hash, uint32_t &error_code) {
    uint32_t idx=0;
    bn_t number;
    get_number(number);
    if(cgbn_compare(_arith._env, number, previous_number) < 1) {
      error_code=ERR_BLOCK_INVALID_NUMBER;
      return;
    }
    cgbn_sub(_arith._env, number, number, previous_number);
    idx=cgbn_get_ui32(_arith._env, number) - 1;
    if (idx > 255) {
      error_code=ERR_BLOCK_INVALID_NUMBER;
      return;
    }
    cgbn_load(_arith._env, previous_hash, &(_content->previous_blocks[idx].hash));
  }

    
  __host__ __device__ void print() {
    uint32_t idx=0, jdx=0;
      printf("BLOCK: \n");
      printf("COINBASE: ");
      print_bn<params>(_content->coin_base);
      printf(", TIMESTAMP: ");
      print_bn<params>(_content->time_stamp);
      printf(", NUMBER: ");
      print_bn<params>(_content->number);
      printf(", DIFICULTY: ");
      print_bn<params>(_content->difficulty);
      printf(", GASLIMIT: ");
      print_bn<params>(_content->gas_limit);
      printf(", CHAINID: ");
      print_bn<params>(_content->chain_id);
      printf(", BASE_FEE: ");
      print_bn<params>(_content->base_fee);
      printf("PREVIOUS_BLOCKS: \n");
      for(idx=0; idx<256; idx++) {
        printf("NUMBER: ");
        print_bn<params>(_content->previous_blocks[idx].number);
        printf(", HASH: ");
        print_bn<params>(_content->previous_blocks[idx].hash);
        printf("\n");
      }
  }


  __host__ cJSON *to_json() {
    uint32_t idx=0;
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

    to_mpz(coin_base, _content->coin_base._limbs, params::BITS/32);
    strcpy(hex_string+2, mpz_get_str(NULL, 16, coin_base));
    cJSON_AddStringToObject(block_json, "currentCoinbase", hex_string);
    
    to_mpz(time_stamp, _content->time_stamp._limbs, params::BITS/32);
    strcpy(hex_string+2, mpz_get_str(NULL, 16, time_stamp));
    cJSON_AddStringToObject(block_json, "currentTimestamp", hex_string);

    to_mpz(number, _content->number._limbs, params::BITS/32);
    strcpy(hex_string+2, mpz_get_str(NULL, 16, number));
    cJSON_AddStringToObject(block_json, "currentNumber", hex_string);

    to_mpz(difficulty, _content->difficulty._limbs, params::BITS/32);
    strcpy(hex_string+2, mpz_get_str(NULL, 16, difficulty));
    cJSON_AddStringToObject(block_json, "currentDifficulty", hex_string);

    to_mpz(gas_limit, _content->gas_limit._limbs, params::BITS/32);
    strcpy(hex_string+2, mpz_get_str(NULL, 16, gas_limit));
    cJSON_AddStringToObject(block_json, "currentGasLimit", hex_string);

    to_mpz(chain_id, _content->chain_id._limbs, params::BITS/32);
    strcpy(hex_string+2, mpz_get_str(NULL, 16, chain_id));
    cJSON_AddStringToObject(block_json, "currentChainId", hex_string);

    to_mpz(base_fee, _content->base_fee._limbs, params::BITS/32);
    strcpy(hex_string+2, mpz_get_str(NULL, 16, base_fee));
    cJSON_AddStringToObject(block_json, "currentBaseFee", hex_string);

    previous_blocks_json=cJSON_CreateArray();
    for(idx=0; idx<256; idx++) {
      previous_block_json=cJSON_CreateObject();
      
      to_mpz(number_prev, _content->previous_blocks[idx].number._limbs, params::BITS/32);
      strcpy(hex_string+2, mpz_get_str(NULL, 16, number_prev));
      cJSON_AddStringToObject(previous_block_json, "number", hex_string);

      to_mpz(hash_prev, _content->previous_blocks[idx].hash._limbs, params::BITS/32);
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
};

#endif