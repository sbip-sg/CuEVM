// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2024-06-20
// SPDX-License-Identifier: MIT
#include "include/block.cuh"
#include "include/error_codes.h"
#include "include/utils.cuh"

namespace cuEVM {
  namespace block {
    __host__ __device__ EVMBlockInfo::EVMBlockInfo(
        ArithEnv arith,
        block_data_t *content
    ) : _arith(arith),
        content(content)
    {
    }
    __host__ EVMBlockInfo::EVMBlockInfo(
        ArithEnv arith,
        const cJSON *test
    ) : _arith(arith)
    {
      cJSON *block_json = NULL;
      cJSON *element_json = NULL;
      cJSON *previous_blocks_json = NULL;
      size_t idx = 0;
      content = NULL;
    #ifndef ONLY_CPU
      CUDA_CHECK(cudaMallocManaged(
          (void **)&(content),
          sizeof(block_data_t)));
    #else
      content = new block_data_t;
    #endif

      block_json = cJSON_GetObjectItemCaseSensitive(test, "env");

      element_json = cJSON_GetObjectItemCaseSensitive(block_json, "currentCoinbase");
      content->coin_base.from_hex(element_json->valuestring);

      element_json = cJSON_GetObjectItemCaseSensitive(block_json, "currentTimestamp");
      content->time_stamp.from_hex(element_json->valuestring);

      element_json = cJSON_GetObjectItemCaseSensitive(block_json, "currentNumber");
      content->number.from_hex(element_json->valuestring);
  
      element_json = cJSON_GetObjectItemCaseSensitive(block_json, "currentDifficulty");
      content->difficulty.from_hex(element_json->valuestring);

      element_json = cJSON_GetObjectItemCaseSensitive(block_json, "currentRandom");
      if (element_json != NULL)
      {
        content->prevrandao.from_hex(element_json->valuestring);
      }

      element_json = cJSON_GetObjectItemCaseSensitive(block_json, "currentGasLimit");
      content->gas_limit.from_hex(element_json->valuestring);

      // element_json=cJSON_GetObjectItemCaseSensitive(block_json, "currentChainId");
      //_arith.cgbn_memory_from_hex_string(content->chain_id, element_json->valuestring);
      content->chain_id.from_size_t(1);

      element_json = cJSON_GetObjectItemCaseSensitive(block_json, "currentBaseFee");
      content->base_fee.from_hex(element_json->valuestring);

      previous_blocks_json = cJSON_GetObjectItemCaseSensitive(block_json, "previousHashes");
      if (previous_blocks_json != NULL and cJSON_IsArray(previous_blocks_json))
      {
        idx = 0;
        cJSON_ArrayForEach(element_json, previous_blocks_json)
        {
          element_json = cJSON_GetObjectItemCaseSensitive(element_json, "number");
          content->previous_blocks[idx].number.from_hex(element_json->valuestring);

          element_json = cJSON_GetObjectItemCaseSensitive(element_json, "hash");
          content->previous_blocks[idx].hash.from_hex(element_json->valuestring);
          idx++;
        }
      }
      else
      {
        idx = 0;
        // TODO: maybe fill with something else
        content->previous_blocks[0].number.from_size_t(0);

        element_json = cJSON_GetObjectItemCaseSensitive(block_json, "previousHash");

        if (element_json != NULL){
          content->previous_blocks[0].hash.from_hex(element_json->valuestring);
        } else {
          content->previous_blocks[0].hash.from_size_t(0);
        }

        idx++;
      }

      // fill the remaing parents with 0
      for (size_t jdx = idx; jdx < 256; jdx++)
      {
        content->previous_blocks[jdx].number.from_size_t(0);
        content->previous_blocks[jdx].hash.from_size_t(0);
      }
    }


    __host__ __device__ EVMBlockInfo::~EVMBlockInfo()
    {
      content = NULL;
    }


    __host__ void EVMBlockInfo::free_content()
    {
    #ifndef ONLY_CPU
      CUDA_CHECK(cudaFree(content));
    #else
      delete content;
    #endif
      content = NULL;
    }


    __host__ __device__ void EVMBlockInfo::get_coin_base(
        bn_t &coin_base)
    {
      cgbn_load(_arith.env, coin_base, &(content->coin_base));
    }


    __host__ __device__ void EVMBlockInfo::get_time_stamp(
        bn_t &time_stamp)
    {
      cgbn_load(_arith.env, time_stamp, &(content->time_stamp));
    }

    __host__ __device__ void EVMBlockInfo::get_number(
      bn_t &number)
    {
      cgbn_load(_arith.env, number, &(content->number));
    }


    __host__ __device__ void EVMBlockInfo::get_difficulty(
      bn_t &difficulty)
    {
      cgbn_load(_arith.env, difficulty, &(content->difficulty));
    }

    __host__ __device__ void EVMBlockInfo::get_prevrandao(
      bn_t &val)
    {
      cgbn_load(_arith.env, val, &(content->prevrandao));
    }


    __host__ __device__ void EVMBlockInfo::get_gas_limit(
      bn_t &gas_limit)
    {
      cgbn_load(_arith.env, gas_limit, &(content->gas_limit));
    }


    __host__ __device__ void EVMBlockInfo::get_chain_id(
      bn_t &chain_id)
    {
      cgbn_load(_arith.env, chain_id, &(content->chain_id));
    }


    __host__ __device__ void EVMBlockInfo::get_base_fee(
      bn_t &base_fee)
    {
      cgbn_load(_arith.env, base_fee, &(content->base_fee));
    }

    __host__ __device__ void EVMBlockInfo::get_previous_hash(
        bn_t &previous_hash,
        bn_t &previous_number,
        uint32_t &error_code)
    {
      uint32_t idx = 0;
      bn_t number;
      // ge tthe current number
      get_number(number);
      // if the rquest number is greater than the current block number
      if (cgbn_compare(_arith.env, number, previous_number) < 1)
      {
        error_code = ERR_BLOCK_INVALID_NUMBER;
      }
      // get the distance from the current block number to the requested block number
      cgbn_sub(_arith.env, number, number, previous_number);
      idx = cgbn_get_ui32(_arith.env, number) - 1;
      // only the last 256 blocks are stored
      if (idx > 255)
      {
        error_code = ERR_BLOCK_INVALID_NUMBER;
      }
      if (error_code == ERR_NONE)
        cgbn_load(_arith.env, previous_hash, &(content->previous_blocks[idx].hash));
      else
        cgbn_set_ui32(_arith.env, previous_hash, 0);
    }

    __host__ __device__ void EVMBlockInfo::print()
    {
      uint32_t idx = 0;
      bn_t number;
      printf("BLOCK: \n");
      printf("COINBASE: ");
      content->coin_base.print();
      printf("TIMESTAMP: ");
      content->time_stamp.print();
      printf("NUMBER: ");
      content->number.print();
      printf("DIFICULTY: ");
      content->difficulty.print();
      printf("GASLIMIT: ");
      content->gas_limit.print();
      printf("CHAINID: ");
      content->chain_id.print();
      printf("BASE_FEE: ");
      content->base_fee.print();
      printf("PREVIOUS_BLOCKS: \n");
      for (idx = 0; idx < 256; idx++)
      {
        printf("NUMBER: ");
        content->previous_blocks[idx].number.print();
        printf("HASH: ");
        content->previous_blocks[idx].hash.print();
        printf("\n");
        cgbn_load(_arith.env, number, &(content->previous_blocks[idx].number));
        if (cgbn_compare_ui32(_arith.env, number, 0) == 0)
        {
          break;
        }
      }
    }

    __host__ cJSON * EVMBlockInfo::to_json()
    {
      uint32_t idx = 0;
      char *hex_string_ptr = new char[EVM_WORD_SIZE * 2 + 3];
      cJSON *block_json = NULL;
      cJSON *previous_blocks_json = NULL;
      cJSON *previous_block_json = NULL;

      block_json = cJSON_CreateObject();

      hex_string_ptr = content->coin_base.to_hex(hex_string_ptr, 0, 5);
      cJSON_AddStringToObject(block_json, "currentCoinbase", hex_string_ptr);

      hex_string_ptr = content->time_stamp.to_hex(hex_string_ptr);
      cJSON_AddStringToObject(block_json, "currentTimestamp", hex_string_ptr);

      hex_string_ptr = content->number.to_hex(hex_string_ptr);
      cJSON_AddStringToObject(block_json, "currentNumber", hex_string_ptr);

      hex_string_ptr = content->difficulty.to_hex(hex_string_ptr);
      cJSON_AddStringToObject(block_json, "currentDifficulty", hex_string_ptr);

      hex_string_ptr = content->prevrandao.to_hex(hex_string_ptr);
      cJSON_AddStringToObject(block_json, "currentGasLimit", hex_string_ptr);

      hex_string_ptr = content->gas_limit.to_hex(hex_string_ptr);
      cJSON_AddStringToObject(block_json, "currentChainId", hex_string_ptr);

      hex_string_ptr = content->base_fee.to_hex(hex_string_ptr);
      cJSON_AddStringToObject(block_json, "currentBaseFee", hex_string_ptr);

      previous_blocks_json = cJSON_CreateArray();
      bn_t number;
      for (idx = 0; idx < 256; idx++)
      {
        previous_block_json = cJSON_CreateObject();

        
        hex_string_ptr = content->previous_blocks[idx].number.to_hex(hex_string_ptr);
        cJSON_AddStringToObject(previous_block_json, "number", hex_string_ptr);

        
        hex_string_ptr = content->previous_blocks[idx].hash.to_hex(hex_string_ptr);
        cJSON_AddStringToObject(previous_block_json, "hash", hex_string_ptr);

        cJSON_AddItemToArray(previous_blocks_json, previous_block_json);

        cgbn_load(_arith.env, number, &(content->previous_blocks[idx].number));
        if (cgbn_compare_ui32(_arith.env, number, 0) == 0)
        {
          break;
        }
      }

      cJSON_AddItemToObject(block_json, "previousHashes", previous_blocks_json);
      
      delete[] hex_string_ptr;
      hex_string_ptr = NULL;
      return block_json;
    }
  }
}
