// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2024-06-20
// SPDX-License-Identifier: MIT
#include "include/block_info.cuh"
#include "include/error_codes.h"
#include "include/utils.cuh"

namespace cuEVM {
  namespace block {
    __host__ __device__ block_info_t::block_info_t()
    {
      coin_base.from_uint32_t(0);
      difficulty.from_uint32_t(0);
      prevrandao.from_uint32_t(0);
      number.from_uint32_t(0);
      gas_limit.from_uint32_t(0);
      time_stamp.from_uint32_t(0);
      base_fee.from_uint32_t(0);
      chain_id.from_uint32_t(0);
      for (size_t idx = 0; idx < 256; idx++)
      {
        previous_blocks[idx].number.from_uint32_t(0);
        previous_blocks[idx].hash.from_uint32_t(0);
      }
    }

    __host__ __device__ block_info_t::~block_info_t()
    {
    }

    __host__ block_info_t::block_info_t(const cJSON* json) {
      from_json(json);
    }

    __host__ int32_t block_info_t::from_json(const cJSON* json) {
      cJSON *block_json = nullptr;
      cJSON *element_json = nullptr;
      cJSON *previous_blocks_json = nullptr;
      size_t idx = 0;

      block_json = cJSON_GetObjectItemCaseSensitive(json, "env");

      element_json = cJSON_GetObjectItemCaseSensitive(block_json, "currentCoinbase");
      coin_base.from_hex(element_json->valuestring);

      element_json = cJSON_GetObjectItemCaseSensitive(block_json, "currentTimestamp");
      time_stamp.from_hex(element_json->valuestring);

      element_json = cJSON_GetObjectItemCaseSensitive(block_json, "currentNumber");
      number.from_hex(element_json->valuestring);
  
      element_json = cJSON_GetObjectItemCaseSensitive(block_json, "currentDifficulty");
      difficulty.from_hex(element_json->valuestring);

      element_json = cJSON_GetObjectItemCaseSensitive(block_json, "currentRandom");
      if (element_json != nullptr)
      {
        prevrandao.from_hex(element_json->valuestring);
      }

      element_json = cJSON_GetObjectItemCaseSensitive(block_json, "currentGasLimit");
      gas_limit.from_hex(element_json->valuestring);

      // element_json=cJSON_GetObjectItemCaseSensitive(block_json, "currentChainId");
      //arith.cgbn_memory_from_hex_string(chain_id, element_json->valuestring);
      chain_id.from_uint32_t(1);

      element_json = cJSON_GetObjectItemCaseSensitive(block_json, "currentBaseFee");
      base_fee.from_hex(element_json->valuestring);

      previous_blocks_json = cJSON_GetObjectItemCaseSensitive(block_json, "previousHashes");
      if (
        (previous_blocks_json != nullptr) && cJSON_IsArray(previous_blocks_json))
      {
        idx = 0;
        cJSON_ArrayForEach(element_json, previous_blocks_json)
        {
          element_json = cJSON_GetObjectItemCaseSensitive(element_json, "number");
          previous_blocks[idx].number.from_hex(element_json->valuestring);

          element_json = cJSON_GetObjectItemCaseSensitive(element_json, "hash");
          previous_blocks[idx].hash.from_hex(element_json->valuestring);
          idx++;
        }
      }
      else
      {
        idx = 0;
        // TODO: maybe fill with something else
        previous_blocks[0].number.from_uint32_t(0);

        element_json = cJSON_GetObjectItemCaseSensitive(block_json, "previousHash");

        if (element_json != nullptr){
          previous_blocks[0].hash.from_hex(element_json->valuestring);
        } else {
          previous_blocks[0].hash.from_uint32_t(0);
        }

        idx++;
      }

      // fill the remaing parents with 0
      for (size_t jdx = idx; jdx < 256; jdx++)
      {
        previous_blocks[jdx].number.from_uint32_t(0);
        previous_blocks[jdx].hash.from_uint32_t(0);
      }
    }






    __host__ __device__ void block_info_t::get_coin_base(
        ArithEnv &arith,
        bn_t &coin_base)
    {
      cgbn_load(arith.env, coin_base, &(this->coin_base));
    }


    __host__ __device__ void block_info_t::get_time_stamp(
        ArithEnv &arith,
        bn_t &time_stamp)
    {
      cgbn_load(arith.env, time_stamp, &(this->time_stamp));
    }

    __host__ __device__ void block_info_t::get_number(
        ArithEnv &arith,
      bn_t &number)
    {
      cgbn_load(arith.env, number, &(this->number));
    }


    __host__ __device__ void block_info_t::get_difficulty(
        ArithEnv &arith,
      bn_t &difficulty)
    {
      cgbn_load(arith.env, difficulty, &(this->difficulty));
    }

    __host__ __device__ void block_info_t::get_prevrandao(
        ArithEnv &arith,
      bn_t &val)
    {
      cgbn_load(arith.env, val, &(this->prevrandao));
    }


    __host__ __device__ void block_info_t::get_gas_limit(
        ArithEnv &arith,
      bn_t &gas_limit)
    {
      cgbn_load(arith.env, gas_limit, &(this->gas_limit));
    }


    __host__ __device__ void block_info_t::get_chain_id(
        ArithEnv &arith,
      bn_t &chain_id)
    {
      cgbn_load(arith.env, chain_id, &(this->chain_id));
    }


    __host__ __device__ void block_info_t::get_base_fee(
        ArithEnv &arith,
      bn_t &base_fee)
    {
      cgbn_load(arith.env, base_fee, &(this->base_fee));
    }

    __host__ __device__ int32_t block_info_t::get_previous_hash(
        ArithEnv &arith,
        bn_t &previous_hash,
        const bn_t &previous_number)
    {
      uint32_t idx = 0;
      bn_t number;
      // ge tthe current number
      get_number(arith, number);
      // if the rquest number is greater than the current block number
      if (cgbn_compare(arith.env, number, previous_number) < 1)
      {
        cgbn_set_ui32(arith.env, previous_hash, 0);
        return 0;
      }
      // get the distance from the current block number to the requested block number
      cgbn_sub(arith.env, number, number, previous_number);
      idx = cgbn_get_ui32(arith.env, number) - 1;
      // only the last 256 blocks are stored
      if (idx > 255)
      {
        cgbn_set_ui32(arith.env, previous_hash, 0);
        return 0;
      }
      cgbn_load(arith.env, previous_hash, &(previous_blocks[idx].hash));
      return 1;
    }

    __host__ __device__ void block_info_t::print()
    {
      uint32_t idx = 0;
      printf("BLOCK: \n");
      printf("COINBASE: ");
      coin_base.print();
      printf("TIMESTAMP: ");
      time_stamp.print();
      printf("NUMBER: ");
      number.print();
      printf("DIFICULTY: ");
      difficulty.print();
      printf("GASLIMIT: ");
      gas_limit.print();
      printf("CHAINID: ");
      chain_id.print();
      printf("BASE_FEE: ");
      base_fee.print();
      printf("PREVIOUS_BLOCKS: \n");
      for (idx = 0; idx < 256; idx++)
      {
        printf("NUMBER: ");
        previous_blocks[idx].number.print();
        printf("HASH: ");
        previous_blocks[idx].hash.print();
        printf("\n");
        if (previous_blocks[idx].number == 0)
        {
          break;
        }
      }
    }

    __host__ cJSON * block_info_t::to_json()
    {
      uint32_t idx = 0;
      char *hex_string_ptr = new char[EVM_WORD_SIZE * 2 + 3];
      cJSON *block_json = nullptr;
      cJSON *previous_blocks_json = nullptr;
      cJSON *previous_block_json = nullptr;

      block_json = cJSON_CreateObject();

      hex_string_ptr = coin_base.to_hex(hex_string_ptr, 0, 5);
      cJSON_AddStringToObject(block_json, "currentCoinbase", hex_string_ptr);

      hex_string_ptr = time_stamp.to_hex(hex_string_ptr);
      cJSON_AddStringToObject(block_json, "currentTimestamp", hex_string_ptr);

      hex_string_ptr = number.to_hex(hex_string_ptr);
      cJSON_AddStringToObject(block_json, "currentNumber", hex_string_ptr);

      hex_string_ptr = difficulty.to_hex(hex_string_ptr);
      cJSON_AddStringToObject(block_json, "currentDifficulty", hex_string_ptr);

      hex_string_ptr = prevrandao.to_hex(hex_string_ptr);
      cJSON_AddStringToObject(block_json, "currentGasLimit", hex_string_ptr);

      hex_string_ptr = gas_limit.to_hex(hex_string_ptr);
      cJSON_AddStringToObject(block_json, "currentChainId", hex_string_ptr);

      hex_string_ptr = base_fee.to_hex(hex_string_ptr);
      cJSON_AddStringToObject(block_json, "currentBaseFee", hex_string_ptr);

      previous_blocks_json = cJSON_CreateArray();
      bn_t number;
      for (idx = 0; idx < 256; idx++)
      {
        previous_block_json = cJSON_CreateObject();

        
        hex_string_ptr = previous_blocks[idx].number.to_hex(hex_string_ptr);
        cJSON_AddStringToObject(previous_block_json, "number", hex_string_ptr);

        
        hex_string_ptr = previous_blocks[idx].hash.to_hex(hex_string_ptr);
        cJSON_AddStringToObject(previous_block_json, "hash", hex_string_ptr);

        cJSON_AddItemToArray(previous_blocks_json, previous_block_json);

        if (previous_blocks[idx].number == 0)
        {
          break;
        }
      }

      cJSON_AddItemToObject(block_json, "previousHashes", previous_blocks_json);
      
      delete[] hex_string_ptr;
      hex_string_ptr = nullptr;
      return block_json;
    }

    __host__ int32_t get_block_info(
        block_info_t* &block_info_ptr,
        const cJSON *json,
        int32_t managed) {
      
      if (managed == 1) {
        CUDA_CHECK(cudaMallocManaged(&block_info_ptr, sizeof(block_info_t)));
        block_info_ptr->from_json(json);
      } else {
        block_info_ptr = new block_info_t(json);
      }
      return 1;
    }

    __host__ int32_t free_block_info(
        block_info_t* &block_info_ptr,
        int32_t managed) {
      if (managed == 1) {
        CUDA_CHECK(cudaFree(block_info_ptr));
      } else {
        delete block_info_ptr;
      }
      return 1;
    }
  }
}
