// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2024-06-20
// SPDX-License-Identifier: MIT
#include "include/block.cuh"


__host__ __device__ block_t::block_t(
    arith_t arith,
    block_data_t *content
) : _arith(arith),
    _content(content)
{
}

__host__ block_t::block_t(
    arith_t arith,
    const cJSON *test
) : _arith(arith)
{
  cJSON *block_json = NULL;
  cJSON *element_json = NULL;
  cJSON *previous_blocks_json = NULL;
  size_t idx = 0;
  _content = NULL;
#ifndef ONLY_CPU
  CUDA_CHECK(cudaMallocManaged(
      (void **)&(_content),
      sizeof(block_data_t)));
#else
  _content = new block_data_t;
#endif

  block_json = cJSON_GetObjectItemCaseSensitive(test, "env");

  element_json = cJSON_GetObjectItemCaseSensitive(block_json, "currentCoinbase");
  _arith.cgbn_memory_from_hex_string(
    _content->coin_base,
    element_json->valuestring
  );

  element_json = cJSON_GetObjectItemCaseSensitive(block_json, "currentTimestamp");
  _arith.cgbn_memory_from_hex_string(
    _content->time_stamp,
    element_json->valuestring
  );

  element_json = cJSON_GetObjectItemCaseSensitive(block_json, "currentNumber");
  _arith.cgbn_memory_from_hex_string(
    _content->number,
    element_json->valuestring
  );

  element_json = cJSON_GetObjectItemCaseSensitive(block_json, "currentDifficulty");
  _arith.cgbn_memory_from_hex_string(
    _content->difficulty,
    element_json->valuestring
  );

  element_json = cJSON_GetObjectItemCaseSensitive(block_json, "currentRandom");
  if (element_json != NULL)
  {
    _arith.cgbn_memory_from_hex_string(
      _content->prevrandao,
      element_json->valuestring
    );
  }

  element_json = cJSON_GetObjectItemCaseSensitive(block_json, "currentGasLimit");
  _arith.cgbn_memory_from_hex_string(
    _content->gas_limit,
    element_json->valuestring
  );

  // element_json=cJSON_GetObjectItemCaseSensitive(block_json, "currentChainId");
  //_arith.cgbn_memory_from_hex_string(_content->chain_id, element_json->valuestring);
  _arith.cgbn_memory_from_size_t(_content->chain_id, 1);

  element_json = cJSON_GetObjectItemCaseSensitive(block_json, "currentBaseFee");
  _arith.cgbn_memory_from_hex_string(
    _content->base_fee,
    element_json->valuestring
  );

  previous_blocks_json = cJSON_GetObjectItemCaseSensitive(block_json, "previousHashes");
  if (previous_blocks_json != NULL and cJSON_IsArray(previous_blocks_json))
  {
    idx = 0;
    cJSON_ArrayForEach(element_json, previous_blocks_json)
    {
      element_json = cJSON_GetObjectItemCaseSensitive(element_json, "number");
      _arith.cgbn_memory_from_hex_string(
        _content->previous_blocks[idx].number,
        element_json->valuestring
      );

      element_json = cJSON_GetObjectItemCaseSensitive(element_json, "hash");
      _arith.cgbn_memory_from_hex_string(
        _content->previous_blocks[idx].hash,
        element_json->valuestring
      );
      idx++;
    }
  }
  else
  {
    idx = 0;
    // TODO: maybe fill with something else
    _arith.cgbn_memory_from_size_t(_content->previous_blocks[0].number, 0);

    element_json = cJSON_GetObjectItemCaseSensitive(block_json, "previousHash");

    if (element_json != NULL){
      _arith.cgbn_memory_from_hex_string(_content->previous_blocks[0].hash, element_json->valuestring);
    } else {
      _arith.cgbn_memory_from_size_t(_content->previous_blocks[0].hash, 0);
    }

    idx++;
  }

  // fill the remaing parents with 0
  for (size_t jdx = idx; jdx < 256; jdx++)
  {
    _arith.cgbn_memory_from_size_t(_content->previous_blocks[jdx].number, 0);
    _arith.cgbn_memory_from_size_t(_content->previous_blocks[jdx].hash, 0);
  }
}


__host__ __device__ block_t::~block_t()
{
  _content = NULL;
}


__host__ void block_t::free_content()
{
#ifndef ONLY_CPU
  CUDA_CHECK(cudaFree(_content));
#else
  delete _content;
#endif
  _content = NULL;
}


__host__ __device__ void block_t::get_coin_base(
    bn_t &coin_base)
{
  cgbn_load(_arith._env, coin_base, &(_content->coin_base));
}


__host__ __device__ void block_t::get_time_stamp(
    bn_t &time_stamp)
{
  cgbn_load(_arith._env, time_stamp, &(_content->time_stamp));
}

__host__ __device__ void block_t::get_number(
  bn_t &number)
{
  cgbn_load(_arith._env, number, &(_content->number));
}


__host__ __device__ void block_t::get_difficulty(
  bn_t &difficulty)
{
  cgbn_load(_arith._env, difficulty, &(_content->difficulty));
}

__host__ __device__ void block_t::get_prevrandao(
  bn_t &val)
{
  cgbn_load(_arith._env, val, &(_content->prevrandao));
}


__host__ __device__ void block_t::get_gas_limit(
  bn_t &gas_limit)
{
  cgbn_load(_arith._env, gas_limit, &(_content->gas_limit));
}


__host__ __device__ void block_t::get_chain_id(
  bn_t &chain_id)
{
  cgbn_load(_arith._env, chain_id, &(_content->chain_id));
}


__host__ __device__ void block_t::get_base_fee(
  bn_t &base_fee)
{
  cgbn_load(_arith._env, base_fee, &(_content->base_fee));
}

__host__ __device__ void block_t::get_previous_hash(
    bn_t &previous_hash,
    bn_t &previous_number,
    uint32_t &error_code)
{
  uint32_t idx = 0;
  bn_t number;
  // ge tthe current number
  get_number(number);
  // if the rquest number is greater than the current block number
  if (cgbn_compare(_arith._env, number, previous_number) < 1)
  {
    error_code = ERR_BLOCK_INVALID_NUMBER;
  }
  // get the distance from the current block number to the requested block number
  cgbn_sub(_arith._env, number, number, previous_number);
  idx = cgbn_get_ui32(_arith._env, number) - 1;
  // only the last 256 blocks are stored
  if (idx > 255)
  {
    error_code = ERR_BLOCK_INVALID_NUMBER;
  }
  if (error_code == ERR_NONE)
    cgbn_load(_arith._env, previous_hash, &(_content->previous_blocks[idx].hash));
  else
    cgbn_set_ui32(_arith._env, previous_hash, 0);
}

__host__ __device__ void block_t::print()
{
  uint32_t idx = 0;
  bn_t number;
  printf("BLOCK: \n");
  printf("COINBASE: ");
  _arith.print_cgbn_memory(_content->coin_base);
  printf("TIMESTAMP: ");
  _arith.print_cgbn_memory(_content->time_stamp);
  printf("NUMBER: ");
  _arith.print_cgbn_memory(_content->number);
  printf("DIFICULTY: ");
  _arith.print_cgbn_memory(_content->difficulty);
  printf("GASLIMIT: ");
  _arith.print_cgbn_memory(_content->gas_limit);
  printf("CHAINID: ");
  _arith.print_cgbn_memory(_content->chain_id);
  printf("BASE_FEE: ");
  _arith.print_cgbn_memory(_content->base_fee);
  printf("PREVIOUS_BLOCKS: \n");
  for (idx = 0; idx < 256; idx++)
  {
    printf("NUMBER: ");
    _arith.print_cgbn_memory(_content->previous_blocks[idx].number);
    printf("HASH: ");
    _arith.print_cgbn_memory(_content->previous_blocks[idx].hash);
    printf("\n");
    cgbn_load(_arith._env, number, &(_content->previous_blocks[idx].number));
    if (cgbn_compare_ui32(_arith._env, number, 0) == 0)
    {
      break;
    }
  }
}

__host__ cJSON * block_t::json()
{
  uint32_t idx = 0;
  char *hex_string_ptr = new char[arith_t::BYTES * 2 + 3];
  cJSON *block_json = NULL;
  cJSON *previous_blocks_json = NULL;
  cJSON *previous_block_json = NULL;

  block_json = cJSON_CreateObject();

  _arith.hex_string_from_cgbn_memory(hex_string_ptr, _content->coin_base, 5);
  cJSON_AddStringToObject(block_json, "currentCoinbase", hex_string_ptr);

  _arith.hex_string_from_cgbn_memory(hex_string_ptr, _content->time_stamp);
  cJSON_AddStringToObject(block_json, "currentTimestamp", hex_string_ptr);

  _arith.hex_string_from_cgbn_memory(hex_string_ptr, _content->number);
  cJSON_AddStringToObject(block_json, "currentNumber", hex_string_ptr);

  _arith.hex_string_from_cgbn_memory(hex_string_ptr, _content->difficulty);
  cJSON_AddStringToObject(block_json, "currentDifficulty", hex_string_ptr);

  _arith.hex_string_from_cgbn_memory(hex_string_ptr, _content->gas_limit);
  cJSON_AddStringToObject(block_json, "currentGasLimit", hex_string_ptr);

  _arith.hex_string_from_cgbn_memory(hex_string_ptr, _content->chain_id);
  cJSON_AddStringToObject(block_json, "currentChainId", hex_string_ptr);

  _arith.hex_string_from_cgbn_memory(hex_string_ptr, _content->base_fee);
  cJSON_AddStringToObject(block_json, "currentBaseFee", hex_string_ptr);

  previous_blocks_json = cJSON_CreateArray();
  bn_t number;
  for (idx = 0; idx < 256; idx++)
  {
    previous_block_json = cJSON_CreateObject();

    _arith.hex_string_from_cgbn_memory(
      hex_string_ptr,
      _content->previous_blocks[idx].number
    );
    cJSON_AddStringToObject(previous_block_json, "number", hex_string_ptr);

    _arith.hex_string_from_cgbn_memory(
      hex_string_ptr,
      _content->previous_blocks[idx].hash
    );
    cJSON_AddStringToObject(previous_block_json, "hash", hex_string_ptr);

    cJSON_AddItemToArray(previous_blocks_json, previous_block_json);

    cgbn_load(_arith._env, number, &(_content->previous_blocks[idx].number));
    if (cgbn_compare_ui32(_arith._env, number, 0) == 0)
    {
      break;
    }
  }

  cJSON_AddItemToObject(block_json, "previousHashes", previous_blocks_json);
  
  delete[] hex_string_ptr;
  hex_string_ptr = NULL;
  return block_json;
}
