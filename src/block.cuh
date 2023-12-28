// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#ifndef _BLOCK_H_
#define _BLOCK_H_

#include "utils.h"

/**
 * The block class is used to store the block information
 * before the transaction are done. YP: H
 */
template <class params>
class block_t
{
public:
  /**
   * The arithmetical environment used by the arbitrary length
   * integer library.
   */
  typedef arith_env_t<params> arith_t;
  /**
   * The arbitrary length integer type.
   */
  typedef typename arith_t::bn_t bn_t;
  /**
   * The arbitrary length integer type used for the storage.
   * It is defined as the EVM word type.
   */
  typedef cgbn_mem_t<params::BITS> evm_word_t;

  /**
   * The previous block hash information.
   *  (YP: \f$P(h, n, a)\f$)
  */
  typedef struct
  {
    evm_word_t number; /**< The number of the block 0 if none */
    evm_word_t hash;  /**< The hash of the block */
  } block_hash_t;

  /**
   * The block information.
   *  (YP: \f$H\f$)
   * It does NOT contains:
   *  the ommers or their hases (YP: \f$U, H_{o}\f$)
   *  the state root (YP: \f$S, H_{r}\f$)
   *  the transactions root (YP: \f$T, H_{t}\f$)
   *  the receipts root (YP: \f$R, H_{e}\f$)
   *  the logs bloom (YP: \f$H_{b}\f$)
   *  the gas used (YP: \f$H_{g}\f$)
   *  the extra data (YP: \f$H_{x}\f$)
   *  the mix hash (YP: \f$H_{m}\f$)
   *  the nonce (YP: \f$H_{n}\f$)
  */
  typedef struct alignas(32)
  {
    evm_word_t coin_base; /**< The address of the block miner (YP: \f$H_{c}\f$) */
    evm_word_t difficulty; /**< The difficulty of the block (YP: \f$H_{d}\f$) */
    evm_word_t number; /**< The number of the block (YP: \f$H_{i}\f$) */
    evm_word_t gas_limit; /**< The gas limit of the block (YP: \f$H_{l}\f$) */
    evm_word_t time_stamp; /**< The timestamp of the block (YP: \f$H_{s}\f$) */
    evm_word_t base_fee; /**< The base fee of the block (YP: \f$H_{f}\f$)*/
    evm_word_t chain_id; /**< The chain id of the block */
    block_hash_t previous_blocks[256]; /**< The previous block hashes (YP: \f$H_{p}\f$) */
  } block_data_t;

  block_data_t *_content; /**< The block information content */
  arith_t _arith; /**< The arithmetical environment */

  /**
   * The constructor of the block class.
   * @param arith The arithmetical environment
   * @param content The block information content
   */
  __device__ __forceinline__ block_t(
      arith_t arith,
      block_data_t *content
  ) : _arith(arith),
      _content(content)
  {
  }

  /**
   * The constructor of the block class.
   * @param arith The arithmetical environment
   * @param test The block information in JSON format
   */
  __host__ block_t(
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
      _arith.cgbn_memory_from_hex_string(
        _content->previous_blocks[0].hash,
        element_json->valuestring
      );
      idx++;
    }

    // fill the remaing parents with 0
    for (size_t jdx = idx; jdx < 256; jdx++)
    {
      _arith.cgbn_memory_from_size_t(_content->previous_blocks[jdx].number, 0);
      _arith.cgbn_memory_from_size_t(_content->previous_blocks[jdx].hash, 0);
    }
  }

  /**
   * The destructor of the block class.
  */
  __host__ __device__ __forceinline__ ~block_t()
  {
    _content = NULL;
  }

  /**
   * deallocates the block information content
  */
  __host__ void free_content()
  {
#ifndef ONLY_CPU
    CUDA_CHECK(cudaFree(_content));
#else
    delete _content;
#endif
    _content = NULL;
  }

  /**
   * Get the coin base of the block.
   * @param[out] coin_base The coin base of the block
  */
  __host__ __device__ __forceinline__ void get_coin_base(
      bn_t &coin_base)
  {
    cgbn_load(_arith._env, coin_base, &(_content->coin_base));
  }

  /**
   * Get the time stamp of the block.
   * @param[out] time_stamp The time stamp of the block
  */
  __host__ __device__ __forceinline__ void get_time_stamp(
      bn_t &time_stamp)
  {
    cgbn_load(_arith._env, time_stamp, &(_content->time_stamp));
  }

  /**
   * Get the number of the block.
   * @param[out] number The number of the block
  */
  __host__ __device__ __forceinline__ void get_number(
    bn_t &number)
  {
    cgbn_load(_arith._env, number, &(_content->number));
  }

  /**
   * Get the difficulty of the block.
   * @param[out] difficulty The difficulty of the block
  */
  __host__ __device__ __forceinline__ void get_difficulty(
    bn_t &difficulty)
  {
    cgbn_load(_arith._env, difficulty, &(_content->difficulty));
  }

  /**
   * Get the gas limit of the block.
   * @param[out] gas_limit The gas limit of the block
  */
  __host__ __device__ __forceinline__ void get_gas_limit(
    bn_t &gas_limit)
  {
    cgbn_load(_arith._env, gas_limit, &(_content->gas_limit));
  }

  /**
   * Get the chain id of the block.
  */
  __host__ __device__ __forceinline__ void get_chain_id(
    bn_t &chain_id)
  {
    cgbn_load(_arith._env, chain_id, &(_content->chain_id));
  }

  /**
   * Get the base fee of the block.
   * @param[out] base_fee The base fee of the block
  */
  __host__ __device__ __forceinline__ void get_base_fee(
    bn_t &base_fee)
  {
    cgbn_load(_arith._env, base_fee, &(_content->base_fee));
  }

  /**
   * Get the has of a previous block given by the number
   * @param[out] previous_hash The hash of the previous block
   * @param[in] previous_number The number of the previous block
   * @param[out] error_code The error code
  */
  __host__ __device__ __forceinline__ void get_previous_hash(
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

  /**
   * Print the block information.
  */
  __host__ __device__ void print()
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

  /**
   * Get the block information in JSON format.
   * @return The block information in JSON format
  */
  __host__ cJSON *json()
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
};

#endif