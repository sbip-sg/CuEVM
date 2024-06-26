// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#ifndef _BLOCK_H_
#define _BLOCK_H_

#include "arith.cuh"
#include <cjson/cJSON.h>
#include <cuda.h>
#include <stdint.h>

namespace cuEVM {
  namespace block {
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
    typedef struct
    {
      evm_word_t coin_base; /**< The address of the block miner (YP: \f$H_{c}\f$) */
      evm_word_t difficulty; /**< The difficulty of the block (YP: \f$H_{d}\f$) */
      evm_word_t prevrandao; /**< The prevrandao EIP-4399 */
      evm_word_t number; /**< The number of the block (YP: \f$H_{i}\f$) */
      evm_word_t gas_limit; /**< The gas limit of the block (YP: \f$H_{l}\f$) */
      evm_word_t time_stamp; /**< The timestamp of the block (YP: \f$H_{s}\f$) */
      evm_word_t base_fee; /**< The base fee of the block (YP: \f$H_{f}\f$)*/
      evm_word_t chain_id; /**< The chain id of the block */
      block_hash_t previous_blocks[256]; /**< The previous block hashes (YP: \f$H_{p}\f$) */
    } block_data_t;
    /**
     * The block class is used to store the block information
     * before the transaction are done. YP: H
     */
    class EVMBlockInfo
    {
    private:
      ArithEnv _arith; /**< The arithmetical environment */

    public:
      block_data_t *content; /**< The block information content */

      /**
       * The constructor of the block class.
       * @param arith The arithmetical environment
       * @param content The block information content
       */
      __host__ __device__  EVMBlockInfo(
          ArithEnv arith,
          block_data_t *content
      );

      /**
       * The constructor of the block class.
       * @param arith The arithmetical environment
       * @param test The block information in JSON format
       */
      __host__ EVMBlockInfo(
          ArithEnv arith,
          const cJSON *test
      );

      /**
       * The destructor of the block class.
      */
      __host__ __device__ ~EVMBlockInfo();

      /**
       * deallocates the block information content
      */
      __host__ void free_content();

      /**
       * Get the coin base of the block.
       * @param[out] coin_base The coin base of the block
      */
      __host__ __device__ void get_coin_base(
          bn_t &coin_base);

      /**
       * Get the time stamp of the block.
       * @param[out] time_stamp The time stamp of the block
      */
      __host__ __device__ void get_time_stamp(
          bn_t &time_stamp);
      /**
       * Get the number of the block.
       * @param[out] number The number of the block
      */
      __host__ __device__ void get_number(
        bn_t &number);

      /**
       * Get the difficulty of the block.
       * @param[out] difficulty The difficulty of the block
      */
      __host__ __device__ void get_difficulty(
        bn_t &difficulty);

      __host__ __device__ void get_prevrandao(
        bn_t &val);

      /**
       * Get the gas limit of the block.
       * @param[out] gas_limit The gas limit of the block
      */
      __host__ __device__ void get_gas_limit(
        bn_t &gas_limit);

      /**
       * Get the chain id of the block.
      */
      __host__ __device__ void get_chain_id(
        bn_t &chain_id);

      /**
       * Get the base fee of the block.
       * @param[out] base_fee The base fee of the block
      */
      __host__ __device__ void get_base_fee(
        bn_t &base_fee);

      /**
       * Get the has of a previous block given by the number
       * @param[out] previous_hash The hash of the previous block
       * @param[in] previous_number The number of the previous block
       * @param[out] error_code The error code
      */
      __host__ __device__ void get_previous_hash(
          bn_t &previous_hash,
          bn_t &previous_number,
          uint32_t &error_code);

      /**
       * Print the block information.
      */
      __host__ __device__ void print();

      /**
       * Get the block information in JSON format.
       * @return The block information in JSON format
      */
      __host__ cJSON *to_json();
    };
  }
}
#endif
