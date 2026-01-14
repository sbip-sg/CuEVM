#pragma once

#include <cjson/cJSON.h>
#include <cuda.h>
#include <stdint.h>

#include <CuEVM/core/evm_word.cuh>

namespace CuEVM {

/**
 * The previous block hash information.
 *  (YP: \f$P(h, n, a)\f$)
 */
struct block_hash_t {
    evm_word_t number; /**< The number of the block 0 if none */
    evm_word_t hash;   /**< The hash of the block */
};
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
struct block_info_t {
    evm_word_t coin_base;              /**< The address of the block miner (YP: \f$H_{c}\f$) */
    evm_word_t difficulty;             /**< The difficulty of the block (YP: \f$H_{d}\f$) */
    evm_word_t prevrandao;             /**< The prevrandao EIP-4399 */
    gas_t number;                      /**< The number of the block (YP: \f$H_{i}\f$) */
    evm_word_t gas_limit;              /**< The gas limit of the block (YP: \f$H_{l}\f$) */
    gas_t time_stamp;                  /**< The timestamp of the block (YP: \f$H_{s}\f$) */
    evm_word_t base_fee;               /**< The base fee of the block (YP: \f$H_{f}\f$)*/
    evm_word_t chain_id;               /**< The chain id of the block */
    block_hash_t previous_blocks[256]; /**< The previous block hashes (YP: \f$H_{p}\f$) */

    /**
     * The default constructor of the block information.
     */
    __host__ __device__ block_info_t();

    /**
     * The destructor of the block information.
     */
    __host__ __device__ ~block_info_t();

    /**
     * The constructor of the block information from json
     */
    __host__ block_info_t(const cJSON *json);

    /**
     * Get the block information from json
     * @param[in] json The json object
     * @return 1 if the operation is succesfull, 0 otherwise
     */
    __host__ int32_t from_json(const cJSON *json);

    /**
     * Get the previous block hash.
     * @param[in] arith The arithmetical environment
     * @param[out] previous_hash The previous block hash
     * @param[in] previous_number The number of the previous block
     * @return the error code, 0 if the previous block hash is found
     */
    __host__ __device__ int32_t get_previous_hash(evm_word_t &previous_hash, const evm_word_t &previous_number) const;

    /**
     * Print the block information.
     */
    __host__ __device__ void print() const;
};

/**
 * Get the block information.
 * @param[out] block_info_ptr The block information
 * @param[in] json The json object
 * @param[in] managed 1 if the memory is managed, 0 otherwise
 * @return 1 if the operation is succesfull, 0 otherwise
 */
__host__ int32_t get_block_info(const cJSON *json);
extern __device__ block_info_t *global_block_info;
}  // namespace CuEVM
