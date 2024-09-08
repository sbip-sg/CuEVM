#ifndef _CUEVM_EVM_UTILS_H_
#define _CUEVM_EVM_UTILS_H_

#include <stdint.h>
#include <cuda.h>
#include <CuEVM/utils/arith.cuh>
#include <CuEVM/core/byte_array.cuh>

namespace CuEVM {
    namespace utils {

    /**
     * Get the contract address from the sender address and the sender nonce.
     * For the simple CREATE operation.
     * @param[in] arith The arithmetic environment
     * @param[out] contract_address The contract address
     * @param[in] sender_address The sender address
     * @param[in] sender_nonce The sender nonce
     */
    __host__ __device__ int32_t get_contract_address_create(
        ArithEnv &arith,
        bn_t &contract_address,
        const bn_t &sender_address,
        const bn_t &sender_nonce);

    /**
     * Get the contract address from the sender address, the salt, and the init code.
     * For the CREATE2 operation.
     * @param[in] arith The arithmetic environment
     * @param[out] contract_address The contract address
     * @param[in] sender_address The sender address
     * @param[in] salt The salt
     * @param[in] init_code The init code
     */
    __host__ __device__ int32_t get_contract_address_create2(
      ArithEnv &arith,
      bn_t &contract_address,
      const bn_t &sender_address,
      const bn_t &salt,
      const CuEVM::byte_array_t &init_code);
    

    /**
     * If it is a hex character.
     * @param[in] hex The character.
     * @return 1 if it is a hex character, 0 otherwise.
     */
    __host__ __device__ int32_t is_hex(
      const char hex);
    /**
     * Get the hex string from a nibble.
     * @param[in] nibble The nibble.
     * @return The hex string.
     */
    __host__ __device__ char hex_from_nibble(const uint8_t nibble);
    /**
     * Get the nibble from a hex string.
     * @param[in] hex The hex string.
     * @return The nibble.
     */
    __host__ __device__ uint8_t nibble_from_hex(const char hex);
    /**
     * Check if a character is a hex character.
     * @param[in] hex The character.
     * @return 1 if it is a hex character, 0 otherwise.
     */
    __host__ __device__ int32_t is_hex(const char hex);
    /**
     * Get the byte from two nibbles.
     * @param[in] high The high nibble.
     * @param[in] low The low nibble.
     */
    __host__ __device__ uint8_t byte_from_nibbles(const uint8_t high, const uint8_t low);
    /**
     * Get the hex string from a byte.
     * @param[in] byte The byte.
     * @param[out] dst The destination hex string.
     */
    __host__ __device__ void hex_from_byte(char *dst, const uint8_t byte);
    /**
     * Get the byte from two hex characters.
     * @param[in] high The high hex character.
     * @param[in] low The low hex character.
     * @return The byte.
    */
    __host__ __device__ uint8_t byte_from_two_hex_char(const char high, const char low);
    /**
     * Get the number of bytes oh a string
     * @param[in] hex_string
     * @return the number of bytes
     */
    __host__ __device__ int32_t hex_string_length(
      const char *hex_string);
    /**
     * Clean the hex string from prefix and return the length
     * @param[inout] hex_string
     * @return the length of the hex string
     */
    __host__ __device__ int32_t clean_hex_string(
      char **hex_string);
    /**
     * Remove the leading zeros of an hex string
     * @param[inout] hex_string
     * @return the length of the hex string
     */
    __host__ __device__ int32_t hex_string_without_leading_zeros(
      char *hex_string);
    

    /**
     * Get the json object from a file.
     * @param[in] filepath The file path.
     * @return The json object.
     */
    __host__ cJSON *get_json_from_file(const char *filepath);
    } // namespace utils
} // namespace CuEVM

#endif // _CUEVM_EVM_UTILS_H_