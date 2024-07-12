// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#ifndef _CUEVM_MESSAGE_H_
#define _CUEVM_MESSAGE_H_

#include <CuCrypto/keccak.cuh>

#include "../utils/arith.cuh"
#include "byte_array.cuh"
#include "jump_destinations.cuh"
#include "../state/state.cuh"

#include <CuCrypto/keccak.cuh>

namespace cuEVM {
  /**
   * The message call struct.
   * YP: \f$M\f$
   */
  struct evm_message_call_t {
    evm_word_t sender;           /**< The sender address YP: \f$s\f$ */
    evm_word_t recipient;        /**< The recipient address YP: \f$r\f$ also \f$I_{a}\f$ */
    evm_word_t contract_address; /**< The contract address YP: \f$c\f$ */
    evm_word_t gas_limit;        /**< The gas limit YP: \f$g\f$ */
    evm_word_t value;            /**< The value YP: \f$v\f$ or \f$v^{'}\f$ for DelegateCALL */
    uint32_t depth;              /**< The depth YP: \f$e\f$ */
    uint32_t call_type;           /**< The call type internal has the opcode */
    evm_word_t storage_address;  /**< The storage address YP: \f$a\f$ */
    cuEVM::byte_array_t data;         /**< The data YP: \f$d\f$ */
    cuEVM::byte_array_t byte_code;    /**< The byte code YP: \f$b\f$ or \f$I_{b}\f$*/
    evm_word_t return_data_offset; /**< The return data offset in memory */
    evm_word_t return_data_size;   /**< The return data size in memory */
    uint32_t static_env;         /**< The static flag (STATICCALL) YP: \f$w\f$ */
    cuEVM::jump_destinations_t *jump_destinations; /**< The jump destinations */


    /**
     * The constructor.
     * @param[in] arith The arithmetical environment.
     * @param[in] sender The sender address YP: \f$s\f$.
     * @param[in] recipient The recipient address YP: \f$r\f$.
     * @param[in] contract_address The contract address YP: \f$c\f$.
     * @param[in] gas_limit The gas limit YP: \f$g\f$.
     * @param[in] value The value YP: \f$v\f$ or \f$v^{'}\f$ for DelegateCALL.
     * @param[in] depth The depth YP: \f$e\f$.
     * @param[in] call_type The call type internal has the opcode.
     * @param[in] storage_address The storage address YP: \f$a\f$.
     * @param[in] data The data YP: \f$d\f$.
     * @param[in] byte_code The byte code YP: \f$b\f$.
     * @param[in] return_data_offset The return data offset in memory.
     * @param[in] return_data_size The return data size in memory.
     * @param[in] static_env The static flag (STATICCALL) YP: \f$w\f$.
     */
    __host__ __device__ evm_message_call_t(
        ArithEnv &arith,
        bn_t &sender,
        bn_t &recipient,
        bn_t &contract_address,
        bn_t &gas_limit,
        bn_t &value,
        uint32_t depth,
        uint32_t call_type,
        bn_t &storage_address,
        cuEVM::byte_array_t &data,
        cuEVM::byte_array_t &byte_code,
        bn_t &return_data_offset,
        bn_t &return_data_size,
        uint32_t static_env = 0
        );

    /**
     * The destructor.
     */
    __host__ __device__ ~evm_message_call_t();

    /**
     * Get the sender address.
     * @param[in] arith The arithmetical environment.
     * @param[out] sender The sender address YP: \f$s\f$.
    */
    __host__ __device__ void get_sender(
        ArithEnv &arith,
        bn_t &sender);

    /**
     * Get the recipient address.
     * @param[in] arith The arithmetical environment.
     * @param[out] recipient The recipient address YP: \f$r\f$.
    */
    __host__ __device__ void get_recipient(
        ArithEnv &arith,
        bn_t &recipient);

    /**
     * Get the contract address.
     * @param[in] arith The arithmetical environment.
     * @param[out] contract_address The contract address YP: \f$c\f$.
    */
    __host__ __device__ void get_contract_address(
        ArithEnv &arith,
        bn_t &contract_address);

    /**
     * Get the gas limit.
     * @param[in] arith The arithmetical environment.
     * @param[out] gas_limit The gas limit YP: \f$g\f$.
    */
    __host__ __device__ void get_gas_limit(
        ArithEnv &arith,
        bn_t &gas_limit);
    /**
     * Get the value.
     * @param[in] arith The arithmetical environment.
     * @param[out] value The value YP: \f$v\f$ or \f$v^{'}\f$ for DelegateCALL.
     */
    __host__ __device__ void get_value(
        ArithEnv &arith,
        bn_t &value);

    /**
     * Get the depth.
     * @return The depth YP: \f$e\f$.
    */
    __host__ __device__ uint32_t get_depth();

    /**
     * Get the call type.
     * @return The call type internal has the opcode YP: \f$w\f$.
    */
    __host__ __device__ uint32_t get_call_type();

    /**
     * Get the storage address.
     * @param[in] arith The arithmetical environment.
     * @param[out] storage_address The storage address YP: \f$a\f$.
    */
    __host__ __device__ void get_storage_address(
        ArithEnv &arith,
        bn_t &storage_address);

    /**
     * Get the call/init data.
     * @return The data YP: \f$d\f$.
     */
    __host__ __device__ cuEVM::byte_array_t get_data();

    /**
     * Get the byte code.
     * @return The byte code YP: \f$b\f$.
     */
    __host__ __device__ cuEVM::byte_array_t get_byte_code();

    /**
     * Get the return data offset.
     * @param[in] arith The arithmetical environment.
     * @param[out] return_data_offset The return data offset in memory.
     */
    __host__ __device__ void get_return_data_offset(
        ArithEnv &arith,
        bn_t &return_data_offset);
    /**
     * Get the return data size.
     * @param[in] arith The arithmetical environment.
     * @param[out] return_data_size The return data size in memory.
     */
    __host__ __device__ void get_return_data_size(
        ArithEnv &arith,
        bn_t &return_data_size);
    /**
     * Get the static flag.
     * @return The static flag (STATICCALL) YP: \f$w\f$.
     */
    __host__ __device__ uint32_t get_static_env();
    /**
     * Set the gas limit.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit YP: \f$g\f$.
     */
    __host__ __device__ void set_gas_limit(
        ArithEnv &arith,
        bn_t &gas_limit);

    /**
     * Set the call data.
     * @param[in] data The data YP: \f$d\f$.
     */
    __host__ __device__ void set_data(
        cuEVM::byte_array_t &data);

    /**
     * Set the byte code.
     * @param[in] byte_code The byte code YP: \f$b\f$.
     */
    __host__ __device__ void set_byte_code(
        cuEVM::byte_array_t &byte_code);

    /**
     * Set the return data offset.
     * @param[in] arith The arithmetical environment.
     * @param[in] return_data_offset The return data offset in memory.
     */
    __host__ __device__ void set_return_data_offset(
        ArithEnv &arith,
        bn_t &return_data_offset);
    /**
     * Set the return data size.
     * @param[in] arith The arithmetical environment.
     * @param[in] return_data_size The return data size in memory.
     */
    __host__ __device__ void set_return_data_size(
        ArithEnv &arith,
        bn_t &return_data_size);
    /**
     * Get the jump destinations.
     * @return The jump destinations.
     */
    __host__ __device__ cuEVM::jump_destinations_t* get_jump_destinations();

    /**
     * Print the message.
     */
    __host__ __device__ void print();
  };

}

#endif
