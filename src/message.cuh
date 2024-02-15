// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#ifndef _MESSAGE_H_
#define _MESSAGE_H_

#include "include/utils.h"
#include "state.cuh"
#include "keccak.cuh"

/**
 * The message call class.
 * YP: \f$M\f$
 */

class message_t
{
public:
  /**
   * The arithmetical environment used by the arbitrary length
   * integer library.
   */
  typedef arith_env_t<evm_params> arith_t;
  /**
   * The arbitrary length integer type.
   */
  typedef typename arith_t::bn_t bn_t;
  /**
   * The arbitrary length integer type used for the storage.
   * It is defined as the EVM word type.
   */
  typedef cgbn_mem_t<evm_params::BITS> evm_word_t;
  /**
   * THe keccak class
  */
  typedef keccak::keccak_t keccak_t;
  static const uint32_t HASH_BYTES = 32; /**< the number of byte in hash*/

  /**
   * The message data.
   */
  typedef struct
  {
    evm_word_t sender;           /**< The sender address YP: \f$s\f$ */
    evm_word_t recipient;        /**< The recipient address YP: \f$r\f$ also \f$I_{a}\f$ */
    evm_word_t contract_address; /**< The contract address YP: \f$c\f$ */
    evm_word_t gas_limit;        /**< The gas limit YP: \f$g\f$ */
    evm_word_t value;            /**< The value YP: \f$v\f$ or \f$v^{'}\f$ for DelegateCALL */
    uint32_t depth;              /**< The depth YP: \f$e\f$ */
    uint8_t call_type;           /**< The call type internal has the opcode */
    evm_word_t storage_address;  /**< The storage address YP: \f$a\f$ */
    data_content_t data;         /**< The data YP: \f$d\f$ */
    data_content_t byte_code;    /**< The byte code YP: \f$b\f$ or \f$I_{b}\f$*/
    evm_word_t return_data_offset; /**< The return data offset in memory */
    evm_word_t return_data_size;   /**< The return data size in memory */
    uint32_t static_env;         /**< The static flag (STATICCALL) YP: \f$w\f$ */
  } message_data_t;

  message_data_t *_content; /**< The message content */
  arith_t _arith;           /**< The arithmetical environment */

  /**
   * The constructor. Takes the message parameters.
   * @param[in] arith The arithmetical environment.
   * @param[in] sender The sender address YP: \f$s\f$.
   * @param[in] recipient The recipient address YP: \f$r\f$.
   * @param[in] contract_address The contract address YP: \f$c\f$.
   * @param[in] gas_limit The gas limit YP: \f$g\f$.
   * @param[in] value The value YP: \f$v\f$ or \f$v^{'}\f$ for DelegateCALL.
   * @param[in] depth The depth YP: \f$e\f$.
   * @param[in] call_type The call type internal has the opcode YP: \f$w\f$.
   * @param[in] storage_address The storage address YP: \f$a\f$.
   * @param[in] data The data YP: \f$d\f$.
   * @param[in] data_size The data size YP: \f$|d|\f$.
   * @param[in] byte_code The byte code YP: \f$b\f$.
   * @param[in] byte_code_size The byte code size YP: \f$|b|\f$.
   * @param[in] return_data_offset The return data offset in memory.
   * @param[in] return_data_size The return data size in memory.
   * @param[in] static_env The static flag (STATICCALL) YP: \f$w\f$.
  */
  __host__ __device__ __forceinline__ message_t(
      arith_t &arith,
      bn_t &sender,
      bn_t &recipient,
      bn_t &contract_address,
      bn_t &gas_limit,
      bn_t &value,
      uint32_t depth,
      uint8_t call_type,
      bn_t &storage_address,
      uint8_t *data,
      size_t data_size,
      uint8_t *byte_code,
      size_t byte_code_size,
      bn_t &return_data_offset,
      bn_t &return_data_size,
      uint32_t static_env = 0
      ) : _arith(arith)
  {
    SHARED_MEMORY message_data_t *content;
    ONE_THREAD_PER_INSTANCE(
      content = new message_data_t;)
    _content = content;
    cgbn_store(_arith._env, &(_content->sender), sender);
    cgbn_store(_arith._env, &(_content->recipient), recipient);
    cgbn_store(_arith._env, &(_content->contract_address), contract_address);
    cgbn_store(_arith._env, &(_content->gas_limit), gas_limit);
    cgbn_store(_arith._env, &(_content->value), value);
    _content->depth = depth;
    _content->call_type = call_type;
    cgbn_store(_arith._env, &(_content->storage_address), storage_address);
    _content->data.size = data_size;
    _content->byte_code.size = byte_code_size;
    ONE_THREAD_PER_INSTANCE(
      if (data_size > 0) {
        _content->data.data = new uint8_t[data_size];
        memcpy(_content->data.data, data, sizeof(uint8_t) * data_size);
      } else {
        _content->data.data = NULL;
      }
      if (byte_code_size > 0) {
        _content->byte_code.data = new uint8_t[byte_code_size];
        memcpy(_content->byte_code.data, byte_code, sizeof(uint8_t) * byte_code_size);
      } else {
        _content->byte_code.data = NULL;
      })
    cgbn_store(_arith._env, &(_content->return_data_offset), return_data_offset);
    cgbn_store(_arith._env, &(_content->return_data_size), return_data_size);
    _content->static_env = static_env;
  }

  /**
   * The destructor.
   */
  __host__ __device__ __forceinline__ ~message_t()
  {
    ONE_THREAD_PER_INSTANCE(
        if (_content->data.size > 0) {
          delete[] _content->data.data;
          _content->data.size = 0;
          _content->data.data = NULL;
        }
        if (_content->byte_code.size > 0) {
          delete[] _content->byte_code.data;
          _content->byte_code.size = 0;
          _content->byte_code.data = NULL;
        }
        delete _content;)
    _content = NULL;
  }

  /**
   * Get the sender address.
   * @param[out] sender The sender address YP: \f$s\f$.
  */
  __host__ __device__ __forceinline__ void get_sender(
      bn_t &sender)
  {
    cgbn_load(_arith._env, sender, &(_content->sender));
  }

  /**
   * Get the recipient address.
   * @param[out] recipient The recipient address YP: \f$r\f$.
  */
  __host__ __device__ __forceinline__ void get_recipient(
      bn_t &recipient)
  {
    cgbn_load(_arith._env, recipient, &(_content->recipient));
  }

  /**
   * Get the contract address.
   * @param[out] contract_address The contract address YP: \f$c\f$.
  */
  __host__ __device__ __forceinline__ void get_contract_address(
      bn_t &contract_address)
  {
    cgbn_load(_arith._env, contract_address, &(_content->contract_address));
  }

  /**
   * Get the gas limit.
   * @param[out] gas_limit The gas limit YP: \f$g\f$.
  */
  __host__ __device__ __forceinline__ void get_gas_limit(
      bn_t &gas_limit)
  {
    cgbn_load(_arith._env, gas_limit, &(_content->gas_limit));
  }

  /**
   * Get the value.
   * @param[out] value The value YP: \f$v\f$ or \f$v^{'}\f$ for DelegateCALL.
  */
  __host__ __device__ __forceinline__ void get_value(
      bn_t &value)
  {
    cgbn_load(_arith._env, value, &(_content->value));
  }

  /**
   * Get the depth.
   * @return The depth YP: \f$e\f$.
  */
  __host__ __device__ __forceinline__ uint32_t get_depth()
  {
    return _content->depth;
  }

  /**
   * Get the call type.
   * @return The call type internal has the opcode YP: \f$w\f$.
  */
  __host__ __device__ __forceinline__ uint8_t get_call_type()
  {
    return _content->call_type;
  }

  /**
   * Get the storage address.
   * @param[out] storage_address The storage address YP: \f$a\f$.
  */
  __host__ __device__ __forceinline__ void get_storage_address(
      bn_t &storage_address)
  {
    cgbn_load(_arith._env, storage_address, &(_content->storage_address));
  }

  /**
   * Get the call/init data size.
   * @return The data size YP: \f$|d|\f$.
  */
  __host__ __device__ __forceinline__ size_t get_data_size()
  {
    return _content->data.size;
  }

  /**
   * Get the call/init data.
   * @param[in] index The index of the first byte to be returned.
   * @param[in] length The number of bytes to be returned.
   * @param[out] available_size The number of bytes available starting from index.
   * @return The pointer to the data.
  */
  __host__ __device__ __forceinline__ uint8_t *get_data(
      bn_t &index,
      bn_t &length,
      size_t &available_size)
  {
    available_size = 0;
    size_t index_s;
    int32_t overflow = _arith.size_t_from_cgbn(index_s, index);
    if (
        (overflow != 0) ||
        (index_s >= _content->data.size))
    {
      return NULL;
    }
    else
    {
      size_t length_s;
      overflow = _arith.size_t_from_cgbn(length_s, length);
      if (
          (overflow != 0) ||
          (length_s > _content->data.size - index_s))
      {
        available_size = _content->data.size - index_s;
        return _content->data.data + index_s;
      }
      else
      {
        available_size = length_s;
        return _content->data.data + index_s;
      }
    }
  }

  /**
   * Get the byte code size.
   * @return The byte code size YP: \f$|b|\f$.
  */
  __host__ __device__ __forceinline__ size_t get_code_size()
  {
    return _content->byte_code.size;
  }

  /**
   * Get the byte code.
   * @return The pointer to the byte code YP: \f$b\f$.
  */
  __host__ __device__ __forceinline__ uint8_t *get_byte_code()
  {
    return _content->byte_code.data;
  }

  /**
   * Get the code at the given index for the given length.
   * If the index is greater than the code size, it returns NULL.
   * If the length is greater than the code size - index, it returns
   * the code data from index to the end of the code and sets the
   * available size to the code size - index. Otherwise, it returns
   * the code data from index to index + length and sets the available
   * size to length.
   * @param[in] index The index of the code data
   * @param[in] length The length of the code data
   * @param[out] available_size The available size of the code data
  */
  __host__ __device__ __forceinline__ uint8_t *get_byte_code_data(
    bn_t &index,
    bn_t &length,
    size_t &available_size
  )
  {
    return _arith.get_data(
      _content->byte_code,
      index,
      length,
      available_size);
  }

  /**
   * Get the return data offset.
   * @param[out] return_data_offset The return data offset in memory.
  */
  __host__ __device__ __forceinline__ void get_return_data_offset(
      bn_t &return_data_offset)
  {
    cgbn_load(_arith._env, return_data_offset, &(_content->return_data_offset));
  }

  /**
   * Get the return data size.
   * @param[out] return_data_size The return data size in memory.
  */
  __host__ __device__ __forceinline__ void get_return_data_size(
      bn_t &return_data_size)
  {
    cgbn_load(_arith._env, return_data_size, &(_content->return_data_size));
  }

  /**
   * Get the static flag.
   * @return The static flag (STATICCALL) YP: \f$w\f$.
  */
  __host__ __device__ __forceinline__ uint32_t get_static_env()
  {
    return _content->static_env;
  }

  /**
   * Set the gas limit.
   * @param[in] gas_limit The gas limit YP: \f$g\f$.
  */
  __host__ __device__ __forceinline__ void set_gas_limit(
      bn_t &gas_limit)
  {
    cgbn_store(_arith._env, &(_content->gas_limit), gas_limit);
  }

  /**
   * Set the call data.
   * @param[in] data The data YP: \f$d\f$.
   * @param[in] data_size The data size YP: \f$|d|\f$.
  */
  __host__ __device__ __forceinline__ void set_data(
      uint8_t *data,
      size_t data_size)
  {
    ONE_THREAD_PER_INSTANCE(
      if (_content->data.size > 0) {
        delete[] _content->data.data;
        _content->data.size = 0;
        _content->data.data = NULL;
      }
      if (data_size > 0) {
        _content->data.data = new uint8_t[data_size];
        memcpy(_content->data.data, data, sizeof(uint8_t) * data_size);
      } else {
        _content->data.data = NULL;
      }
      _content->data.size = data_size;)
  }

  /**
   * Set the byte code.
   * @param[in] byte_code The byte code YP: \f$b\f$.
   * @param[in] byte_code_size The byte code size YP: \f$|b|\f$.
  */
  __host__ __device__ __forceinline__ void set_byte_code(
      uint8_t *byte_code,
      size_t byte_code_size)
  {
    ONE_THREAD_PER_INSTANCE(
      if (_content->byte_code.size > 0) {
        delete[] _content->byte_code.data;
        _content->byte_code.size = 0;
        _content->byte_code.data = NULL;
      }
      if (byte_code_size > 0) {
        _content->byte_code.data = new uint8_t[byte_code_size];
        memcpy(_content->byte_code.data, byte_code, sizeof(uint8_t) * byte_code_size);
      } else {
        _content->byte_code.data = NULL;
      })
      _content->byte_code.size = byte_code_size;
  }

  /**
   * Set the return data offset.
   * @param[in] return_data_offset The return data offset in memory.
  */
  __host__ __device__ __forceinline__ void set_return_data_offset(
      bn_t &return_data_offset)
  {
    cgbn_store(_arith._env, &(_content->return_data_offset), return_data_offset);
  }

  /**
   * Set the return data size.
   * @param[in] return_data_size The return data size in memory.
  */
  __host__ __device__ __forceinline__ void set_return_data_size(
      bn_t &return_data_size)
  {
    cgbn_store(_arith._env, &(_content->return_data_size), return_data_size);
  }

  __host__ __device__ __forceinline__ static void get_create_contract_address(
    arith_t &arith,
    bn_t &contract_address,
    bn_t &sender_address,
    bn_t &sender_nonce,
    keccak_t &keccak
  )
  {
    SHARED_MEMORY uint8_t sender_address_bytes[arith_t::BYTES];
    arith.memory_from_cgbn(
        &(sender_address_bytes[0]),
        sender_address);
    SHARED_MEMORY uint8_t sender_nonce_bytes[arith_t::BYTES];
    arith.memory_from_cgbn(
        &(sender_nonce_bytes[0]),
        sender_nonce);
    
    uint8_t nonce_bytes;
    for (nonce_bytes = arith_t::BYTES; nonce_bytes > 0; nonce_bytes--)
    {
      if (sender_nonce_bytes[arith_t::BYTES - nonce_bytes] != 0)
      {
        break;
      }
    }

    // TODO: this might work only for arith_t::BYTES == 32

    SHARED_MEMORY uint8_t rlp_list[1 + 1 + arith_t::ADDRESS_BYTES + 1 + arith_t::BYTES];

    // the adress has only 20 bytes
    rlp_list[1] = 0x80 + arith_t::ADDRESS_BYTES;
    for (uint8_t idx = 0; idx < arith_t::ADDRESS_BYTES; idx++)
    {
      rlp_list[2 + idx] = sender_address_bytes[arith_t::BYTES - arith_t::ADDRESS_BYTES + idx];
    }

    uint8_t rlp_list_length;
    // 21 is from the address the 20 bytes is the length of the address
    // and the 1 byte is the 0x80 + length of the address (20)
    if (cgbn_compare_ui32(arith._env, sender_nonce, 128) < 0)
    {
      rlp_list_length = 1 + arith_t::ADDRESS_BYTES + 1;
      if (cgbn_compare_ui32(arith._env, sender_nonce, 0)  == 0)
      {
        rlp_list[2 + arith_t::ADDRESS_BYTES] = 0x80; // special case for nonce 0
      }
      else
      {
        rlp_list[2 + arith_t::ADDRESS_BYTES] = sender_nonce_bytes[arith_t::BYTES - 1];
      }
    }
    else
    {
      // 1 byte for the length of the nonce
      // 0x80 + length of the nonce
      rlp_list_length = 21 + 1 + nonce_bytes;
      rlp_list[2 + arith_t::ADDRESS_BYTES] = 0x80 + nonce_bytes;
      for (uint8_t idx = 0; idx < nonce_bytes; idx++)
      {
        rlp_list[2 + arith_t::ADDRESS_BYTES + 1 + idx] = sender_nonce_bytes[arith_t::BYTES - nonce_bytes + idx];
      }
    }
    rlp_list[0] = 0xc0 + rlp_list_length;

    /*
    ONE_THREAD_PER_INSTANCE(
    print_bytes(&(rlp_list[0]), rlp_list_length + 1);
    )
    */

    SHARED_MEMORY uint8_t address_bytes[HASH_BYTES];
    keccak.sha3(
        &(rlp_list[0]),
        rlp_list_length + 1,
        &(address_bytes[0]),
        HASH_BYTES);
    for (uint8_t idx = 0; idx < arith_t::BYTES - arith_t::ADDRESS_BYTES; idx++)
    {
      address_bytes[idx] = 0;
    }
    arith.cgbn_from_memory(
        contract_address,
        &(address_bytes[0]));
  }

  /**
   * Get the CREATE2 contract address.
   * @param[in] arith The arithmetical environment.
   * @param[out] contract_address The contract address YP: \f$a\f$.
   * @param[in] sender_address The sender address YP: \f$s\f$.
   * @param[in] salt The salt YP: \f$n\f$.
   * @param[in] byte_code The byte code YP: \f$b\f$.
   * @param[in] keccak The keccak class.
  */
  __host__ __device__ __forceinline__ static void get_create2_contract_address(
    arith_t &arith,
    bn_t &contract_address,
    bn_t &sender_address,
    bn_t &salt,
    data_content_t &byte_code,
    keccak_t &keccak
  )
  {
    SHARED_MEMORY uint8_t sender_address_bytes[arith_t::BYTES];
    arith.memory_from_cgbn(
        &(sender_address_bytes[0]),
        sender_address);
    SHARED_MEMORY uint8_t salt_bytes[arith_t::BYTES];
    arith.memory_from_cgbn(
        &(salt_bytes[0]),
        salt);

    size_t total_bytes = 1 + arith_t::ADDRESS_BYTES + arith_t::BYTES + HASH_BYTES;

    SHARED_MEMORY uint8_t hash_code[HASH_BYTES];
    keccak.sha3(
        byte_code.data,
        byte_code.size,
        &(hash_code[0]),
        HASH_BYTES);
    
    SHARED_MEMORY uint8_t input_data[1 + arith_t::ADDRESS_BYTES + arith_t::BYTES + HASH_BYTES];
    input_data[0] = 0xff;
    ONE_THREAD_PER_INSTANCE(
    memcpy(
        &(input_data[1]),
        &(sender_address_bytes[arith_t::BYTES - arith_t::ADDRESS_BYTES]),
        arith_t::ADDRESS_BYTES);
    memcpy(
        &(input_data[1 + arith_t::ADDRESS_BYTES]),
        &(salt_bytes[0]),
        arith_t::BYTES);
    memcpy(
        &(input_data[1 + arith_t::ADDRESS_BYTES + arith_t::BYTES]),
        &(hash_code[0]),
        HASH_BYTES);
    )
    SHARED_MEMORY uint8_t address_bytes[HASH_BYTES];
    keccak.sha3(
        input_data,
        total_bytes,
        &(address_bytes[0]),
        HASH_BYTES);
    for (uint8_t idx = 0; idx < arith_t::BYTES - arith_t::ADDRESS_BYTES; idx++)
    {
      address_bytes[idx] = 0;
    }
    arith.cgbn_from_memory(
        contract_address,
        &(address_bytes[0]));
    
  }

  /**
   * Print the message.
  */
  __host__ __device__ __forceinline__ void print()
  {
    printf("SENDER: ");
    _arith.print_cgbn_memory(_content->sender);
    printf("RECIPIENT: ");
    _arith.print_cgbn_memory(_content->recipient);
    printf("CONTRACT_ADDRESS: ");
    _arith.print_cgbn_memory(_content->contract_address);
    printf("GAS_LIMIT: ");
    _arith.print_cgbn_memory(_content->gas_limit);
    printf("VALUE: ");
    _arith.print_cgbn_memory(_content->value);
    printf("DEPTH: %d\n", _content->depth);
    printf("CALL_TYPE: %d\n", _content->call_type);
    printf("STORAGE_ADDRESS: ");
    _arith.print_cgbn_memory(_content->storage_address);
    printf("DATA_SIZE: %lu\n", _content->data.size);
    if (_content->data.size > 0)
    {
      printf("DATA: ");
      print_bytes(_content->data.data, _content->data.size);
      printf("\n");
    }
    printf("BYTE_CODE_SIZE: %lu\n", _content->byte_code.size);
    if (_content->byte_code.size > 0)
    {
      printf("BYTE_CODE: ");
      print_bytes(_content->byte_code.data, _content->byte_code.size);
      printf("\n");
    }
    printf("RETURN_DATA_OFFSET: ");
    _arith.print_cgbn_memory(_content->return_data_offset);
    printf("RETURN_DATA_SIZE: ");
    _arith.print_cgbn_memory(_content->return_data_size);
    printf("STATIC_ENV: %d\n", _content->static_env);
  }
};

/**
 * The transaction type
 * YP: \f$T\f$
 */
class transaction_t
{
public:
  /**
   * The arithmetical environment used by the arbitrary length
   * integer library.
   */
  typedef arith_env_t<evm_params> arith_t;
  /**
   * The arbitrary length integer type.
   */
  typedef typename arith_t::bn_t bn_t;
  /**
   * The arbitrary length integer type used for the storage.
   * It is defined as the EVM word type.
   */
  typedef cgbn_mem_t<evm_params::BITS> evm_word_t;

  /**
   * The account class.
  */
  typedef world_state_t::account_t account_t;
  /**
   * THe keccak class
  */
  typedef keccak::keccak_t keccak_t;
  /**
   * The maximum number of transactions per test.
   */
  static const size_t MAX_TRANSACTION_COUNT = 2000;

  /**
   * The access list account.
   * YP: \f$E_{a}\f$
   */
  typedef struct
  {
    evm_word_t address;       /**< The address YP: \f$a\f$ */
    uint32_t no_storage_keys; /**< The number of storage keys YP: \f$|E_{s}|\f$ */
    evm_word_t *storage_keys; /**< The storage keys YP: \f$E_{s}\f$ */
  } access_list_account_t;

  /**
   * The access list.
   * YP: \f$T_{A}\f$
   */
  typedef struct
  {
    uint32_t no_accounts;            /**< The number of accounts YP: \f$|T_{A}|\f$ */
    access_list_account_t *accounts; /**< The accounts YP: \f$T_{A}\f$ */
  } access_list_t;

  /**
   * The transaction data.
   * It does not contains:
   * - chain id (YP: \f$T_{c}\f$)
   * - yParity (YP: \f$T_{y}\f$)
   * - w (YP: \f$T_{w}\f$)
   */
  typedef struct alignas(32)
  {
    uint8_t type;                        /**< The transaction type EIP-2718 (YP: YP: \f$\T_{x}\f$) */
    evm_word_t nonce;                    /**< The nonce YP: \f$T_{n}\f$ */
    evm_word_t gas_limit;                /**< The gas limit YP: \f$T_{g}\f$ */
    evm_word_t to;                       /**< The recipient address YP: \f$T_{t}\f$ */
    evm_word_t value;                    /**< The value YP: \f$T_{v}\f$ */
    evm_word_t sender;                   /**< The sender address YP: \f$T_{s}\f$ or \f$T_{r}\f$*/
    access_list_t access_list;           /**< The state YP: \f$T_{A}\f$ entry are \f$E=(E_{a}, E_{s})\f$*/
    evm_word_t max_fee_per_gas;          /**< The max fee per gas YP: \f$T_{m}\f$ */
    evm_word_t max_priority_fee_per_gas; /**< The max priority fee per gas YP: \f$T_{f}\f$ */
    evm_word_t gas_price;                /**< The gas proce YP: \f$T_{p}\f$ */
    data_content_t data_init;            /**< The init or data YP:\f$T_{i}\f$ or \f$T_{d}\f$ */
  } transaction_data_t;

  transaction_data_t *_content; /**< The transaction content */
  arith_t _arith; /**< The arithmetic  */

  /**
   * The constructor. Takes the transaction content.
   * @param[in] arith The arithmetical environment.
   * @param[in] content The transaction content.
  */
  __host__ __device__ __forceinline__ transaction_t(
      arith_t arith,
      transaction_data_t *content) : _arith(arith),
                                     _content(content)
  {
  }

  /**
   * The destructor.
  */
  __host__ __device__ __forceinline__ ~transaction_t()
  {
    _content = NULL;
  }

  /**
   * Get the transation nonce.
   * @param[out] nonce The nonce YP: \f$T_{n}\f$.
  */
  __host__ __device__ __forceinline__ void get_nonce(
      bn_t &nonce)
  {
    cgbn_load(_arith._env, nonce, &(_content->nonce));
  }

  /**
   * Get the transation gas limit.
   * @param[out] gas_limit The gas limit YP: \f$T_{g}\f$.
  */
  __host__ __device__ __forceinline__ void get_gas_limit(
      bn_t &gas_limit)
  {
    cgbn_load(_arith._env, gas_limit, &(_content->gas_limit));
  }

  /**
   * Get the transaction receiver.
   * @param[out] to The receiver address YP: \f$T_{t}\f$.
  */
  __host__ __device__ __forceinline__ void get_to(
      bn_t &to)
  {
    cgbn_load(_arith._env, to, &(_content->to));
  }

  /**
   * Get the transaction value.
   * @param[out] value The value YP: \f$T_{v}\f$.
  */
  __host__ __device__ __forceinline__ void get_value(
      bn_t &value)
  {
    cgbn_load(_arith._env, value, &(_content->value));
  }

  /**
   * Get the transaction sender.
   * @param[out] sender The sender address YP: \f$T_{s}\f$ or \f$T_{r}\f$.
  */
  __host__ __device__ __forceinline__ void get_sender(
      bn_t &sender)
  {
    cgbn_load(_arith._env, sender, &(_content->sender));
  }

  /**
   * Get the maximum fee per gas.
   * @param[out] max_fee_per_gas The maximum fee per gas YP: \f$T_{m}\f$.
  */
  __host__ __device__ __forceinline__ void get_max_fee_per_gas(
      bn_t &max_fee_per_gas)
  {
    cgbn_load(_arith._env, max_fee_per_gas, &(_content->max_fee_per_gas));
  }

  /**
   * Get the maximum priority fee per gas.
   * @param[out] max_priority_fee_per_gas The maximum priority fee per gas YP: \f$T_{f}\f$.
  */
  __host__ __device__ __forceinline__ void get_max_priority_fee_per_gas(
      bn_t &max_priority_fee_per_gas)
  {
    cgbn_load(_arith._env, max_priority_fee_per_gas, &(_content->max_priority_fee_per_gas));
  }

  /**
   * Get the transaction gas price.
   * @param[out] gas_price The gas price YP: \f$T_{p}\f$.
  */
  __host__ __device__ __forceinline__ void get_gas_price(
      bn_t &gas_price)
  {
    cgbn_load(_arith._env, gas_price, &(_content->gas_price));
  }

  /**
   * Get the intrisinc gas value for the transacation.
   * Add the accounts and storage keys from access list
   * to the warm accessed accounts and storage keys.
   * @param[out] intrisinc_gas The intrisinc gas value YP: \f$g_{0}\f$.
   * @param[out] accessed_state The accessed state.
  */
  __host__ __device__ __forceinline__ void get_intrisinc_gas(
      bn_t &intrisinc_gas,
      accessed_state_t &accessed_state)
  {
    // set the initial cost to the transaction cost
    cgbn_set_ui32(_arith._env, intrisinc_gas, GAS_TRANSACTION);
    // see if a contract is being created
    bn_t to;
    get_to(to);
    if (cgbn_compare_ui32(_arith._env, to, 0) == 0)
    {
        // contract creation
        cgbn_add_ui32(_arith._env, intrisinc_gas, intrisinc_gas, GAS_TX_CREATE);
    }
    // go through call data cost
    for (size_t idx=0; idx < _content->data_init.size; idx++)
    {
        if (_content->data_init.data[idx] == 0)
        {
            cgbn_add_ui32(_arith._env, intrisinc_gas, intrisinc_gas, GAS_TX_DATA_ZERO);
        }
        else
        {
            cgbn_add_ui32(_arith._env, intrisinc_gas, intrisinc_gas, GAS_TX_DATA_NONZERO);
        }
    }
    // if has the access list add the cost and
    // add the accounts and storage keys to the warm accessed accounts and storage keys
    bn_t address;
    bn_t key;
    bn_t value;
    if (_content->type > 0)
    {
        for (size_t idx=0; idx < _content->access_list.no_accounts; idx++)
        {
          cgbn_add_ui32(
              _arith._env,
              intrisinc_gas,
              intrisinc_gas,
              GAS_ACCESS_LIST_ADDRESS);
          cgbn_load(
              _arith._env,
              address,
              &(_content->access_list.accounts[idx].address));
          // get the account in warm accessed accounts
          accessed_state.get_account(address, READ_NONE);
          for (
              size_t jdx=0;
              jdx < _content->access_list.accounts[idx].no_storage_keys;
              jdx++)
          {
            cgbn_add_ui32(
                _arith._env,
                intrisinc_gas,
                intrisinc_gas,
                GAS_ACCESS_LIST_STORAGE);
            cgbn_load(
                _arith._env,
                key,
                &(_content->access_list.accounts[idx].storage_keys[jdx]));
            // get the storage in warm accessed storage keys
            accessed_state.get_value(address, key, value);
          }
        }
    }

    // if create transaction add the cost
    // EIP-3869
    if (cgbn_compare_ui32(_arith._env, to, 0) == 0)
    {
      // compute the dynamic gas cost for initcode
      bn_t initcode_gas;
      bn_t initcode_length;
      _arith.cgbn_from_size_t(initcode_length, _content->data_init.size);
      // evm word INITCODE_WORD_COST * ceil(len(initcode) / 32)
      cgbn_add_ui32(_arith._env, initcode_gas, initcode_length, 31);
      cgbn_div_ui32(_arith._env, initcode_gas, initcode_gas, 32);
      cgbn_mul_ui32(_arith._env, initcode_gas, initcode_gas, GAS_INITCODE_WORD_COST);
      cgbn_add(_arith._env, intrisinc_gas, intrisinc_gas, initcode_gas);
    }
  }

  /**
   * Get the transaction fees. The fees are computed
   * based on the transaction information and block
   * base fee.
   * @param[out] gas_value The WEI gas value YP: \f$p \dot T_{g}\f$
   * @param[out] gas_limit The gas limit YP: \f$T_{g}\f$
   * @param[out] gas_price The gas price YP: \f$p\f$
   * @param[out] gas_priority_fee The gas priority fee YP: \f$f\f$
   * @param[out] up_front_cost The up front cost YP: \f$v_{0}\f$
   * @param[out] m The max gas fee YP: \f$m\f$
   * @param[in] block_base_fee The block base fee YP: \f$H_{f}\f$
   * @return The error code.
  */
  __host__ __device__ __forceinline__ uint32_t get_transaction_fees(
    bn_t &gas_value,
    bn_t &gas_limit,
    bn_t &gas_price,
    bn_t &gas_priority_fee,
    bn_t &up_front_cost,
    bn_t &m,
    bn_t &block_base_fee
  )
  {
    bn_t max_priority_fee_per_gas; // YP: \f$T_{f}\f$
    bn_t max_fee_per_gas; // YP: \f$T_{m}\f$
    bn_t value; // YP: \f$T_{v}\f$
    get_max_priority_fee_per_gas(max_priority_fee_per_gas);
    get_max_fee_per_gas(max_fee_per_gas);
    get_value(value);
    get_gas_limit(gas_limit); // YP: \f$T_{g}\f$
    if (
        (_content->type == 0) ||
        (_content->type == 1)
    )
    {
        // \f$p = T_{p}\f$
        get_gas_price(gas_price);
        // \f$f = T_{p} - H_{f}\f$
        cgbn_sub(_arith._env, gas_priority_fee, gas_price, block_base_fee);
        // \f$v_{0} = T_{p} * T_{g} + T_{v}\f$
        cgbn_mul(_arith._env, up_front_cost, gas_price, gas_limit);
        cgbn_add(_arith._env, up_front_cost, up_front_cost, value);
        // \f$m = T_{p}\f$
        cgbn_set(_arith._env, m, gas_price);
    } else if (_content->type == 2)
    {
        // \f$T_{m} - H_{f}\f$
        cgbn_sub(_arith._env, gas_priority_fee, max_fee_per_gas, block_base_fee);
        // \f$f=min(T_{f}, T_{m} - H_{f})\f$
        if (cgbn_compare(_arith._env, gas_priority_fee, max_priority_fee_per_gas) > 0)
        {
            cgbn_set(_arith._env, gas_priority_fee, max_priority_fee_per_gas);
        }
        // \f$p = f + H_{f}\f$
        cgbn_add(_arith._env, gas_price, gas_priority_fee, block_base_fee);
        // \f$v_{0} = T_{m} * T_{g} + T_{v}\f$
        cgbn_mul(_arith._env, up_front_cost, max_fee_per_gas, gas_limit);
        cgbn_add(_arith._env, up_front_cost, up_front_cost, value);
        // \f$m = T_{m}\f$
        cgbn_set(_arith._env, m, max_fee_per_gas);
    } else {
        return ERROR_TRANSACTION_TYPE;
    }
    // gas value \f$= T_{g} \dot p\f$
    cgbn_mul(_arith._env, gas_value, gas_limit, gas_price);
    return ERR_NONE;
  }

  __host__ __device__ __forceinline__ void get_computed_gas_price(
    bn_t &gas_price,
    bn_t &block_base_fee,
    uint32_t &error_code
  )
  {
    bn_t max_priority_fee_per_gas; // YP: \f$T_{f}\f$
    bn_t max_fee_per_gas; // YP: \f$T_{m}\f$
    bn_t gas_priority_fee; // YP: \f$f\f$
    get_max_priority_fee_per_gas(max_priority_fee_per_gas);
    get_max_fee_per_gas(max_fee_per_gas);
    if (
        (_content->type == 0) ||
        (_content->type == 1)
    )
    {
        // \f$p = T_{p}\f$
        get_gas_price(gas_price);
    } else if (_content->type == 2)
    {
        // \f$T_{m} - H_{f}\f$
        cgbn_sub(_arith._env, gas_priority_fee, max_fee_per_gas, block_base_fee);
        // \f$f=min(T_{f}, T_{m} - H_{f})\f$
        if (cgbn_compare(_arith._env, gas_priority_fee, max_priority_fee_per_gas) > 0)
        {
            cgbn_set(_arith._env, gas_priority_fee, max_priority_fee_per_gas);
        }
        // \f$p = f + H_{f}\f$
        cgbn_add(_arith._env, gas_price, gas_priority_fee, block_base_fee);
    } else {
        error_code = ERROR_TRANSACTION_TYPE;
    }
  }

  /**
   * Validate the transaction. The validation is done
   * based on the transaction information and block
   * base fee. It gives the gas price, the gas used for
   * the transaction, the gas priority fee, the error code
   * (if the transaction is invalid). If the transaction is
   * valid it updates the touch state by subtracting the
   * gas value from the sender balance and incrementing
   * the sender nonce.
   * @param[out] touch_state The touch state.
   * @param[out] gas_used The gas used YP: \f$g_{0}\f$
   * @param[out] gas_price The gas price YP: \f$p\f$
   * @param[out] gas_priority_fee The gas priority fee YP: \f$f\f$
   * @param[out] error_code The error code.
   * @param[in] block_base_fee The block base fee YP: \f$H_{f}\f$
   * @param[in] block_gas_limit The block gas limit YP: \f$H_{l}\f$
  */
  __host__ __device__ void validate_transaction(
    touch_state_t &touch_state,
    bn_t &gas_used,
    bn_t &gas_price,
    bn_t &gas_priority_fee,
    uint32_t &error_code,
    bn_t &block_base_fee,
    bn_t &block_gas_limit
  )
  {
    bn_t intrisinc_gas; /**< YP: \f$g_{0}\f$*/
    // get the intrisinc gas and update the accessed state
    // with the access list if present
    get_intrisinc_gas(intrisinc_gas, *touch_state._accessed_state);
    bn_t up_front_cost; // YP: \f$v_{0}\f$
    bn_t m; // YP: \f$m\f$
    bn_t gas_value; // YP: \f$p \dot T_{g}\f$
    bn_t gas_limit; // YP: \f$T_{g}\f$
    error_code = get_transaction_fees(
        gas_value,
        gas_limit,
        gas_price,
        gas_priority_fee,
        up_front_cost,
        m,
        block_base_fee
    );
    
    bn_t sender_address;
    account_t *sender_account;
    get_sender(sender_address);
    error_code = ERR_NONE;
    // get the world state account
    sender_account = touch_state._accessed_state->_world_state->get_account(
      sender_address,
      error_code
    );
    bn_t sender_balance;
    cgbn_load(_arith._env, sender_balance, &(sender_account->balance));
    bn_t sender_nonce;
    cgbn_load(_arith._env, sender_nonce, &(sender_account->nonce));
    bn_t transaction_nonce;
    get_nonce(transaction_nonce);
    bn_t max_fee_per_gas;
    get_max_fee_per_gas(max_fee_per_gas);
    bn_t max_priority_fee_per_gas;
    get_max_priority_fee_per_gas(max_priority_fee_per_gas);

    // sender is an empty account YP: \f$\sigma(T_{s}) \neq \varnothing\f$
    error_code = (error_code != ERR_NONE) ?
      error_code :
      ((error_code == ERR_STATE_INVALID_ADDRESS) ?
      ERROR_TRANSACTION_SENDER_EMPTY : ERR_NONE);
    // sender is a contract YP: \f$\sigma(T_{s})_{c} \eq KEC(())\f$
    error_code = (error_code != ERR_NONE) ?
      error_code :
      ( (sender_account->code_size > 0) ?
        ERROR_TRANSACTION_SENDER_CODE : ERR_NONE);
    // nonce are different YP: \f$T_{n} \eq \sigma(T_{s})_{n}\f$
    error_code = (error_code != ERR_NONE) ?
      error_code :
      ((cgbn_compare(_arith._env, transaction_nonce, sender_nonce) != 0) ?
        ERROR_TRANSACTION_NONCE : ERR_NONE);
    // sent gas is less than intrinisec gas YP: \f$T_{g} \geq g_{0}\f$
    error_code = (error_code != ERR_NONE) ?
      error_code :
      ((cgbn_compare(_arith._env, gas_limit, intrisinc_gas) < 0) ?
        ERROR_TRANSACTION_GAS : ERR_NONE);
    // balance is less than up front cost YP: \f$\sigma(T_{s})_{b} \geq v_{0}\f$
    error_code = (error_code != ERR_NONE) ?
      error_code :
      ((cgbn_compare(_arith._env, sender_balance, up_front_cost) < 0) ?
        ERROR_TRANSACTION_SENDER_BALANCE : ERR_NONE);
    // gas fee is less than than block base fee YP: \f$m \geq H_{f}\f$
    error_code = (error_code != ERR_NONE) ?
      error_code :
      ((cgbn_compare(_arith._env, m, block_base_fee) < 0) ?
        ERROR_TRANSACTION_GAS_PRICE : ERR_NONE);
    // Max priority fee per gas is higher than max fee per gas YP: \f$T_{m} \geq T_{f}\f$
    error_code = (error_code != ERR_NONE) ?
      error_code :
      (((_content->type == 2) &&
      (cgbn_compare(_arith._env, max_fee_per_gas, max_priority_fee_per_gas) < 0)) ?
        ERROR_TRANSACTION_GAS_PRIORITY : ERR_NONE);
    
    // the other verification is about the block gas limit
    // YP: \f$T_{g} \leq H_{l}\f$ ... different because it takes in account
    // previous transactions
    error_code = (error_code != ERR_NONE) ?
      error_code :
      ((cgbn_compare(_arith._env, gas_limit, block_gas_limit) > 0) ?
        ERROR_TRANSACTION_BLOCK_GAS_LIMIT : ERR_NONE);
    
    // if transaction is valid update the touch state
    if (error_code == ERR_NONE)
    {
      // \f$\simga(T_{s})_{b} = \simga(T_{s})_{b} - (p \dot T_{g})\f$
      cgbn_sub(_arith._env, sender_balance, sender_balance, gas_value);
      touch_state.set_account_balance(sender_address, sender_balance);
      // \f$\simga(T_{s})_{n} = \simga(T_{s})_{n} + 1\f$
      cgbn_add_ui32(_arith._env, sender_nonce, sender_nonce, 1);
      touch_state.set_account_nonce(sender_address, sender_nonce);
      // set the gas used to the intrisinc gas
      cgbn_set(_arith._env, gas_used, intrisinc_gas);
    }
  }

  /**
   * Get the message call from the transaction.
   * @return The message call.
  */
  __host__ __device__ message_t *get_message_call(
      accessed_state_t &accessed_state,
      keccak_t &keccak
  )
  {
    bn_t sender, to, gas_limit, value;
    get_sender(sender);
    get_to(to);
    get_gas_limit(gas_limit);
    get_value(value);
    uint32_t depth = 0;
    uint8_t call_type = OP_CALL;
    uint8_t *byte_code = NULL;
    size_t byte_code_size = 0;
    if (cgbn_compare_ui32(_arith._env, to, 0) == 0)
    {
      // is CREATE type not CREATE2 because no salt
      call_type = OP_CREATE;
      byte_code = _content->data_init.data;
      byte_code_size = _content->data_init.size;
      // TODO: code size does not execede the maximum allowed
      account_t *account = accessed_state.get_account(sender, READ_NONCE);
      bn_t sender_nonce;
      // nonce is -1 in YP but here is before validating the transaction
      // and increasing the nonce
      cgbn_load(_arith._env, sender_nonce, &(account->nonce));
      message_t::get_create_contract_address(
          _arith,
          to,
          sender,
          sender_nonce,
          keccak);
    }
    else
    {
      account_t *account = accessed_state.get_account(to, READ_CODE);
      byte_code = account->bytecode;
      byte_code_size = account->code_size;
    }
    uint32_t static_env = 0;
    bn_t return_data_offset;
    cgbn_set_ui32(_arith._env, return_data_offset, 0);
    bn_t return_data_size;
    cgbn_set_ui32(_arith._env, return_data_size, 0);
    return new message_t(
        _arith,
        sender,
        to,
        to,
        gas_limit,
        value,
        depth,
        call_type,
        to,
        _content->data_init.data,
        _content->data_init.size,
        byte_code,
        byte_code_size,
        return_data_offset,
        return_data_size,
        static_env);
  }


  /**
   * Print the transaction data structure.
   * @param[in] arith The arithmetical environment.
   * @param[in] transaction_data The transaction data.
  */
  __host__ __device__ static void print_transaction_data_t(
    arith_t &arith,
    transaction_data_t &transaction_data
  )
  {
    printf("TYPE: %hhu\nNONCE: ", transaction_data.type);
    arith.print_cgbn_memory(transaction_data.nonce);
    printf("GAS_LIMIT: ");
    arith.print_cgbn_memory(transaction_data.gas_limit);
    printf("TO: ");
    arith.print_cgbn_memory(transaction_data.to);
    printf("VALUE: ");
    arith.print_cgbn_memory(transaction_data.value);
    printf("SENDER: ");
    arith.print_cgbn_memory(transaction_data.sender);
    if (transaction_data.type >= 1)
    {
      printf("ACCESS_LIST: ");
      for (size_t idx = 0; idx < transaction_data.access_list.no_accounts; idx++)
      {
        printf("ADDRESS: ");
        arith.print_cgbn_memory(transaction_data.access_list.accounts[idx].address);
        printf("NO_STORAGE_KEYS: %d", transaction_data.access_list.accounts[idx].no_storage_keys);
        for (size_t jdx = 0; jdx < transaction_data.access_list.accounts[idx].no_storage_keys; jdx++)
        {
          printf("STORAGE_KEY[%lu]: ", jdx);
          arith.print_cgbn_memory(transaction_data.access_list.accounts[idx].storage_keys[jdx]);
        }
      }
    }
    if (transaction_data.type == 2)
    {
      printf("MAX_FEE_PER_GAS: ");
      arith.print_cgbn_memory(transaction_data.max_fee_per_gas);
      printf("MAX_PRIORITY_FEE_PER_GAS: ");
      arith.print_cgbn_memory(transaction_data.max_priority_fee_per_gas);
    }
    else
    {
      printf("GAS_PRICE: ");
      arith.print_cgbn_memory(transaction_data.gas_price);
    }
    printf("DATA_INIT: ");
    print_data_content_t(transaction_data.data_init);
  }

  /**
   * Print the transaction information.
  */
  __host__ __device__ void print()
  {
    print_transaction_data_t(_arith, *_content);
  }

  /**
   * Get the transaction in json format.
   * @return The transaction in json format.
  */
  __host__ cJSON *json()
  {
    cJSON *transaction_json = cJSON_CreateObject();
    char *hex_string_ptr = new char[arith_t::BYTES * 2 + 3];
    char *bytes_string = NULL;

    // set the type
    cJSON_AddNumberToObject(transaction_json, "type", _content->type);

    // set the nonce
    _arith.hex_string_from_cgbn_memory(hex_string_ptr, _content->nonce);
    cJSON_AddStringToObject(transaction_json, "nonce", hex_string_ptr);

    // set the gas limit
    _arith.hex_string_from_cgbn_memory(hex_string_ptr, _content->gas_limit);
    cJSON_AddStringToObject(transaction_json, "gasLimit", hex_string_ptr);

    // set the to
    _arith.hex_string_from_cgbn_memory(hex_string_ptr, _content->to, 5);
    cJSON_AddStringToObject(transaction_json, "to", hex_string_ptr);

    // set the value
    _arith.hex_string_from_cgbn_memory(hex_string_ptr, _content->value);
    cJSON_AddStringToObject(transaction_json, "value", hex_string_ptr);

    // set the sender
    _arith.hex_string_from_cgbn_memory(hex_string_ptr, _content->sender, 5);
    cJSON_AddStringToObject(transaction_json, "sender", hex_string_ptr);
    // TODO: delete this from revmi comparator
    cJSON_AddStringToObject(transaction_json, "origin", hex_string_ptr);

    // set the access list
    if (_content->type >= 1)
    {
      cJSON *access_list_json = cJSON_CreateArray();
      cJSON_AddItemToObject(transaction_json, "accessList", access_list_json);
      for (size_t idx = 0; idx < _content->access_list.no_accounts; idx++)
      {
        cJSON *account_json = cJSON_CreateObject();
        cJSON_AddItemToArray(access_list_json, account_json);
        _arith.hex_string_from_cgbn_memory(
            hex_string_ptr,
            _content->access_list.accounts[idx].address,
            5);
        cJSON_AddStringToObject(account_json, "address", hex_string_ptr);
        cJSON *storage_keys_json = cJSON_CreateArray();
        cJSON_AddItemToObject(account_json, "storageKeys", storage_keys_json);
        for (size_t jdx = 0; jdx < _content->access_list.accounts[idx].no_storage_keys; jdx++)
        {
          _arith.hex_string_from_cgbn_memory(
              hex_string_ptr,
              _content->access_list.accounts[idx].storage_keys[jdx]);
          cJSON_AddItemToArray(storage_keys_json, cJSON_CreateString(hex_string_ptr));
        }
      }
    }

    // set the gas price
    if (_content->type == 2)
    {
      _arith.hex_string_from_cgbn_memory(hex_string_ptr, _content->max_fee_per_gas);
      cJSON_AddStringToObject(transaction_json, "maxFeePerGas", hex_string_ptr);

      // set the max priority fee per gas
      _arith.hex_string_from_cgbn_memory(hex_string_ptr, _content->max_priority_fee_per_gas);
      cJSON_AddStringToObject(transaction_json, "maxPriorityFeePerGas", hex_string_ptr);
    }
    else
    {
      // set the gas price
      _arith.hex_string_from_cgbn_memory(hex_string_ptr, _content->gas_price);
      cJSON_AddStringToObject(transaction_json, "gasPrice", hex_string_ptr);
    }

    // set the data init
    if (_content->data_init.size > 0)
    {
      bytes_string = hex_from_data_content(_content->data_init);
      cJSON_AddStringToObject(transaction_json, "data", bytes_string);
      delete[] bytes_string;
      bytes_string = NULL;
    }
    else
    {
      cJSON_AddStringToObject(transaction_json, "data", "0x");
    }

    delete[] hex_string_ptr;
    hex_string_ptr = NULL;
    return transaction_json;
  }

  /**
   * Get the number of transactions from a test in json format.
   * @param[in] test The json format of the test.
   * @return The number of transactions.
  */
  __host__ static size_t get_no_transaction(
      const cJSON *test)
  {
    const cJSON *transaction_json = cJSON_GetObjectItemCaseSensitive(test, "transaction");
    const cJSON *data_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "data");
    size_t data_counts = cJSON_GetArraySize(data_json);
    const cJSON *gas_limit_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "gasLimit");
    size_t gas_limit_counts = cJSON_GetArraySize(gas_limit_json);
    const cJSON *value_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "value");
    size_t value_counts = cJSON_GetArraySize(value_json);
    return data_counts * gas_limit_counts * value_counts;
  }

  /**
   * Get the transactions from a test in json format.
   * The number of transaction is limited by MAX_TRANSACTION_COUNT.
   * @param[out] transactions The transactions.
   * @param[in] test The json format of the test.
   * @param[out] count The number of transactions.
   * @param[in] start_index The index of the first transaction to be returned.
  */
  __host__ static void get_transactions(
      transaction_data_t *&transactions,
      const cJSON *test,
      size_t &count,
      size_t start_index = 0)
  {
    const cJSON *transaction_json = cJSON_GetObjectItemCaseSensitive(test, "transaction");
    arith_t arith(cgbn_report_monitor, 0);
    //transaction_data_t *transactions = NULL;
    size_t available_no_transactions = get_no_transaction(test);
    if (start_index >= available_no_transactions)
    {
      count = 0;
      transactions = NULL;
      return;
    }
    // set the number of transactions
    count = available_no_transactions - start_index;
    count = (count > MAX_TRANSACTION_COUNT) ? MAX_TRANSACTION_COUNT : count;
#ifndef ONLY_CPU
    CUDA_CHECK(cudaMallocManaged(
        (void **)&(transactions),
        count * sizeof(transaction_data_t)));
#else
    transactions = new transaction_data_t[count];
#endif
    transaction_data_t *template_transaction = new transaction_data_t;
    memset(template_transaction, 0, sizeof(transaction_data_t));

    uint8_t type;
    size_t data_index, gas_limit_index, value_index;
    size_t idx = 0, jdx = 0;

    type = 0;
    const cJSON *nonce_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "nonce");
    arith.cgbn_memory_from_hex_string(template_transaction->nonce, nonce_json->valuestring);

    const cJSON *gas_limit_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "gasLimit");
    size_t gas_limit_counts = cJSON_GetArraySize(gas_limit_json);

    const cJSON *to_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "to");
    if (strlen(to_json->valuestring) == 0)
    {
      arith.cgbn_memory_from_size_t(template_transaction->to, 0);
    }
    else
    {
      arith.cgbn_memory_from_hex_string(template_transaction->to, to_json->valuestring);
    }

    const cJSON *value_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "value");
    size_t value_counts = cJSON_GetArraySize(value_json);

    const cJSON *sender_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "sender");
    arith.cgbn_memory_from_hex_string(template_transaction->sender, sender_json->valuestring);

    const cJSON *access_list_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "accessList");
    if (access_list_json != NULL)
    {
      size_t accounts_counts = cJSON_GetArraySize(access_list_json);
      template_transaction->access_list.no_accounts = accounts_counts;
#ifndef ONLY_CPU
      CUDA_CHECK(cudaMallocManaged(
          (void **)&(template_transaction->access_list.accounts),
          accounts_counts * sizeof(access_list_account_t)));
#else
      template_transaction->access_list.accounts = new access_list_account_t[accounts_counts];
#endif
      for (idx = 0; idx < accounts_counts; idx++)
      {
        const cJSON *account_json = cJSON_GetArrayItem(access_list_json, idx);
        const cJSON *address_json = cJSON_GetObjectItemCaseSensitive(account_json, "address");
        arith.cgbn_memory_from_hex_string(
            template_transaction->access_list.accounts[idx].address,
            address_json->valuestring);
        const cJSON *storage_keys_json = cJSON_GetObjectItemCaseSensitive(account_json, "storageKeys");
        size_t storage_keys_counts = cJSON_GetArraySize(storage_keys_json);
        template_transaction->access_list.accounts[idx].no_storage_keys = storage_keys_counts;
#ifndef ONLY_CPU
        CUDA_CHECK(cudaMallocManaged(
            (void **)&(template_transaction->access_list.accounts[idx].storage_keys),
            storage_keys_counts * sizeof(evm_word_t)));
#else
        template_transaction->access_list.accounts[idx].storage_keys = new evm_word_t[storage_keys_counts];
#endif
        for (jdx = 0; jdx < storage_keys_counts; jdx++)
        {
          const cJSON *storage_key_json = cJSON_GetArrayItem(storage_keys_json, jdx);
          arith.cgbn_memory_from_hex_string(
              template_transaction->access_list.accounts[idx].storage_keys[jdx],
              storage_key_json->valuestring);
        }
      }
    }

    const cJSON *max_fee_per_gas_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "maxFeePerGas");

    const cJSON *max_priority_fee_per_gas_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "maxPriorityFeePerGas");

    const cJSON *gas_price_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "gasPrice");
    if (
        (max_fee_per_gas_json != NULL) &&
        (max_priority_fee_per_gas_json != NULL) &&
        (gas_price_json == NULL))
    {
      type = 2;
      arith.cgbn_memory_from_hex_string(
          template_transaction->max_fee_per_gas,
          max_fee_per_gas_json->valuestring);
      arith.cgbn_memory_from_hex_string(
          template_transaction->max_priority_fee_per_gas,
          max_priority_fee_per_gas_json->valuestring);
      arith.cgbn_memory_from_size_t(template_transaction->gas_price, 0);
    }
    else if (
        (max_fee_per_gas_json == NULL) &&
        (max_priority_fee_per_gas_json == NULL) &&
        (gas_price_json != NULL))
    {
      if (access_list_json == NULL)
      {
        type = 0;
      }
      else
      {
        type = 1;
      }
      arith.cgbn_memory_from_size_t(template_transaction->max_fee_per_gas, 0);
      arith.cgbn_memory_from_size_t(template_transaction->max_priority_fee_per_gas, 0);
      arith.cgbn_memory_from_hex_string(
          template_transaction->gas_price,
          gas_price_json->valuestring);
    }
    else
    {
      printf("ERROR: invalid transaction type\n");
      exit(1);
    }

    const cJSON *data_json = cJSON_GetObjectItemCaseSensitive(transaction_json, "data");
    size_t data_counts = cJSON_GetArraySize(data_json);

    template_transaction->type = type;

    size_t index;
    char *bytes_string = NULL;
    for (idx = 0; idx < count; idx++)
    {
      index = start_index + idx;
      data_index = index % data_counts;
      gas_limit_index = (index / data_counts) % gas_limit_counts;
      value_index = (index / (data_counts * gas_limit_counts)) % value_counts;
      memcpy(&(transactions[idx]), template_transaction, sizeof(transaction_data_t));
      arith.cgbn_memory_from_hex_string(
          transactions[idx].gas_limit,
          cJSON_GetArrayItem(gas_limit_json, gas_limit_index)->valuestring);
      arith.cgbn_memory_from_hex_string(
          transactions[idx].value,
          cJSON_GetArrayItem(value_json, value_index)->valuestring);
      bytes_string = cJSON_GetArrayItem(data_json, data_index)->valuestring;
      transactions[idx].data_init.size = adjusted_length(&bytes_string);
      if (transactions[idx].data_init.size > 0)
      {
#ifndef ONLY_CPU
        CUDA_CHECK(cudaMallocManaged(
            (void **)&(transactions[idx].data_init.data),
            transactions[idx].data_init.size * sizeof(uint8_t)));
#else
        transactions[idx].data_init.data = new uint8_t[transactions[idx].data_init.size];
#endif
        hex_to_bytes(
            bytes_string,
            transactions[idx].data_init.data,
            2 * transactions[idx].data_init.size);
      }
      else
      {
        transactions[idx].data_init.data = NULL;
      }
    }
    delete template_transaction;
    template_transaction = NULL;
    //return transactions;
  }

  /**
   * Free the memory allocated for transactions.
   * @param[in] transactions The transactions.
   * @param[in] count The number of transactions.
  */
  __host__ static void free_instances(
      transaction_data_t *transactions,
      size_t count)
  {
#ifndef ONLY_CPU
    if (transactions[0].access_list.no_accounts > 0)
    {
      for (size_t jdx = 0; jdx < transactions[0].access_list.no_accounts; jdx++)
      {
        if (transactions[0].access_list.accounts[jdx].no_storage_keys > 0)
        {
          CUDA_CHECK(cudaFree(transactions[0].access_list.accounts[jdx].storage_keys));
          transactions[0].access_list.accounts[jdx].no_storage_keys = 0;
          transactions[0].access_list.accounts[jdx].storage_keys = NULL;
        }
      }
      CUDA_CHECK(cudaFree(transactions[0].access_list.accounts));
    }
    for (size_t idx = 0; idx < count; idx++)
    {
      if (transactions[idx].data_init.size > 0)
      {
        CUDA_CHECK(cudaFree(transactions[idx].data_init.data));
        transactions[idx].data_init.size = 0;
      }
      if (transactions[idx].access_list.no_accounts > 0)
      {
        transactions[idx].access_list.accounts = NULL;
        transactions[idx].access_list.no_accounts = 0;
      }
    }
    CUDA_CHECK(cudaFree(transactions));
#else
    if (transactions[0].access_list.no_accounts > 0)
    {
      for (size_t jdx = 0; jdx < transactions[0].access_list.no_accounts; jdx++)
      {
        if (transactions[0].access_list.accounts[jdx].no_storage_keys > 0)
        {
          delete[] transactions[0].access_list.accounts[jdx].storage_keys;
          transactions[0].access_list.accounts[jdx].no_storage_keys = 0;
          transactions[0].access_list.accounts[jdx].storage_keys = NULL;
        }
      }
      delete[] transactions[0].access_list.accounts;
    }
    for (size_t idx = 0; idx < count; idx++)
    {
      if (transactions[idx].data_init.size > 0)
      {
        delete[] transactions[idx].data_init.data;
        transactions[idx].data_init.size = 0;
      }
      if (transactions[idx].access_list.no_accounts > 0)
      {
        transactions[idx].access_list.accounts = NULL;
        transactions[idx].access_list.no_accounts = 0;
      }
    }
    delete[] transactions;
#endif
  }
};

#endif