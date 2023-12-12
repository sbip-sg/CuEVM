// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#ifndef _STACK_H_
#define _STACK_H_

#include "utils.h"

template <class params>
class stack_t
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
   * The arbitrary length integer type with double the size of the
   * EVM word type.
   */
  typedef typename arith_env_t<params>::bn_wide_t bn_wide_t;
  /**
   * The size of the stack.
   */
  static const uint32_t STACK_SIZE = params::STACK_SIZE;

  /**
   * The stack data structure.
   */
  typedef struct
  {
    evm_word_t *stack_base;
    uint32_t stack_offset;
  } stack_data_t;

  stack_data_t *_content;
  arith_t _arith;

  // constructor
  __host__ __device__ __forceinline__ stack_t(
      arith_t arith,
      stack_data_t *content) : _arith(arith),
                               _content(content)
  {
  }
  __host__ __device__ __forceinline__ stack_t(
      arith_t arith) : _arith(arith)
  {
    SHARED_MEMORY stack_data_t *content;
    ONE_THREAD_PER_INSTANCE(
        content = new stack_data_t;
        content->stack_base = new evm_word_t[STACK_SIZE];
        content->stack_offset = 0;)
    _content = content;
  }

  __host__ __device__ __forceinline__ ~stack_t()
  {
    ONE_THREAD_PER_INSTANCE(
        if (_content->stack_base != NULL) {
          delete[] _content->stack_base;
          _content->stack_base = NULL;
          _content->stack_offset = 0;
        } if (_content != NULL) {
          delete _content;
        })
    _content = NULL;
  }

  __host__ __device__ __forceinline__ uint32_t size()
  {
    return _content->stack_offset;
  }

  __host__ __device__ __forceinline__ evm_word_t *top()
  {
    return _content->stack_base + _content->stack_offset;
  }

  //
  __host__ __device__ __forceinline__ void push(const bn_t &value, uint32_t &error_code)
  {
    if (size() >= STACK_SIZE)
    {
      error_code = ERR_STACK_OVERFLOW;
      return;
    }
    cgbn_store(_arith._env, top(), value);
    _content->stack_offset++;
  }

  __host__ __device__ __forceinline__ void pop(bn_t &y, uint32_t &error_code)
  {
    if (size() == 0)
    {
      error_code = ERR_STACK_UNDERFLOW;
      cgbn_set_ui32(_arith._env, y, 0);
      return;
    }
    _content->stack_offset--;
    cgbn_load(_arith._env, y, top());
  }

  __host__ __device__ __forceinline__ void add(uint32_t &error_code)
  {
    bn_t a, b, r;
    pop(a, error_code);
    pop(b, error_code);
    cgbn_add(_arith._env, r, a, b);
    push(r, error_code);
  }

  __host__ __device__ __forceinline__ void sub(uint32_t &error_code)
  {
    bn_t a, b, r;
    pop(a, error_code);
    pop(b, error_code);
    cgbn_sub(_arith._env, r, a, b);
    push(r, error_code);
  }

  __host__ __device__ __forceinline__ void negate(uint32_t &error_code)
  {
    bn_t a, r;
    pop(a, error_code);
    cgbn_negate(_arith._env, r, a);
    push(r, error_code);
  }

  __host__ __device__ __forceinline__ void mul(uint32_t &error_code)
  {
    bn_t a, b, r;
    pop(a, error_code);
    pop(b, error_code);
    cgbn_mul(_arith._env, r, a, b);
    push(r, error_code);
  }

  __host__ __device__ __forceinline__ void div(uint32_t &error_code)
  {
    bn_t a, b, r;
    pop(a, error_code);
    pop(b, error_code);
    if (cgbn_compare_ui32(_arith._env, b, 0) == 0)
      cgbn_set_ui32(_arith._env, r, 0); // division by zero no error
    else
      cgbn_div(_arith._env, r, a, b);
    push(r, error_code);
  }

  __host__ __device__ __forceinline__ void sdiv(uint32_t &error_code)
  {
    bn_t a, b, r;
    pop(a, error_code);
    pop(b, error_code);
    bn_t d;
    bn_t e;
    // d = -1
    cgbn_set_ui32(_arith._env, d, 0);
    cgbn_sub_ui32(_arith._env, d, d, 1);
    // e = -2^254
    cgbn_set_ui32(_arith._env, e, 1);
    cgbn_shift_left(_arith._env, e, e, arith_t::BITS - 1);
    uint32_t sign_a = cgbn_extract_bits_ui32(_arith._env, a, arith_t::BITS - 1, 1);
    uint32_t sign_b = cgbn_extract_bits_ui32(_arith._env, b, arith_t::BITS - 1, 1);
    uint32_t sign = sign_a ^ sign_b;
    if (cgbn_compare_ui32(_arith._env, b, 0) == 0)
      cgbn_set_ui32(_arith._env, r, 0);
    else if (
        (cgbn_compare(_arith._env, b, d) == 0) &&
        (cgbn_compare(_arith._env, a, e) == 0))
    {
      cgbn_set(_arith._env, r, e); // -2^254 / -1 = -2^254
    }
    else
    {
      // div between absolute values
      if (sign_a == 1)
      {
        cgbn_negate(_arith._env, a, a);
      }
      if (sign_b == 1)
      {
        cgbn_negate(_arith._env, b, b);
      }
      cgbn_div(_arith._env, r, a, b);
      if (sign)
      {
        cgbn_negate(_arith._env, r, r);
      }
    }
    push(r, error_code);
  }

  __host__ __device__ __forceinline__ void mod(uint32_t &error_code)
  {
    bn_t a, b, r;
    pop(a, error_code);
    pop(b, error_code);
    if (cgbn_compare_ui32(_arith._env, b, 0) == 0)
      cgbn_set_ui32(_arith._env, r, 0); // rem by zero
    else
      cgbn_rem(_arith._env, r, a, b);
    push(r, error_code);
  }

  __host__ __device__ __forceinline__ void smod(uint32_t &error_code)
  {
    bn_t a, b, r;
    pop(a, error_code);
    pop(b, error_code);
    uint32_t sign_a = cgbn_extract_bits_ui32(_arith._env, a, arith_t::BITS - 1, 1);
    uint32_t sign_b = cgbn_extract_bits_ui32(_arith._env, b, arith_t::BITS - 1, 1);
    uint32_t sign = sign_a ^ sign_b;
    if (cgbn_compare_ui32(_arith._env, b, 0) == 0)
      cgbn_set_ui32(_arith._env, r, 0);
    else
    {
      // mod between absolute values
      if (sign_a == 1)
      {
        cgbn_negate(_arith._env, a, a);
      }
      if (sign_b == 1)
      {
        cgbn_negate(_arith._env, b, b);
      }
      cgbn_rem(_arith._env, r, a, b);
      if (sign)
      {
        cgbn_negate(_arith._env, r, r);
      }
    }
    push(r, error_code);
  }

  __host__ __device__ __forceinline__ void addmod(uint32_t &error_code)
  {
    bn_t a, b, c, N, r;
    pop(a, error_code);
    pop(b, error_code);
    pop(N, error_code);
    if (cgbn_compare_ui32(_arith._env, N, 0) == 0)
    {
      cgbn_set_ui32(_arith._env, r, 0);
    }
    else if (cgbn_compare_ui32(_arith._env, N, 1) == 0)
    {
      cgbn_set_ui32(_arith._env, r, 0);
    }
    else
    {
      int32_t carry = cgbn_add(_arith._env, c, a, b);
      bn_wide_t d;
      if (carry == 1)
      {
        cgbn_set_ui32(_arith._env, d._high, 1);
        cgbn_set(_arith._env, d._low, c);
        cgbn_rem_wide(_arith._env, r, d, N);
      }
      else
      {
        cgbn_rem(_arith._env, r, c, N);
      }
    }
    push(r, error_code);
  }

  __host__ __device__ __forceinline__ void mulmod(uint32_t &error_code)
  {
    bn_t a, b, N, r;
    pop(a, error_code);
    pop(b, error_code);
    pop(N, error_code);
    if (cgbn_compare_ui32(_arith._env, N, 0) == 0)
    {
      cgbn_set_ui32(_arith._env, r, 0);
    }
    else
    {
      bn_wide_t d;
      cgbn_rem(_arith._env, a, a, N);
      cgbn_rem(_arith._env, b, b, N);
      cgbn_mul_wide(_arith._env, d, a, b);
      cgbn_rem_wide(_arith._env, r, d, N);
    }
    push(r, error_code);
  }

  __host__ __device__ __forceinline__ void exp(uint32_t &error_code, bn_t &gas_cost)
  {
    bn_t a, exponent, r;
    pop(a, error_code);
    pop(exponent, error_code);
    bn_t current, square;
    int32_t bit, last_bit;
    cgbn_set_ui32(_arith._env, current, 1); // r=1
    cgbn_set(_arith._env, square, a);       // square=a
    last_bit = params::BITS - 1 - cgbn_clz(_arith._env, exponent);
    //^0=1 even for 0^0
    if (last_bit == -1)
    {
      cgbn_set_ui32(_arith._env, r, 1);
    }
    else
    {
      uint32_t exponent_byte_size = (last_bit) / 8 + 1;
      bn_t c;
      cgbn_set_ui32(_arith._env, c, exponent_byte_size);
      cgbn_mul_ui32(_arith._env, c, c, 50);
      cgbn_add(_arith._env, gas_cost, gas_cost, c);
      for (bit = 0; bit <= last_bit; bit++)
      {
        if (cgbn_extract_bits_ui32(_arith._env, exponent, bit, 1) == 1)
        {
          cgbn_mul(_arith._env, current, current, square); // r=r*square
        }
        cgbn_mul(_arith._env, square, square, square); // square=square*square
      }
      cgbn_set(_arith._env, r, current);
    }
    push(r, error_code);
  }

  /*
  Even if x has more bytes than the value b, the operation consider only the first
  (b+1) bytes of x and the other are considered zero and they don't have any influence
  on the final result.
  Optimised: use cgbn_bitwise_mask_ior instead of cgbn_insert_bits_ui32
  */
  __host__ __device__ __forceinline__ void signextend(uint32_t &error_code)
  {
    bn_t b, x, r;
    pop(b, error_code);
    pop(x, error_code);
    if (cgbn_compare_ui32(_arith._env, b, 31) == 1)
    {
      cgbn_set(_arith._env, r, x);
    }
    else
    {
      uint32_t c = cgbn_get_ui32(_arith._env, b) + 1;
      uint32_t sign = cgbn_extract_bits_ui32(_arith._env, x, c * 8 - 1, 1);
      int32_t numbits = int32_t(c);
      if (sign == 1)
      {
        numbits = int32_t(arith_t::BITS) - 8 * numbits;
        numbits = -numbits;
        cgbn_bitwise_mask_ior(_arith._env, r, x, numbits);
        /*
        c=c-1;
        for(uint32_t i=0;i<=params::BITS/8-c;i++) {
          cgbn_insert_bits_ui32(_arith._env, r, x, params::BITS - 8 * i, 8, 0xff);
        }
        */
      }
      else
      {
        // needs and it seems
        cgbn_bitwise_mask_and(_arith._env, r, x, 8 * numbits);
        // cgbn_set(_arith._env, r, x);
      }
    }
    push(r, error_code);
  }

  __host__ __device__ __forceinline__ int32_t compare(uint32_t &error_code)
  {
    bn_t a, b;
    pop(a, error_code);
    pop(b, error_code);
    return cgbn_compare(_arith._env, a, b);
  }

  __host__ __device__ __forceinline__ void lt(uint32_t &error_code)
  {
    int32_t int_result = compare(error_code);
    uint32_t result = (int_result < 0) ? 1 : 0;
    bn_t r;
    cgbn_set_ui32(_arith._env, r, result);
    push(r, error_code);
  }

  __host__ __device__ __forceinline__ void gt(uint32_t &error_code)
  {
    int32_t int_result = compare(error_code);
    uint32_t result = (int_result > 0) ? 1 : 0;
    bn_t r;
    cgbn_set_ui32(_arith._env, r, result);
    push(r, error_code);
  }

  __host__ __device__ __forceinline__ int32_t scompare(uint32_t &error_code)
  {
    bn_t a, b;
    pop(a, error_code);
    pop(b, error_code);
    uint32_t sign_a = cgbn_extract_bits_ui32(_arith._env, a, arith_t::BITS - 1, 1);
    uint32_t sign_b = cgbn_extract_bits_ui32(_arith._env, b, arith_t::BITS - 1, 1);
    if (sign_a == 0 && sign_b == 1)
    {
      return 1;
    }
    else if (sign_a == 1 && sign_b == 0)
    {
      return -1;
    }
    else
    {
      return cgbn_compare(_arith._env, a, b);
    }
  }

  __host__ __device__ __forceinline__ void slt(uint32_t &error_code)
  {
    int32_t int_result = scompare(error_code);
    uint32_t result = (int_result < 0) ? 1 : 0;
    bn_t r;
    cgbn_set_ui32(_arith._env, r, result);
    push(r, error_code);
  }

  __host__ __device__ __forceinline__ void sgt(uint32_t &error_code)
  {
    int32_t int_result = scompare(error_code);
    uint32_t result = (int_result > 0) ? 1 : 0;
    bn_t r;
    cgbn_set_ui32(_arith._env, r, result);
    push(r, error_code);
  }

  __host__ __device__ __forceinline__ void eq(uint32_t &error_code)
  {
    int32_t int_result = compare(error_code);
    uint32_t result = (int_result == 0) ? 1 : 0;
    bn_t r;
    cgbn_set_ui32(_arith._env, r, result);
    push(r, error_code);
  }

  __host__ __device__ __forceinline__ void iszero(uint32_t &error_code)
  {
    bn_t a, r;
    pop(a, error_code);
    int32_t compare = cgbn_compare_ui32(_arith._env, a, 0);
    if (compare == 0)
    {
      cgbn_set_ui32(_arith._env, r, 1);
    }
    else
    {
      cgbn_set_ui32(_arith._env, r, 0);
    }
    push(r, error_code);
  }

  __host__ __device__ __forceinline__ void bitwise_and(uint32_t &error_code)
  {
    bn_t a, b, r;
    pop(a, error_code);
    pop(b, error_code);
    cgbn_bitwise_and(_arith._env, r, a, b);
    push(r, error_code);
  }

  __host__ __device__ __forceinline__ void bitwise_or(uint32_t &error_code)
  {
    bn_t a, b, r;
    pop(a, error_code);
    pop(b, error_code);
    cgbn_bitwise_ior(_arith._env, r, a, b);
    push(r, error_code);
  }

  __host__ __device__ __forceinline__ void bitwise_xor(uint32_t &error_code)
  {
    bn_t a, b, r;
    pop(a, error_code);
    pop(b, error_code);
    cgbn_bitwise_xor(_arith._env, r, a, b);
    push(r, error_code);
  }

  __host__ __device__ __forceinline__ void bitwise_not(uint32_t &error_code)
  {
    bn_t a, r;
    pop(a, error_code);
    cgbn_bitwise_mask_xor(_arith._env, r, a, arith_t::BITS);
    push(r, error_code);
  }

  __host__ __device__ __forceinline__ void get_byte(uint32_t &error_code)
  {
    bn_t i, x, r;
    pop(i, error_code);
    pop(x, error_code);
    if (cgbn_compare_ui32(_arith._env, i, 31) == 1)
    {
      cgbn_set_ui32(_arith._env, r, 0);
    }
    else
    {
      uint32_t index = cgbn_get_ui32(_arith._env, i);
      uint32_t byte = cgbn_extract_bits_ui32(_arith._env, x, 8 * ((params::BITS / 8 - 1) - index), 8);
      cgbn_set_ui32(_arith._env, r, byte);
    }
    push(r, error_code);
  }

  __host__ __device__ __forceinline__ void shl(uint32_t &error_code)
  {
    bn_t shift, value, r;
    pop(shift, error_code);
    pop(value, error_code);
    if (cgbn_compare_ui32(_arith._env, shift, arith_t::BITS - 1) == 1)
    {
      cgbn_set_ui32(_arith._env, r, 0);
    }
    else
    {
      uint32_t shift_left = cgbn_get_ui32(_arith._env, shift);
      cgbn_shift_left(_arith._env, r, value, shift_left);
    }
    push(r, error_code);
  }

  __host__ __device__ __forceinline__ void shr(uint32_t &error_code)
  {
    bn_t shift, value, r;
    pop(shift, error_code);
    pop(value, error_code);
    if (cgbn_compare_ui32(_arith._env, shift, arith_t::BITS - 1) == 1)
    {
      cgbn_set_ui32(_arith._env, r, 0);
    }
    else
    {
      uint32_t shift_right = cgbn_get_ui32(_arith._env, shift);
      cgbn_shift_right(_arith._env, r, value, shift_right);
    }
    push(r, error_code);
  }

  __host__ __device__ __forceinline__ void sar(uint32_t &error_code)
  {
    bn_t shift, value, r;
    pop(shift, error_code);
    pop(value, error_code);
    uint32_t sign_b = cgbn_extract_bits_ui32(_arith._env, value, arith_t::BITS - 1, 1);
    uint32_t shift_right = cgbn_get_ui32(_arith._env, shift);

    if (cgbn_compare_ui32(_arith._env, shift, arith_t::BITS - 1) == 1)
      shift_right = arith_t::BITS;

    cgbn_shift_right(_arith._env, r, value, shift_right);
    if (sign_b == 1)
    {
      cgbn_bitwise_mask_ior(_arith._env, r, r, -shift_right);
    }
    push(r, error_code);
  }

  __host__ __device__ __forceinline__ void pushx(uint8_t *value, uint32_t size, uint32_t &error_code)
  {
    if (size > 32)
    {
      error_code = ERR_STACK_INVALID_SIZE;
      return;
    }
    bn_t r;
    cgbn_set_ui32(_arith._env, r, 0);
    for (uint32_t i = 0; i < size; i++)
    {
      cgbn_insert_bits_ui32(_arith._env, r, r, i * 8, 8, value[size - 1 - i]);
    }
    push(r, error_code);
  }

  __host__ __device__ __forceinline__ evm_word_t *get_index(uint32_t index, uint32_t &error_code)
  {
    if (size() < index)
    {
      error_code = ERR_STACK_UNDERFLOW;
      return NULL;
    }
    return _content->stack_base + (size() - index);
  }

  __host__ __device__ __forceinline__ void dupx(uint32_t index, uint32_t &error_code)
  {
    if ((index < 1) || (index > 16))
    {
      error_code = ERR_STACK_INVALID_SIZE;
      return;
    }
    bn_t r;
    evm_word_t *value = get_index(index, error_code);
    if (value == NULL)
    {
      return;
    }
    cgbn_load(_arith._env, r, value);
    push(r, error_code);
  }

  __host__ __device__ __forceinline__ void swapx(uint32_t index, uint32_t &error_code)
  {
    if ((index < 1) || (index > 16))
    {
      error_code = ERR_STACK_INVALID_SIZE;
      return;
    }
    bn_t a, b;
    evm_word_t *value_a = get_index(1, error_code);
    evm_word_t *value_b = get_index(index + 1, error_code);
    if ((value_a == NULL) || (value_b == NULL))
    {
      return;
    }
    cgbn_load(_arith._env, a, value_b);
    cgbn_load(_arith._env, b, value_a);
    cgbn_store(_arith._env, value_a, a);
    cgbn_store(_arith._env, value_b, b);
  }

  __host__ __device__ __forceinline__ void to_stack_data_t(
    stack_data_t &dst
  )
  {
    ONE_THREAD_PER_INSTANCE(
      if (
          (dst.stack_offset > 0) &&
          (dst.stack_base != NULL))
      {
        delete[] dst.stack_base;
        dst.stack_base = NULL;
      }
      dst.stack_offset = _content->stack_offset;
      if (dst.stack_offset == 0)
      {
        dst.stack_base = NULL;
      }
      else
      {
        dst.stack_base = new evm_word_t[dst.stack_offset];
        memcpy(
            dst.stack_base,
            _content->stack_base,
            sizeof(evm_word_t) * dst.stack_offset);
      }
    )
  }

  __host__ __device__ __forceinline__ static void print_stack_data_t(
    arith_t &arith,
    stack_data_t &stack_data)
  {
    printf("Stack size: %d, data:\n", stack_data.stack_offset);
    for (uint32_t idx = 0; idx < stack_data.stack_offset; idx++)
    {
      arith.print_cgbn_memory(stack_data.stack_base[idx]);
    }
  }

  __host__ __device__ void print(
      bool full = false)
  {
    printf("Stack size: %d, data:\n", size());
    uint32_t print_size = full ? STACK_SIZE : size();
    for (uint32_t idx = 0; idx < print_size; idx++)
    {
      _arith.print_cgbn_memory(_content->stack_base[idx]);
    }
  }

  __host__ static cJSON *json_from_stack_data_t(
    arith_t &arith,
    stack_data_t &stack_data
  )
  {
    char *hex_string_ptr = new char[arith_t::BYTES * 2 + 3];
    cJSON *stack_json = cJSON_CreateObject();

    cJSON *stack_data_json = cJSON_CreateArray();
    for (uint32_t idx = 0; idx < stack_data.stack_offset; idx++)
    {
      arith.hex_string_from_cgbn_memory(hex_string_ptr, stack_data.stack_base[idx]);
      cJSON_AddItemToArray(stack_data_json, cJSON_CreateString(hex_string_ptr));
    }
    cJSON_AddItemToObject(stack_json, "data", stack_data_json);
    delete[] hex_string_ptr;
    return stack_json;
  }

  __host__ cJSON *json(bool full = false)
  {
    char *hex_string_ptr = new char[arith_t::BYTES * 2 + 3];
    cJSON *stack_json = cJSON_CreateObject();

    cJSON *stack_data_json = cJSON_CreateArray();
    uint32_t print_size = full ? STACK_SIZE : size();
    for (uint32_t idx = 0; idx < print_size; idx++)
    {
      _arith.hex_string_from_cgbn_memory(hex_string_ptr, _content->stack_base[idx]);
      cJSON_AddItemToArray(stack_data_json, cJSON_CreateString(hex_string_ptr));
    }
    cJSON_AddItemToObject(stack_json, "data", stack_data_json);
    delete[] hex_string_ptr;
    return stack_json;
  }

  // support routine to generate instances
  __host__ static stack_data_t *get_cpu_instances(
      uint32_t count)
  {
    stack_data_t *cpu_instances = new stack_data_t[count];
    for (uint32_t idx = 0; idx < count; idx++)
    {
      cpu_instances[idx].stack_base = NULL;
      cpu_instances[idx].stack_offset = 0;
    }
    return cpu_instances;
  }

  __host__ static void free_cpu_instances(
      stack_data_t *cpu_instances,
      uint32_t count)
  {
    for (int index = 0; index < count; index++)
    {
      if (cpu_instances[index].stack_base != NULL)
      {
        delete[] cpu_instances[index].stack_base;
        cpu_instances[index].stack_base = NULL;
      }
      cpu_instances[index].stack_offset = 0;
    }
    delete[] cpu_instances;
  }

  __host__ static stack_data_t *get_gpu_instances_from_cpu_instances(
      stack_data_t *cpu_instances,
      uint32_t count)
  {
    stack_data_t *gpu_instances, *tmp_cpu_instances;
    tmp_cpu_instances = new stack_data_t[count];
    memcpy(
        tmp_cpu_instances,
        cpu_instances,
        sizeof(stack_data_t) * count);
    for (uint32_t idx = 0; idx < count; idx++)
    {
      if (cpu_instances[idx].stack_offset > 0)
      {
        CUDA_CHECK(cudaMalloc(
            (void **)&tmp_cpu_instances[idx].stack_base,
            sizeof(evm_word_t) * cpu_instances[idx].stack_offset));
        CUDA_CHECK(cudaMemcpy(
            tmp_cpu_instances[idx].stack_base,
            cpu_instances[idx].stack_base,
            sizeof(evm_word_t) * cpu_instances[idx].stack_offset,
            cudaMemcpyHostToDevice));
      } else {
        tmp_cpu_instances[idx].stack_base = NULL;
      }
    }
    CUDA_CHECK(cudaMalloc(
        (void **)&gpu_instances,
        sizeof(stack_data_t) * count));
    CUDA_CHECK(cudaMemcpy(
        gpu_instances,
        tmp_cpu_instances,
        sizeof(stack_data_t) * count,
        cudaMemcpyHostToDevice));
    delete[] tmp_cpu_instances;
    tmp_cpu_instances = NULL;
    return gpu_instances;
  }

  __host__ static void free_gpu_instances(
      stack_data_t *gpu_instances,
      uint32_t count)
  {
    stack_data_t *tmp_cpu_instances;
    tmp_cpu_instances = new stack_data_t[count];
    CUDA_CHECK(cudaMemcpy(
        tmp_cpu_instances,
        gpu_instances,
        sizeof(stack_data_t) * count,
        cudaMemcpyDeviceToHost));
    for (uint32_t idx = 0; idx < count; idx++)
    {
      if (tmp_cpu_instances[idx].stack_base != NULL)
        CUDA_CHECK(cudaFree(tmp_cpu_instances[idx].stack_base));
    }
    delete[] tmp_cpu_instances;
    CUDA_CHECK(cudaFree(gpu_instances));
  }

  __host__ static stack_data_t *get_cpu_instances_from_gpu_instances(
      stack_data_t *gpu_instances,
      uint32_t count)
  {
    stack_data_t *cpu_instances, *tmp_cpu_instances, *tmp_gpu_instances;
    cpu_instances = new stack_data_t[count];
    tmp_cpu_instances = new stack_data_t[count];
    CUDA_CHECK(cudaMemcpy(
        cpu_instances,
        gpu_instances,
        sizeof(stack_data_t) * count,
        cudaMemcpyDeviceToHost));
    memcpy(
        tmp_cpu_instances,
        cpu_instances,
        sizeof(stack_data_t) * count);
    for (uint32_t idx = 0; idx < count; idx++)
    {
      tmp_cpu_instances[idx].stack_offset = cpu_instances[idx].stack_offset;
      if (cpu_instances[idx].stack_offset > 0)
      {
        CUDA_CHECK(cudaMalloc(
            (void **)&tmp_cpu_instances[idx].stack_base,
            sizeof(evm_word_t) * cpu_instances[idx].stack_offset));
      } else {
        tmp_cpu_instances[idx].stack_base = NULL;
      }
    }
    CUDA_CHECK(cudaMalloc(
        (void **)&tmp_gpu_instances,
        sizeof(stack_data_t) * count));
    CUDA_CHECK(cudaMemcpy(
        tmp_gpu_instances,
        tmp_cpu_instances,
        sizeof(stack_data_t) * count,
        cudaMemcpyHostToDevice));
    delete[] tmp_cpu_instances;
    tmp_cpu_instances = NULL;
    kernel_stacks<params><<<1, count>>>(
        tmp_gpu_instances,
        gpu_instances,
        count);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(gpu_instances));
    gpu_instances = tmp_gpu_instances;

    CUDA_CHECK(cudaMemcpy(
        cpu_instances,
        gpu_instances,
        sizeof(stack_data_t) * count,
        cudaMemcpyDeviceToHost));
    tmp_cpu_instances = new stack_data_t[count];
    memcpy(
        tmp_cpu_instances,
        cpu_instances,
        sizeof(stack_data_t) * count);
    
    for (uint32_t idx = 0; idx < count; idx++)
    {
      tmp_cpu_instances[idx].stack_offset = cpu_instances[idx].stack_offset;
      if (cpu_instances[idx].stack_offset > 0)
      {
        tmp_cpu_instances[idx].stack_base = new evm_word_t[cpu_instances[idx].stack_offset];
        CUDA_CHECK(cudaMemcpy(
            tmp_cpu_instances[idx].stack_base,
            cpu_instances[idx].stack_base,
            sizeof(evm_word_t) * cpu_instances[idx].stack_offset,
            cudaMemcpyDeviceToHost));
      } else {
        tmp_cpu_instances[idx].stack_base = NULL;
      }
    }

    memcpy(
        cpu_instances,
        tmp_cpu_instances,
        sizeof(stack_data_t) * count);
    delete[] tmp_cpu_instances;
    tmp_cpu_instances = NULL;
    free_gpu_instances(gpu_instances, count);
    return cpu_instances;
  }
};


template <class params>
__global__ void kernel_stacks(
  typename stack_t<params>::stack_data_t *dst,
  typename stack_t<params>::stack_data_t *src,
  uint32_t count
)
{
  typedef typename stack_t<params>::evm_word_t evm_word_t;
  uint32_t instance = blockIdx.x * blockDim.x + threadIdx.x;
  if (instance >= count)
  {
    return;
  }
  dst[instance].stack_offset = src[instance].stack_offset;
  if (dst[instance].stack_offset > 0)
  {
    memcpy(
        dst[instance].stack_base,
        src[instance].stack_base,
        sizeof(evm_word_t) * src[instance].stack_offset);
    delete[] src[instance].stack_base;
    src[instance].stack_base = NULL;
    src[instance].stack_offset = 0;
  }
}


#endif