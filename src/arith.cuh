#ifndef _ARITH_H_
#define _ARITH_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <gmp.h>
#ifndef __CGBN_H__
#define __CGBN_H__
#include <cgbn/cgbn.h>
#endif


template<class params>
class arith_env_t {
  public:

  typedef cgbn_context_t<params::TPI, params>    context_t;
  typedef cgbn_env_t<context_t, params::BITS>    env_t;
  typedef typename env_t::cgbn_t                 bn_t;
  typedef typename env_t::cgbn_wide_t            bn_wide_t;
  typedef cgbn_mem_t<params::BITS>               evm_word_t;
  static const uint32_t                          BITS = params::BITS;
  static const uint32_t                          BYTES = params::BITS/8;
  static const uint32_t                          LIMBS = params::BITS/32;
  
  context_t _context;
  env_t     _env;
  uint32_t   _instance;
  
  //constructor
  __device__ __forceinline__ arith_env_t(cgbn_monitor_t monitor, cgbn_error_report_t *report, uint32_t instance) : _context(monitor, report, instance), _env(_context), _instance(instance) {
  }
  __device__ __forceinline__ arith_env_t(cgbn_monitor_t monitor) : _context(monitor), _env(_context), _instance(0) {
  }

  //constructor
  __host__ __forceinline__ arith_env_t(cgbn_monitor_t monitor, uint32_t instance) : _context(monitor), _env(_context), _instance(instance) {
  }

  //clone constructor
  __host__ __device__ __forceinline__ arith_env_t(const arith_env_t &env) : _context(env._context), _env(_context), _instance(env._instance) {
  }
  
  // get memory from cgbn
  __host__ __device__ __forceinline__ void from_cgbn_to_memory(uint8_t *dst, bn_t &a) {
    for(uint32_t idx = 0; idx < params::BITS/8; idx++) {
      dst[idx] = cgbn_extract_bits_ui32(_env, a, params::BITS - (idx+1)*8, 8);
    }
  }

  //set cgbn from memory
  __host__ __device__ __forceinline__ void from_memory_to_cgbn(bn_t &a, uint8_t *src) {
    for(uint32_t idx = 0; idx < params::BITS/8; idx++) {
      cgbn_insert_bits_ui32(_env, a, a, params::BITS - (idx+1)*8, 8, src[idx]);
    }
  }

  __host__ __device__ __forceinline__ void from_size_t_to_cgbn(bn_t &a, size_t src) {
    cgbn_set_ui32(_env, a, 0);
    cgbn_insert_bits_ui32(_env, a, a, 32, 32, (src >> 32));
    cgbn_insert_bits_ui32(_env, a, a, 0, 32, src);
  }

  __host__ __device__ __forceinline__ size_t from_cgbn_to_size_t(bn_t &a) {
    size_t dst = 0;
    dst = cgbn_extract_bits_ui32(_env, a, 0, 32);
    dst |= ((size_t)cgbn_extract_bits_ui32(_env, a, 32, 32)) << 32;
    return dst;
  }

  __host__ __device__ __forceinline__ int32_t size_t_from_cgbn(size_t &dst, bn_t &src) {
    bn_t MAX_SIZE_T;
    cgbn_set_ui32(_env, MAX_SIZE_T, 1);
    cgbn_shift_left(_env, MAX_SIZE_T, MAX_SIZE_T, 64);
    dst = 0;
    dst = cgbn_extract_bits_ui32(_env, src, 0, 32);
    dst |= ((size_t)cgbn_extract_bits_ui32(_env, src, 32, 32)) << 32;
    if(cgbn_compare(_env, src, MAX_SIZE_T) >= 0) {
      return 1;
    } else {
      return 0;
    }
  }

  __host__ void from_cgbn_memory_to_hex(evm_word_t &a, char *hex_string, size_t count=params::BITS/32) {
    //char *hex_string = (char *) malloc(sizeof(char) * (count*8+3));
    hex_string[0] = '0';
    hex_string[1] = 'x';
    for(size_t idx = 0; idx < count; idx++) {
      sprintf(hex_string+2+idx*8, "%08x", a._limbs[count-1-idx]);
    }
    hex_string[count*8+2] = '\0';
  }

  __host__ int32_t cgbn_memory_from_hex_string(evm_word_t &a, char *hex_string) {
    mpz_t value;
    size_t written;
    mpz_init(value);
    if ((hex_string[0] == '0') &&
    ((hex_string[1] == 'x') || (hex_string[1] == 'X'))) {
      mpz_set_str(value, hex_string+2, 16);
    } else {
      mpz_set_str(value, hex_string, 16);
    }
    if(mpz_sizeinbase(value, 2) > BITS) {
      return 1;
    }
    mpz_export(a._limbs, &written, -1, sizeof(uint32_t), 0, 0, value);
    while(written < LIMBS) {
      a._limbs[written++] = 0;
    }
    mpz_clear(value);
    return 0;
  }

};

#endif