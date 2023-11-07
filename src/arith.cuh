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
  
  context_t _context;
  env_t     _env;
  uint32_t   _instance;
  
  //constructor
  __device__ __forceinline__ arith_env_t(cgbn_monitor_t monitor, cgbn_error_report_t *report, uint32_t instance) : _context(monitor, report, instance), _env(_context), _instance(instance) {
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
    cgbn_insert_bits_ui32(_env, a, a, 32, 32, (src >> 32));
    cgbn_insert_bits_ui32(_env, a, a, 0, 32, src);
  }

  __host__ __device__ __forceinline__ size_t from_cgbn_to_size_t(bn_t &a) {
    size_t dst = 0;
    dst = cgbn_extract_bits_ui32(_env, a, 0, 32);
    dst |= ((size_t)cgbn_extract_bits_ui32(_env, a, 32, 32)) << 32;
    return dst;
  }

};

#endif