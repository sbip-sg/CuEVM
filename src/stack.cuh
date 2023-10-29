#ifndef _GPU_STACK_H_
#define _GPU_STACK_H_

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
class gpu_stack_t {
  public:

  typedef cgbn_context_t<params::TPI, params>    context_t;
  typedef cgbn_env_t<context_t, params::BITS>    env_t;
  typedef typename env_t::cgbn_t                 bn_t;
  typedef typename env_t::cgbn_wide_t            bn_wide_t;
  
  typedef struct {
    cgbn_mem_t<params::BITS> values[params::STACK_SIZE];
  } instance_t;


  cgbn_mem_t<params::BITS> *_stack;
  uint32_t _top;
  context_t _context;
  env_t     _env;
  uint32_t   _instance;
  
  //constructor
  __device__ __forceinline__ gpu_stack_t(cgbn_monitor_t monitor, cgbn_error_report_t *report, uint32_t instance, cgbn_mem_t<params::BITS> *stack, uint32_t top) : _context(monitor, report, instance), _env(_context), _instance(instance), _stack(stack), _top(top) {
  }

  //
  __device__ __forceinline__ void push(const bn_t &value) {
    if (_top == 0) {
      printf("Stack overflow\n");
      return;
    }
    _top--;
    cgbn_store(_env, _stack + _top, value);
  }

  __device__ __forceinline__ void pop(bn_t &value) {
    if (_top == params::STACK_SIZE) {
      printf("Stack underflow\n");
      return;
    }
    cgbn_load(_env, value, &(_stack[_top]));
    _top++;
  }

  __device__ __forceinline__ void add() {
    if (_top > params::STACK_SIZE - 2) {
      printf("Stack underflow\n");
      return;
    }
    bn_t  a, b, r;
    cgbn_load(_env, a, &(_stack[_top]));
    cgbn_load(_env, b, &(_stack[_top+1]));
    cgbn_add(_env, r, a, b);
    cgbn_store(_env, &(_stack[_top+1]), r);
    _top++;
  }

  __device__ __forceinline__ void sub() {
    if (_top > params::STACK_SIZE - 2) {
      printf("Stack underflow\n");
      return;
    }
    bn_t  a, b, r;
    cgbn_load(_env, a, &(_stack[_top]));
    cgbn_load(_env, b, &(_stack[_top+1]));
    cgbn_sub(_env, r, a, b);
    cgbn_store(_env, &(_stack[_top+1]), r);
    _top++;
  }
  
  
  __device__ __forceinline__ void negate() {
    if (_top > params::STACK_SIZE - 1) {
      printf("Stack underflow\n");
      return;
    }
    bn_t  a, r;
    cgbn_load(_env, a, &(_stack[_top]));
    cgbn_negate(_env, r, a);
    cgbn_store(_env, &(_stack[_top]), r);
  }


  __device__ __forceinline__ void mul() {
    if (_top > params::STACK_SIZE - 2) {
      printf("Stack underflow\n");
      return;
    }
    bn_t  a, b, r;
    cgbn_load(_env, a, &(_stack[_top]));
    cgbn_load(_env, b, &(_stack[_top+1]));
    cgbn_mul(_env, r, a, b);
    cgbn_store(_env, &(_stack[_top+1]), r);
    _top++;
  }

  __device__ __forceinline__ void div() {
    if (_top > params::STACK_SIZE - 2) {
      printf("Stack underflow\n");
      return;
    }
    bn_t  a, b, r;
    cgbn_load(_env, a, &(_stack[_top]));
    cgbn_load(_env, b, &(_stack[_top+1]));
    if (cgbn_compare_ui32(_env, b, 0) == 0)
      cgbn_set_ui32(_env, r, 0);
    else
      cgbn_div(_env, r, a, b);
    cgbn_store(_env, &(_stack[_top+1]), r);
    _top++;
  }

  __device__ __forceinline__ void sdiv() {
    if (_top > params::STACK_SIZE - 2) {
      printf("Stack underflow\n");
      return;
    }
    bn_t  a, b, r;
    cgbn_load(_env, a, &(_stack[_top]));
    cgbn_load(_env, b, &(_stack[_top+1]));
    bn_t d;
    bn_t e;
    cgbn_set_ui32(_env, d, 0);
    cgbn_sub_ui32(_env, d, d, 1);
    cgbn_set_ui32(_env, e, 1);
    cgbn_shift_left(_env, e, e, params::BITS-1);
    uint32_t sign_a = cgbn_extract_bits_ui32(_env, a, params::BITS-1, 1);
    uint32_t sign_b = cgbn_extract_bits_ui32(_env, b, params::BITS-1, 1);
    uint32_t sign = sign_a ^ sign_b;
    if (cgbn_compare_ui32(_env, b, 0) == 0)
      cgbn_set_ui32(_env, r, 0);
    else if(
      (cgbn_compare(_env, b, d) == 0) &&
      (cgbn_compare(_env, a, e) == 0) ) {
        cgbn_set(_env, r, e);
    } else {
      cgbn_div(_env, r, a, b);
      if (sign) {
        cgbn_negate(_env, r, r);
      }

    }
    cgbn_store(_env, &(_stack[_top+1]), r);
    _top++;
  }

  __device__ __forceinline__ void mod() {
    if (_top > params::STACK_SIZE - 2) {
      printf("Stack underflow\n");
      return;
    }
    bn_t  a, b, r;
    cgbn_load(_env, a, &(_stack[_top]));
    cgbn_load(_env, b, &(_stack[_top+1]));
    cgbn_rem(_env, r, a, b);
    cgbn_store(_env, &(_stack[_top+1]), r);
    _top++;
  }
  
  __device__ __forceinline__ void smod() {
    if (_top > params::STACK_SIZE - 2) {
      printf("Stack underflow\n");
      return;
    }
    bn_t  a, b, r;
    cgbn_load(_env, a, &(_stack[_top]));
    cgbn_load(_env, b, &(_stack[_top+1]));
    uint32_t sign_a = cgbn_extract_bits_ui32(_env, a, params::BITS-1, 1);
    uint32_t sign_b = cgbn_extract_bits_ui32(_env, b, params::BITS-1, 1);
    uint32_t sign = sign_a ^ sign_b;
    if (cgbn_compare_ui32(_env, b, 0) == 0)
      cgbn_set_ui32(_env, r, 0);
    else {
      cgbn_rem(_env, r, a, b);
      if (sign) {
        cgbn_negate(_env, r, r);
      }
    }
    cgbn_store(_env, &(_stack[_top+1]), r);
    _top++;
  }

  __device__ __forceinline__ void addmod() {
    if (_top > params::STACK_SIZE - 3) {
      printf("Stack underflow\n");
      return;
    }
    bn_t  a, b, c, m, r;
    cgbn_load(_env, a, &(_stack[_top]));
    cgbn_load(_env, b, &(_stack[_top+1]));
    cgbn_load(_env, m, &(_stack[_top+2]));
    int32_t carry=cgbn_add(_env, c, a, b);
    bn_wide_t d;
    if (carry) {
      cgbn_set_ui32(_env, d._high, 1);
      cgbn_set(_env, d._low, c);
      cgbn_rem_wide(_env, r, d, m);
    } else {
      cgbn_rem(_env, r, c, m);
    }
    cgbn_store(_env, &(_stack[_top+2]), r);
    _top=_top+2;
  }

  __device__ __forceinline__ void mulmod() {
    if (_top > params::STACK_SIZE - 3) {
      printf("Stack underflow\n");
      return;
    }
    bn_t  a, b, m, r;
    cgbn_load(_env, a, &(_stack[_top]));
    cgbn_load(_env, b, &(_stack[_top+1]));
    cgbn_load(_env, m, &(_stack[_top+2]));
    bn_wide_t d;
    cgbn_mul_wide(_env, d, a, b);
    cgbn_rem_wide(_env, r, d, m);
    cgbn_store(_env, &(_stack[_top+2]), r);
    _top=_top+2;
  }

  __device__ __forceinline__ void exp() {
    if (_top > params::STACK_SIZE - 2) {
      printf("Stack underflow\n");
      return;
    }
    bn_t  a, b, r;
    cgbn_load(_env, a, &(_stack[_top]));
    cgbn_load(_env, b, &(_stack[_top+1]));
    bn_t current, square;
    int32_t bit, last_bit;
    cgbn_set_ui32(_env, current, 1);
    cgbn_set(_env, square, a);
    last_bit=params::BITS-1-cgbn_clz(_env, b);
    //^0=1 even for 0^0
    if (last_bit == -1) {
      cgbn_set_ui32(_env, r, 1);
    } else {
      for(bit=0;bit<last_bit;bit++) {
        if(cgbn_extract_bits_ui32(_env, b, bit, 1) == 1) {
          cgbn_mul(_env, current, current, square);
        }
        cgbn_mul(_env, square, square, square);
      }
      cgbn_set(_env, r, current);
    }
    cgbn_store(_env, &(_stack[_top+1]), r);
    _top++;
  }

  // TODO: maybe use cgbn_bitwise_mask_ior (more elegant solution and faster)
  __device__ __forceinline__ void signextend() {
    if (_top > params::STACK_SIZE - 2) {
      printf("Stack underflow\n");
      return;
    }
    bn_t  a, b, r;
    cgbn_load(_env, a, &(_stack[_top]));
    cgbn_load(_env, b, &(_stack[_top+1]));
    uint32_t sign;
    uint32_t c;
    c = cgbn_get_ui32(_env, b);
    cgbn_set(_env, r, a);
    sign = cgbn_extract_bits_ui32(_env, a, params::BITS-1 - 8 * (params::BITS/8-c), 1);
    if ((c <= (params::BITS/8 - 1)) && (sign==1)) {
      for(int i=0;i<=params::BITS/8-c;i++) {
        cgbn_insert_bits_ui32(_env, r, r, params::BITS - 8 * i, 8, 0xff);
      }
    }
    cgbn_store(_env, &(_stack[_top+1]), r);
    _top++;
  }


  __device__ __forceinline__ int32_t compare() {
    if (_top > params::STACK_SIZE - 2) {
      printf("Stack underflow\n");
      return;
    }
    bn_t  a, b;
    cgbn_load(_env, a, &(_stack[_top]));
    cgbn_load(_env, b, &(_stack[_top+1]));
    return cgbn_compare(_env, a, b);
  }

  __device__ __forceinline__ void lt() {
    int32_t int_result = compare();
    uint32_t result = (int_result < 0) ? 1 : 0;
    bn_t  r;
    cgbn_set_ui32(_env, r, result);
    cgbn_store(_env, &(_stack[_top+1]), r);
    _top++;
  }

  __device__ __forceinline__ void gt() {
    int32_t int_result = compare();
    uint32_t result = (int_result > 0) ? 1 : 0;
    bn_t  r;
    cgbn_set_ui32(_env, r, result);
    cgbn_store(_env, &(_stack[_top+1]), r);
    _top++;
  }
  
  __device__ __forceinline__ int32_t scompare() {
    if (_top > params::STACK_SIZE - 2) {
      printf("Stack underflow\n");
      return;
    }
    bn_t  a, b;
    cgbn_load(_env, a, &(_stack[_top]));
    cgbn_load(_env, b, &(_stack[_top+1]));
    uint32_t sign_a = cgbn_extract_bits_ui32(_env, a, params::BITS-1, 1);
    uint32_t sign_b = cgbn_extract_bits_ui32(_env, b, params::BITS-1, 1);
    if (sign_a == 0 && sign_b == 1) {
      return 1;
    } else if (sign_a == 1 && sign_b == 0) {
      return -1;
    } else {
      return cgbn_compare(_env, a, b);
    }
  }

  __device__ __forceinline__ void slt() {
    int32_t int_result = scompare();
    uint32_t result = (int_result < 0) ? 1 : 0;
    bn_t  r;
    cgbn_set_ui32(_env, r, result);
    cgbn_store(_env, &(_stack[_top+1]), r);
    _top++;
  }

  __device__ __forceinline__ void sgt() {
    int32_t int_result = scompare();
    uint32_t result = (int_result > 0) ? 1 : 0;
    bn_t  r;
    cgbn_set_ui32(_env, r, result);
    cgbn_store(_env, &(_stack[_top+1]), r);
    _top++;
  }

  __device__ __forceinline__ void eq() {
    int32_t int_result = compare();
    uint32_t result = (int_result == 0) ? 1 : 0;
    bn_t  r;
    cgbn_set_ui32(_env, r, result);
    cgbn_store(_env, &(_stack[_top+1]), r);
    _top++;
  }

  __device__ __forceinline__ void iszero() {
    if (_top > params::STACK_SIZE - 1) {
      printf("Stack underflow\n");
      return;
    }
    bn_t  a, r;
    cgbn_load(_env, a, &(_stack[_top]));
    int32_t compare = cgbn_compare_ui32(_env, a, 0);
    if (compare == 0) {
      cgbn_set_ui32(_env, r, 1);
    } else {
      cgbn_set_ui32(_env, r, 1);
    }
    cgbn_store(_env, &(_stack[_top]), r);
  }

  __device__ __forceinline__ void bitwise_and() {
    if (_top > params::STACK_SIZE - 2) {
      printf("Stack underflow\n");
      return;
    }
    bn_t  a, b, r;
    cgbn_load(_env, a, &(_stack[_top]));
    cgbn_load(_env, b, &(_stack[_top+1]));
    cgbn_bitwise_and(_env, r, a, b);
    cgbn_store(_env, &(_stack[_top+1]), r);
    _top++;
  }

  __device__ __forceinline__ void bitwise_or() {
    if (_top > params::STACK_SIZE - 2) {
      printf("Stack underflow\n");
      return;
    }
    bn_t  a, b, r;
    cgbn_load(_env, a, &(_stack[_top]));
    cgbn_load(_env, b, &(_stack[_top+1]));
    cgbn_bitwise_ior(_env, r, a, b);
    cgbn_store(_env, &(_stack[_top+1]), r);
    _top++;
  }

  __device__ __forceinline__ void bitwise_xor() {
    if (_top > params::STACK_SIZE - 2) {
      printf("Stack underflow\n");
      return;
    }
    bn_t  a, b, r;
    cgbn_load(_env, a, &(_stack[_top]));
    cgbn_load(_env, b, &(_stack[_top+1]));
    cgbn_bitwise_xor(_env, r, a, b);
    cgbn_store(_env, &(_stack[_top+1]), r);
    _top++;
  }

  __device__ __forceinline__ void bitwise_not() {
    if (_top > params::STACK_SIZE - 1) {
      printf("Stack underflow\n");
      return;
    }
    bn_t  a, r;
    cgbn_load(_env, a, &(_stack[_top]));
    cgbn_bitwise_mask_xor(_env, r, a, params::BITS);
    cgbn_store(_env, &(_stack[_top]), r);
  }

  __device__ __forceinline__ void get_byte() {
    if (_top > params::STACK_SIZE - 2) {
      printf("Stack underflow\n");
      return;
    }
    bn_t  a, b, r;
    cgbn_load(_env, a, &(_stack[_top]));
    cgbn_load(_env, b, &(_stack[_top+1]));
    uint32_t index = cgbn_get_ui32(_env, a);
    uint32_t byte = cgbn_extract_bits_ui32(_env, b, 8 * ((params::BITS/8-1)-index), 8);
    cgbn_set_ui32(_env, r, byte);
    cgbn_store(_env, &(_stack[_top+1]), r);
    _top++;
  }

  __device__ __forceinline__ void shl() {
    if (_top > params::STACK_SIZE - 2) {
      printf("Stack underflow\n");
      return;
    }
    bn_t  a, b, r;
    cgbn_load(_env, a, &(_stack[_top]));
    cgbn_load(_env, b, &(_stack[_top+1]));
    // TODO maybe compare with params::BITS
    uint32_t shift = cgbn_get_ui32(_env, a);
    cgbn_shift_left(_env, r, b, shift);
    cgbn_store(_env, &(_stack[_top+1]), r);
    _top++;
  }

  __device__ __forceinline__ void shr() {
    if (_top > params::STACK_SIZE - 2) {
      printf("Stack underflow\n");
      return;
    }
    bn_t  a, b, r;
    cgbn_load(_env, a, &(_stack[_top]));
    cgbn_load(_env, b, &(_stack[_top+1]));
    // TODO maybe compare with params::BITS
    uint32_t shift = cgbn_get_ui32(_env, a);
    cgbn_shift_right(_env, r, b, shift);
    cgbn_store(_env, &(_stack[_top+1]), r);
    _top++;
  }

  __device__ __forceinline__ void sar() {
    if (_top > params::STACK_SIZE - 2) {
      printf("Stack underflow\n");
      return;
    }
    bn_t  a, b, r;
    cgbn_load(_env, a, &(_stack[_top]));
    cgbn_load(_env, b, &(_stack[_top+1]));
    // TODO maybe compare with params::BITS
    uint32_t shift = cgbn_get_ui32(_env, a);
    uint32_t sign_b = cgbn_extract_bits_ui32(_env, b, params::BITS-1, 1);
    cgbn_shift_right(_env, r, b, shift);
    if (sign_b == 1) {
      cgbn_bitwise_mask_ior(_env, r, r, -shift);
    }
    cgbn_store(_env, &(_stack[_top+1]), r);
    _top++;
  }


  __device__ __forceinline__ void pushx(uint8_t *value, uint32_t size) {
    if (_top == 0) {
      printf("Stack overflow\n");
      return;
    }
    if (size > 32) {
      printf("Invalid size\n");
      return;
    }
    bn_t  r;
    cgbn_set_ui32(_env, r, 0);
    for(int i=0;i<size;i++) {
      cgbn_insert_bits_ui32(_env, r, r, params::BITS - 8 * (i+1), 8, value[size-1-i]);
    }
    _top--;
    cgbn_store(_env, _stack + _top, r);
  }
  

  __device__ __forceinline__ void dupx(uint32_t index) {
    if (_top == 0) {
      printf("Stack overflow\n");
      return;
    }
    if (index < 1 || index > 16) {
      printf("Invalid index\n");
      return;
    }
    if (_top > params::STACK_SIZE - index) {
      printf("Stack underflow\n");
      return;
    }
    bn_t  r;
    cgbn_load(_env, r, &(_stack[_top + index - 1]));
    _top--;
    cgbn_store(_env, _stack + _top, r);
  }
  

  __device__ __forceinline__ void swapx(uint32_t index) {
    if (_top == 0) {
      printf("Stack overflow\n");
      return;
    }
    if (index < 1 || index > 16) {
      printf("Invalid index\n");
      return;
    }
    if (_top > params::STACK_SIZE - index) {
      printf("Stack underflow\n");
      return;
    }
    bn_t  a, b;
    cgbn_load(_env, a, &(_stack[_top + index - 1]));
    cgbn_load(_env, b, &(_stack[_top]));
    cgbn_store(_env, &(_stack[_top]), a);
    cgbn_store(_env, &(_stack[_top + index - 1]), b);
  }
  
// support routine to generate instances
  __host__ static instance_t *generate_instances(uint32_t count) {
    instance_t *instances=(instance_t *)malloc(sizeof(instance_t)*count);

    for(int index=0;index<count;index++) {
      for(int i=0;i<params::STACK_SIZE;i++) {
        instances[index].values[i]._limbs[0]=params::STACK_SIZE-i;
        for(int j=1;j<params::BITS/32;j++)
          instances[index].values[i]._limbs[j]=0;
      }
    }
    return instances;
  }

  // support routine to show the stack on CPU
  __host__ static void show_results(instance_t *instances, uint32_t count) {
    
    for(int index=0;index<count;index++) {
      for (int i = 0; i < 10; i++)
      {
        for(int j=0;j<params::BITS/32;j++) {
          printf("%08x ", instances[index].values[i]._limbs[j]);
        }
        printf("\n");
      }
      
      
    }
    printf("All results match\n");
  }
};

#endif