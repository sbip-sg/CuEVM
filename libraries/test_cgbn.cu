// test cgbn_add

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <CGBN/cgbn.h>
 
// IMPORTANT:  DO NOT DEFINE TPI OR BITS BEFORE INCLUDING CGBN
#define TPI 32
#define BITS 256

// helpful typedefs for the kernel
typedef cgbn_context_t<TPI>         context_t;
typedef cgbn_env_t<context_t, BITS> env_t;


int main() {
  context_t      bn_context(cgbn_report_monitor); 
  env_t          bn_env(bn_context.env<env_t>());                     // construct an environment for 1024-bit math
  env_t::cgbn_t  a, b, r;

    cgbn_set_ui32(bn_env, a, 0x1);
    cgbn_set_ui32(bn_env, b, 0x3);
    cgbn_add(bn_env, r, a, b);
    printf("r = %x\n", cgbn_get_ui32(bn_env, r));
    return 0;
}