/***

Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

***/


#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <gmp.h>
#ifndef __CGBN_H__
#define __CGBN_H__
#include <cgbn/cgbn.h>
#endif
#include "../../CGBN/samples/utility/cpu_support.h"
#include "../../CGBN/samples/utility/cpu_simple_bn_math.h"
#include "../../CGBN/samples/utility/gpu_support.h"
#include "../stack.cuh"

 
template<uint32_t tpi, uint32_t bits, uint32_t window_bits, uint32_t stack_size>
class mr_params_t {
  public:
  // parameters used by the CGBN context
  static const uint32_t TPB=0;                     // get TPB from blockDim.x  
  static const uint32_t MAX_ROTATION=4;            // good default value
  static const uint32_t SHM_LIMIT=0;               // no shared mem available
  static const bool     CONSTANT_TIME=false;       // constant time implementations aren't available yet

  // parameters used locally in the application
  static const uint32_t TPI=tpi;                   // threads per instance
  static const uint32_t BITS=bits;                 // instance size
  static const uint32_t WINDOW_BITS=window_bits;   // window size
  static const uint32_t STACK_SIZE=stack_size;
};

template<class params>
__global__ void kernel_stack(cgbn_error_report_t *report, typename gpu_stack_t<params>::instance_t *instances, uint32_t instance_count) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;
  
  if(instance>=instance_count)
    return;

  typedef gpu_stack_t<params> local_stack_t;
 
  local_stack_t                     stack(cgbn_report_monitor, report, instance, &(instances[instance].values[0]), 0);
  
  // some test operations
  //stack.negate();
  stack.add();
  stack.sub();
  stack.mul();
  //stack.div();
  stack.sdiv();

  //push pop some values
  //typename local_stack_t::bn_t a, b;
  typedef cgbn_context_t<params::TPI, params>    context_t;
  typedef cgbn_env_t<context_t, params::BITS>    env_t;
  typedef typename env_t::cgbn_t                 bn_t;
  bn_t a, b;
  stack.pop(a);
  stack.pop(b);
  stack.push(a);

  cgbn_set_ui32(stack._env, a, 1);
  stack.push(a);
  cgbn_set_ui32(stack._env, a, 0xff);
  stack.push(a);
  stack.signextend();
  
}

template<class params>
void run_test(uint32_t instance_count) {
  typedef typename gpu_stack_t<params>::instance_t instance_t;
  
  instance_t          *instances, *gpuInstances;
  cgbn_error_report_t *report;
  int32_t              TPB=(params::TPB==0) ? 8 : params::TPB;
  int32_t              TPI=params::TPI, IPB=TPB/TPI;
  
  printf("Genereating primes and instances ...\n");
  instances=gpu_stack_t<params>::generate_instances(instance_count);
  
  printf("Copying primes and instances to the GPU ...\n");
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaMalloc((void **)&gpuInstances, sizeof(instance_t)*instance_count));
  CUDA_CHECK(cudaMemcpy(gpuInstances, instances, sizeof(instance_t)*instance_count, cudaMemcpyHostToDevice));
  
  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  
  printf("Running GPU kernel ...\n");

  // launch kernel with blocks=ceil(instance_count/IPB) and threads=TPB
  kernel_stack<params><<<(instance_count+IPB-1)/IPB, TPB>>>(report, gpuInstances, instance_count);

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
    
  // copy the instances back from gpuMemory
  printf("Copying results back to CPU ...\n");
  CUDA_CHECK(cudaMemcpy(instances, gpuInstances, sizeof(instance_t)*instance_count, cudaMemcpyDeviceToHost));
  
  printf("Verifying the results ...\n");
  gpu_stack_t<params>::show_results(instances, instance_count);
  
  // clean up
  free(instances);
  CUDA_CHECK(cudaFree(gpuInstances));
  CUDA_CHECK(cgbn_error_report_free(report));
}

#define INSTANCES 1
#define STACK_SIZE 1024

// IMPORTANT:  DO NOT DEFINE TPI OR BITS BEFORE INCLUDING CGBN
#define TPI 8
#define BITS 256
#define INSTANCES 1
#define STACK_SIZE 1024


int main() {
  typedef mr_params_t<TPI, BITS, 1, STACK_SIZE> params;
  
  run_test<params>(INSTANCES);
}