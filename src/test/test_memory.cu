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
#include "../memory.cuh"
#include "../arith.cuh"


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

typedef typename gpu_memory_t::memory_data_t memory_data_t;
 
template<class params>
__global__ void kernel_memory_run(cgbn_error_report_t *report, memory_data_t *instances, uint32_t instance_count) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;
  __shared__ memory_data_t memory_data;
  __shared__ uint8_t       tmp[params::BITS/8];
  memory_data._size=0;
  memory_data._data=NULL;
  
  if(instance>=instance_count)
    return;
 
  typedef arith_env_t<params> local_arith_t;
  typedef typename arith_env_t<params>::bn_t  bn_t;
  local_arith_t arith(cgbn_report_monitor, report, instance);

  gpu_memory_t  memory(&memory_data);
  
  //printf("Instance %d:  memory size=%d\n", instance, memory.size());
  
  bn_t          a, b, c;
  cgbn_set_ui32(arith._env, a, 1);
  cgbn_set_ui32(arith._env, b, 10);
  cgbn_set_ui32(arith._env, c, 100);

  arith.from_cgbn_to_memory(&(tmp[0]), a);
  memory.set(0, 32, &(tmp[0]));
  arith.from_cgbn_to_memory(&(tmp[0]), b);
  memory.set(32, 32, &(tmp[0]));
  arith.from_memory_to_cgbn(c, memory.get(0, 32));
  printf("RUN S CGBN lowestbit=%08x\n", cgbn_get_ui32(arith._env, c));
  __syncthreads();
  memory.copy_content_info_to(&(instances[instance]));
  __syncthreads();
  if (threadIdx.x == 0) {
    printf("RUN S data address=%08x\n", memory._content->_data);
    printf("RUN S lowestbit=%02x\n", memory._content->_data[31]);
    printf("RUN D data address=%08x\n", instances[instance]._data);
    printf("RUN D lowestbit=%02x\n", instances[instance]._data[31]);
  }
}

template<class params>
void run_test(uint32_t instance_count) {
  
  memory_data_t           *cpu_instances, *gpu_instances;
  cgbn_error_report_t     *report;
  int32_t                 TPB=(params::TPB==0) ? 8 : params::TPB;
  int32_t                 TPI=params::TPI, IPB=TPB/TPI;
  
  printf("Genereating content info instances ...\n");
  cpu_instances=gpu_memory_t::generate_memory_info_data(instance_count);
  
  printf("Copying the info instaces on the GPU ...\n");
  CUDA_CHECK(cudaSetDevice(0));
  gpu_instances=gpu_memory_t::generate_gpu_memory_info_data(cpu_instances, instance_count);
  
  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 

  printf("Running GPU RUN kernel ...\n");

  // launch kernel with blocks=ceil(instance_count/IPB) and threads=TPB
  kernel_memory_run<params><<<(instance_count+IPB-1)/IPB, TPB>>>(report, gpu_instances, instance_count);

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
  
  get_memory_from_gpu(cpu_instances, gpu_instances, instance_count);

  printf("Free the error report\n");
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