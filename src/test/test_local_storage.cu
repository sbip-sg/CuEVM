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
#include "../storage.cuh"
#include "../arith.cuh"
#include "../utils.h"
 
template<class params>
__global__ void kernel_local_storage(cgbn_error_report_t *report, typename gpu_fixed_storage_t<params>::fixed_storage_data_t *instances, uint32_t instance_count) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;
  
  if(instance>=instance_count)
    return;
 
  typedef arith_env_t<params> local_arith_t;
  typedef typename arith_env_t<params>::bn_t  bn_t;
  local_arith_t arith(cgbn_report_monitor, report, instance);

  typedef gpu_fixed_storage_t<params> local_storage_t;

  local_storage_t  local_storage(arith, &(instances[instance]));

  bn_t          a, b, c;
  cgbn_set_ui32(arith._env, a, 1);
  cgbn_set_ui32(arith._env, b, instance);
  cgbn_set_ui32(arith._env, c, 100);
  bn_t          address, key, value;
  cgbn_set_ui32(arith._env, address, 25);
  cgbn_set_ui32(arith._env, key, 5);
  local_storage.set(address, key, a);
  cgbn_set_ui32(arith._env, key, 6);
  local_storage.set(address, key, b);
  cgbn_set_ui32(arith._env, address, 30);
  local_storage.set(address, key, c);
  cgbn_set_ui32(arith._env, address, 25);
  local_storage.get(address, key, value);

  printf("instace %d: %d\n", instance, cgbn_get_ui32(arith._env, value));
}

template<class params>
void run_test(uint32_t instance_count) {
  typedef typename gpu_fixed_storage_t<params>::fixed_storage_data_t storage_data_t;
  storage_data_t           *cpu_instances, *gpu_instances;
  cgbn_error_report_t     *report;
  
  printf("Genereating content info instances ...\n");
  cpu_instances=gpu_fixed_storage_t<params>::generate_storage_data(instance_count);
  
  printf("Copying the info instaces on the GPU ...\n");
  gpu_instances=gpu_fixed_storage_t<params>::generate_gpu_storage_data(cpu_instances, instance_count);
  
  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 

  printf("Running GPU RUN kernel ...\n");

  // launch kernel with blocks=ceil(instance_count/IPB) and threads=TPB
  kernel_local_storage<params><<<2, params::TPI>>>(report, gpu_instances, instance_count);

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
  printf("show storage\n");
  CUDA_CHECK(cudaMemcpy(cpu_instances, gpu_instances, sizeof(storage_data_t) * instance_count, cudaMemcpyDeviceToHost));
  gpu_fixed_storage_t<params>::write_storage(stdout, cpu_instances, instance_count);
  
  printf("clean up\n");
  //get_memory_from_gpu(cpu_instances, gpu_instances, instance_count);
  gpu_fixed_storage_t<params>::free_storage_data(cpu_instances, instance_count);
  gpu_fixed_storage_t<params>::free_gpu_storage_data(gpu_instances, instance_count);
  CUDA_CHECK(cgbn_error_report_free(report));
}


int main() {
  run_test<utils_params>(2);
}