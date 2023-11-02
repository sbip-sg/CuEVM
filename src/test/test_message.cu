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
#include "../utils.h"
#include "../message.cuh"
#include "../arith.cuh"


template<class params>
__global__ void kernel_message(cgbn_error_report_t *report, typename gpu_message<params>::gpu_message *msgs, uint32_t instance_count) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;

  if(instance>=instance_count)
    return;
  typedef arith_env_t<params> local_arith_t;
  typedef typename arith_env_t<params>::bn_t  bn_t;
  local_arith_t arith(cgbn_report_monitor, report, instance);
  typedef typename gpu_global_storage_t<params>::gpu_contract_t gpu_contract_t;
  gpu_contract_t *contract;


  bn_t caller, value, to, tx_origin, tx_gasprice;
  uint32_t error;
  error = 0;
  cgbn_load(arith._env, caller, &(msgs[instance].caller));
  cgbn_load(arith._env, value, &(msgs[instance].value));
  cgbn_load(arith._env, to, &(msgs[instance].to));
  cgbn_load(arith._env, tx_origin, &(msgs[instance].tx.origin));
  cgbn_load(arith._env, tx_gasprice, &(msgs[instance].tx.gasprice));
  contract=msgs[instance].contract;
  printf("caller: %08x, value: %08x, to: %08x, tx_origin: %08x, tx_gasprice: %08x, data_size: %lx\n", cgbn_get_ui32(arith._env, caller), cgbn_get_ui32(arith._env, value), cgbn_get_ui32(arith._env, to), cgbn_get_ui32(arith._env, tx_origin), cgbn_get_ui32(arith._env, tx_gasprice), msgs[instance].data.size);
  bn_t contract_address, contract_balance;
  cgbn_load(arith._env, contract_address, &(contract->address));
  cgbn_load(arith._env, contract_balance, &(contract->balance));
  printf("contract: %p\n", contract);
  printf("address: %08x, balance: %08x, code_size: %lx, storage_size: %lx\n", cgbn_get_ui32(arith._env, contract_address), cgbn_get_ui32(arith._env, contract_balance), contract->code_size, contract->storage_size);
}

template<class params>
void run_test(uint32_t instance_count, uint32_t storage_count) {
  typedef typename gpu_global_storage_t<params>::gpu_contract_t gpu_contract_t;
  typedef typename gpu_message<params>::gpu_message gpu_message_t;
  
  gpu_contract_t          *cpu_global_storage, *gpu_global_storage;
  gpu_message_t           *cpu_messages, *gpu_messages;
  cgbn_error_report_t     *report;
  
  printf("Genereating primes and instances ...\n");
  cpu_global_storage=gpu_global_storage_t<params>::generate_global_storage(storage_count);
  cpu_messages=generate_host_messages<params>(instance_count);
  //write_messages<params>(stdout, cpu_messages, instance_count);
  
  printf("Copying primes and instances to the GPU ...\n");
  gpu_global_storage=gpu_global_storage_t<params>::generate_gpu_global_storage(cpu_global_storage, storage_count);
  for(size_t idx=0; idx<instance_count; idx++) {
    cpu_messages[idx].contract=&(gpu_global_storage[1]);
  }
  gpu_messages=generate_gpu_messages<params>(cpu_messages, instance_count);
  
  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  
  printf("Running GPU kernel ...\n");

  // launch kernel with blocks=ceil(instance_count/IPB) and threads=TPB
  kernel_message<params><<<1, params::TPI>>>(report, gpu_messages, instance_count);

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
    
    
  //read-only memory
  
  printf("Verifying the results ...\n");
  
  // clean up
  gpu_global_storage_t<params>::free_global_storage(cpu_global_storage, storage_count);
  gpu_global_storage_t<params>::free_gpu_global_storage(gpu_global_storage, storage_count);
  free_host_messages<params>(cpu_messages, instance_count);
  free_gpu_messages<params>(gpu_messages, instance_count);
  CUDA_CHECK(cgbn_error_report_free(report));
}

#define INSTANCES 1


int main() {
  run_test<utils_params>(INSTANCES, 100);
}