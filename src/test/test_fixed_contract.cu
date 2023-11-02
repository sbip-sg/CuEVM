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
#include "../contract.cuh"
#include "../arith.cuh"

template<class params>
__global__ void kernel_global_storage(cgbn_error_report_t *report, typename gpu_fixed_global_storage_t<params>::gpu_contract_t *contracts, uint32_t no_contracts) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;

  typedef arith_env_t<params> local_arith_t;
  typedef typename arith_env_t<params>::bn_t  bn_t;
  local_arith_t arith(cgbn_report_monitor, report, instance);

  typedef gpu_fixed_global_storage_t<params> local_global_storage_t;
 
  local_global_storage_t  global_storage(arith, contracts, no_contracts);

  bn_t address, key, value;
  uint32_t error;
  error = 0;
  typedef typename gpu_fixed_global_storage_t<params>::gpu_contract_t gpu_contract_t;
  gpu_contract_t *contract;
  cgbn_set_ui32(arith._env, address, 25);
  cgbn_set_ui32(arith._env, key, 5);
  bn_t contract_address, contract_balance;
  
  printf("address: %08x, key: %08x, no_accounts: %lx\n", cgbn_get_ui32(arith._env, address), cgbn_get_ui32(arith._env, key), global_storage._no_accounts);
  contract=global_storage.get_account(address, error);
  printf("error: %08x\n", error);
  cgbn_load(arith._env, contract_address, &(contract->address));
  cgbn_load(arith._env, contract_balance, &(contract->balance));
  printf("address: %08x, balance: %08x, code_size: %lx, storage_size: %lx\n", cgbn_get_ui32(arith._env, contract_address), cgbn_get_ui32(arith._env, contract_balance), contract->code_size, contract->storage_size);
  global_storage.get_value(address, key, value, error);
  printf("error: %08x\n", error);
  printf("address: %08x, key: %08x, value: %08x\n", cgbn_get_ui32(arith._env, address), cgbn_get_ui32(arith._env, key), cgbn_get_ui32(arith._env, value));
}

template<class params>
void run_test(uint32_t instance_count) {
  typedef typename gpu_fixed_global_storage_t<params>::gpu_contract_t gpu_contract_t;
  
  gpu_contract_t          *cpu_instances, *gpu_instances;
  cgbn_error_report_t     *report;
  size_t test;
  mpz_t gmp_test;
  test=1;
  char size_t_string[11]; // '0x' + 8 bytes + '\0'
  mpz_init(gmp_test);
  printf("hex sintrg: 0x%lx", test);
  snprintf (size_t_string, 11, "%lx", test);
  mpz_set_str(gmp_test, size_t_string, 16);
  printf("hex gmp: %s\n", mpz_get_str(NULL, 16, gmp_test));
  printf("value gmp: %lu\n", mpz_get_ui(gmp_test));
  printf("size_t_string: %s\n", size_t_string);
  
  printf("Genereating primes and instances ...\n");
  cpu_instances=gpu_fixed_global_storage_t<params>::generate_global_storage(instance_count);
  
  printf("Copying primes and instances to the GPU ...\n");
  gpu_instances=gpu_fixed_global_storage_t<params>::generate_gpu_global_storage(cpu_instances, instance_count);
  
  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  
  printf("Running GPU kernel ...\n");

  // launch kernel with blocks=ceil(instance_count/IPB) and threads=TPB
  kernel_global_storage<params><<<1, params::TPI>>>(report, gpu_instances, instance_count);

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
    
  // copy the instances back from gpuMemory
  printf("Copying results back to CPU ...\n");
  CUDA_CHECK(cudaMemcpy(cpu_instances, gpu_instances, sizeof(gpu_contract_t)*instance_count, cudaMemcpyDeviceToHost));
  
  printf("Verifying the results ...\n");
  
  // clean up
  gpu_fixed_global_storage_t<params>::free_global_storage(cpu_instances);
  gpu_fixed_global_storage_t<params>::free_gpu_global_storage(gpu_instances);
  CUDA_CHECK(cgbn_error_report_free(report));
}

#define INSTANCES 100


int main() {
  run_test<utils_params>(INSTANCES);
}