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

__device__ state_t<utils_params>::state_data_t global_state;

template<class params>
__global__ void kernel_storage(cgbn_error_report_t *report, typename state_t<params>::state_data_t *local_states, uint32_t instance_count) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;

  if(instance >= instance_count)
    return;

  typedef arith_env_t<params> arith_t;
  typedef typename arith_t::bn_t  bn_t;
  typedef state_t<params> state_t;
  typedef typename state_t::contract_t contract_t;

  // setup arithmetic
  arith_t arith(cgbn_report_monitor, report, instance);

  // setup the global and local state
  state_t  global(arith, &global_state);
  state_t  local(arith, &(local_states[instance]));

  bn_t address, key, nonce, value;
  uint32_t error;
  error = 0;
  contract_t *contract;
  cgbn_set_ui32(arith._env, address, 1);
  cgbn_set_ui32(arith._env, key, 1);
  bn_t contract_address, contract_balance, contract_nonce;

  // 1. test the global state
  printf("address: %08x, key: %08x, no_accounts: %lx\n", cgbn_get_ui32(arith._env, address), cgbn_get_ui32(arith._env, key), global._content->no_contracts);
  // 1.1 get the contract
  contract=global.get_account(address, error);
  printf("error: %08x\n", error);
  // 1.2 load the contract address and balance
  cgbn_load(arith._env, contract_address, &(contract->address));
  cgbn_load(arith._env, contract_balance, &(contract->balance));
  cgbn_load(arith._env, contract_nonce, &(contract->nonce));
  printf("address: %08x, balance: %08x, nonce: %08x, code_size: %lx, storage_size: %lx\n", cgbn_get_ui32(arith._env, contract_address), cgbn_get_ui32(arith._env, contract_balance), cgbn_get_ui32(arith._env, contract_nonce),contract->code_size, contract->storage_size);
  // 1.3 get the value
  global.get_value(address, key, value, error);
  printf("error: %08x\n", error);
  printf("address: %08x, key: %08x, value: %08x\n", cgbn_get_ui32(arith._env, address), cgbn_get_ui32(arith._env, key), cgbn_get_ui32(arith._env, value));

  // 2. test the local state
  printf("address: %08x, key: %08x, no_accounts: %lx\n", cgbn_get_ui32(arith._env, address), cgbn_get_ui32(arith._env, key), local._content->no_contracts);
  // 2.1 get the contract
  contract=local.get_account(address, error);
  printf("error: %08x\n", error);
  error = ERR_SUCCESS;
  // 2.2 load the contract from global state and set it on local state
  contract=global.get_account(address, error);
  printf("error: %08x\n", error);
  local.set_local_account(address, contract, error);
  // 2.3 load the contract address and balance
  contract=local.get_account(address, error);
  printf("error: %08x\n", error);
  cgbn_load(arith._env, contract_address, &(contract->address));
  cgbn_load(arith._env, contract_balance, &(contract->balance));
  cgbn_load(arith._env, contract_nonce, &(contract->nonce));
  printf("address: %08x, balance: %08x, nonce: %08x, code_size: %lx, storage_size: %lx\n", cgbn_get_ui32(arith._env, contract_address), cgbn_get_ui32(arith._env, contract_balance), cgbn_get_ui32(arith._env, contract_nonce),contract->code_size, contract->storage_size);
  // 2.4 get the value
  local.get_value(address, key, value, error);
  printf("error: %08x\n", error);
  error = ERR_SUCCESS;
  // 2.5 get the value from global state and set it on local state
  global.get_value(address, key, value, error);
  printf("error: %08x\n", error);
  local.set_value(address, key, value, error);
  // 2.6 get the value
  local.get_value(address, key, value, error);
  printf("error: %08x\n", error);
  printf("address: %08x, key: %08x, value: %08x\n", cgbn_get_ui32(arith._env, address), cgbn_get_ui32(arith._env, key), cgbn_get_ui32(arith._env, value));
  // 2.7 modify the value and set it on local state
  cgbn_set_ui32(arith._env, value, 4);
  local.set_value(address, key, value, error);
  // 2.8 get the value
  local.get_value(address, key, value, error);
  printf("error: %08x\n", error);
  printf("address: %08x, key: %08x, value: %08x\n", cgbn_get_ui32(arith._env, address), cgbn_get_ui32(arith._env, key), cgbn_get_ui32(arith._env, value));
  // 2.9 add a new key-value in the local state
  cgbn_set_ui32(arith._env, key, 2);
  cgbn_set_ui32(arith._env, value, 3);
  local.set_value(address, key, value, error);
  // 2.10 get the value
  local.get_value(address, key, value, error);
  printf("error: %08x\n", error);
  printf("address: %08x, key: %08x, value: %08x\n", cgbn_get_ui32(arith._env, address), cgbn_get_ui32(arith._env, key), cgbn_get_ui32(arith._env, value));
  // 2.11 set a new bytecode in the local state
  size_t code_size=10;
  uint8_t *code=(uint8_t *)malloc(code_size);
  for(int i=0; i<code_size; i++)
    code[i]=i;
  local.set_local_bytecode(address, code, code_size, error);
  // 2.12 get the code
  contract=local.get_account(address, error);
  cgbn_load(arith._env, contract_address, &(contract->address));
  cgbn_load(arith._env, contract_balance, &(contract->balance));
  cgbn_load(arith._env, contract_nonce, &(contract->nonce));
  printf("error: %08x\n", error);
  printf("address: %08x, modfied_bytecode %lx, code_size: %lx, code:\n", cgbn_get_ui32(arith._env, contract_address), contract->modfied_bytecode, contract->code_size);
  for(int i=0; i<contract->code_size; i++)
    printf("%02x ", contract->bytecode[i]);
  printf("\n");
}

template<class params>
void run_test(uint32_t instance_count) {
  typedef typename state_t<params>::state_data_t state_data_t;
  
  state_data_t            *cpu_global_state, *gpu_global_state;
  state_data_t            *cpu_local_states, *gpu_local_states;
  cgbn_error_report_t     *report;
  
  //read the json file with the global state
  cJSON *root = get_json_from_file("input/evm_test.json");
  if(root == NULL) {
    printf("Error: could not read the json file\n");
    exit(1);
  }
  const cJSON *test = NULL;
  test = cJSON_GetObjectItemCaseSensitive(root, "sstoreGas");


  printf("Generating global state\n");
  cpu_global_state=state_t<params>::from_json(test);
  gpu_global_state=state_t<params>::get_gpu_world_data(cpu_global_state);
  CUDA_CHECK(cudaMemcpyToSymbol(global_state, gpu_global_state, sizeof(state_data_t)));
  printf("Global state generated\n");

  printf("Generating local states\n");
  cpu_local_states=state_t<params>::generate_local_states(instance_count);
  gpu_local_states=state_t<params>::get_gpu_local_states(cpu_local_states, instance_count);
  printf("Local states generated\n");
  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  
  printf("Running GPU kernel ...\n");

  // launch kernel with blocks=ceil(instance_count/IPB) and threads=TPB
  // TODO: modify with instance_count
  kernel_storage<params><<<1, params::TPI>>>(report, gpu_local_states, instance_count);

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);

  // copy the results back to the CPU
  state_t<params>::free_local_states(cpu_local_states, instance_count);
  cpu_local_states=state_t<params>::get_local_states_from_gpu(gpu_local_states, instance_count);

  // print the results
  printf("Printing the results ...\n");
  state_t<params>::print_local_states(cpu_local_states, instance_count);
  state_t<params>::print_world_data(cpu_global_state);
  printf("Results printed\n");

  // free the memory
  printf("Freeing the memory ...\n");
  state_t<params>::free_local_states(cpu_local_states, instance_count);
  state_t<params>::free_world_data(cpu_global_state);
  state_t<params>::free_gpu_world_data(gpu_global_state);
  cJSON_Delete(root);
  CUDA_CHECK(cgbn_error_report_free(report));
  
}

#define INSTANCES 2


int main() {
  run_test<utils_params>(INSTANCES);
}