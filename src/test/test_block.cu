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
#include "../block.cuh"


__device__ __constant__ block_t<utils_params>::block_data_t current_block;

template<class params>
__global__ void kernel_block(cgbn_error_report_t *report) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;

  typedef arith_env_t<params> local_arith_t;
  typedef typename arith_env_t<params>::bn_t  bn_t;
  typedef block_t<params> block_t;
  local_arith_t arith(cgbn_report_monitor, report, instance);

  //get the current block
  block_t block(arith, &current_block);
  
  bn_t a, b;
  //current block
  block.get_coin_base(a);
  printf("coin_base: %08x\n", cgbn_get_ui32(arith._env, a));

  block.get_difficulty(a);
  printf("difficulty: %08x\n", cgbn_get_ui32(arith._env, a));

  block.get_gas_limit(a);
  printf("gas_limit: %08x\n", cgbn_get_ui32(arith._env, a));

  block.get_number(a);
  printf("number: %08x\n", cgbn_get_ui32(arith._env, a));

  block.get_time_stamp(a);
  printf("time_stamp: %08x\n", cgbn_get_ui32(arith._env, a));

  block.get_chain_id(a);
  printf("chain_id: %08x\n", cgbn_get_ui32(arith._env, a));

  block.get_base_fee(a);
  printf("base_fee: %08x\n", cgbn_get_ui32(arith._env, a));

  //block hash
  cgbn_set_ui32(arith._env, b, 0);
  uint32_t error_code=0;
  block.get_previous_hash(b, a, error_code);
  printf("previous hash %08x for number %08x, with error %d\n", cgbn_get_ui32(arith._env, a), cgbn_get_ui32(arith._env, b), error_code);

  //block.print();
}

void run_test() {
  typedef block_t<utils_params> block_t;
  typedef typename block_t::block_data_t block_data_t;
  typedef arith_env_t<utils_params> arith_t;
  block_data_t            *gpu_block;
  cgbn_error_report_t     *report;
  arith_t arith(cgbn_report_monitor, 0);
  
  //read the json file with the global state
  cJSON *root = get_json_from_file("input/evm_test.json");
  if(root == NULL) {
    printf("Error: could not read the json file\n");
    exit(1);
  }
  const cJSON *test = NULL;
  test = cJSON_GetObjectItemCaseSensitive(root, "sstoreGas");

  printf("Generating the global block\n");
  block_t cpu_block(arith, test);
  cpu_block.print();
  gpu_block=cpu_block.to_gpu();
  CUDA_CHECK(cudaMemcpyToSymbol(current_block, gpu_block, sizeof(block_data_t)));
  printf("Global block generated\n");

  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  
  printf("Running GPU kernel ...\n");

  kernel_block<utils_params><<<1, utils_params::TPI>>>(report);

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
    
  printf("Printing to json files ...\n");
  cJSON_Delete(root);
  root = cJSON_CreateObject();
  cJSON_AddItemToObject(root, "env", cpu_block.to_json());


  char *json_str=cJSON_Print(root);
  FILE *fp=fopen("output/evm_block.json", "w");
  fprintf(fp, "%s", json_str);
  fclose(fp);
  free(json_str);
  cJSON_Delete(root);
  printf("Json files printed\n");
    
  //read-only memory
  printf("Clean up\n");
  
  // clean up
  cpu_block.free_memory();
  block_t::free_gpu(gpu_block);
  CUDA_CHECK(cgbn_error_report_free(report));
}

int main() {
  run_test();
}