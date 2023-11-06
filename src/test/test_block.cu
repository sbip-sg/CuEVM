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
#include "../arith.cuh"


typedef typename block_t<utils_params>::block_t block_t;

__device__ __constant__ block_t current_block;

template<class params>
__global__ void kernel_block(cgbn_error_report_t *report) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;

  typedef arith_env_t<params> local_arith_t;
  typedef typename arith_env_t<params>::bn_t  bn_t;
  local_arith_t arith(cgbn_report_monitor, report, instance);
  
  bn_t a;
  //current block
  cgbn_load(arith._env, a, &(gpu_current_block.coin_base));
  printf("coin_base: %08x\n", cgbn_get_ui32(arith._env, a));
  cgbn_load(arith._env, a, &(gpu_current_block.time_stamp));
  printf("time_stamp: %08x\n", cgbn_get_ui32(arith._env, a));
  cgbn_load(arith._env, a, &(gpu_current_block.number));
  printf("number: %08x\n", cgbn_get_ui32(arith._env, a));
  cgbn_load(arith._env, a, &(gpu_current_block.difficulty));
  printf("difficulty: %08x\n", cgbn_get_ui32(arith._env, a));
  cgbn_load(arith._env, a, &(gpu_current_block.gas_limit));
  printf("gas_limit: %08x\n", cgbn_get_ui32(arith._env, a));
  cgbn_load(arith._env, a, &(gpu_current_block.chain_id));
  printf("chain_id: %08x\n", cgbn_get_ui32(arith._env, a));
  cgbn_load(arith._env, a, &(gpu_current_block.base_fee));
  printf("base_fee: %08x\n", cgbn_get_ui32(arith._env, a));

  //block hash
  cgbn_load(arith._env, a, &(cpu_block.previous_blocks[0].number));
  printf("BH number: %08x\n", cgbn_get_ui32(arith._env, a));
  cgbn_load(arith._env, a, &(cpu_block.previous_blocks[0].hash));
  printf("BH hash: %08x\n", cgbn_get_ui32(arith._env, a));
}

void run_test() {  
  block_t          *cpu_block;
  cgbn_error_report_t     *report;
  
  //read the json file with the global state
  cJSON *root = get_json_from_file("input/evm_test.json");
  if(root == NULL) {
    printf("Error: could not read the json file\n");
    exit(1);
  }
  const cJSON *test = NULL;
  test = cJSON_GetObjectItemCaseSensitive(root, "sstoreGas");

  printf("Generating the global block\n");
  cpu_block=cpu_block_from_json<utils_params>(test);
  print_block<utils_params>(cpu_block);
  CUDA_CHECK(cudaMemcpyToSymbol(current_block, cpu_block, sizeof(block_t)));
  printf("Global block generated\n");

  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  
  printf("Running GPU kernel ...\n");

  kernel_block<utils_params><<<1, utils_params::TPI>>>(report);

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
    
  printf("Printing to json files ...\n");
  cJSON *root = cJSON_CreateObject();
  cJSON_AddItemToObject(root, "env", block_to_json(cpu_block));


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
  free_host_block(cpu_block);
  CUDA_CHECK(cgbn_error_report_free(report));
}

int main() {
  run_test();
}