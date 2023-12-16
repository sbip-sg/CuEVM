
// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#include "../utils.h"
#include "../block.cuh"

template <class params>
__host__ __device__ __forceinline__ void test_block(
    arith_env_t<params> &arith,
    typename block_t<params>::block_data_t *current_block_data,
    uint32_t &instance)
{
  typedef block_t<params> block_t;
  typedef arith_env_t<params> arith_t;
  typedef typename arith_t::bn_t bn_t;
  // get the current block
  block_t *block;
  block = new block_t(arith, current_block_data);

  bn_t a, b;
  // current block
  block->get_coin_base(a);
  printf("coin_base: %08x\n", cgbn_get_ui32(arith._env, a));

  block->get_difficulty(a);
  printf("difficulty: %08x\n", cgbn_get_ui32(arith._env, a));

  block->get_gas_limit(a);
  printf("gas_limit: %08x\n", cgbn_get_ui32(arith._env, a));

  block->get_number(a);
  printf("number: %08x\n", cgbn_get_ui32(arith._env, a));

  block->get_time_stamp(a);
  printf("time_stamp: %08x\n", cgbn_get_ui32(arith._env, a));

  block->get_chain_id(a);
  printf("chain_id: %08x\n", cgbn_get_ui32(arith._env, a));

  block->get_base_fee(a);
  printf("base_fee: %08x\n", cgbn_get_ui32(arith._env, a));

  // block hash
  cgbn_set_ui32(arith._env, b, 0);
  uint32_t error_code = 0;
  block->get_previous_hash(a, b, error_code);
  printf("previous hash %08x for number %08x, with error %d\n", cgbn_get_ui32(arith._env, a), cgbn_get_ui32(arith._env, b), error_code);

  // block->print();
  delete block;
  block = NULL;
  printf("Block deleted\n");
}

template <class params>
__global__ void kernel_block(
    cgbn_error_report_t *report,
    typename block_t<params>::block_data_t *current_block_data)
{
  uint32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / params::TPI;

  typedef arith_env_t<params> arith_t;

  arith_t arith(cgbn_report_monitor, report, instance);
  
  printf("kernel %p\n", current_block_data);

  test_block(arith, current_block_data, instance);
}

void run_test()
{
  #ifndef ONLY_CPU
  CUDA_CHECK(cudaDeviceReset());
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024*1024*1024));
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 64*1024));
  #endif

  typedef block_t<utils_params> block_t;
  typedef typename block_t::block_data_t block_data_t;
  typedef arith_env_t<utils_params> arith_t;
  block_data_t *block_data = NULL;
  block_t *cpu_block = NULL;
  arith_t arith(cgbn_report_monitor, 0);


  // read the json file with the global state
  cJSON *root = get_json_from_file("input/evm_test.json");
  if (root == NULL)
  {
    printf("Error: could not read the json file\n");
    exit(1);
  }
  const cJSON *test = NULL;
  test = cJSON_GetObjectItemCaseSensitive(root, "mstore8");

  printf("Generating the global block\n");
  cpu_block = new block_t(arith, test);
  block_data = cpu_block->_content;
  printf("Global block generated\n");
  cpu_block->print();

// create a cgbn_error_report for CGBN to report back errors
#ifndef ONLY_CPU
  cgbn_error_report_t *report;
  CUDA_CHECK(cgbn_error_report_alloc(&report));

  printf("Running GPU kernel ...\n");

  kernel_block<utils_params><<<1, utils_params::TPI>>>(report, block_data);

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("GPU kernel finished\n");
  CGBN_CHECK(report);
  // clean up
  CUDA_CHECK(cgbn_error_report_free(report));

#else

  printf("Running CPU kernel ...\n");
  uint32_t instance = 0;
  test_block<utils_params>(arith, block_data, instance);
  printf("CPU kernel finished\n");

#endif

  printf("Printing  ...\n");
  cpu_block->print();
  printf("Printed\n");

  printf("Printing to json files ...\n");
  cJSON_Delete(root);
  root = cJSON_CreateObject();
  cJSON_AddItemToObject(root, "env", cpu_block->json());

  char *json_str = cJSON_Print(root);
  FILE *fp = fopen("output/evm_block.json", "w");
  fprintf(fp, "%s", json_str);
  fclose(fp);
  free(json_str);
  cJSON_Delete(root);
  printf("Json files printed\n");

  cpu_block->free_content();
  delete cpu_block;
  cpu_block = NULL;
  #ifndef ONLY_CPU
  CUDA_CHECK(cudaDeviceReset());
  #endif
}

int main()
{
  run_test();
}