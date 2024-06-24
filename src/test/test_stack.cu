// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#include "../stack.cuh"

template <class params>
__host__ __device__ __forceinline__ void test_stack(
    arith_env_t<params> &arith,
    typename stack_t<params>::stack_data_t *stack_data,
    uint32_t &instance)
{
  typedef arith_env_t<params> ArithEnv;
  typedef typename ArithEnv::bn_t bn_t;
  typedef cgbn_mem_t<params::BITS> evm_word_t;
  typedef stack_t<params> stack_t;
  typedef typename stack_t::stack_data_t stack_data_t;

  stack_t *stack;
  stack = new stack_t(arith);
  bn_t a, b, c;
  uint32_t error_code;

  cgbn_set_ui32(arith._env, a, instance);
  cgbn_set_ui32(arith._env, b, 0xFF);
  stack->push(b, error_code);
  stack->push(a, error_code);
  printf("Stack before pop: \n");
  stack->print();
  stack->pop(c, error_code);
  printf("Stack after pop: \n");
  stack->print();
  stack->push(c, error_code);
  printf("Stack after push: \n");
  stack->print();
  printf("stack->size(): %d\n", stack->size());

  stack->dupx(1, error_code);
  printf("Stack after dupx: \n");
  stack->print();
  printf("stack->size(): %d\n", stack->size());

  stack->swapx(2, error_code);
  printf("Stack after swapx: \n");
  stack->print();
  printf("stack->size(): %d\n", stack->size());

  SHARED_MEMORY uint8_t tmp[32];
  arith.memory_from_cgbn(&(tmp[0]), a);

  stack->pushx(32, error_code, &(tmp[0]), 32);
  printf("Stack after pushx: \n");
  stack->print();
  printf("stack->size(): %d\n", stack->size());


  stack->to_stack_data_t(*stack_data);
  delete stack;
  stack = NULL;
}

template <class params>
__global__ void kernel_stack(
    cgbn_error_report_t *report,
    typename stack_t<params>::stack_data_t *stacks_data,
    uint32_t count)
{
  typedef arith_env_t<params> ArithEnv;

  uint32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / params::TPI;

  if (instance >= count)
    return;

  // setup arithmetic
  ArithEnv arith(cgbn_report_monitor, report, instance);

  test_stack(arith, &(stacks_data[instance]), instance);
}

template <class params>
void run_test(uint32_t instance_count)
{
  typedef arith_env_t<params> ArithEnv;
  typedef stack_t<params> stack_t;
  typedef typename stack_t::stack_data_t stack_data_t;
  typedef cgbn_mem_t<params::BITS> evm_word_t;

  stack_data_t *cpu_stacks;
  ArithEnv arith(cgbn_report_monitor, 0);

#ifndef ONLY_CPU
  CUDA_CHECK(cudaDeviceReset());
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 1024));
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 64 * 1024));
  cgbn_error_report_t *report;
  stack_data_t *gpu_stacks;
#endif

  printf("Geenerating stack data ...\n");
  cpu_stacks = stack_t::get_cpu_instances(instance_count);
  printf("Stack data generated\n");

#ifndef ONLY_CPU
  printf("Copying stack data to the GPU ...\n");
  gpu_stacks = stack_t::get_gpu_instances_from_cpu_instances(cpu_stacks, instance_count);

  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report));

  printf("Running GPU kernel ...\n");

  kernel_stack<params><<<instance_count, params::TPI>>>(report, gpu_stacks, instance_count);

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
  CUDA_CHECK(cgbn_error_report_free(report));
  printf("GPU kernel finished\n");

  // copy the instances back from gpuMemory
  printf("Copying results back to CPU ...\n");
  stack_t::free_cpu_instances(cpu_stacks, instance_count);
  cpu_stacks = stack_t::get_cpu_instances_from_gpu_instances(gpu_stacks, instance_count);
#else
  printf("Running CPU kernel ...\n");
  for (uint32_t instance = 0; instance < instance_count; instance++)
  {
    test_stack(arith, &(cpu_stacks[instance]), instance);
  }
  printf("CPU kernel finished\n");
#endif

  // print the results
  printf("Printing the results stdout/json...\n");
  cJSON *root = cJSON_CreateObject();
  cJSON *post = cJSON_CreateArray();
  cJSON *stack_json = NULL;
  for (uint32_t instance = 0; instance < instance_count; instance++)
  {
    
    printf("Instance %d:  \n", instance);
    stack_t::print_stack_data_t(arith, cpu_stacks[instance]);
    stack_json = stack_t::json_from_stack_data_t(arith, cpu_stacks[instance]);
    cJSON_AddItemToArray(post, stack_json);
  }
  stack_t::free_cpu_instances(cpu_stacks, instance_count);
  cJSON_AddItemToObject(root, "post", post);
  char *json_str = cJSON_Print(root);
  FILE *fp = fopen("output/evm_stack.json", "w");
  fprintf(fp, "%s", json_str);
  fclose(fp);
  free(json_str);
  json_str = NULL;
  cJSON_Delete(root);
  root = NULL;
  printf("Results printed\n");

  // clean up
  printf("Freeing stack data on CPU ...\n");
#ifndef ONLY_CPU
  CUDA_CHECK(cudaDeviceReset());
#endif
}

int main()
{
  run_test<utils_params>(2);
}