
// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#include "../memory.cuh"

template <class params>
__host__ __device__ __forceinline__ void test_memory(
    arith_env_t<params> &arith,
    typename memory_t<params>::memory_data_t *memory_data,
    uint32_t &instance)
{
  typedef arith_env_t<params> arith_t;
  typedef memory_t<params> memory_t;
  typedef typename arith_t::bn_t bn_t;
  memory_t *memory;
  SHARED_MEMORY uint8_t tmp[32];
  memory = new memory_t(arith);

  // printf("Instance %d:  memory size=%d\n", instance, memory.size());

  bn_t a, b, c, gas;
  bn_t index, length;
  uint32_t error_code;
  uint8_t *data;
  error_code = ERR_NONE;
  cgbn_set_ui32(arith._env, gas, 0);
  cgbn_set_ui32(arith._env, a, instance + 200);
  cgbn_set_ui32(arith._env, b, 10);
  cgbn_set_ui32(arith._env, c, 100);
  cgbn_set_ui32(arith._env, index, 30);
  cgbn_set_ui32(arith._env, length, 20);

  arith.memory_from_cgbn(&(tmp[0]), a);
  memory->grow_cost(index, length, gas, error_code);
  printf("error_code=%d gas=%08x\n", error_code, cgbn_get_ui32(arith._env, gas));
  memory->set(&(tmp[0]), index, length, error_code);
  printf("error_code=%d\n", error_code);
  printf("Memory set:\n");
  print_bytes(&(tmp[0]), 32);
  data = memory->get(index, length, error_code);
  printf("error_code=%d\n", error_code);
  printf("Memory get:\n");
  print_bytes(data, 32);
  printf("Memory size=%lu\n", memory->size());
  arith.memory_from_cgbn(&(tmp[0]), b);
  cgbn_set_ui32(arith._env, index, 40);
  cgbn_set_ui32(arith._env, length, 32);
  memory->grow_cost(index, length, gas, error_code);
  printf("error_code=%d gas=%08x\n", error_code, cgbn_get_ui32(arith._env, gas));
  memory->set(&(tmp[0]), index, length, error_code);
  printf("error_code=%d\n", error_code);

  memory->to_memory_data_t(*memory_data);

  delete memory;
  memory = NULL;
}

template <class params>
__global__ void kernel_memory_run(
    cgbn_error_report_t *report,
    typename memory_t<params>::memory_data_t *memory_data,
    uint32_t instance_count)
{

  typedef memory_t<params> memory_t;
  typedef arith_env_t<params> arith_t;
  typedef typename memory_t::memory_data_t memory_data_t;
  typedef typename arith_t::bn_t bn_t;

  uint32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / params::TPI;

  if (instance >= instance_count)
    return;

  // setup arithmetic
  arith_t arith(cgbn_report_monitor, report, instance);

  // test memory
  test_memory(arith, &(memory_data[instance]), instance);
}

template <class params>
void run_test(uint32_t instance_count)
{
  typedef memory_t<params> memory_t;
  typedef arith_env_t<params> arith_t;
  typedef typename memory_t::memory_data_t memory_data_t;

  memory_data_t *cpu_memories;
  arith_t arith(cgbn_report_monitor, 0);
#ifndef ONLY_CPU
  memory_data_t *gpu_memories;
  cgbn_error_report_t *report;
  CUDA_CHECK(cudaDeviceReset());
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 1024));
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 64 * 1024));
#endif

  printf("Generating memories info\n");
  cpu_memories = memory_t::get_cpu_instances(instance_count);
#ifndef ONLY_CPU
  gpu_memories = memory_t::get_gpu_instances_from_cpu_instances(cpu_memories, instance_count);
#endif
  printf("Memories info generated\n");

#ifndef ONLY_CPU
  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report));

  printf("Running GPU RUN kernel ...\n");

  // launch kernel with blocks=ceil(instance_count/IPB) and threads=TPB
  kernel_memory_run<params><<<instance_count, params::TPI>>>(report, gpu_memories, instance_count);

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
  CUDA_CHECK(cgbn_error_report_free(report));

  printf("GPU RUN kernel finished\n");

  // copy the results back to the CPU
  printf("Copying results back to CPU\n");
  memory_t::free_cpu_instances(cpu_memories, instance_count);
  cpu_memories = memory_t::get_cpu_instances_from_gpu_instances(gpu_memories, instance_count);
  printf("Results copied back to CPU\n");

#else
  printf("Running CPU RUN kernel ...\n");
  for (uint32_t instance = 0; instance < instance_count; instance++)
  {
    test_memory(arith, &(cpu_memories[instance]), instance);
  }
  printf("CPU RUN kernel finished\n");
#endif

  // print the results
  printf("Printing the results stdout/json...\n");
  cJSON *root = cJSON_CreateObject();
  cJSON *post = cJSON_CreateArray();
  cJSON *memory_json = NULL;
  for (uint32_t instance = 0; instance < instance_count; instance++)
  {
    cJSON *instance_json = cJSON_CreateObject();
    cJSON_AddItemToArray(post, instance_json);
    cJSON_AddNumberToObject(instance_json, "instance", instance);
    memory_json = memory_t::json_from_memory_data_t(arith, cpu_memories[instance]);
    cJSON_AddItemToObject(instance_json, "memory", memory_json);
    printf("Instance %d:  ", instance);
    memory_t::print_memory_data_t(arith, cpu_memories[instance]);
  }
  cJSON_AddItemToObject(root, "post", post);
  delete[] cpu_memories;
  cpu_memories = NULL;
  char *json_str = cJSON_Print(root);
  FILE *fp = fopen("output/evm_memory.json", "w");
  fprintf(fp, "%s", json_str);
  fclose(fp);
  fp = NULL;
  free(json_str);
  json_str = NULL;
  cJSON_Delete(root);
  root = NULL;
  printf("Results printed\n");
}

int main()
{
  run_test<utils_params>(3);
}