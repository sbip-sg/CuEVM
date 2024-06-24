// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#include "../tracer.cuh"

template<class params>
__host__ __device__ __forceinline__ void test_tracer(
  arith_env_t<params> &arith,
  typename world_state_t<params>::state_data_t *world_state_data,
  typename tracer_t<params>::tracer_data_t *tracer_data,
  uint32_t &instance
)
{
  typedef world_state_t<params> world_state_t;
  typedef accessed_state_t<params> accessed_state_t;
  typedef touch_state_t<params> touch_state_t;
  typedef arith_env_t<params> ArithEnv;
  typedef typename ArithEnv::bn_t  bn_t;
  typedef typename world_state_t::account_t account_t;
  typedef typename accessed_state_t::accessed_state_data_t accessed_state_data_t;
  typedef tracer_t<params> tracer_t;
  typedef typename tracer_t::tracer_data_t tracer_data_t;
  typedef stack_t<params> stack_t;
  typedef typename stack_t::stack_data_t stack_data_t;
  typedef memory_t<params> memory_t;
  typedef typename memory_t::memory_data_t memory_data_t;

  world_state_t *world_state;
  accessed_state_t *access_state;
  touch_state_t *touch_state;
  tracer_t *tracer;
  stack_t *stack;
  memory_t *memory;

  world_state=new world_state_t(arith, world_state_data);
  access_state=new accessed_state_t(world_state);
  touch_state=new touch_state_t(access_state, NULL);
  tracer=new tracer_t(arith, tracer_data);
  stack=new stack_t(arith);
  memory=new memory_t(arith);
  /*
  */
  bn_t address, key, nonce, value;
  bn_t gas_cost;
  bn_t gas_refund;
  bn_t gas_limit;
  account_t *account;
  bn_t account_address, account_balance, account_nonce;
  uint32_t error;
  uint32_t pc=0;
  uint8_t opcode=0x52;
  error=ERR_SUCCESS;

  
  cgbn_set_ui32(arith._env, gas_cost, 0);
  cgbn_set_ui32(arith._env, gas_refund, 0);
  cgbn_set_ui32(arith._env, gas_limit, 10*(instance+1));
  account=NULL;

  
  cgbn_set_ui32(arith._env, address, 1);
  cgbn_set_ui32(arith._env, key, 1);
  cgbn_set_ui32(arith._env, nonce, 1);
  cgbn_set_ui32(arith._env, value, instance);
  stack->push(value, error);
  stack->push(nonce, error);
  touch_state->set_value(address, key, value);
  tracer->push(
    address,
    pc,
    opcode,
    *stack,
    *memory,
    *touch_state,
    gas_cost,
    gas_limit,
    gas_refund,
    error
  );
  pc++;
  cgbn_set_ui32(arith._env, address, 2);
  cgbn_set_ui32(arith._env, key, 2);
  cgbn_set_ui32(arith._env, nonce, 2);
  cgbn_set_ui32(arith._env, value, 3*(instance+2));
  stack->push(value, error);
  stack->push(nonce, error);
  touch_state->set_value(address, key, value);
  uint8_t tmp[32];
  arith.memory_from_cgbn(tmp, value);
  bn_t index;
  bn_t length;
  cgbn_set_ui32(arith._env, index, 10+instance);
  cgbn_set_ui32(arith._env, length, 32);
  size_t available_size;
  available_size = EVM_WORD_SIZE;
  memory->set(
    &(tmp[0]),
    index,
    length,
    available_size,
    error
  );
  tracer->push(
    address,
    pc,
    opcode,
    *stack,
    *memory,
    *touch_state,
    gas_cost,
    gas_limit,
    gas_refund,
    error
  );

  // free the memory
  delete touch_state;
  touch_state=NULL;
  delete access_state;
  access_state=NULL;
  delete world_state;
  world_state=NULL;
  delete tracer;
  tracer=NULL;
  delete stack;
  stack=NULL;
  delete memory;
  memory=NULL;
}

template<class params>
__global__ void kernel_test_tracer(
  cgbn_error_report_t *report,
  typename world_state_t<params>::state_data_t *world_state_data,
  typename tracer_t<params>::tracer_data_t *tracers_data,
  uint32_t instance_count
) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;

  if(instance >= instance_count)
    return;

  typedef arith_env_t<params> ArithEnv;

  // setup arithmetic
  ArithEnv arith(cgbn_report_monitor, report, instance);

  test_tracer(arith, world_state_data, &(tracers_data[instance]), instance);
}

template<class params>
void run_test(uint32_t instance_count) {
  typedef arith_env_t<params> ArithEnv;
  typedef world_state_t<params> world_state_t;
  typedef typename world_state_t::state_data_t state_data_t;
  typedef tracer_t<params> tracer_t;
  typedef typename tracer_t::tracer_data_t tracer_data_t;

  state_data_t *world_state_data;
  tracer_data_t *cpu_tracers_data;
  #ifndef ONLY_CPU
  tracer_data_t *gpu_tracers_data;
  CUDA_CHECK(cudaDeviceReset());
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024*1024*1024));
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 64*1024));
  cgbn_error_report_t     *report;
  #endif
  ArithEnv arith(cgbn_report_monitor, 0);
  
  //read the json file with the world state
  cJSON *root = get_json_from_file("input/evm_test.json");
  if(root == NULL) {
    printf("Error: could not read the json file\n");
    exit(1);
  }
  const cJSON *test = NULL;
  test = cJSON_GetObjectItemCaseSensitive(root, "mstore8");


  printf("Generating world state\n");
  world_state_t *cpu_world_state;
  cpu_world_state = new world_state_t(arith, test);
  world_state_data = cpu_world_state->_content;
  printf("Global state generated\n");

  printf("Generating tracers\n");
  cpu_tracers_data=tracer_t::get_cpu_instances(instance_count);
  #ifndef ONLY_CPU
  gpu_tracers_data=tracer_t::get_gpu_instances_from_cpu_instances(cpu_tracers_data, instance_count);
  #endif
  printf("access tracers generated\n");

  #ifndef ONLY_CPU
  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  
  printf("Running GPU kernel ...\n");
  // launch kernel with blocks=ceil(instance_count/IPB) and threads=TPB
  kernel_test_tracer<params><<<instance_count, params::TPI>>>(
    report,
    world_state_data,
    gpu_tracers_data,
    instance_count
  );

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
  printf("GPU kernel finished\n");

  printf("Copying the results back to the CPU ...\n");
  tracer_t::free_cpu_instances(cpu_tracers_data, instance_count);
  cpu_tracers_data=tracer_t::get_cpu_instances_from_gpu_instances(gpu_tracers_data, instance_count);
  printf("Results copied\n");
  CUDA_CHECK(cgbn_error_report_free(report));
  #else
  printf("Running CPU kernel ...\n");
  for(uint32_t instance=0; instance<instance_count; instance++)
  {
    test_tracer(arith, world_state_data, &(cpu_tracers_data[instance]), instance);
  }
  printf("CPU kernel finished\n");
  #endif


  // print the results
  printf("Printing the results stdout/json...\n");
  cJSON_Delete(root);
  root = cJSON_CreateObject();
  cJSON *post = cJSON_CreateArray();
  cJSON *tracer_json = NULL;
  for(uint32_t instance=0; instance<instance_count; instance++)
  {
    cJSON *instance_json = cJSON_CreateObject();
    cJSON_AddItemToArray(post, instance_json);
    cJSON_AddNumberToObject(instance_json, "instance", instance);
    printf("Tracer %d:\n", instance);
    tracer_t::print_tracer_data_t(arith, cpu_tracers_data[instance]);
    tracer_json = tracer_t::json_from_tracer_data_t(arith, cpu_tracers_data[instance]);
    cJSON_AddItemToObject(instance_json, "tracer", tracer_json);
  }
  cJSON_AddItemToObject(root, "post", post);
  cpu_world_state->free_content();
  delete cpu_world_state;
  cpu_world_state=NULL;
  tracer_t::free_cpu_instances(cpu_tracers_data, instance_count);
  cpu_tracers_data=NULL;
  char *json_str=cJSON_Print(root);
  FILE *fp=fopen("output/evm_tracer.json", "w");
  fprintf(fp, "%s", json_str);
  fclose(fp);
  fp=NULL;
  free(json_str);
  json_str=NULL;
  cJSON_Delete(root);
  root=NULL;
  printf("Results printed\n");
  #ifndef ONLY_CPU
  CUDA_CHECK(cudaDeviceReset());
  #endif
  
}

#define INSTANCES 2


int main() {
  run_test<utils_params>(INSTANCES);
}