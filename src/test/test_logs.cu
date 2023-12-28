// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#include "../logs.cuh"

template<class params>
__host__ __device__ __forceinline__ void test_logs(
  arith_env_t<params> &arith,
  typename log_state_t<params>::log_state_data_t *log_state_data,
  uint32_t &instance
)
{
  typedef arith_env_t<params> arith_t;
  typedef typename arith_t::bn_t  bn_t;
  typedef log_state_t<params> log_state_t;
  typedef typename log_state_t::log_state_data_t log_state_data_t;
  typedef typename log_state_t::log_data_t log_data_t;

  log_state_t *log_state;
  log_state_t *parent_1_log_state;

  log_state=new log_state_t(arith);
  parent_1_log_state=new log_state_t(arith);
  
  bn_t address, topic_1, topic_2, topic_3, topic_4;
  SHARED_MEMORY data_content_t record;
  uint32_t no_topics;

  //test 1
  printf("Test %u 1\n", instance);
  cgbn_set_ui32(arith._env, address, instance);
  cgbn_set_ui32(arith._env, topic_1, (instance+1) * 10);
  cgbn_set_ui32(arith._env, topic_2, (instance+1) * 20);
  cgbn_set_ui32(arith._env, topic_3, (instance+1) * 30);
  cgbn_set_ui32(arith._env, topic_4, (instance+1) * 40);
  record.size=32;
  ONE_THREAD_PER_INSTANCE(
  record.data=new uint8_t[32];
  )
  for(uint32_t idx=0; idx<32; idx++)
    record.data[idx]=idx + instance;

  no_topics=4;

  printf("Test %u 2\n", instance);
  log_state->push(address, record, topic_1, topic_2, topic_3, topic_4, no_topics);

  printf("Test %u 3\n", instance);
  
  cgbn_set_ui32(arith._env, address, instance+1);
  cgbn_set_ui32(arith._env, topic_1, (instance+2) * 10);
  cgbn_set_ui32(arith._env, topic_2, (instance+2) * 20);
  cgbn_set_ui32(arith._env, topic_3, (instance+2) * 30);
  cgbn_set_ui32(arith._env, topic_4, (instance+2) * 40);

  for (uint32_t idx=0; idx<32; idx++)
    record.data[idx]=idx + instance + 1;
  
  no_topics=2;

  printf("Test %u 4\n", instance);
  parent_1_log_state->push(address, record, topic_1, topic_2, topic_3, topic_4, no_topics);
  printf("Test %u 5\n", instance);

  log_state->print();
  log_state->update_with_child_state(*parent_1_log_state);
  log_state->print();

  delete parent_1_log_state;
  parent_1_log_state=NULL;

  log_state->to_log_state_data_t(*log_state_data);

  delete log_state;
  log_state=NULL;

  ONE_THREAD_PER_INSTANCE(
  delete[] record.data;
  )
  record.data=NULL;
}

template<class params>
__global__ void kernel_logs(
  cgbn_error_report_t *report,
  typename log_state_t<params>::log_state_data_t *log_states_data,
  uint32_t instance_count
) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;

  if(instance >= instance_count)
    return;

  typedef arith_env_t<params> arith_t;

  // setup arithmetic
  arith_t arith(cgbn_report_monitor, report, instance);

  test_logs<params>(arith,  &(log_states_data[instance]), instance);
}

template<class params>
void run_test(uint32_t instance_count) {
  typedef arith_env_t<params> arith_t;
  typedef log_state_t<params> log_state_t;
  typedef typename log_state_t::log_state_data_t log_state_data_t;

  log_state_data_t *cpu_log_states_data;
  #ifndef ONLY_CPU
  CUDA_CHECK(cudaDeviceReset());
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024*1024*1024));
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 64*1024));
  log_state_data_t *gpu_log_states_data;
  cgbn_error_report_t     *report;
  #endif
  arith_t arith(cgbn_report_monitor, 0);

  printf("Generating log states\n");
  cpu_log_states_data=log_state_t::get_cpu_instances(instance_count);
  #ifndef ONLY_CPU
  gpu_log_states_data=log_state_t::get_gpu_instances_from_cpu_instances(cpu_log_states_data, instance_count);
  #endif
  printf("log states generated\n");

  #ifndef ONLY_CPU
  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  
  printf("Running GPU kernel ...\n");
  // launch kernel with blocks=ceil(instance_count/IPB) and threads=TPB
  kernel_logs<params><<<instance_count, params::TPI>>>(
    report,
    gpu_log_states_data,
    instance_count
  );

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
  printf("GPU kernel finished\n");

  printf("Copying the results back to the CPU ...\n");
  log_state_t::free_cpu_instances(cpu_log_states_data, instance_count);
  cpu_log_states_data=log_state_t::get_cpu_instances_from_gpu_instances(gpu_log_states_data, instance_count);
  printf("Results copied\n");
  CUDA_CHECK(cgbn_error_report_free(report));
  #else
  printf("Running CPU kernel ...\n");
  for(uint32_t instance=0; instance<instance_count; instance++)
  {
    test_logs(arith, &(cpu_log_states_data[instance]), instance);
  }
  printf("CPU kernel finished\n");
  #endif


  // print the results
  printf("Printing the results stdout/json...\n");
  cJSON *root;
  root = cJSON_CreateObject();
  cJSON *logs = cJSON_CreateArray();
  cJSON *log_json = NULL;
  printf("Logs:\n");
  for(uint32_t instance=0; instance<instance_count; instance++)
  {
    cJSON *instance_json = cJSON_CreateObject();
    cJSON_AddItemToArray(logs, instance_json);
    cJSON_AddNumberToObject(instance_json, "instance", instance);
    printf("Log state %d:\n", instance);
    log_state_t::print_log_state_data_t(arith, cpu_log_states_data[instance]);
    log_json = log_state_t::json_from_log_state_data_t(arith, cpu_log_states_data[instance]);
    cJSON_AddItemToObject(instance_json, "logs", log_json);
  }
  cJSON_AddItemToObject(root, "logs", logs);
  log_state_t::free_cpu_instances(cpu_log_states_data, instance_count);
  cpu_log_states_data=NULL;
  char *json_str=cJSON_Print(root);
  FILE *fp=fopen("output/evm_logs.json", "w");
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

#define INSTANCES 1


int main() {
  run_test<utils_params>(INSTANCES);
}