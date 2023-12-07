// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#include "../state.cuh"

template<class params>
__host__ __device__ __forceinline__ void test_storage(
  arith_env_t<params> &arith,
  typename world_state_t<params>::state_data_t *world_state_data,
  typename accessed_state_t<params>::accessed_state_data_t *access_state_data,
  typename touch_state_t<params>::touch_state_data_t *touch_state_data,
  uint32_t &instance
)
{
  typedef world_state_t<params> world_state_t;
  typedef accessed_state_t<params> accessed_state_t;
  typedef touch_state_t<params> touch_state_t;
  typedef arith_env_t<params> arith_t;
  typedef typename arith_t::bn_t  bn_t;
  typedef typename world_state_t::account_t account_t;
  typedef typename accessed_state_t::accessed_state_data_t accessed_state_data_t;

  world_state_t *world_state;
  accessed_state_t *access_state;
  touch_state_t *touch_state;
  touch_state_t *parent_1_touch_state;
  touch_state_t *parent_2_touch_state;

  /*
  SHARED_MEMORY accessed_state_data_t *shared_access_state_data;
  ONE_THREAD_PER_INSTANCE(
    shared_access_state_data=new accessed_state_data_t;
    shared_access_state_data->accessed_accounts.no_accounts = 0;
    shared_access_state_data->accessed_accounts.accounts = NULL;
    shared_access_state_data->reads = NULL;

  )
  */
  world_state=new world_state_t(arith, world_state_data);
  //access_state=new accessed_state_t(shared_access_state_data, world_state);
  access_state=new accessed_state_t(world_state);
  parent_2_touch_state=new touch_state_t(access_state, NULL);
  parent_1_touch_state=new touch_state_t(access_state, parent_2_touch_state);
  touch_state=new touch_state_t(access_state, parent_1_touch_state);
  /*
  */
  bn_t address, key, nonce, value;
  bn_t gas_cost;
  bn_t gas_refund;
  account_t *account;
  bn_t account_address, account_balance, account_nonce;
  uint32_t error;

  
  cgbn_set_ui32(arith._env, gas_cost, 0);
  cgbn_set_ui32(arith._env, gas_refund, 0);
  account=NULL;

  
  cgbn_set_ui32(arith._env, address, 1);
  cgbn_set_ui32(arith._env, key, 1);
  printf("A address: %08x, gas_cost: %08x\n", cgbn_get_ui32(arith._env, address), cgbn_get_ui32(arith._env, gas_cost));
  // get the gas cost
  //access_state->get_access_account_gas_cost(address, gas_cost);
  account = world_state->get_account(address, error);
  printf("A address: %08x, gas_cost: %08x\n", cgbn_get_ui32(arith._env, address), cgbn_get_ui32(arith._env, gas_cost));
  access_state->get_access_account_gas_cost(address, gas_cost);
  printf("A address: %08x, gas_cost: %08x\n", cgbn_get_ui32(arith._env, address), cgbn_get_ui32(arith._env, gas_cost));
  printf("A address: %08x, gas_cost: %08x\n", cgbn_get_ui32(arith._env, address), cgbn_get_ui32(arith._env, gas_cost));
  // get the account
  account=touch_state->get_account(address, READ_BALANCE | READ_NONCE | READ_CODE);
  // load the account address, balance and nonce
  cgbn_load(arith._env, account_address, &(account->address));
  cgbn_load(arith._env, account_balance, &(account->balance));
  cgbn_load(arith._env, account_nonce, &(account->nonce));
  printf("A address: %08x, balance: %08x, nonce: %08x, code_size: %lu, storage_size: %lu\n", cgbn_get_ui32(arith._env, account_address), cgbn_get_ui32(arith._env, account_balance), cgbn_get_ui32(arith._env, account_nonce),account->code_size, account->storage_size);
  // get the gas cost for the storage
  access_state->get_access_storage_gas_cost(address, key, gas_cost);
  printf("A address: %08x, key: %08x, gas_cost: %08x\n", cgbn_get_ui32(arith._env, address), cgbn_get_ui32(arith._env, key), cgbn_get_ui32(arith._env, gas_cost));
  // get the value
  touch_state->get_value(address, key, value);
  printf("A address: %08x, key: %08x, value: %08x\n", cgbn_get_ui32(arith._env, address), cgbn_get_ui32(arith._env, key), cgbn_get_ui32(arith._env, value));

  // set the value in one of the parents
  cgbn_set_ui32(arith._env, value, instance+10);
  parent_2_touch_state->get_storage_set_gas_cost_gas_refund(address, key, value, gas_cost, gas_refund);
  printf("P2 address: %08x, key: %08x, value: %08x, gas_cost: %08x, gas_refund: %08x\n", cgbn_get_ui32(arith._env, address), cgbn_get_ui32(arith._env, key), cgbn_get_ui32(arith._env, value), cgbn_get_ui32(arith._env, gas_cost), cgbn_get_ui32(arith._env, gas_refund));
  parent_2_touch_state->set_value(address, key, value);
  // get the value
  touch_state->get_value(address, key, value);
  printf("T address: %08x, key: %08x, value: %08x\n", cgbn_get_ui32(arith._env, address), cgbn_get_ui32(arith._env, key), cgbn_get_ui32(arith._env, value));
  // set the value in the current touch state
  cgbn_set_ui32(arith._env, value, instance+20);
  touch_state->get_storage_set_gas_cost_gas_refund(address, key, value, gas_cost, gas_refund);
  printf("T address: %08x, key: %08x, value: %08x, gas_cost: %08x, gas_refund: %08x\n", cgbn_get_ui32(arith._env, address), cgbn_get_ui32(arith._env, key), cgbn_get_ui32(arith._env, value), cgbn_get_ui32(arith._env, gas_cost), cgbn_get_ui32(arith._env, gas_refund));
  touch_state->set_value(address, key, value);
  // get the value
  touch_state->get_value(address, key, value);
  printf("T address: %08x, key: %08x, value: %08x\n", cgbn_get_ui32(arith._env, address), cgbn_get_ui32(arith._env, key), cgbn_get_ui32(arith._env, value));
  // upstream the change in touch states
  parent_1_touch_state->update_with_child_state(*touch_state);
  parent_2_touch_state->update_with_child_state(*parent_1_touch_state);
  // save the states back
  ONE_THREAD_PER_INSTANCE(
    parent_2_touch_state->to_touch_state_data_t(*touch_state_data);
    access_state->to_accessed_state_data_t(*access_state_data);
  )

  // free the memory
  delete touch_state;
  touch_state=NULL;
  delete parent_1_touch_state;
  parent_1_touch_state=NULL;
  delete parent_2_touch_state;
  parent_2_touch_state=NULL;
  printf("A address: %08x\n", cgbn_get_ui32(arith._env, address));
  delete access_state;
  access_state=NULL;
  printf("A address: %08x, gas_cost: %08x\n", cgbn_get_ui32(arith._env, address), cgbn_get_ui32(arith._env, gas_cost));
  delete world_state;
  world_state=NULL;
  printf("A address: %08x, gas_cost: %08x\n", cgbn_get_ui32(arith._env, address), cgbn_get_ui32(arith._env, gas_cost));


}

template<class params>
__global__ void kernel_storage(
  cgbn_error_report_t *report,
  typename world_state_t<params>::state_data_t *world_state_data,
  typename accessed_state_t<params>::accessed_state_data_t *access_state_data,
  typename touch_state_t<params>::touch_state_data_t *touch_state_data,
  uint32_t instance_count
) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;

  if(instance >= instance_count)
    return;

  typedef arith_env_t<params> arith_t;

  // setup arithmetic
  arith_t arith(cgbn_report_monitor, report, instance);

  test_storage<params>(arith, world_state_data, &(access_state_data[instance]), &(touch_state_data[instance]), instance);
}

template<class params>
void run_test(uint32_t instance_count) {
  typedef arith_env_t<params> arith_t;
  typedef world_state_t<params> world_state_t;
  typedef typename world_state_t::state_data_t state_data_t;
  typedef accessed_state_t<params> accessed_state_t;
  typedef typename accessed_state_t::accessed_state_data_t accessed_state_data_t;
  typedef touch_state_t<params> touch_state_t;
  typedef typename touch_state_t::touch_state_data_t touch_state_data_t;

  state_data_t *world_state_data;
  accessed_state_data_t *cpu_access_states_data;
  touch_state_data_t *cpu_touch_states_data;
  #ifndef ONLY_CPU
  CUDA_CHECK(cudaDeviceReset());
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024*1024*1024));
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 64*1024));
  accessed_state_data_t *gpu_access_states_data;
  touch_state_data_t *gpu_touch_states_data;
  cgbn_error_report_t     *report;
  #endif
  arith_t arith(cgbn_report_monitor, 0);
  
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

  printf("Generating access states\n");
  cpu_access_states_data=accessed_state_t::get_cpu_instances(instance_count);
  #ifndef ONLY_CPU
  gpu_access_states_data=accessed_state_t::get_gpu_instances_from_cpu_instances(cpu_access_states_data, instance_count);
  #endif
  printf("access states generated\n");
  printf("Generating touch states\n");
  cpu_touch_states_data=touch_state_t::get_cpu_instances(instance_count);
  #ifndef ONLY_CPU
  gpu_touch_states_data=touch_state_t::get_gpu_instances_from_cpu_instances(cpu_touch_states_data, instance_count);
  #endif
  printf("touch states generated\n");

  #ifndef ONLY_CPU
  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  
  printf("Running GPU kernel ...\n");
  // launch kernel with blocks=ceil(instance_count/IPB) and threads=TPB
  kernel_storage<params><<<instance_count, params::TPI>>>(
    report,
    world_state_data,
    gpu_access_states_data,
    gpu_touch_states_data,
    instance_count
  );

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
  printf("GPU kernel finished\n");

  printf("Copying the results back to the CPU ...\n");
  cpu_access_states_data=accessed_state_t::get_cpu_instances_from_gpu_instances(gpu_access_states_data, instance_count);
  printf("Copying the results back to the CPU ...\n");
  cpu_touch_states_data=touch_state_t::get_cpu_instances_from_gpu_instances(gpu_touch_states_data, instance_count);
  printf("Results copied\n");
  CUDA_CHECK(cgbn_error_report_free(report));
  #else
  printf("Running CPU kernel ...\n");
  for(uint32_t instance=0; instance<instance_count; instance++)
  {
    test_storage(arith, world_state_data, &(cpu_access_states_data[instance]), &(cpu_touch_states_data[instance]), instance);
  }
  printf("CPU kernel finished\n");
  #endif


  // print the results
  printf("Printing the results stdout/json...\n");
  cJSON_Delete(root);
  root = cJSON_CreateObject();
  accessed_state_t *access_state;
  touch_state_t *touch_state;
  printf("World state:\n");
  cpu_world_state->print();
  cJSON_AddItemToObject(root, "pre", cpu_world_state->json());
  cJSON *post = cJSON_CreateArray();
  for(uint32_t instance=0; instance<instance_count; instance++)
  {
    cJSON *instance_json = cJSON_CreateObject();
    cJSON_AddItemToArray(post, instance_json);
    cJSON_AddNumberToObject(instance_json, "instance", instance);
    accessed_state_data_t *access_state_data = new accessed_state_data_t;
    memcpy(access_state_data, &(cpu_access_states_data[instance]), sizeof(accessed_state_data_t));
    access_state = new accessed_state_t(access_state_data, cpu_world_state);
    touch_state_data_t *touch_state_data = new touch_state_data_t;
    memcpy(touch_state_data, &(cpu_touch_states_data[instance]), sizeof(touch_state_data_t));
    touch_state = new touch_state_t(touch_state_data, access_state, NULL);
    printf("Access state %d:\n", instance);
    access_state->print();
    cJSON_AddItemToObject(instance_json, "access", access_state->json());
    printf("Touch state %d:\n", instance);
    touch_state->print();
    cJSON_AddItemToObject(instance_json, "touch", touch_state->json());
    delete touch_state;
    touch_state=NULL;
    delete access_state;
    access_state=NULL;
  }
  cJSON_AddItemToObject(root, "post", post);
  cpu_world_state->free_world_state_data();
  delete cpu_world_state;
  cpu_world_state=NULL;
  delete[] cpu_access_states_data;
  cpu_access_states_data=NULL;
  delete[] cpu_touch_states_data;
  cpu_touch_states_data=NULL;
  char *json_str=cJSON_Print(root);
  FILE *fp=fopen("output/evm_state.json", "w");
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