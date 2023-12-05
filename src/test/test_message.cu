


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
#include "../message.cuh"
#include "../arith.cuh"


__device__ state_t<utils_params>::state_data_t global_state;

template<class params>
__global__ void kernel_message(cgbn_error_report_t *report, typename message_t<params>::message_content_t *msgs, uint32_t instance_count) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;

  if(instance>=instance_count)
    return;
  typedef state_t<params>                 state_t;
  typedef message_t<params>               message_t;
  typedef arith_env_t<params>             arith_t;
  typedef typename arith_t::bn_t          bn_t;
  typedef typename state_t::contract_t    contract_t;
  arith_t arith(cgbn_report_monitor, report, instance);
  contract_t *contract;
  message_t message(arith, &(msgs[instance]));
  state_t global(arith, &(global_state));


  bn_t caller, value, nonce, to, tx_origin, tx_gasprice;
  uint32_t error_code;
  error_code = ERR_SUCCESS;
  message.get_caller(caller);
  message.get_value(value);
  message.get_nonce(nonce);
  message.get_to(to);
  message.get_tx_origin(tx_origin);
  message.get_tx_gasprice(tx_gasprice);
  printf("caller: %08x, value: %08x, to: %08x, nonce: %08x, tx_origin: %08x, tx_gasprice: %08x, data_size: %lx, depth: %d, call_type: %d\n", cgbn_get_ui32(arith._env, caller), cgbn_get_ui32(arith._env, value), cgbn_get_ui32(arith._env, to), cgbn_get_ui32(arith._env, nonce), cgbn_get_ui32(arith._env, tx_origin), cgbn_get_ui32(arith._env, tx_gasprice), message.get_data_size(), message.get_depth(), message.get_call_type());
  uint8_t *data=message.get_data(0, 32, error_code);
  if (error_code!=ERR_SUCCESS) {
    printf("Error: %d\n", error_code);
    error_code=ERR_SUCCESS;
  } else {
    printf("data: ");
    print_bytes(data, 32);
    printf("\n");
  }
  bn_t address;
  cgbn_set_ui32(arith._env, address, 0x00000001);
  contract=global.get_account(address, error_code);
  message.set_contract(contract);
  contract=message.get_contract();
  bn_t contract_address, contract_balance;
  cgbn_load(arith._env, contract_address, &(contract->address));
  cgbn_load(arith._env, contract_balance, &(contract->balance));
  printf("contract: %p\n", contract);
  printf("address: %08x, balance: %08x, code_size: %lx, storage_size: %lx\n", cgbn_get_ui32(arith._env, contract_address), cgbn_get_ui32(arith._env, contract_balance), contract->code_size, contract->storage_size);
}

template<class params>
void run_test() {
  typedef state_t<params> state_t;
  typedef typename state_t::state_data_t state_data_t;
  typedef arith_env_t<params> arith_t;
  typedef message_t<params> message_t;
  typedef typename message_t::message_content_t message_content_t;
  
  state_data_t            *gpu_global_state;
  message_content_t       *cpu_messages, *gpu_messages;
  cgbn_error_report_t     *report;
  arith_t arith(cgbn_report_monitor, 0);
  
  //read the json file with the transactions
  cJSON *root = get_json_from_file("input/evm_test.json");
  if(root == NULL) {
    printf("Error: could not read the json file\n");
    exit(1);
  }
  const cJSON *test = NULL;
  test = cJSON_GetObjectItemCaseSensitive(root, "sstoreGas");


  printf("Generating global state\n");
  state_t cpu_global_state(arith, test);
  gpu_global_state=cpu_global_state.to_gpu();
  CUDA_CHECK(cudaMemcpyToSymbol(global_state, gpu_global_state, sizeof(state_data_t)));
  printf("Global state generated\n");

  printf("Generating messages\n");
  size_t messages_count=1;
  message_t message(arith, test);
  cpu_messages=message_t::get_messages(test, messages_count);
  gpu_messages=message_t::get_gpu_messages(cpu_messages, messages_count);
  message.print();
  printf("Messages generated\n");
  
  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  
  printf("Running GPU kernel ...\n");

  // launch kernel with blocks=ceil(instance_count/IPB) and threads=TPB
  kernel_message<params><<<1, params::TPI>>>(report, gpu_messages, messages_count);

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
    
    
  // print to json files
  printf("Printing to json files ...\n");
  cJSON_Delete(root);
  root = cJSON_CreateObject();
  cJSON_AddItemToObject(root, "transaction", message.to_json());
  cJSON *post = cJSON_CreateArray();
  for(uint32_t idx=0; idx<messages_count; idx++) {
    message_t local_message(arith, &(cpu_messages[idx]));
    cJSON_AddItemToArray(post, local_message.to_json());
  }
  cJSON_AddItemToObject(root, "post", post);
  char *json_str=cJSON_Print(root);
  FILE *fp=fopen("output/evm_message.json", "w");
  fprintf(fp, "%s", json_str);
  fclose(fp);
  free(json_str);
  printf("Json files printed\n");
  
  // free the memory
  printf("Freeing the memory ...\n");
  message.free_memory();
  message_t::free_messages(cpu_messages, messages_count);
  message_t::free_gpu_messages(gpu_messages, messages_count);
  //cJSON_Delete(root);
  CUDA_CHECK(cgbn_error_report_free(report));
}

int main() {
  run_test<utils_params>();
}