
// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#include "../message.cuh"

template<class params>
__host__ __device__ __forceinline__ void test_message(
    arith_env_t<params> &arith,
    typename transaction_t<params>::transaction_data_t *transaction_data,
    uint32_t &instance)
{
  typedef transaction_t<params> transaction_t;
  typedef message_t<params> message_t;
  typedef arith_env_t<params>             arith_t;
  typedef typename arith_t::bn_t          bn_t;

  transaction_t *transaction;
  message_t *message;
  transaction = new transaction_t(arith, transaction_data);
  //transaction->print();

  uint8_t type;
  //uint32_t error_code;
  //error_code = ERR_SUCCESS;
  // transaction values
  bn_t nonce, gas_limit, to, value, sender, max_fee_per_gas, max_priority_fee_per_gas, gas_price;

  transaction->get_nonce(nonce);
  printf("nonce: %08x\n", cgbn_get_ui32(arith._env, nonce));
  transaction->get_gas_limit(gas_limit);
  printf("gas_limit: %08x\n", cgbn_get_ui32(arith._env, gas_limit));
  transaction->get_to(to);
  printf("to: %08x\n", cgbn_get_ui32(arith._env, to));
  transaction->get_value(value);
  printf("value: %08x\n", cgbn_get_ui32(arith._env, value));
  transaction->get_sender(sender);
  printf("sender: %08x\n", cgbn_get_ui32(arith._env, sender));
  transaction->get_max_fee_per_gas(max_fee_per_gas);
  printf("max_fee_per_gas: %08x\n", cgbn_get_ui32(arith._env, max_fee_per_gas));
  transaction->get_max_priority_fee_per_gas(max_priority_fee_per_gas);
  printf("max_priority_fee_per_gas: %08x\n", cgbn_get_ui32(arith._env, max_priority_fee_per_gas));
  transaction->get_gas_price(gas_price);
  printf("gas_price: %08x\n", cgbn_get_ui32(arith._env, gas_price));
  type = transaction->_content->type;
  printf("type: %u\n", type);

  uint8_t opcode=OP_CALL;
  size_t size=transaction->_content->data_init.size;
  uint8_t *data=transaction->_content->data_init.data;
  uint32_t depth=0;
  bn_t ret_offset, ret_size;
  cgbn_set_ui32(arith._env, ret_offset, 0);
  cgbn_set_ui32(arith._env, ret_size, 0);

  message = new message_t(
    arith,
    sender,
    to,
    to,
    gas_limit,
    value,
    depth,
    opcode,
    to,
    data,
    size,
    data,
    size,
    ret_offset,
    ret_size,
    0);
  
  ONE_THREAD_PER_INSTANCE(
    message->print();
  )

  delete message;
  message = NULL;
  delete transaction;
  transaction = NULL;
}


template<class params>
__global__ void kernel_message(
  cgbn_error_report_t *report,
  typename transaction_t<params>::transaction_data_t *transanctions_data,
  uint32_t count
) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;

  if(instance>=count)
    return;
  typedef arith_env_t<params>             arith_t;
  arith_t arith(cgbn_report_monitor, report, instance);
  
  test_message(arith, &(transanctions_data[instance]), instance);
}

template<class params>
void run_test() {
  typedef arith_env_t<params> arith_t;
  typedef transaction_t<params> transaction_t;
  typedef typename transaction_t::transaction_data_t transaction_data_t;
  
  transaction_data_t            *transactions_data;
  arith_t arith(cgbn_report_monitor, 0);
  
  #ifndef ONLY_CPU
  CUDA_CHECK(cudaDeviceReset());
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024*1024*1024));
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 64*1024));
  cgbn_error_report_t           *report;
  #endif
  //read the json file with the transactions
  cJSON *root = get_json_from_file("input/evm_arith.json");
  if(root == NULL) {
    printf("Error: could not read the json file\n");
    exit(1);
  }
  const cJSON *test = NULL;
  test = cJSON_GetObjectItemCaseSensitive(root, "arith");


  printf("Generating transactions\n");
  size_t transactions_count=1;
  transaction_t::get_transactions(transactions_data, test, transactions_count);
  printf("no_transactions: %lu\n", transactions_count);
  printf("Global transactions generated\n");

  #ifndef ONLY_CPU
  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  
  printf("Running GPU kernel ...\n");

  // launch kernel with blocks=ceil(instance_count/IPB) and threads=TPB
  kernel_message<params><<<transactions_count, params::TPI>>>(report, transactions_data, transactions_count);
  //kernel_message<params><<<1, params::TPI>>>(report, transactions_data, 1);

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
  CUDA_CHECK(cgbn_error_report_free(report));
  printf("GPU kernel finished\n");
  #else
  printf("Running CPU kernel ...\n");
  for(uint32_t idx=0; idx<transactions_count; idx++) {
    test_message(arith, &(transactions_data[idx]), idx);
  }
  //test_message(arith, &(transactions_data[0]), 0);
  printf("CPU kernel finished\n");
  #endif
    
    
  // print the results
  printf("Printing the results stdout/json...\n");
  cJSON_Delete(root);
  root = cJSON_CreateObject();
  transaction_t *transaction;
  cJSON *post = cJSON_CreateArray();
  for(uint32_t idx=0; idx<transactions_count; idx++) {
    transaction = new transaction_t(arith, &(transactions_data[idx]));
    cJSON_AddItemToArray(post, transaction->json());
    transaction->print();
    delete transaction;
  }
  transaction_t::free_instances(transactions_data, transactions_count);
  transactions_data = NULL;
  transactions_count = 0;
  transaction = NULL;
  cJSON_AddItemToObject(root, "post", post);
  char *json_str=cJSON_Print(root);
  FILE *fp=fopen("output/evm_message.json", "w");
  fprintf(fp, "%s", json_str);
  fclose(fp);
  free(json_str);
  json_str=NULL;
  cJSON_Delete(root);
  root=NULL;
  printf("Results printed\n");
  #ifndef ONLY_CPU
  CUDA_CHECK(cudaDeviceReset());
  #endif
}

int main() {
  run_test<utils_params>();
}