#include "../returndata.cuh"
#include "../utils.h"
 

__host__ __device__ __forceinline__ void test_return_data(
  byte_array_t *data,
  uint32_t instance
)
{
  EVMReturnData  *return_data;
  return_data = new EVMReturnData();

  // printf("Instance %d:  ", instance);
  uint8_t tmp[32];
  for(uint32_t idx=0; idx<32; idx++) {
    tmp[idx]=(instance+1)*idx;
  }
  return_data->set(&(tmp[0]), 32);

  printf("size %lu:  ", return_data->size());

  return_data->print();

  return_data->to_byte_array_t(*data);

  delete return_data;
  return_data = NULL;
}

__global__ void kernel_return_run(cgbn_error_report_t *report, byte_array_t *instances, uint32_t instance_count) {
  
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x);
  
  if(instance>=instance_count)
    return;
  
  test_return_data(&(instances[instance]), instance);
  
}

void run_test(uint32_t instance_count) {

  byte_array_t   *cpu_returns;
  
  
  printf("Generating returns info\n");
  cpu_returns=EVMReturnData::get_cpu_instances(instance_count);
  #ifndef ONLY_CPU
  byte_array_t   *gpu_returns;
  cgbn_error_report_t     *report;
  gpu_returns=EVMReturnData::get_gpu_instances_from_cpu_instances(cpu_returns, instance_count);
  #endif
  printf("returns info generated\n");

  #ifndef ONLY_CPU
  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 

  printf("Running GPU RUN kernel ...\n");

  kernel_return_run<<<1, instance_count>>>(report, gpu_returns, instance_count);

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);

  printf("GPU RUN kernel finished\n");

  // copy the results back to the CPU
  printf("Copying results back to CPU\n");
  EVMReturnData::free_cpu_instances(cpu_returns, instance_count);
  cpu_returns=EVMReturnData::get_cpu_instances_from_gpu_instances(gpu_returns, instance_count);
  printf("Results copied back to CPU\n");
  #else
  printf("Running CPU RUN kernel ...\n");
  for(uint32_t instance=0; instance<instance_count; instance++) {
    test_return_data(&(cpu_returns[instance]), instance);
  }
  printf("CPU RUN kernel finished\n");
  #endif


  // print the results
  printf("Printing results and create jsons\n");
  cJSON *root = cJSON_CreateObject();
  cJSON *post = cJSON_CreateArray();
  for(uint32_t instance=0; instance<instance_count; instance++) {
    printf("Instance %d:  ", instance);
    print_byte_array_t(cpu_returns[instance]);
    printf("\n");
    cJSON_AddItemToArray(post, json_from_byte_array_t(cpu_returns[instance]));
  }
  printf("Results printed\n");
  cJSON_AddItemToObject(root, "post", post);
  char *json_str=cJSON_Print(root);
  FILE *fp=fopen("output/evm_returns.json", "w");
  fprintf(fp, "%s", json_str);
  fclose(fp);
  free(json_str);
  cJSON_Delete(root);
  printf("Json files printed\n");

  printf("Freeing the memory ...\n");
  EVMReturnData::free_cpu_instances(cpu_returns, instance_count);
  // free the memory
  #ifndef ONLY_CPU
  CUDA_CHECK(cgbn_error_report_free(report));
  #endif
}


int main() {
  run_test(1);
}