#include "../returndata.cuh"
#include "../utils.h"
 
__global__ void kernel_return_run(cgbn_error_report_t *report, data_content_t *instances, uint32_t instance_count) {
  
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x);
  
  if(instance>=instance_count)
    return;
  
  return_data_t  return_data(&instances[instance]);

  // printf("Instance %d:  ", instance);
  uint8_t tmp[32];
  for(uint32_t idx=0; idx<32; idx++) {
    tmp[idx]=(instance+1)*idx;
  }
  return_data.set(&(tmp[0]), 32);

  printf("size %lu:  ", return_data.size());

  return_data.print();
  
}

void run_test(uint32_t instance_count) {

  data_content_t   *cpu_returns, *gpu_returns;
  cgbn_error_report_t     *report;
  
  
  printf("Generating returns info\n");
  cpu_returns=return_data_t::get_returns(instance_count);
  gpu_returns=return_data_t::get_gpu_returns(cpu_returns, instance_count);
  return_data_t::free_host_returns(cpu_returns, instance_count);
  printf("returns info generated\n");

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
  cpu_returns=return_data_t::get_cpu_returns_from_gpu(gpu_returns, instance_count);
  printf("Results copied back to CPU\n");

  // print the results
  printf("Printing results and create jsons\n");
  cJSON *root = cJSON_CreateObject();
  cJSON *post = cJSON_CreateArray();
  for(uint32_t instance=0; instance<instance_count; instance++) {
    return_data_t local_returns(&(cpu_returns[instance]));
    printf("Instance %d:  ", instance);
    local_returns.print();
    printf("\n");
    cJSON_AddItemToArray(post, local_returns.to_json());
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

  // free the memory
  printf("Freeing the memory ...\n");
  CUDA_CHECK(cgbn_error_report_free(report));
  return_data_t::free_host_returns(cpu_returns, instance_count);
}


int main() {
  run_test(1);
}