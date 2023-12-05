

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <gmp.h>
#ifndef __CGBN_H__
#define __CGBN_H__
#include <cgbn/cgbn.h>
#endif
#include "../memory.cuh"
#include "../utils.h"
 
template<class params>
__global__ void kernel_memory_run(cgbn_error_report_t *report, typename memory_t<params>::memory_data_t *instances, uint32_t instance_count) {
  
  typedef memory_t<params>                  memory_t;
  typedef arith_env_t<params>               arith_t;
  typedef typename memory_t::memory_data_t  memory_data_t;
  typedef typename arith_t::bn_t            bn_t;

  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;
  __shared__ memory_data_t memory_data;
  __shared__ uint8_t       tmp[params::BITS/8];
  memory_data.size=0;
  memory_data.alocated_size=0;
  memory_data.data=NULL;
  
  if(instance>=instance_count)
    return;
  
  
  // setup arithmetic
  arith_t arith(cgbn_report_monitor, report, instance);

  memory_t  memory(arith, &memory_data);
  
  //printf("Instance %d:  memory size=%d\n", instance, memory.size());
  
  bn_t          a, b, c, gas;
  uint32_t     error_code;
  error_code=0;
  cgbn_set_ui32(arith._env, gas, 0);
  cgbn_set_ui32(arith._env, a, instance + 1);
  cgbn_set_ui32(arith._env, b, 10);
  cgbn_set_ui32(arith._env, c, 100);

  arith.from_cgbn_to_memory(&(tmp[0]), a);
  memory.set(&(tmp[0]), 0, 32, gas, error_code);
  printf("error_code=%d gas=%08x\n", error_code, cgbn_get_ui32(arith._env, gas));
  arith.from_cgbn_to_memory(&(tmp[0]), b);
  memory.set(&(tmp[0]), 33, 32, gas, error_code);
  printf("error_code=%d gas=%08x\n", error_code, cgbn_get_ui32(arith._env, gas));
  arith.from_memory_to_cgbn(c, memory.get(0, 32, gas, error_code));
  printf("error_code=%d gas=%08x\n", error_code, cgbn_get_ui32(arith._env, gas));
  printf("RUN S CGBN lowestbit=%08x\n", cgbn_get_ui32(arith._env, c));
  memory.copy_info(&(instances[instance]));
  if (threadIdx.x == 0) {
    printf("RUN S data address=%p\n", memory._content->data);
    printf("RUN S lowestbit=%02x\n", memory._content->data[31]);
    printf("RUN D data address=%p\n", instances[instance].data);
    printf("RUN D lowestbit=%02x\n", instances[instance].data[31]);
  }
}

template<class params>
void run_test(uint32_t instance_count) {
  typedef memory_t<params>                  memory_t;
  typedef arith_env_t<params>               arith_t;
  typedef typename memory_t::memory_data_t  memory_data_t;
  typedef typename arith_t::bn_t            bn_t;
  
  memory_data_t           *cpu_memories, *gpu_memories;
  cgbn_error_report_t     *report;
  arith_t arith(cgbn_report_monitor, 0);
  
  
  printf("Generating memories info\n");
  cpu_memories=memory_t::get_memories_info(instance_count);
  gpu_memories=memory_t::get_gpu_memories_info(cpu_memories, instance_count);
  memory_t::free_memories_info(cpu_memories, instance_count);
  printf("Memories info generated\n");

  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 

  printf("Running GPU RUN kernel ...\n");

  // launch kernel with blocks=ceil(instance_count/IPB) and threads=TPB
  kernel_memory_run<params><<<instance_count, params::TPI>>>(report, gpu_memories, instance_count);

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);

  printf("GPU RUN kernel finished\n");

  // copy the results back to the CPU
  printf("Copying results back to CPU\n");
  cpu_memories=memory_t::get_memories_from_gpu(gpu_memories, instance_count);
  printf("Results copied back to CPU\n");

  // print the results
  printf("Printing results and create jsons\n");
  cJSON *root = cJSON_CreateObject();
  cJSON *post = cJSON_CreateArray();
  for(uint32_t instance=0; instance<instance_count; instance++) {
    memory_t local_memory(arith, &(cpu_memories[instance]));
    printf("Instance %d:  ", instance);
    local_memory.print();
    printf("\n");
    cJSON_AddItemToArray(post, local_memory.to_json());
  }
  printf("Results printed\n");
  cJSON_AddItemToObject(root, "post", post);
  char *json_str=cJSON_Print(root);
  FILE *fp=fopen("output/evm_memory.json", "w");
  fprintf(fp, "%s", json_str);
  fclose(fp);
  free(json_str);
  cJSON_Delete(root);
  printf("Json files printed\n");

  // free the memory
  printf("Freeing the memory ...\n");
  CUDA_CHECK(cgbn_error_report_free(report));
  memory_t::free_memory_data(cpu_memories, instance_count);
}


int main() {
  run_test<utils_params>(2);
}