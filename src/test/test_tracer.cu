

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <gmp.h>
#ifndef __CGBN_H__
#define __CGBN_H__
#include <cgbn/cgbn.h>
#endif
#include "../stack.cuh"
#include "../tracer.cuh"
#include "../utils.h"

 
template<class params>
__global__ void kernel_tracer(cgbn_error_report_t *report,typename tracer_t<params>::tracer_content_t *tracers, typename stack_t<params>::stack_data_t *stacks, uint32_t instance_count) {
  typedef arith_env_t<params>                     arith_t;
  typedef typename arith_t::bn_t                  bn_t;
  typedef cgbn_mem_t<params::BITS>                evm_word_t;
  typedef stack_t<params>                         stack_t;
  typedef typename stack_t::stack_data_t          stack_data_t;
  typedef typename stack_t::stack_content_data_t  stack_content_data_t;
  typedef tracer_t<params>                        tracer_t;
  typedef typename tracer_t::tracer_content_t     tracer_content_t;
  typedef typename tracer_t::tracer_data_t        tracer_data_t;

  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;
  __shared__ stack_content_data_t                 stack_content_data;
  
  if(instance>=instance_count)
    return;


  __syncthreads();
  if (threadIdx.x == 0)
    memcpy(&(stack_content_data.values[0]), stacks[instance].stack_base, sizeof(stack_content_data_t));
  __syncthreads();

  // setup arithmetic
  arith_t arith(cgbn_report_monitor, report, instance);
 
  //local_stack_t  stack(arith, &(instances[instance].values[0]), 0);
  stack_data_t   stack_data;
  stack_data.stack_offset = 0;
  stack_data.stack_base = &(stack_content_data.values[0]);

  stack_t  stack(arith, &(stack_data));
  tracer_t tracer(arith, &(tracers[instance]));

  uint32_t pc;
  pc=0;
  uint8_t opcode;
  opcode=OP_PUSH32;
  bn_t a, b, c, gas_cost, address;
  cgbn_set_ui32(arith._env, address, 1);
  cgbn_set_ui32(arith._env, gas_cost, 0);
  uint32_t error_code;
  cgbn_set_ui32(arith._env, a, instance);
  cgbn_set_ui32(arith._env, b, 0xFF);
  stack.push(b, error_code);
  tracer.push(address, pc, opcode, &stack);
  pc=1;
  stack.push(a, error_code);
  tracer.push(address, pc, opcode, &stack);
  pc=2;
  opcode=OP_SIGNEXTEND;
  stack.signextend(error_code);
  tracer.push(address, pc, opcode, &stack);
  stack.copy_stack_data(&(stacks[instance]), 0);
}

template<class params>
void run_test(uint32_t instance_count) {
  typedef arith_env_t<params>                     arith_t;
  typedef typename arith_t::bn_t                  bn_t;
  typedef cgbn_mem_t<params::BITS>                evm_word_t;
  typedef stack_t<params>                         stack_t;
  typedef typename stack_t::stack_data_t          stack_data_t;
  typedef typename stack_t::stack_content_data_t  stack_content_data_t;
  typedef tracer_t<params>                        tracer_t;
  typedef typename tracer_t::tracer_content_t     tracer_content_t;
  typedef typename tracer_t::tracer_data_t        tracer_data_t;
  
  stack_data_t          *cpu_stacks, *gpu_stacks;
  tracer_content_t      *cpu_tracers, *gpu_tracers;
  cgbn_error_report_t *report;
  arith_t arith(cgbn_report_monitor, 0);
  

  printf("Geenerating stack data ...\n");
  cpu_stacks=stack_t::get_stacks(instance_count);
  printf("Copying stack data to the GPU ...\n");
  gpu_stacks=stack_t::get_gpu_stacks(cpu_stacks, instance_count);
  printf("Freeing stack data on CPU ...\n");
  stack_t::free_stacks(cpu_stacks, instance_count);
  printf("Generating tracer data ...\n");
  cpu_tracers=tracer_t::get_tracers(instance_count);
  printf("Copying tracer data to the GPU ...\n");
  gpu_tracers=tracer_t::get_gpu_tracers(cpu_tracers, instance_count);
  printf("Freeing tracer data on CPU ...\n");
  tracer_t::free_tracers(cpu_tracers, instance_count);

  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 
  
  printf("Running GPU kernel ...\n");

  kernel_tracer<params><<<instance_count, params::TPI>>>(report, gpu_tracers, gpu_stacks, instance_count);

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);
    
  // copy the instances back from gpuMemory
  printf("Copying results back to CPU ...\n");
  cpu_stacks=stack_t::get_cpu_stacks_from_gpu(gpu_stacks, instance_count);
  stack_t::free_gpu_stacks(gpu_stacks, instance_count);
  cpu_tracers=tracer_t::get_cpu_tracers_from_gpu(gpu_tracers, instance_count);
  
  // print the results
  printf("Printing results and create jsons\n");
  cJSON *root = cJSON_CreateObject();
  cJSON *post = cJSON_CreateArray();
  for(uint32_t instance=0; instance<instance_count; instance++) {
    stack_t local_stack(arith, &(cpu_stacks[instance]));
    tracer_t local_tracer(arith, &(cpu_tracers[instance]));
    printf("Instance %d:  ", instance);
    local_stack.print();
    printf("\n");
    local_tracer.print();
    printf("\n");
    cJSON *instance_json = cJSON_CreateObject();
    cJSON_AddItemToObject(instance_json, "stack", local_stack.to_json());
    cJSON_AddItemToObject(instance_json, "tracer", local_tracer.to_json());
    cJSON_AddItemToArray(post, instance_json);
  }
  printf("Results printed\n");
  cJSON_AddItemToObject(root, "post", post);
  char *json_str=cJSON_Print(root);
  FILE *fp=fopen("output/evm_tracer.json", "w");
  fprintf(fp, "%s", json_str);
  fclose(fp);
  free(json_str);
  cJSON_Delete(root);
  printf("Json files printed\n");
  
  
  // clean up
  printf("Freeing stack data on CPU ...\n");
  stack_t::free_stacks(cpu_stacks, instance_count);
  printf("Freeing tracer data on CPU ...\n");
  tracer_t::free_tracers(cpu_tracers, instance_count);
  printf("Freeing error report ...\n");
  CUDA_CHECK(cgbn_error_report_free(report));
}

int main() {
  run_test<utils_params>(2);
}