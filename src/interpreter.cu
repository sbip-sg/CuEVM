
#include "utils.h"
#include "evm.cuh"
#include <getopt.h>


template<class params>
__global__ void kernel_evm(cgbn_error_report_t *report, typename evm_t<params>::evm_instances_t *instances) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;

  if(instance >= instances->count)
    return;

  typedef arith_env_t<params> arith_t;
  typedef typename arith_t::bn_t  bn_t;
  typedef evm_t<params> evm_t;
  
  // setup evm
  evm_t evm(cgbn_report_monitor, report, instance, instances->block, instances->world_state);

  // run the evm
  evm.run(
    &(instances->msgs[instance]),
    &(instances->stacks[instance]),
    &(instances->return_datas[instance]),
    &(instances->memories[instance]),
    &(instances->access_states[instance]),
    &(instances->parents_write_states[instance]),
    &(instances->write_states[instance]),
    #ifdef GAS
    &(instances->gas_left_a[instance]),
    #endif
    #ifdef TRACER
    &(instances->tracers[instance]),
    #endif
    instances->errors[instance]
  );
}

template<class params>
void run_test(char *read_json_filename, char *write_json_filename) {
  typedef evm_t<params> evm_t;
  typedef typename evm_t::evm_instances_t evm_instances_t;
  typedef arith_env_t<params> arith_t;
  
  evm_instances_t         cpu_instances, tmp_gpu_instances, *gpu_instances;
  cgbn_error_report_t     *report;

  arith_t arith(cgbn_report_monitor, 0);
  
  //read the json file with the global state
  cJSON *root = get_json_from_file(read_json_filename);
  if(root == NULL) {
    printf("Error: could not read the json file\n");
    exit(EXIT_FAILURE);
  }
  const cJSON *test = NULL;
  test = cJSON_GetObjectItemCaseSensitive(root, "arith");

  // get instaces to run
  printf("Generating instances\n");
  evm_t::get_instances(cpu_instances, test);
  evm_t::get_gpu_instances(tmp_gpu_instances, cpu_instances);
  CUDA_CHECK(cudaMalloc(&gpu_instances, sizeof(evm_instances_t)));
  CUDA_CHECK(cudaMemcpy(gpu_instances, &tmp_gpu_instances, sizeof(evm_instances_t), cudaMemcpyHostToDevice));
  printf("Instances generated\n");

  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report)); 

  printf("Running GPU kernel ...\n");
  kernel_evm<params><<<cpu_instances.count, params::TPI>>>(report, gpu_instances);

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("GPU kernel finished\n");
  CGBN_CHECK(report);

  // copy the results back to the CPU
  printf("Copying results back to CPU\n");
  CUDA_CHECK(cudaMemcpy(&tmp_gpu_instances, gpu_instances, sizeof(evm_instances_t), cudaMemcpyDeviceToHost));
  evm_t::get_cpu_from_gpu_instances(cpu_instances, tmp_gpu_instances);
  printf("Results copied\n");

  // print the results
  printf("Printing the results ...\n");
  evm_t evm(cgbn_report_monitor, 0, cpu_instances.block, cpu_instances.world_state);
  evm.print_instances(cpu_instances);
  printf("Results printed\n");

  // print to json files
  printf("Printing to json files ...\n");
  cJSON_Delete(root);
  root = cJSON_CreateObject();
  cJSON_AddItemToObject(root, "test", evm.instances_to_json(cpu_instances));
  char *json_str=cJSON_Print(root);
  FILE *fp=fopen(write_json_filename, "w");
  fprintf(fp, "%s", json_str);
  fclose(fp);
  free(json_str);
  cJSON_Delete(root);
  printf("Json files printed\n");

  // free the memory
  printf("Freeing the memory ...\n");
  evm_t::free_instances(cpu_instances);
  CUDA_CHECK(cudaFree(gpu_instances));
  CUDA_CHECK(cgbn_error_report_free(report));
  
}

int main(int argc, char *argv[]) {//getting the input
  char *read_json_filename = NULL;
  char *write_json_filename = NULL;
  static struct option long_options[] = {
        {"input", required_argument, 0, 'i'},
        {"output", required_argument, 0, 'o'},
        {0, 0, 0, 0}};

  int opt;
  int option_index = 0;
  while ((opt = getopt_long(argc, argv, "i:o:", long_options, &option_index)) != -1)
  {
      switch (opt)
      {
      case 'i':
          read_json_filename = optarg;
          break;
      case 'o':
          write_json_filename = optarg;
          break;
      default:
          fprintf(stderr, "Usage: %s --input <json_filename> --output <json_filename>\n", argv[0]);
          exit(EXIT_FAILURE);
      }
  }
  if (!read_json_filename || !write_json_filename)
  {
      fprintf(stderr, "Both --input and --output flags are required\n");
      exit(EXIT_FAILURE);
  }
  run_test<utils_params>(read_json_filename, write_json_filename);
}