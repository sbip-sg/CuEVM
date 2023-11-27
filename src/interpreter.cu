
#include "utils.h"
#include "evm.cuh"
#include <getopt.h>


template<class params>
void run_interpreter(char *read_json_filename, char *write_json_filename) {
  typedef evm_t<params> evm_t;
  typedef typename evm_t::evm_instances_t evm_instances_t;
  typedef arith_env_t<params> arith_t;
  
  evm_instances_t         cpu_instances;
  #ifndef ONLY_CPU
  evm_instances_t tmp_gpu_instances, *gpu_instances;
  #endif
  cgbn_error_report_t     *report;

  arith_t arith(cgbn_report_monitor, 0);
  
  //read the json file with the global state
  cJSON *read_root = get_json_from_file(read_json_filename);
  if(read_root == NULL) {
    printf("Error: could not read the json file\n");
    exit(EXIT_FAILURE);
  }
  cJSON *write_root = cJSON_CreateObject();
  const cJSON *test = NULL;
  cJSON_ArrayForEach(test, read_root) {
    // get instaces to run
    printf("Generating instances\n");
    evm_t::get_instances(cpu_instances, test);
    #ifndef ONLY_CPU
    evm_t::get_gpu_instances(tmp_gpu_instances, cpu_instances);
    CUDA_CHECK(cudaMalloc(&gpu_instances, sizeof(evm_instances_t)));
    CUDA_CHECK(cudaMemcpy(gpu_instances, &tmp_gpu_instances, sizeof(evm_instances_t), cudaMemcpyHostToDevice));
    #endif
    printf("Instances generated\n");

    // create a cgbn_error_report for CGBN to report back errors
    #ifndef ONLY_CPU
    CUDA_CHECK(cgbn_error_report_alloc(&report)); 
    size_t heap_size;
    cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024*1024*1024));
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 64*1024));
    printf("Heap size: %zu\n", heap_size);
    cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);
    printf("Heap size: %zu\n", heap_size);
    #endif

    // evm
    evm_t evm(cgbn_report_monitor, 0, &(cpu_instances.sha3_parameters[0]), cpu_instances.block, cpu_instances.world_state);


    #ifndef ONLY_CPU
    printf("Running GPU kernel ...\n");
    kernel_evm<params><<<cpu_instances.count, params::TPI>>>(report, gpu_instances);
    //CUDA_CHECK(cudaPeekAtLastError());
    // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("GPU kernel finished\n");
    CGBN_CHECK(report);

    // copy the results back to the CPU
    printf("Copying results back to CPU\n");
    CUDA_CHECK(cudaMemcpy(&tmp_gpu_instances, gpu_instances, sizeof(evm_instances_t), cudaMemcpyDeviceToHost));
    evm_t::get_cpu_from_gpu_instances(cpu_instances, tmp_gpu_instances);
    printf("Results copied\n");
    #else
    printf("Running CPU EVM\n");
    // run the evm
    for(size_t instance = 0; instance < cpu_instances.count; instance++) {
      evm.run(
        &(cpu_instances.msgs[instance]),
        &(cpu_instances.stacks[instance]),
        &(cpu_instances.return_datas[instance]),
        &(cpu_instances.memories[instance]),
        &(cpu_instances.access_states[instance]),
        &(cpu_instances.parents_write_states[instance]),
        &(cpu_instances.write_states[instance]),
        #ifdef GAS
        &(cpu_instances.gas_left_a[instance]),
        #endif
        #ifdef TRACER
        &(cpu_instances.tracers[instance]),
        #endif
        cpu_instances.errors[instance]
      );
    }
    printf("CPU EVM finished\n");
    #endif


    // print the results
    printf("Printing the results ...\n");
    evm.print_instances(cpu_instances);
    printf("Results printed\n");

    // print to json files
    printf("Printing to json files ...\n");
    cJSON_AddItemToObject(write_root, test->string, evm.instances_to_json(cpu_instances));
    printf("Json files printed\n");

    // free the memory
    printf("Freeing the memory ...\n");
    evm_t::free_instances(cpu_instances);
    #ifndef ONLY_CPU
    CUDA_CHECK(cudaFree(gpu_instances));
    CUDA_CHECK(cgbn_error_report_free(report));
    #endif
  }
  cJSON_Delete(read_root);
  char *json_str=cJSON_Print(write_root);
  FILE *fp=fopen(write_json_filename, "w");
  fprintf(fp, "%s", json_str);
  fclose(fp);
  free(json_str);
  cJSON_Delete(write_root);
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
  run_interpreter<utils_params>(read_json_filename, write_json_filename);
}