#include <getopt.h>
#include <fstream>

#include "utils.cu"
#include "evm.cuh"


void run_interpreter(char *read_json_filename, char *write_json_filename) {
  // typedef evm_t<evm_params> evm_t;
  typedef typename evm_t::evm_instances_t evm_instances_t;
  typedef arith_env_t<evm_params> arith_t;

  evm_instances_t         cpu_instances;
  #ifndef ONLY_CPU
  evm_instances_t tmp_gpu_instances, *gpu_instances;
  cgbn_error_report_t     *report;
  CUDA_CHECK(cudaDeviceReset());
  #endif

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
    evm_t::get_cpu_instances(cpu_instances, test);
    printf("%d instances generated\n", cpu_instances.count);

    #ifndef ONLY_CPU
    evm_t::get_gpu_instances(tmp_gpu_instances, cpu_instances);
    CUDA_CHECK(cudaMalloc(&gpu_instances, sizeof(evm_instances_t)));
    CUDA_CHECK(cudaMemcpy(gpu_instances, &tmp_gpu_instances, sizeof(evm_instances_t), cudaMemcpyHostToDevice));
    #endif

    // create a cgbn_error_report for CGBN to report back errors
    #ifndef ONLY_CPU
    size_t heap_size, stack_size;
    CUDA_CHECK(cgbn_error_report_alloc(&report));
    cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);
    heap_size = (size_t(2)<<30); // 2GB
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_size));
    // CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 256*1024));
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 64*1024));
    printf("Heap size: %zu\n", heap_size);
    cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);
    printf("Heap size: %zu\n", heap_size);
    printf("Running GPU kernel ...\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    kernel_evm<evm_params><<<cpu_instances.count, evm_params::TPI>>>(report, gpu_instances);
    //CUDA_CHECK(cudaPeekAtLastError());
    // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("GPU kernel finished\n");
    CGBN_CHECK(report);

    // copy the results back to the CPU
    printf("Copying results back to CPU\n");
    CUDA_CHECK(cudaMemcpy(&tmp_gpu_instances, gpu_instances, sizeof(evm_instances_t), cudaMemcpyDeviceToHost));
    evm_t::get_cpu_instances_from_gpu_instances(cpu_instances, tmp_gpu_instances);
    printf("Results copied\n");
    #else
    printf("Running CPU EVM\n");
    // run the evm
    evm_t *evm = NULL;
    uint32_t tmp_error;
    for(uint32_t instance = 0; instance < cpu_instances.count; instance++) {
      printf("Running instance %d\n", instance);
      evm = new evm_t(
          arith,
          cpu_instances.world_state_data,
          cpu_instances.block_data,
          cpu_instances.sha3_parameters,
          &(cpu_instances.transactions_data[instance]),
          &(cpu_instances.accessed_states_data[instance]),
          &(cpu_instances.touch_states_data[instance]),
          &(cpu_instances.logs_data[instance]),
          #ifdef TRACER
          &(cpu_instances.tracers_data[instance]),
          #endif
          instance,
          &(cpu_instances.errors[instance]));
      evm->run(tmp_error);
      delete evm;
      evm = NULL;
    }
    printf("CPU EVM finished\n");
    #endif


    // print the results
    printf("Printing the results ...\n");
    evm_t::print_evm_instances_t(arith, cpu_instances);
    printf("Results printed\n");

    // print to json files
    printf("Printing to json files ...\n");
    cJSON_AddItemToObject(
      write_root,
      test->string,
      evm_t::json_from_evm_instances_t(arith, cpu_instances));
    printf("Json files printed\n");

    // free the memory
    printf("Freeing the memory ...\n");
    evm_t::free_instances(cpu_instances);
    #ifndef ONLY_CPU
    CUDA_CHECK(cudaFree(gpu_instances));
    CUDA_CHECK(cgbn_error_report_free(report));
    CUDA_CHECK(cudaDeviceReset());
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

  // check if the file exists
  std::ifstream file(read_json_filename);
  if (!file) {
    fprintf(stderr, "File '%s' does not exist\n", read_json_filename);
    exit(EXIT_FAILURE);
  }
  run_interpreter(read_json_filename, write_json_filename);
}
