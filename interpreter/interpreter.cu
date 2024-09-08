#include <getopt.h>
#include <fstream>
#include <chrono>
#include <CGBN/cgbn.h>
#include <CuEVM/utils/arith.cuh>
#include <CuEVM/utils/cuda_utils.cuh>
#include <CuEVM/utils/evm_utils.cuh>
#include <CuEVM/utils/evm_defines.cuh>
#include <CuEVM/evm.cuh>
#include <cjson/cJSON.h>


// define the kernel function
__global__ void kernel_evm(cgbn_error_report_t *report, CuEVM::evm_instance_t *instances, uint32_t count) {
  int32_t instance = (blockIdx.x*blockDim.x+threadIdx.x) / CuEVM::cgbn_tpi;
  if(instance >= count)
    return;
  CuEVM::ArithEnv arith(cgbn_no_checks, report, instance);
  CuEVM::evm_t *evm = nullptr;
  evm = new CuEVM::evm_t(arith, instances[instance]);
  evm->run(arith);
  #ifdef EIP_3155
  evm->tracer_ptr->print_err();
  #endif
  delete evm;
  evm = nullptr;
}

void run_interpreter(char *read_json_filename, char *write_json_filename, size_t clones, bool verbose=false) {
  CuEVM::evm_instance_t *instances_data;
  CuEVM::ArithEnv arith(cgbn_no_checks, 0);
  printf("Running the interpreter\n");
  #ifdef GPU
  printf("Running on GPU\n");
  cgbn_error_report_t *report;
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaDeviceReset());
  cudaEvent_t start, stop;
  float milliseconds = 0;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  #endif

  //read the json file with the global state
  cJSON *read_root = CuEVM::utils::get_json_from_file(read_json_filename);
  if(read_root == nullptr) {
    printf("Error: could not read the json file\n");
    exit(EXIT_FAILURE);
  }
  cJSON *write_root = nullptr;
  if (write_json_filename != nullptr) {
    write_root = cJSON_CreateObject();
  }
  int32_t error_code = 0;
  uint32_t num_instances = 0;
  int32_t managed = 0;
  #ifdef GPU
  managed = 1;
  #endif
  
  const cJSON *test_json = nullptr;
  cJSON_ArrayForEach(test_json, read_root) {
    CuEVM::get_evm_instances(arith, instances_data, test_json, num_instances, managed);

    #ifdef GPU
      printf("Running on GPU\n");
      // run the evm
      kernel_evm<<<num_instances, CuEVM::cgbn_tpi>>>(report, instances_data, num_instances);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaGetLastError());
      printf("GPU kernel finished\n");
      CGBN_CHECK(report);
      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));
      CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    #else
      const cJSON *test = nullptr;
      printf("Running CPU EVM\n");
      // run the evm
      CuEVM::evm_t *evm = nullptr;
      uint32_t tmp_error;
      cJSON* final_state = nullptr;
      auto cpu_start = std::chrono::high_resolution_clock::now();
      for(uint32_t instance = 0; instance < num_instances; instance++) {
        evm = new CuEVM::evm_t(arith, instances_data[instance]);
        evm->run(arith);
        #ifdef EIP_3155
        evm->tracer_ptr->print_err();
        #endif
        // printf("DEBUG: CPU EVM instance %d finished - START\n", instance);
        // printf("DEBUG: CPU EVM instance %d world state\n", instance);
        // cpu_instances_data[instance].world_state_data_ptr->print();
        // printf("DEBUG: CPU EVM instance %d touch state\n", instance);
        // cpu_instances_data[instance].touch_state_data_ptr->print();
        // printf("DEBUG: CPU EVM instance %d access state\n", instance);
        // cpu_instances_data[instance].access_state_data_ptr->print();
        // printf("DEBUG: CPU EVM instance %d finished - END\n", instance);
        final_state = CuEVM::state::state_merge_json(
          *instances_data[instance].world_state_data_ptr,
          *instances_data[instance].touch_state_data_ptr
        );
        char *final_state_root_json_str = cJSON_PrintUnformatted(final_state);
        fprintf(stderr, "%s\n", final_state_root_json_str);
        cJSON_Delete(final_state);
        free(final_state_root_json_str);
        delete evm;
        evm = nullptr;
    }
    #endif
    #ifdef GPU
      printf("GPU EVM finished\n");
      printf("Main GPU kernel execution took %f ms\n", milliseconds);
    #else
      printf("CPU EVM finished\n");
      auto cpu_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
      printf("CPU EVM execution took %f ms\n", cpu_duration.count());
    #endif
  }


  printf("Freeing the memory ...\n");
  CuEVM::free_evm_instances(instances_data, num_instances, managed);

  #ifdef GPU
      CUDA_CHECK(cudaDeviceReset());
  #endif

  
  cJSON_Delete(read_root);
  if (write_json_filename != nullptr){
    char *json_str=cJSON_Print(write_root);
    FILE *fp=fopen(write_json_filename, "w");
    fprintf(fp, "%s", json_str);
    fclose(fp);
    free(json_str);
    cJSON_Delete(write_root);
  }
}

int main(int argc, char *argv[]) {//getting the input
  char *read_json_filename = NULL;
  char *write_json_filename = NULL;
  size_t clones = 1;
  bool verbose = false; // Verbose flag
  static struct option long_options[] = {
        {"input", required_argument, 0, 'i'},
        {"output", optional_argument, 0, 'o'},
        {"clones", required_argument, 0, 'c'},
        {"verbose", no_argument, 0, 'v'},
        {0, 0, 0, 0}};

  int opt;
  int option_index = 0;
  while ((opt = getopt_long(argc, argv, "i:o:c:v", long_options, &option_index)) != -1) {
      switch (opt)
      {
      case 'i':
          read_json_filename = optarg;
          break;
      case 'o':
          write_json_filename = optarg;
          break;
      case 'c':
            clones = strtoul(optarg, NULL, 10);
            break;
      case 'v': // Case for verbose flag
          verbose = true;
          break;
      default:
          fprintf(stdout, "Usage: %s --input <json_filename> --output <json_filename> --clones <number_of_clones> [--verbose]\n", argv[0]);
          exit(EXIT_FAILURE);
      }
  }
  if (!read_json_filename)
  {
      fprintf(stdout, "--input argument is required\n");
      exit(EXIT_FAILURE);
  }

  // check if the file exists
  std::ifstream file(read_json_filename);
  if (!file) {
    fprintf(stdout, "File '%s' does not exist\n", read_json_filename);
    exit(EXIT_FAILURE);
  }
  run_interpreter(read_json_filename, write_json_filename, clones, verbose);
}
