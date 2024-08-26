#include <getopt.h>
#include <fstream>
#include <chrono>
#include "include/utils/arith.cuh"
#include "include/utils/cuda_utils.cuh"
#include "include/utils/evm_utils.cuh"
#include "include/evm.cuh"


void run_interpreter(char *read_json_filename, char *write_json_filename, size_t clones, bool verbose=false) {
  cuEVM::evm_instance_t *cpu_instances_data;
  cuEVM::ArithEnv arith(cgbn_report_monitor, 0);
  printf("Running the interpreter\n");

  //read the json file with the global state
  cJSON *read_root = cuEVM::utils::get_json_from_file(read_json_filename);
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
  
  const cJSON *test_json = nullptr;
  cJSON_ArrayForEach(test_json, read_root) {
    cuEVM::get_cpu_evm_instances(arith, cpu_instances_data, test_json, num_instances);


    const cJSON *test = nullptr;
    printf("Running CPU EVM\n");
    // run the evm
    cuEVM::evm_t *evm = nullptr;
    uint32_t tmp_error;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for(uint32_t instance = 0; instance < num_instances; instance++) {
      // printf("Running instance %d\n", instance);
      evm = new cuEVM::evm_t(arith, cpu_instances_data[instance]);
      evm->run(arith);
      delete evm;
      evm = nullptr;
    }
    printf("CPU EVM finished\n");
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
    printf("CPU EVM execution took %f ms\n", cpu_duration.count());
  }

  
  printf("Freeing the memory ...\n");
  cuEVM::free_cpu_evm_instances(cpu_instances_data, num_instances);
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
