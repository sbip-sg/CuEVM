#include <cjson/cJSON.h>
#include <getopt.h>

#include <CuEVM/core/memory_pool.cuh>
#include <CuEVM/core/message.cuh>
#include <CuEVM/evm.cuh>
#include <CuEVM/tracer.cuh>
#include <CuEVM/utils/cuda_utils.cuh>
#include <CuEVM/utils/evm_defines.cuh>
#include <CuEVM/utils/evm_utils.cuh>
#include <chrono>
#include <fstream>

void run_interpreter(char *read_json_filename, char *write_json_filename, size_t clones, bool verbose = false) {
    CuEVM::evm_instance_t *instances_data;

    printf("Running the interpreter\n");

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaDeviceReset());
    printf("Running on GPU\n");
    cudaEvent_t start, stop;
    float milliseconds = 0;

    size_t size_value;
    cudaDeviceGetLimit(&size_value, cudaLimitStackSize);
    printf("current stack size %zu\n", size_value);
    cudaDeviceGetLimit(&size_value, cudaLimitStackSize);
    printf("current heap size %zu\n", size_value);
    size_t heap_size = (size_t(100) << 20);  // 100MB
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_size));
    // CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 4 * 1024));
    // cudaDeviceGetLimit(&size_value, cudaLimitStackSize);
    // printf("current stack size %zu\n", size_value);
    CUDA_CHECK(cudaDeviceSynchronize());
    // CUDA_CHECK(cudaEventCreate(&start));
    // CUDA_CHECK(cudaEventCreate(&stop));

    // read the json file with the global state
    cJSON *read_root = CuEVM::utils::get_json_from_file(read_json_filename);
    if (read_root == nullptr) {
        printf("Error: could not read the json file\n");
        exit(EXIT_FAILURE);
    }
    cJSON *write_root = nullptr;
    if (write_json_filename != nullptr) {
        write_root = cJSON_CreateObject();
    }
    uint32_t num_instances = 0;
    int32_t managed = 1;

    const cJSON *test_json = nullptr;
    test_json = cJSON_GetArrayItem(read_root, 0);
    if (test_json != nullptr) {
        CuEVM::get_evm_instances(instances_data, test_json, num_instances, clones, managed);
        CuEVM::memory_pool::create_memory_pool(num_instances);
        uint32_t num_blocks = (num_instances + CGBN_IBP - 1) / (CGBN_IBP);
        printf("\n\n ----------\n\n");
        printf("Running %d instances on GPU, num blocks %d, threads per block %d\n", num_instances, num_blocks,
               CGBN_IBP);
        // run the evm
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        CuEVM::kernel_evm_multiple_instances<<<num_blocks, CGBN_IBP>>>(
            instances_data->state_db_ptr, instances_data->transaction_list_ptr, num_instances);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Kernel execution time: %f milliseconds\n", milliseconds);

        CUDA_CHECK(cudaGetLastError());
        printf("GPU kernel finished\n");
    }

    printf("Freeing the memory ...\n");
    CuEVM::free_evm_instances(instances_data, num_instances, managed);

    CUDA_CHECK(cudaDeviceReset());

    cJSON_Delete(read_root);
    if (write_json_filename != nullptr) {
        char *json_str = cJSON_Print(write_root);
        FILE *fp = fopen(write_json_filename, "w");
        fprintf(fp, "%s", json_str);
        fclose(fp);
        free(json_str);
        cJSON_Delete(write_root);
    }
}

int main(int argc, char *argv[]) {  // getting the input
    char *read_json_filename = NULL;
    char *write_json_filename = NULL;
    size_t clones = 1;
    bool verbose = false;  // Verbose flag
    static struct option long_options[] = {{"input", required_argument, 0, 'i'},
                                           {"output", optional_argument, 0, 'o'},
                                           {"clones", required_argument, 0, 'c'},
                                           {"verbose", no_argument, 0, 'v'},
                                           {0, 0, 0, 0}};

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "i:o:c:v", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'i':
                read_json_filename = optarg;
                break;
            case 'o':
                write_json_filename = optarg;
                break;
            case 'c':
                clones = strtoul(optarg, NULL, 10);
                break;
            case 'v':  // Case for verbose flag
                verbose = true;
                break;
            default:
                fprintf(stdout,
                        "Usage: %s --input <json_filename> --output <json_filename> --clones <number_of_clones> "
                        "[--verbose]\n",
                        argv[0]);
                exit(EXIT_FAILURE);
        }
    }
    if (!read_json_filename) {
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
