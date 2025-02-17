#include <cjson/cJSON.h>
#include <getopt.h>

#include <CuEVM/core/memory_pool.cuh>
#include <CuEVM/evm.cuh>
#include <CuEVM/tracer.cuh>
#include <CuEVM/utils/cuda_utils.cuh>
#include <CuEVM/utils/evm_defines.cuh>
#include <CuEVM/utils/evm_utils.cuh>
#include <chrono>
#include <fstream>

void run_interpreter(char *read_json_filename, char *write_json_filename, size_t clones, bool verbose = false) {
    // CuEVM::evm_instance_t *instances_data;

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
    size_t heap_size = (size_t(1) << 32);  // 4GB
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_size));
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 4 * 1024));
    cudaDeviceGetLimit(&size_value, cudaLimitStackSize);
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
        auto start_cpu = std::chrono::high_resolution_clock::now();
        uint32_t num_accounts = 0;

        CuEVM::transaction::TransactionList *transaction_list_ptr =
            CuEVM::get_evm_instances(test_json, num_instances, num_accounts, clones);
        CuEVM::memory_pool::create_memory_pool(num_instances, num_accounts);
        printf("num_accounts: %d\n", num_accounts);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        auto duration_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu);
        printf("CPU setup time: %lld milliseconds\n", duration_cpu.count());
        // CuEVM::memory_pool::preallocate_stack(num_instances);
        uint32_t num_blocks = (num_instances + INSTANCES_PER_BLOCK - 1) / (INSTANCES_PER_BLOCK);
        printf("\n\n ----------\n\n");
        printf("Running %d instances on GPU, num blocks %d, threads per block %d\n", num_instances, num_blocks,
               INSTANCES_PER_BLOCK);
        // run the evm
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        CuEVM::kernel_evm_multiple_instances<<<num_blocks, INSTANCES_PER_BLOCK>>>(transaction_list_ptr, num_instances);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Kernel execution time: %f milliseconds\n", milliseconds);

        CUDA_CHECK(cudaGetLastError());
        printf("GPU kernel finished\n");
    }

    printf("Freeing the memory ...\n");
    // CuEVM::free_evm_instances(instances_data, num_instances);

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
    size_t clones = 2;
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

    return 0;
}
