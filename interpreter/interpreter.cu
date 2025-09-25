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
#include <vector>

void run_interpreter(char *read_json_filename, char *write_json_filename, size_t clones, bool verbose = false) {
    // CuEVM::evm_instance_t *instances_data;

    printf("Running the interpreter\n");

    int num_gpus = 0;
    cudaError_t err = cudaGetDeviceCount(&num_gpus);
    if (err != cudaSuccess || num_gpus <= 0) {
        printf("Error getting GPU count or no GPUs found: %s. Defaulting to 1 GPU.\n", cudaGetErrorString(err));
        num_gpus = 1;  // Fallback to 1 GPU if detection fails
    }
    printf("Found %d GPUs.\n", num_gpus);
    std::vector<cudaEvent_t> start_events(0);
    std::vector<cudaEvent_t> stop_events(0);
    std::vector<cudaStream_t> streams(0);

    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaDeviceReset());
        printf("Running on GPU %d\n", i);

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        start_events.push_back(start);
        stop_events.push_back(stop);
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
    }

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
// tracer
#ifdef EIP_3155
    const size_t BUFFER_SIZE = 100 * 1024 * 1024;  // 100 MB
    // char *d_buffer;
    std::vector<char *> d_buffers(num_gpus);
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMalloc(&d_buffers[i], BUFFER_SIZE));
    }
#endif
    if (test_json == nullptr) {
        exit(EXIT_FAILURE);
    }
    auto start_cpu = std::chrono::high_resolution_clock::now();
    uint32_t num_accounts = 0;

    auto transaction_list_ptrs = CuEVM::get_evm_instances(test_json, num_instances, num_accounts, num_gpus, clones);

    // Move CPU timing end point here to measure only setup time
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu);
    printf("CPU setup time: %lld milliseconds\n", duration_cpu.count());

    // Launch kernels on each GPU with proper timing

    CuEVM::memory_pool::create_memory_pool(num_instances, num_accounts, num_gpus);
    printf("num_accounts: %d\n", num_accounts);

    uint32_t num_blocks = (num_instances + INSTANCES_PER_BLOCK - 1) / (INSTANCES_PER_BLOCK);

    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        printf("\n ----------\n");
        printf("Running %d instances on GPU %d, num blocks %d, threads per block %d\n", num_instances, i, num_blocks,
               INSTANCES_PER_BLOCK);

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        streams.push_back(stream);

        // Record start event on current device with the stream
        CUDA_CHECK(cudaEventRecord(start_events[i], stream));

        // Launch kernel on current GPU
        CuEVM::kernel_evm_multiple_instances<<<num_blocks, INSTANCES_PER_BLOCK>>>(transaction_list_ptrs[i],
                                                                                  num_instances
#ifdef EIP_3155
                                                                                  ,
                                                                                  d_buffers[i], BUFFER_SIZE
#endif
        );

        // Record stop event on current device
        CUDA_CHECK(cudaEventRecord(stop_events[i], stream));
    }
    // Wait for all GPUs to finish and measure times
    float max_time = 0.0f;
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaEventSynchronize(stop_events[i]));

        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_events[i], stop_events[i]));
        printf("Kernel execution time on GPU %d: %f milliseconds\n", i, milliseconds);

        if (milliseconds > max_time) {
            max_time = milliseconds;
        }
    }

    printf("Total execution time (longest GPU): %f milliseconds\n", max_time);

    // Clean up CUDA events
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaEventDestroy(start_events[i]));
        CUDA_CHECK(cudaEventDestroy(stop_events[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }

    CUDA_CHECK(cudaGetLastError());
    printf("GPU kernel finished\n");

    printf("Freeing the memory ...\n");
    // CuEVM::free_evm_instances(instances_data, num_instances);

#ifdef EIP_3155
    // After kernel execution, copy the buffer back to the host
    char *h_buffer = new char[BUFFER_SIZE];
    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        cudaMemcpy(h_buffer, d_buffers[i], BUFFER_SIZE, cudaMemcpyDeviceToHost);

        // Parse and print the data (implemented later)
        CuEVM::utils::print_tracer_data(h_buffer);
        cudaFree(d_buffers[i]);
    }

    // Clean up
    delete[] h_buffer;

#endif

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
    size_t clones = 32;
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
    if (clones < 32 && clones > 1) clones = 32;
    // check if the file exists
    std::ifstream file(read_json_filename);
    if (!file) {
        fprintf(stdout, "File '%s' does not exist\n", read_json_filename);
        exit(EXIT_FAILURE);
    }
    run_interpreter(read_json_filename, write_json_filename, clones, verbose);

    return 0;
}
