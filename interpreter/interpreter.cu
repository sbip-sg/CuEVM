#include <CGBN/cgbn.h>
#include <cjson/cJSON.h>
#include <getopt.h>

#include <CuEVM/evm.cuh>
#include <CuEVM/tracer.cuh>
#include <CuEVM/utils/arith.cuh>
#include <CuEVM/utils/cuda_utils.cuh>
#include <CuEVM/utils/evm_defines.cuh>
#include <CuEVM/utils/evm_utils.cuh>
#include <chrono>
#include <fstream>

// define the kernel function
__global__ void kernel_evm(cgbn_error_report_t *report, CuEVM::evm_instance_t *instances, uint32_t count) {
    int32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / CuEVM::cgbn_tpi;
    if (instance >= count) return;
    CuEVM::ArithEnv arith(cgbn_no_checks, report, instance);
    CuEVM::bn_t test;

// printf("new instance %d\n", instance);
#ifdef EIP_3155
    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    printf("instance %d\n", instance);
    printf("world state\n");
    instances[instance].world_state_data_ptr->print();
    printf("touch state\n");
    instances[instance].touch_state_data_ptr->print();
    printf("instance %d\n", instance);
    printf("transaction\n");
    instances[instance].transaction_ptr->print();
    __ONE_GPU_THREAD_WOSYNC_END__
#endif
    CuEVM::evm_t *evm = new CuEVM::evm_t(arith, instances[instance]);
    // printf("\nevm->run(arith) instance %d\n", instance);
    __SYNC_THREADS__
    evm->run(arith);
#ifdef EIP_3155
    if (instance == 0) {
        __ONE_GPU_THREAD_BEGIN__
        // instances[0].tracer_ptr->print(arith);
        instances[0].tracer_ptr->print_err();
        __ONE_GPU_THREAD_WOSYNC_END__
    }
#endif
    // delete evm;
    // evm = nullptr;
}

void run_interpreter(char *read_json_filename, char *write_json_filename, size_t clones, bool verbose = false) {
    CuEVM::evm_instance_t *instances_data;
    CuEVM::ArithEnv arith(cgbn_no_checks, 0);
    printf("Running the interpreter\n");
#ifdef GPU
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaDeviceReset());
    printf("Running on GPU\n");
    cgbn_error_report_t *report;
    CUDA_CHECK(cgbn_error_report_alloc(&report));
    cudaEvent_t start, stop;
    float milliseconds = 0;

    size_t stack_size;
    cudaDeviceGetLimit(&stack_size, cudaLimitStackSize);
    printf("current stack size %zu\n", stack_size);
    size_t heap_size = (size_t(2) << 30);  // 2GB
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_size));
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 90 * 1024));
    cudaDeviceGetLimit(&stack_size, cudaLimitStackSize);
    printf("current stack size %zu\n", stack_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    // CUDA_CHECK(cudaEventCreate(&start));
    // CUDA_CHECK(cudaEventCreate(&stop));
#endif

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
    int32_t managed = 0;
#ifdef GPU
    managed = 1;
#endif

    const cJSON *test_json = nullptr;
    cJSON_ArrayForEach(test_json, read_root) {
        CuEVM::get_evm_instances(arith, instances_data, test_json, num_instances, managed);

#ifdef GPU
        // TODO remove DEBUG num instances
        // num_instances = 1;
        printf("Running on GPU %d %d\n", num_instances, CuEVM::cgbn_tpi);
        // run the evm
        kernel_evm<<<num_instances, CuEVM::cgbn_tpi>>>(report, instances_data, num_instances);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        printf("GPU kernel finished\n");
        CGBN_CHECK(report);
#ifdef EIP_3155
        // print only the first instance

        // CuEVM::utils::print_err_device_data(instances_data[0].tracer_ptr);
#endif
        // CUDA_CHECK(cudaEventRecord(stop));
        // CUDA_CHECK(cudaEventSynchronize(stop));
        // CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
#else
        printf("Running CPU EVM\n");
        // run the evm
        CuEVM::evm_t *evm = nullptr;
        cJSON *final_state = nullptr;
        auto cpu_start = std::chrono::high_resolution_clock::now();
        for (uint32_t instance = 0; instance < num_instances; instance++) {
            evm = new CuEVM::evm_t(arith, instances_data[instance]);
            evm->run(arith);
#ifdef EIP_3155
            evm->tracer_ptr->print_err();
#endif
            // printf("DEBUG: CPU EVM instance %d finished - START\n", instance);
            // printf("DEBUG: CPU EVM instance %d world state\n", instance);
            // instances_data[instance].world_state_data_ptr->print();
            // printf("DEBUG: CPU EVM instance %d touch state\n", instance);
            // instances_data[instance].touch_state_data_ptr->print();
            // printf("DEBUG: CPU EVM instance %d access state\n", instance);
            // instances_data[instance].access_state_data_ptr->print();
            // printf("DEBUG: CPU EVM instance %d finished - END\n", instance);
            final_state = CuEVM::state_access_t::merge_json(*instances_data[instance].world_state_data_ptr,
                                                            *instances_data[instance].touch_state_data_ptr);
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
    CUDA_CHECK(cgbn_error_report_free(report));
    CUDA_CHECK(cudaDeviceReset());
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
