
#include <cjson/cJSON.h>
#include <getopt.h>

#include <CuEVM/core/message.cuh>
#include <CuEVM/evm.cuh>
#include <CuEVM/evm_call_state.cuh>
#include <CuEVM/tracer.cuh>
#include <CuEVM/utils/arith.cuh>
#include <CuEVM/utils/cuda_utils.cuh>
#include <CuEVM/utils/evm_defines.cuh>
#include <CuEVM/utils/evm_utils.cuh>
#include <chrono>
#include <fstream>

__managed__ CuEVM::flatten_state *flatten_state_ptr = nullptr;

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

    size_t size_value;
    cudaDeviceGetLimit(&size_value, cudaLimitStackSize);
    printf("current stack size %zu\n", size_value);
    cudaDeviceGetLimit(&size_value, cudaLimitStackSize);
    printf("current heap size %zu\n", size_value);
    size_t heap_size = (size_t(500) << 20);  // 500MB
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_size));
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 4 * 1024));
    cudaDeviceGetLimit(&size_value, cudaLimitStackSize);
    printf("current stack size %zu\n", size_value);
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
        CUDA_CHECK(cudaMallocManaged(&flatten_state_ptr, num_instances * sizeof(CuEVM::flatten_state)));
        // run the evm
        CuEVM::kernel_evm_multiple_instances<<<num_instances, CuEVM::cgbn_tpi>>>(report, instances_data, num_instances);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        printf("GPU kernel finished\n");
        CGBN_CHECK(report);

        printf("Found %d accounts\n", flatten_state_ptr->no_accounts);

        CuEVM::flatten_state *host_flatten_data = nullptr, *device_flatten_data = nullptr;
        CuEVM::plain_account *host_accounts = nullptr, *device_accounts = nullptr;
        CuEVM::plain_storage *host_storage = nullptr, *device_storage = nullptr;
        auto accounts_size = flatten_state_ptr->no_accounts * sizeof(CuEVM::plain_account);
        auto storage_size = flatten_state_ptr->no_storage_elements * sizeof(CuEVM::plain_storage);
        CUDA_CHECK(cudaMalloc(&device_flatten_data, sizeof(CuEVM::flatten_state)));
        CUDA_CHECK(cudaMalloc(&device_accounts, accounts_size));
        CUDA_CHECK(cudaMalloc(&device_storage, storage_size));
        CUDA_CHECK(cudaMemcpy(&device_flatten_data->accounts, device_accounts, sizeof(device_accounts), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(&device_flatten_data->storage_elements, device_storage, sizeof(device_storage), cudaMemcpyDeviceToDevice));

        CUDA_CHECK(cudaDeviceSynchronize());

        CuEVM::copy_state_kernel<<<1, 1>>>(device_flatten_data);

        CUDA_CHECK(cudaDeviceSynchronize());

        host_flatten_data = (CuEVM::flatten_state *)malloc(sizeof(CuEVM::flatten_state));
        host_accounts = (CuEVM::plain_account *)malloc(accounts_size);
        host_storage = (CuEVM::plain_storage *)malloc(storage_size);

        CUDA_CHECK(cudaMemcpy(host_flatten_data, device_flatten_data, sizeof(CuEVM::flatten_state), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_accounts, device_accounts, accounts_size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_storage, device_storage, storage_size, cudaMemcpyDeviceToHost));

        host_flatten_data->accounts = host_accounts;
        host_flatten_data->storage_elements = host_storage;

        for (auto i =0; i< host_flatten_data->no_accounts; i++){
            printf("Account %d\n", i);
            printf("Address %s\n", host_flatten_data->accounts[i].address);
            printf("Balance %s\n", host_flatten_data->accounts[i].balance);
            printf("Nonce %d\n", host_flatten_data->accounts[i].nonce);
            printf("Code hash %s\n", host_flatten_data->accounts[i].code_hash);
        }

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
        break;
        // run only one test
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
