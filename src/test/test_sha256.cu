#include "../sha256.cuh"
#include "../utils.cu"

#define INPUT_DATA_COUNT 1

__host__ __device__ __forceinline__ void test_sha256(
    typename sha256::sha256_t::sha256_parameters_t *parameters,
    uint8_t *input_data,
    uint8_t *hash_data,
    uint32_t instance)
{
    printf("Test sha256\n");
    sha256::sha256_t *sha256_obj;

    sha256_obj = new sha256::sha256_t(parameters);

    sha256_obj->sha(
        &(input_data[INPUT_DATA_COUNT * instance]),
        INPUT_DATA_COUNT,
        &(hash_data[32 * instance]));

    delete sha256_obj;
    sha256_obj = NULL;
}

template <class params>
__global__ void kernel_sha256(
    cgbn_error_report_t *report,
    typename sha256::sha256_t::sha256_parameters_t *parameters,
    uint8_t *input_data,
    uint8_t *hash_data,
    uint32_t instance_count)
{
    uint32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / params::TPI;

    if (instance >= instance_count)
        return;

    test_sha256(parameters, input_data, hash_data, instance);
}

template <class params>
void run_test()
{
    typedef sha256::sha256_t sha256_t;
    typedef typename sha256_t::sha256_parameters_t sha256_parameters_t;
    sha256_parameters_t *parameters;
    uint32_t instance_count = 1;
    sha256_t *sha256_obj;

#ifndef ONLY_CPU
    CUDA_CHECK(cudaDeviceReset());
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 1024));
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 64 * 1024));
    cgbn_error_report_t *report;
    CUDA_CHECK(cgbn_error_report_alloc(&report));
#endif

    printf("Generating parameters\n");
    sha256_obj = new sha256_t();
    parameters = sha256_obj->_parameters;
    // create a cgbn_error_report for CGBN to report back errors
    printf("Parameters generated\n");

    printf("Generating input data\n");
    uint8_t *cpu_input_data;
    size_t input_data_count = INPUT_DATA_COUNT;
    cpu_input_data = (uint8_t *)malloc(instance_count * input_data_count * sizeof(uint8_t));
    for (uint32_t jdx = 0; jdx < instance_count; jdx++)
    {
        for (uint32_t idx = 0; idx < input_data_count; idx++)
        {
            cpu_input_data[jdx * input_data_count + idx] = 0xFF; //(uint8_t)idx;
        }
    }
#ifndef ONLY_CPU
    uint8_t *gpu_input_data;
    CUDA_CHECK(cudaMalloc(
        (void **)&gpu_input_data,
        instance_count * input_data_count * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpy(
        gpu_input_data,
        cpu_input_data,
        instance_count * input_data_count * sizeof(uint8_t), cudaMemcpyHostToDevice));
#endif
    printf("Input data generated\n");

    printf("Generating hash data\n");
    size_t hash_data_count = 32;
    uint8_t *cpu_hash_data;
    cpu_hash_data = (uint8_t *)malloc(instance_count * hash_data_count * sizeof(uint8_t));
#ifndef ONLY_CPU
    uint8_t *gpu_hash_data;
    CUDA_CHECK(cudaMalloc(
        (void **)&gpu_hash_data,
        instance_count * hash_data_count * sizeof(uint8_t)));
#endif
    printf("Hash data generated\n");

#ifndef ONLY_CPU
    printf("Running GPU kernel ...\n");
    kernel_sha256<params><<<instance_count, params::TPI>>>(
        report,
        parameters,
        gpu_input_data,
        gpu_hash_data,
        instance_count);
    // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
    CUDA_CHECK(cudaDeviceSynchronize());
    CGBN_CHECK(report);

    printf("Get hash data from GPU\n");
    cudaMemcpy(cpu_hash_data, gpu_hash_data, instance_count * hash_data_count * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    printf("Finished getting hash data\n");
#else
    printf("Running CPU kernel ...\n");
    test_sha256(parameters, cpu_input_data, cpu_hash_data, 0);
    printf("Finished running CPU kernel\n");

#endif

    printf("Print input data\n");
    for (uint32_t jdx = 0; jdx < instance_count; jdx++)
    {
        printf("Input data for instance %d\n", jdx);
        for (uint32_t idx = 0; idx < input_data_count; idx++)
        {
            printf("%02x", cpu_input_data[jdx * input_data_count + idx]);
        }
        printf("\n");
    }

    printf("Print hash data\n");
    for (uint32_t jdx = 0; jdx < instance_count; jdx++)
    {
        printf("Hash data for instance %d\n", jdx);
        for (uint32_t idx = 0; idx < hash_data_count; idx++)
        {
            printf("%02x", cpu_hash_data[jdx * hash_data_count + idx]);
        }
        printf("\n");
    }
    printf("Hash data printed\n");

    // free the memory
    printf("Freeing the memory ...\n");
    sha256_obj->free_parameters();
    delete sha256_obj;
    sha256_obj = NULL;
    free(cpu_input_data);
    free(cpu_hash_data);
#ifndef ONLY_CPU
    cudaFree(gpu_input_data);
    cudaFree(gpu_hash_data);
    CUDA_CHECK(cgbn_error_report_free(report));
#endif
    printf("Memory freed\n");
}

int main()
{
    run_test<utils_params>();
}