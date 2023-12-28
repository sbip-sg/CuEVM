#include "../utils.h"
#include "../keccak.cuh"

#define INPUT_DATA_COUNT 4
#define HASH_DATA_COUNT 32

__host__ __device__ __forceinline__ void test_keccak(
    typename keccak::keccak_t::sha3_parameters_t *parameters,
    uint8_t *input_data,
    uint8_t *hash_data,
    uint32_t instance)
{
    printf("Test keccak\n");
    keccak::keccak_t *keccak_obj;

    keccak_obj = new keccak::keccak_t(parameters);

    keccak_obj->sha3(
        &(input_data[INPUT_DATA_COUNT * instance]),
        INPUT_DATA_COUNT,
        &(hash_data[HASH_DATA_COUNT * instance]),
        HASH_DATA_COUNT);

    delete keccak_obj;
    keccak_obj = NULL;
}

template <class params>
__global__ void kernel_keccak(
    cgbn_error_report_t *report,
    typename keccak::keccak_t::sha3_parameters_t *parameters,
    uint8_t *input_data,
    uint8_t *hash_data,
    uint32_t instance_count)
{
    uint32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / params::TPI;

    if (instance >= instance_count)
        return;

    test_keccak(parameters, input_data, hash_data, instance);
}

template <class params>
void run_test()
{
    typedef keccak::keccak_t keccak_t;
    typedef typename keccak_t::sha3_parameters_t sha3_parameters_t;
    sha3_parameters_t *parameters;
    uint32_t instance_count = 1;
    keccak_t *keccak_obj;

#ifndef ONLY_CPU
    CUDA_CHECK(cudaDeviceReset());
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 1024));
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 64 * 1024));
    cgbn_error_report_t *report;
    CUDA_CHECK(cgbn_error_report_alloc(&report));
#endif

    printf("Generating parameters\n");
    keccak_obj = new keccak_t();
    parameters = keccak_obj->_parameters;
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
    size_t hash_data_count = HASH_DATA_COUNT;
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
    kernel_keccak<params><<<instance_count, params::TPI>>>(
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
    test_keccak(parameters, cpu_input_data, cpu_hash_data, 0);
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
    keccak_obj->free_parameters();
    delete keccak_obj;
    keccak_obj = NULL;
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