#include "../utils.h"
#include "../keccak.cuh"

#define INPUT_DATA_COUNT 4
#define HASH_DATA_COUNT 32

template<class params>
__global__ void kernel_keccak(cgbn_error_report_t *report, typename keccak::keccak_t::sha3_parameters_t *parameters, uint8_t *input_data, uint8_t *hash_data, uint32_t instance_count) {
    uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;

    if(instance>=instance_count)
        return;
    keccak::keccak_t    keccak_obj(parameters[instance].rndc, parameters[instance].rotc, parameters[instance].piln, parameters[instance].state);

    keccak_obj.sha3(&(input_data[INPUT_DATA_COUNT*instance]), INPUT_DATA_COUNT, &(hash_data[HASH_DATA_COUNT*instance]), HASH_DATA_COUNT);
  
}

template<class params>
void run_test() {
    typedef keccak::keccak_t                keccak_t;
    typedef typename keccak_t::sha3_parameters_t sha3_parameters_t;
    sha3_parameters_t *cpu_parameters, *gpu_parameters;
    cgbn_error_report_t     *report;
    uint32_t instance_count=1;

    printf("Generating parameters\n");
    cpu_parameters=keccak_t::get_cpu_instances(instance_count);
    gpu_parameters=keccak_t::get_gpu_instances(cpu_parameters, instance_count);
    // create a cgbn_error_report for CGBN to report back errors
    CUDA_CHECK(cgbn_error_report_alloc(&report)); 
    printf("Parameters generated\n");

    printf("Generating input data\n");
    uint8_t *cpu_input_data, *gpu_input_data;
    size_t input_data_count=INPUT_DATA_COUNT;
    cpu_input_data=(uint8_t *)malloc(instance_count*input_data_count*sizeof(uint8_t));
    for (uint32_t jdx=0; jdx<instance_count; jdx++) {
        for(uint32_t idx=0; idx<input_data_count; idx++) {
            cpu_input_data[jdx*input_data_count+idx]=0xFF;//(uint8_t)idx;
        }
    }
    cudaMalloc((void **)&gpu_input_data, instance_count*input_data_count*sizeof(uint8_t));
    cudaMemcpy(gpu_input_data, cpu_input_data, instance_count*input_data_count*sizeof(uint8_t), cudaMemcpyHostToDevice);
    printf("Input data generated\n");

    printf("Generating hash data\n");
    size_t hash_data_count=HASH_DATA_COUNT;
    uint8_t *cpu_hash_data, *gpu_hash_data;
    cpu_hash_data=(uint8_t *)malloc(instance_count*hash_data_count*sizeof(uint8_t));
    cudaMalloc((void **)&gpu_hash_data, instance_count*hash_data_count*sizeof(uint8_t));
    printf("Hash data generated\n");

    printf("Running GPU kernel ...\n");
    kernel_keccak<params><<<instance_count, params::TPI>>>(report, gpu_parameters, gpu_input_data, gpu_hash_data, 1);
    // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
    CUDA_CHECK(cudaDeviceSynchronize());
    CGBN_CHECK(report);

    printf("Get hash data from GPU\n");
    cudaMemcpy(cpu_hash_data, gpu_hash_data, instance_count*hash_data_count*sizeof(uint8_t), cudaMemcpyDeviceToHost);
    printf("Finished getting hash data\n");

    printf("Print input data\n");
    for (uint32_t jdx=0; jdx<instance_count; jdx++) {
        printf("Input data for instance %d\n", jdx);
        for(uint32_t idx=0; idx<input_data_count; idx++) {
            printf("%02x", cpu_input_data[jdx*input_data_count+idx]);
        }
        printf("\n");
    }

    printf("Print hash data\n");
    for (uint32_t jdx=0; jdx<instance_count; jdx++) {
        printf("Hash data for instance %d\n", jdx);
        for(uint32_t idx=0; idx<hash_data_count; idx++) {
            printf("%02x", cpu_hash_data[jdx*hash_data_count+idx]);
        }
        printf("\n");
    }
    printf("Hash data printed\n");

    // free the memory
    printf("Freeing the memory ...\n");
    keccak_t::free_cpu_instances(cpu_parameters, instance_count);
    keccak_t::free_gpu_instances(gpu_parameters, instance_count);
    free(cpu_input_data);
    cudaFree(gpu_input_data);
    free(cpu_hash_data);
    cudaFree(gpu_hash_data);
    CUDA_CHECK(cgbn_error_report_free(report)); 
    printf("Memory freed\n");
}

int main() {
  run_test<utils_params>();
}