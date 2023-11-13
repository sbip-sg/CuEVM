#ifndef _TRACER_T_H_
#define _TRACER_T_H_

#include "utils.h"
#include "stack.cuh"


template<class params>
class tracer_t {
    public:
    typedef arith_env_t<params>                     arith_t;
    typedef typename arith_t::bn_t                  bn_t;
    typedef cgbn_mem_t<params::BITS>                evm_word_t;
    typedef stack_t<params>                         stack_t;
    typedef typename stack_t::stack_data_t          stack_data_t;
    static const size_t                             PAGE_SIZE=128;

    typedef struct {
        evm_word_t      address;
        uint32_t        pc;
        uint8_t         opcode;
        stack_data_t    stack;
    } tracer_data_t;

    typedef struct {
        size_t          size;
        size_t          capacity;
        tracer_data_t   *data;
    } tracer_content_t;

    tracer_content_t *_content;
    arith_t     _arith;

    __host__ __device__ __forceinline__ tracer_t(arith_t arith, tracer_content_t *content) : _arith(arith),  _content(content) {
    }

    __host__ __device__ __forceinline__ void grow() {
        _content->capacity += PAGE_SIZE;
        #ifdef __CUDA_ARCH__
        __syncthreads();
        if (threadIdx.x == 0) {
        #endif
            tracer_data_t *new_data = (tracer_data_t *)malloc(sizeof(tracer_data_t) * _content->capacity);
            memcpy(new_data, _content->data, sizeof(tracer_data_t) * _content->size);
            if (_content->data != NULL) {
                free(_content->data);
            }
            _content->data = new_data;
        #ifdef __CUDA_ARCH__
        }
        __syncthreads();
        #endif
    }

    __host__ __device__ __forceinline__ void push(bn_t &address, uint32_t pc, uint8_t opcode, stack_t *stack) {
        if (_content->size == _content->capacity) {
            grow();
        }
        cgbn_store(_arith._env, &(_content->data[_content->size].address), address);
        _content->data[_content->size].pc = pc;
        _content->data[_content->size].opcode = opcode;
        stack->copy_stack_data(&_content->data[_content->size].stack, 1);
        _content->size++;
    }

    __host__ __device__ void print() {
        printf("Tracer:\n");
        for(size_t idx=0; idx<_content->size;idx++) {
            printf("Address: ");
            print_bn<params>(_content->data[idx].address);
            printf("\n");
            printf("PC: %d, OPCODE: %02x\n", _content->data[idx].pc, _content->data[idx].opcode);
            stack_t local_stack(_arith, &_content->data[idx].stack);
            local_stack.print();
        }
    }

    __host__ cJSON *to_json() {
        char hex_string[67]="0x";
        mpz_t address;
        mpz_init(address);
        cJSON *tracer_json = cJSON_CreateArray();
        for(size_t idx=0; idx<_content->size;idx++) {
            cJSON *item = cJSON_CreateObject();
            to_mpz(address, _content->data[idx].address._limbs, params::BITS/32);
            strcpy(hex_string+2, mpz_get_str(NULL, 16, address));
            cJSON_AddStringToObject(item, "address", hex_string);
            cJSON_AddNumberToObject(item, "pc", _content->data[idx].pc);
            cJSON_AddNumberToObject(item, "opcode", _content->data[idx].opcode);
            stack_t local_stack(_arith, &_content->data[idx].stack);
            cJSON_AddItemToObject(item, "stack", local_stack.to_json());
            cJSON_AddItemToArray(tracer_json, item);
        }
        return tracer_json;
    }

    __host__ static tracer_content_t *get_tracers(uint32_t count) {
        tracer_content_t *cpu_tracers = (tracer_content_t *)malloc(sizeof(tracer_content_t) * count);
        for(uint32_t idx=0; idx<count; idx++) {
            cpu_tracers[idx].data = NULL;
            cpu_tracers[idx].size = 0;
            cpu_tracers[idx].capacity = 0;
        }
        return cpu_tracers;
    }

    __host__ static tracer_content_t *get_gpu_tracers(tracer_content_t *cpu_tracers, uint32_t count) {
        tracer_content_t *gpu_tracers;
        cudaMalloc(&gpu_tracers, sizeof(tracer_content_t) * count);
        cudaMemcpy(gpu_tracers, cpu_tracers, sizeof(tracer_content_t) * count, cudaMemcpyHostToDevice);
        return gpu_tracers;
    }

    __host__ static void free_tracers(tracer_content_t *cpu_tracers, uint32_t count) {
        for(uint32_t idx=0; idx<count; idx++) {
            if (cpu_tracers[idx].data != NULL) {
                for (size_t jdx=0; jdx<cpu_tracers[idx].size; jdx++) {
                    if (cpu_tracers[idx].data[jdx].stack.stack_base != NULL) {
                        free(cpu_tracers[idx].data[jdx].stack.stack_base);
                    }
                }
                free(cpu_tracers[idx].data);
            }
        }
        free(cpu_tracers);
    }

    

    __host__ static void free_gpu_tracers(tracer_content_t *gpu_tracers, uint32_t count) {
        tracer_content_t *tmp_cpu_tracers;
        tmp_cpu_tracers = (tracer_content_t *)malloc(count*sizeof(tracer_content_t));
        cudaMemcpy(tmp_cpu_tracers, gpu_tracers, count*sizeof(tracer_content_t), cudaMemcpyDeviceToHost);
        for (size_t idx=0; idx<count; idx++) {
            if (tmp_cpu_tracers[idx].data != NULL) {
                tracer_data_t *tmp_cpu_data;
                tmp_cpu_data = (tracer_data_t *)malloc(tmp_cpu_tracers[idx].size*sizeof(tracer_data_t));
                cudaMemcpy(tmp_cpu_data, tmp_cpu_tracers[idx].data, tmp_cpu_tracers[idx].size*sizeof(tracer_data_t), cudaMemcpyDeviceToHost);
                for (size_t jdx=0; jdx<tmp_cpu_tracers[idx].size; jdx++) {
                    if (tmp_cpu_data[jdx].stack.stack_base != NULL) {
                        cudaFree(tmp_cpu_data[jdx].stack.stack_base);
                    }
                }
                cudaFree(tmp_cpu_tracers[idx].data);
                free(tmp_cpu_data);
            }
        }
        free(tmp_cpu_tracers);
        cudaFree(gpu_tracers);
    }
    

    __host__ static tracer_content_t *get_cpu_tracers_from_gpu(tracer_content_t *gpu_tracers, uint32_t count) {
        // STATE 1.1 I can only see the data values and number of data
        tracer_content_t *cpu_tracers;
        cpu_tracers = (tracer_content_t *)malloc(count*sizeof(tracer_content_t));
        cudaMemcpy(cpu_tracers, gpu_tracers, count*sizeof(tracer_content_t), cudaMemcpyDeviceToHost);
        // STATE 1.2 I can alocate the data array
        tracer_content_t *new_gpu_tracers, *tmp_cpu_tracers;
        tmp_cpu_tracers = (tracer_content_t *)malloc(count*sizeof(tracer_content_t));
        memcpy(tmp_cpu_tracers, cpu_tracers, count*sizeof(tracer_content_t));
        for (size_t idx=0; idx<count; idx++) {
            tmp_cpu_tracers[idx].capacity = tmp_cpu_tracers[idx].size;
            if (tmp_cpu_tracers[idx].data != NULL) {
                cudaMalloc((void **)&(tmp_cpu_tracers[idx].data), tmp_cpu_tracers[idx].size*sizeof(tracer_data_t));
            }
        }
        cudaMalloc((void **)&new_gpu_tracers, count*sizeof(tracer_content_t));
        cudaMemcpy(new_gpu_tracers, tmp_cpu_tracers, count*sizeof(tracer_content_t), cudaMemcpyHostToDevice);
        free(tmp_cpu_tracers);
        // STATE 1.3 call the kernel
        kernel_get_tracers_S1<params><<<1, count>>>(new_gpu_tracers, gpu_tracers, count);
        // STATE 1.4 free unnecasry memory
        cudaFree(gpu_tracers);
        gpu_tracers = new_gpu_tracers;

        // STATE 2.1 copy the data array
        cudaMemcpy(cpu_tracers, gpu_tracers, count*sizeof(tracer_content_t), cudaMemcpyDeviceToHost);
        // STATE 2.2 allocate the data array
        tmp_cpu_tracers = (tracer_content_t *)malloc(count*sizeof(tracer_content_t));
        memcpy(tmp_cpu_tracers, cpu_tracers, count*sizeof(tracer_content_t));
        for (size_t idx=0; idx<count; idx++) {
            if (tmp_cpu_tracers[idx].data != NULL) {
                tracer_data_t *tmp_cpu_data;
                tmp_cpu_data = (tracer_data_t *)malloc(tmp_cpu_tracers[idx].size*sizeof(tracer_data_t));
                cudaMemcpy(tmp_cpu_data, tmp_cpu_tracers[idx].data, tmp_cpu_tracers[idx].size*sizeof(tracer_data_t), cudaMemcpyDeviceToHost);
                for (size_t jdx=0; jdx<tmp_cpu_tracers[idx].size; jdx++) {
                    if (tmp_cpu_data[jdx].stack.stack_offset > 0) {
                        cudaMalloc((void **)&(tmp_cpu_data[jdx].stack.stack_base), tmp_cpu_data[jdx].stack.stack_offset*sizeof(evm_word_t));
                    } else {
                        tmp_cpu_data[jdx].stack.stack_base = NULL;
                    }
                }
                cudaMalloc((void **)&(tmp_cpu_tracers[idx].data), tmp_cpu_tracers[idx].size*sizeof(tracer_data_t));
                cudaMemcpy(tmp_cpu_tracers[idx].data, tmp_cpu_data, tmp_cpu_tracers[idx].size*sizeof(tracer_data_t), cudaMemcpyHostToDevice);
                free(tmp_cpu_data);
            }
        }
        cudaMalloc((void **)&new_gpu_tracers, count*sizeof(tracer_content_t));
        cudaMemcpy(new_gpu_tracers, tmp_cpu_tracers, count*sizeof(tracer_content_t), cudaMemcpyHostToDevice);
        free(tmp_cpu_tracers);
        // STATE 2.3 call the kernel
        kernel_get_tracers_S2<params><<<1, count>>>(new_gpu_tracers, gpu_tracers, count);
        // STATE 2.4 free unnecasry memory
        for (size_t idx=0; idx<count; idx++) {
            if (cpu_tracers[idx].data != NULL) {
                cudaFree(cpu_tracers[idx].data);
            }
        }
        cudaFree(gpu_tracers);
        gpu_tracers = new_gpu_tracers;

        // STATE 3.1 copy the data array
        cudaMemcpy(cpu_tracers, gpu_tracers, count*sizeof(tracer_content_t), cudaMemcpyDeviceToHost);
        // STATE 3.2 allocate the data array
        tmp_cpu_tracers = (tracer_content_t *)malloc(count*sizeof(tracer_content_t));
        memcpy(tmp_cpu_tracers, cpu_tracers, count*sizeof(tracer_content_t));
        for (size_t idx=0; idx<count; idx++) {
            if (tmp_cpu_tracers[idx].data != NULL) {
                tracer_data_t *tmp_cpu_data, *aux_tmp_cpu_data;
                tmp_cpu_data = (tracer_data_t *)malloc(tmp_cpu_tracers[idx].size*sizeof(tracer_data_t));
                aux_tmp_cpu_data = (tracer_data_t *)malloc(tmp_cpu_tracers[idx].size*sizeof(tracer_data_t));
                cudaMemcpy(tmp_cpu_data, tmp_cpu_tracers[idx].data, tmp_cpu_tracers[idx].size*sizeof(tracer_data_t), cudaMemcpyDeviceToHost);
                cudaMemcpy(aux_tmp_cpu_data, tmp_cpu_tracers[idx].data, tmp_cpu_tracers[idx].size*sizeof(tracer_data_t), cudaMemcpyDeviceToHost);
                for (size_t jdx=0; jdx<tmp_cpu_tracers[idx].size; jdx++) {
                    if (tmp_cpu_data[jdx].stack.stack_offset > 0) {
                        tmp_cpu_data[jdx].stack.stack_base = (evm_word_t *)malloc(tmp_cpu_data[jdx].stack.stack_offset*sizeof(evm_word_t));
                        cudaMemcpy(tmp_cpu_data[jdx].stack.stack_base, aux_tmp_cpu_data[jdx].stack.stack_base, tmp_cpu_data[jdx].stack.stack_offset*sizeof(evm_word_t), cudaMemcpyDeviceToHost);
                    } else {
                        tmp_cpu_data[jdx].stack.stack_base = NULL;
                    }
                }
                free(aux_tmp_cpu_data);
                tmp_cpu_tracers[idx].data = tmp_cpu_data;
            }
        }
        // STATE 3.3 free gpu local states
        free_gpu_tracers(gpu_tracers, count);
        // STATE 3.4 copy to cpu final
        memcpy(cpu_tracers, tmp_cpu_tracers, count*sizeof(tracer_content_t));
        free(tmp_cpu_tracers);
        return cpu_tracers;
    }


};

template<class params>
__global__ void kernel_get_tracers_S1(typename tracer_t<params>::tracer_content_t *dst_instances, typename tracer_t<params>::tracer_content_t *src_instances, uint32_t instance_count) {
    uint32_t instance=blockIdx.x*blockDim.x + threadIdx.x;
    typedef typename tracer_t<params>::tracer_data_t tracer_data_t;

    if(instance>=instance_count)
        return;

    if (src_instances[instance].data != NULL) {
        memcpy(dst_instances[instance].data, src_instances[instance].data, src_instances[instance].size*sizeof(tracer_data_t));
        free(src_instances[instance].data);
    }
}


template<class params>
__global__ void kernel_get_tracers_S2(typename tracer_t<params>::tracer_content_t *dst_instances, typename tracer_t<params>::tracer_content_t *src_instances, uint32_t instance_count) {
    uint32_t instance=blockIdx.x*blockDim.x + threadIdx.x;
    typedef typename tracer_t<params>::tracer_data_t tracer_data_t;
    typedef cgbn_mem_t<params::BITS>                evm_word_t;

    if(instance>=instance_count)
        return;

    if (src_instances[instance].data != NULL) {
        for(size_t idx=0; idx<src_instances[instance].size; idx++) {
            if (src_instances[instance].data[idx].stack.stack_base != NULL) {
                memcpy(dst_instances[instance].data[idx].stack.stack_base, src_instances[instance].data[idx].stack.stack_base, src_instances[instance].data[idx].stack.stack_offset*sizeof(evm_word_t));
                free(src_instances[instance].data[idx].stack.stack_base);
            }
        }
    }
}


#endif