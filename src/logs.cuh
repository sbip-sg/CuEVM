// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#ifndef _LOGS_T_H_
#define _LOGS_T_H_

#include "include/utils.h"



/**
 * Class to represent the log state.
 * The log state is the state which 
 * contains the log recorded
 * by the execution of the transaction.
 * YP: accrued transaction substate
 *  \f$A_{l}\f$ for log series
*/
template <class params>
class log_state_t
{
public:
    /**
     * The arithmetical environment used by the arbitrary length
     * integer library.
    */
    typedef arith_env_t<params> arith_t;
    /**
     * The arbitrary length integer type.
    */
    typedef typename arith_t::bn_t bn_t;
    /**
     * The arbitrary length integer type used for the storage.
     * It is defined as the EVM word type.
    */
    typedef cgbn_mem_t<params::BITS> evm_word_t;
    /**
     * The log data type for one log.
    */
    typedef struct
    {
        evm_word_t address; /**< The address of the code executing */
        data_content_t record; /**< The record of the log */
        evm_word_t topics[4]; /**< The topics of the log */
        uint32_t no_topics; /**< The number of topics */
    } log_data_t;
    /**
     * The log state data type. Contains the logs.
    */
    typedef struct
    {
        log_data_t *logs; /**< The logs */
        uint32_t no_logs; /**< The number of logs */
    } log_state_data_t;

    static const uint32_t LOG_PAGE_SIZE = 20; /**< The write operation for log */

    log_state_data_t *_content; /**< The content of the touch state */
    arith_t _arith;             /**< The arithmetical environment */
    uint32_t _allocated_size;   /**< The allocated size */


    /**
     * Constructor with given content.
     * @param[in] arith The arithmetical environment
     * @param[in] content The content of the log state
    */
    __host__ __device__ __forceinline__ log_state_t(
        arith_t &arith,
        log_state_data_t *content
    ) : _arith(arith),
        _content(content),
        _allocated_size(content->no_logs)
    {
    }

    /**
     * Constructor without content.
     * @param[in] arith The arithmetical environment
    */
    __host__ __device__ __forceinline__ log_state_t(
        arith_t &arith
    ) : _arith(arith)
    {
        // aloocate the memory for the log state
        // and initialize it
        SHARED_MEMORY log_state_data_t *tmp_content;
        ONE_THREAD_PER_INSTANCE(
            tmp_content = new log_state_data_t;
            tmp_content->no_logs = 0;
            tmp_content->logs = new log_data_t[LOG_PAGE_SIZE];
        )
        _allocated_size = LOG_PAGE_SIZE;
        _content = tmp_content;
    }

    /**
     * The destructor.
     * It frees the memory allocated for the log state.
    */
    __host__ __device__ __forceinline__ ~log_state_t()
    {
        ONE_THREAD_PER_INSTANCE(
            if (_content != NULL)
            {
                if (_allocated_size > 0)
                {
                    for (uint32_t idx = 0; idx < _content->no_logs; idx++)
                    {
                        if (_content->logs[idx].record.size > 0)
                        {
                            delete[] _content->logs[idx].record.data;
                            _content->logs[idx].record.data = NULL;
                            _content->logs[idx].record.size = 0;
                        }
                    }
                    delete[] _content->logs;
                    _content->logs = NULL;
                    _content->no_logs = 0;
                }
                delete _content;
            }
        )
        _allocated_size = 0;
        _content = NULL;
    }

    __host__ __device__ __forceinline__ void grow()
    {
        ONE_THREAD_PER_INSTANCE(
            log_data_t *tmp_logs = new log_data_t[_allocated_size + LOG_PAGE_SIZE];
            if (_allocated_size > 0)
            {
                memcpy(
                    tmp_logs,
                    _content->logs,
                    _allocated_size * sizeof(log_data_t)
                );
                delete[] _content->logs;
            }
            _content->logs = tmp_logs;
        )
        _allocated_size = _allocated_size + LOG_PAGE_SIZE;
    }

    __host__ __device__ __forceinline__ void push(
        bn_t &address,
        data_content_t &record,
        bn_t &topic_1,
        bn_t &topic_2,
        bn_t &topic_3,
        bn_t &topic_4,
        uint32_t &no_topics
    )
    {
        if (_content->no_logs == _allocated_size)
        {
            grow();
        }
        cgbn_store(_arith._env, &(_content->logs[_content->no_logs].address), address);
        cgbn_store(_arith._env, &(_content->logs[_content->no_logs].topics[0]), topic_1);
        cgbn_store(_arith._env, &(_content->logs[_content->no_logs].topics[1]), topic_2);
        cgbn_store(_arith._env, &(_content->logs[_content->no_logs].topics[2]), topic_3);
        cgbn_store(_arith._env, &(_content->logs[_content->no_logs].topics[3]), topic_4);
        _content->logs[_content->no_logs].no_topics = no_topics;
        _content->logs[_content->no_logs].record.size = record.size;
        ONE_THREAD_PER_INSTANCE(
            if (record.size > 0)
            {
                _content->logs[_content->no_logs].record.data = new uint8_t[record.size];
                memcpy(
                    _content->logs[_content->no_logs].record.data,
                    record.data,
                    record.size
                );
            }
            else
            {
                _content->logs[_content->no_logs].record.data = NULL;
            }
            _content->no_logs = _content->no_logs + 1;
        )
    }

    /**
     * Update the current touch state with the touch state of a children
     * @param[in] child The touch state of the child
    */
    __host__ __device__ __forceinline__ void update_with_child_state(
        log_state_t &child
    )
    {
        uint32_t idx;
        bn_t address;
        SHARED_MEMORY data_content_t record;
        bn_t topic_1, topic_2, topic_3, topic_4;
        
        // go through all the logs of the child
        for (idx = 0; idx < child._content->no_logs; idx++)
        {
            // get the address of the log
            cgbn_load(_arith._env, address, &(child._content->logs[idx].address));
            // get the topics of the log
            cgbn_load(_arith._env, topic_1, &(child._content->logs[idx].topics[0]));
            cgbn_load(_arith._env, topic_2, &(child._content->logs[idx].topics[1]));
            cgbn_load(_arith._env, topic_3, &(child._content->logs[idx].topics[2]));
            cgbn_load(_arith._env, topic_4, &(child._content->logs[idx].topics[3]));
            // get the record of the log
            record.size = child._content->logs[idx].record.size;
            record.data = child._content->logs[idx].record.data;
            // add the log to the current touch state
            push(
                address,
                record,
                topic_1,
                topic_2,
                topic_3,
                topic_4,
                child._content->logs[idx].no_topics
            );
        }
    }

    /**
     * Copy the content of the touch state to the given touch state data.
     * @param[out] touch_state_data The touch state data
    */
    __host__ __device__ __forceinline__ void to_log_state_data_t(
        log_state_data_t &log_state_data
    )
    {
        ONE_THREAD_PER_INSTANCE(
        // free the memory if it is already allocated
        if (log_state_data.no_logs > 0)
        {
            for (uint32_t idx = 0; idx < log_state_data.no_logs; idx++)
            {
                if (log_state_data.logs[idx].record.size > 0)
                {
                    delete[] log_state_data.logs[idx].record.data;
                    log_state_data.logs[idx].record.data = NULL;
                    log_state_data.logs[idx].record.size = 0;
                }
            }
            delete[] log_state_data.logs;
            log_state_data.logs = NULL;
            log_state_data.no_logs = 0;
        }

        // copy the content and alocate the necessary memory
        log_state_data.no_logs = _content->no_logs;
        if (log_state_data.no_logs > 0)
        {
            log_state_data.logs = new log_data_t[log_state_data.no_logs];
            memcpy(
                log_state_data.logs,
                _content->logs,
                log_state_data.no_logs * sizeof(log_data_t)
            );
            for (uint32_t idx = 0; idx < log_state_data.no_logs; idx++)
            {
                if (log_state_data.logs[idx].record.size > 0)
                {
                    log_state_data.logs[idx].record.data = new uint8_t[log_state_data.logs[idx].record.size];
                    memcpy(
                        log_state_data.logs[idx].record.data,
                        _content->logs[idx].record.data,
                        log_state_data.logs[idx].record.size * sizeof(uint8_t)
                    );
                }
                else
                {
                    log_state_data.logs[idx].record.data = NULL;
                }
            }
        }
        )
    }

    /**
     * Generate the CPU instances of the log state data.
     * @param[in] count The number of instances
    */
    __host__ static log_state_data_t *get_cpu_instances(
        uint32_t count
    )
    {
        // allocate the instances and initialize them
        log_state_data_t *cpu_instances = new log_state_data_t[count];
        for (size_t idx = 0; idx < count; idx++)
        {
            cpu_instances[idx].no_logs = 0;
            cpu_instances[idx].logs = NULL;
        }
        return cpu_instances;
    }

    /**
     * Free the CPU instances of the log state data.
     * @param[in] cpu_instances The CPU instances
     * @param[in] count The number of instances
    */
    __host__ static void free_cpu_instances(
        log_state_data_t *cpu_instances,
        uint32_t count
    )
    {
        for (uint32_t idx = 0; idx < count; idx++)
        {
            if (cpu_instances[idx].no_logs > 0)
            {
                for (uint32_t jdx = 0; jdx < cpu_instances[idx].no_logs; jdx++)
                {
                    if (cpu_instances[idx].logs[jdx].record.size > 0)
                    {
                        delete[] cpu_instances[idx].logs[jdx].record.data;
                        cpu_instances[idx].logs[jdx].record.data = NULL;
                        cpu_instances[idx].logs[jdx].record.size = 0;
                    }
                }
                delete[] cpu_instances[idx].logs;
                cpu_instances[idx].logs = NULL;
                cpu_instances[idx].no_logs = 0;
            }
        }
        delete[] cpu_instances;
    }

    /**
     * Generate the GPU instances of the log state data from
     * the CPU counterparts.
     * @param[in] cpu_instances The CPU instances
     * @param[in] count The number of instances
    */
    __host__ static log_state_data_t *get_gpu_instances_from_cpu_instances(
        log_state_data_t *cpu_instances,
        uint32_t count
    )
    {

        log_state_data_t *gpu_instances, *tmp_cpu_instances;
        // allocate the GPU memory for instances
        CUDA_CHECK(cudaMalloc(
            (void **)&(gpu_instances),
            count * sizeof(log_state_data_t)
        ));
        // use a temporary CPU memory to allocate the GPU memory for the accounts
        // and storage
        tmp_cpu_instances = new log_state_data_t[count];
        memcpy(
            tmp_cpu_instances,
            cpu_instances,
            count * sizeof(log_state_data_t)
        );
        for (uint32_t idx = 0; idx < count; idx++)
        {
            if (
                (cpu_instances[idx].logs != NULL) &&
                (cpu_instances[idx].no_logs > 0)
            )
            {
                CUDA_CHECK(cudaMalloc(
                    (void **)&(tmp_cpu_instances[idx].logs),
                    cpu_instances[idx].no_logs * sizeof(log_data_t)
                ));
                log_data_t *tmp_logs = new log_data_t[cpu_instances[idx].no_logs];
                memcpy(
                    tmp_logs,
                    cpu_instances[idx].logs,
                    cpu_instances[idx].no_logs * sizeof(log_data_t)
                );
                for (uint32_t jdx = 0; jdx < cpu_instances[idx].no_logs; jdx++)
                {
                    if (
                        (cpu_instances[idx].logs[jdx].record.data != NULL) &&
                        (cpu_instances[idx].logs[jdx].record.size > 0)
                    )
                    {
                        CUDA_CHECK(cudaMalloc(
                            (void **)&(tmp_logs[jdx].record.data),
                            cpu_instances[idx].logs[jdx].record.size * sizeof(uint8_t)
                        ));
                        CUDA_CHECK(cudaMemcpy(
                            tmp_logs[jdx].record.data,
                            cpu_instances[idx].logs[jdx].record.data,
                            cpu_instances[idx].logs[jdx].record.size * sizeof(uint8_t),
                            cudaMemcpyHostToDevice
                        ));
                    }
                    else
                    {
                        tmp_logs[jdx].record.data = NULL;
                    }
                }
                CUDA_CHECK(cudaMemcpy(
                    tmp_cpu_instances[idx].logs,
                    tmp_logs,
                    cpu_instances[idx].no_logs * sizeof(log_data_t),
                    cudaMemcpyHostToDevice
                ));
                delete[] tmp_logs;
            }
        }

        CUDA_CHECK(cudaMemcpy(
            gpu_instances,
            tmp_cpu_instances,
            count * sizeof(log_state_data_t),
            cudaMemcpyHostToDevice
        ));
        delete[] tmp_cpu_instances;
        return gpu_instances;
    }

    /**
     * Free the GPU instances of the log state data.
     * @param[in] gpu_instances The GPU instances
     * @param[in] count The number of instances
    */
    __host__ static void free_gpu_instances(
        log_state_data_t *gpu_instances,
        uint32_t count
    )
    {
        log_state_data_t *tmp_cpu_instances = new log_state_data_t[count];
        CUDA_CHECK(cudaMemcpy(
            tmp_cpu_instances,
            gpu_instances,
            count * sizeof(log_state_data_t),
            cudaMemcpyDeviceToHost
        ));
        for (uint32_t idx = 0; idx < count; idx++)
        {
            if (
                (tmp_cpu_instances[idx].logs != NULL) &&
                (tmp_cpu_instances[idx].no_logs > 0)
            )
            {
                log_data_t *tmp_logs = new log_data_t[tmp_cpu_instances[idx].no_logs];
                CUDA_CHECK(cudaMemcpy(
                    tmp_logs,
                    tmp_cpu_instances[idx].logs,
                    tmp_cpu_instances[idx].no_logs * sizeof(log_data_t),
                    cudaMemcpyDeviceToHost
                ));
                for (uint32_t jdx = 0; jdx < tmp_cpu_instances[idx].no_logs; jdx++)
                {
                    if (
                        (tmp_logs[jdx].record.data != NULL) &&
                        (tmp_logs[jdx].record.size > 0)
                    )
                    {
                        CUDA_CHECK(cudaFree(tmp_logs[jdx].record.data));
                        tmp_logs[jdx].record.data = NULL;
                        tmp_logs[jdx].record.size = 0;
                    }
                }
                CUDA_CHECK(cudaFree(tmp_cpu_instances[idx].logs));
                tmp_cpu_instances[idx].logs = NULL;
                tmp_cpu_instances[idx].no_logs = 0;
            }
        }
        delete[] tmp_cpu_instances;
        tmp_cpu_instances = NULL;
        CUDA_CHECK(cudaFree(gpu_instances));
    }

    /**
     * Get the CPU instances of the log state data from
     * the GPU counterparts.
     * @param[in] gpu_instances The GPU instances
     * @param[in] count The number of instances
    */
    __host__ static log_state_data_t *get_cpu_instances_from_gpu_instances(
        log_state_data_t *gpu_instances,
        uint32_t count
    )
    {
        // temporary instances
        log_state_data_t *cpu_instances, *tmp_gpu_instances, *tmp_cpu_instances;
        // allocate the CPU memory for instances
        // and copy the initial details of the touch state
        // like the number of accounts and the pointer to the accounts
        // and their touch
        cpu_instances = new log_state_data_t[count];
        CUDA_CHECK(cudaMemcpy(
            cpu_instances,
            gpu_instances,
            count * sizeof(log_state_data_t),
            cudaMemcpyDeviceToHost
        ));
        // STEP 1: get the accounts details and read operations from GPU
        // use an axiliary emmory to alocate the necesarry memory on GPU which can be touch from
        // the host to copy the accounts details and read operations done on the accounts.
        tmp_cpu_instances = new log_state_data_t[count];
        memcpy(
            tmp_cpu_instances,
            cpu_instances,
            count * sizeof(log_state_data_t)
        );
        for (uint32_t idx = 0; idx < count; idx++)
        {
            // if the instance has accounts
            if (
                (cpu_instances[idx].logs != NULL) &&
                (cpu_instances[idx].no_logs > 0)
            )
            {
                CUDA_CHECK(cudaMalloc(
                    (void **)&(tmp_cpu_instances[idx].logs),
                    cpu_instances[idx].no_logs * sizeof(log_data_t)
                ));
            }
        }
        CUDA_CHECK(cudaMalloc(
            (void **)&(tmp_gpu_instances),
            count * sizeof(log_state_data_t)
        ));
        CUDA_CHECK(cudaMemcpy(
            tmp_gpu_instances,
            tmp_cpu_instances,
            count * sizeof(log_state_data_t),
            cudaMemcpyHostToDevice
        ));
        delete[] tmp_cpu_instances;

        // run the first kernel which copy the accoutns details and read operations
        kernel_log_state_S1<params><<<count, 1>>>(tmp_gpu_instances, gpu_instances, count);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(gpu_instances));

        
        // STEP 2: get the accounts storage and bytecode from GPU
        gpu_instances = tmp_gpu_instances;

        CUDA_CHECK(cudaMemcpy(
            cpu_instances,
            gpu_instances,
            count * sizeof(log_state_data_t),
            cudaMemcpyDeviceToHost
        ));
        tmp_cpu_instances = new log_state_data_t[count];
        memcpy(
            tmp_cpu_instances,
            cpu_instances,
            count * sizeof(log_state_data_t)
        );

        for (uint32_t idx = 0; idx < count; idx++)
        {
            // if the instance has logs
            if (
                (tmp_cpu_instances[idx].logs != NULL) &&
                (tmp_cpu_instances[idx].no_logs > 0)
            )
            {
                CUDA_CHECK(cudaMalloc(
                    (void **)&(tmp_cpu_instances[idx].logs),
                    cpu_instances[idx].no_logs * sizeof(log_data_t)
                ));
                log_data_t *tmp_logs;
                tmp_logs = new log_data_t[cpu_instances[idx].no_logs];
                CUDA_CHECK(cudaMemcpy(
                    tmp_logs,
                    cpu_instances[idx].logs,
                    cpu_instances[idx].no_logs * sizeof(log_data_t),
                    cudaMemcpyDeviceToHost
                ));
                // go through the logs and alocate record data
                for (uint32_t jdx = 0; jdx < cpu_instances[idx].no_logs; jdx++)
                {
                    if (
                        (tmp_logs[jdx].record.data != NULL) &&
                        (tmp_logs[jdx].record.size > 0)
                    )
                    {
                        CUDA_CHECK(cudaMalloc(
                            (void **)&(tmp_logs[jdx].record.data),
                            tmp_logs[jdx].record.size * sizeof(uint8_t)
                        ));
                    }
                }
                CUDA_CHECK(cudaMemcpy(
                    tmp_cpu_instances[idx].logs,
                    tmp_logs,
                    cpu_instances[idx].no_logs * sizeof(log_data_t),
                    cudaMemcpyHostToDevice
                ));
                delete[] tmp_logs;
                tmp_logs = NULL;
            }
        }

        CUDA_CHECK(cudaMalloc(
            (void **)&(tmp_gpu_instances),
            count * sizeof(log_state_data_t)
        ));
        CUDA_CHECK(cudaMemcpy(
            tmp_gpu_instances,
            tmp_cpu_instances,
            count * sizeof(log_state_data_t),
            cudaMemcpyHostToDevice
        ));
        delete[] tmp_cpu_instances;

        // run the second kernel which copy the bytecode and storage
        kernel_log_state_S2<params><<<count, 1>>>(tmp_gpu_instances, gpu_instances, count);
        CUDA_CHECK(cudaDeviceSynchronize());

        // free the memory on GPU for the first kernel (accounts details)
        // the write operations can be kept because they don not have
        // more depth
        for (size_t idx = 0; idx < count; idx++)
        {
            if (
                (cpu_instances[idx].logs != NULL) &&
                (cpu_instances[idx].no_logs > 0)
            )
            {
                CUDA_CHECK(cudaFree(cpu_instances[idx].logs));
                cpu_instances[idx].logs = NULL;
                cpu_instances[idx].no_logs = 0;
            }
        }

        CUDA_CHECK(cudaFree(gpu_instances));
        gpu_instances = tmp_gpu_instances;

        // STEP 3: copy the the entire touch state data from GPU to CPU
        CUDA_CHECK(cudaMemcpy(
            cpu_instances,
            gpu_instances,
            count * sizeof(log_state_data_t),
            cudaMemcpyDeviceToHost
        ));
        tmp_cpu_instances = new log_state_data_t[count];
        memcpy(
            tmp_cpu_instances,
            cpu_instances,
            count * sizeof(log_state_data_t)
        );

        for (uint32_t idx = 0; idx < count; idx++)
        {
            // if the instance has logs
            if (
                (tmp_cpu_instances[idx].logs != NULL) &&
                (tmp_cpu_instances[idx].no_logs > 0)
            )
            {
                log_data_t *tmp_logs, *aux_tmp_logs;
                tmp_logs = new log_data_t[cpu_instances[idx].no_logs];
                aux_tmp_logs = new log_data_t[cpu_instances[idx].no_logs];
                CUDA_CHECK(cudaMemcpy(
                    tmp_logs,
                    cpu_instances[idx].logs,
                    cpu_instances[idx].no_logs * sizeof(log_data_t),
                    cudaMemcpyDeviceToHost
                ));
                CUDA_CHECK(cudaMemcpy(
                    aux_tmp_logs,
                    cpu_instances[idx].logs,
                    cpu_instances[idx].no_logs * sizeof(log_data_t),
                    cudaMemcpyDeviceToHost
                ));
                // go through the logs and copy the record data
                for (uint32_t jdx = 0; jdx < cpu_instances[idx].no_logs; jdx++)
                {
                    if (
                        (tmp_logs[jdx].record.data != NULL) &&
                        (tmp_logs[jdx].record.size > 0)
                    )
                    {
                        tmp_logs[jdx].record.data = new uint8_t[tmp_logs[jdx].record.size];
                        CUDA_CHECK(cudaMemcpy(
                            tmp_logs[jdx].record.data,
                            aux_tmp_logs[jdx].record.data,
                            tmp_logs[jdx].record.size * sizeof(uint8_t),
                            cudaMemcpyDeviceToHost
                        ));
                    }
                }
                delete[] aux_tmp_logs;
                aux_tmp_logs = NULL;
                tmp_cpu_instances[idx].logs = tmp_logs;
            }
        }

        free_gpu_instances(gpu_instances, count);
        memcpy(
            cpu_instances,
            tmp_cpu_instances,
            count * sizeof(log_state_data_t)
        );
        delete[] tmp_cpu_instances;
        return cpu_instances;
    }

    /**
     * Print the log state data structure
     * @param[in] arith The arithemtic instance
     * @param[in] log_state_data The log state data
    */
    __host__ __device__ __forceinline__ static void print_log_state_data_t(
        arith_t &arith,
        log_state_data_t &log_state_data
    )
    {
        printf("no_logs: %u\n", log_state_data.no_logs);
        for (uint32_t idx = 0; idx < log_state_data.no_logs; idx++)
        {
            printf("logs[%u]:\n", idx);
            printf("address: ");
            arith.print_cgbn_memory(log_state_data.logs[idx].address);
            printf("\n");
            printf("no_topics: %u\n", log_state_data.logs[idx].no_topics);
            for (uint32_t jdx = 0; jdx < log_state_data.logs[idx].no_topics; jdx++)
            {
                printf("topics[%u]: ", jdx);
                arith.print_cgbn_memory(log_state_data.logs[idx].topics[jdx]);
            }
            print_data_content_t(log_state_data.logs[idx].record);
        }
    }

    /**
     * Print the state.
    */
    __host__ __device__ __forceinline__ void print()
    {
        print_log_state_data_t(_arith, *_content);
    }

    /**
     * Get json of the lof state data structure.
     * @param[in] arith The arithemtic instance
     * @param[in] log_state_data The log state data
     * @return The json of the lof state data
    */
    __host__ static cJSON *json_from_log_state_data_t(
        arith_t &arith,
        log_state_data_t &log_state_data
    )
    {
        cJSON *log_data_json = NULL;
        cJSON *logs_json = NULL;
        cJSON *log_json = NULL;
        cJSON *topics_json = NULL;
        char *hex_string_ptr = new char[arith_t::BYTES * 2 + 3];
        log_data_json = cJSON_CreateObject();
        logs_json = cJSON_CreateArray();
        for (uint32_t idx = 0; idx < log_state_data.no_logs; idx++)
        {
            log_json = cJSON_CreateObject();

            arith.hex_string_from_cgbn_memory(
                hex_string_ptr,
                log_state_data.logs[idx].address,
                5);
            cJSON_AddStringToObject(log_json, "address", hex_string_ptr);

            topics_json = cJSON_CreateArray();
            for (uint32_t jdx = 0; jdx < log_state_data.logs[idx].no_topics; jdx++)
            {
                arith.hex_string_from_cgbn_memory(
                    hex_string_ptr,
                    log_state_data.logs[idx].topics[jdx]);
                cJSON_AddItemToArray(topics_json, cJSON_CreateString(hex_string_ptr));
            }
            cJSON_AddItemToObject(log_json, "topics", topics_json);

            cJSON_AddItemToObject(log_json, "record", json_from_data_content_t(log_state_data.logs[idx].record));

            cJSON_AddItemToArray(logs_json, log_json);
            
        }
        cJSON_AddItemToObject(log_data_json, "logs", logs_json);
        delete[] hex_string_ptr;
        hex_string_ptr = NULL;
        return log_data_json;
    }

    /**
     * Get json of the state
     * @return The json of the state
    */
    __host__ __forceinline__ cJSON *json()
    {
        return json_from_log_state_data_t(_arith, *_content);
    }


};


/**
 * Kernel to copy the logs details without record data
 * between two instances of the log state data.
 * @param[out] dst_instances The destination instances
 * @param[in] src_instances The source instances
 * @param[in] count The number of instances
*/
template <class params>
__global__ void kernel_log_state_S1(
    typename log_state_t<params>::log_state_data_t *dst_instances,
    typename log_state_t<params>::log_state_data_t *src_instances,
    uint32_t count)
{
    uint32_t instance = blockIdx.x * blockDim.x + threadIdx.x;
    typedef typename log_state_t<params>::log_data_t log_data_t;

    if (instance >= count)
        return;

    if (
        (src_instances[instance].logs != NULL) &&
        (src_instances[instance].no_logs > 0)
    )
    {
        memcpy(
            dst_instances[instance].logs,
            src_instances[instance].logs,
            src_instances[instance].no_logs * sizeof(log_data_t)
        );
        delete[] src_instances[instance].logs;
        src_instances[instance].logs = NULL;
        src_instances[instance].no_logs = 0;
    }
}

/**
 * Kernel to copy the record data
 * between two instances of the log state data.
 * @param[out] dst_instances The destination instances
 * @param[in] src_instances The source instances
 * @param[in] count The number of instances
*/
template <class params>
__global__ void kernel_log_state_S2(
    typename log_state_t<params>::log_state_data_t *dst_instances,
    typename log_state_t<params>::log_state_data_t *src_instances,
    uint32_t count)
{
    uint32_t instance = blockIdx.x * blockDim.x + threadIdx.x;

    if (instance >= count)
        return;

    if (
        (src_instances[instance].logs != NULL) &&
        (src_instances[instance].no_logs > 0)
    )
    {
        for (uint32_t idx = 0; idx < src_instances[instance].no_logs; idx++)
        {
            if (
                (src_instances[instance].logs[idx].record.data != NULL) &&
                (src_instances[instance].logs[idx].record.size > 0)
            )
            {
                memcpy(
                    dst_instances[instance].logs[idx].record.data,
                    src_instances[instance].logs[idx].record.data,
                    src_instances[instance].logs[idx].record.size * sizeof(uint8_t)
                );
                delete[] src_instances[instance].logs[idx].record.data;
                src_instances[instance].logs[idx].record.data = NULL;
                src_instances[instance].logs[idx].record.size = 0;
            }
        }
    }
}

#endif