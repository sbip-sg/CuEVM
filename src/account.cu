#include "include/account.cuh"
#include "include/utils.cuh"

namespace cuEVM
{
  namespace account
  {

    __global__ void transfer_kernel(
        account_t *dst_instances,
        account_t *src_instances,
        uint32_t count)
    {
        uint32_t instance = blockIdx.x * blockDim.x + threadIdx.x;

        if (instance >= count)
            return;
        
        if (
            (src_instances[instance].byte_code.data != NULL) &&
            (src_instances[instance].byte_code.size > 0)
        ) {
            memcpy(
                dst_instances[instance].byte_code.data,
                src_instances[instance].byte_code.data,
                src_instances[instance].byte_code.size * sizeof(uint8_t)
            );
            dst_instances[instance].byte_code.size = src_instances[instance].byte_code.size;
            delete[] src_instances[instance].byte_code.data;
            src_instances[instance].byte_code.data = NULL;
            src_instances[instance].byte_code.size = 0;
        } else {
            dst_instances[instance].byte_code.data = NULL;
            dst_instances[instance].byte_code.size = 0;
        }

        if (
            (src_instances[instance].storage.storage != NULL) &&
            (src_instances[instance].storage.capacity > 0) &&
            (src_instances[instance].storage.size > 0)
        ) {
            memcpy(
                dst_instances[instance].storage.storage,
                src_instances[instance].storage.storage,
                src_instances[instance].storage.size *
                    sizeof(cuEVM::storage::storage_element_t)
            );
            dst_instances[instance].storage.size = src_instances[instance].storage.size;
            delete[] src_instances[instance].storage.storage;
            src_instances[instance].storage.storage = NULL;
            src_instances[instance].storage.capacity = 0;
            src_instances[instance].storage.size = 0;
        } else {
            dst_instances[instance].storage = NULL;
            dst_instances[instance].storage.capacity = 0;
            dst_instances[instance].storage.size = 0;
        }
        // copy the others not necesary
        memcpy(
            &dst_instances[instance].nonce,
            &src_instances[instance].nonce,
            sizeof(evm_word_t)
        );
        memcpy(
            &dst_instances[instance].balance,
            &src_instances[instance].balance,
            sizeof(evm_word_t)
        );
        memcpy(
            &dst_instances[instance].address,
            &src_instances[instance].address,
            sizeof(evm_word_t)
        );
    }

    __host__ account_t::account_t(
        const cJSON *account_json,
        int32_t managed)
    {
        from_json(account_json, managed);
    }

    __host__ __device__ account_t::account_t(
        const account_t &account)
    {
        address = account.address;
        balance = account.balance;
        nonce = account.nonce;
        byte_code = account.byte_code;
        storage = account.storage;
    }

    __host__ __device__ account_t::account_t(
        const account_t &account,
        const account_flags_t &flags)
    {
        address = account.address;
        balance = account.balance;
        nonce = account.nonce;
        if(flags.has_byte_code()) {
            byte_code = account.byte_code;
        }
        if(flags.has_storage()) {
            storage = account.storage;
        }
    }

    __host__ __device__ int32_t account_t::free_internals(
        int32_t managed)
    {
        // TODO:
        return;
    }

    __host__ __device__ int32_t account_t::get_storage_value(
        ArithEnv &arith,
        const bn_t &key,
        bn_t &value)
    {
        return storage.get_value(arith, key, value);
    }

    __host__ __device__ int32_t account_t::set_storage_value(
        ArithEnv &arith,
        const bn_t &key,
        const bn_t &value)
    {
        return storage.set_value(arith, key, value);
    }

    __host__ __device__ void account_t::get_address(
        ArithEnv &arith,
        bn_t &address)
    {
        cgbn_load(arith.env, address, &this->address);
    }

    __host__ __device__ void account_t::get_balance(
        ArithEnv &arith,
        bn_t &balance)
    {
        cgbn_load(arith.env, balance, &this->balance);
    }

    __host__ __device__ void account_t::get_nonce(
        ArithEnv &arith,
        bn_t &nonce)
    {
        cgbn_load(arith.env, nonce, &this->nonce);
    }

    __host__ __device__ void account_t::set_nonce(
        ArithEnv &arith,
        const bn_t &nonce)
    {
        cgbn_store(arith.env, &this->nonce, nonce);
    }

    __host__ __device__ void account_t::set_balance(
        ArithEnv &arith,
        const bn_t &balance)
    {
        cgbn_store(arith.env, &this->balance, balance);
    }

    __host__ __device__ void account_t::set_address(
        ArithEnv &arith,
        const bn_t &address)
    {
        cgbn_store(arith.env, &this->address, address);
    }

    __host__ __device__ void account_t::set_byte_code(
        const byte_array_t &byte_code)
    {
        this->byte_code = byte_code;
    }

    __host__ __device__ int32_t account_t::has_address(
        ArithEnv &arith,
        const bn_t &address)
    {
        bn_t local_address;
        cgbn_load(arith.env, local_address, &this->address);
        return (cgbn_compare(arith.env, local_address, address) == 0);
    }

    __host__ __device__ int32_t account_t::is_empty(
        ArithEnv &arith)
    {
        bn_t balance, nonce;
        cgbn_load(arith.env, balance, &this->balance);
        cgbn_load(arith.env, nonce, &this->nonce);
        return (
            (cgbn_compare_ui32(arith.env, balance, 0) == 0) &&
            (cgbn_compare_ui32(arith.env, nonce, 0) == 0) &&
            (this->byte_code.size == 0)
        );
    }

    __host__ __device__ int32_t account_t::is_empty()
    {
        return (
            (this->balance == 0) &&
            (this->nonce == 0) &&
            (this->byte_code.size == 0)
        );
    }

    __host__ __device__ void account_t::empty()
    {
        memset(this, 0, sizeof(account_t));
    }

    __host__ void account_t::from_json(
        const cJSON *account_json,
        int32_t managed)
    {
        cJSON *balance_json, *nonce_json, *code_json, *storage_json, *key_value_json;
        char *hex_string;
        
        address.from_hex(account_json->string);

        // set the balance
        balance_json = cJSON_GetObjectItemCaseSensitive(
            account_json,
            "balance");
        balance.from_hex(balance_json->valuestring);

        // set the nonce
        nonce_json = cJSON_GetObjectItemCaseSensitive(
            account_json,
            "nonce");
        nonce.from_hex(nonce_json->valuestring);

        byte_code.from_hex(
            cJSON_GetObjectItemCaseSensitive(
                account_json,
                "code")->valuestring,
            LITTLE_ENDIAN,
            NO_PADDING,
            managed
        );

        if (managed) {
            if (byte_code.size > 0) {
                CUDA_CHECK(cudaMemPrefetchAsync(
                    (void **)&(byte_code.data),
                    byte_code.size * sizeof(uint8_t),
                    0
                ));
            }
        }

        storage.from_json(
            cJSON_GetObjectItemCaseSensitive(
                account_json,
                "storage"),
            managed
        );
    }

    __host__ __device__ void free_internals_account(
        account_t &account,
        int32_t managed = 0)
    {
        if (
            (account.byte_code.data != NULL) &&
            (account.byte_code.size > 0)
        ) {
            if (managed)
            {
                CUDA_CHECK(cudaFree(account.byte_code.data));
            } else {
                delete[] account.byte_code.data;
            }
            account.byte_code.data = NULL;
            account.byte_code.size = 0;
        }
        if (
            (account.storage.storage != NULL) &&
            (account.storage.capacity > 0)
        ) {
            if (managed)
            {
                CUDA_CHECK(cudaFree(account.storage.storage));
            } else {
                delete[] account.storage.storage;
            }
            account.storage.storage = NULL;
            account.storage.capacity = 0;
            account.storage.size = 0;
        }
    }

    __host__ account_t *get_cpu_instances(
        uint32_t count)
    {
        account_t *cpu_instances = new account_t[count];
        for(uint32_t index; index < count; index++)
        {
            cpu_instances[index].byte_code.data = NULL;
            cpu_instances[index].byte_code.size = 0;
            cpu_instances[index].storage = NULL;
            cpu_instances[index].storage_size = 0;
        }
        return cpu_instances;
    }
    
    __host__ void free_cpu_instances(
        account_t *cpu_instances,
        uint32_t count)
    {
        for(uint32_t index; index < count; index++)
        {
            free_internals_account(cpu_instances[index]);
        }
        delete[] cpu_instances;
    }
    
    __host__ account_t *get_gpu_instances_from_cpu_instances(
        account_t *cpu_instances,
        uint32_t count)
    {
        account_t *gpu_instances, *tmp_cpu_instances;
        tmp_cpu_instances = new account_t[count];
        memcpy(
            tmp_cpu_instances,
            cpu_instances,
            count * sizeof(account_t)
        );
        for(uint32_t index; index < count; index++) {
            if (
                (tmp_cpu_instances[index].byte_code.data != NULL) &&
                (tmp_cpu_instances[index].byte_code.size > 0)
            ) {
                CUDA_CHECK(cudaMalloc(
                    &tmp_cpu_instances[index].byte_code.data,
                    tmp_cpu_instances[index].byte_code.size * sizeof(uint8_t)
                ));
                CUDA_CHECK(cudaMemcpy(
                    tmp_cpu_instances[index].byte_code.data,
                    cpu_instances[index].byte_code.data,
                    cpu_instances[index].byte_code.size * sizeof(uint8_t),
                    cudaMemcpyHostToDevice
                ));
            }
            if (
                (tmp_cpu_instances[index].storage != NULL) &&
                (tmp_cpu_instances[index].storage_size > 0)
            ) {
                CUDA_CHECK(cudaMalloc(
                    &tmp_cpu_instances[index].storage,
                    tmp_cpu_instances[index].storage_size *
                        sizeof(contract_storage_t)
                ));
                CUDA_CHECK(cudaMemcpy(
                    tmp_cpu_instances[index].storage,
                    cpu_instances[index].storage,
                    cpu_instances[index].storage_size *
                        sizeof(contract_storage_t),
                    cudaMemcpyHostToDevice
                ));
            }
        }
        CUDA_CHECK(cudaMalloc(
            &gpu_instances,
            count * sizeof(account_t)
        ));
        CUDA_CHECK(cudaMemcpy(
            gpu_instances,
            tmp_cpu_instances,
            count * sizeof(account_t),
            cudaMemcpyHostToDevice
        ));
        delete[] tmp_cpu_instances;
        return gpu_instances;
    }
    
    __host__ void free_gpu_instances(
        account_t *gpu_instances,
        uint32_t count)
    {
        account_t *tmp_cpu_instances = new account_t[count];
        CUDA_CHECK(cudaMemcpy(
            tmp_cpu_instances,
            gpu_instances,
            count * sizeof(account_t),
            cudaMemcpyDeviceToHost
        ));
        for(uint32_t index; index < count; index++)
        {
            if (
                (tmp_cpu_instances[index].byte_code.data != NULL) &&
                (tmp_cpu_instances[index].byte_code.size > 0)
            ) {
                CUDA_CHECK(cudaFree(tmp_cpu_instances[index].byte_code.data));
            }
            if (
                (tmp_cpu_instances[index].storage != NULL) &&
                (tmp_cpu_instances[index].storage_size > 0)
            ) {
                CUDA_CHECK(cudaFree(tmp_cpu_instances[index].storage));
            }
        }
        delete[] tmp_cpu_instances;
        CUDA_CHECK(cudaFree(gpu_instances));
    }
    
    __host__ account_t *get_cpu_from_gpu_instances(
        account_t *gpu_instances,
        uint32_t count)
    {
        // we consider that the byte code and storage were not allocated
        // from the cpu side
        account_t *cpu_instances = new account_t[count];
        account_t *tmp_cpu_instances = new account_t[count];
        CUDA_CHECK(cudaMemcpy(
            tmp_cpu_instances,
            gpu_instances,
            count * sizeof(account_t),
            cudaMemcpyDeviceToHost
        ));
        memcpy(
            cpu_instances,
            tmp_cpu_instances,
            count * sizeof(account_t)
        );
        for(uint32_t index; index < count; index++)
        {
            if (
                (tmp_cpu_instances[index].byte_code.data != NULL) &&
                (tmp_cpu_instances[index].byte_code.size > 0)
            ) {
                cpu_instances[index].byte_code.data = new uint8_t[tmp_cpu_instances[index].byte_code.size];
            } else {
                cpu_instances[index].byte_code.data = NULL;
                cpu_instances[index].byte_code.size = 0;
            }
            if (
                (tmp_cpu_instances[index].storage != NULL) &&
                (tmp_cpu_instances[index].storage_size > 0)
            ) {
                cpu_instances[index].storage = new contract_storage_t[tmp_cpu_instances[index].storage_size];
            } else {
                cpu_instances[index].storage = NULL;
                cpu_instances[index].storage_size = 0;
            }
        }
        delete[] tmp_cpu_instances;
        account_t *tmp_gpu_instaces;
        tmp_gpu_instaces = get_gpu_instances_from_cpu_instances(
            cpu_instances,
            count
        );

        // call the transfer kernel
        transfer_kernel<<<
            count,
            1>>>(
            tmp_gpu_instaces,
            gpu_instances,
            count
        );

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(gpu_instances));


        CUDA_CHECK(cudaMemcpy(
            tmp_cpu_instances,
            tmp_gpu_instaces,
            count * sizeof(account_t),
            cudaMemcpyDeviceToHost
        ));

        for(uint32_t index; index < count; index++)
        {
            if (
                (tmp_cpu_instances[index].byte_code.data != NULL) &&
                (tmp_cpu_instances[index].byte_code.size > 0)
            ) {
                CUDA_CHECK(
                    cudaMemcpy(
                        cpu_instances[index].byte_code.data,
                        tmp_cpu_instances[index].byte_code.data,
                        tmp_cpu_instances[index].byte_code.size * sizeof(uint8_t),
                        cudaMemcpyDeviceToHost
                    )
                );
                cpu_instances[index].byte_code.size = tmp_cpu_instances[index].byte_code.size;
            } else {
                cpu_instances[index].byte_code.data = NULL;
                cpu_instances[index].byte_code.size = 0;
            }
            if (
                (tmp_cpu_instances[index].storage != NULL) &&
                (tmp_cpu_instances[index].storage_size > 0)
            ) {
                CUDA_CHECK(
                    cudaMemcpy(
                        cpu_instances[index].storage,
                        tmp_cpu_instances[index].storage,
                        tmp_cpu_instances[index].storage_size *
                            sizeof(contract_storage_t),
                        cudaMemcpyDeviceToHost
                    )
                );
                cpu_instances[index].storage_size = tmp_cpu_instances[index].storage_size;
            } else {
                cpu_instances[index].storage = NULL;
                cpu_instances[index].storage_size = 0;
            }
        }
        delete[] tmp_cpu_instances;

        free_gpu_instances(
            tmp_gpu_instaces,
            count
        );
        return cpu_instances;
    }

    __host__ account_t *get_managed_instances(
        uint32_t count)
    {
        account_t *managed_instances;
        CUDA_CHECK(cudaMallocManaged(
            &managed_instances,
            count * sizeof(account_t)
        ));
        return managed_instances;
    }


    __host__ void free_internals_managed_instance(
        account_t &managed_instance)
    {
        if (
            (managed_instance.byte_code.data != NULL) &&
            (managed_instance.byte_code.size > 0)
        ) {
            CUDA_CHECK(cudaFree(managed_instance.byte_code.data));
            managed_instance.byte_code.data = NULL;
            managed_instance.byte_code.size = 0;
        }
        if (
            (managed_instance.storage != NULL) &&
            (managed_instance.storage_size > 0)
        ) {
            CUDA_CHECK(cudaFree(managed_instance.storage));
            managed_instance.storage = NULL;
            managed_instance.storage_size = 0;
        }
    }

    __host__ void free_managed_instances(
        account_t *managed_instances,
        uint32_t count)
    {
        for(uint32_t index; index < count; index++)
        {
            free_internals_managed_instance(managed_instances[index]);
        }
        CUDA_CHECK(cudaFree(managed_instances));
    }
    
    // UTILITY Functions
    __host__ void from_json(
        account_t &account,
        const cJSON *account_json,
        bool managed = false)
    {
        cJSON *balance_json, *nonce_json, *code_json, *storage_json, *key_value_json;
        char *hex_string;
        // set the address
        evm_word_t_from_hex_string(
            account.address,
            account_json->string);

        // set the balance
        balance_json = cJSON_GetObjectItemCaseSensitive(
            account_json,
            "balance");
        evm_word_t_from_hex_string(
            account.balance,
            balance_json->valuestring);

        // set the nonce
        nonce_json = cJSON_GetObjectItemCaseSensitive(
            account_json,
            "nonce");
        evm_word_t_from_hex_string(
            account.nonce,
            nonce_json->valuestring);

        // set the code
        code_json = cJSON_GetObjectItemCaseSensitive(
            account_json,
            "code");
        hex_string = code_json->valuestring;
        int32_t length = cuEVM::byte_array::clean_hex_string(&hex_string);
        if (length < 0)
        {
            printf("Error: Invalid hex string\n");
            exit(EXIT_FAILURE);
        }
        account.byte_code.size = (length + 1) / 2;
        if (account.byte_code.size > 0)
        {
            if (managed)
            {
                CUDA_CHECK(cudaMallocManaged(
                    (void **)&(account.byte_code.data),
                    account.byte_code.size * sizeof(uint8_t)
                ));
            }
            else
            {
                account.byte_code.data = new uint8_t[account.byte_code.size];
            }
            cuEVM::byte_array::byte_array_t_from_hex_set_le(
                account.byte_code,
                hex_string,
                length
            );
            if (managed) {
                CUDA_CHECK(cudaMemPrefetchAsync(
                    (void **)&(account.byte_code.data),
                    account.byte_code.size * sizeof(uint8_t),
                    0
                ));
            }
        }
        else
        {
            account.byte_code.data = NULL;
        }

        // set the storage
        storage_json = cJSON_GetObjectItemCaseSensitive(
            account_json,
            "storage");
        account.storage_size = cJSON_GetArraySize(
            storage_json);
        
        uint32_t index;
        if (account.storage_size > 0)
        {
            if (managed)
            {
                CUDA_CHECK(cudaMallocManaged(
                    (void **)&(account.storage),
                    account.storage_size * sizeof(contract_storage_t)
                ));
            }
            else
            {
                account.storage = new contract_storage_t[account.storage_size];
            }
            // iterate through the storage
            index = 0;
            cJSON_ArrayForEach(key_value_json, storage_json)
            {
                // set the key
                evm_word_t_from_hex_string(
                    account.storage[index].key,
                    key_value_json->string);

                // set the value
                evm_word_t_from_hex_string(
                    account.storage[index].value,
                    key_value_json->valuestring);

                index++;
            }
            if (managed) {
                CUDA_CHECK(cudaMemPrefetchAsync(
                    (void **)&(account.storage),
                    account.storage_size * sizeof(contract_storage_t),
                    0
                ));
            }
        }
        else
        {
            account.storage = NULL;
        }
    }
    
    __host__ cJSON* json(
        account_t &account)
    {
        cJSON *account_json = NULL;
        cJSON *storage_json = NULL;
        char *bytes_string = NULL;
        char *hex_string_ptr = new char[EVM_WORD_SIZE * 2 + 3];
        char *value_hex_string_ptr = new char[EVM_WORD_SIZE * 2 + 3];
        size_t jdx = 0;
        account_json = cJSON_CreateObject();
        // set the address
        hex_string_from_evm_word_t(hex_string_ptr, account.address, 5);
        cJSON_SetValuestring(account_json, hex_string_ptr);
        // set the balance
        hex_string_from_evm_word_t(hex_string_ptr, account.balance);
        cJSON_AddStringToObject(account_json, "balance", hex_string_ptr);
        // set the nonce
        hex_string_from_evm_word_t(hex_string_ptr, account.nonce);
        cJSON_AddStringToObject(account_json, "nonce", hex_string_ptr);
        // set the code
        if (account.byte_code.size > 0)
        {
            bytes_string = cuEVM::byte_array::hex_from_bytes(account.byte_code.data, account.byte_code.size);
            cJSON_AddStringToObject(account_json, "code", bytes_string);
            delete[] bytes_string;
        }
        else
        {
            cJSON_AddStringToObject(account_json, "code", "0x");
        }
        // set the storage
        storage_json = cJSON_CreateObject();
        cJSON_AddItemToObject(account_json, "storage", storage_json);
        if (account.storage_size > 0)
        {
            for (jdx = 0; jdx < account.storage_size; jdx++)
            {
                hex_string_from_evm_word_t(hex_string_ptr, account.storage[jdx].key);
                hex_string_from_evm_word_t(value_hex_string_ptr, account.storage[jdx].value);
                cJSON_AddStringToObject(storage_json, hex_string_ptr, value_hex_string_ptr);
            }
        }
        delete[] hex_string_ptr;
        hex_string_ptr = NULL;
        delete[] value_hex_string_ptr;
        value_hex_string_ptr = NULL;
        return account_json;
    }
    
    __host__ __device__ void print(
        account_t &account)
    {
        printf("address: ");
        print_evm_word_t(account.address);
        printf("balance: ");
        print_evm_word_t(account.balance);
        printf("nonce: ");
        print_evm_word_t(account.nonce);
        cuEVM::byte_array::print_byte_array_t(account.byte_code);
        printf("storage_size: %lu\n", account.storage_size);
        for (size_t idx = 0; idx < account.storage_size; idx++)
        {
            printf("storage[%lu].key: ", idx);
            print_evm_word_t(account.storage[idx].key);
            printf("storage[%lu].value: ", idx);
            print_evm_word_t(account.storage[idx].value);
        }
    }
    
    __host__ __device__ int32_t get_storage_index(
        int32_t &index,
        ArithEnv &arith,
        const account_t &account,
        bn_t &key)
    {
        bn_t local_key;
        for (index = 0; index < account.storage_size; index++)
        {
            cgbn_load(arith.env, local_key, &(account.storage[index].key));
            if (cgbn_compare(arith.env, local_key, key) == 0)
            {
                return 1;
            }
        }
        return 0;
    }
    
    __host__ __device__ void empty(
        account_t &account)
    {
        memset(&account, 0, sizeof(account_t));
    }
    
    __host__ __device__ void duplicate(
        account_t &dst,
        const account_t &src,
        bool with_storage)
    {
        memcpy(&dst, &src, sizeof(account_t));
        if (
            (src.byte_code.data != NULL) &&
            (src.byte_code.size > 0)
        ) {
            dst.byte_code.data = new uint8_t[src.byte_code.size];
            memcpy(
                dst.byte_code.data,
                src.byte_code.data,
                src.byte_code.size * sizeof(uint8_t)
            );
        } else {
            dst.byte_code.data = NULL;
            dst.byte_code.size = 0;
        }
        if (
            (src.storage != NULL) &&
            (src.storage_size > 0) &&
            with_storage
        ) {
            dst.storage = new contract_storage_t[src.storage_size];
            memcpy(
                dst.storage,
                src.storage,
                src.storage_size * sizeof(contract_storage_t)
            );
        } else {
            dst.storage = NULL;
            dst.storage_size = 0;
        }
    }
    
    __host__ __device__ int32_t is_empty(
        ArithEnv &arith,
        account_t &account)
    {
        bn_t balance, nonce;
        cgbn_load(arith.env, balance, &(account.balance));
        cgbn_load(arith.env, nonce, &(account.nonce));
        return (
            (cgbn_compare_ui32(arith.env, balance, 0) == 0) &&
            (cgbn_compare_ui32(arith.env, nonce, 0) == 0) &&
            (account.byte_code.size == 0)
        );
    }
  }
}