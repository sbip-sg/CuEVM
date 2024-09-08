// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2024-06-20
// SPDX-License-Identifier: MIT

#include <CuEVM/state/account.cuh>
#include <CuCrypto/keccak.cuh>

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
            (src_instances[instance].byte_code.data != nullptr) &&
            (src_instances[instance].byte_code.size > 0)
        ) {
            memcpy(
                dst_instances[instance].byte_code.data,
                src_instances[instance].byte_code.data,
                src_instances[instance].byte_code.size * sizeof(uint8_t)
            );
            dst_instances[instance].byte_code.size = src_instances[instance].byte_code.size;
            delete[] src_instances[instance].byte_code.data;
            src_instances[instance].byte_code.data = nullptr;
            src_instances[instance].byte_code.size = 0;
        } else {
            dst_instances[instance].byte_code.data = nullptr;
            dst_instances[instance].byte_code.size = 0;
        }

        if (
            (src_instances[instance].storage.storage != nullptr) &&
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
            src_instances[instance].storage.storage = nullptr;
            src_instances[instance].storage.capacity = 0;
            src_instances[instance].storage.size = 0;
        } else {
            dst_instances[instance].storage = nullptr;
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

    __host__ __device__ account_t::account_t(
        const account_t* account_ptr,
        const account_flags_t &flags)
    {
        address = account_ptr->address;
        balance = account_ptr->balance;
        nonce = account_ptr->nonce;
        if(flags.has_byte_code()) {
            byte_code = account_ptr->byte_code;
        }
        if(flags.has_storage()) {
            storage = account_ptr->storage;
        }
    }

    __host__ __device__ account_t::account_t(
        ArithEnv &arith,
        const bn_t &address)
    {
        cgbn_store(arith.env, &this->address, address);
        bn_t tmp;
        cgbn_set_ui32(arith.env, tmp, 0);
        cgbn_store(arith.env, &this->balance, tmp);
        cgbn_store(arith.env, &this->nonce, tmp);
        byte_code.size = 0;
        byte_code.data = nullptr;
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

    
    __host__ __device__ byte_array_t  account_t::get_byte_code() const {
        return byte_code;
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

    __host__ __device__ int32_t account_t::has_address(
        ArithEnv &arith,
        const evm_word_t &address)
    {
        bn_t local_address, target_address;
        cgbn_load(arith.env, local_address, &this->address);
        cgbn_load(arith.env, target_address, (cgbn_evm_word_t_ptr) &address);
        return (cgbn_compare(arith.env, local_address, target_address) == 0);
    }

    __host__ __device__ void account_t::update(
        ArithEnv &arith,
        const account_t &other,
        const account_flags_t &flags) {
        if(flags.has_address()) {
            address = other.address;
        }
        if(flags.has_balance()) {
            balance = other.balance;
        }
        if(flags.has_nonce()) {
            nonce = other.nonce;
        }
        if(flags.has_byte_code()) {
            byte_code = other.byte_code;
        }
        if(flags.has_storage()) {
            storage.update(arith, other.storage);
        }
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
            (balance == 0) &&
            (nonce == 0) &&
            (byte_code.size == 0)
        );
    }

    __host__ __device__ int32_t account_t::is_contract()
    {
        return (byte_code.size > 0);
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

        // TODO: check later the device id for gpu
        // if (managed) {
        //     if (byte_code.size > 0) {
        //         CUDA_CHECK(cudaMemPrefetchAsync(
        //             (void **)&(byte_code.data),
        //             byte_code.size * sizeof(uint8_t),
        //             0
        //         ));
        //     }
        // }

        storage.from_json(
            cJSON_GetObjectItemCaseSensitive(
                account_json,
                "storage"),
            managed
        );
    }

    __host__ cJSON* account_t::to_json() const
    {
        cJSON *account_json = cJSON_CreateObject();
        char *bytes_string = nullptr;
        char *hex_string_ptr = new char[cuEVM::word_size * 2 + 3];
        size_t jdx = 0;
        address.to_hex(hex_string_ptr, 0, 5);
        cJSON_SetValuestring(account_json, hex_string_ptr);
        balance.to_hex(hex_string_ptr);
        cJSON_AddStringToObject(account_json, "balance", hex_string_ptr);
        nonce.to_hex(hex_string_ptr);
        cJSON_AddStringToObject(account_json, "nonce", hex_string_ptr);
        bytes_string = byte_code.to_hex();
        cJSON_AddStringToObject(account_json, "code", bytes_string);
        delete[] bytes_string;
        cJSON_AddItemToObject(account_json, "storage", storage.to_json(1));
        delete[] hex_string_ptr;
        hex_string_ptr = nullptr;
        return account_json;
    }

    __host__ __device__ void account_t::print()
    {
        printf("Account:\n");
        address.print();
        printf("Balance: ");
        balance.print();
        printf("Nonce: ");
        nonce.print();
        printf("Byte code: ");
        byte_code.print();
        printf("Storage: \n");
        storage.print();
    }

    __host__ cJSON* account_merge_json(
        const account_t *&account1_ptr,
        const account_t *&account2_ptr,
        const account_flags_t &flags) {
        
        cJSON *account_json = cJSON_CreateObject();
        char *hex_string_ptr = new char[cuEVM::word_size * 2 + 3];
        account1_ptr->address.to_hex(hex_string_ptr, 0, 5);
        cJSON_AddStringToObject(account_json, "address", hex_string_ptr);

        if(flags.has_balance()) {
            account2_ptr->balance.to_hex(hex_string_ptr, 1);
        } else {
            account1_ptr->balance.to_hex(hex_string_ptr, 1);
        }
        cJSON_AddStringToObject(account_json, "balance", hex_string_ptr);

        if(flags.has_nonce()) {
            account2_ptr->nonce.to_hex(hex_string_ptr, 1);
        } else {
            account1_ptr->nonce.to_hex(hex_string_ptr, 1);
        }
        cJSON_AddStringToObject(account_json, "nonce", hex_string_ptr);

        char *code_hash_hex_string_ptr = nullptr;
        char *code_hex_string_ptr = nullptr;
        cuEVM::byte_array_t *hash;
        hash = new cuEVM::byte_array_t(cuEVM::hash_size);
        if(flags.has_byte_code()) {
            CuCrypto::keccak::sha3(
                account2_ptr->byte_code.data,
                account2_ptr->byte_code.size,
                hash->data,
                hash->size
            );
            code_hex_string_ptr = account2_ptr->byte_code.to_hex();
        } else {
            CuCrypto::keccak::sha3(
                account1_ptr->byte_code.data,
                account1_ptr->byte_code.size,
                hash->data,
                hash->size
            );
            code_hex_string_ptr = account1_ptr->byte_code.to_hex();
        }
        cJSON_AddStringToObject(account_json, "code", code_hex_string_ptr);
        delete[] code_hex_string_ptr;
        code_hash_hex_string_ptr = hash->to_hex();
        cJSON_AddStringToObject(account_json, "codeHash", code_hash_hex_string_ptr);
        delete[] code_hash_hex_string_ptr;
        delete hash;

        cJSON *storage_json = nullptr;
        
        if (flags.has_storage()) {
            storage_json = cuEVM::storage::storage_merge_json(
                account1_ptr->storage,
                account2_ptr->storage,
                1
            );
        } else {
            storage_json = cuEVM::storage::storage_merge_json(
                account1_ptr->storage,
                account1_ptr->storage,
                1
            );
        }

        cJSON_AddItemToObject(account_json, "storage", storage_json);

        delete[] hex_string_ptr;
        return account_json;
    }

    __host__ __device__ void free_internals_account(
        account_t &account,
        int32_t managed)
    {
        if (
            (account.byte_code.data != nullptr) &&
            (account.byte_code.size > 0)
        ) {
            if (managed)
            {
                CUDA_CHECK(cudaFree(account.byte_code.data));
            } else {
                delete[] account.byte_code.data;
            }
            account.byte_code.data = nullptr;
            account.byte_code.size = 0;
        }
        if (
            (account.storage.storage != nullptr) &&
            (account.storage.capacity > 0)
        ) {
            if (managed)
            {
                CUDA_CHECK(cudaFree(account.storage.storage));
            } else {
                delete[] account.storage.storage;
            }
            account.storage.storage = nullptr;
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
            cpu_instances[index].empty();
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
                (tmp_cpu_instances[index].byte_code.data != nullptr) &&
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
                (tmp_cpu_instances[index].storage.storage != nullptr) &&
                (tmp_cpu_instances[index].storage.size > 0)
            ) {
                CUDA_CHECK(cudaMalloc(
                    &tmp_cpu_instances[index].storage.storage,
                    tmp_cpu_instances[index].storage.size *
                        sizeof(cuEVM::storage::storage_element_t)
                ));
                CUDA_CHECK(cudaMemcpy(
                    tmp_cpu_instances[index].storage.storage,
                    cpu_instances[index].storage.storage,
                    cpu_instances[index].storage.size *
                        sizeof(cuEVM::storage::storage_element_t),
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
                (tmp_cpu_instances[index].byte_code.data != nullptr) &&
                (tmp_cpu_instances[index].byte_code.size > 0)
            ) {
                CUDA_CHECK(cudaFree(tmp_cpu_instances[index].byte_code.data));
            }
            if (
                (tmp_cpu_instances[index].storage.storage != nullptr) &&
                (tmp_cpu_instances[index].storage.size > 0)
            ) {
                CUDA_CHECK(cudaFree(tmp_cpu_instances[index].storage.storage));
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
                (tmp_cpu_instances[index].byte_code.data != nullptr) &&
                (tmp_cpu_instances[index].byte_code.size > 0)
            ) {
                cpu_instances[index].byte_code.data = new uint8_t[tmp_cpu_instances[index].byte_code.size];
            } else {
                cpu_instances[index].byte_code.data = nullptr;
                cpu_instances[index].byte_code.size = 0;
            }
            if (
                (tmp_cpu_instances[index].storage.storage != nullptr) &&
                (tmp_cpu_instances[index].storage.size > 0)
            ) {
                cpu_instances[index].storage.storage = new cuEVM::storage::storage_element_t[tmp_cpu_instances[index].storage.size];
            } else {
                cpu_instances[index].storage.storage = nullptr;
                cpu_instances[index].storage.size = 0;
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
                (tmp_cpu_instances[index].byte_code.data != nullptr) &&
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
                cpu_instances[index].byte_code.data = nullptr;
                cpu_instances[index].byte_code.size = 0;
            }
            if (
                (tmp_cpu_instances[index].storage.storage != nullptr) &&
                (tmp_cpu_instances[index].storage.size > 0)
            ) {
                CUDA_CHECK(
                    cudaMemcpy(
                        cpu_instances[index].storage.storage,
                        tmp_cpu_instances[index].storage.storage,
                        tmp_cpu_instances[index].storage.size *
                            sizeof(cuEVM::storage::storage_element_t),
                        cudaMemcpyDeviceToHost
                    )
                );
                cpu_instances[index].storage.size = tmp_cpu_instances[index].storage.size;
            } else {
                cpu_instances[index].storage.storage = nullptr;
                cpu_instances[index].storage.size = 0;
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
            (managed_instance.byte_code.data != nullptr) &&
            (managed_instance.byte_code.size > 0)
        ) {
            CUDA_CHECK(cudaFree(managed_instance.byte_code.data));
            managed_instance.byte_code.data = nullptr;
            managed_instance.byte_code.size = 0;
        }
        if (
            (managed_instance.storage.storage != nullptr) &&
            (managed_instance.storage.size > 0)
        ) {
            CUDA_CHECK(cudaFree(managed_instance.storage.storage));
            managed_instance.storage.storage = nullptr;
            managed_instance.storage.size = 0;
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
  }
}