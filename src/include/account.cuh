#ifndef _CUEVM_ACCOUNT_H_
#define _CUEVM_ACCOUNT_H_

#include "arith.cuh"
#include "storage.cuh"

namespace cuEVM
{
  namespace account
  {
    /**
     * The account type.
    */
    struct account_t
    {
        evm_word_t address; /**< The address of the account (YP: \f$a\f$) */
        evm_word_t balance; /**< The balance of the account (YP: \f$\sigma[a]_{b}\f$) */
        evm_word_t nonce; /**< The nonce of the account (YP: \f$\sigma[a]_{n}\f$) */
        byte_array_t byte_code; /**< The bytecode of the account (YP: \f$b\f$) */
        cuEVM::storage::contract_storage_t storage; /**< The storage of the account (YP: \f$\sigma[a]_{s}\f$) */

        /**
         * The default constructor for the account data structure.
         */
        __host__ __device__ account_t() = default;

        /**
         * The constructor for the account data structure.
         * @param[in] account_json The json object
         * @param[in] managed The flag to indicate if the memory is managed
         */
        __host__ account_t(
            const cJSON *account_json,
            bool managed = false);
        
        /**
         * The copy constructor for the account data structure.
         * @param[in] account The account data structure
         */
        __host__ __device__ account_t(
            const account_t &account);

        
        __host__ __device__ account_t(
            const account_t &account,
            bool with_storage);
        
        /**
         * The destructor for the account data structure.
         */
        __host__ __device__ int32_t free_internals(
            int32_t managed = 0);


        /**
         * Get the storage value for the given key.
         * @param[in] arith The arithmetical environment
         * @param[in] key The key
         * @param[out] value The value
         * @return If found 1, otherwise 0
         */
        __host__ __device__ int32_t get_storage_value(
            ArithEnv &arith,
            const bn_t &key,
            bn_t &value);
        /**
         * Set the storage value for the given key.
         * @param[in] arith The arithmetical environment
         * @param[in] key The key of the storage
         * @param[in] value The value of the storage
         * @return If found succesfull, otherwise 0
         */
        __host__ __device__ int32_t set_storage_value(
            ArithEnv &arith,
            const bn_t &key,
            const bn_t &value);
        
        /**
         * Get the address of the account.
         * @param[in] arith The arithmetical environment
         * @param[out] address The address of the account
         */
        __host__ __device__ void get_address(
            ArithEnv &arith,
            bn_t &address);
        
        /**
         * Get the balance of the account.
         * @param[in] arith The arithmetical environment
         * @param[out] balance The balance of the account
         */
        __host__ __device__ void get_balance(
            ArithEnv &arith,
            bn_t &balance);
        
        /**
         * Get the nonce of the account.
         * @param[in] arith The arithmetical environment
         * @param[out] nonce The nonce of the account
         */
        __host__ __device__ void get_nonce(
            ArithEnv &arith,
            bn_t &nonce);
        
        /**
         * Verify if the account has the the given address.
         * @param[in] arith The arithmetical environment
         * @param[in] address The address
         * @return If found 1, otherwise 0
         */
        __host__ __device__ int32_t has_address(
            ArithEnv &arith,
            const bn_t &address);
            
        /**
         * Setup the account data structure from the json object.
         * @param[in] account_json The json object
         * @param[in] managed The flag to indicate if the memory is managed
         */
        __host__ void from_json(
            const cJSON *account_json,
            bool managed = false);
        
        /**
         * Generate a JSON object from the account data structure.
         * @return The JSON object
         */
        __host__ cJSON* to_json(
            int32_t managed = 0);

        /**
         * Print the account data structure.
         */
        __host__ __device__ void print();

    };

    /**
     * The kernel to copy the account data structures.
     * @param[out] dst The destination account data structure
     * @param[in] src The source account data structure
     * @param[in] count The number of instances
     */
    __global__ void transfer_kernel(
        account_t *dst_instances,
        account_t *src_instances,
        uint32_t count);
   
    /**
     * Free the memory of the internal byte_code and storage
     * of the account.
     * @param[inout] account The account data structure
     * @param[in] managed The flag to indicate if the memory is managed
     */
    __host__ __device__ void free_internals_account(
        account_t &account,
        int32_t managed = 0);

    /**
     * Generate the cpu data structures for the account.
     * @param[in] count The number of instances
     * @return The cpu data structures
     */
    __host__ account_t *get_cpu_instances(
        uint32_t count);
    
    /**
     * Free the cpu data structures for the account.
     * @param[inout] cpu_instances The cpu data structures
     * @param[in] count The number of instances
     */
    __host__ void free_cpu_instances(
        account_t *cpu_instances,
        uint32_t count);
    
    /**
     * Generate the gpu data structures for the account.
     * @param[in] cpu_instances The cpu data structures
     * @param[in] count The number of instances
     */
    __host__ account_t *get_gpu_instances_from_cpu_instances(
        account_t *cpu_instances,
        uint32_t count);
    
    /**
     * Free the gpu data structures for the account.
     * @param[inout] gpu_instances The gpu data structures
     * @param[in] count The number of instances
     */
    __host__ void free_gpu_instances(
        account_t *gpu_instances,
        uint32_t count);
    
    /**
     * Generate the cpu data structures for the account from the gpu data structures.
     * @param[in] gpu_instances The gpu data structures
     * @param[in] count The number of instances
     * @return The cpu data structures
     */
    __host__ account_t *get_cpu_from_gpu_instances(
        account_t *gpu_instances,
        uint32_t count);
    
    /**
     * Get the managed instances.
     * @param[in] count The number of instances
     * @return The managed instances
     */
    __host__ account_t *get_managed_instances(
        uint32_t count);
    /**
     * Free the internal for managed instance.
     * @param[inout] managed_instance The managed instance
     */
    __host__ void free_internals_managed_instance(
        account_t &managed_instance);
    /**
     * Free the managed instances.
     * @param[inout] managed_instances The managed instances
     * @param[in] count The number of instances
     */
    __host__ void free_managed_instances(
        account_t *managed_instances,
        uint32_t count);
    
    /**
     * Fill the account data structure from the json.
     * @param[out] account The account data structure
     * @param[in] json The json object
     * @param[in] managed The flag to indicate if the memory is managed
     */
    __host__ void from_json(
        account_t &account,
        const cJSON *account_json,
        bool managed = false);
    
    /**
     * Generate a JSON object from the account data structure.
     * @param[in] account The account data structure
     * @return The JSON object
     */
    __host__ cJSON* json(
        account_t &account);
    
    /**
     * Print the account data structure.
     * @param[in] account The account data structure
     */
    __host__ __device__ void print(
        account_t &account);
    
    /**
     * Get the storage index for the given key.
     * @param[out] index The index of the storage
     * @param[in] arith The arithmetical environment
     * @param[in] account The account data structure
     * @param[in] key The key
     * @return If found 1, otherwise 0
     */
    __host__ __device__ int32_t get_storage_index(
        int32_t &index,
        ArithEnv &arith,
        const account_t &account,
        bn_t &key);
    
    /**
     * Make the account empty.
     * @param[inout] account The account data structure
     */
    __host__ __device__ void empty(
        account_t &account);
    
    /**
     * Duplicate the account data structure.
     * @param[out] dst The destination account data structure
     * @param[in] src The source account data structure
     * @param[in] with_storage The flag to indicate if the storage should be duplicated
     */
    __host__ __device__ void duplicate(
        account_t &dst,
        const account_t &src,
        bool with_storage = false);
    
    /**
     * Check if the account is empty.
     * @param[in] arith The arithmetical environment
     * @param[in] account The account data structure
     * @return 1 if the account is empty, 0 otherwise
     */
    __host__ __device__ int32_t is_empty(
        ArithEnv &arith,
        account_t &account);
  }
}
#endif