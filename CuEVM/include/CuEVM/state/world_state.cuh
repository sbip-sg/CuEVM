#ifndef _CUEVM_WORLD_STATE_H_
#define _CUEVM_WORLD_STATE_H_

#include <CuEVM/utils/arith.cuh>
#include <CuEVM/state/account.cuh>
#include <CuEVM/state/state.cuh>


namespace cuEVM {
    namespace state {
        /**
         * The world state classs
         */
        class WorldState {
        private:
            state_t* _state; /**< The state with the accounts */
        public:
            /**
             * The default constructor
             */
            __host__ __device__ WorldState() : _state(nullptr) {}
            /**
             * The constructor with a pointer to the state struct
             * @param[in] state pointer to the state
             */
            __host__ __device__ WorldState(state_t* state) : _state(state) {}
            /**
             * The destuctor
             */
            __host__ __device__ ~WorldState() {_state = nullptr;}
            
            /**
             * Get the account pointer given the address
             * @param[in] arith The arithmetic value to be processed.
             * @param[in] address The address for the account
             * @param[out] account_ptr  The pointer to the account
             * @return if found 0, otherwise error.
             */
            __host__ __device__ int32_t get_account(
                ArithEnv &arith,
                const bn_t &address,
                cuEVM::account::account_t* &account_ptr);

            /**
             * Get the value of a storage element
             * @param[in] arith The arithemtic environment.
             * @param[in] address The blockchain address of the account whose storage value is being queried. This parameter specifies the account uniquely.
             * @param[in] key The key corresponding to the storage value within the account's storage to be retrieved.
             * @param[out] value The storage value
             * @return Returns 0 if the storage value for the given account and key was successfully found and value has been updated accordingly. Returns error if the account or the key within the account's storage could not be found and the value is set to 0.
             */
            __host__ __device__ int32_t get_value(
                ArithEnv &arith,
                const bn_t &address,
                const bn_t &key,
                bn_t &value);
        };
    }
}


#endif