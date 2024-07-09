#include <CuCrypto/keccak.cuh>

#include "include/evm_utils.cuh"
#include "include/evm_defines.h"

namespace cuEVM {
    namespace utils {
        
        __host__ __device__ int32_t get_contract_address_create(
            ArithEnv &arith,
            bn_t &contract_address,
            const bn_t &sender_address,
            const bn_t &sender_nonce) {
            
            evm_word_t sender_address_word;
            cgbn_store(arith.env, &sender_address_word, sender_address);
            evm_word_t sender_nonce_word;
            cgbn_store(arith.env, &sender_nonce_word, sender_nonce);
            cuEVM::byte_array_t sender_address_bytes, sender_nonce_bytes;
            sender_address_word.to_byte_array_t(&sender_address_bytes);
            sender_nonce_word.to_byte_array_t(&sender_nonce_bytes);

            uint32_t nonce_bytes;
            for (nonce_bytes = EVM_WORD_SIZE; nonce_bytes > 0; nonce_bytes--) {
                if (sender_nonce_bytes.data[EVM_WORD_SIZE - nonce_bytes] != 0) {
                break;
                }
            }
            // TODO: this might work only for EVM_WORD_SIZE == 32

            uint8_t rlp_list[1 + 1 + EVM_ADDRESS_SIZE + 1 + EVM_WORD_SIZE];
            rlp_list[1] = 0x80 + EVM_ADDRESS_SIZE;
            for (uint32_t idx = 0; idx < EVM_ADDRESS_SIZE; idx++)
            {
                rlp_list[2 + idx] = sender_address_bytes.data[EVM_WORD_SIZE - EVM_ADDRESS_SIZE + idx];
            }

            uint32_t rlp_list_length;
            // 21 is from the address the 20 bytes is the length of the address
            // and the 1 byte is the 0x80 + length of the address (20)
            if (cgbn_compare_ui32(arith.env, sender_nonce, 128) < 0)
            {
                rlp_list_length = 1 + EVM_ADDRESS_SIZE + 1;
                if (cgbn_compare_ui32(arith.env, sender_nonce, 0)  == 0)
                {
                rlp_list[2 + EVM_ADDRESS_SIZE] = 0x80; // special case for nonce 0
                }
                else
                {
                rlp_list[2 + EVM_ADDRESS_SIZE] = sender_nonce_bytes.data[EVM_WORD_SIZE - 1];
                }
            }
            else
            {
                // 1 byte for the length of the nonce
                // 0x80 + length of the nonce
                rlp_list_length = 21 + 1 + nonce_bytes;
                rlp_list[2 + EVM_ADDRESS_SIZE] = 0x80 + nonce_bytes;
                for (uint8_t idx = 0; idx < nonce_bytes; idx++)
                {
                rlp_list[2 + EVM_ADDRESS_SIZE + 1 + idx] = sender_nonce_bytes.data[EVM_WORD_SIZE - nonce_bytes + idx];
                }
            }
            rlp_list[0] = 0xc0 + rlp_list_length;

            uint8_t address_bytes[EVM_HASH_SIZE];
            cuEVM::byte_array_t hash_address_bytes(EVM_HASH_SIZE);
            CuCrypto::keccak::sha3(
                &(rlp_list[0]),
                rlp_list_length + 1,
                hash_address_bytes.data,
                EVM_HASH_SIZE);
            for (uint8_t idx = 0; idx < EVM_WORD_SIZE - EVM_ADDRESS_SIZE; idx++)
            {
                address_bytes[idx] = 0;
            }
            evm_word_t contract_address_word;
            
            contract_address_word.from_byte_array_t(hash_address_bytes);
            cgbn_load(arith.env, contract_address, &contract_address_word);
            return 1;
        }

        __host__ __device__ int32_t get_contract_address_create2(
            ArithEnv &arith,
            bn_t &contract_address,
            const bn_t &sender_address,
            const bn_t &salt,
            const cuEVM::byte_array_t &init_code) {
            evm_word_t sender_address_word;
            cgbn_store(arith.env, &sender_address_word, sender_address);
            evm_word_t salt_word;
            cgbn_store(arith.env, &salt_word, salt);
            cuEVM::byte_array_t sender_address_bytes, salt_bytes;
            sender_address_word.to_byte_array_t(&sender_address_bytes);
            salt_word.to_byte_array_t(&salt_bytes);

            uint32_t total_bytes = 1 + EVM_ADDRESS_SIZE + EVM_WORD_SIZE + EVM_HASH_SIZE;

            cuEVM::byte_array_t hash_code(EVM_HASH_SIZE);
            CuCrypto::keccak::sha3(
                init_code.data,
                init_code.size,
                hash_code.data,
                EVM_HASH_SIZE);
            
            cuEVM::byte_array_t input_data(total_bytes);
            input_data.data[0] = 0xff;
            for (uint32_t idx = 0; idx < EVM_ADDRESS_SIZE; idx++)
            {
                input_data.data[1 + idx] = sender_address_bytes.data[EVM_WORD_SIZE - EVM_ADDRESS_SIZE + idx];
            }
            for (uint32_t idx = 0; idx < EVM_WORD_SIZE; idx++)
            {
                input_data.data[1 + EVM_ADDRESS_SIZE + idx] = salt_bytes.data[EVM_WORD_SIZE - EVM_WORD_SIZE + idx];
            }
            for (uint32_t idx = 0; idx < EVM_HASH_SIZE; idx++)
            {
                input_data.data[1 + EVM_ADDRESS_SIZE + EVM_WORD_SIZE + idx] = hash_code.data[idx];
            }

            cuEVM::byte_array_t hash_input_data(EVM_HASH_SIZE);
            CuCrypto::keccak::sha3(
                input_data.data,
                total_bytes,
                hash_input_data.data,
                EVM_HASH_SIZE);
            
            for (uint32_t idx = 0; idx < EVM_WORD_SIZE - EVM_ADDRESS_SIZE; idx++)
            {
                hash_input_data.data[idx] = 0;
            }

            evm_word_t contract_address_word;
            contract_address_word.from_byte_array_t(hash_input_data);
            cgbn_load(arith.env, contract_address, &contract_address_word);
            return 1;
        }
    }
}