#include <CuCrypto/keccak.cuh>
#include <CGBN/cgbn.h>

#include <CuEVM/utils/evm_utils.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace cuEVM {
    namespace utils {
        __host__ __device__ int32_t get_contract_address_create(
            ArithEnv &arith,
            bn_t &contract_address,
            const bn_t &sender_address,
            const bn_t &sender_nonce) {
            
            evm_word_t sender_address_word;
            cgbn_store(
                arith.env,
                reinterpret_cast<cgbn_evm_word_t_ptr>(&sender_address_word),
                sender_address);
            evm_word_t sender_nonce_word;
            cgbn_store(arith.env, &sender_nonce_word, sender_nonce);
            cuEVM::byte_array_t sender_address_bytes, sender_nonce_bytes;
            sender_address_word.to_byte_array_t(&sender_address_bytes);
            sender_nonce_word.to_byte_array_t(&sender_nonce_bytes);

            uint32_t nonce_bytes;
            for (nonce_bytes = cuEVM::word_size; nonce_bytes > 0; nonce_bytes--) {
                if (sender_nonce_bytes.data[cuEVM::word_size - nonce_bytes] != 0) {
                break;
                }
            }
            // TODO: this might work only for cuEVM::word_size == 32

            uint8_t rlp_list[1 + 1 + cuEVM::address_size + 1 + cuEVM::word_size];
            rlp_list[1] = 0x80 + cuEVM::address_size;
            for (uint32_t idx = 0; idx < cuEVM::address_size; idx++)
            {
                rlp_list[2 + idx] = sender_address_bytes.data[cuEVM::word_size - cuEVM::address_size + idx];
            }

            uint32_t rlp_list_length;
            // 21 is from the address the 20 bytes is the length of the address
            // and the 1 byte is the 0x80 + length of the address (20)
            if (cgbn_compare_ui32(arith.env, sender_nonce, 128) < 0)
            {
                rlp_list_length = 1 + cuEVM::address_size + 1;
                if (cgbn_compare_ui32(arith.env, sender_nonce, 0)  == 0)
                {
                rlp_list[2 + cuEVM::address_size] = 0x80; // special case for nonce 0
                }
                else
                {
                rlp_list[2 + cuEVM::address_size] = sender_nonce_bytes.data[cuEVM::word_size - 1];
                }
            }
            else
            {
                // 1 byte for the length of the nonce
                // 0x80 + length of the nonce
                rlp_list_length = 21 + 1 + nonce_bytes;
                rlp_list[2 + cuEVM::address_size] = 0x80 + nonce_bytes;
                for (uint8_t idx = 0; idx < nonce_bytes; idx++)
                {
                rlp_list[2 + cuEVM::address_size + 1 + idx] = sender_nonce_bytes.data[cuEVM::word_size - nonce_bytes + idx];
                }
            }
            rlp_list[0] = 0xc0 + rlp_list_length;

            uint8_t address_bytes[cuEVM::hash_size];
            cuEVM::byte_array_t hash_address_bytes(cuEVM::hash_size);
            CuCrypto::keccak::sha3(
                &(rlp_list[0]),
                rlp_list_length + 1,
                hash_address_bytes.data,
                cuEVM::hash_size);
            for (uint8_t idx = 0; idx < cuEVM::word_size - cuEVM::address_size; idx++)
            {
                address_bytes[idx] = 0;
            }
            evm_word_t contract_address_word;
            
            contract_address_word.from_byte_array_t(hash_address_bytes);
            cgbn_load(arith.env, contract_address, &contract_address_word);
            return ERROR_SUCCESS;
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

            uint32_t total_bytes = 1 + cuEVM::address_size + cuEVM::word_size + cuEVM::hash_size;

            cuEVM::byte_array_t hash_code(cuEVM::hash_size);
            CuCrypto::keccak::sha3(
                init_code.data,
                init_code.size,
                hash_code.data,
                cuEVM::hash_size);
            
            cuEVM::byte_array_t input_data(total_bytes);
            input_data.data[0] = 0xff;
            for (uint32_t idx = 0; idx < cuEVM::address_size; idx++)
            {
                input_data.data[1 + idx] = sender_address_bytes.data[cuEVM::word_size - cuEVM::address_size + idx];
            }
            for (uint32_t idx = 0; idx < cuEVM::word_size; idx++)
            {
                input_data.data[1 + cuEVM::address_size + idx] = salt_bytes.data[cuEVM::word_size - cuEVM::word_size + idx];
            }
            for (uint32_t idx = 0; idx < cuEVM::hash_size; idx++)
            {
                input_data.data[1 + cuEVM::address_size + cuEVM::word_size + idx] = hash_code.data[idx];
            }

            cuEVM::byte_array_t hash_input_data(cuEVM::hash_size);
            CuCrypto::keccak::sha3(
                input_data.data,
                total_bytes,
                hash_input_data.data,
                cuEVM::hash_size);
            
            for (uint32_t idx = 0; idx < cuEVM::word_size - cuEVM::address_size; idx++)
            {
                hash_input_data.data[idx] = 0;
            }

            evm_word_t contract_address_word;
            contract_address_word.from_byte_array_t(hash_input_data);
            cgbn_load(arith.env, contract_address, &contract_address_word);
            return ERROR_SUCCESS;
        }

        __host__ __device__ int32_t is_hex(const char hex) {
            return hex >= '0' && hex <= '9' ? 1 : (
                hex >= 'a' && hex <= 'f' ? 1 : (
                hex >= 'A' && hex <= 'F' ? 1 : 0
                )
            );
        }

        __host__ __device__ char hex_from_nibble(const uint8_t nibble) {
            return nibble < 10 ? '0' + nibble : 'a' + nibble - 10;
        }
        
        __host__ __device__ uint8_t nibble_from_hex(const char hex) {
            return hex >= '0' && hex <= '9' ? hex - '0' : (
                hex >= 'a' && hex <= 'f' ? hex - 'a' + 10 : (
                hex >= 'A' && hex <= 'F' ? hex - 'A' + 10 : 0
                )
            );
        }

        __host__ __device__ uint8_t byte_from_nibbles(const uint8_t high, const uint8_t low) {
            return (high << 4) | low;
        }
        
        __host__ __device__ void hex_from_byte(char *dst, const uint8_t byte){
            if (dst == NULL)
                return;
            dst[0] = hex_from_nibble(byte >> 4);
            dst[1] = hex_from_nibble(byte & 0x0F);
        }

        __host__ __device__ uint8_t byte_from_two_hex_char(const char high, const char low) {
            return byte_from_nibbles(nibble_from_hex(high), nibble_from_hex(low));
        }
        __host__ __device__ int32_t hex_string_length(
        const char *hex_string)
        {
            int32_t length;
            int32_t error = 0;
            char *current_char;
            current_char = (char *)hex_string;
            if (
                (hex_string[0] == '0') &&
                ((hex_string[1] == 'x') || (hex_string[1] == 'X'))
            ) {
                current_char += 2; // Skip the "0x" prefix
            }
            length = 0;
            do {
                length++;
                error = error | (nibble_from_hex(current_char[length]) == 0);
            } while(current_char[length] != '\0');
            return error ? -1 : length;
        }

        __host__ __device__ int32_t clean_hex_string(
        char **hex_string)
        {
            char *current_char;
            current_char = (char *)*hex_string;
            if (current_char == NULL || current_char[0] == '\0')
            {
                return -1;
            }
            if (
                (current_char[0] == '0') &&
                ((current_char[1] == 'x') || (current_char[1] == 'X'))
            ) {
                current_char += 2; // Skip the "0x" prefix
                *hex_string += 2;
            }
            if (current_char[0] == '\0')
            {
                return 0;
            }
            int32_t length = 0;
            int32_t error = 0;
            do {
                error = error || (is_hex(current_char[length++]) == 0);
            } while(current_char[length] != '\0');
            return error ? -1 : length;
        }

        __host__ __device__ int32_t hex_string_without_leading_zeros(
        char *hex_string) {
            int32_t length;
            char *current_char;
            current_char = (char *)hex_string;
            length = clean_hex_string(&current_char);
            if (length <= 0)
            {
                return 1;
            }
            int32_t prefix = current_char - hex_string;
            int32_t idx;
            for (idx = 0; idx < length; idx++)
            {
                if (*current_char++ != '0')
                {
                break;
                }
            }
            if (idx == length)
            {
                hex_string[prefix] = '0';
                hex_string[prefix + 1] = '\0';
            }
            else
            {
                char *dst_char;
                char *src_char;
                dst_char = (char *)hex_string;
                clean_hex_string(&dst_char);
                src_char = dst_char + idx;
                for (int32_t i = 0; i < length - idx; i++)
                {
                *(dst_char++) = *(src_char++);
                }
                *dst_char = '\0';
            }
            return 0;
        }

        __host__ cJSON *get_json_from_file(const char *filepath) {
            FILE *fp = fopen(filepath, "r");
            fseek(fp, 0, SEEK_END);
            long size = ftell(fp);
            fseek(fp, 0, SEEK_SET);
            char *buffer = (char *)malloc(size + 1);
            fread(buffer, 1, size, fp);
            fclose(fp);
            buffer[size] = '\0';
            // parse
            cJSON *root = cJSON_Parse(buffer);
            free(buffer);
            return root;
        }
    }
}