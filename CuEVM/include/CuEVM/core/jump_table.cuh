#pragma once

#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <unordered_map>
#include <vector>
#include <functional>

#include <CuEVM/core/byte_array.cuh>
#include <CuEVM/core/evm_word.cuh>


#define GLOBAL_JUMP_TABLE_SIZE (1024 * 1024 * 1)


namespace CuEVM {
// Structure to store valid jump destinations for a contract
// Hash function for evm_word_t to use in unordered_map
struct EVM_Word_Hash {
    std::size_t operator()(const evm_word_t& value) const {
        // Combine multiple parts of the address
        return std::hash<uint64_t>()(value.words[0]) ^ 
               (std::hash<uint64_t>()(value.words[1]) << 1) ^
               (std::hash<uint64_t>()(value.words[2]) << 2) ^
               (std::hash<uint64_t>()(value.words[3]) << 3);
    }
};

// Equality function for evm_word_t to use in unordered_map
struct EVM_Word_Equal {
    bool operator()(const evm_word_t& a, const evm_word_t& b) const {
        return a == b;
    }
};

// Global container for valid jump destinations
using ContractPCsMap = std::unordered_map<evm_word_t, std::vector<uint8_t>, EVM_Word_Hash, EVM_Word_Equal>;


struct GlobalJumpTable {
    uint8_t bitmap[GLOBAL_JUMP_TABLE_SIZE];

    // Analyze the bytecode and fill the jump table. Returns non-zero when failed
    __host__ int32_t analyze(uint8_t* bytecode, uint32_t bitmap_offset_start, uint32_t size
#ifdef BUILD_GO_LIBRARY
    , evm_word_t contract_addr
#endif
    );

    // Check if the pc is a valid jump destination, assuming the `bitmap_offset_start` contains valid value
    // Returns
    //  - 0 if the pc is a valid jumpdest
    //  - ERROR_INVALID_JUMP_DESTINATION if it's not valid jumpdest
    //  - other values if there is an error
    __host__ __device__ int32_t validate_jumpdest(uint32_t bitmap_offset_start, uint32_t pc);

};

}
#ifdef BUILD_GO_LIBRARY
    extern CuEVM::ContractPCsMap contract_pcs_map;
#endif
