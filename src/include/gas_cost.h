#ifndef __GAS_COST_H__
#define __GAS_COST_H__

#define GAS_ZERO 0
#define GAS_JUMP_DEST 1
#define GAS_BASE 2
#define GAS_VERY_LOW 3
#define GAS_LOW 5
#define GAS_MID 8
#define GAS_HIGH 10
#define GAS_WARM_ACCESS 100
#define GAS_WARM_SLOAD GAS_WARM_ACCESS
#define GAS_SLOAD GAS_WARM_SLOAD
#define GAS_ACCESS_LIST_ADDRESS 2400
#define GAS_ACCESS_LIST_STORAGE 1900
#define GAS_COLD_ACCOUNT_ACCESS 2600
#define GAS_COLD_SLOAD 2100
#define GAS_STORAGE_SET 20000
#define GAS_STORAGE_RESET 2900
#define GAS_STORAGE_CLEAR_REFUND 4800 //can be defined as GAS_SRESET + GAS_ACCESS_LIST_STORAGE
#define GAS_SSTORE_RESET 5000
#define GAS_SSTORE_CLEARS_SCHEDULE 4800 // EIP-3529 SSTORE_RESET - COLD_SLOAD_COST + ACCESS_LIST_STORAGE_KEY = 5000 - 2100 + 1900 = 4800
#define GAS_WARM_SSOTRE_RESET 1900 // SSTORE_RESET - COLD_SLOAD_COST
#define GAS_SELFDESTRUCT 5000
#define GAS_CREATE 32000
#define GAS_CODE_DEPOSIT 200
#define GAS_CALL_VALUE 9000
#define GAS_CALL_STIPEND 2300
#define GAS_NEW_ACCOUNT 25000
#define GAS_EXP 10
#define GAS_EXP_BYTE 50
#define GAS_MEMORY 3
#define GAS_TX_CREATE 32000
#define GAS_TX_DATA_ZERO 4
#define GAS_TX_DATA_NONZERO 16
#define GAS_TRANSACTION 21000
#define GAS_LOG 375
#define GAS_LOG_DATA 8
#define GAS_LOG_TOPIC 375
#define GAS_KECCAK256 30
#define GAS_KECCAK256_WORD 6
#define GAS_COPY 3
#define GAS_BLOCKHASH 20
#define GAS_STIPEND 2300
#define GAS_INITCODE_WORD_COST 2
#define GAS_PRECOMPILE_ECRECOVER 3000
#define GAS_PRECOMPILE_SHA256 60
#define GAS_PRECOMPILE_SHA256_WORD 12
#define GAS_PRECOMPILE_RIPEMD160 600
#define GAS_PRECOMPILE_RIPEMD160_WORD 120
#define GAS_PRECOMPILE_IDENTITY 15
#define GAS_PRECOMPILE_IDENTITY_WORD 3
#define GAS_PRECOMPILE_MODEXP_MAX 200
#define GAS_PRECOMPILE_ECADD 150
#define GAS_PRECOMPILE_ECMUL 6000
#define GAS_PRECOMPILE_ECPAIRING 45000
#define GAS_PRECOMPILE_ECPAIRING_PAIR 34000
#define GAS_PRECOMPILE_BLAKE2_ROUND 1


#endif  // __GAS_COST_H__
