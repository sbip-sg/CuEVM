
import json
import argparse
import os
import subprocess
from pprint import pprint
from solc_json_parser.combined_json_parser import CombinedJsonParser
from eth_abi import encode
from eth_utils import function_abi_to_4byte_selector

SAMPLE_OPCODES = {1 : "+", 2 : "*", 3 : "-"} # samples for testing of overflow, underflow

def parse_trace_and_detect_bug(trace_file, contract_name, ast_parser = None):
    evm_output = json.loads(open(trace_file).read())
    traces = list(evm_output.values())[0].get("post",[])[0].get("traces", [])
    all_bugs = []
    for idx,item in enumerate(traces):
        if item.get("opcode") in SAMPLE_OPCODES:
            if idx <2: # skip the first two items
                continue
            prev_stack = traces[idx-1].get("stack").get("data")
            curr_stack = item.get("stack").get("data")
            # print (prev_stack, curr_stack)
            a, b = int(prev_stack[-1],16), int(prev_stack[-2],16)
            res = int(curr_stack[-1],16)
            print ("-"*80)
            print (f"found operation {a} {SAMPLE_OPCODES[item.get('opcode')]} {b} = {res}")
            bug_pc = None
            if item.get("opcode") in [1,2]:
                if res < a or res < b:
                    print (f"overflow detected at program counter {item.get('pc')}")
                    all_bugs.append({"overflow": item.get("pc")})
                    bug_pc = item.get("pc")
            if item.get("opcode") == 3:
                if res > a and res > b:
                    print (f"underflow detected at program counter {item.get('pc')}")
                    all_bugs.append({"underflow": item.get("pc")})
                    bug_pc = item.get("pc")
            if bug_pc and ast_parser :
                try:
                    frag = ast_parser.source_by_pc(contract_name, bug_pc, deploy=False)
                    lines = frag.get("linenums",[0,0])
                    if lines[0] == lines[1]:
                        print (f"Line: {lines[0]} : Source {frag.get('fragment')}")
                    else:
                        print (f"Line: {lines[0]} - {lines[1]} : Source {frag.get('fragment')}")
                    # print (frag)
                except:
                    pass
            print ("-"*80)
    return all_bugs


def get_transaction_data_from_config(item, contract_instance):
    print(item)
    # print (contract_instance)
    target_function = item["function"]
    inputs = item["input"]
    print("Target function inputs ", target_function, inputs)
    # print (contract_instance['abi'])
    function_abi = [
        i
        for i in contract_instance["abi"]
        if i.get("name") == target_function and i.get("type") == "function"
    ]
    input_types = item["input_types"]
    # print (function_abi[0])
    function_abi = function_abi[0]
    four_byte = "0x" + function_abi_to_4byte_selector(function_abi).hex()
    print(four_byte)
    if len(inputs) > 0:
        encoded_data = encode(input_types, inputs).hex()
    else:
        encoded_data = ""

    return [four_byte + encoded_data]



def decode_abi_bin_from_compiled_json(compiled_sol):
    m = {}
    for contract_name in compiled_sol.keys():
        abi = compiled_sol.get(contract_name).get("abi")
        binary = compiled_sol.get(contract_name).get("bin")
        binary_runtime = compiled_sol.get(contract_name).get("bin-runtime")
        storage_layout = compiled_sol.get(contract_name).get("storage-layout")
        m[contract_name] = dict(
            abi=abi,
            binary_runtime=binary_runtime,
            binary=binary,
            contract=contract_name,
            storage_layout=storage_layout,
        )
    return m


def compile_file(
    file_path="contracts/state_change.sol", contract_name="TestStateChange"
):
    print ("compile file ", file_path, contract_name)
    # The input can be a file path or source code
    parser = CombinedJsonParser(file_path, try_install_solc=True)
    compiler_output = parser.original_compilation_output
    # print (compiler_output)
    compiled_sol = {k.split(":")[-1]: v for k, v in compiler_output.items()}

    m_contracts = decode_abi_bin_from_compiled_json(compiled_sol)
    contract_instance = m_contracts.get(contract_name)
    contract_bin_runtime = contract_instance.get("binary_runtime")
    return contract_instance, parser

