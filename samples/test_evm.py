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
    parser = CombinedJsonParser(file_path)
    compiler_output = parser.original_compilation_output
    # print (compiler_output)
    compiled_sol = {k.split(":")[-1]: v for k, v in compiler_output.items()}

    m_contracts = decode_abi_bin_from_compiled_json(compiled_sol)
    contract_instance = m_contracts.get(contract_name)
    contract_bin_runtime = contract_instance.get("binary_runtime")
    return contract_instance, parser


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


def contruct_next_test_case(config_file, next_config):
    with open(config_file + ".out", "r") as f:
        config = json.load(f)
    # print (config)
    post_state = list(config.values())[0].get("post")[0]
    touch_state = post_state.get("touch_state")
    # print("touch_state")
    # print(touch_state)
    next_config_key = list(next_config.keys())[0]
    next_config[next_config_key]["pre"].update(touch_state)
    sender = next_config[next_config_key]["transaction"]["sender"]
    next_config[next_config_key]["transaction"]["nonce"] = touch_state.get(sender).get(
        "nonce"
    )
    return next_config


def run_evm(config_file, executable="../out/cpu_interpreter", log_file_path='/tmp/evm_test.log'):
    with open(log_file_path, 'w') as log_file:
        subprocess.run(
            [executable, "--input", config_file, "--output", config_file + ".out"],
            stdout=log_file, stderr=log_file
        )


def run_test(source, config, evm_executable, output_path="/tmp/evm_test", detect_bug=False):
    default_config = json.loads(open("configurations/default.json").read())
    print(default_config)
    # tx_sequence_list
    tx_sequence_config = json.loads(open(config).read())
    contract_name = tx_sequence_config.get("contract_name")
    contract_instance, ast_parser = compile_file(source, contract_name)
    contract_bin_runtime = contract_instance.get("binary_runtime")
    merged_config = []
    # the merged config fields : "env", "pre" (populated with code), "transaction" (populated with tx data and value)
    pre_env = tx_sequence_config.get("pre", {})
    for idx, test_case in enumerate(tx_sequence_config.get("test_cases")):
        new_test = {}
        new_test["env"] = default_config["env"].copy()
        new_test["pre"] = default_config["pre"].copy()
        target_address = default_config["target_address"]
        new_test["pre"][target_address]["code"] = contract_bin_runtime
        new_test["pre"][target_address]["storage"] = tx_sequence_config.get(
            "storage", {}
        )
        new_test["pre"].update(pre_env)
        new_test["transaction"] = default_config["transaction"].copy()

        new_test["transaction"]["to"] = target_address
        new_test["transaction"]["data"] = get_transaction_data_from_config(
            test_case, contract_instance
        )  # must return an array
        new_test["transaction"]["value"] = [hex(test_case.get("value", 0))]
        new_test["transaction"]["nonce"] = "0x00"
        merged_config.append({f"{contract_name}_test_{idx}": new_test})

    if not os.path.exists(output_path):
        os.makedirs(output_path)


    next_test = merged_config[0]
    # print("next test ")
    # print(next_test)
    all_config_files = []
    for idx in range(len(merged_config)):
        # 1 test case per file
        test_name = list(next_test.keys())[0]
        # test_config = config_item.get(test_name)
        config_file = f"{output_path}/{test_name}.json"
        all_config_files.append(config_file)
        log_file_path = f"{output_path}/{test_name}.log"
        with open(f"{output_path}/{test_name}.json", "w") as f:
            f.write(json.dumps(next_test, indent=2))
        run_evm(config_file, evm_executable, log_file_path)
        if idx < len(merged_config) - 1:
            next_test = contruct_next_test_case(config_file, merged_config[idx + 1])

    if detect_bug:
        for config_file in all_config_files:
            parse_trace_and_detect_bug(config_file + ".out", contract_name, ast_parser)

if __name__ == "__main__":
    # run ./test_tx_sequence.py --source ./contracts/state_change.sol --config ./configurations/state_change.json --evm-executable ../out/cpu_interpreter --output_path ./evm_test
    # --detect-bug
    parser = argparse.ArgumentParser(description="Run EVM test cases")
    parser.add_argument(
        "--source", default="contracts/state_change.sol", help="source file"
    )
    parser.add_argument(
        "--config", default="configurations/state_change.json", help="config file"
    )
    parser.add_argument(
        "--evm-executable", default="../out/cpu_interpreter", help="evm executable"
    )
    parser.add_argument("--output-path", default="/tmp/evm_test", help="output path")
    parser.add_argument("--detect-bug", action="store_true", help="detect bug")
    args = parser.parse_args()
    print(args)
    run_test(args.source, args.config, args.evm_executable, args.output_path, args.detect_bug)
