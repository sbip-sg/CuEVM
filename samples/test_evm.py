import json
import argparse
import os
import subprocess
from pprint import pprint
from solc_json_parser.combined_json_parser import CombinedJsonParser
from eth_abi import encode
from eth_utils import function_abi_to_4byte_selector



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
