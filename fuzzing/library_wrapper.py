"""
library wrapper to maintain state of EVM instances and run tx on them
"""

import sys
import ctypes
import json
import copy
from pprint import pprint
import time
from utils import *

# Add the directory containing your .so file to the Python path
sys.path.append("../build/")
# sys.path.append("./binary/")

import libcuevm  # Now you can import your module as usual

class CuEVMLib:
    def __init__(
        self,
        source_file,
        num_instances,
        config=None,
        contract_name=None,
        detect_bug=False,
        sender=int("0x1111111111111111111111111111111111111111",16),
        contract_bin_runtime=None,
        run_eth_tests=False,
    ):
        self.initiate_instance_data(
            source_file,
            num_instances,
            config,
            contract_name,
            detect_bug,
            contract_bin_runtime,
            run_eth_tests,
        )
        self.sender = sender

    def update_persistent_state(self, json_result):
        trace_values = json_result
        # print ("trace value result")
        # pprint(json_result)

        if trace_values is None or trace_values.get("states") is None:
            print("Skipping updating state")
            return
        for i in range(len(trace_values.get("states"))):
            post_state = trace_values.get("states")[i]
            if post_state is None:
                # print(f"Skipping updating state for instance {i}")
                continue
            # print("\n\n post_state %d \n\n" % i)
            # pprint(post_state)
            # self.instances[i]["pre"] = copy.deepcopy(post_state)
            # Nov update: copy nonce balance storage and not code
            for key, value in post_state.items():
                if key in self.instances[i]["pre"]:
                    self.instances[i]["pre"][key].update(value)
                else:
                    self.instances[i]["pre"][key] = value
                    self.instances[i]["pre"][key]["code"] = b""
                # self.instances[i]["pre"][key]["nonce"] = hex(
                #     self.instances[i]["pre"][key]["nonce"]
                # )
                self.instances[i]["pre"][key]["nonce"] = self.instances[i]["pre"][key]["nonce"]

                # self.instances[i]["pre"][key]["storage"] = value.get("storage")

            # sender = next_config["transaction"]["sender"]
            
            self.instances[i]["transaction"]["nonce"] = post_state.get(self.sender, {}).get("nonce", 0)
            
    def run_eth_tests(self):
        result_state = libcuevm.run_dict(self.instances)
        return result_state
    ## 1. run transactions on the EVM instances
    ## 2. update the persistent state of the EVM instances
    ## 3. return the simplified trace during execution
    def run_transactions(self, tx_data, skip_trace_parsing=False, copy_state_data=True, reuse_state_data=False, measure_performance=False):
        self.build_instance_data(tx_data)
        # print("instances")
        # pprint(self.instances)
        if measure_performance:
            time_start = time.time()
        result_state = libcuevm.run_dict(self.instances, skip_trace_parsing, copy_state_data, reuse_state_data)
        if measure_performance:
            time_end = time.time()
            print(f"Time taken: {time_end - time_start} seconds")
        if copy_state_data:
            self.update_persistent_state(result_state)
        # print("result after running transactions")
        # pprint(self.instances)
        return self.post_process_trace(result_state)

    # post process the trace to detect integer bugs and simplify the distance
    def post_process_trace(self, json_result):
        if json_result is None or json_result.get("traces") is None:
            print("Skipping post processing")
            return []
        final_trace = json_result.get("traces")
        storage_write = []
        # print("\ntrace\n")
        # pprint(trace)
        if self.detect_bug:
            for tx_trace in final_trace:
                bugs = []
                for current_event in tx_trace.get("events", []):
                    if current_event.opcode == OP_SSTORE:
                        storage_write.append(
                            EVMStorageWrite(
                                pc=current_event.pc,
                                key=current_event.operand_1,
                                value=current_event.operand_2,
                            )
                        )
                    if (current_event.opcode == OPADD and current_event.operand_1 + current_event.operand_2 >= 2**256):
                        bugs.append(EVMBug(current_event.pc, current_event.opcode, "integer overflow"))
                    elif (current_event.opcode == OPMUL and current_event.operand_1 * current_event.operand_2 >= 2**256):
                        bugs.append(EVMBug(current_event.pc, current_event.opcode, "integer overflow"))
                    elif (current_event.opcode == OPSUB and current_event.operand_1 < current_event.operand_2):
                        bugs.append(EVMBug(current_event.pc, current_event.opcode, "integer underflow"))
                    elif (current_event.opcode == OPEXP and current_event.operand_1 ** current_event.operand_2 >= 2**256):
                        bugs.append(EVMBug(current_event.pc, current_event.opcode, "integer overflow"))
                    elif current_event.opcode == OP_SELFDESTRUCT:
                        bugs.append(EVMBug(current_event.pc, current_event.opcode, "selfdestruct"))

                all_call = tx_trace.get("calls", [])
                for call in all_call:
                    if self.detect_bug:
                        if call.value > 0 and call.pc != 0:
                            bugs.append(
                                EVMBug(
                                    pc=call.pc,
                                    opcode=call.opcode,
                                    bug_type="Leaking Ether",
                                )
                            )
                tx_trace["bugs"] = bugs
                tx_trace["storage_write"] = storage_write
        return final_trace

    def convert_pre_state_to_int(self, pre_state):
        block_info = pre_state.get("env", {})
        for key, value in block_info.items():
            if isinstance(value, str):
                block_info[key] = int(value, 16)
        pre_state["env"] = block_info
        pre_state["target_address"] = int(pre_state["target_address"], 16)
        pre = pre_state.get("pre", {})
        new_pre = {}
        for key, value in pre.items():
            int_key = int(key, 16)
            # If a 'storage' dictionary is present, convert its keys from hex to int as well.
            if "storage" in value and isinstance(value["storage"], dict):
                new_storage = {}
                for stor_key, stor_value in value["storage"].items():
                    new_storage[int(stor_key, 16)] = int(stor_value, 16)
                value["storage"] = new_storage
            if "nonce" in value:
                value["nonce"] = int(value["nonce"], 16)
            if "balance" in value:
                value["balance"] = int(value["balance"], 16)
            if "code" in value:
                value["code"] = convert_hexstr_to_bytes(value["code"])
            new_pre[int_key] = value
        pre_state["pre"] = new_pre

        transaction = pre_state.get("transaction", {})
        transaction["to"] = int(transaction["to"], 16)
        transaction["sender"] = int(transaction["sender"], 16)
        transaction["value"] = [int(value, 16) for value in transaction["value"]]
        pre_state["transaction"] = transaction
        return pre_state
    
    def convert_tx_sequence_config_to_int(self, tx_sequence_config):
        storage = tx_sequence_config.get("storage", {})
        new_storage = {}
        for key, value in storage.items():
            new_storage[int(key, 16)] = int(value, 16)
        tx_sequence_config["storage"] = new_storage
        pre = tx_sequence_config.get("pre", {})
        new_pre = {}
        for key, value in pre.items():
            int_key = int(key, 16)
            # If a 'storage' dictionary is present, convert its keys from hex to int as well.
            if "storage" in value and isinstance(value["storage"], dict):
                new_storage = {}
                for stor_key, stor_value in value["storage"].items():
                    new_storage[int(stor_key, 16)] = int(stor_value, 16)
                value["storage"] = new_storage
            if "nonce" in value:
                value["nonce"] = int(value["nonce"], 16)
            if "balance" in value:
                value["balance"] = int(value["balance"], 16)
            if "code" in value:
                value["code"] = convert_hexstr_to_bytes(value["code"])
            new_pre[int_key] = value
        tx_sequence_config["pre"] = new_pre
        return tx_sequence_config

    ## initiate num_instances clones of the initial state
    def initiate_instance_data(
        self,
        source_file,
        num_instances,
        config=None,
        contract_name=None,
        detect_bug=False,
        contract_bin_runtime=None,
        run_eth_tests=False
    ):
        
        default_config = json.loads(open("configurations/default.json").read())
        default_config = self.convert_pre_state_to_int(default_config)
        # print(default_config)
        self.detect_bug = detect_bug
        # tx_sequence_list
        tx_sequence_config = json.loads(open(config).read())
        tx_sequence_config = self.convert_tx_sequence_config_to_int(tx_sequence_config)
        if run_eth_tests:
            self.instances = [copy.deepcopy(tx_sequence_config) for _ in range(num_instances)]
            return
        if contract_name is None:
            self.contract_name = tx_sequence_config.get("contract_name")
        else:
            self.contract_name = contract_name
        # print(f" source file {source_file} contract_name {self.contract_name} \n\n")
        self.contract_instance, self.ast_parser = compile_file(
            source_file, self.contract_name
        )
        if self.contract_instance is None:
            print("Error in compiling the contract {self.contract_name} {source_file}")
            return
        if contract_bin_runtime is None:
            contract_bin_runtime = self.contract_instance.get("binary_runtime")
        contract_bin_runtime = convert_hexstr_to_bytes(contract_bin_runtime)
        # the merged config fields : "env", "pre" (populated with code), "transaction" (populated with tx data and value)
        pre_env = tx_sequence_config.get("pre", {})
        
        new_test = {}
        new_test["env"] = default_config["env"].copy()
        new_test["pre"] = default_config["pre"].copy()
        
        
        new_test["pre"].update(pre_env)

        target_address = default_config["target_address"]

        new_test["pre"][target_address]["code"] = contract_bin_runtime

        new_test["pre"][target_address]["storage"] = tx_sequence_config.get(
            "storage", {}
        )

        new_test["transaction"] = default_config["transaction"].copy()

        new_test["transaction"]["to"] = target_address
        new_test["transaction"]["data"] = [b""]
        new_test["transaction"]["value"] = [0]
        new_test["transaction"]["nonce"] = 0
    
        self.instances = [copy.deepcopy(new_test) for _ in range(num_instances)]

    def print_instance_data(self):
        for idx, instance in enumerate(self.instances):
            print(f"\n\n Instance data {idx}\n\n")
            print_hex(instance)

    ## build instances data from new tx data
    ## tx_data is a list of tx data
    def build_instance_data(self, tx_data):
        # todo consider clearing pre_state data
        if len(tx_data) < len(self.instances):
            tx_data = tx_data + [tx_data[-1]] * (len(self.instances) - len(tx_data))
        if len(tx_data) > len(self.instances):
            tx_data = tx_data[: len(self.instances)]
        # print (f"tx_data_rebuilt {tx_data}")
        for i in range(len(tx_data)):
            self.instances[i]["transaction"]["data"] = tx_data[i]["data"]
            self.instances[i]["transaction"]["value"] = tx_data[i]["value"]
            if tx_data[i].get("sender"):
                self.instances[i]["transaction"]["sender"] = tx_data[i]["sender"]

            # TODO: add other fuzz-able fields


def test_state_change():
    my_lib = CuEVMLib(
        "contracts/state_change.sol",
        3,
        "configurations/state_change.json",
        # contract_bin_runtime="6011602201600460110260005560015561123460015561ffff60ff5500",
        # contract_bin_runtime="6042611234621234567F123456789101112131415161718192021000"
    )
    test_case = {
        "function": "increase",
        "type": "exec",
        "input_types": [],
        "input": [],
        "sender": 0,
    }

    tx_1 = {
        "data": get_transaction_data_from_config(
            test_case, my_lib.contract_instance
        ),  # must return an array
        "value": [0],
    }
    tx_2 = {
        "data": get_transaction_data_from_config(
            test_case, my_lib.contract_instance
        ),  # must return an array
        "value": [0],
    }

    # for debugging, altering tx2 data
    tx_2["data"] = [b"\x12"]
    tx_2["value"] = [10]
    my_lib.instances[0]["pre"][int("0xcccccccccccccccccccccccccccccccccccccccc",16)]["storage"][
        int("0x00",16)
    ] = int("0x10",16)
    # for debugging, altering the state 2
    my_lib.instances[1]["pre"][int("0xcccccccccccccccccccccccccccccccccccccccc",16)]["storage"][
        int("0x00",16)
    ] = int("0x2000",16)
    # my_lib.instances[1]["pre"]["0xcccccccccccccccccccccccccccccccccccccccc"][
    #     "balance"
    # ] = "0x00"
    my_lib.instances[2]["pre"][int("0xcccccccccccccccccccccccccccccccccccccccc",16)]["storage"][
        int("0x00",16)
    ] = int("0x30",16)
    
    trace_res = my_lib.run_transactions([tx_1],)
    # trace_res = my_lib.run_transactions([tx_1])
    # print("\n\n trace res \n\n")
    # pprint(trace_res)
    print("\n\n Updated instance data \n\n")
    my_lib.print_instance_data()

    trace_res = my_lib.run_transactions([tx_1, tx_2, tx_1])
    print("\n\n Updated instance data \n\n")
    my_lib.print_instance_data()

    # # trace_res = my_lib.run_transactions([tx_2, tx_1, tx_2])
    # # # trace_res = my_lib.run_transactions([tx_1, tx_1])
    # # print("\n\n trace res \n\n")
    # # pprint(trace_res)
    # print("\n\n Updated instance data \n\n")
    # my_lib.print_instance_data()



def test_erc20():
    my_lib = CuEVMLib(
        "contracts/erc20.sol",
        10000,
        "configurations/erc20.json",
        contract_name="ERC20",
        detect_bug=False,
    )
    test_case = {
        "function": "transfer",
        "type": "exec",
        "input_types": ["address", "uint256"],
        "input": ["0x0000000000000000000000000000000000000001", 512],
        "sender": 0,
    }

    tx_1 = {
        "data": get_transaction_data_from_config(
            test_case, my_lib.contract_instance
        ),  # must return an array
        "value": [512],
    }
    tx_2 = {
        "data": get_transaction_data_from_config(
            test_case, my_lib.contract_instance
        ),  # must return an array
        "value": [512],
    }
    # trace_res = my_lib.run_transactions([tx_1, tx_2], skip_trace_parsing=True)
    trace_res = my_lib.run_transactions([tx_1,tx_2])
    # print("\n\n trace res \n\n")
    # pprint(trace_res)


def test_branching():
    my_lib = CuEVMLib(
        "contracts/branching.sol",
        2,
        "configurations/test_branching.json",
    )
    test_case_1 = {
        "function": "test_branch",
        "type": "exec",
        "input_types": ["uint256"],
        "input": [12345],
        "sender": 0,
    }
    test_case_2 = {
        "function": "test_branch",
        "type": "exec",
        "input_types": ["uint256"],
        "input": [50],
        "sender": 0,
    }

    tx_1 = {
        "data": get_transaction_data_from_config(test_case_1, my_lib.contract_instance),
        "value": [0],
    }

    tx_2 = {
        "data": get_transaction_data_from_config(test_case_2, my_lib.contract_instance),
        "value": [0],
    }
    trace_res = my_lib.run_transactions([tx_1, tx_2])
    print("\n\n trace res \n\n")
    pprint(trace_res)


def test_system_operation():
    my_lib = CuEVMLib(
        "contracts/system_operations.sol",
        2,
        "configurations/system_operation.json",
    )
    test_case = {
        "function": "test_call",
        "type": "exec",
        "input_types": ["address", "uint256"],
        "input": ["0x1000000000000000000000000000000000000000", 0x1],
        "sender": 0,
    }

    tx_1 = {
        "data": get_transaction_data_from_config(test_case, my_lib.contract_instance),
        "value": [1234],
    }
    print("instance data")
    pprint(my_lib.instances)
    # my_lib.print_instance_data()
    print("tx_1")
    pprint(tx_1)
    trace_res = my_lib.run_transactions([tx_1])
    print("\n\n trace res \n\n")
    pprint(trace_res)

def test_bugs_simple():
    my_lib = CuEVMLib(
        "contracts/test_bugs_simple.sol",
        2,
        "configurations/default.json",
        contract_name="TestBug",
        detect_bug=True,
    )
    test_case = {
        "function": "bug_combined",
        "type": "exec",
        "input_types": [],
        "input": [],
        "sender": 0,
        
    }
    # print("instance data")
    # pprint(my_lib.instances)

    tx_1 = {
        "data": get_transaction_data_from_config(test_case, my_lib.contract_instance),
        "value": [0],
    }

    trace_res = my_lib.run_transactions([tx_1], measure_performance=True, skip_trace_parsing=False)
    print("\n\n trace res \n\n")
    if trace_res is not None and len(trace_res) > 0:
        pprint(trace_res[0])

def test_run_eth_tests():
    my_lib = CuEVMLib(
        "contracts/test_bugs_simple.sol",
        2,
        "configurations/calldata_load.json",
        run_eth_tests=True,
    )
    # print("\n\n instance data \n\n")
    # pprint(my_lib.instances)

    trace_res = my_lib.run_eth_tests()
    print("\n\n trace res \n\n")
    pprint(trace_res)
# def test_erc20():
#     my_lib = CuEVMLib(
#         "contracts/erc20.sol",
#         1,
#         "configurations/erc20.json",
#         contract_name="ERC20",
#         detect_bug=False,
#     )
#     test_case = {
#         "function": "mint",
#         "type": "exec",
#         "input_types": [],
#         "input": [],
#         "sender": 0,
#     }
#     test_case_2 = {
#         "function": "transfer",
#         "type": "exec",
#         "input_types": ["address", "uint256"],
#         "input": ["0x0000000000000000000000000000000000000001", 512],
#         "sender": 0,
#     }
#     # print("instance data")
#     # pprint(my_lib.instances)

#     tx_1 = {
#         "data": get_transaction_data_from_config(test_case, my_lib.contract_instance),
#         "value": [hex(0)],
#     }
#     tx_2 = {
#         "data": get_transaction_data_from_config(test_case_2, my_lib.contract_instance),
#         "value": [hex(0)],
#     }
#     trace_res = my_lib.run_transactions([tx_1], measure_performance=True, skip_trace_parsing=True)
#     # trace_res_2 = my_lib.run_transactions([tx_2], measure_performance=True, skip_trace_parsing=True)
#     print("\n\n trace res \n\n")
#     if trace_res is not None and len(trace_res) > 0:
#         pprint(trace_res[0])

def test_cross_contract():
    my_lib = CuEVMLib(
        "contracts/cross_contract.sol",
        1,
        "configurations/cross_contract.json",
        detect_bug=True,
    )
    test_case = {
        "function": "underflow",
        "type": "exec",
        "input_types": ["address", "address"],
        "input": [
            "0x1000000000000000000000000000000000000000",
            "0x2000000000000000000000000000000000000000",
        ],
        "value": 300,
        "sender": 0,
        "receiver": "0x1000000000000000000000000000000000000000",
    }

    tx_1 = {
        "data": get_transaction_data_from_config(test_case, my_lib.contract_instance),
        "value": [300],
    }

    trace_res = my_lib.run_transactions([tx_1])
    print("\n\n trace res \n\n")
    pprint(trace_res)

if __name__ == "__main__":

    # test_system_operation()
    # test_cross_contract()
    # test_erc20()

    def run_test_case(test_case_name):
        if test_case_name == "system_operation":
            test_system_operation()
        elif test_case_name == "cross_contract":
            test_cross_contract()
        elif test_case_name == "erc20":
            test_erc20()
        elif test_case_name == "state_change":
            test_state_change()
        elif test_case_name == "bugs_simple":
            test_bugs_simple()
        elif test_case_name == "branching":
            test_branching()
        elif test_case_name == "run_eth_tests":
            test_run_eth_tests()
        else:
            print(f"Unknown test case: {test_case_name}")

    if len(sys.argv) > 1:
        run_test_case(sys.argv[1])
    else:
        print("Please provide a test case name as a command line argument.")
    # test_state_change()
