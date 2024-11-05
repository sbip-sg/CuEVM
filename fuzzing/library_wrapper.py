"""
library wrapper to maintain state of EVM instances and run tx on them
"""

import sys
import ctypes
import json
import copy
from pprint import pprint
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
        sender="0x1111111111111111111111111111111111111111",
    ):
        self.initiate_instance_data(
            source_file, num_instances, config, contract_name, detect_bug
        )
        self.sender = sender

    def update_persistent_state(self, json_result):
        trace_values = json_result
        # print ("trace value result")
        # pprint(json_result)
        for i in range(len(trace_values.get("post"))):
            post_state = trace_values.get("post")[i].get("state")
            # print("\n\n post_state %d \n\n" % i)
            # pprint(post_state)
            # self.instances[i]["pre"] = copy.deepcopy(post_state)
            # Nov update: copy nonce balance storage and not code
            for key, value in post_state.items():
                if key in self.instances[i]["pre"]:
                    self.instances[i]["pre"][key].update(value)
                else:
                    self.instances[i]["pre"][key] = value
                    self.instances[i]["pre"][key]["code"] = "0x"
                self.instances[i]["pre"][key]["nonce"] = hex(
                    self.instances[i]["pre"][key]["nonce"]
                )

                # self.instances[i]["pre"][key]["storage"] = value.get("storage")

            # sender = next_config["transaction"]["sender"]
            self.instances[i]["transaction"]["nonce"] = hex(
                post_state.get(self.sender).get("nonce")
            )

    ## 1. run transactions on the EVM instances
    ## 2. update the persistent state of the EVM instances
    ## 3. return the simplified trace during execution
    def run_transactions(self, tx_data):
        self.build_instance_data(tx_data)
        # self.print_instance_data()
        # print ("before running")

        result_state = libcuevm.run_dict(self.instances)
        self.update_persistent_state(result_state)
        return self.post_process_trace(result_state)

    # post process the trace to detect integer bugs and simplify the distance
    def post_process_trace(self, trace):
        final_trace = []
        # print("\ntrace\n")
        # pprint(trace)
        for i in range(len(trace.get("post", []))):
            post_state = trace.get("post")[i].get("traces", {})
            missed_branches = []
            covered_branches = []
            bugs = []
            storage_write = []
            for branch in post_state.get("branches", [])[3:]:
                missed_branches.append(
                    [
                        str(branch.get("pc"))
                        + ","
                        + str(branch.get("missed_destination")),
                        branch.get("distance"),
                    ]
                )
                covered_branches.append(
                    [str(branch.get("pc")) + "," + str(branch.get("destination"))]
                )
            for bug in post_state.get("bugs", []):
                current_opcode = bug.get("opcode")
                if current_opcode in arith_ops:
                    op_1 = int(bug.get("operand_1"), 16)
                    op_2 = int(bug.get("operand_2"), 16)
                    if current_opcode == OPADD:
                        if op_1 + op_2 > 2**256:
                            bugs.append([str(bug.get("pc")), "overflow"])
                    if current_opcode == OPSUB:
                        if op_1 < op_2:
                            bugs.append([str(bug.get("pc")), "underflow"])
                    if current_opcode == OPMUL:
                        if op_1 * op_2 > 2**256:
                            bugs.append([str(bug.get("pc")), "overflow"])
                    if current_opcode == OPEXP:
                        if op_1**op_2 > 2**256:
                            bugs.append([str(bug.get("pc")), "overflow"])
                else:
                    if current_opcode == OP_SELFDESTRUCT:
                        bugs.append([str(bug.get("pc")), "self destruct"])
                    elif current_opcode == OP_ORIGIN:
                        bugs.append([str(bug.get("pc")), "tx.origin"])

            all_call = []
            for call in post_state.get("calls", []):
                if call.get("pc") != 0:
                    all_call.append(
                        EVMCall(
                            call.get("pc"),
                            call.get("opcode"),
                            call.get("to"),
                            int(call.get("value"), 16),
                            False,
                        )
                    )
                else:
                    for i in range(len(all_call) - 1, -1, -1):
                        if all_call[i].revert == False:
                            all_call[i].revert = True
                            break
            # check unauthorized send
            for call in all_call:
                if call.value != 0 and call.revert == False:
                    bugs.append([str(call.pc), "unauthorized send"])
            for slot in post_state.get("storage_write", []):
                storage_write.append(
                    EVMStore(slot.get("pc"), slot.get("key"), slot.get("value"))
                )
            final_trace.append(
                {
                    "missed_branches": missed_branches,
                    "covered_branches": covered_branches,
                    "bugs": bugs,
                    "calls": all_call,
                    "storage_write": storage_write,
                }
            )

        return final_trace

    ## initiate num_instances clones of the initial state
    def initiate_instance_data(
        self,
        source_file,
        num_instances,
        config=None,
        contract_name=None,
        detect_bug=False,
    ):
        default_config = json.loads(open("configurations/default.json").read())
        print(default_config)
        # tx_sequence_list
        tx_sequence_config = json.loads(open(config).read())
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
        contract_bin_runtime = self.contract_instance.get("binary_runtime")
        # the merged config fields : "env", "pre" (populated with code), "transaction" (populated with tx data and value)
        pre_env = tx_sequence_config.get("pre", {})

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
        new_test["transaction"]["data"] = ["0x00"]
        new_test["transaction"]["value"] = [0]
        new_test["transaction"]["nonce"] = "0x00"
        # print (f"new_test {new_test}")
        self.instances = [copy.deepcopy(new_test) for _ in range(num_instances)]

    def print_instance_data(self):
        for idx, instance in enumerate(self.instances):
            print(f"\n\n Instance data {idx}\n\n")
            pprint(instance)

    ## build instances data from new tx data
    ## tx_data is a list of tx data
    def build_instance_data(self, tx_data):
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


if __name__ == "__main__":
    my_lib = CuEVMLib(
        "contracts/state_change.sol",
        2,
        "configurations/state_change.json",
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
        "value": [hex(0)],
    }
    tx_2 = {
        "data": get_transaction_data_from_config(
            test_case, my_lib.contract_instance
        ),  # must return an array
        "value": [hex(0)],
    }

    # for debugging, altering tx2 data
    tx_2["data"] = ["0x22"]
    tx_2["value"] = [hex(10)]
    # for debugging, altering the state 2
    my_lib.instances[1]["pre"]["0xcccccccccccccccccccccccccccccccccccccccc"]["storage"][
        "0x00"
    ] = "0x22"
    my_lib.instances[1]["pre"]["0xcccccccccccccccccccccccccccccccccccccccc"][
        "balance"
    ] = "0x00"
    # trace_res = my_lib.run_transactions([tx_1, tx_2])
    trace_res = my_lib.run_transactions([tx_1])
    print("\n\n trace res \n\n")
    pprint(trace_res)
    print("\n\n Updated instance data \n\n")
    my_lib.print_instance_data()

    trace_res = my_lib.run_transactions([tx_1])

    trace_res = my_lib.run_transactions([tx_1])
    # trace_res = my_lib.run_transactions([tx_1, tx_1])
    print("\n\n trace res \n\n")
    pprint(trace_res)
    print("\n\n Updated instance data \n\n")
    my_lib.print_instance_data()

    exit(0)
    my_lib = CuEVMLib(
        "contracts/erc20.sol",
        2,
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
        "value": [hex(512)],
    }
    tx_2 = {
        "data": get_transaction_data_from_config(
            test_case, my_lib.contract_instance
        ),  # must return an array
        "value": [hex(512)],
    }
    trace_res = my_lib.run_transactions([tx_1, tx_2])
    print("\n\n trace res \n\n")
    pprint(trace_res)
