'''
library wrapper to maintain state of EVM instances and run tx on them
'''
import sys
import ctypes
import json
import copy
from pprint import pprint
from utils import *

# Add the directory containing your .so file to the Python path
sys.path.append('../build/')

import libcuevm  # Now you can import your module as usual

class CuEVMLib:
    def __init__(self, source_file, num_instances, config = None,
                 detect_bug=False, sender = "0x1111111111111111111111111111111111111111"):
        self.initiate_instance_data(source_file, num_instances, config, detect_bug)
        self.sender = sender


    def update_persistent_state(self, json_result):
        trace_values = json_result
        print ("trace value result")
        pprint(json_result)
        for i in range(len(trace_values.get("post"))):
            post_state = trace_values.get("post")[i].get("state")
            # print("\n\n post_state %d \n\n" % i)
            # pprint(post_state)
            self.instances[i]["pre"] = copy.deepcopy(post_state)
            # sender = next_config["transaction"]["sender"]
            self.instances[i]["transaction"]["nonce"] = post_state.get(self.sender).get(
                "nonce"
            )

    ## 1. run transactions on the EVM instances
    ## 2. update the persistent state of the EVM instances
    ## 3. return the simplified trace during execution
    def run_transactions(self, tx_data):
        self.build_instance_data(tx_data)
        result_state = libcuevm.run_dict(self.instances)
        self.update_persistent_state(result_state)

    ## initiate num_instances clones of the initial state
    def initiate_instance_data(self, source_file, num_instances, config = None, detect_bug=False):
        default_config = json.loads(open("configurations/default.json").read())
        print(default_config)
        # tx_sequence_list
        tx_sequence_config = json.loads(open(config).read())
        self.contract_name = tx_sequence_config.get("contract_name")
        self.contract_instance, self.ast_parser = compile_file(source_file, self.contract_name)
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

        self.instances = [copy.deepcopy(new_test) for _ in range(num_instances)]

    def print_instance_data(self):
        for idx,instance in enumerate(self.instances):
            print(f"\n\n Instance data {idx}\n\n")
            pprint(instance)



    ## build instances data from new tx data
    ## tx_data is a list of tx data
    def build_instance_data(self, tx_data):
        assert len(tx_data) == len(self.instances)
        for i in range(len(tx_data)):
            self.instances[i]["transaction"]["data"] = tx_data[i]["data"]
            self.instances[i]["transaction"]["value"] = tx_data[i]["value"]
            if (tx_data[i].get("sender")):
                self.instances[i]["transaction"]["sender"] = tx_data[i]["sender"]
            # TODO: add other fuzz-able fields


if __name__ == "__main__":
    my_lib = CuEVMLib("contracts/state_change.sol", 2, "configurations/state_change.json", False)
    test_case = {
                    "function": "increase",
                    "type": "exec",
                    "input_types": [],
                    "input": [],
                    "sender": 0
                }


    tx_1 = {
        "data" : get_transaction_data_from_config(test_case, my_lib.contract_instance),  # must return an array
        "value" : [hex(0)]
    }
    tx_2 = {
        "data" : get_transaction_data_from_config(test_case, my_lib.contract_instance),  # must return an array
        "value" : [hex(0)]
    }

    # for debugging, altering tx2 data
    tx_2["data"] = ["0x22"]
    tx_2["value"] = [hex(10)]
    # for debugging, altering the state 2
    my_lib.instances[1]["pre"]["0xcccccccccccccccccccccccccccccccccccccccc"]["storage"]["0x00"] = "0x22"
    my_lib.instances[1]["pre"]["0xcccccccccccccccccccccccccccccccccccccccc"]["balance"] = "0x00"
    my_lib.run_transactions([tx_1 , tx_2])
    print ("\n\n Updated instance data \n\n")
    my_lib.print_instance_data()

    my_lib.run_transactions([tx_1 , tx_1])
    print ("\n\n Updated instance data \n\n")
    my_lib.print_instance_data()
