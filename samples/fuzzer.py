import library_wrapper
from library_wrapper import CuEVMLib
import json
import time
import argparse
import random

import copy
from collections import deque
from utils import *

BRANCH_POPULATION = 3

class SimpleMutator:
    def __init__(self, literal_values):
        self.literal_values = literal_values

    def generate_random_input(self, type_):
        if "int" in type_:
            return random.randint(0, 1000)
        if "string" in type_:
            return "test string"
        if "bool" in type_:
            return random.choice([True, False])
        if "list" in type_:
            return []
    def mutate(self, value):
        if type(value) == int:
            return random.randint(0, 1000)
        if type(value) == str:
            return "test string"
        if type(value) == bool:
            return random.choice([True, False])
        if type(value) == list:
            return [self.mutate(val) for val in value]
        else:
            return value
@dataclass
class Seed:
    function: str
    inputs: list
    distance: int

class Fuzzer:

    def __init__(self, contract_source, num_instances=2, timeout=10, \
                 config="configurations/default.json", contract_name=None, output=None, test_case_file = None) -> None:
        random.seed(0)
        self.library = CuEVMLib(contract_source, num_instances, config, contract_name=contract_name, detect_bug=True)
        self.num_instances = num_instances
        self.ast_parser = self.library.ast_parser
        self.contract_name = self.library.contract_name
        self.timeout = timeout # in seconds
        self.parse_fuzzing_confg(config)
        self.abi_list = {} # mapping from function to input types for abi encoding
        if test_case_file:
            self.run_test_case(test_case_file)

        self.function_list = self.ast_parser.functions_in_contract_by_name(self.contract_name, name_only=True)
        self.literal_values = self.ast_parser.get_literals(self.contract_name, only_value=True)
        self.fuzzer = SimpleMutator(self.literal_values)
        # self.abi_list = self.ast_parser.original_compilation_output
        for k_, v_ in self.ast_parser.original_compilation_output.items():
            if k_.split(":")[1] == self.contract_name:
                self.prepare_abi(v_["abi"])
                break

        self.covered_branches = set()
        self.missed_branches = set()
        self.population = {} # store the population for each branch
        self.raw_inputs = []
        self.run_seed_round()


    def prepare_abi(self, abi):
        print("prepare_abi")
        for item in abi:
            if item.get("type") == "function":
                print(item.get("name"))
                print(item.get("inputs"))
                print(item.get("outputs"))
                print(item.get("stateMutability"))
                if item.get("stateMutability") != "view":
                    input_list = []
                    for input_ in item.get("inputs"):
                        input_list.append(input_.get("type"))
                    self.abi_list[item.get("name")] = {
                        "input_types" : input_list,
                        "4byte" : function_abi_to_4byte_selector(item).hex()
                        }

        print ("after processing")
        print (self.abi_list)

    def process_tx_trace(self, tx_trace):
        print (self.raw_inputs)

        for idx,trace in enumerate(tx_trace):
            for branch in trace.get("missed_branches", []):
                self.missed_branches.add(branch[0])
                if branch[0] in self.population:
                    if self.population[branch[0]].distance > branch[1]:
                        self.population[branch[0]] = Seed(self.raw_inputs[idx].get("function"), self.raw_inputs[idx].get("inputs"), branch[1])
                else:
                    self.population[branch[0]] = Seed(self.raw_inputs[idx].get("function"), self.raw_inputs[idx].get("inputs"), branch[1])
            for branch in trace.get("covered_branches", []):
                self.covered_branches.add(branch[0])
                if branch[0] not in self.population:
                    self.population[branch[0]] = Seed(self.raw_inputs[idx].get("function"), self.raw_inputs[idx].get("inputs"), 0)

    def print_population(self):
        print("Printing Population")
        for k,v in self.population.items():
            print("\n\n ==========")
            print(f"Branch {k} : ")
            pprint(v)

    def run_seed_round(self):

        for function in self.function_list:
            print("running seed round on function ", function)
            tx_data = []
            self.raw_inputs = []
            for idx in range(self.num_instances):
                inputs = []
                for input_ in self.abi_list[function].get("input_types"):
                    inputs.append(self.fuzzer.generate_random_input(input_))
                print(f"Function {function} : {inputs}")
                tx_data.append({
                    "data":  get_transaction_data_from_processed_abi(self.abi_list, function, inputs),
                    "value": [hex(0)]
                })
                self.raw_inputs.append({
                    "function": function,
                    "inputs": copy.deepcopy(inputs)
                })

                tx_trace = self.library.run_transactions(tx_data)
                # print(f"Trace result : ")
                # pprint(tx_trace)
                self.process_tx_trace(tx_trace)
            print ("Seed round completed")
            self.print_population()

    def select_next_branch(self):
        return random.choice(list(self.missed_branches))

    def select_next_input(self):
        next_branch = self.select_next_branch()
        if len(self.population.get(next_branch, [])) == 0:
            self.population[next_branch] = self.generate_population(next_branch)

        return random.choice(self.population[next_branch])

    def generate_population(self, branch):
        for i in range(BRANCH_POPULATION):
            self.fuzzer.generate_random_input()

        ...

    def mutate(self, test_case):
        ...

    def run_test_case(self, test_case_file):
        with open(test_case_file) as f:
            data = json.load(f)
        test_cases = data.get("test_cases")
        for (idx, test_case) in enumerate(test_cases):

            tx_data = self.prepare_tx(test_case)

            print(f"Test case {idx} : {test_case}")
            print(f"Transaction data : {tx_data}")

            # self.library.build_instance_data(tx_data)
            trace_res = self.library.run_transactions(tx_data)


            print(f"Trace result : ")
            pprint(trace_res)

    def prepare_tx(self, test_case):
        tx = []
        temp_val = test_case.get("value", 0)
        if type(temp_val) == int:
            temp_val = hex(temp_val)
        tx.append({
            "data":  get_transaction_data_from_config(test_case, self.library.contract_instance),
            "value": [temp_val]
        })
        # print ("testcase" , test_case)
        return tx
    def parse_fuzzing_confg(self, config):
        ...

    def run_fuzzing(self, num_iterations=10):
        for i in range(num_iterations):
            self.select_next_input()
            self.mutate()
            self.run()


    def finalize_report(self):
        ...


# python fuzzer.py --input sample.sol --config sample_config.json --timeout 10 --contract_name Test --output report.json \
#                       --test_case test_case.json --num_instaces 10 --num_iterations 100
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EVM fuzzer")
    parser.add_argument(
        "--input", default="contracts/overflow.sol", help="source file"
    )
    parser.add_argument(
        "--config", default="configurations/default.json", help="config file"
    )
    parser.add_argument(
        "--timeout", default=10, help="timeout in seconds"
    )
    parser.add_argument(
        "--contract_name", help="contract name"
    )
    parser.add_argument(
        "--output", help="output file"
    )
    parser.add_argument(
        "--test_case", help="test case file"
    )
    parser.add_argument(
        "--num_instances", default=2, help="number of instances"
    )
    args = parser.parse_args()
    fuzzer = Fuzzer(args.input, int(args.num_instances), args.timeout, args.config,
                    contract_name= args.contract_name , output=args.output, test_case_file=args.test_case)
    fuzzer.finalize_report()