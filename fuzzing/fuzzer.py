import library_wrapper
from library_wrapper import CuEVMLib
import json
import time
import argparse
import random

import copy
from collections import deque
from utils import *
import os
BRANCH_POPULATION = 3

SMALL_DELTA = 16

MAXIMUM_INT = int(os.environ.get("MAXIMUM_INT", 2**16))
DEBUG = os.environ.get("DEBUG_MODE", "NA")


class SimpleMutator:
    def __init__(self, literal_values, maximum_int = MAXIMUM_INT):
        self.literal_values = literal_values
        self.maximum_int = maximum_int

    def generate_random_input(self, type_):
        if "int" in type_:
            return random.randint(0, self.maximum_int)
        if "string" in type_:
            return "test string"
        if "bool" in type_:
            return random.choice([True, False])
        if "list" in type_:
            return []
    def random_int(self, input):
        return random.randint(0, self.maximum_int)
    def small_delta(self, input):
        diff = random.randint(0, SMALL_DELTA)
        return random.choice([input+diff, max(0,input-diff)])

    def flip_random_bit(self, input):
        # flip a random bit in 256-bit representation
        return input ^ (1 << random.randint(0, 255))
    def flip_random_byte(self, input):
        # flip a random byte in 256-bit representation
        byte = random.randint(0, 31)
        return input ^ (0xff << (byte*8))
    def integer_mutator(self, input):
        # avaliable_mutators = [self.random_int, self.small_delta, self.flip_random_bit, self.flip_random_byte]
        avaliable_mutators = [self.random_int, self.small_delta, self.flip_random_bit]
        return random.choice(avaliable_mutators)(input)

    def mutate(self, value):
        if type(value) == int:
            return self.integer_mutator(value)

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

@dataclass
class DetectedBug:
    pc: int
    bug_type: str
    input: dict
    line_info: list

class Fuzzer:

    def __init__(self, contract_source, num_instances=2, timeout=10, \
                 config="configurations/default.json", contract_name=None, output=None, test_case_file = None, random_seed = 0, branch_heuristic=False) -> None:
        random.seed(random_seed)
        self.library = CuEVMLib(contract_source, num_instances, config, contract_name=contract_name, detect_bug=True)
        self.branch_heuristic = branch_heuristic
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
        self.detected_bugs = {}
        self.raw_inputs = []
        self.branch_source_mapping = {}
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
    def get_branch_source_mapping(self, branch, branch_id):
        if not self.branch_heuristic:
            return
        try:
            frag = self.library.ast_parser.source_by_pc(self.contract_name, branch.pc_src, deploy=False)
            print(f"find source code: {branch_id} { frag.get('fragment')}")
            lines = frag.get("linenums",[0,0])
            print(f"lines: {lines}")
            if lines[1]<= lines[0]+1:
                self.branch_source_mapping[branch_id] = frag.get('fragment')
                return True
            else:
                self.branch_source_mapping[branch_id] = "NA"
        except:
            self.branch_source_mapping[branch_id] = "NA"
        return False
        
    def process_tx_trace(self, tx_trace):
        # print (self.raw_inputs)
        # class EVMBranch:
        #     pc_src: int
        #     pc_dst: int
        #     pc_missed: int
        #     distance: int
        for idx,trace in enumerate(tx_trace):
            for branch in trace.get("branches", []):
                covered_branch = f"{branch.pc_src},{branch.pc_dst}"
                if covered_branch not in self.covered_branches:
                    self.covered_branches.add(covered_branch)
                    if covered_branch not in self.population:
                        self.population[covered_branch] = Seed(self.raw_inputs[idx].get("function"), self.raw_inputs[idx].get("inputs"), 0)
                missed_branch = f"{branch.pc_src},{branch.pc_missed}"
                if missed_branch not in self.branch_source_mapping:
                    self.get_branch_source_mapping(branch, missed_branch)
                if self.branch_heuristic and self.branch_source_mapping[missed_branch] == "NA":
                    continue
                if missed_branch not in self.covered_branches:
                    self.missed_branches.add(missed_branch)
                if missed_branch in self.population:
                    if self.population[missed_branch].distance > branch.distance:
                        self.population[missed_branch] = Seed(self.raw_inputs[idx].get("function"), self.raw_inputs[idx].get("inputs"), branch.distance)
                else:
                    self.population[missed_branch] = Seed(self.raw_inputs[idx].get("function"), self.raw_inputs[idx].get("inputs"), branch.distance)
                
            # for branch in trace.get("missed_branches", []):
            #     # print ("missed branch ", branch)
            #     if branch[0] not in self.covered_branches:
            #         self.missed_branches.add(branch[0])
            #     if branch[0] in self.population:
            #         if self.population[branch[0]].distance > branch[1]:
            #             self.population[branch[0]] = Seed(self.raw_inputs[idx].get("function"), self.raw_inputs[idx].get("inputs"), branch[1])
            #     else:
            #         self.population[branch[0]] = Seed(self.raw_inputs[idx].get("function"), self.raw_inputs[idx].get("inputs"), branch[1])
            # for branch in trace.get("covered_branches", []):
            #     # print ("covered branch ", branch)
            #     self.covered_branches.add(branch[0])
            #     if branch[0] not in self.population:
            #         self.population[branch[0]] = Seed(self.raw_inputs[idx].get("function"), self.raw_inputs[idx].get("inputs"), 0)
            #     elif self.population[branch[0]].distance != 0 :
            #         self.population[branch[0]] = Seed(self.raw_inputs[idx].get("function"), self.raw_inputs[idx].get("inputs"), 0)
            #     if branch[0] in self.missed_branches:
            #         self.missed_branches.remove(branch[0])

            for bug in trace.get("bugs", []):
                bug_id = str(bug.pc) + "_" + str(bug.bug_type)
                if bug_id not in self.detected_bugs:
                    self.detected_bugs[bug_id] = DetectedBug(bug.pc, bug.bug_type, self.raw_inputs[idx],[])


    def print_population(self):
        print("Printing Population")
        for k,v in self.population.items():
            print("\n\n ==========")
            print(f"Branch {k} : ")
            pprint(v)
            if k in self.branch_source_mapping:
                print(f"Source code: {self.branch_source_mapping[k]}")
        print("Missed branches: ", self.missed_branches)
    def mutate(self, inputs, function, generate_random=False):
        if (generate_random):
            return [self.fuzzer.generate_random_input(input_) for input_ in self.abi_list[function].get("input_types")]
        else:
            return [self.fuzzer.mutate(input) for input in inputs]

    def post_process_input(self, tx_data, inputs, function):
        self.raw_inputs.append({
            "function": function,
            "inputs": copy.deepcopy(inputs)
        })

        tx_data.append({
            "data":  get_transaction_data_from_processed_abi(self.abi_list, function, inputs),
            "value": [hex(0)]
        })

    def run_seed_round(self):

        for function in self.function_list:
            print("running seed round on function ", function)
            tx_data = []
            self.raw_inputs = []
            for idx in range(self.num_instances):
                # inputs = []
                # for input_ in self.abi_list[function].get("input_types"):
                #     inputs.append(self.fuzzer.generate_random_input(input_))
                inputs = self.mutate([], function, generate_random=True)
                # print(f"Function {function} : {inputs}")
                self.post_process_input(tx_data, inputs, function)


            tx_trace = self.library.run_transactions(tx_data)
            # print(f"Seed round {function} : {tx_data}")
            # print(f"Trace result : ")
            # pprint(tx_trace)
            self.process_tx_trace(tx_trace)
            print ("Seed round completed")

        self.print_population()

    def select_next_branch(self):
        # debug :
        # return "141,142"
        # return random.choice(list(self.missed_branches))
        return random.choice(list(self.population.keys()))

    def select_next_input(self):
        next_branch = self.select_next_branch()
        seed = self.population[next_branch]
        return seed.inputs, seed.function
        # return random.choice(self.population[next_branch])


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

    def run(self, num_iterations=10):
        for i in range(num_iterations):
            if DEBUG[0] == "v":
                print ("\n" + "-"*80)
                print(f"Iteration {i}\n")
            tx_data = []
            self.raw_inputs = []
            for idx in range(self.num_instances):
                input, function = self.select_next_input()
                new_input = self.mutate(input, function)
                if DEBUG[0] == "v":
                    print(f"Function {function} : {new_input}")
                self.post_process_input(tx_data, new_input, function)

            tx_trace = self.library.run_transactions(tx_data)
            self.process_tx_trace(tx_trace)
            if len(DEBUG) > 1 and DEBUG[1] == "v":
                print(f"Iteration {i} : {tx_data}")
                pprint(tx_trace)

        print ("\n\n Final Population \n\n")
        self.print_population()
        


    def finalize_report(self):
        for k_, bug in self.detected_bugs.items():
            print ("\n")
            print ("-"*80)
            # print(bug)
            print (f" 🚨 Bug Detected: {bug.bug_type} PC: {bug.pc}")
            try:
                frag = self.library.ast_parser.source_by_pc(self.contract_name, int(bug.pc), deploy=False)
                lines = frag.get("linenums",[0,0])
                self.detected_bugs[k_].line_info = lines
                if lines[0] == lines[1]:
                    print (f" Line: {lines[0]} \n Function: {bug.input.get('function')} \n Inputs: {bug.input.get('inputs')} \n Source code:\n \t {frag.get('fragment')}")
                else:
                    print (f" Line: {lines[0]} - {lines[1]} \n Function: {bug.input.get('function')} \n Inputs: {bug.input.get('inputs')} \n Source code:\n \t {frag.get('fragment')}")
                # print (frag)
            except:
                print (f"Failed to get solidity source line")
                pass



# python fuzzer.py --input sample.sol --config sample_config.json --timeout 10 --contract_name Test --output report.json \
#                       --test_case test_case.json --num_instaces 10 --num_iterations 100 --random_seed 0
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
    parser.add_argument(
        "--num_iterations", default=100, help="number of iterations"
    )
    parser.add_argument(
        "--random_seed", default=0, help="random seed"
    )
    parser.add_argument(
        "--branch_heuristic", action="store_true", help="branch heuristic"
    )
    args = parser.parse_args()
    fuzzer = Fuzzer(args.input, int(args.num_instances), args.timeout, args.config,  contract_name= args.contract_name
                   , output=args.output, test_case_file=args.test_case, random_seed= int(args.random_seed), branch_heuristic=args.branch_heuristic)
    fuzzer.run(num_iterations=int(args.num_iterations))
    fuzzer.finalize_report()