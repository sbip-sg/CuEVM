import library_wrapper
from library_wrapper import CuEVMLib
import json
import time
import argparse
from utils import *
class Fuzzer:

    def __init__(self, contract_source, num_instances=2, timeout=10, \
                 config="configurations/default.json", contract_name=None, output=None, test_case_file = None) -> None:
        self.library = CuEVMLib(contract_source, num_instances, config, contract_name=contract_name, detect_bug=True)
        self.ast_parser = self.library.ast_parser
        self.contract_name = self.library.contract_name
        self.timeout = timeout # in seconds
        self.parse_fuzzing_confg(config)
        if test_case_file:
            self.run_test_case(test_case_file)

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

    def run_fuzzing(self):
        ...

    def finalize_report(self):
        ...


# python fuzzer.py --input sample.sol --config sample_config.json --timeout 10 --contract_name Test --output report.json --test_case test_case.json
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
    args = parser.parse_args()
    fuzzer = Fuzzer(args.input, 2, args.timeout, args.config, args.contract_name , args.output, args.test_case)
    fuzzer.finalize_report()