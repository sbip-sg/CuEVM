import library_wrapper
from library_wrapper import CuEVMLib
import json
import time
class Fuzzer:

    def __init__(self, contract_source, num_instances=2, timeout=10, config="configurations/default.json") -> None:
        self.library = CuEVMLib(contract_source, num_instances, config, True)
        self.ast_parser = self.library.ast_parser
        self.contract_name = self.library.contract_name
        self.timeout = timeout # in seconds
        self.parse_fuzzing_confg(config)
    def parse_fuzzing_confg(self, config):
        ...

    def run_fuzzing(self):
        ...

    def finalize_report(self):
        ...



if __name__ == "__main__":
    ...