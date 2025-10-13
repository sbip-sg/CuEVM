import json
import time
import argparse
import random
import copy
from collections import deque
from utils import *
import os
from pprint import pprint
from gpu_library_wrapper import CuEVMGPULib, GPUExecutionResult
from pprint import pprint


SMALL_DELTA = 16

MAXIMUM_INT = int(os.environ.get("MAXIMUM_INT", 2**255))
DEBUG = os.environ.get("DEBUG_MODE", "NA")


def pretty_print_state(state_data, title="Blockchain State"):
    """Pretty print blockchain state for debugging"""
    print(f"\n{title}")
    print("=" * len(title))

    if isinstance(state_data, str):
        try:
            state_data = json.loads(state_data)
        except:
            print(state_data)
            return

    if "pre" in state_data:
        print(f"\nAccounts ({len(state_data['pre'])}):")
        for i, (addr, account) in enumerate(state_data["pre"].items()):
            print(f"  {addr}")
            print(f"    Balance: {account.get('balance', '0x0')}")
            print(f"    Nonce: {account.get('nonce', '0x0')}")
            code = account.get("code", "0x")
            code_len = len(code) // 2 - 1 if len(code) > 2 else 0
            print(f"    Code: {code[:50]}{'...' if len(code) > 50 else ''} ({code_len} bytes)")
            print()

    if "env" in state_data:
        print("Environment:")
        env = state_data["env"]
        for key, value in env.items():
            print(f"  {key}: {value}")


def pretty_print_constants(constants_data, title="Fuzzing Constants"):
    """Pretty print constants for debugging"""
    print(f"\n{title}")
    print("=" * len(title))

    if isinstance(constants_data, str):
        try:
            constants_data = json.loads(constants_data)
        except:
            print(constants_data)
            return

    for const_type, values in constants_data.items():
        print(f"\n{const_type.upper()} ({len(values)}):")
        for i, value in enumerate(values[:3]):  # Show only first 3
            print(f"  [{i}] {value}")
        if len(values) > 3:
            print(f"  ... and {len(values) - 3} more")


def pretty_print_transactions(tx_data, title="Transaction Data"):
    """Pretty print transaction data for debugging"""
    print(f"\n{title}")
    print("=" * len(title))

    if not tx_data:
        print("No transaction data")
        return

    num_tx = len(tx_data.get("block_numbers", []))
    print(f"Transactions: {num_tx}")

    call_data = tx_data.get("call_data", b"")
    print(f"Call data: {len(call_data)} bytes")

    if DEBUG.startswith("v"):
        data_offsets = tx_data.get("data_offsets", [])
        data_sizes = tx_data.get("data_sizes", [])
        print(f"Data offsets: {data_offsets[:5]}{'...' if len(data_offsets) > 5 else ''}")
        print(f"Data sizes: {data_sizes[:5]}{'...' if len(data_sizes) > 5 else ''}")


def pretty_print_gpu_results(result, title="GPU Execution Results"):
    """Pretty print GPU execution results"""
    print(f"\n{title}")
    print("=" * len(title))

    if not result:
        print("No results returned")
        return

    total_coverage = sum(len(batch) for batch in result.new_coverage_thread_idx)
    total_bugs = sum(len(batch) for batch in result.new_bug_thread_idx)
    total_storage = sum(len(batch) for batch in result.new_storage_thread_idx)

    print(f"Coverage: {total_coverage}, Bugs: {total_bugs}, Storage: {total_storage}")

    if DEBUG.startswith("v"):
        for batch_idx in range(len(result.new_coverage_thread_idx)):
            coverage_count = len(result.new_coverage_thread_idx[batch_idx])
            bug_count = len(result.new_bug_thread_idx[batch_idx])

            if coverage_count > 0 or bug_count > 0:
                print(f"Batch {batch_idx}: {coverage_count} coverage, {bug_count} bugs")


class SimpleMutator:
    def __init__(self, literal_values, maximum_int=MAXIMUM_INT):
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
        return random.choice([input + diff, max(0, input - diff)])

    def flip_random_bit(self, input):
        # flip a random bit in 256-bit representation
        return input ^ (1 << random.randint(0, 255))

    def flip_random_byte(self, input):
        # flip a random byte in 256-bit representation
        byte = random.randint(0, 31)
        return input ^ (0xFF << (byte * 8))

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

    def __init__(
        self,
        contract_source,
        num_instances=32,
        skip_tx_size=32,
        timeout=10,
        config="configurations/default.json",
        contract_name=None,
        output=None,
        test_case_file=None,
        random_seed=0,
        branch_heuristic=False,
        sequence_length=1,
    ) -> None:
        random.seed(random_seed)

        print("Initializing GPU Fuzzer")
        print(f"Contract: {contract_source}, Instances: {num_instances}, Sequence: {sequence_length}")

        # Initialize GPU library
        self.gpu_lib = CuEVMGPULib()
        self.gpu_initialized = False

        # Original parameters
        self.contract_source = contract_source
        self.skip_tx_size = skip_tx_size
        if num_instances > self.skip_tx_size:
            num_instances = num_instances // self.skip_tx_size * self.skip_tx_size
        else:
            num_instances = self.skip_tx_size
        self.num_instances = num_instances
        self.config_path = config
        self.contract_name = contract_name
        self.branch_heuristic = branch_heuristic
        self.sequence_length = sequence_length
        self.timeout = timeout  # in seconds
        self.parse_fuzzing_confg(config)
        self.abi_list = {}  # mapping from function to input types for abi encoding

        # Load contract info using utils functions
        try:
            self.contract_instance, self.ast_parser = compile_file(contract_source, contract_name)
            if self.contract_instance is None:
                raise Exception(f"Failed to compile contract {contract_name} from {contract_source}")

            self.function_list = self.ast_parser.functions_in_contract_by_name(self.contract_name, name_only=True)
            self.literal_values = self.ast_parser.get_literals(self.contract_name, only_value=True)
            self.fuzzer = SimpleMutator(self.literal_values)

            # Parse ABI
            for k_, v_ in self.ast_parser.original_compilation_output.items():
                if k_.split(":")[1] == self.contract_name:
                    self.prepare_abi(v_["abi"])
                    break

            print(f"Contract compiled: {self.contract_name}")
            print(f"Functions: {self.function_list}")
            print(f"ABI methods: {list(self.abi_list.keys())}")

        except Exception as e:
            print(f"Error loading contract: {e}")
            raise

        # Initialize fuzzer state
        self.covered_branches = set()
        self.missed_branches = set()
        self.population = {}  # store the population for each branch
        self.detected_bugs = {}
        self.raw_inputs = []
        self.branch_source_mapping = {}

        # Initialize GPU state
        self.initialize_gpu_state()

        if test_case_file:
            self.run_test_case(test_case_file)

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
                        "input_types": input_list,
                        "4byte": function_abi_to_4byte_selector(item),
                    }

        print("after processing")
        print(self.abi_list)

    def initialize_gpu_state(self):
        """Initialize GPU state with contract and environment data"""
        print("\nInitializing GPU state...")

        try:
            state_data = self.create_blockchain_state()
            state_json = json.dumps(state_data)
            constants_data = self.create_fuzzing_constants()
            constants_json = json.dumps(constants_data)

            if DEBUG.startswith("v"):
                pretty_print_state(state_data, "Blockchain State")
                pretty_print_constants(constants_data, "Constants")

            success = self.gpu_lib.initialize_gpu_state(
                state_json=state_json,
                num_instances=self.num_instances,
                constants_json=constants_json,
                skip_tx_size=self.skip_tx_size,
            )

            if success:
                gpu_instances = self.gpu_lib.get_num_instances_per_device()
                print(f"GPU initialized: {gpu_instances} instances per device")
                self.gpu_initialized = True
            else:
                print("GPU initialization failed")
                self.gpu_initialized = False

        except Exception as e:
            print(f"Error initializing GPU: {e}")
            self.gpu_initialized = False

    def create_blockchain_state(self):
        """Create initial blockchain state from config and contract"""
        try:
            # Load default config
            with open(self.config_path, "r") as f:
                config_data = json.load(f)

            # Extract contract bytecode - try multiple fields
            contract_bytecode = None

            # Try different bytecode fields
            for field in ["binary_runtime", "bin-runtime", "binary", "bin"]:
                if field in self.contract_instance:
                    contract_bytecode = self.contract_instance[field]
                    if DEBUG.startswith("v"):
                        print(f"Found bytecode in '{field}': {len(str(contract_bytecode))} chars")
                    break

            if not contract_bytecode or contract_bytecode in ["0x", ""]:
                print("No valid contract bytecode found!")
                contract_bytecode = "0x"

            # Ensure proper hex format
            if isinstance(contract_bytecode, str):
                if not contract_bytecode.startswith("0x"):
                    contract_bytecode = "0x" + contract_bytecode
            else:
                contract_bytecode = "0x" + contract_bytecode.hex() if isinstance(contract_bytecode, bytes) else "0x"

            # Create state
            state = {
                "pre": {
                    "0x1234567890123456789012345678901234567890": {  # Contract address
                        "balance": "0x1000000000000000000",
                        "nonce": "0x0",
                        "code": contract_bytecode,
                        "storage": config_data.get("storage", {}),
                    },
                    "0x1111111111111111111111111111111111111111": {  # Sender address
                        "balance": "0x1000000000000000000",
                        "nonce": "0x0",
                        "code": "0x",
                        "storage": {},
                    },
                },
                "env": {
                    "currentCoinbase": "0x0000000000000000000000000000000000000000",
                    "currentTimestamp": "0x1",
                    "currentNumber": "0x1",
                    "currentDifficulty": "0x1",
                    "currentGasLimit": "0x1000000",
                    "currentBaseFee": "0xa",
                    "currentRandom": "0x0000000000000000000000000000000000000000000000000000000000000000",
                    "chainId": "0x1",
                },
            }

            return state

        except Exception as e:
            print(f"Error creating blockchain state: {e}")
            # Return minimal state
            return {
                "pre": {
                    "0x1234567890123456789012345678901234567890": {
                        "balance": "0x1000000000000000000",
                        "nonce": "0x0",
                        "code": "0x6000600055",  # Simple SSTORE
                        "storage": {},
                    }
                },
                "env": {
                    "currentCoinbase": "0x0000000000000000000000000000000000000000",
                    "currentTimestamp": "0x1",
                    "currentNumber": "0x1",
                    "currentDifficulty": "0x1",
                    "currentGasLimit": "0x1000000",
                    "currentBaseFee": "0xa",
                    "currentRandom": "0x0000000000000000000000000000000000000000000000000000000000000000",
                    "chainId": "0x1",
                },
            }

    def create_fuzzing_constants(self):
        """Create constants for GPU fuzzing"""
        constants = {
            "address": [
                "0x1234567890123456789012345678901234567890",  # Contract
                "0x1111111111111111111111111111111111111111",  # Sender
            ],
            "integer": [
                "0x0000000000000000000000000000000000000000000000000000000000000000",  # 0
                "0x0000000000000000000000000000000000000000000000000000000000000001",  # 1
            ],
            "sender": [
                "0x1111111111111111111111111111111111111111",
            ],
        }

        # Add literal values from contract
        if self.literal_values and hasattr(self.literal_values, "__iter__"):
            try:
                # Convert to list if not already and limit to first 10
                literal_list = (
                    list(self.literal_values)[:10]
                    if not isinstance(self.literal_values, list)
                    else self.literal_values[:10]
                )
                for val in literal_list:
                    if isinstance(val, (int, float)):
                        val = int(val)
                        hex_val = (
                            f"0x{val:064x}"
                            if val >= 0
                            else "0x"
                            + format(
                                val & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,
                                "064x",
                            )
                        )
                        if hex_val not in constants["integer"]:
                            constants["integer"].append(hex_val)
            except Exception as e:
                print(f"Warning: Could not process literal values: {e}")
                # Continue with default constants

        return constants

    def create_gpu_transaction_batch(self, function_name, raw_inputs):
        """Create GPU transaction batch from function calls and inputs"""
        num_tx = len(raw_inputs)

        # Prepare call data for each transaction
        call_data_list = []
        data_offsets = []
        data_sizes = []
        current_offset = 0

        for input_data in raw_inputs:
            # Generate transaction data using ABI encoding
            if function_name in self.abi_list:
                tx_data = get_transaction_data_from_processed_abi(self.abi_list, function_name, input_data["inputs"])
                # Handle both string and list returns
                if isinstance(tx_data, list) and len(tx_data) > 0:
                    tx_data = tx_data[0]  # Take first element if it's a list

                if isinstance(tx_data, str):
                    call_data_bytes = bytes.fromhex(tx_data[2:]) if tx_data.startswith("0x") else bytes.fromhex(tx_data)
                else:
                    call_data_bytes = tx_data if isinstance(tx_data, bytes) else b""
            else:
                # Fallback: use empty call data
                call_data_bytes = b""

            call_data_list.append(call_data_bytes)
            data_offsets.append(current_offset)
            data_sizes.append(len(call_data_bytes) if call_data_bytes else 1)  # Ensure minimum size of 1
            current_offset += max(1, len(call_data_bytes))

        # Concatenate all call data
        if call_data_list and any(call_data_list):
            combined_call_data = b"".join(call_data_list)
        else:
            # Fallback: use null bytes
            combined_call_data = b"\x00" * num_tx
            data_offsets = list(range(num_tx))
            data_sizes = [1] * num_tx

        # Create transaction batch data
        tx_batch = {
            "block_numbers": [1] * num_tx,
            "timestamps": [int(time.time())] * num_tx,
            "from_addresses": [0] * num_tx,  # Use address index 0 (sender)
            "to_address": bytes.fromhex("1234567890123456789012345678901234567890"),  # Contract address
            "call_data": combined_call_data,
            "data_offsets": data_offsets,
            "data_sizes": data_sizes,
            "marker_offsets": [0] * num_tx,  # Simple marker offsets
            "marker_data": [0] * num_tx,  # Datamarker's length = 0, no on_device mutation
        }

        return tx_batch

    def create_gpu_sequence_batch(self, sequences):
        """Create GPU transaction batch from transaction sequences (similar to fuzzer.go)"""
        # Flatten all sequences into individual transactions
        all_transactions = []
        for tx_idx, _ in enumerate(sequences[0]):
            for seq_idx, sequence in enumerate(sequences):
                tx = sequence[tx_idx]
                all_transactions.append(tx)

        total_tx = len(all_transactions)

        # Prepare call data for each transaction
        call_data_list = []
        data_offsets = []
        data_sizes = []
        current_offset = 0

        for tx_idx, tx in enumerate(all_transactions):
            if tx_idx % len(sequences) == 0:
                current_offset = 0
            function_name = tx["function"]
            inputs = tx["inputs"]

            # Generate transaction data using ABI encoding
            if function_name in self.abi_list:
                tx_data = get_transaction_data_from_processed_abi(self.abi_list, function_name, inputs)
                # Handle both string and list returns
                if isinstance(tx_data, list) and len(tx_data) > 0:
                    tx_data = tx_data[0]  # Take first element if it's a list

                if isinstance(tx_data, str):
                    call_data_bytes = bytes.fromhex(tx_data[2:]) if tx_data.startswith("0x") else bytes.fromhex(tx_data)
                else:
                    call_data_bytes = tx_data if isinstance(tx_data, bytes) else b""
            else:
                # Fallback: use empty call data
                call_data_bytes = b""

            call_data_list.append(call_data_bytes)
            data_offsets.append(current_offset)
            data_sizes.append(len(call_data_bytes) if call_data_bytes else 0)
            current_offset += len(call_data_bytes)

        # Concatenate all call data
        if call_data_list and any(call_data_list):
            combined_call_data = b"".join(call_data_list)
        else:
            # Fallback: use null bytes
            combined_call_data = b"\x00" * total_tx
            data_offsets = list(range(total_tx))
            data_sizes = [1] * total_tx

        # Create transaction batch data (flattened for GPU processing)
        tx_batch = {
            "block_numbers": [1] * total_tx,
            "timestamps": [int(time.time())] * total_tx,
            "from_addresses": [0] * total_tx,  # Use address index 0 (sender)
            "to_address": bytes.fromhex("1234567890123456789012345678901234567890"),  # Contract address
            "call_data": combined_call_data,
            "data_offsets": data_offsets,
            "data_sizes": data_sizes,
            "marker_offsets": [0] * total_tx,  # Simple marker offsets
            "marker_data": [0] * total_tx,  # Datamarker's length = 0, no on_device mutation
        }

        return tx_batch

    def process_gpu_results(self, result):
        """Process GPU execution results and update fuzzer state"""
        if not result or not hasattr(result, "new_coverage_thread_idx"):
            return

        # Store raw_inputs for this batch (flatten all sequences)
        self.raw_inputs = []

        # Process coverage information
        new_coverage = 0
        for batch_idx in range(len(result.new_coverage_thread_idx)):
            coverage_list = result.new_coverage_thread_idx[batch_idx]
            coverage_ids = result.new_coverage_ids[batch_idx] if batch_idx < len(result.new_coverage_ids) else []

            for i, thread_idx in enumerate(coverage_list):
                if i < len(coverage_ids):
                    coverage_id = coverage_ids[i]
                    branch_key = f"gpu_{coverage_id}"

                    if branch_key not in self.covered_branches:
                        self.covered_branches.add(branch_key)
                        new_coverage += 1

        # Process bug information - decode bug IDs like in fuzzer.go
        new_bugs = 0
        for batch_idx in range(len(result.new_bug_thread_idx)):
            bug_list = result.new_bug_thread_idx[batch_idx]
            bug_pcs = result.new_bug_pcs[batch_idx] if batch_idx < len(result.new_bug_pcs) else []
            bug_types = result.new_bug_types[batch_idx] if batch_idx < len(result.new_bug_types) else []
            bug_contract_ids = (
                result.new_bug_contract_ids[batch_idx] if batch_idx < len(result.new_bug_contract_ids) else []
            )

            print(f"Processing batch {batch_idx}: {len(bug_list)} bugs")
            for i, thread_idx in enumerate(bug_list):
                if i < len(bug_pcs) and i < len(bug_types):
                    pc = bug_pcs[i]
                    bug_type = bug_types[i]
                    contract_id = bug_contract_ids[i] if i < len(bug_contract_ids) else 0

                    bug_id = f"{pc}_{bug_type}_{contract_id}"
                    print(f"Bug found: PC={pc}, Type={bug_type}, Contract={contract_id}, Thread={thread_idx}")

                    if bug_id not in self.detected_bugs:
                        # Create a dummy input for the bug since we don't have raw_inputs populated
                        dummy_input = {"function": "unknown", "inputs": []}
                        self.detected_bugs[bug_id] = DetectedBug(pc, bug_type, dummy_input, [])
                        new_bugs += 1

                        # Try to get source location immediately
                        try:
                            frag = self.ast_parser.source_by_pc(self.contract_name, int(pc), deploy=False)
                            lines = frag.get("linenums", [0, 0])
                            source_code = frag.get("fragment", "Source not available")
                            print(f"Bug location: PC={pc}, Lines={lines}, Source: {source_code.strip()}")
                        except Exception as e:
                            print(f"Could not get source location for PC {pc}: {e}")

        if new_bugs > 0:
            print(f"Found {new_bugs} new bugs!")
        if new_coverage > 0:
            print(f"Found {new_coverage} new coverage!")

    def get_branch_source_mapping(self, branch, branch_id):
        if not self.branch_heuristic:
            return
        try:
            frag = self.ast_parser.source_by_pc(self.contract_name, branch.pc_src, deploy=False)
            print(f"find source code: {branch_id} { frag.get('fragment')}")
            lines = frag.get("linenums", [0, 0])
            print(f"lines: {lines}")
            if lines[1] <= lines[0] + 1:
                self.branch_source_mapping[branch_id] = frag.get("fragment")
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
        for idx, trace in enumerate(tx_trace):
            for branch in trace.get("branches", []):
                covered_branch = f"{branch.pc_src},{branch.pc_dst}"
                if covered_branch not in self.covered_branches:
                    self.covered_branches.add(covered_branch)
                    if covered_branch not in self.population:
                        self.population[covered_branch] = Seed(
                            self.raw_inputs[idx].get("function"),
                            self.raw_inputs[idx].get("inputs"),
                            0,
                        )
                missed_branch = f"{branch.pc_src},{branch.pc_missed}"
                if missed_branch not in self.branch_source_mapping:
                    self.get_branch_source_mapping(branch, missed_branch)
                if self.branch_heuristic and self.branch_source_mapping[missed_branch] == "NA":
                    continue
                if missed_branch not in self.covered_branches:
                    self.missed_branches.add(missed_branch)
                if missed_branch in self.population:
                    if self.population[missed_branch].distance > branch.distance:
                        self.population[missed_branch] = Seed(
                            self.raw_inputs[idx].get("function"),
                            self.raw_inputs[idx].get("inputs"),
                            branch.distance,
                        )
                else:
                    self.population[missed_branch] = Seed(
                        self.raw_inputs[idx].get("function"),
                        self.raw_inputs[idx].get("inputs"),
                        branch.distance,
                    )

            for bug in trace.get("bugs", []):
                bug_id = str(bug.pc) + "_" + str(bug.bug_type)
                if bug_id not in self.detected_bugs:
                    self.detected_bugs[bug_id] = DetectedBug(bug.pc, bug.bug_type, self.raw_inputs[idx], [])

    def print_population(self):
        print("Printing Population")
        for k, v in self.population.items():
            print("\n\n ==========")
            print(f"Branch {k} : ")
            pprint(v)
            if k in self.branch_source_mapping:
                print(f"Source code: {self.branch_source_mapping[k]}")
        print("Missed branches: ", self.missed_branches)

    def mutate(self, inputs, function, generate_random=False):
        if generate_random:
            return [self.fuzzer.generate_random_input(input_) for input_ in self.abi_list[function].get("input_types")]
        else:
            return [self.fuzzer.mutate(input) for input in inputs]

    def sequence_mutation(self, sequence_length):
        """Generate a transaction sequence of length N with random functions and inputs"""
        sequence = []
        for i in range(sequence_length):
            function = random.choice(self.function_list)
            # Debug: use specific functions for testing
            if sequence_length > 1:
                function = "set_state" if i == 0 else "multiply"
            inputs = self.mutate([], function, generate_random=True)
            sequence.append({"function": function, "inputs": inputs})
        return sequence

    def post_process_input(self, tx_data, inputs, function):
        self.raw_inputs.append({"function": function, "inputs": copy.deepcopy(inputs)})

        tx_data.append(
            {
                "data": get_transaction_data_from_processed_abi(self.abi_list, function, inputs),
                "value": [0],
            }
        )

    def select_next_branch(self):
        # debug :
        # return "141,142"
        # return random.choice(list(self.missed_branches))
        if self.population:
            return random.choice(list(self.population.keys()))
        else:
            # Fallback: return a dummy branch key
            return "fallback_branch"

    def select_next_input(self):
        if not self.population:
            # Fallback: generate random inputs for a random function
            function = random.choice(self.function_list)
            inputs = self.mutate([], function, generate_random=True)
            return inputs, function

        next_branch = self.select_next_branch()
        seed = self.population[next_branch]
        return seed.inputs, seed.function

    def run_test_case(self, test_case_file):
        with open(test_case_file) as f:
            data = json.load(f)
        test_cases = data.get("test_cases")
        for idx, test_case in enumerate(test_cases):

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
        tx.append(
            {
                "data": get_transaction_data_from_config(test_case, self.library.contract_instance),
                "value": [temp_val],
            }
        )
        # print ("testcase" , test_case)
        return tx

    def parse_fuzzing_confg(self, config): ...

    def run(self, num_iterations=10):
        print(f"\nStarting fuzzing ({num_iterations} iterations)...")

        if not self.gpu_initialized:
            print("GPU not initialized")
            return

        for i in range(num_iterations):
            print(f"\nIteration {i+1}/{num_iterations}")
            print("-" * 40)

            # Reset GPU state for iterations > 0
            if i > 0:
                success = self.gpu_lib.initialize_gpu_state(
                    state_json=None,
                    num_instances=self.num_instances,
                    constants_json=None,
                    skip_tx_size=self.skip_tx_size,
                    reset_state=True,
                )
                if not success:
                    print("Failed to reset GPU state")
                    continue

            # Generate sequences
            num_unique_sequences = max(1, self.num_instances // self.skip_tx_size)
            all_sequences = []

            for seq_idx in range(num_unique_sequences):
                base_sequence = self.sequence_mutation(self.sequence_length)
                for replica in range(self.skip_tx_size):
                    replicated_sequence = []
                    for tx in base_sequence:
                        new_inputs = self.mutate([], tx["function"], generate_random=True)
                        replicated_sequence.append({"function": tx["function"], "inputs": new_inputs})
                    all_sequences.append(replicated_sequence)

            print(f"Generated {len(all_sequences)} sequences (length {self.sequence_length})")

            # Execute on GPU
            tx_data = self.create_gpu_sequence_batch(all_sequences)

            if DEBUG.startswith("v"):
                pretty_print_transactions(tx_data, f"Iteration {i+1}")

            try:
                total_tx = len(tx_data["block_numbers"])
                print(f"Executing {total_tx} transactions on GPU...")

                result = self.gpu_lib.process_batch_transactions(
                    block_numbers=tx_data["block_numbers"],
                    timestamps=tx_data["timestamps"],
                    from_addresses=tx_data["from_addresses"],
                    to_address=tx_data["to_address"],
                    call_data=tx_data["call_data"],
                    data_offsets=tx_data["data_offsets"],
                    data_sizes=tx_data["data_sizes"],
                    marker_offsets=tx_data["marker_offsets"],
                    marker_data=tx_data["marker_data"],
                    tx_batch_count=self.num_instances,
                    sequence_length=self.sequence_length,
                )

                if result:
                    self.process_gpu_results(result)
                    pretty_print_gpu_results(result, f"Results")

            except Exception as e:
                print(f"GPU execution error: {e}")
                import traceback

                traceback.print_exc()

        print("\nFinal Results")
        print("=" * 40)
        self.print_population()

    def finalize_report(self):
        """Generate final bug report with source code locations"""
        if not self.detected_bugs:
            print("No bugs detected")
            return

        print(f"\n{len(self.detected_bugs)} bugs detected:")
        print("=" * 80)

        for bug_id, bug in self.detected_bugs.items():
            print(f"\nBug {bug_id}: {bug.bug_type} at PC {bug.pc}")
            print(f"Function: {bug.input.get('function', 'unknown')}")
            print(f"Inputs: {bug.input.get('inputs', [])}")

            try:
                # Get source code location
                frag = self.ast_parser.source_by_pc(self.contract_name, int(bug.pc), deploy=False)
                lines = frag.get("linenums", [0, 0])
                self.detected_bugs[bug_id].line_info = lines

                if lines[0] == lines[1]:
                    print(f"Location: Line {lines[0]}")
                else:
                    print(f"Location: Lines {lines[0]}-{lines[1]}")

                source_code = frag.get("fragment", "Source not available")
                if source_code and source_code != "Source not available":
                    print(f"Source: {source_code.strip()}")

            except Exception as e:
                print(f"Location: Unable to determine source location ({e})")

            print("-" * 40)


# python fuzzer.py --input sample.sol --config sample_config.json --timeout 10 --contract_name Test --output report.json \
#                       --test_case test_case.json --num_instaces 10 --num_iterations 100 --random_seed 0
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EVM fuzzer")
    parser.add_argument("--input", default="contracts/overflow.sol", help="source file")
    parser.add_argument("--config", default="configurations/default.json", help="config file")
    parser.add_argument("--timeout", default=10, help="timeout in seconds")
    parser.add_argument("--contract_name", help="contract name")
    parser.add_argument("--output", help="output file")
    parser.add_argument("--test_case", help="test case file")
    parser.add_argument("--num_instances", default=2, help="number of instances")
    parser.add_argument("--num_iterations", default=100, help="number of iterations")
    parser.add_argument("--skip_tx_size", default=32, help="number of transactions per instance")
    parser.add_argument("--random_seed", default=0, help="random seed")
    parser.add_argument("--branch_heuristic", action="store_true", help="branch heuristic")
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=1,
        help="length of transaction sequences (default: 1)",
    )
    args = parser.parse_args()
    fuzzer = Fuzzer(
        args.input,
        int(args.num_instances),
        int(args.skip_tx_size),
        args.timeout,
        args.config,
        contract_name=args.contract_name,
        output=args.output,
        test_case_file=args.test_case,
        random_seed=int(args.random_seed),
        branch_heuristic=args.branch_heuristic,
        sequence_length=int(args.sequence_length),
    )
    fuzzer.run(num_iterations=int(args.num_iterations))
    fuzzer.finalize_report()
