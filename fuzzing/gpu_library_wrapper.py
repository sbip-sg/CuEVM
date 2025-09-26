"""
GPU-optimized library wrapper that mirrors fuzzer.go functionality
"""

import sys
import ctypes
from ctypes import (
    Structure,
    POINTER,
    c_uint32,
    c_uint64,
    c_uint8,
    c_int32,
    c_char_p,
    c_bool,
)
from typing import List, Optional

# Add the build directory to path
sys.path.insert(0, "../build/")


# Define C structures that match fuzzer.go
class ReturnDataEntry(Structure):
    _fields_ = [("data", POINTER(c_uint8)), ("length", c_uint32)]


class BugInfoEntry(Structure):
    _fields_ = [
        ("bug_thread_idx", c_uint32),
        ("bug_id", c_uint32),  # pc << 16 | bug_type
    ]


class BranchInfoEntry(Structure):
    _fields_ = [
        ("branch_thread_idx", c_uint32),
        ("branch_id", c_uint32),  # pc_src << 16 | pc_dst
    ]


class StorageInfoEntry(Structure):
    _fields_ = [
        ("storage_thread_idx", c_uint32),
        (
            "storage_id",
            c_uint32,
        ),  # account_idx (8bit) | storage_type (8bit) | storage_slot (16bit)
    ]


class SimplifiedGPUResultSingleBatchC(Structure):
    _fields_ = [
        ("new_branch_info", POINTER(BranchInfoEntry)),
        ("num_new_branch", c_uint32),
        ("new_bug_info", POINTER(BugInfoEntry)),
        ("num_new_bug", c_uint32),
        ("new_storage_info", POINTER(StorageInfoEntry)),
        ("num_new_storage", c_uint32),
    ]


class SimplifiedGPUResultC(Structure):
    _fields_ = [
        ("results", POINTER(SimplifiedGPUResultSingleBatchC)),
        ("num_results", c_uint32),
    ]


class GPUExecutionResult:
    """Python equivalent of coverage.GPUExecutionResult"""

    def __init__(self):
        self.new_coverage_thread_idx: List[List[int]] = []
        self.new_coverage_ids: List[List[int]] = []
        self.new_bug_thread_idx: List[List[int]] = []
        self.new_bug_pcs: List[List[int]] = []
        self.new_bug_contract_ids: List[List[int]] = []
        self.new_bug_types: List[List[int]] = []
        self.new_storage_thread_idx: List[List[int]] = []
        self.new_storage_ids: List[List[int]] = []


class CuEVMGPULib:
    """GPU-optimized library wrapper that mirrors fuzzer.go functionality"""

    def __init__(self, library_path: str = "../build/libcuevm_go.so"):
        self.lib = None
        self.gpu_initialized = False
        self.num_instances_per_device = 0

        # Load the shared library
        try:
            self.lib = ctypes.CDLL(library_path)
            self._setup_function_signatures()
            print(f"Successfully loaded library: {library_path}")
        except Exception as e:
            print(f"Failed to load library {library_path}: {e}")
            raise

    def _setup_function_signatures(self):
        """Setup C function signatures to match fuzzer.go"""

        # Check if required functions exist
        required_functions = [
            "process_batch_transactions",
            "process_json_state_gpu",
            "get_num_instances_per_device",
            "free_simplified_gpu_result",
        ]

        for func_name in required_functions:
            if not hasattr(self.lib, func_name):
                raise AttributeError(
                    f"Required function {func_name} not found in library"
                )
            print(f"✓ Found function: {func_name}")

        # process_batch_transactions
        self.lib.process_batch_transactions.argtypes = [
            POINTER(c_uint64),  # blockNumber
            POINTER(c_uint64),  # timeStamp
            POINTER(c_uint8),  # fromAddr
            POINTER(c_uint8),  # toAddr
            POINTER(c_uint8),  # values
            POINTER(c_uint8),  # callData
            c_uint32,  # callDataLen
            POINTER(c_uint32),  # dataOffsets
            POINTER(c_uint32),  # dataSizes
            POINTER(c_int32),  # markerOffsets
            POINTER(c_uint32),  # markerData
            c_uint32,  # markerDataLen
            c_uint32,  # txBatchCount
            c_uint32,  # sequenceLength
            c_uint32,  # start_seed
        ]
        self.lib.process_batch_transactions.restype = POINTER(SimplifiedGPUResultC)

        # process_json_state_gpu
        self.lib.process_json_state_gpu.argtypes = [
            c_char_p,  # json_state
            c_uint32,  # num_instances
            c_bool,  # reset_state
            c_uint32,  # skipTxSize
            c_char_p,  # constants
            POINTER(c_uint32),  # markerData
            c_uint32,  # markerDataLen
        ]
        self.lib.process_json_state_gpu.restype = ctypes.c_int

        # get_num_instances_per_device
        self.lib.get_num_instances_per_device.argtypes = []
        self.lib.get_num_instances_per_device.restype = c_uint32

        # free_simplified_gpu_result
        self.lib.free_simplified_gpu_result.argtypes = [POINTER(SimplifiedGPUResultC)]
        self.lib.free_simplified_gpu_result.restype = None

    def initialize_gpu_state(
        self,
        state_json: str = None,
        num_instances: int = 0,
        constants_json: str = None,
        marker_data: List[int] = None,
        skip_tx_size: int = 2,
        reset_state: bool = False,
    ) -> bool:
        """Initialize GPU state - equivalent to prepareAndProcessChainStateInGPU"""

        if self.lib is None:
            print("Library not loaded")
            return False

        # For reset operations, skip validation
        if not reset_state:
            # Validate inputs for initialization
            if not state_json or not state_json.strip():
                print("Empty state JSON provided")
                return False

            if num_instances <= 0:
                print("Invalid number of instances")
                return False

        # Convert strings to C strings (or use NULL for reset)
        state_c_str = None
        constants_c_str = None
        if not reset_state:
            state_c_str = ctypes.c_char_p(state_json.encode("utf-8"))
            constants_c_str = ctypes.c_char_p(
                constants_json.encode("utf-8") if constants_json else b"{}"
            )
        else:
            # For reset, pass NULL pointers
            state_c_str = None
            constants_c_str = None

        # Convert marker data to C array if provided
        marker_data_ptr = None
        marker_data_len = 0
        if marker_data:
            marker_array = (c_uint32 * len(marker_data))(*marker_data)
            marker_data_ptr = ctypes.cast(marker_array, POINTER(c_uint32))
            marker_data_len = len(marker_data)

        # Call the C function
        try:
            if reset_state:
                print("🔄 Resetting GPU state...")
            else:
                print(
                    f"Calling process_json_state_gpu with {num_instances} instances, skip_tx_size={skip_tx_size}"
                )

            result = self.lib.process_json_state_gpu(
                state_c_str,
                c_uint32(num_instances) if not reset_state else c_uint32(0),
                c_bool(reset_state),  # Pass the actual reset_state parameter
                c_uint32(skip_tx_size),
                constants_c_str,
                marker_data_ptr if not reset_state else None,
                c_uint32(marker_data_len) if not reset_state else c_uint32(0),
            )
            print(f"C function returned: {result}")
        except Exception as e:
            print(f"Exception during GPU initialization: {e}")
            return False

        if result == 0:
            self.gpu_initialized = True
            try:
                self.num_instances_per_device = self.lib.get_num_instances_per_device()
                print(
                    f"GPU initialized successfully with {self.num_instances_per_device} instances per device"
                )
            except Exception as e:
                print(f"Warning: Could not get instances per device: {e}")
                self.num_instances_per_device = 0
            return True
        else:
            print(f"GPU initialization failed with error code: {result}")
            return False

    def process_batch_transactions(
        self,
        block_numbers: List[int],
        timestamps: List[int],
        from_addresses: List[int],
        to_address: bytes,
        call_data: bytes,
        data_offsets: List[int],
        data_sizes: List[int],
        marker_offsets: List[int] = None,
        marker_data: List[int] = None,
        tx_batch_count: int = 1,
        sequence_length: int = 1,
        random_seed: int = 0,
    ) -> Optional[GPUExecutionResult]:
        """Process batch transactions on GPU - equivalent to runTransactionsGPU"""

        if not self.gpu_initialized:
            print("GPU not initialized. Call initialize_gpu_state() first.")
            return None

        # Convert Python lists to C arrays
        block_numbers_array = (c_uint64 * len(block_numbers))(*block_numbers)
        timestamps_array = (c_uint64 * len(timestamps))(*timestamps)
        from_addr_array = (c_uint8 * len(from_addresses))(*from_addresses)

        # Prepare to_address (32 bytes, right-padded)
        to_addr_array = (c_uint8 * 32)()
        if len(to_address) >= 20:
            # Copy address to position 12-31 (right-padded)
            for i in range(20):
                to_addr_array[12 + i] = to_address[i] if i < len(to_address) else 0

        # Call data
        call_data_array = (c_uint8 * len(call_data))(*call_data) if call_data else None

        # Data offsets and sizes
        data_offsets_array = (c_uint32 * len(data_offsets))(*data_offsets)
        data_sizes_array = (c_uint32 * len(data_sizes))(*data_sizes)

        # Marker offsets and data (optional)
        marker_offsets_ptr = None
        marker_data_ptr = None
        marker_data_len = 0

        if marker_offsets:
            marker_offsets_array = (c_int32 * len(marker_offsets))(*marker_offsets)
            marker_offsets_ptr = ctypes.cast(marker_offsets_array, POINTER(c_int32))

        if marker_data:
            marker_data_array = (c_uint32 * len(marker_data))(*marker_data)
            marker_data_ptr = ctypes.cast(marker_data_array, POINTER(c_uint32))
            marker_data_len = len(marker_data)

        # Call the C function
        c_result = self.lib.process_batch_transactions(
            ctypes.cast(block_numbers_array, POINTER(c_uint64)),
            ctypes.cast(timestamps_array, POINTER(c_uint64)),
            ctypes.cast(from_addr_array, POINTER(c_uint8)),
            ctypes.cast(to_addr_array, POINTER(c_uint8)),
            None,  # values (not used in current implementation)
            ctypes.cast(call_data_array, POINTER(c_uint8)) if call_data_array else None,
            c_uint32(len(call_data)),
            ctypes.cast(data_offsets_array, POINTER(c_uint32)),
            ctypes.cast(data_sizes_array, POINTER(c_uint32)),
            marker_offsets_ptr,
            marker_data_ptr,
            c_uint32(marker_data_len),
            c_uint32(tx_batch_count),
            c_uint32(sequence_length),
            c_uint32(random_seed),
        )

        if not c_result:
            print("GPU processing failed (C function returned null)")
            return None

        # Convert C result to Python
        return self._convert_c_result_to_python(c_result)

    def _convert_c_result_to_python(
        self, c_result: POINTER(SimplifiedGPUResultC)
    ) -> GPUExecutionResult:
        """Convert C result structure to Python GPUExecutionResult"""

        result = GPUExecutionResult()
        num_results = c_result.contents.num_results

        if num_results == 0:
            self.lib.free_simplified_gpu_result(c_result)
            return result

        # Initialize result lists
        for i in range(num_results):
            result.new_coverage_thread_idx.append([])
            result.new_coverage_ids.append([])
            result.new_bug_thread_idx.append([])
            result.new_bug_pcs.append([])
            result.new_bug_contract_ids.append([])
            result.new_bug_types.append([])
            result.new_storage_thread_idx.append([])
            result.new_storage_ids.append([])

        # Process each batch result
        results_array_type = SimplifiedGPUResultSingleBatchC * num_results
        results_array = ctypes.cast(
            c_result.contents.results, POINTER(results_array_type)
        ).contents

        for i in range(num_results):
            batch_result = results_array[i]

            # Process coverage data
            if batch_result.num_new_branch > 0 and batch_result.new_branch_info:
                branch_array = ctypes.cast(
                    batch_result.new_branch_info,
                    POINTER(BranchInfoEntry * batch_result.num_new_branch),
                ).contents
                for j in range(batch_result.num_new_branch):
                    branch = branch_array[j]
                    result.new_coverage_thread_idx[i].append(branch.branch_thread_idx)
                    result.new_coverage_ids[i].append(branch.branch_id)

            # Process bug data
            if batch_result.num_new_bug > 0 and batch_result.new_bug_info:
                bug_array = ctypes.cast(
                    batch_result.new_bug_info,
                    POINTER(BugInfoEntry * batch_result.num_new_bug),
                ).contents
                for j in range(batch_result.num_new_bug):
                    bug = bug_array[j]
                    result.new_bug_thread_idx[i].append(bug.bug_thread_idx)
                    result.new_bug_pcs[i].append(bug.bug_id >> 16)
                    result.new_bug_types[i].append((bug.bug_id & 0xFFFF) >> 8)
                    result.new_bug_contract_ids[i].append(bug.bug_id & 0xFF)

            # Process storage data
            if batch_result.num_new_storage > 0 and batch_result.new_storage_info:
                storage_array = ctypes.cast(
                    batch_result.new_storage_info,
                    POINTER(StorageInfoEntry * batch_result.num_new_storage),
                ).contents
                for j in range(batch_result.num_new_storage):
                    storage = storage_array[j]
                    result.new_storage_thread_idx[i].append(storage.storage_thread_idx)
                    result.new_storage_ids[i].append(storage.storage_id)

        # Free the C memory
        self.lib.free_simplified_gpu_result(c_result)

        return result

    def reset_gpu_state(self, num_instances: int, skip_tx_size: int = 32) -> bool:
        """Reset GPU state without re-initializing JSON data"""
        if not self.gpu_initialized:
            return False

        result = self.lib.process_json_state_gpu(
            None,  # json_state = null for reset
            c_uint32(num_instances),
            c_bool(True),  # reset_state = True for reset operation
            c_uint32(skip_tx_size),
            None,  # constants = null
            None,  # markerData = null
            c_uint32(0),  # markerDataLen = 0
        )

        return result == 0

    def get_num_instances_per_device(self) -> int:
        """Get number of instances per device"""
        return self.num_instances_per_device

    def __del__(self):
        """Cleanup when object is destroyed"""
        # The C library should handle cleanup internally
        if hasattr(self, "lib"):
            self.lib = None
