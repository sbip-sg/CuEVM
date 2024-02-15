import sys
import ctypes
import json


def process_json(lib, input_data):
    # Convert Python dictionary to JSON string
    input_json_string = json.dumps(input_data).encode('utf-8')

    # Call the library function with the JSON string
    result_json_string = lib.run_json_string(input_json_string)

    # Convert the result JSON string back to a Python dictionary
    if result_json_string is not None:
        result_data = json.loads(result_json_string.decode('utf-8'))
        # lib.free_json_string(result_json_string)
        return result_data
    else:
        return None



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_library.py <json_file> <library_path>")
        sys.exit(1)
    json_file = sys.argv[1]
    library_path = sys.argv[2] or './libjsonprocessor.so'
    # Load the shared library
    lib = ctypes.CDLL(library_path)

    # Define the function return type and argument types
    lib.run_json_string.restype = ctypes.c_char_p
    lib.run_json_string.argtypes = [ctypes.c_char_p]

    input_data = json.loads(open(json_file).read())
    output_data = process_json(lib, input_data)
    print ("after calling lib, before printing output_data")
    print(output_data)