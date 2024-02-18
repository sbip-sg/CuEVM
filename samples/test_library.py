import sys
import ctypes
import json

# Add the directory containing your .so file to the Python path
sys.path.append('../build/')

import libcuevm  # Now you can import your module as usual

def process_json(input_data):
    # Convert Python dictionary to JSON string
    # input_json_string = json.dumps(input_data).encode('utf-8')

    input_data = input_data[list(input_data.keys())[0]] # extract first value
    # Call the library function with the JSON string
    print ("input data")
    print (input_data)
    # result_json = libcuevm.print_dict(input_data)
    result_json = libcuevm.run_dict(input_data)
    print ("result_json")
    print (result_json)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python test_library.py <json_file> <library_path> <num_instances>")
        sys.exit(1)
    json_file = sys.argv[1]
    library_path = sys.argv[2] or './libjsonprocessor.so'
    num_instances = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    # Load the shared library

    input_data = json.loads(open(json_file).read())
    instance_data = input_data[list(input_data.keys())[0]]
    instance_data["transaction"]["data"] = instance_data["transaction"]["data"] * (num_instances//len(instance_data["transaction"]["data"]))
    instance_data["transaction"]["gasLimit"] = instance_data["transaction"]["gasLimit"] * (num_instances//len(instance_data["transaction"]["gasLimit"]))
    instance_data["transaction"]["value"] = instance_data["transaction"]["value"] * (num_instances//len(instance_data["transaction"]["value"]))
    print (f'generated {len(instance_data["transaction"]["data"])} instances')
    output_data = process_json(input_data)
    print ("after calling lib, before printing output_data")
    # print(output_data)