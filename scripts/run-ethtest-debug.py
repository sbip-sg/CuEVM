# python3 ./scripts/run-ethtest-debug.py -i ./input  -t ./tmp --runtest-bin runtest --geth gethvm --cuevm ./build/cuevm --ignore-errors
import copy
import json
import subprocess
import shutil
import os

def assert_command_in_path(cmd):
    if not shutil.which(cmd):
        raise ValueError(f"{cmd} not found in PATH")

def clean_test_out():
    subprocess.run('rm *.jsonl', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def read_as_json_lines(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

def compare_output():
    geth_traces = list(read_as_json_lines('geth-0-output.jsonl'))
    cuevm_traces = list(read_as_json_lines('cuevm-0-output.jsonl'))

    if len(geth_traces) != len(cuevm_traces):
        raise ValueError("\033[91m💥\033[0m Mismatched trace length")

    if geth_traces[:-1] != cuevm_traces[:-1]:
        raise ValueError("\033[91m💥\033[0m Mismatched traces")

def run_single_test(output_filepath, runtest_bin, geth_bin, cuevm_bin):
    command = [runtest_bin, f'--outdir=./', f'--geth={geth_bin}', f'--cuevm={cuevm_bin}', output_filepath]

    print(' '.join(command))

    clean_test_out()
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    compare_output()

    print(f"\033[92m🎉\033[0m Test passed for {output_filepath}")

def runtest_fork(input_directory, output_directory, fork='Shanghai', runtest_bin='runtest', geth_bin='geth', cuevm_bin='cuevm', ignore_errors=False, result={}):
    result = result or {'n_total': 0, 'n_success': 0, 'failed_files': []}
    output_filepath = None
    for dirpath, dirnames, filenames in os.walk(input_directory):
        rel_path = os.path.relpath(dirpath, input_directory)
        for filename in filenames:
            print("Processing", dirpath, filename)
            rootname = filename.split('.')[0]
            if filename.endswith(".json"):
                input_filepath = os.path.join(dirpath, filename)

                with open(input_filepath, 'r', encoding='utf-8') as file:
                    data = json.load(file)

                for rootname in list(data.keys()):
                    transaction_data = data[rootname]['transaction']['data']
                    transaction_gaslimit = data[rootname]['transaction']['gasLimit']
                    transaction_value = data[rootname]['transaction']['value']
                    transaction = data[rootname]['transaction']

                    data_len = len(transaction_data)
                    gaslimit_len = len(transaction_gaslimit)
                    value_len = len(transaction_value)

                    for data_index in range(data_len):
                        for gas_index in range(gaslimit_len):
                            for value_index in range(value_len):
                                data = copy.deepcopy(data)
                                data[rootname]['post'] = {fork: [{}]}
                                new_transaction = copy.deepcopy(transaction)
                                new_transaction['data'] = [transaction_data[data_index]]
                                new_transaction['gasLimit'] = [transaction_gaslimit[gas_index]]
                                new_transaction['value'] = [transaction_value[value_index]]

                                output_filepath = os.path.join(output_directory, rel_path, f'{rootname}-{data_index}-{gas_index}-{value_index}', filename)
                                os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

                                data[rootname]['transaction'] = new_transaction

                                with open(output_filepath, 'w', encoding='utf-8') as file:
                                    json.dump(data, file, ensure_ascii=False, indent=2)

                                print(f"Processed and saved {output_filepath} successfully.")
                                try:
                                    run_single_test(output_filepath, runtest_bin, geth_bin, cuevm_bin)
                                    if result:
                                        result['n_success'] += 1
                                        result['n_total'] += 1
                                except Exception as e:
                                    if result:
                                        result['n_total'] += 1
                                        result['failed_files'].append(output_filepath)
                                    if ignore_errors:
                                        print(f"{str(e)}")
                                    else:
                                        raise



def main():
    import argparse
    parser = argparse.ArgumentParser(description='Filter JSON files for entries related to "Shanghai"')
    parser.add_argument('--input', '-i',  type=str, required=True, help='Input directory containing JSON files')
    parser.add_argument('--temporary-path', '-t', type=str, required=True, help='Temporary directory to save the test files')
    parser.add_argument('--runtest-bin', type=str, required=True, help='goevmlab runtest binary path')
    parser.add_argument('--geth', type=str, required=True, help='geth binary path')
    parser.add_argument('--cuevm', type=str, required=True, help='cuevm binary path')
    parser.add_argument('--ignore-errors', action='store_true', help='Continue testing even when test errors occur')

    args = parser.parse_args()

    for cmd in [args.runtest_bin, args.geth, args.cuevm]:
        assert_command_in_path(cmd)

    result = {'n_total': 0, 'n_success': 0, 'failed_files': []}
    try:
        runtest_fork(args.input, args.temporary_path, fork='Shanghai', runtest_bin=args.runtest_bin, geth_bin=args.geth, cuevm_bin=args.cuevm, ignore_errors=args.ignore_errors, result=result)
    finally:
        print(f"Total tests: {result['n_total']}, Passed: {result['n_success']}, Failed files: {result['failed_files']}")

if __name__ == "__main__":
    main()