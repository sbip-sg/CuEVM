# python run-ethtest-by-fork.py -i . -t /tmp/out/ --runtest-bin runtest --geth geth --cuevm cuevm --ignore-errors
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

    if len(geth_traces) != len(cuevm_traces) or len(geth_traces) < 2:
        raise ValueError("\033[91m💥\033[0m Mismatched trace length")

    if geth_traces[0] != cuevm_traces[0]:
        raise ValueError("\033[91m💥\033[0m Mismatched traces")

def run_single_test(output_filepath, runtest_bin, geth_bin, cuevm_bin):
    command = [runtest_bin, f'--geth={geth_bin}', f'--cuevm={cuevm_bin}', output_filepath]

    print(' '.join(command))

    clean_test_out()
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    compare_output()

    print(f"\033[92m🎉\033[0m Test passed for {output_filepath}")

def runtest_fork(input_directory, output_directory, fork='Shanghai', runtest_bin='runtest', geth_bin='geth', cuevm_bin='cuevm', ignore_errors=False, result={}):
    result = result or {'n_total': 0, 'n_success': 0}
    for dirpath, dirnames, filenames in os.walk(input_directory):
        rel_path = os.path.relpath(dirpath, input_directory)
        for filename in filenames:
            rootname = filename.split('.')[0]
            try:
                if filename.endswith(".json"):
                    input_filepath = os.path.join(dirpath, filename)

                    with open(input_filepath, 'r', encoding='utf-8') as file:
                        data = json.load(file)

                    post_data = data[rootname]['post']

                    if fork not in post_data:
                        print(f"No entries related to Shanghai found in {input_filepath}. Skipping...")
                        continue

                    post_data = {key: value for key, value in post_data.items() if key == fork}

                    # unwrap the transactions because cuevm does not lookup by index currently
                    transaction_data = data[rootname]['transaction']['data']
                    transaction_gaslimit = data[rootname]['transaction']['gasLimit']
                    transaction_value = data[rootname]['transaction']['value']
                    transaction = data[rootname]['transaction']

                    test_index = 0
                    for test in post_data[fork]:
                        data = copy.deepcopy(data)
                        data[rootname]['post'] = {fork: [test]}
                        indexes = test.get('indexes', {})
                        test_data_index = indexes.get('data')
                        test_value_index = indexes.get('value')
                        test_gas_index =  indexes.get('gas')

                        new_transaction = copy.deepcopy(transaction)

                        if test_data_index is not None:
                            new_transaction['data'] = [ transaction_data[test_data_index], ]
                            indexes['data'] = 0

                        if test_value_index is not None:
                            new_transaction['value'] = [ transaction_value[test_value_index], ]
                            indexes['value'] = 0

                        if test_gas_index is not None:
                            new_transaction['gasLimit'] = [ transaction_gaslimit[test_gas_index], ]
                            indexes['gas'] = 0

                        output_filepath = os.path.join(output_directory, rel_path, f'{test_index}', filename)
                        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

                        data[rootname]['transaction'] = new_transaction

                        with open(output_filepath, 'w', encoding='utf-8') as file:
                            json.dump(data, file, ensure_ascii=False, indent=2)

                        print(f"Processed and saved {output_filepath} successfully.")
                        run_single_test(output_filepath, runtest_bin, geth_bin, cuevm_bin)
                        test_index += 1
                        if result:
                            result['n_success'] += 1
                            result['n_total'] += 1
            except Exception as e:
                if result:
                    result['n_total'] += 1
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

    result = {'n_total': 0, 'n_success': 0}
    try:
        runtest_fork(args.input, args.temporary_path, fork='Shanghai', runtest_bin=args.runtest_bin, geth_bin=args.geth, cuevm_bin=args.cuevm, ignore_errors=args.ignore_errors, result=result)
    finally:
        print(f"Total tests: {result['n_total']}, Passed: {result['n_success']}")


if __name__ == "__main__":
    main()
