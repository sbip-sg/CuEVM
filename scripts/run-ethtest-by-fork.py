# python ~/projects/sbip-sg/CuEVM/scripts/run-ethtest-by-fork.py   -t /tmp/out --runtest-bin runtest --geth geth --cuevm ~/projects/sbip-sg/CuEVM/out/cpu_debug_interpreter  --ignore-errors  -i GeneralStateTests/
import copy
import json
import subprocess
import shutil
import os

log_file = open('run-ethtest-by-fork.log', 'a')

def debug_print(*args, **kwargs):
    print(*args, **kwargs)

def assert_command_in_path(cmd):
    if not shutil.which(cmd):
        raise ValueError(f"{cmd} not found in PATH")

def clean_test_out():
    subprocess.run('rm *.jsonl', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def read_as_json_lines(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

def check_output(output, error):
    has_str = lambda s: s in (output + error)
    if has_str('error'): # and not has_str('stateRoot')
        raise ValueError(f"\033[91m💥\033[0m Mismatch found {output}")

def run_single_test(output_filepath, runtest_bin, geth_bin, cuevm_bin):
    command = [runtest_bin, f'--outdir=./', f'--geth={geth_bin}', f'--cuevm={cuevm_bin}', output_filepath]

    debug_print(' '.join(command))

    clean_test_out()
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    check_output(result.stdout, result.stderr)

    debug_print(f"\033[92m🎉\033[0m Test passed for {output_filepath}")

def runtest_fork(input_directory, output_directory, fork='Shanghai', runtest_bin='runtest', geth_bin='geth', cuevm_bin='cuevm', ignore_errors=False, result={}):
    result = result or {'n_success': 0, 'failed_files': []}
    output_filepath = None
    for dirpath, dirnames, filenames in os.walk(input_directory):
        rel_path = os.path.relpath(dirpath, input_directory)
        for filename in filenames:
            debug_print("Processing", dirpath, filename)
            rootname = filename.split('.')[0]
            try:
                if filename.endswith(".json"):
                    input_filepath = os.path.join(dirpath, filename)

                    with open(input_filepath, 'r', encoding='utf-8') as file:
                        data = json.load(file)

                    for rootname in list(data.keys()):
                        debug_print(f'rootname: {rootname}')
                        if 'transaction' not in data[rootname]:
                            debug_print(f"Skipping {rootname} as it does not have a `transaction`")
                            if result: result['skip_files'].append(input_filepath)
                            continue
                        transaction = data[rootname]['transaction']
                        transaction_data = transaction['data']
                        transaction_gaslimit = transaction['gasLimit']
                        transaction_value = transaction['value']

                        post_by_fork = data[rootname]['post'].get(fork)

                        if not post_by_fork:
                            debug_print(f"Skipping {rootname} as it does not have a `:post` for {fork}")
                            if result: result['skip_files'].append(input_filepath)
                            continue

                        for tx in post_by_fork:
                            indexes = tx['indexes']
                            data_index = indexes['data']
                            gas_index = indexes['gas']
                            value_index = indexes['value']

                            tx = copy.deepcopy(tx)
                            tx['indexes'] = dict(data=0, gas=0, value=0)
                            data = copy.deepcopy(data)
                            data[rootname]['post'] = {fork: [tx]}
                            new_transaction = copy.deepcopy(transaction)
                            new_transaction['data'] = [transaction_data[data_index]]
                            new_transaction['gasLimit'] = [transaction_gaslimit[gas_index]]
                            new_transaction['value'] = [transaction_value[value_index]]

                            output_filepath = os.path.join(output_directory, rel_path, f'{rootname}-{data_index}-{gas_index}-{value_index}', filename)
                            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

                            data[rootname]['transaction'] = new_transaction

                            with open(output_filepath, 'w', encoding='utf-8') as file:
                                json.dump(data, file, ensure_ascii=False, indent=2)

                            debug_print(f"Processed and saved {output_filepath} successfully.")

                            run_single_test(output_filepath, runtest_bin, geth_bin, cuevm_bin)
                            result['n_success'] += 1
            except Exception as e:
                result['failed_files'].append(output_filepath)
                if ignore_errors:
                    debug_print(f"{str(e)}")
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

    result = {'n_success': 0, 'failed_files': [], 'skip_files': []}
    try:
        test_root = args.input
        print(f"Running tests for {test_root}")
        runtest_fork(test_root, args.temporary_path, fork='Shanghai', runtest_bin=args.runtest_bin, geth_bin=args.geth, cuevm_bin=args.cuevm, ignore_errors=args.ignore_errors, result=result)
    except Exception:
        pass
    finally:
        skipped = result['skip_files']
        n_skipped = len(skipped)
        n_failed = len(result['failed_files'])

        debug_print(f"Test result, Passed: {result['n_success']}, Failed: {n_failed}, Skipped: {n_skipped}")
        debug_print("Skipped files:")
        debug_print(skipped)
        debug_print("Failed files:")
        debug_print(result['failed_files'])

if __name__ == "__main__":
    main()
