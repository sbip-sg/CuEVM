#!/usr/bin/env python3

import copy
import json
import subprocess
import shutil
import os
from datetime import datetime
import time

log_file_prefix = "run-ethtest-by-fork-traces-only"
TIMEOUT = 90

# Performance related tests are excluded
exclude_tests = [
    "vmPerformance",
    "stCreateTest",
    "stCreateTest",
    "stQuadraticComplexityTest",
    "stStaticCall",
    "stTimeConsuming",
]


log_file = open('run-ethtest-by-fork.log', 'a')

def current_time():
    return datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')

def debug_print(*args, **kwargs):
    now = current_time()
    args = [now] + list(args)
    print(*args, **kwargs, file=log_file, flush=True)
    print(*args, **kwargs, flush=True)

def assert_command_in_path(cmd):
    if not shutil.which(cmd):
        raise ValueError(f"{cmd} not found in PATH")

def clean_test_out():
    subprocess.run('rm *.jsonl', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def read_as_json_lines(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

def check_output(output, error, without_state_root):
    all_output = output + error
    print(all_output)
    has_str = lambda s: s in all_output
    if without_state_root:
        if has_str('error') and all_output.count('stateRoot') < 2:
            raise ValueError(f"💥 Mismatch found {output}")
    else:
        if has_str('error'):
            raise ValueError(f"💥 Mismatch found {output}")


def run_single_test(output_filepath, runtest_bin, geth_bin, cuevm_bin, without_state_root):
    command = [runtest_bin, f'--outdir=./', f'--geth={geth_bin}', f'--cuevm={cuevm_bin}', output_filepath]

    debug_print(' '.join(command))

    clean_test_out()
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, preexec_fn=os.setsid)
    try:
        stdout, stderr = proc.communicate(timeout=TIMEOUT)
        check_output(stdout, stderr, without_state_root=without_state_root)
        debug_print(f"\033[92m🎉\033[0m Test passed for {output_filepath}")
    finally:
        try:
            os.killpg(proc.pid, 9)
            proc.wait()
        except ProcessLookupError:
            pass

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

                    if any(exclude in input_filepath for exclude in exclude_tests):
                        debug_print(f"Skipping {rootname} as it is in exclude list")
                        if result: result['skip_files'].append(input_filepath)
                        continue

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

                            try:
                                run_single_test(output_filepath, runtest_bin, geth_bin, cuevm_bin, True)
                                result['n_success'] += 1
                            except subprocess.TimeoutExpired:
                                result['timeout_files'].append(output_filepath)
                                debug_print(f"Test timed out for {output_filepath}")
                            except Exception as e:
                                result['failed_files'].append(output_filepath)
                                if ignore_errors:
                                    debug_print(f"{str(e)}")
                                else:
                                    raise
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

    test_root_folder = args.input
    test_folders = os.listdir(test_root_folder)

    global log_file
    summary_output_file = f"summary-{current_time()}.md"
    with open(summary_output_file, 'w') as f:
        f.write(f"Test result summary\n\n")
        f.write(f"| Test folder | Passed | Failed | Skipped | Timeout | Time taken (seconds) |\n")
        f.write(f"| --- | --- | --- | --- | --- | --- |\n")

    for folder in test_folders:
        start = time.time()
        log_file = open(f'{log_file_prefix}-{folder}.log', 'a')
        result = {'n_success': 0, 'failed_files': [], 'skip_files': [], 'timeout_files': []}
        try:
            test_root = os.path.join(test_root_folder, folder)
            print(f"Running tests for {test_root}")
            runtest_fork(test_root, args.temporary_path, fork='Shanghai', runtest_bin=args.runtest_bin, geth_bin=args.geth, cuevm_bin=args.cuevm, ignore_errors=args.ignore_errors, result=result)
        except Exception:
            pass
        finally:
            period = time.time() - start
            skipped = result['skip_files']
            n_skipped = len(skipped)
            n_failed = len(result['failed_files'])
            n_timeout = len(result['timeout_files'])

            debug_print(f"Test result, Passed: {result['n_success']}, Failed: {n_failed}, Skipped: {n_skipped}, Timeout: {n_timeout}")
            debug_print("Skipped files:")
            debug_print(skipped)
            debug_print("Failed files:")
            debug_print(result['failed_files'])
            debug_print("Timeout files:")
            debug_print(result['timeout_files'])
            debug_print(f"Time taken: {period:.2f} seconds")
            with open(summary_output_file, 'a') as f:
                f.write(f"| {folder} | {result['n_success']} | {n_failed} | {n_skipped} | {n_timeout} | {period:.2f} |\n")

if __name__ == "__main__":
    main()
