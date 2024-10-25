import subprocess
import concurrent.futures
import os
import time

run_id = os.getenv("GITHUB_RUN_ID")
workspace = os.getenv("GITHUB_WORKSPACE")
workdir = f'{workspace}/{run_id}'
container_name = f"cuevm-test-runner-{run_id}"
timeout_secs = 1800
max_workers = 1 # limited by RAM size set for each docker process and CPU cores

# Run these most time-consuming folders first to have a better chance of completing them
slow_folders_with_time = [
    # ("stTimeConsuming", 1500,),
    # ("stRandom2", 1500,),
    # ("stBadOpcode", 1500,),
    # ("stRandom", 1500,),
    # ("stStaticFlagEnabled", 1500,),
    # ("stMemoryTest", 1500,),
]

folders_to_test = [f for f in os.listdir(f"{workdir}/ethereum/tests/GeneralStateTests") if f not in [f[0] for f in slow_folders_with_time]]

# Function to run the docker command for each folder
def run_test(folder, timeout_value, run_id, workspace):
    print(f"⏰ Running tests for {folder} with a timeout of {timeout_value} seconds...")

    process = None
    start_time = time.time()
    try:
        # Run the test inside the container
        docker_cmd = [
            "docker", "exec", container_name,
            "python3", "scripts/run-ethtest-by-fork.py", "-t", "/tmp/",
            "--runtest-bin", "/goevmlab/runtest",
            "--geth", "/goevmlab/gethvm",
            "--cuevm", "/workspaces/CuEVM/build/cuevm_GPU",
            "-i", f"/workspaces/CuEVM/ethereum/tests/GeneralStateTests/{folder}",
            "--ignore-errors", "--without-state-root"
        ]
        log_file = os.path.join(workspace, f"{run_id}/test-outputs/{folder}.log")
        with open(log_file, 'w') as log:
            process = subprocess.Popen(docker_cmd, stdout=log, stderr=log, text=True, preexec_fn=os.setsid)
            process.wait(timeout=timeout_value)
        print(f"👌 Test for {folder} completed.")
    except subprocess.TimeoutExpired as e:
        print(f"🚫 Test for {folder} failed with timeout error {e}")
    except Exception as e:
        print(f"🚫 Test for {folder} failed with error {e}")
    finally:
        end_time = time.time()
        print(f"➡️ Test for {folder} took {int(end_time - start_time)} seconds")
        while process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, 9)
                if process.poll() is None:
                    time.sleep(1)
            except Exception as e:
                print(f"⛔ Failed to kill process for test {folder} Error: {e}")
                break

# Main function to run the tests
def main():
    subprocess.run(["docker", "start", container_name])

    # Create the test outputs directory
    os.makedirs(f"{workdir}/test-outputs", exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for folder, long_timeout_secs in slow_folders_with_time:
            futures.append(executor.submit(run_test, folder, long_timeout_secs, run_id, workspace))

        for folder in folders_to_test:
            futures.append(executor.submit(run_test, folder, timeout_secs, run_id, workspace))

        # Wait for all futures to complete
        concurrent.futures.wait(futures)

if __name__ == "__main__":
    main()
