import os
import re

def generate_test_summary(workspace, run_id):
    test_outputs_dir = f"{workspace}/{run_id}/test-outputs"
    summary_file = os.path.join(test_outputs_dir, "summary.md")

    # Create or overwrite the summary file
    with open(summary_file, 'w') as summary:
        summary.write("| Folder | Status | Passed | Failed | Skipped |\n")
        summary.write("|---|---|---|---|---|\n")

        # Process each log file in the test-outputs directory
        for log_file in sorted(os.listdir(test_outputs_dir)):
            if log_file.endswith(".log"):
                folder = os.path.splitext(log_file)[0]  # Get the folder name from the log file name
                log_file_path = os.path.join(test_outputs_dir, log_file)

                with open(log_file_path, 'r') as log:
                    log_content = log.read()

                    passed_count = log_content.count("🎉")
                    failed_count = log_content.count("💥")
                    skipped_count = "?"

                    # Check if the log file contains test results
                    if "Test result" in log_content:
                        passed = re.search(r'Passed: (\d+)', log_content)
                        failed = re.search(r'Failed: (\d+)', log_content)
                        skipped = re.search(r'Skipped: (\d+)', log_content)

                        passed_count = passed.group(1) if passed else "0"
                        failed_count = failed.group(1) if failed else "0"
                        skipped_count = skipped.group(1) if skipped else "0"

                        status = "Completed"
                        summary.write(f"| {folder} | {status} | {passed_count} | {failed_count} | {skipped_count} |\n")

                    elif "timed out after" in log_content:
                        status = "Timed out"
                        summary.write(f"| {folder} | {status} | {passed_count} | {failed_count} | {skipped_count} |\n")

                    elif "terminated due to out-of-memory" in log_content:
                        status = "Out of memory"
                        summary.write(f"| {folder} | {status} | {passed_count} | {failed_count} | {skipped_count} |\n")

                    else:
                        status = "Unknown"
                        summary.write(f"| {folder} | {status} | {passed_count} | {failed_count} | {skipped_count} |\n")


if __name__ == "__main__":
    # Define workspace and run_id, normally passed from environment variables
    workspace = os.getenv('GITHUB_WORKSPACE', '/path/to/default/workspace')  # Replace with a default if needed
    run_id = os.getenv('GITHUB_RUN_ID', 'default_run_id')

    # Ensure test-outputs directory exists
    os.makedirs(f"{workspace}/{run_id}/test-outputs", exist_ok=True)

    # Generate the test summary
    generate_test_summary(workspace, run_id)
