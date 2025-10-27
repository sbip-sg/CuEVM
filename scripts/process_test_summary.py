#!/usr/bin/env python3
"""
Script to process test summary markdown and create a better formatted version for GitHub README.
"""

import sys
from typing import List, Dict


def create_progress_bar(percentage: float, width: int = 20, use_html: bool = False) -> str:
    """Create a visual progress bar using Unicode block characters or HTML."""

    if use_html:
        # HTML progress bar for GitHub README - modern muted palette with gradient
        if percentage >= 90:
            color = "#8ac48d"  # Soft sage green
        elif percentage >= 80:
            color = "#b8d88e"  # Yellow-green (between green and amber)
        elif percentage >= 50:
            color = "#f4d06f"  # Soft amber
        else:
            color = "#f09e8d"  # Soft coral

        # Create HTML progress bar
        html_bar = f'<div style="background: #e9ecef; border-radius: 4px; overflow: hidden; width: 200px; height: 20px; display: inline-block; margin: 2px 0;"><div style="background: {color}; height: 100%; width: {percentage}%; border-radius: 4px; text-align: center; line-height: 20px; color: white; font-size: 12px; font-weight: bold;">{percentage:.1f}%</div></div>'
        return html_bar
    else:
        # Unicode text bar for compatibility
        filled_blocks = int((percentage / 100) * width)
        empty_blocks = width - filled_blocks

        bar = "█" * filled_blocks + "░" * empty_blocks
        text_bar = f"`{bar}` {percentage:.1f}%"

        return text_bar


def parse_markdown_table(content: str) -> List[Dict]:
    """Parse the markdown table and extract test data."""
    lines = content.strip().split("\n")

    # Find the table header and data rows
    header_found = False
    data_rows = []

    for line in lines:
        if "| Test folder |" in line:
            header_found = True
            continue
        elif header_found and line.startswith("| --- |"):
            continue
        elif header_found and line.startswith("|") and line.endswith("|"):
            # Parse data row
            parts = [part.strip() for part in line.split("|")[1:-1]]  # Remove empty first and last elements
            if len(parts) >= 6:
                try:
                    test_folder = parts[0]
                    passed = int(parts[1])
                    failed = int(parts[2])
                    skipped = int(parts[3])
                    timeout = int(parts[4])
                    time_taken = float(parts[5])

                    data_rows.append(
                        {
                            "test_folder": test_folder,
                            "passed": passed,
                            "failed": failed,
                            "skipped": skipped,
                            "timeout": timeout,
                            "time_taken": time_taken,
                        }
                    )
                except ValueError:
                    continue

    return data_rows


def create_summary_table(data: List[Dict], use_html: bool = False) -> str:
    """Create a summary table with the requested format."""

    # Filter out entries with 0 total tests (skipped experiments)
    filtered_data = []
    for row in data:
        total_tests = row["passed"] + row["failed"] + row["timeout"]
        if total_tests > 0:
            filtered_data.append(
                {**row, "total_tests": total_tests, "passed_percentage": (row["passed"] / total_tests) * 100}
            )

    # Sort by pass percentage (high to low), then by total tests (high to low)
    filtered_data.sort(key=lambda x: (x["passed_percentage"], x["total_tests"]), reverse=True)

    # Calculate totals for summary
    total_passed = sum(row["passed"] for row in filtered_data)
    total_failed = sum(row["failed"] for row in filtered_data)
    total_timeout = sum(row["timeout"] for row in filtered_data)

    # Create markdown table
    markdown = []
    markdown.append("## Test Results Summary")
    markdown.append("")

    grand_total = total_passed + total_failed + total_timeout
    if grand_total > 0:
        overall_percentage = (total_passed / grand_total) * 100
    else:
        overall_percentage = 0

    overall_bar = create_progress_bar(overall_percentage, use_html=use_html)

    if use_html:
        # HTML collapsible version
        markdown.append("| Test Folder | Total Tests | Passed (%) |")
        markdown.append("| --- | --- | --- |")
        markdown.append(f"| **TOTAL** | **{grand_total}** | {overall_bar} |")

        # Find VMTests row if it exists
        vmtest_row = None
        other_rows = []
        for row in filtered_data:
            if "VMTests" in row["test_folder"]:
                vmtest_row = row
            else:
                other_rows.append(row)

        # Display VMTests row if exists
        if vmtest_row:
            test_folder = vmtest_row["test_folder"]
            total_tests = vmtest_row["total_tests"]
            passed_percentage = vmtest_row["passed_percentage"]
            progress_bar = create_progress_bar(passed_percentage, use_html=use_html)
            markdown.append(f"| {test_folder} | {total_tests} | {progress_bar} |")

        markdown.append("")
        markdown.append("")

        # Add collapsible section for other test folders
        markdown.append("<details>")
        markdown.append("<summary><strong>📊 Click to view detailed results for all test folders</strong></summary>")
        markdown.append("")
        markdown.append("| Test Folder | Total Tests | Passed (%) |")
        markdown.append("| --- | --- | --- |")

        # Add all rows in the collapsible section
        for row in filtered_data:
            test_folder = row["test_folder"]
            total_tests = row["total_tests"]
            passed_percentage = row["passed_percentage"]
            progress_bar = create_progress_bar(passed_percentage, use_html=use_html)
            markdown.append(f"| {test_folder} | {total_tests} | {progress_bar} |")

        markdown.append("")
        markdown.append("</details>")
    else:
        # Text version - simple table
        markdown.append("| Test Folder | Total Tests | Passed (%) |")
        markdown.append("| --- | --- | --- |")
        markdown.append(f"| **TOTAL** | **{grand_total}** | {overall_bar} |")

        # Process each test folder
        for row in filtered_data:
            test_folder = row["test_folder"]
            total_tests = row["total_tests"]
            passed_percentage = row["passed_percentage"]
            progress_bar = create_progress_bar(passed_percentage, use_html=use_html)
            markdown.append(f"| {test_folder} | {total_tests} | {progress_bar} |")

    # Add summary statistics (without skipped)
    markdown.append("")
    markdown.append("**Summary Statistics:**")
    markdown.append(f"- Total Tests Run: {grand_total:,}")
    markdown.append(f"- Passed: {total_passed:,} ({overall_percentage:.1f}%)")
    markdown.append(f"- Failed: {total_failed:,}")
    markdown.append(f"- Timeout: {total_timeout:,}")

    return "\n".join(markdown)


def main():
    """Main function to process the markdown file."""
    import argparse

    parser = argparse.ArgumentParser(description="Process test summary markdown for GitHub README")
    parser.add_argument("input_file", help="Input markdown file to process")
    parser.add_argument("--html", action="store_true", help="Use HTML progress bars (better for GitHub)")
    parser.add_argument("--output", "-o", help="Output file (default: input_processed.md)")

    args = parser.parse_args()

    input_file = args.input_file
    use_html = args.html

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse the markdown table
        data = parse_markdown_table(content)

        if not data:
            print("No valid test data found in the markdown file.")
            sys.exit(1)

        # Create the summary table
        summary = create_summary_table(data, use_html=use_html)

        # Determine output file
        if args.output:
            output_file = args.output
        else:
            output_file = input_file.replace(".md", "_processed.md")

        # Write to output file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(summary)

        print(f"Processed summary written to: {output_file}")
        if use_html:
            print("Note: HTML progress bars work best when viewed on GitHub")

        print("\nPreview:")
        print("=" * 50)
        print(summary)

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
