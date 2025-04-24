import os
import re
import subprocess
import json
from github import Github
from datetime import datetime, timedelta

def get_changed_files():
    """Get the list of files changed in the latest commit."""
    try:
        result = subprocess.run(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip().split('\n')
    except subprocess.CalledProcessError:
        print("Error getting changed files")
        return []

def get_file_content(file_path):
    """Get the content of a file."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return ""

def find_issues_fixed_by_changes(repo, changed_files):
    """Find issues that might be fixed by the changes in the latest commit."""
    # Get all open issues
    open_issues = list(repo.get_issues(state='open'))
    potentially_fixed_issues = []

    # Extract file paths and error messages from issues
    for issue in open_issues:
        # Skip issues that don't look like test failures or bug reports
        if not ('Test Failure' in issue.title or 'bug' in [label.name for label in issue.labels]):
            continue

        # Extract error information from the issue body
        file_paths = re.findall(r'File:\s*`([^`]+)`', issue.body)
        error_messages = re.findall(r'Error:\s*([^\n]+)', issue.body)

        # If we have both file paths and error messages, check if the changes might fix the issue
        if file_paths and error_messages:
            for file_path in file_paths:
                # Convert from test file path to implementation file path if needed
                if file_path.startswith('tests/'):
                    impl_path = file_path.replace('tests/', '').replace('test_', '')
                    possible_paths = [
                        impl_path,
                        f"neural/{impl_path}",
                        f"neural/{os.path.dirname(impl_path)}/{os.path.basename(impl_path).replace('test_', '')}"
                    ]
                else:
                    possible_paths = [file_path]

                # Check if any of the changed files match the possible paths
                for changed_file in changed_files:
                    if any(possible_path in changed_file for possible_path in possible_paths):
                        # This change might fix the issue
                        potentially_fixed_issues.append((issue, changed_file))
                        break

    return potentially_fixed_issues

def check_if_issue_fixed(issue, changed_file):
    """Check if the issue is likely fixed by the changes in the file."""
    # Get the content of the changed file
    file_content = get_file_content(changed_file)

    # Extract error messages from the issue body
    error_messages = re.findall(r'Error:\s*([^\n]+)', issue.body)

    # Special case for HPO parser issues
    if 'HPO' in issue.title and 'parser' in issue.title.lower() and 'parser.py' in changed_file:
        return True

    # Special case for Conv2D filters issues
    if 'Conv2D' in issue.body and 'filters' in issue.body and 'parser.py' in changed_file:
        return True

    # Check if any of the error messages are addressed in the file content
    for error_message in error_messages:
        # Clean up the error message for better matching
        clean_error = re.sub(r'[^\w\s]', '', error_message).lower()
        words = clean_error.split()

        # Check if the key words from the error message appear in the file content
        if all(word in file_content.lower() for word in words if len(word) > 3):
            return True

    return False

def close_fixed_issues(repo, issues_to_close):
    """Close the specified issues on GitHub."""
    for issue, changed_file in issues_to_close:
        try:
            # Close the issue
            issue.edit(state='closed')

            # Add a comment
            issue.create_comment(
                f"This issue was automatically closed by the CI system because it appears to be fixed "
                f"by changes to `{changed_file}` in a recent commit.\n\n"
                f"If this is incorrect, please reopen the issue."
            )

            print(f"Closed issue #{issue.number}: {issue.title}")
        except Exception as e:
            print(f"Error closing issue #{issue.number}: {str(e)}")

def check_test_results():
    """Check if the test results indicate that issues are fixed."""
    try:
        # Check if test-results.xml exists
        if not os.path.exists('test-results.xml'):
            print("No test results file found")
            return []

        # Parse the test results
        import xml.etree.ElementTree as ET
        tree = ET.parse('test-results.xml')
        root = tree.getroot()

        # Get all test cases
        test_cases = root.findall('.//testcase')

        # Get the names of all passing tests
        passing_tests = []
        for test_case in test_cases:
            if test_case.find('failure') is None and test_case.find('error') is None:
                test_name = test_case.get('name')
                class_name = test_case.get('classname')
                passing_tests.append(f"{class_name}::{test_name}")

        return passing_tests
    except Exception as e:
        print(f"Error checking test results: {str(e)}")
        return []

def find_issues_fixed_by_tests(repo, passing_tests):
    """Find issues that might be fixed based on passing tests."""
    # Get all open issues
    open_issues = list(repo.get_issues(state='open'))
    potentially_fixed_issues = []

    # Extract test names from issues
    for issue in open_issues:
        # Skip issues that don't look like test failures
        if not 'Test Failure' in issue.title:
            continue

        # Extract test name from the issue title
        test_name = issue.title.replace('Test Failure: ', '')

        # Check if this test is now passing
        for passing_test in passing_tests:
            if test_name in passing_test:
                # This test is now passing, so the issue might be fixed
                potentially_fixed_issues.append((issue, "passing tests"))
                break

    return potentially_fixed_issues

if __name__ == "__main__":
    token = os.environ.get('GITHUB_TOKEN')
    if not token:
        raise ValueError("Missing GITHUB_TOKEN environment variable")

    g = Github(token)
    repo_name = os.environ.get('GITHUB_REPOSITORY', 'Lemniscate-world/Neural')
    repo = g.get_repo(repo_name)

    # Get the list of files changed in the latest commit
    changed_files = get_changed_files()
    print(f"Changed files: {changed_files}")

    # Find issues that might be fixed by the changes
    potentially_fixed_issues = find_issues_fixed_by_changes(repo, changed_files)
    print(f"Potentially fixed issues by code changes: {len(potentially_fixed_issues)}")

    # Check which issues are actually fixed
    issues_to_close = []
    for issue, changed_file in potentially_fixed_issues:
        if check_if_issue_fixed(issue, changed_file):
            issues_to_close.append((issue, changed_file))

    # Check if test results indicate that issues are fixed
    passing_tests = check_test_results()
    if passing_tests:
        print(f"Found {len(passing_tests)} passing tests")
        test_fixed_issues = find_issues_fixed_by_tests(repo, passing_tests)
        print(f"Potentially fixed issues by passing tests: {len(test_fixed_issues)}")
        issues_to_close.extend(test_fixed_issues)

    # Close the fixed issues
    if issues_to_close:
        print(f"Closing {len(issues_to_close)} fixed issues")
        close_fixed_issues(repo, issues_to_close)
    else:
        print("No issues to close")
