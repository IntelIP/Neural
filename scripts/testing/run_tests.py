#!/usr/bin/env python3
"""
Test runner for Neural SDK.

Run all tests or specific test modules.
"""

import sys
import subprocess
from pathlib import Path


def run_tests(test_path=None, verbose=False):
    """
    Run tests using pytest.
    
    Args:
        test_path: Specific test file or directory (None for all)
        verbose: Enable verbose output
    """
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if test_path:
        cmd.append(test_path)
    else:
        cmd.append("tests/")
    
    # Add coverage if available
    cmd.extend(["--cov=neural", "--cov-report=term-missing"])
    
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def main():
    """Main test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Neural SDK tests")
    parser.add_argument(
        "path",
        nargs="?",
        help="Specific test file or directory"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--no-cov",
        action="store_true",
        help="Disable coverage reporting"
    )
    
    args = parser.parse_args()
    
    # Check if pytest is installed
    try:
        import pytest
    except ImportError:
        print("Error: pytest not installed")
        print("Install with: pip install pytest pytest-asyncio pytest-cov")
        sys.exit(1)
    
    # Run tests
    exit_code = run_tests(args.path, args.verbose)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()