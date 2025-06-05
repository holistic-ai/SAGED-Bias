#!/usr/bin/env python3
"""
Simple test runner script for SAGED-Bias project.
Usage: python run_tests.py [options]
"""

import subprocess
import sys
import argparse

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="Run tests for SAGED-Bias project")
    parser.add_argument("--coverage", action="store_true", help="Run tests with coverage report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Run tests in verbose mode")
    parser.add_argument("--specific", "-s", help="Run specific test file or test function")
    
    args = parser.parse_args()
    
    # Base test command
    cmd = "uv run pytest tests/"
    
    if args.verbose:
        cmd += " -v"
    
    if args.coverage:
        cmd += " --cov=saged --cov-report=term-missing --cov-report=html"
    
    if args.specific:
        cmd = cmd.replace("tests/", f"tests/{args.specific}")
    
    # Run tests
    success = run_command(cmd, "Running tests")
    
    if success:
        print("\n🎉 All tests passed!")
        if args.coverage:
            print("📊 Coverage report generated in htmlcov/")
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 