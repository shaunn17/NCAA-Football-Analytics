#!/usr/bin/env python3
"""
Test Runner for NCAA Football Analytics

This script runs all tests and provides comprehensive test reporting.
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse

def run_tests(test_type="all", verbose=False, coverage=True):
    """Run tests with specified options"""
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    
    # Add coverage
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term-missing"])
    
    # Select test type
    if test_type == "unit":
        cmd.extend(["-m", "unit"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "api":
        cmd.extend(["-m", "api"])
    elif test_type == "data":
        cmd.extend(["-m", "data"])
    elif test_type == "ml":
        cmd.extend(["-m", "ml"])
    elif test_type == "fast":
        cmd.extend(["-m", "not slow"])
    elif test_type == "slow":
        cmd.extend(["-m", "slow"])
    
    # Add test directory
    cmd.append("tests/")
    
    print(f"🧪 Running tests: {' '.join(cmd)}")
    print("=" * 60)
    
    # Run tests
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    return result.returncode

def run_specific_test(test_file):
    """Run a specific test file"""
    cmd = ["python", "-m", "pytest", f"tests/{test_file}", "-v"]
    
    print(f"🧪 Running specific test: {' '.join(cmd)}")
    print("=" * 60)
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode

def run_pre_deployment_tests():
    """Run critical tests before deployment"""
    print("🚀 Running Pre-Deployment Tests")
    print("=" * 60)
    
    critical_tests = [
        "test_data_quality.py",
        "test_api_connections.py", 
        "test_ml_models.py",
        "test_database.py"
    ]
    
    failed_tests = []
    
    for test_file in critical_tests:
        print(f"\n📊 Running {test_file}...")
        result = run_specific_test(test_file)
        
        if result != 0:
            failed_tests.append(test_file)
            print(f"❌ {test_file} failed")
        else:
            print(f"✅ {test_file} passed")
    
    if failed_tests:
        print(f"\n❌ Pre-deployment tests failed: {', '.join(failed_tests)}")
        return 1
    else:
        print(f"\n✅ All pre-deployment tests passed!")
        return 0

def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(
        description="NCAA Football Analytics Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python scripts/run_tests.py

  # Run only fast tests
  python scripts/run_tests.py --type fast

  # Run specific test file
  python scripts/run_tests.py --file test_data_quality.py

  # Run pre-deployment tests
  python scripts/run_tests.py --pre-deployment

  # Run with verbose output
  python scripts/run_tests.py --verbose
        """
    )
    
    parser.add_argument(
        "--type", 
        choices=["all", "unit", "integration", "api", "data", "ml", "fast", "slow"],
        default="all",
        help="Type of tests to run"
    )
    
    parser.add_argument(
        "--file",
        help="Run specific test file"
    )
    
    parser.add_argument(
        "--pre-deployment",
        action="store_true",
        help="Run critical pre-deployment tests"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Skip coverage reporting"
    )
    
    args = parser.parse_args()
    
    # Change to project directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("🏈 NCAA Football Analytics - Test Runner")
    print("=" * 60)
    
    # Run pre-deployment tests
    if args.pre_deployment:
        return run_pre_deployment_tests()
    
    # Run specific test file
    if args.file:
        return run_specific_test(args.file)
    
    # Run tests by type
    return run_tests(
        test_type=args.type,
        verbose=args.verbose,
        coverage=not args.no_coverage
    )

if __name__ == "__main__":
    sys.exit(main())
