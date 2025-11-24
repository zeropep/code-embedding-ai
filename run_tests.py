#!/usr/bin/env python3
"""
Test runner script for the code embedding AI project
"""

import sys
import subprocess
import argparse
from pathlib import Path
import os
import io

# Fix Windows Unicode encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}")
    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error code: {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Run tests for the code embedding AI project")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--html", action="store_true", help="Generate HTML coverage report")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--module", "-m", help="Run tests for specific module")
    parser.add_argument("--parallel", "-n", type=int, help="Number of parallel processes")

    args = parser.parse_args()

    # Use current Python interpreter (prefer .venv if available)
    python_exe = sys.executable
    venv_path = Path(".venv")
    if venv_path.exists():
        # Check if we're in a venv or use .venv
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            # Already in a virtual environment
            pass
        else:
            # Try to use .venv if it exists
            if os.name == 'nt':  # Windows
                venv_python = venv_path / "Scripts" / "python.exe"
            else:  # Unix/Linux/Mac
                venv_python = venv_path / "bin" / "python"
            
            if venv_python.exists():
                python_exe = str(venv_python)
                print(f"📦 Using virtual environment: {python_exe}")

    # Base pytest command
    pytest_cmd = [python_exe, "-m", "pytest"]

    # Add verbosity
    if args.verbose:
        pytest_cmd.extend(["-v", "-s"])

    # Add parallel execution
    if args.parallel:
        pytest_cmd.extend(["-n", str(args.parallel)])

    # Filter tests
    if args.unit:
        pytest_cmd.extend(["-m", "unit"])
    elif args.integration:
        pytest_cmd.extend(["-m", "integration"])

    # Skip slow tests
    if args.fast:
        pytest_cmd.extend(["-m", "not slow"])

    # Module-specific tests
    if args.module:
        pytest_cmd.append(f"tests/test_{args.module}.py")

    # Coverage options
    if args.coverage:
        pytest_cmd.extend([
            "--cov=src",
            "--cov-report=term-missing"
        ])

        if args.html:
            pytest_cmd.append("--cov-report=html:htmlcov")

    # Run tests
    success = run_command(pytest_cmd, "Running tests")

    if not success:
        print("\n❌ Tests failed!")
        sys.exit(1)

    # Additional checks
    print("\n" + "="*50)
    print("🧪 TEST SUMMARY")
    print("="*50)

    # Check if we should run linting
    if not args.module:  # Only run full checks when not testing specific module
        # Run type checking
        print("\n🔍 Running type checking...")
        mypy_cmd = [python_exe, "-m", "mypy", "src/", "--ignore-missing-imports"]
        mypy_success = run_command(mypy_cmd, "Type checking")

        # Run code style checking
        print("\n🎨 Running code style checks...")
        flake8_cmd = [python_exe, "-m", "flake8", "src/", "tests/"]
        flake8_success = run_command(flake8_cmd, "Code style checking")

        # Run security checks
        print("\n🔒 Running security checks...")
        bandit_cmd = [python_exe, "-m", "bandit", "-r", "src/", "-f", "json"]
        bandit_success = run_command(bandit_cmd, "Security checking")

        if not all([mypy_success, flake8_success, bandit_success]):
            print("\n⚠️  Some quality checks failed, but tests passed")

    # Coverage report
    if args.coverage and args.html:
        coverage_path = Path("htmlcov/index.html")
        if coverage_path.exists():
            print(f"\n📊 Coverage report generated: {coverage_path.absolute()}")

    print("\n✅ All tests completed successfully!")


if __name__ == "__main__":
    main()