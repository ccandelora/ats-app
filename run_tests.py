#!/usr/bin/env python3
"""
Test runner for ATS Resume Checker application
"""

import os
import sys
import unittest
import logging
from argparse import ArgumentParser

def setup_test_environment():
    """Set up the environment for running tests."""
    # Create necessary directories
    for directory in ['logs', 'temp_storage', 'cache', 'data', 'tests/fixtures']:
        os.makedirs(directory, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename='logs/tests.log'
    )
    logger = logging.getLogger('test_runner')
    logger.info("Setting up test environment")

def discover_and_run_tests(verbosity=1, pattern='test_*.py', start_dir='tests'):
    """Discover and run all tests matching the pattern."""
    # Ensure the tests directory is in the path
    if not os.path.exists(start_dir):
        os.makedirs(start_dir, exist_ok=True)
        print(f"Created tests directory: {start_dir}")
    
    # Discover tests
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=start_dir, pattern=pattern)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)

def main():
    """Parse arguments and run tests."""
    parser = ArgumentParser(description="Run tests for ATS Resume Checker")
    parser.add_argument("-v", "--verbose", action="store_true", help="Run tests with verbose output")
    parser.add_argument("-p", "--pattern", default="test_*.py", help="Pattern to match test files")
    parser.add_argument("-d", "--directory", default="tests", help="Directory to start test discovery")
    
    args = parser.parse_args()
    
    # Set up environment
    setup_test_environment()
    
    # Run tests with appropriate verbosity
    verbosity = 2 if args.verbose else 1
    result = discover_and_run_tests(verbosity, args.pattern, args.directory)
    
    # Exit with appropriate status code
    if result.wasSuccessful():
        print("All tests passed! 🚀")
        return 0
    else:
        print("Tests failed! 😞")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 