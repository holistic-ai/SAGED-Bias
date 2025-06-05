# SAGED-Bias Testing

This directory contains the test suite for the SAGED-Bias project.

## Running Tests

### Using uv (Recommended)

```bash
# Run all tests
uv run pytest tests/

# Run tests with verbose output
uv run pytest tests/ -v

# Run tests with coverage
uv run pytest tests/ --cov=saged --cov-report=term-missing

# Run tests with HTML coverage report
uv run pytest tests/ --cov=saged --cov-report=html
```

### Using the Test Runner Script

```bash
# Basic test run
python run_tests.py

# Verbose output with coverage
python run_tests.py --verbose --coverage

# Run specific test file
python run_tests.py --specific test_saged_data.py

# Run specific test function
python run_tests.py --specific test_saged_data.py::test_init
```

## Test Structure

- `test_saged_data.py` - Tests for the core SAGEDData class
- `test_scrape.py` - Tests for scraping functionality
- `test_utility.py` - Tests for utility functions

## Coverage

Current test coverage is around 16% of the total codebase. The tests primarily cover:

- ✅ Core data structures and validation (SAGEDData)
- ✅ Utility functions
- ✅ Basic scraping functionality
- ❌ Pipeline orchestration (needs more tests)
- ❌ Feature extraction (needs more tests)
- ❌ Bias diagnosis (needs more tests)

## Adding New Tests

When adding new tests:

1. Follow the existing naming convention (`test_*.py`)
2. Use pytest fixtures for common test data
3. Mock external dependencies (APIs, file systems, etc.)
4. Test both success and failure cases
5. Update this README if adding new test categories

## Dependencies

Test dependencies are managed through uv and include:

- pytest >= 8.4.0
- pytest-cov >= 6.1.1
- All production dependencies

## Environment Setup

```bash
# Create virtual environment
uv venv --python 3.10

# Install dependencies
uv sync

# Add test dependencies (if not already included)
uv add pytest pytest-cov
```
